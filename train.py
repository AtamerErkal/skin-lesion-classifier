import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="Train a skin cancer classification model.")
parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer.")
args = parser.parse_args()

# --- 2. Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 3. Data Paths and Class Definitions ---
DATA_DIR = "data"
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
class_to_idx = {cls: i for i, cls in enumerate(lesion_type_dict.keys())}

# --- 4. Data Loading and Preprocessing ---
df = pd.read_csv(METADATA_PATH)
df['label'] = df['dx'].map(class_to_idx)

# Create full image paths and handle both image directories
def get_image_path(image_id):
    path1 = os.path.join(DATA_DIR, 'HAM10000_images_part_1', f'{image_id}.jpg')
    path2 = os.path.join(DATA_DIR, 'HAM10000_images_part_2', f'{image_id}.jpg')
    return path1 if os.path.exists(path1) else path2

df['image_path'] = df['image_id'].apply(get_image_path)

# Split data into training, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['dx'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dx'])

# Image transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Custom Dataset class
class SkinLesionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        label = torch.tensor(row['label'], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

# Handle class imbalance with WeightedRandomSampler
class_counts = train_df['label'].value_counts().sort_index()
weights = 1. / torch.tensor(class_counts.values, dtype=torch.float)
sample_weights = weights[train_df['label'].values]
sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Create DataLoaders
train_dataset = SkinLesionDataset(train_df, transform=data_transforms['train'])
val_dataset = SkinLesionDataset(val_df, transform=data_transforms['val'])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# --- 5. Model Definition ---
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_to_idx))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# --- 6. Training Loop with MLflow ---
mlflow.set_experiment("Skin Cancer Classification")

with mlflow.start_run() as run:
    print(f"MLflow Run ID: {run.info.run_id}")
    mlflow.log_params({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "model_architecture": "EfficientNet-B0"
    })

    best_val_accuracy = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)

        # Validation phase
        model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_accuracy = corrects.double() / len(val_loader.dataset)
        
        mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", epoch_val_accuracy.item(), step=epoch)

        print(f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}")

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            mlflow.pytorch.log_model(model, "best_model")
            print("New best model saved!")

print("Training finished.")