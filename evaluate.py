import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
from tqdm import tqdm

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="Evaluate the model and generate results for Power BI.")
parser.add_argument("--run-id", type=str, required=True, help="MLflow Run ID of the model to evaluate.")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")
args = parser.parse_args()

# --- 2. Device and Data Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "data"
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

lesion_type_dict = {
    'nv': 'Melanocytic nevi', 'mel': 'Melanoma', 'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma', 'akiec': 'Actinic keratoses', 'vasc': 'Vascular lesions', 'df': 'Dermatofibroma'
}
class_to_idx = {cls: i for i, cls in enumerate(lesion_type_dict.keys())}
idx_to_class = {i: cls for i, cls in class_to_idx.items()}

# --- 3. Load Test Data ---
df = pd.read_csv(METADATA_PATH)
# Ensure the same train/test split as in training
_, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['dx'])
_, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dx'])

test_df['label'] = test_df['dx'].map(class_to_idx)
def get_image_path(image_id):
    path1 = os.path.join(DATA_DIR, 'HAM10000_images_part_1', f'{image_id}.jpg')
    path2 = os.path.join(DATA_DIR, 'HAM10000_images_part_2', f'{image_id}.jpg')
    return path1 if os.path.exists(path1) else path2
test_df['image_path'] = test_df['image_id'].apply(get_image_path)

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
        image_id = row['image_id']
        return image, label, image_id

test_dataset = SkinLesionDataset(test_df, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# --- 4. Load Model from MLflow ---
print(f"Loading model from MLflow run: {args.run_id}")
model_uri = f"runs:/{args.run_id}/best_model"
model = mlflow.pytorch.load_model(model_uri)
model = model.to(device)
model.eval()

# --- 5. Generate Predictions ---
results = []
with torch.no_grad():
    for images, labels, image_ids in tqdm(test_loader, desc="Evaluating on test set"):
        images = images.to(device)
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_p, top_class = probabilities.topk(1, dim=1)
        
        for i in range(len(image_ids)):
            results.append({
                'image_id': image_ids[i],
                'true_label_id': labels[i].item(),
                'true_label_short': idx_to_class[labels[i].item()],
                'true_label_full': lesion_type_dict[idx_to_class[labels[i].item()]],
                'predicted_label_id': top_class[i].item(),
                'predicted_label_short': idx_to_class[top_class[i].item()],
                'predicted_label_full': lesion_type_dict[idx_to_class[top_class[i].item()]],
                'confidence': top_p[i].item()
            })

# --- 6. Save Results to CSV ---
results_df = pd.DataFrame(results)
output_path = "test_results_for_powerbi.csv"
results_df.to_csv(output_path, index=False)

print(f"Evaluation finished. Results saved to {output_path}")