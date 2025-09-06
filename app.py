import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Skin Lesion Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. Configuration and Model Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"

# Class Definitions
lesion_type_dict = {
    'nv': 'Melanocytic nevi (Mole)',
    'mel': 'Melanoma (Malignant)',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
idx_to_class = {i: cls for i, cls in enumerate(lesion_type_dict.keys())}

@st.cache_resource
def load_model():
    """Load the PyTorch model from a file, cached for performance."""
    try:
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Please make sure the 'best_model.pth' file is in the root directory.")
        return None

# --- 3. Image Processing and XAI Functions ---
def preprocess_image(image):
    """Apply transformations to an image to prepare it for the model."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def generate_grad_cam(model, input_tensor, target_category=None):
    """Generate a Grad-CAM visualization for the model's prediction."""
    # Correctly identify the last convolutional block in EfficientNet-B0
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    rgb_img = input_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
    
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.6)
    return visualization

# --- 4. Sidebar Content ---
st.sidebar.title("Project Details")
st.sidebar.markdown("""
    This project is an end-to-end demonstration of a deep learning workflow for a computer vision task, specifically for classifying skin lesions. It is designed to showcase skills relevant for a Data Scientist position.
""")
st.sidebar.header("Model & Performance")
st.sidebar.markdown(f"""
- **Model Architecture:** `EfficientNet-B0`
- **Dataset:** HAM10000 ("Human Against Machine with 10000 Training Images")
- **Framework:** `PyTorch`
""")

# IMPORTANT: Manually update these metrics from your MLflow run!
st.sidebar.subheader("Test Set Metrics")
col1, col2 = st.sidebar.columns(2)
col1.metric("Accuracy", "83.8%")
col2.metric("F1-Score (Macro)", "0.78") 

st.sidebar.markdown("""
    *Metrics are based on the held-out test set after training the model. The F1-Score (Macro) gives equal weight to each class, which is important for imbalanced datasets.*
""")
st.sidebar.header("Technology Stack")
st.sidebar.markdown("""
- **Experiment Tracking:** `MLflow`
- **Deployment:** `Streamlit`
- **Data Handling:** `Pandas`, `Scikit-learn`
- **Version Control:** `Git` & `GitHub`
""")

# --- 5. Main Page Content ---
st.markdown("<h1 style='text-align: center;'>🔬 AI-Powered Skin Lesion Analyzer</h1>", unsafe_allow_html=True)
st.write("Upload an image of a skin lesion to classify it into one of 7 types and visualize the model's reasoning using Grad-CAM.")

model = load_model()

if model:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner('Analyzing the image...'):
            input_tensor = preprocess_image(image)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                top_prob, top_class_idx = torch.max(probabilities, 0)
            
            predicted_class_short = idx_to_class[top_class_idx.item()]
            predicted_class_full = lesion_type_dict[predicted_class_short]
            
            grad_cam_image = generate_grad_cam(model, input_tensor, target_category=top_class_idx.item())

        # Create two columns for the images
        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.header("Uploaded Image")
            st.image(image, use_container_width=True)
        
        with col_img2:
            st.header("Model's Focus (Grad-CAM)")
            st.image(grad_cam_image, use_container_width=True, caption="Warmer colors indicate areas the model focused on most for its decision.")

        st.markdown("---") # A separator for visual clarity
        st.markdown("<h2 style='text-align: center;'>AI Analysis Results</h2>", unsafe_allow_html=True)

        # Create a DataFrame with all probabilities
        prob_df = pd.DataFrame({
            'Lesion Type': [lesion_type_dict[idx_to_class[i]] for i in range(len(probabilities))],
            'Probability': [p.item() for p in probabilities]
        })
        
        # Sort the DataFrame and get the top prediction
        prob_df_sorted = prob_df.sort_values(by='Probability', ascending=False)
        top_prediction = prob_df_sorted.iloc[0]
        other_predictions = prob_df_sorted.iloc[1:]

        # Create columns for the main prediction and others
        col_top_pred, col_other_preds = st.columns([1, 2])

        with col_top_pred:
            st.subheader("Top Prediction")
            st.markdown(f"""
            <div style="background-color: #e0f2f1; padding: 20px; border-radius: 10px; border: 2px solid #00796b; text-align: center;">
                <h3 style="color: #004d40; margin-bottom: 5px; font-size: 1.5em;">{top_prediction['Lesion Type']}</h3>
                <h1 style="color: #004d40; font-size: 3.5em; margin-top: 5px;">{top_prediction['Probability']:.2%}</h1>
            </div>
            """, unsafe_allow_html=True)
            
        with col_other_preds:
            st.subheader("Other Probabilities")
            # Display other probabilities in a compact DataFrame
            st.dataframe(
                other_predictions.style.format({'Probability': '{:.2%}'}),
                use_container_width=True,
                hide_index=True
            )
            
else:
    st.warning("Application is not functional until the model file is available.")
