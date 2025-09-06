import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os

# OpenCV'yi güvenli şekilde import et
try:
    import cv2
except ImportError:
    st.error("OpenCV not found. Please check requirements.txt")
    st.stop()

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
    if not os.path.exists(MODEL_PATH):
        st.error(f"""
        🚨 **Model file not found!**
        
        The model file `{MODEL_PATH}` is missing from your repository.
        
        **To fix this:**
        1. Make sure `best_model.pth` is in your GitHub repository root
        2. If the file is too large (>100MB), use Git LFS:
           ```
           git lfs track "*.pth"
           git add .gitattributes
           git add best_model.pth
           git commit -m "Add model with Git LFS"
           ```
        3. Push to GitHub and redeploy
        """)
        return None
    
    try:
        # Streamlit Cloud için güvenli model loading
        with st.spinner("Loading AI model... Please wait."):
            model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            model.eval()
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"""
        🚨 **Error loading model:**
        ```
        {str(e)}
        ```
        
        **Possible solutions:**
        - Check if the model was trained with a compatible PyTorch version
        - Verify the model file is not corrupted
        - Try re-uploading the model file
        """)
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

def generate_gradcam(model, input_tensor, target_class_idx, original_image):
    """
    Generate a Grad-CAM visualization that properly aligns with the original image.
    """
    try:
        # Get the last convolutional layer
        target_layer = model.features[-1]

        # Register hooks
        gradients = []
        def save_gradient(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        feature_maps = []
        def save_feature_map(module, input, output):
            feature_maps.append(output)

        handle1 = target_layer.register_forward_hook(save_feature_map)
        handle2 = target_layer.register_full_backward_hook(save_gradient)

        # Forward pass
        outputs = model(input_tensor)
        model.zero_grad()

        # Backward pass
        one_hot = torch.zeros_like(outputs)
        one_hot[0, target_class_idx] = 1
        outputs.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and feature maps
        guided_gradients = gradients[0].cpu().data.numpy()[0]
        target_feature_map = feature_maps[0].cpu().data.numpy()[0]
        
        # Calculate weights
        weights = np.mean(guided_gradients, axis=(1, 2))
        
        # Create CAM
        cam = np.zeros(target_feature_map.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target_feature_map[i, :, :]
        
        # Process CAM
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam -= np.min(cam)
            cam /= np.max(cam)
        
        # Use original image dimensions
        original_np = np.array(original_image)
        original_height, original_width = original_np.shape[:2]
        
        # Resize CAM to original image size
        cam_resized = cv2.resize(cam, (original_width, original_height))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255.0
        
        # Normalize original image
        original_normalized = np.float32(original_np) / 255.0
        
        # Combine
        visualization = 0.4 * heatmap + 0.6 * original_normalized
        visualization = np.clip(visualization, 0, 1)
        visualization = np.uint8(255 * visualization)
        
        # Clean up
        handle1.remove()
        handle2.remove()
        
        return visualization
        
    except Exception as e:
        # Fallback: return original with red tint
        original_np = np.array(original_image)
        overlay = np.zeros_like(original_np)
        if len(original_np.shape) == 3:
            overlay[:, :, 0] = 30
        return cv2.addWeighted(original_np, 0.9, overlay, 0.1, 0)

# --- 4. Sidebar Content ---
st.sidebar.title("🔬 Project Details")
st.sidebar.markdown("""
This project demonstrates an end-to-end deep learning workflow for skin lesion classification, 
showcasing skills relevant for Data Science and ML Engineering positions.
""")

st.sidebar.header("📊 Model & Performance")
st.sidebar.markdown("""
- **Model Architecture:** EfficientNet-B0
- **Dataset:** HAM10000 (Human Against Machine with 10000 Training Images)
- **Framework:** PyTorch
- **Deployment:** Streamlit Cloud
""")

# Performance metrics
st.sidebar.subheader("📈 Test Set Metrics")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Accuracy", "83.8%", delta="2.1%")
with col2:
    st.metric("F1-Score (Macro)", "0.78", delta="0.05")

st.sidebar.info("""
💡 **Note:** Metrics are based on the held-out test set. 
The F1-Score (Macro) gives equal weight to each class, 
important for handling class imbalance.
""")

st.sidebar.header("🛠️ Technology Stack")
st.sidebar.markdown("""
- **ML Framework:** PyTorch, Torchvision
- **Web App:** Streamlit
- **Computer Vision:** OpenCV, PIL
- **Data Processing:** Pandas, NumPy
- **Deployment:** Streamlit Cloud
- **Version Control:** Git & GitHub
""")

st.sidebar.header("⚠️ Medical Disclaimer")
st.sidebar.error("""
**This tool is for educational purposes only.**
Never use this for actual medical diagnosis.
Always consult qualified healthcare professionals 
for medical advice.
""")

# --- 5. Main Page Content ---
st.markdown("""
<div style="text-align: center;">
    <h1>🔬 AI-Powered Skin Lesion Analyzer</h1>
    <p style="font-size: 1.2em; color: #666;">
        Upload an image of a skin lesion to classify it into one of 7 types 
        and visualize the model's reasoning using Grad-CAM technology.
    </p>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()

if model is not None:
    # File uploader
    st.markdown("### 📤 Upload Skin Lesion Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, PNG)", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear, well-lit image of the skin lesion for best results."
    )

    if uploaded_file is not None:
        try:
            # Load and display the uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Show image info
            st.info(f"📷 **Image uploaded:** {uploaded_file.name} ({image.size[0]}×{image.size[1]} pixels)")
            
            with st.spinner('🤖 Analyzing the image... Please wait.'):
                # Preprocess the image
                input_tensor = preprocess_image(image)
                
                # Make prediction
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    top_prob, top_class_idx = torch.max(probabilities, 0)
                
                # Get class names
                predicted_class_short = idx_to_class[top_class_idx.item()]
                predicted_class_full = lesion_type_dict[predicted_class_short]
                
                # Generate Grad-CAM visualization
                grad_cam_image = generate_gradcam(model, input_tensor, top_class_idx.item(), image)
                grad_cam_pil = Image.fromarray(grad_cam_image)

            # Display results
            st.markdown("---")
            st.markdown("## 🔍 Analysis Results")

            # Create two columns for the images
            col_img1, col_img2 = st.columns(2)

            with col_img1:
                st.markdown("### 📷 Original Image")
                st.image(image, use_container_width=True)
            
            with col_img2:
                st.markdown("### 🎯 AI Focus Areas (Grad-CAM)")
                st.image(
                    grad_cam_pil, 
                    use_container_width=True, 
                    caption="🔥 Red/warm colors show where the AI focused most for its decision"
                )

            # Results section
            st.markdown("### 📊 Prediction Results")
            
            # Create a DataFrame with all probabilities
            prob_df = pd.DataFrame({
                'Lesion Type': [lesion_type_dict[idx_to_class[i]] for i in range(len(probabilities))],
                'Probability': [p.item() for p in probabilities],
                'Confidence': [f"{p.item():.1%}" for p in probabilities]
            })
            
            # Sort by probability
            prob_df_sorted = prob_df.sort_values(by='Probability', ascending=False)
            top_prediction = prob_df_sorted.iloc[0]

            # Display top prediction prominently
            confidence_color = "green" if top_prediction['Probability'] > 0.7 else "orange" if top_prediction['Probability'] > 0.4 else "red"
            
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #f0f9ff 0%, #e0f2fe 100%); 
                        padding: 25px; border-radius: 15px; border-left: 5px solid {confidence_color}; 
                        margin: 20px 0px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h2 style="color: #1e40af; margin-bottom: 10px; text-align: center;">
                    🎯 Primary Prediction
                </h2>
                <h1 style="color: #1e40af; font-size: 2.5em; text-align: center; margin: 10px 0;">
                    {top_prediction['Lesion Type']}
                </h1>
                <h2 style="color: {confidence_color}; font-size: 2em; text-align: center; margin-top: 5px;">
                    {top_prediction['Confidence']} confidence
                </h2>
            </div>
            """, unsafe_allow_html=True)

            # Show all predictions in a nice table
            st.markdown("### 📈 All Predictions")
            
            # Style the dataframe
            styled_df = prob_df_sorted.style.format({'Probability': '{:.2%}'}).hide(subset=['Probability'], axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Add interpretation help
            with st.expander("🤔 How to interpret these results"):
                st.markdown("""
                - **High confidence (>70%):** The model is quite certain about its prediction
                - **Medium confidence (40-70%):** The model has moderate certainty; consider multiple possibilities  
                - **Low confidence (<40%):** The model is uncertain; the image may be unclear or atypical
                
                **Remember:** This is an AI tool for educational purposes only. 
                Always consult healthcare professionals for medical concerns.
                """)

        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("💡 Try uploading a different image or check if the file is corrupted.")

else:
    st.error("""
    🚨 **Application not ready**
    
    The AI model could not be loaded. Please check:
    1. Model file is present in the repository
    2. Dependencies are correctly installed
    3. No file corruption occurred
    
    Contact the developer if this issue persists.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>
        🔬 <strong>Skin Lesion Analyzer</strong> | 
        Built with ❤️ using PyTorch & Streamlit | 
        🚨 For Educational Use Only
    </p>
</div>
""", unsafe_allow_html=True)
