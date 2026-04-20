from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import io
import base64
from typing import List, Dict
import os

app = FastAPI(title="Skin Lesion Classifier API", version="2.0")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_PATH = "../best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class definitions
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

# Risk levels
risk_levels = {
    'nv': 'LOW',
    'mel': 'HIGH',
    'bkl': 'MEDIUM',
    'bcc': 'HIGH',
    'akiec': 'HIGH',
    'vasc': 'LOW',
    'df': 'LOW'
}

# Model loading
model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval()
    return model

# Image preprocessing
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Grad-CAM generation
def generate_gradcam(model, input_tensor, target_class_idx, original_image):
    try:
        target_layer = model.features[-1]
        
        gradients = []
        def save_gradient(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        feature_maps = []
        def save_feature_map(module, input, output):
            feature_maps.append(output)
        
        handle1 = target_layer.register_forward_hook(save_feature_map)
        handle2 = target_layer.register_full_backward_hook(save_gradient)
        
        outputs = model(input_tensor)
        model.zero_grad()
        
        one_hot = torch.zeros_like(outputs)
        one_hot[0, target_class_idx] = 1
        outputs.backward(gradient=one_hot, retain_graph=True)
        
        guided_gradients = gradients[0].cpu().data.numpy()[0]
        target_feature_map = feature_maps[0].cpu().data.numpy()[0]
        
        weights = np.mean(guided_gradients, axis=(1, 2))
        
        cam = np.zeros(target_feature_map.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target_feature_map[i, :, :]
        
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam -= np.min(cam)
            cam /= np.max(cam)
        
        original_np = np.array(original_image)
        original_height, original_width = original_np.shape[:2]
        
        cam_resized = cv2.resize(cam, (original_width, original_height))
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255.0
        
        original_normalized = np.float32(original_np) / 255.0
        
        visualization = 0.4 * heatmap + 0.6 * original_normalized
        visualization = np.clip(visualization, 0, 1)
        visualization = np.uint8(255 * visualization)
        
        handle1.remove()
        handle2.remove()
        
        return visualization
        
    except Exception as e:
        original_np = np.array(original_image)
        overlay = np.zeros_like(original_np)
        if len(original_np.shape) == 3:
            overlay[:, :, 0] = 30
        return cv2.addWeighted(original_np, 0.9, overlay, 0.1, 0)

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    pil_image = Image.fromarray(image_array)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Pydantic models
class ProbabilityItem(BaseModel):
    class_name: str
    probability: float

class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_class_full: str
    confidence: float
    risk_level: str
    all_probabilities: List[ProbabilityItem]
    grad_cam_image: str  # Base64 encoded

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        load_model()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            device=str(device)
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device=str(device)
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}, content_type: {file.content_type}")
        
        # Load model
        model = load_model()
        
        # Read and process image
        contents = await file.read()
        print(f"File size: {len(contents)} bytes")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print(f"Image size: {image.size}")
        
        # Preprocess
        try:
            input_tensor = preprocess_image(image)
            print("Preprocessing successful")
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")
        
        # Make prediction
        try:
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                top_prob, top_class_idx = torch.max(probabilities, 0)
            print("Prediction successful")
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        
        # Get class names
        predicted_class_short = idx_to_class[top_class_idx.item()]
        predicted_class_full = lesion_type_dict[predicted_class_short]
        risk_level = risk_levels[predicted_class_short]
        
        # Generate Grad-CAM
        try:
            grad_cam_image = generate_gradcam(model, input_tensor, top_class_idx.item(), image)
            grad_cam_base64 = image_to_base64(grad_cam_image)
            print("Grad-CAM generation successful")
        except Exception as e:
            print(f"Grad-CAM error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Grad-CAM error: {str(e)}")
        
        # All probabilities
        all_probs = []
        for i, prob in enumerate(probabilities):
            class_short = idx_to_class[i]
            all_probs.append({
                "class_name": lesion_type_dict[class_short],
                "probability": prob.item()
            })
        
        # Sort by probability
        all_probs.sort(key=lambda x: x["probability"], reverse=True)
        
        try:
            response = PredictionResponse(
                predicted_class=predicted_class_short,
                predicted_class_full=predicted_class_full,
                confidence=top_prob.item(),
                risk_level=risk_level,
                all_probabilities=all_probs,
                grad_cam_image=grad_cam_base64
            )
            print("Response creation successful")
            return response
        except Exception as e:
            print(f"Response creation error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Response creation error: {str(e)}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
