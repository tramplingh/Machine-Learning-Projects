import streamlit as st
import torch
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Flood Mapping AI")
st.title("🌊 AI-Powered Flood Mapping Tool")
st.write("Upload a satellite or aerial image to detect and visualize flooded areas.")

# --- MODEL AND TRANSFORMS (Copy from your training script) ---
DEVICE = torch.device("cpu") # Use CPU for deployment
IMG_SIZE = 256
VAL_TFMS = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

@st.cache_resource # Cache the model for faster re-runs
def load_model():
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)

    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "best_unet.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# --- PREDICTION AND VISUALIZATION FUNCTION ---
def predict_and_visualize(image_bytes, alpha=0.5):
    # Load image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Preprocess and predict
    aug = VAL_TFMS(image=img)
    x = aug["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(model(x))[0][0].cpu().numpy()

    # Resize prediction to original image size
    pred_resized = cv2.resize(pred, (w, h))

    # Create heatmap
    heatmap = (pred_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Create overlay
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    return img, heatmap_color, overlay

# --- STREAMLIT UI ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
overlay_alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.5, 0.05)

if uploaded_file is not None:
    st.info("Processing image... Please wait.")
    original, heatmap, overlay = predict_and_visualize(uploaded_file.getvalue(), overlay_alpha)

    st.success("Processing complete!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(original, caption="Original Image", use_column_width=True)
    with col2:
        st.image(heatmap, caption="Flood Heatmap", use_column_width=True)
    with col3:
        st.image(overlay, caption="Overlay", use_column_width=True)
