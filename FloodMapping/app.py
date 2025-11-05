import streamlit as st
import torch
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="Flood Mapping AI",
    page_icon="🌊",
    menu_items={
        'About': "# AI-Powered Flood Mapping Tool\nThis application uses deep learning to detect flooded areas in satellite imagery."
    }
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #0083B8;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 5px;
    }
    .success-text {
        color: #0083B8;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/tramplingh/Machine-Learning-Projects/main/FloodMapping/images/logo.png", width=100)
    st.title("🌊 Flood Mapping AI")
    st.markdown("---")
    st.markdown("""
    ### About
    This AI-powered tool helps identify flooded areas in satellite or aerial imagery using deep learning.
    
    ### How to use:
    1. Upload a satellite/aerial image
    2. Adjust visualization settings
    3. Analyze the results
    
    ### Technical Details:
    - Model: UNet with ResNet34 backbone
    - Input size: 256x256 pixels
    - Output: Flood probability map
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ by [Anish Dev Edward](https://github.com/tramplingh)")

# Main Content
st.title("🌊 AI-Powered Flood Mapping Tool")
st.markdown("### Transform satellite imagery into actionable flood insights")

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
def predict_and_visualize(image_bytes, alpha=0.5, colormap=cv2.COLORMAP_JET):
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
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Create overlay
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    return img, heatmap_color, overlay, pred_resized

# --- STREAMLIT UI ---
st.markdown("## 📤 Upload Image")
with st.expander("ℹ️ Upload Guidelines", expanded=True):
    st.markdown("""
    - Supported formats: JPG, JPEG, PNG
    - Best results with satellite or aerial imagery
    - Recommended resolution: at least 256x256 pixels
    - Clear, daytime images work best
    """)

uploaded_file = st.file_uploader("Choose a satellite or aerial image", type=["jpg", "jpeg", "png"])

# Settings in columns
if uploaded_file is not None:
    settings_col1, settings_col2 = st.columns(2)
    
    with settings_col1:
        overlay_alpha = st.slider(
            "🔍 Overlay Transparency",
            0.0, 1.0, 0.5, 0.05,
            help="Adjust the transparency of the flood detection overlay"
        )
    
    with settings_col2:
        colormap_option = st.selectbox(
            "🎨 Heatmap Color Scheme",
            ["jet", "viridis", "plasma", "inferno"],
            help="Choose the color scheme for flood visualization"
        )
        colormap_dict = {
            "jet": cv2.COLORMAP_JET,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
            "inferno": cv2.COLORMAP_INFERNO
        }

    # Process Button
    if st.button("🔍 Analyze Image", help="Click to start flood detection"):
        with st.spinner("🔄 Processing image... Please wait."):
            try:
                original, heatmap, overlay, pred_probs = predict_and_visualize(
                    uploaded_file.getvalue(),
                    overlay_alpha,
                    colormap=colormap_dict[colormap_option]
                )
                
                st.markdown("### 📊 Analysis Results")
                st.markdown(f"*Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
                
                # Display metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    # Using Otsu's method for adaptive thresholding
                    blur = cv2.GaussianBlur(heatmap, (5, 5), 0)
                    thresh_value, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    flood_percentage = np.mean(heatmap > thresh_value) * 100
                    st.metric("Estimated Flood Coverage", f"{flood_percentage:.1f}%",
                            help="Percentage of image area identified as flooded using adaptive thresholding")
                with metric_col2:
                    image_size = f"{original.shape[1]}x{original.shape[0]}"
                    megapixels = (original.shape[1] * original.shape[0]) / 1_000_000
                    st.metric("Image Resolution", f"{image_size} ({megapixels:.1f}MP)",
                            help="Image dimensions in pixels (width x height) and megapixels")
                with metric_col3:
                    # Calculate confidence using prediction probability distribution
                    pred_probs = pred_resized.flatten()
                    high_conf_mask = (pred_probs > 0.8) | (pred_probs < 0.2)  # Areas with clear predictions
                    confidence = (np.mean(high_conf_mask) * 100)
                    st.metric("Detection Confidence", f"{confidence:.1f}%",
                            help="Percentage of pixels where the model makes strong predictions (>80% certain)")
                
                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["🖼️ Original", "🌊 Flood Heatmap", "🎯 Overlay"])
                
                with tab1:
                    st.image(original, caption="Original Image", use_column_width=True)
                with tab2:
                    st.image(heatmap, caption="Flood Detection Heatmap", use_column_width=True)
                with tab3:
                    st.image(overlay, caption="Overlay Visualization", use_column_width=True)
                
                # Download section
                st.markdown("### 💾 Download Results")
                download_col1, download_col2, download_col3 = st.columns(3)
                with download_col1:
                    st.download_button(
                        "📥 Download Heatmap",
                        cv2.imencode('.png', cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))[1].tobytes(),
                        "flood_heatmap.png",
                        "image/png"
                    )
                with download_col2:
                    st.download_button(
                        "📥 Download Overlay",
                        cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))[1].tobytes(),
                        "flood_overlay.png",
                        "image/png"
                    )
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
else:
    st.info("👆 Please upload an image to begin the analysis")
