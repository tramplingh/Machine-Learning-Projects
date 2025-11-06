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

# Custom CSS for presentation
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white !important;
    }
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #00bfff;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0099cc;
        transform: translateY(-2px);
    }
    .stAlert {
        padding: 1rem;
        border-radius: 5px;
        background-color: rgba(255, 255, 255, 0.1);
    }
    .success-text {
        color: #00bfff;
        font-size: 18px;
        font-weight: bold;
    }
    .css-1d391kg, .css-1lcbmhc {
        background-color: rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        color: white;
        background-color: transparent;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.2);
    }
    h1, h2, h3, h4, h5, h6, .st-emotion-cache-ztfqz8 {
        color: white !important;
    }
    .st-emotion-cache-16txtl3 {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: rgba(0, 0, 0, 0.2);
    }
    .sidebar .sidebar-content {
        background-color: rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with presentation styling
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='color: white; font-size: 2.5em; margin-bottom: 10px;'>🌊</h1>
            <h2 style='color: white; margin-top: 0;'>Flood Mapping AI</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: white; margin-top: 0;'>Project Overview</h3>
        <p style='color: white;'>
        Advanced deep learning solution for real-time flood detection in satellite imagery using state-of-the-art computer vision techniques.
        </p>
    </div>
    
    <div style='background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: white; margin-top: 0;'>Key Features</h3>
        <ul style='color: white;'>
            <li>Real-time flood detection</li>
            <li>Interactive visualization</li>
            <li>High-precision mapping</li>
            <li>Multiple visualization modes</li>
        </ul>
    </div>
    
    <div style='background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;'>
        <h3 style='color: white; margin-top: 0;'>Technology Stack</h3>
        <ul style='color: white;'>
            <li>Model: UNet Architecture</li>
            <li>Backbone: ResNet34</li>
            <li>Framework: PyTorch</li>
            <li>Interface: Streamlit</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: white;'>
            Presented by<br>
            <b>Anish Dev Edward</b>
        </div>
    """, unsafe_allow_html=True)

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

    return img, heatmap_color, overlay

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
                original, heatmap, overlay = predict_and_visualize(
                    uploaded_file.getvalue(),
                    overlay_alpha,
                    colormap=colormap_dict[colormap_option]
                )
                
                st.markdown("### 🎯 Analysis Results")
                st.markdown(
                    """
                    <div style='background-color: rgba(255, 255, 255, 0.1); 
                             padding: 20px; 
                             border-radius: 10px; 
                             margin-bottom: 20px;
                             border: 1px solid rgba(255, 255, 255, 0.2);'>
                        <h4 style='color: white; margin: 0;'>
                            🔍 Deep Learning Flood Detection Analysis
                        </h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Tabs for different views with enhanced styling
                tab1, tab2, tab3 = st.tabs(["🖼️ Original Image", "🌊 Flood Detection", "🎯 Combined View"])
                
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
