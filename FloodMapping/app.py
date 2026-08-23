import os

import streamlit as st
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Aerial Flood Water Mapping")
st.title("Aerial Flood Water Mapping")
st.write(
    "Upload an aerial image of a flood-affected area. The model segments "
    "standing water at pixel level and reports how much of the frame it covers."
)

DEVICE = torch.device("cpu")
IMG_SIZE = 256
MAX_SIDE = 1600  # cap input size so a huge upload can't stall the CPU container

VAL_TFMS = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


@st.cache_resource
def load_model():
    # encoder_weights=None -- the checkpoint overwrites them anyway, so
    # downloading ImageNet weights at startup is wasted time and a
    # needless network dependency.
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=1)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best_unet.pth")
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


model = load_model()


def decode_image(image_bytes):
    """Decode an upload to RGB, or return None if it isn't a readable image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


@st.cache_data(show_spinner=False)
def predict(image_bytes):
    """Return the RGB image and a float probability map at the image's own size."""
    img = decode_image(image_bytes)
    if img is None:
        return None, None
    h, w = img.shape[:2]

    x = VAL_TFMS(image=img)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0][0].cpu().numpy()

    return img, cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def build_views(img, prob, threshold, alpha):
    """Binary mask, thresholded heatmap, and an overlay that only paints detections."""
    mask = prob >= threshold

    # Heatmap of confidence, but only where the model actually detected water.
    # Below-threshold pixels stay black instead of rendering as confident blue.
    shown = np.where(mask, prob, 0.0)
    heatmap = cv2.applyColorMap((shown * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap[~mask] = 0

    # Overlay: original everywhere, blended colour only inside the mask.
    blended = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    overlay = img.copy()
    overlay[mask] = blended[mask]

    # Outline the detected regions so the boundary is legible.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)

    return mask, heatmap, overlay


# --- SIDEBAR CONTROLS ---
st.sidebar.header("Controls")
threshold = st.sidebar.slider(
    "Detection threshold", 0.05, 0.95, 0.50, 0.05,
    help="Lower catches more water but raises false alarms. Higher is stricter.",
)
overlay_alpha = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.55, 0.05)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Scope.** Trained on aerial photographs of flooded scenes only. It segments "
    "*standing water*, so permanent rivers, lakes and ponds are also detected. "
    "It cannot tell normal water from floodwater. Images are not georeferenced, "
    "so coverage is a share of the frame, not an area on the ground."
)

# --- MAIN ---
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload an aerial image to begin.")
    st.stop()

with st.spinner("Running the model..."):
    img, prob = predict(uploaded_file.getvalue())

if img is None:
    st.error("That file could not be read as an image. Try a JPG or PNG.")
    st.stop()

mask, heatmap, overlay = build_views(img, prob, threshold, overlay_alpha)

# --- COVERAGE STATISTICS ---
coverage = float(mask.mean())
mean_conf = float(prob[mask].mean()) if mask.any() else 0.0
n_regions = len(cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)[0])

c1, c2, c3 = st.columns(3)
c1.metric("Water coverage", f"{coverage * 100:.1f}%")
c2.metric("Mean confidence", f"{mean_conf:.2f}" if mask.any() else "—")
c3.metric("Distinct regions", f"{n_regions}")

if coverage < 0.005:
    st.warning(
        f"No water detected at threshold {threshold:.2f}. Either the scene is dry, "
        "or it looks unlike anything in the training data. Try lowering the threshold."
    )
elif coverage > 0.85:
    st.warning(
        "Almost the entire frame is flagged. Check the overlay before trusting this — "
        "uniform brown or grey scenes can trigger a blanket detection."
    )

# --- VISUALS ---
t1, t2, t3 = st.tabs(["Overlay", "Confidence heatmap", "Binary mask"])
with t1:
    st.image(overlay, caption=f"Detections outlined at threshold {threshold:.2f}",
             use_container_width=True)
with t2:
    st.image(heatmap, caption="Model confidence inside detected regions",
             use_container_width=True)
with t3:
    st.image(mask.astype(np.uint8) * 255, caption="Binary water mask",
             use_container_width=True)

with st.expander("Original image"):
    st.image(img, use_container_width=True)

# --- DOWNLOAD ---
ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
if ok:
    st.download_button(
        "Download binary mask (PNG)",
        data=buf.tobytes(),
        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_mask.png",
        mime="image/png",
    )
