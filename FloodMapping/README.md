
### TL;DR 
This project implements an image segmentation system for automated flood detection in satellite and aerial imagery using _deep learning and transfer learning_. The primary objective is to identify and visualize regions affected by flooding by generating a probability heatmap overlay on the input image.

*Check it out here by uploading* an aerial image of a flood affected area: https://floodmapping-xnqphjv4amhud88gbgfd2n.streamlit.app/ 

---

## Problem Statement

Floods are destructive natural disasters, and rapid flood assessment is critical for disaster response. Traditional manual analysis of satellite data is time-consuming. The goal of this project is to develop a deep learning–based segmentation model that can highlight flood-affected areas in images, enabling faster decision-making.


---

## Dataset

Flood-images (aerial imagery with pixel-level flood annotations
https://www.kaggle.com/datasets/saiharshitjami/flood-images-mask-segmentation?select=Images

Input: RGB images of flood-affected regions.

Ground-truth: Segmentation masks labeling flooded vs. non-flooded pixels.

Preprocessing:

Resizing to fixed dimensions (e.g., 256×256).

Normalization using ImageNet mean/std.

Data augmentation (rotation, flipping, brightness/contrast adjustment) for generalization.




---

## Model Architecture

U-Net convolutional neural network for semantic segmentation.

Encoder: Pretrained backbone (e.g., ResNet34, EfficientNet-B0) for feature extraction.

Decoder: Upsampling layers with skip connections to reconstruct pixel-level predictions.


Output: A probability map (H×W) where each pixel’s value represents the likelihood of flooding.



---

## Training Setup

Framework: PyTorch with segmentation_models_pytorch.

Loss Function: Combination of Binary Cross-Entropy (BCE) and Dice Loss to handle class imbalance.

Optimizer: Adam with learning rate scheduling.

## Metrics:

Intersection over Union (IoU)

Dice Coefficient (F1 Score)

Visual overlays for qualitative assessment.




---

## Inference & Visualization

For a new image, the trained model generates a flood probability map.

Postprocessing:

Convert probability map (0–1) into a heatmap using color mapping (blue → low, red → high).

Overlay heatmap onto original satellite image for interpretability.


## Output Formats:

Heatmap alone

Overlay visualization

Flood coverage statistics (% of pixels above threshold)




---

## Deployment

Built a lightweight Streamlit web application with:

File uploader (for satellite/aerial images).

Real-time model inference (1–3 seconds per image on GPU/CPU).

Side-by-side display: Original Image | Flood Heatmap | Overlay.

Adjustable overlay transparency for better visualization.


---

## Real-Life Use Cases:

*   Disaster Response and Management: Quickly identify flooded areas after heavy rainfall or natural disasters to assess the impact, prioritize rescue efforts, and allocate resources effectively.
*   Urban Planning: Analyze historical flood data and predict potential flood zones to inform urban development, infrastructure planning, and land-use regulations.
*   Environmental Monitoring: Track changes in water bodies and identify areas prone to flooding due to climate change or other environmental factors.
*   Insurance and Risk Assessment: Help insurance companies assess flood risk for properties and inform policy decisions.
*   Agriculture: Identify flooded agricultural land to estimate crop damage and plan for recovery.

## Next Steps:

*   Improve Model Performance: Experiment with different encoder architectures, hyperparameters, and loss functions to further improve the model's accuracy.
*   Expand Dataset: Train the model on a larger and more diverse dataset to improve its generalization capabilities.
*   Implement More Sophisticated Post-processing: Explore advanced techniques for refining the model's output, such as conditional random fields (CRFs).
*   Deploy as a Web Service: Deploy the trained model as a robust web service for wider accessibility and integration into other applications.
*   Real-time Processing: Investigate methods for near real-time flood detection using streaming data.
*   Explore Different Data Sources: Incorporate other data sources like satellite radar data or social media reports to enhance flood detection.
