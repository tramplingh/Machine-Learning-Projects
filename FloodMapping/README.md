# Aerial Water Segmentation for Flood Assessment

A U-Net segmentation model that identifies standing water in aerial photographs of
flood-affected areas, producing a pixel-level mask, a confidence heatmap, and a
coverage statistic.

**Live demo:** https://floodmapping-xnqphjv4amhud88gbgfd2n.streamlit.app/
**Methodology write-up:** https://docs.google.com/document/d/1A4e_wOdkY6JHDK0CG1uS1G-A6v38XIRbClofJLuHoXo/edit?usp=drivesdk

**Test-set result: 0.881 mean IoU across 341 held-out images**, against 0.451 for a tuned
classical colour-threshold baseline.

---

## What this model does and does not do

**It does:** segment visible standing water in optical aerial imagery, at pixel level,
and report what share of the frame that water covers.

**It does not:** distinguish floodwater from normal water. The training set contains
only flooded scenes, so the model was never shown a dry landscape and never had the
chance to learn what "abnormal" water looks like. Rivers, lakes, ponds and reservoirs
are detected as confidently as floodwater. Turning this into a true flood detector would
require either before/after image pairs of the same location, or a baseline water mask
to difference against.

**It also does not** work on georeferenced data. Inputs are ordinary photographs with no
coordinates, so coverage is reported as a percentage of the image, not as an area on the
ground. Flooded area in km² is not something this system can produce.

This is stated up front because the distinction matters for anyone considering the
outputs operationally.

---

## Problem

Rapid post-disaster damage assessment depends on quickly identifying inundated areas.
Manual inspection of aerial imagery is slow. This project trains a semantic segmentation
model to automate the pixel-level water delineation step, so an analyst can triage a
batch of images by water coverage rather than reviewing each one.

---

## Dataset

[Flood images with segmentation masks](https://www.kaggle.com/datasets/saiharshitjami/flood-images-mask-segmentation)
— RGB aerial photographs of flood-affected regions with hand-labelled binary masks.

| | |
|---|---|
| Total paired image/mask files | 3,402 |
| Train / Validation / Test | 2,721 / 340 / 341 (80 / 10 / 10) |
| Split method | Random shuffle, `random.seed(42)` |
| Input resolution | Resized to 256 × 256 |
| Normalisation | ImageNet mean/std |

**Known limitation of the split.** Files are shuffled individually, not grouped by source
event. The dataset contains multiple frames of the same location — two of the ten
worst-scoring test images are visibly the same neighbourhood photographed from different
angles — so near-duplicates may straddle the train/test boundary. The scores below should
be read as an upper bound until a perceptual-hash audit rules this out.

---

## Model

- **Architecture:** U-Net (`segmentation_models_pytorch`)
- **Encoder:** ResNet-34, ImageNet-pretrained
- **Output:** single-channel logit map, sigmoid to a per-pixel probability
- **Loss:** Binary Cross-Entropy + Dice, summed
- **Optimiser:** Adam, fixed learning rate 1e-4 (no scheduler)
- **Augmentation:** horizontal flip (p=0.5), random brightness/contrast (p=0.2)
- **Epochs:** 10, with the best-validation-IoU checkpoint retained

Validation loss reaches its minimum around epoch 8–9 and rises afterwards, so 10 epochs
is roughly the useful ceiling for this configuration. Best-checkpoint saving prevents the
final overfitted weights from being used.

---

## Results

Held-out test set, 341 images, evaluated at 256 × 256 to match the training resolution.

| Metric | U-Net | HSV baseline |
|---|---|---|
| **Mean IoU** | **0.8810** | 0.4513 |
| Mean Dice | 0.9348 | 0.5593 |
| Mean Precision | 0.9333 | 0.6879 |
| Mean Recall | 0.9376 | 0.5728 |
| Median IoU | 0.8945 | — |
| Std. dev. of per-image IoU | 0.0754 | — |

**The U-Net outperforms the baseline on 341 of 341 test images** — every case, without
exception.

### Why the baseline is a fair comparison

A learned model should have to prove it beats the obvious heuristic. Floodwater in these
scenes is a distinctive muddy brown, so a colour threshold is the natural thing to try
first, and if it were competitive the deep model would not be justified.

The baseline is an HSV colour-range threshold with morphological open/close cleanup. Its
six parameters were selected by grid search over 192 combinations, scored on 100
**training** images, and frozen before the test set was touched. It was never tuned on
test data.

It is not a strawman: at 0.688 precision it is right about two-thirds of the time when it
flags a pixel. Its failure is recall — 0.573, meaning it misses close to half the water
present. Muddy water, wet soil, bare earth and shadowed vegetation occupy overlapping
regions of HSV space that no fixed threshold can separate. The U-Net's advantage comes
from spatial context and texture, not from better colour sensitivity.

The distribution makes the gap concrete: the baseline has a long tail of near-zero scores,
while the U-Net is tightly clustered above 0.80. See `Eval/iou_distribution.png`.

**Threshold.** A sweep across 0.05–0.95 confirms 0.50 as the optimal operating point, with
a clean precision/recall crossover there. It was verified rather than assumed. See
`Eval/threshold_sweep.png`.

**Consistency.** A standard deviation of 0.075 with a minimum of 0.33 means performance is
stable across the test set rather than an average of very good and very bad cases.

### A note on the evaluation code

An earlier version of the IoU function summed over the wrong tensor dimensions
(`.sum((1, 2))` on a `(B, 1, H, W)` tensor), computing IoU per *column of pixels* rather
than per image. On this model it produced 0.864 against a true 0.881 — a modest distortion,
but the reported quantity was not IoU. The corrected version sums over `(1, 2, 3)`.
`evaluate.py` prints both so the difference is reproducible.

---

## Known failure modes

`evaluate.py` produces a gallery of the ten worst-scoring test images with a colour-coded
error map (green = correct detection, red = false positive, blue = missed water). See
`Eval/failure_gallery.png`.

**Most "failures" are boundary error, not misdetection.** Eight of the ten worst cases show
green cores with thin red and blue fringes: the model located the water correctly and was
imprecise at the edge by a few pixels. On narrow channels between buildings the union is
small, so a few pixels of edge error costs a disproportionate amount of IoU.

The genuine failures are two cases of **wide, shallow inundation over vegetation** — flooded
fields and cropland where the water surface is partly obscured and reads green rather than
brown. These produce the large blue regions in the two lowest-scoring images (0.33 and 0.45)
and represent the model's real weakness: it keys on visible water, so water it cannot see
directly is water it cannot segment.

A third factor is **label ambiguity**. Some ground-truth masks mark visually green, wet
fields as water, which is defensible for flood assessment but inconsistent with the
appearance-based cue the model learned. Part of the residual error is annotation variance
rather than model error.

---

## Application

A Streamlit web app for single-image inference:

- Upload a JPG or PNG aerial image
- Adjustable detection threshold and overlay opacity
- Three views: outlined overlay, confidence heatmap, binary mask
- Water coverage percentage, mean confidence, and count of distinct regions
- Downloadable binary mask as PNG
- Explicit warnings when nothing is detected or when the entire frame is flagged

Detections below the threshold render as transparent rather than as a coloured
low-confidence region, so a scene with no water produces a visibly empty result instead of
a full-frame colour map.

---

## Repository

```
FloodMapping/
├── app.py                      # Streamlit inference app
├── evaluate.py                 # Test-set evaluation, baseline comparison, failure analysis
├── floodmapdl.py               # Training script (originally a Colab notebook)
├── requirements.txt            # Pinned dependencies (CPU PyTorch)
├── best_unet.pth               # Trained weights (Git LFS)
├── Eval/                       # Evaluation outputs
│   ├── summary.json            #   Headline metrics
│   ├── per_image_metrics.json  #   Per-image IoU / Dice / precision / recall
│   ├── baseline_metrics.json   #   Baseline params and per-image scores
│   ├── iou_distribution.png
│   ├── threshold_sweep.png
│   └── failure_gallery.png
└── images/                     # Qualitative predictions and training curves
```

`best_unet.pth` is tracked with Git LFS. Clone with `git lfs install` first, or the file
arrives as a text pointer rather than weights.

### Running locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Reproducing the evaluation

Edit the paths at the top of `evaluate.py`, then:

```bash
python evaluate.py
```

This does not retrain. It runs one forward pass over the test set, caches per-image
probability maps, and writes metrics and figures to `eval_out/`. Roughly 10–15 minutes on
CPU for 341 images.

---

## Where this could go next

1. **Audit the split for near-duplicates.** Perceptual-hash the train and test sets and
   re-split by source event if leakage is found. This determines whether 0.881 is real.
2. **Retrain on FloodNet** — drone imagery from Hurricane Harvey with classes that separate
   *flooded* buildings and roads from *non-flooded* ones. This is the change that would make
   the model a genuine flood detector rather than a water segmenter.
3. **Move to SAR (Sen1Floods11)** — Sentinel-1 radar penetrates cloud cover. Optical imagery
   fails precisely during the storms that cause floods, and SAR data is georeferenced, which
   unlocks area measurement.

---

## Stack

Python · PyTorch · segmentation-models-pytorch · Albumentations · OpenCV · NumPy ·
Matplotlib · Streamlit
