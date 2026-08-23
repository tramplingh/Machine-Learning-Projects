"""
evaluate.py -- honest evaluation for the flood mapping project.


  1. Corrected per-image metrics (IoU, Dice, precision, recall)
  2. The old buggy IoU alongside it, so you can state the exact inflation
  3. A failure gallery: the 10 worst test images
  4. A distribution plot of per-image IoU
  5. A classical HSV baseline, tuned on TRAIN, scored on TEST
  6. A threshold sweep

Outputs go to OUT_DIR. Everything is saved to disk so you can re-plot
without re-running the model.
"""

import os
import glob
import random
import json

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIG -- edit these three paths
# ----------------------------------------------------------------------
IMG_DIR = r"C:/Users/Anish Dev Edward/Downloads/archive/Masks"
MASK_DIR = r"C:/Users/Anish Dev Edward/Downloads/archive/Masks"
CKPT = r"C:/Users/Anish Dev Edward/Downloads/FloodMapping/best_unet.pth"

OUT_DIR = r"C:/Users/Anish Dev Edward/Downloads/FloodMapping/Eval"
IMG_SIZE = 256
BATCH_SIZE = 8
THRESHOLD = 0.5
N_WORST = 10
N_BASELINE_TUNE = 100  # how many TRAIN images to tune the HSV baseline on

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "probs"), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Reproducibility (the original script only seeded `random`)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ----------------------------------------------------------------------
# STAGE 0 -- rebuild the split exactly as training did
# ----------------------------------------------------------------------
# IMPORTANT: this only reproduces your original split if the contents of
# IMG_DIR / MASK_DIR are unchanged since you trained. Check the printed
# counts against what your training run printed.

images = sorted([os.path.basename(p) for p in glob.glob(os.path.join(IMG_DIR, "*.png"))])
paired = [f for f in images if os.path.exists(os.path.join(MASK_DIR, f))]

random.seed(42)  # re-seed immediately before the shuffle, as in training
random.shuffle(paired)

n = len(paired)
n_train, n_val = int(0.8 * n), int(0.1 * n)
splits = {
    "train": paired[:n_train],
    "val": paired[n_train:n_train + n_val],
    "test": paired[n_train + n_val:],
}

print("\n" + "=" * 60)
print("SPLIT CHECK -- compare these to your original training output")
print("=" * 60)
print(f"Total paired files: {n}")
print({k: len(v) for k, v in splits.items()})
print("If these numbers differ from your training run, the split has")
print("shifted and some test images were seen during training.")
print("=" * 60 + "\n")

# The original SegDataset listed files with sorted(os.listdir(...)),
# so sort within each split to match that ordering.
test_files = sorted(splits["test"])
train_files = sorted(splits["train"])


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
val_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class EvalDataset(Dataset):
    """Reads directly from the source dirs -- no file copying, no full decode at init."""

    def __init__(self, img_dir, mask_dir, files, tfm):
        self.img_dir, self.mask_dir, self.tfm = img_dir, mask_dir, tfm
        self.files = [f for f in files if os.path.exists(os.path.join(mask_dir, f))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, f))
        mask = cv2.imread(os.path.join(self.mask_dir, f), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise RuntimeError(f"Could not read {f}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype("float32")
        aug = self.tfm(image=img, mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0), f


test_ds = EvalDataset(IMG_DIR, MASK_DIR, test_files, val_tfms)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Test set: {len(test_ds)} images")


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
EPS = 1e-6


def binary_metrics(pred_bin, gt_bin):
    """Per-image IoU / Dice / precision / recall on 2D boolean arrays."""
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return {
        "iou": float((tp + EPS) / (union + EPS)),
        "dice": float((2 * tp + EPS) / (2 * tp + fp + fn + EPS)),
        "precision": float((tp + EPS) / (tp + fp + EPS)),
        "recall": float((tp + EPS) / (tp + fn + EPS)),
    }


def iou_buggy(outputs, labels, threshold=0.5):
    """The ORIGINAL function. Kept only to quantify how much it inflated the score."""
    preds = (torch.sigmoid(outputs) > threshold).int()
    labels = labels.int()
    intersection = (preds & labels).float().sum((1, 2))   # <-- wrong dims
    union = (preds | labels).float().sum((1, 2))
    return ((intersection + EPS) / (union + EPS)).mean()


def iou_fixed(outputs, labels, threshold=0.5):
    """The corrected function: sums over channel + height + width."""
    preds = (torch.sigmoid(outputs) > threshold).int()
    labels = labels.int()
    intersection = (preds & labels).float().sum((1, 2, 3))
    union = (preds | labels).float().sum((1, 2, 3))
    return ((intersection + EPS) / (union + EPS)).mean()


# ----------------------------------------------------------------------
# STAGE 1 -- one forward pass over the test set
# ----------------------------------------------------------------------
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
state = torch.load(CKPT, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.to(DEVICE).eval()

records = []
buggy_sum, fixed_sum, n_batches = 0.0, 0.0, 0

with torch.no_grad():
    for x, y, fnames in test_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)

        buggy_sum += iou_buggy(logits, y).item()
        fixed_sum += iou_fixed(logits, y).item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]     # (B, H, W)
        gts = y.cpu().numpy()[:, 0]

        for p, g, f in zip(probs, gts, fnames):
            # save probabilities as uint8 (4x smaller, plenty for a threshold sweep)
            np.save(os.path.join(OUT_DIR, "probs", f + ".npy"), (p * 255).astype(np.uint8))
            m = binary_metrics(p > THRESHOLD, g > 0.5)
            m["file"] = f
            m["gt_coverage"] = float((g > 0.5).mean())
            m["pred_coverage"] = float((p > THRESHOLD).mean())
            records.append(m)

per_image_iou = np.array([r["iou"] for r in records])

print("\n" + "=" * 60)
print("MODEL RESULTS ON TEST SET")
print("=" * 60)
print(f"Old (buggy) IoU     : {buggy_sum / n_batches:.4f}   <- what you reported")
print(f"Corrected IoU       : {fixed_sum / n_batches:.4f}   <- the real number")
print(f"  inflation         : {buggy_sum / n_batches - fixed_sum / n_batches:+.4f}")
print("-" * 60)
print(f"Mean IoU (per-image): {per_image_iou.mean():.4f}")
print(f"Median IoU          : {np.median(per_image_iou):.4f}")
print(f"Std dev             : {per_image_iou.std():.4f}")
print(f"Mean Dice           : {np.mean([r['dice'] for r in records]):.4f}")
print(f"Mean Precision      : {np.mean([r['precision'] for r in records]):.4f}")
print(f"Mean Recall         : {np.mean([r['recall'] for r in records]):.4f}")
print(f"Images below 0.5 IoU: {(per_image_iou < 0.5).sum()} / {len(per_image_iou)}")
print("=" * 60)

with open(os.path.join(OUT_DIR, "per_image_metrics.json"), "w") as fh:
    json.dump(records, fh, indent=2)


# ----------------------------------------------------------------------
# STAGE 2 -- classical HSV baseline (tuned on TRAIN, scored on TEST)
# ----------------------------------------------------------------------
def hsv_predict(img_rgb, params):
    """Threshold muddy-water colours in HSV, then clean up with morphology."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lo = np.array([params["h_lo"], params["s_lo"], params["v_lo"]], np.uint8)
    hi = np.array([params["h_hi"], params["s_hi"], params["v_hi"]], np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask > 0


def load_resized(fname):
    img = cv2.cvtColor(cv2.imread(os.path.join(IMG_DIR, fname)), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(MASK_DIR, fname), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return img, mask > 127


print("\nTuning HSV baseline on TRAIN images (never on test)...")
tune_files = train_files[:N_BASELINE_TUNE]
tune_data = [load_resized(f) for f in tune_files]

grid = []
for h_lo in (0, 5, 10):
    for h_hi in (20, 25, 30, 35):
        for s_lo in (10, 30):
            for s_hi in (90, 130, 170, 210):
                for v_lo in (40, 80):
                    grid.append({"h_lo": h_lo, "h_hi": h_hi, "s_lo": s_lo,
                                 "s_hi": s_hi, "v_lo": v_lo, "v_hi": 255})

best_params, best_train_iou = None, -1.0
for params in grid:
    ious = [binary_metrics(hsv_predict(im, params), gt)["iou"] for im, gt in tune_data]
    score = float(np.mean(ious))
    if score > best_train_iou:
        best_train_iou, best_params = score, params

print(f"Best baseline params : {best_params}")
print(f"Baseline IoU on train: {best_train_iou:.4f}")

baseline_records = []
for f in test_files:
    img, gt = load_resized(f)
    m = binary_metrics(hsv_predict(img, best_params), gt)
    m["file"] = f
    baseline_records.append(m)

baseline_iou = np.array([r["iou"] for r in baseline_records])

print("\n" + "=" * 60)
print("MODEL vs BASELINE (test set)")
print("=" * 60)
print(f"{'':<14}{'IoU':>8}{'Dice':>8}{'Prec':>8}{'Recall':>8}")
for name, recs in (("U-Net", records), ("HSV baseline", baseline_records)):
    print(f"{name:<14}"
          f"{np.mean([r['iou'] for r in recs]):>8.4f}"
          f"{np.mean([r['dice'] for r in recs]):>8.4f}"
          f"{np.mean([r['precision'] for r in recs]):>8.4f}"
          f"{np.mean([r['recall'] for r in recs]):>8.4f}")
print("-" * 60)
print(f"U-Net wins on {(per_image_iou > baseline_iou).sum()} / {len(test_files)} images")
print("=" * 60)

with open(os.path.join(OUT_DIR, "baseline_metrics.json"), "w") as fh:
    json.dump({"params": best_params, "train_iou": best_train_iou,
               "per_image": baseline_records}, fh, indent=2)


# ----------------------------------------------------------------------
# STAGE 3 -- plots
# ----------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 140, "font.size": 10,
                     "axes.titlesize": 12, "axes.titleweight": "bold"})

# --- 3a. Failure gallery: the N worst test images ---
worst = sorted(records, key=lambda r: r["iou"])[:N_WORST]

fig, axes = plt.subplots(N_WORST, 4, figsize=(13, 3.1 * N_WORST))
for row, rec in enumerate(worst):
    f = rec["file"]
    img, gt = load_resized(f)
    prob = np.load(os.path.join(OUT_DIR, "probs", f + ".npy")).astype(np.float32) / 255.0
    pred = prob > THRESHOLD

    # error map: green = correct flood, red = false positive, blue = missed flood
    err = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    err[np.logical_and(pred, gt)] = (60, 180, 75)
    err[np.logical_and(pred, ~gt)] = (230, 25, 75)
    err[np.logical_and(~pred, gt)] = (60, 90, 230)

    for col, (data, title, kw) in enumerate([
        (img, "Input", {}),
        (gt, "Ground truth", {"cmap": "gray"}),
        (pred, "Prediction", {"cmap": "gray"}),
        (err, "Errors", {}),
    ]):
        ax = axes[row, col]
        ax.imshow(data, **kw)
        ax.axis("off")
        if row == 0:
            ax.set_title(title)
    axes[row, 0].set_ylabel(f"IoU {rec['iou']:.2f}")
    axes[row, 0].axis("on")
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])

fig.suptitle("Worst test predictions  |  green = hit, red = false alarm, blue = missed flood",
             fontsize=13, fontweight="bold", y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "failure_gallery.png"), bbox_inches="tight")
plt.close()

# --- 3b. Per-image IoU distribution ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(per_image_iou, bins=25, color="#4C72B0", edgecolor="white", alpha=0.85,
        label="U-Net")
ax.hist(baseline_iou, bins=25, color="#DD8452", edgecolor="white", alpha=0.55,
        label="HSV baseline")
ax.axvline(per_image_iou.mean(), color="#4C72B0", ls="--", lw=1.6,
           label=f"U-Net mean {per_image_iou.mean():.3f}")
ax.axvline(baseline_iou.mean(), color="#DD8452", ls="--", lw=1.6,
           label=f"Baseline mean {baseline_iou.mean():.3f}")
ax.set_title(f"Per-image IoU is spread out, not uniform (n={len(per_image_iou)} test images)")
ax.set_xlabel("IoU")
ax.set_ylabel("Number of images")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "iou_distribution.png"), bbox_inches="tight")
plt.close()

# --- 3c. Threshold sweep (reuses saved probability maps -- no model needed) ---
thresholds = np.arange(0.05, 0.96, 0.05)
sweep = {"iou": [], "precision": [], "recall": []}
gts_cache = {f: load_resized(f)[1] for f in test_files}

for t in thresholds:
    ms = []
    for f in test_files:
        prob = np.load(os.path.join(OUT_DIR, "probs", f + ".npy")).astype(np.float32) / 255.0
        ms.append(binary_metrics(prob > t, gts_cache[f]))
    for k in sweep:
        sweep[k].append(float(np.mean([m[k] for m in ms])))

best_t = thresholds[int(np.argmax(sweep["iou"]))]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, sweep["iou"], lw=2, color="#4C72B0", label="IoU")
ax.plot(thresholds, sweep["precision"], lw=2, ls="--", color="#DD8452", label="Precision")
ax.plot(thresholds, sweep["recall"], lw=2, ls=":", color="#55A868", label="Recall")
ax.axvline(best_t, color="grey", lw=1.2, label=f"Best IoU @ {best_t:.2f}")
ax.set_title(f"0.5 is not necessarily the right threshold (best IoU at {best_t:.2f})")
ax.set_xlabel("Probability threshold")
ax.set_ylabel("Score")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "threshold_sweep.png"), bbox_inches="tight")
plt.close()

with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
    json.dump({
        "n_test": len(test_files),
        "buggy_iou": buggy_sum / n_batches,
        "corrected_iou": fixed_sum / n_batches,
        "mean_iou": float(per_image_iou.mean()),
        "median_iou": float(np.median(per_image_iou)),
        "std_iou": float(per_image_iou.std()),
        "mean_dice": float(np.mean([r["dice"] for r in records])),
        "mean_precision": float(np.mean([r["precision"] for r in records])),
        "mean_recall": float(np.mean([r["recall"] for r in records])),
        "baseline_mean_iou": float(baseline_iou.mean()),
        "baseline_params": best_params,
        "best_threshold": float(best_t),
    }, fh, indent=2)

print(f"\nDone. Figures and JSON written to {OUT_DIR}")
print("  failure_gallery.png")
print("  iou_distribution.png")
print("  threshold_sweep.png")
print("  summary.json / per_image_metrics.json / baseline_metrics.json")
