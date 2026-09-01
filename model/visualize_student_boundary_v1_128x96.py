import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset_distill_128x96 import NYUDistillation128x96
from student_boundary_274k_128x96 import DepthStudentBoundary274K128x96


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = (
    r"D:\research\student48\data\nyu256"
)

CHECKPOINT = (
    r"D:\research\student48\scripts"
    r"\depth_anything_student\checkpoints"
    r"\student_boundary_274k_128x96_best.pth"
)

OUTPUT_DIR = (
    r"D:\research\student48\data\nyu256"
    r"\visualizations_boundary_v1_128x96"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CHANGE THIS TO VISUALIZE DIFFERENT TEST SAMPLES
# ============================================================

SAMPLE_INDEX = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BOUNDARY-PRESERVING V1 — 128×96 VISUALIZATION")
print("=" * 70)

print("Device    :", DEVICE)
print("Checkpoint:", CHECKPOINT)


# ============================================================
# DATASET
# ============================================================

dataset = NYUDistillation128x96(
    DATASET_ROOT,
    split="test"
)

print("Test samples:", len(dataset))
print("Sample index:", SAMPLE_INDEX)


# ============================================================
# MODEL
# ============================================================

model = DepthStudentBoundary274K128x96()

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


print("\nCheckpoint epoch:",
      checkpoint["epoch"])

print("Checkpoint RMSE:",
      checkpoint["val_rmse"])


# ============================================================
# LOAD SAMPLE
# ============================================================

sample = dataset[SAMPLE_INDEX]

rgb = sample["rgb"]
gt = sample["depth"]
teacher = sample["da_teacher"]
valid_mask = sample["valid_mask"]

original_index = sample["index"]


# ============================================================
# INFERENCE
# ============================================================

input_tensor = (
    rgb
    .unsqueeze(0)
    .to(DEVICE)
)


with torch.no_grad():

    prediction = model(
        input_tensor
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

prediction = (
    prediction[0, 0]
    .cpu()
    .numpy()
)

gt = (
    gt[0]
    .cpu()
    .numpy()
)

teacher = (
    teacher[0]
    .cpu()
    .numpy()
)

valid_mask = (
    valid_mask[0]
    .cpu()
    .numpy()
)

rgb_display = (
    rgb
    .permute(1, 2, 0)
    .cpu()
    .numpy()
)


# ============================================================
# SHAPE CHECK
# ============================================================

print("\n" + "=" * 70)
print("SHAPE VERIFICATION")
print("=" * 70)

print("RGB       :", rgb_display.shape)
print("GT        :", gt.shape)
print("Teacher   :", teacher.shape)
print("V1 Student:", prediction.shape)
print("Valid mask:", valid_mask.shape)


assert rgb_display.shape == (
    96,
    128,
    3
)

assert gt.shape == (
    96,
    128
)

assert teacher.shape == (
    96,
    128
)

assert prediction.shape == (
    96,
    128
)


# ============================================================
# VALID REGION
# ============================================================

valid = (
    valid_mask > 0.5
)


# ============================================================
# ERROR
# ============================================================

error = np.abs(
    prediction - gt
)

error = error * valid_mask


# ============================================================
# METRICS FOR THIS SAMPLE
# ============================================================

sample_mae = np.mean(
    error[valid]
)

sample_rmse = np.sqrt(
    np.mean(
        (prediction[valid] - gt[valid]) ** 2
    )
)


print("\n" + "=" * 70)
print("SAMPLE METRICS")
print("=" * 70)

print(
    f"RMSE : {sample_rmse:.6f}"
)

print(
    f"MAE  : {sample_mae:.6f}"
)


# ============================================================
# DEPTH STATISTICS
# ============================================================

print("\nGT:")
print(
    " Min :",
    gt[valid].min()
)
print(
    " Max :",
    gt[valid].max()
)
print(
    " Mean:",
    gt[valid].mean()
)


print("\nTeacher:")
print(
    " Min :",
    teacher[valid].min()
)
print(
    " Max :",
    teacher[valid].max()
)
print(
    " Mean:",
    teacher[valid].mean()
)


print("\nBoundary V1 Student:")
print(
    " Min :",
    prediction[valid].min()
)
print(
    " Max :",
    prediction[valid].max()
)
print(
    " Mean:",
    prediction[valid].mean()
)


# ============================================================
# COMMON DEPTH SCALE
#
# IMPORTANT:
# All depth maps use exactly the same color scale.
# ============================================================

depth_min = min(
    gt[valid].min(),
    teacher[valid].min(),
    prediction[valid].min()
)

depth_max = max(
    gt[valid].max(),
    teacher[valid].max(),
    prediction[valid].max()
)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(16, 8)
)


# ============================================================
# ROW 1
# ============================================================


# RGB

axes[0, 0].imshow(
    rgb_display
)

axes[0, 0].set_title(
    "RGB"
)

axes[0, 0].axis(
    "off"
)


# GT

im_depth = axes[0, 1].imshow(
    gt,
    cmap="plasma",
    vmin=depth_min,
    vmax=depth_max
)

axes[0, 1].set_title(
    "NYU Ground Truth"
)

axes[0, 1].axis(
    "off"
)


# Teacher

axes[0, 2].imshow(
    teacher,
    cmap="plasma",
    vmin=depth_min,
    vmax=depth_max
)

axes[0, 2].set_title(
    "Depth Anything V2 Teacher"
)

axes[0, 2].axis(
    "off"
)


# V1 Student

axes[0, 3].imshow(
    prediction,
    cmap="plasma",
    vmin=depth_min,
    vmax=depth_max
)

axes[0, 3].set_title(
    f"Boundary V1 — RMSE {sample_rmse:.4f}"
)

axes[0, 3].axis(
    "off"
)


# ============================================================
# ROW 2
# ============================================================


# GT detail

axes[1, 0].imshow(
    gt,
    cmap="plasma",
    vmin=depth_min,
    vmax=depth_max
)

axes[1, 0].set_title(
    "GT — Detail"
)

axes[1, 0].axis(
    "off"
)


# Teacher detail

axes[1, 1].imshow(
    teacher,
    cmap="plasma",
    vmin=depth_min,
    vmax=depth_max
)

axes[1, 1].set_title(
    "Teacher — Detail"
)

axes[1, 1].axis(
    "off"
)


# V1 detail

axes[1, 2].imshow(
    prediction,
    cmap="plasma",
    vmin=depth_min,
    vmax=depth_max
)

axes[1, 2].set_title(
    "Boundary V1 — Detail"
)

axes[1, 2].axis(
    "off"
)


# Error

im_error = axes[1, 3].imshow(
    error,
    cmap="inferno"
)

axes[1, 3].set_title(
    f"V1 Absolute Error — MAE {sample_mae:.4f}"
)

axes[1, 3].axis(
    "off"
)


# ============================================================
# COLORBAR — DEPTH
# ============================================================

fig.colorbar(
    im_depth,
    ax=[
        axes[0, 1],
        axes[0, 2],
        axes[0, 3],
        axes[1, 0],
        axes[1, 1],
        axes[1, 2]
    ],
    fraction=0.02,
    pad=0.02,
    label="Normalized inverse depth"
)


# ============================================================
# COLORBAR — ERROR
# ============================================================

fig.colorbar(
    im_error,
    ax=[
        axes[1, 3]
    ],
    fraction=0.046,
    pad=0.04,
    label="Absolute error"
)


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    (
        f"NYU Sample {original_index:04d} — "
        f"Boundary-Preserving V1 — 128×96"
    ),
    fontsize=16
)


fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

figure_file = os.path.join(
    OUTPUT_DIR,
    f"{original_index:04d}_boundary_v1_comparison.png"
)

plt.savefig(
    figure_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print(
    "Figure:",
    figure_file
)

print(
    "Sample RMSE:",
    f"{sample_rmse:.6f}"
)

print(
    "Sample MAE:",
    f"{sample_mae:.6f}"
)
