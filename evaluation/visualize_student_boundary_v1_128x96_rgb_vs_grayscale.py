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
# IMPORTANT:
# The ESP32-S3 test currently uses:
#     grayscale -> R=G=B
#
# The original Python visualization below used the normal RGB image.
# We therefore run BOTH versions so we can determine whether the
# ESP32 grayscale input is responsible for the poor spatial match.
# ============================================================

rgb_input = (
    rgb
    .unsqueeze(0)
    .to(DEVICE)
)

# Convert the Python RGB tensor to grayscale, then replicate it
# into 3 channels exactly like the ESP32-S3 input path.
gray = (
    0.299 * rgb[0] +
    0.587 * rgb[1] +
    0.114 * rgb[2]
)

gray_rgb = (
    gray.unsqueeze(0)
    .repeat(3, 1, 1)
)

gray_input = (
    gray_rgb
    .unsqueeze(0)
    .to(DEVICE)
)

with torch.no_grad():

    prediction_rgb = model(
        rgb_input
    )

    prediction_gray = model(
        gray_input
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

prediction_rgb = (
    prediction_rgb[0, 0]
    .cpu()
    .numpy()
)

prediction_gray = (
    prediction_gray[0, 0]
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
print("V1 Student RGB   :", prediction_rgb.shape)
print("V1 Student Gray  :", prediction_gray.shape)
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

assert prediction_rgb.shape == (
    96,
    128
)

assert prediction_gray.shape == (
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
# ERROR + METRICS
# ============================================================

error_rgb = np.abs(
    prediction_rgb - gt
) * valid_mask

error_gray = np.abs(
    prediction_gray - gt
) * valid_mask

sample_mae_rgb = np.mean(
    error_rgb[valid]
)

sample_rmse_rgb = np.sqrt(
    np.mean(
        (prediction_rgb[valid] - gt[valid]) ** 2
    )
)

sample_mae_gray = np.mean(
    error_gray[valid]
)

sample_rmse_gray = np.sqrt(
    np.mean(
        (prediction_gray[valid] - gt[valid]) ** 2
    )
)

# Spatial Pearson correlation is useful for checking whether the
# predicted structures follow the same spatial pattern as GT/teacher.
corr_rgb_gt = np.corrcoef(
    prediction_rgb[valid].ravel(),
    gt[valid].ravel()
)[0, 1]

corr_gray_gt = np.corrcoef(
    prediction_gray[valid].ravel(),
    gt[valid].ravel()
)[0, 1]

corr_rgb_teacher = np.corrcoef(
    prediction_rgb[valid].ravel(),
    teacher[valid].ravel()
)[0, 1]

corr_gray_teacher = np.corrcoef(
    prediction_gray[valid].ravel(),
    teacher[valid].ravel()
)[0, 1]


print("\n" + "=" * 70)
print("SAMPLE METRICS — RGB VS GRAYSCALE INPUT")
print("=" * 70)

print(f"RGB input       RMSE: {sample_rmse_rgb:.6f}")
print(f"RGB input       MAE : {sample_mae_rgb:.6f}")
print(f"RGB vs GT       r   : {corr_rgb_gt:.6f}")
print(f"RGB vs Teacher  r   : {corr_rgb_teacher:.6f}")

print()

print(f"Gray input      RMSE: {sample_rmse_gray:.6f}")
print(f"Gray input      MAE : {sample_mae_gray:.6f}")
print(f"Gray vs GT      r   : {corr_gray_gt:.6f}")
print(f"Gray vs Teacher r   : {corr_gray_teacher:.6f}")


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


print("\nBoundary V1 Student — RGB:")
print(" Min :", prediction_rgb[valid].min())
print(" Max :", prediction_rgb[valid].max())
print(" Mean:", prediction_rgb[valid].mean())

print("\nBoundary V1 Student — Grayscale:")
print(" Min :", prediction_gray[valid].min())
print(" Max :", prediction_gray[valid].max())
print(" Mean:", prediction_gray[valid].mean())


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
    prediction_rgb[valid].max(),
    prediction_gray[valid].max()
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
# ROW 1 — MAIN COMPARISON
# ============================================================

axes[0, 0].imshow(
    rgb_display
)
axes[0, 0].set_title("Original RGB")
axes[0, 0].axis("off")

axes[0, 1].imshow(
    gray,
    cmap="gray",
    vmin=0,
    vmax=1
)
axes[0, 1].set_title("Grayscale Input → R=G=B")
axes[0, 1].axis("off")

im_depth = axes[0, 2].imshow(
    prediction_rgb,
    cmap="plasma",
    vmin=depth_min,
    vmax=depth_max
)
axes[0, 2].set_title(
    f"Student — RGB | RMSE {sample_rmse_rgb:.4f}"
)
axes[0, 2].axis("off")

axes[0, 3].imshow(
    prediction_gray,
    cmap="plasma",
    vmin=depth_min,
    vmax=depth_max
)
axes[0, 3].set_title(
    f"Student — Gray | RMSE {sample_rmse_gray:.4f}"
)
axes[0, 3].axis("off")


# ============================================================
# ROW 2
# ============================================================

axes[1, 0].imshow(
    gt,
    cmap="plasma",
    vmin=depth_min,
    vmax=depth_max
)
axes[1, 0].set_title("NYU Ground Truth")
axes[1, 0].axis("off")

axes[1, 1].imshow(
    teacher,
    cmap="plasma",
    vmin=depth_min,
    vmax=depth_max
)
axes[1, 1].set_title("Depth Anything V2 Teacher")
axes[1, 1].axis("off")

im_error = axes[1, 2].imshow(
    error_rgb,
    cmap="inferno"
)
axes[1, 2].set_title(
    f"RGB Absolute Error | MAE {sample_mae_rgb:.4f}"
)
axes[1, 2].axis("off")

axes[1, 3].imshow(
    error_gray,
    cmap="inferno"
)
axes[1, 3].set_title(
    f"Gray Absolute Error | MAE {sample_mae_gray:.4f}"
)
axes[1, 3].axis("off")

# ============================================================
# COLORBAR — DEPTH
# ============================================================

fig.colorbar(
    im_depth,
    ax=[
        axes[0, 2],
        axes[0, 3],
        axes[1, 0],
        axes[1, 1]
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
        axes[1, 2],
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
        f"Boundary V1 — RGB vs Grayscale — 128×96"
    ),
    fontsize=16
)


fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

figure_file = os.path.join(
    OUTPUT_DIR,
    f"{original_index:04d}_boundary_v1_rgb_vs_grayscale.png"
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

print("RGB Sample RMSE:", f"{sample_rmse_rgb:.6f}")
print("RGB Sample MAE :", f"{sample_mae_rgb:.6f}")
print("Gray Sample RMSE:", f"{sample_rmse_gray:.6f}")
print("Gray Sample MAE :", f"{sample_mae_gray:.6f}")
print("RGB vs GT r    :", f"{corr_rgb_gt:.6f}")
print("Gray vs GT r   :", f"{corr_gray_gt:.6f}")
