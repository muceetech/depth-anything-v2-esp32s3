
import os
import numpy as np
import torch

from dataset_distill_128x96 import NYUDistillation128x96
from student_boundary_274k_128x96 import DepthStudentBoundary274K128x96


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = r"D:\research\student48\data\nyu256"

CHECKPOINT = (
    r"D:\research\student48\scripts"
    r"\depth_anything_student\checkpoints"
    r"\student_boundary_274k_128x96_best.pth"
)

SAMPLE_INDICES = list(range(10))


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BOUNDARY-PRESERVING V1 — 128x96")
print("10-SAMPLE RGB VS GRAYSCALE EVALUATION")
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
print("Samples     :", SAMPLE_INDICES)


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

print("\nCheckpoint epoch:", checkpoint["epoch"])
print("Checkpoint RMSE :", checkpoint["val_rmse"])


# ============================================================
# HELPERS
# ============================================================

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def prepare_rgb(rgb):
    """
    Convert dataset RGB to C,H,W float tensor in [0,1].

    The existing visualization showed RGB as H,W,3, so handle both
    HWC and CHW safely.
    """
    if not torch.is_tensor(rgb):
        rgb = torch.from_numpy(np.asarray(rgb))

    rgb = rgb.float()

    if rgb.ndim != 3:
        raise ValueError(f"Unexpected RGB shape: {tuple(rgb.shape)}")

    # H,W,3 -> 3,H,W
    if rgb.shape[-1] == 3:
        rgb = rgb.permute(2, 0, 1)

    elif rgb.shape[0] == 3:
        pass

    else:
        raise ValueError(f"Cannot identify RGB channels: {tuple(rgb.shape)}")

    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    return rgb


# ============================================================
# EVALUATION
# ============================================================

results = []

print("\n" + "=" * 70)
print("PER-SAMPLE RESULTS")
print("=" * 70)
print(
    "Sample | RGB RMSE | Gray RMSE | RGB MAE | Gray MAE | "
    "RGB r | Gray r"
)
print("-" * 70)


for sample_index in SAMPLE_INDICES:

    sample = dataset[sample_index]

    # IMPORTANT:
    # These are the actual keys used by your dataset/original script.
    rgb = sample["rgb"]
    gt = sample["depth"]
    teacher = sample["da_teacher"]
    valid_mask = sample["valid_mask"]

    # --------------------------------------------------------
    # Prepare RGB
    # --------------------------------------------------------

    rgb_chw = prepare_rgb(rgb)

    # --------------------------------------------------------
    # Create grayscale -> R=G=B
    # This matches the ESP32 grayscale input concept.
    # --------------------------------------------------------

    gray = (
        0.299 * rgb_chw[0] +
        0.587 * rgb_chw[1] +
        0.114 * rgb_chw[2]
    )

    gray_chw = gray.unsqueeze(0).repeat(3, 1, 1)

    rgb_input = rgb_chw.unsqueeze(0).to(DEVICE)
    gray_input = gray_chw.unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        pred_rgb = model(rgb_input)[0, 0]
        pred_gray = model(gray_input)[0, 0]

    pred_rgb = pred_rgb.cpu().numpy()
    pred_gray = pred_gray.cpu().numpy()

    # --------------------------------------------------------
    # GT / mask
    # --------------------------------------------------------

    gt_np = to_numpy(gt).squeeze()
    mask_np = to_numpy(valid_mask).squeeze().astype(bool)

    if gt_np.shape != pred_rgb.shape:
        raise ValueError(
            f"Shape mismatch sample {sample_index}: "
            f"GT={gt_np.shape}, prediction={pred_rgb.shape}"
        )

    # Include only finite valid GT pixels.
    valid = (
        mask_np
        & np.isfinite(gt_np)
        & np.isfinite(pred_rgb)
        & np.isfinite(pred_gray)
    )

    if valid.sum() == 0:
        print(f"{sample_index:04d} | NO VALID PIXELS")
        continue

    gt_v = gt_np[valid]
    rgb_v = pred_rgb[valid]
    gray_v = pred_gray[valid]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    rmse_rgb = np.sqrt(np.mean((rgb_v - gt_v) ** 2))
    mae_rgb = np.mean(np.abs(rgb_v - gt_v))

    rmse_gray = np.sqrt(np.mean((gray_v - gt_v) ** 2))
    mae_gray = np.mean(np.abs(gray_v - gt_v))

    # Correlation can become NaN for a constant array.
    if np.std(rgb_v) > 1e-12 and np.std(gt_v) > 1e-12:
        corr_rgb = np.corrcoef(rgb_v, gt_v)[0, 1]
    else:
        corr_rgb = np.nan

    if np.std(gray_v) > 1e-12 and np.std(gt_v) > 1e-12:
        corr_gray = np.corrcoef(gray_v, gt_v)[0, 1]
    else:
        corr_gray = np.nan

    results.append([
        sample_index,
        rmse_rgb,
        mae_rgb,
        corr_rgb,
        rmse_gray,
        mae_gray,
        corr_gray
    ])

    print(
        f"{sample_index:04d} | "
        f"{rmse_rgb:.4f}   | "
        f"{rmse_gray:.4f}   | "
        f"{mae_rgb:.4f}  | "
        f"{mae_gray:.4f}  | "
        f"{corr_rgb:.4f} | "
        f"{corr_gray:.4f}"
    )


# ============================================================
# SUMMARY
# ============================================================

if not results:

    print("\nNo valid samples were evaluated.")
    raise SystemExit


arr = np.asarray(results, dtype=np.float64)

rgb_rmse = arr[:, 1]
rgb_mae = arr[:, 2]
rgb_corr = arr[:, 3]

gray_rmse = arr[:, 4]
gray_mae = arr[:, 5]
gray_corr = arr[:, 6]


print("\n" + "=" * 70)
print("AVERAGE OVER 10 TEST SAMPLES")
print("=" * 70)

print(f"Samples evaluated : {len(results)}")
print()
print(f"RGB RMSE          : {np.nanmean(rgb_rmse):.6f}")
print(f"RGB MAE           : {np.nanmean(rgb_mae):.6f}")
print(f"RGB correlation   : {np.nanmean(rgb_corr):.6f}")
print()
print(f"Gray RMSE         : {np.nanmean(gray_rmse):.6f}")
print(f"Gray MAE          : {np.nanmean(gray_mae):.6f}")
print(f"Gray correlation  : {np.nanmean(gray_corr):.6f}")


print("\n" + "=" * 70)
print("RGB vs GRAYSCALE DIFFERENCE")
print("=" * 70)

rmse_diff = np.nanmean(gray_rmse) - np.nanmean(rgb_rmse)
mae_diff = np.nanmean(gray_mae) - np.nanmean(rgb_mae)
corr_diff = np.nanmean(gray_corr) - np.nanmean(rgb_corr)

print(f"Gray - RGB RMSE   : {rmse_diff:+.6f}")
print(f"Gray - RGB MAE    : {mae_diff:+.6f}")
print(f"Gray - RGB corr.  : {corr_diff:+.6f}")


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    os.getcwd(),
    "student_v1_128x96_rgb_vs_grayscale_10samples.csv"
)

np.savetxt(
    csv_path,
    arr,
    delimiter=",",
    header=(
        "sample,rgb_rmse,rgb_mae,rgb_corr,"
        "gray_rmse,gray_mae,gray_corr"
    ),
    comments=""
)

print("\nSaved:", csv_path)
print("=" * 70)
