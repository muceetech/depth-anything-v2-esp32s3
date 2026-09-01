import os
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_distill_128x96 import NYUDistillation128x96
from student_boundary_274k_128x96 import (
    DepthStudentBoundary274K128x96,
    initialize_boundary_fusion_as_identity
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"D:\research\student48\data\nyu256"

CHECKPOINT_DIR = (
    r"D:\research\student48\scripts"
    r"\depth_anything_student\checkpoints"
)

BASELINE_CHECKPOINT = (
    r"D:\research\student48\scripts"
    r"\depth_anything_student\checkpoints"
    r"\student269k_128x96_best.pth"
)

OUTPUT_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "student_boundary_274k_128x96_best.pth"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 50
BATCH_SIZE = 16
NUM_WORKERS = 0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Keep exactly the same loss recipe as the 269K baseline.
LAMBDA_GT = 1.0
LAMBDA_TEACHER = 1.0
LAMBDA_GRADIENT = 0.25

SEED = 42


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
print("BOUNDARY-PRESERVING STUDENT — 128x96")
print("=" * 70)

print("Device:", DEVICE)
print("Input : (1, 3, 96, 128)")
print("Output: (1, 1, 96, 128)")


# ============================================================
# DATASET
# ============================================================

train_dataset = NYUDistillation128x96(
    DATASET_ROOT,
    split="train"
)

val_dataset = NYUDistillation128x96(
    DATASET_ROOT,
    split="val"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print("\nDataset:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))


# ============================================================
# MODEL
# ============================================================

model = DepthStudentBoundary274K128x96()

# First initialize the new layer as identity-like so the new
# model starts close to the proven 269K baseline.
initialize_boundary_fusion_as_identity(model)


# ============================================================
# LOAD COMPATIBLE 269K WEIGHTS
# ============================================================

print("\nLoading compatible 269K baseline weights:")
print(BASELINE_CHECKPOINT)

baseline_checkpoint = torch.load(
    BASELINE_CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

baseline_state = baseline_checkpoint["model_state_dict"]

current_state = model.state_dict()

compatible_state = {}
skipped = []

for key, value in baseline_state.items():

    if (
        key in current_state
        and current_state[key].shape == value.shape
    ):
        compatible_state[key] = value
    else:
        skipped.append(key)

load_result = model.load_state_dict(
    compatible_state,
    strict=False
)

model = model.to(DEVICE)


# Reinitialize the new fusion layer after loading because it does
# not exist in the 269K checkpoint.
initialize_boundary_fusion_as_identity(model)


num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("\nWeight transfer:")
print("Compatible tensors loaded:", len(compatible_state))
print("Skipped baseline tensors:", len(skipped))
print("Missing model tensors:", len(load_result.missing_keys))
print("Unexpected tensors:", len(load_result.unexpected_keys))

if load_result.missing_keys:
    print("\nMissing keys:")
    for key in load_result.missing_keys:
        print(" ", key)

print("\nStudent:")
print("Parameters:", f"{num_parameters:,}")
print(
    "FP32 size:",
    f"{num_parameters * 4 / 1024 / 1024:.3f} MB"
)
print(
    "INT8 weight size:",
    f"{num_parameters / 1024 / 1024:.3f} MB"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)


# ============================================================
# AMP
# ============================================================

use_amp = DEVICE.type == "cuda"

if use_amp:
    scaler = torch.amp.GradScaler("cuda")
else:
    scaler = None


# ============================================================
# MASKED SMOOTH L1
# ============================================================

def masked_smooth_l1(prediction, target, mask):

    loss = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none"
    )

    loss = loss * mask

    denominator = mask.sum().clamp_min(1.0)

    return loss.sum() / denominator


# ============================================================
# GRADIENT LOSS
# Same as the 269K baseline.
# ============================================================

def gradient_loss(prediction, target, mask):

    pred_dx = (
        prediction[:, :, :, 1:]
        - prediction[:, :, :, :-1]
    )

    target_dx = (
        target[:, :, :, 1:]
        - target[:, :, :, :-1]
    )

    pred_dy = (
        prediction[:, :, 1:, :]
        - prediction[:, :, :-1, :]
    )

    target_dy = (
        target[:, :, 1:, :]
        - target[:, :, :-1, :]
    )

    mask_x = (
        mask[:, :, :, 1:]
        * mask[:, :, :, :-1]
    )

    mask_y = (
        mask[:, :, 1:, :]
        * mask[:, :, :-1, :]
    )

    loss_x = (
        torch.abs(pred_dx - target_dx)
        * mask_x
    )

    loss_y = (
        torch.abs(pred_dy - target_dy)
        * mask_y
    )

    denominator_x = mask_x.sum().clamp_min(1.0)
    denominator_y = mask_y.sum().clamp_min(1.0)

    loss_x = loss_x.sum() / denominator_x
    loss_y = loss_y.sum() / denominator_y

    return (loss_x + loss_y) / 2.0


# ============================================================
# TOTAL LOSS
# ============================================================

def compute_loss(prediction, gt, teacher, mask):

    gt_loss = masked_smooth_l1(
        prediction,
        gt,
        mask
    )

    teacher_loss = masked_smooth_l1(
        prediction,
        teacher,
        mask
    )

    edge_loss = gradient_loss(
        prediction,
        gt,
        mask
    )

    total_loss = (
        LAMBDA_GT * gt_loss
        + LAMBDA_TEACHER * teacher_loss
        + LAMBDA_GRADIENT * edge_loss
    )

    return (
        total_loss,
        gt_loss,
        teacher_loss,
        edge_loss
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(prediction, target, mask):

    prediction = prediction.detach()
    target = target.detach()

    valid = mask > 0.5

    pred = prediction[valid]
    gt = target[valid]

    if pred.numel() == 0:
        return {
            "rmse": 0.0,
            "mae": 0.0,
            "absrel": 0.0,
            "delta1": 0.0,
            "delta2": 0.0,
            "delta3": 0.0
        }

    error = pred - gt

    mae = torch.mean(torch.abs(error))

    rmse = torch.sqrt(
        torch.mean(error ** 2)
    )

    relative_valid = torch.abs(gt) > 0.05

    pred_rel = pred[relative_valid]
    gt_rel = gt[relative_valid]

    if pred_rel.numel() > 0:

        absrel = torch.mean(
            torch.abs(pred_rel - gt_rel)
            / torch.clamp(
                torch.abs(gt_rel),
                min=1e-6
            )
        )

        pred_safe = torch.clamp(
            torch.abs(pred_rel),
            min=1e-6
        )

        gt_safe = torch.clamp(
            torch.abs(gt_rel),
            min=1e-6
        )

        ratio = torch.maximum(
            pred_safe / gt_safe,
            gt_safe / pred_safe
        )

        delta1 = torch.mean(
            (ratio < 1.25).float()
        )

        delta2 = torch.mean(
            (ratio < 1.25 ** 2).float()
        )

        delta3 = torch.mean(
            (ratio < 1.25 ** 3).float()
        )

    else:

        absrel = torch.tensor(
            0.0,
            device=prediction.device
        )

        delta1 = torch.tensor(
            0.0,
            device=prediction.device
        )

        delta2 = torch.tensor(
            0.0,
            device=prediction.device
        )

        delta3 = torch.tensor(
            0.0,
            device=prediction.device
        )

    return {
        "rmse": rmse.item(),
        "mae": mae.item(),
        "absrel": absrel.item(),
        "delta1": delta1.item(),
        "delta2": delta2.item(),
        "delta3": delta3.item()
    }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_gt = 0.0
    total_teacher = 0.0
    total_edge = 0.0
    batches = 0

    start_time = time.time()

    for batch in train_loader:

        rgb = batch["rgb"].to(
            DEVICE,
            non_blocking=True
        )

        gt = batch["depth"].to(
            DEVICE,
            non_blocking=True
        )

        teacher = batch["da_teacher"].to(
            DEVICE,
            non_blocking=True
        )

        mask = batch["valid_mask"].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        if use_amp:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                prediction = model(rgb)

                (
                    loss,
                    gt_loss,
                    teacher_loss,
                    edge_loss
                ) = compute_loss(
                    prediction,
                    gt,
                    teacher,
                    mask
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:

            prediction = model(rgb)

            (
                loss,
                gt_loss,
                teacher_loss,
                edge_loss
            ) = compute_loss(
                prediction,
                gt,
                teacher,
                mask
            )

            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_gt += gt_loss.item()
        total_teacher += teacher_loss.item()
        total_edge += edge_loss.item()
        batches += 1

    return {
        "loss": total_loss / batches,
        "gt": total_gt / batches,
        "teacher": total_teacher / batches,
        "edge": total_edge / batches,
        "time": time.time() - start_time
    }


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0
    total_gt = 0.0
    total_teacher = 0.0
    total_edge = 0.0

    metric_sum = {
        "rmse": 0.0,
        "mae": 0.0,
        "absrel": 0.0,
        "delta1": 0.0,
        "delta2": 0.0,
        "delta3": 0.0
    }

    batches = 0

    with torch.no_grad():

        for batch in val_loader:

            rgb = batch["rgb"].to(
                DEVICE,
                non_blocking=True
            )

            gt = batch["depth"].to(
                DEVICE,
                non_blocking=True
            )

            teacher = batch["da_teacher"].to(
                DEVICE,
                non_blocking=True
            )

            mask = batch["valid_mask"].to(
                DEVICE,
                non_blocking=True
            )

            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):
                    prediction = model(rgb)

            else:
                prediction = model(rgb)

            (
                loss,
                gt_loss,
                teacher_loss,
                edge_loss
            ) = compute_loss(
                prediction,
                gt,
                teacher,
                mask
            )

            total_loss += loss.item()
            total_gt += gt_loss.item()
            total_teacher += teacher_loss.item()
            total_edge += edge_loss.item()

            metrics = compute_metrics(
                prediction,
                gt,
                mask
            )

            for key in metric_sum:
                metric_sum[key] += metrics[key]

            batches += 1

    for key in metric_sum:
        metric_sum[key] /= batches

    return {
        "loss": total_loss / batches,
        "gt": total_gt / batches,
        "teacher": total_teacher / batches,
        "edge": total_edge / batches,
        "metrics": metric_sum
    }


# ============================================================
# TRAINING
# ============================================================

best_rmse = float("inf")
best_epoch = 0

print("\n" + "=" * 70)
print("STARTING BOUNDARY-PRESERVING 274K TRAINING")
print("=" * 70)

print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)

print("GT loss weight:", LAMBDA_GT)
print("Teacher loss weight:", LAMBDA_TEACHER)
print("Edge loss weight:", LAMBDA_GRADIENT)

print("\nBaseline checkpoint:")
print(BASELINE_CHECKPOINT)

print("\nNew checkpoint:")
print(OUTPUT_CHECKPOINT)


for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    train_stats = train_one_epoch()
    val_stats = validate()

    scheduler.step()

    metrics = val_stats["metrics"]

    print("\n" + "-" * 70)
    print(f"Epoch {epoch:03d}/{EPOCHS}")
    print(
        f"Time: {time.time() - epoch_start:.1f}s"
    )
    print(
        f"LR: {optimizer.param_groups[0]['lr']:.7f}"
    )

    print("\nTrain loss:")
    print(f"  Total   : {train_stats['loss']:.6f}")
    print(f"  GT      : {train_stats['gt']:.6f}")
    print(f"  Teacher : {train_stats['teacher']:.6f}")
    print(f"  Edge    : {train_stats['edge']:.6f}")

    print("\nValidation loss:")
    print(f"  Total   : {val_stats['loss']:.6f}")
    print(f"  GT      : {val_stats['gt']:.6f}")
    print(f"  Teacher : {val_stats['teacher']:.6f}")
    print(f"  Edge    : {val_stats['edge']:.6f}")

    print("\nValidation metrics:")
    print(f"  RMSE    : {metrics['rmse']:.6f}")
    print(f"  MAE     : {metrics['mae']:.6f}")
    print(f"  AbsRel  : {metrics['absrel']:.6f}")
    print(f"  δ1.25   : {metrics['delta1']:.6f}")
    print(f"  δ1.25²  : {metrics['delta2']:.6f}")
    print(f"  δ1.25³  : {metrics['delta3']:.6f}")

    if metrics["rmse"] < best_rmse:

        best_rmse = metrics["rmse"]
        best_epoch = epoch

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),

                "val_rmse": metrics["rmse"],
                "val_mae": metrics["mae"],
                "val_absrel": metrics["absrel"],
                "val_delta1": metrics["delta1"],
                "val_delta2": metrics["delta2"],
                "val_delta3": metrics["delta3"],

                "num_parameters": num_parameters,

                "input_height": 96,
                "input_width": 128,

                "baseline_checkpoint": BASELINE_CHECKPOINT,

                "architecture": (
                    "269K baseline + 32-channel high-resolution "
                    "x0 skip + 1x1 64-to-32 boundary fusion"
                )
            },
            OUTPUT_CHECKPOINT
        )

        print("\n*** NEW BEST MODEL ***")
        print("Saved:", OUTPUT_CHECKPOINT)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("Best epoch:", best_epoch)
print("Best validation RMSE:", best_rmse)
print("Checkpoint:", OUTPUT_CHECKPOINT)
