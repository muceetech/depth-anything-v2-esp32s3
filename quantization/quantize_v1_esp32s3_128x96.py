import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------------
# Make the local model file importable.
# Keep this script in the same directory as:
#   student_boundary_274k_128x96.py
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from student_boundary_274k_128x96 import DepthStudentBoundary274K128x96


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"D:\research\student48\data\nyu256"

CHECKPOINT = (
    r"D:\research\student48\scripts\depth_anything_student"
    r"\checkpoints\student_boundary_274k_128x96_best.pth"
)

OUTPUT_DIR = Path(
    r"D:\research\student48\esp32s3_depth_v1\quantized"
)

ESPDL_MODEL = OUTPUT_DIR / "student_boundary_v1_128x96_esp32s3.espdl"

INPUT_SHAPE = [1, 3, 96, 128]

TARGET = "esp32s3"
NUM_OF_BITS = 8

# Start conservatively. This does not create a new dataset.
CALIBRATION_SAMPLES = 128
CALIBRATION_BATCH_SIZE = 1

# Use CPU for the first ESP-PPQ conversion. This avoids CUDA/PPQ
# backend compatibility problems and is sufficient for 128 samples.
DEVICE = "cpu"


# ============================================================
# CALIBRATION DATASET
# ============================================================

class RGBCalibrationDataset(Dataset):
    """
    Uses existing RGB PNGs from:
        DATASET_ROOT/rgb

    The training pipeline already stores the images at 128x96.
    We only load RGB tensors here; GT/teacher depth is not needed
    for calibration.
    """

    def __init__(self, root, max_samples=None):
        self.rgb_dir = Path(root) / "rgb"

        if not self.rgb_dir.exists():
            raise FileNotFoundError(
                f"RGB directory not found: {self.rgb_dir}"
            )

        self.files = sorted(
            list(self.rgb_dir.glob("*.png"))
            + list(self.rgb_dir.glob("*.jpg"))
            + list(self.rgb_dir.glob("*.jpeg"))
        )

        if not self.files:
            raise RuntimeError(
                f"No RGB images found in {self.rgb_dir}"
            )

        if max_samples is not None:
            self.files = self.files[:max_samples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np

        path = self.files[idx]

        image = Image.open(path).convert("RGB")

        # Existing RGB images are expected to be 128x96.
        if image.size != (128, 96):
            image = image.resize((128, 96))

        arr = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        # HWC -> CHW
        tensor = torch.from_numpy(
            arr.transpose(2, 0, 1)
        ).contiguous()

        return tensor


def collate_fn(batch):
    return torch.stack(batch, dim=0).to(DEVICE)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print("=" * 72)
    print("V1 → ESP32-S3 INT8 QUANTIZATION")
    print("=" * 72)

    print("Checkpoint:")
    print(CHECKPOINT)

    print("\nInput shape :", INPUT_SHAPE)
    print("Target      :", TARGET)
    print("Bits        :", NUM_OF_BITS)
    print("Device      :", DEVICE)

    model = DepthStudentBoundary274K128x96()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nModel parameters:", f"{parameters:,}")

    expected = 271617
    if parameters != expected:
        raise RuntimeError(
            f"Unexpected parameter count: {parameters}; "
            f"expected {expected}. "
            "This indicates that the wrong model architecture may "
            "have been imported."
        )

    # Verify exact output shape before quantization.
    example = torch.zeros(
        INPUT_SHAPE,
        dtype=torch.float32
    )

    with torch.no_grad():
        output = model(example)

    print("Model input :", tuple(example.shape))
    print("Model output:", tuple(output.shape))

    expected_output = (1, 1, 96, 128)

    if tuple(output.shape) != expected_output:
        raise RuntimeError(
            f"Unexpected output shape {tuple(output.shape)}; "
            f"expected {expected_output}"
        )

    print("Model verification: PASSED")

    return model


# ============================================================
# MAIN QUANTIZATION
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model = load_model()

    # Import through PPQ's public API. ESP-PPQ exposes the ESP-DL
    # quantization API through the ppq.api namespace.
    try:
        from esp_ppq.api import espdl_quantize_torch
    except ImportError as exc:
        raise RuntimeError(
            "Could not import espdl_quantize_torch from ppq.api.\n"
            "Your installed esp-ppq package may expose a different "
            "API layout. Run:\n\n"
            "python -c \"from ppq.api import espdl_quantize_torch; "
            "print(espdl_quantize_torch)\"\n"
        ) from exc

    # ----------------------------------------------------------
    # Calibration
    # ----------------------------------------------------------

    calibration_dataset = RGBCalibrationDataset(
        DATASET_ROOT,
        max_samples=CALIBRATION_SAMPLES
    )

    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=CALIBRATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False
    )

    print("\nCalibration images:",
          len(calibration_dataset))

    print("Calibration directory:",
          calibration_dataset.rgb_dir)

    print("\nNo new dataset is being generated.")
    print("Existing RGB images are used directly.")

    # ----------------------------------------------------------
    # Quantize and export
    # ----------------------------------------------------------

    print("\n" + "=" * 72)
    print("STARTING ESP-PPQ QUANTIZATION")
    print("=" * 72)

    print("\nExport file:")
    print(ESPDL_MODEL)

    print("\nIMPORTANT:")
    print("- target = esp32s3")
    print("- num_of_bits = 8")
    print("- batch size = 1")
    print("- export_test_values = True")
    print("- CPU-side PPQ reference will be generated")

    quant_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=str(ESPDL_MODEL),
        calib_dataloader=calibration_loader,
        calib_steps=len(calibration_loader),
        input_shape=INPUT_SHAPE,
        inputs=None,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )

    print("\n" + "=" * 72)
    print("ESP-PPQ QUANTIZATION COMPLETE")
    print("=" * 72)

    print("\nESP-DL model:")
    print(ESPDL_MODEL)

    for suffix in [".info", ".json"]:
        path = ESPDL_MODEL.with_suffix(suffix)
        print(f"{suffix} file:", path)

    # ----------------------------------------------------------
    # PC-side quantized verification
    # ----------------------------------------------------------

    print("\n" + "=" * 72)
    print("PC-SIDE ESP-PPQ QUANTIZED MODEL CHECK")
    print("=" * 72)

    try:
        from esp_ppq import TorchExecutor

        executor = TorchExecutor(
            graph=quant_graph,
            device=DEVICE
        )

        test_input = next(
            iter(calibration_loader)
        )

        with torch.no_grad():
            quant_output = executor(test_input)

        if isinstance(quant_output, (tuple, list)):
            quant_output = quant_output[0]

        print("Reference input shape :",
              tuple(test_input.shape))
        print("Quantized output shape:",
              tuple(quant_output.shape))

        print(
            "Quantized output min  :",
            float(quant_output.min())
        )
        print(
            "Quantized output max  :",
            float(quant_output.max())
        )

        # Save one reference tensor for later board comparison.
        torch.save(
            {
                "input": test_input.cpu(),
                "output": quant_output.cpu(),
            },
            OUTPUT_DIR / "pc_esp_ppq_reference.pt"
        )

        print(
            "\nSaved PC reference:",
            OUTPUT_DIR / "pc_esp_ppq_reference.pt"
        )

    except Exception as exc:
        print("\nWARNING:")
        print("ESP-PPQ export succeeded, but the optional")
        print("PC-side TorchExecutor check failed:")
        print(repr(exc))
        print(
            "\nThe .espdl may still have been exported. "
            "Do not deploy it until we inspect the error."
        )

    print("\n" + "=" * 72)
    print("NEXT STEP")
    print("=" * 72)

    print(
        "\nInspect the complete quantization log and the generated:"
    )
    print("  *.espdl")
    print("  *.info")
    print("  *.json")

    print(
        "\nDo NOT flash the model yet."
    )
    print(
        "First send the complete console output to ChatGPT so we "
        "can verify operator support and quantization success."
    )


if __name__ == "__main__":
    main()
