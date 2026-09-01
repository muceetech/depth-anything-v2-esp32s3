# depth-anything-v2-esp32s3 v1.0-working-esp32s3
First fully working ESP32-S3 deployment of the 128×96 Boundary-Preserving Depth Anything V2 student model. Live OV2640 RGB camera, INT8 ESP-DL inference, Wi-Fi browser streaming, and 128×96 depth output confirmed. Depth quality improvement remains the next research task.

# Depth Anything V2 Student — ESP32-S3

Lightweight Depth Anything V2 student model designed for real-time
monocular depth estimation on the XIAO ESP32-S3 Sense.

## Project Status

Current working deployment:

- Board: XIAO ESP32-S3 Sense
- Camera: OV2640
- MCU: ESP32-S3
- PSRAM: 8 MB Octal PSRAM
- ESP-IDF: v5.5.3
- ESP-DL: INT8
- Input: 128 × 96 RGB
- Output: 128 × 96 depth
- Model: Boundary-Preserving Student V1
- Parameters: 271,617
- Validation RMSE: 0.20777
- Inference time: approximately 500–530 ms
- Approximate inference rate: 1.9 FPS

## Hardware

XIAO ESP32-S3 Sense with OV2640 camera.

Camera pins:

| Signal | GPIO |
|---|---:|
| XCLK | 10 |
| SIOD | 40 |
| SIOC | 39 |
| Y9 | 48 |
| Y8 | 11 |
| Y7 | 17 |
| Y6 | 16 |
| Y5 | 15 |
| Y4 | 14 |
| Y3 | 13 |
| Y2 | 12 |
| VSYNC | 38 |
| HREF | 47 |
| PCLK | 21 |

The board uses 8 MB Octal PSRAM.

## Model

The current student model accepts:

    [1, 3, 96, 128]

and produces:

    [1, 1, 96, 128]

The model was trained from the 269K student baseline and enhanced with
a high-resolution boundary-preserving feature path.

Current model:

    student_boundary_274k_128x96_best.pth

Exact parameter count:

    271,617

Best validation RMSE:

    0.2077689856

## Training

Training uses the existing NYU distillation dataset.

Loss components:

- Ground-truth depth loss
- Depth Anything teacher distillation loss
- Gradient loss

Current V1 weights:

    Lambda GT       = 1.0
    Lambda Teacher  = 1.0
    Lambda Gradient = 0.25

Training configuration:

    Epochs       = 50
    Batch size   = 16
    Learning rate = 1e-3
    Weight decay  = 1e-4

## Quantization

The PyTorch model is converted to ESP-DL INT8 using ESP-PPQ.

Target:

    ESP32-S3

Quantization:

    8-bit
    per-tensor symmetric
    power-of-two scale

Calibration uses RGB images normalized to:

    float32 / 255.0

Generated deployment model:

    student_boundary_v1_128x96_esp32s3.espdl

## ESP32 Deployment

The ESP32-S3:

1. Captures an RGB image using OV2640.
2. Converts the camera image to RGB.
3. Resizes it to 128 × 96.
4. Normalizes RGB values to [0,1].
5. Runs the INT8 ESP-DL model.
6. Produces a 128 × 96 INT8 depth map.
7. Streams the RGB image and depth visualization through Wi-Fi.

## Wi-Fi

The ESP32 creates an access point:

    SSID: DEPTH_V1_ESP32S3
    Password: depth123
    IP: 192.168.4.1

Open:

    http://192.168.4.1

The browser displays:

- Live RGB camera
- Live depth colormap
- Depth statistics
- Inference information

## Current Results

The model successfully runs on the ESP32-S3.

Typical measured inference:

    ~516 ms
    ~1.9 FPS

The ESP-DL embedded model test passes successfully.

However, the current depth map is relatively smooth and does not preserve
object boundaries strongly enough.

The next research step is to compare:

    Live camera frame
            ↓
    PyTorch student prediction
            VS
    ESP32 INT8 prediction

using the exact same camera frame.

This will determine whether the remaining depth-quality problem is caused
by:

1. Student model quality,
2. ESP32 preprocessing,
3. INT8 quantization, or
4. deployment differences.

## Research Roadmap

### Completed

- [x] Depth Anything V2 student model
- [x] 128 × 96 conversion
- [x] Boundary-preserving V1
- [x] ESP-PPQ INT8 quantization
- [x] ESP-DL model generation
- [x] ESP32-S3 deployment
- [x] OV2640 live camera
- [x] Live depth visualization
- [x] Wi-Fi browser interface

### Next

- [ ] Compare identical camera frame on PC and ESP32
- [ ] Verify preprocessing equivalence
- [ ] Verify INT8 vs FP32 output
- [ ] Improve depth boundaries
- [ ] Evaluate V2/V3 training variants
- [ ] Quantize the best model
- [ ] Compare against ToF depth sensors
- [ ] Final research evaluation

## Repository Structure

```text
esp32s3/
    ESP32 deployment project

model/
    PyTorch student model and checkpoint

training/
    Training scripts

quantization/
    ESP-PPQ / ESP-DL quantization scripts

evaluation/
    Evaluation and visualization scripts

docs/
    Hardware and research notes
