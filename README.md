# YOLO TensorRT Tracking and Jetson Camera Toolkit

A production-ready toolkit for real-time YOLO detection, multi-object tracking, and interactive object locking with TensorRT acceleration. Optimized for NVIDIA Jetson devices and discrete GPUs, with first-class support for the Jetson ARGUS camera via GStreamer.

![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/ammarmalik17/Jetson-Yolo-Tracking-TensorRT)

## Features
- TensorRT-accelerated YOLO inference (Ultralytics)
- Multi-object tracking with BoT-SORT/ByteTrack
- Interactive object locking (mouse click, class-based, or ID-based)
- Real-time overlays: FPS, status, track trails
- Jetson ARGUS camera pipelines with multiple fallbacks
- Video/file stream support and optional saving of annotated video and tracks

## Repository Layout
- `scripts/yolo_track_lock.py`: Main script for detection, tracking, and interactive object locking across video files, streams, and Jetson camera.
- `scripts/jetson_camera_inference.py`: Minimal example to run YOLO TensorRT inference on the Jetson ARGUS camera, with performance overlays.
- `scripts/jetson_camera_test.py`: Quick camera diagnostics to find a working GStreamer pipeline for the ARGUS camera.
- `scripts/setup_jetson.sh`: Automated setup validation and system diagnostics for Jetson devices.
- `tools/export_tensorrt_engine.py`: Exports a YOLO model to a TensorRT `.engine` with conservative, Jetson-friendly settings.
- `models/`: Directory for storing YOLO model files (`.pt`, `.engine`, `.onnx`).
- `requirements.txt`: Pinned dependencies for x86_64 and Jetson (with platform markers).

## Clone and Setup
```bash
# Clone the repository
git clone https://github.com/ammarmalik17/Yolo-TensorRT-Tracking-and-Jetson-Camera-Toolkit.git
cd Yolo-TensorRT-Tracking-and-Jetson-Camera-Toolkit

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Quick Setup & Validation (Jetson Only)
```bash
# Run the automated setup and validation script
bash scripts/setup_jetson.sh
```
This script will:
- Check JetPack version and system compatibility
- Validate TensorRT, PyTorch, and OpenCV installations
- Test GStreamer camera support
- Verify model files
- Run a quick inference test
- Provide performance recommendations

Notes for Jetson:
- Prefer system OpenCV with GStreamer (provided by JetPack). The requirements file avoids pip-OpenCV on aarch64.
- Install TensorRT and CUDA via JetPack. The Python package `nvidia-tensorrt` is generally not needed on Jetson.
- Tested on JetPack 6.2 (L4T r36.x) on Jetson Orin NX.

### Prefer Jetson AI Lab PyPI mirror (Jetson-only)
For faster, more reliable installs on Jetson, prefer prebuilt wheels from the Jetson AI Lab PyPI mirror. It hosts wheels tailored to specific JetPack (jp6/jp7) and CUDA variants (cu126, cu129, cu130), reducing build times and preventing ABI mismatches.

- Why use it:
  - Prebuilt, Jetson-compatible wheels (PyTorch/TorchVision/Ultralytics deps, etc.)
  - Avoids slow or failing source builds on-device
  - Ensures CUDA/TensorRT compatibility aligned with your JetPack

- Mirror: [Jetson AI Lab PyPI mirror](https://pypi.jetson-ai-lab.io/)

- Quick usage (choose the index matching your JetPack/CUDA, e.g., jp6 + cu130):
  ```bash
  # Prefer the mirror but allow fallback to PyPI when a package isn't available
  pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu130/+simple/ -r requirements.txt
  ```

- Optional pip config (persistent): create `~/.config/pip/pip.conf` with:
  ```ini
  [global]
  extra-index-url = https://pypi.jetson-ai-lab.io/jp6/cu130/+simple/
  ```
  Then run the usual install:
  ```bash
  pip install -r requirements.txt
  ```

If you're unsure which index to use: `jp6` corresponds to JetPack 6, `jp7` to JetPack 7; pick the `cuXXX` that matches your CUDA (e.g., 12.6 → cu126, 12.9 → cu129, 13.0 → cu130). If a package isn't present on the mirror, pip will fall back to PyPI when using `--extra-index-url`.

Jetson best practices and tips:
- Enable MAX power mode and clocks for consistent benchmarking and maximum performance:
  ```bash
  sudo nvpmodel -m 0
  sudo jetson_clocks
  ```
- Monitor system and confirm JetPack details with jetson-stats:
  ```bash
  sudo apt update
  sudo pip install jetson-stats
  sudo reboot
  jtop
  ```
- Install Ultralytics, PyTorch, and TorchVision versions compatible with your JetPack release. Prefer NVIDIA/JetPack-provided wheels for PyTorch/TorchVision. TensorRT is bundled with JetPack.
- For best inference performance, convert to TensorRT and use FP16 when possible. INT8 requires calibration but can provide further speedups.
- Ensure OpenCV has GStreamer enabled for ARGUS camera and network streams (RTSP/RTMP/HTTP).


## Model Preparation
You can use either a prebuilt TensorRT engine (`.engine`) or export one from a YOLO model (`.pt`).

### 1) Export a TensorRT Engine
```bash
python tools/export_tensorrt_engine.py
```
This script loads `yolo11n.pt` and creates `models/yolo11n.engine` with static shapes and FP16, suitable for Jetson.

### 2) Use an Existing Engine
Place your engine file in the `models/` directory, e.g., `models/best.engine` or `models/yolo11n.engine` (or pass a path via `--model`).

## Usage

### yolo_track_lock.py
Multi-object tracking with interactive object locking.

```bash
python scripts/yolo_track_lock.py \
  --model models/yolo11n.engine \
  --source 0 \
  --conf-thres 0.25 \
  --iou-thres 0.7 \
  --imgsz 640 \
  --tracker botsort.yaml \
  --enable-lock \
  --show-tracks \
  --save-vid
```

Key arguments:
- `--model`: Path to YOLO model (`.engine` preferred, `.pt` also supported)
- `--source`: Video file, directory, camera index (e.g., `0`), or stream URL (`rtsp://`, `http://`, `https://`, `rtmp://`).
- `--conf-thres`, `--iou-thres`, `--max-det`, `--agnostic-nms`, `--classes`
- `--imgsz [H W]` or single `--imgsz 640`, `--half`, `--batch`, `--visualize`, `--augment`, `--retina-masks`
- `--tracker botsort.yaml|bytetrack.yaml`
- `--output-dir`, `--save-vid`, `--save-tracks`
- `--track-history`, `--show-tracks`, `--track-thickness`
- `--device` (e.g., `0`, `cpu`), `--display`, `--verbose`, `--show-fps`
- Locking: `--enable-lock`, `--lock-id <int>`, `--lock-class <name|id>`, `--highlight-color R,G,B`

Source modes:
- Video file: path ending with `.mp4`, `.avi`, `.mov`, `.mkv`
- Jetson ARGUS camera: `--source camera` or `jetson` or `camsrc=1`
- Generic camera/stream: numeric index (e.g., `0`) or URL starting with `rtsp://`, `http://`, `https://`, or `rtmp://`

Controls during runtime:
- `l`: toggle lock mode
- `c`: clear locked object
- `i`: enter a track ID (press Enter to confirm)
- Mouse click: lock object under cursor
- `p`/Space: pause/resume
- `s`: save current frame
- `q`: quit

Examples:
- From a video file:
  ```bash
  python scripts/yolo_track_lock.py --model models/yolo11n.engine --source sample.mp4 --save-vid --enable-lock
  ```
- From a USB camera:
  ```bash
  python scripts/yolo_track_lock.py --model models/yolo11n.engine --source 0 --show-tracks --enable-lock
  ```
- From Jetson ARGUS camera:
  ```bash
  python scripts/yolo_track_lock.py --model models/yolo11n.engine --source jetson --enable-lock --display
  ```

### jetson_camera_inference.py
Minimal camera inference with FPS overlays (TensorRT engine required):
```bash
python scripts/jetson_camera_inference.py
```
If loading fails, ensure `models/yolo11n.engine` exists or export via `tools/export_tensorrt_engine.py`.

### jetson_camera_test.py
Probe for a working GStreamer pipeline on Jetson and show a short preview:
```bash
python scripts/jetson_camera_test.py
```
On success, it prints a working pipeline and suggests running `scripts/jetson_camera_inference.py`.

### export_tensorrt_engine.py
Export a YOLO model to TensorRT with static shapes and FP16:
```bash
python tools/export_tensorrt_engine.py
```
Outputs: `models/yolo11n.engine` in the models directory.

## Outputs
- Annotated video: `output/videos/<source_name>_locked.mp4` (when `--save-vid`)
- Track data (CSV): `output/tracks/<source_name>_tracks.txt` (when `--save-tracks`)
- Run summary: `output/summaries/<source_name>_tracking_summary.txt`

## Tips and Troubleshooting
- If OpenCV fails to open ARGUS pipelines on Jetson, ensure no `nvgstcapture-1.0` process is running and OpenCV is built with GStreamer.
- If FPS is low, reduce `--imgsz`, increase `--vid-stride`, or use FP16 (`--half`).
- For trackers, ensure dependencies are installed (see `requirements.txt`, includes `lapx`, `scipy`).
- If `.engine` loading fails, regenerate the engine to match your device and TensorRT version.

Additional Jetson resources:
- Installing Ultralytics, PyTorch/TorchVision, and onnxruntime-gpu on specific JetPack versions
- TensorRT usage guidance (conversion and running), optional DLA usage, and best practices
- YOLO11 benchmark charts for multiple Jetson devices (AGX Orin, Orin NX, Orin Nano)

See the Ultralytics NVIDIA Jetson guide for details: [Ultralytics YOLO Docs – NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/).

## License
This repository is provided for research and production use. See source headers for details. 

Acknowledgment: This project’s Jetson notes and recommendations were informed by the Ultralytics NVIDIA Jetson guide: [https://docs.ultralytics.com/guides/nvidia-jetson/](https://docs.ultralytics.com/guides/nvidia-jetson/).
