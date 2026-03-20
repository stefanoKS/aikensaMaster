# AIKENSA (aikensa_dcam) MASTER

Internal inspection software built with **Qt (PyQt5)** for multi-camera visual inspection workflows.
Designed for production-line use: camera capture → preprocessing → ML inference (detection / keypoints / segmentation / anomaly) → OK/NG decision → UI display + logging.

> This repository is intended for **internal use**.

---

## Features

- **Qt GUI** with multiple screens / widgets for displaying inspection images and status
- **Multi-threaded architecture** (UI thread + camera thread(s) + inspection thread + optional PLC/Modbus thread)
- **Multiple model types per part**
  - clip / defect detection (YOLO detect)
  - keypoint detection (YOLO pose)
  - segmentation (YOLO seg)
  - anomaly scoring (optional)
- **Config-driven model paths** via YAML (per-part, per-task models)
- **Per-part processing logic** (currently in Python modules under `aikensa/parts_config/`)
- Image preprocessing utilities (crop, undistort, homography/planarize, etc.)
- Sound feedback (OK/NG, notifications)
- Debug image outputs (optional)

---

## Requirements

### OS
- Primary target: **Windows 10/11**
- Development often works on Linux, but camera SDK/driver availability may differ.

### Python
- Python 3.8+ recommended (match your deployed environment)

### Core dependencies (typical)
- PyQt5
- numpy
- opencv-python
- ultralytics (YOLO)
- pyyaml
- (optional) pymodbus (if PLC/Modbus integration is enabled)
- (optional) camera SDK bindings (e.g., Imaging Source IC4 / other)

> Exact versions depend on your environment. If you don’t have a `requirements.txt` yet, create one once your environment is stable.

---

## Quick Start

### 1) Create an environment (Conda example)

```bash
conda create -n aikensa python=3.10 -y
conda activate aikensa

pip install -U pip
pip install pyqt5 numpy opencv-python pyyaml ultralytics
# optional:
# pip install pymodbus
```
