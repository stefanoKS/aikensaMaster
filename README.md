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

## 設備設定 (Machine settings)

日常的な設備パラメータ（カメラのシリアル番号・解像度・露光・校正・位置補正・
品番別調整）は **UI から変更できます**。Python / YAML を直接編集する必要はありません。

メイン画面の **設備設定** ボタン →
カメラ設定 / 校正・位置調整 / 全体微調整 / 品番別設定 / 検査条件 / AIモデル設定 / 設定履歴

### ドキュメント（日本語）

- [docs/settings/README.md](docs/settings/README.md) — 目次と全体像
- [docs/settings/install_and_start.md](docs/settings/install_and_start.md) — インストール、新規設備の初期設定
- [docs/settings/camera_assignment.md](docs/settings/camera_assignment.md) — カメラのシリアル割当
- [docs/settings/calibration.md](docs/settings/calibration.md) — 校正とホモグラフィ
- [docs/settings/fine_adjustment.md](docs/settings/fine_adjustment.md) — 全体／品番別の微調整
- [docs/settings/save_and_restore.md](docs/settings/save_and_restore.md) — 保存・状態・復元
- [docs/settings/git_pull_protection.md](docs/settings/git_pull_protection.md) — `git pull` からの保護（重要）
- [docs/settings/parameter_reference.md](docs/settings/parameter_reference.md) — パラメータ一覧
- [docs/settings/troubleshooting.md](docs/settings/troubleshooting.md) — 異常時の対処
- [docs/settings/hardware_checklist.md](docs/settings/hardware_checklist.md) — 実機検証チェックリスト

### 重要: 設備固有ファイルは Git 管理外

```
aikensa/core/config/cam/cam_id.yaml          # カメラのシリアル番号
aikensa/core/config/cam/cam_config.yaml      # 解像度・露光など
aikensa/core/param/camera/                   # 校正・ホモグラフィ・平面化
aikensa/core/param/adjustments/              # 微調整・履歴
```

`git pull` の前に **必ず** バックアップしてください。

```bat
tools\backup_machine_config.cmd     :: "BACKUP VERIFIED" を確認してから git pull
```

Git 管理のテンプレートは `aikensa/core/config/defaults/` にあります（`null` 初期値）。

## アーキテクチャ（主要モジュール）

| モジュール | 責務 |
|------------|------|
| `aikensa/core/settings/` | 型付き設定・検証・原子的保存・版管理・バックアップ |
| `aikensa/core/camera/` | カメラの単一所有権・状態遷移・シリアル解決 |
| `aikensa/core/geometry/` | null 安全な校正利用・スケール変換・変換合成・マスク合成 |
| `aikensa/gui/pages/SETTINGS/` | Qt Designer `.ui`（設備設定の 8 画面） |
| `aikensa/gui/settings_pages/` | 上記 `.ui` のコントローラ |
| `aikensa/core/thread/` | 校正スレッド / 検査スレッド（共有カメラサービスを使用） |

### テスト

```bat
tools\run_tests.cmd
```

実機カメラ・GPU・PLC は不要です（UI は `QT_QPA_PLATFORM=offscreen`）。
実機でしか確認できない項目は
[docs/settings/hardware_checklist.md](docs/settings/hardware_checklist.md) を参照してください。

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
