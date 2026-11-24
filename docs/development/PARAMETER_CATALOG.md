# Parameter catalog

This is the starting inventory, not an approved list of defaults. See the
[operator reference](../settings/parameter_reference.md) for current Settings
fields. WP3 must classify and source every effective value before making it
editable.

| Group | Source to trace | Required verification |
|---|---|---|
| Camera assignment and acquisition | Settings repository, `CameraService`, IC4 helper | Distinguish requested serial, resolution, FPS, exposure, gain and white balance from hardware readback. |
| Calibration and geometry | Machine calibration files, global and widget adjustments, legacy worker | Confirm calibration resolution, transform order, rotations, ROI and coordinate spaces. |
| Production inference and measurements | `PartInspectorRegistry`, model settings, strategy and legacy part checks | Record weights, thresholds, class rules, measurement ordering, nominal values, units and tolerances per profile. |
| Daily inspection | Legacy paths for positions 21-23 | Inventory each position's reference target, detection rules, geometry and tolerances separately. |
| Result recording and I/O | Legacy worker, database, PLC/SiO and result presentation | Preserve trigger semantics and counter/recording field meanings; do not expose them as generic editable thresholds. |

## Daily reference (widgets 21–23)

| Parameter ID / Japanese label | Default and unit | Scope and source | Validated by typed strategy | Editable |
|---|---|---|---|---|
| `pitchSpec` / 基準寸法 | `100.0` mm | Shared `parts_specifications.yaml`, `NICHIJOU_TENKEN` | Positive finite number, used after rounding the actual measurement | No |
| `pitchTolerance` / 許容差 | `1.0` mm | Shared `NICHIJOU_TENKEN` | Nonnegative finite number, inclusive absolute difference | No |
| `pixelMultiplier` / 画素換算係数 | `0.1604477` mm/px | Shared `NICHIJOU_TENKEN` | Positive finite number, horizontal extent only | No |
| `inference_crop` / 検出範囲 | `[0, 0, 1890, 1342]`, `[2100, 0, 1890, 1342]`, `[4200, 0, 1890, 1342]` px | Per position in `part_registry.yaml` | Integer rectangle in native cropped merge coordinates. Legacy bottom clipping is required: the shared merge crop is 431 px high. | No |
| `NICHIJOU_TENKEN_Model` / 使用AIモデル | YOLO pose | Shared `models.yaml` loader/weights; legacy worker assignment | Model must be present and callable | No |
| `confidence` / 検出信頼度 | `0.4` | Shared legacy worker argument | Number in [0, 1] | No |
| `keypoint_model_imgsz` / 推論画像サイズ | `1980` px | Per position in `part_registry.yaml` | Positive integer in supported strategy range | No |
| `debug_save_images` / 診断画像保存 | `true` | Shared `NICHIJOU_TENKEN`; legacy worker | Diagnostic only; no effect on measurement | No |

Changes to the reference, tolerance, conversion, crop or inference are
judgment-affecting and require approved activation. The legacy path reads its
shared parameters at process/module start and still owns rendering and the
worker lifecycle. `debug_mode` no longer replaces accepted measured values.
