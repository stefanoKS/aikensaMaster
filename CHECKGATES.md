# Inspection strategy checkgates

## Gate IDs and status

Use **PASS**, **FAIL**, **BLOCKED**, or **NOT TESTED**. Track software and
physical-machine qualification separately; a software PASS does not confer a
machine PASS.

| ID | Software gate |
|---|---|
| A | Reproduce the baseline; document rules and preserve machine configuration |
| B | 1920×1080 unscrolled main menu; legacy indices and fullscreen preserved |
| C | Camera placeholders, fail-closed acquisition, calibration and reconnect |
| D | Consistent, usable Settings; unchanged configuration behavior |
| E | Data-driven widget-17 strategy with original behavior |
| F | Authoritative part registry and validation |
| G | Generic worker dispatch with explicit failures |
| H | One result, presentation, counters and persistence |
| I | Generic part navigation and presentation |
| J | Verified incremental legacy strategy migrations |
| K | Complete new-part integration with no central product-specific edits |
| L | Executable software checkgates |
| M | Readability and operator/developer documentation |

## Requirements

- Preserve widget 17 inference parameters, partcheck outputs, tolerances,
  geometry, triggers, counters, signals and persisted fields.
- Unknown profiles, unavailable strategies/models, invalid inputs/results and
  exceptions must not yield OK or fall back to another inspector.
- A new profile must run through the coordinator and be dynamically registered
  for real UI presentation and persistence without product-specific edits to
  the two central modules. Keep pages 0-49 unchanged.
- Test against the original partcheck using identical synthetic fixtures.
  Software tests do not establish camera, GPU, PLC or machine behavior.

## Fixed inspection pages (CG-10 through CG-13)

- Every populated production and daily Designer root is 1920×1080.
- The main menu, calibration, inspection and reserved pages are direct stack
  children at the original indices 0-49. Settings remains at 50 and can
  scroll internally.
- At 1920×1080, all visible control and label geometry fits inside the Qt
  page contents. At smaller logical resolutions, fixed pages may clip
  without resizing the window. Navigation preserves fullscreen.
- Exercise daily navigation through 21-23, calibration controls, and camera
  and model diagnostics with fake workers. Inspect representative offscreen
  screenshots; confirm the actual display separately on the machine.
- Regression checks: `tests/test_settings_ui.py`.

## Commands

```sh
python -m compileall -q aikensa tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 QT_QPA_PLATFORM=offscreen python -m pytest -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 QT_QPA_PLATFORM=offscreen python -m pytest -q tests/test_inspection_strategy.py
```

Windows CMD:

```bat
set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
set QT_QPA_PLATFORM=offscreen
python -m pytest -q
```

## Widget 17 baseline

The full-resolution path remaps calibrated camera frames, rotates if needed,
merges with the widget-adjusted homography, crops at native resolution, and
returns BGR plus RGB. Partcheck receives BGR; SAHI receives RGB. SAHI uses
512x1980 slices, overlap 0/0.2, IOS 0.2, class-agnostic GREEDYNMM,
`verbose=0`, and `perform_standard_pred=False`. The keypoint model receives
the first and last 512 BGR columns at `conf=0.6`, `imgsz=512`, `verbose=False`.
The anomaly model is passed when available; the partcheck currently forces its
hanire judgment off. Its signature is `(image, detections, left, right, anomaly)`.

The six values are annotated image, ordered pitch measurements (left edge,
inter-clip pitches, right edge, total length), pitch judgments, ID judgments,
status and NG reason. The thread historically stores ID judgments in
`DeltaPitch` and persists them as `delta_pitch`; this is intentionally retained.
The single result increments the OK or NG current and daily counter, emits
preview, counter and pitch signals, saves raw/result images and records the
existing database fields including operator, PPMS, status and NG reason.

## Current gaps

- Dynamic UI registration and generic production dispatch for a new widget are
  **BLOCKED**; the synthetic profile test only proves coordinator execution
  and profile metadata resolution. It does not satisfy the new-part gate.
- Full camera-to-database production-cycle equivalence is **NOT TESTED**;
  strategy tests compare partcheck outcomes and inference arguments in isolation.
- Physical cameras, calibration, GPU inference, PLC, sound and database behavior
  on the production machine are **NOT TESTED**.
