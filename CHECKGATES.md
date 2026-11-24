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

## Current main-menu policy (CG-P2-00)

- The four standard routes to pages 5–8 are absent from the menu; their
  Designer pages and slots remain intact.
- Test active production, daily, calibration and Settings navigation; keep
  the fixed-canvas, fullscreen and no-outer-scroll checks.

## IC4 camera configuration (CG-20)

- With no injected capture factory, `CameraService.acquire()` forwards the
  assigned serial, logical index, size, FPS, exposure, gain, white balance
  and rotation to the IC4 helper.
- Settings readback uses device-reported values, not requested dimensions
  or fallback zeroes. Unavailable properties display `-`; measured zero
  remains zero.
- Regression checks: `tests/test_camera_lifecycle.py`. Device acceptance
  of the requested values and physical frame orientation are separate gates.

## Camera rectification (CG-P2-10, partial)

- `tests/test_camera_lifecycle.py` exercises unequal native camera sizes and
  distinct intrinsics in both workers. Full maps use each source's own
  dimensions. Preview maps use the established common downsample dimensions.
- Missing or invalid intrinsics, incompatible acquired dimensions and
  missing/invalid homographies block accepted triggers (widgets 17 and 22)
  and leave product counters and the inspection database untouched.
- Pair timing is host receive skew, not hardware synchronization. Legacy
  profiles without calibration metadata cannot be checked against an
  original calibration resolution. Production crop and high-homography
  preparation failures and physical orientation remain open checks.

## Daily reference strategy (CG-P2-30, partial)

- `tests/test_daily_inspection.py` tests horizontal measurement, rounding before
  judgment, all three ROI offsets, valid detection NG, and configuration,
  model and inference failures with synthetic poses.
- Widgets 21–23 use the strict shared strategy. The legacy compatibility
  wrapper also preserves measured values instead of substituting the nominal.

## Daily configuration revisions (CG-P2-20, partial)

- Software tests prove that a Position 2 draft does not change the active
  configuration, approval changes only Position 2, the revision persists on
  restart, and rollback restores the previous values.
- Validation and activation failure leave the previous active snapshot intact.

## Daily application and Settings (CG-P2-40/50, partial)

- `test_daily_settings_change_drives_real_worker_and_rolls_back` runs the
  actual Settings controller and worker with fake camera/model inputs and
  isolated SQLite and PNG output. Position 2 changes NG → OK → NG across
  approval and rollback without changing the 101.4 mm measurement; the
  approved value also survives recreating the Settings service.
- Check deduplication, failed writes and position-specific Qt presentation
  in `tests/test_daily_inspection.py` and `tests/test_settings_ui.py`.
- ROI/model/debug settings are not editable. A Settings reference preview,
  dry-run measurement, widget-36 parity and machine qualification are open.

## Commands

`python tools/checkgates.py CG-BASELINE CG-DAILY CG-REGRESSION`
runs executable software checks and reports command, duration, exit status
and pytest counts for each. Windows CMD: `tools\checkgates.cmd` with the
same gate arguments. The default is `CG-BASELINE CG-REGRESSION`; see
`--help` for focused gates. A failure or a required skipped test returns
nonzero. A PASS only covers the commands run, not all phase acceptance
criteria. The performance gate is explicitly NOT TESTED.

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
