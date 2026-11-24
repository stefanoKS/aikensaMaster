# Agent handover

## Phase 3

Started at clean `9cb91787b74d5c4ca812e637bd913774d1c8616e`, matching remote
`main`. Independent baseline: 212 passed, zero failures/skips, compilation
PASS (Conda `aikensamaster`, Python 3.11.14). Machine calibration and
overrides still reside under `aikensa/core/param`; the software tests use
temporary configuration, images and databases.

CG-P3-10: reproduced the blank Position 2 label after clicking Next. The
worker only sent a placeholder when a fault changed; the UI only painted
the selected page and reset discarded the old images. The worker now
publishes deduplicated equipment state independently of diagnostics, and
the active page presents that state upon entry or session reset. A tagged
live image can replace it only for the current page activation and equipment
sequence. Production, calibration and camera Settings remain separate.
Software checks: camera 51, offscreen Qt 67, full regression 219 passed
with no failures/skips; seven screenshots at 1920×1080.
Physical cameras and Windows display remain NOT TESTED. Next: structured
daily session outcomes, including reinspection invalidation.

CG-P3-20: daily positions now track typed attempt and recording states.
Starting a new attempt immediately drops the old all-pass summary.
An equipment interruption, local/remote recording failure, obsolete
configuration, or old session/cycle cannot appear as a current pass.
The session compares effective profile values rather than invalidating
unaffected positions merely because another position's revision changed.
Existing local and remote persistence mechanisms are unchanged; MySQL
retry, reconciliation and live equipment timing still require later gates.
Software regression: 221 passed, including 28 daily checkgate tests, no
failures or skips.

## Phase 2

Local and remote `main` started at `c8d486874a8ff633135a4c54a48b1cf179be588b`
with a clean tree. The `aikensamaster` Conda environment (Python 3.11.14)
passed `compileall`; the initial offscreen suite had 159 passes and two
failures, both expecting buttons intentionally removed in `c8d4868`.
CG-P2-00 reconciles those expectations without changing Designer files or
machine configuration. Software PASS: 60 Qt tests and 161 full-suite tests,
zero failures or skips.

CG-P2-10 now uses each camera's native frame dimensions and validated
intrinsics in both workers. Inspection checks both cameras' full and scaled
maps and homographies before accepting a cycle. A failed merge-calibration
map leaves raw camera previews available but prevents calibration saves.
Software tests use distinct intrinsics at 320×240 and 400×300 with fake
cameras and isolated persistence; full regression: 178 passed, no failures or
skips. No production capture, transform order or
machine configuration files were changed. Gate remains partial: move
production crop/high-homography and late preparation errors out of the
legacy black-image fallback; verify transform parity and reconnect metadata.

CG-P2-20/30 partial: validated daily reference, ROI and inference types plus
a Qt/audio-independent strategy. The legacy daily wrapper now uses the
shared measurement function, so a passing 100.6 mm is still 100.6 mm even
with debug enabled. Default ROIs are in native cropped-merge coordinates:
the shared widget crop is 6528×431, so the legacy 1342 px inference height
clips at the bottom. The strict strategy explicitly supports that case and
rejects horizontal overflow or a degenerate crop. Next: resolve registered
profiles and immutable approved daily revisions through `SettingsService`,
then wire the strategy and frozen cycle identity into the worker before
exposing editable Settings. Do not activate the strict path in isolation.
Software regression at this milestone: 22 daily strategy cases, 200 full
suite passes, no failures or skips.

The daily configuration service now resolves the three registered profiles
from shared YAML and applies validated per-position reference, tolerance
and scale overrides. `RevisionManager` stores daily overrides as an immutable
scoped snapshot selected by one atomic pointer. Daily rollback does not
alter camera or alignment settings. The active revision is loaded separately
from production adjustment revisions; Settings does not yet show daily
history or expose the editor. A shared YAML deployment remains separate from
machine-local override history. Software regression: 202 passed, no failures
or skips.

The worker now uses the shared strategy and strict geometry for widgets
21–23. Each accepted cycle freezes operator, position, session, and
configuration revision. SQLite retains the legacy fields and records the
extra metadata in `daily_inspection_metadata`. Saved PNGs preserve the
original directory structure; failed writes do not create a successful local
record, and the cycle ID prevents duplicate local inserts. The optional
legacy MySQL mirror remains; its physical deployment and failure recovery
are NOT TESTED. Widget 36's product-specific daily mode is unchanged.

`設備設定 → 日常点検設定` edits baseline length, tolerance and pixel conversion
per position, with validation, confirmation, history, and rollback. The
synthetic controller/worker regression used the same 101.4 mm input:
Position 2 was NG at 1.0 mm (including with an unsaved 2.0 mm draft),
OK at 2.0 mm after approval and after reloading the worker's Settings
service, and NG after rollback. Positions 1 and 3 remained NG. The test
checked SQLite metadata, images, frozen identity during a page change and
an unchanged production counter. It also renders the Japanese annotation
with the tracked font and decodes the saved result PNG. Separate Qt coverage
checks daily navigation and position-specific presentation. `compileall` passed; the
full suite had 209 passes, no failures or skips.

Next: qualify real camera orientation, daily targets, inference, MySQL and
display; finish the remaining production geometry failure paths. Gate 5
still needs editable ROI/inference controls, reference preview and a
side-effect-free test measurement using the existing model owner. Gates
6–8 (production migration, editable production settings, preview benchmark)
are not started.

Executable software checkgates: `tools/checkgates.py` and the Windows CMD
wrapper. Baseline compilation, camera (47), geometry (35), configuration
(38), daily (27), widget-17 parity (14), Settings/Qt (64), local recording
(1), and full regression (212) passed. The groups overlap, and none of
their elapsed times is an inspection performance benchmark.
The baseline, daily and full regression gates also passed from a temporary
archive containing only tracked sources.

## Prior evidence

Starting head: `7f031789e8a4b11b9ddd61e14737b4177a4b41eb` on `main`;
verified against GitHub `main`.
Ending implementation head: `121b2aaeb823768c3fbba8f8dc9f63f8f49e26f9`.
This handover has a separate documentation commit.

| Package | Implementation | Software evidence | Machine |
|---|---|---|---|
| WP0 clean-checkout gate | SOFTWARE VERIFIED (`86f0a3c`) | PASS: 157 tests after replacing the local calibration dependency | NOT TESTED |
| WP0 static metrics | IN PROGRESS | See the [baseline](./PERFORMANCE_BASELINE.md) for measured and missing metrics | Not applicable |
| WP1 fixed inspection pages | SOFTWARE VERIFIED (`f2885c7`) | PASS: 159 tests and `compileall` against tracked sources; nine Qt page captures reviewed | NOT TESTED |
| WP2 camera and calibration | IN PROGRESS; CG-20 SOFTWARE VERIFIED (`121b2aa`) | PASS for CG-20: 28 camera tests, 161 tests and `compileall` against tracked sources overall | NOT TESTED |
| WP3–WP11 | TODO for remaining program requirements | NOT TESTED as work packages | NOT TESTED |

Baseline failures: clean tracked sources had 156 passing tests and one
test requiring an untracked calibration file; the working copy had 155
passing tests and two menu failures caused by the pre-existing removal of
four navigation buttons. The menu edit was not changed or staged. During
WP1, geometry checks were refined to exclude the reserved pages' absent
central widget and a combo-box popup's hidden, framework-created scrollbars;
the final check covers all populated pages and no outer wrapper.

WP0 files: `tests/test_calibration_and_revisions.py`, hardware checklist,
and the four development inventories. WP1 files: `AIKensa.py`, the shared page resolver,
14 inspection Designer files, Qt regression tests, Designer guidance,
repository rules, checkgates, plan and this handover. CG-20 files:
`camera_service.py`, `cam_init_ic4.py`, `test_camera_lifecycle.py`,
camera assignment guidance, checkgates and [PLANS.md](../../PLANS.md).
See `git show --stat` of the named commits for the exact file list.

Invariants: slots 0-49 and Settings at 50; no inspection algorithm or
approved product parameters changed; no writes to operator configuration,
production databases or hardware. Small logical displays may clip contents
by design. The real display, calibration/geometry, daily results and
trigger-to-record lifecycle still need qualification.

CG-20 repro: `CameraService.acquire()` called the real-path IC4 helper without
the requested `exposure_us=12000.0` or `gain_db=4.5`. The regression patches
that opening helper without injecting `capture_factory`. A second test
verifies that an unavailable SDK value is not substituted with the requested
size or a fallback zero, while a real measured gain of 0 is preserved.
No cameras were opened in these tests.

Final working-tree suite: **FAIL** (159 passed, 2 failed), both from the same
pre-existing menu edit. Tracked-source candidate: **PASS** (161 passed,
0 skipped). GitHub CI **NOT TESTED**; no tracked workflow was found.

**Next: WP2 CG-22.** Add fake-frame worker regressions with different frame
shapes per camera (e.g. 64×48 and 80×60) and different temporary calibration
matrices. Capture the `cv2.initUndistortRectifyMap` arguments in both loops;
prove camera 2 receives its own dimensions and calibration. Replace the
inspection loop's single `mapCalculated[1]` gate with readiness of both
required maps; check calibration metadata and fail closed if the approved
merge geometry cannot support the acquired dimensions. In a missing-map
test, assert that a trigger cannot reach inference, counters or persistence.
Preserve the existing coordinate and rotation conventions; legacy workers
still use the global rotation, and Settings does not persist a differing
per-camera override. Do not change that behavior without an orientation
parity test. CG-21 lifecycle gaps, CG-23 timestamps and CG-24 geometry
unification remain open.
