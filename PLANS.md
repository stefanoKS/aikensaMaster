# Milestones

## Active sequence — Phase 3C

1. Verify the fixed merge page, native footprint projection and diagnostics
   without modifying production transforms or the measurement canvas.
2. Commission native and scaled coverage, rotation and daily ROI on hardware
   before changing any approved geometry.

## Phase 3B (historical)

1. Baseline: software PASS (233 tests at `6f488af`).
2. Daily Settings: v3 common/position overrides, prepared-image capture,
   ROI verification, and worker-owned draft measurement software verified.
3. Capture daily model and geometry identities. Qualified alternate model
   selection, legacy geometry, recording recovery and production migration
   remain open.

## Phase 3 (historical)

1. Baseline and equipment preview across navigation: software PASS.
2. Typed daily session state and reinspection: software PASS.
3. Harden production geometry (strict preparation verified for widgets
   17–18; other legacy families pending) and effective configuration
   revisions (daily v2 approval and backup verified; artifact identity pending).
4. Complete daily Settings and recording before production migration.

## Phase 2 (historical)

1. CG-P2-00: current main menu and baseline reconciled (software PASS:
   161 tests; machine NOT TESTED).
2. CG-P2-10: independent native rectification and trigger guards implemented;
   production crop/high-homography fail-closed coverage remains open.
3. CG-P2-20 through CG-P2-50: typed configuration, daily strategy, three-position
   workflow, and editable Settings with restart and rollback evidence. Typed
   daily strategy, per-position approved revisions, worker/recording, and
   editing the reference/tolerance/scale are software verified in the
   synthetic workflow. Advanced draft preview and machine checks remain open.
4. CG-P2-60 through CG-P2-80: production migration and preview work, after the
   daily workflow passes.
5. The executable software checkgates in [CHECKGATES.md](./CHECKGATES.md)
   track the verified subset; physical qualification remains separate.

The sections below are historical evidence, not a competing work order.

The program's current work order and evidence are recorded in the
[agent handover](./docs/development/AGENT_HANDOVER.md). WP0's clean-checkout
fixture work and WP1's inspection page policy extend the historical gates
below; neither changes which inspection algorithms run.

1. Establish repository rules and software checkgates.
2. Introduce and validate the widget-17 strategy without migrating other families.
3. Implement generic dynamic presentation, camera geometry, counters, and persistence
   for new profiles after the existing pages; prove end-to-end new-part registration.
4. Validate the migrated path on the physical machine before production release.

Milestone 3 is not implemented. Synthetic coordinator execution alone is not
dynamic UI registration. Milestone 4 requires the actual machine.

## Implementation sequence

A baseline → B main menu → C camera safety → D Settings → E widget-17 strategy
→ F registry → G worker dispatch → H results → I generic UI → K new-part
integration → J legacy migrations → L executable gates → M documentation.
Each change must pass targeted and applicable full software tests, preserve
production configuration, and be committed independently. No incomplete
migration is to be enabled.

### A — Baseline (software PASS)

- Local and remote `main` both started at `ebbf9d66d1ad8590b9a566258bd0650672cbc315`.
- Offscreen suite in the `aikensamaster` Conda environment: 131 passed,
  4 failed, 0 skipped. The failures were all menu tests for four standard
  buttons removed in an existing local edit of `mainPage.ui`; the baseline
  runner in Conda `base` separately lacked PyQt5 and pygame.
- A tracked-only archive of the starting commit ran 134 passed, 1 failed,
  0 skipped. `test_existing_repository_calibration_files_still_load` uses
  local, untracked calibration files and fails in a fresh checkout.
- `compileall` passed. No tracked production configuration differed from HEAD.
- Starting sizes: `AIKensa.py` 1,735 lines, `inspection_thread.py` 4,168,
  `calibration_thread.py` 740. The widget-17 behavior is recorded in
  [CHECKGATES.md](./CHECKGATES.md#widget-17-baseline); parity tests passed.
- Tracked menu canvas: 1920×1190, four standard controls at y=1100, and page 0
  hosted in a scroll area. The local edit already reduced the canvas height
  to 1080 but removed the four controls.
- IC4 has a dummy fallback. Its placeholder renderer is separate from the
  worker fallback, and existing tests do not establish disconnect safety
  across both worker loops.
- Measured offscreen minimum size hints: Settings hub 525×220; camera page
  1362×1006. The hub scrolls internally rather than propagating that camera
  page minimum to the window.

### B — Main menu (software PASS)

- Page 0 is hosted directly; pages 1–49 retain their scroll viewports,
  and Settings remains at index 50. Startup fullscreen is unchanged.
- Designer canvas is 1920×1080. The four standard 220×70 controls now sit
  at x=50, 280, 510 and 740, y=984. Qt verifies containment, no overlaps
  with existing controls, original navigation targets, Settings access,
  no main menu scrollbars, and fullscreen at the actual offscreen size.
- Targeted Settings/Qt suite: 50 passed. Full suite: 135 passed,
  0 failed, 0 skipped. `compileall` passed. No production config edits.

### C — Camera placeholders and safety (software PASS)

- A shared font-independent BGR renderer draws `cam 1 placeholder` or
  `cam 2 placeholder`. Settings, IC4 fallback, and calibration/inspection
  displays use it. Settings reports connection OK only after a valid real
  snapshot and does not take an active inspection lease.
- Worker loops consume fresh validated BGR snapshots only. Failed reads clear
  stale frames and discard pending inspection/calibration actions. Reopening
  a failed camera is rate-limited to one attempt per two seconds. A recovered
  pair reloads calibration; no parameters or production configuration changed.
- Targeted camera/inspection/Qt suite: 88 passed. Full suite: 149 passed,
  0 failed, 0 skipped. Tests include a simulated disconnect during a widget-17
  trigger (no model execution, counter update or SQLite result) and failed
  single/dual calibration runs (no detection or save). `compileall` passed.
- The receive-time skew check uses the IC4 fresh-snapshot contract; physical
  frame timing, reconnect behavior, calibration, and production I/O remain
  **NOT TESTED**.

### D — Settings layout and preview foundation (partial)

- All seven pages and Designer control names remain. The hub has grouped,
  non-selectable headings, a compact readiness summary with a scrollable
  details dialog, and shared styling. Camera tabs retain both unsaved drafts.
- At 1280×720 the hub stays 1280×720. Minimum hints fell to 358×196 for
  the hub and 560×704 for the camera page, with no camera-page horizontal
  scroll. All pages were also exercised at 1600×900 and 1920×1080.
- Only the active adjustment page renders valid live previews. Full-resolution
  crop coordinates are mapped and clamped using merged-image size metadata.
  Fine adjustment identifies the merged-image preview as a reference guide,
  not a verified camera-2 correction. Backup restore and history rollback
  now require confirmation.
- Targeted Settings/camera/inspection suite: 96 passed. Full suite: 157
  passed, 0 failed, 0 skipped. `compileall` passed. Production configuration
  is unchanged.

Next unfinished gate: D5, implement camera-2-only before/after preview from
equivalent source frames using the production transform order and verify
translation and rotation on synthetic markers. Also finish D's page-specific
layout/details (advanced camera controls, conditional Katabu UI, and fuller
inspection/model metadata). Then proceed to E: inject the widget-17
specification explicitly and test the reusable clip/keypoint strategy with
a second independent specification before generalizing registration.
Gates E–M and physical qualification have not been executed.

### WP1 — Fixed inspection pages (software PASS)

- All 36 populated inspection Designer roots are 1920×1080. This includes
  three daily pages and one legacy production page not enabled by navigation.
  Only the root size changed in 14 files; child geometry and names are intact.
- Pages 0–49 are hosted directly, including calibration and reserved pages.
  At 1920×1080 all visible controls fit. Settings retains internal scrolling
  and the same index. Camera and model errors reach the selected page.
- Tracked-source sweep: 159 tests passed; `compileall` passed. Reviewed
  offscreen Qt screenshots of the menu, calibration merge, production,
  clip-picking, Katabu, clip selection and all three daily positions.
- Physical display and hardware operation are **NOT TESTED**.

### WP2 — Camera and calibration correctness (partial)

- CG-20 software **PASS**: the IC4 opening path forwards the saved exposure
  and gain alongside the other camera arguments. The readback display no
  longer uses requested FPS/size or fallback zeroes when SDK values are
  unavailable. Camera lifecycle suite: 28 passed.
- Physical IC4 acceptance and measured image orientation are **NOT TESTED**.
  CG-22 is next; CG-21, CG-23 and CG-24 still need the program-level evidence
  described in the handover.
