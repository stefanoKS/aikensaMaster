# Milestones

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

Next: C, separate display placeholders from camera acquisition. Physical
machine qualification remains **NOT TESTED**.
