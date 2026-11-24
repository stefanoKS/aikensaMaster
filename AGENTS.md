# Repository development rules

- New parts must not require product-specific changes in `inspection_thread.py` or `AIKensa.py`.
- Never renumber existing UI indices 0-49.
- Never silently substitute inspection algorithms.
- Preserve original judgment tolerances, calibration, and machine-specific settings.
- Preserve PLC triggers, counters, and database fields.
- Add behavioral tests for inspection changes.
- Separate software validation from hardware validation.
- Report PASS, FAIL, BLOCKED, or NOT TESTED honestly.
- Perform implementation when explicitly requested; do not stop after writing an audit.
- Put part identity and capabilities in the part registry, model settings in
  configuration, and algorithm-specific work in strategies with explicit inputs
  and results. Do not activate incomplete legacy paths.
- Keep the main menu and all production and daily inspection pages on a
  1920×1080 Designer canvas with no outer scrolling or automatic scaling.
  Start fullscreen at the actual screen geometry. Keep Settings layout-based
  and preserve its Designer object names.
- Treat camera placeholders as display-only. Invalid, disconnected, or stale
  frames must not be used for inference or calibration or counted as products.
- Run Settings draft measurements through the owning inspection worker, never
  through ordinary recording, counters, sounds, or PLC completion.
- Preserve local operator configuration, including camera assignments,
  calibrated dimensions, model parameters, and alignment. Use temporary
  configuration and databases plus fake hardware for automated tests.
- Update `PLANS.md` and run the relevant `CHECKGATES.md` checks for each
  independently reviewable milestone before committing.
