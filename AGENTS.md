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
