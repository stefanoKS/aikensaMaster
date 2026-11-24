# Performance and maintainability baseline

Static baseline from tracked `7f031789e8a4b11b9ddd61e14737b4177a4b41eb`.
Counts are physical and nonblank lines plus AST counts. Broad handlers are
bare `except`, `Exception` and `BaseException`; signal counts include all
`pyqtSignal` declarations, not just product-specific ones.

| Module | Lines | Nonblank | Functions | Longest function (lines) | Broad handlers | Signals |
|---|---:|---:|---:|---|---:|---:|
| `aikensa/AIKensa.py` | 1747 | 1459 | 87 | `_setup_ui` (250) | 6 | 0 |
| `aikensa/core/thread/inspection_thread.py` | 4089 | 3423 | 81 | `_run_inspection` (1940) | 54 | 38 |
| `aikensa/core/thread/calibration_thread.py` | 696 | 571 | 19 | `_run_calibration` (334) | 4 | 5 |

At `121b2aa`, both worker modules are unchanged. `AIKensa.py` has 1750
physical lines and 1462 nonblank lines; `_setup_ui` has 249 lines. Its
function, handler and signal counts are unchanged.

Cyclomatic complexity, maintainability index, product-specific dispatch/signal
counts, registry duplication, fixed-size array inventory, coverage and timed
workload benchmarks are **NOT TESTED** for this baseline. Test suite elapsed
time is not an inspection latency benchmark.

Phase 3B performs a one-shot prepared capture only on a Settings request;
the Settings-only worker does not initialize production result storage.
Native preparation/inference latency, memory and live preview throughput
remain **NOT TESTED** as performance metrics.
