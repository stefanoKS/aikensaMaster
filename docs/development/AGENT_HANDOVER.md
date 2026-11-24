# Agent handover

Starting head: `7f031789e8a4b11b9ddd61e14737b4177a4b41eb` on `main`;
verified against GitHub `main`.

| Package | Implementation | Software evidence | Machine |
|---|---|---|---|
| WP0 clean-checkout gate | SOFTWARE VERIFIED (`86f0a3c`) | PASS: 157 tests after replacing the local calibration dependency | NOT TESTED |
| WP0 static metrics | IN PROGRESS | See the [baseline](./PERFORMANCE_BASELINE.md) for measured and missing metrics | Not applicable |
| WP1 fixed inspection pages | SOFTWARE VERIFIED (commit `Host inspection pages directly on the fixed Designer canvas`) | PASS: 159 tests and `compileall` against tracked sources; nine Qt page captures reviewed | NOT TESTED |
| WP2–WP11 | TODO for remaining program requirements | NOT TESTED as work packages | NOT TESTED |

Baseline failures: clean tracked sources had 156 passing tests and one
test requiring an untracked calibration file; the working copy had 155
passing tests and two menu failures caused by the pre-existing removal of
four navigation buttons. The menu edit was not changed or staged. During
WP1, geometry checks were refined to exclude the reserved pages' absent
central widget and a combo-box popup's hidden, framework-created scrollbars;
the final check covers all populated pages and no outer wrapper.

WP0 files: `tests/test_calibration_and_revisions.py`, hardware checklist,
and the four development inventories linked in [PLANS.md](../../PLANS.md)
and this directory. WP1 files: `AIKensa.py`, the shared page resolver,
14 inspection Designer files, Qt regression tests, Designer guidance,
repository rules, checkgates, plan and this handover. See `git show --stat`
of the named commits for the exact file list.

Invariants: slots 0-49 and Settings at 50; no inspection algorithm or
approved product parameters changed; no writes to operator configuration,
production databases or hardware. Small logical displays may clip contents
by design. The real display, calibration/geometry, daily results and
trigger-to-record lifecycle still need qualification.

Next: WP2 CG-20. Add a public-acquire regression test that patches
`open_ic4_camera_or_placeholder` without using `capture_factory`. Save
non-default camera settings in temporary storage and require every requested
argument, including exposure and gain, at the real opening boundary.
Confirm that requested settings and readback are independently represented.
