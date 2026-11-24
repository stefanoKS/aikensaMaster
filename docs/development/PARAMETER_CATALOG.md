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
