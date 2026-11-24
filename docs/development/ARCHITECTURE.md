# Inspection architecture

Keep worker I/O and the inspection algorithm separate. A trigger selects an
approved profile; the profile selects its strategy, models, geometry and
measurement rules. A missing configuration or strategy is an equipment
failure, never a fallback to another inspector.

| Boundary | Existing implementation | Migration rule |
|---|---|---|
| Camera ownership and acquisition | `CameraService`, `CameraStream` | Pass only fresh, validated real frames to geometry. Placeholders are display-only. |
| Geometry | `GeometryPipeline` and legacy worker paths | Prove equivalence before switching a profile. Preserve frame orientation, calibration, anchoring, crop coordinates and color spaces. |
| Configuration | `SettingsService`, its repository and `RevisionManager` | Resolve and validate a coherent active revision; do not rewrite local operator files when introducing defaults. |
| Profile registry | Settings registry and `PartInspectorRegistry` | Consolidate product identity, capabilities, and strategy selection without moving indices 0-49. |
| Inspection | `InspectionCoordinator`, widget-17 strategy and legacy worker branches | Strategies consume prepared inputs and return measurements or an explicit failure; they own no Qt widgets. |
| Results and GUI | Legacy signals/counters/persistence and `AIKensa` | Route one qualified result to presentation, counters and storage. An equipment failure must not count as an inspection. |
| Inspection preview | `EquipmentPreview` and the active Qt page | Publish availability changes once; replay the current diagnostic upon page entry. Tag live images with page activation and equipment sequence. Do not feed rendered diagnostics into the worker's image buffers. |
| Daily Settings measurement | One-shot draft request queued to the inspection worker | Use the same native prepared image, ROI and strategy as actual daily inspection; return only diagnostics, never pass through the result lifecycle. |
| Merge coverage | One-shot request to the calibration worker | Analyze native footprints and render their union separately. The normal preview, Settings preview, calibration saves, and production inference must not use the diagnostic canvas. |

Existing paths remain active until a replacement is verified for that
profile. See the [migration status](./MIGRATION_STATUS.md) for the handoff.
