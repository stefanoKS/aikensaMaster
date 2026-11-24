"""Group F: inspection safety nets (no false OK, no hardcoded cameras)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

THREADS = {
    "calibration": REPO_ROOT / "aikensa" / "core" / "thread" / "calibration_thread.py",
    "inspection": REPO_ROOT / "aikensa" / "core" / "thread" / "inspection_thread.py",
}
GENERIC_INSPECTOR = REPO_ROOT / "aikensa" / "core" / "scripts" / "generic_part_inspector.py"
DEBUG_PART = (
    REPO_ROOT
    / "aikensa"
    / "core"
    / "config"
    / "parts"
    / "NISSAN"
    / "M_J42U"
    / "P828387YA1A_SEALRRDOORPARTINGLOCK.py"
)


@pytest.mark.parametrize("name", sorted(THREADS))
def test_no_hardcoded_camera_serials_in_threads(name):
    source = THREADS[name].read_text(encoding="utf-8")
    for serial in ("11620526", "11620167"):
        assert serial not in source, f"{name}_thread still hardcodes serial {serial}"


@pytest.mark.parametrize("name", sorted(THREADS))
def test_threads_resolve_cameras_through_the_shared_service(name):
    source = THREADS[name].read_text(encoding="utf-8")
    assert "get_camera_service" in source
    assert "CameraStream(self.camera_service" in source


def test_generic_inspector_never_returns_ok_for_unimplemented_logic(qapp):
    from aikensa.core.scripts.generic_part_inspector import GenericPartInspector

    specs = REPO_ROOT / "aikensa" / "core" / "config" / "parts_specifications.yaml"
    if not specs.is_file():
        pytest.skip("parts_specifications.yaml not present")

    import yaml

    data = yaml.safe_load(specs.read_text(encoding="utf-8")) or {}
    part = next(iter((data.get("parts") or {}).keys()), None)
    if part is None:
        pytest.skip("no part specification available")

    inspector = GenericPartInspector(part)
    with pytest.raises(NotImplementedError):
        inspector.inspect_keypoint_based(None, None)
    with pytest.raises(NotImplementedError):
        inspector.inspect_segmentation_based(None, None)


def test_debug_marking_bypass_is_disabled_by_default():
    source = DEBUG_PART.read_text(encoding="utf-8")
    executable = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "DEBUG_ACCEPT_ANY_MARKING = True" not in executable

    tree = ast.parse(source)
    assignment = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "DEBUG_ACCEPT_ANY_MARKING" for t in node.targets
        ):
            assignment = node
    assert assignment is not None, "DEBUG_ACCEPT_ANY_MARKING definition not found"
    # It must be computed, not a hardcoded True literal.
    assert not (
        isinstance(assignment.value, ast.Constant) and assignment.value.value is True
    )
    assert "AIKENSA_DEBUG_ACCEPT_ANY_MARKING" in source


def test_debug_bypass_logs_when_enabled():
    source = DEBUG_PART.read_text(encoding="utf-8")
    assert "WARNING DEBUG_ACCEPT_ANY_MARKING active" in source


def test_part_registry_covers_the_inspection_widgets():
    import yaml

    registry = yaml.safe_load(
        (REPO_ROOT / "aikensa" / "core" / "config" / "part_registry.yaml").read_text(encoding="utf-8")
    )
    widget_ids = {int(entry["widget_id"]) for entry in registry["parts"]}
    registered = {5, 6, 7, 8, 17, 18, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 48, 49}
    assert registered <= widget_ids

    # Known, documented coverage gap: widgets 44-47 (P5902A510 / P5902A509 /
    # P5819A107 / P8462284S00) are still dispatched by the legacy hardcoded
    # branches in inspection_thread.py.  Recording it here means a future
    # change that silently drops them fails the suite.
    legacy_only = {44, 45, 46, 47}
    assert not (widget_ids & legacy_only), "these widgets unexpectedly moved into the registry"
    assert widget_ids | legacy_only  # no-op, keeps intent readable


def test_cam2_widget_offset_is_composed_not_patched_into_h02():
    import numpy as np
    from aikensa.core.geometry.merge_diagnostic import inspection_cam2_transform
    from aikensa.core.geometry.transforms import translation_matrix

    base = np.array([[1., 0., 8.], [0., 1., 9.], [.1, .2, 1.]])
    adjusted = inspection_cam2_transform(base, (3, -2), None)
    np.testing.assert_allclose(adjusted, translation_matrix(3, -2) @ base)


def test_warp_template_mask_is_built_in_source_coordinates():
    from aikensa.core.scripts.img_processing import cameracalibrate

    source = Path(cameracalibrate.__file__).read_text(encoding="utf-8")
    assert "mask2 = np.ones((h2, w2), dtype=np.uint8) * 255" in source
    assert "np.putmask" not in source


def test_homography_calculation_initialises_matrix_before_use():
    from aikensa.core.scripts.img_processing import cameracalibrate

    source = Path(cameracalibrate.__file__).read_text(encoding="utf-8")
    start = source.index("def calculateHomography_template(")
    body = source[start : source.index("def warpTwoImages(")]
    initialiser = body.index("M = None")
    first_use = body.index("if M is not None")
    assert initialiser < first_use
