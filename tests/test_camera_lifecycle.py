"""Group D: camera lifecycle and ownership."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from aikensa.core.camera.camera_service import (
    CameraBusyError,
    CameraRole,
    CameraService,
    CameraStream,
    CameraUnavailableError,
)
from tests.conftest import write_serial


class FakeCapture:
    """Camera-shaped object without any hardware dependency."""

    _W, _H, _FPS, _EXP, _GAIN, _WB = 3, 4, 5, 15, 14, 45

    instances: list = []

    def __init__(self, identifier, **kwargs):
        self.identifier = identifier
        self.kwargs = kwargs
        self.width = kwargs.get("width", 64)
        self.height = kwargs.get("height", 48)
        self.released = False
        self.reads = 0
        FakeCapture.instances.append(self)

    def read(self):
        self.reads += 1
        return True, np.zeros((self.height, self.width, 3), np.uint8)

    def release(self):
        self.released = True

    def get(self, prop):
        if prop == self._W:
            return float(self.width)
        if prop == self._H:
            return float(self.height)
        if prop == self._FPS:
            return 15.0
        return 0.0


class PlaceholderCapture(FakeCapture):
    """Named like a placeholder so the service recognises it."""


PlaceholderCapture.__name__ = "DummyCapture"


@pytest.fixture()
def service(settings):
    FakeCapture.instances.clear()
    return CameraService(settings, capture_factory=FakeCapture, allow_placeholders=False)


def test_acquire_and_release(service, settings):
    write_serial(settings, {0: "SN-0", 1: "SN-1"})
    handle = service.acquire(0, CameraRole.INSPECTION)
    assert handle.is_placeholder is False
    assert service.is_open(0)
    assert service.state(0) is CameraRole.INSPECTION

    service.release(0, CameraRole.INSPECTION)
    assert not service.is_open(0)
    assert FakeCapture.instances[0].released is True


def test_unassigned_serial_is_refused_with_japanese_reason(service):
    with pytest.raises(CameraUnavailableError) as excinfo:
        service.acquire(0, CameraRole.INSPECTION)

    assert "シリアル番号が未設定" in excinfo.value.reason_ja
    assert service.state(0) is CameraRole.ERROR


def test_two_roles_cannot_own_the_same_camera(service, settings):
    write_serial(settings, {0: "SN-0", 1: "SN-1"})
    FakeCapture.instances.clear()
    service = CameraService(settings, capture_factory=FakeCapture, allow_placeholders=False)

    service.acquire(0, CameraRole.CALIBRATION)

    with pytest.raises(CameraBusyError) as excinfo:
        service.acquire(0, CameraRole.INSPECTION)
    assert "使用中" in excinfo.value.reason_ja

    # The owner can re-acquire its own lease without reopening the device.
    again = service.acquire(0, CameraRole.CALIBRATION)
    assert again.capture is FakeCapture.instances[0]
    assert len(FakeCapture.instances) == 1


def test_same_role_can_leave_and_reenter(service, settings):
    write_serial(settings, {0: "SN-0", 1: "SN-1"})
    FakeCapture.instances.clear()
    service = CameraService(settings, capture_factory=FakeCapture, allow_placeholders=False)

    service.acquire(0, CameraRole.CALIBRATION)
    service.release(0, CameraRole.CALIBRATION)
    service.acquire(0, CameraRole.INSPECTION)

    assert len(FakeCapture.instances) == 2
    assert FakeCapture.instances[0].released is True


def test_placeholder_is_not_a_production_source(settings):
    write_serial(settings, {0: "SN-0", 1: "SN-1"})
    service = CameraService(settings, capture_factory=PlaceholderCapture, allow_placeholders=False)

    with pytest.raises(CameraUnavailableError):
        service.acquire(0, CameraRole.INSPECTION)
    assert service.is_production_source(0) is False


def test_placeholder_allowed_only_in_offline_mode(settings):
    write_serial(settings, {0: "SN-0", 1: "SN-1"})
    service = CameraService(settings, capture_factory=PlaceholderCapture, allow_placeholders=True)

    handle = service.acquire(0, CameraRole.CAMERA_PREVIEW)

    assert handle.is_placeholder is True
    assert service.is_production_source(0) is False
    assert service.read_frame(0, CameraRole.CAMERA_PREVIEW) == (False, None)
    service.release(0, CameraRole.CAMERA_PREVIEW)
    for role in (CameraRole.INSPECTION, CameraRole.CALIBRATION):
        with pytest.raises(CameraUnavailableError):
            service.acquire(0, role)


def test_read_after_release_is_safe(service, settings):
    write_serial(settings, {0: "SN-0", 1: "SN-1"})
    FakeCapture.instances.clear()
    service = CameraService(settings, capture_factory=FakeCapture, allow_placeholders=False)

    service.acquire(0, CameraRole.INSPECTION)
    service.release(0, CameraRole.INSPECTION)

    ok, frame = service.read_frame(0, CameraRole.INSPECTION)
    assert (ok, frame) == (False, None)


def test_release_by_wrong_role_is_refused(service, settings):
    write_serial(settings, {0: "SN-0", 1: "SN-1"})
    FakeCapture.instances.clear()
    service = CameraService(settings, capture_factory=FakeCapture, allow_placeholders=False)

    service.acquire(0, CameraRole.INSPECTION)
    with pytest.raises(CameraBusyError):
        service.release(0, CameraRole.CALIBRATION)
    assert service.is_open(0)


def test_no_simultaneous_read_and_release(service, settings):
    """A release must wait for an in-flight frame read."""

    write_serial(settings, {0: "SN-0", 1: "SN-1"})
    FakeCapture.instances.clear()

    read_started = threading.Event()
    release_finished = threading.Event()

    class SlowCapture(FakeCapture):
        def read(self):
            read_started.set()
            import time

            time.sleep(0.15)
            return super().read()

    service = CameraService(settings, capture_factory=SlowCapture, allow_placeholders=False)
    service.acquire(0, CameraRole.INSPECTION)

    def reader():
        service.read_frame(0, CameraRole.INSPECTION)

    def releaser():
        read_started.wait(timeout=2)
        service.release(0, CameraRole.INSPECTION)
        release_finished.set()

    reader_thread = threading.Thread(target=reader)
    releaser_thread = threading.Thread(target=releaser)
    reader_thread.start()
    releaser_thread.start()
    reader_thread.join(timeout=5)
    releaser_thread.join(timeout=5)

    assert release_finished.is_set()
    # The capture was released only after the read completed.
    assert SlowCapture.instances[0].released is True
    assert SlowCapture.instances[0].reads == 1


def test_configured_settings_reach_the_capture(service, settings):
    cameras, _ = settings.repository.load_camera_settings()
    cameras.assignments[0].serial = "SERIAL-0"
    cameras.assignments[1].serial = "SERIAL-1"
    cameras.hardware_for(0).width_px = 2592
    cameras.hardware_for(0).height_px = 1944
    cameras.hardware_for(0).exposure_us = 12000.0
    cameras.hardware_for(0).gain_db = 4.5
    settings.save_camera_settings(cameras)

    FakeCapture.instances.clear()
    service = CameraService(settings, capture_factory=FakeCapture, allow_placeholders=False)
    handle = service.acquire(0, CameraRole.INSPECTION)

    capture = FakeCapture.instances[0]
    assert capture.identifier == "SERIAL-0"
    assert capture.kwargs["width"] == 2592
    assert capture.kwargs["height"] == 1944
    assert capture.kwargs["exposure_us"] == 12000.0
    assert capture.kwargs["gain_db"] == 4.5
    assert handle.serial == "SERIAL-0"


def test_saved_settings_reach_the_ic4_opening_helper(settings, monkeypatch):
    import sys
    from types import ModuleType
    from unittest.mock import Mock

    cameras, _ = settings.repository.load_camera_settings()
    cameras.assignments[0].serial = "SERIAL-0"
    cameras.assignments[1].serial = "SERIAL-1"
    hardware = cameras.hardware_for(1)
    hardware.width_px, hardware.height_px, hardware.fps = 2592, 1944, 9.0
    hardware.exposure_us, hardware.gain_db = 12000.0, 4.5
    hardware.wb_temperature_k = 5100
    hardware.wb_ratio_r, hardware.wb_ratio_g, hardware.wb_ratio_b = 1.7, 1.1, 2.4
    cameras.rotate_180_global = False
    for index in (0, 1):
        cameras.hardware_for(index).rotate_180 = False
    settings.save_camera_settings(cameras)

    name = "aikensa.core.scripts.cam_init.cam_init_ic4"
    ic4_module = ModuleType(name)
    capture = FakeCapture("SERIAL-1")
    report = {"width": 2500, "exposure_us": 11950.0, "gain_db": None}
    capture.readback_settings = Mock(return_value=report)
    ic4_module.open_ic4_camera_or_placeholder = Mock(return_value=capture)
    monkeypatch.setitem(sys.modules, name, ic4_module)

    service = CameraService(settings)
    handle = service.acquire(1, CameraRole.CAMERA_PREVIEW)
    try:
        ic4_module.open_ic4_camera_or_placeholder.assert_called_once_with(
            "SERIAL-1", camera_index=1, width=2592, height=1944, fps=9.0,
            exposure_us=12000.0, gain_db=4.5, wb_temperature=5100,
            wb_ratio_r=1.7, wb_ratio_g=1.1, wb_ratio_b=2.4, rotate_180=False,
        )
        assert handle.readback == report
    finally:
        service.release(1, CameraRole.CAMERA_PREVIEW)


def test_ic4_readback_keeps_unavailable_measurements_unknown(monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace

    from tests.conftest import REPO_ROOT

    class SDKError(Exception):
        pass

    sdk = SimpleNamespace(
        DeviceInfo=object, PropertyMap=object, IC4Exception=SDKError,
        PropId=SimpleNamespace(WIDTH="width", HEIGHT="height", GAIN="gain", EXPOSURE_TIME="exposure"),
        PixelFormat=SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "imagingcontrol4", sdk)
    spec = importlib.util.spec_from_file_location(
        "_test_ic4_readback", REPO_ROOT / "aikensa/core/scripts/cam_init/cam_init_ic4.py"
    )
    ic4_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ic4_module)

    class Properties:
        def get_value_int(self, name):
            if name == "height":
                return 1944
            raise SDKError(name)

        def get_value_float(self, name):
            if name == "gain":
                return 0.0
            raise SDKError(name)

    capture = ic4_module.IC4Capture.__new__(ic4_module.IC4Capture)
    capture._grab = SimpleNamespace(device_property_map=Properties())
    capture._w, capture._h, capture._fps = 2592, 1944, 9.0
    assert capture.readback_settings() == {
        "width": None, "height": 1944, "fps": None, "exposure_us": None,
        "gain_db": 0.0, "wb_temperature_k": None,
    }


def test_repeated_switching_does_not_leak_handles(settings):
    write_serial(settings, {0: "SN-0", 1: "SN-1"})
    FakeCapture.instances.clear()
    service = CameraService(settings, capture_factory=FakeCapture, allow_placeholders=False)

    for _ in range(5):
        service.acquire(0, CameraRole.CALIBRATION)
        service.release(0, CameraRole.CALIBRATION)
        service.acquire(0, CameraRole.INSPECTION)
        service.release(0, CameraRole.INSPECTION)

    assert service.is_open(0) is False
    assert all(instance.released for instance in FakeCapture.instances)
    assert len(FakeCapture.instances) == 10


@pytest.mark.parametrize("size", [(246, 163), (1280, 960), (1920, 1080)])
def test_camera_placeholders_have_exact_centered_text(monkeypatch, size):
    from aikensa.core.camera import placeholder

    calls = []
    put_text = placeholder.cv2.putText

    def record(image, text, *args):
        calls.append(text)
        return put_text(image, text, *args)

    monkeypatch.setattr(placeholder.cv2, "putText", record)
    width, height = size
    for index in (0, 1):
        image = placeholder.make_camera_placeholder(index, width, height)
        assert image.shape == (height, width, 3)
        ys, xs = np.where(image[:, :, 0] > 100)
        assert abs((xs.min() + xs.max()) / 2 - width / 2) < width * 0.05
        assert abs((ys.min() + ys.max()) / 2 - height / 2) < height * 0.05
    assert calls == ["cam 1 placeholder", "cam 2 placeholder"]


@pytest.mark.parametrize("role", [CameraRole.INSPECTION, CameraRole.CALIBRATION])
def test_stream_drops_failed_frame_then_reopens_with_backoff(monkeypatch, settings, role):
    from aikensa.core.camera import camera_service

    now = [100.0]
    monkeypatch.setattr(camera_service.time, "monotonic", lambda: now[0])
    write_serial(settings, {0: "SN-0", 1: "SN-1"})
    service = CameraService(settings, capture_factory=FakeCapture)
    stream = CameraStream(service, role)
    first, second = stream.read_pair()
    assert (first.index, second.index) == (0, 1)
    initial = service.handle(1).capture
    # A success flag with an invalid format must fail as well.
    initial.read = lambda: (True, np.zeros((2, 2), dtype=np.uint8))
    first, second = stream.read_pair()
    assert first is not None and second is None
    assert initial.released
    assert "カメラ2" in stream.errors[1]
    assert stream.read(1) is None
    assert service.handle(1) is None
    now[0] += 2.1
    first, second = stream.read_pair()
    assert second.index == 1
    assert service.handle(1).capture is not initial
    assert not stream.errors
    stream.close()


def test_stream_drops_first_frame_if_second_arrives_too_late(monkeypatch, service, settings):
    from aikensa.core.camera import camera_service

    now = [100.0]
    monkeypatch.setattr(camera_service.time, "monotonic", lambda: now[0])
    write_serial(settings, {0: "SN-0", 1: "SN-1"})
    stream = CameraStream(service, CameraRole.INSPECTION)
    capture = service.acquire(1, CameraRole.INSPECTION).capture
    real_read = capture.read

    def delayed_read():
        now[0] += 1.5
        return real_read()

    capture.read = delayed_read
    first, second = stream.read_pair()
    assert first is None and second.index == 1
    assert "古すぎます" in stream.errors[0]


@pytest.mark.parametrize("mode", ["unassigned", "placeholder", "busy", "real"])
def test_settings_connection_preview_reports_real_sources_only(qapp, monkeypatch, settings, mode):
    from aikensa.gui.settings_pages import camera_settings_page

    if mode != "unassigned":
        write_serial(settings, {0: "SN-0", 1: "SN-1"})
    factory = PlaceholderCapture if mode == "placeholder" else FakeCapture
    service = CameraService(settings, capture_factory=factory, allow_placeholders=True)
    if mode == "busy":
        inspection_handle = service.acquire(1, CameraRole.INSPECTION)
    monkeypatch.setattr(camera_settings_page, "get_camera_service", lambda _: service)
    page = camera_settings_page.CameraSettingsPage(settings)
    assert not page.camera0PreviewLabel.pixmap().isNull()
    assert not page.camera1PreviewLabel.pixmap().isNull()
    page.test_connection(1)
    assert ("接続OK" in page.camera1StatusLabel.text()) == (mode == "real")
    assert not page.camera1PreviewLabel.pixmap().isNull()
    if mode == "busy":
        assert service.handle(1) is inspection_handle
        assert not inspection_handle.capture.released
    else:
        assert service.owner(1) is None


@pytest.fixture()
def worker_modules(monkeypatch):
    """Import workers without importing model engines or opening the sound device."""
    import sys
    from types import ModuleType
    from unittest.mock import Mock
    import pygame

    monkeypatch.setattr(pygame.mixer, "init", lambda: None)
    monkeypatch.setattr(pygame.mixer, "Sound", Mock())
    yolo = ModuleType("ultralytics")
    yolo.YOLO = Mock()
    sahi = ModuleType("sahi")
    sahi.AutoDetectionModel = Mock()
    prediction = ModuleType("sahi.predict")
    prediction.predict = Mock()
    prediction.get_prediction = Mock()
    prediction.get_sliced_prediction = Mock()
    mysql = ModuleType("mysql")
    mysql.connector = ModuleType("mysql.connector")
    mysql.connector.Error = type("FakeMySQLError", (Exception,), {})
    mysql.connector.connect = Mock(side_effect=RuntimeError("No MySQL in software tests"))
    for name, module in (
        ("ultralytics", yolo), ("sahi", sahi), ("sahi.predict", prediction),
        ("mysql", mysql), ("mysql.connector", mysql.connector),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    from importlib import import_module

    calibration_thread, inspection_thread = (
        import_module(f"aikensa.core.thread.{name}")
        for name in ("calibration_thread", "inspection_thread")
    )

    yield calibration_thread, inspection_thread
    for module in (calibration_thread, inspection_thread):
        sys.modules.pop(module.__name__, None)


@pytest.mark.parametrize("missing", [(0,), (1,), (0, 1)])
@pytest.mark.parametrize("widget", [17, 22])
def test_inspection_run_blocks_inference_and_result_on_camera_loss(
    qapp, settings, monkeypatch, tmp_path, worker_modules, missing, widget,
):
    from unittest.mock import Mock
    from PyQt5.QtCore import QThread

    _, module = worker_modules
    write_serial(settings, {0: "SN-0", 1: "SN-1"})
    service = CameraService(settings, capture_factory=FakeCapture)
    stream = CameraStream(service, CameraRole.INSPECTION)
    assert all(frame is not None for frame in stream.read_pair())
    for index in missing:
        service.handle(index).capture.read = lambda: (False, None)

    worker = module.InspectionThread.__new__(module.InspectionThread)
    QThread.__init__(worker)
    worker.inspection_config = module.InspectionConfig(widget=widget, doInspection=True)
    original_current = [counts.copy() for counts in worker.inspection_config.current_numofPart]
    worker.camera_service = service
    worker.camera_stream = stream
    worker._camera_diagnostic = ("", ())
    worker.camera_ready = True
    worker.running = True
    worker.mergeframe1 = worker.mergeframe2 = np.ones((4, 4, 3), np.uint8)
    worker.widget_dir_map = {}
    worker.initialize_model = Mock()
    worker._load_geometry_from_settings = Mock()
    worker._check_date_rollover = Mock()
    worker.downSampling = Mock()
    worker.inspection_coordinator = Mock()
    worker.NICHIJOU_TENKEN_Model = Mock()
    iterations = iter((True, True, False))
    worker.msleep = lambda _: setattr(worker, "running", next(iterations))
    for name in ("mysqlHost", "mysqlID", "mysqlPassword", "mysqlHostPort"):
        setattr(worker, name, None)
    errors = []
    worker.cameraErrorSignal.connect(errors.append)
    states = []
    worker.equipmentPreviewSignal.connect(lambda sequence, state: states.append((sequence, state)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.cv2, "imread", lambda _: np.zeros((10, 10, 3), np.uint8))

    worker.run()
    assert worker.mergeframe1 is None and worker.mergeframe2 is None
    assert not worker.inspection_config.doInspection
    assert worker.inspection_config.current_numofPart == original_current
    worker.downSampling.assert_not_called()
    worker.inspection_coordinator.execute.assert_not_called()
    worker.NICHIJOU_TENKEN_Model.assert_not_called()
    assert len(errors) == 1
    assert [state.readiness.name for _, state in states] == ["CAMERA_UNAVAILABLE", "CONNECTING"]
    assert states[0][1].missing_cameras == missing
    assert [camera.name for camera in states[0][1].cameras] == [
        "READ_FAILED" if index in missing else "AVAILABLE" for index in range(2)
    ]
    for index in missing:
        assert f"カメラ{index + 1}" in errors[0]
    try:
        assert worker.conn.execute("SELECT count(*) FROM inspection_results").fetchone() == (0,)
    finally:
        worker.conn.close()


@pytest.mark.parametrize("page", [1, 2, 3])
def test_calibration_run_drops_invalid_frames_and_save_requests(
    qapp, settings, monkeypatch, tmp_path, worker_modules, page,
):
    from unittest.mock import Mock
    from PyQt5.QtCore import QThread

    module, _ = worker_modules
    worker = module.CalibrationThread.__new__(module.CalibrationThread)
    QThread.__init__(worker)
    worker.calib_config = module.CalibrationConfig(
        widget=page, calculateSingeFrameMatrix=True, calculateCamMatrix=True,
        calculateHomo_cam1=True, calculateHomo_cam2=True,
        savePlanarize=True, savePlanarizeHigh=True,
    )
    worker.settings = settings
    worker.camera_stream = CameraStream(CameraService(settings, capture_factory=FakeCapture), CameraRole.CALIBRATION)
    worker._load_calibration_data_from_settings = Mock()
    worker._persist_homography = Mock()
    worker._persist_planarize = Mock()
    worker.running = True
    worker.scale_factor = 5.0
    worker.frame_width, worker.frame_height = 10, 10
    worker.frame = worker.mergeframe1 = worker.mergeframe2 = np.ones((4, 4, 3), np.uint8)
    worker.msleep = lambda _: setattr(worker, "running", False)
    for name in ("detectCharucoBoard", "calculatecameramatrix", "calculateHomography_template"):
        monkeypatch.setattr(module, name, Mock())
    errors = []
    worker.cameraErrorSignal.connect(errors.append)
    previews = []
    worker.CalibCamStream.connect(lambda index, image: previews.append((index, image)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.cv2, "imread", lambda _: np.zeros((10, 10, 3), np.uint8))

    worker.run()
    assert len(errors) == 1
    for name in ("detectCharucoBoard", "calculatecameramatrix", "calculateHomography_template"):
        getattr(module, name).assert_not_called()
    worker._persist_homography.assert_not_called()
    worker._persist_planarize.assert_not_called()
    if page == 3:
        assert worker.mergeframe1 is None and worker.mergeframe2 is None
        assert not worker.calib_config.savePlanarize
    else:
        assert worker.frame is None
        assert not worker.calib_config.calculateCamMatrix
        assert len(previews) == 1 and previews[0][0] == page - 1
        assert not previews[0][1].isNull()


@pytest.fixture()
def camera_geometry(settings):
    sizes = ((320, 240), (400, 300))
    cameras, _ = settings.repository.load_camera_settings()
    for index, (width, height) in enumerate(sizes):
        cameras.assignment(index).serial = f"SN-{index}"
        hardware = cameras.hardware_for(index)
        hardware.width_px, hardware.height_px = width, height
    settings.repository.save_camera_settings(cameras)
    matrices = [np.array([[50.0 + i, 0, 32], [0, 50, 24], [0, 0, 1]]) for i in (0, 1)]
    for index, matrix in enumerate(matrices):
        settings.repository.save_intrinsic(index, matrix, np.zeros(5))
        settings.repository.save_intrinsic(index, matrix, np.zeros(5), scaled=True)
        settings.repository.save_homography(index, np.eye(3))
        settings.repository.save_homography(index, np.eye(3), scaled=True)

    def capture_factory(identifier, **kwargs):
        index = int(identifier[-1])
        kwargs["width"], kwargs["height"] = sizes[index]
        return FakeCapture(identifier, **kwargs)

    return CameraService(settings, capture_factory=capture_factory), sizes, matrices


@pytest.mark.parametrize("widget", [17, 22])
@pytest.mark.parametrize("problem", [
    "missing_cam1", "missing_cam2", "scaled", "homography", "scaled_homography",
    "invalid_matrix", "invalid_distortion", "acquired_size",
])
def test_inspection_run_blocks_cycle_and_record_on_geometry_error(
    qapp, settings, monkeypatch, tmp_path, worker_modules, camera_geometry, widget, problem,
):
    from unittest.mock import Mock
    from PyQt5.QtCore import QThread

    service, _, _ = camera_geometry
    if problem == "missing_cam1":
        settings.paths.calibration_file(0).unlink()
    elif problem == "missing_cam2":
        settings.paths.calibration_file(1).unlink()
    elif problem == "scaled":
        settings.paths.calibration_scaled_file(0).unlink()
    elif problem in ("homography", "scaled_homography"):
        settings.paths.homography_file(0, scaled=(problem == "scaled_homography")).write_text(
            "[[1, 0, 0], [0, 0, 0], [0, 0, 1]]", encoding="utf-8"
        )
    elif problem == "acquired_size":
        cameras, _ = settings.repository.load_camera_settings()
        cameras.hardware_for(0).width_px += 4
        settings.repository.save_camera_settings(cameras)
        settings.record_calibration_metadata(0)
    else:
        import yaml

        path = settings.paths.calibration_file(0)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if problem == "invalid_matrix":
            payload["camera_matrix"][0][0] = -1
        else:
            payload["distortion_coefficients"] = [0.0]
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    _, module = worker_modules
    worker = module.InspectionThread.__new__(module.InspectionThread)
    QThread.__init__(worker)
    worker.inspection_config = module.InspectionConfig(widget=widget, doInspection=True)
    original_counts = [counts.copy() for counts in worker.inspection_config.current_numofPart]
    worker.settings = settings
    worker.camera_service = service
    worker.camera_stream = CameraStream(service, CameraRole.INSPECTION)
    worker._camera_diagnostic = ("", ())
    worker.running = True
    worker._last_color_debug_ts = float("inf")
    worker._color_debug_interval_sec = 1
    worker.scaled_width = worker.scaled_height = 8
    worker.widget_dir_map = {}
    worker.inspection_widget_indices = [widget]
    worker.initialize_model = Mock()
    worker._check_date_rollover = Mock()
    worker._begin_inspection_cycle = Mock()
    worker.msleep = lambda _: setattr(worker, "running", False)
    for name in ("mysqlHost", "mysqlID", "mysqlPassword", "mysqlHostPort"):
        setattr(worker, name, None)
    errors = []
    worker.cameraErrorSignal.connect(errors.append)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.cv2, "imread", lambda _: np.zeros((40, 40, 3), np.uint8))

    worker.run()
    try:
        worker._begin_inspection_cycle.assert_not_called()
        assert not worker.inspection_config.doInspection
        assert worker.inspection_config.current_numofPart == original_counts
        assert len(errors) == 1 and errors[0]
        assert worker.conn.execute("SELECT count(*) FROM inspection_results").fetchone() == (0,)
    finally:
        worker.conn.close()


@pytest.mark.parametrize("role", [CameraRole.CALIBRATION, CameraRole.INSPECTION])
def test_workers_build_maps_at_each_source_frame_size(
    qapp, settings, monkeypatch, worker_modules, camera_geometry, role,
):
    from PyQt5.QtCore import QThread

    service, sizes, matrices = camera_geometry
    calibration, inspection = worker_modules
    module = calibration if role is CameraRole.CALIBRATION else inspection
    calls = []
    orig = module.cv2.initUndistortRectifyMap

    def capture_maps(matrix, coeffs, rotation, new_matrix, size, dtype):
        calls.append((matrix.copy(), size))
        return orig(matrix, coeffs, rotation, new_matrix, size, dtype)

    monkeypatch.setattr(module.cv2, "initUndistortRectifyMap", capture_maps)
    if role is CameraRole.CALIBRATION:
        worker = module.CalibrationThread.__new__(module.CalibrationThread)
        QThread.__init__(worker)
        worker.calib_config = module.CalibrationConfig(widget=3, calculateHomo_cam1=True)
        worker._persist_homography = lambda *_args, **_kwargs: False
        for name in ("H1_high", "H2_high", "H1_high_scaled", "H2_high_scaled"):
            setattr(worker, name, None)
        worker.frame_width, worker.frame_height = sizes[0]
        worker.scale_factor = 5
        worker.running = True
        worker.CamMergeAll.connect(lambda *_: setattr(worker, "running", False))
        monkeypatch.setattr(module.cv2, "imread", lambda _: np.zeros((40, 40, 3), np.uint8))
        monkeypatch.setattr(module, "calculateHomography_template", lambda *_: (None, None))
    else:
        worker = module.InspectionThread.__new__(module.InspectionThread)
        QThread.__init__(worker)
        worker.inspection_config = module.InspectionConfig(widget=17)
        worker._camera_diagnostic = ("", ())
        worker.homography_size = (40, 40)
        worker.homography_size_scaled = (8, 8)
    worker.settings = settings
    worker.camera_service = service
    worker.camera_stream = CameraStream(service, role)
    if role is CameraRole.CALIBRATION:
        worker.run()
    else:
        try:
            worker._load_geometry_from_settings()
            assert worker._capture_inspection_frames()
            worker.mergeframe1_scaled = worker.downSampling(worker.mergeframe1, 8, 8)
            worker.mergeframe2_scaled = worker.downSampling(worker.mergeframe2, 8, 8)
            assert worker._prepare_camera_maps(), worker._camera_diagnostic
        finally:
            worker.release_all_camera()
    assert [size for _, size in calls[:2]] == list(sizes)
    for (actual, _), expected in zip(calls, matrices):
        np.testing.assert_array_equal(actual, expected)


def test_merge_worker_exports_one_capture_without_changing_calibration(
    qapp, settings, monkeypatch, tmp_path, worker_modules, camera_geometry,
):
    import json
    from PyQt5.QtCore import QThread

    module, _ = worker_modules
    service, sizes, _ = camera_geometry
    settings.repository.save_homography(0, np.array([[1., 0., -30.], [0., 1., 0.], [0., 0., 1.]]))
    settings.repository.save_homography(1, np.array([[1., 0., 350.], [0., 1., 0.], [0., 0., 1.]]))
    paths = [settings.paths.homography_file(i, scaled=scaled) for i in range(2) for scaled in (False, True)]
    before = [path.read_bytes() for path in paths]
    worker = module.CalibrationThread.__new__(module.CalibrationThread)
    QThread.__init__(worker)
    worker.calib_config = module.CalibrationConfig(widget=3, coverage_request=(1, None, True))
    worker.frame_width, worker.frame_height = sizes[0]
    worker.scale_factor = 5
    worker.running = True
    worker.settings = settings
    worker.camera_service = service
    worker.camera_stream = CameraStream(service, CameraRole.CALIBRATION)
    for name in ("H1_high", "H2_high", "H1_high_scaled", "H2_high_scaled"):
        setattr(worker, name, None)
    cv_read = module.cv2.imread
    monkeypatch.setattr(module.cv2, "imread", lambda _: np.zeros((400, 600, 3), np.uint8))
    monkeypatch.chdir(tmp_path)
    published, diagnostics = [], []
    worker.CamMergeAll.connect(lambda image, size: published.append((image.size(), size)))

    def received(serial, image, report):
        diagnostics.append((serial, image, report))
        worker.running = False

    worker.mergeCoverageSignal.connect(received)
    worker.run()
    assert len(diagnostics) == 1 and diagnostics[0][0] == 1
    assert not diagnostics[0][1].isNull()
    assert published[0][0].width() == 120 and published[0][0].height() == 80
    assert published[0][1].width() == 600 and published[0][1].height() == 400
    results = list((settings.paths.root / "aikensa/diagnostic_results").iterdir())
    assert len(results) == 1
    payload = json.loads((results[0] / "report.json").read_text(encoding="utf-8"))
    assert payload["coverage"]["footprints"][0]["overflow"] == pytest.approx([30, 0, 0, 0])
    assert payload["coverage"]["footprints"][1]["overflow"] == pytest.approx([0, 150, 0, 0])
    assert payload["image_sizes_wh"]["camera1_native"] == [320, 240]
    assert payload["image_sizes_wh"]["camera2_native"] == [400, 300]
    assert cv_read(str(results[0] / "fixed_merge.png")).shape[:2] == (400, 600)
    assert [path.read_bytes() for path in paths] == before
    assert not (tmp_path / "combinedImagev2.png").exists()


@pytest.mark.parametrize("widget", [17, 18])
@pytest.mark.parametrize("problem", ["crop", "remap", "warp"])
def test_production_geometry_failure_skips_cycle_and_worker_recovers(
    qapp, settings, monkeypatch, tmp_path, worker_modules, camera_geometry, widget, problem,
):
    from unittest.mock import Mock
    from PyQt5.QtCore import QThread
    from aikensa.core.camera.preview_state import PreviewReadiness

    _, module = worker_modules
    service, _, _ = camera_geometry
    worker = module.InspectionThread.__new__(module.InspectionThread)
    QThread.__init__(worker)
    worker.inspection_config = module.InspectionConfig(
        widget=widget, doInspection=True, preview_target=(widget, 1),
    )
    worker.inspection_config.current_numofPart[widget] = [7, 2]
    worker.settings = settings
    worker.camera_service = service
    worker.camera_stream = CameraStream(service, CameraRole.INSPECTION)
    worker._camera_diagnostic = ("", ())
    worker.running = True
    worker._last_color_debug_ts = float("inf")
    worker._color_debug_interval_sec = 1
    worker.scaled_width, worker.scaled_height = 80, 60
    worker.scale_factor = 5
    worker.wide_planarize = (40, 640)
    worker.widget_h2_offsets = {}
    worker.widget_dir_map = {}
    worker.inspection_widget_indices = [widget]
    worker.inspection_widget_katabu = []
    worker.narrow_height_widget = [17, 18]
    worker.InspectionImages, worker.InspectionImages_bgr = [None], [None]
    worker.InspectionTimeStart = 0
    worker.InspectionWaitTime = 0
    worker.bool_keep_measurement = False
    for name in ("mysqlHost", "mysqlID", "mysqlPassword", "mysqlHostPort"):
        setattr(worker, name, None)
    for name in ("initialize_model", "_check_date_rollover", "_handle_manual_adjustment_and_reset",
                 "_emit_kensa_status", "_finalize_inspection_common", "_check_model_available",
                 "inspection_coordinator"):
        setattr(worker, name, Mock())
    crop = [0, 0, 640, 0, 640, 40, 0, 40]
    worker.widget_crop_points = {widget: [0] * 8 if problem == "crop" else crop}
    failures = []

    def on_equipment(_sequence, state):
        if state.readiness is PreviewReadiness.GEOMETRY_ERROR:
            failures.append(state.message)
            worker.widget_crop_points[widget] = crop
        elif state.readiness is PreviewReadiness.READY and failures:
            worker.running = False

    worker.equipmentPreviewSignal.connect(on_equipment)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.cv2, "imread", lambda _: np.zeros((40, 640, 3), np.uint8))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    if problem == "remap":
        remap = module.cv2.remap

        def fail_full_remap(image, *args, **kwargs):
            if image.shape[0] > 60 and not failures:
                raise module.cv2.error("injected full-resolution remap failure")
            return remap(image, *args, **kwargs)

        monkeypatch.setattr(module.cv2, "remap", fail_full_remap)
    elif problem == "warp":
        warp = module.warpTwoImages_template

        def fail_full_warp(canvas, *args, **kwargs):
            if canvas.shape[1] == 640 and not failures:
                raise module.cv2.error("injected full-resolution warp failure")
            return warp(canvas, *args, **kwargs)

        monkeypatch.setattr(module, "warpTwoImages_template", fail_full_warp)

    def count_loop():
        assert count_loop.calls < 3, "worker did not recover after failed preparation"
        count_loop.calls += 1

    count_loop.calls = 0
    worker._check_date_rollover = count_loop
    worker.run()
    try:
        assert len(failures) == 1 and "全解像度" in failures[0]
        assert worker.inspection_config.current_numofPart[widget] == [7, 2]
        assert not worker.inspection_config.doInspection
        worker.inspection_coordinator.inspect.assert_not_called()
        worker._check_model_available.assert_not_called()
        worker._finalize_inspection_common.assert_not_called()
        assert worker.conn.execute("SELECT count(*) FROM inspection_results").fetchone() == (0,)
    finally:
        worker.conn.close()


@pytest.mark.parametrize("hardware_rotation", [False, True])
def test_strict_production_geometry_preserves_native_pixels_and_color_order(
    qapp, worker_modules, hardware_rotation,
):
    from PyQt5.QtCore import QThread

    _, module = worker_modules
    worker = module.InspectionThread.__new__(module.InspectionThread)
    QThread.__init__(worker)
    worker.inspection_config = module.InspectionConfig(widget=17)
    worker.InspectionImages, worker.InspectionImages_bgr = [None], [None]
    worker.ic4_rotate_180 = hardware_rotation
    worker.wide_planarize = (6, 10)
    worker.planarizeTransform_wide = None
    worker.mergeframe1 = np.zeros((4, 5, 3), np.uint8)
    worker.mergeframe2 = np.zeros((4, 5, 3), np.uint8)
    worker.mergeframe1[1, 1] = (10, 20, 30)
    worker.mergeframe2[1, 1] = (40, 50, 60)
    worker.H1 = np.eye(3)
    worker.H2 = np.eye(3)
    worker.widget_h2_offsets = {17: (5, 0)}
    worker.widget_crop_points = {17: [0, 0, 10, 0, 10, 4, 0, 4]}
    worker.homography_blank_canvas = np.zeros((6, 10, 3), np.uint8)
    y, x = np.indices((4, 5), dtype=np.float32)
    for index in (0, 1):
        worker.inspection_config.map1[index], worker.inspection_config.map2[index] = module.cv2.convertMaps(
            x, y, module.cv2.CV_16SC2
        )

    assert worker._prepare_production_image(17)
    expected = np.zeros((4, 10, 3), np.uint8)
    marker_y, marker_x = (1, 1) if hardware_rotation else (2, 3)
    expected[marker_y, marker_x] = (10, 20, 30)
    expected[marker_y, marker_x + 5] = (40, 50, 60)
    np.testing.assert_array_equal(worker.InspectionImages[0], expected)
    np.testing.assert_array_equal(worker.InspectionImages_bgr[0], expected[:, :, ::-1])


def test_daily_settings_change_drives_real_worker_and_rolls_back(
    qapp, settings, monkeypatch, tmp_path, worker_modules, camera_geometry,
):
    from shutil import copyfile
    import os
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock
    import sqlite3
    from queue import Queue
    from threading import Event

    from PyQt5.QtCore import QEventLoop, QThread, QTimer
    from aikensa.core.inspection.daily_identity import DailyModelArtifact
    from aikensa.core.inspection.daily_session import DailyPositionState, RecordingState
    from aikensa.core.settings.service import SettingsService
    from aikensa.gui.settings_pages import SettingsController
    from aikensa.gui.settings_pages import daily_inspection_settings_page
    from tests.conftest import REPO_ROOT

    service, _, _ = camera_geometry
    _, module = worker_modules
    config_dir = settings.paths.root / "aikensa/core/config"
    for name in ("part_registry.yaml", "parts_specifications.yaml", "models.yaml"):
        copyfile(REPO_ROOT / "aikensa/core/config" / name, config_dir / name)

    worker = module.InspectionThread.__new__(module.InspectionThread)
    QThread.__init__(worker)
    worker.inspection_config = module.InspectionConfig(widget=22, doInspection=True)
    worker.inspection_config.current_numofPart[17] = [7, 2]
    worker.settings = settings
    worker.camera_service = service
    worker.camera_stream = CameraStream(service, CameraRole.INSPECTION)
    worker._camera_diagnostic = ("", ())
    worker.running = True
    worker._last_color_debug_ts = float("inf")
    worker._color_debug_interval_sec = 1
    worker.scaled_width, worker.scaled_height = 80, 60
    worker.scale_factor = 5
    worker.wide_planarize = (431, 6550)
    worker.widget_crop_points = {
        position: [0, 0, 6528, 0, 6528, 431, 0, 431] for position in (21, 22, 23)
    }
    worker.widget_h2_offsets = {}
    worker._load_widget_crop_points = Mock()
    worker._load_widget_h2_offsets = Mock()
    worker.widget_dir_map = {}
    worker.inspection_widget_indices = [21, 22, 23]
    worker.inspection_widget_katabu = []
    worker.initialize_model = Mock()
    worker._check_date_rollover = Mock()
    worker.InspectionTimeStart = 0
    worker.InspectionWaitTime = 0
    worker.bool_keep_measurement = False
    worker._daily_drafts = Queue(maxsize=1)
    worker._emit_kensa_status = Mock()
    worker.msleep = lambda _: setattr(worker, "running", False)
    for name in ("mysqlHost", "mysqlID", "mysqlPassword", "mysqlHostPort"):
        setattr(worker, name, None)

    calls = []

    def infer(*, source, conf, imgsz, verbose):
        calls.append((source.shape, conf, imgsz, verbose, QThread.currentThread()))
        if worker.inspection_config.widget > 0:
            worker.inspection_config.widget = 23
        return [SimpleNamespace(keypoints=SimpleNamespace(xy=np.array([[[0, 1], [632, 4]]])))]

    infer.task = "pose"
    worker.NICHIJOU_TENKEN_Model = infer
    worker._loaded_daily_model_path = settings.load_daily_defaults().profile_for_widget(22).model_path
    signals = []
    live_previews = []
    worker.inspectionPreviewSignal.connect(lambda *event: live_previews.append(event))

    def on_result(update, image):
        signals.append((update, image))
        if update.state is not DailyPositionState.INSPECTING:
            worker.running = False

    worker.dailyResultSignal.connect(on_result)
    monkeypatch.chdir(tmp_path)
    artifact = Path(worker._loaded_daily_model_path)
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"synthetic model identifier")
    worker._daily_model_artifact = DailyModelArtifact.read(worker._loaded_daily_model_path)
    monkeypatch.setattr(module.cv2, "imread", lambda _: np.zeros((431, 6550, 3), np.uint8))
    worker.save_debug_image = Mock()
    from aikensa.core.config.parts import NICHIJOU_TENKEN

    monkeypatch.setattr(
        NICHIJOU_TENKEN, "kanjiFontPath",
        str(REPO_ROOT / "aikensa/gui/resources/font/NotoSansJP-ExtraBold.ttf"),
    )
    monkeypatch.setattr(module, "play_ok_sound", Mock())
    monkeypatch.setattr(module, "play_ng_sound", Mock())
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    def drafts(controller, *, missing_model=False, capture_only=False):
        page = controller.daily_page
        worker.inspection_config.widget = 0
        worker.inspection_config.preview_target = (50, 10)
        worker.inspection_config.doInspection = False
        # No SQL, production outputs, or counters should be consulted by a cold draft-only worker.
        worker._initialize_inspection_records = Mock(side_effect=AssertionError("draft touched ordinary records"))
        session_events = len(signals)
        debug_saves = worker.save_debug_image.call_count
        counts = [entry.copy() for entry in worker.inspection_config.current_numofPart]
        completed = Event()
        loop = QEventLoop()
        watchdog = QTimer()
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(loop.quit)

        def on_draft(result):
            page.show_draft_result(result)
            worker.running = False
            completed.set()
            loop.quit()

        worker.dailyDraftSignal.connect(on_draft)
        page.set_draft_submitter(worker.request_daily_draft)
        original_model = worker.NICHIJOU_TENKEN_Model
        if missing_model:
            worker.NICHIJOU_TENKEN_Model = None
        before_calls = len(calls)
        try:
            (page.dailyCaptureButton if capture_only else page.dailyTestButton).click()
            watchdog.start(10000)
            loop.exec_()
            assert completed.is_set()
            assert page.dailyFullPreviewLabel.pixmap() is not None
            assert page.dailyCropPreviewLabel.pixmap() is not None
            assert "実効 (2100, 0, 900, 431)" in page.dailyTestResultLabel.text()
            if missing_model:
                assert "MODEL_ERROR" in page.dailyTestResultLabel.text()
            elif not capture_only:
                assert "設定テスト結果（記録なし）" in page.dailyTestResultLabel.text()
                assert "OK / 実測 101.4 mm" in page.dailyTestResultLabel.text()
                assert calls[-1][:4] == ((431, 900, 3), 0.55, 1600, False)
                assert calls[-1][4] is worker
            if missing_model or capture_only:
                assert len(calls) == before_calls
            assert len(signals) == session_events
            assert worker.save_debug_image.call_count == debug_saves
            assert worker.inspection_config.current_numofPart == counts
            worker._initialize_inspection_records.assert_not_called()
        finally:
            watchdog.stop()
            worker.running = False
            assert worker.wait(5000)
            worker.dailyDraftSignal.disconnect(on_draft)
            worker.NICHIJOU_TENKEN_Model = original_model
            del worker._initialize_inspection_records

    def screenshot(name):
        destination = os.environ.get("AIKENSA_DAILY_SETTINGS_EVIDENCE")
        if destination:
            controller.resize(1920, 1080)
            controller.show()
            qapp.processEvents()
            image = controller.grab()
            assert (image.width(), image.height()) == (1920, 1080)
            assert image.save(str(Path(destination) / f"{name}.png"))

    def run_cycle(widget, expected_status, expected_revision):
        worker.inspection_config.widget = widget
        worker.inspection_config.preview_target = (widget, 8)
        worker.inspection_config.doInspection = True
        worker.InspectionTimeStart = 0
        worker.running = True
        signals.clear()
        live_previews.clear()
        worker.run()
        worker.conn.close()
        assert worker.inspection_config.current_numofPart[17] == [7, 2]
        assert len(signals) == 2
        assert live_previews and live_previews[0][:2] == (widget, 8)
        assert not live_previews[0][3].isNull()
        assert signals[0][0].widget == signals[1][0].widget == widget
        assert signals[0][0].cycle_id == signals[1][0].cycle_id
        assert signals[1][1].width() == 1791 and signals[1][0].state.name == expected_status
        assert signals[1][0].local is RecordingState.RECORDED
        root = tmp_path / "aikensa/inspection_results"
        con = sqlite3.connect(root / "database_results.db")
        try:
            row = con.execute("""
                SELECT i.partName, i.detected_pitch, i.status, m.profile_id, m.revision_id, m.actual_mm,
                       m.raw_path, m.result_path, m.profile_digest, m.model_digest, m.geometry_digest
                FROM inspection_results i JOIN daily_inspection_metadata m ON i.id = m.result_id
                ORDER BY i.id DESC LIMIT 1
            """).fetchone()
        finally:
            con.close()
        profile = f"dailyTenken_0{widget - 20}"
        assert row[:6] == (profile, "[101.4]", expected_status, profile, expected_revision, 101.4)
        for name in row[6:8]:
            path = tmp_path / "aikensa/inspection_results" / name
            assert path.is_file() and path.parts[-5] == profile
        annotated = module.cv2.imdecode(np.fromfile(root / row[7], np.uint8), module.cv2.IMREAD_COLOR)
        assert annotated is not None and np.count_nonzero(annotated[:, :150]) > 0
        assert row[8] == signals[1][0].profile.configuration_digest
        assert row[9] == worker._daily_model_artifact.digest
        assert len(row[10]) == 64
        return row[10]

    original_geometry = run_cycle(22, "NG", "shared-defaults")
    monkeypatch.setattr(daily_inspection_settings_page, "confirm_change", lambda *_: True)
    controller = SettingsController(settings)
    controller.select_page(6)
    page = controller.daily_page
    assert controller.settingsNavList.currentItem().text() == "日常点検設定"
    page.dailyPositionCombo.setCurrentIndex(page.dailyPositionCombo.count() - 1)
    screenshot("01-common-scope")
    page.dailyPositionCombo.setCurrentIndex(1)
    screenshot("02-position-2")
    page.dailyToleranceEdit.setText("2.0")
    page.dailyWidthEdit.setText("900")
    page.dailyConfidenceEdit.setText("0.55")
    page.dailyImageSizeEdit.setText("1600")
    page.dailyDiagnosticCheck.setChecked(False)
    page.dailyOperatorEdit.setText("engineer")
    page.dailyReasonEdit.setText("Verify synthetic CAD reference")
    page.dailyConfidenceEdit.setText("1.1")
    page.dailyValidateButton.click()
    assert "設定を確認" in page.dailyDraftLabel.text()
    screenshot("03-invalid-confidence")
    page.dailyConfidenceEdit.setText("0.55")
    page.dailyValidateButton.click()
    assert "下書きは有効" in page.dailyDraftLabel.text()
    run_cycle(22, "NG", "shared-defaults")
    initial_images = sorted((tmp_path / "aikensa/inspection_results").rglob("*.png"))
    with sqlite3.connect(tmp_path / "aikensa/inspection_results/database_results.db") as connection:
        ordinary_results = connection.execute("SELECT count(*) FROM inspection_results").fetchone()
    assert calls[-1][:4] == ((431, 1890, 3), 0.4, 1980, False)
    page.dailySaveButton.click()
    assert "基準画像で確認" in page.dailyDraftLabel.text()
    drafts(controller, capture_only=True)
    screenshot("04-prepared-image-and-roi")
    # A valid old capture is not approval evidence after a calibration change.
    settings.repository.save_homography(0, np.array([[1., 0., 1.], [0., 1., 0.], [0., 0., 1.]]))
    page.dailySaveButton.click()
    assert "基準画像で確認" in page.dailyDraftLabel.text()
    assert settings.load_daily_configuration().revision_id == "shared-defaults"
    settings.repository.save_homography(0, np.eye(3))
    drafts(controller, missing_model=True)
    screenshot("05-model-unavailable")
    drafts(controller)
    screenshot("06-draft-measurement")
    assert sorted((tmp_path / "aikensa/inspection_results").rglob("*.png")) == initial_images
    with sqlite3.connect(tmp_path / "aikensa/inspection_results/database_results.db") as connection:
        assert connection.execute("SELECT count(*) FROM inspection_results").fetchone() == ordinary_results
    assert module.play_ok_sound.call_count == 0 and module.play_ng_sound.call_count == 2
    page.dailySaveButton.click()
    revision_id = settings.load_daily_configuration().revision_id
    assert revision_id != "shared-defaults" and revision_id in page.dailyRevisionLabel.text()

    for widget, expected in ((21, "NG"), (22, "OK"), (23, "NG")):
        diagnostics = worker.save_debug_image.call_count
        run_cycle(widget, expected, revision_id)
        assert calls[-1][:4] == (
            ((431, 900, 3), 0.55, 1600, False) if widget == 22 else ((431, 1890, 3), 0.4, 1980, False)
        )
        assert worker.save_debug_image.call_count == diagnostics + int(widget != 22)
    assert settings.load_daily_configuration().profile_for_widget(22).reference.mm_per_px == 0.1604477
    worker.settings = SettingsService(settings.paths)
    assert run_cycle(22, "OK", revision_id) == original_geometry
    worker.widget_h2_offsets[22] = (2.0, 0.0)
    assert run_cycle(22, "OK", revision_id) != original_geometry
    page.dailyHistoryCombo.setCurrentIndex(0)
    page.dailyRollbackButton.click()
    reverted = worker.settings.load_daily_configuration().revision_id
    assert reverted != revision_id
    run_cycle(22, "NG", reverted)
    assert module.play_ok_sound.call_count == 3 and module.play_ng_sound.call_count == 5
