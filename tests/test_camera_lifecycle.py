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


@pytest.mark.parametrize("missing", [(1,), (0, 1)])
def test_inspection_run_blocks_inference_and_result_on_camera_loss(
    qapp, settings, monkeypatch, tmp_path, worker_modules, missing,
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
    worker.inspection_config = module.InspectionConfig(widget=17, doInspection=True)
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
    worker.msleep = lambda _: setattr(worker, "running", False)
    for name in ("mysqlHost", "mysqlID", "mysqlPassword", "mysqlHostPort"):
        setattr(worker, name, None)
    errors = []
    worker.cameraErrorSignal.connect(errors.append)
    previews = []
    worker.partCam.connect(previews.append)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.cv2, "imread", lambda _: np.zeros((10, 10, 3), np.uint8))

    worker.run()
    assert worker.mergeframe1 is None and worker.mergeframe2 is None
    assert not worker.inspection_config.doInspection
    assert worker.inspection_config.current_numofPart == original_current
    worker.downSampling.assert_not_called()
    worker.inspection_coordinator.execute.assert_not_called()
    assert len(errors) == 1
    assert len(previews) == 1 and previews[0].width() == 690 * len(missing)
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
