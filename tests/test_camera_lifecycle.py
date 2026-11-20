"""Group D: camera lifecycle and ownership."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from aikensa.core.camera.camera_service import (
    CameraBusyError,
    CameraRole,
    CameraService,
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
