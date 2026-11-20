"""Null-aware, atomic YAML persistence.

Design rules:

* ``null`` is a first-class value.  A calibration file whose matrix is ``null``
  must load as ``None`` and must never be coerced into a plausible matrix.
* Writing is atomic: a temporary file on the same filesystem is flushed,
  ``fsync``-ed and then ``os.replace``-d over the target.  A crash can never
  leave a half-written calibration behind.
* NumPy objects never reach the serialiser.  ``numpy.float64`` and friends are
  rejected by PyYAML's safe dumper, so :func:`to_plain` converts to lists and
  builtins first.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import yaml

PathLike = Union[os.PathLike, str]


class SettingsFileError(RuntimeError):
    """Raised when a settings file cannot be read or parsed."""


class SettingsValidationError(RuntimeError):
    """Raised when in-memory settings fail validation before writing."""

    def __init__(self, issues):
        self.issues = list(issues)
        detail = "; ".join(str(i) for i in self.issues) or "no detail"
        super().__init__(f"Settings validation failed: {detail}")


def to_plain(value: Any) -> Any:
    """Recursively convert numpy / dataclass values into YAML-safe builtins."""

    if isinstance(value, np.ndarray):
        return [to_plain(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {to_plain(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def read_yaml(path: PathLike) -> Any:
    """Read a YAML file.

    Returns ``None`` when the file does not exist.  Raises
    :class:`SettingsFileError` for unreadable or malformed content so callers
    can classify the state as ``INVALID`` instead of silently using defaults.
    """

    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise SettingsFileError(f"{p}: {exc}") from exc


def read_yaml_result(path: PathLike) -> tuple[Any, Optional[str]]:
    """Read YAML without raising; returns ``(data, error_message)``."""

    try:
        return read_yaml(path), None
    except SettingsFileError as exc:
        return None, str(exc)


def write_yaml_atomic(path: PathLike, data: Any) -> Path:
    """Atomically write *data* as YAML to *path*."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(
        to_plain(data), allow_unicode=True, sort_keys=False, default_flow_style=False
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        _fsync_directory(target.parent)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory fsync (unsupported on some Windows filesystems)."""

    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except (OSError, AttributeError):
        pass
    finally:
        os.close(dir_fd)


def matrix_or_none(data: Any, expected_shape: Optional[tuple[int, ...]] = None) -> Optional[np.ndarray]:
    """Convert YAML list-of-lists into a float64 array, or ``None`` when null.

    Raises :class:`ValueError` when the content is not a numeric matrix of the
    expected shape.  ``None`` in means ``None`` out: callers must treat that as
    "not calibrated".
    """

    if data is None:
        return None
    try:
        array = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"not a numeric matrix: {exc}") from exc
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"expected shape {expected_shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix contains NaN or infinite values")
    return array


def is_null_value(value: Any) -> bool:
    """True when *value* means "not configured" (None, empty list, nulls only)."""

    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0 or all(is_null_value(v) for v in value)
    if isinstance(value, dict):
        return all(is_null_value(v) for v in value.values())
    return False


def ensure_null_template(path: PathLike, template: Any) -> bool:
    """Create *path* from *template* only when it is absent.

    Never overwrites an existing file, so a valid machine calibration can never
    be replaced by a ``null`` template.  Returns ``True`` when the file was
    created.
    """

    target = Path(path)
    if target.exists():
        return False
    write_yaml_atomic(target, template)
    return True
