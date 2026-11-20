"""Backup and restore of machine-specific configuration.

Removing tracked runtime files from the index (Stage B of the migration) does
**not** protect deployed machines: pulling a commit that deletes a tracked file
deletes that file from the working tree.  The only safe procedure is to back up
outside the work tree first, verify the backup, and only then pull.

This module implements that backup, verification and restore in a way that
works on Windows as well as Linux (``Path`` operations plus ``shutil`` only).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from aikensa.core.settings.paths import (
    AppPaths,
    RUNTIME_ADJUSTMENT_FILES,
    RUNTIME_CAMERA_FILES,
    RUNTIME_CAM_FILES,
)

MANIFEST_NAME = "backup_manifest.json"


@dataclass
class BackupResult:
    backup_dir: Path
    copied: List[str] = field(default_factory=list)
    verified: bool = False
    problems: List[str] = field(default_factory=list)
    manifest_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.verified and not self.problems


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_file_groups(paths: AppPaths) -> List[tuple[str, Path, Iterable[str]]]:
    """``(label, directory, filenames)`` for every machine-specific file."""

    return [
        ("cam", paths.runtime_cam_dir, RUNTIME_CAM_FILES),
        ("camera", paths.runtime_camera_dir, RUNTIME_CAMERA_FILES),
        ("adjustments", paths.runtime_adjustments_dir, RUNTIME_ADJUSTMENT_FILES),
    ]


def collect_runtime_files(paths: AppPaths) -> List[Path]:
    """Every machine-specific file that currently exists."""

    existing: List[Path] = []
    for _label, directory, filenames in runtime_file_groups(paths):
        for name in filenames:
            candidate = directory / name
            if candidate.is_file():
                existing.append(candidate)
    return existing


def backup_machine_files(
    paths: AppPaths,
    destination: Optional[Path] = None,
    *,
    reason: str = "manual",
) -> BackupResult:
    """Copy every machine-specific file outside the work tree and verify it."""

    root = Path(destination) if destination is not None else paths.backup_dir
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = root / f"{stamp}_{reason}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    result = BackupResult(backup_dir=backup_dir)
    entries: List[Dict[str, str]] = []

    for label, directory, filenames in runtime_file_groups(paths):
        target_dir = backup_dir / label
        for name in filenames:
            source = directory / name
            if not source.is_file():
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / name
            shutil.copy2(source, target)
            digest = _sha256(target)
            entries.append(
                {
                    "group": label,
                    "name": name,
                    "source": str(source),
                    "backup": str(target),
                    "sha256": digest,
                    "size": str(target.stat().st_size),
                }
            )
            result.copied.append(str(source))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reason": reason,
        "app_root": str(paths.root),
        "files": entries,
    }
    manifest_path = backup_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    result.manifest_path = manifest_path

    result.problems = verify_backup(backup_dir)
    result.verified = not result.problems
    return result


def verify_backup(backup_dir: Path) -> List[str]:
    """Re-hash every file in a backup and report any mismatch or loss."""

    backup_dir = Path(backup_dir)
    manifest_path = backup_dir / MANIFEST_NAME
    problems: List[str] = []
    if not manifest_path.is_file():
        return [f"manifest missing: {manifest_path}"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"manifest unreadable: {exc}"]

    for entry in manifest.get("files", []):
        stored = Path(entry["backup"])
        if not stored.is_file():
            problems.append(f"missing backup file: {stored}")
            continue
        if _sha256(stored) != entry["sha256"]:
            problems.append(f"checksum mismatch: {stored}")
    return problems


def restore_machine_files(
    paths: AppPaths,
    backup_dir: Path,
    *,
    dry_run: bool = False,
) -> List[str]:
    """Restore a verified backup into the runtime locations.

    Returns the list of restored file paths.  Raises ``RuntimeError`` when the
    backup does not verify, so an unverifiable backup is never applied.
    """

    backup_dir = Path(backup_dir)
    problems = verify_backup(backup_dir)
    if problems:
        raise RuntimeError("backup did not verify: " + "; ".join(problems))

    manifest = json.loads((backup_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    restored: List[str] = []
    for entry in manifest.get("files", []):
        target = Path(entry["source"])
        stored = Path(entry["backup"])
        restored.append(str(target))
        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "wb") as out_handle, stored.open("rb") as in_handle:
                shutil.copyfileobj(in_handle, out_handle)
                out_handle.flush()
                os.fsync(out_handle.fileno())
            os.replace(tmp_name, target)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
    return restored


def list_backups(root: Optional[Path], paths: AppPaths) -> List[Path]:
    base = Path(root) if root is not None else paths.backup_dir
    if not base.is_dir():
        return []
    return sorted(
        (entry for entry in base.iterdir() if (entry / MANIFEST_NAME).is_file()),
        reverse=True,
    )


def migration_status(paths: AppPaths, tracked: Optional[Iterable[str]] = None) -> Dict[str, object]:
    """Summarise whether untracking machine files is safe.

    ``tracked`` must be supplied by the caller (normally from
    ``git ls-files``) because this module never shells out to Git itself.
    """

    tracked_set = set(tracked or ())
    runtime = {str(p.relative_to(paths.root)) for p in collect_runtime_files(paths)}
    still_tracked = sorted(runtime & tracked_set)

    backups = list_backups(None, paths)
    verified_backups = [b for b in backups if not verify_backup(b)]

    return {
        "runtime_file_count": len(runtime),
        "still_tracked": still_tracked,
        "backup_count": len(backups),
        "verified_backup_count": len(verified_backups),
        "untracking_safe": bool(verified_backups) and not still_tracked,
    }
