"""Versioned, machine-local settings revisions with rollback.

A single YAML file is written atomically in
:mod:`aikensa.core.settings.yaml_store`.  Multi-file changes (for example a
global alignment plus several widget overrides) need more: either *all* files
become active or none does.  This module implements that with a staging
directory plus a revision manifest:

1. ``stage``  - write candidate files into ``revisions/<id>/`` and hash them.
2. ``activate`` - verify every staged hash, then replace every runtime file;
   on any failure the previous file contents are restored.
3. ``rollback`` - activate an older revision again.

Revision history lives under ``aikensa/core/param/adjustments/revisions`` and
is machine specific, therefore never committed to Git.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from threading import Lock
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import yaml

from aikensa.core.settings.paths import AppPaths
from aikensa.core.settings.schema import SettingsRevision
from aikensa.core.settings.yaml_store import to_plain

_MANIFEST_NAME = "manifest.json"
_ACTIVE_NAME = "active.json"
DAILY_REVISION_LOCK = Lock()


class RevisionError(RuntimeError):
    """Raised when a revision cannot be staged, activated or rolled back."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def machine_id() -> str:
    """Stable identifier for this installation (no network / hardware probing)."""

    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        try:
            if candidate.is_file():
                value = candidate.read_text(encoding="utf-8").strip()
                if value:
                    return value
        except OSError:
            continue
    return os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "unknown-machine"


class RevisionManager:
    """Stage / activate / roll back machine-local settings revisions."""

    def __init__(self, paths: AppPaths, *, snapshot_scope: Optional[str] = None):
        self.paths = paths
        self.snapshot_only = snapshot_scope is not None
        self.history_dir = (
            paths.runtime_history_dir if snapshot_scope is None
            else paths.runtime_history_dir / snapshot_scope
        )

    # ---- staging -----------------------------------------------------------
    def stage(
        self,
        files: Mapping[Path, object],
        *,
        operator: str = "",
        changes: Iterable[str] = (),
        note: str = "",
    ) -> str:
        """Write candidate files into a staging revision and return its id."""

        if not files:
            raise RevisionError("no files to stage")

        revision_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        revision_dir = self.history_dir / revision_id
        payload_dir = revision_dir / "files"
        payload_dir.mkdir(parents=True, exist_ok=False)

        manifest_files: List[Dict[str, str]] = []
        try:
            for index, (target, data) in enumerate(files.items()):
                target_path = Path(target)
                stored_name = f"{index:02d}_{target_path.name}"
                stored_path = payload_dir / stored_name
                stored_path.write_text(
                    yaml.safe_dump(
                        to_plain(data),
                        allow_unicode=True,
                        sort_keys=False,
                        default_flow_style=False,
                    ),
                    encoding="utf-8",
                )
                manifest_files.append(
                    {
                        "target": str(target_path),
                        "stored": stored_name,
                        "sha256": _sha256(stored_path),
                    }
                )
        except BaseException:
            shutil.rmtree(revision_dir, ignore_errors=True)
            raise

        manifest = {
            "revision_id": revision_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "operator": operator,
            "machine_id": machine_id(),
            "files": manifest_files,
            "changes": [str(c) for c in changes],
            "note": note,
        }
        self._write_json(revision_dir / _MANIFEST_NAME, manifest)
        return revision_id

    # ---- activation --------------------------------------------------------
    def activate(self, revision_id: str) -> SettingsRevision:
        """Make *revision_id* the active revision.

        Every staged file is verified before anything is replaced.  If a write
        fails midway the already-replaced files are restored.
        """

        revision_dir = self.history_dir / revision_id
        manifest_path = revision_dir / _MANIFEST_NAME
        if not manifest_path.is_file():
            raise RevisionError(f"revision not found: {revision_id}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload_dir = revision_dir / "files"

        for entry in manifest["files"]:
            stored = payload_dir / entry["stored"]
            if not stored.is_file():
                raise RevisionError(f"staged file missing: {stored}")
            if _sha256(stored) != entry["sha256"]:
                raise RevisionError(f"staged file corrupted: {stored}")

        if self.snapshot_only:
            self._mark_active(revision_id)
            return self._to_revision(manifest)

        backups: Dict[Path, Optional[bytes]] = {}
        replaced: List[Path] = []
        try:
            for entry in manifest["files"]:
                target = Path(entry["target"])
                stored = payload_dir / entry["stored"]
                backups[target] = target.read_bytes() if target.is_file() else None
                self._replace_atomically(target, stored.read_bytes())
                replaced.append(target)
            self._mark_active(revision_id)
        except BaseException:
            for target in reversed(replaced):
                previous = backups.get(target)
                try:
                    if previous is None:
                        target.unlink(missing_ok=True)
                    else:
                        self._replace_atomically(target, previous)
                except OSError:
                    pass
            raise

        return self._to_revision(manifest)

    def _mark_active(self, revision_id: str) -> None:
        self._write_json(
            self.history_dir / _ACTIVE_NAME,
            {"revision_id": revision_id, "activated_at": datetime.now().isoformat(timespec="seconds")},
        )
    def rollback(self, revision_id: str) -> SettingsRevision:
        return self.activate(revision_id)

    # ---- querying ----------------------------------------------------------
    def active_revision_id(self, *, strict: bool = False) -> Optional[str]:
        record = self.history_dir / _ACTIVE_NAME
        if not record.is_file():
            return None
        try:
            revision_id = json.loads(record.read_text(encoding="utf-8"))["revision_id"]
            if (not isinstance(revision_id, str) or not revision_id
                    or revision_id in (".", "..") or "/" in revision_id or "\\" in revision_id):
                raise ValueError("invalid revision id")
            return revision_id
        except (KeyError, OSError, TypeError, ValueError) as exc:
            if strict:
                raise RevisionError("invalid active revision") from exc
            return None

    def list_revisions(self) -> List[SettingsRevision]:
        if not self.history_dir.is_dir():
            return []
        revisions: List[SettingsRevision] = []
        for entry in sorted(self.history_dir.iterdir(), reverse=True):
            manifest_path = entry / _MANIFEST_NAME
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            revisions.append(self._to_revision(manifest))
        return revisions

    def get_revision(self, revision_id: str) -> Optional[SettingsRevision]:
        manifest_path = self.history_dir / revision_id / _MANIFEST_NAME
        if not manifest_path.is_file():
            return None
        return self._to_revision(json.loads(manifest_path.read_text(encoding="utf-8")))

    def staged_data(self, revision_id: str) -> Dict[str, object]:
        """Return ``{target_path: parsed_yaml}`` for a staged revision."""

        revision_dir = self.history_dir / revision_id
        manifest_path = revision_dir / _MANIFEST_NAME
        if not manifest_path.is_file():
            raise RevisionError(f"revision not found: {revision_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result: Dict[str, object] = {}
        for entry in manifest["files"]:
            stored = revision_dir / "files" / entry["stored"]
            if not stored.is_file() or _sha256(stored) != entry["sha256"]:
                raise RevisionError(f"staged file missing or corrupted: {stored}")
            result[entry["target"]] = yaml.safe_load(stored.read_text(encoding="utf-8"))
        return result

    # ---- internals ---------------------------------------------------------
    @staticmethod
    def _replace_atomically(target: Path, payload: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    @staticmethod
    def _write_json(path: Path, payload: Mapping) -> None:
        RevisionManager._replace_atomically(
            path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        )

    @staticmethod
    def _to_revision(manifest: Mapping) -> SettingsRevision:
        return SettingsRevision(
            revision_id=manifest["revision_id"],
            created_at=datetime.fromisoformat(manifest["created_at"]),
            operator=manifest.get("operator", "") or "",
            machine_id=manifest.get("machine_id", "") or "",
            changed_files=tuple(entry["target"] for entry in manifest.get("files", [])),
            changes=tuple(manifest.get("changes", []) or ()),
            activated=True,
            note=manifest.get("note", "") or "",
        )
