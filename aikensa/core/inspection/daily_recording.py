"""Record daily results without changing production counter or history semantics."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3
from uuid import uuid4

import cv2
import numpy as np

from aikensa.core.inspection.daily import DailyMeasurement
from aikensa.core.settings.daily_config import DailyProfileSpec


@dataclass(frozen=True)
class DailyCycle:
    profile: DailyProfileSpec
    revision_id: str
    session_id: str
    operator: str
    cycle_id: str
    timestamp: datetime
    current_counts: tuple[int, int]
    today_counts: tuple[int, int]
    model_digest: str | None = None
    geometry_digest: str | None = None

    @classmethod
    def begin(
        cls, profile, revision_id, session_id, operator, current_counts, today_counts,
        *, cycle_id=None, timestamp=None, model_digest=None, geometry_digest=None,
    ):
        return cls(
            profile, revision_id, session_id, operator, cycle_id or uuid4().hex, timestamp or datetime.now(),
            tuple(current_counts), tuple(today_counts),
            model_digest, geometry_digest,
        )


_RESULT_FIELDS = (
    "partName, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, "
    "kensainName, detected_pitch, delta_pitch, total_length, resultpitch, status, NGreason, PPMS"
)


def legacy_row(cycle: DailyCycle, measurement: DailyMeasurement) -> tuple:
    measured = [] if measurement.actual_mm is None else [measurement.actual_mm]
    return (
        cycle.profile.directory, str(list(cycle.today_counts)), str(list(cycle.current_counts)),
        cycle.timestamp.strftime("%H:%M:%S"), cycle.timestamp.strftime("%Y%m%d"), 0.0,
        cycle.operator, str(measured), "None", 0.0, str([int(measurement.status == "OK")]),
        measurement.status, measurement.reason, "Null",
    )


def write_daily_result(
    connection: sqlite3.Connection,
    root: Path,
    cycle: DailyCycle,
    measurement: DailyMeasurement,
    raw: np.ndarray,
    annotated: np.ndarray,
) -> bool:
    """Return False only for a cycle already recorded. Raise on image or DB failure."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS daily_inspection_metadata (
            cycle_id TEXT PRIMARY KEY, result_id INTEGER NOT NULL UNIQUE,
            session_id TEXT NOT NULL, profile_id TEXT NOT NULL, widget_id INTEGER NOT NULL,
            revision_id TEXT NOT NULL, actual_mm REAL, nominal_mm REAL NOT NULL,
            tolerance_mm REAL NOT NULL, mm_per_px REAL NOT NULL, difference_mm REAL,
            raw_path TEXT NOT NULL, result_path TEXT NOT NULL,
            profile_digest TEXT, model_digest TEXT, geometry_digest TEXT,
            FOREIGN KEY(result_id) REFERENCES inspection_results(id)
        )
    """)
    written = []
    row = legacy_row(cycle, measurement)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(daily_inspection_metadata)")}
            for name in ("profile_digest", "model_digest", "geometry_digest"):
                if name not in columns:
                    connection.execute(f"ALTER TABLE daily_inspection_metadata ADD COLUMN {name} TEXT")
            if connection.execute(
                "SELECT 1 FROM daily_inspection_metadata WHERE cycle_id = ?", (cycle.cycle_id,)
            ).fetchone():
                return False
            base = root / cycle.profile.directory / cycle.timestamp.strftime("%Y%m%d") / measurement.status
            filename = f"{cycle.timestamp:%Y%m%d_%H%M%S}_{cycle.cycle_id}.png"
            raw_path, result_path = base / "nama" / filename, base / "kekka" / filename
            height, width = annotated.shape[:2]
            result_small = cv2.resize(annotated, (max(1, width // 2), max(1, height // 2)))
            for path, image in ((raw_path, raw), (result_path, result_small)):
                path.parent.mkdir(parents=True, exist_ok=True)
                written.append(path)
                if not cv2.imwrite(str(path), image):
                    raise OSError(f"日常点検の画像を保存できません: {path.name}")
            record_id = connection.execute(
                f"INSERT INTO inspection_results ({_RESULT_FIELDS}) VALUES ({','.join(['?'] * 14)})", row
            ).lastrowid
            ref = measurement.reference
            connection.execute("""
                INSERT INTO daily_inspection_metadata (
                    cycle_id, result_id, session_id, profile_id, widget_id, revision_id, actual_mm,
                    nominal_mm, tolerance_mm, mm_per_px, difference_mm, raw_path, result_path, profile_digest,
                    model_digest, geometry_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cycle.cycle_id, record_id, cycle.session_id, cycle.profile.profile_id,
                cycle.profile.widget_id, cycle.revision_id, measurement.actual_mm,
                ref.nominal_mm, ref.tolerance_mm, ref.mm_per_px, measurement.difference_mm,
                str(raw_path.relative_to(root)), str(result_path.relative_to(root)),
                cycle.profile.configuration_digest,
                cycle.model_digest, cycle.geometry_digest,
            ))
    except (OSError, sqlite3.Error, cv2.error) as exc:
        for path in written:
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                exc.add_note(f"Unable to remove incomplete daily image: {cleanup_error}")
        raise
    return True
