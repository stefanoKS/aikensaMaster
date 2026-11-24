"""Identity of the model file and geometry used for a daily measurement."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from aikensa.core.inspection.daily import DailyFailure, DailyInspectionError


def _file_stamp(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass(frozen=True)
class DailyModelArtifact:
    path: Path
    stamp: tuple[int, ...]
    digest: str

    @classmethod
    def read(cls, path: str) -> "DailyModelArtifact":
        resolved = Path(path).resolve()
        stamp = _file_stamp(resolved)
        with resolved.open("rb") as stream:
            digest = sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        artifact = cls(resolved, stamp, digest.hexdigest())
        artifact.ensure_current(path)
        return artifact

    def ensure_current(self, path: str) -> None:
        try:
            resolved = Path(path).resolve()
            if resolved == self.path and _file_stamp(resolved) == self.stamp:
                return
        except OSError as exc:
            raise DailyInspectionError(DailyFailure.MODEL_ERROR, "日常点検モデルのファイルを確認できません。") from exc
        raise DailyInspectionError(DailyFailure.MODEL_ERROR, "日常点検モデルが変更されました。再起動が必要です。")


def daily_geometry_digest(intrinsics, source_shapes, homographies, crop, canvas_shape, software_rotation) -> str:
    payload = {
        "transform": "native-remap-rotate-warp-crop-v1",
        "intrinsics": [
            (intrinsics[i].camera_matrix.tolist(), intrinsics[i].distortion_coefficients.tolist())
            for i in range(2)
        ],
        "source_shapes": source_shapes,
        "homographies": [matrix.tolist() for matrix in homographies],
        "crop": crop,
        "canvas_shape": canvas_shape,
        "software_rotation": software_rotation,
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
