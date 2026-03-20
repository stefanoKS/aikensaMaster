from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class AppPaths:
    root: Path
    resources: Path
    data: Path

    @staticmethod
    def detect() -> "AppPaths":
        # Optional override for factory PCs
        root_env = os.getenv("AIKENSA_ROOT")
        if root_env:
            root = Path(root_env).resolve()
        else:
            root = Path(__file__).resolve().parents[2]

        resources = root / "resources"
        data = root / "data"

        resources.mkdir(exist_ok=True)
        data.mkdir(exist_ok=True)

        return AppPaths(root, resources, data)


PATHS = AppPaths.detect()

