"""Read the shared part registry (git tracked) for the settings UI.

This is a read-only view: the settings pages must not rewrite the registry,
because it is shared across every machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Optional

from aikensa.core.settings.yaml_store import SettingsFileError, read_yaml


class PartRegistryEntry(NamedTuple):
    widget_id: int
    part_number: str
    label: str

    def __iter__(self):  # keep tuple-unpacking in callers working
        return iter((self.widget_id, self.part_number, self.label))


def load_part_registry_entries(root: Path) -> List[PartRegistryEntry]:
    """Return ``(widget_id, part_number, label)`` for every registered part."""

    registry_path = Path(root) / "aikensa" / "core" / "config" / "part_registry.yaml"
    try:
        data = read_yaml(registry_path)
    except SettingsFileError:
        return []
    if not isinstance(data, dict):
        return []

    entries: List[PartRegistryEntry] = []
    for item in data.get("parts", []) or []:
        if not isinstance(item, dict):
            continue
        widget_id = item.get("widget_id")
        if widget_id is None:
            continue
        try:
            widget_id = int(widget_id)
        except (TypeError, ValueError):
            continue
        part_number = str(item.get("part_number", "") or "")
        label = str(item.get("part_name") or part_number or f"widget {widget_id}")
        entries.append(PartRegistryEntry(widget_id, part_number, label))

    entries.sort(key=lambda e: e.widget_id)
    return entries


def part_label_for_widget(entries: List[PartRegistryEntry], widget_id: int) -> Optional[str]:
    for entry in entries:
        if entry.widget_id == widget_id:
            return entry.part_number or entry.label
    return None
