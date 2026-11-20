"""Qt controllers for the machine-settings pages.

Each page is a real Qt Designer ``.ui`` file under
``aikensa/gui/pages/SETTINGS`` plus a thin controller here.  Controllers only
translate between widgets and the typed settings service; they never parse YAML
themselves.
"""

from aikensa.gui.settings_pages.controller import SettingsController

__all__ = ["SettingsController"]
