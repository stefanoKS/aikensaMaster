"""Current daily inspection attempts, separate from the persisted history."""

from dataclasses import dataclass, field, replace
from enum import Enum, auto

from aikensa.core.settings.daily_config import DailyProfileSpec


DAILY_POSITIONS = (21, 22, 23)


class DailyPositionState(Enum):
    NOT_INSPECTED = "未点検"
    INSPECTING = "点検中"
    OK = "OK"
    NG = "NG"
    EQUIPMENT_ERROR = "設備異常"
    RECORDING_FAILED = "記録失敗"
    REINSPECTION_REQUIRED = "再点検が必要"


class RecordingState(Enum):
    NOT_ATTEMPTED = auto()
    NOT_CONFIGURED = auto()
    RECORDED = auto()
    FAILED = auto()


@dataclass(frozen=True)
class DailyPositionUpdate:
    widget: int
    session_id: str
    cycle_id: str
    state: DailyPositionState
    details: str = ""
    revision_id: str | None = None
    profile: DailyProfileSpec | None = None
    judgment: str | None = None
    local: RecordingState = RecordingState.NOT_ATTEMPTED
    remote: RecordingState = RecordingState.NOT_ATTEMPTED

    @property
    def message(self) -> str:
        return f"{self.state.value} / {self.details}" if self.details else self.state.value


@dataclass
class DailySession:
    session_id: str
    positions: dict[int, DailyPositionUpdate] = field(default_factory=dict)
    approved_profiles: dict[int, DailyProfileSpec] | None = None

    def apply(self, update: DailyPositionUpdate) -> bool:
        if update.session_id != self.session_id or update.widget not in DAILY_POSITIONS:
            return False
        previous = self.positions.get(update.widget)
        if update.state is not DailyPositionState.INSPECTING and (
            previous is None or previous.cycle_id != update.cycle_id
            or previous.state is not DailyPositionState.INSPECTING
        ):
            return False
        if (update.profile is not None and self.approved_profiles is not None
                and self.approved_profiles.get(update.widget) != update.profile):
            update = replace(update, state=DailyPositionState.REINSPECTION_REQUIRED, details="")
        self.positions[update.widget] = update
        return True

    def update_configuration(self, profiles: dict[int, DailyProfileSpec]) -> tuple[int, ...]:
        self.approved_profiles = profiles
        invalidated = []
        for widget, update in self.positions.items():
            if update.profile is not None and update.profile != profiles.get(widget):
                self.positions[widget] = replace(
                    update, state=DailyPositionState.REINSPECTION_REQUIRED, details="",
                )
                invalidated.append(widget)
        return tuple(invalidated)

    def interrupt(self, message: str) -> None:
        for widget, update in self.positions.items():
            if update.state is DailyPositionState.INSPECTING:
                self.positions[widget] = replace(update, state=DailyPositionState.EQUIPMENT_ERROR, details=message)

    @property
    def summary(self) -> str:
        states = {widget: result.state for widget, result in self.positions.items()}
        values = states.values()
        completed = sum(state in (DailyPositionState.OK, DailyPositionState.NG) for state in values)
        if completed == len(DAILY_POSITIONS):
            heading = "全位置実施・NGあり" if DailyPositionState.NG in values else "全位置OK"
        elif DailyPositionState.INSPECTING in values:
            heading = "点検中"
        elif DailyPositionState.REINSPECTION_REQUIRED in values:
            heading = "再点検が必要"
        elif states:
            heading = "一部実施・NGあり" if DailyPositionState.NG in values else "一部実施"
        else:
            heading = "未完了"
        positions = " / ".join(
            f"{widget - 20}: {states.get(widget, DailyPositionState.NOT_INSPECTED).value}"
            for widget in DAILY_POSITIONS
        )
        return f"{heading} / {positions}"
