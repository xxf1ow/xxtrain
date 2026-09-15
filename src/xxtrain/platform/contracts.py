from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from xxtrain.data import Bbox

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type Status = Literal['pending', 'preparing', 'annotating', 'sync_failed', 'saved']


@dataclass(frozen=True)
class DetectionBox:
    geometry: Bbox
    extra: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class ImageInput:
    sample_id: str
    image_path: Path
    width: int
    height: int
    boxes: tuple[DetectionBox, ...]


@dataclass(frozen=True)
class FrameResult:
    sample_id: str
    boxes: tuple[DetectionBox, ...]


@dataclass(frozen=True)
class JobRef:
    task_id: int
    job_id: int
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class WorkspaceView:
    workspace_id: str
    name: str
    image_count: int
    status: Status
    error: str | None = None


class PlatformError(Exception):
    pass


class PlatformAccessError(PlatformError):
    pass
