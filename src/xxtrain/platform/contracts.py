from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from xxtrain.data import Bbox

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type Status = Literal['pending', 'preparing', 'annotating', 'sync_failed', 'saved']


@dataclass(frozen=True)
class DetectionBox:
    """Detection geometry plus loss-preserving raw LabelMe rectangle metadata.

    ``extra`` preserves every raw rectangle field except ``label``, ``points``, and ``shape_type``. Group
    metadata remains transport metadata and is not a business association ID.
    """

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
class UploadResult:
    """Counts outcomes from one staged-image admission attempt."""

    received_count: int
    accepted_count: int
    exact_duplicate_count: int
    similar_duplicate_count: int


@dataclass(frozen=True)
class DetectionSummary:
    """Counts workspace images and their completed detection annotations."""

    image_count: int
    annotated_image_count: int
    boxed_image_count: int


@dataclass(frozen=True)
class FrameResult:
    sample_id: str
    boxes: tuple[DetectionBox, ...]


@dataclass(frozen=True)
class JobRef:
    """CVAT task, job, and workspace sample identifiers for a disposable runtime entry."""

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
