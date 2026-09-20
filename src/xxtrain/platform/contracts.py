from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from xxtrain.data import Bbox

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type WorkspaceEditGuard = Callable[[str, str | None], None]
type WorkspaceCacheRebuildGuard = Callable[[str, str, str], None]


@dataclass(frozen=True)
class ImageRecord:
    """Durable identity, location, dimensions, and unsigned 64-bit pHash for an image."""

    id: str
    relative_path: str
    width: int
    height: int
    perceptual_hash: int


@dataclass(frozen=True)
class AnnotationRecord:
    """Durable task-owned annotation with kind-specific content and parent rules.

    Negative records use null labels and geometry. Classification records use an allowed label and null
    geometry. Shape records use an allowed label and kind-specific JSON points. Task steps decide whether a
    parent is forbidden or required; every parent must be an annotation from an allowed step on the same
    image. ``AnnotationRepository`` raises ``ValueError`` when a record violates these rules.
    """

    id: UUID
    image_id: str
    step_key: str
    parent_id: UUID | None
    kind: str
    label: str | None
    geometry: JsonValue


@dataclass(frozen=True)
class AnnotationChanges:
    """One transient set of annotation upserts, deletions, and invalidated task steps."""

    upserts: tuple[AnnotationRecord, ...]
    delete_ids: frozenset[UUID]
    invalidated_steps: frozenset[str]


@dataclass(frozen=True)
class CvatBinding:
    """Bind one CVAT object identity to an annotation under its original image ID."""

    sample_id: str
    object_type: str
    object_id: int
    annotation_id: UUID


@dataclass(frozen=True)
class DetectionBox:
    """Detection geometry plus loss-preserving raw LabelMe rectangle metadata.

    ``extra`` preserves every raw rectangle field except ``label``, ``points``, and ``shape_type``. Group
    metadata remains transport metadata and is not a business association ID. ``cvat_id`` is the native shape
    identity returned by CVAT, or null when the source has no server identity.
    """

    geometry: Bbox
    extra: JsonObject = field(default_factory=dict)
    cvat_id: int | None = None


@dataclass(frozen=True)
class ImageInput:
    """Registered image with current detection boxes or explicit negative confirmation."""

    sample_id: str
    image_path: Path
    width: int
    height: int
    boxes: tuple[DetectionBox, ...]
    negative: bool = False


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
    """Current rectangles and explicit image-level negative confirmation from CVAT.

    No confirmation means unfinished when boxes are empty. Boxes plus confirmation are a validation conflict.
    """

    sample_id: str
    boxes: tuple[DetectionBox, ...]
    negative: bool = False


@dataclass(frozen=True)
class JobRef:
    """CVAT task and job IDs plus ordered unique original image IDs for a disposable runtime entry."""

    task_id: int
    job_id: int
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class PreparedJob:
    """A disposable CVAT job reference and its platform annotation bindings."""

    ref: JobRef
    bindings: tuple[CvatBinding, ...]


@dataclass(frozen=True)
class DetectionSync:
    """One validated detection synchronization prepared under the workspace write lock."""

    changes: AnnotationChanges
    bindings: tuple[CvatBinding, ...]
    fingerprint: str


@dataclass(frozen=True)
class TargetSync:
    """One validated downstream synchronization and its predicted final fingerprint."""

    changes: AnnotationChanges
    bindings: tuple[CvatBinding, ...]
    fingerprint: str


@dataclass(frozen=True)
class EditAnnotation:
    """One classification, shape, or negative annotation in edit-frame coordinates.

    Negative annotations use null label and geometry; all other kinds use a string business label.
    """

    id: UUID | None
    kind: str
    label: str | None
    geometry: JsonValue
    cvat_id: int | None = None


@dataclass(frozen=True)
class FrameMapping:
    """Map an ordered edit frame to its original image and integer source-pixel bounds.

    ``frame_id`` is the original image SHA for a full-image frame and the parent annotation UUID string for
    a crop. ``image_id`` is always the original image SHA. Full-image bounds are ``(0, 0, width, height)``;
    crop bounds are the actual clamped pixel rectangle. ``parent_id`` is null only for full-image frames.
    """

    frame_id: str
    image_id: str
    parent_id: UUID | None
    bounds: tuple[int, int, int, int]


@dataclass(frozen=True)
class EditFrame:
    """One image frame and its annotations for a target-specific edit job."""

    mapping: FrameMapping
    image_path: Path
    width: int
    height: int
    annotations: tuple[EditAnnotation, ...]


@dataclass(frozen=True)
class EditFrameResult:
    """Annotations returned for one edit frame identity."""

    frame_id: str
    annotations: tuple[EditAnnotation, ...]


@dataclass(frozen=True)
class EditJob:
    """A CVAT job reference plus frame-to-source mappings in exact CVAT frame order."""

    ref: JobRef
    frames: tuple[FrameMapping, ...]


@dataclass(frozen=True)
class TargetSummary:
    """Count target samples, completed samples, and completed positive samples."""

    sample_count: int
    annotated_sample_count: int
    positive_sample_count: int


@dataclass(frozen=True)
class TargetView:
    """One model target's fact-derived progress and available actions."""

    id: str
    sample_count: int
    annotated_sample_count: int
    can_annotate: bool
    can_generate_cache: bool
    cache_ready: bool
    display_name: str = ''
    sample_unit: str = '个样本'


@dataclass(frozen=True)
class WorkspaceView:
    workspace_id: str
    name: str
    image_count: int
    targets: tuple[TargetView, ...] = ()
    task_id: str = 'task'
    task_name: str = 'Task'


class PlatformError(Exception):
    pass


class PlatformConflictError(PlatformError):
    """An expected resource conflict that is safe to report without backend details."""


class TargetValidationError(PlatformError):
    """A safe annotation validation failure with its server-owned correction link."""

    def __init__(self, message: str, annotation_url: str) -> None:
        super().__init__(message)
        self.annotation_url = annotation_url


class PlatformAccessError(PlatformError):
    pass
