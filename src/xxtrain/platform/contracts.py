from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from xxtrain.data import Bbox

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


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
    """Bind one CVAT object identity to an annotation within a job sample."""

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
class WorkspaceView:
    workspace_id: str
    name: str
    image_count: int
    annotated_image_count: int
    boxed_image_count: int
    can_generate_detection_cache: bool
    detection_cache_ready: bool


class PlatformError(Exception):
    pass


class PlatformAccessError(PlatformError):
    pass
