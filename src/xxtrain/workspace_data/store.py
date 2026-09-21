import os
from pathlib import Path
from uuid import uuid4

from PIL import Image

from xxtrain.business_tasks.definition import TaskDefinition
from xxtrain.data import Bbox
from xxtrain.platform.contracts import (
    AnnotationRecord,
    CvatBinding,
    DetectionBox,
    DetectionSummary,
    DetectionSync,
    EditFrame,
    EditFrameResult,
    EditJob,
    FrameResult,
    ImageInput,
    ImageRecord,
    JobRef,
    PlatformAccessError,
    PreparedJob,
    TargetSummary,
    TargetSync,
    UploadResult,
)
from xxtrain.workspace_data.changes import plan_changes
from xxtrain.workspace_data.dedup import SIMILARITY_DISTANCE, hamming_distance, image_sha256, perceptual_hash
from xxtrain.workspace_data.editing import (
    fingerprint_target,
    fingerprint_training,
    prepare_sync,
    project_target_frames,
    summarize_target,
)
from xxtrain.workspace_data.repository import AnnotationRepository

_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


class WorkspaceData:
    """Access registered images and authoritative SQLite annotations beneath one workspace root."""

    def __init__(self, workspace_dir: Path, task: TaskDefinition) -> None:
        self._root = Path(workspace_dir)
        self._images_dir = self._root / 'images'
        self._require_directory(self._images_dir, writable=True)
        self._task = task
        self._repository = AnnotationRepository(self._root / 'annotations.db', self._task)

    @property
    def task(self) -> TaskDefinition:
        """Return the immutable task definition supplied by the composition root."""
        return self._task

    @staticmethod
    def _require_directory(path: Path, *, writable: bool) -> None:
        access = os.R_OK | os.X_OK | (os.W_OK if writable else 0)
        if not path.is_dir():
            raise PlatformAccessError(f'Directory does not exist: {path}')
        if not os.access(path, access):
            raise PlatformAccessError(f'Directory is not accessible: {path}')

    def images(self, step_key: str | None = None) -> tuple[ImageInput, ...]:
        """Return raw registered images, optionally projected with one rectangle-bearing step."""
        annotations: tuple[AnnotationRecord, ...] = ()
        if step_key is not None:
            step = self._task.step(step_key)
            if 'rectangle' not in step.kinds:
                raise ValueError(f'Task step {step_key!r} does not contain rectangle inputs')
            annotations = self._repository.annotations(step_key=step_key)
        negative_ids = {record.image_id for record in annotations if record.kind == 'negative'}
        by_image: dict[str, list[DetectionBox]] = {}
        for record in annotations:
            if record.kind != 'rectangle':
                continue
            points = record.geometry
            if not isinstance(points, list) or len(points) != 2:
                raise ValueError('Detection rectangle geometry requires two points')
            first, second = points
            by_image.setdefault(record.image_id, []).append(
                DetectionBox(
                    Bbox(id=record.id, label=record.label, x1=first[0], y1=first[1], x2=second[0], y2=second[1])
                )
            )
        return tuple(
            ImageInput(
                record.id,
                self._root / record.relative_path,
                record.width,
                record.height,
                tuple(by_image.get(record.id, ())),
                record.id in negative_ids,
            )
            for record in self._repository.images()
        )

    def admit(self, staged: tuple[Path, ...]) -> UploadResult:
        """Admit decodable staged images once, deleting every staged file before returning."""
        try:
            existing = self._repository.images()
            existing_shas = {record.id for record in existing}
            existing_hashes = tuple(record.perceptual_hash for record in existing)
            candidates = []
            for staged_path in staged:
                path = Path(staged_path)
                if path.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                try:
                    digest = image_sha256(path)
                    image_hash = perceptual_hash(path)
                    with Image.open(path) as image:
                        width, height = image.size
                except (OSError, ValueError):
                    continue
                candidates.append((digest, image_hash, width, height, path))

            accepted_count = 0
            exact_duplicate_count = 0
            similar_duplicate_count = 0
            accepted_hashes: list[int] = []
            batch_shas: set[str] = set()
            records: list[ImageRecord] = []
            for digest, image_hash, width, height, path in sorted(candidates, key=lambda candidate: candidate[0]):
                if digest in existing_shas or digest in batch_shas:
                    exact_duplicate_count += 1
                    continue
                batch_shas.add(digest)
                if any(
                    hamming_distance(image_hash, compared_hash) <= SIMILARITY_DISTANCE
                    for compared_hash in (*existing_hashes, *accepted_hashes)
                ):
                    similar_duplicate_count += 1
                    continue
                filename = f'{digest}{path.suffix.lower()}'
                os.replace(path, self._images_dir / filename)
                records.append(ImageRecord(digest, f'images/{filename}', width, height, image_hash))
                existing_shas.add(digest)
                accepted_hashes.append(image_hash)
                accepted_count += 1
            self._repository.register_images(tuple(records))
            return UploadResult(len(staged), accepted_count, exact_duplicate_count, similar_duplicate_count)
        finally:
            for path in staged:
                Path(path).unlink(missing_ok=True)

    def detection_summary(self) -> DetectionSummary:
        """Return image-level detection counts aggregated from registered database facts."""
        summary = self.target_summary('detect')
        return DetectionSummary(summary.sample_count, summary.annotated_sample_count, summary.positive_sample_count)

    def target_frames(self, target: str, runtime_root: Path) -> tuple[EditFrame, ...]:
        """Return current crop frames populated with one downstream target's annotations."""
        return project_target_frames(target, self.images(), self._repository.annotations(), runtime_root, self._task)

    def target_summary(self, target: str) -> TargetSummary:
        """Return crop and completed-crop counts for a downstream annotation target."""
        return summarize_target(target, self.images(), self._repository.annotations(), self._task)

    def target_fingerprint(self, target: str) -> str:
        """Hash one downstream target's current input facts and stable associations."""
        return fingerprint_target(target, self.images(), self._repository.annotations(), self._task)

    def training_fingerprint(self, target: str) -> str:
        """Hash current target facts plus task-declared dataset conversion semantics."""
        return fingerprint_training(self.target_fingerprint(target), target, self._task)

    def detection_fingerprint(self) -> str:
        """Hash registered image identities and normalized detection business content."""
        return self.target_fingerprint('detect')

    def bind_job(self, prepared: PreparedJob) -> None:
        """Persist a prepared CVAT job's native-object bindings atomically."""
        self._repository.bind_job(prepared)

    def prepare_detection_sync(self, ref: JobRef, results: tuple[FrameResult, ...]) -> DetectionSync:
        """Validate exact image coverage and native CVAT identities, then plan one object-scoped update."""
        image_ids = tuple(record.id for record in self._repository.images())
        if len(set(ref.sample_ids)) != len(ref.sample_ids) or set(ref.sample_ids) != set(image_ids):
            raise ValueError('Detection job must cover every current workspace image exactly once')
        received = tuple(result.sample_id for result in results)
        if len(set(received)) != len(received) or set(received) != set(image_ids):
            raise ValueError('Detection results must cover every workspace image exactly once')

        current = self._repository.annotations()
        current_detection = tuple(record for record in current if record.step_key == 'detect')
        current_negative = {record.image_id: record for record in current_detection if record.kind == 'negative'}
        existing_bindings = self._repository.bindings(ref)
        if any(binding.object_type != 'shape' for binding in existing_bindings):
            raise ValueError('Detection jobs support only CVAT shape bindings')
        by_object_id = {binding.object_id: binding for binding in existing_bindings}

        object_ids: set[int] = set()
        incoming: list[AnnotationRecord] = []
        bindings: list[CvatBinding] = []
        for result in results:
            if result.negative and result.boxes:
                raise ValueError('Detection boxes and negative confirmation cannot coexist')
            if not result.boxes:
                if result.negative:
                    negative = current_negative.get(result.sample_id) or AnnotationRecord(
                        uuid4(), result.sample_id, 'detect', None, 'negative', None, None
                    )
                    incoming.append(negative)
                continue
            for box in result.boxes:
                object_id = box.cvat_id
                if isinstance(object_id, bool) or not isinstance(object_id, int) or object_id <= 0:
                    raise ValueError('Persistent detection synchronization requires a native CVAT shape ID')
                if object_id in object_ids:
                    raise ValueError('CVAT shape IDs must be unique within a detection job')
                object_ids.add(object_id)
                existing = by_object_id.get(object_id)
                if existing is not None and existing.sample_id != result.sample_id:
                    raise ValueError('A known CVAT shape ID cannot move to another frame')
                annotation_id = existing.annotation_id if existing is not None else uuid4()
                geometry = box.geometry
                incoming.append(
                    AnnotationRecord(
                        annotation_id,
                        result.sample_id,
                        'detect',
                        None,
                        'rectangle',
                        geometry.label,
                        [[geometry.x1, geometry.y1], [geometry.x2, geometry.y2]],
                    )
                )
                bindings.append(CvatBinding(result.sample_id, 'shape', object_id, annotation_id))

        incoming_ids = {record.id for record in incoming}
        delete_ids = frozenset(record.id for record in current_detection if record.id not in incoming_ids)
        changes = plan_changes(current, tuple(incoming), delete_ids, self._task)
        final = {record.id: record for record in current}
        for annotation_id in changes.delete_ids:
            final.pop(annotation_id, None)
        final.update((record.id, record) for record in changes.upserts)
        return DetectionSync(
            changes, tuple(bindings), fingerprint_target('detect', self.images(), tuple(final.values()), self._task)
        )

    def commit_detection_sync(self, ref: JobRef, sync: DetectionSync) -> None:
        """Atomically commit a prepared detection change set and its native CVAT bindings."""
        self._repository.apply_changes(sync.changes, ref=ref, bindings=sync.bindings)

    def prepare_target_sync(self, target: str, job: EditJob, results: tuple[EditFrameResult, ...]) -> TargetSync:
        """Validate exact crop coverage and plan one target-scoped synchronization."""
        images = self.images()
        current = self._repository.annotations()
        return prepare_sync(target, job, results, images, current, self._repository.bindings(job.ref), self._task)

    def commit_target_sync(self, job: EditJob, sync: TargetSync) -> None:
        """Atomically commit prepared downstream annotations and their native CVAT bindings."""
        self._repository.apply_changes(sync.changes, ref=job.ref, bindings=sync.bindings)
