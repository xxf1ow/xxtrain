import json
import os
import shutil
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from PIL import Image

from xxtrain.business_tasks.point import point_task_definition
from xxtrain.data import Bbox
from xxtrain.platform.contracts import (
    AnnotationRecord,
    CvatBinding,
    DetectionBox,
    DetectionSummary,
    DetectionSync,
    FrameResult,
    ImageInput,
    ImageRecord,
    JobRef,
    PlatformAccessError,
    PreparedJob,
    UploadResult,
)
from xxtrain.workspace_data.changes import plan_changes
from xxtrain.workspace_data.dedup import SIMILARITY_DISTANCE, hamming_distance, image_sha256, perceptual_hash
from xxtrain.workspace_data.labelme import detection_document
from xxtrain.workspace_data.repository import AnnotationRepository

_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


class WorkspaceData:
    """Access registered images and authoritative SQLite annotations beneath one workspace root."""

    def __init__(self, workspace_dir: Path) -> None:
        self._root = Path(workspace_dir)
        self._images_dir = self._root / 'images'
        self._require_directory(self._images_dir, writable=True)
        self._task = point_task_definition()
        self._repository = AnnotationRepository(self._root / 'annotations.db', self._task)

    @staticmethod
    def _require_directory(path: Path, *, writable: bool) -> None:
        access = os.R_OK | os.X_OK | (os.W_OK if writable else 0)
        if not path.is_dir():
            raise PlatformAccessError(f'Directory does not exist: {path}')
        if not os.access(path, access):
            raise PlatformAccessError(f'Directory is not accessible: {path}')

    def images(self) -> tuple[ImageInput, ...]:
        """Return registered image facts and detection boxes in registration order."""
        annotations = self._repository.annotations(step_key='detect')
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
        return self._repository.detection_summary()

    def detection_fingerprint(self) -> str:
        """Hash registered image identities and normalized detection business content."""
        return self._fingerprint(self._repository.annotations())

    def materialize_detection_source(self, root: Path) -> Path:
        """Export completed database samples into a disposable LabelMe source tree."""
        source_root = Path(root) / 'src'
        group_root = source_root / 'workspace'
        images_dir = group_root / 'imgs'
        annotations_dir = group_root / 'anns_seg'
        images_dir.mkdir(parents=True)
        annotations_dir.mkdir()
        (source_root / 'labels.txt').write_text('Point\n', encoding='utf-8')

        annotations = self._repository.annotations(step_key='detect')
        by_image: dict[str, list[AnnotationRecord]] = {}
        for record in annotations:
            by_image.setdefault(record.image_id, []).append(record)
        for image in self._repository.images():
            records = tuple(by_image.get(image.id, ()))
            if not any(record.kind in {'rectangle', 'negative'} for record in records):
                continue
            image_path = self._root / image.relative_path
            document = detection_document(
                image_path=f'../imgs/{image_path.name}', width=image.width, height=image.height, annotations=records
            )
            shutil.copy2(image_path, images_dir / image_path.name)
            (annotations_dir / f'{image.id}.json').write_text(
                json.dumps(document, ensure_ascii=False, allow_nan=False), encoding='utf-8'
            )
        return Path(root)

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
            if not result.boxes:
                negative = current_negative.get(result.sample_id)
                if negative is not None:
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
        return DetectionSync(changes, tuple(bindings), self._fingerprint(tuple(final.values())))

    def commit_detection_sync(self, ref: JobRef, sync: DetectionSync) -> None:
        """Atomically commit a prepared detection change set and its native CVAT bindings."""
        self._repository.apply_changes(sync.changes, ref=ref, bindings=sync.bindings)

    def _fingerprint(self, annotations: tuple[AnnotationRecord, ...]) -> str:
        by_image: dict[str, list[AnnotationRecord]] = {}
        for record in annotations:
            if record.step_key == 'detect':
                by_image.setdefault(record.image_id, []).append(record)
        records = []
        for image in self._repository.images():
            detection_records = by_image.get(image.id, ())
            boxes = [
                {'label': record.label, 'points': [[float(value) for value in point] for point in record.geometry]}
                for record in detection_records
                if record.kind == 'rectangle'
            ]
            if boxes:
                detection = {
                    'boxes': sorted(
                        boxes,
                        key=lambda box: json.dumps(box, ensure_ascii=False, sort_keys=True, separators=(',', ':')),
                    )
                }
            elif any(record.kind == 'negative' for record in detection_records):
                detection = {'negative': True}
            else:
                detection = None
            records.append({'sha256': image.id, 'detection': detection})
        payload = json.dumps(
            sorted(records, key=lambda record: record['sha256']),
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        ).encode('utf-8')
        return sha256(payload).hexdigest()
