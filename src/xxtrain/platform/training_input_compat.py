import math
from uuid import UUID

from xxtrain.business_tasks.definition import TaskDefinition
from xxtrain.business_tasks.loader import LEGACY_POINT_TASK_ENTRY
from xxtrain.platform.contracts import AnnotationRecord, FrameMapping, ImageInput
from xxtrain.workspace_data.editing import fingerprint_target, fingerprint_training
from xxtrain.workspace_data.inputs import AxisAlignedRectangleInputs, OriginalImageInputs
from xxtrain.workspace_data.legacy_fingerprints import legacy_point_fingerprint
from xxtrain.workspace_data.repository import AnnotationRepository

_FROZEN_POINT_CONVERSIONS = {
    'detect': ('point-detect-single-class-yolo-v1', ('Point',), 'detect', True),
    'classify': ('point-classify-indexed-directories-v1', ('tl', 'tc', 'cl', 'cc'), 'classify', False),
    'segment': ('point-segment-triangle-mask-v1', ('Point',), 'segment', False),
}


def initialize_input_compatibility(training_service) -> None:
    """Associate current Point inputs with old runs only when frozen projection and conversion semantics match.

    The caller invokes this once before request handling or training reconciliation. Repeated startup calls are
    idempotent. Historical runs whose input cannot be proven remain accessible only by run ID.
    """
    annotations = training_service.annotations
    config = training_service.config
    with annotations.mutation(config.owner_user_id):
        task = annotations.data.task
        if not _is_frozen_point_conversion(task):
            return
        images = annotations.data.images()
        records = AnnotationRepository(config.workspace_dir / 'annotations.db', task).annotations()
        identities = {}
        for target in _FROZEN_POINT_CONVERSIONS:
            if not _has_frozen_point_projection(task, target, images, records):
                continue
            identities[target] = (
                legacy_point_fingerprint(target, images, records),
                fingerprint_training(fingerprint_target(target, images, records, task), target, task),
            )
        for run in training_service.store.list_workspace(config.workspace_id):
            if run.task_entry != LEGACY_POINT_TASK_ENTRY or run.target not in identities:
                continue
            legacy_fingerprint, current_fingerprint = identities[run.target]
            if run.fingerprint != legacy_fingerprint:
                continue
            training_service.store.associate_input(
                run.user_id, run.workspace_id, run.target, current_fingerprint, run.id
            )


def _is_frozen_point_conversion(task: TaskDefinition) -> bool:
    if task.key != 'point' or {step.key for step in task.steps} != set(_FROZEN_POINT_CONVERSIONS):
        return False
    for target, expected in _FROZEN_POINT_CONVERSIONS.items():
        step = task.step(target)
        training = step.training
        if training is None or step.annotation is None:
            return False
        actual = (
            training.conversion_key,
            training.labels,
            training.settings.task_type.value,
            step.annotation.negative_label is not None,
        )
        if actual != expected:
            return False
    return True


def _has_frozen_point_projection(task: TaskDefinition, target: str, images, records) -> bool:
    step = task.step(target)
    adapter = step.input_adapter
    expected_type = OriginalImageInputs if target == 'detect' else AxisAlignedRectangleInputs
    if type(adapter) is not expected_type:
        return False
    return adapter.mappings(images, records, step) == _frozen_point_mappings(target, images, records)


def _frozen_point_mappings(
    target: str, images: tuple[ImageInput, ...], records: tuple[AnnotationRecord, ...]
) -> tuple[FrameMapping, ...]:
    if target == 'detect':
        return tuple(
            FrameMapping(image.sample_id, image.sample_id, None, (0, 0, image.width, image.height)) for image in images
        )

    by_image: dict[str, list[AnnotationRecord]] = {}
    for record in records:
        if record.step_key == 'detect' and record.kind == 'rectangle':
            by_image.setdefault(record.image_id, []).append(record)
    mappings = []
    frame_ids: set[str] = set()
    for image in images:
        for record in by_image.get(image.sample_id, ()):
            parent_id = record.id
            if not isinstance(parent_id, UUID):
                raise ValueError('Crop source annotations require UUID identities')
            frame_id = str(parent_id)
            if frame_id in frame_ids:
                raise ValueError(f'Duplicate crop frame ID {frame_id}')
            frame_ids.add(frame_id)
            geometry = record.geometry
            if not isinstance(geometry, list) or len(geometry) != 2:
                raise ValueError(f'Crop source {record.id} requires rectangle geometry')
            first, second = geometry
            if not isinstance(first, list) or not isinstance(second, list) or len(first) != 2 or len(second) != 2:
                raise ValueError(f'Crop source {record.id} requires rectangle geometry')
            x1, y1, x2, y2 = float(first[0]), float(first[1]), float(second[0]), float(second[1])
            bounds = (
                max(0, min(image.width, math.floor(min(x1, x2)))),
                max(0, min(image.height, math.floor(min(y1, y2)))),
                max(0, min(image.width, math.ceil(max(x1, x2)))),
                max(0, min(image.height, math.ceil(max(y1, y2)))),
            )
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise ValueError(f'Crop frame {frame_id} has no pixels after clamping')
            mappings.append(FrameMapping(frame_id, image.sample_id, parent_id, bounds))
    return tuple(mappings)
