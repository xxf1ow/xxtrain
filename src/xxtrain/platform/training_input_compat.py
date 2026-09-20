from xxtrain.business_tasks.definition import TaskDefinition
from xxtrain.business_tasks.loader import LEGACY_POINT_TASK_ENTRY
from xxtrain.workspace_data.editing import fingerprint_target, fingerprint_training
from xxtrain.workspace_data.legacy_fingerprints import legacy_point_fingerprint
from xxtrain.workspace_data.repository import AnnotationRepository

_FROZEN_POINT_CONVERSIONS = {
    'detect': ('point-detect-single-class-yolo-v1', ('Point',), 'detect', True),
    'classify': ('point-classify-indexed-directories-v1', ('tl', 'tc', 'cl', 'cc'), 'classify', False),
    'segment': ('point-segment-triangle-mask-v1', ('Point',), 'segment', False),
}


def initialize_input_compatibility(training_service) -> None:
    """Associate exact current Point inputs with immutable pre-definition run identities.

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
        identities = {
            target: (
                legacy_point_fingerprint(target, images, records),
                fingerprint_training(fingerprint_target(target, images, records, task), target, task),
            )
            for target in _FROZEN_POINT_CONVERSIONS
        }
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
