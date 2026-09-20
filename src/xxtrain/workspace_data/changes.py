from uuid import UUID

from xxtrain.business_tasks.definition import TaskDefinition
from xxtrain.platform.contracts import AnnotationChanges, AnnotationRecord, JsonValue


def plan_changes(
    current: tuple[AnnotationRecord, ...],
    incoming: tuple[AnnotationRecord, ...],
    delete_ids: frozenset[UUID],
    task: TaskDefinition,
) -> AnnotationChanges:
    """Plan object-scoped changes without persisting records or retaining history.

    Existing IDs keep their image, task step, and parent assignment. Invalid records, duplicate IDs, and an
    ID requested for both upsert and deletion raise ``ValueError``. Content changes and additions invalidate
    dependent task steps even when no downstream annotation currently exists.
    """
    current_by_id = _records_by_id(current, subject='Current annotations')
    incoming_by_id = _records_by_id(incoming, subject='Incoming annotations')
    if any(not isinstance(annotation_id, UUID) for annotation_id in delete_ids):
        raise ValueError('Deleted annotation ids must be UUIDs')
    if incoming_by_id.keys() & delete_ids:
        raise ValueError('An annotation cannot be saved and deleted together')

    children: dict[UUID, set[UUID]] = {}
    for record in current:
        if record.parent_id is not None:
            children.setdefault(record.parent_id, set()).add(record.id)

    upserts: list[AnnotationRecord] = []
    mutations: list[AnnotationRecord] = []
    for record in incoming:
        existing = current_by_id.get(record.id)
        if existing is not None:
            if _assignment(existing) != _assignment(record):
                raise ValueError('Existing annotation ids cannot change image, step, or parent assignment')
            if _content(existing) == _content(record):
                continue
        upserts.append(record)
        mutations.append(record)

    deleted: set[UUID] = set()
    invalidated: set[str] = set()
    for annotation_id in delete_ids:
        existing = current_by_id.get(annotation_id)
        if existing is None:
            continue
        deleted.update(_subtree(annotation_id, children))
        mutations.append(existing)

    for record in mutations:
        dependents = task.dependent_steps(record.step_key)
        invalidated.update(dependents)
        if not dependents:
            continue
        anchor_ids = {record.id}
        if record.parent_id is not None:
            anchor_ids.add(record.parent_id)
        for candidate in current:
            shares_source = candidate.image_id == record.image_id and candidate.parent_id == record.parent_id
            if candidate.step_key in dependents and (candidate.parent_id in anchor_ids or shares_source):
                deleted.update(_subtree(candidate.id, children))

    return AnnotationChanges(
        tuple(record for record in upserts if record.id not in deleted), frozenset(deleted), frozenset(invalidated)
    )


def _records_by_id(records: tuple[AnnotationRecord, ...], *, subject: str) -> dict[UUID, AnnotationRecord]:
    by_id = {record.id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError(f'{subject} contain duplicate ids')
    if any(not isinstance(annotation_id, UUID) for annotation_id in by_id):
        raise ValueError(f'{subject} ids must be UUIDs')
    return by_id


def _assignment(record: AnnotationRecord) -> tuple[str, str, UUID | None]:
    return record.image_id, record.step_key, record.parent_id


def _content(record: AnnotationRecord) -> tuple[str, str | None, object]:
    return record.kind, record.label, _normalized_json(record.geometry)


def _normalized_json(value: JsonValue) -> object:
    if value is None:
        return ('null',)
    if isinstance(value, bool):
        return ('boolean', value)
    if isinstance(value, str):
        return ('string', value)
    if isinstance(value, (int, float)):
        return ('number', float(value))
    if isinstance(value, list):
        return ('array', tuple(_normalized_json(item) for item in value))
    if isinstance(value, dict):
        return ('object', tuple(sorted((key, _normalized_json(item)) for key, item in value.items())))
    raise ValueError('Annotation geometry must be valid JSON')


def _subtree(root_id: UUID, children: dict[UUID, set[UUID]]) -> set[UUID]:
    result: set[UUID] = set()
    pending = [root_id]
    while pending:
        annotation_id = pending.pop()
        if annotation_id in result:
            continue
        result.add(annotation_id)
        pending.extend(children.get(annotation_id, ()))
    return result
