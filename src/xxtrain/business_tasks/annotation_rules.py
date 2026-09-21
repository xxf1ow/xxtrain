from math import isfinite

from xxtrain.platform.contracts import EditAnnotation, JsonValue

from .definition import StepDefinition


class AnnotationValidationError(ValueError):
    """A task-rule validation failure with a stable, presentation-safe reason code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def validate_step_annotations(step: StepDefinition, annotations: tuple[EditAnnotation, ...]) -> None:
    """Validate annotations against one task-owned policy; empty partial work is valid."""
    policy = step.annotation
    assert policy is not None
    if any(annotation.kind == 'negative' for annotation in annotations) and len(annotations) != 1:
        raise AnnotationValidationError(
            'negative_conflict', f'{step.display_name} negative annotation conflicts with other annotations'
        )
    if policy.maximum_annotations is not None and len(annotations) > policy.maximum_annotations:
        raise AnnotationValidationError(
            'cardinality', f'{step.display_name} allows at most {policy.maximum_annotations} annotations'
        )
    for annotation in annotations:
        if annotation.kind == 'negative':
            if policy.negative_label is None:
                raise AnnotationValidationError(
                    'annotation_type', f'{step.display_name} does not allow negative annotations'
                )
            if annotation.label is not None or annotation.geometry is not None:
                raise AnnotationValidationError(
                    'geometry', f'{step.display_name} negative annotations require null label and geometry'
                )
            continue
        if annotation.kind not in step.kinds:
            raise AnnotationValidationError(
                'annotation_type', f'{step.display_name} does not allow annotation kind {annotation.kind!r}'
            )
        if annotation.label not in step.labels:
            raise AnnotationValidationError('label', f'{step.display_name} does not allow label {annotation.label!r}')
        if annotation.kind == 'classification':
            if annotation.geometry is not None:
                raise AnnotationValidationError(
                    'geometry', f'{step.display_name} classification annotations require null geometry'
                )
        elif annotation.kind == 'polyline':
            points = _points(annotation.geometry, policy.point_count)
            if len(points) == 2 and points[0] == points[1]:
                raise AnnotationValidationError(
                    'coincident_points', f'{step.display_name} line points must be distinct'
                )
        elif annotation.kind == 'rectangle':
            _points(annotation.geometry, 2)
        elif annotation.kind == 'polygon':
            if len(_points(annotation.geometry, None)) < 3:
                raise AnnotationValidationError(
                    'point_count', f'{step.display_name} polygons require at least three points'
                )


def step_complete(step: StepDefinition, annotations: tuple[EditAnnotation, ...]) -> bool:
    validate_step_annotations(step, annotations)
    assert step.annotation is not None
    return len(annotations) >= step.annotation.minimum_annotations


def _points(geometry: JsonValue, point_count: int | None) -> tuple[tuple[float, float], ...]:
    if not isinstance(geometry, list) or (point_count is not None and len(geometry) != point_count):
        raise AnnotationValidationError('point_count', f'Annotation geometry requires exactly {point_count} points')
    points: list[tuple[float, float]] = []
    for point in geometry:
        if not isinstance(point, list) or len(point) != 2:
            raise AnnotationValidationError('geometry', 'Annotation points require exactly two coordinates')
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point):
            raise AnnotationValidationError('geometry', 'Annotation coordinates must be numbers')
        numeric = (float(point[0]), float(point[1]))
        if not all(isfinite(value) for value in numeric):
            raise AnnotationValidationError('geometry', 'Annotation coordinates must be finite')
        points.append(numeric)
    return tuple(points)


def validate_target_annotations(target: str, annotations: tuple[EditAnnotation, ...]) -> None:
    from .point import point_task_definition

    if target not in {'classify', 'segment'}:
        raise ValueError(f'Unknown annotation target: {target!r}')
    try:
        validate_step_annotations(point_task_definition().step(target), annotations)
    except ValueError as error:
        legacy_messages = {
            '分类 allows at most 1 annotations': 'Point classification requires at most one annotation',
            'Annotation geometry requires exactly 2 points': 'Point segmentation lines require exactly two points',
            '指针分割 line points must be distinct': 'Point segmentation lines require two distinct points',
        }
        raise ValueError(legacy_messages.get(str(error), str(error))) from error


def target_complete(target: str, annotations: tuple[EditAnnotation, ...]) -> bool:
    from .point import point_task_definition

    if target not in {'classify', 'segment'}:
        raise ValueError(f'Unknown annotation target: {target!r}')
    return step_complete(point_task_definition().step(target), annotations)
