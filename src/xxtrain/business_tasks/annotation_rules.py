from math import isfinite

from xxtrain.platform.contracts import EditAnnotation, JsonValue

_CLASS_LABELS = frozenset({'tl', 'tc', 'cl', 'cc'})


def validate_target_annotations(target: str, annotations: tuple[EditAnnotation, ...]) -> None:
    """Validate Point classification cardinality or pointer-line content.

    Empty input is valid partial work. Unknown targets and malformed annotations raise ``ValueError``.
    """
    if target == 'classify':
        _validate_classifications(annotations)
    elif target == 'segment':
        _validate_lines(annotations)
    else:
        raise ValueError(f'Unknown annotation target: {target!r}')


def target_complete(target: str, annotations: tuple[EditAnnotation, ...]) -> bool:
    """Return whether valid Point target annotations satisfy their cardinality rule."""
    validate_target_annotations(target, annotations)
    return bool(annotations)


def _validate_classifications(annotations: tuple[EditAnnotation, ...]) -> None:
    if len(annotations) > 1:
        raise ValueError('Point classification requires at most one annotation')
    for annotation in annotations:
        if annotation.kind != 'classification':
            raise ValueError('Point classification annotations require kind classification')
        if annotation.label not in _CLASS_LABELS:
            raise ValueError(f'Unknown Point classification label: {annotation.label!r}')
        if annotation.geometry is not None:
            raise ValueError('Point classification annotations require null geometry')


def _validate_lines(annotations: tuple[EditAnnotation, ...]) -> None:
    for annotation in annotations:
        if annotation.kind != 'polyline':
            raise ValueError('Point segmentation annotations require kind polyline')
        if annotation.label != '1':
            raise ValueError(f'Unknown Point segmentation label: {annotation.label!r}')
        first, second = _line_points(annotation.geometry)
        if first == second:
            raise ValueError('Point segmentation lines require two distinct points')


def _line_points(geometry: JsonValue) -> tuple[tuple[float, float], tuple[float, float]]:
    if not isinstance(geometry, list) or len(geometry) != 2:
        raise ValueError('Point segmentation lines require exactly two points')
    points: list[tuple[float, float]] = []
    for point in geometry:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError('Each Point segmentation line point requires exactly two coordinates')
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point):
            raise ValueError('Point segmentation line coordinates must be numbers')
        numeric = (float(point[0]), float(point[1]))
        if not all(isfinite(value) for value in numeric):
            raise ValueError('Point segmentation line coordinates must be finite')
        points.append(numeric)
    return points[0], points[1]
