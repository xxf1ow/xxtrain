import math
from collections.abc import Sequence
from uuid import UUID

from .annotation import Circle, Shape

BboxTuple = tuple[float, float, float, float]
PointTuple = tuple[float, float]


def rectangle_contains_point(rect: BboxTuple, point: PointTuple, wide: float = 0) -> bool:
    return (
        point[0] >= rect[0] - wide
        and point[0] <= rect[2] + wide
        and point[1] >= rect[1] - wide
        and point[1] <= rect[3] + wide
    )


def calculate_wide(area_rect: float, area_shape: float, base_length: float, base_wide: float) -> float:
    if base_wide <= 0:
        return 0
    ratio = area_shape / area_rect
    if ratio >= 0.5:
        scale = 1.0
    elif ratio < 0.1:
        scale = 0
    else:
        scale = ratio / 0.5
    return base_length * base_wide * scale


def rectangle_contains_shape(rect: BboxTuple, shape: Shape, wide: float = 0) -> bool:
    rect_width = rect[2] - rect[0]
    rect_height = rect[3] - rect[1]
    area_rect = rect_width * rect_height
    assert area_rect > 0
    base_length = max(rect_width, rect_height)

    if isinstance(shape, Circle):
        area_shape = math.pi * shape.radius**2
        tolerance = calculate_wide(area_rect, area_shape, base_length, wide)
        xmin, ymin, xmax, ymax = shape.bbox
        return (
            xmin >= rect[0] - tolerance
            and ymin >= rect[1] - tolerance
            and xmax <= rect[2] + tolerance
            and ymax <= rect[3] + tolerance
        )

    xmin, ymin, xmax, ymax = shape.bbox
    area_shape = (xmax - xmin) * (ymax - ymin)
    tolerance = calculate_wide(area_rect, area_shape, base_length, wide)
    return (
        xmin >= rect[0] - tolerance
        and ymin >= rect[1] - tolerance
        and xmax <= rect[2] + tolerance
        and ymax <= rect[3] + tolerance
    )


def calculate_iou(box1: BboxTuple, box2: BboxTuple) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0


def match_parent_children(
    parents: Sequence[Shape], children: Sequence[Shape], image_path: str = '', wide: float = 0, strict: bool = True
) -> dict[UUID, list[UUID]]:
    mapping: dict[UUID, list[UUID]] = {}
    for child in children:
        matched = [parent for parent in parents if rectangle_contains_shape(parent.bbox, child, wide)]
        if strict and not matched:
            raise ValueError(
                f'Child annotation (path: {image_path}, bbox: {child.bbox}) does not belong to any parent.'
            )
        if len(matched) > 1:
            raise ValueError(f'Child annotation (path: {image_path}) is ambiguous: it matches multiple parents')
        if matched:
            mapping.setdefault(matched[0].id, []).append(child.id)
    if strict:
        for parent in parents:
            if parent.id not in mapping:
                raise ValueError(f'Parent annotation (path: {image_path}) has no matching children.')
    return mapping
