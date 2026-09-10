import math
from collections.abc import Sequence
from uuid import UUID

from .annotation import Circle, ImageInfo, Polygon, Shape

BboxTuple = tuple[float, float, float, float]
PointTuple = tuple[float, float]


def clip_polygon_to_image(polygon: Polygon, image_info: ImageInfo) -> Polygon | None:
    """Clip a polygon to the image rectangle, returning None for zero visible area."""
    points = polygon.points
    bounds = ((0, 0.0, True), (0, image_info.width, False), (1, 0.0, True), (1, image_info.height, False))

    for axis, limit, keep_greater in bounds:

        def inside(point: PointTuple) -> bool:
            return point[axis] >= limit if keep_greater else point[axis] <= limit

        def intersection(start: PointTuple, end: PointTuple) -> PointTuple:
            ratio = (limit - start[axis]) / (end[axis] - start[axis])
            return (start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio)

        clipped: list[PointTuple] = []
        for index, end in enumerate(points):
            start = points[index - 1]
            start_inside, end_inside = inside(start), inside(end)
            if end_inside != start_inside:
                clipped.append(intersection(start, end))
            if end_inside:
                clipped.append(end)
        points = tuple(clipped)
        if not points:
            return None

    cleaned = tuple(point for index, point in enumerate(points) if point != points[index - 1])
    if len(cleaned) < 3:
        return None
    area = (
        abs(
            sum(
                first[0] * second[1] - second[0] * first[1] for first, second in zip(cleaned, cleaned[1:] + cleaned[:1])
            )
        )
        / 2
    )
    if area == 0:
        return None
    return Polygon(label=polygon.label, id=polygon.id, group=polygon.group, points=cleaned)


def validate_obb(polygon: Polygon, *, tolerance: float = 1e-6) -> None:
    """Raise ValueError unless *polygon* is a four-point oriented bounding box."""
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError):
        raise ValueError('OBB tolerance must be a positive finite number') from None
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('OBB tolerance must be a positive finite number')

    points = polygon.points
    if len(points) != 4:
        raise ValueError('OBB requires exactly four points (invalid point count)')

    edges = tuple(
        (next_point[0] - point[0], next_point[1] - point[1])
        for point, next_point in zip(points, points[1:] + points[:1])
    )
    lengths = tuple(math.hypot(*edge) for edge in edges)
    if any(not math.isfinite(length) for length in lengths):
        raise ValueError('OBB has a non-finite edge')
    if any(length == 0 for length in lengths):
        raise ValueError('OBB has a zero-length edge')
    directions = tuple((edge[0] / length, edge[1] / length) for edge, length in zip(edges, lengths))

    def cross(first: PointTuple, second: PointTuple) -> float:
        return first[0] * second[1] - first[1] * second[0]

    def subtract(first: PointTuple, second: PointTuple) -> PointTuple:
        return first[0] - second[0], first[1] - second[1]

    def orientation(start: PointTuple, end: PointTuple, point: PointTuple) -> float:
        edge = subtract(end, start)
        point_offset = subtract(point, start)
        point_length = math.hypot(*point_offset)
        if point_length == 0:
            return 0
        edge_length = math.hypot(*edge)
        return cross(
            (edge[0] / edge_length, edge[1] / edge_length),
            (point_offset[0] / point_length, point_offset[1] / point_length),
        )

    def segments_intersect(
        first_start: PointTuple, first_end: PointTuple, second_start: PointTuple, second_end: PointTuple
    ) -> bool:
        first_start_turn = orientation(first_start, first_end, second_start)
        first_end_turn = orientation(first_start, first_end, second_end)
        second_start_turn = orientation(second_start, second_end, first_start)
        second_end_turn = orientation(second_start, second_end, first_end)

        def on_segment(point: PointTuple, start: PointTuple, end: PointTuple) -> bool:
            return min(start[0], end[0]) <= point[0] <= max(start[0], end[0]) and min(start[1], end[1]) <= point[
                1
            ] <= max(start[1], end[1])

        if abs(first_start_turn) <= tolerance and on_segment(second_start, first_start, first_end):
            return True
        if abs(first_end_turn) <= tolerance and on_segment(second_end, first_start, first_end):
            return True
        if abs(second_start_turn) <= tolerance and on_segment(first_start, second_start, second_end):
            return True
        if abs(second_end_turn) <= tolerance and on_segment(first_end, second_start, second_end):
            return True
        return (
            (first_start_turn > tolerance and first_end_turn < -tolerance)
            or (first_start_turn < -tolerance and first_end_turn > tolerance)
        ) and (
            (second_start_turn > tolerance and second_end_turn < -tolerance)
            or (second_start_turn < -tolerance and second_end_turn > tolerance)
        )

    if segments_intersect(points[0], points[1], points[2], points[3]) or segments_intersect(
        points[1], points[2], points[3], points[0]
    ):
        raise ValueError('OBB has a self-intersection')

    turns = tuple(cross(direction, directions[(index + 1) % 4]) for index, direction in enumerate(directions))
    if any(abs(turn) <= tolerance for turn in turns) or not (
        all(turn > 0 for turn in turns) or all(turn < 0 for turn in turns)
    ):
        raise ValueError('OBB must be convex')

    if any(
        abs(direction[0] * directions[(index + 1) % 4][0] + direction[1] * directions[(index + 1) % 4][1]) > tolerance
        for index, direction in enumerate(directions)
    ):
        raise ValueError('OBB has non-perpendicular adjacent edges')
    if abs(cross(directions[0], directions[2])) > tolerance or abs(cross(directions[1], directions[3])) > tolerance:
        raise ValueError('OBB has non-parallel opposite edges')


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
