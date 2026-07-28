from collections.abc import Iterable

from .annotation import Annotation, Bbox, Keypoint, Points, Pose
from .geometry import rectangle_contains_point


def assemble_poses(
    annotations: tuple[Annotation, ...],
    keypoint_labels: tuple[str, ...] | None = None,
    *,
    match_ungrouped: bool = False,
) -> tuple[Annotation, ...]:
    strict = bool(keypoint_labels)
    catalog = keypoint_labels or ()
    if strict and len(set(catalog)) != len(catalog):
        raise ValueError('Keypoint labels must be unique')

    consumed: set[int] = set()
    poses: dict[int, Pose] = {}
    groups: dict[int | str | object, list[int]] = {}
    for index, annotation in enumerate(annotations):
        if annotation.group is not None:
            groups.setdefault(annotation.group, []).append(index)

    for indices in groups.values():
        members = tuple(annotations[index] for index in indices)
        bboxes = tuple(member for member in members if isinstance(member, Bbox))
        points = tuple(member for member in members if isinstance(member, Points) and len(member.points) == 1)
        valid = len(bboxes) == 1 and len(points) > 0 and len(bboxes) + len(points) == len(members)
        if not valid:
            if strict:
                raise ValueError('Grouped annotations do not form a pose')
            continue
        bbox = bboxes[0]
        poses[indices[members.index(bbox)]] = _make_pose(bbox, points, catalog)
        consumed.update(indices)

    if match_ungrouped:
        bbox_indices = [
            index
            for index, annotation in enumerate(annotations)
            if index not in consumed and annotation.group is None and isinstance(annotation, Bbox)
        ]
        points_by_bbox: dict[int, list[Points]] = {}
        for index, annotation in enumerate(annotations):
            if (
                index in consumed
                or annotation.group is not None
                or not isinstance(annotation, Points)
                or len(annotation.points) != 1
            ):
                continue
            matches = [
                bbox_index
                for bbox_index in bbox_indices
                if rectangle_contains_point(annotations[bbox_index].bbox, annotation.points[0])
            ]
            if len(matches) != 1:
                raise ValueError('Ungrouped keypoint must match exactly one Bbox')
            points_by_bbox.setdefault(matches[0], []).append(annotation)
            consumed.add(index)
        for bbox_index, points in points_by_bbox.items():
            bbox = annotations[bbox_index]
            assert isinstance(bbox, Bbox)
            poses[bbox_index] = _make_pose(bbox, points, catalog)
            consumed.add(bbox_index)

    return tuple(
        poses[index] if index in poses else annotation
        for index, annotation in enumerate(annotations)
        if index not in consumed or index in poses
    )


def _make_pose(bbox: Bbox, points: Iterable[Points], catalog: tuple[str, ...]) -> Pose:
    keypoints = tuple(Keypoint(label=point.label, x=point.points[0][0], y=point.points[0][1]) for point in points)
    if catalog:
        labels = tuple(keypoint.label for keypoint in keypoints)
        if len(set(labels)) != len(labels) or set(labels) != set(catalog):
            raise ValueError('Pose keypoint labels do not match the catalog')
        keypoints = tuple(sorted(keypoints, key=lambda keypoint: catalog.index(keypoint.label)))
    return Pose(
        label=bbox.label,
        id=bbox.id,
        group=bbox.group,
        x1=bbox.x1,
        y1=bbox.y1,
        x2=bbox.x2,
        y2=bbox.y2,
        keypoints=keypoints,
    )
