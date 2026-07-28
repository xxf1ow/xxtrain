import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..annotation import Annotation, Bbox, ImageInfo, Keypoint, Polygon, Pose
from ..labels import LabelCatalog


@dataclass(frozen=True, slots=True)
class CocoImage:
    file_name: str
    info: ImageInfo
    annotations: tuple[Annotation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.file_name, str):
            raise TypeError('COCO image file_name must be a string')
        if not self.file_name:
            raise ValueError('COCO image file_name must not be empty')
        if not isinstance(self.info, ImageInfo):
            raise TypeError('COCO image info must be ImageInfo')
        if type(self.annotations) is not tuple:
            raise TypeError('COCO image annotations must be a tuple')
        if not all(isinstance(annotation, Annotation) for annotation in self.annotations):
            raise TypeError('COCO image annotations must contain only Annotation values')


@dataclass(frozen=True, slots=True)
class CocoDoc:
    labels: LabelCatalog
    images: tuple[CocoImage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.labels, LabelCatalog):
            raise TypeError('COCO document labels must be LabelCatalog')
        if type(self.images) is not tuple:
            raise TypeError('COCO document images must be a tuple')
        if not all(isinstance(image, CocoImage) for image in self.images):
            raise TypeError('COCO document images must contain only CocoImage values')
        file_names = tuple(image.file_name for image in self.images)
        if len(file_names) != len(set(file_names)):
            raise ValueError('COCO image file_name values must be unique')


def _collections(root: object) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if not isinstance(root, dict):
        raise TypeError('COCO document root must be an object')

    result: list[list[dict[str, object]]] = []
    for name in ('categories', 'images', 'annotations'):
        if name not in root:
            raise ValueError(f'COCO document is missing {name}')
        values = root[name]
        if type(values) is not list:
            raise TypeError(f'COCO {name} must be a list')
        if not all(isinstance(value, dict) for value in values):
            raise TypeError(f'COCO {name} entries must be objects')
        result.append(values)
    return result[0], result[1], result[2]


def _required(value: dict[str, object], name: str) -> object:
    if name not in value:
        raise ValueError(f'COCO entry is missing {name}')
    return value[name]


def _id(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'COCO {name} must be an integer')
    return value


def _name(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f'COCO {name} must be a string')
    if not value:
        raise ValueError(f'COCO {name} must not be empty')
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'COCO {name} must be numeric')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'COCO {name} must be finite')
    return result


def _bbox(value: object) -> tuple[float, float, float, float]:
    if type(value) is not list:
        raise TypeError('COCO bbox must be a list')
    if len(value) != 4:
        raise ValueError('COCO bbox must contain exactly four values')
    x, y, width, height = (_number(token, 'bbox coordinate') for token in value)
    if width <= 0 or height <= 0:
        raise ValueError('COCO bbox width and height must be positive')
    return x, y, x + width, y + height


def _keypoint_names(category: dict[str, object]) -> tuple[str, ...]:
    values = category.get('keypoints', [])
    if type(values) is not list:
        raise TypeError('COCO category keypoints must be a list')
    names = tuple(_name(value, 'keypoint name') for value in values)
    if len(names) != len(set(names)):
        raise ValueError('COCO category keypoint names must be unique')
    return names


def _keypoints(values: object, names: tuple[str, ...]) -> tuple[Keypoint, ...]:
    if type(values) is not list:
        raise TypeError('COCO annotation keypoints must be a list')
    if len(values) != 3 * len(names):
        raise ValueError('COCO annotation keypoints must contain exactly three values per category keypoint')

    keypoints = []
    for index, name in enumerate(names):
        x = _number(values[3 * index], 'keypoint coordinate')
        y = _number(values[3 * index + 1], 'keypoint coordinate')
        visibility = values[3 * index + 2]
        if isinstance(visibility, bool) or not isinstance(visibility, int) or visibility not in (0, 1, 2):
            raise ValueError('COCO keypoint visibility must be the integer 0, 1, or 2')
        keypoints.append(Keypoint(label=name, x=x, y=y, visibility=visibility))
    return tuple(keypoints)


def _contours(value: object) -> tuple[tuple[tuple[float, float], ...], ...]:
    if isinstance(value, dict):
        raise ValueError('COCO RLE segmentation is not supported')
    if type(value) is not list:
        raise TypeError('COCO segmentation must be a list')

    contours = []
    for contour in value:
        if type(contour) is not list:
            raise TypeError('COCO polygon contour must be a list')
        if len(contour) < 6 or len(contour) % 2:
            raise ValueError('COCO polygon contour must contain an even number of at least six values')
        coordinates = tuple(_number(token, 'polygon coordinate') for token in contour)
        contours.append(tuple(zip(coordinates[::2], coordinates[1::2], strict=True)))
    return tuple(contours)


def _decode_annotation(
    value: dict[str, object], annotation_id: int, category: tuple[str, tuple[str, ...]]
) -> tuple[Annotation, ...]:
    label, keypoint_names = category
    bbox = _bbox(value['bbox']) if 'bbox' in value else None

    raw_keypoints = value.get('keypoints', [])
    if type(raw_keypoints) is not list:
        raise TypeError('COCO annotation keypoints must be a list')
    has_keypoints = bool(raw_keypoints)

    raw_segmentation = value.get('segmentation', [])
    contours = _contours(raw_segmentation)
    outputs: list[Annotation] = []

    if has_keypoints:
        if bbox is None:
            raise ValueError('COCO pose annotation requires bbox')
        x1, y1, x2, y2 = bbox
        outputs.append(
            Pose(label=label, x1=x1, y1=y1, x2=x2, y2=y2, keypoints=_keypoints(raw_keypoints, keypoint_names))
        )

    outputs.extend(Polygon(label=label, points=contour) for contour in contours)
    if not outputs:
        if bbox is None:
            raise ValueError('COCO annotation requires bbox, keypoints, or segmentation')
        x1, y1, x2, y2 = bbox
        outputs.append(Bbox(label=label, x1=x1, y1=y1, x2=x2, y2=y2))

    if len(outputs) > 1:
        return tuple(annotation.wrap(group=annotation_id) for annotation in outputs)
    return tuple(outputs)


def read_coco(json_path: str | Path) -> CocoDoc:
    root = json.loads(Path(json_path).read_text(encoding='utf-8'))
    categories, images, annotations = _collections(root)

    category_by_id: dict[int, tuple[str, tuple[str, ...]]] = {}
    category_names = []
    for category in categories:
        category_id = _id(_required(category, 'id'), 'category id')
        if category_id in category_by_id:
            raise ValueError(f'Duplicate COCO category id: {category_id}')
        name = _name(_required(category, 'name'), 'category name')
        category_by_id[category_id] = name, _keypoint_names(category)
        category_names.append(name)

    image_by_id: dict[int, tuple[str, ImageInfo]] = {}
    image_annotations: dict[int, list[Annotation]] = {}
    image_order = []
    for image in images:
        image_id = _id(_required(image, 'id'), 'image id')
        if image_id in image_by_id:
            raise ValueError(f'Duplicate COCO image id: {image_id}')
        file_name = _name(_required(image, 'file_name'), 'image file_name')
        info = ImageInfo(
            width=_number(_required(image, 'width'), 'image width'),
            height=_number(_required(image, 'height'), 'image height'),
        )
        image_by_id[image_id] = file_name, info
        image_annotations[image_id] = []
        image_order.append(image_id)

    annotations_with_id = []
    annotation_ids = set()
    for annotation in annotations:
        annotation_id = _id(_required(annotation, 'id'), 'annotation id')
        if annotation_id in annotation_ids:
            raise ValueError(f'Duplicate COCO annotation id: {annotation_id}')
        annotation_ids.add(annotation_id)
        annotations_with_id.append((annotation, annotation_id))

    for annotation, annotation_id in annotations_with_id:
        image_id = _id(_required(annotation, 'image_id'), 'annotation image_id')
        category_id = _id(_required(annotation, 'category_id'), 'annotation category_id')
        if image_id not in image_by_id:
            raise ValueError(f'Unknown COCO annotation image id: {image_id}')
        if category_id not in category_by_id:
            raise ValueError(f'Unknown COCO annotation category id: {category_id}')
        image_annotations[image_id].extend(_decode_annotation(annotation, annotation_id, category_by_id[category_id]))

    labels = LabelCatalog(tuple(category_names))
    coco_images = tuple(
        CocoImage(
            file_name=image_by_id[image_id][0],
            info=image_by_id[image_id][1],
            annotations=tuple(image_annotations[image_id]),
        )
        for image_id in image_order
    )
    return CocoDoc(labels=labels, images=coco_images)


def _finite_derived(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _positive_finite_derived(value: float, name: str) -> float:
    result = _finite_derived(value, name)
    if result <= 0:
        raise ValueError(f'{name} must be positive')
    return result


def _xywh(annotation: Bbox | Pose) -> list[float]:
    width = _positive_finite_derived(annotation.x2 - annotation.x1, 'COCO bbox width')
    height = _positive_finite_derived(annotation.y2 - annotation.y1, 'COCO bbox height')
    return [annotation.x1, annotation.y1, width, height]


def _polygon_area(polygon: Polygon) -> float:
    points = polygon.points
    origin_x, origin_y = points[0]
    shifted = tuple(
        (
            _finite_derived(x - origin_x, 'COCO shifted polygon coordinate'),
            _finite_derived(y - origin_y, 'COCO shifted polygon coordinate'),
        )
        for x, y in points
    )
    terms = []
    for (x1, y1), (x2, y2) in zip(shifted, shifted[1:] + shifted[:1], strict=True):
        first_product = _finite_derived(x1 * y2, 'COCO polygon cross product')
        second_product = _finite_derived(x2 * y1, 'COCO polygon cross product')
        terms.append(_finite_derived(first_product - second_product, 'COCO polygon cross product'))
    try:
        double_area = math.fsum(terms)
    except OverflowError as error:
        raise ValueError('COCO polygon area must be finite') from error
    area = abs(_finite_derived(double_area, 'COCO polygon area')) / 2
    return _positive_finite_derived(area, 'COCO polygon area')


def _segmentation(polygons: tuple[Polygon, ...]) -> list[list[float]]:
    return [[coordinate for point in polygon.points for coordinate in point] for polygon in polygons]


def _polygon_bbox(polygons: tuple[Polygon, ...]) -> list[float]:
    points = tuple(point for polygon in polygons for point in polygon.points)
    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    width = _positive_finite_derived(x2 - x1, 'COCO polygon bbox width')
    height = _positive_finite_derived(y2 - y1, 'COCO polygon bbox height')
    return [x1, y1, width, height]


def _annotation_units(annotations: tuple[Annotation, ...]) -> list[list[Annotation]]:
    units: list[list[Annotation]] = []
    group_indexes: dict[tuple[type[object], object], int] = {}
    for annotation in annotations:
        if annotation.group is None:
            units.append([annotation])
        else:
            group_key = type(annotation.group), annotation.group
            if group_key in group_indexes:
                units[group_indexes[group_key]].append(annotation)
            else:
                group_indexes[group_key] = len(units)
                units.append([annotation])
    return units


def _pose_schemas(doc: CocoDoc, category_ids: dict[str, int]) -> dict[str, tuple[str, ...]]:
    schemas: dict[str, tuple[str, ...]] = {}
    supported_types = (Bbox, Polygon, Pose)
    for image in doc.images:
        for annotation in image.annotations:
            if annotation.label not in category_ids:
                raise ValueError(f'Unknown annotation label: {annotation.label}')
            if type(annotation) not in supported_types:
                raise TypeError(f'Unsupported COCO annotation type: {type(annotation).__name__}')
            if type(annotation) is Pose:
                schema = tuple(keypoint.label for keypoint in annotation.keypoints)
                if annotation.label in schemas and schemas[annotation.label] != schema:
                    raise ValueError(f'Inconsistent COCO keypoint schema for label: {annotation.label}')
                schemas[annotation.label] = schema
    return schemas


def _encode_unit(
    unit: list[Annotation], annotation_id: int, image_id: int, category_ids: dict[str, int]
) -> dict[str, object]:
    labels = {annotation.label for annotation in unit}
    if len(labels) != 1:
        raise ValueError('Grouped COCO annotations must use one label')

    bboxes = tuple(annotation for annotation in unit if type(annotation) is Bbox)
    polygons = tuple(annotation for annotation in unit if type(annotation) is Polygon)
    poses = tuple(annotation for annotation in unit if type(annotation) is Pose)
    if len(unit) > 1:
        if bboxes:
            raise ValueError('A grouped COCO annotation cannot contain a Bbox')
        if len(poses) > 1:
            raise ValueError('A grouped COCO annotation cannot contain more than one Pose')

    label = unit[0].label
    encoded: dict[str, object] = {
        'id': annotation_id,
        'image_id': image_id,
        'category_id': category_ids[label],
        'iscrowd': 0,
    }
    if poses:
        pose = poses[0]
        encoded['bbox'] = _xywh(pose)
        encoded['keypoints'] = [
            value for keypoint in pose.keypoints for value in (keypoint.x, keypoint.y, keypoint.visibility)
        ]
        encoded['num_keypoints'] = sum(keypoint.visibility > 0 for keypoint in pose.keypoints)
        encoded['area'] = _finite_derived(
            (
                sum(_polygon_area(polygon) for polygon in polygons)
                if polygons
                else (pose.x2 - pose.x1) * (pose.y2 - pose.y1)
            ),
            'COCO annotation area',
        )
        if polygons:
            encoded['segmentation'] = _segmentation(polygons)
        return encoded

    if polygons:
        encoded['bbox'] = _polygon_bbox(polygons)
        encoded['segmentation'] = _segmentation(polygons)
        encoded['area'] = _finite_derived(sum(_polygon_area(polygon) for polygon in polygons), 'COCO annotation area')
        return encoded

    bbox = bboxes[0]
    encoded['bbox'] = _xywh(bbox)
    encoded['area'] = _finite_derived((bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1), 'COCO annotation area')
    return encoded


def _validate_payload_numbers(value: object) -> None:
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError('COCO payload numbers must be finite')
    elif isinstance(value, list):
        for item in value:
            _validate_payload_numbers(item)
    elif isinstance(value, dict):
        for item in value.values():
            _validate_payload_numbers(item)


def _encode_coco(doc: CocoDoc) -> dict[str, object]:
    category_ids = {label: index for index, label in enumerate(doc.labels, start=1)}
    schemas = _pose_schemas(doc, category_ids)
    categories = []
    for label, category_id in category_ids.items():
        category: dict[str, object] = {'id': category_id, 'name': label}
        if label in schemas:
            category['keypoints'] = list(schemas[label])
        categories.append(category)

    images = [
        {'id': image_id, 'file_name': image.file_name, 'width': image.info.width, 'height': image.info.height}
        for image_id, image in enumerate(doc.images, start=1)
    ]
    annotations = []
    annotation_id = 1
    for image_id, image in enumerate(doc.images, start=1):
        for unit in _annotation_units(image.annotations):
            annotations.append(_encode_unit(unit, annotation_id, image_id, category_ids))
            annotation_id += 1
    payload = {'categories': categories, 'images': images, 'annotations': annotations}
    _validate_payload_numbers(payload)
    return payload


def write_coco(doc: CocoDoc, json_path: str | Path) -> None:
    payload = _encode_coco(doc)
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
