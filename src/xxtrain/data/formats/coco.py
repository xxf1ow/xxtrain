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


def _decode_annotation(value: dict[str, object], category: tuple[str, tuple[str, ...]]) -> tuple[Annotation, ...]:
    annotation_id = _id(_required(value, 'id'), 'annotation id')
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

    for annotation in annotations:
        image_id = _id(_required(annotation, 'image_id'), 'annotation image_id')
        category_id = _id(_required(annotation, 'category_id'), 'annotation category_id')
        if image_id not in image_by_id:
            raise ValueError(f'Unknown COCO annotation image id: {image_id}')
        if category_id not in category_by_id:
            raise ValueError(f'Unknown COCO annotation category id: {category_id}')
        image_annotations[image_id].extend(_decode_annotation(annotation, category_by_id[category_id]))

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
