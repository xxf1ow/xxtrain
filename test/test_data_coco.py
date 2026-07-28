import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from xxtrain.data import Bbox, CocoDoc, CocoImage, ImageInfo, LabelCatalog, Polygon, Pose
from xxtrain.data.formats import read_coco


def _category(category_id: int = 17, *, keypoints: object = None) -> dict[str, object]:
    category: dict[str, object] = {'id': category_id, 'name': 'object'}
    if keypoints is not None:
        category['keypoints'] = keypoints
    return category


def _image(image_id: int = 31, *, file_name: str = 'image.jpg') -> dict[str, object]:
    return {'id': image_id, 'file_name': file_name, 'width': 100, 'height': 80}


def _annotation(**changes: object) -> dict[str, object]:
    annotation: dict[str, object] = {'id': 41, 'image_id': 31, 'category_id': 17, 'bbox': [1, 2, 3, 4]}
    annotation.update(changes)
    return annotation


def _document(*, categories: object = None, images: object = None, annotations: object = None) -> dict[str, object]:
    return {
        'categories': [_category()] if categories is None else categories,
        'images': [_image()] if images is None else images,
        'annotations': [_annotation()] if annotations is None else annotations,
    }


class CocoStructuresTest(unittest.TestCase):
    def test_structures_are_immutable_and_keep_exact_tuples(self) -> None:
        annotation = Bbox(label='object', x1=1, y1=2, x2=4, y2=6)
        image = CocoImage(file_name='nested/image.jpg', info=ImageInfo(width=100, height=80), annotations=(annotation,))
        document = CocoDoc(labels=LabelCatalog(('object',)), images=(image,))

        self.assertEqual('nested/image.jpg', image.file_name)
        self.assertEqual((annotation,), image.annotations)
        self.assertEqual((image,), document.images)
        with self.assertRaises(FrozenInstanceError):
            image.file_name = 'changed.jpg'  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            document.images = ()  # type: ignore[misc]

    def test_coco_image_rejects_invalid_fields(self) -> None:
        info = ImageInfo(width=10, height=20)
        annotation = Bbox(label='object', x1=1, y1=2, x2=3, y2=4)
        invalid = [
            {'file_name': '', 'info': info, 'annotations': ()},
            {'file_name': 1, 'info': info, 'annotations': ()},
            {'file_name': 'a.jpg', 'info': object(), 'annotations': ()},
            {'file_name': 'a.jpg', 'info': info, 'annotations': [annotation]},
            {'file_name': 'a.jpg', 'info': info, 'annotations': (object(),)},
        ]

        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises((TypeError, ValueError)):
                CocoImage(**fields)  # type: ignore[arg-type]

    def test_coco_doc_rejects_invalid_fields_and_duplicate_file_names(self) -> None:
        image = CocoImage(file_name='a.jpg', info=ImageInfo(width=10, height=20), annotations=())
        duplicate = CocoImage(file_name='a.jpg', info=ImageInfo(width=30, height=40), annotations=())
        invalid = [
            {'labels': object(), 'images': ()},
            {'labels': LabelCatalog(('object',)), 'images': [image]},
            {'labels': LabelCatalog(('object',)), 'images': (object(),)},
            {'labels': LabelCatalog(('object',)), 'images': (image, duplicate)},
        ]

        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises((TypeError, ValueError)):
                CocoDoc(**fields)  # type: ignore[arg-type]


class CocoReaderTest(unittest.TestCase):
    def _read(self, payload: object) -> CocoDoc:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'annotations.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            return read_coco(path)

    def test_reads_all_shapes_in_source_order_and_preserves_sparse_id_order(self) -> None:
        payload = {
            'categories': [{'id': 42, 'name': 'person', 'keypoints': ['nose', 'tail']}, {'id': 7, 'name': 'parcel'}],
            'images': [
                {'id': 30, 'file_name': 'nested/z.jpg', 'width': 640, 'height': 480},
                {'id': 10, 'file_name': 'a.jpg', 'width': 320, 'height': 200},
            ],
            'annotations': [
                {'id': 90, 'image_id': 10, 'category_id': 7, 'bbox': [5, 6, 7, 8]},
                {
                    'id': 91,
                    'image_id': 30,
                    'category_id': 42,
                    'bbox': [10, 20, 30, 40],
                    'keypoints': [11, 22, 2, 33, 44, 1],
                    'segmentation': [[0, 0, 20, 0, 20, 20]],
                },
                {
                    'id': 92,
                    'image_id': 30,
                    'category_id': 7,
                    'bbox': [1, 2, 30, 40],
                    'segmentation': [[1, 2, 11, 2, 11, 12], [20, 21, 30, 21, 30, 31]],
                },
                {'id': 93, 'image_id': 30, 'category_id': 42, 'bbox': [2, 3, 4, 5], 'keypoints': [3, 4, 0, 5, 6, 2]},
                {
                    'id': 94,
                    'image_id': 30,
                    'category_id': 7,
                    'bbox': [3, 4, 5, 6],
                    'segmentation': [[2, 3, 4, 5, 6, 7]],
                },
                {'id': 95, 'image_id': 30, 'category_id': 7, 'bbox': [4, 5, 6, 7], 'segmentation': []},
            ],
        }

        document = self._read(payload)

        self.assertEqual(('person', 'parcel'), document.labels.names)
        self.assertEqual(('nested/z.jpg', 'a.jpg'), tuple(image.file_name for image in document.images))
        self.assertEqual((640.0, 480.0), (document.images[0].info.width, document.images[0].info.height))
        annotations = document.images[0].annotations
        self.assertEqual(
            (Pose, Polygon, Polygon, Polygon, Pose, Polygon, Bbox), tuple(type(item) for item in annotations)
        )
        pose = annotations[0]
        self.assertEqual(('nose', 'tail'), tuple(keypoint.label for keypoint in pose.keypoints))
        self.assertEqual(
            ((11.0, 22.0, 2), (33.0, 44.0, 1)),
            tuple((keypoint.x, keypoint.y, keypoint.visibility) for keypoint in pose.keypoints),
        )
        self.assertEqual((10.0, 20.0, 40.0, 60.0), pose.bbox)
        self.assertEqual(((0.0, 0.0), (20.0, 0.0), (20.0, 20.0)), annotations[1].points)
        self.assertEqual((91, 91, 92, 92), tuple(item.group for item in annotations[:4]))
        self.assertIsNone(annotations[4].group)
        self.assertIsNone(annotations[5].group)
        self.assertIsNone(annotations[6].group)
        self.assertEqual((4.0, 5.0, 10.0, 12.0), annotations[6].bbox)
        self.assertEqual((Bbox,), tuple(type(item) for item in document.images[1].annotations))
        self.assertEqual((5.0, 6.0, 12.0, 14.0), document.images[1].annotations[0].bbox)

    def test_rejects_duplicate_ids_unknown_references_and_file_names(self) -> None:
        cases = {
            'duplicate category id': _document(categories=[_category(17), {'id': 17, 'name': 'other'}]),
            'duplicate image id': _document(images=[_image(31), _image(31, file_name='other.jpg')]),
            'unknown category id': _document(annotations=[_annotation(category_id=999)]),
            'unknown image id': _document(annotations=[_annotation(image_id=999)]),
            'duplicate file name': _document(images=[_image(31), _image(32)]),
        }

        for name, payload in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self._read(payload)

    def test_rejects_invalid_bbox_values_and_geometry(self) -> None:
        invalid_bboxes = (
            [1, 2, 3],
            [1, 2, '3', 4],
            [1, 2, float('nan'), 4],
            [1, 2, 0, 4],
            [1, 2, 3, -1],
            [True, 2, 3, 4],
        )

        for bbox in invalid_bboxes:
            with self.subTest(bbox=bbox), self.assertRaises((TypeError, ValueError)):
                self._read(_document(annotations=[_annotation(bbox=bbox)]))

    def test_rejects_invalid_category_keypoint_names(self) -> None:
        invalid_names = ('nose', ['', 'tail'], ['nose', 'nose'], [1, 'tail'])

        for names in invalid_names:
            with self.subTest(names=names), self.assertRaises((TypeError, ValueError)):
                self._read(_document(categories=[_category(keypoints=names)], annotations=[]))

    def test_rejects_invalid_annotation_keypoints_and_requires_bbox(self) -> None:
        category = _category(keypoints=['nose', 'tail'])
        invalid_keypoints = (
            [1, 2, 2],
            [1, 2, 2, 3, float('inf'), 2],
            [1, 2, 3, 3, 4, 2],
            [1, 2, 1.0, 3, 4, 2],
            [True, 2, 1, 3, 4, 2],
        )

        for keypoints in invalid_keypoints:
            with self.subTest(keypoints=keypoints), self.assertRaises((TypeError, ValueError)):
                self._read(_document(categories=[category], annotations=[_annotation(keypoints=keypoints)]))

        annotation = _annotation(keypoints=[1, 2, 2, 3, 4, 1])
        del annotation['bbox']
        with self.assertRaises(ValueError):
            self._read(_document(categories=[category], annotations=[annotation]))

    def test_rejects_rle_and_invalid_polygon_contours(self) -> None:
        invalid_segmentations = (
            {'counts': [1, 2], 'size': [80, 100]},
            [1, 2, 3, 4, 5, 6],
            [[1, 2, 3, 4]],
            [[1, 2, 3, 4, 5, 6, 7]],
            [[1, 2, 3, 4, '5', 6]],
            [[1, 2, 3, 4, float('-inf'), 6]],
        )

        for segmentation in invalid_segmentations:
            with self.subTest(segmentation=segmentation), self.assertRaises((TypeError, ValueError)):
                self._read(_document(annotations=[_annotation(segmentation=segmentation)]))

    def test_rejects_malformed_root_and_collections_without_assertions(self) -> None:
        type_errors = (
            [],
            _document(categories={}),
            _document(images={}),
            _document(annotations={}),
            _document(categories=[1]),
            _document(images=[1]),
            _document(annotations=[1]),
        )
        for payload in type_errors:
            with self.subTest(payload=payload), self.assertRaises(TypeError):
                self._read(payload)

        for missing in ('categories', 'images', 'annotations'):
            payload = _document()
            del payload[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self._read(payload)


if __name__ == '__main__':
    unittest.main()
