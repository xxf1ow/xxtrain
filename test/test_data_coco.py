import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from xxtrain.data import (
    Annotation,
    Bbox,
    Circle,
    CocoDoc,
    CocoImage,
    ImageInfo,
    Keypoint,
    LabelCatalog,
    Points,
    Polygon,
    Polyline,
    Pose,
)
from xxtrain.data.formats import read_coco
from xxtrain.data.formats.coco import write_coco


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

    def test_rejects_duplicate_annotation_ids_across_images(self) -> None:
        payload = _document(
            images=[_image(31), _image(32, file_name='other.jpg')],
            annotations=[_annotation(id=55, image_id=31), _annotation(id=55, image_id=32)],
        )

        with self.assertRaisesRegex(ValueError, 'Duplicate COCO annotation id: 55'):
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


class _UnknownAnnotation(Annotation):
    def _validate_geometry(self) -> None:
        pass


class CocoWriterTest(unittest.TestCase):
    def _round_trip(self, document: CocoDoc) -> tuple[CocoDoc, dict[str, object]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'nested' / 'annotations.json'
            write_coco(document, path)
            payload = json.loads(path.read_text(encoding='utf-8'))
            return read_coco(path), payload

    def test_round_trip_preserves_independent_bbox_polygon_and_pose(self) -> None:
        annotations = (
            Bbox(label='parcel', x1=1, y1=2, x2=11, y2=22),
            Polygon(label='parcel', points=((30, 5), (40, 5), (38, 15))),
            Pose(
                label='person',
                x1=50,
                y1=10,
                x2=90,
                y2=70,
                keypoints=(
                    Keypoint(label='nose', x=60, y=20, visibility=2),
                    Keypoint(label='tail', x=80, y=60, visibility=0),
                ),
            ),
        )
        document = CocoDoc(
            labels=LabelCatalog(('person', 'parcel')),
            images=(CocoImage(file_name='image.jpg', info=ImageInfo(width=100, height=80), annotations=annotations),),
        )

        loaded, payload = self._round_trip(document)

        self.assertEqual(('person', 'parcel'), loaded.labels.names)
        self.assertEqual((Bbox, Polygon, Pose), tuple(type(item) for item in loaded.images[0].annotations))
        self.assertEqual((1.0, 2.0, 11.0, 22.0), loaded.images[0].annotations[0].bbox)
        self.assertEqual(((30.0, 5.0), (40.0, 5.0), (38.0, 15.0)), loaded.images[0].annotations[1].points)
        pose = loaded.images[0].annotations[2]
        self.assertEqual((50.0, 10.0, 90.0, 70.0), pose.bbox)
        self.assertEqual(
            (('nose', 60.0, 20.0, 2), ('tail', 80.0, 60.0, 0)),
            tuple((keypoint.label, keypoint.x, keypoint.y, keypoint.visibility) for keypoint in pose.keypoints),
        )
        self.assertEqual(1, payload['annotations'][2]['num_keypoints'])
        self.assertEqual(0, payload['annotations'][2]['iscrowd'])

    def test_grouped_pose_and_polygon_become_one_annotation_with_pose_bbox(self) -> None:
        annotations = (
            Polygon(label='person', group='subject', points=((0, 0), (20, 0), (20, 10))),
            Pose(
                label='person',
                group='subject',
                x1=5,
                y1=6,
                x2=35,
                y2=46,
                keypoints=(Keypoint(label='nose', x=10, y=12, visibility=2),),
            ),
        )
        document = CocoDoc(
            labels=LabelCatalog(('person',)),
            images=(CocoImage(file_name='image.jpg', info=ImageInfo(width=50, height=50), annotations=annotations),),
        )

        loaded, payload = self._round_trip(document)

        self.assertEqual(1, len(payload['annotations']))
        self.assertEqual([5.0, 6.0, 30.0, 40.0], payload['annotations'][0]['bbox'])
        self.assertEqual(100.0, payload['annotations'][0]['area'])
        self.assertEqual((Pose, Polygon), tuple(type(item) for item in loaded.images[0].annotations))
        self.assertEqual((5.0, 6.0, 35.0, 46.0), loaded.images[0].annotations[0].bbox)
        self.assertEqual((1, 1), tuple(item.group for item in loaded.images[0].annotations))

    def test_grouped_polygons_become_ordered_multi_contour_segmentation(self) -> None:
        annotations = (
            Polygon(label='parcel', group=9, points=((10, 10), (20, 10), (20, 20))),
            Polygon(label='parcel', group=9, points=((30, 5), (40, 5), (40, 25), (30, 25))),
        )
        document = CocoDoc(
            labels=LabelCatalog(('parcel',)),
            images=(CocoImage(file_name='image.jpg', info=ImageInfo(width=50, height=30), annotations=annotations),),
        )

        loaded, payload = self._round_trip(document)

        encoded = payload['annotations'][0]
        self.assertEqual(
            [[10.0, 10.0, 20.0, 10.0, 20.0, 20.0], [30.0, 5.0, 40.0, 5.0, 40.0, 25.0, 30.0, 25.0]],
            encoded['segmentation'],
        )
        self.assertEqual([10.0, 5.0, 30.0, 20.0], encoded['bbox'])
        self.assertEqual(250.0, encoded['area'])
        self.assertEqual((Polygon, Polygon), tuple(type(item) for item in loaded.images[0].annotations))
        self.assertEqual((1, 1), tuple(item.group for item in loaded.images[0].annotations))

    def test_large_translated_polygon_keeps_local_area_and_round_trip_geometry(self) -> None:
        points = ((-1e16, 1e16), (-1e16 + 1024, 1e16), (-1e16 + 1024, 1e16 + 4), (-1e16, 1e16 + 4))
        document = CocoDoc(
            labels=LabelCatalog(('object',)),
            images=(
                CocoImage(
                    file_name='image.jpg',
                    info=ImageInfo(width=100, height=80),
                    annotations=(Polygon(label='object', points=points),),
                ),
            ),
        )

        loaded, payload = self._round_trip(document)

        self.assertEqual(4096.0, payload['annotations'][0]['area'])
        self.assertEqual([-1e16, 1e16, 1024.0, 4.0], payload['annotations'][0]['bbox'])
        self.assertEqual((Polygon,), tuple(type(item) for item in loaded.images[0].annotations))
        self.assertEqual(points, loaded.images[0].annotations[0].points)
        self.assertEqual((-1e16, 1e16, -1e16 + 1024, 1e16 + 4), loaded.images[0].annotations[0].bbox)

    def test_writes_stable_one_based_ids_in_document_and_first_source_order(self) -> None:
        first_image = CocoImage(
            file_name='z.jpg',
            info=ImageInfo(width=100, height=80),
            annotations=(
                Bbox(label='second', x1=1, y1=2, x2=4, y2=6),
                Polygon(label='first', group='joined', points=((0, 0), (4, 0), (4, 4))),
                Polygon(label='first', group='joined', points=((10, 10), (12, 10), (12, 12))),
                Bbox(label='first', x1=20, y1=20, x2=30, y2=30),
            ),
        )
        second_image = CocoImage(
            file_name='a.jpg',
            info=ImageInfo(width=40, height=30),
            annotations=(Bbox(label='second', x1=2, y1=3, x2=8, y2=9),),
        )
        document = CocoDoc(labels=LabelCatalog(('first', 'second')), images=(first_image, second_image))

        _, payload = self._round_trip(document)

        self.assertEqual([{'id': 1, 'name': 'first'}, {'id': 2, 'name': 'second'}], payload['categories'])
        self.assertEqual([1, 2], [image['id'] for image in payload['images']])
        self.assertEqual(['z.jpg', 'a.jpg'], [image['file_name'] for image in payload['images']])
        self.assertEqual([1, 2, 3, 4], [annotation['id'] for annotation in payload['annotations']])
        self.assertEqual([1, 1, 1, 2], [annotation['image_id'] for annotation in payload['annotations']])
        self.assertEqual([2, 1, 1, 2], [annotation['category_id'] for annotation in payload['annotations']])

    def test_rejects_unsupported_annotations_without_creating_or_overwriting_target(self) -> None:
        unsupported = (
            Circle(label='object', center=(5, 5), edge=(6, 5)),
            Polyline(label='object', points=((1, 1), (2, 2))),
            Points(label='object', points=((1, 1),)),
            _UnknownAnnotation(label='object'),
        )

        for index, annotation in enumerate(unsupported):
            with self.subTest(annotation=type(annotation).__name__), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                path = root / f'missing-{index}' / 'annotations.json'
                document = CocoDoc(
                    labels=LabelCatalog(('object',)),
                    images=(
                        CocoImage(
                            file_name='image.jpg', info=ImageInfo(width=10, height=10), annotations=(annotation,)
                        ),
                    ),
                )
                with self.assertRaises(TypeError):
                    write_coco(document, path)
                self.assertFalse(path.parent.exists())

                existing = root / f'existing-{index}.json'
                existing.write_text('keep me', encoding='utf-8')
                with self.assertRaises(TypeError):
                    write_coco(document, existing)
                self.assertEqual('keep me', existing.read_text(encoding='utf-8'))

    def test_rejects_unknown_labels_group_ambiguity_and_inconsistent_pose_schemas(self) -> None:
        cases = {
            'unknown label': (LabelCatalog(('object',)), (Bbox(label='missing', x1=1, y1=1, x2=2, y2=2),)),
            'mixed labels': (
                LabelCatalog(('person', 'object')),
                (
                    Polygon(label='person', group=1, points=((0, 0), (2, 0), (2, 2))),
                    Polygon(label='object', group=1, points=((3, 3), (5, 3), (5, 5))),
                ),
            ),
            'multiple poses': (
                LabelCatalog(('person',)),
                (
                    Pose(
                        label='person', group=1, x1=0, y1=0, x2=10, y2=10, keypoints=(Keypoint(label='nose', x=1, y=1),)
                    ),
                    Pose(
                        label='person',
                        group=1,
                        x1=10,
                        y1=10,
                        x2=20,
                        y2=20,
                        keypoints=(Keypoint(label='nose', x=11, y=11),),
                    ),
                ),
            ),
            'bbox mixed into group': (
                LabelCatalog(('object',)),
                (
                    Bbox(label='object', group=1, x1=0, y1=0, x2=10, y2=10),
                    Polygon(label='object', group=1, points=((0, 0), (2, 0), (2, 2))),
                ),
            ),
            'inconsistent schemas': (
                LabelCatalog(('person',)),
                (
                    Pose(
                        label='person',
                        x1=0,
                        y1=0,
                        x2=10,
                        y2=10,
                        keypoints=(Keypoint(label='nose', x=1, y=1), Keypoint(label='tail', x=2, y=2)),
                    ),
                    Pose(
                        label='person',
                        x1=10,
                        y1=10,
                        x2=20,
                        y2=20,
                        keypoints=(Keypoint(label='tail', x=12, y=12), Keypoint(label='nose', x=11, y=11)),
                    ),
                ),
            ),
        }

        for name, (labels, annotations) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / 'missing' / 'annotations.json'
                document = CocoDoc(
                    labels=labels,
                    images=(
                        CocoImage(file_name='image.jpg', info=ImageInfo(width=30, height=30), annotations=annotations),
                    ),
                )
                with self.assertRaises(ValueError):
                    write_coco(document, path)
                self.assertFalse(path.parent.exists())

    def test_rejects_degenerate_polygons_without_creating_or_overwriting_target(self) -> None:
        bowtie = Polygon(label='object', points=((0, 0), (2, 2), (0, 2), (2, 0)))
        vertical = Polygon(label='object', group='shape', points=((1, 0), (1, 1), (1, 2)))
        documents = (
            CocoDoc(
                labels=LabelCatalog(('object',)),
                images=(CocoImage(file_name='image.jpg', info=ImageInfo(width=20, height=20), annotations=(bowtie,)),),
            ),
            CocoDoc(
                labels=LabelCatalog(('object',)),
                images=(
                    CocoImage(
                        file_name='image.jpg',
                        info=ImageInfo(width=20, height=20),
                        annotations=(Polygon(label='object', group='shape', points=((0, 0), (4, 0), (4, 4))), vertical),
                    ),
                ),
            ),
            CocoDoc(
                labels=LabelCatalog(('object',)),
                images=(
                    CocoImage(
                        file_name='image.jpg',
                        info=ImageInfo(width=20, height=20),
                        annotations=(
                            Pose(
                                label='object',
                                group='pose',
                                x1=0,
                                y1=0,
                                x2=10,
                                y2=10,
                                keypoints=(Keypoint(label='center', x=5, y=5),),
                            ),
                            bowtie.wrap(group='pose'),
                        ),
                    ),
                ),
            ),
        )

        for index, document in enumerate(documents):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                missing = root / f'missing-{index}' / 'annotations.json'
                with self.assertRaises(ValueError):
                    write_coco(document, missing)
                self.assertFalse(missing.parent.exists())

                existing = root / f'existing-{index}.json'
                existing.write_text('keep me', encoding='utf-8')
                with self.assertRaises(ValueError):
                    write_coco(document, existing)
                self.assertEqual('keep me', existing.read_text(encoding='utf-8'))

    def test_rejects_extreme_finite_geometry_that_overflows_derived_values_before_io(self) -> None:
        annotations = (
            Bbox(label='object', x1=-1e308, y1=0, x2=1e308, y2=1),
            Bbox(label='object', x1=0, y1=0, x2=1e200, y2=1e200),
            Polygon(label='object', points=((0, 0), (1e200, 0), (0, 1e200))),
            Polygon(label='object', points=((-1e308, 0), (1e308, 0), (0, 1))),
        )

        for index, annotation in enumerate(annotations):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                document = CocoDoc(
                    labels=LabelCatalog(('object',)),
                    images=(
                        CocoImage(
                            file_name='image.jpg', info=ImageInfo(width=20, height=20), annotations=(annotation,)
                        ),
                    ),
                )
                missing = root / f'missing-{index}' / 'annotations.json'
                with self.assertRaises(ValueError):
                    write_coco(document, missing)
                self.assertFalse(missing.parent.exists())

                existing = root / f'existing-{index}.json'
                existing.write_text('keep me', encoding='utf-8')
                with self.assertRaises(ValueError):
                    write_coco(document, existing)
                self.assertEqual('keep me', existing.read_text(encoding='utf-8'))

    def test_type_distinct_equal_group_values_remain_separate(self) -> None:
        document = CocoDoc(
            labels=LabelCatalog(('first', 'second')),
            images=(
                CocoImage(
                    file_name='image.jpg',
                    info=ImageInfo(width=20, height=20),
                    annotations=(
                        Polygon(label='first', group=True, points=((0, 0), (4, 0), (4, 4))),
                        Polygon(label='second', group=1, points=((10, 10), (14, 10), (14, 14))),
                    ),
                ),
            ),
        )

        loaded, payload = self._round_trip(document)

        self.assertEqual([1, 2], [annotation['id'] for annotation in payload['annotations']])
        self.assertEqual([1, 2], [annotation['category_id'] for annotation in payload['annotations']])
        self.assertEqual(('first', 'second'), tuple(item.label for item in loaded.images[0].annotations))
        self.assertTrue(all(item.group is None for item in loaded.images[0].annotations))

    def test_same_group_value_in_different_images_is_not_merged(self) -> None:
        images = (
            CocoImage(
                file_name='first.jpg',
                info=ImageInfo(width=10, height=10),
                annotations=(Polygon(label='object', group=7, points=((0, 0), (4, 0), (4, 4))),),
            ),
            CocoImage(
                file_name='second.jpg',
                info=ImageInfo(width=10, height=10),
                annotations=(Polygon(label='object', group=7, points=((5, 5), (9, 5), (9, 9))),),
            ),
        )
        document = CocoDoc(labels=LabelCatalog(('object',)), images=images)

        loaded, payload = self._round_trip(document)

        self.assertEqual(2, len(payload['annotations']))
        self.assertEqual([1, 2], [annotation['image_id'] for annotation in payload['annotations']])
        self.assertIsNone(loaded.images[0].annotations[0].group)
        self.assertIsNone(loaded.images[1].annotations[0].group)


if __name__ == '__main__':
    unittest.main()
