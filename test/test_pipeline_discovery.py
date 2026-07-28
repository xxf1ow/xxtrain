import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xxtrain.data import LabelCatalog
from xxtrain.pipeline.core import Context, ConversionConfig, ConversionReport
from xxtrain.pipeline.discovery import CocoSource, DirectorySource, validate_classification_source
from xxtrain.task import TaskType


class PipelineDiscoveryTest(unittest.TestCase):
    def make_context(self, root: Path, labels: LabelCatalog = LabelCatalog(('a', 'b'))) -> Context:
        return Context(
            config=ConversionConfig(
                task_name='detect',
                task_type=TaskType.DETECT,
                root_path=root,
                split=10,
                labels=labels,
                reserve_no_label=True,
            ),
            report=ConversionReport(),
        )

    def test_directory_source_orders_groups_and_images_without_reading_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for group, names in {'b': ('2.PNG', '1.jpg'), 'a': ('z.bmp', 'note.txt')}.items():
                images = root / 'src' / group / 'imgs'
                images.mkdir(parents=True)
                for name in names:
                    (images / name).write_bytes(b'not-an-opened-image')
            (root / 'src' / 'skip').mkdir()

            samples = list(DirectorySource().read(self.make_context(root)))

            self.assertEqual(['a/z', 'b/1', 'b/2'], [sample.id for sample in samples])
            self.assertEqual(
                [('a', 0), ('b', 0), ('b', 1)], [(sample.source_group, sample.source_index) for sample in samples]
            )
            self.assertTrue(all(sample.image.info is None for sample in samples))
            self.assertTrue(all(sample.image.path.is_absolute() for sample in samples))

    def test_directory_source_preserves_logical_image_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / 'src' / 'group' / 'imgs' / 'logical.jpg'
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b'image')
            resolved_target = root / 'external' / 'physical.jpg'

            with patch.object(Path, 'resolve', return_value=resolved_target):
                samples = list(DirectorySource().read(self.make_context(root)))

            self.assertEqual(image_path.absolute(), samples[0].image.path)

    def test_classification_validation_rejects_mismatched_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'src' / 'a' / 'imgs').mkdir(parents=True)
            (root / 'src' / 'a' / 'imgs' / '0.jpg').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, 'Classification labels mismatch'):
                validate_classification_source(root, LabelCatalog(('a', 'b')), split=10)

    def test_classification_validation_rejects_unusable_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / 'src' / 'a' / 'imgs'
            images.mkdir(parents=True)
            (images / '0.jpg').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, "class 'a'.*without train or val samples"):
                validate_classification_source(root, LabelCatalog(('a',)), split=10)

    def write_coco(self, root: Path, payload: dict[str, object]) -> Path:
        json_path = root / 'annotations.json'
        json_path.write_text(json.dumps(payload), encoding='utf-8')
        return json_path

    def test_coco_source_reads_samples_with_catalog_and_stable_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / 'dataset'
            json_path = self.write_coco(
                root,
                {
                    'categories': [{'id': 1, 'name': 'dial'}],
                    'images': [
                        {'id': 20, 'file_name': 'images/a.jpg', 'width': 16, 'height': 8},
                        {'id': 10, 'file_name': 'images/b.jpg', 'width': 16, 'height': 8},
                    ],
                    'annotations': [
                        {'id': 1, 'image_id': 20, 'category_id': 1, 'bbox': [1, 2, 3, 4], 'segmentation': []}
                    ],
                },
            )
            source = CocoSource(json_path, image_root=image_root)

            samples = list(source.read(self.make_context(root, LabelCatalog(('dial',)))))

            self.assertEqual(LabelCatalog(('dial',)), source.catalog_for(TaskType.DETECT))
            self.assertEqual(['coco/000000', 'coco/000001'], [sample.id for sample in samples])
            self.assertEqual(
                [('coco', 0), ('coco', 1)], [(sample.source_group, sample.source_index) for sample in samples]
            )
            self.assertEqual(image_root / 'images' / 'a.jpg', samples[0].image.path)
            self.assertEqual((16.0, 8.0), (samples[0].image.info.width, samples[0].image.info.height))

    def test_coco_source_derives_pose_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = self.write_coco(
                root,
                {
                    'categories': [{'id': 1, 'name': 'person', 'keypoints': ['nose', 'wrist']}],
                    'images': [{'id': 1, 'file_name': 'a.jpg', 'width': 16, 'height': 8}],
                    'annotations': [
                        {
                            'id': 1,
                            'image_id': 1,
                            'category_id': 1,
                            'bbox': [1, 2, 3, 4],
                            'keypoints': [2, 3, 2, 3, 4, 2],
                            'segmentation': [],
                        }
                    ],
                },
            )

            self.assertEqual(
                LabelCatalog(('person', 'nose', 'wrist')), CocoSource(json_path).catalog_for(TaskType.POSE)
            )

    def test_coco_pose_without_annotations_requires_explicit_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = self.write_coco(
                root,
                {
                    'categories': [{'id': 1, 'name': 'person', 'keypoints': ['nose']}],
                    'images': [{'id': 1, 'file_name': 'a.jpg', 'width': 16, 'height': 8}],
                    'annotations': [],
                },
            )

            with self.assertRaisesRegex(ValueError, 'Pose catalog'):
                CocoSource(json_path).catalog_for(TaskType.POSE)
            self.assertEqual(
                LabelCatalog(('person', 'nose')),
                CocoSource(json_path, catalog=LabelCatalog(('person', 'nose'))).catalog_for(TaskType.POSE),
            )

    def test_coco_pose_without_annotations_rejects_catalog_without_keypoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = self.write_coco(
                root,
                {
                    'categories': [{'id': 1, 'name': 'person', 'keypoints': ['nose']}],
                    'images': [{'id': 1, 'file_name': 'a.jpg', 'width': 16, 'height': 8}],
                    'annotations': [],
                },
            )

            with self.assertRaisesRegex(ValueError, 'keypoint'):
                CocoSource(json_path, catalog=LabelCatalog(('person',))).catalog_for(TaskType.POSE)

    def test_coco_source_resolves_relative_image_root_to_absolute_image_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            image_root = root / 'dataset'
            json_path = self.write_coco(
                root,
                {
                    'categories': [{'id': 1, 'name': 'dial'}],
                    'images': [{'id': 1, 'file_name': 'images/a.jpg', 'width': 16, 'height': 8}],
                    'annotations': [],
                },
            )
            relative_image_root = image_root.relative_to(Path.cwd())

            sample = next(
                CocoSource(json_path, image_root=relative_image_root).read(
                    self.make_context(root, LabelCatalog(('dial',)))
                )
            )

            self.assertEqual((image_root / 'images' / 'a.jpg').absolute(), sample.image.path)

    def test_coco_source_rejects_inconsistent_explicit_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = self.write_coco(
                root,
                {
                    'categories': [{'id': 1, 'name': 'dial'}],
                    'images': [{'id': 1, 'file_name': 'a.jpg', 'width': 16, 'height': 8}],
                    'annotations': [],
                },
            )

            with self.assertRaisesRegex(ValueError, 'catalog'):
                CocoSource(json_path, catalog=LabelCatalog(('other',))).catalog_for(TaskType.DETECT)

    def test_coco_source_validates_obb_annotations_at_read_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = self.write_coco(
                root,
                {
                    'categories': [{'id': 1, 'name': 'dial'}],
                    'images': [{'id': 1, 'file_name': 'a.jpg', 'width': 16, 'height': 8}],
                    'annotations': [
                        {'id': 1, 'image_id': 1, 'category_id': 1, 'segmentation': [[0, 0, 3, 0, 4, 1, 3, 3, 0, 3]]}
                    ],
                },
            )
            context = Context(
                config=ConversionConfig(
                    task_name='obb',
                    task_type=TaskType.OBB,
                    root_path=root,
                    split=10,
                    labels=LabelCatalog(('dial',)),
                    reserve_no_label=True,
                ),
                report=ConversionReport(),
            )

            with self.assertRaisesRegex(ValueError, 'OBB'):
                list(CocoSource(json_path).read(context))
