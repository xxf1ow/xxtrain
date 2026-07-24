import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xxtrain.data import LabelCatalog
from xxtrain.pipeline.core import Context, ConversionConfig, ConversionReport
from xxtrain.pipeline.discovery import DirectorySource, validate_classification_source
from xxtrain.task import TaskType


class PipelineDiscoveryTest(unittest.TestCase):
    def make_context(self, root: Path) -> Context:
        return Context(
            config=ConversionConfig(
                task_name='detect',
                task_type=TaskType.DETECT,
                root_path=root,
                split=10,
                labels=LabelCatalog(('a', 'b')),
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
                [('a', 0), ('b', 0), ('b', 1)],
                [(sample.source_group, sample.source_index) for sample in samples],
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
