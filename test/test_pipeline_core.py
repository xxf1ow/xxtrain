import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from xxtrain.data import Bbox, ImageInfo, LabelCatalog
from xxtrain.pipeline.core import (
    ClassifyOutput,
    ConversionConfig,
    ConversionReport,
    CropOutput,
    EncodeOutput,
    ImageRef,
    Sample,
)
from xxtrain.task import TaskType


class PipelineValueObjectsTest(unittest.TestCase):
    def make_sample(self, root: Path) -> Sample:
        return Sample(
            id='group/image',
            source_group='group',
            source_index=3,
            image=ImageRef(path=root / 'image.png', info=ImageInfo(width=20, height=10)),
        )

    def test_sample_wrap_returns_a_new_immutable_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample = self.make_sample(Path(temp_dir))
            updated = sample.wrap(annotations=(Bbox(label='dial', x1=1, y1=2, x2=8, y2=9),))
            self.assertEqual((), sample.annotations)
            self.assertEqual(1, len(updated.annotations))
            with self.assertRaises(FrozenInstanceError):
                sample.id = 'changed'

    def test_image_ref_requires_explicitly_loaded_info(self) -> None:
        image = ImageRef(path=Path('image.jpg'))
        with self.assertRaisesRegex(ValueError, 'Image metadata is not loaded'):
            image.require_info()

    def test_stage_outputs_expose_stable_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample = self.make_sample(Path(temp_dir))
            parent = Bbox(label='parent', x1=1, y1=1, x2=10, y2=9)
            crop = CropOutput(sample=sample, parent=parent)
            encoded = EncodeOutput(sample=sample, lines=('0 0.5 0.5 0.2 0.2',))
            classified = ClassifyOutput(sample=sample, class_name='dial', output_name='image.jpg')
            self.assertEqual(parent, crop.parent)
            self.assertEqual(1, encoded.annotation_count)
            self.assertEqual('image.jpg', classified.output_name)

    def test_report_records_train_and_val_independently(self) -> None:
        report = ConversionReport()
        report.record_output(train_item='train.jpg', val_item=None, annotation_count=2)
        report.record_output(train_item=None, val_item='val.jpg', annotation_count=3)
        report.record_missing_annotations('group')
        self.assertEqual(['train.jpg'], report.train_items)
        self.assertEqual(['val.jpg'], report.val_items)
        self.assertEqual((1, 1), (report.train_image_count, report.val_image_count))
        self.assertEqual((2, 3), (report.train_annotation_count, report.val_annotation_count))
        self.assertEqual({'group': 1}, report.missing_annotation_counts)

    def test_config_is_immutable(self) -> None:
        config = ConversionConfig(
            task_name='detect',
            task_type=TaskType.DETECT,
            root_path=Path('dataset'),
            split=10,
            labels=LabelCatalog(('dial',)),
            reserve_no_label=True,
        )
        with self.assertRaises(FrozenInstanceError):
            config.split = 2


if __name__ == '__main__':
    unittest.main()
