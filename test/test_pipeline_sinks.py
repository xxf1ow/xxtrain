import io
import shutil
import tempfile
import unittest
from contextlib import chdir, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from xxtrain.data import ImageInfo, LabelCatalog
from xxtrain.pipeline.core import (
    ClassifyOutput,
    Context,
    ConversionConfig,
    ConversionReport,
    EncodeOutput,
    ImageRef,
    Sample,
)
from xxtrain.pipeline.sinks import ClassificationDatasetSink, YoloDatasetSink, _symlink_or_copy, print_conversion_report
from xxtrain.task import TaskType


class PipelineSinkTest(unittest.TestCase):
    def make_context(
        self, root: Path, *, task_name='detect', task_type=TaskType.DETECT, reserve_no_label=True
    ) -> Context:
        return Context(
            config=ConversionConfig(
                task_name=task_name,
                task_type=task_type,
                root_path=root,
                split=10,
                labels=LabelCatalog(('label',)),
                reserve_no_label=reserve_no_label,
            ),
            report=ConversionReport(),
        )

    def make_image(self, root: Path, *, size=(10, 10), name='image.png') -> Path:
        path = root / 'src' / 'group' / 'imgs' / name
        path.parent.mkdir(parents=True)
        with Image.new('RGBA', size, color=(10, 20, 30, 255)) as image:
            image.save(path)
        return path

    def make_sample(
        self, image_path: Path, *, sample_id='group/image', source_index=1, crop_box=None, info=None
    ) -> Sample:
        return Sample(
            id=sample_id,
            source_group='group',
            source_index=source_index,
            image=ImageRef(path=image_path, crop_box=crop_box, info=info),
        )

    def test_yolo_sink_links_whole_image_and_writes_text_and_lists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            context = self.make_context(root)
            sink = YoloDatasetSink()
            sink.write(EncodeOutput(sample=self.make_sample(source), lines=('0 0.5 0.5 0.2 0.2',)), context)
            sink.finalize(context)
            target = root / 'detect' / 'group' / 'image.png'
            self.assertTrue(target.is_file())
            self.assertEqual('0 0.5 0.5 0.2 0.2', target.with_suffix('.txt').read_text(encoding='utf-8'))
            self.assertEqual(str(target.absolute()), (root / 'detect' / 'train.txt').read_text(encoding='utf-8'))
            self.assertEqual('', (root / 'detect' / 'val.txt').read_text(encoding='utf-8'))
            self.assertTrue((root / 'detect' / 'dataset.yaml').is_file())
            self.assertEqual((1, 1), (context.report.train_image_count, context.report.train_annotation_count))

    def test_yolo_sink_preserves_relative_output_paths_in_split_lists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, chdir(temp_dir):
            root = Path('dataset')
            source = self.make_image(root)
            context = self.make_context(root)
            sink = YoloDatasetSink()

            sink.write(EncodeOutput(sample=self.make_sample(source), lines=('0 0.5 0.5 0.2 0.2',)), context)
            sink.finalize(context)

            expected = str(root / 'detect' / 'group' / 'image.png')
            self.assertEqual(expected, (root / 'detect' / 'train.txt').read_text(encoding='utf-8'))

    def test_yolo_sink_materializes_deferred_crop_as_rgb_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            source_bytes = source.read_bytes()
            context = self.make_context(root)
            sample = self.make_sample(
                source,
                sample_id='group/image_0',
                source_index=0,
                crop_box=(1, 2, 7, 8),
                info=ImageInfo(width=6, height=6),
            )
            YoloDatasetSink().write(EncodeOutput(sample=sample, lines=('0 0.5 0.5 1 1',)), context)
            target = root / 'detect' / 'group' / 'image_0.jpg'
            with Image.open(target) as image:
                self.assertEqual((6, 6), image.size)
                self.assertEqual('RGB', image.mode)
            self.assertEqual(source_bytes, source.read_bytes())

    def test_yolo_sink_preserves_dotted_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root, name='image.v1.png')
            context = self.make_context(root)
            sample = self.make_sample(source, sample_id='group/image.v1')

            YoloDatasetSink().write(EncodeOutput(sample=sample, lines=('0 0.5 0.5 1 1',)), context)

            self.assertTrue((root / 'detect' / 'group' / 'image.v1.png').is_file())
            self.assertEqual('0 0.5 0.5 1 1', (root / 'detect' / 'group' / 'image.v1.txt').read_text(encoding='utf-8'))

    def test_zero_annotation_files_are_written_but_not_listed_when_not_reserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            context = self.make_context(root, reserve_no_label=False)
            YoloDatasetSink().write(EncodeOutput(sample=self.make_sample(source), lines=()), context)
            self.assertTrue((root / 'detect' / 'group' / 'image.png').exists())
            self.assertEqual('', (root / 'detect' / 'group' / 'image.txt').read_text(encoding='utf-8'))
            self.assertEqual({'group': 1}, context.report.missing_annotation_counts)
            self.assertEqual(([], []), (context.report.train_items, context.report.val_items))

    def test_symlink_failure_uses_copy2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            context = self.make_context(root)
            with patch('xxtrain.pipeline.sinks.os.symlink', side_effect=OSError('unavailable')):
                YoloDatasetSink().write(
                    EncodeOutput(sample=self.make_sample(source), lines=('0 0.5 0.5 1 1',)), context
                )
            target = root / 'detect' / 'group' / 'image.png'
            self.assertTrue(target.is_file())
            self.assertFalse(target.is_symlink())

    def test_existing_same_source_symlink_is_already_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            target = root / 'classify' / 'train' / 'label' / 'image.png'
            with (
                patch('xxtrain.pipeline.sinks.os.symlink', side_effect=FileExistsError),
                patch.object(Path, 'is_symlink', return_value=True),
                patch.object(Path, 'samefile', return_value=True),
                patch(
                    'xxtrain.pipeline.sinks.shutil.copy2', side_effect=shutil.SameFileError(source, target, 'same file')
                ) as copy2,
            ):
                _symlink_or_copy(source, target)

            copy2.assert_not_called()
            self.assertTrue(source.is_file())

    def test_existing_different_or_broken_symlink_is_replaced_without_following_it(self) -> None:
        for samefile_result in (False, FileNotFoundError()):
            with self.subTest(samefile_result=samefile_result):
                source = Path('source.png')
                target = Path('target.png')
                samefile = (
                    patch.object(Path, 'samefile', side_effect=samefile_result)
                    if isinstance(samefile_result, OSError)
                    else patch.object(Path, 'samefile', return_value=samefile_result)
                )
                with (
                    patch('xxtrain.pipeline.sinks.os.symlink', side_effect=FileExistsError),
                    patch.object(Path, 'is_symlink', return_value=True),
                    samefile,
                    patch.object(Path, 'unlink') as unlink,
                    patch('xxtrain.pipeline.sinks.shutil.copy2') as copy2,
                ):
                    _symlink_or_copy(source, target)

                unlink.assert_called_once_with()
                copy2.assert_called_once_with(source, target)

    def test_classification_sink_can_repeat_an_interrupted_write_when_symlinks_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            context = self.make_context(root, task_name='classify', task_type=TaskType.CLASSIFY)
            output = ClassifyOutput(sample=self.make_sample(source), class_name='label', output_name='image.png')

            with patch('xxtrain.pipeline.sinks.os.symlink', side_effect=OSError('unavailable')):
                ClassificationDatasetSink().write(output, context)
                restarted_context = self.make_context(root, task_name='classify', task_type=TaskType.CLASSIFY)
                ClassificationDatasetSink().write(output, restarted_context)

            target = root / 'classify' / 'train' / 'label' / 'image.png'
            self.assertEqual(source.read_bytes(), target.read_bytes())
            self.assertFalse(target.is_symlink())
            self.assertEqual([str(target.absolute())], restarted_context.report.train_items)

    def test_standard_classification_preserves_original_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root, name='source.png')
            context = self.make_context(root, task_name='classify', task_type=TaskType.CLASSIFY)
            output = ClassifyOutput(sample=self.make_sample(source), class_name='label', output_name='source.png')
            ClassificationDatasetSink().write(output, context)
            target = root / 'classify' / 'train' / 'label' / 'source.png'
            self.assertTrue(target.is_file())
            self.assertFalse((root / 'classify' / 'val' / 'label' / 'source.png').exists())
            self.assertEqual([str(target.absolute())], context.report.train_items)

    def test_classification_sink_preserves_relative_output_paths_in_split_lists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, chdir(temp_dir):
            root = Path('dataset')
            source = self.make_image(root)
            context = self.make_context(root, task_name='classify', task_type=TaskType.CLASSIFY)
            output = ClassifyOutput(sample=self.make_sample(source), class_name='label', output_name='image.png')

            ClassificationDatasetSink().write(output, context)
            ClassificationDatasetSink().finalize(context)

            expected = str(root / 'classify' / 'train' / 'label' / 'image.png')
            self.assertEqual(expected, (root / 'classify' / 'train.txt').read_text(encoding='utf-8'))

    def test_point_classification_uses_integer_crop_and_square_padding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            context = self.make_context(root, task_name='point-classify', task_type=TaskType.CLASSIFY)
            sample = self.make_sample(
                source,
                sample_id='group/image_0',
                source_index=0,
                crop_box=(1.9, 1.9, 7.9, 5.9),
                info=ImageInfo(width=6, height=4),
            )
            output = ClassifyOutput(sample=sample, class_name='label', output_name='group_0_0.jpg')
            ClassificationDatasetSink(indexed_class_directories=True).write(output, context)
            target = root / 'point-classify' / 'val' / '00-label' / 'group_0_0.jpg'
            with Image.open(target) as image:
                self.assertEqual((6, 6), image.size)
                self.assertLess(sum(image.getpixel((3, 0))), 30)
                self.assertGreater(sum(image.getpixel((3, 2))), 30)
                self.assertLess(sum(image.getpixel((3, 5))), 30)
            self.assertEqual((0, 1), (context.report.train_image_count, context.report.val_image_count))
            self.assertEqual(1, context.report.val_annotation_count)

    def test_classification_writes_and_records_each_selected_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            context = self.make_context(root, task_name='classify', task_type=TaskType.CLASSIFY)
            context = Context(
                config=ConversionConfig(
                    task_name=context.config.task_name,
                    task_type=context.config.task_type,
                    root_path=context.config.root_path,
                    split=0,
                    labels=context.config.labels,
                    reserve_no_label=context.config.reserve_no_label,
                ),
                report=context.report,
            )
            ClassificationDatasetSink().write(
                ClassifyOutput(
                    sample=self.make_sample(source, source_index=0), class_name='label', output_name='image.png'
                ),
                context,
            )
            train = root / 'classify' / 'train' / 'label' / 'image.png'
            val = root / 'classify' / 'val' / 'label' / 'image.png'
            self.assertTrue(train.is_file())
            self.assertTrue(val.is_file())
            self.assertEqual(
                ([str(train.absolute())], [str(val.absolute())]), (context.report.train_items, context.report.val_items)
            )
            self.assertNotEqual(context.report.train_items, context.report.val_items)
            self.assertEqual(
                (1, 1, 1, 1),
                (
                    context.report.train_image_count,
                    context.report.val_image_count,
                    context.report.train_annotation_count,
                    context.report.val_annotation_count,
                ),
            )

    def test_sinks_reject_wrong_runtime_input_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self.make_context(Path(temp_dir))
            with self.assertRaisesRegex(TypeError, 'YoloDatasetSink expected EncodeOutput, got ClassifyOutput'):
                YoloDatasetSink().write(
                    ClassifyOutput(
                        sample=self.make_sample(Path('image.png')), class_name='label', output_name='image.png'
                    ),
                    context,
                )
            with self.assertRaisesRegex(
                TypeError, 'ClassificationDatasetSink expected ClassifyOutput, got EncodeOutput'
            ):
                ClassificationDatasetSink().write(
                    EncodeOutput(sample=self.make_sample(Path('image.png')), lines=()), context
                )

    def test_finalize_creates_an_empty_dataset_root_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = self.make_context(root)
            YoloDatasetSink().finalize(context)
            task_root = root / 'detect'
            self.assertEqual('', (task_root / 'train.txt').read_text(encoding='utf-8'))
            self.assertEqual('', (task_root / 'val.txt').read_text(encoding='utf-8'))
            self.assertIn('train: detect/train.txt', (task_root / 'dataset.yaml').read_text(encoding='utf-8'))

    def test_conversion_report_preserves_legacy_summary_and_sorts_files(self) -> None:
        context = self.make_context(Path('.'))
        context.report.record_output(train_item='train.jpg', val_item=None, annotation_count=2)
        context.report.record_output(train_item=None, val_item='val.jpg', annotation_count=3)
        context.report.record_missing_annotations('group')
        context.report.record_skipped_label('undefined')
        context.report.record_skipped_file(Path('z.jpg'))
        context.report.record_skipped_file(Path('a.jpg'))
        output = io.StringIO()

        with redirect_stdout(output):
            print_conversion_report(context)

        self.assertEqual(
            '\n\x1b[1;32m[Convert Summary]\x1b[0m\n'
            '训练集图片总数: 1, 标注总数: 2\n'
            '验证集图片总数: 1, 标注总数: 3\n'
            "类别列表: ['label']\n\n"
            '\x1b[1;31m[Warning] 以下目录包含没有标注的图片\x1b[0m\n'
            '  - group: 1张图片\n'
            '\x1b[1;33m[Warning] 以下类别在标签列表中未定义\x1b[0m\n'
            '  - undefined\n'
            '\x1b[1;33m[Warning] 以下图片因包含未定义类别而被跳过:\x1b[0m\n'
            '  - a.jpg\n'
            '  - z.jpg\n',
            output.getvalue(),
        )


if __name__ == '__main__':
    unittest.main()
