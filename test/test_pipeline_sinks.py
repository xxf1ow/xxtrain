import io
import json
import shutil
import tempfile
import unittest
from contextlib import chdir, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from xxtrain.data import Bbox, ImageInfo, Keypoint, LabelCatalog, Polygon, Polyline, Pose
from xxtrain.pipeline.core import (
    ClassifyOutput,
    Context,
    ConversionConfig,
    ConversionReport,
    EncodeOutput,
    ImageRef,
    Sample,
)
from xxtrain.pipeline.sinks import (
    ClassificationDatasetSink,
    LabelImgSink,
    LabelMeSink,
    YoloDatasetSink,
    _symlink_or_copy,
    print_conversion_report,
)
from xxtrain.task import TaskType


class PipelineSinkTest(unittest.TestCase):
    def make_context(
        self, root: Path, *, task_name='detect', task_type=TaskType.DETECT, reserve_no_label=True, labels=('label',)
    ) -> Context:
        return Context(
            config=ConversionConfig(
                task_name=task_name,
                task_type=task_type,
                root_path=root,
                split=10,
                labels=LabelCatalog(labels),
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
            self.assertEqual({'label': 1}, context.report.output_label_counts)

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

    def test_labelimg_sink_writes_xml_without_copying_whole_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            sample = self.make_sample(source, info=ImageInfo(width=10, height=10)).wrap(
                annotations=(Bbox(label='label', x1=1, y1=2, x2=7, y2=8),)
            )

            LabelImgSink().write(sample, self.make_context(root))

            annotation = root / 'detect' / 'group' / 'image.xml'
            self.assertTrue(annotation.is_file())
            self.assertFalse((root / 'detect' / 'group' / 'image.png').exists())
            self.assertIn(b'<name>label</name>', annotation.read_bytes())

    def test_labelme_sink_writes_json_referencing_whole_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            sample = self.make_sample(source, info=ImageInfo(width=10, height=10)).wrap(
                annotations=(Bbox(label='label', x1=1, y1=2, x2=7, y2=8),)
            )

            LabelMeSink().write(sample, self.make_context(root))

            annotation = root / 'detect' / 'group' / 'image.json'
            self.assertTrue(annotation.is_file())
            self.assertFalse((root / 'detect' / 'group' / 'image.png').exists())
            payload = json.loads(annotation.read_text(encoding='utf-8'))
            self.assertEqual('../../src/group/imgs/image.png', payload['imagePath'])

    def test_labelme_sink_materializes_crop_and_uses_its_actual_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            sample = self.make_sample(
                source, sample_id='group/image_0', crop_box=(1, 2, 7, 8), info=ImageInfo(width=99, height=99)
            ).wrap(annotations=(Bbox(label='label', x1=1, y1=1, x2=5, y2=5),))

            LabelMeSink().write(sample, self.make_context(root))

            image_path = root / 'detect' / 'group' / 'image_0.jpg'
            annotation = root / 'detect' / 'group' / 'image_0.json'
            with Image.open(image_path) as image:
                self.assertEqual(('RGB', (6, 6)), (image.mode, image.size))
            payload = json.loads(annotation.read_text(encoding='utf-8'))
            self.assertEqual('image_0.jpg', payload['imagePath'])
            self.assertEqual((6, 6), (payload['imageWidth'], payload['imageHeight']))

    def test_annotation_sinks_skip_unlabelled_samples_when_not_reserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            sample = self.make_sample(source, info=ImageInfo(width=10, height=10))
            context = self.make_context(root, reserve_no_label=False)

            for sink, suffix in ((LabelImgSink(), '.xml'), (LabelMeSink(), '.json')):
                with self.subTest(sink=type(sink).__name__):
                    sink.write(sample, context)
                    self.assertFalse((root / 'detect' / 'group' / f'image{suffix}').exists())

            self.assertEqual({'group': 2}, context.report.missing_annotation_counts)

    def test_annotation_sinks_write_empty_annotations_when_reserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            sample = self.make_sample(source, info=ImageInfo(width=10, height=10))
            context = self.make_context(root)

            LabelImgSink().write(sample, context)
            LabelMeSink().write(sample, context)

            self.assertIn(b'<annotation', (root / 'detect' / 'group' / 'image.xml').read_bytes())
            payload = json.loads((root / 'detect' / 'group' / 'image.json').read_text(encoding='utf-8'))
            self.assertEqual([], payload['shapes'])

    def test_annotation_sinks_retry_after_writer_failure_without_committing_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            sample = self.make_sample(
                source, sample_id='group/crop', crop_box=(1, 1, 6, 6), info=ImageInfo(width=5, height=5)
            ).wrap(annotations=(Bbox(label='label', x1=1, y1=1, x2=4, y2=4),))

            for sink, suffix, writer in (
                (LabelImgSink(), '.xml', 'write_labelimg'),
                (LabelMeSink(), '.json', 'write_labelme'),
            ):
                with self.subTest(sink=type(sink).__name__):
                    output_root = root / type(sink).__name__
                    context = self.make_context(output_root)
                    annotation = output_root / 'detect' / 'group' / f'crop{suffix}'
                    crop = output_root / 'detect' / 'group' / 'crop.jpg'
                    with patch(f'xxtrain.pipeline.sinks.{writer}', side_effect=OSError('write failed')):
                        with self.assertRaisesRegex(OSError, 'write failed'):
                            sink.write(sample, context)
                    self.assertEqual(set(), sink._claimed_paths)
                    self.assertEqual({}, context.report.missing_annotation_counts)
                    self.assertFalse(annotation.exists())
                    self.assertFalse(crop.exists())

                    sink.write(sample, context)
                    self.assertTrue(annotation.exists())
                    self.assertTrue(crop.exists())

    def test_annotation_sinks_retry_after_crop_failure_without_committing_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            sample = self.make_sample(
                source, sample_id='group/crop', crop_box=(1, 1, 6, 6), info=ImageInfo(width=5, height=5)
            ).wrap(annotations=(Bbox(label='label', x1=1, y1=1, x2=4, y2=4),))

            for sink, suffix in ((LabelImgSink(), '.xml'), (LabelMeSink(), '.json')):
                with self.subTest(sink=type(sink).__name__):
                    output_root = root / type(sink).__name__
                    context = self.make_context(output_root)
                    annotation = output_root / 'detect' / 'group' / f'crop{suffix}'
                    crop = output_root / 'detect' / 'group' / 'crop.jpg'
                    with patch('xxtrain.pipeline.sinks._materialize_crop', side_effect=OSError('crop failed')):
                        with self.assertRaisesRegex(OSError, 'crop failed'):
                            sink.write(sample, context)
                    self.assertEqual(set(), sink._claimed_paths)
                    self.assertEqual({}, context.report.missing_annotation_counts)
                    self.assertFalse(annotation.exists())
                    self.assertFalse(crop.exists())

                    sink.write(sample, context)
                    self.assertTrue(annotation.exists())
                    self.assertTrue(crop.exists())

    def test_annotation_sinks_clean_temporary_annotation_files_after_writer_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            sample = self.make_sample(
                source, sample_id='group/crop', crop_box=(1, 1, 6, 6), info=ImageInfo(width=5, height=5)
            ).wrap(annotations=(Bbox(label='label', x1=1, y1=1, x2=4, y2=4),))

            def write_then_fail(*args, **kwargs) -> None:
                Path(args[1]).write_text('partial', encoding='utf-8')
                raise OSError('write failed')

            for sink, suffix, writer in (
                (LabelImgSink(), '.xml', 'write_labelimg'),
                (LabelMeSink(), '.json', 'write_labelme'),
            ):
                with self.subTest(sink=type(sink).__name__):
                    output_root = root / type(sink).__name__
                    context = self.make_context(output_root)
                    annotation = output_root / 'detect' / 'group' / f'crop{suffix}'
                    crop = output_root / 'detect' / 'group' / 'crop.jpg'
                    with patch(f'xxtrain.pipeline.sinks.{writer}', side_effect=write_then_fail):
                        with self.assertRaisesRegex(OSError, 'write failed'):
                            sink.write(sample, context)
                    self.assertFalse(annotation.exists())
                    self.assertFalse(crop.exists())
                    self.assertEqual([], list(annotation.parent.glob('*.tmp')))

    def test_annotation_sinks_retry_after_temporary_annotation_creation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            sample = self.make_sample(
                source, sample_id='group/crop', crop_box=(1, 1, 6, 6), info=ImageInfo(width=5, height=5)
            ).wrap(annotations=(Bbox(label='label', x1=1, y1=1, x2=4, y2=4),))

            for sink, suffix in ((LabelImgSink(), '.xml'), (LabelMeSink(), '.json')):
                with self.subTest(sink=type(sink).__name__):
                    output_root = root / type(sink).__name__
                    context = self.make_context(output_root)
                    annotation = output_root / 'detect' / 'group' / f'crop{suffix}'
                    crop = output_root / 'detect' / 'group' / 'crop.jpg'
                    with patch('xxtrain.pipeline.sinks.tempfile.mkstemp', side_effect=OSError('temp failed')):
                        with self.assertRaisesRegex(OSError, 'temp failed'):
                            sink.write(sample, context)
                    self.assertEqual(set(), sink._claimed_paths)
                    self.assertEqual({}, context.report.missing_annotation_counts)
                    self.assertFalse(annotation.exists())
                    self.assertFalse(crop.exists())
                    self.assertEqual([], list(annotation.parent.glob('*.tmp')))

                    sink.write(sample, context)
                    self.assertTrue(annotation.exists())
                    self.assertTrue(crop.exists())

    def test_annotation_sinks_retry_after_annotation_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            sample = self.make_sample(
                source, sample_id='group/crop', crop_box=(1, 1, 6, 6), info=ImageInfo(width=5, height=5)
            ).wrap(annotations=(Bbox(label='label', x1=1, y1=1, x2=4, y2=4),))

            for sink, suffix in ((LabelImgSink(), '.xml'), (LabelMeSink(), '.json')):
                with self.subTest(sink=type(sink).__name__):
                    output_root = root / type(sink).__name__
                    context = self.make_context(output_root)
                    annotation = output_root / 'detect' / 'group' / f'crop{suffix}'
                    crop = output_root / 'detect' / 'group' / 'crop.jpg'
                    with patch('xxtrain.pipeline.sinks.os.replace', side_effect=OSError('replace failed')):
                        with self.assertRaisesRegex(OSError, 'replace failed'):
                            sink.write(sample, context)
                    self.assertEqual(set(), sink._claimed_paths)
                    self.assertEqual({}, context.report.missing_annotation_counts)
                    self.assertFalse(annotation.exists())
                    self.assertFalse(crop.exists())
                    self.assertEqual([], list(annotation.parent.glob('*.tmp')))

                    sink.write(sample, context)
                    self.assertTrue(annotation.exists())
                    self.assertTrue(crop.exists())

    def test_reserved_empty_annotation_retries_without_recording_missing_until_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            sample = self.make_sample(source, info=ImageInfo(width=10, height=10))

            for sink, suffix, writer in (
                (LabelImgSink(), '.xml', 'write_labelimg'),
                (LabelMeSink(), '.json', 'write_labelme'),
            ):
                with self.subTest(sink=type(sink).__name__):
                    output_root = root / type(sink).__name__
                    context = self.make_context(output_root, reserve_no_label=True)
                    annotation = output_root / 'detect' / 'group' / f'image{suffix}'
                    with patch(f'xxtrain.pipeline.sinks.{writer}', side_effect=OSError('write failed')):
                        with self.assertRaisesRegex(OSError, 'write failed'):
                            sink.write(sample, context)
                    self.assertEqual({}, context.report.missing_annotation_counts)
                    self.assertEqual(set(), sink._claimed_paths)
                    self.assertFalse(annotation.exists())

                    sink.write(sample, context)
                    self.assertEqual({'group': 1}, context.report.missing_annotation_counts)

    def test_annotation_sinks_preserve_preexisting_annotation_on_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            sample = self.make_sample(source, info=ImageInfo(width=10, height=10)).wrap(
                annotations=(Bbox(label='label', x1=1, y1=1, x2=4, y2=4),)
            )

            for sink, suffix in ((LabelImgSink(), '.xml'), (LabelMeSink(), '.json')):
                with self.subTest(sink=type(sink).__name__):
                    annotation = root / 'detect' / 'group' / f'image{suffix}'
                    annotation.parent.mkdir(parents=True, exist_ok=True)
                    annotation.write_text('existing', encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, 'collision'):
                        sink.write(sample, self.make_context(root))
                    self.assertEqual('existing', annotation.read_text(encoding='utf-8'))

    def test_annotation_sinks_preserve_dotted_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root, name='image.v1.png')
            sample = self.make_sample(source, sample_id='group/image.v1', info=ImageInfo(width=10, height=10))

            LabelImgSink().write(sample, self.make_context(root))
            LabelMeSink().write(sample, self.make_context(root))

            self.assertTrue((root / 'detect' / 'group' / 'image.v1.xml').is_file())
            self.assertTrue((root / 'detect' / 'group' / 'image.v1.json').is_file())

    def test_annotation_sinks_reject_unsafe_or_colliding_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            valid = self.make_sample(source, info=ImageInfo(width=10, height=10))
            unsafe = (
                self.make_sample(source, sample_id='../escape', info=ImageInfo(width=10, height=10)),
                self.make_sample(source, sample_id=str(root / 'escape'), info=ImageInfo(width=10, height=10)),
                self.make_sample(source, sample_id='.', info=ImageInfo(width=10, height=10)),
            )

            for sink in (LabelImgSink(), LabelMeSink()):
                with self.subTest(sink=type(sink).__name__, case='unsafe'):
                    for sample in unsafe:
                        with self.assertRaisesRegex(ValueError, 'Unsafe sample id'):
                            sink.write(sample, self.make_context(root))
                with self.subTest(sink=type(sink).__name__, case='collision'):
                    sink.write(valid, self.make_context(root))
                    with self.assertRaisesRegex(ValueError, 'collision'):
                        sink.write(valid, self.make_context(root))

    def test_annotation_sinks_reject_root_relative_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            sample = self.make_sample(source, sample_id=r'\escape', info=ImageInfo(width=10, height=10)).wrap(
                annotations=(Bbox(label='label', x1=1, y1=1, x2=5, y2=5),)
            )

            for sink, writer in ((LabelImgSink(), 'write_labelimg'), (LabelMeSink(), 'write_labelme')):
                with self.subTest(sink=type(sink).__name__), patch(f'xxtrain.pipeline.sinks.{writer}'):
                    with self.assertRaisesRegex(ValueError, 'Unsafe sample id'):
                        sink.write(sample, self.make_context(root))

    def test_annotation_sinks_reject_unsafe_empty_samples_when_not_reserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            unsafe_ids = ('', '.', r'..\escape', r'\escape', str(root / 'escape'))

            for sink in (LabelImgSink(), LabelMeSink()):
                for sample_id in unsafe_ids:
                    with self.subTest(sink=type(sink).__name__, sample_id=sample_id):
                        sample = self.make_sample(source, sample_id=sample_id, info=ImageInfo(width=10, height=10))
                        with self.assertRaisesRegex(ValueError, 'Unsafe sample id'):
                            sink.write(sample, self.make_context(root, reserve_no_label=False))

    def test_labelme_sink_rejects_polyline_and_labelimg_rejects_non_detect_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            segment_context = self.make_context(root, task_name='segment', task_type=TaskType.SEGMENT)
            polyline = self.make_sample(source, info=ImageInfo(width=10, height=10)).wrap(
                annotations=(Polyline(label='label', points=((1, 1), (5, 5))),)
            )

            with self.assertRaisesRegex(TypeError, 'segment does not support Polyline'):
                LabelMeSink().write(polyline, segment_context)
            with self.assertRaisesRegex(ValueError, 'detect'):
                LabelImgSink().write(self.make_sample(source, info=ImageInfo(width=10, height=10)), segment_context)

    def test_labelme_sink_supports_segment_pose_and_obb(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            samples = (
                (
                    self.make_sample(source, sample_id='group/segment', info=ImageInfo(width=10, height=10)).wrap(
                        annotations=(Polygon(label='label', points=((1, 1), (5, 1), (3, 5))),)
                    ),
                    self.make_context(root, task_name='segment', task_type=TaskType.SEGMENT),
                    'polygon',
                ),
                (
                    self.make_sample(source, sample_id='group/pose', info=ImageInfo(width=10, height=10)).wrap(
                        annotations=(
                            Pose(label='person', x1=1, y1=1, x2=5, y2=5, keypoints=(Keypoint(label='nose', x=3, y=2),)),
                        )
                    ),
                    self.make_context(root, task_name='pose', task_type=TaskType.POSE, labels=('person', 'nose')),
                    'rectangle',
                ),
                (
                    self.make_sample(source, sample_id='group/obb', info=ImageInfo(width=10, height=10)).wrap(
                        annotations=(Polygon(label='label', points=((1, 1), (5, 1), (5, 4), (1, 4))),)
                    ),
                    self.make_context(root, task_name='obb', task_type=TaskType.OBB),
                    'rotation',
                ),
            )

            for sample, context, shape_type in samples:
                with self.subTest(task_type=context.config.task_type):
                    LabelMeSink().write(sample, context)
                    annotation_path = root / context.config.task_name / 'group' / f'{Path(sample.id).name}.json'
                    payload = json.loads(annotation_path.read_text(encoding='utf-8'))
                    self.assertEqual(shape_type, payload['shapes'][0]['shape_type'])

    def test_annotation_sink_finalize_creates_only_the_task_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = self.make_context(root)

            LabelImgSink().finalize(context)
            LabelMeSink().finalize(context)

            task_root = root / 'detect'
            self.assertTrue(task_root.is_dir())
            self.assertFalse((task_root / 'dataset.yaml').exists())
            self.assertFalse((task_root / 'train.txt').exists())

    def test_annotation_sinks_reject_wrong_runtime_input_type(self) -> None:
        context = self.make_context(Path('.'))
        output = EncodeOutput(sample=self.make_sample(Path('image.png')), lines=())
        for sink in (LabelImgSink(), LabelMeSink()):
            with self.subTest(sink=type(sink).__name__):
                with self.assertRaisesRegex(TypeError, f'{type(sink).__name__} expected Sample, got EncodeOutput'):
                    sink.write(output, context)

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

    def test_classification_sink_can_repeat_an_interrupted_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root, size=(10, 7))
            source_bytes = source.read_bytes()
            output = ClassifyOutput(sample=self.make_sample(source), class_name='label', output_name='image.png')

            first_context = self.make_context(root, task_name='classify', task_type=TaskType.CLASSIFY)
            ClassificationDatasetSink().write(output, first_context)
            target = root / 'classify' / 'train' / 'label' / 'image.png'
            first_output = target.read_bytes()

            restarted_context = self.make_context(root, task_name='classify', task_type=TaskType.CLASSIFY)
            ClassificationDatasetSink().write(output, restarted_context)

            self.assertEqual(first_output, target.read_bytes())
            self.assertEqual(source_bytes, source.read_bytes())
            self.assertEqual([str(target.absolute())], restarted_context.report.train_items)

    def test_standard_classification_letterboxes_wide_image_to_224(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root, size=(10, 7), name='source.png')
            source_bytes = source.read_bytes()
            context = self.make_context(root, task_name='classify', task_type=TaskType.CLASSIFY)
            output = ClassifyOutput(sample=self.make_sample(source), class_name='label', output_name='source.png')

            ClassificationDatasetSink().write(output, context)

            target = root / 'classify' / 'train' / 'label' / 'source.png'
            with Image.open(target).convert('RGB') as image:
                self.assertEqual((224, 224), image.size)
                self.assertEqual((114, 114, 114), image.getpixel((0, 32)))
                self.assertEqual((10, 20, 30), image.getpixel((0, 33)))
                self.assertEqual((10, 20, 30), image.getpixel((223, 189)))
                self.assertEqual((114, 114, 114), image.getpixel((0, 190)))
            self.assertEqual(source_bytes, source.read_bytes())
            self.assertEqual([str(target.absolute())], context.report.train_items)

    def test_standard_classification_letterboxes_tall_image_to_224(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root, size=(7, 10), name='source.png')
            context = self.make_context(root, task_name='classify', task_type=TaskType.CLASSIFY)
            output = ClassifyOutput(sample=self.make_sample(source), class_name='label', output_name='source.png')

            ClassificationDatasetSink().write(output, context)

            target = root / 'classify' / 'train' / 'label' / 'source.png'
            with Image.open(target).convert('RGB') as image:
                self.assertEqual((224, 224), image.size)
                self.assertEqual((114, 114, 114), image.getpixel((32, 0)))
                self.assertEqual((10, 20, 30), image.getpixel((33, 0)))
                self.assertEqual((10, 20, 30), image.getpixel((189, 223)))
                self.assertEqual((114, 114, 114), image.getpixel((190, 0)))

    def test_standard_classification_downscale_uses_linear_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / 'src' / 'group' / 'imgs' / 'source.png'
            source.parent.mkdir(parents=True)
            checkerboard = bytes(255 if (x + y) % 2 else 0 for y in range(224) for x in range(448))
            with Image.frombytes('L', (448, 224), checkerboard).convert('RGB') as image:
                image.save(source)
            context = self.make_context(root, task_name='classify', task_type=TaskType.CLASSIFY)
            output = ClassifyOutput(sample=self.make_sample(source), class_name='label', output_name='source.png')

            ClassificationDatasetSink().write(output, context)

            target = root / 'classify' / 'train' / 'label' / 'source.png'
            with Image.open(target).convert('RGB') as image:
                self.assertEqual((114, 114, 114), image.getpixel((0, 55)))
                self.assertEqual((128, 128, 128), image.getpixel((0, 56)))
                self.assertEqual((128, 128, 128), image.getpixel((223, 167)))
                self.assertEqual((114, 114, 114), image.getpixel((0, 168)))

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
            output = ClassifyOutput(sample=sample, class_name='label', output_name='group_0_0.png')
            ClassificationDatasetSink(indexed_class_directories=True).write(output, context)
            target = root / 'point-classify' / 'val' / '00-label' / 'group_0_0.png'
            with Image.open(target).convert('RGB') as image:
                self.assertEqual((224, 224), image.size)
                self.assertEqual((114, 114, 114), image.getpixel((112, 36)))
                self.assertEqual((10, 20, 30), image.getpixel((112, 37)))
                self.assertEqual((10, 20, 30), image.getpixel((112, 185)))
                self.assertEqual((114, 114, 114), image.getpixel((112, 186)))
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
            self.assertEqual(train.read_bytes(), val.read_bytes())
            with Image.open(train) as image:
                self.assertEqual((224, 224), image.size)
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
            self.assertEqual({'label': 1}, context.report.output_label_counts)

    def test_classification_write_failure_raises_before_recording_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            context = self.make_context(root, task_name='classify', task_type=TaskType.CLASSIFY)
            output = ClassifyOutput(sample=self.make_sample(source), class_name='label', output_name='image.png')
            target = root / 'classify' / 'train' / 'label' / 'image.png'

            with patch('xxtrain.pipeline.sinks.cv2.imwrite', return_value=False), self.assertRaises(OSError) as raised:
                ClassificationDatasetSink().write(output, context)

            self.assertEqual(f'failed to write classification image: {target}', str(raised.exception))
            self.assertEqual(([], []), (context.report.train_items, context.report.val_items))
            self.assertEqual((0, 0), (context.report.train_image_count, context.report.val_image_count))
            self.assertEqual({}, context.report.output_label_counts)

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

    def test_conversion_report_prints_label_counts_and_sorts_entries(self) -> None:
        context = self.make_context(Path('.'))
        context.report.record_output(train_item='train.jpg', val_item=None, annotation_count=2)
        context.report.record_output(train_item=None, val_item='val.jpg', annotation_count=3)
        context.report.record_missing_annotations('group')
        context.report.record_skipped_label('z-label')
        context.report.record_skipped_file(Path('z.jpg'))
        context.report.record_skipped_file(Path('a.jpg'))
        context.report.record_source_label('z-label', Path('z.jpg'))
        context.report.record_source_label('a-label', Path('b.jpg'))
        context.report.record_source_label('a-label', Path('a.jpg'))
        context.report.record_ignored_label('z-label')
        context.report.record_output_label('label')
        context.report.set_missing_output_labels(('missing',))
        output = io.StringIO()

        with redirect_stdout(output):
            print_conversion_report(context)

        text = output.getvalue()
        self.assertIn('a-label: 2', text)
        self.assertIn('z-label: 1', text)
        self.assertIn('label: 1', text)
        self.assertIn('missing', text)
        self.assertLess(text.index('a-label: 2'), text.index('z-label: 1'))
        self.assertLess(text.index('a.jpg'), text.index('z.jpg'))


if __name__ == '__main__':
    unittest.main()
