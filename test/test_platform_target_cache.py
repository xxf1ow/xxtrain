import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import yaml
from PIL import Image

from xxtrain.business_tasks.point import point_task_definition
from xxtrain.data import Bbox, ImageInfo, LabelCatalog, Polyline
from xxtrain.pipeline.core import (
    ClassifyOutput,
    Context,
    ConversionConfig,
    ConversionReport,
    CropOutput,
    EncodeOutput,
    ImageRef,
    Sample,
)
from xxtrain.pipeline.processors import EncodePointSegment
from xxtrain.pipeline.sinks import ClassificationDatasetSink
from xxtrain.platform.contracts import AnnotationRecord, ImageRecord
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.target_cache import build_target_cache
from xxtrain.task import TaskType
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.repository import AnnotationRepository


class PlatformTargetCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='xxtrain-target-cache-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace_root = self.root / 'workspace'
        images_root = self.workspace_root / 'images'
        images_root.mkdir(parents=True)
        image_path = images_root / 'source.png'
        with Image.new('RGB', (100, 80)) as image:
            for y in range(image.height):
                for x in range(image.width):
                    image.putpixel((x, y), ((x * 3) % 256, (y * 5) % 256, (x + y) % 256))
            image.save(image_path)

        self.data = WorkspaceData(self.workspace_root, point_task_definition())
        self.repository = AnnotationRepository(self.workspace_root / 'annotations.db', point_task_definition())
        self.image = ImageRecord('a' * 64, 'images/source.png', 100, 80, 0)
        self.repository.register_images((self.image,))
        self.first = AnnotationRecord(
            UUID('10000000-0000-0000-0000-000000000001'),
            self.image.id,
            'detect',
            None,
            'rectangle',
            'tl',
            [[10, 20], [50, 60]],
        )
        self.second = AnnotationRecord(
            UUID('10000000-0000-0000-0000-000000000002'),
            self.image.id,
            'detect',
            None,
            'rectangle',
            'cc',
            [[60, 10], [90, 40]],
        )
        records = (
            self.first,
            self.second,
            AnnotationRecord(
                UUID('20000000-0000-0000-0000-000000000001'),
                self.image.id,
                'classify',
                self.first.id,
                'classification',
                'tl',
                None,
            ),
            AnnotationRecord(
                UUID('20000000-0000-0000-0000-000000000002'),
                self.image.id,
                'classify',
                self.second.id,
                'classification',
                'cc',
                None,
            ),
            AnnotationRecord(
                UUID('30000000-0000-0000-0000-000000000001'),
                self.image.id,
                'segment',
                self.first.id,
                'polyline',
                '1',
                [[20, 30], [30, 40]],
            ),
            AnnotationRecord(
                UUID('30000000-0000-0000-0000-000000000002'),
                self.image.id,
                'segment',
                self.second.id,
                'polyline',
                '1',
                [[65, 15], [80, 25]],
            ),
        )
        self.repository.save_annotations(records)
        self.runtime_root = self.root / 'runtime'

    def test_builds_classification_from_shared_crops_with_existing_preprocessing(self) -> None:
        frames = self.data.target_frames('classify', self.runtime_root)
        crop_mtimes = {frame.image_path: frame.image_path.stat().st_mtime_ns for frame in frames}
        database_before = (self.workspace_root / 'annotations.db').read_bytes()
        destination = self.runtime_root / 'cache' / self.data.target_fingerprint('classify')

        report = build_target_cache(self.data, 'classify', self.runtime_root, destination)

        self.assertEqual(database_before, (self.workspace_root / 'annotations.db').read_bytes())
        self.assertEqual(crop_mtimes, {path: path.stat().st_mtime_ns for path in crop_mtimes})
        self.assertEqual(2, len(report.train_items) + len(report.val_items))
        self.assertEqual(
            {'00-tl', '03-cc'},
            {path.name for split in ('train', 'val') for path in (destination / 'classify' / split).glob('*')},
        )

        expected_root = self.root / 'expected-classification'
        expected_context = Context(
            config=ConversionConfig(
                task_name='classify',
                task_type=TaskType.CLASSIFY,
                root_path=expected_root,
                split=10,
                labels=LabelCatalog(('tl', 'tc', 'cl', 'cc')),
                reserve_no_label=False,
            ),
            report=ConversionReport(),
        )
        expected_sink = ClassificationDatasetSink(indexed_class_directories=True)
        labels = ('tl', 'cc')
        for index, (frame, label) in enumerate(zip(frames, labels, strict=True)):
            sample = Sample(
                id=frame.mapping.frame_id,
                source_group=label,
                source_index=index,
                image=ImageRef(
                    path=frame.image_path, info=ImageInfo(width=frame.width, height=frame.height), crop_box=None
                ),
            )
            expected_sink.write(
                ClassifyOutput(sample=sample, class_name=label, output_name=f'{frame.mapping.frame_id}.png'),
                expected_context,
            )

        actual = sorted(Path(item) for item in (*report.train_items, *report.val_items))
        expected = sorted(
            Path(item) for item in (*expected_context.report.train_items, *expected_context.report.val_items)
        )
        self.assertEqual([path.read_bytes() for path in expected], [path.read_bytes() for path in actual])
        self.assertTrue(all(path.is_file() and path.is_relative_to(destination) for path in actual))
        dataset = yaml.safe_load((destination / 'classify' / 'dataset.yaml').read_text(encoding='utf-8'))
        self.assertEqual(str(destination), dataset['path'])
        split_paths = [destination / dataset[name] for name in ('train', 'val')]
        listed = [Path(line) for path in split_paths for line in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(set(actual), set(listed))

    def test_builds_segment_with_existing_triangle_encoder_and_independent_crop_copies(self) -> None:
        frames = self.data.target_frames('segment', self.runtime_root)
        database_before = (self.workspace_root / 'annotations.db').read_bytes()
        destination = self.runtime_root / 'cache' / self.data.target_fingerprint('segment')

        report = build_target_cache(self.data, 'segment', self.runtime_root, destination)

        self.assertEqual(database_before, (self.workspace_root / 'annotations.db').read_bytes())
        self.assertEqual(2, len(report.train_items) + len(report.val_items))
        expected_lines = (
            self._encode_lines(((10, 10), (20, 20)), (40, 40)),
            self._encode_lines(((5, 5), (20, 15)), (30, 30)),
        )
        output_images = [Path(item) for item in (*report.train_items, *report.val_items)]
        output_by_id = {path.stem: path for path in output_images}
        for frame, lines in zip(frames, expected_lines, strict=True):
            output = output_by_id[frame.mapping.frame_id]
            self.assertEqual(frame.image_path.read_bytes(), output.read_bytes())
            self.assertFalse(output.is_symlink())
            self.assertEqual(lines, output.with_suffix('.txt').read_text(encoding='utf-8').splitlines())
            for row in lines:
                values = [float(value) for value in row.split()]
                self.assertEqual(0.0, values[0])
                self.assertTrue(all(0.0 <= value <= 1.0 for value in values[1:]))

        shutil.rmtree(self.runtime_root / 'crops')
        self.assertTrue(all(path.is_file() for path in output_images))
        self.assertTrue(all(path.with_suffix('.txt').is_file() for path in output_images))

    def test_preserves_every_line_for_one_crop(self) -> None:
        self.repository.save_annotations(
            (
                AnnotationRecord(
                    UUID('30000000-0000-0000-0000-000000000003'),
                    self.image.id,
                    'segment',
                    self.first.id,
                    'polyline',
                    '1',
                    [[25, 35], [35, 45]],
                ),
            )
        )
        destination = self.runtime_root / 'cache' / 'all-lines'

        report = build_target_cache(self.data, 'segment', self.runtime_root, destination)

        outputs = {Path(item).stem: Path(item) for item in (*report.train_items, *report.val_items)}
        lines = outputs[str(self.first.id)].with_suffix('.txt').read_text(encoding='utf-8').splitlines()
        self.assertEqual(2, len(lines))

    def test_runtime_readiness_requires_exact_manifest_and_target_directory(self) -> None:
        fingerprint = self.data.target_fingerprint('classify')
        destination = self.runtime_root / 'cache' / fingerprint
        runtime = RuntimeCache(self.runtime_root)
        self.assertFalse(runtime.has_target_cache('classify', fingerprint))

        build_target_cache(self.data, 'classify', self.runtime_root, destination)

        self.assertTrue(runtime.has_target_cache('classify', fingerprint))
        manifest = destination / 'manifest.json'
        self.assertEqual({'fingerprint': fingerprint, 'target': 'classify'}, json.loads(manifest.read_text()))
        manifest.write_bytes(b'\xff')
        self.assertFalse(runtime.has_target_cache('classify', fingerprint))
        manifest.write_text('{', encoding='utf-8')
        self.assertFalse(runtime.has_target_cache('classify', fingerprint))
        manifest.write_text(json.dumps({'fingerprint': fingerprint, 'target': 'segment'}), encoding='utf-8')
        self.assertFalse(runtime.has_target_cache('classify', fingerprint))
        manifest.write_text(json.dumps({'fingerprint': fingerprint, 'target': 'classify'}), encoding='utf-8')
        shutil.rmtree(destination / 'classify')
        self.assertFalse(runtime.has_target_cache('classify', fingerprint))

        rebuilt = build_target_cache(self.data, 'classify', self.runtime_root, destination)
        self.assertEqual(2, len(rebuilt.train_items) + len(rebuilt.val_items))
        self.assertTrue(runtime.has_target_cache('classify', fingerprint))

    def test_publication_failure_does_not_mark_cache_ready(self) -> None:
        fingerprint = self.data.target_fingerprint('segment')
        destination = self.runtime_root / 'cache' / fingerprint
        with patch('xxtrain.platform.target_cache.os.replace', side_effect=OSError('publish failed')):
            with self.assertRaisesRegex(OSError, 'publish failed'):
                build_target_cache(self.data, 'segment', self.runtime_root, destination)

        self.assertFalse(RuntimeCache(self.runtime_root).has_target_cache('segment', fingerprint))
        self.assertFalse(destination.with_name(f'.{destination.name}.building').exists())

    def test_rejects_unsupported_target_and_out_of_bounds_triangle(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported target cache: 'detect'"):
            build_target_cache(self.data, 'detect', self.runtime_root, self.root / 'detect')
        with self.assertRaisesRegex(ValueError, "Unsupported target cache: 'detect'"):
            RuntimeCache(self.runtime_root).has_target_cache('detect', 'fingerprint')

        self.repository.save_annotations(
            (
                AnnotationRecord(
                    UUID('30000000-0000-0000-0000-000000000003'),
                    self.image.id,
                    'segment',
                    self.first.id,
                    'polyline',
                    '1',
                    [[10, 30], [10, 40]],
                ),
            )
        )
        destination = self.root / 'invalid-segment'
        with self.assertRaisesRegex(ValueError, 'outside crop bounds'):
            build_target_cache(self.data, 'segment', self.runtime_root, destination)
        self.assertFalse(destination.exists())

    def test_rejects_invalid_normalized_encoder_output_without_assertions(self) -> None:
        destination = self.root / 'invalid-encoded-segment'

        def invalid_output(_encoder, item, _context):
            return EncodeOutput(sample=item.sample, lines=('0 -0.100000 0.200000 0.300000 0.400000 0.500000 0.600000',))

        with patch.object(EncodePointSegment, 'transform', invalid_output):
            with self.assertRaisesRegex(ValueError, 'outside crop bounds'):
                build_target_cache(self.data, 'segment', self.runtime_root, destination)
        self.assertFalse(destination.exists())

    @staticmethod
    def _encode_lines(points: tuple[tuple[int, int], tuple[int, int]], size: tuple[int, int]) -> list[str]:
        context = Context(
            config=ConversionConfig(
                task_name='segment',
                task_type=TaskType.SEGMENT,
                root_path=Path('.'),
                split=10,
                labels=LabelCatalog(('Point',)),
                reserve_no_label=False,
            ),
            report=ConversionReport(),
        )
        sample = Sample(
            id='workspace/frame',
            source_group='workspace',
            source_index=0,
            image=ImageRef(path=Path('crop.png'), info=ImageInfo(width=size[0], height=size[1]), crop_box=None),
            annotations=(Polyline(label='1', points=points),),
        )
        parent = Bbox(label='Point', x1=0, y1=0, x2=size[0], y2=size[1])
        return list(EncodePointSegment().transform(CropOutput(sample=sample, parent=parent), context).lines)


if __name__ == '__main__':
    unittest.main()
