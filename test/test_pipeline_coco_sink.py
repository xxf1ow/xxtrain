import json
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from xxtrain.data import Bbox, ImageInfo, Keypoint, LabelCatalog, Polygon, Polyline, Pose
from xxtrain.data.formats.coco import read_coco
from xxtrain.pipeline import sinks
from xxtrain.pipeline.core import Context, ConversionConfig, ConversionReport, ImageRef, Sample
from xxtrain.task import TaskType


class CocoSinkTest(unittest.TestCase):
    def make_context(
        self,
        root: Path,
        *,
        task_name: str = 'coco',
        task_type: TaskType = TaskType.DETECT,
        reserve_no_label: bool = True,
        labels: tuple[str, ...] = ('label',),
        split: int = 10,
    ) -> Context:
        return Context(
            config=ConversionConfig(
                task_name=task_name,
                task_type=task_type,
                root_path=root,
                split=split,
                labels=LabelCatalog(labels),
                reserve_no_label=reserve_no_label,
            ),
            report=ConversionReport(),
        )

    def make_image(self, root: Path, *, size: tuple[int, int] = (10, 8), name: str = 'image.png') -> Path:
        path = root / 'src' / 'group' / 'images' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.new('RGBA', size, color=(10, 20, 30, 255)) as image:
            image.save(path)
        return path

    def make_sample(
        self,
        image_path: Path,
        *,
        sample_id: str = 'group/image',
        source_index: int = 1,
        crop_box: tuple[float, float, float, float] | None = None,
        info: ImageInfo | None = None,
        annotations: tuple[object, ...] = (),
    ) -> Sample:
        return Sample(
            id=sample_id,
            source_group='group',
            source_index=source_index,
            image=ImageRef(path=image_path, crop_box=crop_box, info=info),
            annotations=annotations,  # type: ignore[arg-type]
        )

    def make_sink(self) -> object:
        self.assertTrue(hasattr(sinks, 'CocoSink'), 'CocoSink must be exported by xxtrain.pipeline.sinks')
        return sinks.CocoSink

    def test_detect_bbox_round_trips_through_coco(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            sample = self.make_sample(
                source, info=ImageInfo(width=10, height=8), annotations=(Bbox(label='label', x1=1, y1=2, x2=7, y2=6),)
            )
            sink = self.make_sink()()

            sink.write(sample, self.make_context(root))
            sink.finalize(self.make_context(root))

            document = read_coco(root / 'coco' / 'annotations.json')
            annotation = document.images[0].annotations[0]
            self.assertEqual(('label',), document.labels.names)
            self.assertEqual((1.0, 2.0, 7.0, 6.0), annotation.bbox)

    def test_pose_uses_single_object_category_and_pose_keypoint_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            pose = Pose(
                label='person',
                x1=1,
                y1=2,
                x2=8,
                y2=7,
                keypoints=(Keypoint(label='nose', x=3, y=4), Keypoint(label='tail', x=6, y=5, visibility=1)),
            )
            context = self.make_context(root, task_type=TaskType.POSE, labels=('person', 'nose', 'tail'))
            sink = self.make_sink()()

            sink.write(self.make_sample(source, info=ImageInfo(width=10, height=8), annotations=(pose,)), context)
            sink.finalize(context)

            payload = json.loads((root / 'coco' / 'annotations.json').read_text(encoding='utf-8'))
            self.assertEqual([{'id': 1, 'name': 'person', 'keypoints': ['nose', 'tail']}], payload['categories'])
            self.assertEqual([3.0, 4.0, 2, 6.0, 5.0, 1], payload['annotations'][0]['keypoints'])

    def test_invalid_obb_is_rejected_before_cache_or_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            context = self.make_context(root, task_type=TaskType.OBB)
            sink = self.make_sink()()
            invalid = Polygon(label='label', points=((1, 1), (5, 1), (4, 4)))

            with self.assertRaises(ValueError):
                sink.write(
                    self.make_sample(source, info=ImageInfo(width=10, height=8), annotations=(invalid,)), context
                )

            self.assertEqual([], sink._images)
            self.assertEqual((0, 0), (context.report.train_image_count, context.report.val_image_count))

    def test_crop_materializes_rgb_jpeg_and_uses_actual_image_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            context = self.make_context(root)
            sample = self.make_sample(
                source,
                sample_id='group/crop',
                crop_box=(1, 2, 7, 7),
                info=ImageInfo(width=99, height=99),
                annotations=(Bbox(label='label', x1=1, y1=1, x2=5, y2=4),),
            )
            sink = self.make_sink()()

            sink.write(sample, context)
            sink.finalize(context)

            image_path = root / 'coco' / 'group' / 'crop.jpg'
            with Image.open(image_path) as image:
                self.assertEqual(('RGB', (6, 5)), (image.mode, image.size))
            payload = json.loads((root / 'coco' / 'annotations.json').read_text(encoding='utf-8'))
            self.assertEqual('group/crop.jpg', payload['images'][0]['file_name'])
            self.assertEqual((6, 5), (payload['images'][0]['width'], payload['images'][0]['height']))

    def test_whole_image_is_referenced_without_copying_and_falls_back_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, chdir(temp_dir):
            root = Path('dataset')
            source = self.make_image(root)
            context = self.make_context(root)
            sink = self.make_sink()()

            sink.write(
                self.make_sample(
                    source,
                    info=ImageInfo(width=10, height=8),
                    annotations=(Bbox(label='label', x1=1, y1=1, x2=4, y2=4),),
                ),
                context,
            )
            sink.finalize(context)

            self.assertFalse((root / 'coco' / 'group' / 'image.png').exists())
            payload = json.loads((root / 'coco' / 'annotations.json').read_text(encoding='utf-8'))
            self.assertEqual('../src/group/images/image.png', payload['images'][0]['file_name'])
            with patch('xxtrain.pipeline.sinks.os.path.relpath', side_effect=ValueError):
                self.assertEqual(
                    str(source.absolute()).replace('\\', '/'), sinks._image_reference(source, root / 'coco')
                )

    def test_empty_samples_are_recorded_and_only_reserved_images_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            skipped_context = self.make_context(root, reserve_no_label=False)
            reserved_context = self.make_context(root, task_name='reserved', reserve_no_label=True)
            skipped = self.make_sink()()
            reserved = self.make_sink()()
            sample = self.make_sample(source, info=ImageInfo(width=10, height=8))

            skipped.write(sample, skipped_context)
            reserved.write(sample, reserved_context)
            reserved.finalize(reserved_context)

            self.assertEqual({'group': 1}, skipped_context.report.missing_annotation_counts)
            self.assertEqual([], skipped._images)
            self.assertEqual({'group': 1}, reserved_context.report.missing_annotation_counts)
            self.assertEqual(1, len(read_coco(root / 'reserved' / 'annotations.json').images))

    def test_failed_empty_crop_write_leaves_state_and_report_clean_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            context = self.make_context(root, reserve_no_label=True)
            sample = self.make_sample(
                source, sample_id='group/crop', crop_box=(1, 1, 5, 5), info=ImageInfo(width=4, height=4)
            )
            sink = self.make_sink()()

            with patch('xxtrain.pipeline.sinks._materialize_crop', side_effect=OSError('crop failed')):
                with self.assertRaisesRegex(OSError, 'crop failed'):
                    sink.write(sample, context)

            self.assertEqual([], sink._images)
            self.assertEqual(set(), sink._claimed_paths)
            self.assertEqual({}, context.report.missing_annotation_counts)
            self.assertEqual((0, 0), (context.report.train_image_count, context.report.val_image_count))

            sink.write(sample, context)

            self.assertEqual(1, len(sink._images))
            self.assertEqual({'group': 1}, context.report.missing_annotation_counts)

    def test_duplicate_whole_image_reference_is_rejected_without_polluting_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            context = self.make_context(root)
            sample = self.make_sample(
                source, info=ImageInfo(width=10, height=8), annotations=(Bbox(label='label', x1=1, y1=1, x2=4, y2=4),)
            )
            sink = self.make_sink()()

            sink.write(sample, context)
            with self.assertRaisesRegex(ValueError, 'COCO image file_name collision'):
                sink.write(sample, context)

            self.assertEqual(1, len(sink._images))
            self.assertEqual(set(), sink._claimed_paths)
            self.assertEqual(
                (1, 0, 1, 0),
                (
                    context.report.train_image_count,
                    context.report.val_image_count,
                    context.report.train_annotation_count,
                    context.report.val_annotation_count,
                ),
            )

    def test_successful_finalize_resets_state_for_the_next_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_root = root / 'first'
            second_root = root / 'second'
            first_context = self.make_context(first_root)
            second_context = self.make_context(second_root)
            first_source = self.make_image(first_root)
            second_source = self.make_image(second_root)
            first = self.make_sample(
                first_source,
                info=ImageInfo(width=10, height=8),
                annotations=(Bbox(label='label', x1=1, y1=1, x2=4, y2=4),),
            )
            second = self.make_sample(
                second_source,
                info=ImageInfo(width=10, height=8),
                annotations=(Bbox(label='label', x1=2, y1=2, x2=6, y2=5),),
            )
            sink = self.make_sink()()

            sink.write(first, first_context)
            sink.finalize(first_context)
            sink.write(second, second_context)
            sink.finalize(second_context)

            document = read_coco(second_root / 'coco' / 'annotations.json')
            self.assertEqual(1, len(document.images))
            self.assertEqual((2.0, 2.0, 6.0, 5.0), document.images[0].annotations[0].bbox)

    def test_safe_ids_preserve_dots_and_reject_unsafe_paths_and_crop_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            context = self.make_context(root)
            sink = self.make_sink()()
            annotated = (Bbox(label='label', x1=1, y1=1, x2=4, y2=4),)

            sink.write(
                self.make_sample(
                    source,
                    sample_id='group/image.v1',
                    crop_box=(1, 1, 5, 5),
                    info=ImageInfo(width=4, height=4),
                    annotations=annotated,
                ),
                context,
            )
            self.assertEqual('group/image.v1.jpg', sink._images[0].file_name)
            for sample_id in ('/absolute', 'C:/absolute', '../escape', '', '.'):
                with self.subTest(sample_id=sample_id), self.assertRaises(ValueError):
                    sink.write(
                        self.make_sample(
                            source, sample_id=sample_id, info=ImageInfo(width=10, height=8), annotations=annotated
                        ),
                        context,
                    )

            crop = self.make_sample(
                source,
                sample_id='group/collision',
                crop_box=(1, 1, 5, 5),
                info=ImageInfo(width=4, height=4),
                annotations=annotated,
            )
            sink.write(crop, context)
            with self.assertRaisesRegex(ValueError, 'collision'):
                sink.write(crop, context)
            self.assertEqual(2, len(sink._images))

    def test_finalize_writes_empty_document_and_write_defers_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = self.make_context(root)
            sink = self.make_sink()()

            sink.finalize(context)
            payload = json.loads((root / 'coco' / 'annotations.json').read_text(encoding='utf-8'))
            self.assertEqual(([], []), (payload['images'], payload['annotations']))

            source = self.make_image(root)
            deferred_context = self.make_context(root, task_name='deferred')
            deferred = self.make_sink()()
            deferred.write(
                self.make_sample(
                    source,
                    info=ImageInfo(width=10, height=8),
                    annotations=(Bbox(label='label', x1=1, y1=1, x2=4, y2=4),),
                ),
                deferred_context,
            )
            self.assertFalse((root / 'deferred' / 'annotations.json').exists())

    def test_rejects_wrong_runtime_type_and_unsupported_polyline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = self.make_context(root)
            sink = self.make_sink()()

            with self.assertRaisesRegex(TypeError, 'CocoSink expected Sample, got object'):
                sink.write(object(), context)
            source = self.make_image(root)
            polyline = Polyline(label='label', points=((1, 1), (5, 5)))
            with self.assertRaisesRegex(TypeError, 'detect does not support Polyline'):
                sink.write(
                    self.make_sample(source, info=ImageInfo(width=10, height=8), annotations=(polyline,)), context
                )

    def test_split_report_counts_match_existing_sink_convention(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.make_image(root)
            context = self.make_context(root, split=2)
            sink = self.make_sink()()
            sink.write(
                self.make_sample(
                    source,
                    source_index=2,
                    info=ImageInfo(width=10, height=8),
                    annotations=(Bbox(label='label', x1=1, y1=1, x2=4, y2=4),),
                ),
                context,
            )

            self.assertEqual(
                (0, 1, 0, 1),
                (
                    context.report.train_image_count,
                    context.report.val_image_count,
                    context.report.train_annotation_count,
                    context.report.val_annotation_count,
                ),
            )
            self.assertEqual([str(source)], context.report.val_items)


if __name__ == '__main__':
    unittest.main()
