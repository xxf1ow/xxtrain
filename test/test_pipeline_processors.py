import tempfile
import unittest
from pathlib import Path

from PIL import Image

from xxtrain.data import Bbox, Circle, ImageInfo, LabelCatalog, Line, Points, Polygon, Polyline, RotatedBbox
from xxtrain.pipeline.core import Context, ConversionConfig, ConversionReport, CropOutput, ImageRef, MatchInput, Sample
from xxtrain.pipeline.processors import (
    CropDetectionBoxes,
    CropMatches,
    FilterLabels,
    FilterMatchingAnnotations,
    MatchAnnotations,
    PartitionAnnotations,
    PrepareClassification,
    PrepareMatchChildren,
    PrepareSegmentShapes,
    ReadImageInfo,
    ReadLabelImg,
    ReadLabelMe,
    ReadMatchingAnnotations,
    RelabelAnnotations,
    RelabelCropAnnotations,
)
from xxtrain.task import TaskType

FIXTURES = Path(__file__).resolve().parent / 'fixtures'


class PipelineProcessorTest(unittest.TestCase):
    def make_context(self, root: Path, labels: tuple[str, ...] = ('known',)) -> Context:
        return Context(
            config=ConversionConfig(
                task_name='detect',
                task_type=TaskType.DETECT,
                root_path=root,
                split=10,
                labels=LabelCatalog(labels),
                reserve_no_label=True,
            ),
            report=ConversionReport(),
        )

    def make_sample(self, image_path: Path, *, annotations=()) -> Sample:
        return Sample(
            id=f'group/{image_path.stem}',
            source_group='group',
            source_index=3,
            image=ImageRef(path=image_path),
            annotations=annotations,
        )

    def test_read_image_info_wraps_without_mutating_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / 'image.png'
            with Image.new('RGB', (10, 8)) as image:
                image.save(image_path)
            sample = self.make_sample(image_path)
            output = ReadImageInfo().transform(sample, self.make_context(Path(temp_dir)))
            self.assertIsNone(sample.image.info)
            self.assertEqual(ImageInfo(width=10, height=8), output.image.info)

    def test_read_labelimg_uses_sibling_anns_path(self) -> None:
        image_path = FIXTURES / 'standard-detect' / 'src' / '20260620' / 'imgs' / '0000.jpg'
        sample = self.make_sample(image_path)
        context = self.make_context(FIXTURES / 'standard-detect', ('cc',))
        sized = ReadImageInfo().transform(sample, context)
        output = ReadLabelImg().transform(sized, context)
        self.assertIsInstance(output.annotations[0], Bbox)
        self.assertEqual('cc', output.annotations[0].label)

    def test_read_matching_annotations_keeps_file_order(self) -> None:
        image_path = FIXTURES / 'standard-pose' / 'src' / 'rw400' / 'imgs' / '0000.jpg'
        sample = self.make_sample(image_path)
        context = self.make_context(FIXTURES / 'standard-pose', ('rw400',))
        sized = ReadImageInfo().transform(sample, context)
        output = ReadMatchingAnnotations().transform(sized, context)
        self.assertEqual(('rw400',), tuple(parent.label for parent in output.parents))
        self.assertEqual(('rw400',), tuple(child.label for child in output.children))

    def test_read_labelme_uses_sibling_anns_seg_path(self) -> None:
        image_path = FIXTURES / 'standard-segment' / 'src' / '251010' / 'imgs' / '0000.jpg'
        sample = self.make_sample(image_path)
        context = self.make_context(FIXTURES / 'standard-segment', ('switch',))
        sized = ReadImageInfo().transform(sample, context)
        output = ReadLabelMe().transform(sized, context)
        self.assertEqual(('switch', 'switch', 'switch', 'switch'), tuple(value.label for value in output.annotations))

    def test_strict_filter_records_file_and_unknown_label(self) -> None:
        image_path = Path('image.jpg')
        annotations = (Bbox(label='known', x1=0, y1=0, x2=2, y2=2), Bbox(label='unknown', x1=3, y1=3, x2=5, y2=5))
        sample = self.make_sample(image_path, annotations=annotations)
        context = self.make_context(Path('.'))
        output = FilterLabels(labels=('known',), strict=True).transform(sample, context)
        self.assertEqual(('known',), tuple(annotation.label for annotation in output.annotations))
        self.assertEqual({'unknown'}, context.report.skipped_labels)
        self.assertEqual({image_path}, context.report.skipped_files)

    def test_matching_filter_filters_both_sides_without_reporting_when_not_strict(self) -> None:
        sample = self.make_sample(Path('image.jpg'))
        item = MatchInput(
            sample=sample,
            parents=(Bbox(label='parent', x1=0, y1=0, x2=2, y2=2), Bbox(label='other-parent', x1=3, y1=3, x2=5, y2=5)),
            children=(Points(label='child', points=((1, 1),)), Points(label='other-child', points=((4, 4),))),
        )
        context = self.make_context(Path('.'))
        output = FilterMatchingAnnotations(parent_labels=('parent',), child_labels=('child',), strict=False).transform(
            item, context
        )
        self.assertEqual(('parent',), tuple(value.label for value in output.parents))
        self.assertEqual(('child',), tuple(value.label for value in output.children))
        self.assertEqual(set(), context.report.skipped_labels)
        self.assertEqual(set(), context.report.skipped_files)

    def test_relabel_returns_new_annotations(self) -> None:
        annotation = Bbox(label='source', x1=0, y1=0, x2=2, y2=2)
        sample = self.make_sample(Path('image.jpg'), annotations=(annotation,))
        output = RelabelAnnotations(label='target').transform(sample, self.make_context(Path('.')))
        self.assertEqual('source', sample.annotations[0].label)
        self.assertEqual('target', output.annotations[0].label)
        self.assertEqual(annotation.id, output.annotations[0].id)

    def test_prepare_segment_converts_bbox_and_circle_to_polygon(self) -> None:
        annotations = (
            Bbox(label='known', x1=0, y1=0, x2=4, y2=3),
            Circle(label='known', center=(10, 10), edge=(20, 10)),
        )
        sample = self.make_sample(Path('image.jpg'), annotations=annotations)
        output = PrepareSegmentShapes().transform(sample, self.make_context(Path('.')))
        self.assertEqual((Polygon, Polygon), tuple(type(value) for value in output.annotations))
        self.assertEqual(((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)), output.annotations[0].points)
        self.assertGreaterEqual(len(output.annotations[1].points), 12)

    def test_prepare_match_children_accepts_pose_shapes_without_conversion(self) -> None:
        sample = self.make_sample(Path('image.jpg'))
        children = (
            Polygon(label='known', points=((0, 0), (2, 0), (1, 1))),
            RotatedBbox(label='known', points=((0, 0), (2, 0), (2, 2), (0, 2))),
            Line(label='known', points=((0, 0), (1, 1))),
            Polyline(label='known', points=((0, 0), (1, 1))),
            Points(label='known', points=((0, 0),)),
        )
        item = MatchInput(sample=sample, parents=(), children=children)
        output = PrepareMatchChildren(TaskType.POSE).transform(item, self.make_context(Path('.')))
        self.assertEqual(children, output.children)
        self.assertTrue(all(original is prepared for original, prepared in zip(children, output.children)))

    def test_prepare_match_children_rejects_pose_bbox(self) -> None:
        sample = self.make_sample(Path('image.jpg'))
        item = MatchInput(sample=sample, parents=(), children=(Bbox(label='known', x1=0, y1=0, x2=2, y2=2),))
        with self.assertRaisesRegex(Exception, "Task pose usually doesn't use"):
            PrepareMatchChildren(TaskType.POSE).transform(item, self.make_context(Path('.')))

    def test_partition_annotations_selects_labels_without_reporting(self) -> None:
        annotations = (
            Bbox(label='parent', x1=0, y1=0, x2=2, y2=2),
            Points(label='child', points=((1, 1),)),
            Points(label='ignored', points=((4, 4),)),
        )
        sample = self.make_sample(Path('image.jpg'), annotations=annotations)
        context = self.make_context(Path('.'))
        output = PartitionAnnotations(parent_labels=('parent',), child_labels=('child',)).transform(sample, context)
        self.assertEqual((annotations[0],), output.parents)
        self.assertEqual((annotations[1],), output.children)
        self.assertEqual(set(), context.report.skipped_labels)
        self.assertEqual(set(), context.report.skipped_files)

    def test_prepare_classification_does_not_open_image(self) -> None:
        sample = self.make_sample(Path('missing.jpg'))
        output = PrepareClassification().transform(sample, self.make_context(Path('.')))
        self.assertEqual('group', output.class_name)
        self.assertEqual('missing.jpg', output.output_name)
        self.assertIs(output.sample, sample)

    def test_match_order_follows_first_child_hit(self) -> None:
        sample = self.make_sample(Path('image.jpg'))
        context = self.make_context(Path('.'))
        parents = (Bbox(label='p', x1=0, y1=0, x2=20, y2=20), Bbox(label='p', x1=30, y1=0, x2=50, y2=20))
        children = (
            Polygon(label='c', points=((32, 2), (40, 2), (40, 10))),
            Polygon(label='c', points=((2, 2), (10, 2), (10, 10))),
            Polygon(label='c', points=((34, 12), (42, 12), (42, 18))),
        )
        output = MatchAnnotations().transform(MatchInput(sample=sample, parents=parents, children=children), context)
        self.assertEqual((parents[1], parents[0]), tuple(match.parent for match in output.matches))
        self.assertEqual((children[0], children[2]), output.matches[0].children)
        self.assertEqual((children[1],), output.matches[1].children)

    def test_crop_matches_builds_deferred_views_in_match_order(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        output_path = Path(temp_dir.name) / 'output'
        sample = self.make_sample(Path('image.jpg'))
        context = self.make_context(output_path)
        parents = (Bbox(label='p', x1=0, y1=0, x2=20, y2=20), Bbox(label='p', x1=30, y1=0, x2=50, y2=20))
        children = (
            Polygon(label='c', points=((32, 2), (40, 2), (40, 10))),
            Polygon(label='c', points=((2, 2), (10, 2), (10, 10))),
        )
        match_output = MatchAnnotations().transform(
            MatchInput(sample=sample, parents=parents, children=children), context
        )
        crops = tuple(CropMatches().expand(match_output, context))
        self.assertEqual(('group/image_0', 'group/image_1'), tuple(crop.sample.id for crop in crops))
        self.assertEqual((sample.source_index, sample.source_index), tuple(crop.sample.source_index for crop in crops))
        self.assertEqual((sample.source_group, sample.source_group), tuple(crop.sample.source_group for crop in crops))
        self.assertEqual(parents[1].bbox, crops[0].sample.image.crop_box)
        self.assertEqual(ImageInfo(width=20, height=20), crops[0].sample.image.info)
        self.assertFalse(output_path.exists())
        self.assertEqual(((2.0, 2.0), (10.0, 2.0), (10.0, 10.0)), crops[0].sample.annotations[0].points)

    def test_crop_matches_preserves_fractional_coordinate_extent(self) -> None:
        sample = self.make_sample(Path('image.jpg'))
        context = self.make_context(Path('.'))
        parent = Bbox(label='p', x1=0.2, y1=0.4, x2=20.8, y2=10.9)
        child = Polygon(label='c', points=((2.2, 2.4), (10.2, 2.4), (10.2, 8.4)))
        match_output = MatchAnnotations().transform(
            MatchInput(sample=sample, parents=(parent,), children=(child,)), context
        )

        crop = next(CropMatches().expand(match_output, context))

        self.assertEqual(parent.bbox, crop.sample.image.crop_box)
        self.assertAlmostEqual(20.6, crop.sample.image.require_info().width)
        self.assertAlmostEqual(10.5, crop.sample.image.require_info().height)
        self.assertEqual(((2.0, 2.0), (10.0, 2.0), (10.0, 8.0)), crop.sample.annotations[0].points)

    def test_crop_detection_boxes_builds_deferred_views_with_legacy_names(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        output_path = Path(temp_dir.name) / 'output'
        annotations = (
            Bbox(label='known', x1=0.2, y1=0.3, x2=20.8, y2=10.9),
            Bbox(label='known', x1=30, y1=0, x2=50, y2=20),
        )
        sample_with_two_boxes = self.make_sample(Path('image.jpg'), annotations=annotations)
        context = self.make_context(output_path)
        outputs = tuple(CropDetectionBoxes().expand(sample_with_two_boxes, context))
        self.assertEqual(('group_3_0.jpg', 'group_3_1.jpg'), tuple(output.output_name for output in outputs))
        self.assertEqual(('group/image_0', 'group/image_1'), tuple(output.sample.id for output in outputs))
        self.assertEqual((3, 3), tuple(output.sample.source_index for output in outputs))
        self.assertEqual(('known', 'known'), tuple(output.class_name for output in outputs))
        self.assertEqual(
            (annotations[0].bbox, annotations[1].bbox), tuple(output.sample.image.crop_box for output in outputs)
        )
        self.assertAlmostEqual(20.6, outputs[0].sample.image.require_info().width)
        self.assertAlmostEqual(10.6, outputs[0].sample.image.require_info().height)
        self.assertEqual(ImageInfo(width=20, height=20), outputs[1].sample.image.info)
        self.assertTrue(all(not output.sample.annotations for output in outputs))
        self.assertFalse(output_path.exists())

    def test_relabel_crop_annotations_preserves_crop_parent(self) -> None:
        annotation = Polygon(label='source', points=((1, 1), (2, 1), (2, 2)))
        parent = Bbox(label='parent', x1=0, y1=0, x2=4, y2=4)
        crop = CropOutput(sample=self.make_sample(Path('image.jpg'), annotations=(annotation,)), parent=parent)
        output = RelabelCropAnnotations('target').transform(crop, self.make_context(Path('.')))
        self.assertEqual('source', crop.sample.annotations[0].label)
        self.assertEqual('target', output.sample.annotations[0].label)
        self.assertEqual(annotation.id, output.sample.annotations[0].id)
        self.assertIs(parent, output.parent)


if __name__ == '__main__':
    unittest.main()
