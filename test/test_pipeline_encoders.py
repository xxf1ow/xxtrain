import unittest
from pathlib import Path

from xxtrain.data import Bbox, ImageInfo, LabelCatalog, Points, Polygon, Polyline
from xxtrain.pipeline.core import (
    AnnotationMatch,
    Context,
    ConversionConfig,
    ConversionReport,
    CropOutput,
    ImageRef,
    MatchOutput,
    Sample,
)
from xxtrain.pipeline.processors import (
    EncodeCropDetection,
    EncodeDetection,
    EncodeKnobSegment,
    EncodePointSegment,
    EncodePose,
    EncodeSegment,
)
from xxtrain.task import TaskType


class PipelineEncoderTest(unittest.TestCase):
    def make_context(self, labels=('dial',), task_type=TaskType.DETECT) -> Context:
        return Context(
            config=ConversionConfig(
                task_name=task_type.value,
                task_type=task_type,
                root_path=Path('.'),
                split=10,
                labels=LabelCatalog(tuple(labels)),
                reserve_no_label=True,
            ),
            report=ConversionReport(),
        )

    def make_sample(self, annotations, size=(100, 100)) -> Sample:
        return Sample(
            id='group/image',
            source_group='group',
            source_index=1,
            image=ImageRef(path=Path('image.jpg'), info=ImageInfo(width=size[0], height=size[1])),
            annotations=tuple(annotations),
        )

    def test_standard_detection_and_segment_encoders(self) -> None:
        context = self.make_context()
        detection = self.make_sample((Bbox(label='dial', x1=25, y1=25, x2=75, y2=75),))
        segment = self.make_sample((Polygon(label='dial', points=((10, 10), (90, 10), (50, 90))),))
        self.assertEqual(
            ('0 0.500000 0.500000 0.500000 0.500000',), EncodeDetection().transform(detection, context).lines
        )
        self.assertEqual(
            ('0 0.100000 0.100000 0.900000 0.100000 0.500000 0.900000',),
            EncodeSegment().transform(segment, context).lines,
        )
        self.assertEqual(ConversionReport(), context.report)

    def test_crop_detection_uses_crop_sample_info(self) -> None:
        context = self.make_context()
        sample = self.make_sample((Bbox(label='dial', x1=2.5, y1=2.5, x2=7.5, y2=7.5),), size=(10, 10))
        crop = CropOutput(sample=sample, parent=Bbox(label='parent', x1=0, y1=0, x2=10, y2=10))
        self.assertEqual(
            ('0 0.500000 0.500000 0.500000 0.500000',), EncodeCropDetection().transform(crop, context).lines
        )

    def test_point_segment_reproduces_legacy_triangle(self) -> None:
        context = self.make_context(('Point',), TaskType.SEGMENT)
        line = Polyline(label='1', points=((10, 10), (20, 30)))
        sample = self.make_sample((line,), size=(40, 40))
        crop = CropOutput(sample=sample, parent=Bbox(label='cc', x1=0, y1=0, x2=40, y2=40))
        output = EncodePointSegment().transform(crop, context)
        self.assertEqual(('0 0.182918 0.283541 0.317082 0.216459 0.500000 0.750000',), output.lines)
        self.assertEqual(((10.0, 10.0), (20.0, 30.0)), line.points)

    def test_knob_segment_clamps_normalized_coordinates(self) -> None:
        context = self.make_context(('switch',), TaskType.SEGMENT)
        polygon = Polygon(label='switch', points=((-1, 1), (11, 1), (5, 12)))
        sample = self.make_sample((polygon,), size=(10, 10))
        parent = Bbox(label='switch', x1=0, y1=0, x2=10, y2=10)
        output = EncodeKnobSegment().transform(CropOutput(sample=sample, parent=parent), context)
        self.assertEqual(('0 0.000000 0.100000 1.000000 0.100000 0.500000 1.000000',), output.lines)

    def test_pose_match_adapter_preserves_legacy_bytes_in_catalog_order(self) -> None:
        context = self.make_context(('start', 'end'), TaskType.POSE)
        sample = self.make_sample((), size=(100, 100))
        parent = Bbox(label='object', x1=10, y1=10, x2=90, y2=90)
        children = (Points(label='end', points=((80, 70),)), Points(label='start', points=((20, 30),)))
        item = MatchOutput(sample=sample, matches=(AnnotationMatch(parent=parent, children=children),))
        output = EncodePose().transform(item, context)
        self.assertEqual(
            ('0 0.500000 0.500000 0.800000 0.800000 0.200000 0.300000 2 0.800000 0.700000 2',), output.lines
        )


if __name__ == '__main__':
    unittest.main()
