import json
import tempfile
import unittest
from pathlib import Path

from xxtrain.data import Bbox, ImageInfo, LabelCatalog, Polygon, Pose
from xxtrain.pipeline.annotation_io import ExtractSample, ReadAnnotations, ValidateObb
from xxtrain.pipeline.core import Context, ConversionConfig, ConversionReport, CropOutput, ImageRef, Sample
from xxtrain.task import TaskType


class ReadAnnotationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix='xxtrain-annotation-io-')
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name) / 'src' / 'group'
        self.image_path = self.root / 'imgs' / 'image.jpg'
        self.image_path.parent.mkdir(parents=True)
        self.image_path.touch()
        self.sample = Sample(
            id='group/image',
            source_group='group',
            source_index=0,
            image=ImageRef(path=self.image_path, info=ImageInfo(width=100, height=100)),
        )

    def context(self, task_type: TaskType, labels: tuple[str, ...]) -> Context:
        return Context(
            config=ConversionConfig(
                task_name=task_type.value,
                task_type=task_type,
                root_path=self.root.parents[1],
                split=10,
                labels=LabelCatalog(labels),
                reserve_no_label=False,
            ),
            report=ConversionReport(),
        )

    def path(self, directory: str, suffix: str) -> Path:
        path = self.root / directory / f'image{suffix}'
        path.parent.mkdir(exist_ok=True)
        return path

    def write_xml(self, *, label: str = 'object') -> None:
        self.path('anns', '.xml').write_text(
            f'<annotation><size><width>100</width><height>100</height></size>'
            f'<object><name>{label}</name><bndbox><xmin>10</xmin><ymin>10</ymin>'
            f'<xmax>90</xmax><ymax>90</ymax></bndbox></object></annotation>',
            encoding='utf-8',
        )

    def write_json(self, shapes: list[dict[str, object]]) -> None:
        self.path('anns_seg', '.json').write_text(
            json.dumps({'imageWidth': 100, 'imageHeight': 100, 'shapes': shapes}), encoding='utf-8'
        )

    def shape(
        self, label: str, shape_type: str, points: list[list[float]], group: int | None = None
    ) -> dict[str, object]:
        return {'label': label, 'shape_type': shape_type, 'points': points, 'group_id': group}

    def test_detect_rejects_two_nonempty_formats(self) -> None:
        self.write_xml()
        self.path('labels', '.txt').write_text('0 0.5 0.5 0.8 0.8\n', encoding='utf-8')

        with self.assertRaisesRegex(ValueError, 'multiple non-empty annotation formats'):
            ReadAnnotations().transform(self.sample, self.context(TaskType.DETECT, ('object',)))

    def test_empty_yolo_is_empty_but_empty_xml_and_json_are_invalid(self) -> None:
        context = self.context(TaskType.DETECT, ('object',))
        self.path('labels', '.txt').write_text(' \n', encoding='utf-8')
        self.assertEqual((), ReadAnnotations().transform(self.sample, context).annotations)

        self.path('anns', '.xml').write_bytes(b'')
        with self.assertRaisesRegex(Exception, 'Failed to parse annotation'):
            ReadAnnotations().transform(self.sample, context)

        self.path('anns', '.xml').unlink()
        self.path('anns_seg', '.json').write_bytes(b'')
        with self.assertRaisesRegex(Exception, 'Failed to parse annotation'):
            ReadAnnotations().transform(self.sample, context)

    def test_missing_files_and_empty_xml_json_are_empty(self) -> None:
        context = self.context(TaskType.DETECT, ('object',))
        self.assertEqual((), ReadAnnotations().transform(self.sample, context).annotations)

        self.path('anns', '.xml').write_text(
            '<annotation><size><width>100</width><height>100</height></size></annotation>', encoding='utf-8'
        )
        self.assertEqual((), ReadAnnotations().transform(self.sample, context).annotations)

        self.path('anns', '.xml').unlink()
        self.write_json([])
        self.assertEqual((), ReadAnnotations().transform(self.sample, context).annotations)

    def test_pose_assembles_labelimg_box_and_labelme_points(self) -> None:
        self.write_xml(label='person')
        self.write_json(
            [
                self.shape('wrist', 'point', [[80, 70]]),
                self.shape('nose', 'point', [[20, 30]]),
            ]
        )

        output = ReadAnnotations().transform(self.sample, self.context(TaskType.POSE, ('person', 'nose', 'wrist')))

        self.assertEqual((Pose,), tuple(type(annotation) for annotation in output.annotations))
        self.assertEqual('person', output.annotations[0].label)
        self.assertEqual(('nose', 'wrist'), tuple(point.label for point in output.annotations[0].keypoints))

    def test_pose_rejects_complete_pose_plus_other_nonempty_source(self) -> None:
        self.path('labels', '.txt').write_text('0 0.5 0.5 0.8 0.8 0.2 0.3 2 0.8 0.7 2\n', encoding='utf-8')
        self.write_xml(label='person')

        with self.assertRaisesRegex(ValueError, 'duplicate pose representations'):
            ReadAnnotations().transform(self.sample, self.context(TaskType.POSE, ('person', 'nose', 'wrist')))

    def test_pose_rejects_missing_duplicate_unknown_and_multiple_boxes(self) -> None:
        context = self.context(TaskType.POSE, ('person', 'nose', 'wrist'))
        cases = (
            [self.shape('nose', 'point', [[20, 30]])],
            [self.shape('nose', 'point', [[20, 30]]), self.shape('nose', 'point', [[30, 40]])],
            [self.shape('nose', 'point', [[20, 30]]), self.shape('unknown', 'point', [[80, 70]])],
        )
        for shapes in cases:
            with self.subTest(shapes=shapes):
                self.write_xml(label='person')
                self.write_json(shapes)
                with self.assertRaises(ValueError):
                    ReadAnnotations().transform(self.sample, context)
                self.path('anns', '.xml').unlink()

        self.write_json(
            [
                self.shape('person', 'rectangle', [[10, 10], [90, 90]]),
                self.shape('person', 'rectangle', [[15, 15], [85, 85]]),
                self.shape('nose', 'point', [[20, 30]]),
                self.shape('wrist', 'point', [[80, 70]]),
            ]
        )
        with self.assertRaises(ValueError):
            ReadAnnotations().transform(self.sample, context)

    def test_segment_rejects_labelme_line_at_pipeline_boundary(self) -> None:
        self.write_json([self.shape('object', 'line', [[10, 10], [90, 90]])])

        with self.assertRaisesRegex(TypeError, 'Polyline'):
            ReadAnnotations().transform(self.sample, self.context(TaskType.SEGMENT, ('object',)))

    def test_obb_validates_input_polygon(self) -> None:
        self.write_json([self.shape('object', 'polygon', [[10, 10], [90, 10], [50, 90]])])

        with self.assertRaisesRegex(ValueError, 'OBB'):
            ReadAnnotations().transform(self.sample, self.context(TaskType.OBB, ('object',)))

    def test_pose_normalizes_complete_labelme_keypoint_order(self) -> None:
        self.write_json(
            [
                self.shape('person', 'rectangle', [[10, 10], [90, 90]], group=7),
                self.shape('wrist', 'point', [[80, 70]], group=7),
                self.shape('nose', 'point', [[20, 30]], group=7),
            ]
        )

        output = ReadAnnotations().transform(self.sample, self.context(TaskType.POSE, ('person', 'nose', 'wrist')))

        self.assertEqual(('nose', 'wrist'), tuple(point.label for point in output.annotations[0].keypoints))

    def test_validate_obb_and_extract_sample_interfaces(self) -> None:
        invalid = self.sample.wrap(
            annotations=(Polygon(label='object', points=((10, 10), (90, 10), (50, 90))),)
        )
        context = self.context(TaskType.OBB, ('object',))
        with self.assertRaisesRegex(ValueError, 'OBB'):
            ValidateObb().transform(invalid, context)

        crop = CropOutput(sample=self.sample, parent=Bbox(label='object', x1=0, y1=0, x2=100, y2=100))
        self.assertIs(self.sample, ExtractSample().transform(crop, context))


if __name__ == '__main__':
    unittest.main()
