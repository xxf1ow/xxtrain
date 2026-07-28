import json
import shutil
import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO

from test.support.annotation_projection import project_annotations
from xxtrain.data import LabelCatalog, Pose
from xxtrain.data.formats import decode_pose, decode_segment, read_labelimg, read_labelme
from xxtrain.pipeline import CocoSink, CocoSource, LabelImgSink, LabelMeSink, ReadAnnotations
from xxtrain.pipeline.core import Context, ConversionConfig, ConversionReport, Pipeline
from xxtrain.pipeline.discovery import DirectorySource
from xxtrain.pipeline.processors import EncodePose, EncodeSegment, ReadImageInfo
from xxtrain.pipeline.sinks import YoloDatasetSink
from xxtrain.task import TaskType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_PATH = PROJECT_ROOT / 'test' / 'fixtures'


def normalized_json(path: Path) -> str:
    return json.dumps(json.loads(path.read_text(encoding='utf-8')), ensure_ascii=False, sort_keys=True)


def normalized_xml(path: Path) -> str:
    return ET.canonicalize(xml_data=path.read_text(encoding='utf-8'))


class DatasetIoRoundTripTest(unittest.TestCase):
    def copy_fixture(self, fixture_name: str, root: Path) -> Path:
        target = root / fixture_name
        shutil.copytree(FIXTURES_PATH / fixture_name, target)
        return target

    def context(self, root: Path, task_name: str, task_type: TaskType, labels: tuple[str, ...]) -> Context:
        return Context(
            config=ConversionConfig(
                task_name=task_name,
                task_type=task_type,
                root_path=root,
                split=2,
                labels=LabelCatalog(labels),
                reserve_no_label=True,
            ),
            report=ConversionReport(),
        )

    def labels_for(self, root: Path) -> tuple[str, ...]:
        return tuple((root / 'src' / 'labels.txt').read_text(encoding='utf-8').splitlines())

    def read_directory(self, context: Context):
        return tuple(Pipeline((ReadImageInfo(), ReadAnnotations())).run(DirectorySource().read(context), context))

    def write(self, sink, items, context: Context) -> None:
        for item in items:
            sink.write(item, context)
        sink.finalize(context)

    def assert_projects_equal(self, expected, actual) -> None:
        self.assertEqual(
            tuple(project_annotations(sample.annotations, sample.image.require_info()) for sample in expected),
            tuple(project_annotations(sample.annotations, sample.image.require_info()) for sample in actual),
        )

    def assert_coco(self, path: Path, *, images: int, annotations: int, categories: int, mask: bool = False) -> None:
        coco = COCO(str(path))
        self.assertEqual(images, len(coco.imgs))
        self.assertEqual(annotations, len(coco.anns))
        self.assertEqual(categories, len(coco.cats))
        if mask:
            annotation = next(iter(coco.anns.values()))
            image = coco.imgs[annotation['image_id']]
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message="__array__ implementation doesn't accept a copy keyword",
                    category=DeprecationWarning,
                    module='pycocotools.mask',
                )
                decoded = coco.annToMask(annotation)
            self.assertEqual((image['height'], image['width']), decoded.shape)
            self.assertGreater(int(decoded.sum()), 0)

    def test_detect_pipeline_coco_and_labelimg_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix='xxtrain-io-detect-') as temp_dir:
            root = self.copy_fixture('standard-detect', Path(temp_dir))
            labels = self.labels_for(root)
            source_context = self.context(root, 'source', TaskType.DETECT, labels)
            samples = self.read_directory(source_context)
            output_root = root.parent / 'first'
            context = self.context(output_root, 'detect', TaskType.DETECT, labels)

            coco_sink = CocoSink()
            self.write(coco_sink, samples, context)
            coco_path = output_root / 'detect' / 'annotations.json'
            self.assert_coco(coco_path, images=2, annotations=2, categories=1)
            reread = tuple(CocoSource(coco_path).read(context))
            self.assert_projects_equal(samples, reread)

            second = root.parent / 'second'
            second_context = self.context(second, 'detect', TaskType.DETECT, labels)
            self.write(CocoSink(), reread, second_context)
            self.assertEqual(normalized_json(coco_path), normalized_json(second / 'detect' / 'annotations.json'))

            labelme_context = self.context(root.parent / 'labelimg-one', 'detect', TaskType.DETECT, labels)
            self.write(LabelImgSink(), samples, labelme_context)
            labelme_again_context = self.context(root.parent / 'labelimg-two', 'detect', TaskType.DETECT, labels)
            self.write(LabelImgSink(), samples, labelme_again_context)
            for sample in samples:
                name = Path(sample.id).name
                first = labelme_context.config.root_path / 'detect' / sample.source_group / f'{name}.xml'
                second = labelme_again_context.config.root_path / 'detect' / sample.source_group / f'{name}.xml'
                self.assertEqual(normalized_xml(first), normalized_xml(second))
                self.assertEqual(
                    project_annotations(sample.annotations, sample.image.require_info()),
                    project_annotations(read_labelimg(first, sample.image.require_info()), sample.image.require_info()),
                )

    def test_segment_pipeline_yolo_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix='xxtrain-io-segment-') as temp_dir:
            root = self.copy_fixture('standard-segment', Path(temp_dir))
            labels = self.labels_for(root)
            context = self.context(root, 'segment', TaskType.SEGMENT, labels)
            samples = self.read_directory(context)

            coco_context = self.context(root, 'segment-coco', TaskType.SEGMENT, labels)
            self.write(CocoSink(), samples, coco_context)
            coco_path = root / 'segment-coco' / 'annotations.json'
            self.assert_coco(coco_path, images=2, annotations=6, categories=1, mask=True)
            self.assert_projects_equal(samples, tuple(CocoSource(coco_path).read(coco_context)))

            encoded = tuple(Pipeline((EncodeSegment(),)).run(samples, context))
            self.write(YoloDatasetSink(), encoded, context)

            reread = tuple(
                sample.wrap(
                    annotations=tuple(
                        decode_segment(line, sample.image.require_info(), context.config.labels)
                        for line in (root / 'segment' / f'{sample.id}.txt').read_text(encoding='utf-8').splitlines()
                    )
                )
                for sample in samples
            )
            self.assert_projects_equal(samples, reread)
            second = root / 'second'
            second_context = self.context(second, 'segment', TaskType.SEGMENT, labels)
            reencoded = tuple(Pipeline((EncodeSegment(),)).run(reread, second_context))
            self.write(YoloDatasetSink(), reencoded, second_context)
            for sample in samples:
                self.assertEqual(
                    (root / 'segment' / f'{sample.id}.txt').read_text(encoding='utf-8'),
                    (second / 'segment' / f'{sample.id}.txt').read_text(encoding='utf-8'),
                )

    def test_pose_pipeline_assembles_pose_and_round_trips_sinks(self) -> None:
        with tempfile.TemporaryDirectory(prefix='xxtrain-io-pose-') as temp_dir:
            root = self.copy_fixture('standard-pose', Path(temp_dir))
            labels = self.labels_for(root)
            context = self.context(root, 'pose-coco', TaskType.POSE, labels)
            samples = self.read_directory(context)
            self.assertTrue(
                all(isinstance(annotation, Pose) for sample in samples for annotation in sample.annotations)
            )

            self.write(CocoSink(), samples, context)
            coco_path = root / 'pose-coco' / 'annotations.json'
            self.assert_coco(coco_path, images=2, annotations=2, categories=1)
            self.assert_projects_equal(samples, tuple(CocoSource(coco_path).read(context)))

            labelme_context = self.context(root / 'labelme', 'pose', TaskType.POSE, labels)
            self.write(LabelMeSink(), samples, labelme_context)
            for sample in samples:
                path = labelme_context.config.root_path / 'pose' / sample.source_group / f'{Path(sample.id).name}.json'
                self.assertEqual(
                    project_annotations(sample.annotations, sample.image.require_info()),
                    project_annotations(read_labelme(path, sample.image.require_info()), sample.image.require_info()),
                )

            yolo_context = self.context(root / 'yolo', 'pose', TaskType.POSE, labels)
            encoded = tuple(Pipeline((EncodePose(),)).run(samples, yolo_context))
            self.write(YoloDatasetSink(), encoded, yolo_context)
            yolo_reread = tuple(
                sample.wrap(
                    annotations=tuple(
                        decode_pose(line, sample.image.require_info(), yolo_context.config.labels)
                        for line in (yolo_context.config.root_path / 'pose' / f'{sample.id}.txt')
                        .read_text(encoding='utf-8')
                        .splitlines()
                    )
                )
                for sample in samples
            )
            yolo_expected = tuple(
                sample.wrap(
                    annotations=tuple(
                        decode_pose(line, sample.image.require_info(), yolo_context.config.labels)
                        for line in item.lines
                    )
                )
                for sample, item in zip(samples, encoded, strict=True)
            )
            self.assert_projects_equal(yolo_expected, yolo_reread)

    def test_obb_pipeline_round_trips_labelme_yolo_and_coco(self) -> None:
        with tempfile.TemporaryDirectory(prefix='xxtrain-io-obb-') as temp_dir:
            root = self.copy_fixture('standard-obb', Path(temp_dir))
            labels = self.labels_for(root)
            context = self.context(root, 'obb-coco', TaskType.OBB, labels)
            samples = self.read_directory(context)

            self.write(CocoSink(), samples, context)
            coco_path = root / 'obb-coco' / 'annotations.json'
            self.assert_coco(coco_path, images=1, annotations=1, categories=1, mask=True)
            self.assert_projects_equal(samples, tuple(CocoSource(coco_path).read(context)))

            labelme_context = self.context(root / 'labelme', 'obb', TaskType.OBB, labels)
            self.write(LabelMeSink(), samples, labelme_context)
            yolo_context = self.context(root / 'yolo', 'obb', TaskType.OBB, labels)
            encoded = tuple(Pipeline((EncodeSegment(),)).run(samples, yolo_context))
            self.write(YoloDatasetSink(), encoded, yolo_context)
            for sample in samples:
                labelme_path = (
                    labelme_context.config.root_path / 'obb' / sample.source_group / f'{Path(sample.id).name}.json'
                )
                self.assertEqual(
                    project_annotations(sample.annotations, sample.image.require_info()),
                    project_annotations(
                        read_labelme(labelme_path, sample.image.require_info()), sample.image.require_info()
                    ),
                )
                yolo_path = yolo_context.config.root_path / 'obb' / f'{sample.id}.txt'
                yolo_annotations = tuple(
                    decode_segment(line, sample.image.require_info(), yolo_context.config.labels)
                    for line in yolo_path.read_text(encoding='utf-8').splitlines()
                )
                self.assertEqual(
                    project_annotations(sample.annotations, sample.image.require_info()),
                    project_annotations(yolo_annotations, sample.image.require_info()),
                )

    def test_literal_coco_source_writes_labelme(self) -> None:
        with tempfile.TemporaryDirectory(prefix='xxtrain-io-literal-coco-') as temp_dir:
            root = Path(temp_dir)
            image_path = FIXTURES_PATH / 'standard-segment' / 'src' / '251010' / 'imgs' / '0000.jpg'
            with Image.open(image_path) as image:
                width, height = image.size
            payload = {
                'categories': [{'id': 1, 'name': 'switch'}],
                'images': [{'id': 1, 'file_name': str(image_path.resolve()), 'width': width, 'height': height}],
                'annotations': [
                    {
                        'id': 1,
                        'image_id': 1,
                        'category_id': 1,
                        'bbox': [100, 100, 300, 200],
                        'segmentation': [[100, 100, 400, 100, 400, 300, 100, 300]],
                        'area': 60000,
                        'iscrowd': 0,
                    }
                ],
            }
            coco_path = root / 'literal.json'
            coco_path.write_text(json.dumps(payload), encoding='utf-8')
            context = self.context(root, 'literal', TaskType.SEGMENT, ('switch',))
            samples = tuple(CocoSource(coco_path).read(context))
            self.write(LabelMeSink(), samples, context)
            sample = samples[0]
            output = root / 'literal' / 'coco' / '000000.json'
            self.assertEqual(
                project_annotations(sample.annotations, sample.image.require_info()),
                project_annotations(read_labelme(output, sample.image.require_info()), sample.image.require_info()),
            )


if __name__ == '__main__':
    unittest.main()
