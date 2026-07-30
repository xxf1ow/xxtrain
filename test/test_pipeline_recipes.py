import copy
import json
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test.support.scenarios import load_case_scenario
from test.test_conversion_baseline import FIXTURES_PATH, PROJECT_ROOT
from xxtrain.data import LabelCatalog
from xxtrain.pipeline import DatasetRecipe, convert_dataset, standard_recipe
from xxtrain.pipeline.annotation_io import ReadAnnotations, ValidateObb
from xxtrain.pipeline.core import Context, Pipeline, Sample
from xxtrain.pipeline.discovery import CocoSource, DirectorySource
from xxtrain.pipeline.processors import (
    EncodeDetection,
    EncodePose,
    EncodeSegment,
    FilterLabels,
    PrepareClassification,
    PrepareSegmentShapes,
    ReadImageInfo,
)
from xxtrain.pipeline.sinks import YoloDatasetSink
from xxtrain.task import TaskType


class CatalogOnlySink:
    def __init__(self, input_type: type):
        self.input_type = input_type

    def write(self, item, context: Context) -> None:
        raise AssertionError('catalog-only conversion must not write outputs')

    def finalize(self, context: Context) -> None:
        pass


class PipelineRecipeTest(unittest.TestCase):
    def copy_fixture(self, fixture_name: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory(prefix='xxtrain-recipe-')
        self.addCleanup(temp_dir.cleanup)
        root_path = Path(temp_dir.name) / fixture_name
        shutil.copytree(FIXTURES_PATH / fixture_name, root_path)
        return root_path

    def converted_labels(self, task_type: TaskType, root_path: Path) -> LabelCatalog:
        recipe = standard_recipe(task_type)
        recipe = replace(recipe, sink=CatalogOnlySink(recipe.sink.input_type))
        with (
            patch.object(DirectorySource, 'read', return_value=iter(())),
            patch('xxtrain.pipeline.workflow.print_conversion_report') as print_report,
        ):
            convert_dataset(recipe, root_path)
        (context,) = print_report.call_args.args
        return context.config.labels

    def test_pipeline_public_api_exports_stable_symbols(self) -> None:
        from xxtrain.pipeline import (
            Context,
            ConversionConfig,
            ConversionReport,
            DatasetRecipe,
            ExpandProcessor,
            ImageRef,
            ItemProcessor,
            Pipeline,
            Sample,
            convert_dataset,
            standard_recipe,
        )

        self.assertTrue(callable(convert_dataset))
        self.assertTrue(callable(standard_recipe))
        self.assertTrue(
            all(
                value is not None
                for value in (
                    Context,
                    ConversionConfig,
                    ConversionReport,
                    DatasetRecipe,
                    ExpandProcessor,
                    ImageRef,
                    ItemProcessor,
                    Pipeline,
                    Sample,
                )
            )
        )

    def test_standard_recipe_processor_sequences(self) -> None:
        expected = {
            TaskType.DETECT: (ReadImageInfo, ReadAnnotations, FilterLabels, EncodeDetection),
            TaskType.SEGMENT: (ReadImageInfo, ReadAnnotations, FilterLabels, PrepareSegmentShapes, EncodeSegment),
            TaskType.POSE: (ReadImageInfo, ReadAnnotations, FilterLabels, EncodePose),
            TaskType.OBB: (ReadImageInfo, ReadAnnotations, FilterLabels, ValidateObb, EncodeSegment),
            TaskType.CLASSIFY: (PrepareClassification,),
        }
        for task_type, processor_types in expected.items():
            with self.subTest(task_type=task_type):
                recipe = standard_recipe(task_type)
                self.assertEqual(processor_types, tuple(type(processor) for processor in recipe.pipeline.processors))
                self.assertIsNone(recipe.labels)

    def test_convert_dataset_uses_recipe_and_false_reserve_default(self) -> None:
        root_path = self.copy_fixture('standard-detect')
        recipe = standard_recipe(TaskType.DETECT)

        report = convert_dataset(recipe, root_path)

        self.assertEqual(2, report.train_image_count + report.val_image_count)
        self.assertTrue((root_path / 'detect' / 'dataset.yaml').is_file())

    def test_labelimg_extra_label_is_filtered_and_reported(self) -> None:
        root_path = self.copy_fixture('standard-detect')
        xml_path = root_path / 'src' / '20260620' / 'anns' / '0000.xml'
        tree = ET.parse(xml_path)
        original = tree.getroot().find('object')
        assert original is not None
        for label in ('历史标签', '1008'):
            extra = copy.deepcopy(original)
            name = extra.find('name')
            assert name is not None
            name.text = label
            tree.getroot().append(extra)
        tree.write(xml_path, encoding='utf-8')

        report = convert_dataset(standard_recipe(TaskType.DETECT), root_path)

        self.assertEqual(1, report.ignored_label_counts['历史标签'])
        self.assertEqual(1, report.ignored_label_counts['1008'])
        self.assertEqual(1, report.source_label_counts['历史标签'])
        self.assertEqual(1, report.source_label_counts['1008'])
        self.assertNotIn('历史标签', report.output_label_counts)
        self.assertNotIn('1008', report.output_label_counts)
        self.assertGreater(report.output_label_counts['cc'], 0)

    def test_labelme_extra_label_is_filtered_and_reported(self) -> None:
        root_path = self.copy_fixture('standard-segment')
        json_path = root_path / 'src' / '251010' / 'anns_seg' / '0000.json'
        payload = json.loads(json_path.read_text(encoding='utf-8'))
        extra = copy.deepcopy(payload['shapes'][0])
        extra['label'] = 'legacy'
        payload['shapes'].append(extra)
        json_path.write_text(json.dumps(payload), encoding='utf-8')

        report = convert_dataset(standard_recipe(TaskType.SEGMENT), root_path)

        self.assertEqual(1, report.ignored_label_counts['legacy'])
        self.assertNotIn('legacy', report.output_label_counts)
        self.assertGreater(report.output_label_counts['switch'], 0)

    def test_missing_final_target_label_fails_before_finalize(self) -> None:
        root_path = self.copy_fixture('standard-detect')
        (root_path / 'src' / 'labels.txt').write_text('cc\nmissing\n', encoding='utf-8')

        with self.assertRaisesRegex(ValueError, 'Converted dataset has no output samples for target labels: missing'):
            convert_dataset(standard_recipe(TaskType.DETECT), root_path)

        self.assertFalse((root_path / 'detect' / 'dataset.yaml').exists())

    def test_pose_coverage_requires_object_label_not_keypoint_names(self) -> None:
        root_path = self.copy_fixture('standard-pose')

        report = convert_dataset(standard_recipe(TaskType.POSE), root_path)

        object_label = (root_path / 'src' / 'labels.txt').read_text(encoding='utf-8').splitlines()[0]
        self.assertGreater(report.output_label_counts[object_label], 0)
        self.assertEqual((), report.missing_output_labels)

    def test_special_recipe_validates_projected_output_label(self) -> None:
        root_path = self.copy_fixture('point')
        scenario = load_case_scenario('point-detect')

        report = convert_dataset(scenario.dataset, root_path)

        self.assertEqual({'Point'}, set(report.output_label_counts))
        self.assertGreater(report.output_label_counts['Point'], 0)
        self.assertTrue({'tl', 'tc', 'cl', 'cc'} & set(report.source_label_counts))

    def test_pipeline_package_does_not_import_legacy_modules(self) -> None:
        pipeline_root = PROJECT_ROOT / 'src' / 'xxtrain' / 'pipeline'
        source = '\n'.join(path.read_text(encoding='utf-8') for path in pipeline_root.glob('*.py'))
        for legacy_name in ('annparser', 'annprocessor', 'annconverter'):
            self.assertNotIn(legacy_name, source)

    def test_pipeline_recipes_has_no_task_name_registry(self) -> None:
        from xxtrain.pipeline import recipes

        for name in ('Recipe', '_CUSTOM_TASK_TYPES', 'build_recipe'):
            with self.subTest(name=name):
                self.assertFalse(hasattr(recipes, name))

    def test_detection_segmentation_and_pose_preserve_label_order(self) -> None:
        fixtures = {'detect': 'standard-detect', 'segment': 'standard-segment', 'pose': 'standard-pose'}
        for task_name, fixture_name in fixtures.items():
            with self.subTest(task_name=task_name):
                root_path = self.copy_fixture(fixture_name)
                expected = tuple(
                    line.strip() for line in (root_path / 'src' / 'labels.txt').read_text(encoding='utf-8').splitlines()
                )
                labels = self.converted_labels(TaskType(task_name), root_path)
                self.assertEqual(expected, labels.names)

    def test_non_classification_recipes_preserve_legacy_empty_and_duplicate_labels(self) -> None:
        fixtures = {'detect': 'standard-detect', 'segment': 'standard-segment', 'pose': 'standard-pose'}
        for task_name, fixture_name in fixtures.items():
            with self.subTest(task_name=task_name):
                root_path = self.copy_fixture(fixture_name)
                labels_path = root_path / 'src' / 'labels.txt'
                original_label = labels_path.read_text(encoding='utf-8').splitlines()[0]
                expected = ('unused', '', original_label, original_label)
                labels_path.write_text('\n'.join(expected), encoding='utf-8')

                try:
                    labels = self.converted_labels(TaskType(task_name), root_path)
                except ValueError as error:
                    self.fail(f'legacy labels were rejected: {error}')

                self.assertEqual(expected, labels.names)

    def test_classification_labels_are_sorted(self) -> None:
        root_path = self.copy_fixture('standard-classify')
        (root_path / 'src' / 'labels.txt').write_text('Uab\nIA\nP\n', encoding='utf-8')

        labels = self.converted_labels(TaskType.CLASSIFY, root_path)

        self.assertEqual(('IA', 'P', 'Uab'), labels.names)

    def test_recipe_validates_fixed_source_and_sink_boundaries(self) -> None:
        with self.assertRaisesRegex(TypeError, 'PrepareClassification produces ClassifyOutput'):
            DatasetRecipe(
                name='classify',
                task_type=TaskType.CLASSIFY,
                labels=LabelCatalog(('label',)),
                pipeline=Pipeline((PrepareClassification(),)),
                sink=YoloDatasetSink(),
            )

    def test_convert_dataset_uses_coco_source_catalog_when_recipe_labels_are_none(self) -> None:
        class RecordingSampleSink:
            input_type = Sample

            def __init__(self) -> None:
                self.context: Context | None = None

            def write(self, item: Sample, context: Context) -> None:
                self.context = context

            def finalize(self, context: Context) -> None:
                self.context = context

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = root / 'annotations.json'
            json_path.write_text(
                '{"categories":[{"id":1,"name":"dial"}],"images":[],"annotations":[]}', encoding='utf-8'
            )
            sink = RecordingSampleSink()
            recipe = DatasetRecipe(
                name='coco-detect',
                task_type=TaskType.DETECT,
                labels=None,
                source=CocoSource(json_path),
                pipeline=Pipeline(()),
                sink=sink,
            )

            convert_dataset(recipe, root)

            self.assertIsNotNone(sink.context)
            self.assertEqual(('dial',), sink.context.config.labels.names)

    def test_convert_dataset_preserves_explicit_classification_labels_for_group_sources(self) -> None:
        class RecordingSampleSink:
            input_type = Sample

            def __init__(self) -> None:
                self.context: Context | None = None

            def write(self, item: Sample, context: Context) -> None:
                self.context = context

            def finalize(self, context: Context) -> None:
                self.context = context

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / 'src' / '20260620' / 'imgs'
            images.mkdir(parents=True)
            (images / 'sample.jpg').write_bytes(b'image')
            labels = LabelCatalog(('cc', 'cl', 'tc', 'tl'))
            sink = RecordingSampleSink()
            recipe = DatasetRecipe(
                name='point-classify',
                task_type=TaskType.CLASSIFY,
                labels=labels,
                source=DirectorySource(),
                pipeline=Pipeline(()),
                sink=sink,
            )

            convert_dataset(recipe, root)

            self.assertIsNotNone(sink.context)
            self.assertIs(labels, sink.context.config.labels)


if __name__ == '__main__':
    unittest.main()
