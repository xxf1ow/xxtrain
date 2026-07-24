import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test.test_conversion_baseline import FIXTURES_PATH, PROJECT_ROOT
from xxtrain.data import LabelCatalog
from xxtrain.pipeline import DatasetRecipe, convert_dataset, standard_recipe
from xxtrain.pipeline.core import Pipeline
from xxtrain.pipeline.discovery import DirectorySource
from xxtrain.pipeline.processors import (
    CropDetectionBoxes,
    CropMatches,
    EncodeCropDetection,
    EncodeDetection,
    EncodeKnobSegment,
    EncodePointSegment,
    EncodePose,
    EncodeSegment,
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
from xxtrain.pipeline.recipes import build_recipe
from xxtrain.pipeline.sinks import ClassificationDatasetSink, YoloDatasetSink
from xxtrain.task import TaskType

EXPECTED_CUSTOM_PROCESSORS = {
    'point-detect': (ReadImageInfo, ReadLabelImg, FilterLabels, RelabelAnnotations, EncodeDetection),
    'point-classify': (ReadImageInfo, ReadLabelImg, FilterLabels, CropDetectionBoxes),
    'point-segment': (
        ReadImageInfo,
        ReadMatchingAnnotations,
        FilterMatchingAnnotations,
        PrepareMatchChildren,
        MatchAnnotations,
        CropMatches,
        EncodePointSegment,
    ),
    'knob-detect': (ReadImageInfo, ReadLabelImg, FilterLabels, EncodeDetection),
    'knob-segment': (
        ReadImageInfo,
        ReadMatchingAnnotations,
        FilterMatchingAnnotations,
        PrepareMatchChildren,
        MatchAnnotations,
        CropMatches,
        EncodeKnobSegment,
    ),
    'light1-detect': (ReadImageInfo, ReadLabelImg, FilterLabels, EncodeDetection),
    'light2-detect': (
        ReadImageInfo,
        ReadLabelImg,
        PartitionAnnotations,
        MatchAnnotations,
        CropMatches,
        RelabelCropAnnotations,
        EncodeCropDetection,
    ),
}

EXPECTED_CUSTOM_LABELS = {
    'point-detect': ('Point',),
    'point-classify': ('tl', 'tc', 'cl', 'cc'),
    'point-segment': ('Point',),
    'knob-detect': ('switch',),
    'knob-segment': ('switch',),
    'light1-detect': ('1008',),
    'light2-detect': ('0',),
}

CUSTOM_FIXTURES = {
    'point-detect': 'point',
    'point-classify': 'point',
    'point-segment': 'point',
    'knob-detect': 'knob',
    'knob-segment': 'knob',
    'light1-detect': 'light',
    'light2-detect': 'light',
}


class PipelineRecipeTest(unittest.TestCase):
    def copy_fixture(self, fixture_name: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory(prefix='xxtrain-recipe-')
        self.addCleanup(temp_dir.cleanup)
        root_path = Path(temp_dir.name) / fixture_name
        shutil.copytree(FIXTURES_PATH / fixture_name, root_path)
        return root_path

    def converted_labels(self, task_type: TaskType, root_path: Path) -> LabelCatalog:
        with (
            patch.object(DirectorySource, 'read', return_value=iter(())),
            patch('xxtrain.pipeline.workflow.print_conversion_report') as print_report,
        ):
            convert_dataset(standard_recipe(task_type), root_path)
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
        self.assertTrue(all(value is not None for value in (
            Context,
            ConversionConfig,
            ConversionReport,
            DatasetRecipe,
            ExpandProcessor,
            ImageRef,
            ItemProcessor,
            Pipeline,
            Sample,
        )))

    def test_standard_recipe_processor_sequences(self) -> None:
        expected = {
            TaskType.DETECT: (ReadImageInfo, ReadLabelImg, FilterLabels, EncodeDetection),
            TaskType.SEGMENT: (
                ReadImageInfo,
                ReadLabelMe,
                FilterLabels,
                PrepareSegmentShapes,
                EncodeSegment,
            ),
            TaskType.POSE: (
                ReadImageInfo,
                ReadMatchingAnnotations,
                FilterMatchingAnnotations,
                PrepareMatchChildren,
                MatchAnnotations,
                EncodePose,
            ),
            TaskType.CLASSIFY: (PrepareClassification,),
        }
        for task_type, processor_types in expected.items():
            with self.subTest(task_type=task_type):
                recipe = standard_recipe(task_type)
                self.assertEqual(
                    processor_types,
                    tuple(type(processor) for processor in recipe.pipeline.processors),
                )
                self.assertIsNone(recipe.labels)

    def test_convert_dataset_uses_recipe_and_false_reserve_default(self) -> None:
        root_path = self.copy_fixture('standard-detect')
        recipe = standard_recipe(TaskType.DETECT)

        report = convert_dataset(recipe, root_path)

        self.assertEqual(2, report.train_image_count + report.val_image_count)
        self.assertTrue((root_path / 'detect' / 'dataset.yaml').is_file())

    def test_custom_recipe_processor_sequences_and_label_catalogs(self) -> None:
        for task_name, expected_types in EXPECTED_CUSTOM_PROCESSORS.items():
            with self.subTest(task_name=task_name):
                recipe, context = build_recipe(task_name, self.copy_fixture(CUSTOM_FIXTURES[task_name]))
                self.assertEqual(expected_types, tuple(type(value) for value in recipe.pipeline.processors))
                self.assertEqual(EXPECTED_CUSTOM_LABELS[task_name], recipe.labels.names)
                self.assertIs(recipe.labels, context.config.labels)

    def test_custom_recipe_processor_configuration(self) -> None:
        for task_name in EXPECTED_CUSTOM_PROCESSORS:
            with self.subTest(task_name=task_name):
                recipe, _ = build_recipe(task_name, self.copy_fixture(CUSTOM_FIXTURES[task_name]))
                filter_labels_processors = tuple(
                    processor
                    for processor in recipe.pipeline.processors
                    if isinstance(processor, FilterLabels)
                )
                if not filter_labels_processors:
                    continue
                (filter_labels,) = filter_labels_processors
                expected_input_labels = (
                    ('tl', 'tc', 'cl', 'cc')
                    if task_name.startswith('point-')
                    else EXPECTED_CUSTOM_LABELS[task_name]
                )
                self.assertEqual(expected_input_labels, filter_labels.labels)
                self.assertEqual(task_name != 'light1-detect', filter_labels.strict)

                if task_name == 'point-detect':
                    relabel = next(
                        processor
                        for processor in recipe.pipeline.processors
                        if isinstance(processor, RelabelAnnotations)
                    )
                    self.assertEqual('Point', relabel.label)

    def test_matching_crop_recipe_processor_configuration(self) -> None:
        expected = {
            'point-segment': (('tl', 'tc', 'cl', 'cc'), ('1',), 0),
            'knob-segment': (('switch',), ('switch',), 0.15),
        }
        for task_name, (parent_labels, child_labels, wide) in expected.items():
            with self.subTest(task_name=task_name):
                recipe, _ = build_recipe(task_name, self.copy_fixture(CUSTOM_FIXTURES[task_name]))
                matching_filter = next(
                    processor
                    for processor in recipe.pipeline.processors
                    if isinstance(processor, FilterMatchingAnnotations)
                )
                prepare = next(
                    processor
                    for processor in recipe.pipeline.processors
                    if isinstance(processor, PrepareMatchChildren)
                )
                matcher = next(
                    processor
                    for processor in recipe.pipeline.processors
                    if isinstance(processor, MatchAnnotations)
                )
                self.assertEqual(parent_labels, matching_filter.parent_labels)
                self.assertEqual(child_labels, matching_filter.child_labels)
                self.assertTrue(matching_filter.strict)
                self.assertIs(TaskType.SEGMENT, prepare.task_type)
                self.assertEqual(wide, matcher.wide)
                self.assertFalse(matcher.strict)
                self.assertIsInstance(recipe.sink, YoloDatasetSink)

    def test_light2_detection_partitions_one_labelimg_read(self) -> None:
        recipe, _ = build_recipe('light2-detect', self.copy_fixture('light'))

        processors = recipe.pipeline.processors
        self.assertEqual(1, sum(isinstance(value, ReadLabelImg) for value in processors))
        partition = next(
            processor for processor in processors if isinstance(processor, PartitionAnnotations)
        )
        matcher = next(processor for processor in processors if isinstance(processor, MatchAnnotations))
        relabel = next(
            processor for processor in processors if isinstance(processor, RelabelCropAnnotations)
        )
        self.assertEqual({'1008'}, partition.parent_labels)
        self.assertEqual({'0', '1', '2'}, partition.child_labels)
        self.assertEqual(0.1, matcher.wide)
        self.assertFalse(matcher.strict)
        self.assertEqual('0', relabel.label)
        self.assertIsInstance(recipe.sink, YoloDatasetSink)

    def test_pipeline_package_does_not_import_legacy_modules(self) -> None:
        pipeline_root = PROJECT_ROOT / 'src' / 'xxtrain' / 'pipeline'
        source = '\n'.join(path.read_text(encoding='utf-8') for path in pipeline_root.glob('*.py'))
        for legacy_name in ('annparser', 'annprocessor', 'annconverter'):
            self.assertNotIn(legacy_name, source)

    def test_point_classification_uses_indexed_class_directories(self) -> None:
        recipe, _ = build_recipe('point-classify', self.copy_fixture('point'))

        self.assertIsInstance(recipe.sink, ClassificationDatasetSink)
        self.assertTrue(recipe.sink.indexed_class_directories)

    def test_detection_segmentation_and_pose_preserve_label_order(self) -> None:
        fixtures = {
            'detect': 'standard-detect',
            'segment': 'standard-segment',
            'pose': 'standard-pose',
        }
        for task_name, fixture_name in fixtures.items():
            with self.subTest(task_name=task_name):
                root_path = self.copy_fixture(fixture_name)
                expected = tuple(
                    line.strip()
                    for line in (root_path / 'src' / 'labels.txt').read_text(encoding='utf-8').splitlines()
                )
                labels = self.converted_labels(TaskType(task_name), root_path)
                self.assertEqual(expected, labels.names)

    def test_non_classification_recipes_preserve_legacy_empty_and_duplicate_labels(self) -> None:
        fixtures = {
            'detect': 'standard-detect',
            'segment': 'standard-segment',
            'pose': 'standard-pose',
        }
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

    def test_standard_recipe_rejects_unsupported_basic_task(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Unsupported standard task type: obb'):
            standard_recipe(TaskType.OBB)


if __name__ == '__main__':
    unittest.main()
