import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
