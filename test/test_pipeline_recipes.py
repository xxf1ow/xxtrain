import shutil
import tempfile
import unittest
from pathlib import Path

from test.test_conversion_baseline import FIXTURES_PATH
from xxtrain.data import LabelCatalog
from xxtrain.pipeline.core import Pipeline
from xxtrain.pipeline.discovery import DirectorySource
from xxtrain.pipeline.processors import (
    EncodeDetection,
    EncodePose,
    EncodeSegment,
    FilterLabels,
    FilterMatchingAnnotations,
    MatchAnnotations,
    PrepareClassification,
    PrepareMatchChildren,
    PrepareSegmentShapes,
    ReadImageInfo,
    ReadLabelImg,
    ReadLabelMe,
    ReadMatchingAnnotations,
)
from xxtrain.pipeline.recipes import Recipe, build_recipe
from xxtrain.pipeline.sinks import ClassificationDatasetSink, YoloDatasetSink
from xxtrain.task import TaskType

EXPECTED_STANDARD_PROCESSORS = {
    'detect': (ReadImageInfo, ReadLabelImg, FilterLabels, EncodeDetection),
    'segment': (ReadImageInfo, ReadLabelMe, FilterLabels, PrepareSegmentShapes, EncodeSegment),
    'pose': (
        ReadImageInfo,
        ReadMatchingAnnotations,
        FilterMatchingAnnotations,
        PrepareMatchChildren,
        MatchAnnotations,
        EncodePose,
    ),
    'classify': (PrepareClassification,),
}


class PipelineRecipeTest(unittest.TestCase):
    def copy_fixture(self, fixture_name: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory(prefix='xxtrain-recipe-')
        self.addCleanup(temp_dir.cleanup)
        root_path = Path(temp_dir.name) / fixture_name
        shutil.copytree(FIXTURES_PATH / fixture_name, root_path)
        return root_path

    def test_standard_recipe_processor_sequences(self) -> None:
        fixtures = {
            'detect': 'standard-detect',
            'segment': 'standard-segment',
            'pose': 'standard-pose',
            'classify': 'standard-classify',
        }
        for task_name, expected_types in EXPECTED_STANDARD_PROCESSORS.items():
            with self.subTest(task_name=task_name):
                root_path = self.copy_fixture(fixtures[task_name])
                recipe, _ = build_recipe(
                    task_name,
                    root_path,
                    split=10,
                    reserve_no_label=False,
                )
                self.assertEqual(expected_types, tuple(type(value) for value in recipe.pipeline.processors))

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
                recipe, _ = build_recipe(task_name, root_path)
                self.assertEqual(expected, recipe.labels.names)

    def test_classification_labels_are_sorted(self) -> None:
        root_path = self.copy_fixture('standard-classify')
        (root_path / 'src' / 'labels.txt').write_text('Uab\nIA\nP\n', encoding='utf-8')

        recipe, context = build_recipe('classify', root_path)

        self.assertEqual(('IA', 'P', 'Uab'), recipe.labels.names)
        self.assertIs(recipe.labels, context.config.labels)

    def test_standard_recipes_use_directory_source_and_task_specific_sinks(self) -> None:
        fixtures = {
            'detect': 'standard-detect',
            'segment': 'standard-segment',
            'pose': 'standard-pose',
            'classify': 'standard-classify',
        }
        for task_name, fixture_name in fixtures.items():
            with self.subTest(task_name=task_name):
                recipe, _ = build_recipe(task_name, self.copy_fixture(fixture_name))
                self.assertIsInstance(recipe.source, DirectorySource)
                expected_sink = ClassificationDatasetSink if task_name == 'classify' else YoloDatasetSink
                self.assertIsInstance(recipe.sink, expected_sink)

    def test_recipe_validates_pipeline_sink_boundary(self) -> None:
        with self.assertRaisesRegex(TypeError, 'PrepareClassification produces ClassifyOutput'):
            Recipe(
                name='classify',
                task_type=TaskType.CLASSIFY,
                labels=LabelCatalog(('label',)),
                source=DirectorySource(),
                pipeline=Pipeline((PrepareClassification(),)),
                sink=YoloDatasetSink(),
            )

    def test_unsupported_task_is_rejected_before_filesystem_access(self) -> None:
        with tempfile.TemporaryDirectory(prefix='xxtrain-recipe-') as temp_dir:
            missing_root = Path(temp_dir) / 'does-not-exist'
            for task_name in ('obb', 'point-detect', 'custom'):
                with self.subTest(task_name=task_name):
                    with self.assertRaisesRegex(ValueError, 'Unsupported standard task'):
                        build_recipe(task_name, missing_root)


if __name__ == '__main__':
    unittest.main()
