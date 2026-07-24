import shutil
import tempfile
import unittest
from pathlib import Path

from test.support.scenarios import SCENARIO_PATHS, load_case_scenario
from xxtrain.pipeline import convert_dataset
from xxtrain.pipeline.processors import (
    CropDetectionBoxes,
    CropMatches,
    EncodeCropDetection,
    EncodeDetection,
    EncodeKnobSegment,
    EncodePointSegment,
    FilterLabels,
    FilterMatchingAnnotations,
    MatchAnnotations,
    PartitionAnnotations,
    PrepareMatchChildren,
    ReadImageInfo,
    ReadLabelImg,
    ReadMatchingAnnotations,
    RelabelAnnotations,
    RelabelCropAnnotations,
)
from xxtrain.pipeline.sinks import ClassificationDatasetSink, YoloDatasetSink
from xxtrain.task import TaskType

FIXTURES_PATH = Path(__file__).resolve().parent / 'fixtures'

EXPECTED_SPECIAL_PROCESSORS = {
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

EXPECTED_SPECIAL_LABELS = {
    'point-detect': ('Point',),
    'point-classify': ('tl', 'tc', 'cl', 'cc'),
    'point-segment': ('Point',),
    'knob-detect': ('switch',),
    'knob-segment': ('switch',),
    'light1-detect': ('1008',),
    'light2-detect': ('0',),
}


class TrainingPresetTest(unittest.TestCase):
    def test_all_twelve_presets_load(self) -> None:
        self.assertEqual(12, len(SCENARIO_PATHS))
        for name in SCENARIO_PATHS:
            with self.subTest(name=name):
                self.assertIsNotNone(load_case_scenario(name))

    def test_first_eleven_presets_use_standard_training_args(self) -> None:
        for name in SCENARIO_PATHS:
            if name == 'digit-cls':
                continue
            with self.subTest(name=name):
                self.assertEqual({}, load_case_scenario(name).train_args)

    def test_digit_cls_reuses_standard_classify_recipe_with_overrides(self) -> None:
        standard = load_case_scenario('classify')
        digit = load_case_scenario('digit-cls')
        self.assertIs(TaskType.CLASSIFY, digit.dataset.task_type)
        self.assertEqual(standard.dataset.name, digit.dataset.name)
        self.assertEqual(
            tuple(type(value) for value in standard.dataset.pipeline.processors),
            tuple(type(value) for value in digit.dataset.pipeline.processors),
        )
        self.assertIs(type(standard.dataset.sink), type(digit.dataset.sink))
        self.assertEqual(
            {
                'epochs': 80,
                'batch': 16,
                'imgsz': 320,
                'patience': 15,
                'optimizer': 'AdamW',
                'lr0': 0.0005,
                'lrf': 0.05,
                'weight_decay': 0.001,
                'warmup_epochs': 3.0,
                'cos_lr': True,
                'dropout': 0.15,
                'fliplr': 0.0,
                'flipud': 0.0,
                'auto_augment': None,
                'erasing': 0.0,
            },
            digit.train_args,
        )

    def test_special_preset_processor_sequences_and_label_catalogs(self) -> None:
        for name, expected_types in EXPECTED_SPECIAL_PROCESSORS.items():
            with self.subTest(name=name):
                recipe = load_case_scenario(name).dataset
                self.assertEqual(expected_types, tuple(type(value) for value in recipe.pipeline.processors))
                self.assertEqual(EXPECTED_SPECIAL_LABELS[name], recipe.labels.names)

    def test_special_preset_processor_configuration(self) -> None:
        for name in EXPECTED_SPECIAL_PROCESSORS:
            with self.subTest(name=name):
                recipe = load_case_scenario(name).dataset
                filter_processors = tuple(
                    processor for processor in recipe.pipeline.processors if isinstance(processor, FilterLabels)
                )
                if not filter_processors:
                    continue
                (filter_labels,) = filter_processors
                expected_input_labels = (
                    ('tl', 'tc', 'cl', 'cc') if name.startswith('point-') else EXPECTED_SPECIAL_LABELS[name]
                )
                self.assertEqual(expected_input_labels, filter_labels.labels)
                self.assertEqual(name != 'light1-detect', filter_labels.strict)

                if name == 'point-detect':
                    relabel = next(
                        processor
                        for processor in recipe.pipeline.processors
                        if isinstance(processor, RelabelAnnotations)
                    )
                    self.assertEqual('Point', relabel.label)

    def test_matching_crop_preset_processor_configuration(self) -> None:
        expected = {
            'point-segment': (('tl', 'tc', 'cl', 'cc'), ('1',), 0),
            'knob-segment': (('switch',), ('switch',), 0.15),
        }
        for name, (parent_labels, child_labels, wide) in expected.items():
            with self.subTest(name=name):
                recipe = load_case_scenario(name).dataset
                matching_filter = next(
                    processor
                    for processor in recipe.pipeline.processors
                    if isinstance(processor, FilterMatchingAnnotations)
                )
                prepare = next(
                    processor for processor in recipe.pipeline.processors if isinstance(processor, PrepareMatchChildren)
                )
                matcher = next(
                    processor for processor in recipe.pipeline.processors if isinstance(processor, MatchAnnotations)
                )
                self.assertEqual(parent_labels, matching_filter.parent_labels)
                self.assertEqual(child_labels, matching_filter.child_labels)
                self.assertTrue(matching_filter.strict)
                self.assertIs(TaskType.SEGMENT, prepare.task_type)
                self.assertEqual(wide, matcher.wide)
                self.assertFalse(matcher.strict)
                self.assertIsInstance(recipe.sink, YoloDatasetSink)

    def test_light2_detection_partitions_one_labelimg_read(self) -> None:
        recipe = load_case_scenario('light2-detect').dataset

        processors = recipe.pipeline.processors
        self.assertEqual(1, sum(isinstance(value, ReadLabelImg) for value in processors))
        partition = next(processor for processor in processors if isinstance(processor, PartitionAnnotations))
        matcher = next(processor for processor in processors if isinstance(processor, MatchAnnotations))
        relabel = next(processor for processor in processors if isinstance(processor, RelabelCropAnnotations))
        self.assertEqual({'1008'}, partition.parent_labels)
        self.assertEqual({'0', '1', '2'}, partition.child_labels)
        self.assertEqual(0.1, matcher.wide)
        self.assertFalse(matcher.strict)
        self.assertEqual('0', relabel.label)
        self.assertIsInstance(recipe.sink, YoloDatasetSink)

    def test_point_classification_uses_indexed_class_directories(self) -> None:
        recipe = load_case_scenario('point-classify').dataset

        self.assertIsInstance(recipe.sink, ClassificationDatasetSink)
        self.assertTrue(recipe.sink.indexed_class_directories)

    def test_digit_cls_converts_standard_classify_fixture(self) -> None:
        with tempfile.TemporaryDirectory(prefix='xxtrain-digit-cls-') as temp_dir:
            root_path = Path(temp_dir) / 'digit-cls'
            shutil.copytree(FIXTURES_PATH / 'standard-classify', root_path)
            scenario = load_case_scenario('digit-cls')

            report = convert_dataset(
                scenario.dataset, root_path, split=scenario.split, reserve_no_label=scenario.reserve_no_label
            )

            self.assertEqual(6, report.train_image_count + report.val_image_count)
            self.assertTrue((root_path / 'classify' / 'train').is_dir())


if __name__ == '__main__':
    unittest.main()
