import tempfile
import textwrap
import unittest
from pathlib import Path

from xxtrain.pipeline import standard_recipe
from xxtrain.task import TaskType
from xxtrain.training import TrainingScenario, load_scenario


class TrainingScenarioTest(unittest.TestCase):
    def write_scenario(self, root: Path, body: str) -> Path:
        path = root / 'scenario.py'
        path.write_text(textwrap.dedent(body), encoding='utf-8')
        return path

    def test_defaults_match_current_standard_behavior(self) -> None:
        scenario = TrainingScenario(dataset=standard_recipe(TaskType.CLASSIFY))
        self.assertEqual('v8', scenario.model_version)
        self.assertEqual('n', scenario.model_scale)
        self.assertEqual(10, scenario.split)
        self.assertFalse(scenario.reserve_no_label)
        self.assertEqual({}, scenario.train_args)

    def test_load_resolves_nested_path_values_relative_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = self.write_scenario(
                root,
                """
                from pathlib import Path
                from xxtrain.pipeline import standard_recipe
                from xxtrain.task import TaskType
                from xxtrain.training import TrainingScenario

                SCENARIO = TrainingScenario(
                    dataset=standard_recipe(TaskType.CLASSIFY),
                    train_args={
                        'cache': Path('cache'),
                        'nested': [Path('one'), (Path('two'),)],
                        'plain': 'relative-string',
                    },
                )
                """,
            )
            scenario = load_scenario(path)
            self.assertEqual(root / 'cache', scenario.train_args['cache'])
            self.assertEqual(root / 'one', scenario.train_args['nested'][0])
            self.assertEqual(root / 'two', scenario.train_args['nested'][1][0])
            self.assertEqual('relative-string', scenario.train_args['plain'])

    def test_load_rejects_missing_file_export_and_wrong_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(FileNotFoundError):
                load_scenario(root / 'missing.py')
            with self.assertRaisesRegex(ValueError, 'must export SCENARIO'):
                load_scenario(self.write_scenario(root, 'VALUE = 1'))
            with self.assertRaisesRegex(TypeError, 'SCENARIO must be a TrainingScenario'):
                load_scenario(self.write_scenario(root, 'SCENARIO = object()'))

    def test_model_scale_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Unsupported model scale: z'):
            TrainingScenario(dataset=standard_recipe(TaskType.DETECT), model_scale='z')
