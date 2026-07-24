import importlib.util
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


def load_entry(name: str) -> ModuleType:
    path = Path(__file__).parents[1] / 'src' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'test_entry_{name}', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load entry: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainingEntriesTest(unittest.TestCase):
    def test_train_requires_only_scenario(self) -> None:
        module = load_entry('train')

        with patch.object(module, 'train') as run:
            module.main(['data/standard-detect/standard_detect.py'])

        run.assert_called_once_with(Path('data/standard-detect/standard_detect.py'))

    def test_export_requires_scenario_and_weights(self) -> None:
        module = load_entry('export')

        with patch.object(module, 'export') as run:
            module.main(['scenario.py', '--weights', 'best.pt'])

        run.assert_called_once_with(Path('scenario.py'), Path('best.pt'))

    def test_review_requires_scenario_weights_and_directory(self) -> None:
        module = load_entry('review')

        with patch.object(module, 'review') as run:
            module.main(['scenario.py', '--weights', 'best.pt', '--directory', 'images'])

        run.assert_called_once_with(Path('scenario.py'), Path('best.pt'), Path('images'))

    def test_required_arguments_are_enforced(self) -> None:
        invalid_arguments = {
            'train': [[]],
            'export': [['--weights', 'best.pt'], ['scenario.py']],
            'review': [
                ['--weights', 'best.pt', '--directory', 'images'],
                ['scenario.py', '--directory', 'images'],
                ['scenario.py', '--weights', 'best.pt'],
            ],
        }

        for entry, argument_lists in invalid_arguments.items():
            module = load_entry(entry)
            for arguments in argument_lists:
                with self.subTest(entry=entry, arguments=arguments):
                    with self.assertRaises(SystemExit) as raised:
                        module.main(arguments)
                    self.assertNotEqual(0, raised.exception.code)

    def test_export_failure_escapes_main(self) -> None:
        module = load_entry('export')

        with patch.object(module, 'export', side_effect=RuntimeError('export failed')):
            with self.assertRaisesRegex(RuntimeError, 'export failed'):
                module.main(['scenario.py', '--weights', 'best.pt'])


if __name__ == '__main__':
    unittest.main()
