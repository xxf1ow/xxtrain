import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from xxtrain.cli import main


class CliTest(unittest.TestCase):
    @patch('xxtrain.cli.train')
    def test_train_dispatches_scenario(self, train) -> None:
        main(['train', 'data/standard-detect/standard_detect.py'])

        train.assert_called_once_with(Path('data/standard-detect/standard_detect.py'))

    @patch('xxtrain.cli.export')
    def test_export_dispatches_scenario_and_weights(self, export) -> None:
        main(['export', 'scenario.py', '--weights', 'best.pt'])

        export.assert_called_once_with(Path('scenario.py'), Path('best.pt'))

    @patch('xxtrain.cli.review')
    def test_review_dispatches_scenario_weights_and_directory(self, review) -> None:
        main(['review', 'scenario.py', '--weights', 'best.pt', '--directory', 'images'])

        review.assert_called_once_with(Path('scenario.py'), Path('best.pt'), Path('images'))

    def test_required_subcommand_and_arguments_are_enforced(self) -> None:
        invalid_arguments = [
            [],
            ['train'],
            ['export', '--weights', 'best.pt'],
            ['export', 'scenario.py'],
            ['review', '--weights', 'best.pt', '--directory', 'images'],
            ['review', 'scenario.py', '--directory', 'images'],
            ['review', 'scenario.py', '--weights', 'best.pt'],
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                    main(arguments)
                self.assertNotEqual(0, raised.exception.code)

    @patch('xxtrain.cli.export', side_effect=RuntimeError('export failed'))
    def test_workflow_failure_escapes_main(self, export) -> None:
        with self.assertRaisesRegex(RuntimeError, 'export failed'):
            main(['export', 'scenario.py', '--weights', 'best.pt'])


if __name__ == '__main__':
    unittest.main()
