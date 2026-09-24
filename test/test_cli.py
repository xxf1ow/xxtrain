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

        review.assert_called_once_with(Path('scenario.py'), Path('best.pt'), Path('images'), False)

    @patch('xxtrain.cli.review')
    def test_review_dispatches_unlabeled_mode(self, review) -> None:
        main(['review', 'scenario.py', '--weights', 'best.pt', '--directory', 'images', '--unlabeled'])

        review.assert_called_once_with(Path('scenario.py'), Path('best.pt'), Path('images'), True)

    @patch('xxtrain.serverctl.main', return_value=0)
    def test_serverctl_dispatches_sibling_command(self, control) -> None:
        main(['serverctl', 'status'])

        control.assert_called_once_with('status')

    @patch('xxtrain.serverctl.main', return_value=7)
    def test_serverctl_maps_failure_to_exit_status(self, control) -> None:
        with self.assertRaises(SystemExit) as raised:
            main(['serverctl', 'start'])

        self.assertEqual(7, raised.exception.code)
        control.assert_called_once_with('start')

    def test_serverctl_accepts_only_lifecycle_actions(self) -> None:
        for action in ('install', 'start', 'stop', 'status', 'verify'):
            with self.subTest(action=action):
                with patch('xxtrain.serverctl.main', return_value=0) as control:
                    main(['serverctl', action])
                control.assert_called_once_with(action)

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main(['serverctl', 'restart'])

    def test_serverctl_lifecycle_actions_reach_controller_entry(self) -> None:
        from xxtrain import serverctl

        for action, operation in (('install', 'install'), ('start', 'start'), ('stop', 'stop')):
            with self.subTest(action=action), patch.object(serverctl, operation) as controller:
                main(['serverctl', action])
            controller.assert_called_once()

    @patch('xxtrain.serverctl.main', return_value=0)
    def test_internal_bootstrap_accepts_deadline_without_public_help_listing(self, control) -> None:
        main(['serverctl', 'bootstrap', '--timeout', '12'])
        control.assert_called_once_with('bootstrap', timeout=12)
        with patch('sys.stdout', new_callable=StringIO) as output, self.assertRaises(SystemExit):
            main(['serverctl', '--help'])
        self.assertNotIn('bootstrap', output.getvalue())

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
