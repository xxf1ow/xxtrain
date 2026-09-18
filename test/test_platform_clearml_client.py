import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from xxtrain.integrations.clearml.client import ClearMLClient, ClearMLConflictError, parse_execution
from xxtrain.platform.training_contracts import TrainingRun


def training_run(run_id: str = '8f245d4e-0ab1-4eb4-98db-6415249cadf7') -> TrainingRun:
    return TrainingRun(run_id, 1, 'point', 'Point', 'detect', 'abc', 'detect/abc', '2026-09-17T00:00:00Z', None, None)


class ClearMLClientTests(unittest.TestCase):
    def setUp(self):
        self.sdk = Mock()
        self.client = ClearMLClient(
            'xxtrain', 'training', Path('/opt/xxtrain/worker.py'), Path('/shared'), sdk=self.sdk
        )

    def test_unknown_remote_state_does_not_release_workspace(self):
        view = parse_execution({'id': 't1', 'status': 'unrecognized'}, artifact_ready=False)
        self.assertEqual(view.status, 'unknown')
        self.assertTrue(view.active)
        self.assertFalse(view.download_ready)

    def test_completed_requires_published_deployment_artifact(self):
        without_artifact = parse_execution({'id': 't1', 'status': 'completed'}, artifact_ready=False)
        with_artifact = parse_execution({'id': 't1', 'status': 'completed'}, artifact_ready=True)
        self.assertEqual(without_artifact.status, 'unknown')
        self.assertTrue(without_artifact.active)
        self.assertEqual(with_artifact.status, 'completed')
        self.assertFalse(with_artifact.active)
        self.assertTrue(with_artifact.download_ready)

    def test_create_carries_run_identity_and_worker_arguments_in_initial_call(self):
        task = SimpleNamespace(id='task-1')
        self.sdk.create.return_value = task
        run = training_run()

        task_id = self.client.create(run)

        self.assertEqual(task_id, 'task-1')
        self.sdk.create.assert_called_once_with(
            project_name='xxtrain',
            task_name=run.id,
            task_type='training',
            script=str(Path('/opt/xxtrain/worker.py').resolve()),
            working_directory=str(Path('/opt/xxtrain').resolve()),
            packages=False,
            argparse_args=[
                ('task', 'point'),
                ('target', 'detect'),
                ('cache-relative-path', 'detect/abc'),
                ('shared-root', str(Path('/shared').resolve())),
                ('run-id', run.id),
                ('run-root', str(Path('/shared').resolve().parent / 'runs')),
            ],
            add_task_init_call=False,
        )

    def test_create_arguments_match_installed_clearml_sdk_signature(self):
        from clearml import Task

        with patch.object(Task, 'create', autospec=True, return_value=SimpleNamespace(id='task-1')) as create:
            client = ClearMLClient('xxtrain', 'training', Path('/opt/xxtrain/worker.py'), Path('/shared'))
            self.assertEqual(client.create(training_run()), 'task-1')
            create.assert_called_once()

    def test_find_uses_exact_project_and_run_uuid_and_rejects_duplicates(self):
        run_id = training_run().id
        self.sdk.find.return_value = [SimpleNamespace(id='one'), SimpleNamespace(id='two')]

        with self.assertRaises(ClearMLConflictError):
            self.client.find(run_id)

        self.sdk.find.assert_called_once_with(project_name='xxtrain', task_name=f'^{re.escape(run_id)}$')

    def test_ordinary_package_import_does_not_import_clearml_sdk(self):
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                "import sys; import xxtrain.integrations.clearml; raise SystemExit('clearml' in sys.modules)",
            ],
            check=False,
        )
        self.assertEqual(result.returncode, 0)

    def test_enqueue_always_uses_configured_queue_without_force(self):
        self.sdk.get.return_value = {'id': 'task-1', 'status': 'created'}
        self.client.enqueue('task-1')
        self.sdk.enqueue.assert_called_once_with('task-1', queue_name='training', force=False)

    def test_enqueue_does_not_enqueue_existing_remote_execution_twice(self):
        self.sdk.get.return_value = {'id': 'task-1', 'status': 'queued'}
        self.client.enqueue('task-1')
        self.sdk.enqueue.assert_not_called()

    def test_cancelled_task_remains_active_while_worker_owns_it(self):
        self.sdk.get.return_value = {'id': 'task-1', 'status': 'stopped'}
        self.sdk.worker_occupancy.return_value = True

        view = self.client.cancel('task-1')

        self.assertEqual(view.status, 'unknown')
        self.assertTrue(view.active)
        self.assertIn('worker', view.detail.lower())

    def test_cancelled_task_releases_only_after_worker_absence_is_observed(self):
        self.sdk.get.side_effect = [{'id': 'task-1', 'status': 'in_progress'}, {'id': 'task-1', 'status': 'stopped'}]
        self.sdk.worker_occupancy.return_value = False

        view = self.client.cancel('task-1')

        self.sdk.request_stop.assert_called_once_with('task-1')
        self.assertEqual(view.status, 'cancelled')
        self.assertFalse(view.active)

    def test_worker_observation_failure_keeps_cancelled_task_active(self):
        self.sdk.get.return_value = {'id': 'task-1', 'status': 'stopped'}
        self.sdk.worker_occupancy.side_effect = RuntimeError('offline')

        view = self.client.cancel('task-1')

        self.assertEqual(view.status, 'unknown')
        self.assertTrue(view.active)
        self.assertNotIn('offline', view.detail)

    def test_download_copies_only_the_fixed_owned_deployment_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'remote.zip'
            source.write_bytes(b'delivery')
            destination = Path(temporary) / 'private'
            self.sdk.artifact.return_value = SimpleNamespace(
                name='deployment', url='https://files/x', local_path=source
            )

            result = self.client.download('task-1', destination)

            self.sdk.artifact.assert_called_once_with('task-1', 'deployment', project_name='xxtrain')
            self.assertEqual(result.path.read_bytes(), b'delivery')
            self.assertEqual(result.filename, 'model.zip')
            self.assertEqual(result.media_type, 'application/zip')
            self.assertTrue(result.path.is_relative_to(destination.resolve()))


if __name__ == '__main__':
    unittest.main()
