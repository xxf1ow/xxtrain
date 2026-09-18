import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from xxtrain.integrations.clearml.client import ClearMLClient, ClearMLConflictError, _ClearMLSDK, parse_execution
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
        self.sdk.get.return_value = {
            'id': 'task-1',
            'status': 'stopped',
            'last_worker': 'worker-1',
            'status_changed': '2026-09-17T12:00:00Z',
        }
        self.sdk.worker_released.return_value = False

        view = self.client.cancel('task-1')

        self.assertEqual(view.status, 'unknown')
        self.assertTrue(view.active)
        self.assertIn('worker', view.detail.lower())

    def test_cancelled_task_releases_only_after_fresh_worker_release_report(self):
        self.sdk.get.side_effect = [
            {'id': 'task-1', 'status': 'in_progress', 'last_worker': 'worker-1'},
            {'id': 'task-1', 'status': 'stopped', 'last_worker': 'worker-1', 'status_changed': '2026-09-17T12:00:00Z'},
        ]
        self.sdk.worker_released.return_value = True

        view = self.client.cancel('task-1')

        self.sdk.request_stop.assert_called_once_with('task-1')
        self.assertEqual(view.status, 'cancelled')
        self.assertFalse(view.active)

    def test_worker_observation_failure_keeps_cancelled_task_active(self):
        self.sdk.get.return_value = {
            'id': 'task-1',
            'status': 'stopped',
            'last_worker': 'worker-1',
            'status_changed': '2026-09-17T12:00:00Z',
        }
        self.sdk.worker_released.side_effect = RuntimeError('offline')

        view = self.client.cancel('task-1')

        self.assertEqual(view.status, 'unknown')
        self.assertTrue(view.active)
        self.assertNotIn('offline', view.detail)

    def test_missing_worker_record_does_not_prove_release(self):
        self.sdk.get.return_value = {
            'id': 'task-1',
            'status': 'stopped',
            'last_worker': 'worker-1',
            'status_changed': '2026-09-17T12:00:00Z',
        }
        self.sdk.worker_released.return_value = None

        view = self.client.observe('task-1')

        self.assertEqual(view.status, 'unknown')
        self.assertTrue(view.active)

    def test_queued_never_started_task_cancels_after_atomic_dequeue(self):
        self.sdk.get.side_effect = [
            {'id': 'task-1', 'status': 'queued', 'last_worker': None},
            {'id': 'task-1', 'status': 'stopped', 'last_worker': None},
        ]

        view = self.client.cancel('task-1')

        self.sdk.cancel_queued.assert_called_once_with('task-1')
        self.sdk.worker_released.assert_not_called()
        self.assertEqual(view.status, 'cancelled')
        self.assertFalse(view.active)

    def test_sdk_requires_fresh_report_from_the_specific_worker(self):
        sdk = object.__new__(_ClearMLSDK)
        sdk._task = Mock()
        session = sdk._task._get_default_session.return_value
        response = session.send.return_value
        response.ok.return_value = True
        worker = SimpleNamespace(
            id='worker-1', last_report_time='2026-09-17T12:00:01Z', task=SimpleNamespace(id='other-task')
        )
        response.response.workers = [worker]

        self.assertTrue(sdk.worker_released('task-1', worker_id='worker-1', stopped_at='2026-09-17T12:00:00Z'))
        worker.task = SimpleNamespace(id='task-1')
        self.assertFalse(sdk.worker_released('task-1', worker_id='worker-1', stopped_at='2026-09-17T12:00:00Z'))
        worker.task = None
        worker.last_report_time = '2026-09-17T11:59:59Z'
        self.assertIsNone(sdk.worker_released('task-1', worker_id='worker-1', stopped_at='2026-09-17T12:00:00Z'))
        response.response.workers = []
        self.assertIsNone(sdk.worker_released('task-1', worker_id='worker-1', stopped_at='2026-09-17T12:00:00Z'))

    def test_sdk_does_not_stop_queued_task_when_dequeue_loses_race(self):
        sdk = object.__new__(_ClearMLSDK)
        sdk._task = Mock()
        sdk._task.dequeue.return_value = SimpleNamespace(dequeued=0)

        with self.assertRaises(RuntimeError):
            sdk.cancel_queued('task-1')

        sdk._task.get_task.assert_not_called()

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
