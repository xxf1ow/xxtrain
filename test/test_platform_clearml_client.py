import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

import xxtrain.integrations.clearml.client as clearml_client_module
from xxtrain.integrations.clearml.client import (
    CREATED_CANCELLATION_VALUE,
    QUEUED_CANCELLATION_PARAMETER,
    QUEUED_CANCELLATION_VALUE,
    ClearMLClient,
    ClearMLConflictError,
    _ClearMLSDK,
    parse_execution,
)
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

    def test_completed_without_artifact_is_not_active(self):
        without_artifact = parse_execution({'id': 't1', 'status': 'completed'}, artifact_ready=False)
        with_artifact = parse_execution({'id': 't1', 'status': 'completed'}, artifact_ready=True)
        self.assertEqual(without_artifact.status, 'completed')
        self.assertFalse(without_artifact.active)
        self.assertFalse(without_artifact.download_ready)
        self.assertEqual(with_artifact.status, 'completed')
        self.assertFalse(with_artifact.active)
        self.assertTrue(with_artifact.download_ready)

    def test_real_sdk_active_duration_is_projected_as_elapsed_seconds(self):
        from clearml.backend_api.services.v2_20.tasks import Task

        facts = Task(id='task-1', status='in_progress', active_duration=123).to_dict()
        facts['id'] = 'task-1'

        view = parse_execution(facts, artifact_ready=False)

        self.assertEqual(123.0, view.elapsed_seconds)

    def test_missing_sdk_active_duration_remains_unknown(self):
        view = parse_execution({'id': 'task-1', 'status': 'in_progress'}, artifact_ready=False)

        self.assertIsNone(view.elapsed_seconds)

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
                ('cache_relative_path', 'detect/abc'),
                ('shared_root', str(Path('/shared').resolve())),
                ('run_id', run.id),
                ('run_root', str(Path('/shared').resolve().parent / 'runs')),
            ],
            add_task_init_call=False,
        )

    def test_create_carries_the_run_selected_task_entry(self):
        self.sdk.create.return_value = SimpleNamespace(id='task-1')
        run = replace(training_run(), task_entry='package.custom:factory')

        self.client.create(run)

        arguments = dict(self.sdk.create.call_args.kwargs['argparse_args'])
        self.assertEqual('package.custom:factory', arguments['task'])

    def test_create_arguments_match_installed_clearml_sdk_signature(self):
        from clearml import Task

        with (
            patch.object(Task, 'create', autospec=True, return_value=SimpleNamespace(id='task-1')) as create,
            patch.object(_ClearMLSDK, '__init__', lambda sdk: setattr(sdk, '_task', Task)),
        ):
            client = ClearMLClient('xxtrain', 'training', Path('/opt/xxtrain/worker.py'), Path('/shared'))
            self.assertEqual(client.create(training_run()), 'task-1')
            create.assert_called_once()

    def test_find_uses_exact_project_and_run_uuid_and_rejects_duplicates(self):
        run_id = training_run().id
        self.sdk.find.return_value = [SimpleNamespace(id='one'), SimpleNamespace(id='two')]

        with self.assertRaises(ClearMLConflictError):
            self.client.find(run_id)

        self.sdk.find.assert_called_once_with(project_name='xxtrain', task_name=f'^{re.escape(run_id)}$')

    def test_programming_errors_from_sdk_remain_distinguishable(self):
        self.sdk.find.side_effect = TypeError('bad adapter call')

        with self.assertRaisesRegex(TypeError, 'bad adapter call'):
            self.client.find(training_run().id)

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
        self.sdk.get.side_effect = [{'id': 'task-1', 'status': 'created'}, {'id': 'task-1', 'status': 'created'}]
        self.client.enqueue('task-1', training_run())
        self.sdk.prepare.assert_called_once()
        self.sdk.enqueue.assert_called_once_with('task-1', queue_name='training', force=False)

    def test_enqueue_does_not_enqueue_existing_remote_execution_twice(self):
        self.sdk.get.return_value = {'id': 'task-1', 'status': 'queued'}
        self.client.enqueue('task-1', training_run())
        self.sdk.prepare.assert_not_called()
        self.sdk.enqueue.assert_not_called()

    def test_enqueue_does_not_publish_an_incomplete_created_task(self):
        self.sdk.get.return_value = {'id': 'task-1', 'status': 'created'}
        self.sdk.prepare.return_value = False

        with self.assertRaisesRegex(RuntimeError, 'readiness'):
            self.client.enqueue('task-1', training_run())

        self.sdk.enqueue.assert_not_called()

    def test_installed_sdk_completes_the_same_task_after_script_population_failure(self):
        from clearml import Task
        from clearml.backend_api.services.v2_20.tasks import Task as TaskData

        class ControlledRemoteTask:
            def __init__(self):
                self.id = 'task-original'
                self.data = TaskData(id=self.id, status='created', script={})
                self.last_worker = None
                self.parameters = {}
                self.fail_script_update = True

            def get_parameters(self, *args, **kwargs):
                return dict(self.parameters)

            def reload(self):
                return self

            def set_base_docker(self, **kwargs):
                pass

            def set_parameters(self, parameters):
                self.parameters = dict(parameters)

            def update_parameters(self, parameters):
                self.parameters.update(parameters)

            def update_task(self, state):
                if self.fail_script_update:
                    self.fail_script_update = False
                    raise ConnectionError('script publication interrupted')
                self.data.script = state['script']

        with tempfile.TemporaryDirectory() as temporary:
            worker = Path(temporary) / 'worker.py'
            worker.write_text('print("worker")\n', encoding='utf-8')
            remote = ControlledRemoteTask()
            run = training_run()
            arguments = [
                ('task', 'point'),
                ('target', run.target),
                ('cache_relative_path', run.cache_relative_path),
                ('shared_root', str(Path('/shared').resolve())),
                ('run_id', run.id),
                ('run_root', str(Path('/runs').resolve())),
            ]
            with (
                patch.object(Task, '_create', return_value=remote),
                patch.object(Task, 'get_project_id', return_value='project-id'),
                self.assertRaisesRegex(ConnectionError, 'interrupted'),
            ):
                Task.create(
                    project_name='xxtrain',
                    task_name=run.id,
                    task_type='training',
                    script=str(worker),
                    working_directory=str(worker.parent),
                    packages=False,
                    argparse_args=arguments,
                    add_task_init_call=False,
                )

            sdk = object.__new__(_ClearMLSDK)
            sdk._task = Task
            with (
                patch.object(Task, 'get_task', return_value=remote),
                patch.object(Task, 'get_project_id', return_value='project-id'),
            ):
                sdk.prepare(
                    remote.id,
                    project_name='xxtrain',
                    task_name=run.id,
                    task_type='training',
                    script=str(worker),
                    working_directory=str(worker.parent),
                    packages=False,
                    argparse_args=arguments,
                    add_task_init_call=False,
                )

        self.assertEqual('worker.py', remote.data.script.entry_point)
        self.assertEqual('point', remote.parameters['Args/task'])
        self.assertEqual(run.id, remote.parameters['Args/run_id'])

    def test_sdk_population_stops_when_the_created_task_is_cancelled(self):
        from clearml import Task
        from clearml.backend_api.services.v2_20.tasks import Task as TaskData

        class CancelDuringScriptUpdate:
            id = 'task-original'
            last_worker = None
            parameters = {}

            def __init__(self):
                self.data = TaskData(id=self.id, status='created', script={})
                self.parameter_updates = 0

            def get_parameters(self, *args, **kwargs):
                return dict(self.parameters)

            def update_parameters(self, parameters):
                self.parameter_updates += 1

            def update_task(self, state):
                self.data.script = state['script']
                self.data.status = 'stopped'

        with tempfile.TemporaryDirectory() as temporary:
            worker = Path(temporary) / 'worker.py'
            worker.write_text('print("worker")\n', encoding='utf-8')
            remote = CancelDuringScriptUpdate()
            sdk = object.__new__(_ClearMLSDK)
            sdk._task = Task
            with (
                patch.object(Task, 'get_task', return_value=remote),
                patch.object(Task, 'get_project_id', return_value='project-id'),
            ):
                ready = sdk.prepare(
                    remote.id,
                    project_name='xxtrain',
                    task_name=training_run().id,
                    task_type='training',
                    script=str(worker),
                    working_directory=str(worker.parent),
                    packages=False,
                    argparse_args=[('task', 'point')],
                    add_task_init_call=False,
                )

        self.assertFalse(ready)
        self.assertEqual(0, remote.parameter_updates)

    def test_sdk_rejects_a_mismatching_created_task_without_overwriting_it(self):
        from clearml import Task
        from clearml.backend_api.services.v2_20.tasks import Task as TaskData

        remote = Mock()
        remote.id = 'task-original'
        remote.last_worker = None
        remote.data = TaskData(
            id=remote.id,
            status='created',
            script={'entry_point': 'other.py', 'working_dir': '.', 'diff': 'print("other")\n'},
        )
        with tempfile.TemporaryDirectory() as temporary:
            worker = Path(temporary) / 'worker.py'
            worker.write_text('print("worker")\n', encoding='utf-8')
            sdk = object.__new__(_ClearMLSDK)
            sdk._task = Task
            with (
                patch.object(Task, 'get_task', return_value=remote),
                patch.object(Task, 'get_project_id', return_value='project-id'),
                self.assertRaisesRegex(RuntimeError, 'does not match'),
            ):
                sdk.prepare(
                    remote.id,
                    project_name='xxtrain',
                    task_name=training_run().id,
                    task_type='training',
                    script=str(worker),
                    working_directory=str(worker.parent),
                    packages=False,
                    argparse_args=[('task', 'point')],
                    add_task_init_call=False,
                )

        remote.update_task.assert_not_called()
        remote.update_parameters.assert_not_called()

    def test_artifact_query_failure_keeps_completed_execution_inactive(self):
        self.sdk.get.return_value = {'id': 'task-1', 'status': 'completed'}
        self.sdk.has_artifact.side_effect = ConnectionError('artifact store offline')

        view = self.client.observe('task-1')

        self.assertEqual('completed', view.status)
        self.assertFalse(view.active)
        self.assertFalse(view.download_ready)
        self.assertIn('artifact', view.detail.lower())

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
            {
                'id': 'task-1',
                'status': 'stopped',
                'last_worker': None,
                'parameters': {QUEUED_CANCELLATION_PARAMETER: QUEUED_CANCELLATION_VALUE},
            },
        ]

        view = self.client.cancel('task-1')

        self.sdk.cancel_queued.assert_called_once_with('task-1')
        self.sdk.worker_released.assert_not_called()
        self.assertEqual(view.status, 'cancelled')
        self.assertFalse(view.active)

    def test_created_never_started_task_is_stopped_without_enqueue(self):
        self.sdk.get.side_effect = [
            {'id': 'task-1', 'status': 'created', 'last_worker': None},
            {
                'id': 'task-1',
                'status': 'stopped',
                'last_worker': None,
                'parameters': {QUEUED_CANCELLATION_PARAMETER: CREATED_CANCELLATION_VALUE},
            },
        ]

        view = self.client.cancel('task-1')

        self.sdk.cancel_created.assert_called_once_with('task-1')
        self.sdk.enqueue.assert_not_called()
        self.assertEqual('cancelled', view.status)
        self.assertFalse(view.active)

    def test_queued_cancellation_is_stable_across_observe_retry_and_new_client(self):
        facts = {'id': 'task-1', 'status': 'queued', 'last_worker': None, 'parameters': {}}
        sdk = Mock()
        sdk.get.side_effect = lambda task_id: dict(facts, parameters=dict(facts['parameters']))
        sdk.has_artifact.return_value = False

        def cancel_queued(task_id):
            facts['status'] = 'stopped'
            facts['parameters'][QUEUED_CANCELLATION_PARAMETER] = QUEUED_CANCELLATION_VALUE

        sdk.cancel_queued.side_effect = cancel_queued
        client = ClearMLClient('xxtrain', 'training', Path('/worker.py'), Path('/shared'), sdk=sdk)

        cancelled = client.cancel('task-1')
        observed = client.observe('task-1')
        retried = client.cancel('task-1')
        restarted = ClearMLClient('xxtrain', 'training', Path('/worker.py'), Path('/shared'), sdk=sdk).observe('task-1')

        self.assertEqual([cancelled.status, observed.status, retried.status, restarted.status], ['cancelled'] * 4)
        self.assertTrue(all(not view.active for view in (cancelled, observed, retried, restarted)))
        sdk.cancel_queued.assert_called_once_with('task-1')

    def test_manual_stopped_task_without_dequeue_proof_remains_active(self):
        self.sdk.get.return_value = {'id': 'task-1', 'status': 'stopped', 'last_worker': None, 'parameters': {}}

        view = self.client.observe('task-1')

        self.assertEqual(view.status, 'unknown')
        self.assertTrue(view.active)

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

    def test_sdk_does_not_stop_when_dequeue_proof_publication_fails(self):
        sdk = object.__new__(_ClearMLSDK)
        sdk._task = Mock()
        sdk._task.dequeue.return_value = SimpleNamespace(dequeued=1)
        task = sdk._task.get_task.return_value
        task.set_parameter.side_effect = RuntimeError('edit failed')

        with self.assertRaises(RuntimeError):
            sdk.cancel_queued('task-1')

        task.set_parameter.assert_called_once_with(QUEUED_CANCELLATION_PARAMETER, QUEUED_CANCELLATION_VALUE)
        task.stopped.assert_not_called()

    def test_sdk_created_cancellation_records_distinct_never_started_proof(self):
        sdk = object.__new__(_ClearMLSDK)
        sdk._task = Mock()
        task = sdk._task.get_task.return_value
        task.data.status = 'created'
        task.last_worker = None

        sdk.cancel_created('task-1')

        task.set_parameter.assert_called_once_with(QUEUED_CANCELLATION_PARAMETER, CREATED_CANCELLATION_VALUE)
        task.stopped.assert_called_once_with(
            ignore_errors=False, force=False, status_reason='cancelled before execution'
        )

    def test_sdk_owned_session_has_finite_request_configuration(self):
        sdk = object.__new__(_ClearMLSDK)
        sdk._task = Mock()
        sdk._session = Mock()

        self.assertTrue(hasattr(clearml_client_module, '_BoundedSession'))
        self.assertEqual((3.0, 10.0), clearml_client_module._BoundedSession._session_timeout)
        self.assertEqual((3.0, 10.0), clearml_client_module._BoundedSession._session_initial_timeout)
        self.assertEqual((3.0, 10.0), clearml_client_module._BoundedSession._write_session_timeout)

    def test_bounded_session_stops_after_finite_ssl_failures(self):
        from clearml.backend_api.session.session import Session

        bounded_mixin = getattr(clearml_client_module, '_BoundedSession', None)
        self.assertIsNotNone(bounded_mixin)
        session_type = type('TestBoundedSession', (bounded_mixin, Session), {})
        session = object.__new__(session_type)
        session._offline_mode = False
        session._verbose = False
        session._logger = None
        session._session_requests = 1
        session._ssl_error_count_verbosity = 999
        session._Session__worker = 'worker'
        session.client = 'client'
        session._Session__host = 'https://clearml.test'
        session.config = Mock()
        session.config.get.return_value = False
        transport = Mock()
        transport.request.side_effect = requests.exceptions.SSLError('certificate failure')
        session._Session__http_session = transport

        with self.assertRaises(requests.exceptions.SSLError):
            session._send_request('tasks', 'get_all', version='2.20', method='post')

        self.assertEqual(3, transport.request.call_count)

    def test_bounded_session_converts_server_failure_to_finite_sdk_error(self):
        from clearml import Task
        from clearml.backend_api.services.v2_20.tasks import GetAllRequest
        from clearml.backend_interface.session import SendError

        class ParentSession:
            def send(self, req_obj, async_enable=False, headers=None):
                return SimpleNamespace(meta=SimpleNamespace(result_code=503))

        class Session(clearml_client_module._BoundedSession, ParentSession):
            pass

        session = object.__new__(Session)
        request = GetAllRequest()
        session._logger = None

        result = session.send(request)

        self.assertEqual(500, result.meta.result_code)
        with self.assertRaises(SendError):
            Task._send(session, GetAllRequest())

    def test_bounded_session_converts_sdk_decoding_failure_to_finite_error(self):
        from clearml import Task
        from clearml.backend_api.services.v2_20.tasks import GetAllRequest
        from clearml.backend_interface.session import SendError

        class ParentSession:
            def send(self, req_obj, async_enable=False, headers=None):
                raise ValueError('response decoding failed')

        class Session(clearml_client_module._BoundedSession, ParentSession):
            pass

        session = object.__new__(Session)
        session._logger = None

        result = session.send(GetAllRequest())

        self.assertEqual(500, result.meta.result_code)
        with self.assertRaises(SendError):
            Task._send(session, GetAllRequest())

    def test_bounded_session_preserves_programming_error_across_installed_sdk_retry_boundary(self):
        from clearml import Task
        from clearml.backend_api.services.v2_20.tasks import GetAllRequest
        from clearml.backend_interface.session import SendError

        programming_error = TypeError('controlled invalid adapter request')

        class ParentSession:
            def send(self, req_obj, async_enable=False, headers=None):
                raise programming_error

        class Session(clearml_client_module._BoundedSession, ParentSession):
            pass

        session = object.__new__(Session)
        session._logger = None

        def find(**kwargs):
            return Task._send(session, GetAllRequest())

        client = ClearMLClient(
            'xxtrain', 'training', Path('/worker.py'), Path('/shared'), sdk=SimpleNamespace(find=find)
        )

        with self.assertRaises(TypeError) as caught:
            client.find(training_run().id)

        self.assertIs(programming_error, caught.exception)
        self.assertIsInstance(caught.exception.__cause__, SendError)

    def test_sdk_artifact_readiness_uses_metadata_without_downloading(self):
        sdk = object.__new__(_ClearMLSDK)
        sdk._task = Mock()
        task = sdk._task.get_task.return_value
        task.get_project_name.return_value = 'xxtrain'
        artifact = SimpleNamespace(url='https://files/deployment', get_local_copy=Mock())
        task.artifacts = {'deployment': artifact}

        self.assertTrue(sdk.has_artifact('task-1', 'deployment', project_name='xxtrain'))
        artifact.get_local_copy.assert_not_called()

    def test_sdk_artifact_download_rejects_missing_and_empty_local_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdk = object.__new__(_ClearMLSDK)
            sdk._task = Mock()
            task = sdk._task.get_task.return_value
            task.get_project_name.return_value = 'xxtrain'
            artifact = SimpleNamespace(url='https://files/deployment', get_local_copy=Mock(return_value=None))
            task.artifacts = {'deployment': artifact}

            with self.assertRaises(FileNotFoundError):
                sdk.artifact('task-1', 'deployment', project_name='xxtrain')

            empty = Path(temporary) / 'empty.zip'
            empty.touch()
            artifact.get_local_copy.return_value = str(empty)
            with self.assertRaises(FileNotFoundError):
                sdk.artifact('task-1', 'deployment', project_name='xxtrain')

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
