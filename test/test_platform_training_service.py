import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import requests

import xxtrain.integrations.clearml.client as clearml_client_module
from test.platform_fixture import create_fixture
from xxtrain.business_tasks.point import point_task_definition
from xxtrain.integrations.clearml.client import QUEUED_CANCELLATION_PARAMETER, QUEUED_CANCELLATION_VALUE, ClearMLClient
from xxtrain.platform.config import load_config
from xxtrain.platform.contracts import AnnotationChanges, PlatformAccessError, PlatformConflictError, PlatformError
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.service import AnnotationService
from xxtrain.platform.training_contracts import DownloadFile, ExecutionView, TrainingRun
from xxtrain.platform.training_service import TrainingService
from xxtrain.platform.training_store import TrainingRunStore
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.repository import AnnotationRepository


class _UnusedCvat:
    pass


class ControlledClearML:
    def __init__(self) -> None:
        self.tasks = {}
        self.create_calls = 0
        self.enqueue_calls = 0
        self.cancel_calls = 0
        self.download_calls = 0
        self.create_failure = None
        self.enqueue_failure = None
        self.write_calls = []

    def find(self, run_id):
        matches = [task_id for task_id, task in self.tasks.items() if task['run_id'] == run_id]
        return matches[0] if matches else None

    def create(self, run):
        self.create_calls += 1
        self.write_calls.append(('create', run.id))
        task_id = f'task-{self.create_calls}'
        self.tasks[task_id] = {'run_id': run.id, 'status': 'created'}
        if self.create_failure is not None:
            error = self.create_failure
            self.create_failure = None
            raise error
        return task_id

    def enqueue(self, task_id, run):
        if self.tasks[task_id]['status'] != 'created':
            return
        self.enqueue_calls += 1
        self.write_calls.append(('enqueue', task_id))
        self.tasks[task_id]['status'] = 'queued'
        if self.enqueue_failure is not None:
            error = self.enqueue_failure
            self.enqueue_failure = None
            raise error

    def observe(self, task_id):
        status = self.tasks[task_id]['status']
        active = status in {'created', 'queued', 'running', 'unknown'}
        public_status = 'unknown' if status == 'created' else status
        return ExecutionView(task_id, public_status, active, None, None, None, None, status == 'completed', None)

    def cancel(self, task_id):
        if self.tasks[task_id]['status'] not in {'completed', 'failed', 'cancelled'}:
            self.cancel_calls += 1
            self.write_calls.append(('cancel', task_id))
            self.tasks[task_id]['status'] = 'unknown'
        return self.observe(task_id)

    def download(self, task_id, destination):
        self.download_calls += 1
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / 'model.onnx'
        path.write_bytes(b'model')
        return DownloadFile(path, path.name, 'application/octet-stream')

    def complete(self, task_id):
        self.tasks[task_id]['status'] = 'completed'

    def release(self, task_id):
        self.tasks[task_id]['status'] = 'cancelled'


class DequeueProofFailureSDK:
    def __init__(self) -> None:
        self.task = None
        self.enqueue_calls = 0
        self.proof_failure = True

    def find(self, **kwargs):
        return [] if self.task is None else [type('Task', (), {'id': 'task-1'})()]

    def create(self, **kwargs):
        self.task = {'id': 'task-1', 'status': 'created', 'last_worker': None, 'parameters': {}}
        return type('Task', (), {'id': 'task-1'})()

    def enqueue(self, task_id, **kwargs):
        self.enqueue_calls += 1
        self.task['status'] = 'queued'

    def prepare(self, task_id, **kwargs):
        return True

    def get(self, task_id):
        return dict(self.task, parameters=dict(self.task['parameters']))

    def has_artifact(self, task_id, name, **kwargs):
        return False

    def cancel_queued(self, task_id):
        self.task['status'] = 'created'
        if self.proof_failure:
            self.proof_failure = False
            raise RuntimeError('proof write failed')

    def cancel_created(self, task_id):
        self.task['parameters'][QUEUED_CANCELLATION_PARAMETER] = QUEUED_CANCELLATION_VALUE
        self.task['status'] = 'stopped'

    def request_stop(self, task_id):
        raise AssertionError('created task must not use running cancellation')

    def worker_released(self, task_id, **kwargs):
        return None


class TrainingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        receipt = create_fixture(self.root, owner_user_id=17, cvat_internal_url='http://cvat.test')
        self.config = load_config(Path(receipt['config_path']))
        self.workspace_id = self.config.workspace_id
        self.owner = self.config.owner_user_id
        self.data = WorkspaceData(self.config.workspace_dir, point_task_definition())
        self.annotations = AnnotationService(
            self.config, self.data, _UnusedCvat(), RuntimeCache(self.config.runtime_dir)
        )
        for target in ('detect', 'classify', 'segment'):
            self.annotations.generate_target_cache(self.owner, target)
        self.store_path = self.root / 'platform' / 'training-runs.db'
        self.backend = ControlledClearML()
        self.service = self._service()

    def _service(self):
        return TrainingService(
            self.config,
            self.annotations,
            TrainingRunStore(self.store_path),
            self.backend,
            self.config.runtime_dir / 'cache',
        )

    def _store_run(
        self,
        target='detect',
        *,
        desired_action,
        create_attempted_at=None,
        clearml_task_id=None,
        submitted_at='2026-09-17T12:00:00+00:00',
    ):
        fingerprint = self.data.detection_fingerprint() if target == 'detect' else self.data.target_fingerprint(target)
        return TrainingRunStore(self.store_path).create(
            TrainingRun(
                str(uuid4()),
                self.owner,
                self.workspace_id,
                self.config.display_name,
                target,
                fingerprint,
                f'{fingerprint}/{target}',
                submitted_at,
                create_attempted_at,
                clearml_task_id,
                desired_action,
            )
        )

    def test_lock_depends_on_all_runs_not_last_clicked_target(self):
        runs = [self.service.submit(self.owner, target) for target in ('detect', 'classify', 'segment')]
        self.backend.complete(runs[0].run.clearml_task_id)
        self.backend.complete(runs[1].run.clearml_task_id)
        with self.assertRaises(PlatformError):
            self.service.require_editable(self.workspace_id)
        self.backend.complete(runs[2].run.clearml_task_id)
        self.service.require_editable(self.workspace_id)

    def test_workspace_view_uses_workspace_runs_for_lock_and_user_runs_for_visibility(self):
        fingerprint = self.data.detection_fingerprint()
        store = TrainingRunStore(self.store_path)
        other_workspace = store.create(
            TrainingRun(
                str(uuid4()),
                self.owner,
                'line-other',
                'Other line',
                'detect',
                fingerprint,
                f'{fingerprint}/detect',
                '2026-09-17T12:01:00+00:00',
                None,
                None,
                'execute',
            )
        )
        other_owner = store.create(
            TrainingRun(
                str(uuid4()),
                99,
                self.workspace_id,
                self.config.display_name,
                'detect',
                fingerprint,
                f'{fingerprint}/detect',
                '2026-09-17T12:02:00+00:00',
                None,
                None,
                'execute',
            )
        )

        state = self.service.workspace_view(self.owner)

        self.assertFalse(state['editable'])
        self.assertIsNone(state['training']['detect'])
        self.assertEqual((other_workspace,), tuple(view.run for view in self.service.list_runs(self.owner)))
        with self.assertRaises(PlatformAccessError):
            self.service.get_run(self.owner, other_owner.id)

    def test_cancel_keeps_lock_across_restart_until_worker_release(self):
        run = self.service.submit(self.owner, 'detect')
        cancelled = self.service.cancel(self.owner, run.run.id)
        self.assertEqual('unknown', cancelled.execution.status)

        restarted = self._service()
        self.assertEqual('unknown', restarted.get_run(self.owner, run.run.id).execution.status)
        with self.assertRaises(PlatformError):
            restarted.require_editable(self.workspace_id)
        self.backend.release(run.run.clearml_task_id)
        self.assertEqual('cancelled', restarted.get_run(self.owner, run.run.id).execution.status)
        restarted.require_editable(self.workspace_id)

    def test_legacy_unattempted_same_input_remains_observation_only(self):
        run = self._store_run(desired_action=None)

        submitted = self._service().submit(self.owner, 'detect')
        self._service().reconcile_pending()

        self.assertEqual(run, submitted.run)
        self.assertIsNone(submitted.execution)
        self.assertEqual([], self.backend.write_calls)

    def test_same_input_recovers_task_when_create_return_was_lost(self):
        self.backend.create_failure = ConnectionError('response lost')

        recovered = self.service.submit(self.owner, 'detect')
        repeated = self.service.submit(self.owner, 'detect')

        self.assertEqual(recovered.run, repeated.run)
        self.assertEqual(1, self.backend.create_calls)
        self.assertEqual('queued', repeated.execution.status)

    def test_cancel_after_interrupted_population_uses_the_original_task_without_enqueue(self):
        self.backend.create_failure = ConnectionError('response lost after identity creation')

        def incomplete(task_id, run):
            raise ConnectionError('launch population interrupted')

        self.backend.enqueue = incomplete
        with self.assertRaisesRegex(PlatformError, 'queue'):
            self.service.submit(self.owner, 'detect')
        run = TrainingRunStore(self.store_path).list_user(self.owner)[0]

        cancelled = self.service.cancel(self.owner, run.id)

        self.assertEqual('task-1', cancelled.run.clearml_task_id)
        self.assertEqual(1, self.backend.create_calls)
        self.assertEqual([('create', run.id), ('cancel', 'task-1')], self.backend.write_calls)

    def test_uncertain_create_without_remote_match_is_not_retried_blindly(self):
        def missing_create(run):
            self.backend.create_calls += 1
            raise ConnectionError('request state unknown')

        self.backend.create = missing_create
        with self.assertRaisesRegex(PlatformError, 'not confirmed'):
            self.service.submit(self.owner, 'detect')
        with self.assertRaisesRegex(PlatformError, 'not confirmed'):
            self.service.submit(self.owner, 'detect')
        self.assertEqual(1, self.backend.create_calls)

    def test_uncertain_enqueue_reuses_bound_task(self):
        self.backend.enqueue_failure = ConnectionError('response lost')
        with self.assertRaisesRegex(PlatformError, 'queue'):
            self.service.submit(self.owner, 'detect')

        recovered = self.service.submit(self.owner, 'detect')

        self.assertEqual('task-1', recovered.run.clearml_task_id)
        self.assertEqual(1, self.backend.create_calls)

    def test_queries_do_not_resume_created_task(self):
        view = self.service.submit(self.owner, 'detect')
        self.backend.tasks[view.run.clearml_task_id]['status'] = 'created'
        before = list(self.backend.write_calls)

        self.service.list_runs(self.owner)
        self.service.get_run(self.owner, view.run.id)
        self.service.workspace_view(self.owner)

        self.assertEqual(before, self.backend.write_calls)

    def test_unattempted_execute_locks_then_cancel_survives_restart_without_remote_writes(self):
        run = self._store_run(desired_action='execute')
        with self.assertRaises(PlatformConflictError):
            self.service.require_editable(self.workspace_id)

        cancelled = self.service.cancel(self.owner, run.id)
        restarted = self._service()
        restarted.reconcile_pending()

        self.assertEqual('cancel', TrainingRunStore(self.store_path).get(self.owner, run.id).desired_action)
        self.assertEqual('cancelled', cancelled.execution.status)
        self.assertTrue(cancelled.cancellation_requested)
        self.assertEqual([], self.backend.write_calls)
        restarted.require_editable(self.workspace_id)

    def test_read_reconciliation_never_reenqueues_terminal_tasks(self):
        run = self.service.submit(self.owner, 'detect')
        enqueue_calls = self.backend.enqueue_calls

        for status in ('completed', 'failed', 'cancelled'):
            with self.subTest(status=status):
                self.backend.tasks[run.run.clearml_task_id]['status'] = status
                view = self.service.get_run(self.owner, run.run.id)
                self.assertEqual(status, view.execution.status)
                self.assertEqual(enqueue_calls, self.backend.enqueue_calls)

    def test_reconcile_pending_never_restarts_terminal_tasks(self):
        run = self.service.submit(self.owner, 'detect')
        writes = list(self.backend.write_calls)

        for status in ('completed', 'failed', 'cancelled'):
            with self.subTest(status=status):
                self.backend.tasks[run.run.clearml_task_id]['status'] = status
                self.service.reconcile_pending()
                self.assertEqual(writes, self.backend.write_calls)

    def test_reconcile_pending_uses_durable_submission_order(self):
        runs = [
            self._store_run(target, desired_action='execute', submitted_at=timestamp)
            for target, timestamp in (
                ('segment', '2026-09-17T12:03:00+00:00'),
                ('detect', '2026-09-17T12:01:00+00:00'),
                ('classify', '2026-09-17T12:02:00+00:00'),
            )
        ]

        self.service.reconcile_pending()

        self.assertEqual([('create', run.id) for run in runs], self.backend.write_calls[::2])

    def test_reconcile_pending_normalizes_lazy_sdk_login_failure_and_continues(self):
        from clearml.backend_api.session.defs import ENV_ACCESS_KEY, ENV_SECRET_KEY

        runs = [
            self._store_run('detect', desired_action='execute'),
            self._store_run('classify', desired_action='execute'),
        ]
        clearml = ClearMLClient(
            'xxtrain', 'training', Path('/opt/xxtrain/worker.py'), self.config.runtime_dir / 'cache'
        )
        service = TrainingService(
            self.config, self.annotations, TrainingRunStore(self.store_path), clearml, self.config.runtime_dir / 'cache'
        )

        with (
            patch.object(ENV_ACCESS_KEY, 'get', return_value='controlled-key'),
            patch.object(ENV_SECRET_KEY, 'get', return_value='controlled-secret'),
            patch.object(
                clearml_client_module._BoundedSession,
                '_send_request',
                side_effect=requests.Timeout('controlled login timeout'),
            ),
            self.assertLogs('xxtrain.platform.training_service', level='WARNING') as logs,
        ):
            service.reconcile_pending()

        stored = [TrainingRunStore(self.store_path).get(self.owner, run.id) for run in runs]
        self.assertEqual(2, len(logs.output))
        self.assertTrue(all('Training submission recovery failed' in entry for entry in logs.output))
        self.assertTrue(all(run.create_attempted_at is not None for run in stored))
        self.assertTrue(all(run.clearml_task_id is None for run in stored))

    def test_read_reconciliation_keeps_unconfirmed_creation_unknown_and_locked(self):
        def missing_create(run):
            self.backend.create_calls += 1
            raise ConnectionError('request state unknown')

        self.backend.create = missing_create
        with self.assertRaisesRegex(PlatformError, 'not confirmed'):
            self.service.submit(self.owner, 'detect')

        listed = self.service.list_runs(self.owner)

        self.assertEqual('unknown', listed[0].execution.status)
        self.assertEqual(1, self.backend.create_calls)
        with self.assertRaises(PlatformError):
            self.service.require_editable(self.workspace_id)

    def test_cancel_attempted_missing_association_finds_and_cancels_only_original_task(self):
        run = self._store_run(desired_action='execute', create_attempted_at='2026-09-17T12:01:00+00:00')
        self.backend.tasks['remote-original'] = {'run_id': run.id, 'status': 'created'}

        cancelled = self.service.cancel(self.owner, run.id)

        stored = TrainingRunStore(self.store_path).get(self.owner, run.id)
        self.assertEqual('cancel', stored.desired_action)
        self.assertEqual('remote-original', stored.clearml_task_id)
        self.assertEqual([('cancel', 'remote-original')], self.backend.write_calls)
        self.assertEqual('unknown', cancelled.execution.status)

    def test_cancel_attempted_missing_association_never_creates_replacement(self):
        run = self._store_run(desired_action='execute', create_attempted_at='2026-09-17T12:01:00+00:00')

        cancelled = self.service.cancel(self.owner, run.id)

        self.assertEqual('cancel', TrainingRunStore(self.store_path).get(self.owner, run.id).desired_action)
        self.assertTrue(cancelled.cancellation_requested)
        self.assertEqual('unknown', cancelled.execution.status)
        self.assertEqual([], self.backend.write_calls)

    def test_cancelled_input_repeat_submission_does_not_restore_execute_intent(self):
        original = self.service.submit(self.owner, 'detect')
        self.service.cancel(self.owner, original.run.id)
        before = list(self.backend.write_calls)

        repeated = self.service.submit(self.owner, 'detect')

        self.assertEqual(original.run.id, repeated.run.id)
        self.assertEqual('cancel', repeated.run.desired_action)
        self.assertTrue(repeated.cancellation_requested)
        self.assertNotIn(('enqueue', original.run.clearml_task_id), self.backend.write_calls[len(before) :])

    def test_durable_cancel_prevents_reenqueue_after_dequeue_proof_write_failure(self):
        sdk = DequeueProofFailureSDK()
        clearml = ClearMLClient(
            'xxtrain', 'training', Path('/opt/xxtrain/worker.py'), self.config.runtime_dir / 'cache', sdk=sdk
        )
        service = TrainingService(
            self.config, self.annotations, TrainingRunStore(self.store_path), clearml, self.config.runtime_dir / 'cache'
        )
        run = service.submit(self.owner, 'detect')

        interrupted = service.cancel(self.owner, run.run.id)
        reads_before = sdk.enqueue_calls
        restarted = TrainingService(
            self.config, self.annotations, TrainingRunStore(self.store_path), clearml, self.config.runtime_dir / 'cache'
        )
        restarted.get_run(self.owner, run.run.id)
        restarted.list_runs(self.owner)
        restarted.workspace_view(self.owner)
        restarted.reconcile_pending()

        self.assertTrue(interrupted.cancellation_requested)
        self.assertEqual(reads_before, sdk.enqueue_calls)
        self.assertEqual('cancelled', restarted.get_run(self.owner, run.run.id).execution.status)

    def test_unknown_remote_status_keeps_workspace_locked(self):
        run = self.service.submit(self.owner, 'detect')

        def unavailable(task_id):
            raise ConnectionError('offline')

        self.backend.observe = unavailable
        self.assertEqual('unknown', self.service.get_run(self.owner, run.run.id).execution.status)
        with self.assertRaises(PlatformError):
            self.service.require_editable(self.workspace_id)

    def test_foreign_user_cannot_access_runs_or_trigger_external_operations(self):
        run = self.service.submit(self.owner, 'detect')
        before = (self.backend.create_calls, self.backend.cancel_calls, self.backend.download_calls)
        operations = (
            ('get', lambda: self.service.get_run(99, run.run.id)),
            ('cancel', lambda: self.service.cancel(99, run.run.id)),
            ('download', lambda: self.service.download(99, run.run.id)),
        )
        self.assertEqual((), self.service.list_runs(99))
        for name, operation in operations:
            with self.subTest(name=name), self.assertRaises(PlatformAccessError):
                operation()
        self.assertEqual(before, (self.backend.create_calls, self.backend.cancel_calls, self.backend.download_calls))

    def test_changed_input_creates_new_run_and_restored_input_reuses_original(self):
        original = self.service.submit(self.owner, 'classify')
        self.backend.complete(original.run.clearml_task_id)
        repository = AnnotationRepository(self.config.workspace_dir / 'annotations.db', point_task_definition())
        original_annotation = repository.annotations(step_key='classify')[0]
        changed_annotation = replace(original_annotation, label='tc' if original_annotation.label != 'tc' else 'tl')
        repository.apply_changes(AnnotationChanges((changed_annotation,), frozenset(), frozenset()))

        changed = self.service.submit(self.owner, 'classify')
        repository.apply_changes(AnnotationChanges((original_annotation,), frozenset(), frozenset()))
        restored = self.service.submit(self.owner, 'classify')

        self.assertNotEqual(original.run.id, changed.run.id)
        self.assertEqual(original.run.id, restored.run.id)
        self.assertEqual(2, self.backend.create_calls)

    def test_same_input_execution_states_reuse_original_run(self):
        original = self.service.submit(self.owner, 'detect')
        task_id = original.run.clearml_task_id

        for status in ('created', 'queued', 'running', 'completed', 'failed', 'cancelled', 'unknown'):
            with self.subTest(status=status):
                self.backend.tasks[task_id]['status'] = status
                enqueue_calls = self.backend.enqueue_calls

                repeated = self._service().submit(self.owner, 'detect')

                self.assertEqual(original.run.id, repeated.run.id)
                self.assertEqual(1, self.backend.create_calls)
                self.assertEqual(enqueue_calls + (status == 'created'), self.backend.enqueue_calls)
                self.assertEqual('queued' if status == 'created' else status, repeated.execution.status)

    def test_different_target_submission_remains_available_while_another_target_is_active(self):
        detect = self.service.submit(self.owner, 'detect')

        classify = self.service.submit(self.owner, 'classify')

        self.assertNotEqual(detect.run.id, classify.run.id)
        self.assertEqual('queued', classify.execution.status)

    def test_workspace_projection_and_write_guard_share_active_rule(self):
        run = self._store_run(desired_action='execute')

        self.assertFalse(self.service.workspace_view(self.owner)['editable'])
        with self.assertRaises(PlatformConflictError):
            self.service.require_editable(self.workspace_id)

        self.service.cancel(self.owner, run.id)
        self.assertTrue(self.service.workspace_view(self.owner)['editable'])
        self.service.require_editable(self.workspace_id)

    def test_cancel_conflicts_with_inflight_coordination_and_no_enqueue_follows_saved_cancel(self):
        entered = threading.Event()
        release = threading.Event()
        original_create = self.backend.create

        def held_create(run):
            entered.set()
            self.assertTrue(release.wait(5))
            return original_create(run)

        self.backend.create = held_create
        run = self._store_run(desired_action='execute')
        worker = threading.Thread(target=self.service.reconcile_pending)
        worker.start()
        self.assertTrue(entered.wait(5))
        with self.assertRaises(PlatformConflictError):
            self.service.cancel(self.owner, run.id)
        release.set()
        worker.join(5)
        self.assertFalse(worker.is_alive())

        self.service.cancel(self.owner, run.id)
        writes_after_cancel = len(self.backend.write_calls)
        self.service.reconcile_pending()

        self.assertNotIn(('enqueue', 'task-1'), self.backend.write_calls[writes_after_cancel:])

    def test_referenced_cache_damage_is_not_rebuilt_over_historical_input(self):
        original = self.service.submit(self.owner, 'detect')
        self.backend.complete(original.run.clearml_task_id)
        cache = self.config.runtime_dir / 'cache' / original.run.cache_relative_path
        (cache / 'dataset.yaml').unlink()

        repeated = self.service.submit(self.owner, 'detect')

        self.assertEqual(original.run.id, repeated.run.id)
        self.assertFalse((cache / 'dataset.yaml').exists())


if __name__ == '__main__':
    unittest.main()
