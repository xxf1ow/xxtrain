from __future__ import annotations

import asyncio
import re
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from test.platform_fixture import create_fixture
from test.test_platform_point_workflow import HttpxCvatFixture
from test.test_platform_training_http import AsgiClient
from test.test_platform_training_service import ControlledClearML
from xxtrain.business_tasks.point import point_task_definition
from xxtrain.integrations.clearml.client import (
    CREATED_CANCELLATION_VALUE,
    QUEUED_CANCELLATION_PARAMETER,
    QUEUED_CANCELLATION_VALUE,
    ClearMLClient,
)
from xxtrain.platform.app import create_app
from xxtrain.platform.config import load_config
from xxtrain.platform.contracts import AnnotationChanges, PlatformAccessError, PlatformConflictError
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.service import AnnotationService
from xxtrain.platform.training_contracts import TrainingRun
from xxtrain.platform.training_service import TrainingService
from xxtrain.platform.training_store import TrainingRunStore
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.repository import AnnotationRepository


class _BrowserSession:
    def __init__(self, user_id: int) -> None:
        self.user_id = user_id

    def current_user(self, cookie: str) -> int:
        if 'sessionid=active' not in cookie:
            raise PlatformAccessError('expired')
        return self.user_id


class _QueuedClearML(ControlledClearML):
    def __init__(self) -> None:
        super().__init__()
        self.enqueued: list[str] = []

    def enqueue(self, task_id):
        before = self.tasks[task_id]['status']
        super().enqueue(task_id)
        if before == 'created':
            self.enqueued.append(task_id)


class _LifecycleSDK:
    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, object]] = {}
        self.create_calls = 0
        self.enqueue_calls = 0
        self.cancel_queued_calls = 0
        self.cancel_created_calls = 0
        self.request_stop_calls = 0
        self.find_failures = 0
        self.create_response_lost = False
        self.dequeue_proof_failures = 0
        self.enqueued = threading.Event()
        self.cancelled = threading.Event()

    def find(self, **kwargs):
        if self.find_failures:
            self.find_failures -= 1
            raise ConnectionError('controlled lookup failure')
        pattern = kwargs['task_name']
        return [
            SimpleNamespace(id=task_id)
            for task_id, task in self.tasks.items()
            if re.fullmatch(pattern, str(task['run_id']))
        ]

    def create(self, **kwargs):
        self.create_calls += 1
        task_id = f'task-{self.create_calls}'
        self.tasks[task_id] = {
            'id': task_id,
            'run_id': kwargs['task_name'],
            'status': 'created',
            'last_worker': None,
            'parameters': {},
        }
        if self.create_response_lost:
            self.create_response_lost = False
            raise ConnectionError('controlled response loss')
        return SimpleNamespace(id=task_id)

    def enqueue(self, task_id, **kwargs):
        self.enqueue_calls += 1
        self.tasks[task_id]['status'] = 'queued'
        self.enqueued.set()

    def get(self, task_id):
        task = self.tasks[task_id]
        return dict(task, parameters=dict(task['parameters']))

    def has_artifact(self, task_id, name, **kwargs):
        return False

    def cancel_queued(self, task_id):
        self.cancel_queued_calls += 1
        task = self.tasks[task_id]
        task['status'] = 'created'
        if self.dequeue_proof_failures:
            self.dequeue_proof_failures -= 1
            raise ConnectionError('controlled proof failure')
        task['parameters'][QUEUED_CANCELLATION_PARAMETER] = QUEUED_CANCELLATION_VALUE
        task['status'] = 'stopped'
        self.cancelled.set()

    def cancel_created(self, task_id):
        self.cancel_created_calls += 1
        task = self.tasks[task_id]
        task['parameters'][QUEUED_CANCELLATION_PARAMETER] = CREATED_CANCELLATION_VALUE
        task['status'] = 'stopped'
        self.cancelled.set()

    def request_stop(self, task_id):
        self.request_stop_calls += 1
        raise AssertionError('workflow fixture does not run workers')

    def worker_released(self, task_id, **kwargs):
        return None

    def artifact(self, task_id, name, **kwargs):
        raise AssertionError('missing artifacts must not be downloaded')


class PlatformTrainingWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        receipt = create_fixture(self.root, owner_user_id=17, cvat_internal_url='http://cvat.test')
        self.config = load_config(Path(receipt['config_path']))
        self.owner = self.config.owner_user_id
        self.data = WorkspaceData(self.config.workspace_dir)
        self.repository = AnnotationRepository(Path(receipt['database_path']), point_task_definition())
        self.cvat = HttpxCvatFixture()
        self.addCleanup(self.cvat.close)
        self.annotations = AnnotationService(
            self.config, self.data, self.cvat.client, RuntimeCache(self.config.runtime_dir)
        )
        self.store_path = self.root / 'metadata' / 'training-runs.db'
        self.store = TrainingRunStore(self.store_path)
        self.clearml = _QueuedClearML()
        self.training = self._training_service()
        self.annotations.require_editable = self.training.require_editable
        self.annotations.require_cache_rebuild = self.training.require_cache_rebuild
        self.app = create_app(
            self.config, self.annotations, _BrowserSession(self.owner), training_service=self.training
        )
        self.client = self._client()
        self.addCleanup(self.client.close)

    def _training_service(self) -> TrainingService:
        return TrainingService(
            self.config,
            self.annotations,
            TrainingRunStore(self.store_path),
            self.clearml,
            self.config.runtime_dir / 'cache',
        )

    def _sdk_training_service(self, sdk: _LifecycleSDK) -> TrainingService:
        clearml = ClearMLClient(
            'xxtrain', 'training', Path('/opt/xxtrain/worker.py'), self.config.runtime_dir / 'cache', sdk=sdk
        )
        return TrainingService(
            self.config, self.annotations, TrainingRunStore(self.store_path), clearml, self.config.runtime_dir / 'cache'
        )

    def _client(self, app=None) -> AsgiClient:
        client = AsgiClient(self.app if app is None else app)
        client.client.cookies.set('sessionid', 'active')
        client.get('/platform/')
        return client

    @staticmethod
    def _headers(client: AsgiClient) -> dict[str, str]:
        return {'origin': 'http://testserver', 'x-xtrain-csrf': client.client.cookies.get('xxtrain_csrf')}

    def _submit(self, target: str):
        return self.client.post(f'/platform/api/targets/{target}/train', json={}, headers=self._headers(self.client))

    def _stored_run(
        self,
        target: str,
        desired_action: str | None,
        *,
        user_id: int | None = None,
        create_attempted_at: str | None = None,
        clearml_task_id: str | None = None,
    ) -> TrainingRun:
        fingerprint = self.data.detection_fingerprint() if target == 'detect' else self.data.target_fingerprint(target)
        cache_path = self.annotations.ensure_target_cache(self.owner, target)
        cache_relative_path = cache_path.resolve().relative_to((self.config.runtime_dir / 'cache').resolve()).as_posix()
        return TrainingRunStore(self.store_path).create(
            TrainingRun(
                str(uuid4()),
                self.owner if user_id is None else user_id,
                self.config.workspace_id,
                self.config.display_name,
                target,
                fingerprint,
                cache_relative_path,
                '2026-09-18T12:00:00+00:00',
                create_attempted_at,
                clearml_task_id,
                desired_action,
            )
        )

    @staticmethod
    def _run_lifespan(app, entered: threading.Event) -> None:
        async def run() -> None:
            async with app.router.lifespan_context(app):
                observed = await asyncio.to_thread(entered.wait, 2)
                if not observed:
                    raise AssertionError('coordinator did not reach the controlled SDK boundary')

        asyncio.run(run())

    @staticmethod
    def _run_one_lifespan_pass(app, training: TrainingService) -> int:
        completed = threading.Event()
        calls = 0
        real_reconcile = training.reconcile_pending

        def reconcile() -> None:
            nonlocal calls
            calls += 1
            try:
                real_reconcile()
            finally:
                completed.set()

        async def run() -> None:
            with patch.object(training, 'reconcile_pending', side_effect=reconcile):
                async with app.router.lifespan_context(app):
                    observed = await asyncio.to_thread(completed.wait, 2)
                    if not observed:
                        raise AssertionError('coordinator did not complete its first reconciliation pass')

        asyncio.run(run())
        return calls

    @staticmethod
    def _sdk_write_counts(sdk: _LifecycleSDK) -> tuple[int, int, int, int, int]:
        return (
            sdk.create_calls,
            sdk.enqueue_calls,
            sdk.cancel_queued_calls,
            sdk.cancel_created_calls,
            sdk.request_stop_calls,
        )

    def test_three_runs_keep_annotation_database_unchanged_and_fifo_order(self) -> None:
        before = self.repository.annotations()
        run_ids = []

        for target in ('detect', 'classify', 'segment'):
            response = self._submit(target)
            self.assertEqual(200, response.status_code)
            run_ids.append(response.json()['run_id'])

        self.assertEqual(before, self.repository.annotations())
        self.assertEqual(3, len(set(run_ids)))
        self.assertEqual(3, len(self.store.list_user(self.owner)))
        self.assertEqual(['task-1', 'task-2', 'task-3'], self.clearml.enqueued)

    def test_same_input_double_submit_restart_and_concurrency_share_one_run(self) -> None:
        first = self._submit('detect')
        second = self._submit('detect')
        restarted = self._training_service().submit(self.owner, 'detect')

        self.assertEqual(200, first.status_code)
        self.assertEqual(first.json()['run_id'], second.json()['run_id'])
        self.assertEqual(first.json()['run_id'], restarted.run.id)

        barrier = threading.Barrier(3)

        def submit() -> tuple[int, str | None]:
            barrier.wait()
            try:
                return 200, self._training_service().submit(self.owner, 'classify').run.id
            except PlatformConflictError:
                return 409, None

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(submit) for _ in range(2)]
            barrier.wait()
            results = [future.result() for future in futures]

        successful_ids = {run_id for status, run_id in results if status == 200}
        self.assertTrue(successful_ids)
        self.assertTrue(all(status in (200, 409) for status, _ in results))
        retried = self._submit('classify')
        self.assertEqual(200, retried.status_code)
        successful_ids.add(retried.json()['run_id'])
        self.assertEqual(1, len(successful_ids))
        self.assertEqual(2, len(self.store.list_user(self.owner)))

    def test_unknown_cancel_confirmation_last_terminal_and_stale_sync_gate(self) -> None:
        self.cvat.expect(self.data.images())
        started = self.client.post('/platform/api/targets/detect/start', json={}, headers=self._headers(self.client))
        self.assertEqual(200, started.status_code)
        runs = [self._submit(target).json() for target in ('detect', 'classify', 'segment')]

        stale_sync = self.client.post('/platform/api/targets/detect/sync', json={}, headers=self._headers(self.client))
        self.assertEqual(409, stale_sync.status_code)
        self.assertTrue(self.client.get('/platform/api/workspace').json()['editing_locked'])

        detect_task = self.store.get(self.owner, runs[0]['run_id']).clearml_task_id
        classify_task = self.store.get(self.owner, runs[1]['run_id']).clearml_task_id
        segment_task = self.store.get(self.owner, runs[2]['run_id']).clearml_task_id
        self.clearml.tasks[classify_task]['status'] = 'unknown'
        self.clearml.complete(detect_task)
        cancelled = self.client.post(
            f'/platform/api/training-runs/{runs[2]["run_id"]}/cancel', json={}, headers=self._headers(self.client)
        )
        self.assertEqual('unknown', cancelled.json()['execution']['status'])
        self.assertTrue(self.client.get('/platform/api/workspace').json()['editing_locked'])

        self.clearml.release(segment_task)
        self.assertTrue(self.client.get('/platform/api/workspace').json()['editing_locked'])
        self.clearml.complete(classify_task)
        self.assertFalse(self.client.get('/platform/api/workspace').json()['editing_locked'])

    def test_changed_input_creates_new_run_and_restored_input_reuses_old_run(self) -> None:
        original = self._submit('classify').json()['run_id']
        original_run = self.store.get(self.owner, original)
        original_publication = (self.config.runtime_dir / 'cache' / original_run.cache_relative_path).parent
        original_cache = {
            path.relative_to(original_publication).as_posix(): path.read_bytes()
            for path in original_publication.rglob('*')
            if path.is_file()
        }
        self.clearml.complete(original_run.clearml_task_id)
        original_annotation = self.repository.annotations(step_key='classify')[0]
        changed_annotation = replace(original_annotation, label='tc' if original_annotation.label != 'tc' else 'tl')
        self.repository.apply_changes(AnnotationChanges((changed_annotation,), frozenset(), frozenset()))

        changed = self._submit('classify').json()['run_id']
        self.repository.apply_changes(AnnotationChanges((original_annotation,), frozenset(), frozenset()))
        restored = self._submit('classify').json()['run_id']

        self.assertNotEqual(original, changed)
        self.assertEqual(original, restored)
        self.assertEqual([original, changed], [run.id for run in self.store.list_user(self.owner)])
        retained_cache = {
            path.relative_to(original_publication).as_posix(): path.read_bytes()
            for path in original_publication.rglob('*')
            if path.is_file()
        }
        self.assertEqual(original_cache, retained_cache)

    def test_refreshes_are_read_only_and_coordinator_recovers_bound_created_task(self) -> None:
        original_enqueue = self.clearml.enqueue

        def fail_before_enqueue(task_id):
            raise ConnectionError('queue unavailable before enqueue')

        self.clearml.enqueue = fail_before_enqueue
        failed = self._submit('detect')
        self.assertEqual(502, failed.status_code)
        run = self.store.list_user(self.owner)[0]
        self.clearml.enqueue = original_enqueue

        workspace = self.client.get('/platform/api/workspace')
        listed = self.client.get('/platform/api/training-runs')
        detailed = self.client.get(f'/platform/api/training-runs/{run.id}')
        new_session = self._client()
        try:
            restarted = new_session.get('/platform/api/workspace')
        finally:
            new_session.close()

        self.assertEqual('unknown', workspace.json()['training']['detect']['execution']['status'])
        self.assertEqual('unknown', listed.json()[0]['execution']['status'])
        self.assertEqual('unknown', detailed.json()['execution']['status'])
        self.assertEqual('unknown', restarted.json()['training']['detect']['execution']['status'])
        self.assertEqual([], self.clearml.enqueued)

        self.training.reconcile_pending()
        workspace = self.client.get('/platform/api/workspace')
        listed = self.client.get('/platform/api/training-runs')
        detailed = self.client.get(f'/platform/api/training-runs/{run.id}')

        self.assertEqual('queued', workspace.json()['training']['detect']['execution']['status'])
        self.assertEqual('queued', listed.json()[0]['execution']['status'])
        self.assertEqual('queued', detailed.json()['execution']['status'])
        self.assertEqual(1, self.clearml.create_calls)
        self.assertEqual(['task-1'], self.clearml.enqueued)

    def test_lifespan_resumes_http_submission_without_browser_reads_and_read_routes_stay_pure(self) -> None:
        sdk = _LifecycleSDK()
        sdk.create_response_lost = True
        sdk.find_failures = 1
        training = self._sdk_training_service(sdk)
        app = create_app(self.config, self.annotations, _BrowserSession(self.owner), training_service=training)
        client = self._client(app)
        try:
            submitted = client.post('/platform/api/targets/detect/train', json={}, headers=self._headers(client))
        finally:
            client.close()

        self.assertEqual(502, submitted.status_code)
        run = TrainingRunStore(self.store_path).list_user(self.owner)[0]
        self.assertEqual('execute', run.desired_action)
        self.assertIsNotNone(run.create_attempted_at)
        self.assertIsNone(run.clearml_task_id)

        restarted = self._sdk_training_service(sdk)
        restarted_app = create_app(
            self.config, self.annotations, _BrowserSession(self.owner), training_service=restarted
        )
        self._run_lifespan(restarted_app, sdk.enqueued)

        stored = TrainingRunStore(self.store_path).get(self.owner, run.id)
        self.assertEqual('task-1', stored.clearml_task_id)
        self.assertEqual(1, sdk.create_calls)
        self.assertEqual(1, sdk.enqueue_calls)

        attempted = self._stored_run('classify', 'execute', create_attempted_at='2026-09-18T12:01:00+00:00')
        sdk.tasks['task-unbound'] = {
            'id': 'task-unbound',
            'run_id': attempted.id,
            'status': 'created',
            'last_worker': None,
            'parameters': {},
        }
        rows_before_reads = TrainingRunStore(self.store_path).list_user(self.owner)
        writes_before_reads = self._sdk_write_counts(sdk)

        read_app = create_app(self.config, self.annotations, _BrowserSession(self.owner), training_service=restarted)
        read_client = self._client(read_app)
        try:
            with (
                patch.object(sdk, 'create', side_effect=AssertionError('read must not create')),
                patch.object(sdk, 'enqueue', side_effect=AssertionError('read must not enqueue')),
                patch.object(sdk, 'cancel_queued', side_effect=AssertionError('read must not cancel queued')),
                patch.object(sdk, 'cancel_created', side_effect=AssertionError('read must not cancel created')),
                patch.object(sdk, 'request_stop', side_effect=AssertionError('read must not request stop')),
            ):
                workspace = read_client.get('/platform/api/workspace')
                listed = read_client.get('/platform/api/training-runs')
                detailed = read_client.get(f'/platform/api/training-runs/{attempted.id}')
        finally:
            read_client.close()

        self.assertEqual(200, workspace.status_code)
        self.assertEqual(200, listed.status_code)
        self.assertEqual(200, detailed.status_code)
        self.assertEqual('queued', workspace.json()['training']['detect']['execution']['status'])
        self.assertEqual([run.id, attempted.id], [item['id'] for item in listed.json()])
        self.assertEqual('unknown', detailed.json()['execution']['status'])
        self.assertEqual(rows_before_reads, TrainingRunStore(self.store_path).list_user(self.owner))
        self.assertEqual(writes_before_reads, self._sdk_write_counts(sdk))

    def test_cancel_restarts_after_dequeue_failure_without_reenqueue(self) -> None:
        sdk = _LifecycleSDK()
        training = self._sdk_training_service(sdk)
        app = create_app(self.config, self.annotations, _BrowserSession(self.owner), training_service=training)
        client = self._client(app)
        try:
            submitted = client.post('/platform/api/targets/detect/train', json={}, headers=self._headers(client))
            run_id = submitted.json()['run_id']
            sdk.dequeue_proof_failures = 1
            cancelled = client.post(
                f'/platform/api/training-runs/{run_id}/cancel', json={}, headers=self._headers(client)
            )
        finally:
            client.close()

        self.assertEqual(200, submitted.status_code)
        self.assertEqual(200, cancelled.status_code)
        self.assertTrue(cancelled.json()['cancellation_requested'])
        self.assertEqual('unknown', cancelled.json()['execution']['status'])
        self.assertEqual('cancel', TrainingRunStore(self.store_path).get(self.owner, run_id).desired_action)
        writes_before_restart = (sdk.create_calls, sdk.enqueue_calls)

        restarted = self._sdk_training_service(sdk)
        restarted_app = create_app(
            self.config, self.annotations, _BrowserSession(self.owner), training_service=restarted
        )
        self._run_lifespan(restarted_app, sdk.cancelled)

        self.assertEqual(writes_before_restart, (sdk.create_calls, sdk.enqueue_calls))
        self.assertEqual('cancelled', restarted.get_run(self.owner, run_id).execution.status)

    def test_cancel_before_create_never_reaches_the_sdk(self) -> None:
        sdk = _LifecycleSDK()
        restarted = self._sdk_training_service(sdk)
        local_run = self._stored_run('classify', 'execute')
        local_app = create_app(self.config, self.annotations, _BrowserSession(self.owner), training_service=restarted)
        local_client = self._client(local_app)
        try:
            local_cancel = local_client.post(
                f'/platform/api/training-runs/{local_run.id}/cancel', json={}, headers=self._headers(local_client)
            )
        finally:
            local_client.close()

        self.assertEqual(200, local_cancel.status_code)
        self.assertEqual('cancelled', local_cancel.json()['execution']['status'])
        self.assertEqual((0, 0), (sdk.create_calls, sdk.enqueue_calls))

    def test_legacy_row_is_observed_without_external_writes(self) -> None:
        sdk = _LifecycleSDK()
        legacy = self._stored_run('detect', None)
        before = TrainingRunStore(self.store_path).get(self.owner, legacy.id)
        training = self._sdk_training_service(sdk)
        legacy_app = create_app(self.config, self.annotations, _BrowserSession(self.owner), training_service=training)
        self.assertEqual(1, self._run_one_lifespan_pass(legacy_app, training))

        after_reconcile = TrainingRunStore(self.store_path).get(self.owner, legacy.id)
        self.assertEqual(before, after_reconcile)
        self.assertEqual((0, 0, 0, 0, 0), self._sdk_write_counts(sdk))

        legacy_client = self._client(legacy_app)
        try:
            with (
                patch.object(sdk, 'create', side_effect=AssertionError('read must not create')),
                patch.object(sdk, 'enqueue', side_effect=AssertionError('read must not enqueue')),
                patch.object(sdk, 'cancel_queued', side_effect=AssertionError('read must not cancel')),
                patch.object(sdk, 'cancel_created', side_effect=AssertionError('read must not cancel')),
                patch.object(sdk, 'request_stop', side_effect=AssertionError('read must not request stop')),
            ):
                workspace = legacy_client.get('/platform/api/workspace')
                listed = legacy_client.get('/platform/api/training-runs')
                detailed = legacy_client.get(f'/platform/api/training-runs/{legacy.id}')
        finally:
            legacy_client.close()

        self.assertEqual(200, workspace.status_code)
        self.assertEqual(legacy.id, listed.json()[0]['id'])
        self.assertIsNone(listed.json()[0]['execution'])
        self.assertEqual(legacy.id, detailed.json()['id'])
        self.assertIsNone(detailed.json()['execution'])
        self.assertEqual(legacy.id, workspace.json()['training']['detect']['id'])
        self.assertIsNone(workspace.json()['training']['detect']['execution'])
        self.assertFalse(workspace.json()['editing_locked'])
        self.assertEqual(before, TrainingRunStore(self.store_path).get(self.owner, legacy.id))
        self.assertEqual((0, 0, 0, 0, 0), self._sdk_write_counts(sdk))

    def test_three_targets_unlock_on_last_completion_even_when_artifact_is_missing(self) -> None:
        sdk = _LifecycleSDK()
        training = self._sdk_training_service(sdk)
        app = create_app(self.config, self.annotations, _BrowserSession(self.owner), training_service=training)
        client = self._client(app)
        try:
            responses = [
                client.post(f'/platform/api/targets/{target}/train', json={}, headers=self._headers(client))
                for target in ('detect', 'classify', 'segment')
            ]
            self.assertEqual([200, 200, 200], [response.status_code for response in responses])
            runs = [response.json() for response in responses]
            self.assertTrue(client.get('/platform/api/workspace').json()['editing_locked'])
            for index, run in enumerate(runs):
                task_id = TrainingRunStore(self.store_path).get(self.owner, run['run_id']).clearml_task_id
                sdk.tasks[task_id]['status'] = 'completed'
                locked = client.get('/platform/api/workspace').json()['editing_locked']
                self.assertEqual(index < 2, locked)

            completed = client.get(f'/platform/api/training-runs/{runs[0]["run_id"]}')
            download = client.get(f'/platform/api/training-runs/{runs[0]["run_id"]}/download')
        finally:
            client.close()

        self.assertEqual(3, sdk.create_calls)
        self.assertEqual(3, sdk.enqueue_calls)
        self.assertEqual('completed', completed.json()['execution']['status'])
        self.assertFalse(completed.json()['execution']['download_ready'])
        self.assertEqual(502, download.status_code)

    def test_referenced_damaged_target_caches_are_not_rebuilt_through_http(self) -> None:
        for target in ('classify', 'segment'):
            with self.subTest(target=target):
                submitted = self._submit(target)
                self.assertEqual(200, submitted.status_code)
                run = self.store.get(self.owner, submitted.json()['run_id'])
                publication = (self.config.runtime_dir / 'cache' / run.cache_relative_path).parent
                (publication / 'manifest.json').unlink()
                (publication / 'review-marker').write_bytes(b'preserve')
                before = {
                    path.relative_to(publication).as_posix(): path.read_bytes()
                    for path in publication.rglob('*')
                    if path.is_file()
                }

                response = self.client.post(
                    f'/platform/api/targets/{target}/cache', json={}, headers=self._headers(self.client)
                )
                after = {
                    path.relative_to(publication).as_posix(): path.read_bytes()
                    for path in publication.rglob('*')
                    if path.is_file()
                }

                self.assertEqual(502, response.status_code)
                self.assertEqual({'detail': '平台暂时无法完成操作，请重试。'}, response.json())
                self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
