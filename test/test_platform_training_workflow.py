from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from test.platform_fixture import create_fixture
from test.test_platform_point_workflow import HttpxCvatFixture
from test.test_platform_training_http import AsgiClient
from test.test_platform_training_service import ControlledClearML
from xxtrain.business_tasks.point import point_task_definition
from xxtrain.platform.app import create_app
from xxtrain.platform.config import load_config
from xxtrain.platform.contracts import AnnotationChanges, PlatformAccessError, PlatformConflictError
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.service import AnnotationService
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

    def _client(self) -> AsgiClient:
        client = AsgiClient(self.app)
        client.client.cookies.set('sessionid', 'active')
        client.get('/platform/')
        return client

    @staticmethod
    def _headers(client: AsgiClient) -> dict[str, str]:
        return {'origin': 'http://testserver', 'x-xtrain-csrf': client.client.cookies.get('xxtrain_csrf')}

    def _submit(self, target: str):
        return self.client.post(f'/platform/api/targets/{target}/train', json={}, headers=self._headers(self.client))

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
        self.clearml.complete(original_run.clearml_task_id)
        original_annotation = self.repository.annotations(step_key='classify')[0]
        changed_annotation = replace(original_annotation, label='tc' if original_annotation.label != 'tc' else 'tl')
        self.repository.apply_changes(AnnotationChanges((changed_annotation,), frozenset(), frozenset()))

        changed = self._submit('classify').json()['run_id']
        self.repository.apply_changes(AnnotationChanges((original_annotation,), frozenset(), frozenset()))
        restored = self._submit('classify').json()['run_id']

        self.assertNotEqual(original, changed)
        self.assertEqual(original, restored)
        self.assertEqual(2, len(self.store.list_user(self.owner)))

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
