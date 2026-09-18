import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from xxtrain.platform.app import create_app
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import PlatformAccessError, PlatformConflictError, PlatformError, WorkspaceView
from xxtrain.platform.training_contracts import DownloadFile, ExecutionView, TrainingRun, TrainingRunView


class AsgiClient:
    def __init__(self, app: Any) -> None:
        self.loop = asyncio.new_event_loop()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url='http://testserver'
        )

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        return self.loop.run_until_complete(self.client.request(method, path, **kwargs))

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request('GET', path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request('POST', path, **kwargs)

    def close(self) -> None:
        self.loop.run_until_complete(self.client.aclose())
        self.loop.close()


class AnnotationStub:
    def view(self, user_id: int) -> WorkspaceView:
        return WorkspaceView('line-3', 'Line 3', 0, 0, 0, False, False)

    def upload(self, user_id: int, staged: tuple[Path, ...]) -> WorkspaceView:
        return self.view(user_id)

    def sync_target(self, user_id: int, target: str) -> WorkspaceView:
        return self.view(user_id)

    def generate_target_cache(self, user_id: int, target: str) -> WorkspaceView:
        return self.view(user_id)


class CvatSessionStub:
    user_id = 17

    def current_user(self, cookie: str) -> int:
        if 'sessionid=active' not in cookie:
            raise PlatformAccessError('expired')
        return self.user_id


class TrainingStub:
    def __init__(self, download: DownloadFile) -> None:
        run_id = str(uuid4())
        self.view = TrainingRunView(
            TrainingRun(
                run_id,
                17,
                'line-3',
                'Line 3',
                'detect',
                'fingerprint',
                'cache/secret',
                '2026-09-17T00:00:00+00:00',
                'attempt',
                'clearml-secret',
            ),
            ExecutionView('clearml-secret', 'running', True, 2, 10, 4.5, 0.4, False, None),
        )
        self.download_file = download
        self.calls: list[tuple[object, ...]] = []

    def submit(self, user_id: int, target: str) -> TrainingRunView:
        self.calls.append(('submit', user_id, target))
        return self.view

    def list_runs(self, user_id: int) -> tuple[TrainingRunView, ...]:
        self.calls.append(('list', user_id))
        return (self.view,)

    def workspace_view(self, user_id: int) -> dict[str, object]:
        self.calls.append(('workspace', user_id))
        return {
            'workspace': AnnotationStub().view(user_id),
            'editable': False,
            'training': {'detect': self.view, 'classify': None, 'segment': None},
        }

    def get_run(self, user_id: int, run_id: str) -> TrainingRunView:
        self.calls.append(('get', user_id, run_id))
        if user_id != 17 or run_id != self.view.run.id:
            raise PlatformAccessError('denied')
        return self.view

    def cancel(self, user_id: int, run_id: str) -> TrainingRunView:
        self.calls.append(('cancel', user_id, run_id))
        return self.get_run(user_id, run_id)

    def download(self, user_id: int, run_id: str) -> DownloadFile:
        self.calls.append(('download', user_id, run_id))
        self.get_run(user_id, run_id)
        return self.download_file


class PlatformTrainingHttpTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        artifact = root / 'model.onnx'
        artifact.write_bytes(b'onnx')
        config = WorkspaceConfig('line-3', 'Line 3', 17, root, root / 'runtime', 'http://cvat.test')
        self.cvat = CvatSessionStub()
        self.training = TrainingStub(DownloadFile(artifact, 'model.onnx', 'application/octet-stream'))
        self.client = AsgiClient(create_app(config, AnnotationStub(), self.cvat, training_service=self.training))
        self.addCleanup(self.client.close)
        self.client.client.cookies.set('sessionid', 'active')
        self.client.get('/platform/')
        token = self.client.client.cookies.get('xxtrain_csrf')
        self.write_headers = {'origin': 'http://testserver', 'x-xtrain-csrf': token}

    def test_submit_uses_real_auth_csrf_and_strict_empty_body(self) -> None:
        self.client.client.cookies.delete('sessionid')
        self.assertEqual(
            401, self.client.post('/platform/api/targets/detect/train', json={}, headers=self.write_headers).status_code
        )
        self.client.client.cookies.set('sessionid', 'active')
        self.assertEqual(403, self.client.post('/platform/api/targets/detect/train', json={}).status_code)
        response = self.client.post(
            '/platform/api/targets/detect/train', json={'request_id': str(uuid4())}, headers=self.write_headers
        )
        self.assertEqual(422, response.status_code)
        response = self.client.post('/platform/api/targets/detect/train', json={}, headers=self.write_headers)
        self.assertEqual(200, response.status_code)
        self.assertEqual(('submit', 17, 'detect'), self.training.calls[-1])
        self.assertEqual(self.training.view.run.id, response.json()['run_id'])

    def test_list_and_detail_return_only_safe_projection(self) -> None:
        listed = self.client.get('/platform/api/training-runs')
        detail = self.client.get(f'/platform/api/training-runs/{self.training.view.run.id}')
        self.assertEqual(200, listed.status_code)
        self.assertEqual(200, detail.status_code)
        payload = {'list': listed.json(), 'detail': detail.json()}
        serialized = str(payload)
        for secret in (
            'cache/secret',
            'clearml-secret',
            'cache_relative_path',
            'clearml_task_id',
            'user_id',
            'fingerprint',
        ):
            self.assertNotIn(secret, serialized)
        self.assertEqual('running', detail.json()['execution']['status'])
        self.assertEqual('检测效果：mAP50-95', detail.json()['metric_name'])

    def test_annotation_mutations_return_full_training_workspace_projection(self) -> None:
        responses = (
            self.client.post('/platform/api/targets/detect/sync', json={}, headers=self.write_headers),
            self.client.post('/platform/api/targets/detect/cache', json={}, headers=self.write_headers),
            self.client.post(
                '/platform/api/images',
                files=[('images', ('duplicate.png', b'image', 'image/png'))],
                headers=self.write_headers,
            ),
        )

        for response in responses:
            with self.subTest(path=response.request.url.path):
                self.assertEqual(200, response.status_code)
                payload = response.json()
                self.assertTrue(payload['training_enabled'])
                self.assertTrue(payload['editing_locked'])
                self.assertEqual(self.training.view.run.id, payload['training']['detect']['id'])

    def test_unknown_and_cross_user_run_ids_have_the_same_forbidden_response(self) -> None:
        unknown = self.client.get(f'/platform/api/training-runs/{uuid4()}')
        self.cvat.user_id = 18
        foreign = self.client.get(f'/platform/api/training-runs/{self.training.view.run.id}')
        self.assertEqual((403, unknown.json()), (foreign.status_code, foreign.json()))

    def test_cancel_and_download_use_server_owned_run_and_retry_is_removed(self) -> None:
        run_id = self.training.view.run.id
        cancelled = self.client.post(
            f'/platform/api/training-runs/{run_id}/cancel', json={}, headers=self.write_headers
        )
        retried = self.client.post(f'/platform/api/training-runs/{run_id}/retry', json={}, headers=self.write_headers)
        downloaded = self.client.get(f'/platform/api/training-runs/{run_id}/download')
        self.assertEqual(200, cancelled.status_code)
        self.assertIn(retried.status_code, (404, 405))
        self.assertEqual(b'onnx', downloaded.content)
        self.assertEqual('attachment; filename="model.onnx"', downloaded.headers['content-disposition'])

    def test_expected_backend_failures_are_safely_reported(self) -> None:
        for error in (
            PlatformError('SDK payload contains token=secret and C:/cache'),
            OSError('private metadata path'),
            ValueError('private run facts'),
        ):
            with self.subTest(error=type(error).__name__):

                def fail(*args: object) -> TrainingRunView:
                    raise error

                self.training.list_runs = fail  # type: ignore[method-assign]
                response = self.client.get('/platform/api/training-runs')
                self.assertEqual(502, response.status_code)
                self.assertEqual({'detail': '平台暂时无法完成操作，请重试。'}, response.json())

    def test_programming_errors_are_not_normalized_as_backend_failures(self) -> None:
        def fail(*args: object) -> TrainingRunView:
            raise RuntimeError('programming defect')

        self.training.list_runs = fail  # type: ignore[method-assign]
        response = self.client.get('/platform/api/training-runs')
        self.assertEqual(500, response.status_code)

    def test_write_lock_conflict_is_a_safe_conflict(self) -> None:
        def fail(*args: object) -> TrainingRunView:
            raise PlatformConflictError('private lock details')

        self.training.submit = fail  # type: ignore[method-assign]
        response = self.client.post('/platform/api/targets/detect/train', json={}, headers=self.write_headers)
        self.assertEqual(409, response.status_code)
        self.assertEqual({'detail': '现场当前有操作或训练任务正在进行，请稍后重试。'}, response.json())


if __name__ == '__main__':
    unittest.main()
