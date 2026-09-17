import asyncio
import json
import os
import tempfile
import unittest
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from unittest.mock import patch

_MISSING_HTTP_DEPENDENCIES = tuple(name for name in ('fastapi', 'httpx') if find_spec(name) is None)
if _MISSING_HTTP_DEPENDENCIES:
    raise unittest.SkipTest(f'platform extra is required: {", ".join(_MISSING_HTTP_DEPENDENCIES)}')
else:
    import httpx
    from PIL import Image

    from xxtrain.business_tasks import POINT_BOX_LABELS
    from xxtrain.business_tasks.point import point_task_definition
    from xxtrain.integrations.cvat import CvatClient
    from xxtrain.platform.app import create_app
    from xxtrain.platform.config import WorkspaceConfig
    from xxtrain.platform.contracts import JobRef, PlatformAccessError, PlatformError, WorkspaceView
    from xxtrain.platform.runtime import RuntimeCache
    from xxtrain.platform.service import AnnotationService
    from xxtrain.workspace_data import WorkspaceData
    from xxtrain.workspace_data.repository import AnnotationRepository


class AsgiTestClient:
    def __init__(self, app: Any) -> None:
        self.loop = asyncio.new_event_loop()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url='http://testserver'
        )

    @property
    def cookies(self) -> httpx.Cookies:
        return self.client.cookies

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        return self.loop.run_until_complete(self.client.request(method, path, **kwargs))

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request('GET', path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request('POST', path, **kwargs)

    def close(self) -> None:
        self.loop.run_until_complete(self.client.aclose())
        self.loop.close()

    def __enter__(self) -> 'AsgiTestClient':
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.view_result = WorkspaceView('line-3', '三号现场', 2, 0, 0, False, False)
        self.errors: dict[str, Exception] = {}
        self.staged: tuple[Path, ...] = ()
        self.uploaded: tuple[bytes, ...] = ()

    def _result(self, name: str, user_id: int) -> object:
        self.calls.append((name, user_id))
        if name in self.errors:
            raise self.errors[name]
        if name == 'begin':
            return '/tasks/41/jobs/73'
        return self.view_result

    def view(self, user_id: int) -> WorkspaceView:
        return self._result('view', user_id)  # type: ignore[return-value]

    def begin_detection(self, user_id: int) -> str:
        return self._result('begin', user_id)  # type: ignore[return-value]

    def sync_detection(self, user_id: int) -> WorkspaceView:
        return self._result('sync', user_id)  # type: ignore[return-value]

    def upload(self, user_id: int, staged: tuple[Path, ...]) -> WorkspaceView:
        self.staged = staged
        self.uploaded = tuple(path.read_bytes() for path in staged)
        return self._result('upload', user_id)  # type: ignore[return-value]

    def generate_detection_cache(self, user_id: int) -> WorkspaceView:
        return self._result('cache', user_id)  # type: ignore[return-value]


class FakeCvat:
    def __init__(self) -> None:
        self.user_id = 17
        self.calls: list[str] = []
        self.errors: dict[str, Exception] = {}

    def _raise(self, name: str) -> None:
        self.calls.append(name)
        if name in self.errors:
            raise self.errors[name]

    def current_user(self, cookie: str) -> int:
        self._raise('current_user')
        if 'sessionid=active' not in cookie:
            raise PlatformAccessError('missing session')
        return self.user_id

    def login(self, username: str, password: str) -> tuple[str, ...]:
        self._raise('login')
        if (username, password) != ('worker', 'password'):
            raise PlatformAccessError('invalid credentials')
        return ('csrftoken=cvat-csrf; Path=/; SameSite=Lax', 'sessionid=active; Path=/; HttpOnly; SameSite=Lax')

    def logout(self, cookie: str, csrf: str) -> tuple[str, ...]:
        self._raise('logout')
        return ('csrftoken=""; Max-Age=0; Path=/', 'sessionid=""; Max-Age=0; Path=/')


class PlatformHttpTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.config = WorkspaceConfig(
            workspace_id='line-3',
            display_name='三号现场',
            owner_user_id=17,
            workspace_dir=Path('workspace'),
            runtime_dir=Path(directory.name) / 'runtime',
            cvat_internal_url='http://cvat.test',
        )
        self.service = FakeService()
        self.cvat = FakeCvat()
        self.client = AsgiTestClient(create_app(self.config, self.service, self.cvat))
        self.addCleanup(self.client.close)

    def authenticate(self) -> dict[str, str]:
        self.client.cookies.set('sessionid', 'active')
        self.client.cookies.set('csrftoken', 'cvat-csrf')
        response = self.client.get('/platform/')
        self.assertEqual(200, response.status_code)
        token = self.client.cookies.get('xxtrain_csrf')
        self.assertIsNotNone(token)
        return {'origin': 'http://testserver', 'x-xtrain-csrf': token}

    def test_session_and_workspace_require_a_browser_session(self) -> None:
        for path in ('/platform/api/session', '/platform/api/workspace'):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(401, response.status_code)
        self.assertEqual([], self.service.calls)

    def test_workspace_owner_denial_is_distinct_from_authentication(self) -> None:
        self.cvat.user_id = 99
        self.service.errors['view'] = PlatformAccessError('private owner detail')
        self.client.cookies.set('sessionid', 'active')
        response = self.client.get('/platform/api/workspace')

        self.assertEqual(403, response.status_code)
        self.assertNotIn('private owner detail', response.text)

    def test_workspace_returns_the_view_and_exact_point_target_definitions(self) -> None:
        self.client.cookies.set('sessionid', 'active')
        response = self.client.get('/platform/api/workspace')

        self.assertEqual(
            {
                'workspace_id': 'line-3',
                'name': '三号现场',
                'image_count': 2,
                'annotated_image_count': 0,
                'boxed_image_count': 0,
                'can_generate_detection_cache': False,
                'detection_cache_ready': False,
                'task': {'id': 'point', 'name': 'Point'},
                'targets': [
                    {'id': 'detect', 'name': '检测', 'available': True},
                    {'id': 'classify', 'name': '分类', 'available': False},
                    {'id': 'segment', 'name': '分割', 'available': False},
                ],
            },
            response.json(),
        )

    def test_every_write_rejects_cross_origin_requests_before_side_effects(self) -> None:
        headers = self.authenticate() | {'origin': 'http://attacker.test'}
        requests = (
            ('/platform/api/login', {'username': 'worker', 'password': 'password'}),
            ('/platform/api/logout', {}),
            ('/platform/api/detection/start', {}),
            ('/platform/api/detection/sync', {}),
            ('/platform/api/detection/cache', {}),
        )
        cvat_calls = list(self.cvat.calls)
        for path, payload in requests:
            with self.subTest(path=path):
                response = self.client.post(path, headers=headers, json=payload)
                self.assertEqual(403, response.status_code)
        self.assertEqual(cvat_calls, self.cvat.calls)
        self.assertEqual([], self.service.calls)

    def test_every_write_requires_the_page_csrf_token_including_login(self) -> None:
        self.client.get('/platform/')
        requests = (
            ('/platform/api/login', {'username': 'worker', 'password': 'password'}),
            ('/platform/api/logout', {}),
            ('/platform/api/detection/start', {}),
            ('/platform/api/detection/sync', {}),
            ('/platform/api/detection/cache', {}),
        )
        for path, payload in requests:
            with self.subTest(path=path):
                response = self.client.post(path, headers={'origin': 'http://testserver'}, json=payload)
                self.assertEqual(403, response.status_code)
        self.assertEqual([], self.cvat.calls)
        self.assertEqual([], self.service.calls)

    def test_login_session_logout_round_trip_forwards_only_session_cookies(self) -> None:
        self.client.get('/platform/')
        token = self.client.cookies.get('xxtrain_csrf')
        headers = {'origin': 'http://testserver', 'x-xtrain-csrf': token}

        login = self.client.post(
            '/platform/api/login', headers=headers, json={'username': 'worker', 'password': 'password'}
        )
        session = self.client.get('/platform/api/session')
        logout = self.client.post('/platform/api/logout', headers=headers, json={})

        self.assertEqual(204, login.status_code)
        self.assertEqual({'authenticated': True, 'user_id': 17}, session.json())
        self.assertEqual(204, logout.status_code)
        self.assertEqual(['login', 'current_user', 'current_user', 'logout'], self.cvat.calls)

    def test_detection_actions_accept_only_an_empty_object(self) -> None:
        headers = self.authenticate()
        invalid_payloads = (
            {'target': 'classify'},
            {'target': 'segment'},
            {'task_id': 41},
            {'job_id': 73},
            {'image_path': 'C:/private/image.jpg'},
            {'annotations_dir': '/private/annotations'},
            [],
            'text',
            1,
        )
        for path in ('/platform/api/detection/start', '/platform/api/detection/sync', '/platform/api/detection/cache'):
            for payload in invalid_payloads:
                with self.subTest(path=path, payload=payload):
                    response = self.client.post(path, headers=headers, json=payload)
                    self.assertEqual(422, response.status_code)
            with self.subTest(path=path, payload=None):
                response = self.client.post(path, headers=headers)
                self.assertEqual(422, response.status_code)
        self.assertEqual([], self.service.calls)

    def test_start_returns_only_the_server_owned_annotation_path(self) -> None:
        response = self.client.post('/platform/api/detection/start', headers=self.authenticate(), json={})

        self.assertEqual(200, response.status_code)
        self.assertEqual({'annotation_url': '/tasks/41/jobs/73'}, response.json())
        self.assertEqual([('begin', 17)], self.service.calls)

    def test_sync_error_is_not_success(self) -> None:
        self.service.errors['sync'] = PlatformError('Could not save annotations at C:/private/site')
        response = self.client.post('/platform/api/detection/sync', headers=self.authenticate(), json={})

        self.assertEqual(502, response.status_code)
        self.assertNotIn('C:/private/site', response.text)

    def test_malformed_workspace_annotation_is_an_operational_error(self) -> None:
        self.service.errors['view'] = ValueError('LabelMe shapes must be a list')
        self.client.cookies.set('sessionid', 'active')
        response = self.client.get('/platform/api/workspace')

        self.assertEqual(502, response.status_code)

    def test_returned_get_never_syncs(self) -> None:
        self.client.cookies.set('sessionid', 'active')
        response = self.client.get('/platform/?returned=1')

        self.assertEqual(200, response.status_code)
        self.assertEqual([], self.service.calls)

    def test_upload_stages_multiple_files_with_generated_names_and_cleans_them(self) -> None:
        self.service.view_result = replace(self.service.view_result, image_count=4)
        response = self.client.post(
            '/platform/api/images',
            headers=self.authenticate(),
            files=[('images', ('../../private/a.jpg', b'first')), ('images', ('a.PNG', b'second'))],
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(4, response.json()['image_count'])
        self.assertNotIn('status', response.json())
        self.assertEqual((b'first', b'second'), self.service.uploaded)
        for path in self.service.staged:
            self.assertTrue(path.is_relative_to(self.config.runtime_dir / 'staging'))
            self.assertNotEqual('a', path.stem)
            self.assertFalse(path.exists())
        self.assertEqual([], list((self.config.runtime_dir / 'staging').iterdir()))

    def test_upload_requires_origin_csrf_session_and_owner_before_staging(self) -> None:
        headers = self.authenticate()
        cases = (
            (headers | {'origin': 'http://attacker.test'}, True, 17, 403),
            ({'origin': 'http://testserver'}, True, 17, 403),
            (headers, False, 17, 401),
            (headers, True, 99, 403),
        )
        for request_headers, has_session, user_id, expected in cases:
            with self.subTest(expected=expected, user_id=user_id):
                self.client.cookies.clear()
                self.client.cookies.set('xxtrain_csrf', headers['x-xtrain-csrf'])
                if has_session:
                    self.client.cookies.set('sessionid', 'active')
                self.cvat.user_id = user_id
                response = self.client.post(
                    '/platform/api/images', headers=request_headers, files={'images': ('a.jpg', b'jpeg')}
                )
                self.assertEqual(expected, response.status_code)
        self.assertFalse(self.config.runtime_dir.exists())
        self.assertEqual([], self.service.calls)

    def test_upload_requires_nonempty_file_parts_named_images(self) -> None:
        headers = self.authenticate()
        for payload in ({'json': {}}, {'data': {'images': 'path.jpg'}}, {'files': {'other': ('a.jpg', b'jpeg')}}):
            with self.subTest(payload=payload):
                response = self.client.post('/platform/api/images', headers=headers, **payload)
                self.assertEqual(422, response.status_code)
        self.assertEqual([], self.service.calls)

    def test_upload_parsing_and_filesystem_errors_are_sanitized(self) -> None:
        headers = self.authenticate()
        malformed = self.client.post(
            '/platform/api/images', headers=headers | {'content-type': 'multipart/form-data'}, content=b'broken'
        )
        self.assertEqual(502, malformed.status_code)
        self.config.runtime_dir.write_text('not a directory', encoding='utf-8')
        failed = self.client.post('/platform/api/images', headers=headers, files={'images': ('a.jpg', b'jpeg')})
        self.assertEqual(502, failed.status_code)
        for response in (malformed, failed):
            self.assertEqual({'detail': '平台暂时无法完成操作，请重试。'}, response.json())

    def test_upload_service_failure_removes_staging_and_hides_private_errors(self) -> None:
        self.service.errors['upload'] = OSError('C:/private/staged-image.jpg')
        response = self.client.post(
            '/platform/api/images', headers=self.authenticate(), files={'images': ('a.jpg', b'jpeg')}
        )
        self.assertEqual(502, response.status_code)
        self.assertEqual({'detail': '平台暂时无法完成操作，请重试。'}, response.json())
        self.assertTrue(self.service.staged)
        self.assertTrue(all(not path.exists() for path in self.service.staged))

    def test_upload_copy_failure_removes_partial_staging(self) -> None:
        def fail_copy(source: object, destination: object) -> None:
            destination.write(b'partial')
            raise OSError('private upload path')

        with patch('xxtrain.platform.app.shutil.copyfileobj', side_effect=fail_copy):
            response = self.client.post(
                '/platform/api/images', headers=self.authenticate(), files={'images': ('a.jpg', b'jpeg')}
            )
        self.assertEqual(502, response.status_code)
        self.assertEqual({'detail': '平台暂时无法完成操作，请重试。'}, response.json())
        self.assertEqual([], list((self.config.runtime_dir / 'staging').iterdir()))
        self.assertEqual([], self.service.calls)

    def test_upload_form_close_error_is_sanitized(self) -> None:
        from starlette.datastructures import FormData

        original_close = FormData.close

        async def fail_close(form: FormData) -> None:
            await original_close(form)
            raise OSError('private upload spool path')

        with patch.object(FormData, 'close', fail_close):
            response = self.client.post(
                '/platform/api/images', headers=self.authenticate(), files={'images': ('a.jpg', b'jpeg')}
            )
        self.assertEqual(502, response.status_code)
        self.assertEqual({'detail': '平台暂时无法完成操作，请重试。'}, response.json())

    def test_cache_returns_derived_readiness_and_rejects_operational_failure(self) -> None:
        headers = self.authenticate()
        self.service.view_result = WorkspaceView('line-3', '三号现场', 50, 50, 50, True, True)
        for _ in range(2):
            response = self.client.post('/platform/api/detection/cache', headers=headers, json={})
            self.assertEqual(200, response.status_code)
            self.assertTrue(response.json()['detection_cache_ready'])
            self.assertNotIn('status', response.json())
        self.service.errors['cache'] = PlatformError('private cache directory')
        response = self.client.post('/platform/api/detection/cache', headers=headers, json={})
        self.assertEqual(502, response.status_code)
        self.assertNotIn('private cache', response.text)


class PlatformRealWorkflowHttpTest(unittest.TestCase):
    def test_uploaded_image_is_admitted_by_the_real_service_and_ineligible_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / 'workspace'
            (workspace / 'images').mkdir(parents=True)
            candidate = root / 'candidate.png'
            Image.new('RGB', (64, 48), 'white').save(candidate)
            config = WorkspaceConfig('line-3', '三号现场', 17, workspace, root / 'runtime', 'http://cvat.test')
            cvat = FakeCvat()
            service = AnnotationService(config, WorkspaceData(workspace), cvat, RuntimeCache(config.runtime_dir))
            with AsgiTestClient(create_app(config, service, cvat)) as client:
                client.get('/platform/')
                client.cookies.set('sessionid', 'active')
                headers = {'origin': 'http://testserver', 'x-xtrain-csrf': client.cookies.get('xxtrain_csrf')}
                uploaded = client.post(
                    '/platform/api/images',
                    headers=headers,
                    files=[('images', ('a.png', candidate.read_bytes())), ('images', ('broken.jpg', b'broken'))],
                )
                self.assertEqual(200, uploaded.status_code)
                self.assertEqual(1, uploaded.json()['image_count'])
                self.assertEqual(0, uploaded.json()['annotated_image_count'])
                self.assertFalse(uploaded.json()['can_generate_detection_cache'])
                self.assertEqual(1, len(list((workspace / 'images').iterdir())))
                self.assertEqual([], list((config.runtime_dir / 'staging').iterdir()))
                cache = client.post('/platform/api/detection/cache', headers=headers, json={})
                self.assertEqual(502, cache.status_code)

    def test_login_start_and_sync_use_real_workflow_and_workspace_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'
            images.mkdir()
            staged = root / 'frame.jpg'
            Image.new('RGB', (64, 48), 'white').save(staged)
            data = WorkspaceData(root)
            data.admit((staged,))
            sample_id = data.images()[0].sample_id
            runtime = RuntimeCache(root / 'runtime')
            runtime.remember_job('detect', data.detection_fingerprint(), JobRef(41, 73, (sample_id,)))
            config = WorkspaceConfig('line-3', '三号现场', 17, root, root / 'runtime', 'http://cvat.test')
            labels = [
                {'id': 41 + index, 'name': name, 'attributes': [{'id': 71 + index, 'name': 'xxtrain_labelme_extra'}]}
                for index, name in enumerate(POINT_BOX_LABELS)
            ]

            def respond(request: httpx.Request) -> httpx.Response:
                if request.url.path == '/api/auth/login':
                    return httpx.Response(
                        200,
                        json={'key': 'ignored'},
                        headers=[
                            ('set-cookie', 'csrftoken=cvat-csrf; Path=/; SameSite=Lax'),
                            ('set-cookie', 'sessionid=active; Path=/; HttpOnly; SameSite=Lax'),
                        ],
                    )
                if request.url.path == '/api/users/self':
                    return httpx.Response(200, json={'id': 17})
                if request.url.path == '/api/jobs/73':
                    return httpx.Response(200, json={'id': 73, 'state': 'in progress'})
                if request.url.path == '/api/labels':
                    return httpx.Response(200, json={'count': 5, 'next': None, 'results': labels})
                if request.url.path == '/api/jobs/73/annotations':
                    return httpx.Response(
                        200,
                        json={
                            'version': 0,
                            'tracks': [],
                            'tags': [],
                            'shapes': [
                                {
                                    'id': 91,
                                    'type': 'rectangle',
                                    'frame': 0,
                                    'label_id': 42,
                                    'points': [1, 2, 20, 30],
                                    'attributes': [{'spec_id': 72, 'value': '{"description":"kept"}'}],
                                }
                            ],
                        },
                    )
                return httpx.Response(404)

            http = httpx.Client(transport=httpx.MockTransport(respond))
            self.addCleanup(http.close)
            cvat = CvatClient(config.cvat_internal_url, 'service-secret', http)
            service = AnnotationService(config, data, cvat, runtime)
            with AsgiTestClient(create_app(config, service, cvat)) as client:
                client.get('/platform/')
                token = client.cookies.get('xxtrain_csrf')
                headers = {'origin': 'http://testserver', 'x-xtrain-csrf': token}
                login = client.post(
                    '/platform/api/login', headers=headers, json={'username': 'worker', 'password': 'password'}
                )
                start = client.post('/platform/api/detection/start', headers=headers, json={})
                saved = client.post('/platform/api/detection/sync', headers=headers, json={})

            records = AnnotationRepository(root / 'annotations.db', point_task_definition()).annotations()
            self.assertEqual(204, login.status_code)
            self.assertEqual({'annotation_url': '/tasks/41/jobs/73'}, start.json())
            self.assertEqual(1, saved.json()['annotated_image_count'])
            self.assertEqual(1, saved.json()['boxed_image_count'])
            self.assertFalse((root / 'state.json').exists())
            self.assertEqual(['tl'], [record.label for record in records])
            self.assertEqual([[[1.0, 2.0], [20.0, 30.0]]], [record.geometry for record in records])


class PlatformEntrypointTest(unittest.TestCase):
    def test_composition_owns_and_closes_the_dedicated_cvat_http_client(self) -> None:
        from xxtrain.platform.__main__ import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            config_path = root / 'workspace.json'
            config_path.write_text(
                json.dumps(
                    {
                        'workspace_id': 'line-3',
                        'display_name': '三号现场',
                        'owner_user_id': 17,
                        'workspace_dir': '.',
                        'runtime_dir': 'runtime',
                        'cvat_internal_url': 'http://cvat.test:8080',
                    }
                ),
                encoding='utf-8',
            )
            events: list[object] = []

            class RecordingHttp:
                def __enter__(self) -> 'RecordingHttp':
                    events.append('http-enter')
                    return self

                def __exit__(self, *args: object) -> None:
                    events.append('http-close')

            http = RecordingHttp()

            def make_cvat(base_url: str, token: str, http: object) -> FakeCvat:
                events.append(('cvat', base_url, token, http))
                return FakeCvat()

            def run_server(app: object, **kwargs: object) -> None:
                events.append(('serve', kwargs))

            with (
                patch.dict(os.environ, {'XXTRAIN_CVAT_SERVICE_TOKEN': 'service-secret'}, clear=False),
                patch('xxtrain.platform.__main__.httpx.Client', return_value=http),
                patch('xxtrain.platform.__main__.CvatClient', side_effect=make_cvat),
                patch('xxtrain.platform.__main__.uvicorn.run', side_effect=run_server),
            ):
                main(['--config', str(config_path), '--host', '127.0.0.1', '--port', '18001'])

        self.assertEqual('http-enter', events[0])
        self.assertEqual('cvat', events[1][0])
        self.assertEqual('http://cvat.test:8080', events[1][1])
        self.assertEqual('service-secret', events[1][2])
        self.assertIs(http, events[1][3])
        self.assertEqual(
            (
                'serve',
                {'host': '127.0.0.1', 'port': 18001, 'workers': 1, 'proxy_headers': True, 'forwarded_allow_ips': '*'},
            ),
            events[2],
        )
        self.assertEqual('http-close', events[3])


if __name__ == '__main__':
    unittest.main()
