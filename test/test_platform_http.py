import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
from PIL import Image

from xxtrain.business_tasks import POINT_BOX_LABELS
from xxtrain.integrations.cvat import CvatClient
from xxtrain.platform.app import create_app
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import PlatformAccessError, PlatformError, WorkspaceView
from xxtrain.platform.service import AnnotationService
from xxtrain.platform.state import StateStore
from xxtrain.workspace_data import WorkspaceData


class AsgiTestClient:
    def __init__(self, app: Any) -> None:
        self.loop = asyncio.new_event_loop()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver')

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
        self.view_result = WorkspaceView('line-3', '三号现场', 2, 'pending')
        self.errors: dict[str, Exception] = {}

    def _result(self, name: str, user_id: int) -> object:
        self.calls.append((name, user_id))
        if name in self.errors:
            raise self.errors[name]
        if name == 'begin':
            return '/tasks/41/jobs/73'
        return self.view_result

    def view(self, user_id: int) -> WorkspaceView:
        return self._result('view', user_id)  # type: ignore[return-value]

    def begin(self, user_id: int) -> str:
        return self._result('begin', user_id)  # type: ignore[return-value]

    def sync(self, user_id: int) -> WorkspaceView:
        return self._result('sync', user_id)  # type: ignore[return-value]


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
        self.config = WorkspaceConfig(
            workspace_id='line-3',
            display_name='三号现场',
            owner_user_id=17,
            images_dir=Path('images'),
            annotations_dir=Path('annotations'),
            state_path=Path('state.json'),
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
        return {'origin': 'http://testserver', 'x-xxtrain-csrf': token}

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
                'status': 'pending',
                'error': None,
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
            ('/platform/api/annotation/start', {}),
            ('/platform/api/annotation/sync', {}),
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
            ('/platform/api/annotation/start', {}),
            ('/platform/api/annotation/sync', {}),
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
        headers = {'origin': 'http://testserver', 'x-xxtrain-csrf': token}

        login = self.client.post(
            '/platform/api/login', headers=headers, json={'username': 'worker', 'password': 'password'}
        )
        session = self.client.get('/platform/api/session')
        logout = self.client.post('/platform/api/logout', headers=headers, json={})

        self.assertEqual(204, login.status_code)
        self.assertEqual({'authenticated': True, 'user_id': 17}, session.json())
        self.assertEqual(204, logout.status_code)
        self.assertEqual(['login', 'current_user', 'current_user', 'logout'], self.cvat.calls)

    def test_start_and_sync_accept_only_an_empty_object(self) -> None:
        headers = self.authenticate()
        invalid_payloads = (
            {'target': 'classify'},
            {'target': 'segment'},
            {'task_id': 41},
            {'job_id': 73},
            {'image_path': 'C:/private/image.jpg'},
            {'annotations_dir': '/private/annotations'},
        )
        for path in ('/platform/api/annotation/start', '/platform/api/annotation/sync'):
            for payload in invalid_payloads:
                with self.subTest(path=path, payload=payload):
                    response = self.client.post(path, headers=headers, json=payload)
                    self.assertEqual(422, response.status_code)
            with self.subTest(path=path, payload=None):
                response = self.client.post(path, headers=headers)
                self.assertEqual(422, response.status_code)
        self.assertEqual([], self.service.calls)

    def test_start_returns_only_the_server_owned_annotation_path(self) -> None:
        response = self.client.post('/platform/api/annotation/start', headers=self.authenticate(), json={})

        self.assertEqual(200, response.status_code)
        self.assertEqual({'annotation_url': '/tasks/41/jobs/73'}, response.json())
        self.assertEqual([('begin', 17)], self.service.calls)

    def test_sync_error_is_not_success(self) -> None:
        self.service.errors['sync'] = PlatformError('Could not save annotations at C:/private/site')
        response = self.client.post('/platform/api/annotation/sync', headers=self.authenticate(), json={})

        self.assertEqual(502, response.status_code)
        self.assertNotIn('saved', response.json().get('status', ''))
        self.assertNotIn('C:/private/site', response.text)

    def test_malformed_local_state_is_an_operational_error(self) -> None:
        self.service.errors['view'] = ValueError('Workspace state must be a JSON object')
        self.client.cookies.set('sessionid', 'active')
        response = self.client.get('/platform/api/workspace')

        self.assertEqual(502, response.status_code)

    def test_returned_get_never_syncs(self) -> None:
        self.client.cookies.set('sessionid', 'active')
        response = self.client.get('/platform/?returned=1')

        self.assertEqual(200, response.status_code)
        self.assertEqual([], self.service.calls)


class PlatformRealWorkflowHttpTest(unittest.TestCase):
    def test_login_start_and_sync_use_real_workflow_and_workspace_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'
            annotations = root / 'annotations'
            images.mkdir()
            annotations.mkdir()
            image_path = images / 'frame.jpg'
            Image.new('RGB', (64, 48), 'white').save(image_path)
            annotation_path = annotations / 'frame.json'
            annotation_path.write_text(
                json.dumps(
                    {'shapes': [{'label': 'mask', 'shape_type': 'polygon', 'points': [[0, 0], [1, 0], [1, 1]]}]}
                ),
                encoding='utf-8',
            )
            state = StateStore(root / 'state.json')
            state.save({'status': 'annotating', 'job': {'task_id': 41, 'job_id': 73, 'sample_ids': ['frame']}})
            config = WorkspaceConfig(
                'line-3', '三号现场', 17, images, annotations, root / 'state.json', 'http://cvat.test'
            )
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
            service = AnnotationService(config, WorkspaceData(images, annotations), cvat, state)
            with AsgiTestClient(create_app(config, service, cvat)) as client:
                client.get('/platform/')
                token = client.cookies.get('xxtrain_csrf')
                headers = {'origin': 'http://testserver', 'x-xxtrain-csrf': token}
                login = client.post(
                    '/platform/api/login', headers=headers, json={'username': 'worker', 'password': 'password'}
                )
                start = client.post('/platform/api/annotation/start', headers=headers, json={})
                saved = client.post('/platform/api/annotation/sync', headers=headers, json={})

            document = json.loads(annotation_path.read_text(encoding='utf-8'))
            self.assertEqual(204, login.status_code)
            self.assertEqual({'annotation_url': '/tasks/41/jobs/73'}, start.json())
            self.assertEqual('saved', saved.json()['status'])
            self.assertEqual(['mask', 'tl'], [shape['label'] for shape in document['shapes']])
            self.assertEqual('kept', document['shapes'][1]['description'])


class PlatformEntrypointTest(unittest.TestCase):
    def test_composition_owns_and_closes_the_dedicated_cvat_http_client(self) -> None:
        from xxtrain.platform.__main__ import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            (root / 'annotations').mkdir()
            config_path = root / 'workspace.json'
            config_path.write_text(
                json.dumps(
                    {
                        'workspace_id': 'line-3',
                        'display_name': '三号现场',
                        'owner_user_id': 17,
                        'images_dir': 'images',
                        'annotations_dir': 'annotations',
                        'state_path': 'state.json',
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
