import json
import subprocess
import sys
import tempfile
import textwrap
import traceback
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

try:
    import httpx
except ModuleNotFoundError as error:
    raise unittest.SkipTest('platform extra is not installed') from error

from xxtrain.business_tasks import POINT_BOX_LABELS
from xxtrain.data import Bbox
from xxtrain.integrations.cvat import CvatClient, PreparationState
from xxtrain.platform.contracts import (
    CvatBinding,
    DetectionBox,
    EditAnnotation,
    EditFrame,
    EditJob,
    FrameMapping,
    FrameResult,
    ImageInput,
    JobRef,
    PlatformAccessError,
    PlatformError,
    PreparedJob,
)

LABELS = [
    {
        'id': 41 + index,
        'name': name,
        'attributes': [
            {
                'id': 71 + index,
                'name': 'xxtrain_labelme_extra',
                'mutable': True,
                'input_type': 'text',
                'default_value': '{}',
                'values': [],
            }
        ],
    }
    for index, name in enumerate(POINT_BOX_LABELS)
]


def page(results: list[dict], *, next_url: str | None = None) -> dict:
    return {'count': len(results), 'next': next_url, 'results': results}


class CvatClientTest(unittest.TestCase):
    def test_base_url_rejects_username_or_password(self):
        http = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500)))

        for base_url in ('http://user@cvat.test', 'http://:password@cvat.test'):
            with self.subTest(base_url=base_url):
                with self.assertRaisesRegex(ValueError, 'username or password'):
                    CvatClient(base_url, 'private-token', http)

    def test_codec_and_preparation_values_import_without_httpx(self):
        script = textwrap.dedent(
            """
            import builtins

            original_import = builtins.__import__

            def import_without_httpx(name, *args, **kwargs):
                if name == 'httpx' or name.startswith('httpx.'):
                    raise ModuleNotFoundError("blocked optional dependency 'httpx'")
                return original_import(name, *args, **kwargs)

            builtins.__import__ = import_without_httpx
            from xxtrain.integrations.cvat import PreparationState
            from xxtrain.integrations.cvat.codec import encode_annotations

            assert PreparationState().stage == 'new'
            assert callable(encode_annotations)
            """
        )

        completed = subprocess.run([sys.executable, '-c', script], text=True, capture_output=True, check=False)

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_create_task_uses_service_token_and_exact_point_label_schema(self):
        seen = []

        def respond(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            self.assertEqual(request.method, 'POST')
            self.assertEqual(str(request.url), 'http://cvat.test/api/tasks')
            self.assertEqual(request.headers['authorization'], 'Token private-token')
            self.assertEqual(request.headers['accept'], 'application/vnd.cvat+json')
            self.assertNotIn('cookie', request.headers)
            payload = json.loads(request.content)
            self.assertEqual(payload['name'], 'Point detection')
            self.assertEqual(
                payload['labels'],
                [
                    {
                        'name': name,
                        'type': 'rectangle',
                        'attributes': [
                            {
                                'name': 'xxtrain_labelme_extra',
                                'mutable': True,
                                'input_type': 'text',
                                'default_value': '{}',
                                'values': [],
                            }
                        ],
                    }
                    for name in POINT_BOX_LABELS
                ],
            )
            return httpx.Response(201, json={'id': 7})

        http = httpx.Client(transport=httpx.MockTransport(respond), cookies={'browser': 'must-not-leak'})
        client = CvatClient('http://cvat.test', 'private-token', http)

        self.assertEqual(client.create_task('Point detection', POINT_BOX_LABELS), 7)
        self.assertEqual(len(seen), 1)

    def test_create_edit_task_sends_explicit_tag_label_type(self):
        def respond(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, 'POST')
            self.assertEqual(
                json.loads(request.content)['labels'],
                [
                    {
                        'name': 'tl',
                        'type': 'tag',
                        'attributes': [
                            {
                                'name': 'xxtrain_labelme_extra',
                                'mutable': True,
                                'input_type': 'text',
                                'default_value': '{}',
                                'values': [],
                            }
                        ],
                    }
                ],
            )
            return httpx.Response(201, json={'id': 17})

        client = CvatClient('http://cvat.test', 'private-token', httpx.Client(transport=httpx.MockTransport(respond)))

        self.assertTrue(hasattr(client, 'create_edit_task'), 'typed CVAT task creation is not implemented')
        self.assertEqual(client.create_edit_task('Point classification', ('tl',), 'tag'), 17)

    def test_prepare_uploads_numbered_images_initializes_annotations_and_assigns_job(self):
        checkpoints = []
        requests = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / 'camera-a.jpg'
            second = root / 'camera-b.png'
            first.write_bytes(b'first-image')
            second.write_bytes(b'second-image')
            first_annotation = UUID('11111111-1111-1111-1111-111111111111')
            second_annotation = UUID('22222222-2222-2222-2222-222222222222')
            images = (
                ImageInput(
                    'sample-b',
                    first,
                    100,
                    80,
                    (
                        DetectionBox(
                            Bbox(id=first_annotation, label='tl', x1=1, y1=2, x2=11, y2=12), {'description': 'existing'}
                        ),
                    ),
                ),
                ImageInput(
                    'sample-a',
                    second,
                    60,
                    40,
                    (DetectionBox(Bbox(id=second_annotation, label='tc', x1=2, y1=3, x2=12, y2=13)),),
                ),
            )

            def respond(request: httpx.Request) -> httpx.Response:
                requests.append((request.method, request.url.path, request.url.query, request.content))
                if request.url.path == '/api/tasks/7':
                    return httpx.Response(200, json={'id': 7})
                if request.url.path == '/api/tasks/7/data':
                    body = request.content
                    self.assertIn(b'name="client_files[0]"; filename="00000000.jpg"', body)
                    self.assertIn(b'name="client_files[1]"; filename="00000001.png"', body)
                    self.assertIn(b'first-image', body)
                    self.assertIn(b'second-image', body)
                    self.assertIn(b'name="sorting_method"', body)
                    self.assertIn(b'lexicographical', body)
                    return httpx.Response(202, json={'rq_id': 'upload-7'})
                if request.url.path == '/api/requests/upload-7':
                    return httpx.Response(200, json={'id': 'upload-7', 'status': 'finished'})
                if request.url.path == '/api/tasks/7/data/meta':
                    return httpx.Response(
                        200,
                        json={
                            'size': 2,
                            'start_frame': 0,
                            'stop_frame': 1,
                            'deleted_frames': [],
                            'included_frames': None,
                            'frames': [
                                {'name': '00000000.jpg', 'width': 100, 'height': 80, 'related_files': 0},
                                {'name': '00000001.png', 'width': 60, 'height': 40, 'related_files': 0},
                            ],
                        },
                    )
                if request.url.path == '/api/jobs':
                    self.assertEqual(request.url.params['task_id'], '7')
                    return httpx.Response(
                        200,
                        json=page(
                            [
                                {
                                    'id': 8,
                                    'task_id': 7,
                                    'type': 'annotation',
                                    'start_frame': 0,
                                    'stop_frame': 1,
                                    'frame_count': 2,
                                }
                            ]
                        ),
                    )
                if request.url.path == '/api/labels':
                    self.assertEqual(request.url.params['task_id'], '7')
                    return httpx.Response(200, json=page(LABELS))
                if request.url.path == '/api/jobs/8/annotations':
                    payload = json.loads(request.content)
                    self.assertEqual(payload['shapes'][0]['frame'], 0)
                    self.assertEqual(payload['shapes'][0]['label_id'], 42)
                    extras = [json.loads(shape['attributes'][0]['value']) for shape in payload['shapes']]
                    self.assertEqual('existing', extras[0]['description'])
                    self.assertEqual(str(first_annotation), extras[0]['xxtrain_annotation_id'])
                    self.assertEqual(str(second_annotation), extras[1]['xxtrain_annotation_id'])
                    payload['shapes'] = [payload['shapes'][1] | {'id': 92}, payload['shapes'][0] | {'id': 91}]
                    return httpx.Response(200, json=payload)
                if request.url.path == '/api/jobs/8':
                    self.assertEqual(json.loads(request.content), {'assignee': 23})
                    return httpx.Response(200, json={'id': 8})
                return httpx.Response(404)

            http = httpx.Client(transport=httpx.MockTransport(respond))
            client = CvatClient('http://cvat.test', 'private-token', http)

            prepared = client.prepare_task(7, images, 23, checkpoint=checkpoints.append)

        self.assertEqual(JobRef(7, 8, ('sample-b', 'sample-a')), prepared.ref)
        self.assertEqual(
            {
                CvatBinding('sample-b', 'shape', 91, first_annotation),
                CvatBinding('sample-a', 'shape', 92, second_annotation),
            },
            set(prepared.bindings),
        )
        self.assertEqual(
            checkpoints,
            [
                PreparationState('uploading'),
                PreparationState('uploading', 'upload-7'),
                PreparationState('uploaded'),
                PreparationState('initializing'),
                PreparationState('initialized'),
            ],
        )
        self.assertEqual(
            [(method, path) for method, path, _, _ in requests],
            [
                ('GET', '/api/tasks/7'),
                ('POST', '/api/tasks/7/data'),
                ('GET', '/api/requests/upload-7'),
                ('GET', '/api/tasks/7/data/meta'),
                ('GET', '/api/jobs'),
                ('GET', '/api/labels'),
                ('PUT', '/api/jobs/8/annotations'),
                ('PATCH', '/api/jobs/8'),
            ],
        )

    def test_prepare_resumes_known_upload_request_without_uploading_again(self):
        checkpoints = []
        requests = []
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / 'a.jpg'
            image_path.write_bytes(b'image')
            images = (ImageInput('a', image_path, 10, 20, ()),)

            def respond(request: httpx.Request) -> httpx.Response:
                requests.append((request.method, request.url.path))
                responses = {
                    ('GET', '/api/tasks/7'): httpx.Response(200, json={'id': 7, 'size': 0}),
                    ('GET', '/api/requests/saved-rq'): httpx.Response(
                        200, json={'id': 'saved-rq', 'status': 'finished'}
                    ),
                    ('GET', '/api/tasks/7/data/meta'): httpx.Response(
                        200,
                        json={
                            'size': 1,
                            'start_frame': 0,
                            'stop_frame': 0,
                            'deleted_frames': [],
                            'included_frames': None,
                            'frames': [{'name': '00000000.jpg', 'width': 10, 'height': 20, 'related_files': 0}],
                        },
                    ),
                    ('GET', '/api/jobs'): httpx.Response(
                        200,
                        json=page(
                            [
                                {
                                    'id': 8,
                                    'task_id': 7,
                                    'type': 'annotation',
                                    'start_frame': 0,
                                    'stop_frame': 0,
                                    'frame_count': 1,
                                }
                            ]
                        ),
                    ),
                    ('GET', '/api/labels'): httpx.Response(200, json=page(LABELS)),
                    ('PUT', '/api/jobs/8/annotations'): httpx.Response(200, json={}),
                    ('PATCH', '/api/jobs/8'): httpx.Response(200, json={'id': 8}),
                }
                return responses.get((request.method, request.url.path), httpx.Response(404))

            client = CvatClient(
                'http://cvat.test', 'private-token', httpx.Client(transport=httpx.MockTransport(respond))
            )
            ref = client.prepare_task(
                7, images, 23, preparation=PreparationState('uploading', 'saved-rq'), checkpoint=checkpoints.append
            )

        self.assertEqual(ref, PreparedJob(JobRef(7, 8, ('a',)), ()))
        self.assertNotIn(('POST', '/api/tasks/7/data'), requests)
        self.assertEqual(checkpoints[0], PreparationState('uploaded'))

    def test_prepare_edit_task_uploads_crop_frames_and_returns_original_identity_bindings(self):
        requests = []
        labels = [
            {
                'id': 41,
                'name': 'tl',
                'type': 'tag',
                'attributes': [
                    {
                        'id': 71,
                        'name': 'xxtrain_labelme_extra',
                        'mutable': True,
                        'input_type': 'text',
                        'default_value': '{}',
                        'values': [],
                    }
                ],
            }
        ]
        original_id = 'a' * 64
        first_parent = UUID('11111111-1111-1111-1111-111111111111')
        second_parent = UUID('22222222-2222-2222-2222-222222222222')
        first_annotation = UUID('33333333-3333-3333-3333-333333333333')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / 'opaque-first.png'
            second_path = root / 'opaque-second.png'
            first_path.write_bytes(b'first-crop')
            second_path.write_bytes(b'second-crop')
            frames = (
                EditFrame(
                    FrameMapping(str(first_parent), original_id, first_parent, (1, 2, 11, 12)),
                    first_path,
                    10,
                    10,
                    (EditAnnotation(first_annotation, 'classification', 'tl', None),),
                ),
                EditFrame(
                    FrameMapping(str(second_parent), original_id, second_parent, (20, 30, 40, 50)),
                    second_path,
                    20,
                    20,
                    (),
                ),
            )

            def respond(request: httpx.Request) -> httpx.Response:
                requests.append((request.method, request.url.path))
                if request.url.path == '/api/tasks/7':
                    return httpx.Response(200, json={'id': 7})
                if request.url.path == '/api/tasks/7/data':
                    self.assertIn(b'filename="00000000.png"', request.content)
                    self.assertIn(b'filename="00000001.png"', request.content)
                    self.assertIn(b'first-crop', request.content)
                    self.assertIn(b'second-crop', request.content)
                    return httpx.Response(202, json={'rq_id': 'edit-upload-7'})
                if request.url.path == '/api/requests/edit-upload-7':
                    return httpx.Response(200, json={'id': 'edit-upload-7', 'status': 'finished'})
                if request.url.path == '/api/tasks/7/data/meta':
                    return httpx.Response(
                        200,
                        json={
                            'size': 2,
                            'start_frame': 0,
                            'stop_frame': 1,
                            'deleted_frames': [],
                            'frames': [
                                {'name': '00000000.png', 'width': 10, 'height': 10},
                                {'name': '00000001.png', 'width': 20, 'height': 20},
                            ],
                        },
                    )
                if request.url.path == '/api/jobs':
                    return httpx.Response(
                        200,
                        json=page(
                            [
                                {
                                    'id': 8,
                                    'task_id': 7,
                                    'type': 'annotation',
                                    'start_frame': 0,
                                    'stop_frame': 1,
                                    'frame_count': 2,
                                }
                            ]
                        ),
                    )
                if request.url.path == '/api/labels':
                    return httpx.Response(200, json=page(labels))
                if request.url.path == '/api/jobs/8/annotations':
                    payload = json.loads(request.content)
                    self.assertEqual(payload['tags'][0]['frame'], 0)
                    payload['tags'][0]['id'] = 91
                    return httpx.Response(200, json=payload)
                if request.url.path == '/api/jobs/8':
                    self.assertEqual(json.loads(request.content), {'assignee': 23})
                    return httpx.Response(200, json={'id': 8})
                return httpx.Response(404)

            client = CvatClient(
                'http://cvat.test', 'private-token', httpx.Client(transport=httpx.MockTransport(respond))
            )
            self.assertTrue(hasattr(client, 'prepare_edit_task'), 'CVAT edit task preparation is not implemented')
            prepared = client.prepare_edit_task(7, frames, 23)

        self.assertEqual(prepared.ref, JobRef(7, 8, (original_id,)))
        self.assertEqual(prepared.bindings, (CvatBinding(original_id, 'tag', 91, first_annotation),))
        self.assertEqual(requests[-1], ('PATCH', '/api/jobs/8'))

    def test_prepare_does_not_repeat_initialized_annotations(self):
        requests = []
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / 'a.jpg'
            image_path.write_bytes(b'image')
            annotation_id = UUID('12345678-1234-5678-1234-567812345678')
            images = (
                ImageInput(
                    'a',
                    image_path,
                    10,
                    20,
                    (DetectionBox(Bbox(id=annotation_id, label='Point', x1=1, y1=2, x2=8, y2=9)),),
                ),
            )

            def respond(request: httpx.Request) -> httpx.Response:
                requests.append((request.method, request.url.path))
                if request.url.path == '/api/tasks/7':
                    return httpx.Response(200, json={'id': 7, 'size': 1})
                if request.url.path == '/api/tasks/7/data/meta':
                    return httpx.Response(
                        200,
                        json={
                            'size': 1,
                            'start_frame': 0,
                            'stop_frame': 0,
                            'deleted_frames': [],
                            'included_frames': None,
                            'frames': [{'name': '00000000.jpg', 'width': 10, 'height': 20, 'related_files': 0}],
                        },
                    )
                if request.url.path == '/api/jobs':
                    return httpx.Response(
                        200,
                        json=page(
                            [
                                {
                                    'id': 8,
                                    'task_id': 7,
                                    'type': 'annotation',
                                    'start_frame': 0,
                                    'stop_frame': 0,
                                    'frame_count': 1,
                                }
                            ]
                        ),
                    )
                if request.url.path == '/api/labels':
                    return httpx.Response(200, json=page(LABELS))
                if request.url.path == '/api/jobs/8/annotations':
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
                                    'label_id': 41,
                                    'points': [1, 2, 8, 9],
                                    'attributes': [
                                        {
                                            'spec_id': 71,
                                            'value': json.dumps({'xxtrain_annotation_id': str(annotation_id)}),
                                        }
                                    ],
                                }
                            ],
                        },
                    )
                if request.url.path == '/api/jobs/8':
                    return httpx.Response(200, json={'id': 8})
                return httpx.Response(404)

            client = CvatClient(
                'http://cvat.test', 'private-token', httpx.Client(transport=httpx.MockTransport(respond))
            )
            prepared = client.prepare_task(7, images, 23, preparation=PreparationState('initialized'))

        self.assertEqual(PreparedJob(JobRef(7, 8, ('a',)), (CvatBinding('a', 'shape', 91, annotation_id),)), prepared)
        self.assertNotIn(('PUT', '/api/jobs/8/annotations'), requests)
        self.assertIn(('GET', '/api/jobs/8/annotations'), requests)

    def test_prepare_missing_initialization_token_does_not_assign_job(self):
        requests = []
        annotation_id = UUID('12345678-1234-5678-1234-567812345678')
        images = (
            ImageInput(
                'a',
                Path('unused.jpg'),
                10,
                20,
                (DetectionBox(Bbox(id=annotation_id, label='Point', x1=1, y1=2, x2=8, y2=9)),),
            ),
        )
        client = CvatClient(
            'http://cvat.test',
            'private-token',
            httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500))),
        )

        def request(method, path, **kwargs):
            requests.append((method, path))
            if path == '/api/tasks/7':
                return httpx.Response(200, json={'id': 7, 'size': 1})
            if path == '/api/jobs/8/annotations':
                payload = json.loads(json.dumps(kwargs['json']))
                payload['shapes'][0]['id'] = 91
                payload['shapes'][0]['attributes'][0]['value'] = '{}'
                return httpx.Response(200, json=payload)
            if path == '/api/jobs/8':
                return httpx.Response(200, json={'id': 8})
            raise AssertionError(path)

        with (
            patch.object(client, '_service_request', side_effect=request),
            patch.object(client, '_verify_frames'),
            patch.object(client, '_wait_for_job', return_value=8),
            patch.object(client, '_labels', return_value=LABELS),
        ):
            with self.assertRaisesRegex(ValueError, 'token'):
                client.prepare_task(7, images, 23)

        self.assertNotIn(('PATCH', '/api/jobs/8'), requests)

    def test_prepare_stops_ambiguous_initialization_without_an_http_write(self):
        requests = []
        client = CvatClient(
            'http://cvat.test',
            'private-token',
            httpx.Client(
                transport=httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(500))
            ),
        )

        with self.assertRaisesRegex(PlatformError, 'task 7.*administrator'):
            client.prepare_task(7, (), 23, preparation=PreparationState('initializing'))

        self.assertEqual(requests, [])

    def test_prepare_stops_unknown_inflight_upload_when_server_has_no_data(self):
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            return httpx.Response(200, json={'id': 7, 'size': 0})

        client = CvatClient('http://cvat.test', 'private-token', httpx.Client(transport=httpx.MockTransport(respond)))
        with self.assertRaisesRegex(PlatformError, 'task 7.*upload.*administrator'):
            client.prepare_task(7, (), 23, preparation=PreparationState('uploading'))
        self.assertEqual(requests, [('GET', '/api/tasks/7')])

    def test_prepare_timeout_is_bounded_and_names_the_task(self):
        def respond(request: httpx.Request) -> httpx.Response:
            if request.url.path == '/api/tasks/7':
                return httpx.Response(200, json={'id': 7, 'size': 0})
            if request.url.path == '/api/requests/saved-rq':
                return httpx.Response(200, json={'id': 'saved-rq', 'status': 'queued'})
            return httpx.Response(404)

        client = CvatClient('http://cvat.test', 'private-token', httpx.Client(transport=httpx.MockTransport(respond)))
        with patch('xxtrain.integrations.cvat.client.time.monotonic', side_effect=[0.0, 0.0, 121.0]):
            with self.assertRaisesRegex(PlatformError, 'task 7.*120 seconds'):
                client.prepare_task(7, (), 23, preparation=PreparationState('uploading', 'saved-rq'))

    def test_fetch_uses_server_frame_mapping(self):
        def respond(request: httpx.Request) -> httpx.Response:
            if request.url.path == '/api/labels':
                return httpx.Response(200, json={'count': 0, 'next': None, 'results': []})
            if request.url.path == '/api/jobs/8/annotations':
                return httpx.Response(200, json={'version': 0, 'shapes': [], 'tracks': [], 'tags': []})
            return httpx.Response(404)

        http = httpx.Client(transport=httpx.MockTransport(respond))
        client = CvatClient('http://cvat.test', 'private-token', http)
        self.assertEqual(client.fetch_detection(JobRef(7, 8, ('b', 'a'))), (FrameResult('b', ()), FrameResult('a', ())))

    def test_fetch_edit_decodes_pointer_lines_through_the_real_client(self):
        labels = [
            {
                'id': 51,
                'name': '1',
                'type': 'polyline',
                'attributes': [{'id': 81, 'name': 'xxtrain_labelme_extra', 'input_type': 'text'}],
            }
        ]

        def respond(request: httpx.Request) -> httpx.Response:
            if request.url.path == '/api/labels':
                return httpx.Response(200, json=page(labels))
            if request.url.path == '/api/jobs/8/annotations':
                return httpx.Response(
                    200,
                    json={
                        'version': 0,
                        'tags': [],
                        'tracks': [],
                        'shapes': [
                            {
                                'id': 91,
                                'type': 'polyline',
                                'frame': 0,
                                'label_id': 51,
                                'points': [1, 2, 3, 4],
                                'occluded': False,
                                'outside': False,
                                'rotation': 0,
                                'z_order': 0,
                                'attributes': [],
                            }
                        ],
                    },
                )
            return httpx.Response(404)

        parent_id = UUID('11111111-1111-1111-1111-111111111111')
        mapping = FrameMapping(str(parent_id), 'a' * 64, parent_id, (10, 20, 30, 40))
        job = EditJob(JobRef(7, 8, ('a' * 64,)), (mapping,))
        client = CvatClient('http://cvat.test', 'private-token', httpx.Client(transport=httpx.MockTransport(respond)))

        self.assertTrue(hasattr(client, 'fetch_edit'), 'CVAT edit fetching is not implemented')
        results = client.fetch_edit(job)

        self.assertEqual(results[0].frame_id, str(parent_id))
        self.assertEqual(results[0].annotations, (EditAnnotation(None, 'polyline', '1', [[1.0, 2.0], [3.0, 4.0]], 91),))

    def test_fetch_edit_decodes_classification_tags_through_the_real_client(self):
        labels = [
            {
                'id': 41,
                'name': 'tl',
                'type': 'tag',
                'attributes': [{'id': 71, 'name': 'xxtrain_labelme_extra', 'input_type': 'text'}],
            }
        ]

        def respond(request: httpx.Request) -> httpx.Response:
            if request.url.path == '/api/labels':
                return httpx.Response(200, json=page(labels))
            if request.url.path == '/api/jobs/8/annotations':
                return httpx.Response(
                    200,
                    json={
                        'version': 0,
                        'tags': [{'id': 92, 'frame': 0, 'label_id': 41, 'attributes': []}],
                        'tracks': [],
                        'shapes': [],
                    },
                )
            return httpx.Response(404)

        parent_id = UUID('11111111-1111-1111-1111-111111111111')
        mapping = FrameMapping(str(parent_id), 'a' * 64, parent_id, (10, 20, 30, 40))
        job = EditJob(JobRef(7, 8, ('a' * 64,)), (mapping,))
        client = CvatClient('http://cvat.test', 'private-token', httpx.Client(transport=httpx.MockTransport(respond)))

        results = client.fetch_edit(job)

        self.assertEqual(results[0].frame_id, str(parent_id))
        self.assertEqual(results[0].annotations, (EditAnnotation(None, 'classification', 'tl', None, 92),))

    def test_job_is_unfinished_reads_state_without_exposing_response_details(self):
        for state, expected in (('new', True), ('in progress', True), ('completed', False)):
            with self.subTest(state=state):

                def respond(request):
                    self.assertEqual(('GET', '/api/jobs/8'), (request.method, request.url.path))
                    return httpx.Response(200, json={'id': 8, 'state': state})

                with httpx.Client(transport=httpx.MockTransport(respond)) as http:
                    client = CvatClient('http://cvat.test', 'private-token', http)
                    self.assertEqual(expected, client.job_is_unfinished(JobRef(7, 8, ('a',))))

    def test_job_is_unfinished_rejects_malformed_state_and_sanitizes_failures(self):
        responses = [httpx.Response(200, json=body) for body in ({}, [], {'state': None}, {'state': 1})]
        responses.append(httpx.Response(403, text='private-response'))
        for response in responses:
            with self.subTest(response=response):
                with httpx.Client(transport=httpx.MockTransport(lambda _: response)) as http:
                    client = CvatClient('http://cvat.test', 'private-token', http)
                    with self.assertRaises(PlatformError) as caught:
                        client.job_is_unfinished(JobRef(7, 8, ('a',)))
                    self.assertNotIsInstance(caught.exception, PlatformAccessError)
                    self.assertNotIn('private-response', str(caught.exception))

    def test_job_path_is_a_local_same_origin_path(self):
        client = CvatClient(
            'http://cvat.test',
            'private-token',
            httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500))),
        )
        self.assertEqual(client.job_path(JobRef(7, 8, ('a',))), '/tasks/7/jobs/8')

    def test_browser_session_calls_exclude_service_and_client_credentials(self):
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            self.assertNotIn('authorization', request.headers)
            self.assertEqual(request.headers['accept'], 'application/vnd.cvat+json')
            if request.url.path == '/api/auth/login':
                self.assertNotIn('cookie', request.headers)
                return httpx.Response(
                    200,
                    json={'key': 'rest-token-must-not-be-returned'},
                    headers=[
                        ('set-cookie', 'csrftoken=csrf-new; Path=/; SameSite=Lax'),
                        ('set-cookie', 'sessionid=session-new; Path=/; HttpOnly'),
                        ('set-cookie', 'tracking=drop-me; Path=/'),
                    ],
                )
            if request.url.path == '/api/users/self':
                self.assertEqual(request.headers['cookie'], 'sessionid=session-new; csrftoken=csrf-new')
                return httpx.Response(200, json={'id': 23, 'username': 'worker'})
            if request.url.path == '/api/auth/logout':
                self.assertEqual(request.headers['cookie'], 'sessionid=session-new; csrftoken=csrf-new')
                self.assertEqual(request.headers['x-csrftoken'], 'csrf-new')
                return httpx.Response(
                    200,
                    json={'detail': 'Successfully logged out.'},
                    headers=[
                        ('set-cookie', 'sessionid=""; expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=/'),
                        ('set-cookie', 'tracking=still-drop-me; Path=/'),
                    ],
                )
            return httpx.Response(404)

        http = httpx.Client(
            transport=httpx.MockTransport(respond),
            auth=('default-user', 'default-password'),
            cookies={'client-cookie': 'must-not-leak'},
            follow_redirects=True,
        )
        client = CvatClient('http://cvat.test', 'private-token', http)

        cookies = client.login('worker', 'password')
        self.assertEqual(
            cookies, ('csrftoken=csrf-new; Path=/; SameSite=Lax', 'sessionid=session-new; Path=/; HttpOnly')
        )
        self.assertEqual(client.current_user('sessionid=session-new; csrftoken=csrf-new'), 23)
        self.assertEqual(
            client.logout('sessionid=session-new; csrftoken=csrf-new', 'csrf-new'),
            ('sessionid=""; expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=/',),
        )
        self.assertEqual(
            [request.url.path for request in requests], ['/api/auth/login', '/api/users/self', '/api/auth/logout']
        )

    def test_http_errors_are_visible_without_response_secrets(self):
        client = CvatClient(
            'http://cvat.test',
            'private-token',
            httpx.Client(
                transport=httpx.MockTransport(lambda _: httpx.Response(401, text='server-secret-that-must-not-appear'))
            ),
        )

        with self.assertRaises(PlatformAccessError) as caught:
            client.current_user('sessionid=browser-secret')

        self.assertIn('/api/users/self', str(caught.exception))
        self.assertNotIn('server-secret', str(caught.exception))
        self.assertNotIn('browser-secret', str(caught.exception))

    def test_login_rejection_with_cvat_bad_request_is_an_access_error(self):
        client = CvatClient(
            'http://cvat.test',
            'private-token',
            httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(400, text='private-response'))),
        )

        with self.assertRaises(PlatformAccessError) as caught:
            client.login('worker', 'password')

        self.assertIn('/api/auth/login', str(caught.exception))
        self.assertNotIn('private-response', str(caught.exception))

    def test_service_token_access_failure_is_an_operational_error(self):
        client = CvatClient(
            'http://cvat.test',
            'private-token',
            httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(403, text='private-response'))),
        )

        with self.assertRaises(PlatformError) as caught:
            client.create_task('Point detection', POINT_BOX_LABELS)

        self.assertNotIsInstance(caught.exception, PlatformAccessError)
        self.assertIn('/api/tasks', str(caught.exception))
        self.assertNotIn('private-response', str(caught.exception))

    def test_transport_failures_are_operational_platform_errors(self):
        def fail(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError('network-secret-that-must-not-appear', request=request)

        client = CvatClient('http://cvat.test', 'private-token', httpx.Client(transport=httpx.MockTransport(fail)))
        with self.assertRaises(PlatformError) as caught:
            client.current_user('sessionid=browser-secret')

        self.assertNotIsInstance(caught.exception, PlatformAccessError)
        self.assertIn('/api/users/self', str(caught.exception))
        self.assertNotIn('network-secret', str(caught.exception))
        self.assertNotIn('browser-secret', str(caught.exception))
        rendered = ''.join(traceback.format_exception(caught.exception))
        self.assertNotIn('network-secret', rendered)
        self.assertNotIn('httpx.ConnectError', rendered)

    def test_browser_login_does_not_follow_external_redirects(self):
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(302, headers={'location': 'http://attacker.test/collect'})

        client = CvatClient(
            'http://cvat.test',
            'private-token',
            httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True),
        )
        with self.assertRaisesRegex(PlatformError, 'status 302'):
            client.login('worker', 'password')

        self.assertEqual([str(request.url) for request in requests], ['http://cvat.test/api/auth/login'])

    def test_external_label_pagination_is_rejected_without_forwarding_token(self):
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=page([], next_url='http://attacker.test/api/labels?page=2'))

        client = CvatClient('http://cvat.test', 'private-token', httpx.Client(transport=httpx.MockTransport(respond)))
        with self.assertRaisesRegex(PlatformError, 'pagination'):
            client.fetch_detection(JobRef(7, 8, ('a',)))
        self.assertEqual(len(requests), 1)


if __name__ == '__main__':
    unittest.main()
