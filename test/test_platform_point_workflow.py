from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import httpx
except ModuleNotFoundError as error:
    raise unittest.SkipTest('platform extra is not installed') from error

from PIL import Image

from xxtrain.business_tasks.point import point_task_definition
from xxtrain.integrations.cvat import CvatClient
from xxtrain.platform.config import load_config
from xxtrain.platform.contracts import PlatformError
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.service import AnnotationService
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.repository import AnnotationRepository

from .platform_fixture import FIXTURE_MARKER, create_fixture


def _page(results: list[dict]) -> dict:
    return {'count': len(results), 'next': None, 'results': results}


class HttpxCvatFixture:
    """Stateful CVAT API fixture used through the production HTTPX adapter."""

    def __init__(self) -> None:
        self.tasks: dict[int, dict] = {}
        self.jobs: dict[int, dict] = {}
        self.expected_frames = ()
        self.next_task_id = 100
        self.next_object_id = 1000
        self.http = httpx.Client(transport=httpx.MockTransport(self.respond))
        self.client = CvatClient('http://cvat.test', 'fixture-token', self.http)

    def close(self) -> None:
        self.http.close()

    def expect(self, frames) -> None:
        self.expected_frames = tuple(frames)

    def job(self, target: str) -> dict:
        matching = [job for job in self.jobs.values() if self.tasks[job['task_id']]['target'] == target]
        return matching[-1]

    def set_annotations(self, target: str, *, tags: list[dict] | None = None, shapes: list[dict] | None = None) -> None:
        self.job(target)['annotations'] = {
            'version': 0,
            'tags': copy.deepcopy(tags or []),
            'shapes': copy.deepcopy(shapes or []),
            'tracks': [],
        }

    def label_id(self, target: str, name: str) -> int:
        task = self.tasks[self.job(target)['task_id']]
        return next(label['id'] for label in task['labels'] if label['name'] == name)

    def respond(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == 'POST' and path == '/api/tasks':
            payload = json.loads(request.content)
            task_id = self.next_task_id
            self.next_task_id += 1
            labels = []
            for index, supplied in enumerate(payload['labels']):
                labels.append(
                    {
                        **supplied,
                        'id': task_id * 10 + index,
                        'attributes': [{**supplied['attributes'][0], 'id': task_id * 100 + index}],
                    }
                )
            target = {'rectangle': 'detect', 'tag': 'classify', 'polyline': 'segment'}[labels[0]['type']]
            self.tasks[task_id] = {'id': task_id, 'target': target, 'labels': labels, 'frames': None, 'job_id': None}
            return httpx.Response(201, json={'id': task_id})
        if path.startswith('/api/tasks/') and path.endswith('/data') and request.method == 'POST':
            task_id = int(path.split('/')[3])
            task = self.tasks[task_id]
            task['frames'] = tuple(self.expected_frames)
            job_id = task_id + 1000
            task['job_id'] = job_id
            self.jobs[job_id] = {
                'id': job_id,
                'task_id': task_id,
                'state': 'annotation',
                'annotations': {'version': 0, 'tags': [], 'shapes': [], 'tracks': []},
            }
            return httpx.Response(202, json={'rq_id': f'upload-{task_id}'})
        if path.startswith('/api/requests/upload-'):
            return httpx.Response(200, json={'status': 'finished'})
        if path.startswith('/api/tasks/') and path.endswith('/data/meta'):
            task_id = int(path.split('/')[3])
            frames = self.tasks[task_id]['frames']
            return httpx.Response(
                200,
                json={
                    'size': len(frames),
                    'start_frame': 0,
                    'stop_frame': len(frames) - 1,
                    'deleted_frames': [],
                    'frames': [
                        {
                            'name': f'{index:08d}{Path(frame.image_path).suffix.lower()}',
                            'width': frame.width,
                            'height': frame.height,
                        }
                        for index, frame in enumerate(frames)
                    ],
                },
            )
        if path.startswith('/api/tasks/') and request.method == 'GET':
            task_id = int(path.split('/')[3])
            frames = self.tasks[task_id]['frames']
            return httpx.Response(200, json={'id': task_id, 'size': None if frames is None else len(frames)})
        if path == '/api/jobs' and request.method == 'GET':
            task_id = int(request.url.params['task_id'])
            task = self.tasks[task_id]
            frames = task['frames']
            result = []
            if task['job_id'] is not None:
                result.append(
                    {
                        'id': task['job_id'],
                        'task_id': task_id,
                        'type': 'annotation',
                        'start_frame': 0,
                        'stop_frame': len(frames) - 1,
                        'frame_count': len(frames),
                    }
                )
            return httpx.Response(200, json=_page(result))
        if path == '/api/labels' and request.method == 'GET':
            return httpx.Response(200, json=_page(self.tasks[int(request.url.params['task_id'])]['labels']))
        if path.startswith('/api/jobs/') and path.endswith('/annotations'):
            job = self.jobs[int(path.split('/')[3])]
            if request.method == 'PUT':
                payload = json.loads(request.content)
                for item in [*payload.get('tags', []), *payload.get('shapes', [])]:
                    if 'id' not in item:
                        item['id'] = self.next_object_id
                        self.next_object_id += 1
                job['annotations'] = payload
            return httpx.Response(200, json=copy.deepcopy(job['annotations']))
        if path.startswith('/api/jobs/'):
            job = self.jobs[int(path.split('/')[3])]
            if request.method == 'PATCH':
                job.update(json.loads(request.content))
            return httpx.Response(200, json={'id': job['id'], 'state': job['state']})
        return httpx.Response(404, json={'detail': 'unhandled fixture request'})


class PointWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.receipt = create_fixture(Path(temporary.name), owner_user_id=17, cvat_internal_url='http://cvat.test')
        self.config = load_config(Path(self.receipt['config_path']))
        self.data = WorkspaceData(self.config.workspace_dir)
        self.repository = AnnotationRepository(Path(self.receipt['database_path']), point_task_definition())
        self.repository.save_annotations(
            (), delete_ids=frozenset(record.id for record in self.repository.annotations())
        )
        self.cvat = HttpxCvatFixture()
        self.addCleanup(self.cvat.close)
        self.service = AnnotationService(
            self.config, self.data, self.cvat.client, RuntimeCache(self.config.runtime_dir)
        )

    @staticmethod
    def _targets(view):
        return {target.id: target for target in view.targets}

    def _complete_workflow(self) -> dict[str, object]:
        self.cvat.expect(self.data.images())
        self.service.begin_target(17, 'detect')
        detect_shapes = [
            {
                'id': 10_000 + frame,
                'type': 'rectangle',
                'frame': frame,
                'label_id': self.cvat.label_id('detect', 'tl'),
                'points': [40, 30, 280, 210],
                'attributes': [],
            }
            for frame in range(50)
        ]
        self.cvat.set_annotations('detect', shapes=detect_shapes)
        detect_view = self.service.sync_target(17, 'detect')

        self.cvat.expect(self.data.target_frames('classify', self.config.runtime_dir))
        self.service.begin_target(17, 'classify')
        classify_tags = [
            {
                'id': 20_000 + frame,
                'frame': frame,
                'label_id': self.cvat.label_id('classify', ('tl', 'tc', 'cl', 'cc')[frame % 4]),
                'attributes': [],
            }
            for frame in range(50)
        ]
        self.cvat.set_annotations('classify', tags=classify_tags)
        classify_view = self.service.sync_target(17, 'classify')

        self.cvat.expect(self.data.target_frames('segment', self.config.runtime_dir))
        self.service.begin_target(17, 'segment')
        segment_shapes = [self._line(frame, 30_000 + frame, [40, 40, 160, 120]) for frame in range(50)]
        segment_shapes.append(self._line(0, 31_000, [50, 130, 170, 50]))
        self.cvat.set_annotations('segment', shapes=segment_shapes)
        segment_view = self.service.sync_target(17, 'segment')
        return {
            'detect_view': detect_view,
            'classify_view': classify_view,
            'segment_view': segment_view,
            'detect_shapes': detect_shapes,
            'classify_tags': classify_tags,
            'segment_shapes': segment_shapes,
        }

    def _line(self, frame: int, object_id: int, points: list[int]) -> dict:
        return {
            'id': object_id,
            'type': 'polyline',
            'frame': frame,
            'label_id': self.cvat.label_id('segment', '1'),
            'points': points,
            'attributes': [],
        }

    def test_detect_classify_segment_and_both_caches_leave_sqlite_unchanged(self) -> None:
        views = self._complete_workflow()

        self.assertTrue(self._targets(views['detect_view'])['classify'].can_annotate)
        self.assertTrue(self._targets(views['classify_view'])['segment'].can_annotate)
        self.assertTrue(self._targets(views['segment_view'])['segment'].can_generate_cache)
        before = self.repository.annotations()
        jobs = {target: self._runtime_job(target) for target in ('detect', 'classify', 'segment')}
        bindings_before = {
            target: self.repository.bindings(job.ref if hasattr(job, 'ref') else job) for target, job in jobs.items()
        }

        class_cache = self.service.generate_target_cache(17, 'classify')
        segment_cache = self.service.generate_target_cache(17, 'segment')

        self.assertEqual(before, self.repository.annotations())
        for target, job in jobs.items():
            ref = job.ref if hasattr(job, 'ref') else job
            self.assertEqual(bindings_before[target], self.repository.bindings(ref))
        self.assertTrue(self._targets(class_cache)['classify'].cache_ready)
        self.assertTrue(self._targets(segment_cache)['segment'].cache_ready)
        for target in ('classify', 'segment'):
            fingerprint = self.data.target_fingerprint(target)
            publication = self.config.runtime_dir / 'cache' / fingerprint
            self.assertEqual(
                {'fingerprint': fingerprint, 'target': target},
                json.loads((publication / 'manifest.json').read_text(encoding='utf-8')),
            )
            image_path = next(
                path for path in (publication / target).rglob('*') if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}
            )
            with Image.open(image_path) as image:
                image.verify()

    def test_identity_invalidation_stale_restart_and_rollback_retry_are_object_scoped(self) -> None:
        values = self._complete_workflow()
        before_noop = self.repository.annotations()
        segment_job = self._runtime_job('segment')
        bindings_noop = self.repository.bindings(segment_job.ref)

        self.service.sync_target(17, 'segment')

        self.assertEqual(before_noop, self.repository.annotations())
        self.assertEqual(bindings_noop, self.repository.bindings(segment_job.ref))

        parent = next(record for record in before_noop if record.step_key == 'detect')
        before_parent_records = tuple(
            record for record in before_noop if record.parent_id != parent.id and record != parent
        )
        pointer_bindings = {binding.object_id: binding.annotation_id for binding in bindings_noop}
        original_parent_line_ids = {
            record.id for record in before_noop if record.parent_id == parent.id and record.step_key == 'segment'
        }
        unrelated_pointer_bindings = tuple(
            binding for binding in bindings_noop if binding.annotation_id not in original_parent_line_ids
        )
        segment_shapes = copy.deepcopy(values['segment_shapes'])
        segment_shapes[0]['points'] = [45, 45, 165, 125]
        segment_shapes = [shape for shape in segment_shapes if shape['id'] != 31_000]
        segment_shapes.extend((self._line(0, 31_001, [55, 135, 175, 55]), self._line(0, 31_002, [60, 140, 180, 60])))
        self.cvat.set_annotations('segment', shapes=segment_shapes)

        self.service.sync_target(17, 'segment')

        after_pointers = self.repository.annotations()
        self.assertEqual(
            before_parent_records,
            tuple(record for record in after_pointers if record.parent_id != parent.id and record != parent),
        )
        parent_lines = tuple(
            record for record in after_pointers if record.parent_id == parent.id and record.step_key == 'segment'
        )
        self.assertEqual(3, len(parent_lines))
        self.assertIn(pointer_bindings[30_000], {record.id for record in parent_lines})
        self.assertNotIn(pointer_bindings[31_000], {record.id for record in parent_lines})
        current_segment_job = self._runtime_job('segment')
        self.assertEqual(
            unrelated_pointer_bindings,
            tuple(
                binding
                for binding in self.repository.bindings(current_segment_job.ref)
                if binding.annotation_id not in {record.id for record in parent_lines}
            ),
        )

        classify_tags = copy.deepcopy(values['classify_tags'])
        classify_tags[0]['label_id'] = self.cvat.label_id('classify', 'cc')
        self.cvat.set_annotations('classify', tags=classify_tags)
        before_class_change = self.repository.annotations()
        classify_job = self._runtime_job('classify')
        class_bindings_before = self.repository.bindings(classify_job.ref)
        parent_class_id = next(
            record.id
            for record in before_class_change
            if record.parent_id == parent.id and record.step_key == 'classify'
        )
        self.service.sync_target(17, 'classify')
        after_class_change = self.repository.annotations()
        unaffected = tuple(record for record in before_class_change if record.parent_id != parent.id)
        self.assertEqual(unaffected, tuple(record for record in after_class_change if record.parent_id != parent.id))
        self.assertFalse(
            any(record.step_key == 'segment' and record.parent_id == parent.id for record in after_class_change)
        )
        current_classify_job = self._runtime_job('classify')
        self.assertEqual(
            tuple(binding for binding in class_bindings_before if binding.annotation_id != parent_class_id),
            tuple(
                binding
                for binding in self.repository.bindings(current_classify_job.ref)
                if binding.annotation_id != parent_class_id
            ),
        )

        detect_shapes = copy.deepcopy(values['detect_shapes'])
        detect_shapes[0]['points'] = [42, 31, 279, 209]
        self.cvat.set_annotations('detect', shapes=detect_shapes)
        before_parent_change = self.repository.annotations()
        detect_job = self._runtime_job('detect')
        detect_bindings_before = self.repository.bindings(detect_job)
        self.service.sync_target(17, 'detect')
        after_parent_change = self.repository.annotations()
        changed_parent = next(
            record
            for record in after_parent_change
            if record.step_key == 'detect' and record.image_id == parent.image_id
        )
        self.assertEqual(parent.id, changed_parent.id)
        self.assertEqual(
            tuple(record for record in before_parent_change if record.image_id != parent.image_id),
            tuple(record for record in after_parent_change if record.image_id != parent.image_id),
        )
        current_detect_job = self._runtime_job('detect')
        self.assertEqual(
            tuple(binding for binding in detect_bindings_before if binding.sample_id != parent.image_id),
            tuple(
                binding
                for binding in self.repository.bindings(current_detect_job)
                if binding.sample_id != parent.image_id
            ),
        )
        with self.assertRaisesRegex(PlatformError, 'not ready'):
            self.service.sync_target(17, 'classify')

        self.cvat.expect(self.data.target_frames('classify', self.config.runtime_dir))
        self.service.begin_target(17, 'classify')
        new_classify_job = self._runtime_job('classify')
        tags = copy.deepcopy(self.cvat.job('classify')['annotations']['tags'])
        tags.append({'id': 21_000, 'frame': 0, 'label_id': self.cvat.label_id('classify', 'tl'), 'attributes': []})
        self.cvat.set_annotations('classify', tags=tags)
        before_failure = self.repository.annotations()
        bindings_before_failure = self.repository.bindings(new_classify_job.ref)
        with patch.object(self.data._repository, 'apply_changes', side_effect=PlatformError('injected rollback')):
            with self.assertRaisesRegex(PlatformError, '取回或保存'):
                self.service.sync_target(17, 'classify')
        self.assertEqual(before_failure, self.repository.annotations())
        self.assertEqual(bindings_before_failure, self.repository.bindings(new_classify_job.ref))

        restarted = AnnotationService(
            self.config,
            WorkspaceData(self.config.workspace_dir),
            self.cvat.client,
            RuntimeCache(self.config.runtime_dir),
        )
        retried = restarted.sync_target(17, 'classify')
        self.assertEqual(50, self._targets(retried)['classify'].annotated_sample_count)
        self.assertEqual(
            tuple(record for record in before_failure if record.image_id != parent.image_id),
            tuple(record for record in self.repository.annotations() if record.image_id != parent.image_id),
        )

    def _runtime_job(self, target: str):
        runtime = RuntimeCache(self.config.runtime_dir)
        fingerprint = self.data.detection_fingerprint() if target == 'detect' else self.data.target_fingerprint(target)
        job = runtime.job_for(target, fingerprint) if target == 'detect' else runtime.edit_job_for(target, fingerprint)
        self.assertIsNotNone(job)
        return job


class PointWorkflowFixtureTest(unittest.TestCase):
    def test_fixture_is_fresh_complete_and_records_explicit_identity(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            receipt = create_fixture(Path(parent), owner_user_id=17, cvat_internal_url='http://cvat.test')
            repository = AnnotationRepository(Path(receipt['database_path']), point_task_definition())

            self.assertEqual('xxtrain-task-7-point-workflow-v1', FIXTURE_MARKER)
            self.assertEqual(50, len(receipt['images']))
            self.assertEqual({'detect', 'classify', 'segment'}, set(receipt['initial_annotation_ids']))
            for target, ids in receipt['initial_annotation_ids'].items():
                self.assertEqual(ids, [str(record.id) for record in repository.annotations(step_key=target)])
            self.assertEqual(50, len(repository.annotations(step_key='detect')))
            self.assertEqual(50, len(repository.annotations(step_key='classify')))
            self.assertEqual(51, len(repository.annotations(step_key='segment')))


if __name__ == '__main__':
    unittest.main()
