from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

try:
    import httpx
except ModuleNotFoundError as error:
    raise unittest.SkipTest('platform extra is not installed') from error

from PIL import Image

from xxtrain.business_tasks.point import point_task_definition
from xxtrain.integrations.cvat import CvatClient
from xxtrain.platform.config import load_config
from xxtrain.platform.contracts import AnnotationRecord, CvatBinding, PlatformError
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
            return httpx.Response(200, json={'id': task_id, 'size': 0 if frames is None else len(frames)})
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
            if request.method == 'PATCH' and request.url.params.get('action') == 'create':
                payload = json.loads(request.content)
                for key in ('tags', 'shapes'):
                    for item in payload.get(key, []):
                        item['id'] = self.next_object_id
                        self.next_object_id += 1
                        job['annotations'][key].append(item)
                return httpx.Response(200, json=copy.deepcopy(payload))
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
        detect_shapes.insert(
            1,
            {
                'id': 10_500,
                'type': 'rectangle',
                'frame': 0,
                'label_id': self.cvat.label_id('detect', 'tc'),
                'points': [60, 50, 180, 150],
                'attributes': [],
            },
        )
        self.cvat.set_annotations('detect', shapes=detect_shapes)
        detect_view = self.service.sync_target(17, 'detect')

        classify_frames = self.data.target_frames('classify', self.config.runtime_dir)
        self.cvat.expect(classify_frames)
        self.service.begin_target(17, 'classify')
        classify_tags = [
            {
                'id': 20_000 + frame,
                'frame': frame,
                'label_id': self.cvat.label_id('classify', ('tl', 'tc', 'cl', 'cc')[frame % 4]),
                'attributes': [],
            }
            for frame in range(len(classify_frames))
        ]
        self.cvat.set_annotations('classify', tags=classify_tags)
        classify_view = self.service.sync_target(17, 'classify')

        segment_frames = self.data.target_frames('segment', self.config.runtime_dir)
        self.cvat.expect(segment_frames)
        self.service.begin_target(17, 'segment')
        segment_shapes = [self._line(frame, 30_000 + frame, [20, 20, 80, 80]) for frame in range(len(segment_frames))]
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
            'classify_frames': classify_frames,
            'segment_frames': segment_frames,
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
        detect_job = self._runtime_job('detect')
        classify_job = self._runtime_job('classify')
        segment_job = self._runtime_job('segment')
        detect_bindings = self.repository.bindings(detect_job)
        classify_bindings = self.repository.bindings(classify_job.ref)
        segment_bindings = self.repository.bindings(segment_job.ref)
        detect_ids = {binding.object_id: binding.annotation_id for binding in detect_bindings}
        expected_detection = tuple(
            AnnotationRecord(
                detect_ids[shape['id']],
                detect_job.sample_ids[shape['frame']],
                'detect',
                None,
                'rectangle',
                'tc' if shape['id'] == 10_500 else 'tl',
                [shape['points'][:2], shape['points'][2:]],
            )
            for shape in views['detect_shapes']
        )
        self.assertEqual(expected_detection, self.repository.annotations(step_key='detect'))
        self.assertEqual(
            tuple(
                CvatBinding(record.image_id, 'shape', shape['id'], record.id)
                for record, shape in zip(expected_detection, views['detect_shapes'], strict=True)
            ),
            detect_bindings,
        )

        class_ids = {binding.object_id: binding.annotation_id for binding in classify_bindings}
        class_names = ('tl', 'tc', 'cl', 'cc')
        expected_classification = tuple(
            AnnotationRecord(
                class_ids[tag['id']],
                frame.mapping.image_id,
                'classify',
                frame.mapping.parent_id,
                'classification',
                class_names[index % len(class_names)],
                None,
            )
            for index, (frame, tag) in enumerate(zip(views['classify_frames'], views['classify_tags'], strict=True))
        )
        self.assertEqual(expected_classification, self.repository.annotations(step_key='classify'))
        self.assertEqual(
            tuple(
                CvatBinding(record.image_id, 'tag', tag['id'], record.id)
                for record, tag in zip(expected_classification, views['classify_tags'], strict=True)
            ),
            classify_bindings,
        )

        segment_ids = {binding.object_id: binding.annotation_id for binding in segment_bindings}
        expected_segment: list[AnnotationRecord] = []
        expected_segment_bindings: list[CvatBinding] = []
        shapes_by_frame: dict[int, list[dict]] = {}
        for shape in views['segment_shapes']:
            shapes_by_frame.setdefault(shape['frame'], []).append(shape)
        for frame_index, frame in enumerate(views['segment_frames']):
            left, top, _, _ = frame.mapping.bounds
            for shape in shapes_by_frame[frame_index]:
                points = shape['points']
                record = AnnotationRecord(
                    segment_ids[shape['id']],
                    frame.mapping.image_id,
                    'segment',
                    frame.mapping.parent_id,
                    'polyline',
                    '1',
                    [[left + points[0], top + points[1]], [left + points[2], top + points[3]]],
                )
                expected_segment.append(record)
                expected_segment_bindings.append(CvatBinding(record.image_id, 'shape', shape['id'], record.id))
        self.assertEqual(tuple(expected_segment), self.repository.annotations(step_key='segment'))
        self.assertEqual(tuple(expected_segment_bindings), segment_bindings)
        before = self.repository.annotations()
        jobs = {'detect': detect_job, 'classify': classify_job, 'segment': segment_job}
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

        detect_job = self._runtime_job('detect')
        detect_bindings = {binding.object_id: binding.annotation_id for binding in self.repository.bindings(detect_job)}
        parent = next(record for record in before_noop if record.id == detect_bindings[10_000])
        sibling = next(record for record in before_noop if record.id == detect_bindings[10_500])
        self.assertEqual(parent.image_id, sibling.image_id)
        sibling_records_before = tuple(
            record for record in before_noop if record.id == sibling.id or record.parent_id == sibling.id
        )
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
        segment_shapes[0]['points'] = [25, 25, 85, 85]
        segment_shapes = [shape for shape in segment_shapes if shape['id'] != 31_000]
        segment_shapes.extend((self._line(0, 31_001, [30, 100, 150, 30]), self._line(0, 31_002, [35, 105, 155, 35])))
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
        moved_line = next(record for record in parent_lines if record.id == pointer_bindings[30_000])
        self.assertEqual([[65.0, 55.0], [125.0, 115.0]], moved_line.geometry)
        self.assertNotIn(pointer_bindings[31_000], {record.id for record in parent_lines})
        new_line_ids = {record.id for record in parent_lines} - {pointer_bindings[30_000]}
        self.assertEqual(
            {((70.0, 130.0), (190.0, 60.0)), ((75.0, 135.0), (195.0, 65.0))},
            {tuple(tuple(point) for point in record.geometry) for record in parent_lines if record.id in new_line_ids},
        )
        current_segment_job = self._runtime_job('segment')
        current_segment_bindings = self.repository.bindings(current_segment_job.ref)
        self.assertEqual(
            unrelated_pointer_bindings,
            tuple(
                binding
                for binding in current_segment_bindings
                if binding.annotation_id not in {record.id for record in parent_lines}
            ),
        )
        line_ids_by_geometry = {tuple(tuple(point) for point in record.geometry): record.id for record in parent_lines}
        self.assertEqual(
            {
                30_000: pointer_bindings[30_000],
                31_001: line_ids_by_geometry[((70.0, 130.0), (190.0, 60.0))],
                31_002: line_ids_by_geometry[((75.0, 135.0), (195.0, 65.0))],
            },
            {
                binding.object_id: binding.annotation_id
                for binding in current_segment_bindings
                if binding.annotation_id in {record.id for record in parent_lines}
            },
        )
        self.assertEqual(
            sibling_records_before,
            tuple(record for record in after_pointers if record.id == sibling.id or record.parent_id == sibling.id),
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
        parent_class_binding = CvatBinding(parent.image_id, 'tag', 20_000, parent_class_id)
        self.assertIn(parent_class_binding, class_bindings_before)
        self.service.sync_target(17, 'classify')
        after_class_change = self.repository.annotations()
        changed_class = next(record for record in after_class_change if record.id == parent_class_id)
        self.assertEqual(
            AnnotationRecord(parent_class_id, parent.image_id, 'classify', parent.id, 'classification', 'cc', None),
            changed_class,
        )
        unaffected = tuple(record for record in before_class_change if record.parent_id != parent.id)
        self.assertEqual(unaffected, tuple(record for record in after_class_change if record.parent_id != parent.id))
        self.assertFalse(
            any(record.step_key == 'segment' and record.parent_id == parent.id for record in after_class_change)
        )
        current_classify_job = self._runtime_job('classify')
        class_bindings_after = self.repository.bindings(current_classify_job.ref)
        self.assertEqual(class_bindings_before, class_bindings_after)
        self.assertIn(parent_class_binding, class_bindings_after)
        self.assertEqual(
            sibling_records_before,
            tuple(record for record in after_class_change if record.id == sibling.id or record.parent_id == sibling.id),
        )

        detect_shapes = copy.deepcopy(values['detect_shapes'])
        detect_shapes[0]['points'] = [42, 31, 279, 209]
        self.cvat.set_annotations('detect', shapes=detect_shapes)
        before_parent_change = self.repository.annotations()
        detect_job = self._runtime_job('detect')
        detect_bindings_before = self.repository.bindings(detect_job)
        parent_detect_binding = CvatBinding(parent.image_id, 'shape', 10_000, parent.id)
        self.assertIn(parent_detect_binding, detect_bindings_before)
        self.service.sync_target(17, 'detect')
        after_parent_change = self.repository.annotations()
        changed_parent = next(
            record for record in after_parent_change if record.step_key == 'detect' and record.id == parent.id
        )
        self.assertEqual(
            AnnotationRecord(
                parent.id, parent.image_id, 'detect', None, 'rectangle', 'tl', [[42.0, 31.0], [279.0, 209.0]]
            ),
            changed_parent,
        )
        unaffected_parent_records = tuple(
            record for record in before_parent_change if record.id != parent.id and record.parent_id != parent.id
        )
        self.assertEqual(
            unaffected_parent_records,
            tuple(record for record in after_parent_change if record.id != parent.id and record.parent_id != parent.id),
        )
        current_detect_job = self._runtime_job('detect')
        detect_bindings_after = self.repository.bindings(current_detect_job)
        self.assertEqual(detect_bindings_before, detect_bindings_after)
        self.assertIn(parent_detect_binding, detect_bindings_after)
        self.assertEqual((sibling,), tuple(record for record in after_parent_change if record.id == sibling.id))
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
        database_path = Path(self.receipt['database_path'])
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                f"""CREATE TRIGGER reject_workflow_binding BEFORE INSERT ON cvat_annotation_map
                WHEN NEW.job_id = {new_classify_job.ref.job_id} AND NEW.object_id = 21000
                BEGIN SELECT RAISE(ABORT, 'injected workflow binding failure'); END"""
            )
        with self.assertRaisesRegex(PlatformError, '取回或保存') as raised:
            self.service.sync_target(17, 'classify')
        self.assertIsInstance(raised.exception.__cause__, PlatformError)
        self.assertIsInstance(raised.exception.__cause__.__cause__, sqlite3.Error)
        self.assertEqual(before_failure, self.repository.annotations())
        self.assertEqual(bindings_before_failure, self.repository.bindings(new_classify_job.ref))

        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute('DROP TRIGGER reject_workflow_binding')

        restarted = AnnotationService(
            self.config,
            WorkspaceData(self.config.workspace_dir),
            self.cvat.client,
            RuntimeCache(self.config.runtime_dir),
        )
        retried = restarted.sync_target(17, 'classify')
        self.assertEqual(51, self._targets(retried)['classify'].annotated_sample_count)
        after_retry = self.repository.annotations()
        new_parent_classes = tuple(
            record for record in after_retry if record.parent_id == parent.id and record.step_key == 'classify'
        )
        self.assertEqual(1, len(new_parent_classes))
        new_parent_class = new_parent_classes[0]
        self.assertNotIn(new_parent_class.id, {record.id for record in before_failure})
        self.assertEqual(
            AnnotationRecord(new_parent_class.id, parent.image_id, 'classify', parent.id, 'classification', 'tl', None),
            new_parent_class,
        )
        self.assertEqual(
            sorted((*before_failure, new_parent_class), key=lambda record: str(record.id)),
            sorted(after_retry, key=lambda record: str(record.id)),
        )
        new_parent_binding = CvatBinding(parent.image_id, 'tag', 21_000, new_parent_class.id)
        bindings_after_retry = self.repository.bindings(new_classify_job.ref)
        self.assertEqual(
            sorted(
                (*bindings_before_failure, new_parent_binding),
                key=lambda binding: (binding.object_type, binding.object_id, str(binding.annotation_id)),
            ),
            sorted(
                bindings_after_retry,
                key=lambda binding: (binding.object_type, binding.object_id, str(binding.annotation_id)),
            ),
        )
        repeated = restarted.sync_target(17, 'classify')
        self.assertEqual(51, self._targets(repeated)['classify'].annotated_sample_count)
        self.assertEqual(after_retry, self.repository.annotations())
        self.assertEqual(bindings_after_retry, self.repository.bindings(new_classify_job.ref))

    def _runtime_job(self, target: str):
        runtime = RuntimeCache(self.config.runtime_dir)
        fingerprint = self.data.detection_fingerprint() if target == 'detect' else self.data.target_fingerprint(target)
        job = runtime.job_for(target, fingerprint) if target == 'detect' else runtime.edit_job_for(target, fingerprint)
        self.assertIsNotNone(job)
        return job


class PointWorkflowFixtureTest(unittest.TestCase):
    def test_seeded_fixture_round_trips_and_builds_consumable_caches(self) -> None:
        import yaml
        from torchvision.datasets import ImageFolder

        from xxtrain.data import ImageInfo, LabelCatalog
        from xxtrain.data.formats import decode_segment

        with tempfile.TemporaryDirectory() as parent:
            receipt = create_fixture(Path(parent), owner_user_id=17, cvat_internal_url='http://cvat.test')
            config = load_config(Path(receipt['config_path']))
            data = WorkspaceData(config.workspace_dir)
            repository = AnnotationRepository(Path(receipt['database_path']), point_task_definition())
            before = repository.annotations()
            cvat = HttpxCvatFixture()
            self.addCleanup(cvat.close)
            runtime = RuntimeCache(config.runtime_dir)
            service = AnnotationService(config, data, cvat.client, runtime)
            for target in ('classify', 'segment'):
                frames = data.target_frames(target, config.runtime_dir)
                if target == 'segment':
                    self.assertEqual((60, 50, 180, 150), frames[1].mapping.bounds)
                    self.assertEqual([[20, 20], [100, 80]], frames[1].annotations[0].geometry)
                cvat.expect(frames)
                service.begin_target(17, target)
                job = runtime.edit_job_for(target, data.target_fingerprint(target))
                bindings = repository.bindings(job.ref)
                service.sync_target(17, target)
                self.assertEqual(before, repository.annotations())
                self.assertEqual(bindings, repository.bindings(job.ref))
                service.generate_target_cache(17, target)
                publication = config.runtime_dir / 'cache' / data.target_fingerprint(target) / target
                if target == 'classify':
                    datasets = [ImageFolder(publication / split) for split in ('train', 'val')]
                    self.assertEqual(51, sum(len(dataset) for dataset in datasets))
                    for dataset in datasets:
                        for image, label in dataset:
                            self.assertEqual((224, 224), image.size)
                            self.assertIn(dataset.classes[label], ('00-tl', '01-tc', '02-cl', '03-cc'))
                else:
                    dataset = yaml.safe_load((publication / 'dataset.yaml').read_text())
                    image_paths = []
                    for split in ('train', 'val'):
                        image_paths.extend((Path(dataset['path']) / dataset[split]).read_text().splitlines())
                    self.assertEqual(51, len(image_paths))
                    labels = [Path(path).with_suffix('.txt') for path in image_paths]
                    rows = [line.split() for path in labels for line in path.read_text().splitlines()]
                    self.assertEqual(52, len(rows))
                    self.assertTrue(all(row[0] == '0' and len(row) == 7 for row in rows))
                    self.assertTrue(all(0 <= float(value) <= 1 for row in rows for value in row[1:]))
                    for path in image_paths:
                        with Image.open(path) as image:
                            for row in Path(path).with_suffix('.txt').read_text().splitlines():
                                decoded = decode_segment(
                                    row, ImageInfo(width=image.width, height=image.height), LabelCatalog(('Point',))
                                )
                                self.assertEqual('Point', decoded.label)
                            image.verify()

    def test_fixture_is_fresh_complete_and_records_explicit_identity(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            receipt = create_fixture(Path(parent), owner_user_id=17, cvat_internal_url='http://cvat.test')
            repository = AnnotationRepository(Path(receipt['database_path']), point_task_definition())

            self.assertEqual('xxtrain-task-7-point-workflow-v2', FIXTURE_MARKER)
            self.assertEqual(50, len(receipt['images']))
            self.assertEqual({'detect', 'classify', 'segment'}, set(receipt['initial_annotation_ids']))
            for target, ids in receipt['initial_annotation_ids'].items():
                self.assertEqual(ids, [str(record.id) for record in repository.annotations(step_key=target)])
            detections = repository.annotations(step_key='detect')
            self.assertEqual(51, len(detections))
            self.assertEqual(detections[0].image_id, detections[1].image_id)
            self.assertNotEqual(detections[0].id, detections[1].id)
            self.assertEqual(51, len(repository.annotations(step_key='classify')))
            self.assertEqual(52, len(repository.annotations(step_key='segment')))


if __name__ == '__main__':
    unittest.main()
