import copy
import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import httpx
from PIL import Image

from xxtrain.business_tasks.point import point_task_definition
from xxtrain.integrations.cvat.client import CvatClient
from xxtrain.integrations.cvat.codec import decode_annotations, decode_initial_bindings, encode_mapped_annotations
from xxtrain.integrations.cvat.edit_codec import decode_edit_annotations
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import AnnotationRecord, EditJob, FrameResult, JobRef, TargetValidationError
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.service import AnnotationService
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.repository import AnnotationRepository

from .test_platform_point_workflow import HttpxCvatFixture

LABELS = [
    {'id': 41, 'name': 'Point', 'type': 'rectangle', 'attributes': [{'id': 71, 'name': 'xxtrain_labelme_extra'}]},
    {'id': 42, 'name': '无检测目标', 'type': 'tag', 'attributes': [{'id': 72, 'name': 'xxtrain_labelme_extra'}]},
]


class NegativeTagTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'images').mkdir()
        self.data = WorkspaceData(self.root, point_task_definition())
        staged = self.root / 'upload.png'
        Image.new('RGB', (32, 24)).save(staged)
        self.data.admit((staged,))
        self.image = self.data.images()[0]
        self.ref = JobRef(7, 8, (self.image.sample_id,))
        self.repo = AnnotationRepository(self.root / 'annotations.db', point_task_definition())

    def payload(self):
        return {'shapes': [], 'tracks': [], 'tags': [{'id': 101, 'frame': 0, 'label_id': 42, 'attributes': []}]}

    def test_explicit_tag_round_trip_and_removal_recompute_database_facts(self):
        results = decode_annotations(self.payload(), self.ref, LABELS)
        self.data.commit_detection_sync(self.ref, self.data.prepare_detection_sync(self.ref, results))
        record = self.repo.annotations()[0]
        self.assertEqual(('negative', None, None), (record.kind, record.label, record.geometry))
        self.assertEqual(0, self.data.detection_summary().boxed_image_count)
        self.assertEqual(1, self.data.detection_summary().annotated_image_count)
        self.assertEqual((), self.data.target_frames('classify', self.root / 'runtime'))
        fingerprint = self.data.detection_fingerprint()
        payload = encode_mapped_annotations(self.data.images('detect'), LABELS)
        self.assertEqual([], payload['shapes'])
        self.assertEqual([(0, 42)], [(tag['frame'], tag['label_id']) for tag in payload['tags']])
        payload['tags'][0]['id'] = 102
        self.assertEqual((), decode_initial_bindings(payload, self.data.images('detect'), self.ref, LABELS))
        self.data.commit_detection_sync(self.ref, self.data.prepare_detection_sync(self.ref, results))
        self.assertEqual((record,), self.repo.annotations())
        self.assertEqual(fingerprint, self.data.detection_fingerprint())
        self.data.commit_detection_sync(
            self.ref, self.data.prepare_detection_sync(self.ref, (FrameResult(self.image.sample_id, ()),))
        )
        self.assertEqual((), self.repo.annotations())
        self.assertEqual(0, self.data.detection_summary().annotated_image_count)
        self.assertNotEqual(fingerprint, self.data.detection_fingerprint())

    def test_initialization_cannot_silently_lose_a_negative_tag(self):
        self.repo.save_annotations(
            (AnnotationRecord(uuid4(), self.image.sample_id, 'detect', None, 'negative', None, None),)
        )
        with self.assertRaisesRegex(ValueError, 'negative'):
            decode_initial_bindings({'shapes': [], 'tags': []}, self.data.images('detect'), self.ref, LABELS)

    def test_real_client_and_service_rebuild_negative_job_and_revoke_confirmation(self):
        cvat = HttpxCvatFixture()
        self.addCleanup(cvat.close)
        config = WorkspaceConfig('test', 'Test', 17, self.root, self.root / 'runtime', 'http://cvat.test')
        service = AnnotationService(config, self.data, cvat.client, RuntimeCache(config.runtime_dir))
        cvat.expect(self.data.images('detect'))
        service.begin_detection(17)
        label_id = cvat.label_id('detect', '无检测目标')
        cvat.set_annotations('detect', tags=[{'id': 999, 'frame': 0, 'label_id': label_id, 'attributes': []}])
        view = service.sync_detection(17)
        self.assertEqual(
            (1, 0, False), (view.annotated_image_count, view.boxed_image_count, view.can_generate_detection_cache)
        )
        record = self.repo.annotations()[0]
        cvat.job('detect')['state'] = 'completed'
        cvat.expect(self.data.images('detect'))
        service.begin_detection(17)
        self.assertEqual(1, len(cvat.job('detect')['annotations']['tags']))
        service.sync_detection(17)
        self.assertEqual((record,), self.repo.annotations())
        cvat.set_annotations('detect')
        view = service.sync_detection(17)
        self.assertEqual(0, view.annotated_image_count)
        self.assertEqual((), self.repo.annotations())

    def test_negative_label_cannot_be_used_as_a_rectangle(self):
        payload = {'shapes': [{'type': 'rectangle', 'frame': 0, 'label_id': 42, 'points': [1, 1, 9, 9]}]}
        with self.assertRaises(ValueError):
            decode_annotations(payload, self.ref, LABELS)

    def test_wrong_label_type_duplicate_tags_and_wrong_frames_are_rejected(self):
        for change in ('type', 'duplicate', 'frame', 'label'):
            with self.subTest(change=change):
                payload = self.payload()
                labels = copy.deepcopy(LABELS)
                if change == 'type':
                    labels[1]['type'] = 'rectangle'
                elif change == 'duplicate':
                    payload['tags'].append(dict(payload['tags'][0], id=102))
                elif change == 'frame':
                    payload['tags'][0]['frame'] = 1
                else:
                    payload['tags'][0]['label_id'] = 41
                with self.assertRaises(ValueError):
                    decode_annotations(payload, self.ref, labels)

    def test_detection_task_creation_types_negative_label_as_tag(self):
        payloads = []

        def respond(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(201, json={'id': 7})

        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            CvatClient('http://cvat.test', 'token', http).create_task(
                'detection', ('Point',), point_task_definition().step('detect').annotation
            )
        self.assertEqual(
            [('Point', 'rectangle'), ('无检测目标', 'tag')],
            [(label['name'], label['type']) for label in payloads[0]['labels']],
        )

    def test_conflict_returns_correction_link_without_changing_database(self):
        payload = self.payload()
        payload['shapes'] = [{'id': 101, 'type': 'rectangle', 'frame': 0, 'label_id': 41, 'points': [1, 1, 9, 9]}]

        class Cvat:
            def fetch_annotations(inner, job, policy):
                return decode_edit_annotations(payload, job, LABELS, policy)

            def job_path(inner, ref):
                return '/tasks/7/jobs/8'

        runtime = RuntimeCache(self.root / 'runtime')
        fingerprint = self.data.detection_fingerprint()
        frames = self.data.target_frames('detect', self.root / 'runtime')
        runtime.remember_edit_job('detect', fingerprint, EditJob(self.ref, tuple(frame.mapping for frame in frames)))
        config = WorkspaceConfig('test', 'Test', 17, self.root, self.root / 'runtime', 'http://cvat.test')
        service = AnnotationService(config, self.data, Cvat(), runtime)
        with self.assertRaises(TargetValidationError) as raised:
            service.sync_detection(17)
        self.assertIn('无检测目标', str(raised.exception))
        self.assertEqual('/tasks/7/jobs/8?frame=0', raised.exception.annotation_url)
        self.assertEqual(fingerprint, self.data.detection_fingerprint())
        self.assertEqual((), self.repo.annotations())


if __name__ == '__main__':
    unittest.main()
