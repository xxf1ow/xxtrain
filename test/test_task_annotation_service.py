import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import httpx
from PIL import Image

from test.task_definitions import synthetic_task_definition
from xxtrain.business_tasks.definition import (
    AnnotationPolicy,
    DeliveryDefinition,
    TargetTrainingDefinition,
    TaskDefinition,
)
from xxtrain.integrations.cvat import CvatClient
from xxtrain.integrations.cvat.edit_codec import decode_edit_annotations, decode_edit_bindings, encode_edit_annotations
from xxtrain.platform.cache_builders import encode_classification, encode_rectangles
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import AnnotationRecord, EditAnnotation, EditFrame, EditJob, FrameMapping, JobRef
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.service import AnnotationService
from xxtrain.platform.target_cache import build_target_cache
from xxtrain.task import TaskType
from xxtrain.training.settings import TrainingSettings
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.repository import AnnotationRepository

from .test_platform_point_workflow import HttpxCvatFixture


class TaskAnnotationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / 'workspace'
        (self.workspace / 'images').mkdir(parents=True)
        self.task = synthetic_task_definition()
        self.data = WorkspaceData(self.workspace, self.task)
        self.repository = AnnotationRepository(self.workspace / 'annotations.db', self.task)
        self.config = WorkspaceConfig(
            'synthetic-workspace', 'Synthetic workspace', 17, self.workspace, self.root / 'runtime', 'http://cvat.test'
        )
        self.service = AnnotationService(self.config, self.data, object(), RuntimeCache(self.config.runtime_dir))

    def _admit_image(self, name: str = 'candidate.png', seed: int = 19) -> str:
        path = self.root / name
        rng = random.Random(seed)
        image = Image.new('RGB', (32, 24))
        image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(32 * 24)])
        image.save(path)
        self.assertEqual(1, self.data.admit((path,)).accepted_count)
        return self.data.images()[0].sample_id

    def _cache_task(self) -> TaskDefinition:
        task = synthetic_task_definition()
        training = {
            'regions': TargetTrainingDefinition(
                settings=TrainingSettings(TaskType.DETECT),
                metric_key='regions',
                metric_name='Regions',
                delivery=DeliveryDefinition(labels=False, reference_images=False),
                labels=('part',),
                encode_sample=encode_rectangles,
            ),
            'kind': TargetTrainingDefinition(
                settings=TrainingSettings(TaskType.CLASSIFY),
                metric_key='kind',
                metric_name='Kind',
                delivery=DeliveryDefinition(labels=True, reference_images=True),
                labels=('a', 'b'),
                encode_sample=encode_classification,
            ),
            'subregions': TargetTrainingDefinition(
                settings=TrainingSettings(TaskType.DETECT),
                metric_key='subregions',
                metric_name='Subregions',
                delivery=DeliveryDefinition(labels=False, reference_images=False),
                labels=('part',),
                encode_sample=encode_rectangles,
            ),
        }
        return TaskDefinition(
            tuple(replace(step, training=training.get(step.key)) for step in task.steps),
            key=task.key,
            display_name=task.display_name,
        )

    def test_view_uses_actual_dependencies_for_parallel_and_nested_targets(self) -> None:
        image_id = self._admit_image()
        region_id = uuid4()
        self.repository.save_annotations(
            (AnnotationRecord(region_id, image_id, 'regions', None, 'rectangle', 'part', [[2, 3], [26, 20]]),)
        )

        view = self.service.view(17)

        rows = {row.id: row for row in view.targets}
        self.assertEqual(('synthetic', 'Synthetic task'), (view.task_id, view.task_name))
        self.assertEqual(('Kind', '个样本'), (rows['kind'].display_name, rows['kind'].sample_unit))
        self.assertTrue(rows['kind'].can_annotate)
        self.assertTrue(rows['needles'].can_annotate)
        self.assertTrue(rows['subregions'].can_annotate)
        self.assertFalse(rows['details'].can_annotate)

        self.repository.save_annotations(
            (AnnotationRecord(uuid4(), image_id, 'subregions', region_id, 'rectangle', 'part', [[4, 5], [12, 14]]),)
        )
        self.assertTrue({row.id: row for row in self.service.view(17).targets}['details'].can_annotate)

        self._admit_image('second.png', 23)
        self.assertFalse({row.id: row for row in self.service.view(17).targets}['details'].can_annotate)

    def test_cvat_task_creation_uses_policy_native_types_and_negative_label(self) -> None:
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(201, json={'id': 41})

        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            client = CvatClient('http://cvat.test', 'private-token', http)
            task_id = client.create_task(
                'Synthetic regions', ('part',), AnnotationPolicy('rectangle', 'STANDARD', negative_label='无检测目标')
            )

        self.assertEqual(41, task_id)
        labels = json.loads(requests[0].content)['labels']
        self.assertEqual(
            [('part', 'rectangle'), ('无检测目标', 'tag')], [(item['name'], item['type']) for item in labels]
        )

    def test_rectangle_and_negative_tag_share_edit_codec_identity_transport(self) -> None:
        policy = AnnotationPolicy('rectangle', 'STANDARD', negative_label='无检测目标')
        labels = [
            {
                'id': 41,
                'name': 'part',
                'type': 'rectangle',
                'attributes': [{'id': 71, 'name': 'xxtrain_labelme_extra', 'input_type': 'text'}],
            },
            {
                'id': 42,
                'name': '无检测目标',
                'type': 'tag',
                'attributes': [{'id': 72, 'name': 'xxtrain_labelme_extra', 'input_type': 'text'}],
            },
        ]
        first_id = uuid4()
        negative_id = uuid4()
        first_mapping = FrameMapping('a' * 64, 'a' * 64, None, (0, 0, 32, 24))
        second_mapping = FrameMapping('b' * 64, 'b' * 64, None, (0, 0, 32, 24))
        frames = (
            EditFrame(
                first_mapping,
                Path('first.png'),
                32,
                24,
                (EditAnnotation(first_id, 'rectangle', 'part', [[2, 3], [20, 18]]),),
            ),
            EditFrame(
                second_mapping, Path('second.png'), 32, 24, (EditAnnotation(negative_id, 'negative', None, None),)
            ),
        )

        payload = encode_edit_annotations(frames, labels, policy, mapped=True)
        payload['shapes'][0]['id'] = 101
        payload['tags'][0]['id'] = 102
        job = EditJob(JobRef(7, 8, ('a' * 64, 'b' * 64)), (first_mapping, second_mapping))

        bindings = decode_edit_bindings(payload, frames, job, labels, policy)
        results = decode_edit_annotations(payload, job, labels, policy)
        self.assertEqual(
            {('shape', 101, first_id), ('tag', 102, negative_id)},
            {(binding.object_type, binding.object_id, binding.annotation_id) for binding in bindings},
        )
        self.assertEqual(
            (EditAnnotation(None, 'rectangle', 'part', [[2.0, 3.0], [20.0, 18.0]], 101),), results[0].annotations
        )
        self.assertEqual((EditAnnotation(None, 'negative', None, None, 102),), results[1].annotations)

        wrong_shape_type = json.loads(json.dumps(payload))
        wrong_shape_type['shapes'][0]['type'] = 'polyline'
        with self.assertRaisesRegex(ValueError, 'wrong type'):
            decode_edit_annotations(wrong_shape_type, job, labels, policy)

    def test_real_cvat_transport_writes_back_root_and_nested_rectangles(self) -> None:
        image_id = self._admit_image()
        cvat = HttpxCvatFixture()
        self.addCleanup(cvat.close)
        service = AnnotationService(self.config, self.data, cvat.client, self.service.runtime)

        cvat.expect(self.data.target_frames('regions', self.config.runtime_dir))
        service.begin_target(17, 'regions')
        root_task = next(iter(cvat.tasks.values()))
        root_job = cvat.jobs[root_task['job_id']]
        root_label = next(label['id'] for label in root_task['labels'] if label['name'] == 'part')
        root_job['annotations'] = {
            'version': 0,
            'tags': [],
            'tracks': [],
            'shapes': [
                {
                    'id': 501,
                    'type': 'rectangle',
                    'frame': 0,
                    'label_id': root_label,
                    'points': [2, 3, 26, 20],
                    'attributes': [],
                }
            ],
        }
        view = service.sync_target(17, 'regions')
        rows = {row.id: row for row in view.targets}
        self.assertTrue(rows['kind'].can_annotate)
        self.assertTrue(rows['needles'].can_annotate)

        cvat.expect(self.data.target_frames('subregions', self.config.runtime_dir))
        service.begin_target(17, 'subregions')
        nested_task = max(cvat.tasks.values(), key=lambda item: item['id'])
        nested_job = cvat.jobs[nested_task['job_id']]
        nested_label = next(label['id'] for label in nested_task['labels'] if label['name'] == 'part')
        nested_job['annotations'] = {
            'version': 0,
            'tags': [],
            'tracks': [],
            'shapes': [
                {
                    'id': 601,
                    'type': 'rectangle',
                    'frame': 0,
                    'label_id': nested_label,
                    'points': [1, 2, 10, 11],
                    'attributes': [],
                }
            ],
        }
        view = service.sync_target(17, 'subregions')

        nested = self.repository.annotations(step_key='subregions')
        self.assertEqual(1, len(nested))
        self.assertEqual(image_id, nested[0].image_id)
        self.assertEqual([[3.0, 5.0], [12.0, 14.0]], nested[0].geometry)
        self.assertTrue({row.id: row for row in view.targets}['details'].can_annotate)

    def test_generic_cache_publishes_root_category_and_nested_rectangle_inputs(self) -> None:
        task = self._cache_task()
        data = WorkspaceData(self.workspace, task)
        repository = AnnotationRepository(self.workspace / 'annotations.db', task)
        image_ids = []
        for index in range(2):
            path = self.root / f'cache-{index}.png'
            rng = random.Random(100 + index)
            image = Image.new('RGB', (32, 24))
            image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(32 * 24)])
            image.save(path)
            self.assertEqual(1, data.admit((path,)).accepted_count)
        image_ids = [image.sample_id for image in data.images()]
        regions = tuple(
            AnnotationRecord(uuid4(), image_id, 'regions', None, 'rectangle', 'part', [[2, 3], [26, 20]])
            for image_id in image_ids
        )
        repository.save_annotations(regions)
        repository.save_annotations(
            tuple(
                AnnotationRecord(uuid4(), region.image_id, 'kind', region.id, 'classification', 'a', None)
                for region in regions
            )
            + tuple(
                AnnotationRecord(
                    uuid4(), region.image_id, 'subregions', region.id, 'rectangle', 'part', [[3, 5], [12, 14]]
                )
                for region in regions
            )
        )

        for target in ('regions', 'kind', 'subregions'):
            with self.subTest(target=target):
                fingerprint = data.target_fingerprint(target)
                destination = self.config.runtime_dir / 'cache' / fingerprint
                report = build_target_cache(data, target, self.config.runtime_dir, destination)
                self.assertTrue(report.train_items)
                self.assertTrue(report.val_items)
                self.assertEqual(
                    destination / target, RuntimeCache(self.config.runtime_dir).cache_path(target, fingerprint)
                )


if __name__ == '__main__':
    unittest.main()
