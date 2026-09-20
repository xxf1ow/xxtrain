import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from uuid import UUID

from PIL import Image

from xxtrain.business_tasks.definition import AnnotationPolicy, StepDefinition, TaskDefinition
from xxtrain.business_tasks.point import point_task_definition
from xxtrain.data import Bbox
from xxtrain.platform.contracts import (
    AnnotationRecord,
    DetectionBox,
    EditAnnotation,
    EditFrameResult,
    EditJob,
    ImageInput,
    ImageRecord,
    JobRef,
    TargetSummary,
)
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.inputs import AxisAlignedRectangleInputs, OriginalImageInputs
from xxtrain.workspace_data.legacy_fingerprints import legacy_point_fingerprint
from xxtrain.workspace_data.repository import AnnotationRepository


class WorkspaceTaskInputsTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='xxtrain-task-inputs-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / 'workspace'
        (self.workspace / 'images').mkdir(parents=True)
        with Image.new('RGB', (64, 48), (10, 20, 30)) as image:
            image.save(self.workspace / 'images' / 'source.png')
        self.task = self.make_task()
        self.data = WorkspaceData(self.workspace, self.task)
        self.repository = AnnotationRepository(self.workspace / 'annotations.db', self.task)
        self.image = ImageRecord('a' * 64, 'images/source.png', 64, 48, 0)
        self.repository.register_images((self.image,))

    @staticmethod
    def make_task() -> TaskDefinition:
        original = OriginalImageInputs()
        rectangles = AxisAlignedRectangleInputs()
        return TaskDefinition(
            (
                StepDefinition(
                    'regions',
                    frozenset({'rectangle'}),
                    frozenset({'part'}),
                    frozenset(),
                    frozenset(),
                    input_adapter=original,
                    annotation=AnnotationPolicy('rectangle', 'STANDARD'),
                    minimum_samples=1,
                ),
                StepDefinition(
                    'subregions',
                    frozenset({'rectangle'}),
                    frozenset({'part'}),
                    frozenset({'regions'}),
                    frozenset(),
                    input_adapter=rectangles,
                    annotation=AnnotationPolicy('rectangle', 'STANDARD'),
                ),
                StepDefinition(
                    'details',
                    frozenset({'classification'}),
                    frozenset({'x', 'y'}),
                    frozenset({'subregions'}),
                    frozenset(),
                    input_adapter=rectangles,
                    annotation=AnnotationPolicy('tag', 'TAGS', maximum_annotations=1),
                ),
                StepDefinition(
                    'kind',
                    frozenset({'classification'}),
                    frozenset({'a', 'b'}),
                    frozenset({'regions'}),
                    frozenset(),
                    input_adapter=rectangles,
                    annotation=AnnotationPolicy('tag', 'TAGS', maximum_annotations=1),
                ),
                StepDefinition(
                    'needles',
                    frozenset({'polyline'}),
                    frozenset({'line'}),
                    frozenset({'regions'}),
                    frozenset(),
                    input_adapter=rectangles,
                    annotation=AnnotationPolicy('polyline', 'STANDARD', point_count=2),
                ),
            ),
            key='synthetic',
            display_name='Synthetic',
        )

    def seed_nested_records(self) -> tuple[AnnotationRecord, AnnotationRecord, AnnotationRecord]:
        outer = AnnotationRecord(
            UUID('10000000-0000-0000-0000-000000000001'),
            self.image.id,
            'regions',
            None,
            'rectangle',
            'part',
            [[10, 12], [40, 42]],
        )
        inner = AnnotationRecord(
            UUID('20000000-0000-0000-0000-000000000001'),
            self.image.id,
            'subregions',
            outer.id,
            'rectangle',
            'part',
            [[12, 14], [28, 30]],
        )
        detail = AnnotationRecord(
            UUID('30000000-0000-0000-0000-000000000001'),
            self.image.id,
            'details',
            inner.id,
            'classification',
            'x',
            None,
        )
        self.repository.save_annotations((outer, inner, detail))
        return outer, inner, detail

    def test_raw_images_and_root_frames_do_not_require_a_point_step(self) -> None:
        (raw,) = self.data.images()

        self.assertEqual((), raw.boxes)
        self.assertIs(self.task, self.data.task)
        (frame,) = self.data.target_frames('regions', self.root / 'runtime')
        self.assertEqual(self.image.id, frame.mapping.frame_id)
        self.assertIsNone(frame.mapping.parent_id)
        self.assertEqual((0, 0, 64, 48), frame.mapping.bounds)
        self.assertEqual(self.workspace / 'images' / 'source.png', frame.image_path)

    def test_nested_rectangle_input_uses_original_coordinates_and_direct_parent(self) -> None:
        _, inner, detail = self.seed_nested_records()

        (frame,) = self.data.target_frames('details', self.root / 'details')

        self.assertEqual(inner.id, frame.mapping.parent_id)
        self.assertEqual((12, 14, 28, 30), frame.mapping.bounds)
        self.assertEqual((detail.id,), tuple(annotation.id for annotation in frame.annotations))
        self.assertEqual(TargetSummary(1, 1, 1), self.data.target_summary('details'))

    def test_rectangle_child_geometry_is_projected_from_original_coordinates_once(self) -> None:
        _, inner, _ = self.seed_nested_records()

        (frame,) = self.data.target_frames('subregions', self.root / 'subregions')

        self.assertEqual([[2, 2], [18, 18]], frame.annotations[0].geometry)
        self.assertEqual(inner.id, frame.annotations[0].id)

    def test_selected_rectangle_images_are_explicit_and_step_scoped(self) -> None:
        outer, inner, _ = self.seed_nested_records()

        (raw,) = self.data.images()
        (regions,) = self.data.images('regions')
        (subregions,) = self.data.images('subregions')

        self.assertEqual((), raw.boxes)
        self.assertEqual((outer.id,), tuple(box.geometry.id for box in regions.boxes))
        self.assertEqual((inner.id,), tuple(box.geometry.id for box in subregions.boxes))

    def test_generic_sync_assigns_root_and_nested_rectangle_parents_from_mappings(self) -> None:
        (root_frame,) = self.data.target_frames('regions', self.root / 'root-sync')
        root_job = EditJob(JobRef(10, 11, (self.image.id,)), (root_frame.mapping,))
        root_result = EditFrameResult(
            root_frame.mapping.frame_id, (EditAnnotation(None, 'rectangle', 'part', [[10, 12], [40, 42]], 101),)
        )
        root_sync = self.data.prepare_target_sync('regions', root_job, (root_result,))
        self.data.commit_target_sync(root_job, root_sync)
        (outer,) = self.repository.annotations(step_key='regions')
        self.assertIsNone(outer.parent_id)

        (crop_frame,) = self.data.target_frames('subregions', self.root / 'crop-sync')
        crop_job = EditJob(JobRef(12, 13, (self.image.id,)), (crop_frame.mapping,))
        crop_result = EditFrameResult(
            crop_frame.mapping.frame_id, (EditAnnotation(None, 'rectangle', 'part', [[2, 2], [18, 18]], 201),)
        )
        crop_sync = self.data.prepare_target_sync('subregions', crop_job, (crop_result,))
        self.data.commit_target_sync(crop_job, crop_sync)
        (inner,) = self.repository.annotations(step_key='subregions')

        self.assertEqual(outer.id, inner.parent_id)
        self.assertEqual([[12, 14], [28, 30]], inner.geometry)

    def test_fingerprint_uses_declared_inputs_and_ignores_parallel_sibling(self) -> None:
        outer, _, _ = self.seed_nested_records()
        category = AnnotationRecord(
            UUID('40000000-0000-0000-0000-000000000001'), self.image.id, 'kind', outer.id, 'classification', 'a', None
        )
        line = AnnotationRecord(
            UUID('50000000-0000-0000-0000-000000000001'),
            self.image.id,
            'needles',
            outer.id,
            'polyline',
            'line',
            [[12, 14], [20, 22]],
        )
        self.repository.save_annotations((category, line))
        before = self.data.target_fingerprint('needles')

        self.repository.save_annotations((replace(category, label='b'),))

        self.assertEqual(before, self.data.target_fingerprint('needles'))
        self.assertEqual((line,), self.repository.annotations(step_key='needles'))

    def test_root_negative_is_complete_but_not_a_qualified_positive_sample(self) -> None:
        negative = AnnotationRecord(
            UUID('60000000-0000-0000-0000-000000000001'), self.image.id, 'regions', None, 'negative', None, None
        )
        task = replace(
            self.task,
            steps=(
                replace(
                    self.task.step('regions'),
                    kinds=frozenset({'rectangle', 'negative'}),
                    annotation=AnnotationPolicy('rectangle', 'STANDARD', negative_label='negative'),
                ),
                *self.task.steps[1:],
            ),
        )
        data = WorkspaceData(self.workspace, task)
        repository = AnnotationRepository(self.workspace / 'annotations.db', task)
        repository.save_annotations((negative,))

        self.assertEqual(TargetSummary(1, 1, 0), data.target_summary('regions'))

    def test_point_declares_original_and_rectangle_input_adapters(self) -> None:
        task = point_task_definition()

        self.assertIsInstance(task.step('detect').input_adapter, OriginalImageInputs)
        self.assertIsInstance(task.step('classify').input_adapter, AxisAlignedRectangleInputs)
        self.assertIsInstance(task.step('segment').input_adapter, AxisAlignedRectangleInputs)

    def test_legacy_point_fingerprint_vectors_remain_byte_compatible(self) -> None:
        parent_id = UUID(int=1)
        image = ImageInput(
            'a' * 64,
            Path('a.png'),
            64,
            48,
            (DetectionBox(Bbox(id=parent_id, label='Point', x1=1, y1=2, x2=30, y2=40)),),
        )
        records = (
            AnnotationRecord(parent_id, image.sample_id, 'detect', None, 'rectangle', 'Point', [[1, 2], [30, 40]]),
            AnnotationRecord(UUID(int=2), image.sample_id, 'classify', parent_id, 'classification', 'tl', None),
            AnnotationRecord(UUID(int=3), image.sample_id, 'segment', parent_id, 'polyline', '1', [[2, 3], [4, 5]]),
        )

        self.assertEqual(
            '20b718f7cfbff83c1d8203c9eff154296d1ec8677ffc7e75608d3eb7ad9320b0',
            legacy_point_fingerprint('detect', (image,), records),
        )
        self.assertEqual(
            '12984abf3c8306dbd4291e40a50714305de15249e97d8915886e9499b81a710c',
            legacy_point_fingerprint('classify', (image,), records),
        )
        self.assertEqual(
            '37dd707bb54cd17ea599fa1b4e0e112bf7630a45aac8623d89e6acba13c1cb8d',
            legacy_point_fingerprint('segment', (image,), records),
        )


if __name__ == '__main__':
    unittest.main()
