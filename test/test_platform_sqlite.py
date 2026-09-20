import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from xxtrain.business_tasks.definition import StepDefinition, TaskDefinition
from xxtrain.business_tasks.point import POINT_BOX_LABELS, point_task_definition
from xxtrain.platform.contracts import (
    AnnotationChanges,
    AnnotationRecord,
    CvatBinding,
    ImageRecord,
    JobRef,
    PlatformError,
    PreparedJob,
)
from xxtrain.workspace_data.repository import AnnotationRepository


class PlatformSqliteTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='xxtrain-sqlite-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make_task(self) -> TaskDefinition:
        return TaskDefinition(
            (
                StepDefinition(
                    'detect', frozenset({'rectangle', 'negative'}), frozenset({'box'}), frozenset(), frozenset()
                ),
                StepDefinition(
                    'classify',
                    frozenset({'classification'}),
                    frozenset({'good'}),
                    frozenset({'detect'}),
                    frozenset({'detect'}),
                ),
                StepDefinition(
                    'segment',
                    frozenset({'polygon', 'polyline'}),
                    frozenset({'mask'}),
                    frozenset({'detect'}),
                    frozenset({'classify'}),
                ),
            )
        )

    def make_repo(self, task: TaskDefinition | None = None) -> tuple[AnnotationRepository, ImageRecord]:
        repo = AnnotationRepository(self.root / 'annotations.db', task or self.make_task())
        image = ImageRecord('a' * 64, 'a.png', 100, 80, (1 << 64) - 1)
        repo.register_images((image,))
        return repo, image

    def test_registered_image_and_geometry_survive_reopen(self) -> None:
        task = point_task_definition()
        path = self.root / 'annotations.db'
        repo = AnnotationRepository(path, task)
        image = ImageRecord('a' * 64, 'a.png', 100, 80, (1 << 64) - 1)
        repo.register_images((image,))
        box = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'Point', [[1.0, 2.0], [30.0, 40.0]])
        repo.save_annotations((box,))

        reopened = AnnotationRepository(path, task)

        self.assertEqual((image,), reopened.images())
        self.assertEqual((box,), reopened.annotations())
        with closing(sqlite3.connect(path)) as connection:
            self.assertEqual(
                {'images', 'annotations', 'cvat_annotation_map'},
                {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")},
            )
            self.assertEqual(1, connection.execute('PRAGMA user_version').fetchone()[0])

    def test_point_task_declares_steps_labels_and_transitive_dependents(self) -> None:
        task = point_task_definition()

        self.assertEqual(frozenset({'rectangle', 'negative'}), task.step('detect').kinds)
        self.assertEqual(frozenset(POINT_BOX_LABELS), task.step('detect').labels)
        self.assertEqual(frozenset({'tl', 'tc', 'cl', 'cc'}), task.step('classify').labels)
        self.assertEqual(frozenset({'detect'}), task.step('classify').parent_steps)
        self.assertEqual(frozenset({'polyline'}), task.step('segment').kinds)
        self.assertEqual(frozenset({'1'}), task.step('segment').labels)
        self.assertEqual(frozenset({'detect'}), task.step('segment').parent_steps)
        self.assertEqual(frozenset(), task.step('segment').depends_on)
        self.assertEqual(frozenset({'classify', 'segment'}), task.dependent_steps('detect'))
        with self.assertRaises(ValueError):
            task.step('missing')

    def test_image_registration_rejects_unsafe_paths_and_invalid_hashes(self) -> None:
        repo = AnnotationRepository(self.root / 'annotations.db', self.make_task())
        invalid_records = (
            ImageRecord('a' * 64, '../a.png', 100, 80, 0),
            ImageRecord('a' * 64, str(self.root / 'a.png'), 100, 80, 0),
            ImageRecord('a' * 64, 'a.png', 100, 80, -1),
            ImageRecord('a' * 64, 'a.png', 100, 80, 1 << 64),
        )
        for record in invalid_records:
            with self.subTest(record=record), self.assertRaises(ValueError):
                repo.register_images((record,))
        self.assertEqual((), repo.images())

    def test_repository_rejects_unknown_nonzero_schema_version(self) -> None:
        path = self.root / 'annotations.db'
        with closing(sqlite3.connect(path)) as connection:
            connection.execute('PRAGMA user_version=2')

        with self.assertRaises(ValueError):
            AnnotationRepository(path, self.make_task())

    def test_parent_and_child_can_be_saved_in_reverse_input_order(self) -> None:
        repo, image = self.make_repo()
        parent = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'box', [[1, 2], [30, 40]])
        child = AnnotationRecord(uuid4(), image.id, 'classify', parent.id, 'classification', 'good', None)

        repo.save_annotations((child, parent))

        self.assertEqual({parent.id, child.id}, {record.id for record in repo.annotations()})
        self.assertEqual((child,), repo.annotations(image_id=image.id, step_key='classify'))

    def test_records_reject_unknown_steps_wrong_parents_cross_image_and_cycles(self) -> None:
        repo, image = self.make_repo()
        other = ImageRecord('b' * 64, 'b.png', 100, 80, 0)
        repo.register_images((other,))
        parent = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'box', [[1, 2], [30, 40]])
        repo.save_annotations((parent,))
        invalid = (
            AnnotationRecord(uuid4(), image.id, 'missing', None, 'rectangle', 'box', [[1, 2], [30, 40]]),
            AnnotationRecord(uuid4(), image.id, 'classify', None, 'classification', 'good', None),
            AnnotationRecord(uuid4(), other.id, 'classify', parent.id, 'classification', 'good', None),
            AnnotationRecord(uuid4(), image.id, 'segment', None, 'polygon', 'mask', [[0, 0], [1, 0], [0, 1]]),
        )
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                repo.save_annotations((record,))

        classification = AnnotationRecord(uuid4(), image.id, 'classify', parent.id, 'classification', 'good', None)
        wrong_parent_step = AnnotationRecord(
            uuid4(), image.id, 'segment', classification.id, 'polygon', 'mask', [[0, 0], [1, 0], [0, 1]]
        )
        with self.assertRaises(ValueError):
            repo.save_annotations((classification, wrong_parent_step))

    def test_task_definition_rejects_parent_cycle_before_annotation_storage(self) -> None:
        with self.assertRaises(ValueError):
            TaskDefinition(
                (
                    StepDefinition(
                        'chain', frozenset({'classification'}), frozenset({'good'}), frozenset({'chain'}), frozenset()
                    ),
                )
            )

    def test_geometry_must_be_valid_json_with_finite_correctly_shaped_points(self) -> None:
        repo, image = self.make_repo()
        invalid_geometry = (
            ('rectangle', [[1, 2], [float('nan'), 40]]),
            ('rectangle', [[1, 2], [30, 40], [50, 60]]),
            ('polygon', [[0, 0], [1, 1]]),
            ('polyline', [[0, 0]]),
            ('rectangle', {'not': 'points'}),
            ('rectangle', object()),
        )
        for kind, geometry in invalid_geometry:
            label = 'box' if kind == 'rectangle' else 'mask'
            step = 'detect' if kind == 'rectangle' else 'segment'
            parent_id = None
            if step == 'segment':
                parent = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'box', [[1, 2], [30, 40]])
                repo.save_annotations((parent,))
                parent_id = parent.id
            record = AnnotationRecord(uuid4(), image.id, step, parent_id, kind, label, geometry)  # type: ignore[arg-type]
            with self.subTest(kind=kind, geometry=geometry), self.assertRaises(ValueError):
                repo.save_annotations((record,))

        existing = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'box', [[1, 2], [30, 40]])
        repo.save_annotations((existing,))
        with self.assertRaises(ValueError):
            repo.save_annotations((replace(existing, geometry=[[True, 2], [30, 40]]),))

    def test_negative_requires_null_label_and_geometry_and_conflicts_with_boxes(self) -> None:
        repo, image = self.make_repo()
        invalid_negatives = (
            AnnotationRecord(uuid4(), image.id, 'detect', None, 'negative', 'box', None),
            AnnotationRecord(uuid4(), image.id, 'detect', None, 'negative', None, []),
        )
        for record in invalid_negatives:
            with self.subTest(record=record), self.assertRaises(ValueError):
                repo.save_annotations((record,))

        negative = AnnotationRecord(uuid4(), image.id, 'detect', None, 'negative', None, None)
        box = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'box', [[1, 2], [30, 40]])
        with self.assertRaises(ValueError):
            repo.save_annotations((negative, box))
        self.assertEqual((), repo.annotations())

    def test_failed_batch_keeps_existing_annotations_unchanged(self) -> None:
        repo, image = self.make_repo()
        existing = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'box', [[1, 2], [30, 40]])
        repo.save_annotations((existing,))
        valid = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'box', [[2, 3], [40, 50]])
        invalid = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'missing', [[2, 3], [40, 50]])

        with self.assertRaises(ValueError):
            repo.save_annotations((valid, invalid), delete_ids=frozenset({existing.id}))

        self.assertEqual((existing,), repo.annotations())

    def test_save_annotations_applies_object_scoped_downstream_deletion(self) -> None:
        repo, image = self.make_repo()
        first = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'box', [[0, 0], [20, 20]])
        second = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'box', [[30, 0], [50, 20]])
        category = AnnotationRecord(uuid4(), image.id, 'classify', first.id, 'classification', 'good', None)
        first_mask = AnnotationRecord(
            uuid4(), image.id, 'segment', first.id, 'polygon', 'mask', [[1, 1], [10, 1], [10, 10]]
        )
        second_mask = replace(first_mask, id=uuid4(), parent_id=second.id)
        repo.save_annotations((first, second, category, first_mask, second_mask))

        repo.save_annotations((replace(category, label='good'),))
        self.assertEqual(
            {first.id, second.id, category.id, first_mask.id, second_mask.id}, {item.id for item in repo.annotations()}
        )

        changed_task = TaskDefinition(
            (
                self.make_task().step('detect'),
                replace(self.make_task().step('classify'), labels=frozenset({'good', 'bad'})),
                self.make_task().step('segment'),
            )
        )
        changed_repo = AnnotationRepository(self.root / 'annotations.db', changed_task)
        changed_repo.save_annotations((replace(category, label='bad'),))

        self.assertEqual(
            {first.id, second.id, category.id, second_mask.id}, {item.id for item in changed_repo.annotations()}
        )

    def test_job_bindings_are_idempotent_and_survive_reopen(self) -> None:
        task = self.make_task()
        path = self.root / 'annotations.db'
        repo, image = self.make_repo(task)
        box = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'box', [[1, 2], [30, 40]])
        repo.save_annotations((box,))
        ref = JobRef(10, 20, (image.id,))
        binding = CvatBinding(image.id, 'shape', 30, box.id)
        prepared = PreparedJob(ref, (binding,))

        repo.bind_job(prepared)
        repo.bind_job(prepared)

        self.assertEqual((binding,), repo.bindings(ref))
        self.assertEqual((binding,), AnnotationRepository(path, task).bindings(ref))

    def test_job_bindings_reject_scope_and_identity_conflicts(self) -> None:
        repo, image = self.make_repo()
        other = ImageRecord('b' * 64, 'b.png', 100, 80, 0)
        repo.register_images((other,))
        first = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'box', [[1, 2], [30, 40]])
        second = AnnotationRecord(uuid4(), other.id, 'detect', None, 'rectangle', 'box', [[1, 2], [30, 40]])
        repo.save_annotations((first, second))
        ref = JobRef(10, 20, (image.id,))
        invalid_groups = (
            (CvatBinding(other.id, 'shape', 1, second.id),),
            (CvatBinding(image.id, 'shape', 1, second.id),),
            (CvatBinding(image.id, 'shape', 1, first.id), CvatBinding(image.id, 'shape', 1, first.id)),
            (CvatBinding(image.id, 'shape', 1, first.id), CvatBinding(image.id, 'tag', 2, first.id)),
        )

        for bindings in invalid_groups:
            with self.subTest(bindings=bindings), self.assertRaises(ValueError):
                repo.bind_job(PreparedJob(ref, bindings))

        unknown_ref = JobRef(10, 21, ('c' * 64,))
        with self.assertRaises(ValueError):
            repo.bind_job(PreparedJob(unknown_ref, ()))
        with self.assertRaises(ValueError):
            repo.bindings(unknown_ref)

        original = CvatBinding(image.id, 'shape', 1, first.id)
        repo.bind_job(PreparedJob(ref, (original,)))
        with self.assertRaises(ValueError):
            repo.bind_job(PreparedJob(ref, (CvatBinding(image.id, 'shape', 1, uuid4()),)))
        with self.assertRaises(ValueError):
            repo.bind_job(PreparedJob(ref, (CvatBinding(image.id, 'shape', 2, first.id),)))
        self.assertEqual((original,), repo.bindings(ref))

    def test_apply_changes_rolls_back_annotations_and_bindings_after_sqlite_failure(self) -> None:
        repo, image = self.make_repo()
        existing = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'box', [[1, 2], [30, 40]])
        repo.save_annotations((existing,))
        ref = JobRef(10, 20, (image.id,))
        original_binding = CvatBinding(image.id, 'shape', 1, existing.id)
        repo.bind_job(PreparedJob(ref, (original_binding,)))
        new = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'box', [[2, 3], [40, 50]])
        changes = AnnotationChanges((replace(existing, geometry=[[3, 4], [31, 41]]), new), frozenset(), frozenset())
        failing_binding = CvatBinding(image.id, 'shape', 999, new.id)
        before_annotations = repo.annotations()
        before_bindings = repo.bindings(ref)
        db_path = self.root / 'annotations.db'
        with closing(sqlite3.connect(db_path)) as connection:
            connection.execute(
                """CREATE TRIGGER reject_test_binding BEFORE INSERT ON cvat_annotation_map
                WHEN NEW.object_id = 999
                BEGIN SELECT RAISE(ABORT, 'injected binding failure'); END"""
            )

        with self.assertRaises(PlatformError) as raised:
            repo.apply_changes(changes, ref=ref, bindings=(failing_binding,))

        self.assertIsInstance(raised.exception.__cause__, sqlite3.Error)
        self.assertNotIn('injected binding failure', str(raised.exception))
        self.assertNotIn(str(db_path), str(raised.exception))
        self.assertEqual(before_annotations, repo.annotations())
        self.assertEqual(before_bindings, repo.bindings(ref))

        with closing(sqlite3.connect(db_path)) as connection:
            connection.execute('DROP TRIGGER reject_test_binding')
        repo.apply_changes(changes, ref=ref, bindings=(failing_binding,))

        self.assertEqual({existing.id, new.id}, {item.id for item in repo.annotations()})
        self.assertEqual({original_binding, failing_binding}, set(repo.bindings(ref)))

    def test_deleting_annotation_removes_its_job_binding(self) -> None:
        repo, image = self.make_repo()
        box = AnnotationRecord(uuid4(), image.id, 'detect', None, 'rectangle', 'box', [[1, 2], [30, 40]])
        repo.save_annotations((box,))
        ref = JobRef(10, 20, (image.id,))
        repo.bind_job(PreparedJob(ref, (CvatBinding(image.id, 'shape', 1, box.id),)))

        repo.apply_changes(AnnotationChanges((), frozenset({box.id}), frozenset()))

        self.assertEqual((), repo.annotations())
        self.assertEqual((), repo.bindings(ref))

    def test_sqlite_read_failure_is_sanitized_for_platform_callers(self) -> None:
        repo, _ = self.make_repo()
        db_path = self.root / 'annotations.db'
        with closing(sqlite3.connect(db_path)) as connection:
            connection.execute('DROP TABLE annotations')

        with self.assertRaises(PlatformError) as raised:
            repo.annotations()

        self.assertIsInstance(raised.exception.__cause__, sqlite3.Error)
        self.assertEqual('Annotation storage operation failed', str(raised.exception))


if __name__ == '__main__':
    unittest.main()
