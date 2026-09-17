import unittest
from dataclasses import replace
from uuid import uuid4

from xxtrain.business_tasks.definition import StepDefinition, TaskDefinition
from xxtrain.platform.contracts import AnnotationRecord
from xxtrain.workspace_data.changes import plan_changes


class PlatformAnnotationChangesTest(unittest.TestCase):
    def make_task(self) -> TaskDefinition:
        return TaskDefinition(
            (
                StepDefinition('detect', frozenset({'rectangle'}), frozenset({'Point'}), frozenset(), frozenset()),
                StepDefinition(
                    'classify',
                    frozenset({'classification'}),
                    frozenset({'tl', 'tc'}),
                    frozenset({'detect'}),
                    frozenset({'detect'}),
                ),
                StepDefinition(
                    'segment',
                    frozenset({'polygon'}),
                    frozenset({'mask'}),
                    frozenset({'detect'}),
                    frozenset({'classify'}),
                ),
            )
        )

    def make_records(
        self,
    ) -> tuple[AnnotationRecord, AnnotationRecord, AnnotationRecord, AnnotationRecord, AnnotationRecord]:
        image_id = 'a' * 64
        first = AnnotationRecord(uuid4(), image_id, 'detect', None, 'rectangle', 'Point', [[0, 0], [20, 20]])
        second = replace(first, id=uuid4(), geometry=[[30, 0], [50, 20]])
        category = AnnotationRecord(uuid4(), image_id, 'classify', first.id, 'classification', 'tl', None)
        first_mask = AnnotationRecord(
            uuid4(), image_id, 'segment', first.id, 'polygon', 'mask', [[1, 1], [10, 1], [10, 10]]
        )
        second_mask = replace(first_mask, id=uuid4(), parent_id=second.id)
        return first, second, category, first_mask, second_mask

    def test_category_change_clears_sibling_mask_only_for_same_object(self) -> None:
        first, second, category, first_mask, second_mask = self.make_records()

        result = plan_changes(
            (first, second, category, first_mask, second_mask),
            (replace(category, label='tc'),),
            frozenset(),
            self.make_task(),
        )

        self.assertEqual((replace(category, label='tc'),), result.upserts)
        self.assertEqual(frozenset({first_mask.id}), result.delete_ids)
        self.assertEqual(frozenset({'segment'}), result.invalidated_steps)

    def test_equivalent_records_are_no_op_regardless_of_order_or_numeric_representation(self) -> None:
        first, second, category, _, _ = self.make_records()
        incoming = (
            replace(second, geometry=[[30.0, 0.0], [50.0, 20.0]]),
            category,
            replace(first, geometry=[[0.0, 0.0], [20.0, 20.0]]),
        )

        result = plan_changes((first, second, category), incoming, frozenset(), self.make_task())

        self.assertEqual((), result.upserts)
        self.assertEqual(frozenset(), result.delete_ids)
        self.assertEqual(frozenset(), result.invalidated_steps)

    def test_existing_annotation_cannot_change_assignment(self) -> None:
        first, second, category, _, _ = self.make_records()
        replacements = (
            replace(category, image_id='b' * 64),
            replace(category, step_key='segment', kind='polygon', label='mask', geometry=[[0, 0], [1, 0], [0, 1]]),
            replace(category, parent_id=second.id),
        )

        for incoming in replacements:
            with self.subTest(incoming=incoming), self.assertRaises(ValueError):
                plan_changes((first, second, category), (incoming,), frozenset(), self.make_task())

    def test_deleting_root_removes_its_tree_without_touching_another_root(self) -> None:
        first, second, category, first_mask, second_mask = self.make_records()

        result = plan_changes(
            (first, second, category, first_mask, second_mask), (), frozenset({first.id}), self.make_task()
        )

        self.assertEqual(frozenset({first.id, category.id, first_mask.id}), result.delete_ids)
        self.assertEqual(frozenset({'classify', 'segment'}), result.invalidated_steps)
        self.assertNotIn(second.id, result.delete_ids)
        self.assertNotIn(second_mask.id, result.delete_ids)

    def test_new_upstream_record_invalidates_downstream_steps_without_existing_children(self) -> None:
        first, _, _, _, _ = self.make_records()

        result = plan_changes((), (first,), frozenset(), self.make_task())

        self.assertEqual((first,), result.upserts)
        self.assertEqual(frozenset(), result.delete_ids)
        self.assertEqual(frozenset({'classify', 'segment'}), result.invalidated_steps)

    def test_changed_upstream_record_invalidates_downstream_steps_without_existing_children(self) -> None:
        first, _, _, _, _ = self.make_records()

        result = plan_changes((first,), (replace(first, geometry=[[0, 0], [21, 20]]),), frozenset(), self.make_task())

        self.assertEqual(frozenset(), result.delete_ids)
        self.assertEqual(frozenset({'classify', 'segment'}), result.invalidated_steps)

    def test_dependent_child_and_its_subtree_are_removed_for_a_detection_chain(self) -> None:
        task = TaskDefinition(
            (
                StepDefinition('detect', frozenset({'rectangle'}), frozenset({'root'}), frozenset(), frozenset()),
                StepDefinition(
                    'refine',
                    frozenset({'rectangle'}),
                    frozenset({'refined'}),
                    frozenset({'detect'}),
                    frozenset({'detect'}),
                ),
                StepDefinition(
                    'segment', frozenset({'polygon'}), frozenset({'mask'}), frozenset({'refine'}), frozenset({'refine'})
                ),
            )
        )
        root = AnnotationRecord(uuid4(), 'a' * 64, 'detect', None, 'rectangle', 'root', [[0, 0], [20, 20]])
        refined = AnnotationRecord(
            uuid4(), root.image_id, 'refine', root.id, 'rectangle', 'refined', [[1, 1], [19, 19]]
        )
        mask = AnnotationRecord(
            uuid4(), root.image_id, 'segment', refined.id, 'polygon', 'mask', [[2, 2], [10, 2], [10, 10]]
        )

        result = plan_changes(
            (root, refined, mask), (replace(root, label='root', geometry=[[0, 0], [21, 20]]),), frozenset(), task
        )

        self.assertEqual(frozenset({refined.id, mask.id}), result.delete_ids)
        self.assertEqual(frozenset({'refine', 'segment'}), result.invalidated_steps)

        refined_result = plan_changes(
            (root, refined, mask), (replace(refined, geometry=[[1, 1], [18, 19]]),), frozenset(), task
        )

        self.assertEqual(frozenset({mask.id}), refined_result.delete_ids)
        self.assertEqual(frozenset({'segment'}), refined_result.invalidated_steps)

    def test_polygon_vertex_order_is_content_not_annotation_order(self) -> None:
        first, _, category, mask, _ = self.make_records()
        reversed_mask = replace(mask, geometry=list(reversed(mask.geometry)))  # type: ignore[arg-type]

        result = plan_changes((first, category, mask), (reversed_mask,), frozenset(), self.make_task())

        self.assertEqual((reversed_mask,), result.upserts)

    def test_equal_geometry_with_a_new_id_is_delete_and_add_not_identity_matching(self) -> None:
        current, _, _, _, _ = self.make_records()
        replacement = replace(current, id=uuid4())

        result = plan_changes((current,), (replacement,), frozenset({current.id}), self.make_task())

        self.assertEqual((replacement,), result.upserts)
        self.assertEqual(frozenset({current.id}), result.delete_ids)


if __name__ == '__main__':
    unittest.main()
