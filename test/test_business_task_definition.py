import unittest
from dataclasses import FrozenInstanceError
from uuid import uuid4

from test.task_definitions import synthetic_task_definition
from xxtrain.business_tasks.annotation_rules import step_complete, validate_step_annotations
from xxtrain.business_tasks.definition import AnnotationPolicy, StepDefinition, TaskDefinition
from xxtrain.business_tasks.loader import load_task_definition
from xxtrain.business_tasks.point import point_task_definition
from xxtrain.platform.contracts import EditAnnotation


def annotation(kind: str, label: str | None, geometry=None) -> EditAnnotation:
    return EditAnnotation(uuid4(), kind, label, geometry)


class TaskDefinitionTest(unittest.TestCase):
    def test_dependency_closures_include_parent_and_explicit_edges(self) -> None:
        task = synthetic_task_definition()
        self.assertEqual(task.dependencies('needles'), frozenset({'regions'}))
        self.assertEqual(task.input_steps('needles'), frozenset({'regions', 'needles'}))
        self.assertEqual(task.input_steps('details'), frozenset({'regions', 'subregions', 'details'}))
        self.assertEqual(task.dependent_steps('regions'), frozenset({'kind', 'needles', 'subregions', 'details'}))
        self.assertNotIn('needles', task.dependent_steps('kind'))

    def test_combined_dependency_graph_rejects_unknown_edges_and_cycles(self) -> None:
        root = StepDefinition('root', frozenset({'rectangle'}), frozenset({'x'}), frozenset(), frozenset())
        cases = (
            (
                root,
                StepDefinition('leaf', frozenset({'rectangle'}), frozenset({'x'}), frozenset({'missing'}), frozenset()),
            ),
            (
                StepDefinition('root', frozenset({'rectangle'}), frozenset({'x'}), frozenset(), frozenset({'leaf'})),
                StepDefinition('leaf', frozenset({'rectangle'}), frozenset({'x'}), frozenset({'root'}), frozenset()),
            ),
        )
        for steps in cases:
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                TaskDefinition(steps)

    def test_definition_rejects_unsafe_ids_empty_names_and_invalid_thresholds(self) -> None:
        valid_policy = AnnotationPolicy('rectangle', 'STANDARD')
        bad_steps = (
            StepDefinition(
                '../bad', frozenset({'rectangle'}), frozenset({'x'}), frozenset(), frozenset(), annotation=valid_policy
            ),
            StepDefinition(
                'bad', frozenset({'rectangle'}), frozenset({'x'}), frozenset(), frozenset(), display_name=''
            ),
            StepDefinition('bad', frozenset({'rectangle'}), frozenset(), frozenset(), frozenset()),
            StepDefinition(
                'bad', frozenset({'rectangle'}), frozenset({'x'}), frozenset(), frozenset(), minimum_samples=-1
            ),
        )
        for step in bad_steps:
            with self.subTest(step=step), self.assertRaises(ValueError):
                TaskDefinition((step,))
        with self.assertRaises(ValueError):
            AnnotationPolicy('polyline', 'STANDARD', minimum_annotations=-1)

    def test_definition_rejects_non_string_task_and_step_display_names(self) -> None:
        step = StepDefinition('root', frozenset({'rectangle'}), frozenset({'x'}), frozenset(), frozenset())
        with self.assertRaises(ValueError):
            TaskDefinition((step,), display_name=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            TaskDefinition((step,), key=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            TaskDefinition(
                (
                    StepDefinition(
                        'root', frozenset({'rectangle'}), frozenset({'x'}), frozenset(), frozenset(), display_name=1
                    ),
                )
            )  # type: ignore[arg-type]

    def test_legacy_positional_geometry_combinations_receive_generic_policies(self) -> None:
        task = TaskDefinition(
            (
                StepDefinition(
                    'detect', frozenset({'rectangle', 'negative'}), frozenset({'box'}), frozenset(), frozenset()
                ),
                StepDefinition(
                    'mark', frozenset({'polygon', 'polyline'}), frozenset({'shape'}), frozenset({'detect'}), frozenset()
                ),
            )
        )
        self.assertEqual('negative', task.step('detect').annotation.negative_label)
        self.assertEqual('shapes', task.step('mark').annotation.cvat_type)

    def test_policy_is_immutable(self) -> None:
        policy = AnnotationPolicy('tag', 'TAGS', maximum_annotations=1)
        with self.assertRaises(FrozenInstanceError):
            policy.maximum_annotations = 2

    def test_loader_accepts_trusted_factory_and_rejects_invalid_result(self) -> None:
        self.assertEqual('synthetic', load_task_definition('test.task_definitions:synthetic_task_definition').key)
        with self.assertRaises(ValueError):
            load_task_definition('test.task_definitions:not_a_definition')

    def test_generic_annotation_rules_validate_kind_label_geometry_and_cardinality(self) -> None:
        task = synthetic_task_definition()
        cases = (
            ('kind', (annotation('classification', 'a'),), True),
            ('needles', (annotation('polyline', 'line', [[1, 2], [3, 4]]),), True),
            ('needles', (), False),
        )
        for key, values, complete in cases:
            with self.subTest(key=key, values=values):
                validate_step_annotations(task.step(key), values)
                self.assertEqual(complete, step_complete(task.step(key), values))
        invalid = (
            ('kind', (annotation('classification', 'wrong'),)),
            ('kind', (annotation('classification', 'a'), annotation('classification', 'b'))),
            ('kind', (annotation('classification', 'a', []),)),
            ('needles', (annotation('polyline', 'line', [[1, 2]]),)),
            ('needles', (annotation('polyline', 'line', [[1, 2], [1, 2]]),)),
        )
        for key, values in invalid:
            with self.subTest(key=key, values=values), self.assertRaises(ValueError):
                validate_step_annotations(task.step(key), values)

    def test_generic_negative_requires_policy_capability_and_null_content(self) -> None:
        detect = point_task_definition().step('detect')
        validate_step_annotations(detect, (annotation('negative', None),))
        for value in (annotation('negative', 'Point'), annotation('negative', None, [])):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_step_annotations(detect, (value,))
        with self.assertRaises(ValueError):
            validate_step_annotations(
                detect, (annotation('negative', None), annotation('rectangle', 'Point', [[1, 2], [3, 4]]))
            )
        without_negative = synthetic_task_definition().step('regions')
        with self.assertRaises(ValueError):
            validate_step_annotations(without_negative, (annotation('negative', None),))

    def test_point_rules_define_parallel_children_and_detection_threshold(self) -> None:
        task = point_task_definition()
        self.assertEqual(('point', 'Point'), (task.key, task.display_name))
        detect = task.step('detect')
        classify = task.step('classify')
        segment = task.step('segment')
        self.assertEqual(50, detect.minimum_samples)
        self.assertEqual(frozenset({'detect'}), classify.parent_steps)
        self.assertEqual(frozenset({'detect'}), segment.parent_steps)
        self.assertEqual(frozenset(), classify.depends_on)
        self.assertEqual(frozenset(), segment.depends_on)
        self.assertEqual(AnnotationPolicy('tag', 'TAGS', maximum_annotations=1), classify.annotation)
        self.assertEqual(AnnotationPolicy('polyline', 'STANDARD', point_count=2), segment.annotation)


if __name__ == '__main__':
    unittest.main()
