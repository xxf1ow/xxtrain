import unittest

from xxtrain.data import Bbox, Circle, Line, Points
from xxtrain.data.geometry import (
    calculate_iou,
    calculate_wide,
    match_parent_children,
    rectangle_contains_point,
    rectangle_contains_shape,
)


class GeometryTest(unittest.TestCase):
    def test_point_and_shape_containment_preserve_inclusive_edges(self) -> None:
        parent = Bbox(label='parent', x1=0, y1=0, x2=10, y2=10)

        self.assertTrue(rectangle_contains_point(parent.bbox, (10, 10)))
        self.assertTrue(rectangle_contains_shape(parent.bbox, Line(label='line', points=[[1, 1], [9, 9]])))
        self.assertFalse(rectangle_contains_shape(parent.bbox, Points(label='point', points=[[11, 5]])))

    def test_circle_containment_uses_radius_and_wide_formula(self) -> None:
        parent = Bbox(label='parent', x1=0, y1=0, x2=10, y2=10)
        inside = Circle(label='circle', center=[5, 5], edge=[8, 5])
        outside = Circle(label='circle', center=[9, 5], edge=[12, 5])

        self.assertTrue(rectangle_contains_shape(parent.bbox, inside))
        self.assertFalse(rectangle_contains_shape(parent.bbox, outside))
        self.assertEqual(0, calculate_wide(100, 50, 10, 0))
        self.assertEqual(1, calculate_wide(100, 50, 10, 0.1))

    def test_iou_preserves_existing_formula(self) -> None:
        self.assertAlmostEqual(1 / 7, calculate_iou((0, 0, 2, 2), (1, 1, 3, 3)))
        self.assertEqual(0, calculate_iou((0, 0, 1, 1), (2, 2, 3, 3)))

    def test_matching_uses_annotation_ids_and_input_order(self) -> None:
        first_parent = Bbox(label='p1', x1=0, y1=0, x2=10, y2=10)
        second_parent = Bbox(label='p2', x1=20, y1=20, x2=30, y2=30)
        first_child = Points(label='c1', points=[[25, 25]])
        second_child = Points(label='c2', points=[[5, 5]])

        mapping = match_parent_children([first_parent, second_parent], [first_child, second_child])

        self.assertEqual({second_parent.id: [first_child.id], first_parent.id: [second_child.id]}, mapping)

    def test_strict_matching_preserves_failure_conditions(self) -> None:
        parent = Bbox(label='p', x1=0, y1=0, x2=10, y2=10)
        orphan = Points(label='c', points=[[20, 20]])

        with self.assertRaisesRegex(ValueError, 'does not belong to any parent'):
            match_parent_children([parent], [orphan], image_path='sample.jpg')
        self.assertEqual({}, match_parent_children([parent], [orphan], strict=False))

        overlapping = Bbox(label='p2', x1=0, y1=0, x2=12, y2=12)
        child = Points(label='c', points=[[5, 5]])
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            match_parent_children([parent, overlapping], [child])


if __name__ == '__main__':
    unittest.main()
