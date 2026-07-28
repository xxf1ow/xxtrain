import unittest
from dataclasses import FrozenInstanceError
from uuid import UUID, uuid4

import numpy as np

from xxtrain.data import Annotation, AnnotationType, Bbox, Circle, ImageInfo, Points, Polygon, Polyline


class AnnotationTest(unittest.TestCase):
    def test_base_annotation_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            Annotation(label='dial')

    def test_bbox_has_identity_and_immutable_geometry(self) -> None:
        group = uuid4()
        annotation = Bbox(label='dial', x1=1, y1=2, x2=5, y2=8, group=group)

        self.assertIsInstance(annotation.id, UUID)
        self.assertEqual(group, annotation.group)
        self.assertEqual(AnnotationType.BBOX, annotation.type)
        self.assertEqual(((1.0, 2.0), (5.0, 2.0), (5.0, 8.0), (1.0, 8.0)), annotation.points)
        self.assertEqual((1.0, 2.0, 5.0, 8.0), annotation.bbox)
        with self.assertRaises(FrozenInstanceError):
            annotation.label = 'other'

    def test_wrap_and_translate_return_new_values_and_preserve_identity(self) -> None:
        annotation = Polyline(label='needle', points=[[1, 2], [3, 4]], group=7)

        relabeled = annotation.wrap(label='pointer')
        translated = annotation.translate(10, -1)

        self.assertIsNot(annotation, relabeled)
        self.assertEqual(annotation.id, relabeled.id)
        self.assertEqual('pointer', relabeled.label)
        self.assertEqual('needle', annotation.label)
        self.assertEqual(annotation.id, translated.id)
        self.assertEqual(((11.0, 1.0), (13.0, 3.0)), translated.points)
        self.assertEqual(((1.0, 2.0), (3.0, 4.0)), annotation.points)
        with self.assertRaises(TypeError):
            annotation.wrap(points=[[0, 0], [1, 1]])

    def test_numpy_input_cannot_mutate_saved_points(self) -> None:
        source = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
        annotation = Polygon(label='triangle', points=source)

        source[0] = [99.0, 99.0]

        self.assertEqual(((0.0, 0.0), (2.0, 0.0), (1.0, 2.0)), annotation.points)
        with self.assertRaises(TypeError):
            annotation.points[0] = (9.0, 9.0)

    def test_concrete_shape_types_and_bounds(self) -> None:
        cases = [
            (Polygon(label='p', points=[[0, 0], [2, 0], [1, 2]]), AnnotationType.POLYGON, (0, 0, 2, 2)),
            (Polyline(label='ls', points=[[0, 1], [2, 3]]), AnnotationType.POLYLINE, (0, 1, 2, 3)),
            (Points(label='pt', points=[[4, 5]]), AnnotationType.POINTS, (4, 5, 4, 5)),
        ]
        for annotation, expected_type, expected_bbox in cases:
            with self.subTest(annotation=annotation):
                self.assertEqual(expected_type, annotation.type)
                self.assertEqual(tuple(float(value) for value in expected_bbox), annotation.bbox)

    def test_circle_uses_radius_for_complete_bbox(self) -> None:
        circle = Circle(label='circle', center=[10, 10], edge=[13, 14])

        self.assertEqual(((10.0, 10.0), (13.0, 14.0)), circle.points)
        self.assertEqual((5.0, 5.0, 15.0, 15.0), circle.bbox)

    def test_rejects_invalid_common_and_shape_values(self) -> None:
        invalid_factories = [
            lambda: Bbox(label='', x1=0, y1=0, x2=1, y2=1),
            lambda: Bbox(label='x', x1=1, y1=0, x2=1, y2=2),
            lambda: Polygon(label='x', points=[[0, 0], [1, 1]]),
            lambda: Polyline(label='x', points=[[0, 0]]),
            lambda: Points(label='x', points=[]),
            lambda: Circle(label='x', center=[1, 1], edge=[1, 1]),
            lambda: Points(label='x', points=[[float('nan'), 0]]),
        ]
        for factory in invalid_factories:
            with self.subTest(factory=factory), self.assertRaises(ValueError):
                factory()

    def test_rejects_boolean_groups(self) -> None:
        for group in (False, True):
            with self.subTest(group=group), self.assertRaises(ValueError):
                Bbox(label='x', group=group, x1=0, y1=0, x2=1, y2=1)

    def test_image_info_accepts_positive_finite_coordinate_extents(self) -> None:
        integer_info = ImageInfo(width=1920, height=1080)
        fractional_info = ImageInfo(width=20.6, height=10.5)
        self.assertEqual((1920.0, 1080.0), (integer_info.width, integer_info.height))
        self.assertEqual((20.6, 10.5), (fractional_info.width, fractional_info.height))
        self.assertIs(type(integer_info.width), float)
        self.assertIs(type(integer_info.height), float)
        self.assertIs(type(fractional_info.width), float)
        self.assertIs(type(fractional_info.height), float)
        for width, height in ((0, 1), (1, 0), (-1, 2), (float('nan'), 2), (float('inf'), 2), (True, 2)):
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                ImageInfo(width=width, height=height)


if __name__ == '__main__':
    unittest.main()
