import unittest
from dataclasses import FrozenInstanceError
from uuid import uuid4

from xxtrain.data import Annotation, AnnotationType, Bbox, Keypoint, Points, Polygon, Pose, assemble_poses


class PoseTest(unittest.TestCase):
    def test_assemble_grouped_pose_replaces_bbox_and_points_atomically(self) -> None:
        annotations = (
            Bbox(label='person', group=7, x1=0, y1=0, x2=100, y2=100),
            Points(label='nose', group=7, points=((10, 20),)),
            Points(label='wrist', group=7, points=((30, 40),)),
            Polygon(label='mask', points=((0, 0), (2, 0), (1, 1))),
        )

        output = assemble_poses(annotations)

        self.assertEqual((Pose, Polygon), tuple(type(item) for item in output))
        self.assertEqual('person', output[0].label)
        self.assertEqual(7, output[0].group)
        self.assertEqual(('nose', 'wrist'), tuple(point.label for point in output[0].keypoints))

    def test_assemble_grouped_pose_does_not_guess_ambiguous_groups(self) -> None:
        annotations = (
            Bbox(label='a', group=1, x1=0, y1=0, x2=10, y2=10),
            Bbox(label='b', group=1, x1=0, y1=0, x2=10, y2=10),
            Points(label='point', group=1, points=((1, 1),)),
        )

        self.assertEqual(annotations, assemble_poses(annotations))

    def test_assemble_pose_can_match_ungrouped_fragments_by_unique_containment(self) -> None:
        annotations = (
            Bbox(label='person', x1=0, y1=0, x2=100, y2=100),
            Points(label='wrist', points=((30, 40),)),
            Points(label='nose', points=((10, 20),)),
        )

        output = assemble_poses(annotations, keypoint_labels=('nose', 'wrist'), match_ungrouped=True)

        self.assertEqual((Pose,), tuple(type(item) for item in output))
        self.assertEqual(('nose', 'wrist'), tuple(point.label for point in output[0].keypoints))

    def test_assemble_ungrouped_pose_rejects_zero_or_multiple_containment_matches(self) -> None:
        orphan = (Bbox(label='person', x1=0, y1=0, x2=10, y2=10), Points(label='nose', points=((20, 20),)))
        ambiguous = (
            Bbox(label='a', x1=0, y1=0, x2=10, y2=10),
            Bbox(label='b', x1=0, y1=0, x2=10, y2=10),
            Points(label='nose', points=((5, 5),)),
        )

        for annotations in (orphan, ambiguous):
            with self.subTest(annotations=annotations), self.assertRaises(ValueError):
                assemble_poses(annotations, match_ungrouped=True)

    def test_assemble_pose_catalog_orders_and_validates_keypoints(self) -> None:
        valid = (
            Bbox(label='person', group=1, x1=0, y1=0, x2=10, y2=10),
            Points(label='wrist', group=1, points=((5, 5),)),
            Points(label='nose', group=1, points=((2, 2),)),
        )
        invalid = (
            (Bbox(label='person', group=1, x1=0, y1=0, x2=10, y2=10), Points(label='nose', group=1, points=((2, 2),))),
            (
                Bbox(label='person', group=1, x1=0, y1=0, x2=10, y2=10),
                Points(label='nose', group=1, points=((2, 2),)),
                Points(label='nose', group=1, points=((3, 3),)),
            ),
            (
                Bbox(label='person', group=1, x1=0, y1=0, x2=10, y2=10),
                Points(label='nose', group=1, points=((2, 2),)),
                Points(label='unknown', group=1, points=((3, 3),)),
            ),
        )

        output = assemble_poses(valid, keypoint_labels=('nose', 'wrist'))

        self.assertEqual(('nose', 'wrist'), tuple(point.label for point in output[0].keypoints))
        for annotations in invalid:
            with self.subTest(annotations=annotations), self.assertRaises(ValueError):
                assemble_poses(annotations, keypoint_labels=('nose', 'wrist'))

    def test_assemble_pose_strictly_rejects_malformed_candidate_groups(self) -> None:
        malformed = (
            (
                Bbox(label='person', group=1, x1=0, y1=0, x2=10, y2=10),
                Polygon(label='mask', group=1, points=((0, 0), (2, 0), (1, 1))),
            ),
            (
                Bbox(label='person', group=1, x1=0, y1=0, x2=10, y2=10),
                Bbox(label='other', group=1, x1=0, y1=0, x2=10, y2=10),
                Points(label='nose', group=1, points=((1, 1),)),
            ),
            (
                Bbox(label='person', group=1, x1=0, y1=0, x2=10, y2=10),
                Points(label='nose', group=1, points=((1, 1), (2, 2))),
            ),
        )

        for annotations in malformed:
            with self.subTest(annotations=annotations), self.assertRaises(ValueError):
                assemble_poses(annotations, keypoint_labels=('nose',))

    def test_assemble_pose_preserves_stable_annotation_order(self) -> None:
        annotations = (
            Polygon(label='before', points=((0, 0), (2, 0), (1, 1))),
            Points(label='nose', group=3, points=((2, 2),)),
            Bbox(label='person', group=3, x1=0, y1=0, x2=10, y2=10),
            Polygon(label='after', points=((0, 0), (2, 0), (1, 1))),
        )

        output = assemble_poses(annotations)

        self.assertEqual((Polygon, Pose, Polygon), tuple(type(item) for item in output))

    def test_pose_is_an_immutable_annotation_with_ordered_keypoints(self) -> None:
        pose = Pose(
            label='person',
            x1=1,
            y1=2,
            x2=11,
            y2=22,
            keypoints=[Keypoint(label='nose', x=3, y=4), Keypoint(label='eye', x=5, y=6, visibility=1)],
        )

        self.assertIsInstance(pose, Annotation)
        self.assertEqual(AnnotationType.POSE, pose.type)
        self.assertEqual((1.0, 2.0, 11.0, 22.0), pose.bbox)
        self.assertEqual(('nose', 'eye'), tuple(keypoint.label for keypoint in pose.keypoints))
        self.assertEqual((2, 1), tuple(keypoint.visibility for keypoint in pose.keypoints))
        with self.assertRaises(FrozenInstanceError):
            pose.label = 'other'
        with self.assertRaises(FrozenInstanceError):
            pose.keypoints[0].x = 9

    def test_translate_moves_pose_and_keypoints_preserving_identity_and_visibility(self) -> None:
        identifier = uuid4()
        pose = Pose(
            label='person',
            x1=1,
            y1=2,
            x2=11,
            y2=22,
            keypoints=[Keypoint(label='nose', x=3, y=4, visibility=0)],
            id=identifier,
            group='people',
        )

        translated = pose.translate(10, -2)

        self.assertEqual((11.0, 0.0, 21.0, 20.0), translated.bbox)
        translated_keypoint = translated.keypoints[0]
        self.assertEqual((13.0, 2.0, 0), (translated_keypoint.x, translated_keypoint.y, translated_keypoint.visibility))
        self.assertEqual(identifier, translated.id)
        self.assertEqual('people', translated.group)
        self.assertEqual('person', translated.label)

    def test_rejects_invalid_keypoints_and_pose_geometry(self) -> None:
        valid_keypoint = Keypoint(label='nose', x=1, y=2)
        invalid_factories = [
            lambda: Keypoint(label='', x=1, y=2),
            lambda: Keypoint(label='nose', x=float('nan'), y=2),
            lambda: Keypoint(label='nose', x=1, y=2, visibility=3),
            lambda: Keypoint(label='nose', x=1, y=2, visibility=True),
            lambda: Pose(label='person', x1=1, y1=2, x2=1, y2=3, keypoints=[valid_keypoint]),
            lambda: Pose(label='person', x1=1, y1=2, x2=3, y2=2, keypoints=[valid_keypoint]),
            lambda: Pose(label='person', x1=1, y1=2, x2=3, y2=4, keypoints=[]),
            lambda: Pose(
                label='person', x1=1, y1=2, x2=3, y2=4, keypoints=[valid_keypoint, Keypoint(label='nose', x=2, y=3)]
            ),
        ]

        for factory in invalid_factories:
            with self.subTest(factory=factory), self.assertRaises(ValueError):
                factory()


if __name__ == '__main__':
    unittest.main()
