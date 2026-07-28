import unittest
from dataclasses import FrozenInstanceError
from uuid import uuid4

from xxtrain.data import Annotation, AnnotationType, Keypoint, Pose


class PoseTest(unittest.TestCase):
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
