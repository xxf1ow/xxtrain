import copy
import json
import unittest
from pathlib import Path
from uuid import UUID

from xxtrain.integrations.cvat.edit_codec import decode_edit_annotations, decode_edit_bindings, encode_edit_annotations
from xxtrain.platform.contracts import EditAnnotation, EditFrame, EditJob, FrameMapping, JobRef

ORIGINAL_ID = 'a' * 64
TAG_LABELS = [
    {
        'id': 41 + index,
        'name': name,
        'type': 'tag',
        'attributes': [{'id': 71 + index, 'name': 'xxtrain_labelme_extra', 'input_type': 'text'}],
    }
    for index, name in enumerate(('tl', 'tc', 'cl', 'cc'))
]
POLYLINE_LABELS = [
    {
        'id': 51,
        'name': '1',
        'type': 'polyline',
        'attributes': [{'id': 81, 'name': 'xxtrain_labelme_extra', 'input_type': 'text'}],
    }
]


def frame(
    frame_id: str, parent_id: UUID, annotations: tuple[EditAnnotation, ...], *, image_path: str = 'crop.png'
) -> EditFrame:
    mapping = FrameMapping(frame_id, ORIGINAL_ID, parent_id, (10, 20, 50, 60))
    return EditFrame(mapping, Path(image_path), 40, 40, annotations)


class CvatEditCodecTest(unittest.TestCase):
    def test_classification_round_trip_and_initial_binding_use_tag_identity(self) -> None:
        annotation_id = UUID('11111111-1111-1111-1111-111111111111')
        edit_frame = frame(
            '22222222-2222-2222-2222-222222222222',
            UUID('22222222-2222-2222-2222-222222222222'),
            (EditAnnotation(annotation_id, 'classification', 'tc', None),),
        )

        payload = encode_edit_annotations((edit_frame,), TAG_LABELS, mapped=True)

        self.assertEqual(len(payload['tags']), 1)
        self.assertEqual(payload['shapes'], [])
        self.assertEqual(payload['tracks'], [])
        self.assertEqual(payload['tags'][0]['frame'], 0)
        self.assertEqual(payload['tags'][0]['label_id'], 42)
        self.assertEqual(
            json.loads(payload['tags'][0]['attributes'][0]['value']), {'xxtrain_annotation_id': str(annotation_id)}
        )

        payload['tags'][0]['id'] = 101
        job = EditJob(JobRef(7, 8, (ORIGINAL_ID,)), (edit_frame.mapping,))
        self.assertEqual(decode_edit_bindings(payload, (edit_frame,), job, TAG_LABELS)[0].sample_id, ORIGINAL_ID)
        binding = decode_edit_bindings(payload, (edit_frame,), job, TAG_LABELS)[0]
        self.assertEqual((binding.object_type, binding.object_id, binding.annotation_id), ('tag', 101, annotation_id))

        result = decode_edit_annotations(payload, job, TAG_LABELS)
        self.assertEqual(result[0].frame_id, edit_frame.mapping.frame_id)
        self.assertEqual(result[0].annotations, (EditAnnotation(None, 'classification', 'tc', None, 101),))

    def test_copied_initialization_token_never_becomes_edited_identity(self) -> None:
        first_id = UUID('11111111-1111-1111-1111-111111111111')
        first = frame(
            '22222222-2222-2222-2222-222222222222',
            UUID('22222222-2222-2222-2222-222222222222'),
            (EditAnnotation(first_id, 'classification', 'tl', None),),
        )
        second = frame(
            '33333333-3333-3333-3333-333333333333',
            UUID('33333333-3333-3333-3333-333333333333'),
            (),
            image_path='second.png',
        )
        payload = encode_edit_annotations((first, second), TAG_LABELS, mapped=True)
        copied = copy.deepcopy(payload['tags'][0])
        payload['tags'][0]['id'] = 101
        copied.update({'id': 102, 'frame': 1})
        payload['tags'].append(copied)
        job = EditJob(JobRef(7, 8, (ORIGINAL_ID,)), (first.mapping, second.mapping))

        results = decode_edit_annotations(payload, job, TAG_LABELS)

        self.assertEqual(results[0].annotations[0].id, None)
        self.assertEqual(results[1].annotations[0].id, None)
        self.assertEqual([results[0].annotations[0].cvat_id, results[1].annotations[0].cvat_id], [101, 102])
        with self.assertRaisesRegex(ValueError, 'token'):
            decode_edit_bindings(payload, (first, second), job, TAG_LABELS)

    def test_initial_binding_rejects_a_token_moved_to_the_wrong_frame(self) -> None:
        annotation_id = UUID('11111111-1111-1111-1111-111111111111')
        first = frame(
            '22222222-2222-2222-2222-222222222222',
            UUID('22222222-2222-2222-2222-222222222222'),
            (EditAnnotation(annotation_id, 'classification', 'tl', None),),
        )
        second = frame('33333333-3333-3333-3333-333333333333', UUID('33333333-3333-3333-3333-333333333333'), ())
        payload = encode_edit_annotations((first, second), TAG_LABELS, mapped=True)
        payload['tags'][0].update({'id': 101, 'frame': 1})
        job = EditJob(JobRef(7, 8, (ORIGINAL_ID,)), (first.mapping, second.mapping))

        with self.assertRaisesRegex(ValueError, 'frame'):
            decode_edit_bindings(payload, (first, second), job, TAG_LABELS)

    def test_initial_binding_rejects_a_missing_token(self) -> None:
        annotation_id = UUID('11111111-1111-1111-1111-111111111111')
        edit_frame = frame(
            '22222222-2222-2222-2222-222222222222',
            UUID('22222222-2222-2222-2222-222222222222'),
            (EditAnnotation(annotation_id, 'classification', 'tl', None),),
        )
        payload = encode_edit_annotations((edit_frame,), TAG_LABELS, mapped=True)
        payload['tags'][0]['id'] = 101
        payload['tags'][0]['attributes'][0]['value'] = '{}'
        job = EditJob(JobRef(7, 8, (ORIGINAL_ID,)), (edit_frame.mapping,))

        with self.assertRaisesRegex(ValueError, 'token'):
            decode_edit_bindings(payload, (edit_frame,), job, TAG_LABELS)

    def test_initial_binding_rejects_duplicate_native_ids_for_the_same_object_type(self) -> None:
        edit_frame = frame(
            '33333333-3333-3333-3333-333333333333',
            UUID('33333333-3333-3333-3333-333333333333'),
            (
                EditAnnotation(UUID('11111111-1111-1111-1111-111111111111'), 'classification', 'tl', None),
                EditAnnotation(UUID('22222222-2222-2222-2222-222222222222'), 'classification', 'tc', None),
            ),
        )
        payload = encode_edit_annotations((edit_frame,), TAG_LABELS, mapped=True)
        payload['tags'][0]['id'] = 101
        payload['tags'][1]['id'] = 101
        job = EditJob(JobRef(7, 8, (ORIGINAL_ID,)), (edit_frame.mapping,))

        with self.assertRaisesRegex(ValueError, 'unique'):
            decode_edit_bindings(payload, (edit_frame,), job, TAG_LABELS)

    def test_polyline_round_trip_preserves_point_order_and_allows_three_points(self) -> None:
        annotation_id = UUID('11111111-1111-1111-1111-111111111111')
        edit_frame = frame(
            '22222222-2222-2222-2222-222222222222',
            UUID('22222222-2222-2222-2222-222222222222'),
            (EditAnnotation(annotation_id, 'polyline', '1', [[1.5, 2], [3, 4.5], [5, 6]]),),
        )

        payload = encode_edit_annotations((edit_frame,), POLYLINE_LABELS, mapped=True)

        self.assertEqual(payload['tags'], [])
        self.assertEqual(payload['shapes'][0]['type'], 'polyline')
        self.assertEqual(payload['shapes'][0]['points'], [1.5, 2.0, 3.0, 4.5, 5.0, 6.0])
        payload['shapes'][0]['id'] = 101
        job = EditJob(JobRef(7, 8, (ORIGINAL_ID,)), (edit_frame.mapping,))
        decoded = decode_edit_annotations(payload, job, POLYLINE_LABELS)
        self.assertEqual(decoded[0].annotations[0].geometry, [[1.5, 2.0], [3.0, 4.5], [5.0, 6.0]])
        self.assertEqual(decoded[0].annotations[0].cvat_id, 101)

    def test_binding_identity_is_unique_by_object_type_and_native_id(self) -> None:
        tag_id = UUID('11111111-1111-1111-1111-111111111111')
        line_id = UUID('22222222-2222-2222-2222-222222222222')
        edit_frame = frame(
            '33333333-3333-3333-3333-333333333333',
            UUID('33333333-3333-3333-3333-333333333333'),
            (
                EditAnnotation(tag_id, 'classification', 'tl', None),
                EditAnnotation(line_id, 'polyline', '1', [[1, 2], [3, 4]]),
            ),
        )
        labels = TAG_LABELS + POLYLINE_LABELS
        payload = encode_edit_annotations((edit_frame,), labels, mapped=True)
        payload['tags'][0]['id'] = 101
        payload['shapes'][0]['id'] = 101
        job = EditJob(JobRef(7, 8, (ORIGINAL_ID,)), (edit_frame.mapping,))

        bindings = decode_edit_bindings(payload, (edit_frame,), job, labels)

        self.assertEqual(
            {('tag', 101, tag_id), ('shape', 101, line_id)},
            {(binding.object_type, binding.object_id, binding.annotation_id) for binding in bindings},
        )

    def test_decode_rejects_unsupported_tracks_wrong_types_and_missing_ids(self) -> None:
        mapping = FrameMapping(
            '22222222-2222-2222-2222-222222222222',
            ORIGINAL_ID,
            UUID('22222222-2222-2222-2222-222222222222'),
            (10, 20, 50, 60),
        )
        job = EditJob(JobRef(7, 8, (ORIGINAL_ID,)), (mapping,))
        valid_tag = {'id': 101, 'frame': 0, 'label_id': 41, 'attributes': []}
        cases = (
            ({'version': 0, 'tags': [], 'shapes': [], 'tracks': [{}]}, 'tracks'),
            ({'version': 0, 'tags': [valid_tag | {'id': None}], 'shapes': [], 'tracks': []}, 'ID'),
            ({'version': 0, 'tags': [valid_tag | {'label_id': 51}], 'shapes': [], 'tracks': []}, 'type'),
            (
                {
                    'version': 0,
                    'tags': [],
                    'shapes': [
                        {
                            'id': 102,
                            'type': 'polyline',
                            'frame': 0,
                            'label_id': 51,
                            'points': [1, 2, 3],
                            'attributes': [],
                        }
                    ],
                    'tracks': [],
                },
                'coordinates',
            ),
        )

        for payload, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                decode_edit_annotations(payload, job, TAG_LABELS + POLYLINE_LABELS)


if __name__ == '__main__':
    unittest.main()
