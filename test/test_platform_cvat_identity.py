import copy
import json
import unittest
from pathlib import Path
from uuid import UUID

from xxtrain.business_tasks import POINT_BOX_LABELS
from xxtrain.data import Bbox
from xxtrain.integrations.cvat.codec import decode_annotations, decode_initial_bindings, encode_mapped_annotations
from xxtrain.platform.contracts import DetectionBox, ImageInput, JobRef

LABELS = [
    {
        'id': 41 + index,
        'name': name,
        'attributes': [
            {
                'id': 71 + index,
                'name': 'xxtrain_labelme_extra',
                'mutable': True,
                'input_type': 'text',
                'default_value': '{}',
                'values': [],
            }
        ],
    }
    for index, name in enumerate(POINT_BOX_LABELS)
]


class CvatIdentityCodecTest(unittest.TestCase):
    def image_with_two_boxes(self) -> ImageInput:
        return ImageInput(
            'sample-a',
            Path('a.jpg'),
            100,
            80,
            (
                DetectionBox(Bbox(label='tl', x1=1, y1=2, x2=11, y2=12), {'description': 'first'}),
                DetectionBox(Bbox(label='tl', x1=1, y1=2, x2=11, y2=12), {'description': 'second'}),
            ),
        )

    def mapped_payload(self) -> tuple[ImageInput, JobRef, dict]:
        image = self.image_with_two_boxes()
        ref = JobRef(7, 8, (image.sample_id,))
        payload = encode_mapped_annotations((image,), LABELS)
        payload['shapes'][0]['id'] = 101
        payload['shapes'][1]['id'] = 102
        return image, ref, payload

    def test_initial_mapping_uses_token_not_response_order(self):
        image, ref, payload = self.mapped_payload()
        payload['shapes'].reverse()

        bindings = decode_initial_bindings(payload, (image,), ref, LABELS)

        self.assertEqual(
            {101: image.boxes[0].geometry.id, 102: image.boxes[1].geometry.id},
            {binding.object_id: binding.annotation_id for binding in bindings},
        )
        self.assertEqual({'sample-a'}, {binding.sample_id for binding in bindings})
        self.assertEqual({'shape'}, {binding.object_type for binding in bindings})

    def test_initial_mapping_rejects_duplicate_token(self):
        image, ref, payload = self.mapped_payload()
        first_extra = payload['shapes'][0]['attributes'][0]['value']
        payload['shapes'][1]['attributes'][0]['value'] = first_extra

        with self.assertRaisesRegex(ValueError, 'token'):
            decode_initial_bindings(payload, (image,), ref, LABELS)

    def test_initial_mapping_rejects_missing_token(self):
        image, ref, payload = self.mapped_payload()
        extra = json.loads(payload['shapes'][0]['attributes'][0]['value'])
        del extra['xxtrain_annotation_id']
        payload['shapes'][0]['attributes'][0]['value'] = json.dumps(extra)

        with self.assertRaisesRegex(ValueError, 'token'):
            decode_initial_bindings(payload, (image,), ref, LABELS)

    def test_initial_mapping_rejects_duplicate_cvat_id(self):
        image, ref, payload = self.mapped_payload()
        payload['shapes'][1]['id'] = 101

        with self.assertRaisesRegex(ValueError, 'ID'):
            decode_initial_bindings(payload, (image,), ref, LABELS)

    def test_initial_mapping_rejects_non_positive_or_boolean_cvat_id(self):
        for invalid_id in (0, -1, True):
            with self.subTest(invalid_id=invalid_id):
                image, ref, payload = self.mapped_payload()
                payload['shapes'][0]['id'] = invalid_id

                with self.assertRaisesRegex(ValueError, 'ID'):
                    decode_initial_bindings(payload, (image,), ref, LABELS)

    def test_initial_mapping_rejects_token_on_wrong_frame(self):
        first = self.image_with_two_boxes()
        second = ImageInput(
            'sample-b', Path('b.jpg'), 100, 80, (DetectionBox(Bbox(label='cc', x1=3, y1=4, x2=13, y2=14)),)
        )
        images = (first, second)
        ref = JobRef(7, 8, tuple(image.sample_id for image in images))
        payload = encode_mapped_annotations(images, LABELS)
        for object_id, shape in enumerate(payload['shapes'], start=101):
            shape['id'] = object_id
        payload['shapes'][0]['frame'] = 1

        with self.assertRaisesRegex(ValueError, 'frame'):
            decode_initial_bindings(payload, images, ref, LABELS)

    def test_empty_initial_annotations_produce_no_bindings(self):
        image = ImageInput('sample-a', Path('a.jpg'), 100, 80, ())
        ref = JobRef(7, 8, ('sample-a',))
        payload = encode_mapped_annotations((image,), LABELS)

        self.assertEqual((), decode_initial_bindings(payload, (image,), ref, LABELS))

    def test_fetch_uses_native_ids_and_strips_copied_initialization_token(self):
        image = ImageInput(
            'sample-a',
            Path('a.jpg'),
            100,
            80,
            (DetectionBox(Bbox(label='tl', x1=1, y1=2, x2=11, y2=12), {'description': 'keep'}),),
        )
        ref = JobRef(7, 8, ('sample-a',))
        payload = encode_mapped_annotations((image,), LABELS)
        copied = copy.deepcopy(payload['shapes'][0])
        payload['shapes'][0]['id'] = 101
        copied['id'] = 102
        payload['shapes'].append(copied)

        boxes = decode_annotations(payload, ref, LABELS)[0].boxes

        self.assertEqual([101, 102], [box.cvat_id for box in boxes])
        self.assertEqual([{'description': 'keep'}, {'description': 'keep'}], [box.extra for box in boxes])
        self.assertEqual(2, len({box.geometry.id for box in boxes}))
        self.assertNotIn(image.boxes[0].geometry.id, {box.geometry.id for box in boxes})
        self.assertTrue(all(isinstance(box.geometry.id, UUID) for box in boxes))


if __name__ == '__main__':
    unittest.main()
