import json
import unittest
from pathlib import Path

from xxtrain.business_tasks import MODEL_TARGETS, POINT_BOX_LABELS
from xxtrain.data import Bbox
from xxtrain.integrations.cvat.codec import decode_annotations, encode_annotations
from xxtrain.platform.contracts import DetectionBox, FrameResult, ImageInput, JobRef

LABELS = [
    {
        'id': 41 + index,
        'name': name,
        'attributes': [{'id': 71 + index, 'name': 'xxtrain_labelme_extra', 'input_type': 'text'}],
    }
    for index, name in enumerate(('Point', 'tl', 'tc', 'cl', 'cc'))
]


def rectangle(*, frame: int, label_id: int, attribute_id: int | None, extra: dict | None = None) -> dict:
    attributes = []
    if attribute_id is not None:
        attributes.append({'spec_id': attribute_id, 'value': json.dumps(extra, ensure_ascii=False)})
    return {
        'type': 'rectangle',
        'frame': frame,
        'label_id': label_id,
        'points': [1.0 + frame, 2.0, 11.0 + frame, 12.0],
        'occluded': False,
        'outside': False,
        'rotation': 0,
        'z_order': 0,
        'attributes': attributes,
    }


class PointDefinitionTest(unittest.TestCase):
    def test_point_exposes_all_annotation_targets(self):
        self.assertEqual(POINT_BOX_LABELS, ('Point', 'tl', 'tc', 'cl', 'cc'))
        self.assertEqual(MODEL_TARGETS, (('detect', True), ('classify', True), ('segment', True)))


class CvatCodecTest(unittest.TestCase):
    def test_encode_uses_real_label_and_attribute_ids_for_all_point_labels(self):
        boxes = tuple(
            DetectionBox(
                geometry=Bbox(label=name, x1=index + 0.5, y1=2, x2=index + 10.5, y2=12),
                extra={'group_id': index, 'description': f'框 {name}'},
            )
            for index, name in enumerate(POINT_BOX_LABELS)
        )
        images = (ImageInput('sample-a', Path('a.jpg'), 100, 80, boxes),)

        payload = encode_annotations(images, LABELS)

        self.assertEqual(payload['version'], 0)
        self.assertEqual(payload['tracks'], [])
        self.assertEqual(payload['tags'], [])
        self.assertEqual([shape['label_id'] for shape in payload['shapes']], [41, 42, 43, 44, 45])
        self.assertEqual([shape['attributes'][0]['spec_id'] for shape in payload['shapes']], [71, 72, 73, 74, 75])
        self.assertEqual(payload['shapes'][0]['points'], [0.5, 2.0, 10.5, 12.0])
        self.assertEqual(
            json.loads(payload['shapes'][4]['attributes'][0]['value']), {'group_id': 4, 'description': '框 cc'}
        )

    def test_literal_shapes_decode_by_frame_and_preserve_labels_and_extras(self):
        payload = {
            'version': 0,
            'shapes': [
                {
                    'type': 'rectangle',
                    'frame': 1,
                    'label_id': 45,
                    'points': [2.0, 2.0, 12.0, 12.0],
                    'occluded': False,
                    'outside': False,
                    'rotation': 0,
                    'z_order': 0,
                    'attributes': [{'spec_id': 75, 'value': '{"flags": {"reviewed": true}}'}],
                },
                {
                    'type': 'rectangle',
                    'frame': 0,
                    'label_id': 41,
                    'points': [1.0, 2.0, 11.0, 12.0],
                    'occluded': False,
                    'outside': False,
                    'rotation': 0,
                    'z_order': 0,
                    'attributes': [{'spec_id': 71, 'value': '{"group_id": 9, "description": "原始"}'}],
                },
                {
                    'type': 'rectangle',
                    'frame': 1,
                    'label_id': 42,
                    'points': [2.0, 2.0, 12.0, 12.0],
                    'occluded': False,
                    'outside': False,
                    'rotation': 0,
                    'z_order': 0,
                    'attributes': [{'spec_id': 72, 'value': '{"custom": ["x", 2]}'}],
                },
                {
                    'type': 'rectangle',
                    'frame': 0,
                    'label_id': 44,
                    'points': [1.0, 2.0, 11.0, 12.0],
                    'occluded': False,
                    'outside': False,
                    'rotation': 0,
                    'z_order': 0,
                    'attributes': [{'spec_id': 74, 'value': '{}'}],
                },
                {
                    'type': 'rectangle',
                    'frame': 1,
                    'label_id': 43,
                    'points': [2.0, 2.0, 12.0, 12.0],
                    'occluded': False,
                    'outside': False,
                    'rotation': 0,
                    'z_order': 0,
                    'attributes': [{'spec_id': 73, 'value': '{"score": null}'}],
                },
            ],
            'tracks': [],
            'tags': [],
        }
        ref = JobRef(task_id=7, job_id=8, sample_ids=('a', 'b'))

        result = decode_annotations(payload, ref, LABELS)

        self.assertEqual([box.geometry.label for box in result[0].boxes], ['Point', 'cl'])
        self.assertEqual([box.geometry.label for box in result[1].boxes], ['cc', 'tl', 'tc'])
        self.assertEqual(result[0].boxes[0].extra, {'group_id': 9, 'description': '原始'})
        self.assertEqual(result[1].boxes[0].extra, {'flags': {'reviewed': True}})
        self.assertEqual(result[1].boxes[1].extra, {'custom': ['x', 2]})
        self.assertEqual(result[1].boxes[2].geometry.bbox, (2.0, 2.0, 12.0, 12.0))

    def test_empty_frames_are_not_dropped(self):
        ref = JobRef(task_id=7, job_id=8, sample_ids=('a', 'b'))
        payload = {'version': 0, 'shapes': [], 'tracks': [], 'tags': []}
        result = decode_annotations(payload, ref, [])
        self.assertEqual(result, (FrameResult('a', ()), FrameResult('b', ())))

    def test_missing_reserved_attribute_defaults_to_empty_extra(self):
        payload = {
            'version': 0,
            'shapes': [rectangle(frame=0, label_id=41, attribute_id=None)],
            'tracks': [],
            'tags': [],
        }

        result = decode_annotations(payload, JobRef(7, 8, ('a',)), LABELS)

        self.assertEqual(result[0].boxes[0].extra, {})

    def test_out_of_range_frame_is_rejected(self):
        payload = {
            'version': 0,
            'shapes': [rectangle(frame=1, label_id=41, attribute_id=71, extra={})],
            'tracks': [],
            'tags': [],
        }
        with self.assertRaisesRegex(ValueError, 'frame'):
            decode_annotations(payload, JobRef(7, 8, ('a',)), LABELS)

    def test_unknown_label_is_rejected(self):
        payload = {
            'version': 0,
            'shapes': [rectangle(frame=0, label_id=999, attribute_id=None)],
            'tracks': [],
            'tags': [],
        }
        with self.assertRaisesRegex(ValueError, 'label'):
            decode_annotations(payload, JobRef(7, 8, ('a',)), LABELS)

    def test_tracks_are_rejected(self):
        payload = {'version': 0, 'shapes': [], 'tracks': [{'frame': 0}], 'tags': []}
        with self.assertRaisesRegex(ValueError, 'tracks'):
            decode_annotations(payload, JobRef(7, 8, ('a',)), LABELS)

    def test_non_rectangle_shape_is_rejected(self):
        shape = rectangle(frame=0, label_id=41, attribute_id=71, extra={})
        shape['type'] = 'polygon'
        payload = {'version': 0, 'shapes': [shape], 'tracks': [], 'tags': []}
        with self.assertRaisesRegex(ValueError, 'rectangle'):
            decode_annotations(payload, JobRef(7, 8, ('a',)), LABELS)

    def test_unmapped_shape_attribute_is_rejected(self):
        shape = rectangle(frame=0, label_id=41, attribute_id=71, extra={})
        shape['attributes'].append({'spec_id': 999, 'value': 'cannot preserve'})
        payload = {'version': 0, 'shapes': [shape], 'tracks': [], 'tags': []}
        with self.assertRaisesRegex(ValueError, 'attribute'):
            decode_annotations(payload, JobRef(7, 8, ('a',)), LABELS)

    def test_reserved_attribute_must_contain_a_json_object(self):
        shape = rectangle(frame=0, label_id=41, attribute_id=71, extra={})
        shape['attributes'][0]['value'] = '[1, 2]'
        payload = {'version': 0, 'shapes': [shape], 'tracks': [], 'tags': []}
        with self.assertRaisesRegex(ValueError, 'JSON object'):
            decode_annotations(payload, JobRef(7, 8, ('a',)), LABELS)


if __name__ == '__main__':
    unittest.main()
