import json
import os
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from xxtrain.data import Bbox
from xxtrain.platform.cache import build_detection_cache
from xxtrain.platform.contracts import DetectionBox, DetectionSummary, FrameResult, PlatformAccessError, UploadResult
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.dedup import SIMILARITY_DISTANCE, hamming_distance, perceptual_hash
from xxtrain.workspace_data.labelme import merge_detection


class PlatformDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images_dir = self.root / 'images'
        self.annotations_dir = self.root / 'annotations'
        self.staging_dir = self.root / 'staging'
        self.images_dir.mkdir()
        self.annotations_dir.mkdir()
        self.staging_dir.mkdir()
        self.workspace = WorkspaceData(self.root)

    def make_image(self, name: str = 'a.jpg', *, size: tuple[int, int] = (64, 48)) -> Path:
        path = self.images_dir / name
        Image.new('RGB', size, 'white').save(path)
        return path

    def stage_image(self, name: str, *, color: str = 'white') -> Path:
        path = self.staging_dir / name
        Image.new('RGB', (64, 48), color).save(path)
        return path

    def stage_images(self, *names: str) -> tuple[Path, ...]:
        first = self.stage_image(names[0])
        exact_copy = self.staging_dir / names[1]
        exact_copy.write_bytes(first.read_bytes())
        near_copy = self.stage_image(names[2], color='gray')
        return first, exact_copy, near_copy

    def accept_image(self, name: str) -> None:
        self.workspace.admit((self.stage_image(name),))

    def accept_boxed_and_negative_workspace(self) -> None:
        boxed = self.stage_image('boxed.jpg')
        negative = self.staging_dir / 'negative.jpg'
        with Image.new('RGB', (64, 48), 'white') as image:
            for x in range(64):
                for y in range(48):
                    value = (x * 19 + y * 31) % 256
                    image.putpixel((x, y), (value, value, value))
            image.save(negative)
        self.workspace.admit((boxed, negative))
        boxed_sample, negative_sample = self.workspace.images()
        self.write_annotation(
            boxed_sample.sample_id,
            {'shapes': [{'label': 'tl', 'shape_type': 'rectangle', 'points': [[1, 2], [30, 40]]}]},
        )
        self.write_annotation(negative_sample.sample_id, {'flags': {'xxtrain_detection_negative': True}, 'shapes': []})

    def image_names(self) -> list[str]:
        return sorted(path.name for path in self.images_dir.iterdir())

    def write_annotation(self, stem: str, document: dict[str, object]) -> Path:
        path = self.annotations_dir / f'{stem}.json'
        path.write_text(json.dumps(document), encoding='utf-8')
        return path

    def test_admit_renames_by_sha_and_deduplicates_in_deterministic_order(self) -> None:
        first, exact_copy, near_copy = self.stage_images('first.jpg', 'copy.jpg', 'near.jpg')
        expected_name = f'{min(sha256(path.read_bytes()).hexdigest() for path in (first, near_copy))}.jpg'

        result = self.workspace.admit((near_copy, exact_copy, first))

        self.assertEqual(3, result.received_count)
        self.assertEqual(1, result.accepted_count)
        self.assertEqual(1, result.exact_duplicate_count)
        self.assertEqual(1, result.similar_duplicate_count)
        self.assertEqual([expected_name], self.image_names())
        self.assertFalse(any(path.exists() for path in (first, exact_copy, near_copy)))

    def test_admit_rejects_undecodable_and_unsupported_staged_files(self) -> None:
        invalid_image = self.staging_dir / 'invalid.jpg'
        invalid_image.write_text('not an image', encoding='utf-8')
        unsupported = self.staging_dir / 'unsupported.gif'
        unsupported.write_text('not an image', encoding='utf-8')

        result = self.workspace.admit((invalid_image, unsupported))

        self.assertEqual(UploadResult(2, 0, 0, 0), result)
        self.assertEqual([], self.image_names())
        self.assertFalse(invalid_image.exists())
        self.assertFalse(unsupported.exists())

    def test_admit_preserves_existing_images_over_similar_batch_candidates(self) -> None:
        existing = self.make_image('existing.jpg')
        candidate = self.stage_image('candidate.jpg', color='gray')
        existing_bytes = existing.read_bytes()

        result = self.workspace.admit((candidate,))

        self.assertEqual(UploadResult(1, 0, 0, 1), result)
        self.assertEqual(existing_bytes, existing.read_bytes())
        self.assertFalse(candidate.exists())

    def test_images_reenumerate_the_workspace_after_admission(self) -> None:
        self.assertEqual((), self.workspace.images())

        self.accept_image('late.PNG')

        image = self.workspace.images()[0]
        self.assertTrue(image.image_path.name.endswith('.png'))
        self.assertEqual(image.image_path.stem, image.sample_id)

    def test_perceptual_hash_uses_the_inclusive_similarity_boundary(self) -> None:
        image = self.stage_image('hash.png')

        self.assertIsInstance(perceptual_hash(image), int)
        self.assertEqual(SIMILARITY_DISTANCE, hamming_distance(0, 0b11))
        self.assertGreater(hamming_distance(0, 0b111), SIMILARITY_DISTANCE)

    def test_negative_flag_and_boxes_are_the_only_annotated_detection_forms(self) -> None:
        self.accept_image('empty.jpg')
        sample_id = self.workspace.images()[0].sample_id

        self.assertEqual(DetectionSummary(1, 0, 0), self.workspace.detection_summary())
        self.write_annotation(sample_id, {'flags': {'xxtrain_detection_negative': True}, 'shapes': []})
        self.assertEqual(DetectionSummary(1, 1, 0), self.workspace.detection_summary())
        self.write_annotation(
            sample_id, {'flags': {'xxtrain_detection_negative': True}, 'shapes': [{'shape_type': 'line'}]}
        )
        self.assertEqual(DetectionSummary(1, 0, 0), self.workspace.detection_summary())
        self.write_annotation(
            sample_id, {'shapes': [{'label': 'Point', 'shape_type': 'rectangle', 'points': [[1, 2], [3, 4]]}]}
        )
        self.assertEqual(DetectionSummary(1, 1, 1), self.workspace.detection_summary())

    def test_detection_cache_keeps_explicit_negative_background_images(self) -> None:
        self.accept_boxed_and_negative_workspace()
        self.make_image('incomplete.jpg')

        report = build_detection_cache(self.workspace, self.root / 'runtime' / 'cache' / 'fingerprint')

        output = self.root / 'runtime' / 'cache' / 'fingerprint' / 'detect'
        self.assertEqual(2, report.train_image_count + report.val_image_count)
        self.assertTrue(output.is_dir())
        labels = sorted((output / 'workspace').glob('*.txt'))
        self.assertEqual(2, len(labels))
        self.assertEqual(1, sum(bool(path.read_text(encoding='utf-8')) for path in labels))
        self.assertIn('0: Point', (output / 'dataset.yaml').read_text(encoding='utf-8'))

    def test_detection_cache_does_not_publish_an_incomplete_workspace(self) -> None:
        destination = self.root / 'runtime' / 'cache' / 'fingerprint'

        with self.assertRaisesRegex(ValueError, 'no output samples'):
            build_detection_cache(self.workspace, destination)

        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_name('.fingerprint.building').exists())

    def test_detection_fingerprint_ignores_non_detection_labelme_content(self) -> None:
        self.accept_image('annotated.jpg')
        sample_id = self.workspace.images()[0].sample_id
        detection = {'label': 'Point', 'shape_type': 'rectangle', 'points': [[1, 2], [3, 4]]}
        self.write_annotation(
            sample_id, {'version': '5.0.0', 'model': {'name': 'first'}, 'shapes': [detection], 'imageData': 'first'}
        )
        original = self.workspace.detection_fingerprint()
        self.write_annotation(
            sample_id,
            {
                'version': '99.0.0',
                'model': {'name': 'other'},
                'shapes': [{'label': 'ignored', 'shape_type': 'line', 'points': [[0, 0], [1, 1]]}, detection],
                'imageData': 'other',
            },
        )

        self.assertEqual(original, self.workspace.detection_fingerprint())

    def test_detection_edit_preserves_other_annotations(self) -> None:
        polygon = {'label': '1', 'shape_type': 'line', 'points': [[1, 2], [3, 4]], 'flags': {'keep': True}}
        source = {'imagePath': '../images/a.jpg', 'custom': 'keep', 'shapes': [polygon]}
        box = DetectionBox(
            Bbox(label='tl', x1=1.25, y1=2, x2=30, y2=40),
            {'description': 'keep', 'label': 'wrong', 'points': [], 'shape_type': 'polygon'},
        )

        result = merge_detection(source, (box,))

        self.assertEqual(result['shapes'][0], polygon)
        self.assertEqual(result['shapes'][1]['label'], 'tl')
        self.assertEqual(result['shapes'][1]['points'][0], [1.25, 2.0])
        self.assertEqual(result['shapes'][1]['description'], 'keep')
        self.assertEqual(result['custom'], 'keep')
        self.assertEqual(source['shapes'], [polygon])

    def test_images_and_save_support_an_empty_annotation(self) -> None:
        image_path = self.make_image(size=(37, 23))
        original_image = image_path.read_bytes()
        workspace = WorkspaceData(self.root)

        images = workspace.images()

        self.assertEqual(1, len(images))
        self.assertEqual('a', images[0].sample_id)
        self.assertEqual(image_path, images[0].image_path)
        self.assertEqual((37, 23), (images[0].width, images[0].height))
        self.assertEqual((), images[0].boxes)

        workspace.save_detection((FrameResult(sample_id='a', boxes=()),))

        self.assertEqual(
            {
                'version': '5.0.0',
                'flags': {},
                'shapes': [],
                'imagePath': '../images/a.jpg',
                'imageData': None,
                'imageHeight': 23,
                'imageWidth': 37,
            },
            json.loads((self.annotations_dir / 'a.json').read_text(encoding='utf-8')),
        )
        self.assertEqual(original_image, image_path.read_bytes())

    def test_existing_rectangle_label_fractional_geometry_and_metadata_round_trip(self) -> None:
        self.make_image()
        source = {
            'version': '5.4.1',
            'shapes': [
                {
                    'label': 'tl',
                    'shape_type': 'rectangle',
                    'points': [[1.25, 2], [30, 40]],
                    'group_id': 7,
                    'flags': {'reviewed': True},
                    'description': 'keep',
                }
            ],
            'imagePath': '../images/a.jpg',
            'imageHeight': 48,
            'imageWidth': 64,
        }
        self.write_annotation('a', source)
        workspace = WorkspaceData(self.root)

        image = workspace.images()[0]

        self.assertEqual('tl', image.boxes[0].geometry.label)
        self.assertEqual((1.25, 2.0, 30.0, 40.0), image.boxes[0].geometry.bbox)
        self.assertEqual({'group_id': 7, 'flags': {'reviewed': True}, 'description': 'keep'}, image.boxes[0].extra)

        workspace.save_detection((FrameResult(sample_id='a', boxes=image.boxes),))

        saved_shape = json.loads((self.annotations_dir / 'a.json').read_text(encoding='utf-8'))['shapes'][0]
        self.assertEqual('tl', saved_shape['label'])
        self.assertEqual([[1.25, 2.0], [30.0, 40.0]], saved_shape['points'])
        self.assertEqual('rectangle', saved_shape['shape_type'])
        self.assertEqual(7, saved_shape['group_id'])
        self.assertEqual({'reviewed': True}, saved_shape['flags'])
        self.assertEqual('keep', saved_shape['description'])

    def test_deleting_all_rectangles_preserves_lines_polygons_and_root_fields(self) -> None:
        line = {'label': 'line', 'shape_type': 'line', 'points': [[1, 2], [3, 4]]}
        polygon = {'label': 'mask', 'shape_type': 'polygon', 'points': [[1, 2], [3, 4], [5, 2]]}
        source = {'custom': {'keep': True}, 'shapes': [line, {'label': 'tl', 'shape_type': 'rectangle'}, polygon]}

        result = merge_detection(source, ())

        self.assertEqual([line, polygon], result['shapes'])
        self.assertEqual({'keep': True}, result['custom'])

    def test_images_reject_invalid_rectangle_geometry(self) -> None:
        self.make_image()
        self.write_annotation(
            'a', {'shapes': [{'label': 'Point', 'shape_type': 'rectangle', 'points': [[1, 2], [1, 4]]}]}
        )

        with self.assertRaisesRegex(ValueError, 'Bbox'):
            WorkspaceData(self.root).images()

    def test_save_requires_each_image_exactly_once(self) -> None:
        self.make_image('a.jpg')
        self.make_image('b.png')
        workspace = WorkspaceData(self.root)
        valid_box = DetectionBox(Bbox(label='Point', x1=1, y1=2, x2=3, y2=4))

        invalid_results = (
            (FrameResult(sample_id='a', boxes=(valid_box,)),),
            (FrameResult(sample_id='a', boxes=()), FrameResult(sample_id='a', boxes=())),
            (FrameResult(sample_id='a', boxes=()), FrameResult(sample_id='unknown', boxes=())),
        )
        for results in invalid_results:
            with self.subTest(results=results), self.assertRaises(ValueError):
                workspace.save_detection(results)

        self.assertEqual([], list(self.annotations_dir.iterdir()))

    def test_workspace_rejects_duplicate_stems_and_missing_directories(self) -> None:
        self.make_image('same.jpg')
        self.make_image('same.png')

        with self.assertRaisesRegex(ValueError, 'Duplicate image stem'):
            WorkspaceData(self.root).images()
        with self.assertRaises(PlatformAccessError):
            WorkspaceData(self.root / 'missing-images')
        incomplete = self.root / 'incomplete'
        (incomplete / 'images').mkdir(parents=True)
        with self.assertRaises(PlatformAccessError):
            WorkspaceData(incomplete)

    def test_all_documents_are_encoded_before_any_file_is_replaced(self) -> None:
        self.make_image('a.jpg')
        self.make_image('b.jpg')
        a_path = self.write_annotation('a', {'marker': 'old-a', 'shapes': []})
        b_path = self.write_annotation('b', {'marker': 'old-b', 'shapes': []})
        old_a = a_path.read_bytes()
        old_b = b_path.read_bytes()
        workspace = WorkspaceData(self.root)
        invalid_box = DetectionBox(Bbox(label='Point', x1=1, y1=2, x2=3, y2=4), {'confidence': float('nan')})

        with self.assertRaises(ValueError):
            workspace.save_detection(
                (FrameResult(sample_id='a', boxes=()), FrameResult(sample_id='b', boxes=(invalid_box,)))
            )

        self.assertEqual(old_a, a_path.read_bytes())
        self.assertEqual(old_b, b_path.read_bytes())

    def test_replace_failure_does_not_truncate_old_json_and_cleans_temporary_file(self) -> None:
        self.make_image()
        annotation_path = self.write_annotation('a', {'marker': 'old', 'shapes': []})
        old_content = annotation_path.read_bytes()
        workspace = WorkspaceData(self.root)

        with patch('xxtrain.workspace_data.store.os.replace', side_effect=OSError('replace failed')):
            with self.assertRaisesRegex(OSError, 'replace failed'):
                workspace.save_detection((FrameResult(sample_id='a', boxes=()),))

        self.assertEqual(old_content, annotation_path.read_bytes())
        self.assertEqual([annotation_path], list(self.annotations_dir.iterdir()))

        workspace.save_detection((FrameResult(sample_id='a', boxes=()),))

        self.assertEqual({'marker': 'old', 'shapes': []}, json.loads(annotation_path.read_text(encoding='utf-8')))

    def test_later_save_failure_removes_new_annotations_and_preserves_existing_bytes(self) -> None:
        self.make_image('a.jpg')
        self.make_image('b.jpg')
        annotation = self.write_annotation('b', {'marker': 'old', 'shapes': []})
        original = annotation.read_bytes()
        fingerprint = self.workspace.detection_fingerprint()
        real_replace = os.replace

        def fail_second(source, destination):
            if Path(destination) == annotation:
                raise OSError('disk full')
            return real_replace(source, destination)

        with patch('xxtrain.workspace_data.store.os.replace', side_effect=fail_second):
            with self.assertRaises(OSError):
                self.workspace.save_detection((FrameResult('a', ()), FrameResult('b', ())))
        self.assertFalse((self.annotations_dir / 'a.json').exists())
        self.assertEqual(original, annotation.read_bytes())
        self.assertEqual(fingerprint, self.workspace.detection_fingerprint())

    def test_result_fingerprint_predicts_saved_detection_without_writing(self) -> None:
        self.make_image()
        annotation = self.write_annotation('a', {'flags': {'xxtrain_detection_negative': True}, 'shapes': []})
        for boxes in ((DetectionBox(Bbox(label='tl', x1=1, y1=2, x2=20, y2=30)),), ()):
            with self.subTest(boxes=boxes):
                original = annotation.read_bytes()
                results = (FrameResult('a', boxes),)
                fingerprint = self.workspace.detection_fingerprint(results)
                self.assertEqual(original, annotation.read_bytes())
                self.workspace.save_detection(results)
                self.assertEqual(fingerprint, self.workspace.detection_fingerprint())

    def test_result_fingerprint_requires_each_workspace_image_exactly_once(self) -> None:
        self.make_image()
        for results in ((), (FrameResult('unknown', ()),), (FrameResult('a', ()), FrameResult('a', ()))):
            with self.subTest(results=results), self.assertRaises(ValueError):
                self.workspace.detection_fingerprint(results)


if __name__ == '__main__':
    unittest.main()
