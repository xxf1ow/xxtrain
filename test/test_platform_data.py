import json
import shutil
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import yaml
from PIL import Image

from xxtrain.business_tasks.point import point_task_definition
from xxtrain.data import Bbox
from xxtrain.platform.cache import build_detection_cache
from xxtrain.platform.contracts import (
    AnnotationRecord,
    CvatBinding,
    DetectionBox,
    DetectionSummary,
    FrameResult,
    JobRef,
    PlatformAccessError,
    PreparedJob,
    UploadResult,
)
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.dedup import SIMILARITY_DISTANCE, hamming_distance, perceptual_hash
from xxtrain.workspace_data.repository import AnnotationRepository


class PlatformDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images_dir = self.root / 'images'
        self.staging_dir = self.root / 'staging'
        self.images_dir.mkdir()
        self.staging_dir.mkdir()
        self.workspace = WorkspaceData(self.root)

    def repository(self) -> AnnotationRepository:
        return AnnotationRepository(self.root / 'annotations.db', point_task_definition())

    def stage_image(self, name: str, *, seed: int = 0, solid: str | None = None) -> Path:
        path = self.staging_dir / name
        with Image.new('RGB', (64, 48), solid or 'white') as image:
            if solid is None:
                for x in range(64):
                    for y in range(48):
                        value = (x * (17 + seed) + y * (29 + seed * 2) + seed * 41) % 256
                        image.putpixel((x, y), (value, (value + seed * 23) % 256, 255 - value))
            image.save(path)
        return path

    def accept_image(self, name: str = 'a.png', *, seed: int = 0):
        result = self.workspace.admit((self.stage_image(name, seed=seed),))
        self.assertEqual(1, result.accepted_count)
        return self.workspace.images()[-1]

    @staticmethod
    def box_record(image_id: str, annotation_id: UUID | None = None, *, x1: float = 1, label: str = 'tl'):
        return AnnotationRecord(
            annotation_id or uuid4(), image_id, 'detect', None, 'rectangle', label, [[x1, 2], [30, 40]]
        )

    @staticmethod
    def classification_record(image_id: str, parent_id: UUID, *, label: str = 'tl'):
        return AnnotationRecord(uuid4(), image_id, 'classify', parent_id, 'classification', label, None)

    def test_constructor_initializes_database_without_a_labelme_directory(self) -> None:
        self.assertTrue((self.root / 'annotations.db').is_file())
        self.assertFalse((self.root / 'annotations').exists())

    def test_admit_renames_by_sha_and_deduplicates_in_deterministic_order(self) -> None:
        first = self.stage_image('first.jpg', solid='white')
        exact_copy = self.staging_dir / 'copy.jpg'
        exact_copy.write_bytes(first.read_bytes())
        near_copy = self.stage_image('near.jpg', solid='gray')
        expected_name = f'{min(sha256(path.read_bytes()).hexdigest() for path in (first, near_copy))}.jpg'

        result = self.workspace.admit((near_copy, exact_copy, first))

        self.assertEqual(UploadResult(3, 1, 1, 1), result)
        self.assertEqual([expected_name], sorted(path.name for path in self.images_dir.iterdir()))
        self.assertFalse(any(path.exists() for path in (first, exact_copy, near_copy)))

    def test_admit_rejects_undecodable_and_unsupported_staged_files(self) -> None:
        invalid_image = self.staging_dir / 'invalid.jpg'
        invalid_image.write_text('not an image', encoding='utf-8')
        unsupported = self.staging_dir / 'unsupported.gif'
        unsupported.write_text('not an image', encoding='utf-8')

        self.assertEqual(UploadResult(2, 0, 0, 0), self.workspace.admit((invalid_image, unsupported)))
        self.assertEqual((), self.workspace.images())
        self.assertFalse(invalid_image.exists())
        self.assertFalse(unsupported.exists())

    def test_admit_uses_registered_hashes_and_leaves_unregistered_files_out_of_queries(self) -> None:
        accepted = self.accept_image('existing.png', seed=3)
        duplicate = self.staging_dir / 'duplicate.png'
        shutil.copy2(accepted.image_path, duplicate)
        Image.new('RGB', (12, 12)).save(self.images_dir / 'orphan.png')

        result = self.workspace.admit((duplicate,))

        self.assertEqual(UploadResult(1, 0, 1, 0), result)
        self.assertEqual((accepted,), self.workspace.images())

    def test_repeated_summary_and_fingerprint_do_not_rescan_images_or_legacy_json(self) -> None:
        image = self.accept_image('a.png')
        self.repository().save_annotations((self.box_record(image.sample_id),))
        Image.new('RGB', (12, 12)).save(self.images_dir / 'orphan.png')
        legacy = self.root / 'annotations'
        legacy.mkdir()
        (legacy / 'a.json').write_text('{not valid json', encoding='utf-8')

        with (
            patch('xxtrain.workspace_data.store.Image.open', side_effect=AssertionError('unexpected decode')),
            patch('xxtrain.workspace_data.store.image_sha256', side_effect=AssertionError('unexpected hash')),
            patch('xxtrain.workspace_data.store.perceptual_hash', side_effect=AssertionError('unexpected pHash')),
            patch.object(Path, 'read_text', side_effect=AssertionError('unexpected JSON read')),
        ):
            first_summary = self.workspace.detection_summary()
            first_fingerprint = self.workspace.detection_fingerprint()
            self.assertEqual(first_summary, self.workspace.detection_summary())
            self.assertEqual(first_fingerprint, self.workspace.detection_fingerprint())

        self.assertEqual(DetectionSummary(1, 1, 1), first_summary)

    def test_images_use_database_dimensions_and_preserve_annotation_uuid(self) -> None:
        image = self.accept_image()
        record = self.box_record(image.sample_id)
        self.repository().save_annotations((record,))

        with patch('xxtrain.workspace_data.store.Image.open', side_effect=AssertionError('unexpected decode')):
            restored = self.workspace.images()[0]

        self.assertEqual((64, 48), (restored.width, restored.height))
        self.assertEqual(record.id, restored.boxes[0].geometry.id)
        self.assertEqual((1.0, 2.0, 30.0, 40.0), restored.boxes[0].geometry.bbox)

    def test_summary_counts_complete_images_and_not_box_rows(self) -> None:
        boxed = self.accept_image('boxed.png', seed=1)
        negative = self.accept_image('negative.png', seed=7)
        first = self.box_record(boxed.sample_id)
        second = self.box_record(boxed.sample_id, x1=3, label='tc')
        no_object = AnnotationRecord(uuid4(), negative.sample_id, 'detect', None, 'negative', None, None)
        self.repository().save_annotations((first, second, no_object))

        self.assertEqual(DetectionSummary(2, 2, 1), self.workspace.detection_summary())

    def test_fingerprint_uses_business_content_not_uuid_mapping_or_display_extra(self) -> None:
        image = self.accept_image()
        original = self.box_record(image.sample_id)
        repository = self.repository()
        repository.save_annotations((original,))
        before = self.workspace.detection_fingerprint()
        ref = JobRef(7, 8, (image.sample_id,))
        repository.bind_job(PreparedJob(ref, (CvatBinding(image.sample_id, 'shape', 91, original.id),)))
        self.assertEqual(before, self.workspace.detection_fingerprint())

        replacement = self.box_record(image.sample_id)
        repository.save_annotations((replacement,), delete_ids=frozenset({original.id}))
        self.assertEqual(before, self.workspace.detection_fingerprint())

        changed = self.box_record(image.sample_id, replacement.id, x1=4)
        repository.save_annotations((changed,))
        self.assertNotEqual(before, self.workspace.detection_fingerprint())

    def test_detection_export_is_independent_of_database_and_cvat_ids(self) -> None:
        image = self.accept_image()
        record = self.box_record(image.sample_id)
        repository = self.repository()
        repository.save_annotations((record,))
        ref = JobRef(7, 8, (image.sample_id,))
        repository.bind_job(PreparedJob(ref, (CvatBinding(image.sample_id, 'shape', 91, record.id),)))

        before = self.workspace.detection_fingerprint()
        root = self.root / 'conversion'
        self.workspace.materialize_detection_source(root)
        document = json.loads(next((root / 'src/workspace/anns_seg').glob('*.json')).read_text())
        self.assertNotIn('annotation_id', document)
        self.assertTrue(all(shape['shape_type'] == 'rectangle' for shape in document['shapes']))
        self.assertEqual(before, self.workspace.detection_fingerprint())

    def test_every_point_box_label_projects_to_detection_class_zero(self) -> None:
        image = self.accept_image()
        labels = ('Point', 'tl', 'tc', 'cl', 'cc')
        self.repository().save_annotations(
            tuple(
                AnnotationRecord(
                    uuid4(), image.sample_id, 'detect', None, 'rectangle', label, [[index + 1, 2], [30, 40]]
                )
                for index, label in enumerate(labels)
            )
        )

        build_detection_cache(self.workspace, self.root / 'runtime' / 'cache' / 'labels')

        output = self.root / 'runtime' / 'cache' / 'labels' / 'detect' / 'workspace'
        lines = [line for path in output.glob('*.txt') for line in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(5, len(lines))
        self.assertEqual(['0'] * 5, [line.split()[0] for line in lines])

    def test_detection_cache_projects_boxes_and_explicit_negative_from_database(self) -> None:
        boxed = self.accept_image('boxed.png', seed=2)
        negative = self.accept_image('negative.png', seed=9)
        self.repository().save_annotations(
            (
                self.box_record(boxed.sample_id, label='tc'),
                AnnotationRecord(uuid4(), negative.sample_id, 'detect', None, 'negative', None, None),
            )
        )

        destination = self.root / 'runtime' / 'cache' / 'fingerprint'
        report = build_detection_cache(self.workspace, destination)

        output = destination / 'detect'
        self.assertEqual(2, report.train_image_count + report.val_image_count)
        labels = sorted((output / 'workspace').glob('*.txt'))
        self.assertEqual(2, len(labels))
        self.assertEqual(1, sum(bool(path.read_text(encoding='utf-8')) for path in labels))
        self.assertEqual('', (output / 'workspace' / f'{negative.sample_id}.txt').read_text(encoding='utf-8'))
        source_documents = [
            json.loads(path.read_text(encoding='utf-8')) for path in (output.parent / 'src').rglob('*.json')
        ]
        self.assertEqual({'tc'}, {shape['label'] for doc in source_documents for shape in doc['shapes']})

        dataset = yaml.safe_load((output / 'dataset.yaml').read_text(encoding='utf-8'))
        self.assertEqual(str(destination.absolute()), dataset['path'])
        self.assertEqual({'detect/train.txt', 'detect/val.txt'}, {dataset['train'], dataset['val']})
        split_paths = [destination / dataset[name] for name in ('train', 'val')]
        self.assertTrue(all(path.is_file() for path in split_paths))
        published_items = [Path(item) for item in (*report.train_items, *report.val_items)]
        self.assertEqual(2, len(published_items))
        self.assertTrue(all(item.is_file() and item.resolve().is_relative_to(destination) for item in published_items))
        listed_items = [Path(line) for path in split_paths for line in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(set(published_items), set(listed_items))
        published_images = [path for path in (output / 'workspace').iterdir() if path.suffix != '.txt']
        self.assertEqual(2, len(published_images))
        self.assertTrue(all(path.exists() and path.resolve().is_file() for path in published_images))
        for path in published_images:
            if path.is_symlink():
                self.assertFalse(path.readlink().is_absolute())
                self.assertTrue((path.parent / path.readlink()).resolve().is_file())

    def test_prepare_sync_preserves_known_geometry_ids_and_unrelated_downstream_labels(self) -> None:
        image = self.accept_image()
        first = self.box_record(image.sample_id, x1=1)
        second = self.box_record(image.sample_id, x1=10, label='tc')
        first_label = self.classification_record(image.sample_id, first.id)
        second_label = self.classification_record(image.sample_id, second.id, label='tc')
        repository = self.repository()
        repository.save_annotations((first, second, first_label, second_label))
        ref = JobRef(7, 8, (image.sample_id,))
        self.workspace.bind_job(
            PreparedJob(
                ref,
                (
                    CvatBinding(image.sample_id, 'shape', 101, first.id),
                    CvatBinding(image.sample_id, 'shape', 102, second.id),
                ),
            )
        )
        results = (
            FrameResult(
                image.sample_id,
                (
                    DetectionBox(Bbox(label='tl', x1=4, y1=2, x2=30, y2=40), cvat_id=101),
                    DetectionBox(Bbox(label='tc', x1=10, y1=2, x2=30, y2=40), cvat_id=102),
                ),
            ),
        )

        sync = self.workspace.prepare_detection_sync(ref, results)

        self.assertEqual(frozenset({'classify', 'segment'}), sync.changes.invalidated_steps)
        self.assertIn(first_label.id, sync.changes.delete_ids)
        self.assertNotIn(second_label.id, sync.changes.delete_ids)
        self.workspace.commit_detection_sync(ref, sync)
        records = {record.id: record for record in repository.annotations()}
        self.assertEqual([[4.0, 2.0], [30.0, 40.0]], records[first.id].geometry)
        self.assertIn(second.id, records)
        self.assertIn(second_label.id, records)
        self.assertEqual(sync.fingerprint, self.workspace.detection_fingerprint())

    def test_prepare_sync_keeps_an_unchanged_empty_frame_incomplete(self) -> None:
        image = self.accept_image()
        ref = JobRef(7, 8, (image.sample_id,))

        sync = self.workspace.prepare_detection_sync(ref, (FrameResult(image.sample_id, ()),))
        self.workspace.commit_detection_sync(ref, sync)

        self.assertEqual((), self.repository().annotations(step_key='detect'))
        self.assertEqual(DetectionSummary(1, 0, 0), self.workspace.detection_summary())

    def test_prepare_sync_leaves_a_frame_incomplete_after_its_last_box_is_removed(self) -> None:
        image = self.accept_image()
        record = self.box_record(image.sample_id)
        self.repository().save_annotations((record,))
        ref = JobRef(7, 8, (image.sample_id,))
        self.workspace.bind_job(PreparedJob(ref, (CvatBinding(image.sample_id, 'shape', 101, record.id),)))

        sync = self.workspace.prepare_detection_sync(ref, (FrameResult(image.sample_id, ()),))
        self.workspace.commit_detection_sync(ref, sync)

        self.assertEqual((), self.repository().annotations(step_key='detect'))
        self.assertEqual(DetectionSummary(1, 0, 0), self.workspace.detection_summary())

    def test_prepare_sync_removes_an_existing_negative_when_confirmation_is_absent(self) -> None:
        image = self.accept_image()
        negative = AnnotationRecord(uuid4(), image.sample_id, 'detect', None, 'negative', None, None)
        self.repository().save_annotations((negative,))
        ref = JobRef(7, 8, (image.sample_id,))

        sync = self.workspace.prepare_detection_sync(ref, (FrameResult(image.sample_id, ()),))
        self.workspace.commit_detection_sync(ref, sync)

        self.assertEqual((), self.repository().annotations(step_key='detect'))
        self.assertEqual(DetectionSummary(1, 0, 0), self.workspace.detection_summary())

    def test_new_server_identity_gets_a_new_uuid_even_with_a_copied_token(self) -> None:
        image = self.accept_image()
        previous = self.box_record(image.sample_id)
        self.repository().save_annotations((previous,))
        ref = JobRef(7, 8, (image.sample_id,))
        self.workspace.bind_job(PreparedJob(ref, (CvatBinding(image.sample_id, 'shape', 101, previous.id),)))
        copied = DetectionBox(
            Bbox(label='tl', x1=1, y1=2, x2=30, y2=40), extra={'xxtrain_annotation_id': str(previous.id)}, cvat_id=202
        )

        sync = self.workspace.prepare_detection_sync(ref, (FrameResult(image.sample_id, (copied,)),))

        self.assertNotEqual(previous.id, sync.bindings[0].annotation_id)
        self.assertIn(previous.id, sync.changes.delete_ids)

    def test_prepare_sync_rejects_missing_duplicate_and_wrong_frame_server_ids(self) -> None:
        first_image = self.accept_image('first.png', seed=1)
        second_image = self.accept_image('second.png', seed=11)
        record = self.box_record(first_image.sample_id)
        self.repository().save_annotations((record,))
        ref = JobRef(7, 8, (first_image.sample_id, second_image.sample_id))
        self.workspace.bind_job(PreparedJob(ref, (CvatBinding(first_image.sample_id, 'shape', 101, record.id),)))
        missing = DetectionBox(Bbox(label='tl', x1=1, y1=2, x2=3, y2=4))
        duplicate = DetectionBox(Bbox(label='tl', x1=2, y1=2, x2=3, y2=4), cvat_id=202)

        with self.assertRaisesRegex(ValueError, 'ID'):
            self.workspace.prepare_detection_sync(
                ref, (FrameResult(first_image.sample_id, (missing,)), FrameResult(second_image.sample_id, ()))
            )
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.workspace.prepare_detection_sync(
                ref,
                (FrameResult(first_image.sample_id, (duplicate,)), FrameResult(second_image.sample_id, (duplicate,))),
            )
        with self.assertRaisesRegex(ValueError, 'frame'):
            self.workspace.prepare_detection_sync(
                ref,
                (
                    FrameResult(first_image.sample_id, ()),
                    FrameResult(
                        second_image.sample_id, (DetectionBox(Bbox(label='tl', x1=1, y1=2, x2=3, y2=4), cvat_id=101),)
                    ),
                ),
            )

    def test_perceptual_hash_uses_the_inclusive_similarity_boundary(self) -> None:
        image = self.stage_image('hash.png')

        self.assertIsInstance(perceptual_hash(image), int)
        self.assertEqual(SIMILARITY_DISTANCE, hamming_distance(0, 0b11))
        self.assertGreater(hamming_distance(0, 0b111), SIMILARITY_DISTANCE)

    def test_workspace_requires_the_images_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PlatformAccessError):
                WorkspaceData(Path(directory))


if __name__ == '__main__':
    unittest.main()
