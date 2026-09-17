import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml

from xxtrain.business_tasks.definition import DeliveryDefinition
from xxtrain.training import TrainingResult, build_delivery


class TrainingDeliveryTest(unittest.TestCase):
    def test_single_class_delivery_exposes_only_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.onnx'
            source.write_bytes(b'model')
            result = TrainingResult(source, {0: 'Point'}, {})
            output = build_delivery(result, DeliveryDefinition(False, False), root, root / 'output')
            self.assertEqual(output.suffix, '.onnx')
            self.assertEqual(output.read_bytes(), b'model')
            self.assertFalse((root / 'output' / 'labels.txt').exists())

    def test_classification_zip_maps_reindexed_directories_to_business_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'classify'
            samples = {'03-cc': b'cc-image', '00-tl': b'tl-image'}
            for directory_name, content in samples.items():
                path = dataset / 'train' / directory_name / f'{directory_name}.png'
                path.parent.mkdir(parents=True)
                path.write_bytes(content)
            (dataset / 'dataset.yaml').write_text(
                yaml.safe_dump({'names': {0: 'tl', 1: 'tc', 2: 'cl', 3: 'cc'}}), encoding='utf-8'
            )
            source = root / 'source.onnx'
            source.write_bytes(b'model')
            result = TrainingResult(source, {1: '03-cc', 0: '00-tl'}, {})

            output = build_delivery(result, DeliveryDefinition(True, True), dataset, root / 'output')

            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    ['labels.txt', 'model.onnx', 'references/0_tl.png', 'references/1_cc.png'],
                    sorted(archive.namelist()),
                )
                self.assertEqual('tl\ncc\n', archive.read('labels.txt').decode())
                self.assertEqual(b'tl-image', archive.read('references/0_tl.png'))
                self.assertEqual(b'cc-image', archive.read('references/1_cc.png'))

    def test_classification_delivery_rejects_missing_reference_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'classify'
            (dataset / 'train' / '00-tl').mkdir(parents=True)
            (dataset / 'train' / '00-tl' / 'sample.png').write_bytes(b'image')
            (dataset / 'dataset.yaml').write_text(
                yaml.safe_dump({'names': {0: 'tl', 1: 'tc', 2: 'cl', 3: 'cc'}}), encoding='utf-8'
            )
            source = root / 'source.onnx'
            source.write_bytes(b'model')
            result = TrainingResult(source, {0: '00-tl', 1: '03-cc'}, {})

            with self.assertRaisesRegex(ValueError, '03-cc'):
                build_delivery(result, DeliveryDefinition(True, True), dataset, root / 'output')

    def test_classification_delivery_rejects_non_contiguous_model_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.onnx'
            source.write_bytes(b'model')
            dataset = root / 'classify'
            dataset.mkdir()
            (dataset / 'dataset.yaml').write_text(
                yaml.safe_dump({'names': {0: 'tl', 1: 'tc', 2: 'cl', 3: 'cc'}}), encoding='utf-8'
            )
            for class_names in ({1: '00-tl'}, {0: '00-tl', 2: '03-cc'}):
                with self.subTest(class_names=class_names):
                    with self.assertRaisesRegex(ValueError, 'contiguous zero-based'):
                        build_delivery(
                            TrainingResult(source, class_names, {}),
                            DeliveryDefinition(True, True),
                            dataset,
                            root / 'output',
                        )


if __name__ == '__main__':
    unittest.main()
