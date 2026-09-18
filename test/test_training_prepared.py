import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
from PIL import Image
from ultralytics.data.utils import check_image
from ultralytics.nn.tasks import yaml_model_load

from xxtrain.task import TaskType
from xxtrain.training import TrainingProgress, TrainingSettings, train_prepared


class PreparedTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='xxtrain-prepared-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _write_dataset(self, task_type: TaskType) -> Path:
        dataset = self.root / f'{task_type.value}-source'
        dataset.mkdir()
        if task_type is TaskType.CLASSIFY:
            image = dataset / 'classify' / 'train' / '00-tl' / 'sample.png'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'image')
            (dataset / 'classify' / 'val' / '00-tl').mkdir(parents=True)
            (dataset / 'classify' / 'val' / '00-tl' / 'sample.png').write_bytes(b'image')
            data = {'path': str(dataset), 'train': 'classify/train.txt', 'val': 'classify/val.txt', 'names': {0: 'tl'}}
        else:
            image = dataset / task_type.value / 'sample.png'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'image')
            image.with_suffix('.txt').write_text('' if task_type is TaskType.DETECT else '0 0.1 0.1 0.2 0.2 0.3 0.3')
            for split in ('train', 'val'):
                (dataset / task_type.value / f'{split}.txt').write_text(str(image), encoding='utf-8')
            data = {
                'path': str(dataset),
                'train': f'{task_type.value}/train.txt',
                'val': f'{task_type.value}/val.txt',
                'names': {0: 'Point'},
            }
        (dataset / task_type.value / 'dataset.yaml').write_text(yaml.safe_dump(data), encoding='utf-8')
        return dataset / task_type.value

    @staticmethod
    def _snapshot(root: Path) -> dict[str, bytes]:
        return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}

    def test_three_targets_train_from_read_only_prepared_data_in_independent_run_directories(self) -> None:
        for task_type in (TaskType.DETECT, TaskType.CLASSIFY, TaskType.SEGMENT):
            with self.subTest(task_type=task_type):
                dataset = self._write_dataset(task_type)
                before = self._snapshot(dataset)
                run_dir = self.root / f'{task_type.value}-run'
                model = MagicMock()
                trainer = MagicMock(epoch=0, epochs=2, best='')
                model.trainer = trainer
                model.names = {0: '00-tl'} if task_type is TaskType.CLASSIFY else {0: 'Point'}
                model.add_callback.side_effect = lambda _event, callback: callback(trainer)
                exported = run_dir / 'exported.onnx'
                model.export.side_effect = lambda **_kwargs: (exported.write_bytes(b'onnx'), exported)[1]
                validation = MagicMock(results_dict={'metrics/score': 0.75})
                model.val.return_value = validation
                progress = []

                with (
                    patch('xxtrain.training.prepared.YOLO', return_value=model) as yolo,
                    patch(
                        'xxtrain.training.prepared.prepare_pretrained_weights', return_value=self.root / 'default.pt'
                    ) as weights,
                    patch('xxtrain.pipeline.convert_dataset') as convert,
                ):
                    result = train_prepared(
                        TrainingSettings(task_type, train_args={'epochs': 2}),
                        dataset,
                        run_dir,
                        on_progress=progress.append,
                    )

                convert.assert_not_called()
                suffix = '-cls' if task_type is TaskType.CLASSIFY else '-seg' if task_type is TaskType.SEGMENT else ''
                weights.assert_called_once_with(f'yolov8n{suffix}')
                yolo.assert_called_once_with(run_dir / f'yolov8n{suffix}.yaml')
                model.load.assert_called_once_with(self.root / 'default.pt')
                train_args = model.train.call_args.kwargs
                self.assertEqual(run_dir, train_args['project'])
                self.assertEqual('training', train_args['name'])
                self.assertTrue(Path(train_args['data']).is_relative_to(run_dir))
                self.assertEqual(before, self._snapshot(dataset))
                self.assertEqual([TrainingProgress(1, 2)], progress)
                self.assertEqual(run_dir / 'model.onnx', result.onnx_path)
                self.assertEqual(b'onnx', result.onnx_path.read_bytes())
                self.assertEqual({'metrics/score': 0.75}, dict(result.metrics))

    def test_configured_model_basename_selects_the_real_ultralytics_scale(self) -> None:
        dataset = self._write_dataset(TaskType.DETECT)
        for scale in ('n', 's'):
            with self.subTest(scale=scale):
                run_dir = self.root / f'{scale}-run'
                model = MagicMock(names={0: 'Point'})
                model.trainer = MagicMock(best='')
                model.val.return_value = MagicMock(results_dict={})
                exported = run_dir / 'exported.onnx'
                model.export.side_effect = lambda **_kwargs: (exported.write_bytes(b'onnx'), exported)[1]
                loaded_scales = []

                def load_model(path):
                    loaded_scales.append(yaml_model_load(path)['scale'])
                    return model

                with (
                    patch('xxtrain.training.prepared.YOLO', side_effect=load_model),
                    patch(
                        'xxtrain.training.prepared.prepare_pretrained_weights', return_value=self.root / 'default.pt'
                    ),
                ):
                    train_prepared(TrainingSettings(TaskType.DETECT, model_scale=scale), dataset, run_dir)

                self.assertEqual([scale], loaded_scales)

    def test_best_checkpoint_supplies_export_names_and_metrics(self) -> None:
        dataset = self._write_dataset(TaskType.DETECT)
        run_dir = self.root / 'best-run'
        best_path = run_dir / 'best.pt'
        best_path.parent.mkdir()
        best_path.write_bytes(b'best')
        training_model = MagicMock()
        training_model.trainer = MagicMock(best=best_path)
        best_model = MagicMock(names={0: 'Point'})
        best_model.val.return_value = MagicMock(results_dict={'metrics/mAP50-95(B)': 0.81})
        exported = run_dir / 'best.onnx'
        best_model.export.side_effect = lambda **_kwargs: (exported.write_bytes(b'onnx'), exported)[1]

        with (
            patch('xxtrain.training.prepared.YOLO', side_effect=[training_model, best_model]),
            patch('xxtrain.training.prepared.prepare_pretrained_weights', return_value=self.root / 'default.pt'),
        ):
            result = train_prepared(TrainingSettings(TaskType.DETECT), dataset, run_dir)

        best_model.val.assert_called_once()
        best_model.export.assert_called_once_with(format='onnx', simplify=True)
        self.assertEqual(0.81, result.metrics['metrics/mAP50-95(B)'])

    def test_rejects_overlapping_source_and_run_directories(self) -> None:
        dataset = self._write_dataset(TaskType.DETECT)
        with self.assertRaisesRegex(ValueError, 'must not overlap'):
            train_prepared(TrainingSettings(TaskType.DETECT), dataset, dataset / 'run')

    def test_export_failure_propagates_without_a_training_result(self) -> None:
        dataset = self._write_dataset(TaskType.DETECT)
        model = MagicMock(names={0: 'Point'})
        model.trainer = MagicMock(best='')
        model.val.return_value = MagicMock(results_dict={})
        model.export.side_effect = RuntimeError('export failed')
        with (
            patch('xxtrain.training.prepared.YOLO', return_value=model),
            patch('xxtrain.training.prepared.prepare_pretrained_weights', return_value=self.root / 'default.pt'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'export failed'):
                train_prepared(TrainingSettings(TaskType.DETECT), dataset, self.root / 'failed-run')

    def test_real_ultralytics_jpeg_repair_cannot_modify_publication_image(self) -> None:
        dataset = self.root / 'publication' / 'detect'
        image = dataset / 'sample.jpg'
        image.parent.mkdir(parents=True)
        with Image.new('RGB', (16, 16), 'red') as value:
            value.save(image, 'JPEG')
        image.write_bytes(image.read_bytes() + b'junk')
        image.with_suffix('.txt').write_text('', encoding='utf-8')
        for split in ('train', 'val'):
            (dataset / f'{split}.txt').write_text(str(image), encoding='utf-8')
        (dataset / 'dataset.yaml').write_text(
            yaml.safe_dump(
                {
                    'path': str(dataset.parent),
                    'train': 'detect/train.txt',
                    'val': 'detect/val.txt',
                    'names': {0: 'Point'},
                }
            ),
            encoding='utf-8',
        )
        source_before = image.read_bytes()
        model = MagicMock(names={0: 'Point'})
        model.trainer = MagicMock(best='')
        model.val.return_value = MagicMock(results_dict={})
        exported = self.root / 'temporary.onnx'
        model.export.side_effect = lambda **_kwargs: (exported.write_bytes(b'onnx'), exported)[1]

        def scan_local_image(**arguments: object) -> None:
            metadata = yaml.safe_load(Path(arguments['data']).read_text(encoding='utf-8'))
            local_list = Path(arguments['data']).parent / metadata['train']
            local_image = Path(local_list.read_text(encoding='utf-8'))
            message, _shape = check_image(str(local_image))
            self.assertIn('restored and saved', message)

        model.train.side_effect = scan_local_image
        with (
            patch('xxtrain.training.prepared.YOLO', return_value=model),
            patch('xxtrain.training.prepared.prepare_pretrained_weights', return_value=self.root / 'default.pt'),
        ):
            train_prepared(TrainingSettings(TaskType.DETECT), dataset, self.root / 'scanner-run')

        self.assertEqual(source_before, image.read_bytes())


if __name__ == '__main__':
    unittest.main()
