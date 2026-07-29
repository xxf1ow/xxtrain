import os
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics.engine.results import Probs
from ultralytics.models import YOLO

from xxtrain.pipeline.sinks import _letterbox_classification_image

from .scenario import load_scenario


def _review_standard(best_model: YOLO, directory: str | Path) -> None:
    print('🚀 Running inference on validation set using best model ...')
    results = best_model.predict(source=directory, verbose=False, save=True, stream=True)
    if isinstance(results, map) or hasattr(results, '__iter__'):
        for _ in results:
            pass
    assert best_model.predictor and hasattr(best_model.predictor, 'save_dir')
    print(f'✅ Inference completed. Results are saved in {best_model.predictor.save_dir}')


def _review_classification(best_model: YOLO, directory: str | Path, unlabeled: bool = False) -> None:
    assert best_model.task == 'classify'
    class_names = best_model.names
    name_to_idx = {value: key for key, value in class_names.items()}
    if not class_names:
        print('⚠️ 标签列表是空的 ...')
        return

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    abs_root_path = os.path.abspath(directory)
    image_paths = []
    for root, _directories, files in os.walk(abs_root_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))
    if not image_paths:
        print(f'⚠️ 在目录 {directory} 下未找到任何图片文件。')
        return

    print(f'🚀 Running inference using best model, image size: {len(image_paths)} ...')
    total_count = 0
    mismatched = []
    for image_path in tqdm(image_paths, leave=True, colour='CYAN'):
        source = image_path
        if unlabeled:
            image = cv2.imread(image_path)
            assert image is not None
            source = _letterbox_classification_image(image)
        results = best_model.predict(source=source, verbose=False, save=False)
        result = results[0]
        total_count += 1
        assert isinstance(result.probs, Probs)
        pred_idx = result.probs.top1
        pred_name = class_names[pred_idx]
        if unlabeled:
            assert best_model.predictor and hasattr(best_model.predictor, 'save_dir')
            save_dir = Path(best_model.predictor.save_dir) / pred_name
            save_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(image_path, save_dir / os.path.basename(image_path))
            continue
        true_name = os.path.basename(os.path.dirname(image_path))
        true_idx = name_to_idx.get(true_name)
        assert true_idx is not None
        if pred_idx != true_idx:
            assert best_model.predictor and hasattr(best_model.predictor, 'save_dir')
            save_dir = Path(best_model.predictor.save_dir) / f'{true_name} - {pred_name}'
            save_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(image_path, save_dir / os.path.basename(image_path))
            mismatched.append(f'expect: {true_idx}-{true_name}, actual: {pred_idx}-{pred_name} ==> {image_path}')

    print('✅ Validation completed!')
    print(f'📊 Total samples processed: {total_count}')
    if mismatched:
        assert best_model.predictor and hasattr(best_model.predictor, 'save_dir')
        mismatched_txt = Path(best_model.predictor.save_dir) / 'mismatched_samples.txt'
        with mismatched_txt.open('w', encoding='utf-8') as stream:
            stream.write('# Mismatched Samples Report\n')
            stream.write(f'# Total mismatched: {len(mismatched)} / {total_count}\n')
            stream.write('\n'.join(mismatched))
        print(f'❌ Found {len(mismatched)} mismatched samples.')
        print(f'📄 Results saved at: {mismatched_txt}\n')


def review_model(best_model: YOLO, directory: str | Path, unlabeled: bool = False) -> None:
    if unlabeled and best_model.task != 'classify':
        raise ValueError('Unlabeled review is only supported for classification models.')
    if best_model.task == 'classify':
        _review_classification(best_model, directory, unlabeled)
    else:
        _review_standard(best_model, directory)


def review(scenario_path: str | Path, weights: str | Path, directory: str | Path, unlabeled: bool = False) -> None:
    load_scenario(scenario_path)
    best_model = YOLO(weights)
    review_model(best_model, directory, unlabeled)
