import shutil
from datetime import datetime
from pathlib import Path

from ultralytics.models import YOLO

from xxtrain.task import TaskType

from .model import model_name
from .scenario import load_scenario


def copy_class_reference_images(
    root_path: str | Path,
    task_type: str,
    onnx_path: str | Path,
    class_names: dict[int, str],
) -> None:
    root = Path(root_path)
    references = []
    train_path = root / task_type / 'train'
    for class_index, class_name in sorted(class_names.items()):
        class_path = train_path / class_name
        image_path = next((entry for entry in class_path.iterdir() if entry.is_file()), None)
        if image_path is None:
            raise FileNotFoundError(f'No reference image found for class: {class_name}')
        references.append((class_index, class_name, image_path))

    onnx = Path(onnx_path)
    references_path = onnx.with_name(f'{onnx.stem}_references')
    references_path.mkdir(parents=True, exist_ok=True)
    for class_index, class_name, image_path in references:
        shutil.copy(image_path, references_path / f'{class_index}_{class_name}{image_path.suffix}')


def export_model_to_onnx(best_model: YOLO, root_path: str | Path, name: str) -> Path | None:
    print('🚀 Exporting best model to ONNX format ...')
    temp_onnx_path = best_model.export(format='onnx', simplify=True)
    formatted_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    onnx_path = Path(root_path) / 'weights' / f'{name}_{formatted_time}.onnx'
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(temp_onnx_path, onnx_path)
    print(f'✅ Model exported to ONNX format: {onnx_path}')
    return onnx_path


def export(scenario_path: str | Path, weights: str | Path) -> Path:
    scenario = load_scenario(scenario_path)
    root = Path(scenario_path).resolve().parent
    best_model = YOLO(weights)
    onnx_path = export_model_to_onnx(best_model, root, model_name(scenario))
    if onnx_path is not None and scenario.dataset.task_type is TaskType.CLASSIFY:
        copy_class_reference_images(root, scenario.dataset.name, onnx_path, best_model.names)
    return onnx_path
