from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .settings import TrainingResult

if TYPE_CHECKING:
    from xxtrain.business_tasks.definition import DeliveryDefinition

_INDEXED_CLASS = re.compile(r'^(?P<index>\d+)-(?P<label>.+)$')


def build_delivery(result: TrainingResult, definition: DeliveryDefinition, dataset_dir: Path, output_dir: Path) -> Path:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not definition.labels and not definition.reference_images:
        output = output_dir / 'model.onnx'
        shutil.copy2(result.onnx_path, output)
        return output

    indices = sorted(result.class_names)
    if indices != list(range(len(indices))):
        raise ValueError('Model class indices must be contiguous zero-based integers')
    catalog = _load_catalog(dataset_dir)
    classes = [
        (output_index, *_resolve_class(dataset_dir, result.class_names[output_index], catalog))
        for output_index in indices
    ]
    labels = [label for _output_index, label, _image in classes]
    if len(set(labels)) != len(labels):
        raise ValueError(f'Duplicate delivery labels: {labels}')
    output = output_dir / 'model.zip'
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(result.onnx_path, 'model.onnx')
        if definition.labels:
            archive.writestr('labels.txt', ''.join(f'{label}\n' for label in labels))
        if definition.reference_images:
            for output_index, label, image in classes:
                archive.write(image, f'references/{output_index}_{label}{image.suffix}')
    return output


def _load_catalog(dataset_dir: Path) -> dict[int, str]:
    metadata = yaml.safe_load((dataset_dir / 'dataset.yaml').read_text(encoding='utf-8'))
    names = metadata.get('names', {})
    values = names.items() if isinstance(names, dict) else enumerate(names)
    return {int(index): str(label) for index, label in values}


def _resolve_class(dataset_dir: Path, model_name: str, catalog: dict[int, str]) -> tuple[str, Path]:
    match = _INDEXED_CLASS.fullmatch(model_name)
    if match:
        source_index = int(match.group('index'))
        label = catalog.get(source_index)
        if label is None or label != match.group('label'):
            raise ValueError(f'Classification directory does not match dataset names: {model_name}')
    else:
        label = model_name
        if label not in catalog.values():
            raise ValueError(f'Model class not found in dataset names: {model_name}')
    candidates: list[Path] = []
    for split in ('train', 'val'):
        split_root = dataset_dir / split
        if not split_root.is_dir():
            continue
        for class_dir in split_root.iterdir():
            if not class_dir.is_dir():
                continue
            directory_match = _INDEXED_CLASS.fullmatch(class_dir.name)
            directory_label = directory_match.group('label') if directory_match else class_dir.name
            if class_dir.name == model_name or directory_label == label:
                candidates.extend(path for path in class_dir.iterdir() if path.is_file())
    if not candidates:
        raise ValueError(f'No reference image found for model class: {model_name}')
    return label, sorted(candidates)[0]
