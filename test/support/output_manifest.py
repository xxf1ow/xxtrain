from collections.abc import Mapping
from pathlib import Path

from PIL import Image
from ruamel.yaml import YAML

IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png'}
YAML_SUFFIXES = {'.yaml', '.yml'}


def _normalize_string(value: str, resolved_root: str) -> str:
    return value.replace('\r\n', '\n').replace('\r', '\n').replace(resolved_root, '<ROOT>').replace('\\', '/')


def _normalize_yaml(value: object, resolved_root: str) -> object:
    if isinstance(value, Mapping):
        return {
            _normalize_string(str(key), resolved_root): _normalize_yaml(item, resolved_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_yaml(item, resolved_root) for item in value]
    if isinstance(value, str):
        return _normalize_string(value, resolved_root)
    return value


def collect_output_manifest(dataset_root: Path, task_type: str) -> dict[str, object]:
    resolved_root = str(dataset_root.resolve())
    output_root = dataset_root / task_type
    discovered_files = sorted(
        ((path.relative_to(output_root).as_posix(), path) for path in output_root.rglob('*') if path.is_file()),
        key=lambda item: item[0],
    )

    text_files: dict[str, str] = {}
    yaml_files: dict[str, object] = {}
    images: dict[str, dict[str, list[int]]] = {}
    yaml = YAML(typ='safe')

    for relative_path, path in discovered_files:
        suffix = path.suffix.lower()
        if suffix == '.txt':
            text_files[relative_path] = _normalize_string(path.read_bytes().decode('utf-8'), resolved_root)
        elif suffix in YAML_SUFFIXES:
            yaml_files[relative_path] = _normalize_yaml(yaml.load(path.read_bytes().decode('utf-8')), resolved_root)
        elif suffix in IMAGE_SUFFIXES:
            with Image.open(path) as image:
                images[relative_path] = {'size': list(image.size)}

    return {
        'files': [relative_path for relative_path, _ in discovered_files],
        'text_files': text_files,
        'yaml_files': yaml_files,
        'images': images,
    }
