from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from xxtrain.data import LabelCatalog

from .core import Context, ImageRef, Sample

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


def _directories(path: Path) -> list[Path]:
    return sorted((item for item in path.iterdir() if item.is_dir()), key=lambda item: item.name)


def _images(path: Path) -> list[Path]:
    return sorted(
        (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda item: item.name,
    )


class SampleSource(Protocol):
    output_type: type[Sample]

    def read(self, context: Context) -> Iterable[Sample]:
        raise NotImplementedError


class DirectorySource:
    output_type = Sample

    def read(self, context: Context) -> Iterable[Sample]:
        source_root = context.config.root_path / 'src'
        assert source_root.is_dir(), f'数据集不存在: {source_root}'
        for group_path in _directories(source_root):
            images_path = group_path / 'imgs'
            if not images_path.is_dir():
                continue
            for index, image_path in enumerate(_images(images_path)):
                yield Sample(
                    id=f'{group_path.name}/{image_path.stem}',
                    source_group=group_path.name,
                    source_index=index,
                    image=ImageRef(path=image_path.absolute()),
                )


def validate_classification_source(root_path: Path, labels: LabelCatalog, split: int) -> LabelCatalog:
    source_root = root_path / 'src'
    class_paths = _directories(source_root)
    class_names = [path.name for path in class_paths]
    missing = sorted(set(labels.names) - set(class_names))
    extra = sorted(set(class_names) - set(labels.names))
    if missing or extra:
        raise ValueError(f'Classification labels mismatch: missing={missing}, extra={extra}')
    for class_path in class_paths:
        images_path = class_path / 'imgs'
        if not images_path.is_dir():
            raise ValueError(f'Classification image directory does not exist: {images_path}')
        images = _images(images_path)
        if not images:
            raise ValueError(f'Classification class has no images: {class_path.name}')
        if split > 0:
            train_count = sum(index % split != 0 for index in range(len(images)))
            val_count = sum(index % split == 0 for index in range(len(images)))
            if train_count == 0 or val_count == 0:
                raise ValueError(
                    f"Classification split leaves class '{class_path.name}' without train or val samples"
                )
    return LabelCatalog(tuple(sorted(labels.names)))
