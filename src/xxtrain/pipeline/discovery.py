from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from xxtrain.data import LabelCatalog, Pose
from xxtrain.data.formats.coco import read_coco
from xxtrain.task import TaskType

from .annotation_io import validate_annotations_for_task
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

    def catalog_for(self, task_type: TaskType) -> LabelCatalog | None:
        raise NotImplementedError

    def read(self, context: Context) -> Iterable[Sample]:
        raise NotImplementedError


class DirectorySource:
    output_type = Sample

    def catalog_for(self, task_type: TaskType) -> LabelCatalog | None:
        return None

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


class CocoSource:
    output_type = Sample

    def __init__(
        self, json_path: str | Path, *, image_root: str | Path | None = None, catalog: LabelCatalog | None = None
    ):
        self.json_path = Path(json_path)
        self.image_root = Path(image_root) if image_root is not None else self.json_path.parent
        self.doc = read_coco(self.json_path)
        self.explicit_catalog = catalog

    def catalog_for(self, task_type: TaskType) -> LabelCatalog | None:
        if task_type is not TaskType.POSE:
            derived = self.doc.labels
        else:
            poses = tuple(
                annotation
                for image in self.doc.images
                for annotation in image.annotations
                if isinstance(annotation, Pose)
            )
            if not poses:
                if self.explicit_catalog is None:
                    raise ValueError('Pose catalog cannot be derived without Pose annotations')
                if len(self.explicit_catalog) < 2:
                    raise ValueError('Pose catalog requires at least one keypoint label')
                return self.explicit_catalog
            pose_labels = {pose.label for pose in poses}
            if len(pose_labels) != 1:
                raise ValueError('Pose catalog requires exactly one object category')
            schemas = {tuple(keypoint.label for keypoint in pose.keypoints) for pose in poses}
            if len(schemas) != 1:
                raise ValueError('Pose annotations do not share one keypoint schema')
            derived = LabelCatalog((poses[0].label, *schemas.pop()))

        if self.explicit_catalog is not None and self.explicit_catalog != derived:
            raise ValueError('Explicit COCO catalog does not match the derived catalog')
        return self.explicit_catalog or derived

    def read(self, context: Context) -> Iterable[Sample]:
        expected = self.catalog_for(context.config.task_type)
        if expected is not None and expected != context.config.labels:
            raise ValueError('COCO source catalog does not match conversion labels')

        for index, image in enumerate(self.doc.images):
            file_path = Path(image.file_name)
            if not file_path.is_absolute():
                file_path = self.image_root / file_path
            file_path = file_path.absolute()
            annotations = validate_annotations_for_task(
                image.annotations, context.config.task_type, context.config.labels.names
            )
            yield Sample(
                id=f'coco/{index:06d}',
                source_group='coco',
                source_index=index,
                image=ImageRef(path=file_path, info=image.info),
                annotations=annotations,
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
                raise ValueError(f"Classification split leaves class '{class_path.name}' without train or val samples")
    return LabelCatalog(tuple(sorted(labels.names)))
