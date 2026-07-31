from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar

import cv2
import numpy as np
from PIL import Image

from xxtrain.data import CocoDoc, CocoImage, ImageInfo, LabelCatalog
from xxtrain.data.dataset import split_membership, write_dataset_yaml, write_split_lists
from xxtrain.data.formats import write_labelimg, write_labelme
from xxtrain.data.formats.coco import write_coco
from xxtrain.pipeline.annotation_io import validate_annotations_for_task
from xxtrain.pipeline.core import ClassifyOutput, Context, EncodeOutput, Sample
from xxtrain.task import TaskType

OutputT = TypeVar('OutputT')


class DatasetSink(Protocol[OutputT]):
    input_type: type[OutputT]

    def write(self, item: OutputT, context: Context) -> None:
        raise NotImplementedError

    def finalize(self, context: Context) -> None:
        raise NotImplementedError


def _symlink_or_copy(source: Path, target: Path) -> None:
    try:
        os.symlink(source, target)
    except OSError:
        if target.is_symlink():
            try:
                if target.samefile(source):
                    return
            except OSError:
                pass
            target.unlink()
        shutil.copy2(source, target)


def _append_suffix(path: Path, suffix: str) -> Path:
    return path.parent / f'{path.name}{suffix}'


def _output_base(sample: Sample, context: Context) -> Path:
    logical = Path(sample.id)
    if not logical.parts or logical.is_absolute() or logical.drive or logical.root or '..' in logical.parts:
        raise ValueError(f'Unsafe sample id: {sample.id}')
    return context.config.root_path / context.config.task_name / logical


def _materialize_crop(sample: Sample, output_base: Path) -> tuple[Path, ImageInfo] | None:
    if sample.image.crop_box is None:
        return None
    image_path = _append_suffix(output_base, '.jpg')
    image_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(sample.image.path) as image:
        crop = image.crop(sample.image.crop_box)
        if crop.mode != 'RGB':
            crop = crop.convert('RGB')
        crop.save(image_path)
        width, height = crop.size
    return image_path, ImageInfo(width=width, height=height)


def _image_reference(image_path: Path, output_directory: Path) -> str:
    try:
        reference = os.path.relpath(image_path, output_directory)
    except ValueError:
        reference = str(image_path.absolute())
    return reference.replace('\\', '/')


def _finalize_dataset(context: Context) -> None:
    config = context.config
    (config.root_path / config.task_name).mkdir(parents=True, exist_ok=True)
    write_split_lists(config.root_path, config.task_name, context.report.train_items, context.report.val_items)
    write_dataset_yaml(config.root_path, config.task_name, config.task_type, config.labels)


class YoloDatasetSink:
    input_type = EncodeOutput
    tracks_output_labels = True

    def write(self, item: EncodeOutput, context: Context) -> None:
        if not isinstance(item, EncodeOutput):
            raise TypeError(f'{type(self).__name__} expected EncodeOutput, got {type(item).__name__}')

        sample = item.sample
        output_base = context.config.root_path / context.config.task_name / sample.id
        output_base.parent.mkdir(parents=True, exist_ok=True)
        if sample.image.crop_box is None:
            image_path = _append_suffix(output_base, sample.image.path.suffix)
            _symlink_or_copy(sample.image.path, image_path)
        else:
            image_path = _append_suffix(output_base, '.jpg')
            with Image.open(sample.image.path) as image:
                crop = image.crop(sample.image.crop_box)
                if crop.mode != 'RGB':
                    crop = crop.convert('RGB')
                crop.save(image_path)

        _append_suffix(output_base, '.txt').write_text('\n'.join(item.lines), encoding='utf-8')
        for line in item.lines:
            label_index = int(line.split(maxsplit=1)[0])
            context.report.record_output_label(context.config.labels.names[label_index])
        if not item.lines:
            context.report.record_missing_annotations(sample.source_group)
            if not context.config.reserve_no_label:
                return

        in_train, in_val = split_membership(sample.source_index, context.config.split)
        output_path = str(image_path)
        context.report.record_output(
            train_item=output_path if in_train else None,
            val_item=output_path if in_val else None,
            annotation_count=item.annotation_count,
        )

    def finalize(self, context: Context) -> None:
        _finalize_dataset(context)


class CocoSink:
    input_type = Sample

    def __init__(self) -> None:
        self._images: list[CocoImage] = []
        self._claimed_paths: set[Path] = set()
        self._file_names: set[str] = set()

    def _claim(self, path: Path) -> None:
        if path in self._claimed_paths:
            raise ValueError('COCO crop output path collision')
        self._claimed_paths.add(path)

    def _claim_file_name(self, file_name: str) -> None:
        if file_name in self._file_names:
            raise ValueError('COCO image file_name collision')
        self._file_names.add(file_name)

    @staticmethod
    def _remove_crop(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def write(self, item: Sample, context: Context) -> None:
        if not isinstance(item, Sample):
            raise TypeError(f'{type(self).__name__} expected Sample, got {type(item).__name__}')

        output_base = _output_base(item, context)
        if not item.annotations:
            if not context.config.reserve_no_label:
                context.report.record_missing_annotations(item.source_group)
                return

        annotations = validate_annotations_for_task(
            item.annotations, context.config.task_type, context.config.labels.names
        )
        output_root = context.config.root_path / context.config.task_name
        if item.image.crop_box is None:
            image_path = item.image.path
            image_info = item.image.require_info()
            file_name = _image_reference(image_path, output_root)
            image = CocoImage(file_name=file_name, info=image_info, annotations=annotations)
        else:
            crop_path = _append_suffix(output_base, '.jpg')
            if crop_path in self._claimed_paths or crop_path.exists():
                raise ValueError('COCO crop output path collision')
            file_name = _image_reference(crop_path, output_root)
            if file_name in self._file_names:
                raise ValueError('COCO image file_name collision')
            try:
                image_path, image_info = _materialize_crop(item, output_base)
                image = CocoImage(file_name=file_name, info=image_info, annotations=annotations)
            except Exception:
                self._remove_crop(crop_path)
                raise

        self._claim_file_name(file_name)
        if item.image.crop_box is not None:
            self._claim(image_path)
        self._images.append(image)
        if not item.annotations:
            context.report.record_missing_annotations(item.source_group)
        in_train, in_val = split_membership(item.source_index, context.config.split)
        output_item = str(image_path)
        context.report.record_output(
            train_item=output_item if in_train else None,
            val_item=output_item if in_val else None,
            annotation_count=len(annotations),
        )

    def finalize(self, context: Context) -> None:
        labels = (
            LabelCatalog((context.config.labels.names[0],))
            if context.config.task_type is TaskType.POSE
            else context.config.labels
        )
        write_coco(
            CocoDoc(labels=labels, images=tuple(self._images)),
            context.config.root_path / context.config.task_name / 'annotations.json',
        )
        self._images.clear()
        self._claimed_paths.clear()
        self._file_names.clear()


class _AnnotationSink:
    input_type = Sample
    suffix: str

    def __init__(self) -> None:
        self._claimed_paths: set[Path] = set()

    def _ensure_available(self, *paths: Path) -> None:
        if any(path in self._claimed_paths or path.exists() for path in paths):
            raise ValueError('Annotation output path collision')

    def _commit_claim(self, *paths: Path) -> None:
        self._claimed_paths.update(paths)

    @staticmethod
    def _remove_new_file(path: Path | None) -> None:
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def _prepare(self, item: Sample, context: Context) -> tuple[Path, Path | None, ImageInfo] | None:
        if not isinstance(item, Sample):
            raise TypeError(f'{type(self).__name__} expected Sample, got {type(item).__name__}')
        output_base = _output_base(item, context)
        if not item.annotations:
            if not context.config.reserve_no_label:
                context.report.record_missing_annotations(item.source_group)
                return None

        validate_annotations_for_task(item.annotations, context.config.task_type, context.config.labels.names)
        annotation_path = _append_suffix(output_base, self.suffix)
        crop_path = _append_suffix(output_base, '.jpg') if item.image.crop_box is not None else None
        paths = (annotation_path, *((crop_path,) if crop_path is not None else ()))
        self._ensure_available(*paths)
        try:
            materialized = _materialize_crop(item, output_base)
        except Exception:
            self._remove_new_file(crop_path)
            raise
        if materialized is None:
            return annotation_path, None, item.image.require_info()
        image_path, image_info = materialized
        return annotation_path, image_path, image_info

    def _write_annotation(
        self,
        item: Sample,
        context: Context,
        prepared: tuple[Path, Path | None, ImageInfo],
        write: Callable[[Path], None],
    ) -> None:
        annotation_path, image_path, _ = prepared
        crop_path = _append_suffix(_output_base(item, context), '.jpg') if item.image.crop_box is not None else None
        temporary_path: Path | None = None
        try:
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f'.{annotation_path.name}.', suffix='.tmp', dir=annotation_path.parent
            )
            temporary_path = Path(temp_name)
            os.close(descriptor)
            write(temporary_path)
            if annotation_path.exists():
                raise ValueError('Annotation output path collision')
            os.replace(temporary_path, annotation_path)
        except Exception:
            self._remove_new_file(temporary_path)
            self._remove_new_file(crop_path)
            raise

        self._commit_claim(annotation_path, *((crop_path,) if crop_path is not None else ()))
        if not item.annotations:
            context.report.record_missing_annotations(item.source_group)
        self._record_output(item, context, image_path)

    def _record_output(self, item: Sample, context: Context, image_path: Path | None) -> None:
        in_train, in_val = split_membership(item.source_index, context.config.split)
        output_path = str(image_path if image_path is not None else item.image.path)
        context.report.record_output(
            train_item=output_path if in_train else None,
            val_item=output_path if in_val else None,
            annotation_count=len(item.annotations),
        )

    def finalize(self, context: Context) -> None:
        (context.config.root_path / context.config.task_name).mkdir(parents=True, exist_ok=True)


class LabelImgSink(_AnnotationSink):
    suffix = '.xml'

    def write(self, item: Sample, context: Context) -> None:
        if not isinstance(item, Sample):
            raise TypeError(f'{type(self).__name__} expected Sample, got {type(item).__name__}')
        if context.config.task_type is not TaskType.DETECT:
            raise ValueError('LabelImgSink only supports detect tasks')
        prepared = self._prepare(item, context)
        if prepared is None:
            return
        self._write_annotation(
            item, context, prepared, lambda path: write_labelimg(item.annotations, path, prepared[2])
        )


class LabelMeSink(_AnnotationSink):
    suffix = '.json'

    def write(self, item: Sample, context: Context) -> None:
        prepared = self._prepare(item, context)
        if prepared is None:
            return
        annotation_path, image_path, image_info = prepared
        image_reference = (
            _image_reference(item.image.path, annotation_path.parent)
            if image_path is None
            else _image_reference(image_path, annotation_path.parent)
        )
        self._write_annotation(
            item,
            context,
            prepared,
            lambda path: write_labelme(
                item.annotations,
                path,
                image_info,
                image_path=image_reference,
                obb=context.config.task_type is TaskType.OBB,
            ),
        )


def _letterbox_classification_image(image: np.ndarray, image_size: int = 224) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(image_size / width, image_size / height)
    if abs(scale - 1.0) >= 1e-6:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    resized_height, resized_width = image.shape[:2]
    pad_width = image_size - resized_width
    pad_height = image_size - resized_height
    top = pad_height // 2
    left = pad_width // 2
    return cv2.copyMakeBorder(
        image, top, pad_height - top, left, pad_width - left, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )


class ClassificationDatasetSink:
    input_type = ClassifyOutput
    tracks_output_labels = True

    def __init__(self, *, image_size: int = 224, indexed_class_directories: bool = False):
        self.image_size = image_size
        self.indexed_class_directories = indexed_class_directories

    def write(self, item: ClassifyOutput, context: Context) -> None:
        if not isinstance(item, ClassifyOutput):
            raise TypeError(f'{type(self).__name__} expected ClassifyOutput, got {type(item).__name__}')

        class_directory = item.class_name
        if self.indexed_class_directories:
            label_index = context.config.labels.index(item.class_name)
            class_directory = f'{label_index:02d}-{item.class_name}'

        image = self._prepare_image(item)
        in_train, in_val = split_membership(item.sample.source_index, context.config.split)
        selected_splits = (('train', in_train), ('val', in_val))
        wrote_output = False
        for split_name, selected in selected_splits:
            if not selected:
                continue
            target = (
                context.config.root_path / context.config.task_name / split_name / class_directory / item.output_name
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), image):
                raise OSError(f'failed to write classification image: {target}')

            output_path = str(target)
            context.report.record_output(
                train_item=output_path if split_name == 'train' else None,
                val_item=output_path if split_name == 'val' else None,
                annotation_count=1,
            )
            wrote_output = True
        if wrote_output:
            context.report.record_output_label(item.class_name)

    def _prepare_image(self, item: ClassifyOutput) -> np.ndarray:
        image = cv2.imread(str(item.sample.image.path))
        assert image is not None
        crop_box = item.sample.image.crop_box
        if crop_box is not None:
            x1, y1, x2, y2 = map(int, crop_box)
            image = image[y1:y2, x1:x2]
        return _letterbox_classification_image(image, self.image_size)

    def finalize(self, context: Context) -> None:
        _finalize_dataset(context)


def print_conversion_report(context: Context) -> None:
    report = context.report
    print('\n\033[1;32m[Convert Summary]\033[0m')
    print(f'训练集图片总数: {report.train_image_count}, 标注总数: {report.train_annotation_count}')
    print(f'验证集图片总数: {report.val_image_count}, 标注总数: {report.val_annotation_count}')
    print(f'类别列表: {list(context.config.labels)}\n')
    if report.missing_annotation_counts:
        print('\033[1;31m[Warning] 以下目录包含没有标注的图片\033[0m')
        for directory, count in report.missing_annotation_counts.items():
            print(f'  - {directory}: {count}张图片')
    if report.source_label_counts:
        print('\n来源标签统计:')
        for label, count in sorted(report.source_label_counts.items()):
            file_count = len(report.source_label_files[label])
            print(f'  - {label}: {count}条标注, {file_count}张图片')
    if report.ignored_label_counts:
        print('\n已忽略的非目标标签:')
        for label, count in sorted(report.ignored_label_counts.items()):
            print(f'  - {label}: {count}条标注')
    if report.output_label_counts:
        print('\n最终输出类别统计:')
        for label, count in sorted(report.output_label_counts.items()):
            print(f'  - {label}: {count}条样本')
    if report.missing_output_labels:
        print('\n\033[1;31m[Error] 以下目标类别没有最终输出样本\033[0m')
        for label in report.missing_output_labels:
            print(f'  - {label}')
    if report.skipped_files:
        print('\033[1;33m[Warning] 以下图片包含被忽略的非目标标签:\033[0m')
        for path in sorted(report.skipped_files):
            print(f'  - {path}')
