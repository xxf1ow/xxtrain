from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol, TypeVar

import cv2
import numpy as np
from PIL import Image

from xxtrain.data import ImageInfo
from xxtrain.data.dataset import split_membership, write_dataset_yaml, write_split_lists
from xxtrain.data.formats import write_labelimg, write_labelme
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
    if not logical.parts or logical.is_absolute() or logical.drive or '..' in logical.parts:
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


class _AnnotationSink:
    input_type = Sample
    suffix: str

    def __init__(self) -> None:
        self._claimed_paths: set[Path] = set()

    def _claim(self, *paths: Path) -> None:
        if any(path in self._claimed_paths or path.exists() for path in paths):
            raise ValueError('Annotation output path collision')
        self._claimed_paths.update(paths)

    def _prepare(self, item: Sample, context: Context) -> tuple[Path, Path | None, ImageInfo] | None:
        if not isinstance(item, Sample):
            raise TypeError(f'{type(self).__name__} expected Sample, got {type(item).__name__}')
        if not item.annotations:
            context.report.record_missing_annotations(item.source_group)
            if not context.config.reserve_no_label:
                return None

        validate_annotations_for_task(item.annotations, context.config.task_type, context.config.labels.names)
        output_base = _output_base(item, context)
        annotation_path = _append_suffix(output_base, self.suffix)
        crop_path = _append_suffix(output_base, '.jpg') if item.image.crop_box is not None else None
        self._claim(annotation_path, *((crop_path,) if crop_path is not None else ()))
        materialized = _materialize_crop(item, output_base)
        if materialized is None:
            return annotation_path, None, item.image.require_info()
        image_path, image_info = materialized
        return annotation_path, image_path, image_info

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
        annotation_path, image_path, image_info = prepared
        write_labelimg(item.annotations, annotation_path, image_info)
        self._record_output(item, context, image_path)


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
        write_labelme(
            item.annotations,
            annotation_path,
            image_info,
            image_path=image_reference,
            obb=context.config.task_type is TaskType.OBB,
        )
        self._record_output(item, context, image_path)


class ClassificationDatasetSink:
    input_type = ClassifyOutput

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

    def _prepare_image(self, item: ClassifyOutput) -> np.ndarray:
        image = cv2.imread(str(item.sample.image.path))
        assert image is not None
        crop_box = item.sample.image.crop_box
        if crop_box is not None:
            x1, y1, x2, y2 = map(int, crop_box)
            image = image[y1:y2, x1:x2]

        height, width = image.shape[:2]
        scale = min(self.image_size / width, self.image_size / height)
        if abs(scale - 1.0) >= 1e-6:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        resized_height, resized_width = image.shape[:2]
        pad_width = self.image_size - resized_width
        pad_height = self.image_size - resized_height
        top = pad_height // 2
        left = pad_width // 2
        return cv2.copyMakeBorder(
            image, top, pad_height - top, left, pad_width - left, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

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
    if report.skipped_labels:
        print('\033[1;33m[Warning] 以下类别在标签列表中未定义\033[0m')
        for label in report.skipped_labels:
            print(f'  - {label}')
    if report.skipped_files:
        print('\033[1;33m[Warning] 以下图片因包含未定义类别而被跳过:\033[0m')
        for path in sorted(report.skipped_files):
            print(f'  - {path}')
