from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol, TypeVar

import cv2
import numpy as np
from PIL import Image

from xxtrain.data.dataset import split_membership, write_dataset_yaml, write_split_lists
from xxtrain.pipeline.core import ClassifyOutput, Context, EncodeOutput

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
            assert cv2.imwrite(str(target), image)

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
