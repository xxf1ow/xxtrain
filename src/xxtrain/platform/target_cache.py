import json
import os
import shutil
from pathlib import Path

import yaml

from xxtrain.data import LabelCatalog
from xxtrain.pipeline import Context, ConversionConfig, ConversionReport
from xxtrain.pipeline.sinks import ClassificationDatasetSink, YoloDatasetSink
from xxtrain.task import TaskType
from xxtrain.workspace_data import WorkspaceData


def build_target_cache(data: WorkspaceData, target: str, runtime_root: Path, destination: Path) -> ConversionReport:
    """Build and atomically publish one task-defined training cache."""
    step = data.task.step(target)
    training = step.training
    if training is None:
        raise ValueError(f'Task step {target!r} does not define training conversion')
    destination = Path(destination).absolute()
    temporary = destination.with_name(f'.{destination.name}.building')
    shutil.rmtree(temporary, ignore_errors=True)
    try:
        frames = data.target_frames(target, runtime_root)
        context = Context(
            config=ConversionConfig(
                task_name=target,
                task_type=training.settings.task_type,
                root_path=temporary,
                split=10,
                labels=LabelCatalog(training.labels),
                reserve_no_label=step.annotation is not None and step.annotation.negative_label is not None,
            ),
            report=ConversionReport(),
        )
        sink = (
            ClassificationDatasetSink(indexed_class_directories=True)
            if training.settings.task_type is TaskType.CLASSIFY
            else YoloDatasetSink()
        )
        for index, frame in enumerate(frames):
            sink.write(training.encode_sample(frame, index, context), context)
        _materialize_linked_images(context.report)
        _rewrite_report_paths(context.report, temporary, destination)
        sink.finalize(context)
        _rewrite_dataset_root(temporary, target, destination)
        manifest = {'fingerprint': data.training_fingerprint(target), 'target': target}
        (temporary / 'manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(',', ':')), encoding='utf-8'
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        _remove_incomplete(destination)
        os.replace(temporary, destination)
        return context.report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _materialize_linked_images(report: ConversionReport) -> None:
    for item in {*report.train_items, *report.val_items}:
        image_path = Path(item)
        if image_path.is_symlink():
            source = image_path.resolve(strict=True)
            image_path.unlink()
            shutil.copy2(source, image_path)


def _rewrite_report_paths(report: ConversionReport, temporary: Path, destination: Path) -> None:
    report.train_items = [str(destination / Path(item).relative_to(temporary)) for item in report.train_items]
    report.val_items = [str(destination / Path(item).relative_to(temporary)) for item in report.val_items]


def _rewrite_dataset_root(temporary: Path, target: str, destination: Path) -> None:
    dataset_path = temporary / target / 'dataset.yaml'
    dataset = yaml.safe_load(dataset_path.read_text(encoding='utf-8'))
    dataset['path'] = str(destination)
    dataset_path.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding='utf-8')


def _remove_incomplete(destination: Path) -> None:
    if destination.is_dir() and not destination.is_symlink():
        shutil.rmtree(destination)
    else:
        destination.unlink(missing_ok=True)
