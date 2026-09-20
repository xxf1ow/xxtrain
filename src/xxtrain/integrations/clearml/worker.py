from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def train_prepared(*args: object, **kwargs: object) -> Any:
    from xxtrain.training.prepared import train_prepared as implementation

    return implementation(*args, **kwargs)


def build_delivery(*args: object, **kwargs: object) -> Path:
    from xxtrain.training.delivery import build_delivery as implementation

    return implementation(*args, **kwargs)


def _task_init() -> Any:
    from clearml import Task

    return Task.init(auto_connect_arg_parser=False, auto_connect_frameworks=False)


def main(argv: Sequence[str] | None = None) -> None:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if '-h' in raw_arguments or '--help' in raw_arguments:
        parser.parse_args(raw_arguments)
    task = _task_init()
    task.connect(parser)
    arguments = parser.parse_args(raw_arguments)
    try:
        definition = _training_definition(arguments.task, arguments.target)
        dataset_dir = _contained_path(Path(arguments.shared_root), arguments.cache_relative_path)
        run_id = str(uuid.UUID(arguments.run_id))
        run_dir = _contained_output_path(Path(arguments.run_root), run_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        _disable_ultralytics_clearml(run_dir)
        logger = task.get_logger()

        def report_progress(progress: Any) -> None:
            logger.report_scalar(title='training', series='epoch', value=progress.epoch, iteration=progress.epoch)
            logger.report_scalar(
                title='training', series='total_epochs', value=progress.total_epochs, iteration=progress.epoch
            )
            task.set_parameter('xxtrain/epoch', progress.epoch)
            task.set_parameter('xxtrain/total_epochs', progress.total_epochs)

        def report_metric(metric: float | None, epoch: int) -> None:
            if metric is not None:
                logger.report_scalar(title='validation', series=definition.metric_name, value=metric, iteration=epoch)
                task.set_parameter('xxtrain/metric', metric)

        def report_validation(progress: Any, metrics: Mapping[str, float]) -> None:
            if progress.epoch % 5 == 0:
                report_metric(metrics.get(definition.metric_key), progress.epoch)

        result = train_prepared(
            definition.settings, dataset_dir, run_dir, on_progress=report_progress, on_validation=report_validation
        )
        delivery = build_delivery(result, definition.delivery, dataset_dir, run_dir / 'delivery')
        metric = result.metrics.get(definition.metric_key)
        report_metric(metric, int(definition.settings.train_args['epochs']))
        if metric is None:
            task.set_parameter('xxtrain/metric', None)
        manifest = {
            'filename': delivery.name,
            'target': arguments.target,
            'metric_name': definition.metric_name,
            'metric_value': metric,
        }
        uploaded = task.upload_artifact(
            name='deployment',
            artifact_object=delivery,
            metadata={'manifest': json.dumps(manifest, ensure_ascii=False, sort_keys=True)},
            wait_on_upload=True,
        )
        if not uploaded:
            raise RuntimeError('ClearML deployment artifact upload did not complete')
        task.mark_completed(ignore_errors=False)
    except Exception as error:
        task.mark_failed(
            ignore_errors=False, status_reason=type(error).__name__, status_message='Training worker failed'
        )
        raise
    finally:
        task.close()


def _training_definition(task: str, target: str) -> Any:
    if task != 'point':
        raise ValueError(f'Unknown business task: {task!r}')
    from xxtrain.business_tasks.point import point_task_definition

    training = point_task_definition().step(target).training
    if training is None:
        raise ValueError(f'Target does not define training: {target!r}')
    return training


def _contained_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError('Training cache path escapes the shared root')
    if not candidate.is_dir():
        raise FileNotFoundError('Prepared training cache is unavailable')
    return candidate


def _contained_output_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError('Training run path escapes the run root')
    return candidate


def _disable_ultralytics_clearml(run_dir: Path) -> None:
    config_dir = run_dir / 'ultralytics-config'
    config_dir.mkdir()
    os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
    from ultralytics import settings

    dict.__setitem__(settings, 'clearml', False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--cache-relative-path', required=True)
    parser.add_argument('--shared-root', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--run-root', default='runs')
    return parser


if __name__ == '__main__':
    main()
