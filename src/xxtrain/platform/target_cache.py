import json
import os
import shutil
from pathlib import Path

import yaml

from xxtrain.data import Bbox, ImageInfo, LabelCatalog, Polyline
from xxtrain.data.formats import decode_segment
from xxtrain.pipeline import Context, ConversionConfig, ConversionReport, ImageRef, Sample
from xxtrain.pipeline.core import ClassifyOutput, CropOutput
from xxtrain.pipeline.processors import EncodePointSegment
from xxtrain.pipeline.sinks import ClassificationDatasetSink, YoloDatasetSink
from xxtrain.platform.contracts import EditFrame
from xxtrain.task import TaskType
from xxtrain.workspace_data import WorkspaceData

_CLASSIFICATION_LABELS = LabelCatalog(('tl', 'tc', 'cl', 'cc'))
_SEGMENT_LABELS = LabelCatalog(('Point',))
_SUPPORTED_TARGETS = frozenset({'classify', 'segment'})


def build_target_cache(data: WorkspaceData, target: str, runtime_root: Path, destination: Path) -> ConversionReport:
    """Build and atomically publish one Point downstream cache from shared crop files.

    Unsupported targets, incomplete annotations, and segment triangles outside their crop raise ``ValueError``.
    Failed builds do not publish a completion manifest or write derived annotations to the workspace database.
    """
    _validate_target(target)
    destination = Path(destination).absolute()
    temporary = destination.with_name(f'.{destination.name}.building')
    shutil.rmtree(temporary, ignore_errors=True)
    try:
        frames = data.target_frames(target, runtime_root)
        context = _context(target, temporary)
        if target == 'classify':
            sink = ClassificationDatasetSink(indexed_class_directories=True)
            _write_classification(frames, context, sink)
        else:
            sink = YoloDatasetSink()
            _write_segments(frames, context, sink)
            _materialize_segment_images(context.report)

        _rewrite_report_paths(context.report, temporary, destination)
        sink.finalize(context)
        _rewrite_dataset_root(temporary, target, destination)
        manifest = {'fingerprint': data.target_fingerprint(target), 'target': target}
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


def _validate_target(target: str) -> None:
    if target not in _SUPPORTED_TARGETS:
        raise ValueError(f'Unsupported target cache: {target!r}')


def _context(target: str, root: Path) -> Context:
    task_type = TaskType.CLASSIFY if target == 'classify' else TaskType.SEGMENT
    labels = _CLASSIFICATION_LABELS if target == 'classify' else _SEGMENT_LABELS
    return Context(
        config=ConversionConfig(
            task_name=target, task_type=task_type, root_path=root, split=10, labels=labels, reserve_no_label=False
        ),
        report=ConversionReport(),
    )


def _sample(frame: EditFrame, index: int, annotations: tuple[Polyline, ...] = ()) -> Sample:
    return Sample(
        id=f'workspace/{frame.mapping.frame_id}',
        source_group='workspace',
        source_index=index,
        image=ImageRef(
            path=frame.image_path.absolute(), info=ImageInfo(width=frame.width, height=frame.height), crop_box=None
        ),
        annotations=annotations,
    )


def _write_classification(frames: tuple[EditFrame, ...], context: Context, sink: ClassificationDatasetSink) -> None:
    for index, frame in enumerate(frames):
        if len(frame.annotations) != 1:
            raise ValueError(f'Classification frame {frame.mapping.frame_id!r} requires exactly one category')
        annotation = frame.annotations[0]
        sample = _sample(frame, index)
        sink.write(
            ClassifyOutput(sample=sample, class_name=annotation.label, output_name=f'{frame.mapping.frame_id}.png'),
            context,
        )


def _write_segments(frames: tuple[EditFrame, ...], context: Context, sink: YoloDatasetSink) -> None:
    encoder = EncodePointSegment()
    for index, frame in enumerate(frames):
        if not frame.annotations:
            raise ValueError(f'Segment frame {frame.mapping.frame_id!r} requires at least one line')
        annotations = tuple(
            Polyline(id=annotation.id, label=annotation.label, points=annotation.geometry)
            for annotation in frame.annotations
        )
        sample = _sample(frame, index, annotations)
        parent_id = frame.mapping.parent_id
        if parent_id is None:
            raise ValueError(f'Segment frame {frame.mapping.frame_id!r} has no source box identity')
        crop = CropOutput(
            sample=sample, parent=Bbox(id=parent_id, label='Point', x1=0, y1=0, x2=frame.width, y2=frame.height)
        )
        try:
            output = encoder.transform(crop, context)
        except AssertionError as error:
            raise ValueError(
                f'Segment frame {frame.mapping.frame_id!r} generates a triangle outside crop bounds'
            ) from error
        try:
            for line in output.lines:
                decode_segment(line, sample.image.require_info(), context.config.labels)
        except ValueError as error:
            raise ValueError(
                f'Segment frame {frame.mapping.frame_id!r} generates a triangle outside crop bounds'
            ) from error
        sink.write(output, context)


def _materialize_segment_images(report: ConversionReport) -> None:
    for item in (*report.train_items, *report.val_items):
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
