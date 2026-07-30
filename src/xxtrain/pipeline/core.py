from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Generic, Self, TypeVar

from xxtrain.data import Annotation, Bbox, ImageInfo, LabelCatalog, Shape
from xxtrain.task import TaskType


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageRef:
    path: Path
    info: ImageInfo | None = None
    crop_box: tuple[float, float, float, float] | None = None

    def wrap(self, **changes: object) -> Self:
        return replace(self, **changes)

    def require_info(self) -> ImageInfo:
        if self.info is None:
            raise ValueError(f'Image metadata is not loaded: {self.path}')
        return self.info


@dataclass(frozen=True, slots=True, kw_only=True)
class Sample:
    id: str
    source_group: str
    source_index: int
    image: ImageRef
    annotations: tuple[Annotation, ...] = ()

    def wrap(self, **changes: object) -> Self:
        return replace(self, **changes)


@dataclass(frozen=True, slots=True, kw_only=True)
class MatchInput:
    sample: Sample
    parents: tuple[Bbox, ...]
    children: tuple[Shape, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class AnnotationMatch:
    parent: Bbox
    children: tuple[Shape, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class MatchOutput:
    sample: Sample
    matches: tuple[AnnotationMatch, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class CropOutput:
    sample: Sample
    parent: Bbox


@dataclass(frozen=True, slots=True, kw_only=True)
class EncodeOutput:
    sample: Sample
    lines: tuple[str, ...]

    @property
    def annotation_count(self) -> int:
        return len(self.lines)


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassifyOutput:
    sample: Sample
    class_name: str
    output_name: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ConversionConfig:
    task_name: str
    task_type: TaskType
    root_path: Path
    split: int
    labels: LabelCatalog
    reserve_no_label: bool


@dataclass(slots=True)
class ConversionReport:
    train_items: list[str] = field(default_factory=list)
    val_items: list[str] = field(default_factory=list)
    train_image_count: int = 0
    val_image_count: int = 0
    train_annotation_count: int = 0
    val_annotation_count: int = 0
    skipped_labels: set[str] = field(default_factory=set)
    skipped_files: set[Path] = field(default_factory=set)
    missing_annotation_counts: dict[str, int] = field(default_factory=dict)
    source_label_counts: dict[str, int] = field(default_factory=dict)
    source_label_files: dict[str, set[Path]] = field(default_factory=dict)
    ignored_label_counts: dict[str, int] = field(default_factory=dict)
    output_label_counts: dict[str, int] = field(default_factory=dict)
    missing_output_labels: tuple[str, ...] = ()

    def record_output(self, *, train_item: str | None, val_item: str | None, annotation_count: int) -> None:
        if train_item is not None:
            self.train_items.append(train_item)
            self.train_image_count += 1
            self.train_annotation_count += annotation_count
        if val_item is not None:
            self.val_items.append(val_item)
            self.val_image_count += 1
            self.val_annotation_count += annotation_count

    def record_skipped_label(self, label: str) -> None:
        self.skipped_labels.add(label)

    def record_skipped_file(self, path: Path) -> None:
        self.skipped_files.add(path)

    def record_missing_annotations(self, source_group: str) -> None:
        self.missing_annotation_counts[source_group] = self.missing_annotation_counts.get(source_group, 0) + 1

    def record_source_label(self, label: str, source_path: Path) -> None:
        self.source_label_counts[label] = self.source_label_counts.get(label, 0) + 1
        self.source_label_files.setdefault(label, set()).add(source_path)

    def record_ignored_label(self, label: str) -> None:
        self.ignored_label_counts[label] = self.ignored_label_counts.get(label, 0) + 1

    def record_output_label(self, label: str) -> None:
        self.output_label_counts[label] = self.output_label_counts.get(label, 0) + 1

    def set_missing_output_labels(self, labels: tuple[str, ...]) -> None:
        self.missing_output_labels = labels


@dataclass(frozen=True, slots=True, kw_only=True)
class Context:
    config: ConversionConfig
    report: ConversionReport


InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


def _type_name(value_type: type) -> str:
    return value_type.__name__


class ItemProcessor(ABC, Generic[InputT, OutputT]):
    input_type: type[InputT]
    output_type: type[OutputT]

    @abstractmethod
    def transform(self, item: InputT, context: Context) -> OutputT | None:
        raise NotImplementedError

    def apply(self, items: Iterable[InputT], context: Context) -> Iterator[OutputT]:
        for item in items:
            self._validate_input(item)
            output = self.transform(item, context)
            if output is not None:
                self._validate_output(output)
                yield output

    def _validate_input(self, item: object) -> None:
        if not isinstance(item, self.input_type):
            raise TypeError(f'{type(self).__name__} expected {_type_name(self.input_type)}, got {type(item).__name__}')

    def _validate_output(self, output: object) -> None:
        if not isinstance(output, self.output_type):
            raise TypeError(
                f'{type(self).__name__} declared {_type_name(self.output_type)}, got {type(output).__name__}'
            )


class ExpandProcessor(ABC, Generic[InputT, OutputT]):
    input_type: type[InputT]
    output_type: type[OutputT]

    @abstractmethod
    def expand(self, item: InputT, context: Context) -> Iterable[OutputT]:
        raise NotImplementedError

    def apply(self, items: Iterable[InputT], context: Context) -> Iterator[OutputT]:
        for item in items:
            if not isinstance(item, self.input_type):
                raise TypeError(
                    f'{type(self).__name__} expected {_type_name(self.input_type)}, got {type(item).__name__}'
                )
            for output in self.expand(item, context):
                if not isinstance(output, self.output_type):
                    raise TypeError(
                        f'{type(self).__name__} declared {_type_name(self.output_type)}, got {type(output).__name__}'
                    )
                yield output


Processor = ItemProcessor[object, object] | ExpandProcessor[object, object]


@dataclass(frozen=True, slots=True)
class Pipeline:
    processors: tuple[Processor, ...]

    def __post_init__(self) -> None:
        for previous, following in zip(self.processors, self.processors[1:]):
            if not issubclass(previous.output_type, following.input_type):
                raise TypeError(
                    f'{type(previous).__name__} produces {_type_name(previous.output_type)}, '
                    f'but {type(following).__name__} expects {_type_name(following.input_type)}'
                )

    def validate_boundaries(self, source_type: type, sink_type: type) -> None:
        if not self.processors:
            if not issubclass(source_type, sink_type):
                raise TypeError(f'source produces {_type_name(source_type)}, but sink expects {_type_name(sink_type)}')
            return
        first, last = self.processors[0], self.processors[-1]
        if not issubclass(source_type, first.input_type):
            raise TypeError(
                f'source produces {_type_name(source_type)}, but {type(first).__name__} '
                f'expects {_type_name(first.input_type)}'
            )
        if not issubclass(last.output_type, sink_type):
            raise TypeError(
                f'{type(last).__name__} produces {_type_name(last.output_type)}, '
                f'but sink expects {_type_name(sink_type)}'
            )

    def run(self, items: Iterable[object], context: Context) -> Iterator[object]:
        stream: Iterable[object] = items
        for processor in self.processors:
            stream = processor.apply(stream, context)
        yield from stream
