from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Self

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

    def record_output(
        self,
        *,
        train_item: str | None,
        val_item: str | None,
        annotation_count: int,
    ) -> None:
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


@dataclass(frozen=True, slots=True, kw_only=True)
class Context:
    config: ConversionConfig
    report: ConversionReport
