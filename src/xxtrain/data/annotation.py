import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import ClassVar, Self, TypeAlias
from uuid import UUID, uuid4

import numpy as np

Point: TypeAlias = tuple[float, float]
PointLike: TypeAlias = Sequence[float] | np.ndarray
PointSetInput: TypeAlias = Sequence[PointLike] | np.ndarray


class AnnotationType(Enum):
    BBOX = 'bbox'
    POLYGON = 'polygon'
    POLYLINE = 'polyline'
    POINTS = 'points'
    CIRCLE = 'circle'
    POSE = 'pose'


def _finite_float(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Coordinates must be finite')
    return result


def _normalize_point(value: PointLike) -> Point:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError('Point must contain exactly two finite coordinates')
    return float(array[0]), float(array[1])


def _normalize_points(value: PointSetInput) -> tuple[Point, ...]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError('Points must have finite N x 2 coordinates')
    return tuple((float(x), float(y)) for x, y in array)


@dataclass(frozen=True, slots=True, kw_only=True)
class Annotation(ABC):
    label: str
    id: UUID = field(default_factory=uuid4)
    group: int | str | UUID | None = None
    type: ClassVar[AnnotationType]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError('Annotation label must be a non-empty string')
        if not isinstance(self.id, UUID):
            raise ValueError('Annotation id must be a UUID')
        if self.group is not None and not isinstance(self.group, (int, str, UUID)):
            raise ValueError('Annotation group must be int, str, UUID, or None')
        self._validate_geometry()

    @abstractmethod
    def _validate_geometry(self) -> None:
        raise NotImplementedError

    def wrap(self, **changes: object) -> Self:
        invalid = set(changes) - {'id', 'label', 'group'}
        if invalid:
            raise TypeError(f'Unsupported annotation changes: {sorted(invalid)}')
        return replace(self, **changes)


@dataclass(frozen=True, slots=True, kw_only=True)
class Shape(Annotation, ABC):
    @property
    @abstractmethod
    def points(self) -> tuple[Point, ...]:
        raise NotImplementedError

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = tuple(point[0] for point in self.points)
        ys = tuple(point[1] for point in self.points)
        return min(xs), min(ys), max(xs), max(ys)

    @abstractmethod
    def translate(self, dx: float, dy: float) -> Self:
        raise NotImplementedError


@dataclass(frozen=True, slots=True, kw_only=True)
class Bbox(Shape):
    type: ClassVar[AnnotationType] = AnnotationType.BBOX
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        object.__setattr__(self, 'x1', _finite_float(self.x1))
        object.__setattr__(self, 'y1', _finite_float(self.y1))
        object.__setattr__(self, 'x2', _finite_float(self.x2))
        object.__setattr__(self, 'y2', _finite_float(self.y2))
        Annotation.__post_init__(self)

    def _validate_geometry(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError('Bbox requires x2 > x1 and y2 > y1')

    @property
    def points(self) -> tuple[Point, ...]:
        return ((self.x1, self.y1), (self.x2, self.y1), (self.x2, self.y2), (self.x1, self.y2))

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def translate(self, dx: float, dy: float) -> Self:
        dx, dy = _finite_float(dx), _finite_float(dy)
        return replace(self, x1=self.x1 + dx, y1=self.y1 + dy, x2=self.x2 + dx, y2=self.y2 + dy)


@dataclass(frozen=True, slots=True, kw_only=True)
class _PointShape(Shape, ABC):
    points: PointSetInput = field()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'points', _normalize_points(self.points))
        Annotation.__post_init__(self)

    def translate(self, dx: float, dy: float) -> Self:
        dx, dy = _finite_float(dx), _finite_float(dy)
        return replace(self, points=tuple((x + dx, y + dy) for x, y in self.points))


@dataclass(frozen=True, slots=True, kw_only=True)
class Polygon(_PointShape):
    type: ClassVar[AnnotationType] = AnnotationType.POLYGON

    def _validate_geometry(self) -> None:
        if len(self.points) < 3:
            raise ValueError('Polygon requires at least three points')


@dataclass(frozen=True, slots=True, kw_only=True)
class Polyline(_PointShape):
    type: ClassVar[AnnotationType] = AnnotationType.POLYLINE

    def _validate_geometry(self) -> None:
        if len(self.points) < 2:
            raise ValueError('Polyline requires at least two points')


@dataclass(frozen=True, slots=True, kw_only=True)
class Points(_PointShape):
    type: ClassVar[AnnotationType] = AnnotationType.POINTS

    def _validate_geometry(self) -> None:
        if not self.points:
            raise ValueError('Points requires at least one point')


@dataclass(frozen=True, slots=True, kw_only=True)
class Circle(Shape):
    type: ClassVar[AnnotationType] = AnnotationType.CIRCLE
    center: Point
    edge: Point

    def __post_init__(self) -> None:
        object.__setattr__(self, 'center', _normalize_point(self.center))
        object.__setattr__(self, 'edge', _normalize_point(self.edge))
        Annotation.__post_init__(self)

    def _validate_geometry(self) -> None:
        if self.center == self.edge:
            raise ValueError('Circle center and edge must differ')

    @property
    def points(self) -> tuple[Point, ...]:
        return self.center, self.edge

    @property
    def radius(self) -> float:
        return math.dist(self.center, self.edge)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        cx, cy = self.center
        radius = self.radius
        return cx - radius, cy - radius, cx + radius, cy + radius

    def translate(self, dx: float, dy: float) -> Self:
        dx, dy = _finite_float(dx), _finite_float(dy)
        return replace(
            self, center=(self.center[0] + dx, self.center[1] + dy), edge=(self.edge[0] + dx, self.edge[1] + dy)
        )


@dataclass(frozen=True, slots=True)
class Keypoint:
    label: str
    x: float
    y: float
    visibility: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError('Keypoint label must be a non-empty string')
        object.__setattr__(self, 'x', _finite_float(self.x))
        object.__setattr__(self, 'y', _finite_float(self.y))
        if (
            isinstance(self.visibility, bool)
            or not isinstance(self.visibility, int)
            or self.visibility not in (0, 1, 2)
        ):
            raise ValueError('Keypoint visibility must be 0, 1, or 2')


@dataclass(frozen=True, slots=True, kw_only=True)
class Pose(Annotation):
    type: ClassVar[AnnotationType] = AnnotationType.POSE
    x1: float
    y1: float
    x2: float
    y2: float
    keypoints: Sequence[Keypoint]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'x1', _finite_float(self.x1))
        object.__setattr__(self, 'y1', _finite_float(self.y1))
        object.__setattr__(self, 'x2', _finite_float(self.x2))
        object.__setattr__(self, 'y2', _finite_float(self.y2))
        object.__setattr__(self, 'keypoints', tuple(self.keypoints))
        Annotation.__post_init__(self)

    def _validate_geometry(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError('Pose requires x2 > x1 and y2 > y1')
        if not self.keypoints or not all(isinstance(keypoint, Keypoint) for keypoint in self.keypoints):
            raise ValueError('Pose requires at least one Keypoint')
        if len({keypoint.label for keypoint in self.keypoints}) != len(self.keypoints):
            raise ValueError('Pose keypoint labels must be unique')

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def translate(self, dx: float, dy: float) -> Self:
        dx, dy = _finite_float(dx), _finite_float(dy)
        return replace(
            self,
            x1=self.x1 + dx,
            y1=self.y1 + dy,
            x2=self.x2 + dx,
            y2=self.y2 + dy,
            keypoints=tuple(
                Keypoint(label=keypoint.label, x=keypoint.x + dx, y=keypoint.y + dy, visibility=keypoint.visibility)
                for keypoint in self.keypoints
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageInfo:
    width: float
    height: float

    def __post_init__(self) -> None:
        dimensions = (self.width, self.height)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in dimensions):
            raise ValueError('ImageInfo width and height must be positive finite numbers')
        object.__setattr__(self, 'width', _finite_float(self.width))
        object.__setattr__(self, 'height', _finite_float(self.height))
        if self.width <= 0 or self.height <= 0:
            raise ValueError('ImageInfo width and height must be positive finite numbers')
