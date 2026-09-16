import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import UUID

from xxtrain.business_tasks.definition import TaskDefinition
from xxtrain.data import Bbox, Polygon, Polyline
from xxtrain.platform.contracts import AnnotationRecord, ImageRecord, JsonValue

_SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE images (
  id TEXT PRIMARY KEY,
  relative_path TEXT NOT NULL UNIQUE,
  width INTEGER NOT NULL CHECK(width > 0),
  height INTEGER NOT NULL CHECK(height > 0),
  perceptual_hash BLOB NOT NULL CHECK(length(perceptual_hash) = 8)
);
CREATE TABLE annotations (
  id TEXT PRIMARY KEY,
  image_id TEXT NOT NULL REFERENCES images(id),
  step_key TEXT NOT NULL,
  parent_id TEXT REFERENCES annotations(id),
  kind TEXT NOT NULL,
  label TEXT,
  geometry TEXT CHECK(geometry IS NULL OR json_valid(geometry))
);
CREATE INDEX annotations_by_image_step ON annotations(image_id, step_key);
CREATE INDEX annotations_by_parent ON annotations(parent_id);
CREATE TABLE cvat_annotation_map (
  job_id INTEGER NOT NULL,
  object_type TEXT NOT NULL,
  object_id INTEGER NOT NULL,
  annotation_id TEXT NOT NULL REFERENCES annotations(id) ON DELETE CASCADE,
  PRIMARY KEY(job_id, object_type, object_id),
  UNIQUE(job_id, object_type, annotation_id)
);
PRAGMA user_version=1;
"""


class AnnotationRepository:
    """Store validated image and annotation records through short SQLite connections."""

    def __init__(self, path: Path, task: TaskDefinition):
        self._path = path
        self._task = task
        self._initialize()

    def register_images(self, records: tuple[ImageRecord, ...]) -> None:
        """Register immutable image facts atomically, accepting exact repeat registrations."""
        for record in records:
            _validate_image(record)
        if len({record.id for record in records}) != len(records):
            raise ValueError('Image registration contains duplicate ids')
        if len({record.relative_path for record in records}) != len(records):
            raise ValueError('Image registration contains duplicate paths')

        with self._connection() as connection, connection:
            existing = {
                row[0]: ImageRecord(row[0], row[1], row[2], row[3], int.from_bytes(row[4], 'big'))
                for row in connection.execute('SELECT id, relative_path, width, height, perceptual_hash FROM images')
            }
            paths = {record.relative_path: record for record in existing.values()}
            for record in records:
                if record.id in existing:
                    if existing[record.id] != record:
                        raise ValueError(f'Image id {record.id!r} is already registered with different facts')
                    continue
                if record.relative_path in paths:
                    raise ValueError(f'Image path {record.relative_path!r} is already registered')
                connection.execute(
                    'INSERT INTO images(id, relative_path, width, height, perceptual_hash) VALUES (?, ?, ?, ?, ?)',
                    (
                        record.id,
                        record.relative_path,
                        record.width,
                        record.height,
                        record.perceptual_hash.to_bytes(8, 'big'),
                    ),
                )

    def images(self) -> tuple[ImageRecord, ...]:
        """Return registered images in insertion order."""
        with self._connection() as connection:
            return tuple(
                ImageRecord(row[0], row[1], row[2], row[3], int.from_bytes(row[4], 'big'))
                for row in connection.execute(
                    'SELECT id, relative_path, width, height, perceptual_hash FROM images ORDER BY rowid'
                )
            )

    def annotations(self, *, image_id: str | None = None, step_key: str | None = None) -> tuple[AnnotationRecord, ...]:
        """Return annotations in insertion order, optionally filtered by image and step."""
        clauses: list[str] = []
        parameters: list[str] = []
        if image_id is not None:
            clauses.append('image_id = ?')
            parameters.append(image_id)
        if step_key is not None:
            clauses.append('step_key = ?')
            parameters.append(step_key)
        where = f' WHERE {" AND ".join(clauses)}' if clauses else ''
        with self._connection() as connection:
            rows = connection.execute(
                'SELECT id, image_id, step_key, parent_id, kind, label, geometry '
                f'FROM annotations{where} ORDER BY rowid',
                parameters,
            )
            return tuple(_annotation_from_row(row) for row in rows)

    def save_annotations(
        self, records: tuple[AnnotationRecord, ...], *, delete_ids: frozenset[UUID] = frozenset()
    ) -> None:
        """Validate and atomically apply record upserts and deletions against the final set."""
        if len({record.id for record in records}) != len(records):
            raise ValueError('Annotation save contains duplicate ids')
        if any(not isinstance(annotation_id, UUID) for annotation_id in delete_ids):
            raise ValueError('Deleted annotation ids must be UUIDs')
        record_ids = {record.id for record in records}
        if record_ids & delete_ids:
            raise ValueError('An annotation cannot be saved and deleted together')

        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                image_ids = {row[0] for row in connection.execute('SELECT id FROM images')}
                existing = {
                    record.id: record
                    for record in (
                        _annotation_from_row(row)
                        for row in connection.execute(
                            'SELECT id, image_id, step_key, parent_id, kind, label, geometry FROM annotations'
                        )
                    )
                }
                final_records = dict(existing)
                for annotation_id in delete_ids:
                    final_records.pop(annotation_id, None)
                final_records.update((record.id, record) for record in records)
                encoded = _validate_annotations(tuple(final_records.values()), image_ids, self._task)

                connection.execute('PRAGMA defer_foreign_keys=ON')
                connection.executemany(
                    'DELETE FROM annotations WHERE id = ?', ((str(annotation_id),) for annotation_id in delete_ids)
                )
                connection.executemany(
                    """
                    INSERT INTO annotations(id, image_id, step_key, parent_id, kind, label, geometry)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                      image_id=excluded.image_id,
                      step_key=excluded.step_key,
                      parent_id=excluded.parent_id,
                      kind=excluded.kind,
                      label=excluded.label,
                      geometry=excluded.geometry
                    """,
                    (
                        (
                            str(record.id),
                            record.image_id,
                            record.step_key,
                            str(record.parent_id) if record.parent_id is not None else None,
                            record.kind,
                            record.label,
                            encoded[record.id],
                        )
                        for record in records
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _initialize(self) -> None:
        with self._connection() as connection:
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            if version == 0:
                connection.executescript(_SCHEMA)
            elif version != _SCHEMA_VERSION:
                raise ValueError(f'Unsupported annotation database schema version: {version}')

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path)
        try:
            connection.execute('PRAGMA foreign_keys=ON')
            yield connection
        finally:
            connection.close()


def _validate_image(record: ImageRecord) -> None:
    if not isinstance(record.id, str) or len(record.id) != 64 or any(c not in '0123456789abcdef' for c in record.id):
        raise ValueError('Image id must be a lowercase SHA-256 hexadecimal digest')
    if not isinstance(record.relative_path, str) or not record.relative_path:
        raise ValueError('Image relative path must be a non-empty string')
    posix_path = PurePosixPath(record.relative_path.replace('\\', '/'))
    windows_path = PureWindowsPath(record.relative_path)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive or '..' in posix_path.parts:
        raise ValueError('Image path must be relative and must not traverse its workspace')
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (record.width, record.height)
    ):
        raise ValueError('Image width and height must be positive integers')
    if (
        isinstance(record.perceptual_hash, bool)
        or not isinstance(record.perceptual_hash, int)
        or not 0 <= record.perceptual_hash < 1 << 64
    ):
        raise ValueError('Image perceptual hash must be an unsigned 64-bit integer')


def _annotation_from_row(row: tuple[object, ...]) -> AnnotationRecord:
    geometry = json.loads(row[6]) if row[6] is not None else None
    return AnnotationRecord(
        UUID(str(row[0])),
        str(row[1]),
        str(row[2]),
        UUID(str(row[3])) if row[3] is not None else None,
        str(row[4]),
        str(row[5]) if row[5] is not None else None,
        geometry,
    )


def _validate_annotations(
    records: tuple[AnnotationRecord, ...], image_ids: set[str], task: TaskDefinition
) -> dict[UUID, str | None]:
    by_id = {record.id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError('Annotation ids must be unique')
    encoded: dict[UUID, str | None] = {}
    negative_groups: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record.id, UUID):
            raise ValueError('Annotation id must be a UUID')
        if record.image_id not in image_ids:
            raise ValueError(f'Annotation references unknown image: {record.image_id!r}')
        step = task.step(record.step_key)
        if record.kind not in step.kinds:
            raise ValueError(f'Annotation kind {record.kind!r} is not allowed for step {record.step_key!r}')
        encoded[record.id] = _validate_content(record, step.labels)
        group = (record.image_id, record.step_key)
        if record.kind == 'negative':
            if group in negative_groups:
                raise ValueError('An image step cannot contain multiple negative annotations')
            negative_groups.add(group)

        if step.parent_steps:
            if record.parent_id is None:
                raise ValueError(f'Annotation step {record.step_key!r} requires a parent')
            parent = by_id.get(record.parent_id)
            if parent is None:
                raise ValueError(f'Annotation references unknown parent: {record.parent_id}')
            if parent.image_id != record.image_id:
                raise ValueError('Annotation parent must belong to the same image')
            if parent.step_key not in step.parent_steps:
                raise ValueError(
                    f'Annotation step {record.step_key!r} cannot use a parent from step {parent.step_key!r}'
                )
        elif record.parent_id is not None:
            raise ValueError(f'Annotation step {record.step_key!r} does not allow a parent')

    for group in negative_groups:
        if sum((record.image_id, record.step_key) == group for record in records) != 1:
            raise ValueError('A negative annotation conflicts with other annotations for the same image step')
    _validate_parent_cycles(by_id)
    return encoded


def _validate_content(record: AnnotationRecord, labels: frozenset[str]) -> str | None:
    try:
        encoded = (
            json.dumps(record.geometry, allow_nan=False, separators=(',', ':')) if record.geometry is not None else None
        )
    except (TypeError, ValueError) as error:
        raise ValueError('Annotation geometry must be valid finite JSON') from error

    if record.kind == 'negative':
        if record.label is not None or record.geometry is not None:
            raise ValueError('Negative annotations require null label and geometry')
        return encoded
    if not isinstance(record.label, str) or record.label not in labels:
        raise ValueError(f'Annotation label {record.label!r} is not allowed for step {record.step_key!r}')
    if record.kind == 'classification':
        if record.geometry is not None:
            raise ValueError('Classification annotations require null geometry')
    elif record.kind == 'rectangle':
        points = _points(record.geometry, count=2)
        Bbox(label=record.label, x1=points[0][0], y1=points[0][1], x2=points[1][0], y2=points[1][1])
    elif record.kind == 'polygon':
        Polygon(label=record.label, points=_points(record.geometry, minimum=3))
    elif record.kind == 'polyline':
        Polyline(label=record.label, points=_points(record.geometry, minimum=2))
    else:
        raise ValueError(f'Unsupported annotation kind: {record.kind!r}')
    return encoded


def _points(geometry: JsonValue, *, count: int | None = None, minimum: int | None = None) -> list[list[float]]:
    if not isinstance(geometry, list):
        raise ValueError('Shape geometry must be a JSON array of points')
    if count is not None and len(geometry) != count:
        raise ValueError(f'Shape geometry requires exactly {count} points')
    if minimum is not None and len(geometry) < minimum:
        raise ValueError(f'Shape geometry requires at least {minimum} points')
    points: list[list[float]] = []
    for point in geometry:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError('Each shape point requires exactly two coordinates')
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point):
            raise ValueError('Shape coordinates must be numbers')
        points.append([float(point[0]), float(point[1])])
    return points


def _validate_parent_cycles(records: dict[UUID, AnnotationRecord]) -> None:
    for start in records:
        seen: set[UUID] = set()
        current: UUID | None = start
        while current is not None:
            if current in seen:
                raise ValueError('Annotation parent relationships must not contain a cycle')
            seen.add(current)
            record = records.get(current)
            current = record.parent_id if record is not None else None
