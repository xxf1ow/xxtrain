import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import UUID

from xxtrain.business_tasks.definition import TaskDefinition
from xxtrain.data import Bbox, Polygon, Polyline
from xxtrain.platform.contracts import (
    AnnotationChanges,
    AnnotationRecord,
    CvatBinding,
    DetectionSummary,
    ImageRecord,
    JobRef,
    JsonValue,
    PlatformError,
    PreparedJob,
)

from .changes import plan_changes

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
    """Store validated image, annotation, and CVAT identity records through short SQLite connections.

    Caller-supplied records that violate content, parent, identity, or job-scope rules raise ``ValueError``.
    SQLite operational failures raise ``PlatformError`` without exposing SQL or database paths and retain the
    original ``sqlite3.Error`` as their exception cause.
    """

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

    def detection_summary(self) -> DetectionSummary:
        """Aggregate image-level detection completion and boxed-image counts."""
        try:
            with self._connection() as connection:
                row = connection.execute(
                    """
                    SELECT COUNT(*),
                           COALESCE(SUM(CASE WHEN EXISTS (
                               SELECT 1 FROM annotations
                               WHERE annotations.image_id = images.id
                                 AND annotations.step_key = 'detect'
                                 AND annotations.kind IN ('rectangle', 'negative')
                           ) THEN 1 ELSE 0 END), 0),
                           COALESCE(SUM(CASE WHEN EXISTS (
                               SELECT 1 FROM annotations
                               WHERE annotations.image_id = images.id
                                 AND annotations.step_key = 'detect'
                                 AND annotations.kind = 'rectangle'
                           ) THEN 1 ELSE 0 END), 0)
                    FROM images
                    """
                ).fetchone()
        except sqlite3.Error as error:
            raise PlatformError('Annotation storage operation failed') from error
        return DetectionSummary(int(row[0]), int(row[1]), int(row[2]))

    def save_annotations(
        self, records: tuple[AnnotationRecord, ...], *, delete_ids: frozenset[UUID] = frozenset()
    ) -> None:
        """Plan dependent deletion, then validate and atomically apply the resulting final record set.

        Invalid records, reassigned IDs, and conflicting upsert/delete requests raise ``ValueError``.
        """
        changes = plan_changes(self.annotations(), records, delete_ids, self._task)
        self.apply_changes(changes)

    def bindings(self, ref: JobRef) -> tuple[CvatBinding, ...]:
        """Return current bindings for ``ref`` or raise ``ValueError`` when its sample scope is inconsistent."""
        _validate_ref(ref)
        try:
            with self._connection() as connection:
                image_ids = {str(row[0]) for row in connection.execute('SELECT id FROM images')}
                _validate_ref_samples(ref, image_ids)
                rows = tuple(
                    connection.execute(
                        """
                        SELECT annotations.image_id, cvat_annotation_map.object_type,
                               cvat_annotation_map.object_id, cvat_annotation_map.annotation_id
                        FROM cvat_annotation_map
                        JOIN annotations ON annotations.id = cvat_annotation_map.annotation_id
                        WHERE cvat_annotation_map.job_id = ?
                        ORDER BY cvat_annotation_map.rowid
                        """,
                        (ref.job_id,),
                    )
                )
        except sqlite3.Error as error:
            raise PlatformError('Annotation storage operation failed') from error
        sample_ids = frozenset(ref.sample_ids)
        if any(row[0] not in sample_ids for row in rows):
            raise ValueError('Job bindings fall outside the referenced sample scope')
        return tuple(CvatBinding(str(row[0]), str(row[1]), int(row[2]), UUID(str(row[3]))) for row in rows)

    def bind_job(self, prepared: PreparedJob) -> None:
        """Atomically add a prepared job's bindings; exact repeated bindings are idempotent.

        Unknown samples or annotations, cross-image bindings, duplicate CVAT identities, and attempts to
        rebind either identity raise ``ValueError``.
        """
        self.apply_changes(
            AnnotationChanges((), frozenset(), frozenset()), ref=prepared.ref, bindings=prepared.bindings
        )

    def apply_changes(
        self, changes: AnnotationChanges, *, ref: JobRef | None = None, bindings: tuple[CvatBinding, ...] = ()
    ) -> None:
        """Atomically delete, upsert, and bind one planned change set against the final records.

        Invalid records, reassigned IDs, malformed changes, and binding scope or identity conflicts raise
        ``ValueError``. A failed SQLite statement rolls back annotations and bindings together and raises a
        sanitized ``PlatformError`` with the original database exception chained as its cause.
        """
        _validate_change_shape(changes)
        if bindings and ref is None:
            raise ValueError('CVAT bindings require a job reference')
        if ref is not None:
            _validate_ref(ref)
            _validate_binding_shape(bindings)

        try:
            with self._connection() as connection:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    image_ids = {row[0] for row in connection.execute('SELECT id FROM images')}
                    if ref is not None:
                        _validate_ref_samples(ref, image_ids)
                    existing = {
                        record.id: record
                        for record in (
                            _annotation_from_row(row)
                            for row in connection.execute(
                                'SELECT id, image_id, step_key, parent_id, kind, label, geometry FROM annotations'
                            )
                        )
                    }
                    for record in changes.upserts:
                        previous = existing.get(record.id)
                        if previous is not None and _assignment(previous) != _assignment(record):
                            raise ValueError('Existing annotation ids cannot change image, step, or parent assignment')
                    final_records = dict(existing)
                    for annotation_id in changes.delete_ids:
                        final_records.pop(annotation_id, None)
                    final_records.update((record.id, record) for record in changes.upserts)
                    encoded = _validate_annotations(tuple(final_records.values()), image_ids, self._task)

                    connection.execute('PRAGMA defer_foreign_keys=ON')
                    connection.executemany(
                        'DELETE FROM cvat_annotation_map WHERE annotation_id = ?',
                        ((str(annotation_id),) for annotation_id in changes.delete_ids),
                    )
                    connection.executemany(
                        'DELETE FROM annotations WHERE id = ?',
                        ((str(annotation_id),) for annotation_id in _delete_order(changes.delete_ids, existing)),
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
                            for record in changes.upserts
                        ),
                    )
                    if ref is not None:
                        _validate_bindings(connection, ref, bindings, final_records)
                        connection.executemany(
                            """
                            INSERT OR IGNORE INTO cvat_annotation_map(job_id, object_type, object_id, annotation_id)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                (ref.job_id, binding.object_type, binding.object_id, str(binding.annotation_id))
                                for binding in bindings
                            ),
                        )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
        except sqlite3.Error as error:
            raise PlatformError('Annotation storage operation failed') from error

    def _initialize(self) -> None:
        with self._connection() as connection:
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            if version == 0:
                connection.executescript(_SCHEMA)
            elif version != _SCHEMA_VERSION:
                raise ValueError(f'Unsupported annotation database schema version: {version}')

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self._path)
            connection.execute('PRAGMA foreign_keys=ON')
            yield connection
        except sqlite3.Error as error:
            raise PlatformError('Annotation storage operation failed') from error
        finally:
            if connection is not None:
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


def _validate_change_shape(changes: AnnotationChanges) -> None:
    if not isinstance(changes, AnnotationChanges):
        raise ValueError('Annotation changes must use AnnotationChanges')
    upsert_ids = [record.id for record in changes.upserts]
    if len(set(upsert_ids)) != len(upsert_ids):
        raise ValueError('Annotation changes contain duplicate upsert ids')
    if any(not isinstance(annotation_id, UUID) for annotation_id in (*upsert_ids, *changes.delete_ids)):
        raise ValueError('Annotation change ids must be UUIDs')
    if set(upsert_ids) & changes.delete_ids:
        raise ValueError('An annotation cannot be saved and deleted together')
    if any(not isinstance(step, str) or not step for step in changes.invalidated_steps):
        raise ValueError('Invalidated annotation steps must be non-empty strings')


def _validate_ref(ref: JobRef) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (ref.task_id, ref.job_id)):
        raise ValueError('CVAT task and job ids must be non-negative integers')
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ref.sample_ids):
        raise ValueError('Job samples must be non-empty strings')
    if len(set(ref.sample_ids)) != len(ref.sample_ids):
        raise ValueError('Job sample ids must be unique')


def _validate_ref_samples(ref: JobRef, image_ids: set[str]) -> None:
    if not set(ref.sample_ids) <= image_ids:
        raise ValueError('Job references an unknown sample')


def _validate_binding_shape(bindings: tuple[CvatBinding, ...]) -> None:
    keys: set[tuple[str, int]] = set()
    annotation_ids: set[UUID] = set()
    for binding in bindings:
        if not isinstance(binding.sample_id, str) or not binding.sample_id:
            raise ValueError('CVAT binding sample ids must be non-empty strings')
        if not isinstance(binding.object_type, str) or not binding.object_type:
            raise ValueError('CVAT binding object types must be non-empty strings')
        if isinstance(binding.object_id, bool) or not isinstance(binding.object_id, int) or binding.object_id < 0:
            raise ValueError('CVAT binding object ids must be non-negative integers')
        if not isinstance(binding.annotation_id, UUID):
            raise ValueError('CVAT binding annotation ids must be UUIDs')
        key = (binding.object_type, binding.object_id)
        if key in keys:
            raise ValueError('A CVAT object identity can appear only once in a prepared job')
        if binding.annotation_id in annotation_ids:
            raise ValueError('An annotation can appear only once in a prepared job')
        keys.add(key)
        annotation_ids.add(binding.annotation_id)


def _validate_bindings(
    connection: sqlite3.Connection,
    ref: JobRef,
    bindings: tuple[CvatBinding, ...],
    records: dict[UUID, AnnotationRecord],
) -> None:
    sample_ids = frozenset(ref.sample_ids)
    existing_rows = tuple(
        connection.execute(
            'SELECT object_type, object_id, annotation_id FROM cvat_annotation_map WHERE job_id = ?', (ref.job_id,)
        )
    )
    existing_by_key = {(str(row[0]), int(row[1])): UUID(str(row[2])) for row in existing_rows}
    existing_by_annotation: dict[UUID, tuple[str, int]] = {}
    for row in existing_rows:
        annotation_id = UUID(str(row[2]))
        key = (str(row[0]), int(row[1]))
        previous = existing_by_annotation.setdefault(annotation_id, key)
        if previous != key:
            raise ValueError('A job annotation already has multiple CVAT bindings')

    for binding in bindings:
        if binding.sample_id not in sample_ids:
            raise ValueError('CVAT binding references a sample outside the job')
        record = records.get(binding.annotation_id)
        if record is None:
            raise ValueError('CVAT binding references an unknown annotation')
        if record.image_id != binding.sample_id:
            raise ValueError('CVAT binding annotation must belong to its sample image')
        key = (binding.object_type, binding.object_id)
        bound_annotation = existing_by_key.get(key)
        if bound_annotation is not None and bound_annotation != binding.annotation_id:
            raise ValueError('CVAT object identity is already bound to another annotation')
        bound_key = existing_by_annotation.get(binding.annotation_id)
        if bound_key is not None and bound_key != key:
            raise ValueError('Annotation is already bound to another CVAT object in this job')


def _assignment(record: AnnotationRecord) -> tuple[str, str, UUID | None]:
    return record.image_id, record.step_key, record.parent_id


def _delete_order(delete_ids: frozenset[UUID], existing: dict[UUID, AnnotationRecord]) -> tuple[UUID, ...]:
    def depth(annotation_id: UUID) -> int:
        result = 0
        current = existing.get(annotation_id)
        seen: set[UUID] = set()
        while current is not None and current.parent_id in delete_ids and current.parent_id not in seen:
            seen.add(current.parent_id)
            result += 1
            current = existing.get(current.parent_id)
        return result

    return tuple(sorted(delete_ids, key=lambda annotation_id: (depth(annotation_id), str(annotation_id)), reverse=True))


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
