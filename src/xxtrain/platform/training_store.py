import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import UUID

from xxtrain.platform.contracts import PlatformConflictError, PlatformError
from xxtrain.platform.training_contracts import TrainingRun

_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_runs (
  id TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL,
  workspace_id TEXT NOT NULL,
  workspace_name TEXT NOT NULL,
  target TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  cache_relative_path TEXT NOT NULL,
  submitted_at TEXT NOT NULL,
  create_attempted_at TEXT,
  clearml_task_id TEXT UNIQUE,
  desired_action TEXT CHECK (desired_action IN ('execute', 'cancel')),
  task_entry TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS training_runs_by_user ON training_runs(user_id);
CREATE INDEX IF NOT EXISTS training_runs_by_workspace ON training_runs(workspace_id);
"""

_INPUT_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS training_runs_by_input
ON training_runs(user_id, workspace_id, target, fingerprint)
"""

_ALIAS_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_input_aliases (
  user_id INTEGER NOT NULL,
  workspace_id TEXT NOT NULL,
  target TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  run_id TEXT NOT NULL REFERENCES training_runs(id),
  PRIMARY KEY (user_id, workspace_id, target, fingerprint)
)
"""


class TrainingRunStore:
    """Persist immutable training associations outside workspace annotation storage.

    ``get`` enforces user ownership. Creation returns the canonical row for the same input; task binding cannot
    change a recorded ClearML task identity. The creation-attempt timestamp is diagnostic data and does not represent
    execution state.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._initialize()

    def create(self, run: TrainingRun) -> TrainingRun:
        """Store one run or return the canonical row already associated with its input."""
        _validate_run(run)
        with self._connection() as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            existing = _find_run(connection, run.id)
            if existing is not None:
                if _immutable_values(existing) != _immutable_values(run):
                    raise ValueError(f'Training run {run.id!r} is already recorded with different facts')
                return existing
            found = _find_input(connection, run.user_id, run.workspace_id, run.target, run.fingerprint)
            if found is not None:
                existing, aliased = found
                if not aliased and existing.cache_relative_path != run.cache_relative_path:
                    raise ValueError('Training input is already recorded with a different cache association')
                return existing
            connection.execute(
                """INSERT INTO training_runs(
                id, user_id, workspace_id, workspace_name, target, fingerprint, cache_relative_path, submitted_at,
                create_attempted_at, clearml_task_id, desired_action, task_entry
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                _run_values(run),
            )
        return run

    def associate_input(self, user_id: int, workspace_id: str, target: str, fingerprint: str, run_id: str) -> None:
        """Associate an additional exact input identity with one immutable owned run."""
        _validate_user_id(user_id)
        _validate_run_id(run_id)
        for value, name in (
            (workspace_id, 'Workspace id'),
            (target, 'Training target'),
            (fingerprint, 'Training fingerprint'),
        ):
            _validate_non_empty_string(value, name)
        with self._connection() as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            run = _find_run(connection, run_id)
            if run is None or (run.user_id, run.workspace_id, run.target) != (user_id, workspace_id, target):
                raise ValueError('Training input association does not match the run owner, workspace, and target')
            found = _find_input(connection, user_id, workspace_id, target, fingerprint)
            if found is not None:
                existing, _ = found
                if existing.id != run_id:
                    raise PlatformConflictError('Training input is already associated with another run')
                return
            connection.execute(
                """INSERT INTO training_input_aliases(user_id, workspace_id, target, fingerprint, run_id)
                VALUES (?, ?, ?, ?, ?)""",
                (user_id, workspace_id, target, fingerprint, run_id),
            )

    def request_cancel(self, user_id: int, run_id: str) -> TrainingRun:
        """Persist cancellation for an owned run and return its current row."""
        _validate_user_id(user_id)
        _validate_run_id(run_id)
        with self._connection() as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            run = _find_run(connection, run_id)
            if run is None or run.user_id != user_id:
                raise ValueError('Training run access denied')
            connection.execute(
                "UPDATE training_runs SET desired_action = 'cancel' WHERE id = ? AND user_id = ?", (run_id, user_id)
            )
            cancelled = _find_run(connection, run_id)
            assert cancelled is not None
            return cancelled

    def find_input(self, user_id: int, workspace_id: str, target: str, fingerprint: str) -> TrainingRun | None:
        """Return the run associated with one user's exact training input, if present."""
        _validate_user_id(user_id)
        for value, name in (
            (workspace_id, 'Workspace id'),
            (target, 'Training target'),
            (fingerprint, 'Training fingerprint'),
        ):
            _validate_non_empty_string(value, name)
        with self._connection() as connection:
            found = _find_input(connection, user_id, workspace_id, target, fingerprint)
            return None if found is None else found[0]

    def get(self, user_id: int, run_id: str) -> TrainingRun:
        """Return one run only when it belongs to ``user_id``."""
        _validate_user_id(user_id)
        _validate_run_id(run_id)
        with self._connection() as connection:
            run = _find_run(connection, run_id)
        if run is None or run.user_id != user_id:
            raise ValueError('Training run access denied')
        return run

    def list_user(self, user_id: int) -> tuple[TrainingRun, ...]:
        """Return one user's historical training associations in submission order."""
        _validate_user_id(user_id)
        with self._connection() as connection:
            return tuple(
                _run_from_row(row)
                for row in connection.execute(
                    'SELECT * FROM training_runs WHERE user_id = ? ORDER BY rowid', (user_id,)
                )
            )

    def list_workspace(self, workspace_id: str) -> tuple[TrainingRun, ...]:
        """Return a workspace's historical training associations in submission order."""
        _validate_non_empty_string(workspace_id, 'Workspace id')
        with self._connection() as connection:
            return tuple(
                _run_from_row(row)
                for row in connection.execute(
                    'SELECT * FROM training_runs WHERE workspace_id = ? ORDER BY rowid', (workspace_id,)
                )
            )

    def mark_create_attempted(self, run_id: str, attempted_at: str) -> None:
        """Record when a ClearML task creation request was attempted for cross-process diagnosis."""
        _validate_run_id(run_id)
        _validate_non_empty_string(attempted_at, 'Creation attempt timestamp')
        with self._connection() as connection, connection:
            if (
                connection.execute(
                    'UPDATE training_runs SET create_attempted_at = COALESCE(create_attempted_at, ?) WHERE id = ?',
                    (attempted_at, run_id),
                ).rowcount
                != 1
            ):
                raise ValueError(f'Unknown training run {run_id!r}')

    def bind_task(self, run_id: str, task_id: str) -> None:
        """Bind a run to one ClearML task, accepting only an exact retry of that binding."""
        _validate_run_id(run_id)
        _validate_non_empty_string(task_id, 'ClearML task id')
        with self._connection() as connection, connection:
            run = _find_run(connection, run_id)
            if run is None:
                raise ValueError(f'Unknown training run {run_id!r}')
            if run.clearml_task_id is not None:
                if run.clearml_task_id != task_id:
                    raise ValueError(f'Training run {run_id!r} is already bound to a different task')
                return
            owner = connection.execute('SELECT id FROM training_runs WHERE clearml_task_id = ?', (task_id,)).fetchone()
            if owner is not None:
                raise ValueError(f'ClearML task {task_id!r} is already bound to another training run')
            connection.execute('UPDATE training_runs SET clearml_task_id = ? WHERE id = ?', (task_id, run_id))

    def _initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection, connection:
            connection.executescript(_SCHEMA)
            connection.execute('BEGIN IMMEDIATE')
            columns = {row[1] for row in connection.execute('PRAGMA table_info(training_runs)')}
            if 'desired_action' not in columns:
                connection.execute(
                    'ALTER TABLE training_runs ADD COLUMN desired_action TEXT '
                    "CHECK (desired_action IN ('execute', 'cancel'))"
                )
            if 'task_entry' not in columns:
                connection.execute("ALTER TABLE training_runs ADD COLUMN task_entry TEXT NOT NULL DEFAULT 'point'")
            duplicate = connection.execute(
                """SELECT user_id, workspace_id, target, fingerprint, COUNT(*)
                FROM training_runs
                GROUP BY user_id, workspace_id, target, fingerprint
                HAVING COUNT(*) > 1
                LIMIT 1"""
            ).fetchone()
            if duplicate is not None:
                user_id, workspace_id, target, fingerprint, count = duplicate
                raise PlatformConflictError(
                    'Training metadata contains duplicate input runs; resolve the records without deleting task '
                    f'associations before startup (user_id={user_id}, workspace_id={workspace_id!r}, '
                    f'target={target!r}, fingerprint={fingerprint!r}, rows={count})'
                )
            connection.execute(_INPUT_INDEX)
            connection.execute(_ALIAS_SCHEMA)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = None
        try:
            connection = sqlite3.connect(self._path)
            connection.execute('PRAGMA foreign_keys = ON')
            yield connection
        except sqlite3.Error as error:
            error_code = getattr(error, 'sqlite_errorcode', 0) & 0xFF
            if error_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                raise PlatformConflictError('Training metadata is busy') from error
            raise PlatformError('Training metadata is unavailable') from error
        finally:
            if connection is not None:
                connection.close()


def _find_run(connection: sqlite3.Connection, run_id: str) -> TrainingRun | None:
    row = connection.execute('SELECT * FROM training_runs WHERE id = ?', (run_id,)).fetchone()
    return None if row is None else _run_from_row(row)


def _find_input(
    connection: sqlite3.Connection, user_id: int, workspace_id: str, target: str, fingerprint: str
) -> tuple[TrainingRun, bool] | None:
    canonical = connection.execute(
        """SELECT * FROM training_runs
        WHERE user_id = ? AND workspace_id = ? AND target = ? AND fingerprint = ?""",
        (user_id, workspace_id, target, fingerprint),
    ).fetchone()
    alias = connection.execute(
        """SELECT training_runs.* FROM training_input_aliases
        JOIN training_runs ON training_runs.id = training_input_aliases.run_id
        WHERE training_input_aliases.user_id = ?
          AND training_input_aliases.workspace_id = ?
          AND training_input_aliases.target = ?
          AND training_input_aliases.fingerprint = ?""",
        (user_id, workspace_id, target, fingerprint),
    ).fetchone()
    if canonical is not None and alias is not None and canonical[0] != alias[0]:
        raise PlatformConflictError('Training input has conflicting canonical and compatibility associations')
    if canonical is not None:
        return _run_from_row(canonical), False
    if alias is not None:
        return _run_from_row(alias), True
    return None


def _run_from_row(row: tuple[object, ...]) -> TrainingRun:
    return TrainingRun(*row)  # type: ignore[arg-type]


def _run_values(run: TrainingRun) -> tuple[object, ...]:
    return (
        run.id,
        run.user_id,
        run.workspace_id,
        run.workspace_name,
        run.target,
        run.fingerprint,
        run.cache_relative_path,
        run.submitted_at,
        run.create_attempted_at,
        run.clearml_task_id,
        run.desired_action,
        run.task_entry,
    )


def _immutable_values(run: TrainingRun) -> tuple[object, ...]:
    return (
        run.id,
        run.user_id,
        run.workspace_id,
        run.workspace_name,
        run.target,
        run.fingerprint,
        run.cache_relative_path,
        run.submitted_at,
        run.task_entry,
    )


def _validate_run(run: TrainingRun) -> None:
    if not isinstance(run, TrainingRun):
        raise ValueError('Training run must use TrainingRun')
    _validate_run_id(run.id)
    _validate_user_id(run.user_id)
    for value, name in (
        (run.workspace_id, 'Workspace id'),
        (run.workspace_name, 'Workspace name'),
        (run.target, 'Training target'),
        (run.fingerprint, 'Training fingerprint'),
        (run.submitted_at, 'Submission timestamp'),
    ):
        _validate_non_empty_string(value, name)
    _validate_relative_path(run.cache_relative_path)
    for value, name in (
        (run.create_attempted_at, 'Creation attempt timestamp'),
        (run.clearml_task_id, 'ClearML task id'),
    ):
        if value is not None:
            _validate_non_empty_string(value, name)
    if run.desired_action not in {None, 'execute', 'cancel'}:
        raise ValueError("Training intent must be 'execute', 'cancel', or None")
    _validate_non_empty_string(run.task_entry, 'Training task entry')


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str):
        raise ValueError('Training run id must be a UUID')
    try:
        valid = str(UUID(run_id)) == run_id
    except ValueError:
        valid = False
    if not valid:
        raise ValueError('Training run id must be a canonical UUID')


def _validate_user_id(user_id: int) -> None:
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise ValueError('Training user id must be a positive integer')


def _validate_relative_path(value: str) -> None:
    _validate_non_empty_string(value, 'Training cache path')
    posix_path = PurePosixPath(value.replace('\\', '/'))
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive or '..' in posix_path.parts:
        raise ValueError('Training cache path must be relative and must not traverse metadata storage')


def _validate_non_empty_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{name} must be a non-empty string')
