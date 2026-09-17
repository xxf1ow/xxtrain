import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import UUID

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
  clearml_task_id TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS training_runs_by_user ON training_runs(user_id);
CREATE INDEX IF NOT EXISTS training_runs_by_workspace ON training_runs(workspace_id);
"""


class TrainingRunStore:
    """Persist immutable training associations outside workspace annotation storage.

    ``get`` enforces user ownership. Creation accepts an exact retry only; task binding cannot change a recorded
    ClearML task identity. The creation-attempt timestamp is diagnostic data and does not represent execution state.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._initialize()

    def create(self, run: TrainingRun) -> TrainingRun:
        """Store one run or return the existing run when every durable fact is identical."""
        _validate_run(run)
        with self._connection() as connection, connection:
            existing = _find_run(connection, run.id)
            if existing is not None:
                if existing != run:
                    raise ValueError(f'Training run {run.id!r} is already recorded with different facts')
                return existing
            connection.execute(
                """INSERT INTO training_runs(
                id, user_id, workspace_id, workspace_name, target, fingerprint, cache_relative_path, submitted_at,
                create_attempted_at, clearml_task_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                _run_values(run),
            )
        return run

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
                    'UPDATE training_runs SET create_attempted_at = ? WHERE id = ?', (attempted_at, run_id)
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
        with self._connection() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path)
        try:
            yield connection
        finally:
            connection.close()


def _find_run(connection: sqlite3.Connection, run_id: str) -> TrainingRun | None:
    row = connection.execute('SELECT * FROM training_runs WHERE id = ?', (run_id,)).fetchone()
    return None if row is None else _run_from_row(row)


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
