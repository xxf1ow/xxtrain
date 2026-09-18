from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import PlatformAccessError, PlatformError
from xxtrain.platform.service import AnnotationService
from xxtrain.platform.training_contracts import DownloadFile, ExecutionView, TrainingRun, TrainingRunView
from xxtrain.platform.training_store import TrainingRunStore


class TrainingService:
    """Coordinate durable run associations with current annotation and ClearML facts."""

    def __init__(self, config, annotations, store, clearml, shared_root) -> None:
        self.config: WorkspaceConfig = config
        self.annotations: AnnotationService = annotations
        self.store: TrainingRunStore = store
        self.clearml = clearml
        self.shared_root = Path(shared_root).resolve()

    def submit(self, user_id: int, target: str, request_id: str) -> TrainingRunView:
        """Prepare and enqueue one current input, idempotently keyed by ``request_id``."""
        _validate_request_id(request_id)
        with self.annotations.mutation(user_id):
            existing = self._find_user_run(user_id, request_id)
            if existing is not None:
                if existing.target != target or existing.workspace_id != self.config.workspace_id:
                    raise PlatformError('Training request conflicts with its recorded input')
                return self._resume_submission(existing)

            fingerprint = (
                self.annotations.data.detection_fingerprint()
                if target == 'detect'
                else self.annotations.data.target_fingerprint(target)
            )
            workspace_runs = self.store.list_workspace(self.config.workspace_id)
            for run in reversed(workspace_runs):
                if run.target == target and run.fingerprint == fingerprint and self._view(run).execution.active:
                    return self._view(run)
            referenced = next(
                (run for run in workspace_runs if run.target == target and run.fingerprint == fingerprint), None
            )
            if referenced is not None:
                try:
                    cache_path = self.annotations.runtime.cache_path(target, fingerprint)
                except ValueError as error:
                    raise PlatformError('Referenced training cache is unavailable and cannot be replaced') from error
            else:
                cache_path = self.annotations.ensure_target_cache(user_id, target)
            run = TrainingRun(
                request_id,
                user_id,
                self.config.workspace_id,
                self.config.display_name,
                target,
                fingerprint,
                self._relative_cache(cache_path),
                _now(),
                None,
                None,
            )
            self.store.create(run)
            return self._resume_submission(run)

    def list_runs(self, user_id: int) -> tuple[TrainingRunView, ...]:
        return tuple(self._view(run) for run in self.store.list_user(user_id))

    def get_run(self, user_id: int, run_id: str) -> TrainingRunView:
        return self._view(self._get(user_id, run_id))

    def cancel(self, user_id: int, run_id: str) -> TrainingRunView:
        run = self._get(user_id, run_id)
        if run.clearml_task_id is None:
            return self._view(run)
        try:
            execution = self.clearml.cancel(run.clearml_task_id)
        except Exception:
            execution = _unknown(run.clearml_task_id, 'Training cancellation could not be confirmed')
        return TrainingRunView(run, execution)

    def retry(self, user_id: int, run_id: str, request_id: str) -> TrainingRunView:
        """Enqueue a new run against the retained immutable input of a historical run."""
        original = self._get(user_id, run_id)
        _validate_request_id(request_id)
        with self.annotations.mutation(user_id):
            existing = self._find_user_run(user_id, request_id)
            if existing is not None:
                if (existing.target, existing.fingerprint, existing.cache_relative_path) != (
                    original.target,
                    original.fingerprint,
                    original.cache_relative_path,
                ):
                    raise PlatformError('Training request conflicts with its recorded input')
                return self._resume_submission(existing)
            self._retained_cache(original)
            run = TrainingRun(
                request_id,
                user_id,
                original.workspace_id,
                original.workspace_name,
                original.target,
                original.fingerprint,
                original.cache_relative_path,
                _now(),
                None,
                None,
            )
            self.store.create(run)
            return self._resume_submission(run)

    def download(self, user_id: int, run_id: str) -> DownloadFile:
        run = self._get(user_id, run_id)
        view = self._view(run)
        if view.execution is None or not view.execution.download_ready or run.clearml_task_id is None:
            raise PlatformError('Training result is not ready for download')
        try:
            return self.clearml.download(run.clearml_task_id, self.config.runtime_dir / 'downloads' / run.id)
        except Exception as error:
            raise PlatformError('Training result download failed') from error

    def require_editable(self, workspace_id: str) -> None:
        if workspace_id != self.config.workspace_id:
            raise PlatformAccessError('Workspace access denied')
        for run in self.store.list_workspace(workspace_id):
            if run.create_attempted_at is not None and self._view(run).execution.active:
                raise PlatformError('Workspace editing is disabled while training is active')

    def workspace_view(self, user_id: int) -> dict[str, object]:
        workspace = self.annotations.view(user_id)
        runs = self.list_runs(user_id)
        editable = True
        try:
            self.require_editable(workspace.workspace_id)
        except PlatformError:
            editable = False
        latest = {}
        for target in ('detect', 'classify', 'segment'):
            fingerprint = (
                self.annotations.data.detection_fingerprint()
                if target == 'detect'
                else self.annotations.data.target_fingerprint(target)
            )
            latest[target] = next(
                (view for view in reversed(runs) if view.run.target == target and view.run.fingerprint == fingerprint),
                None,
            )
        return {'workspace': workspace, 'editable': editable, 'training': latest}

    def _resume_submission(self, run: TrainingRun) -> TrainingRunView:
        current = self.store.get(run.user_id, run.id)
        task_id = current.clearml_task_id
        if task_id is None:
            if current.create_attempted_at is not None:
                task_id = self._find_remote(current.id)
                if task_id is None:
                    raise PlatformError('Training submission was not confirmed; an administrator must verify it')
            else:
                self.store.mark_create_attempted(current.id, _now())
                current = self.store.get(current.user_id, current.id)
                try:
                    task_id = self.clearml.create(current)
                except Exception:
                    task_id = self._find_remote(current.id)
                    if task_id is None:
                        raise PlatformError(
                            'Training submission was not confirmed; an administrator must verify it'
                        ) from None
            self.store.bind_task(current.id, task_id)
            current = self.store.get(current.user_id, current.id)
        try:
            self.clearml.enqueue(task_id)
        except Exception as error:
            raise PlatformError('Training queue submission could not be confirmed') from error
        return self._view(current)

    def _view(self, run: TrainingRun) -> TrainingRunView:
        if run.clearml_task_id is None:
            execution = _unknown('', 'Training submission has not been confirmed') if run.create_attempted_at else None
            return TrainingRunView(run, execution)
        try:
            execution = self.clearml.observe(run.clearml_task_id)
        except Exception:
            execution = _unknown(run.clearml_task_id, 'Training status is temporarily unavailable')
        return TrainingRunView(run, execution)

    def _get(self, user_id: int, run_id: str) -> TrainingRun:
        try:
            return self.store.get(user_id, run_id)
        except ValueError as error:
            raise PlatformAccessError('Training run access denied') from error

    def _find_user_run(self, user_id: int, run_id: str) -> TrainingRun | None:
        return next((run for run in self.store.list_user(user_id) if run.id == run_id), None)

    def _find_remote(self, run_id: str) -> str | None:
        try:
            return self.clearml.find(run_id)
        except Exception as error:
            raise PlatformError('Training submission recovery failed') from error

    def _relative_cache(self, cache_path: Path) -> str:
        try:
            return cache_path.resolve().relative_to(self.shared_root).as_posix()
        except ValueError as error:
            raise PlatformError('Training cache is outside the configured shared root') from error

    def _retained_cache(self, run: TrainingRun) -> Path:
        path = (self.shared_root / run.cache_relative_path).resolve()
        try:
            path.relative_to(self.shared_root)
        except ValueError as error:
            raise PlatformError('Retained training cache is unavailable') from error
        expected = self.annotations.runtime.cache_path(run.target, run.fingerprint).resolve()
        if path != expected:
            raise PlatformError('Retained training cache is unavailable')
        return path


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_request_id(request_id: str) -> None:
    try:
        valid = str(UUID(request_id)) == request_id
    except (TypeError, ValueError, AttributeError):
        valid = False
    if not valid:
        raise ValueError('Training request id must be a canonical UUID')


def _unknown(task_id: str, detail: str) -> ExecutionView:
    return ExecutionView(task_id, 'unknown', True, None, None, None, None, False, detail)
