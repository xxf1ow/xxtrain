from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import PlatformAccessError, PlatformConflictError, PlatformError
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

    def submit(self, user_id: int, target: str) -> TrainingRunView:
        """Return the canonical run for the user's current target input, creating it when absent."""
        with self.annotations.mutation(user_id):
            fingerprint = (
                self.annotations.data.detection_fingerprint()
                if target == 'detect'
                else self.annotations.data.target_fingerprint(target)
            )
            existing = self.store.find_input(user_id, self.config.workspace_id, target, fingerprint)
            if existing is not None:
                return self._resume_submission(existing)

            cache_path = self.annotations.ensure_target_cache(user_id, target)
            run = TrainingRun(
                str(uuid4()),
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
            return self._resume_submission(self.store.create(run))

    def list_runs(self, user_id: int) -> tuple[TrainingRunView, ...]:
        runs = self.store.list_user(user_id)
        if not runs:
            return ()
        with self.annotations.mutation(user_id):
            return tuple(self._view(self._reconcile_submission(run)) for run in runs)

    def get_run(self, user_id: int, run_id: str) -> TrainingRunView:
        run = self._get(user_id, run_id)
        with self.annotations.mutation(user_id):
            return self._view(self._reconcile_submission(run))

    def cancel(self, user_id: int, run_id: str) -> TrainingRunView:
        run = self._get(user_id, run_id)
        if run.clearml_task_id is None:
            return self._view(run)
        try:
            execution = self.clearml.cancel(run.clearml_task_id)
        except Exception:
            execution = _unknown(run.clearml_task_id, 'Training cancellation could not be confirmed')
        return TrainingRunView(run, execution)

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
                raise PlatformConflictError('Workspace editing is disabled while training is active')

    def require_cache_rebuild(self, workspace_id: str, target: str, fingerprint: str) -> None:
        """Reject rebuilding a publication referenced by any current or historical run."""
        if workspace_id != self.config.workspace_id:
            raise PlatformAccessError('Workspace access denied')
        if any(
            run.target == target and run.fingerprint == fingerprint for run in self.store.list_workspace(workspace_id)
        ):
            raise PlatformError('A training run references this cache publication')

    def workspace_view(self, user_id: int) -> dict[str, object]:
        with self.annotations.mutation(user_id):
            workspace = self.annotations.view(user_id)
            runs = tuple(self._view(self._reconcile_submission(run)) for run in self.store.list_user(user_id))
            editable = not any(view.execution is not None and view.execution.active for view in runs)
            latest = {}
            for target in ('detect', 'classify', 'segment'):
                fingerprint = (
                    self.annotations.data.detection_fingerprint()
                    if target == 'detect'
                    else self.annotations.data.target_fingerprint(target)
                )
                latest[target] = next(
                    (
                        view
                        for view in reversed(runs)
                        if view.run.target == target and view.run.fingerprint == fingerprint
                    ),
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

    def _reconcile_submission(self, run: TrainingRun) -> TrainingRun:
        current = self.store.get(run.user_id, run.id)
        if current.clearml_task_id is None:
            if current.create_attempted_at is None:
                return current
            try:
                task_id = self.clearml.find(current.id)
                if task_id is None:
                    return current
                self.store.bind_task(current.id, task_id)
                current = self.store.get(current.user_id, current.id)
            except Exception:
                return current
        try:
            self.clearml.enqueue(current.clearml_task_id)
        except Exception:
            pass
        return current

    def _get(self, user_id: int, run_id: str) -> TrainingRun:
        try:
            return self.store.get(user_id, run_id)
        except ValueError as error:
            raise PlatformAccessError('Training run access denied') from error

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


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _unknown(task_id: str, detail: str) -> ExecutionView:
    return ExecutionView(task_id, 'unknown', True, None, None, None, None, False, detail)
