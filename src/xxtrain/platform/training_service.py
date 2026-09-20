from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from xxtrain.business_tasks.loader import load_training_task_definition
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import PlatformAccessError, PlatformConflictError, PlatformError
from xxtrain.platform.service import AnnotationService
from xxtrain.platform.training_contracts import DownloadFile, ExecutionView, TrainingRun, TrainingRunView
from xxtrain.platform.training_store import TrainingRunStore

_LOGGER = logging.getLogger(__name__)


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
            fingerprint = self.annotations.data.training_fingerprint(target)
            existing = self.store.find_input(user_id, self.config.workspace_id, target, fingerprint)
            if existing is not None:
                run = existing
            else:
                cache_path = self.annotations.ensure_target_cache(user_id, target)
                run = self.store.create(
                    TrainingRun(
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
                        'execute',
                        self.config.task_entry,
                    )
                )
        self._reconcile_run(run)
        return self._view(self.store.get(user_id, run.id))

    def list_runs(self, user_id: int) -> tuple[TrainingRunView, ...]:
        return tuple(self._view(run) for run in self.store.list_user(user_id))

    def get_run(self, user_id: int, run_id: str) -> TrainingRunView:
        return self._view(self._get(user_id, run_id))

    def cancel(self, user_id: int, run_id: str) -> TrainingRunView:
        try:
            with self.annotations.mutation(user_id):
                run = self.store.request_cancel(user_id, run_id)
        except ValueError as error:
            raise PlatformAccessError('Training run access denied') from error
        try:
            self._reconcile_run(run)
        except PlatformError as error:
            _LOGGER.warning('Training cancellation remains pending for run %s: %s', run.id, error)
        return self._view(self.store.get(user_id, run.id))

    def reconcile_pending(self) -> None:
        """Reconcile configured-workspace runs in durable submission order."""
        for run in self.store.list_workspace(self.config.workspace_id):
            try:
                self._reconcile_run(run)
            except PlatformConflictError:
                continue
            except PlatformError as error:
                _LOGGER.warning('Training reconciliation remains pending for run %s: %s', run.id, error)

    def download(self, user_id: int, run_id: str) -> DownloadFile:
        run = self._get(user_id, run_id)
        view = self._view(run)
        if view.execution is None or not view.execution.download_ready or run.clearml_task_id is None:
            raise PlatformError('Training result is not ready for download')
        try:
            return self.clearml.download(run.clearml_task_id, self.config.runtime_dir / 'downloads' / run.id)
        except (OSError, RuntimeError) as error:
            raise PlatformError('Training result download failed') from error

    def require_editable(self, workspace_id: str, target: str | None = None) -> None:
        if workspace_id != self.config.workspace_id:
            raise PlatformAccessError('Workspace access denied')
        can_upload, locked_targets = self._edit_protection(workspace_id)
        if (target is None and not can_upload) or (target is not None and target in locked_targets):
            raise PlatformConflictError('Workspace editing is disabled while training is active')

    def require_cache_rebuild(self, workspace_id: str, target: str, fingerprint: str) -> None:
        """Reject rebuilding a publication referenced by any current or historical run."""
        if workspace_id != self.config.workspace_id:
            raise PlatformAccessError('Workspace access denied')
        runs = self.store.list_workspace(workspace_id)
        if any(self.store.find_input(run.user_id, workspace_id, target, fingerprint) is not None for run in runs):
            raise PlatformError('A training run references this cache publication')

    def workspace_view(self, user_id: int) -> dict[str, object]:
        with self.annotations.mutation(user_id):
            workspace = self.annotations.view(user_id)
            user_runs = self.store.list_user(user_id)
            workspace_runs = self.store.list_workspace(self.config.workspace_id)
            targets = tuple(step.key for step in self.annotations.data.task.steps)
            training_targets = tuple(step.key for step in self.annotations.data.task.steps if step.training is not None)
            fingerprints = {target: self.annotations.data.training_fingerprint(target) for target in training_targets}
        active_runs = tuple(self._view(run) for run in workspace_runs)
        active_by_id = {view.run.id: view for view in active_runs}
        visible_runs = tuple(active_by_id.get(run.id) or self._view(run) for run in user_runs)
        can_upload, locked_targets = self._edit_protection_from_views(active_runs)
        visible_by_id = {view.run.id: view for view in visible_runs}
        latest = {
            target: (
                None
                if (run := self.store.find_input(user_id, self.config.workspace_id, target, fingerprints[target]))
                is None
                else visible_by_id[run.id]
            )
            for target in training_targets
        }
        return {
            'workspace': workspace,
            'can_upload': can_upload,
            'editable': can_upload,
            'target_editable': {target: target not in locked_targets for target in targets},
            'training': latest,
        }

    def _edit_protection(self, workspace_id: str) -> tuple[bool, frozenset[str]]:
        views = tuple(self._view(run) for run in self.store.list_workspace(workspace_id))
        return self._edit_protection_from_views(views)

    def _edit_protection_from_views(self, views: tuple[TrainingRunView, ...]) -> tuple[bool, frozenset[str]]:
        task = self.annotations.data.task
        all_targets = frozenset(step.key for step in task.steps)
        locked: set[str] = set()
        active = False
        for view in views:
            if not _is_active(view):
                continue
            active = True
            try:
                run_task = load_training_task_definition(view.run.task_entry)
                if run_task.key != task.key:
                    raise ValueError('Training run belongs to another task definition')
                locked.update(run_task.input_steps(view.run.target))
            except ValueError:
                locked.update(all_targets)
        return not active, frozenset(locked)

    def _reconcile_run(self, run: TrainingRun) -> None:
        with self.annotations.mutation(run.user_id):
            current = self.store.get(run.user_id, run.id)
            if current.desired_action is None:
                return
            task_id = current.clearml_task_id
            if task_id is None:
                if current.create_attempted_at is None:
                    if current.desired_action == 'cancel':
                        return
                    self.store.mark_create_attempted(current.id, _now())
                    current = self.store.get(current.user_id, current.id)
                    try:
                        task_id = self.clearml.create(current)
                    except (OSError, RuntimeError) as error:
                        task_id = self._find_remote(current.id)
                        if task_id is None:
                            raise PlatformError(
                                'Training submission was not confirmed; an administrator must verify it'
                            ) from error
                else:
                    task_id = self._find_remote(current.id)
                    if task_id is None:
                        raise PlatformError('Training submission was not confirmed; an administrator must verify it')
                self.store.bind_task(current.id, task_id)
                current = self.store.get(current.user_id, current.id)
            try:
                if current.desired_action == 'execute':
                    self.clearml.enqueue(task_id, current)
                else:
                    self.clearml.cancel(task_id)
            except (OSError, RuntimeError) as error:
                action = (
                    'launch configuration or queue submission'
                    if current.desired_action == 'execute'
                    else 'cancellation'
                )
                raise PlatformError(f'Training {action} could not be confirmed') from error

    def _view(self, run: TrainingRun) -> TrainingRunView:
        if run.clearml_task_id is None:
            if run.create_attempted_at is not None:
                execution = _unknown('', 'Training submission has not been confirmed')
            elif run.desired_action == 'execute':
                execution = ExecutionView('', 'pending', True, None, None, None, None, False, None)
            elif run.desired_action == 'cancel':
                execution = ExecutionView('', 'cancelled', False, None, None, None, None, False, None)
            else:
                execution = None
            return TrainingRunView(run, execution, run.desired_action == 'cancel')
        try:
            execution = self.clearml.observe(run.clearml_task_id)
        except (OSError, RuntimeError):
            execution = _unknown(run.clearml_task_id, 'Training status is temporarily unavailable')
        return TrainingRunView(run, execution, run.desired_action == 'cancel')

    def _get(self, user_id: int, run_id: str) -> TrainingRun:
        try:
            return self.store.get(user_id, run_id)
        except ValueError as error:
            raise PlatformAccessError('Training run access denied') from error

    def _find_remote(self, run_id: str) -> str | None:
        try:
            return self.clearml.find(run_id)
        except (OSError, RuntimeError) as error:
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


def _is_active(view: TrainingRunView) -> bool:
    return view.execution is not None and view.execution.active
