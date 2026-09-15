from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from xxtrain.business_tasks import POINT_BOX_LABELS
from xxtrain.integrations.cvat import PreparationState
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import JobRef, PlatformAccessError, PlatformError, WorkspaceView
from xxtrain.platform.state import StateStore
from xxtrain.workspace_data import WorkspaceData

if TYPE_CHECKING:
    from xxtrain.integrations.cvat.client import CvatClient

_CREATION_UNCERTAIN = 'CVAT task creation result is uncertain; administrator reconciliation is required'
_SYNC_ERROR = '无法取回或保存标注，请重试。'
_STATUSES = {'pending', 'preparing', 'annotating', 'sync_failed', 'saved'}


class AnnotationService:
    """Coordinate one configured workspace across durable local state and CVAT."""

    def __init__(self, config: WorkspaceConfig, data: WorkspaceData, cvat: CvatClient, state: StateStore) -> None:
        self.config = config
        self.data = data
        self.cvat = cvat
        self.state = state
        self.lock = threading.Lock()

    def view(self, user_id: int) -> WorkspaceView:
        """Return the configured owner's current local workflow status.

        Other users receive ``PlatformAccessError`` before workspace files are read. Malformed or non-object durable
        JSON from ``StateStore.load`` raises ``ValueError``; subsequent state validation raises ``PlatformError``.
        """
        self._require_owner(user_id)
        state = self.state.load()
        status = state.get('status', 'pending')
        if not isinstance(status, str) or status not in _STATUSES:
            raise PlatformError('Workspace state contains an invalid status')
        error = state.get('error')
        if error is not None and not isinstance(error, str):
            raise PlatformError('Workspace state contains an invalid error')
        return WorkspaceView(
            workspace_id=self.config.workspace_id,
            name=self.config.display_name,
            image_count=len(self.data.images()),
            status=status,
            error=error,
        )

    def begin(self, user_id: int) -> str:
        """Create or continue the owner's detection Job and return its same-origin CVAT path.

        Remote-write intent and every preparation checkpoint are saved before a Job path is returned. Other users
        receive ``PlatformAccessError``; ambiguous creation or preparation raises ``PlatformError`` and remains
        persisted for retry or administrator reconciliation.
        """
        self._require_owner(user_id)
        with self.lock:
            state = self.state.load()
            ref = self._job_ref(state.get('job'))
            if ref is not None:
                self.state.save({**state, 'status': 'annotating', 'error': None})
                return self.cvat.job_path(ref)

            task_id = self._task_id(state.get('task_id'))
            if task_id is None and state.get('task_creation') == 'creating':
                raise PlatformError(_CREATION_UNCERTAIN)

            images = self.data.images()
            if task_id is None:
                state = {**state, 'status': 'preparing', 'error': None, 'task_creation': 'creating'}
                self.state.save(state)
                try:
                    task_id = self.cvat.create_task(self.config.display_name, POINT_BOX_LABELS)
                except PlatformError as error:
                    state = {**state, 'error': _CREATION_UNCERTAIN}
                    self.state.save(state)
                    raise PlatformError(_CREATION_UNCERTAIN) from error
                task_id = self._required_identifier(task_id, 'task_id')
                state = {key: value for key, value in state.items() if key != 'task_creation'} | {
                    'status': 'preparing',
                    'error': None,
                    'task_id': task_id,
                    'preparation': self._preparation_dict(PreparationState()),
                }
                self.state.save(state)

            preparation = self._preparation(state.get('preparation'))

            def checkpoint(current: PreparationState) -> None:
                nonlocal state
                state = {**state, 'preparation': self._preparation_dict(current)}
                self.state.save(state)

            try:
                ref = self.cvat.prepare_task(task_id, images, user_id, preparation=preparation, checkpoint=checkpoint)
            except PlatformError as error:
                state = {**state, 'status': 'preparing', 'error': str(error)}
                self.state.save(state)
                raise

            state = {
                **state,
                'status': 'annotating',
                'error': None,
                'job': {'task_id': ref.task_id, 'job_id': ref.job_id, 'sample_ids': list(ref.sample_ids)},
            }
            self.state.save(state)
            return self.cvat.job_path(ref)

    def sync(self, user_id: int) -> WorkspaceView:
        """Fetch the Job and report saved only after every local annotation and state replacement succeeds.

        Fetch, validation, and annotation-save failures persist ``sync_failed`` and raise ``PlatformError``. State
        persistence failures remain visible and never produce a successful return.
        """
        self._require_owner(user_id)
        with self.lock:
            state = self.state.load()
            ref = self._job_ref(state.get('job'))
            if ref is None:
                raise PlatformError('Annotation job is not ready')
            try:
                results = self.cvat.fetch_detection(ref)
                self.data.save_detection(results)
                self.state.save({**state, 'status': 'saved', 'error': None})
            except (OSError, ValueError, PlatformError) as error:
                self.state.save({**state, 'status': 'sync_failed', 'error': _SYNC_ERROR})
                raise PlatformError(_SYNC_ERROR) from error
        return self.view(user_id)

    def _require_owner(self, user_id: int) -> None:
        if user_id != self.config.owner_user_id:
            raise PlatformAccessError('Workspace access denied')

    @classmethod
    def _job_ref(cls, record: object) -> JobRef | None:
        if record is None:
            return None
        if not isinstance(record, dict) or set(record) != {'task_id', 'job_id', 'sample_ids'}:
            raise PlatformError('Workspace state contains an invalid Job reference')
        task_id = cls._required_identifier(record['task_id'], 'task_id')
        job_id = cls._required_identifier(record['job_id'], 'job_id')
        sample_ids = record['sample_ids']
        if not isinstance(sample_ids, list) or any(
            not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids
        ):
            raise PlatformError('Workspace state contains invalid Job sample IDs')
        return JobRef(task_id, job_id, tuple(sample_ids))

    @classmethod
    def _task_id(cls, value: object) -> int | None:
        if value is None:
            return None
        return cls._required_identifier(value, 'task_id')

    @staticmethod
    def _required_identifier(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PlatformError(f'Workspace state contains an invalid {name}')
        return value

    @staticmethod
    def _preparation(value: object) -> PreparationState:
        if value is None:
            return PreparationState()
        if not isinstance(value, dict) or set(value) != {'stage', 'request_id'}:
            raise PlatformError('Workspace state contains an invalid CVAT preparation checkpoint')
        try:
            return PreparationState(stage=value['stage'], request_id=value['request_id'])
        except (TypeError, ValueError) as error:
            raise PlatformError('Workspace state contains an invalid CVAT preparation checkpoint') from error

    @staticmethod
    def _preparation_dict(state: PreparationState) -> dict[str, str | None]:
        return {'stage': state.stage, 'request_id': state.request_id}
