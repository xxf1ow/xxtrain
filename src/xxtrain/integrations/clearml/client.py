from __future__ import annotations

import mimetypes
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from xxtrain.platform.training_contracts import DownloadFile, ExecutionView, TrainingRun

QUEUED_CANCELLATION_PARAMETER = 'xxtrain/queued_cancellation'
QUEUED_CANCELLATION_VALUE = 'dequeued-v1'


class ClearMLConflictError(RuntimeError):
    """Raised when one run UUID resolves to multiple ClearML tasks."""


class ClearMLClient:
    """Submit and observe xxtrain tasks without importing ClearML until first use."""

    def __init__(
        self,
        project: str,
        queue: str,
        worker_script: Path,
        shared_root: Path,
        *,
        run_root: Path | None = None,
        sdk: Any | None = None,
    ) -> None:
        self._project = project
        self._queue = queue
        self._worker_script = Path(worker_script).resolve()
        self._shared_root = Path(shared_root).resolve()
        self._run_root = Path(run_root).resolve() if run_root is not None else self._shared_root.parent / 'runs'
        self._sdk = sdk

    @property
    def sdk(self) -> Any:
        if self._sdk is None:
            self._sdk = _ClearMLSDK()
        return self._sdk

    def find(self, run_id: str) -> str | None:
        tasks = self.sdk.find(project_name=self._project, task_name=f'^{re.escape(run_id)}$')
        if len(tasks) > 1:
            raise ClearMLConflictError(f'Multiple ClearML tasks match training run {run_id}')
        return tasks[0].id if tasks else None

    def create(self, run: TrainingRun) -> str:
        task = self.sdk.create(
            project_name=self._project,
            task_name=run.id,
            task_type='training',
            script=str(self._worker_script),
            working_directory=str(self._worker_script.parent),
            packages=False,
            argparse_args=[
                ('task', 'point'),
                ('target', run.target),
                ('cache-relative-path', run.cache_relative_path),
                ('shared-root', str(self._shared_root)),
                ('run-id', run.id),
                ('run-root', str(self._run_root)),
            ],
            add_task_init_call=False,
        )
        return task.id

    def enqueue(self, task_id: str) -> None:
        if self.sdk.get(task_id).get('status') != 'created':
            return
        self.sdk.enqueue(task_id, queue_name=self._queue, force=False)

    def observe(self, task_id: str) -> ExecutionView:
        facts = self.sdk.get(task_id)
        artifact_ready = self.sdk.has_artifact(task_id, 'deployment', project_name=self._project)
        if facts.get('status') == 'stopped':
            return self._stopped_view(facts, artifact_ready)
        return parse_execution(facts, artifact_ready=artifact_ready)

    def cancel(self, task_id: str) -> ExecutionView:
        before = self.sdk.get(task_id)
        if before.get('status') == 'queued':
            self.sdk.cancel_queued(task_id)
        elif before.get('status') == 'in_progress':
            self.sdk.request_stop(task_id)
        facts = self.sdk.get(task_id)
        artifact_ready = self.sdk.has_artifact(task_id, 'deployment', project_name=self._project)
        if facts.get('status') == 'stopped':
            return self._stopped_view(facts, artifact_ready)
        return parse_execution(facts, artifact_ready=artifact_ready)

    def _stopped_view(self, facts: dict[str, object], artifact_ready: bool) -> ExecutionView:
        if _has_queued_cancellation_proof(facts):
            return parse_execution(facts, artifact_ready=artifact_ready, worker_released=True)
        try:
            released = self.sdk.worker_released(
                str(facts['id']), worker_id=facts.get('last_worker'), stopped_at=facts.get('status_changed')
            )
        except Exception:
            released = None
        if released is True:
            return parse_execution(facts, artifact_ready=artifact_ready, worker_released=True)
        detail = (
            'ClearML worker still reports this task' if released is False else 'Worker release could not be confirmed'
        )
        return parse_execution(facts, artifact_ready=artifact_ready, detail=detail)

    def download(self, task_id: str, destination: Path) -> DownloadFile:
        artifact = self.sdk.artifact(task_id, 'deployment', project_name=self._project)
        source = Path(artifact.local_path).resolve()
        if not source.is_file():
            raise FileNotFoundError('ClearML deployment artifact is unavailable')
        destination = Path(destination).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix.lower()
        filename = f'model{suffix}'
        target = destination / filename
        shutil.copy2(source, target)
        media_type = (
            'application/zip' if suffix == '.zip' else mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        )
        return DownloadFile(target, filename, media_type)


def parse_execution(
    facts: dict[str, object], *, artifact_ready: bool, worker_released: bool = False, detail: str | None = None
) -> ExecutionView:
    remote_status = str(facts.get('status', 'unknown'))
    if remote_status == 'queued':
        status, active = 'queued', True
    elif remote_status == 'in_progress':
        status, active = 'running', True
    elif remote_status == 'failed':
        status, active = 'failed', False
    elif remote_status == 'stopped' and worker_released:
        status, active = 'cancelled', False
    elif remote_status == 'completed' and artifact_ready:
        status, active = 'completed', False
    else:
        status, active = 'unknown', True
    parameters = facts.get('parameters') if isinstance(facts.get('parameters'), dict) else {}
    return ExecutionView(
        task_id=str(facts.get('id', '')),
        status=status,
        active=active,
        epoch=_optional_int(parameters.get('xxtrain/epoch')),
        total_epochs=_optional_int(parameters.get('xxtrain/total_epochs')),
        elapsed_seconds=_optional_float(facts.get('active_duration')),
        metric=_optional_float(parameters.get('xxtrain/metric')),
        download_ready=status == 'completed' and artifact_ready,
        detail=detail,
    )


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float, str)) and str(value).isdigit() else None


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class _ClearMLSDK:
    def __init__(self) -> None:
        from clearml import Task

        self._task = Task

    def create(self, **kwargs: object) -> Any:
        return self._task.create(**kwargs)

    def find(self, **kwargs: object) -> list[Any]:
        return self._task.get_tasks(**kwargs)

    def enqueue(self, task_id: str, **kwargs: object) -> None:
        self._task.enqueue(task_id, **kwargs)

    def get(self, task_id: str) -> dict[str, object]:
        task = self._task.get_task(task_id=task_id)
        data = task.data.to_dict()
        data['id'] = task.id
        data['last_worker'] = task.last_worker
        data['parameters'] = task.get_parameters(cast=True)
        return data

    def cancel_queued(self, task_id: str) -> None:
        response = self._task.dequeue(task_id)
        if getattr(response, 'dequeued', None) != 1:
            raise RuntimeError('ClearML task left the queue before cancellation')
        task = self._task.get_task(task_id=task_id)
        task.set_parameter(QUEUED_CANCELLATION_PARAMETER, QUEUED_CANCELLATION_VALUE)
        task.stopped(ignore_errors=False, force=True, status_reason='cancelled before execution')

    def request_stop(self, task_id: str) -> None:
        task = self._task.get_task(task_id=task_id)
        task.stop_request(ignore_errors=False, force=False, status_message='Cancellation requested by xxtrain')

    def worker_released(self, task_id: str, *, worker_id: object, stopped_at: object) -> bool | None:
        from clearml.backend_api.services.v2_20 import workers

        if not isinstance(worker_id, str) or not worker_id:
            return None
        stopped_time = _timestamp(stopped_at)
        if stopped_time is None:
            return None
        session = self._task._get_default_session()
        response = session.send(workers.GetAllRequest(last_seen=None))
        if not response.ok():
            raise RuntimeError('ClearML worker query failed')
        for worker in response.response.workers or ():
            if getattr(worker, 'id', None) != worker_id:
                continue
            report_time = _timestamp(getattr(worker, 'last_report_time', None))
            if report_time is None or report_time < stopped_time:
                return None
            current = getattr(worker, 'task', None)
            return not (current is not None and getattr(current, 'id', None) == task_id)
        return None

    def has_artifact(self, task_id: str, name: str, *, project_name: str) -> bool:
        task = self._task.get_task(task_id=task_id)
        if task.get_project_name() != project_name:
            raise ValueError('ClearML task does not belong to the configured project')
        try:
            artifact = task.artifacts[name]
        except KeyError:
            return False
        return bool(artifact.url)

    def artifact(self, task_id: str, name: str, *, project_name: str) -> Any:
        task = self._task.get_task(task_id=task_id)
        if task.get_project_name() != project_name:
            raise ValueError('ClearML task does not belong to the configured project')
        artifact = task.artifacts[name]
        local_copy = artifact.get_local_copy()
        if not local_copy:
            raise FileNotFoundError('ClearML deployment artifact is unavailable')
        local_path = Path(local_copy).resolve()
        if not local_path.is_file() or local_path.stat().st_size == 0:
            raise FileNotFoundError('ClearML deployment artifact is unavailable')
        return SimpleNamespace(name=name, url=artifact.url, local_path=local_path)


def _timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return None
    else:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _has_queued_cancellation_proof(facts: dict[str, object]) -> bool:
    parameters = facts.get('parameters')
    return (
        not facts.get('last_worker')
        and isinstance(parameters, dict)
        and parameters.get(QUEUED_CANCELLATION_PARAMETER) == QUEUED_CANCELLATION_VALUE
    )
