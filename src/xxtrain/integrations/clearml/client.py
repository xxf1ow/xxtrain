from __future__ import annotations

import mimetypes
import re
import shutil
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from xxtrain.platform.training_contracts import DownloadFile, ExecutionView, TrainingRun

QUEUED_CANCELLATION_PARAMETER = 'xxtrain/queued_cancellation'
QUEUED_CANCELLATION_VALUE = 'dequeued-v1'
CREATED_CANCELLATION_VALUE = 'created-v1'
_REQUEST_ATTEMPTS = 3
_PROGRAMMING_ERRORS = (AssertionError, AttributeError, KeyError, TypeError)
_PROGRAMMING_ERROR_ATTRIBUTE = '_xxtrain_programming_error'


class _BoundedSession:
    """Mixin that bounds the ClearML SDK request loops used by the platform adapter."""

    _session_initial_timeout = (3.0, 10.0)
    _session_timeout = (3.0, 10.0)
    _write_session_timeout = (3.0, 10.0)

    def _send_request(
        self,
        service,
        action,
        version=None,
        method=None,
        headers=None,
        auth=None,
        data=None,
        json=None,
        refresh_token_if_unauthorized=True,
        params=None,
    ):
        from clearml.backend_api.session.request import Request
        from requests import codes
        from requests.exceptions import ChunkedEncodingError, ContentDecodingError, SSLError, StreamConsumedError

        if self._offline_mode:
            return None
        method = method or Request.def_method
        host = self.host
        headers = headers.copy() if headers else {}
        for name in self._WORKER_HEADER:
            headers[name] = self.worker
        for name in self._CLIENT_HEADER:
            headers[name] = self.client
        url = f'{host}/v{version}/{service}.{action}' if version else f'{host}/{service}.{action}'
        token_refreshed = False
        for attempt in range(_REQUEST_ATTEMPTS):
            timeout = self._request_timeout(data)
            try:
                response = self._Session__http_session.request(
                    method, url, headers=headers, auth=auth, data=data, json=json, timeout=timeout, params=params
                )
            except (SSLError, ChunkedEncodingError, ContentDecodingError, StreamConsumedError):
                if attempt + 1 == _REQUEST_ATTEMPTS:
                    raise
                continue
            if refresh_token_if_unauthorized and response.status_code == codes.unauthorized and not token_refreshed:
                self.refresh_token()
                token_refreshed = True
                continue
            self._session_requests += 1
            return response
        raise RuntimeError('ClearML request retry bound was exhausted')

    def send(self, req_obj, async_enable=False, headers=None):
        try:
            result = super().send(req_obj, async_enable=async_enable, headers=headers)
        except _PROGRAMMING_ERRORS as error:
            return self._request_failure(req_obj, error)
        except Exception:
            return self._request_failure(req_obj)
        if result is not None and result.meta.result_code > 500:
            return self._request_failure(req_obj)
        return result

    def _request_failure(self, req_obj, programming_error=None):
        from clearml.backend_api.session.callresult import CallResult

        result = CallResult._make_raw_response(
            request_cls=req_obj.__class__,
            service=req_obj._service,
            action=req_obj._action,
            status_code=500,
            text='ClearML transport request failed',
        )
        if programming_error is not None:
            setattr(result, _PROGRAMMING_ERROR_ATTRIBUTE, programming_error)
        return result

    def _request_timeout(self, data):
        if data and len(data) > self._write_session_data_size:
            return self._write_session_timeout
        if self._session_requests < 1:
            return self._session_initial_timeout
        return self._session_timeout


class ClearMLConflictError(RuntimeError):
    """Raised when one run UUID resolves to multiple ClearML tasks."""


class ClearMLOperationError(RuntimeError):
    """Raised when an installed ClearML SDK operation fails."""


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
        tasks = self._sdk_call('find', project_name=self._project, task_name=f'^{re.escape(run_id)}$')
        if len(tasks) > 1:
            raise ClearMLConflictError(f'Multiple ClearML tasks match training run {run_id}')
        return tasks[0].id if tasks else None

    def create(self, run: TrainingRun) -> str:
        task = self._sdk_call('create', **self._creation_arguments(run))
        return task.id

    def enqueue(self, task_id: str, run: TrainingRun) -> None:
        if self._sdk_call('get', task_id).get('status') != 'created':
            return
        ready = self._sdk_call('prepare', task_id, **self._creation_arguments(run))
        status = self._sdk_call('get', task_id).get('status')
        if not ready:
            if status == 'created':
                raise RuntimeError('ClearML task launch readiness could not be confirmed')
            return
        if status != 'created':
            return
        self._sdk_call('enqueue', task_id, queue_name=self._queue, force=False)

    def _creation_arguments(self, run: TrainingRun) -> dict[str, object]:
        return {
            'project_name': self._project,
            'task_name': run.id,
            'task_type': 'training',
            'script': str(self._worker_script),
            'working_directory': str(self._worker_script.parent),
            'packages': False,
            'argparse_args': [
                ('task', run.task_entry),
                ('target', run.target),
                ('cache_relative_path', run.cache_relative_path),
                ('shared_root', str(self._shared_root)),
                ('run_id', run.id),
                ('run_root', str(self._run_root)),
            ],
            'add_task_init_call': False,
        }

    def observe(self, task_id: str) -> ExecutionView:
        facts = self._sdk_call('get', task_id)
        artifact_ready, detail = self._artifact_observation(task_id)
        if facts.get('status') == 'stopped':
            return self._stopped_view(facts, artifact_ready)
        return parse_execution(facts, artifact_ready=artifact_ready, detail=detail)

    def cancel(self, task_id: str) -> ExecutionView:
        before = self._sdk_call('get', task_id)
        if before.get('status') == 'queued':
            self._sdk_call('cancel_queued', task_id)
        elif before.get('status') == 'created':
            self._sdk_call('cancel_created', task_id)
        elif before.get('status') == 'in_progress':
            self._sdk_call('request_stop', task_id)
        facts = self._sdk_call('get', task_id)
        artifact_ready, detail = self._artifact_observation(task_id)
        if facts.get('status') == 'stopped':
            return self._stopped_view(facts, artifact_ready)
        return parse_execution(facts, artifact_ready=artifact_ready, detail=detail)

    def _artifact_observation(self, task_id: str) -> tuple[bool, str | None]:
        try:
            ready = self._sdk_call('has_artifact', task_id, 'deployment', project_name=self._project)
            return ready, None
        except ClearMLOperationError:
            return False, 'Deployment artifact status is temporarily unavailable'

    def _sdk_call(self, operation, *args, **kwargs):
        try:
            return getattr(self.sdk, operation)(*args, **kwargs)
        except _PROGRAMMING_ERRORS:
            raise
        except Exception as error:
            programming_error = getattr(getattr(error, 'result', None), _PROGRAMMING_ERROR_ATTRIBUTE, None)
            if isinstance(programming_error, _PROGRAMMING_ERRORS):
                raise programming_error from error
            raise ClearMLOperationError('ClearML SDK operation failed') from error

    def _stopped_view(self, facts: dict[str, object], artifact_ready: bool) -> ExecutionView:
        if _has_never_started_cancellation_proof(facts):
            return parse_execution(facts, artifact_ready=artifact_ready, worker_released=True)
        try:
            released = self._sdk_call(
                'worker_released',
                str(facts['id']),
                worker_id=facts.get('last_worker'),
                stopped_at=facts.get('status_changed'),
            )
        except ClearMLOperationError:
            released = None
        if released is True:
            return parse_execution(facts, artifact_ready=artifact_ready, worker_released=True)
        detail = (
            'ClearML worker still reports this task' if released is False else 'Worker release could not be confirmed'
        )
        return parse_execution(facts, artifact_ready=artifact_ready, detail=detail)

    def download(self, task_id: str, destination: Path) -> DownloadFile:
        artifact = self._sdk_call('artifact', task_id, 'deployment', project_name=self._project)
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
    elif remote_status == 'completed':
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
        from clearml.backend_api.session.defs import ENV_ACCESS_KEY, ENV_SECRET_KEY
        from clearml.backend_api.session.session import Session
        from clearml.config import config_obj

        config_obj.get('api.http')
        config = deepcopy(config_obj._config)
        config.set_overrides({'api': {'http': {'wait_on_maintenance_forever': False}}})
        session_type = type('_XXTrainClearMLSession', (_BoundedSession, Session), {})
        self._session = session_type(
            initialize_logging=False,
            config=config,
            api_key=ENV_ACCESS_KEY.get(),
            secret_key=ENV_SECRET_KEY.get(),
            http_retries_config={'total': 0, 'connect': 0, 'read': 0, 'redirect': 0, 'status': 0, 'backoff_factor': 0},
        )
        Task._set_default_session(self._session)

        self._task = Task

    def create(self, **kwargs: object) -> Any:
        return self._task.create(**kwargs)

    def find(self, **kwargs: object) -> list[Any]:
        return self._task.get_tasks(**kwargs)

    def enqueue(self, task_id: str, **kwargs: object) -> None:
        self._task.enqueue(task_id, **kwargs)

    def prepare(self, task_id: str, **creation_arguments: object) -> bool:
        from clearml.backend_interface.task.populate import CreateAndPopulate

        arguments = list(creation_arguments.pop('argparse_args'))
        expected_script = CreateAndPopulate(**creation_arguments, raise_on_missing_entries=False).create_task(
            dry_run=True
        )['script']
        expected_parameters = {f'Args/{name}': value for name, value in arguments}

        task = self._task.get_task(task_id=task_id)
        if not _is_unstarted_created(task):
            return False
        actual_script = _script_state(task)
        if not _script_matches(actual_script, expected_script):
            if _has_launch_script(actual_script):
                raise RuntimeError('ClearML task launch script does not match the configured worker')
            task.update_task({'script': expected_script})

        task = self._task.get_task(task_id=task_id)
        if not _is_unstarted_created(task):
            return False
        actual_parameters = task.get_parameters()
        conflicting = {
            name
            for name, value in expected_parameters.items()
            if name in actual_parameters and str(actual_parameters[name]) != str(value)
        }
        if conflicting:
            raise RuntimeError('ClearML task launch parameters do not match the canonical run')
        missing = {name: value for name, value in expected_parameters.items() if name not in actual_parameters}
        if missing:
            task.update_parameters(missing)

        task = self._task.get_task(task_id=task_id)
        if not _is_unstarted_created(task):
            return False
        if not _script_matches(_script_state(task), expected_script):
            raise RuntimeError('ClearML task launch script could not be confirmed')
        persisted = task.get_parameters()
        if any(str(persisted.get(name)) != str(value) for name, value in expected_parameters.items()):
            raise RuntimeError('ClearML task launch parameters could not be confirmed')
        return True

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

    def cancel_created(self, task_id: str) -> None:
        task = self._task.get_task(task_id=task_id)
        if str(task.data.status) != 'created' or task.last_worker:
            raise RuntimeError('ClearML task execution could not be excluded')
        task.set_parameter(QUEUED_CANCELLATION_PARAMETER, CREATED_CANCELLATION_VALUE)
        task.stopped(ignore_errors=False, force=False, status_reason='cancelled before execution')

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


def _has_never_started_cancellation_proof(facts: dict[str, object]) -> bool:
    parameters = facts.get('parameters')
    return (
        not facts.get('last_worker')
        and isinstance(parameters, dict)
        and parameters.get(QUEUED_CANCELLATION_PARAMETER) in {QUEUED_CANCELLATION_VALUE, CREATED_CANCELLATION_VALUE}
    )


_SCRIPT_FIELDS = ('repository', 'version_num', 'branch', 'diff', 'working_dir', 'entry_point', 'binary', 'requirements')


def _is_unstarted_created(task: Any) -> bool:
    return str(task.data.status) == 'created' and not task.last_worker


def _script_state(task: Any) -> dict[str, object]:
    data = task.data.to_dict()
    script = data.get('script')
    return script if isinstance(script, dict) else {}


def _script_matches(actual: dict[str, object], expected: dict[str, object]) -> bool:
    return all(actual.get(name) == expected.get(name) for name in _SCRIPT_FIELDS)


def _has_launch_script(script: dict[str, object]) -> bool:
    return any(script.get(name) not in (None, '', {}, []) for name in _SCRIPT_FIELDS if name != 'binary')
