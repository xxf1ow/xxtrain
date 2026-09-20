from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import quote

import httpx

from xxtrain.business_tasks.definition import AnnotationPolicy
from xxtrain.platform.contracts import (
    CvatBinding,
    EditFrame,
    EditFrameResult,
    EditJob,
    JobRef,
    PlatformAccessError,
    PlatformError,
    PreparedJob,
)

from .edit_codec import decode_edit_annotations, decode_edit_bindings, encode_edit_annotations

_EXTRA_ATTRIBUTE = 'xxtrain_labelme_extra'
_PREPARE_TIMEOUT_SECONDS = 120.0
_POLL_INTERVAL_SECONDS = 0.25
_REQUEST_TIMEOUT_SECONDS = 10.0

type _BindingDecoder = Callable[[dict, JobRef, list[dict]], tuple[CvatBinding, ...]]


class CvatClient:
    """Use CVAT's HTTP API without sharing service credentials with browser sessions.

    The supplied HTTPX client owns transport and connection lifecycle. This adapter builds credential-isolated
    requests, disables redirects, applies request timeouts, and translates failures without including response bodies
    or credentials. Browser authentication failures raise ``PlatformAccessError``; service-token and operational
    failures raise ``PlatformError``.
    """

    def __init__(self, base_url: str, service_token: str, http: httpx.Client):
        parsed = httpx.URL(base_url)
        if parsed.scheme not in {'http', 'https'} or not parsed.host:
            raise ValueError('CVAT base URL must be an HTTP origin')
        if parsed.username or parsed.password:
            raise ValueError('CVAT base URL must not include a username or password')
        if parsed.query or parsed.fragment or parsed.path not in {'', '/'}:
            raise ValueError('CVAT base URL must not include a path, query, or fragment')
        self._base_url = parsed.copy_with(path='/')
        self._origin = parsed.scheme, parsed.host, parsed.port
        self._service_token = service_token
        self._http = http

    def create_task(self, name: str, labels: tuple[str, ...], policy: AnnotationPolicy) -> int:
        """Create a task with the definition's native CVAT labels and optional negative-image tag."""
        typed_labels = tuple((label, policy.cvat_type) for label in labels)
        if policy.negative_label is not None:
            typed_labels += ((policy.negative_label, 'tag'),)
        return self._create_typed_task(name, typed_labels)

    def _create_typed_task(self, name: str, labels: tuple[tuple[str, str], ...]) -> int:
        payload = {
            'name': name,
            'labels': [
                {
                    'name': label,
                    'type': label_type,
                    'attributes': [
                        {
                            'name': _EXTRA_ATTRIBUTE,
                            'mutable': True,
                            'input_type': 'text',
                            'default_value': '{}',
                            'values': [],
                        }
                    ],
                }
                for label, label_type in labels
            ],
        }
        response = self._service_request('POST', '/api/tasks', json=payload)
        task_id = self._integer_field(self._json(response), 'id', '/api/tasks')
        return task_id

    def prepare_task(
        self, task_id: int, frames: tuple[EditFrame, ...], user_id: int, policy: AnnotationPolicy
    ) -> PreparedJob:
        """Upload ordered edit frames, initialize policy annotations, and return original-image bindings.

        Generated filenames are opaque ordering keys. ``PreparedJob.ref.sample_ids`` contains each original image ID
        once in first-frame order; callers retain the supplied mappings when constructing ``EditJob``.
        """

        sample_ids = tuple(dict.fromkeys(frame.mapping.image_id for frame in frames))

        def decode(payload: dict, ref: JobRef, labels: list[dict]) -> tuple[CvatBinding, ...]:
            job = EditJob(ref, tuple(frame.mapping for frame in frames))
            return decode_edit_bindings(payload, frames, job, labels, policy)

        return self._prepare_frames(
            task_id,
            frames,
            user_id,
            sample_ids,
            lambda labels: encode_edit_annotations(frames, labels, policy, mapped=True),
            decode,
        )

    def fetch_annotations(self, job: EditJob, policy: AnnotationPolicy) -> tuple[EditFrameResult, ...]:
        """Fetch persistent annotations using the Job's verified frame order and task policy."""

        labels = self._labels(job.ref.task_id)
        response = self._service_request('GET', f'/api/jobs/{job.ref.job_id}/annotations')
        return decode_edit_annotations(self._json(response), job, labels, policy)

    def job_is_unfinished(self, ref: JobRef) -> bool:
        """Return whether the Job state differs from completed.

        Malformed states and operational failures raise ``PlatformError`` without response details.
        """
        path = f'/api/jobs/{ref.job_id}'
        job = self._json(self._service_request('GET', path))
        if not isinstance(job, dict) or not isinstance(job.get('state'), str):
            raise PlatformError(f'CVAT {path} returned an invalid state')
        return job['state'] != 'completed'

    def job_path(self, ref: JobRef) -> str:
        """Return the local same-origin CVAT UI path for a Job."""

        return f'/tasks/{ref.task_id}/jobs/{ref.job_id}'

    def current_user(self, cookie: str) -> int:
        """Return the CVAT user ID authenticated by one browser Cookie header.

        No service token, HTTPX cookie-jar value, or configured default authentication is sent. Authentication and
        permission failures raise ``PlatformAccessError``; other CVAT and transport failures raise ``PlatformError``.
        """

        response = self._browser_request('GET', '/api/users/self', cookie=cookie)
        return self._integer_field(self._json(response), 'id', '/api/users/self')

    def login(self, username: str, password: str) -> tuple[str, ...]:
        """Create a CVAT browser session and return only session and CSRF ``Set-Cookie`` values.

        The REST token in CVAT's response body is ignored. Credentials are used only for this request and are never
        included in adapter errors. Rejected credentials raise ``PlatformAccessError``; operational failures raise
        ``PlatformError``.
        """

        response = self._browser_request(
            'POST',
            '/api/auth/login',
            access_error_statuses=(400, 401, 403),
            json={'username': username, 'password': password},
        )
        return self._session_cookies(response)

    def logout(self, cookie: str, csrf: str) -> tuple[str, ...]:
        """End one CVAT browser session and return only session and CSRF cookie updates.

        An absent or expired session raises ``PlatformAccessError``; operational failures raise ``PlatformError``.
        """

        response = self._browser_request('POST', '/api/auth/logout', cookie=cookie, csrf=csrf)
        return self._session_cookies(response)

    def _prepare_frames(
        self,
        task_id: int,
        frames: tuple[EditFrame, ...],
        user_id: int,
        sample_ids: tuple[str, ...],
        encode: Callable[[list[dict]], dict],
        decode_bindings: _BindingDecoder,
    ) -> PreparedJob:
        deadline = time.monotonic() + _PREPARE_TIMEOUT_SECONDS
        filenames = self._upload_filenames(frames)
        task = self._json(
            self._service_request('GET', f'/api/tasks/{task_id}', deadline=deadline, deadline_task_id=task_id)
        )
        if not isinstance(task, dict):
            raise PlatformError(f'CVAT /api/tasks/{task_id} returned an invalid size')
        # CVAT omits data-derived fields until the Task has an attached Data record.
        jobs = task.get('jobs')
        unattached = (
            'size' not in task
            and 'data' not in task
            and task.get('id') == task_id
            and task.get('mode') == ''
            and isinstance(jobs, dict)
            and type(jobs.get('count')) is int
            and jobs['count'] == 0
        )
        size = 0 if unattached else self._integer_field(task, 'size', f'/api/tasks/{task_id}')
        if size != 0:
            raise PlatformError(f'CVAT task {task_id} is not fresh and cannot be prepared')

        request_id = self._upload_images(task_id, frames, filenames, deadline)
        self._wait_for_request(task_id, request_id, deadline)
        self._verify_frames(task_id, frames, filenames, deadline)
        job_id = self._wait_for_job(task_id, len(frames), deadline)
        ref = JobRef(task_id, job_id, sample_ids)
        labels = self._labels(task_id, deadline=deadline)
        annotations = encode(labels)
        response = self._service_request(
            'PUT', f'/api/jobs/{job_id}/annotations', json=annotations, deadline=deadline, deadline_task_id=task_id
        )
        payload = self._json(response)
        bindings = decode_bindings(payload, ref, labels)
        self._service_request(
            'PATCH', f'/api/jobs/{job_id}', json={'assignee': user_id}, deadline=deadline, deadline_task_id=task_id
        )
        return PreparedJob(ref, bindings)

    def _upload_images(
        self, task_id: int, images: tuple[EditFrame, ...], filenames: tuple[str, ...], deadline: float
    ) -> str:
        with ExitStack() as stack:
            files = [
                (
                    f'client_files[{index}]',
                    (filename, stack.enter_context(image.image_path.open('rb')), 'application/octet-stream'),
                )
                for index, (image, filename) in enumerate(zip(images, filenames, strict=True))
            ]
            response = self._service_request(
                'POST',
                f'/api/tasks/{task_id}/data',
                data={'image_quality': '100', 'sorting_method': 'lexicographical'},
                files=files,
                deadline=deadline,
                deadline_task_id=task_id,
            )
        payload = self._json(response)
        request_id = payload.get('rq_id') if isinstance(payload, dict) else None
        if not isinstance(request_id, str) or not request_id:
            raise PlatformError(f'CVAT /api/tasks/{task_id}/data returned an invalid request ID')
        return request_id

    def _wait_for_request(self, task_id: int, request_id: str, deadline: float) -> None:
        request_path = f'/api/requests/{quote(request_id, safe="")}'
        while True:
            response = self._service_request('GET', request_path, deadline=deadline, deadline_task_id=task_id)
            payload = self._json(response)
            status = payload.get('status') if isinstance(payload, dict) else None
            if status == 'finished':
                return
            if status == 'failed':
                raise PlatformError(f'CVAT data preparation failed for task {task_id}')
            if status not in {'queued', 'started'}:
                raise PlatformError(f'CVAT {request_path} returned an invalid request status')
            self._poll_pause(task_id, deadline)

    def _verify_frames(
        self, task_id: int, images: tuple[EditFrame, ...], filenames: tuple[str, ...], deadline: float
    ) -> None:
        path = f'/api/tasks/{task_id}/data/meta'
        metadata = self._json(self._service_request('GET', path, deadline=deadline, deadline_task_id=task_id))
        if not isinstance(metadata, dict):
            raise PlatformError(f'CVAT {path} returned invalid frame metadata')
        frames = metadata.get('frames')
        if (
            metadata.get('size') != len(images)
            or metadata.get('start_frame') != 0
            or metadata.get('stop_frame') != len(images) - 1
            or metadata.get('deleted_frames') not in (None, [])
            or not isinstance(frames, list)
            or len(frames) != len(images)
        ):
            raise PlatformError(f'CVAT task {task_id} frame metadata does not match the uploaded images')
        for frame, image, filename in zip(frames, images, filenames, strict=True):
            if not isinstance(frame, dict) or (frame.get('name'), frame.get('width'), frame.get('height')) != (
                filename,
                image.width,
                image.height,
            ):
                raise PlatformError(f'CVAT task {task_id} frame mapping does not match the uploaded images')

    def _wait_for_job(self, task_id: int, frame_count: int, deadline: float) -> int:
        while True:
            jobs = self._page_results(
                '/api/jobs', params={'task_id': task_id}, deadline=deadline, deadline_task_id=task_id
            )
            annotation_jobs = [job for job in jobs if isinstance(job, dict) and job.get('type') == 'annotation']
            if annotation_jobs:
                if len(annotation_jobs) != 1:
                    raise PlatformError(f'CVAT task {task_id} must have exactly one annotation Job')
                job = annotation_jobs[0]
                expected = (task_id, 0, frame_count - 1, frame_count)
                actual = (job.get('task_id'), job.get('start_frame'), job.get('stop_frame'), job.get('frame_count'))
                if actual != expected:
                    raise PlatformError(f'CVAT task {task_id} annotation Job does not cover every frame exactly once')
                return self._integer_field(job, 'id', '/api/jobs')
            self._poll_pause(task_id, deadline)

    def _labels(self, task_id: int, *, deadline: float | None = None) -> list[dict]:
        return self._page_results(
            '/api/labels', params={'task_id': task_id}, deadline=deadline, deadline_task_id=task_id
        )

    def _page_results(
        self, path: str, *, params: dict[str, int], deadline: float | None, deadline_task_id: int
    ) -> list[dict]:
        results: list[dict] = []
        target: str = path
        query: dict[str, int] | None = params
        while True:
            payload = self._json(
                self._service_request('GET', target, params=query, deadline=deadline, deadline_task_id=deadline_task_id)
            )
            if not isinstance(payload, dict) or not isinstance(payload.get('results'), list):
                raise PlatformError(f'CVAT {path} returned an invalid page')
            results.extend(payload['results'])
            next_url = payload.get('next')
            if next_url is None:
                return results
            if not isinstance(next_url, str):
                raise PlatformError(f'CVAT {path} returned an invalid pagination URL')
            parsed = self._resolve_url(next_url)
            if parsed.path != path:
                raise PlatformError(f'CVAT {path} returned an unsafe pagination URL')
            target = str(parsed)
            query = None

    def _poll_pause(self, task_id: int, deadline: float) -> None:
        remaining = self._remaining(task_id, deadline)
        time.sleep(min(_POLL_INTERVAL_SECONDS, remaining))

    def _remaining(self, task_id: int, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PlatformError(f'CVAT task {task_id} preparation exceeded the 120 seconds deadline')
        return remaining

    def _service_request(
        self, method: str, target: str, *, deadline: float | None = None, deadline_task_id: int = 0, **kwargs
    ) -> httpx.Response:
        headers = {'Accept': 'application/vnd.cvat+json', 'Authorization': f'Token {self._service_token}'}
        return self._request(
            method,
            target,
            headers=headers,
            deadline=deadline,
            deadline_task_id=deadline_task_id,
            access_errors=False,
            **kwargs,
        )

    def _browser_request(
        self,
        method: str,
        target: str,
        *,
        cookie: str | None = None,
        csrf: str | None = None,
        access_error_statuses: tuple[int, ...] = (401, 403),
        **kwargs,
    ) -> httpx.Response:
        headers = {'Accept': 'application/vnd.cvat+json'}
        if cookie is not None:
            headers['Cookie'] = cookie
        if csrf is not None:
            headers['X-CSRFToken'] = csrf
        return self._request(
            method, target, headers=headers, access_errors=True, access_error_statuses=access_error_statuses, **kwargs
        )

    def _request(
        self,
        method: str,
        target: str,
        *,
        headers: dict[str, str],
        access_errors: bool,
        access_error_statuses: tuple[int, ...] = (401, 403),
        deadline: float | None = None,
        deadline_task_id: int = 0,
        **kwargs,
    ) -> httpx.Response:
        url = self._resolve_url(target)
        timeout = _REQUEST_TIMEOUT_SECONDS
        if deadline is not None:
            timeout = min(timeout, self._remaining(deadline_task_id, deadline))
        try:
            request = httpx.Request(
                method, url, headers=headers, extensions={'timeout': httpx.Timeout(timeout).as_dict()}, **kwargs
            )
            response = self._http.send(request, auth=None, follow_redirects=False)
        except httpx.HTTPError:
            raise PlatformError(f'CVAT {method} {url.path} failed') from None
        if access_errors and response.status_code in access_error_statuses:
            raise PlatformAccessError(f'CVAT {method} {url.path} denied access ({response.status_code})')
        if not 200 <= response.status_code < 300:
            raise PlatformError(f'CVAT {method} {url.path} failed with status {response.status_code}')
        return response

    def _resolve_url(self, target: str) -> httpx.URL:
        parsed = httpx.URL(target)
        url = self._base_url.join(parsed) if parsed.is_relative_url else parsed
        if (url.scheme, url.host, url.port) != self._origin or not url.path.startswith('/api/'):
            raise PlatformError('CVAT request or pagination URL is outside the configured API origin')
        return url

    @staticmethod
    def _json(response: httpx.Response) -> object:
        try:
            return response.json()
        except ValueError as error:
            raise PlatformError(f'CVAT {response.request.url.path} returned invalid JSON') from error

    @staticmethod
    def _integer_field(payload: object, field: str, path: str) -> int:
        value = payload.get(field) if isinstance(payload, dict) else None
        if isinstance(value, bool) or not isinstance(value, int):
            raise PlatformError(f'CVAT {path} returned an invalid {field}')
        return value

    @staticmethod
    def _upload_filenames(images: tuple[EditFrame, ...]) -> tuple[str, ...]:
        return tuple(f'{index:08d}{Path(image.image_path).suffix.lower()}' for index, image in enumerate(images))

    @staticmethod
    def _session_cookies(response: httpx.Response) -> tuple[str, ...]:
        allowed = {'csrftoken', 'sessionid'}
        return tuple(
            value
            for value in response.headers.get_list('set-cookie')
            if value.partition('=')[0].strip().casefold() in allowed
        )
