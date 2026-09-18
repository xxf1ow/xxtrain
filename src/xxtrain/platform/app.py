from __future__ import annotations

import secrets
import shutil
from dataclasses import asdict
from hmac import compare_digest
from importlib import resources
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict
from python_multipart.exceptions import MultipartParseError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from xxtrain.business_tasks import MODEL_TARGETS, point_task_definition
from xxtrain.integrations.cvat.client import CvatClient
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import (
    PlatformAccessError,
    PlatformConflictError,
    PlatformError,
    TargetValidationError,
    WorkspaceView,
)
from xxtrain.platform.service import AnnotationService
from xxtrain.platform.training_contracts import TrainingRunView
from xxtrain.platform.training_service import TrainingService

_CSRF_COOKIE = 'xxtrain_csrf'
_CSRF_HEADER = 'x-xtrain-csrf'
_OPERATIONAL_ERROR = '平台暂时无法完成操作，请重试。'
_TARGET_NAMES = {'detect': '检测', 'classify': '分类', 'segment': '分割'}
_TARGET_IDS = frozenset(_TARGET_NAMES)


class _EmptyBody(BaseModel):
    model_config = ConfigDict(extra='forbid')


class _LoginBody(BaseModel):
    model_config = ConfigDict(extra='forbid')

    username: str
    password: str


def create_app(
    config: WorkspaceConfig,
    service: AnnotationService,
    cvat: CvatClient,
    *,
    training_service: TrainingService | None = None,
) -> FastAPI:
    """Create the HTTP application for one configured workspace.

    Authentication delegates to the supplied CVAT browser-session adapter. Routes distinguish missing or rejected
    sessions (401), authenticated workspace-owner denial (403), invalid request objects (422), and operational
    workflow or backend failures (502). The caller owns ``service``, ``cvat``, and the optional ``training_service``
    and must close their external resources.
    """
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    web = resources.files('xxtrain.platform').joinpath('web')
    page = web.joinpath('index.html').read_text(encoding='utf-8')
    training_page = web.joinpath('training.html').read_text(encoding='utf-8')
    style = web.joinpath('style.css').read_text(encoding='utf-8')
    script = web.joinpath('app.js').read_text(encoding='utf-8')
    training_script = web.joinpath('training.js').read_text(encoding='utf-8')

    def operational_error() -> HTTPException:
        return HTTPException(status.HTTP_502_BAD_GATEWAY, _OPERATIONAL_ERROR)

    def conflict_error() -> HTTPException:
        return HTTPException(status.HTTP_409_CONFLICT, '现场当前有操作或训练任务正在进行，请稍后重试。')

    def browser_cookie(request: Request) -> str:
        values = []
        for name in ('sessionid', 'csrftoken'):
            value = request.cookies.get(name)
            if value:
                values.append(f'{name}={value}')
        return '; '.join(values)

    def authenticated_user(request: Request) -> int:
        cookie = browser_cookie(request)
        if not request.cookies.get('sessionid'):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, '请先登录。')
        try:
            return cvat.current_user(cookie)
        except PlatformAccessError:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, '登录已失效，请重新登录。') from None
        except PlatformError:
            raise operational_error() from None

    def write_request(request: Request) -> None:
        origin = request.headers.get('origin')
        parsed = urlsplit(origin) if origin else None
        request_host = request.headers.get('host', '').casefold()
        if (
            parsed is None
            or parsed.scheme.casefold() != request.url.scheme.casefold()
            or parsed.netloc.casefold() != request_host
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise HTTPException(status.HTTP_403_FORBIDDEN, '请求来源无效。')
        cookie_token = request.cookies.get(_CSRF_COOKIE, '')
        header_token = request.headers.get(_CSRF_HEADER, '')
        if not cookie_token or not header_token or not compare_digest(cookie_token, header_token):
            raise HTTPException(status.HTTP_403_FORBIDDEN, '请求令牌无效，请刷新页面。')

    def workspace_payload(view: WorkspaceView) -> dict[str, object]:
        availability = dict(MODEL_TARGETS)
        target_facts = {target.id: asdict(target) for target in view.targets}
        return {
            **asdict(view),
            'task': {'id': 'point', 'name': 'Point'},
            'targets': [
                {
                    **target_facts.get(target, {'id': target}),
                    'name': _TARGET_NAMES[target],
                    'available': availability[target],
                }
                for target in _TARGET_NAMES
            ],
        }

    def target_id(target: str) -> str:
        if target not in _TARGET_IDS:
            raise HTTPException(status.HTTP_404_NOT_FOUND, '未知模型目标。')
        return target

    def validation_response(error: TargetValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT, content={'detail': str(error), 'annotation_url': error.annotation_url}
        )

    def view_for(user_id: int) -> WorkspaceView:
        try:
            return service.view(user_id)
        except PlatformAccessError:
            raise HTTPException(status.HTTP_403_FORBIDDEN, '无权访问此现场。') from None
        except (OSError, ValueError, PlatformError):
            raise operational_error() from None

    def training_payload(view: TrainingRunView) -> dict[str, object]:
        run = view.run
        execution = view.execution
        training = point_task_definition().step(run.target).training
        if training is None:
            raise ValueError(f'Target does not define training: {run.target!r}')
        return {
            'id': run.id,
            'workspace_id': run.workspace_id,
            'workspace_name': run.workspace_name,
            'target': run.target,
            'metric_name': training.metric_name,
            'submitted_at': run.submitted_at,
            'execution': None
            if execution is None
            else {
                'status': execution.status,
                'active': execution.active,
                'epoch': execution.epoch,
                'total_epochs': execution.total_epochs,
                'elapsed_seconds': execution.elapsed_seconds,
                'metric': execution.metric,
                'download_ready': execution.download_ready,
                'detail': execution.detail,
            },
        }

    def full_workspace_payload(user_id: int, view: WorkspaceView | None = None) -> dict[str, object]:
        if training_service is None:
            return workspace_payload(view if view is not None else view_for(user_id))
        state = training_service.workspace_view(user_id)
        payload = workspace_payload(state['workspace'])
        payload['editing_locked'] = not state['editable']
        payload['training_enabled'] = True
        payload['training'] = {
            target: None if run is None else training_payload(run) for target, run in state['training'].items()
        }
        return payload

    def training_access(error: PlatformAccessError) -> HTTPException:
        return HTTPException(status.HTTP_403_FORBIDDEN, '无权访问此训练任务。')

    def training_failure(error: Exception) -> HTTPException:
        if isinstance(error, PlatformConflictError):
            return conflict_error()
        return operational_error()

    @app.get('/platform/', response_class=HTMLResponse)
    def platform_page(request: Request) -> Response:
        response = HTMLResponse(page)
        if not request.cookies.get(_CSRF_COOKIE):
            response.set_cookie(
                _CSRF_COOKIE, secrets.token_urlsafe(32), httponly=False, samesite='strict', path='/platform/'
            )
        return response

    @app.get('/platform/training/', response_class=HTMLResponse)
    def training_page_route(request: Request) -> Response:
        response = HTMLResponse(training_page)
        if not request.cookies.get(_CSRF_COOKIE):
            response.set_cookie(
                _CSRF_COOKIE, secrets.token_urlsafe(32), httponly=False, samesite='strict', path='/platform/'
            )
        return response

    @app.get('/platform/style.css', response_class=PlainTextResponse)
    def platform_style() -> Response:
        return Response(style, media_type='text/css')

    @app.get('/platform/app.js', response_class=PlainTextResponse)
    def platform_script() -> Response:
        return Response(script, media_type='text/javascript')

    @app.get('/platform/training.js', response_class=PlainTextResponse)
    def training_page_script() -> Response:
        return Response(training_script, media_type='text/javascript')

    @app.post('/platform/api/login', status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(write_request)])
    def login(body: _LoginBody) -> Response:
        try:
            cookies = cvat.login(body.username, body.password)
        except PlatformAccessError:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, '用户名或密码不正确。') from None
        except PlatformError:
            raise operational_error() from None
        response = Response(status_code=status.HTTP_204_NO_CONTENT)
        for cookie in cookies:
            response.headers.append('set-cookie', cookie)
        return response

    @app.post('/platform/api/logout', status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(write_request)])
    def logout(request: Request, body: _EmptyBody) -> Response:
        authenticated_user(request)
        csrf = request.cookies.get('csrftoken')
        if not csrf:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, '登录已失效，请重新登录。')
        try:
            cookies = cvat.logout(browser_cookie(request), csrf)
        except PlatformAccessError:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, '登录已失效，请重新登录。') from None
        except PlatformError:
            raise operational_error() from None
        response = Response(status_code=status.HTTP_204_NO_CONTENT)
        for cookie in cookies:
            response.headers.append('set-cookie', cookie)
        return response

    @app.get('/platform/api/session')
    def session(request: Request) -> dict[str, object]:
        return {'authenticated': True, 'user_id': authenticated_user(request)}

    @app.get('/platform/api/workspace')
    def workspace(request: Request) -> dict[str, object]:
        user_id = authenticated_user(request)
        try:
            return full_workspace_payload(user_id)
        except PlatformAccessError:
            raise HTTPException(status.HTTP_403_FORBIDDEN, '无权访问此现场。') from None
        except (OSError, ValueError, PlatformError):
            raise operational_error() from None

    def stage_uploads(user_id: int, images: list[UploadFile]) -> WorkspaceView:
        staging = config.runtime_dir / 'staging'
        staging.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=staging) as directory:
            staged = []
            for image in images:
                suffix = Path(image.filename or '').suffix.lower()
                if not suffix[1:].isalnum():
                    suffix = ''
                path = Path(directory) / f'{secrets.token_hex(16)}{suffix}'
                with path.open('wb') as destination:
                    shutil.copyfileobj(image.file, destination)
                staged.append(path)
            return service.upload(user_id, tuple(staged))

    @app.post('/platform/api/images', dependencies=[Depends(write_request)])
    async def upload_images(request: Request) -> dict[str, object]:
        user_id = authenticated_user(request)
        if user_id != config.owner_user_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, '无权访问此现场。')
        try:
            form = await request.form()
        except (StarletteHTTPException, MultipartParseError, OSError, ValueError):
            raise operational_error() from None
        try:
            try:
                parts = form.multi_items()
                if not parts or any(name != 'images' or not isinstance(value, UploadFile) for name, value in parts):
                    raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, '请选择要上传的图片。')
                view = await run_in_threadpool(stage_uploads, user_id, form.getlist('images'))
                return full_workspace_payload(user_id, view)
            finally:
                await form.close()
        except PlatformAccessError:
            raise HTTPException(status.HTTP_403_FORBIDDEN, '无权访问此现场。') from None
        except PlatformConflictError:
            raise conflict_error() from None
        except (OSError, ValueError, PlatformError):
            raise operational_error() from None

    @app.post('/platform/api/detection/start', dependencies=[Depends(write_request)])
    def start_annotation(request: Request, body: _EmptyBody) -> dict[str, str]:
        user_id = authenticated_user(request)
        try:
            annotation_url = service.begin_detection(user_id)
        except PlatformAccessError:
            raise HTTPException(status.HTTP_403_FORBIDDEN, '无权访问此现场。') from None
        except PlatformConflictError:
            raise conflict_error() from None
        except (OSError, ValueError, PlatformError):
            raise operational_error() from None
        return {'annotation_url': annotation_url}

    @app.post('/platform/api/detection/sync', dependencies=[Depends(write_request)])
    def sync_annotation(request: Request, body: _EmptyBody) -> Response:
        user_id = authenticated_user(request)
        try:
            view = service.sync_detection(user_id)
        except TargetValidationError as error:
            return validation_response(error)
        except PlatformAccessError:
            raise HTTPException(status.HTTP_403_FORBIDDEN, '无权访问此现场。') from None
        except PlatformConflictError:
            raise conflict_error() from None
        except (OSError, ValueError, PlatformError):
            raise operational_error() from None
        return JSONResponse(content=full_workspace_payload(user_id, view))

    @app.post('/platform/api/detection/cache', dependencies=[Depends(write_request)])
    def generate_detection_cache(request: Request, body: _EmptyBody) -> dict[str, object]:
        user_id = authenticated_user(request)
        try:
            view = service.generate_detection_cache(user_id)
        except PlatformAccessError:
            raise HTTPException(status.HTTP_403_FORBIDDEN, '无权访问此现场。') from None
        except PlatformConflictError:
            raise conflict_error() from None
        except (OSError, ValueError, PlatformError):
            raise operational_error() from None
        return full_workspace_payload(user_id, view)

    @app.post('/platform/api/targets/{target}/start', dependencies=[Depends(write_request)])
    def start_target(request: Request, target: str, body: _EmptyBody) -> dict[str, str]:
        user_id = authenticated_user(request)
        target = target_id(target)
        try:
            annotation_url = service.begin_target(user_id, target)
        except PlatformAccessError:
            raise HTTPException(status.HTTP_403_FORBIDDEN, '无权访问此现场。') from None
        except PlatformConflictError:
            raise conflict_error() from None
        except (OSError, ValueError, PlatformError):
            raise operational_error() from None
        return {'annotation_url': annotation_url}

    @app.post('/platform/api/targets/{target}/sync', dependencies=[Depends(write_request)])
    def sync_target(request: Request, target: str, body: _EmptyBody) -> Response:
        user_id = authenticated_user(request)
        target = target_id(target)
        try:
            view = service.sync_target(user_id, target)
        except TargetValidationError as error:
            return validation_response(error)
        except PlatformAccessError:
            raise HTTPException(status.HTTP_403_FORBIDDEN, '无权访问此现场。') from None
        except PlatformConflictError:
            raise conflict_error() from None
        except (OSError, ValueError, PlatformError):
            raise operational_error() from None
        return JSONResponse(content=full_workspace_payload(user_id, view))

    @app.post('/platform/api/targets/{target}/cache', dependencies=[Depends(write_request)])
    def generate_target_cache(request: Request, target: str, body: _EmptyBody) -> dict[str, object]:
        user_id = authenticated_user(request)
        target = target_id(target)
        try:
            view = service.generate_target_cache(user_id, target)
        except PlatformAccessError:
            raise HTTPException(status.HTTP_403_FORBIDDEN, '无权访问此现场。') from None
        except PlatformConflictError:
            raise conflict_error() from None
        except (OSError, ValueError, PlatformError):
            raise operational_error() from None
        return full_workspace_payload(user_id, view)

    if training_service is not None:

        @app.post('/platform/api/targets/{target}/train', dependencies=[Depends(write_request)])
        def submit_training(request: Request, target: str, body: _EmptyBody) -> dict[str, object]:
            user_id = authenticated_user(request)
            target = target_id(target)
            try:
                view = training_service.submit(user_id, target)
            except PlatformAccessError as error:
                raise training_access(error) from None
            except (OSError, ValueError, PlatformError) as error:
                raise training_failure(error) from None
            return {
                'run_id': view.run.id,
                'training_url': f'/platform/training/?run={view.run.id}',
                'run': training_payload(view),
            }

        @app.get('/platform/api/training-runs')
        def list_training(request: Request) -> list[dict[str, object]]:
            try:
                return [training_payload(view) for view in training_service.list_runs(authenticated_user(request))]
            except PlatformAccessError as error:
                raise training_access(error) from None
            except (OSError, ValueError, PlatformError) as error:
                raise training_failure(error) from None

        @app.get('/platform/api/training-runs/{run_id}')
        def get_training(request: Request, run_id: UUID) -> dict[str, object]:
            try:
                return training_payload(training_service.get_run(authenticated_user(request), str(run_id)))
            except PlatformAccessError as error:
                raise training_access(error) from None
            except (OSError, ValueError, PlatformError) as error:
                raise training_failure(error) from None

        @app.post('/platform/api/training-runs/{run_id}/cancel', dependencies=[Depends(write_request)])
        def cancel_training(request: Request, run_id: UUID, body: _EmptyBody) -> dict[str, object]:
            try:
                return training_payload(training_service.cancel(authenticated_user(request), str(run_id)))
            except PlatformAccessError as error:
                raise training_access(error) from None
            except (OSError, ValueError, PlatformError) as error:
                raise training_failure(error) from None

        @app.get('/platform/api/training-runs/{run_id}/download', response_class=FileResponse)
        def download_training(request: Request, run_id: UUID) -> FileResponse:
            try:
                download = training_service.download(authenticated_user(request), str(run_id))
            except PlatformAccessError as error:
                raise training_access(error) from None
            except (OSError, ValueError, PlatformError) as error:
                raise training_failure(error) from None
            return FileResponse(download.path, filename=download.filename, media_type=download.media_type)

    return app
