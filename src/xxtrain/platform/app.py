from __future__ import annotations

import secrets
from dataclasses import asdict
from hmac import compare_digest
from importlib import resources
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict

from xxtrain.business_tasks import MODEL_TARGETS
from xxtrain.integrations.cvat.client import CvatClient
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import PlatformAccessError, PlatformError, WorkspaceView
from xxtrain.platform.service import AnnotationService

_CSRF_COOKIE = 'xxtrain_csrf'
_CSRF_HEADER = 'x-xtrain-csrf'
_OPERATIONAL_ERROR = '平台暂时无法完成操作，请重试。'
_TARGET_NAMES = {'detect': '检测', 'classify': '分类', 'segment': '分割'}


class _EmptyBody(BaseModel):
    model_config = ConfigDict(extra='forbid')


class _LoginBody(BaseModel):
    model_config = ConfigDict(extra='forbid')

    username: str
    password: str


def create_app(config: WorkspaceConfig, service: AnnotationService, cvat: CvatClient) -> FastAPI:
    """Create the HTTP application for one configured workspace.

    Authentication delegates to the supplied CVAT browser-session adapter. Routes distinguish missing or rejected
    sessions (401), authenticated workspace-owner denial (403), invalid request objects (422), and operational
    workflow or CVAT failures (502). The caller owns ``service`` and ``cvat`` and must close their external resources.
    """
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    web = resources.files('xxtrain.platform').joinpath('web')
    page = web.joinpath('index.html').read_text(encoding='utf-8')
    style = web.joinpath('style.css').read_text(encoding='utf-8')
    script = web.joinpath('app.js').read_text(encoding='utf-8')

    def operational_error() -> HTTPException:
        return HTTPException(status.HTTP_502_BAD_GATEWAY, _OPERATIONAL_ERROR)

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
        return {
            **asdict(view),
            'task': {'id': 'point', 'name': 'Point'},
            'targets': [
                {'id': target, 'name': _TARGET_NAMES[target], 'available': available}
                for target, available in MODEL_TARGETS
            ],
        }

    def view_for(user_id: int) -> WorkspaceView:
        try:
            return service.view(user_id)
        except PlatformAccessError:
            raise HTTPException(status.HTTP_403_FORBIDDEN, '无权访问此现场。') from None
        except (OSError, ValueError, PlatformError):
            raise operational_error() from None

    @app.get('/platform/', response_class=HTMLResponse)
    def platform_page(request: Request) -> Response:
        response = HTMLResponse(page)
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
        return workspace_payload(view_for(authenticated_user(request)))

    @app.post('/platform/api/annotation/start', dependencies=[Depends(write_request)])
    def start_annotation(request: Request, body: _EmptyBody) -> dict[str, str]:
        user_id = authenticated_user(request)
        try:
            annotation_url = service.begin_detection(user_id)
        except PlatformAccessError:
            raise HTTPException(status.HTTP_403_FORBIDDEN, '无权访问此现场。') from None
        except (OSError, ValueError, PlatformError):
            raise operational_error() from None
        return {'annotation_url': annotation_url}

    @app.post('/platform/api/annotation/sync', dependencies=[Depends(write_request)])
    def sync_annotation(request: Request, body: _EmptyBody) -> dict[str, object]:
        user_id = authenticated_user(request)
        try:
            view = service.sync_detection(user_id)
        except PlatformAccessError:
            raise HTTPException(status.HTTP_403_FORBIDDEN, '无权访问此现场。') from None
        except (OSError, ValueError, PlatformError):
            raise operational_error() from None
        return workspace_payload(view)

    return app
