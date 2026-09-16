from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from contextlib import suppress
from importlib import resources
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

_MISSING_HTTP_DEPENDENCIES = tuple(name for name in ('fastapi', 'httpx') if find_spec(name) is None)
if _MISSING_HTTP_DEPENDENCIES:
    raise unittest.SkipTest(f'platform extra is required: {", ".join(_MISSING_HTTP_DEPENDENCIES)}')
else:
    from xxtrain.business_tasks.point import point_task_definition
    from xxtrain.platform.app import create_app
    from xxtrain.platform.config import WorkspaceConfig, load_config
    from xxtrain.workspace_data import WorkspaceData
    from xxtrain.workspace_data.repository import AnnotationRepository

    from .platform_fixture import FIXTURE_MARKER, create_fixture
    from .test_platform_http import AsgiTestClient, FakeCvat, FakeService

_LIVE_ENVIRONMENT = ('XXTRAIN_PLATFORM_URL', 'XXTRAIN_PLATFORM_TEST_USER', 'XXTRAIN_PLATFORM_TEST_PASSWORD')
_LIVE_VALUES = {name: os.environ.get(name, '') for name in _LIVE_ENVIRONMENT}
_LIVE_REQUESTED = any(_LIVE_VALUES.values())

if _LIVE_REQUESTED and all(_LIVE_VALUES.values()):
    try:
        from playwright.sync_api import Page, Route, expect, sync_playwright
    except ImportError as error:
        _PLAYWRIGHT_IMPORT_ERROR: ImportError | None = error
    else:
        _PLAYWRIGHT_IMPORT_ERROR = None
else:
    _PLAYWRIGHT_IMPORT_ERROR = None


class PlatformBrowserTest(unittest.TestCase):
    def setUp(self) -> None:
        config = WorkspaceConfig('line-3', '三号现场', 17, Path('workspace'), Path('runtime'), 'http://cvat.test')
        self.service = FakeService()
        self.cvat = FakeCvat()
        self.client = AsgiTestClient(create_app(config, self.service, self.cvat))
        self.addCleanup(self.client.close)

    def test_page_serves_the_point_utility_layout_and_local_assets(self) -> None:
        page = self.client.get('/platform/')
        style = self.client.get('/platform/style.css')
        script = self.client.get('/platform/app.js')

        self.assertEqual(200, page.status_code)
        self.assertEqual('text/html; charset=utf-8', page.headers['content-type'])
        self.assertIn('<main', page.text)
        self.assertIn('id="login-panel"', page.text)
        self.assertIn('method="post" action="/platform/api/login"', page.text)
        self.assertIn('id="workspace-panel"', page.text)
        self.assertIn('aria-live="polite"', page.text)
        self.assertIn('Point', page.text)
        self.assertIn('检测', page.text)
        self.assertIn('分类', page.text)
        self.assertIn('分割', page.text)
        self.assertIn('href="/platform/style.css"', page.text)
        self.assertIn('src="/platform/app.js"', page.text)
        self.assertNotIn('http://', page.text)
        self.assertNotIn('https://', page.text)
        self.assertEqual(200, style.status_code)
        self.assertTrue(style.headers['content-type'].startswith('text/css'))
        self.assertEqual(200, script.status_code)
        self.assertTrue(script.headers['content-type'].startswith('text/javascript'))

    def run_page(
        self, actions: str = '', *, returned: bool = False, ready: bool = False, sync_failure: bool = False
    ) -> dict[str, object]:
        script = self.client.get('/platform/app.js')
        self.assertEqual(200, script.status_code)
        harness = f"""
(async () => {{
const calls = [];
const timers = new Map();
let timerId = 0;
globalThis.setTimeout = (callback) => {{ timers.set(++timerId, callback); return timerId; }};
globalThis.clearTimeout = (id) => timers.delete(id);
const elements = new Map();
const ids = {json.dumps(re.findall(r'id="([^"]+)"', self.client.get('/platform/').text))};
function makeElement(id) {{
  return {{
    id,
    hidden: false,
    disabled: false,
    textContent: '',
    value: '',
    files: [],
    listeners: {{}},
    dataset: {{}},
    classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
    addEventListener(name, callback) {{ this.listeners[name] = callback; }},
    setAttribute() {{}},
    removeAttribute() {{}},
    reset() {{}},
  }};
}}
ids.forEach((id) => elements.set(id, makeElement(id)));
let loaded;
globalThis.document = {{
  cookie: 'xxtrain_csrf=page-token',
  getElementById(id) {{
    return elements.get(id) || null;
  }},
  querySelectorAll() {{ return []; }},
  addEventListener(name, callback) {{ if (name === 'DOMContentLoaded') loaded = callback; }},
}};
let assigned = null;
let replaced = null;
globalThis.location = {{ search: {json.dumps('?returned=1' if returned else '')}, assign(url) {{ assigned = url; }} }};
globalThis.history = {{ replaceState(_state, _unused, url) {{
  replaced = url;
  location.search = new URL(url, 'http://testserver').search;
}} }};
const workspace = {{
  workspace_id: 'line-3', name: '三号现场', image_count: 10, annotated_image_count: 7, boxed_image_count: 7,
  can_generate_detection_cache: {json.dumps(ready)}, detection_cache_ready: false,
  task: {{id: 'point', name: 'Point'}},
  targets: [
    {{id: 'detect', name: '检测', available: true}},
    {{id: 'classify', name: '分类', available: false}},
    {{id: 'segment', name: '分割', available: false}},
  ],
}};
let release;
let hold = false;
let failure = {json.dumps(sync_failure)};
globalThis.fetch = async (url, options = {{}}) => {{
  const multipart = options.body instanceof FormData;
  calls.push({{
    url,
    method: options.method || 'GET',
    body: multipart ? Array.from(options.body.entries()).map(([key, file]) => [key, file.name]) : options.body || null,
    csrf: options.headers?.['X-XTrain-CSRF'] || null,
    contentType: options.headers?.['Content-Type'] || null,
  }});
  if (hold && options.method === 'POST') await new Promise((resolve) => {{ release = resolve; }});
  if (failure && options.method === 'POST') {{
    return {{ok: false, status: 502, json: async () => ({{detail: '平台暂时无法完成操作，请重试。'}})}};
  }}
  const body = url.endsWith('/session')
    ? {{authenticated: true, user_id: 17}}
    : url.endsWith('/sync') ? {{...workspace, image_count: 50, annotated_image_count: 50,
        boxed_image_count: 50, can_generate_detection_cache: true}}
    : url.endsWith('/images') ? {{...workspace, image_count: 12}}
    : url.endsWith('/cache') ? {{...workspace, detection_cache_ready: true}}
    : url.endsWith('/start') ? {{annotation_url: '/tasks/41/jobs/73'}} : workspace;
  return {{ok: true, status: 200, json: async () => body}};
}};
eval({json.dumps(script.text)});
await loaded();
const get = (id) => elements.get(id);
const snapshot = () => ({{
  images: get('image-count')?.textContent,
  annotated: get('annotated-image-count')?.textContent,
  totals: ['detect', 'classify', 'segment'].map((target) => get(`${{target}}-image-total`)?.textContent),
  cacheDisabled: get('cache-action')?.disabled,
  cacheText: get('cache-action')?.textContent,
  error: get('workspace-error')?.textContent,
  uploadProgress: get('upload-progress')?.textContent,
  uploadProgressHidden: get('upload-progress')?.hidden,
  notification: get('workspace-message')?.textContent,
  notificationHidden: get('workspace-message')?.hidden,
  disabled: ['primary-action', 'image-files', 'cache-action', 'logout-button', 'login-button']
    .map((id) => get(id)?.disabled),
}});
const before = snapshot();
{actions}
process.stdout.write(JSON.stringify({{calls, before, after: snapshot(), assigned, replaced}}));
}})().catch((error) => {{ console.error(error); process.exitCode = 1; }});
"""
        completed = subprocess.run(
            ['node', '-e', harness], text=True, encoding='utf-8', capture_output=True, check=False
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        return json.loads(completed.stdout)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_page_shows_derived_counts_and_disables_ineligible_cache(self) -> None:
        result = self.run_page()
        self.assertEqual('10', result['after']['images'])
        self.assertEqual('7', result['after']['annotated'])
        self.assertTrue(result['after']['cacheDisabled'])
        self.assertEqual(['10', '10', '10'], result['after']['totals'])
        self.assertFalse(result['after']['disabled'][1])
        page = self.client.get('/platform/').text
        self.assertNotIn('data-image-thumbnail', page)
        self.assertNotIn('id="status-value"', page)
        self.assertIn('type="file"', page)
        self.assertIn('multiple', page)
        self.assertNotIn('id="upload-button"', page)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_returned_page_syncs_counts_and_enables_cache(self) -> None:
        result = self.run_page(returned=True)
        self.assertEqual('/platform/api/detection/sync', result['calls'][-1]['url'])
        self.assertEqual('{}', result['calls'][-1]['body'])
        self.assertEqual('page-token', result['calls'][-1]['csrf'])
        self.assertEqual('50', result['after']['annotated'])
        self.assertFalse(result['after']['cacheDisabled'])
        self.assertEqual('/platform/', result['replaced'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_selecting_images_uploads_them_without_a_second_action(self) -> None:
        result = self.run_page("""
get('image-files').files = [new File(['first'], 'a.jpg'), new File(['second'], 'b.png')];
const upload = get('image-files').listeners.change();
await upload;
""")
        self.assertEqual(3, len(result['calls']))
        self.assertEqual('/platform/api/images', result['calls'][-1]['url'])
        self.assertEqual([['images', 'a.jpg'], ['images', 'b.png']], result['calls'][-1]['body'])
        self.assertEqual('page-token', result['calls'][-1]['csrf'])
        self.assertIsNone(result['calls'][-1]['contentType'])
        self.assertEqual('12', result['after']['images'])
        self.assertEqual(['12', '12', '12'], result['after']['totals'])
        self.assertEqual('7', result['after']['annotated'])
        self.assertFalse(result['after']['disabled'][0])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_upload_progress_stays_visible_until_completion_then_notification_expires(self) -> None:
        result = self.run_page("""
get('image-files').files = [new File(['first'], 'a.jpg')];
hold = true;
const upload = get('image-files').listeners.change();
before.uploading = snapshot();
release();
await upload;
before.completed = snapshot();
for (const callback of timers.values()) callback();
""")
        self.assertFalse(result['before']['uploading']['uploadProgressHidden'])
        self.assertIn('1 张', result['before']['uploading']['uploadProgress'])
        self.assertEqual([True] * 5, result['before']['uploading']['disabled'])
        self.assertTrue(result['before']['completed']['uploadProgressHidden'])
        self.assertFalse(result['before']['completed']['notificationHidden'])
        self.assertIn('12 张有效图片', result['before']['completed']['notification'])
        self.assertTrue(result['after']['notificationHidden'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_cache_action_renders_generated_result_and_blocks_repeated_click(self) -> None:
        result = self.run_page(
            """
await get('cache-action').listeners.click();
await get('cache-action').listeners.click();
""",
            ready=True,
        )
        self.assertFalse(result['before']['cacheDisabled'])
        self.assertTrue(result['after']['cacheDisabled'])
        self.assertEqual('训练缓存已生成', result['after']['cacheText'])
        self.assertEqual('/platform/api/detection/cache', result['calls'][-1]['url'])
        self.assertEqual('{}', result['calls'][-1]['body'])
        self.assertEqual(3, len(result['calls']))

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_start_navigates_to_the_server_annotation_url(self) -> None:
        result = self.run_page("await get('primary-action').listeners.click();")
        self.assertEqual('/platform/api/detection/start', result['calls'][-1]['url'])
        self.assertEqual('/tasks/41/jobs/73', result['assigned'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_failed_cache_keeps_counts_and_reenables_controls(self) -> None:
        result = self.run_page("failure = true; await get('cache-action').listeners.click();", ready=True)
        self.assertEqual('10', result['after']['images'])
        self.assertEqual('7', result['after']['annotated'])
        self.assertFalse(result['after']['cacheDisabled'])
        self.assertFalse(result['after']['disabled'][0])
        self.assertTrue(result['after']['error'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_failed_return_sync_blocks_workspace_actions_until_refresh(self) -> None:
        result = self.run_page(
            """
get('image-files').files = [new File(['first'], 'a.jpg')];
await get('image-files').listeners.change();
await get('primary-action').listeners.click();
await get('image-files').listeners.change();
await get('cache-action').listeners.click();
""",
            returned=True,
            ready=True,
            sync_failure=True,
        )
        self.assertEqual('10', result['after']['images'])
        self.assertEqual('7', result['after']['annotated'])
        self.assertIn('刷新页面', result['after']['error'])
        self.assertIsNone(result['replaced'])
        self.assertEqual([True, True, True, False, False], result['after']['disabled'])
        self.assertEqual(
            ['/platform/api/session', '/platform/api/workspace', '/platform/api/detection/sync'],
            [call['url'] for call in result['calls']],
        )
        self.assertIsNone(result['assigned'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_successful_return_sync_retry_restores_derived_controls(self) -> None:
        result = self.run_page(
            """
failure = false;
await loaded();
get('image-files').files = [new File(['first'], 'a.jpg')];
await get('image-files').listeners.change();
""",
            returned=True,
            ready=True,
            sync_failure=True,
        )
        self.assertEqual([True, True, True, False, False], result['before']['disabled'])
        self.assertEqual([False] * 5, result['after']['disabled'])
        self.assertEqual('7', result['after']['annotated'])
        self.assertEqual('/platform/', result['replaced'])
        self.assertEqual('', result['after']['error'])

    def test_deployment_example_uses_the_exact_workspace_schema_without_credentials(self) -> None:
        path = Path(__file__).parents[1] / 'deploy' / 'platform' / 'workspace.example.json'
        payload = json.loads(path.read_text(encoding='utf-8'))

        self.assertEqual(
            {'workspace_id', 'display_name', 'owner_user_id', 'workspace_dir', 'runtime_dir', 'cvat_internal_url'},
            set(payload),
        )
        config = load_config(path)
        self.assertEqual(payload['workspace_id'], config.workspace_id)
        self.assertEqual(payload['owner_user_id'], config.owner_user_id)

    def test_deployment_uses_the_cvat_ui_image_listener(self) -> None:
        root = Path(__file__).parents[1]
        dockerfile = (root / 'deploy' / 'platform' / 'cvat-ui.Dockerfile').read_text(encoding='utf-8')
        proxy = (root / 'deploy' / 'platform' / 'nginx.conf').read_text(encoding='utf-8')

        self.assertIn('EXPOSE 8000', dockerfile.splitlines())
        self.assertIn('proxy_pass http://xxtrain_cvat_ui:8000;', proxy)

    def run_plugin(self, active_control: str) -> dict[str, object]:
        try:
            plugin = (
                resources.files('xxtrain.integrations.cvat').joinpath('ui/return-plugin.js').read_text(encoding='utf-8')
            )
        except FileNotFoundError:
            self.fail('the packaged CVAT return plugin is missing')
        harness = f"""
(async () => {{
  const events = [];
  const created = [];
  function element(tag) {{
    const value = {{
      tag, children: [], parentElement: null, style: {{}}, dataset: {{}}, hidden: false, disabled: false,
      textContent: '', removed: false,
      setAttribute() {{}},
      append(...children) {{
        this.children.push(...children);
        children.forEach((child) => child.parentElement = this);
      }},
      before(node) {{ node.parentElement = this.parentElement; }},
      remove() {{ this.removed = true; }},
    }};
    created.push(value);
    return value;
  }}
  const body = element('body');
  const toolbar = element('toolbar');
  const menu = element('menu');
  menu.parentElement = toolbar;
  const root = element('root');
  let disconnected = false;
  globalThis.MutationObserver = class {{
    constructor(callback) {{ this.callback = callback; }}
    observe() {{}}
    disconnect() {{ disconnected = true; }}
  }};
  globalThis.document = {{
    body,
    createElement: element,
    querySelector(selector) {{ return selector === '.cvat-annotation-header-menu-button' ? menu : null; }},
    getElementById(id) {{ return id === 'root' ? root : null; }},
  }};
  let assigned = null;
  globalThis.location = {{
    pathname: '/tasks/41/jobs/73',
    assign(path) {{ assigned = path; events.push('assign'); }},
  }};
  const job = {{
    id: 73,
    taskId: 41,
    frames: {{save: async () => events.push('frames')}},
    annotations: {{save: async () => events.push('annotations')}},
  }};
  const store = {{getState: () => ({{annotation: {{
    job: {{instance: job}},
    annotations: {{saving: {{uploading: false}}}},
    canvas: {{activeControl: {json.dumps(active_control)}}},
  }}}})}};
  let registered = null;
  globalThis.window = {{
    cvatUI: {{registerComponent(factory) {{
      registered = factory({{
        store,
        dispatch(action) {{ events.push('dispatch'); return action; }},
        actionCreators: {{updateJobAsync(instance, payload) {{
          events.push(`update:${{payload.state}}`);
          return {{}};
        }}}},
        core: {{enums: {{JobState: {{COMPLETED: 'completed'}}}}}},
      }});
    }}}},
    addEventListener() {{}},
  }};
  globalThis.fetch = async (path, options) => {{
    events.push(`verify:${{path}}:${{options.credentials}}`);
    return {{ok: true, json: async () => ({{state: 'completed'}})}};
  }};
  eval({json.dumps(plugin)});
  const button = created.find((item) => item.tag === 'button');
  const alert = created.find((item) => item.tag === 'p');
  await button.onclick();
  registered.destructor();
  process.stdout.write(JSON.stringify({{
    events, assigned, alert: alert.textContent, alertHidden: alert.hidden,
    removed: created.find((item) => item.tag === 'aside').removed, disconnected,
  }}));
}})().catch((error) => {{ console.error(error); process.exitCode = 1; }});
"""
        completed = subprocess.run(
            ['node', '-e', harness], text=True, encoding='utf-8', capture_output=True, check=False
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        return json.loads(completed.stdout)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline plugin check')
    def test_cvat_plugin_saves_completes_verifies_and_returns_to_the_fixed_page(self) -> None:
        result = self.run_plugin('cursor')

        self.assertEqual(
            ['frames', 'annotations', 'update:completed', 'dispatch', 'verify:/api/jobs/73:same-origin', 'assign'],
            result['events'],
        )
        self.assertEqual('/platform/?returned=1', result['assigned'])
        self.assertTrue(result['removed'])
        self.assertTrue(result['disconnected'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline plugin check')
    def test_cvat_plugin_stays_on_the_job_while_a_shape_is_being_drawn(self) -> None:
        result = self.run_plugin('draw_rectangle')

        self.assertEqual([], result['events'])
        self.assertIsNone(result['assigned'])
        self.assertFalse(result['alertHidden'])
        self.assertIn('完成当前绘制或编辑', result['alert'])


class PlatformFixtureTest(unittest.TestCase):
    def test_fixture_is_isolated_and_records_sqlite_annotation_identity(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            receipt = create_fixture(Path(parent), owner_user_id=17, cvat_internal_url='http://cvat.test')

            root = Path(receipt['root'])
            self.assertIn('database_path', receipt)
            self.assertNotIn('annotation_path', receipt)
            database_path = Path(receipt['database_path'])
            self.assertEqual(FIXTURE_MARKER, receipt['marker'])
            self.assertEqual(root.parent, Path(parent))
            self.assertTrue(root.name.startswith('xxtrain-point-acceptance-'))
            self.assertEqual(root / 'workspace' / 'annotations.db', database_path)
            self.assertTrue(database_path.is_file())
            self.assertFalse((root / 'workspace' / 'annotations').exists())
            self.assertEqual(2, len(receipt['images']))
            for immutable in [*receipt['images'], receipt['baseline']]:
                path = Path(immutable['path'])
                self.assertTrue(path.is_relative_to(root))
                self.assertEqual(immutable['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

            workspace = WorkspaceData(root / 'workspace')
            self.assertEqual(
                [image['sample_id'] for image in receipt['images']], [image.sample_id for image in workspace.images()]
            )
            records = AnnotationRepository(database_path, point_task_definition()).annotations(step_key='detect')
            self.assertEqual(receipt['initial_annotation_ids'], [str(record.id) for record in records])
            self.assertEqual(receipt['images'][0]['sample_id'], records[0].image_id)
            self.assertEqual('tl', records[0].label)
            self.assertEqual(receipt['original_rectangle_points'], records[0].geometry)


@unittest.skipUnless(_LIVE_REQUESTED, 'real browser acceptance environment is not configured')
class PlatformLiveBrowserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        missing = [name for name, value in _LIVE_VALUES.items() if not value]
        if missing:
            raise RuntimeError(f'incomplete real browser acceptance configuration: {", ".join(missing)}')
        if _PLAYWRIGHT_IMPORT_ERROR is not None:
            raise RuntimeError(
                'platform-test extra is required for configured browser acceptance'
            ) from _PLAYWRIGHT_IMPORT_ERROR

        cls.receipt_path = Path.cwd() / '.superpowers' / 'platform-acceptance' / 'fixture.json'
        if not cls.receipt_path.is_file():
            raise RuntimeError(f'dedicated synthetic fixture receipt is missing: {cls.receipt_path}')
        cls.receipt = json.loads(cls.receipt_path.read_text(encoding='utf-8'))
        cls._validate_receipt()

        cls.artifact_dir = cls.receipt_path.parent / 'screenshots'
        cls.artifact_dir.mkdir(parents=True, exist_ok=True)
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls) -> None:
        with suppress(AttributeError):
            cls.browser.close()
        with suppress(AttributeError):
            cls.playwright.stop()

    @classmethod
    def _validate_receipt(cls) -> None:
        if cls.receipt.get('marker') != FIXTURE_MARKER:
            raise RuntimeError('fixture receipt is not an xxtrain Task 6 synthetic workspace')
        root = Path(cls.receipt.get('root', '')).resolve()
        if not root.is_dir() or not root.name.startswith('xxtrain-point-acceptance-'):
            raise RuntimeError('fixture root is not a generated Task 6 temporary directory')
        required_paths = [cls.receipt.get('database_path', ''), cls.receipt.get('baseline', {}).get('path', '')]
        required_paths.extend(item.get('path', '') for item in cls.receipt.get('images', []))
        if not required_paths or any(not Path(path).resolve().is_relative_to(root) for path in required_paths):
            raise RuntimeError('fixture receipt points outside its generated temporary directory')

    def _new_page(self) -> Page:
        context = self.browser.new_context(viewport={'width': 1440, 'height': 1000})
        self.addCleanup(context.close)
        page = context.new_page()
        page.set_default_timeout(180_000)
        return page

    def _login(self, page: Page) -> None:
        page.goto(_LIVE_VALUES['XXTRAIN_PLATFORM_URL'], wait_until='domcontentloaded')
        expect(page.get_by_role('heading', name='登录后继续现场标注')).to_be_visible()
        cookie_names = page.evaluate(
            "document.cookie.split(';').map((part) => part.trim().split('=', 1)[0]).filter(Boolean)"
        )
        self.assertIn('xxtrain_csrf', cookie_names)
        observed_request: dict[str, bool] = {}
        platform_origin = urlsplit(_LIVE_VALUES['XXTRAIN_PLATFORM_URL'])
        expected_origin = f'{platform_origin.scheme}://{platform_origin.netloc}'

        def observe_login(route: Route) -> None:
            headers = route.request.all_headers()
            cookies = dict(
                part.strip().partition('=')[::2] for part in headers.get('cookie', '').split(';') if '=' in part
            )
            observed_request.update(
                cookie_present='xxtrain_csrf' in cookies,
                header_present=bool(headers.get('x-xtrain-csrf')),
                token_matches=cookies.get('xxtrain_csrf') == headers.get('x-xtrain-csrf'),
                origin_matches=headers.get('origin') == expected_origin,
            )
            route.continue_()

        page.route('**/platform/api/login', observe_login, times=1)
        page.get_by_label('用户名').fill(_LIVE_VALUES['XXTRAIN_PLATFORM_TEST_USER'])
        page.get_by_label('密码').fill(_LIVE_VALUES['XXTRAIN_PLATFORM_TEST_PASSWORD'])
        page.get_by_role('button', name='登录', exact=True).click()
        page.wait_for_function(
            "!document.getElementById('workspace-panel').hidden || !document.getElementById('login-error').hidden"
        )
        if page.locator('#workspace-panel').is_hidden():
            error = page.locator('#login-error').text_content()
            page.get_by_label('用户名').fill('')
            page.get_by_label('密码').fill('')
            self.fail(f'login failed: {error}; request checks: {observed_request}')
        expect(page.locator('#workspace-name')).to_have_text(self.receipt['display_name'])
        expect(page.locator('[data-target="detect"]')).to_have_attribute('aria-disabled', 'false')
        expect(page.locator('[data-target="classify"]')).to_have_attribute('aria-disabled', 'true')
        expect(page.locator('[data-target="segment"]')).to_have_attribute('aria-disabled', 'true')

    def _assert_immutable_inputs(self) -> None:
        for immutable in [*self.receipt['images'], self.receipt['baseline']]:
            path = Path(immutable['path'])
            self.assertEqual(immutable['sha256'], hashlib.sha256(path.read_bytes()).hexdigest(), path)

    def _saved_detection_records(self):
        return AnnotationRepository(Path(self.receipt['database_path']), point_task_definition()).annotations(
            step_key='detect'
        )

    def test_detection_annotation_round_trip(self) -> None:
        page = self._new_page()
        self._login(page)
        expect(page.locator('#image-count')).to_have_text('2')
        expect(page.locator('#annotated-image-count')).to_have_text('1')
        expect(page.locator('#cache-action')).to_be_disabled()
        page.screenshot(path=self.artifact_dir / '01-platform-ready.png', full_page=True)

        with page.expect_navigation(wait_until='domcontentloaded'):
            page.get_by_role('button', name='检测标注', exact=True).click()
        first_job_url = page.url
        self.assertRegex(first_job_url, r'/tasks/\d+/jobs/\d+/?$')
        expect(page.locator('.cvat-header')).to_be_hidden()
        expect(page.locator('.cvat-canvas-container')).to_be_visible()
        expect(page.get_by_role('button', name='完成', exact=True)).to_be_visible()
        page.screenshot(path=self.artifact_dir / '02-cvat-first-job.png', full_page=True)

        existing_shape = page.locator('.cvat_canvas_shape').first
        expect(existing_shape).to_be_visible()
        bounds = existing_shape.bounding_box()
        self.assertIsNotNone(bounds)
        assert bounds is not None
        page.mouse.move(bounds['x'] + bounds['width'] / 2, bounds['y'] + bounds['height'] / 2)
        page.mouse.down()
        page.mouse.move(bounds['x'] + bounds['width'] / 2 + 30, bounds['y'] + bounds['height'] / 2 + 20, steps=5)
        page.mouse.up()

        page.locator('.cvat-draw-rectangle-control').click()
        page.locator('.cvat-draw-rectangle-popover').get_by_role('button', name='Shape', exact=True).click()
        canvas = page.locator('.cvat-canvas-container')
        canvas.click(position={'x': 250, 'y': 230})
        canvas.click(position={'x': 450, 'y': 390})

        annotation_pattern = re.compile(r'/api/jobs/\d+/annotations(?:/.*)?(?:\?.*)?$')
        write_failure = {'injected': False}

        def fail_first_annotation_write(route: Route) -> None:
            if not write_failure['injected'] and route.request.method in {'POST', 'PUT', 'PATCH', 'DELETE'}:
                write_failure['injected'] = True
                route.abort('connectionfailed')
            else:
                route.continue_()

        page.route(annotation_pattern, fail_first_annotation_write)
        page.get_by_role('button', name='完成', exact=True).click()
        expect(page.get_by_test_id('xxtrain-return-error')).to_have_text('CVAT 保存未完成，请留在本页重试。')
        self.assertEqual(first_job_url.rstrip('/'), page.url.rstrip('/'))
        page.screenshot(path=self.artifact_dir / '03-cvat-save-failure.png', full_page=True)
        self.assertTrue(write_failure['injected'])
        page.unroute(annotation_pattern, fail_first_annotation_write)

        with page.expect_navigation(wait_until='domcontentloaded'):
            page.get_by_role('button', name='完成', exact=True).click()
        expect(page.locator('#annotated-image-count')).to_have_text('2')
        expect(page).to_have_url(re.compile(r'/platform/$'))
        page.screenshot(path=self.artifact_dir / '04-first-save.png', full_page=True)

        records = self._saved_detection_records()
        rectangles = [record for record in records if record.kind == 'rectangle']
        self.assertEqual({'Point', 'tl'}, {record.label for record in rectangles})
        classified = next(record for record in rectangles if record.label == 'tl')
        initial_id = UUID(self.receipt['initial_annotation_ids'][0])
        self.assertEqual(initial_id, classified.id)
        self.assertEqual(self.receipt['images'][0]['sample_id'], classified.image_id)
        self.assertNotEqual(self.receipt['original_rectangle_points'], classified.geometry)
        added = next(record for record in rectangles if record.label == 'Point')
        self.assertNotIn(added.id, {UUID(value) for value in self.receipt['initial_annotation_ids']})
        self._assert_immutable_inputs()

        with page.expect_navigation(wait_until='domcontentloaded'):
            page.get_by_role('button', name='检测标注', exact=True).click()
        second_job_url = page.url
        self.assertNotEqual(first_job_url.rstrip('/'), second_job_url.rstrip('/'))
        expect(page.locator('.cvat-header')).to_be_hidden()
        expect(page.locator('.cvat-canvas-container')).to_be_visible()
        page.reload(wait_until='domcontentloaded')
        self.assertEqual(second_job_url.rstrip('/'), page.url.rstrip('/'))
        expect(page.locator('.cvat-header')).to_be_hidden()
        expect(page.locator('.cvat-canvas-container')).to_be_visible()

        shapes = page.locator('.cvat_canvas_shape')
        for remaining in range(shapes.count() - 1, -1, -1):
            shape = shapes.first
            bounds = shape.bounding_box()
            self.assertIsNotNone(bounds)
            assert bounds is not None
            page.mouse.click(bounds['x'] + bounds['width'] / 2, bounds['y'] + bounds['height'] / 2)
            page.keyboard.press('Delete')
            expect(shapes).to_have_count(remaining)

        page.route('**/platform/api/detection/sync', lambda route: route.abort('connectionfailed'), times=1)
        with page.expect_navigation(wait_until='domcontentloaded'):
            page.get_by_role('button', name='完成', exact=True).click()
        expect(page.locator('#workspace-error')).to_contain_text('刷新页面')
        expect(page.locator('#annotated-image-count')).to_have_text('1')
        page.screenshot(path=self.artifact_dir / '05-platform-sync-failure.png', full_page=True)

        page.reload(wait_until='domcontentloaded')
        expect(page.locator('#annotated-image-count')).to_have_text('2')
        expect(page).to_have_url(re.compile(r'/platform/$'))
        page.screenshot(path=self.artifact_dir / '06-empty-detection-save.png', full_page=True)
        final_records = self._saved_detection_records()
        self.assertEqual({'negative'}, {record.kind for record in final_records})
        self.assertTrue(all(record.geometry is None for record in final_records))
        self.assertTrue({classified.id, added.id}.isdisjoint(record.id for record in final_records))
        self._assert_immutable_inputs()


if __name__ == '__main__':
    unittest.main()
