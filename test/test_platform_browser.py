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

_MISSING_HTTP_DEPENDENCIES = tuple(name for name in ('fastapi', 'httpx') if find_spec(name) is None)
if _MISSING_HTTP_DEPENDENCIES:
    raise unittest.SkipTest(f'platform extra is required: {", ".join(_MISSING_HTTP_DEPENDENCIES)}')
else:
    from xxtrain.platform.app import create_app
    from xxtrain.platform.config import WorkspaceConfig, load_config

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

_PRESERVED_ROOT_FIELDS = {
    'version': '5.4.1',
    'flags': {'fixture': 'task-6', 'preserve': True},
    'description': 'synthetic mixed annotation',
    'imagePath': '../images/point-a.jpg',
    'imageData': None,
    'imageHeight': 600,
    'imageWidth': 800,
    'custom': {'preserve': ['root', 'value']},
}
_PRESERVED_NON_RECTANGLES = [
    {
        'label': 'wire',
        'points': [[40.0, 40.0], [90.0, 70.0]],
        'group_id': None,
        'description': 'preserve-line',
        'shape_type': 'line',
        'flags': {'preserve': True},
    },
    {
        'label': 'mask',
        'points': [[500.0, 100.0], [620.0, 130.0], [560.0, 250.0]],
        'group_id': 23,
        'description': 'preserve-polygon',
        'shape_type': 'polygon',
        'flags': {'preserve': True},
    },
]


def assert_preserved_mixed_annotation(test: unittest.TestCase, document: dict[str, object]) -> None:
    for field, expected_value in _PRESERVED_ROOT_FIELDS.items():
        test.assertIn(field, document)
        test.assertEqual(expected_value, document[field])
    test.assertEqual(
        _PRESERVED_NON_RECTANGLES, [shape for shape in document['shapes'] if shape['shape_type'] != 'rectangle']
    )


class PlatformBrowserTest(unittest.TestCase):
    def setUp(self) -> None:
        config = WorkspaceConfig(
            'line-3', '三号现场', 17, Path('images'), Path('annotations'), Path('state.json'), 'http://cvat.test'
        )
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

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_returned_page_posts_sync_and_renders_the_saved_workspace(self) -> None:
        script = self.client.get('/platform/app.js')
        self.assertEqual(200, script.status_code)
        harness = f"""
const calls = [];
const elements = new Map();
function makeElement(id) {{
  return {{
    id,
    hidden: false,
    disabled: false,
    textContent: '',
    value: '',
    dataset: {{}},
    classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
    addEventListener() {{}},
    setAttribute() {{}},
    removeAttribute() {{}},
    reset() {{}},
  }};
}}
globalThis.document = {{
  cookie: 'xxtrain_csrf=page-token',
  getElementById(id) {{
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
  }},
  querySelectorAll() {{ return []; }},
  addEventListener(name, callback) {{ if (name === 'DOMContentLoaded') callback(); }},
}};
globalThis.location = {{ search: '?returned=1', assign() {{ throw new Error('return flow must not navigate'); }} }};
globalThis.history = {{ replaceState() {{}} }};
const workspace = {{
  workspace_id: 'line-3', name: '三号现场', image_count: 2, status: 'pending', error: null,
  task: {{id: 'point', name: 'Point'}},
  targets: [
    {{id: 'detect', name: '检测', available: true}},
    {{id: 'classify', name: '分类', available: false}},
    {{id: 'segment', name: '分割', available: false}},
  ],
}};
globalThis.fetch = async (url, options = {{}}) => {{
  calls.push({{
    url,
    method: options.method || 'GET',
    body: options.body || null,
    csrf: options.headers?.['X-XTrain-CSRF'] || null,
  }});
  const body = url.endsWith('/session')
    ? {{authenticated: true, user_id: 17}}
    : (url.endsWith('/sync') ? {{...workspace, status: 'saved'}} : workspace);
  return {{ok: true, status: 200, json: async () => body}};
}};
eval({json.dumps(script.text)});
setTimeout(() => {{
  process.stdout.write(JSON.stringify({{
    calls,
    status: elements.get('status-value')?.textContent,
    action: elements.get('primary-action')?.textContent,
    disabled: elements.get('primary-action')?.disabled,
  }}));
}}, 25);
"""
        completed = subprocess.run(
            ['node', '-e', harness], text=True, encoding='utf-8', capture_output=True, check=False
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(
            [
                {'url': '/platform/api/session', 'method': 'GET', 'body': None, 'csrf': None},
                {'url': '/platform/api/workspace', 'method': 'GET', 'body': None, 'csrf': None},
                {'url': '/platform/api/annotation/sync', 'method': 'POST', 'body': '{}', 'csrf': 'page-token'},
            ],
            result['calls'],
        )
        self.assertEqual('已保存', result['status'])
        self.assertEqual('继续标注', result['action'])
        self.assertFalse(result['disabled'])

    def test_deployment_example_uses_the_exact_workspace_schema_without_credentials(self) -> None:
        path = Path(__file__).parents[1] / 'deploy' / 'platform' / 'workspace.example.json'
        payload = json.loads(path.read_text(encoding='utf-8'))

        self.assertEqual(
            {
                'workspace_id',
                'display_name',
                'owner_user_id',
                'images_dir',
                'annotations_dir',
                'state_path',
                'cvat_internal_url',
            },
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
    def test_fixture_is_isolated_and_records_immutable_input_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            receipt = create_fixture(Path(parent), owner_user_id=17, cvat_internal_url='http://cvat.test')

            root = Path(receipt['root'])
            self.assertEqual(FIXTURE_MARKER, receipt['marker'])
            self.assertEqual(root.parent, Path(parent))
            self.assertTrue(root.name.startswith('xxtrain-point-acceptance-'))
            self.assertEqual(2, len(receipt['images']))
            for immutable in [*receipt['images'], receipt['baseline']]:
                path = Path(immutable['path'])
                self.assertTrue(path.is_relative_to(root))
                self.assertEqual(immutable['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

            annotation = json.loads(Path(receipt['annotation_path']).read_text(encoding='utf-8'))
            self.assertEqual({'fixture': 'task-6', 'preserve': True}, annotation['flags'])
            self.assertEqual(['rectangle', 'line', 'polygon'], [shape['shape_type'] for shape in annotation['shapes']])
            self.assertEqual('tl', annotation['shapes'][0]['label'])
            self.assertEqual(11, annotation['shapes'][0]['group_id'])
            self.assertEqual({'reviewed': True}, annotation['shapes'][0]['flags'])
            self.assertEqual('classification-metadata', annotation['shapes'][0]['description'])

    def test_preservation_assertion_rejects_missing_root_or_nonrectangle_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            receipt = create_fixture(Path(parent), owner_user_id=17, cvat_internal_url='http://cvat.test')
            annotation = json.loads(Path(receipt['annotation_path']).read_text(encoding='utf-8'))

            annotation.pop('description')
            with self.assertRaises(AssertionError):
                assert_preserved_mixed_annotation(self, annotation)

            annotation = json.loads(Path(receipt['annotation_path']).read_text(encoding='utf-8'))
            annotation['shapes'][1].pop('flags')
            with self.assertRaises(AssertionError):
                assert_preserved_mixed_annotation(self, annotation)


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
        required_paths = [cls.receipt.get('annotation_path', ''), cls.receipt.get('baseline', {}).get('path', '')]
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

    def _saved_document(self) -> dict[str, object]:
        return json.loads(Path(self.receipt['annotation_path']).read_text(encoding='utf-8'))

    def test_detection_annotation_round_trip(self) -> None:
        page = self._new_page()
        self._login(page)
        status = page.locator('#status-value').text_content()
        self.assertIn(status, {'待标注', '正在准备标注任务', '标注中', '已保存'})
        page.screenshot(path=self.artifact_dir / '01-platform-ready.png', full_page=True)

        with page.expect_navigation(wait_until='domcontentloaded'):
            action = '开始标注' if status == '待标注' else '继续标注'
            page.get_by_role('button', name=action, exact=True).click()
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
        expect(page.locator('#status-value')).to_have_text('已保存')
        page.screenshot(path=self.artifact_dir / '04-first-save.png', full_page=True)

        document = self._saved_document()
        assert_preserved_mixed_annotation(self, document)
        rectangles = [shape for shape in document['shapes'] if shape['shape_type'] == 'rectangle']
        self.assertEqual({'Point', 'tl'}, {shape['label'] for shape in rectangles})
        classified = next(shape for shape in rectangles if shape['label'] == 'tl')
        self.assertEqual(11, classified['group_id'])
        self.assertEqual({'reviewed': True}, classified['flags'])
        self.assertEqual('classification-metadata', classified['description'])
        self.assertNotEqual(self.receipt['original_rectangle_points'], classified['points'])
        self._assert_immutable_inputs()

        with page.expect_navigation(wait_until='domcontentloaded'):
            page.get_by_role('button', name='继续标注', exact=True).click()
        self.assertEqual(first_job_url.rstrip('/'), page.url.rstrip('/'))
        expect(page.locator('.cvat-header')).to_be_hidden()
        expect(page.locator('.cvat-canvas-container')).to_be_visible()
        page.reload(wait_until='domcontentloaded')
        self.assertEqual(first_job_url.rstrip('/'), page.url.rstrip('/'))
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

        page.route('**/platform/api/annotation/sync', lambda route: route.abort('connectionfailed'), times=1)
        with page.expect_navigation(wait_until='domcontentloaded'):
            page.get_by_role('button', name='完成', exact=True).click()
        expect(page.locator('#status-value')).to_have_text('平台保存失败')
        expect(page.get_by_role('button', name='重试保存', exact=True)).to_be_enabled()
        page.screenshot(path=self.artifact_dir / '05-platform-sync-failure.png', full_page=True)

        page.get_by_role('button', name='重试保存', exact=True).click()
        expect(page.locator('#status-value')).to_have_text('已保存')
        page.screenshot(path=self.artifact_dir / '06-empty-detection-save.png', full_page=True)
        final_document = self._saved_document()
        assert_preserved_mixed_annotation(self, final_document)
        self.assertEqual([], [shape for shape in final_document['shapes'] if shape['shape_type'] == 'rectangle'])
        self._assert_immutable_inputs()


if __name__ == '__main__':
    unittest.main()
