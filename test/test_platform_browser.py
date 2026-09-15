import json
import shutil
import subprocess
import unittest
from importlib import resources
from pathlib import Path

from xxtrain.platform.app import create_app
from xxtrain.platform.config import WorkspaceConfig, load_config

from .test_platform_http import AsgiTestClient, FakeCvat, FakeService


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


if __name__ == '__main__':
    unittest.main()
