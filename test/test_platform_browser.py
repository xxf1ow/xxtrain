from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from contextlib import nullcontext, suppress
from importlib import resources
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

from PIL import Image

_MISSING_HTTP_DEPENDENCIES = tuple(name for name in ('fastapi', 'httpx') if find_spec(name) is None)
if _MISSING_HTTP_DEPENDENCIES:
    raise unittest.SkipTest(f'platform extra is required: {", ".join(_MISSING_HTTP_DEPENDENCIES)}')
else:
    from xxtrain.business_tasks.point import point_task_definition
    from xxtrain.platform.app import create_app
    from xxtrain.platform.config import WorkspaceConfig, load_config
    from xxtrain.platform.contracts import EditJob, JobRef
    from xxtrain.platform.runtime import RuntimeCache
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


def _acceptance_job(runtime: RuntimeCache, data: WorkspaceData, target: str) -> JobRef | EditJob | None:
    if target == 'detect':
        return runtime.job_for(target, data.detection_fingerprint())
    return runtime.edit_job_for(target, data.target_fingerprint(target))


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
        self.assertIn('每张裁剪图只选择一个分类标签', page.text)
        self.assertIn('href="/platform/style.css"', page.text)
        self.assertIn('src="/platform/app.js"', page.text)
        self.assertNotIn('http://', page.text)
        self.assertNotIn('https://', page.text)
        self.assertEqual(200, style.status_code)
        self.assertTrue(style.headers['content-type'].startswith('text/css'))
        self.assertEqual(200, script.status_code)
        self.assertTrue(script.headers['content-type'].startswith('text/javascript'))

    def run_page(
        self,
        actions: str = '',
        *,
        returned: bool = False,
        ready: bool = False,
        sync_failure: bool = False,
        stored_target: str | None = None,
        validation_failure: bool = False,
        classify_ready: bool = False,
        validation_payload: dict[str, str] | None = None,
        load_actions: str = 'await loaded();',
        training_run: dict[str, object] | None = None,
        submission_run: dict[str, object] | None = None,
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
const stored = new Map();
if ({json.dumps(stored_target)} !== null) stored.set('xxtrain-return-target', {json.dumps(stored_target)});
globalThis.sessionStorage = {{
  getItem(key) {{ return stored.has(key) ? stored.get(key) : null; }},
  setItem(key, value) {{ stored.set(key, String(value)); }},
  removeItem(key) {{ stored.delete(key); }},
}};
const workspace = {{
  workspace_id: 'line-3', name: '三号现场', image_count: 10, annotated_image_count: 7, boxed_image_count: 7,
  can_generate_detection_cache: {json.dumps(ready)}, detection_cache_ready: false,
  task: {{id: 'point', name: 'Point'}},
  targets: [
    {{id: 'detect', name: '检测', available: true, sample_count: 10, annotated_sample_count: 7,
      can_annotate: true, can_generate_cache: {json.dumps(ready)}, cache_ready: false}},
    {{id: 'classify', name: '分类', available: true, sample_count: 7, annotated_sample_count: 4,
      can_annotate: true, can_generate_cache: {json.dumps(classify_ready)}, cache_ready: false}},
    {{id: 'segment', name: '分割', available: true, sample_count: 7, annotated_sample_count: 0,
      can_annotate: false, can_generate_cache: false, cache_ready: false}},
  ],
  training_enabled: {json.dumps(training_run is not None or submission_run is not None)},
  training: {{detect: {json.dumps(training_run)}, classify: null, segment: null}},
}};
let release;
let hold = false;
let failure = {json.dumps(sync_failure)};
let submittedRun = null;
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
  if ({json.dumps(validation_failure or validation_payload is not None)}
      && url.endsWith('/sync') && options.method === 'POST') {{
    return {{ok: false, status: 409, json: async () => ({json.dumps(validation_payload)} || {{
      detail: '裁剪图 3 的标注不符合要求，请返回当前任务修正。',
      annotation_url: '/tasks/42/jobs/74?defaultWorkspace=TAGS&frame=2',
    }})}};
  }}
  if (url.endsWith('/train')) {{
    submittedRun = {json.dumps(submission_run)};
    return {{ok: true, status: 200, json: async () => ({{run_id: submittedRun.id, run: submittedRun}})}};
  }}
  const completedTargets = workspace.targets.map((target) =>
    target.id === 'detect' ? {{...target, sample_count: 50, annotated_sample_count: 50, can_generate_cache: true}}
      : target);
  const body = url.endsWith('/session')
    ? {{authenticated: true, user_id: 17}}
    : url.endsWith('/sync') ? {{...workspace, image_count: 50, annotated_image_count: 50,
        boxed_image_count: 50, can_generate_detection_cache: true, targets: completedTargets}}
    : url.endsWith('/images') ? {{...workspace, image_count: 12,
        targets: workspace.targets.map((target) => target.id === 'detect' ? {{...target, sample_count: 12}} : target)}}
    : url.endsWith('/cache') ? {{...workspace, detection_cache_ready: true,
        targets: workspace.targets.map((target) => url.includes(`/targets/${{target.id}}/`)
          ? {{...target, cache_ready: true}} : target)}}
    : url.endsWith('/start') ? {{annotation_url: '/tasks/41/jobs/73'}}
    : submittedRun ? {{...workspace, training: {{...workspace.training, detect: submittedRun}}}} : workspace;
  return {{ok: true, status: 200, json: async () => body}};
}};
eval({json.dumps(script.text)});
{load_actions}
const get = (id) => elements.get(id);
const snapshot = () => ({{
  images: get('image-count')?.textContent,
  annotated: get('annotated-image-count')?.textContent,
  totals: ['detect', 'classify', 'segment'].map((target) => get(`${{target}}-image-total`)?.textContent),
  annotatedTargets: ['detect', 'classify', 'segment'].map((target) =>
    get(target === 'detect' ? 'annotated-image-count' : `${{target}}-annotated-count`)?.textContent),
  targetButtons: ['detect', 'classify', 'segment'].map((target) => [
    get(target === 'detect' ? 'primary-action' : `${{target}}-annotate-action`)?.disabled,
    get(target === 'detect' ? 'cache-action' : `${{target}}-cache-action`)?.disabled,
    get(target === 'detect' ? 'cache-action' : `${{target}}-cache-action`)?.textContent,
  ]),
  cacheDisabled: get('cache-action')?.disabled,
  cacheText: get('cache-action')?.textContent,
  error: get('workspace-error')?.textContent,
  classifyError: get('classify-error')?.textContent,
  segmentError: get('segment-error')?.textContent,
  segmentCorrectionHref: get('segment-correction')?.href || null,
  correctionHref: get('classify-correction')?.href || null,
  targetProgress: ['detect', 'classify', 'segment'].map((target) => get(`${{target}}-progress`)?.textContent),
  uploadProgress: get('upload-progress')?.textContent,
  uploadProgressHidden: get('upload-progress')?.hidden,
  notification: get('workspace-message')?.textContent,
  notificationHidden: get('workspace-message')?.hidden,
  disabled: ['primary-action', 'image-files', 'cache-action', 'logout-button', 'login-button']
    .map((id) => get(id)?.disabled),
}});
const before = snapshot();
before.syncObservations = globalThis.syncObservations;
{actions}
process.stdout.write(JSON.stringify({{calls, before, after: snapshot(), assigned, replaced,
  storedTarget: sessionStorage.getItem('xxtrain-return-target')}}));
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
        self.assertEqual(['10', '7', '7'], result['after']['totals'])
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
        self.assertEqual('/platform/api/targets/detect/sync', result['calls'][-1]['url'])
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
        self.assertEqual(['12', '7', '7'], result['after']['totals'])
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
        self.assertEqual('/platform/api/targets/detect/cache', result['calls'][-1]['url'])
        self.assertEqual('{}', result['calls'][-1]['body'])
        self.assertEqual(3, len(result['calls']))

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_start_navigates_to_the_server_annotation_url(self) -> None:
        result = self.run_page("await get('primary-action').listeners.click();")
        self.assertEqual('/platform/api/targets/detect/start', result['calls'][-1]['url'])
        self.assertEqual('/tasks/41/jobs/73', result['assigned'])
        self.assertEqual('detect', result['storedTarget'])

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
            ['/platform/api/session', '/platform/api/workspace', '/platform/api/targets/detect/sync'],
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

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_target_rows_use_crop_counts_and_target_specific_actions(self) -> None:
        result = self.run_page("await get('classify-annotate-action').listeners.click();")

        self.assertEqual(['10', '7', '7'], result['after']['totals'])
        self.assertEqual(['7', '4', '0'], result['after']['annotatedTargets'])
        self.assertEqual([False, True, '开始训练'], result['before']['targetButtons'][1])
        self.assertEqual([True, True, '开始训练'], result['before']['targetButtons'][2])
        self.assertEqual('/platform/api/targets/classify/start', result['calls'][-1]['url'])
        self.assertEqual('classify', result['storedTarget'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_workspace_training_label_uses_cancellation_and_terminal_facts(self) -> None:
        base_run = {
            'id': '55555555-5555-4555-8555-555555555555',
            'workspace_id': 'line-3',
            'workspace_name': '三号现场',
            'target': 'detect',
            'submitted_at': '2026-09-17T00:00:00+00:00',
            'metric_name': '检测效果：mAP50-95',
        }
        cancelling = {
            **base_run,
            'cancellation_requested': True,
            'execution': {
                'status': 'queued',
                'active': True,
                'epoch': None,
                'total_epochs': None,
                'elapsed_seconds': None,
                'metric': None,
                'download_ready': False,
                'detail': None,
            },
        }
        completed_without_artifact = {
            **cancelling,
            'execution': {**cancelling['execution'], 'status': 'completed', 'active': False},
        }
        cancellation_result = self.run_page(training_run=cancelling)
        terminal_result = self.run_page(training_run=completed_without_artifact)

        self.assertEqual('取消请求已保存，等待停止', cancellation_result['after']['cacheText'])
        self.assertEqual('训练已结束，产物不可用', terminal_result['after']['cacheText'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_legacy_run_submission_does_not_claim_new_intent_was_saved(self) -> None:
        legacy = {
            'id': '55555555-5555-4555-8555-555555555555',
            'workspace_id': 'line-3',
            'workspace_name': '三号现场',
            'target': 'detect',
            'submitted_at': '2026-09-17T00:00:00+00:00',
            'metric_name': '检测效果：mAP50-95',
            'cancellation_requested': False,
            'execution': None,
        }

        legacy_result = self.run_page("await get('cache-action').listeners.click();", ready=True, submission_run=legacy)

        self.assertEqual('已有历史任务，未保存新的训练请求', legacy_result['after']['notification'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_return_hint_selects_target_sync_and_clears_only_after_success(self) -> None:
        successful = self.run_page(returned=True, stored_target='segment')
        failed = self.run_page(returned=True, stored_target='classify', validation_failure=True)

        self.assertEqual('/platform/api/targets/segment/sync', successful['calls'][-1]['url'])
        self.assertIsNone(successful['storedTarget'])
        self.assertEqual('/platform/api/targets/classify/sync', failed['calls'][-1]['url'])
        self.assertEqual('classify', failed['storedTarget'])
        self.assertIn('裁剪图 3', failed['after']['classifyError'])
        self.assertEqual('/tasks/42/jobs/74?defaultWorkspace=TAGS&frame=2', failed['after']['correctionHref'])
        self.assertIsNone(failed['replaced'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_live_complete_wait_observes_delayed_and_fast_platform_sync(self) -> None:
        for target, delayed in (('detect', True), ('classify', True), ('segment', False)):
            with self.subTest(target=target, delayed=delayed):
                page = Mock()
                page.url = 'http://testserver/platform/' + ('?returned=1' if delayed else '')
                page.expect_navigation.return_value = nullcontext()
                PlatformLiveBrowserTest()._complete_with_plugin(page, target)
                expression = page.wait_for_function.call_args.args[0]
                argument = page.wait_for_function.call_args.kwargs.get('arg')
                result = self.run_page(
                    returned=True,
                    stored_target=target,
                    load_actions=f"""
const complete = () => {{
  const value = eval({json.dumps(expression)});
  return typeof value === 'function' ? value({json.dumps(argument)}) : value;
}};
hold = {json.dumps(delayed)};
const loading = loaded();
globalThis.syncObservations = [];
if (hold) {{
  while (!release) await Promise.resolve();
  syncObservations.push({{visible: !elements.get('workspace-panel').hidden, complete: complete()}});
  release();
}}
await loading;
syncObservations.push({{visible: !elements.get('workspace-panel').hidden, complete: complete()}});
""",
                )
                observations = result['before']['syncObservations']
                if delayed:
                    self.assertEqual({'visible': True, 'complete': False}, observations[0])
                self.assertEqual({'visible': True, 'complete': True}, observations[-1])
                failed = self.run_page(
                    returned=True,
                    stored_target=target,
                    sync_failure=True,
                    load_actions=f"""
await loaded();
globalThis.syncObservations = [({expression})({json.dumps(argument)})];
""",
                )
                self.assertEqual([False], failed['before']['syncObservations'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_validation_reasons_reach_http_and_page_without_private_details(self) -> None:
        import copy

        from xxtrain.platform.contracts import TargetValidationError
        from xxtrain.platform.service import AnnotationService

        from .test_platform_point_workflow import HttpxCvatFixture

        with tempfile.TemporaryDirectory() as parent:
            receipt = create_fixture(Path(parent), owner_user_id=17, cvat_internal_url='http://cvat.test')
            config = load_config(Path(receipt['config_path']))
            data = WorkspaceData(config.workspace_dir)
            repository = AnnotationRepository(Path(receipt['database_path']), point_task_definition())
            cvat = HttpxCvatFixture()
            self.addCleanup(cvat.close)
            service = AnnotationService(config, data, cvat.client, RuntimeCache(config.runtime_dir))
            client = AsgiTestClient(create_app(config, service, FakeCvat()))
            self.addCleanup(client.close)
            client.get('/platform/')
            client.cookies.set('sessionid', 'active')
            headers = {'origin': 'http://testserver', 'x-xtrain-csrf': client.cookies.get('xxtrain_csrf')}
            before = repository.annotations()
            for target in ('classify', 'segment'):
                cvat.expect(data.target_frames(target, config.runtime_dir))
                service.begin_target(17, target)
            cases = (
                ('classify', None, '只能保留一个分类标签'),
                ('segment', [20, 20, 50, 50, 80, 80], '必须恰好有两个点'),
                ('segment', [20, 20, 20, 20], '两个端点不能重合'),
                ('segment', [20, 20, 999, 999], '端点必须位于裁剪图内'),
            )
            originals = {target: copy.deepcopy(cvat.job(target)['annotations']) for target in ('classify', 'segment')}
            for target, points, reason in cases:
                with self.subTest(reason=reason):
                    payload = copy.deepcopy(originals[target])
                    if target == 'classify':
                        payload['tags'].append({**payload['tags'][0], 'id': 9999})
                    else:
                        payload['shapes'][0]['points'] = points
                    cvat.job(target)['annotations'] = payload
                    with self.assertRaises(TargetValidationError) as raised:
                        service.sync_target(17, target)
                    self.assertIn(reason, str(raised.exception))
                    response = client.post(f'/platform/api/targets/{target}/sync', headers=headers, json={})
                    self.assertEqual(409, response.status_code)
                    detail = response.json()
                    self.assertEqual(str(raised.exception), detail['detail'])
                    self.assertIn('裁剪图 1', detail['detail'])
                    self.assertIn('请返回', detail['detail'])
                    self.assertTrue(detail['annotation_url'].endswith('frame=0'))
                    for secret in (str(config.workspace_dir), str(before[0].id), 'ValueError', 'label_id'):
                        self.assertNotIn(secret, response.text)
                    page = self.run_page(returned=True, stored_target=target, validation_payload=detail)
                    self.assertIn(reason, page['after'][f'{target}Error'])
                    link = 'correctionHref' if target == 'classify' else 'segmentCorrectionHref'
                    self.assertEqual(detail['annotation_url'], page['after'][link])
                    self.assertEqual(target, page['storedTarget'])
                    self.assertIsNone(page['replaced'])
                    self.assertEqual(before, repository.annotations())
            secret = f'CVAT response password=secret at {config.workspace_dir} object {before[0].id}'
            with patch('xxtrain.platform.service.validate_target_annotations', side_effect=ValueError(secret)):
                response = client.post('/platform/api/targets/classify/sync', headers=headers, json={})
            self.assertEqual(409, response.status_code)
            self.assertIn('标注类型、标签或坐标不符合要求', response.json()['detail'])
            self.assertNotIn(secret, response.text)
            self.assertNotIn('password', response.text)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_invalid_return_hint_falls_back_to_detection(self) -> None:
        result = self.run_page(returned=True, stored_target='../../segment')

        self.assertEqual('/platform/api/targets/detect/sync', result['calls'][-1]['url'])
        self.assertIsNone(result['storedTarget'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_target_cache_uses_its_row_progress_and_ready_text(self) -> None:
        result = self.run_page(
            """
hold = true;
const operation = get('classify-cache-action').listeners.click();
before.generating = snapshot();
release();
await operation;
""",
            classify_ready=True,
        )

        self.assertEqual(['', '正在生成训练缓存…', ''], result['before']['generating']['targetProgress'])
        self.assertEqual('/platform/api/targets/classify/cache', result['calls'][-1]['url'])
        self.assertEqual([False, True, '训练缓存已生成'], result['after']['targetButtons'][1])

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
    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline acceptance sequence')
    def test_live_scenario_preserves_current_bindings_and_restores_invalidated_pointers(self) -> None:
        import copy

        from xxtrain.platform.contracts import TargetValidationError
        from xxtrain.platform.service import AnnotationService

        from .test_platform_point_workflow import HttpxCvatFixture

        with tempfile.TemporaryDirectory() as parent:
            receipt = create_fixture(Path(parent), owner_user_id=17, cvat_internal_url='http://cvat.test')
            config = load_config(Path(receipt['config_path']))
            data = WorkspaceData(config.workspace_dir)
            cvat = HttpxCvatFixture()
            self.addCleanup(cvat.close)
            service = AnnotationService(config, data, cvat.client, RuntimeCache(config.runtime_dir))
            case = PlatformLiveBrowserTest()
            case.receipt = receipt
            page = Mock()
            page.expect_navigation.side_effect = lambda **kwargs: nullcontext()
            active = {}

            def open_target(_page, target):
                cvat.expect(data.images() if target == 'detect' else data.target_frames(target, config.runtime_dir))
                page.url = service.begin_target(17, target)
                active['target'] = target

            def complete(_page, target='detect'):
                self.assertEqual(target, active['target'])
                cvat.job(active['target'])['state'] = 'completed'
                service.sync_target(17, active['target'])
                page.url = '/platform/'

            def malformed(_page):
                cvat.job(active['target'])['state'] = 'completed'
                with self.assertRaises(TargetValidationError):
                    service.sync_target(17, active['target'])

            def click(selector):
                if selector.endswith('-cache-action'):
                    service.generate_target_cache(17, selector.split('-')[0][1:])
                elif selector.endswith('-correction'):
                    job = cvat.job(active['target'])
                    page.url = f'/tasks/{job["task_id"]}/jobs/{job["id"]}'

            def draw(_page):
                shapes = cvat.job('segment')['annotations']['shapes']
                shapes.append(
                    {
                        'id': cvat.next_object_id,
                        'frame': 0,
                        'type': 'polyline',
                        'label_id': cvat.label_id('segment', '1'),
                        'points': [60, 60, 120, 100],
                        'attributes': [],
                        'occluded': False,
                        'outside': False,
                        'rotation': 0,
                        'z_order': 0,
                    }
                )
                cvat.next_object_id += 1

            def delete(_page):
                payload = cvat.job(active['target'])['annotations']
                items = payload['tags'] if active['target'] == 'classify' else payload['shapes']
                selected = max((item for item in items if item['frame'] == 0), key=lambda item: item['id'])
                items.remove(selected)

            def evaluate(script, argument):
                # Execute the harness's REST JavaScript; only fetch is replaced by the in-process CVAT fixture.
                harness = f"""
globalThis.document = {{cookie: 'csrftoken=offline'}};
let request;
globalThis.fetch = async (url, options = {{}}) => {{
  request = {{url, ...options}};
  return {{ok: true, json: async () => ({{}})}};
}};
(async () => {{ await ({script})({json.dumps(argument)}); process.stdout.write(JSON.stringify(request)); }})();
"""
                result = subprocess.run(['node', '-e', harness], capture_output=True, text=True, check=True)
                request = json.loads(result.stdout)
                method = request.get('method', 'GET')
                payload = json.loads(request['body']) if 'body' in request else None
                if method == 'PUT':
                    # CVAT full replacement assigns new native IDs, including for unchanged objects.
                    for item in [*payload.get('tags', []), *payload.get('shapes', [])]:
                        item.pop('id', None)
                response = cvat.http.request(method, 'http://cvat.test' + request['url'], json=payload)
                self.assertLess(response.status_code, 400)
                return response.json()

            page.evaluate.side_effect = evaluate
            page.locator.side_effect = lambda selector: Mock(click=lambda: click(selector))
            case._new_page = lambda: page
            case._login = lambda _page: None
            case._open_target = open_target
            case._complete_with_plugin = complete
            case._complete_malformed_via_api = malformed
            case._job_annotations = lambda _page: copy.deepcopy(cvat.job(active['target'])['annotations'])
            case._select_first_object_label = lambda _page, label: cvat.job('classify')['annotations']['tags'][
                0
            ].update(label_id=cvat.label_id('classify', label), attributes=[])
            case._draw_two_point_polyline = draw
            case._delete_last_object = delete
            with patch(f'{__name__}.expect', create=True):
                case.test_three_target_native_round_trip_correction_and_cache_readiness()

    def test_live_job_lookup_distinguishes_detection_from_edit_targets(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            receipt = create_fixture(Path(parent), owner_user_id=17, cvat_internal_url='http://cvat.test')
            data = WorkspaceData(Path(receipt['root']) / 'workspace')
            runtime = RuntimeCache(Path(receipt['runtime_dir']))
            detection = JobRef(101, 201, tuple(image.sample_id for image in data.images()))
            runtime.remember_job('detect', data.detection_fingerprint(), detection)
            frames = data.target_frames('classify', Path(receipt['runtime_dir']))
            mappings = tuple(frame.mapping for frame in frames)
            classification = EditJob(
                JobRef(102, 202, tuple(dict.fromkeys(mapping.image_id for mapping in mappings))), mappings
            )
            runtime.remember_edit_job('classify', data.target_fingerprint('classify'), classification)

            try:
                detected = _acceptance_job(runtime, data, 'detect')
            except ValueError as error:
                self.fail(f'detection acceptance lookup used the edit-target fingerprint: {error}')
            self.assertEqual(detection, detected)
            self.assertEqual(classification, _acceptance_job(runtime, data, 'classify'))

    def test_fixture_is_isolated_and_records_sqlite_annotation_identity(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            receipt = create_fixture(Path(parent), owner_user_id=17, cvat_internal_url='http://cvat.test')

            root = Path(receipt['root'])
            self.assertIn('database_path', receipt)
            self.assertNotIn('annotation_path', receipt)
            database_path = Path(receipt['database_path'])
            self.assertEqual(FIXTURE_MARKER, receipt['marker'])
            with patch.object(PlatformLiveBrowserTest, 'receipt', receipt, create=True):
                PlatformLiveBrowserTest._validate_receipt()
            with patch.object(
                PlatformLiveBrowserTest,
                'receipt',
                {**receipt, 'marker': 'xxtrain-task-7-point-workflow-v1'},
                create=True,
            ):
                with self.assertRaisesRegex(RuntimeError, 'fixture receipt'):
                    PlatformLiveBrowserTest._validate_receipt()
            self.assertEqual(root.parent, Path(parent))
            self.assertTrue(root.name.startswith('xxtrain-point-acceptance-'))
            self.assertEqual(root / 'workspace' / 'annotations.db', database_path)
            self.assertTrue(database_path.is_file())
            self.assertFalse((root / 'workspace' / 'annotations').exists())
            self.assertEqual(50, len(receipt['images']))
            for immutable in [*receipt['images'], receipt['baseline']]:
                path = Path(immutable['path'])
                self.assertTrue(path.is_relative_to(root))
                self.assertEqual(immutable['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

            workspace = WorkspaceData(root / 'workspace')
            self.assertEqual(
                [image['sample_id'] for image in receipt['images']], [image.sample_id for image in workspace.images()]
            )
            repository = AnnotationRepository(database_path, point_task_definition())
            records = repository.annotations(step_key='detect')
            self.assertEqual(receipt['initial_annotation_ids']['detect'], [str(record.id) for record in records])
            self.assertEqual(receipt['images'][0]['sample_id'], records[0].image_id)
            self.assertEqual(records[0].image_id, records[1].image_id)
            self.assertNotEqual(records[0].id, records[1].id)
            self.assertEqual('tl', records[0].label)
            self.assertEqual(receipt['original_rectangle_points'], records[0].geometry)
            self.assertEqual(
                receipt['initial_annotation_ids']['classify'],
                [str(record.id) for record in repository.annotations(step_key='classify')],
            )
            self.assertEqual(
                receipt['initial_annotation_ids']['segment'],
                [str(record.id) for record in repository.annotations(step_key='segment')],
            )


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
            raise RuntimeError('fixture receipt is not an xxtrain Task 7 Point workflow workspace')
        root = Path(cls.receipt.get('root', '')).resolve()
        if not root.is_dir() or not root.name.startswith('xxtrain-point-acceptance-'):
            raise RuntimeError('fixture root is not a generated Task 7 temporary directory')
        required_paths = [cls.receipt.get('database_path', ''), cls.receipt.get('baseline', {}).get('path', '')]
        required_paths.append(cls.receipt.get('runtime_dir', ''))
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
        expect(page.locator('[data-target="classify"]')).to_have_attribute('aria-disabled', 'false')
        expect(page.locator('[data-target="segment"]')).to_have_attribute('aria-disabled', 'false')

    def _assert_immutable_inputs(self) -> None:
        for immutable in [*self.receipt['images'], self.receipt['baseline']]:
            path = Path(immutable['path'])
            self.assertEqual(immutable['sha256'], hashlib.sha256(path.read_bytes()).hexdigest(), path)

    def _job_identity(self, page: Page) -> tuple[int, int]:
        matched = re.search(r'/tasks/(\d+)/jobs/(\d+)/?(?:\?[^#]*)?$', page.url)
        self.assertIsNotNone(matched)
        assert matched is not None
        return int(matched.group(1)), int(matched.group(2))

    def _job_annotations(self, page: Page) -> dict[str, object]:
        _, job_id = self._job_identity(page)
        return page.evaluate(
            """async (jobId) => {
              const response = await fetch(`/api/jobs/${jobId}/annotations`, {credentials: 'same-origin'});
              if (!response.ok) throw new Error(`annotations GET failed: ${response.status}`);
              return response.json();
            }""",
            job_id,
        )

    def _create_malformed_annotations(self, page: Page, payload: dict[str, object]) -> dict[str, object]:
        _, job_id = self._job_identity(page)
        return page.evaluate(
            """async ({jobId, payload}) => {
              const csrf = document.cookie.split(';').map((part) => part.trim())
                .find((part) => part.startsWith('csrftoken='))?.split('=', 2)[1];
              const response = await fetch(`/api/jobs/${jobId}/annotations?action=create`, {
                method: 'PATCH', credentials: 'same-origin',
                headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf || ''},
                body: JSON.stringify(payload),
              });
              if (!response.ok) throw new Error(`annotations PATCH failed: ${response.status}`);
              return response.json();
            }""",
            {'jobId': job_id, 'payload': payload},
        )

    def _complete_malformed_via_api(self, page: Page) -> None:
        _, job_id = self._job_identity(page)
        page.evaluate(
            """async (jobId) => {
              const csrf = document.cookie.split(';').map((part) => part.trim())
                .find((part) => part.startsWith('csrftoken='))?.split('=', 2)[1];
              const response = await fetch(`/api/jobs/${jobId}`, {
                method: 'PATCH', credentials: 'same-origin',
                headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf || ''},
                body: JSON.stringify({state: 'completed'}),
              });
              if (!response.ok) throw new Error(`job PATCH failed: ${response.status}`);
            }""",
            job_id,
        )
        page.goto(f'{_LIVE_VALUES["XXTRAIN_PLATFORM_URL"].rstrip("/")}/?returned=1', wait_until='domcontentloaded')
        page.wait_for_function("!document.getElementById('workspace-panel').hidden")

    def _open_target(self, page: Page, target: str) -> None:
        action = '#primary-action' if target == 'detect' else f'#{target}-annotate-action'
        with page.expect_navigation(wait_until='domcontentloaded'):
            page.locator(action).click()
        expect(page.locator('.cvat-canvas-container')).to_be_visible()

    def _complete_with_plugin(self, page: Page, target: str = 'detect') -> None:
        with page.expect_navigation(wait_until='domcontentloaded'):
            page.get_by_role('button', name='完成', exact=True).click()
        page.wait_for_function(
            """(target) => !document.getElementById('workspace-panel').hidden
              && !new URLSearchParams(location.search).has('returned')
              && sessionStorage.getItem('xxtrain-return-target') === null
              && document.getElementById(target === 'detect' ? 'workspace-error' : `${target}-error`).hidden
              && document.getElementById(`${target}-progress`).hidden""",
            arg=target,
        )

    def _select_first_object_label(self, page: Page, label: str) -> None:
        item = page.locator('.cvat-objects-sidebar-state-item:visible').first
        selector = item.locator('.cvat-objects-sidebar-state-item-label-selector')
        expect(selector).to_be_visible()
        selector.click()
        option = page.locator(f'.ant-select-dropdown:visible .ant-select-item-option[title="{label}"]')
        expect(option).to_be_visible()
        option.click()

    def _delete_last_object(self, page: Page) -> None:
        items = page.locator('.cvat-objects-sidebar-state-item:visible')
        before = items.count()
        self.assertGreater(before, 0)
        items.last.hover()
        items.last.click()
        page.keyboard.press('Delete')
        expect(items).to_have_count(before - 1)

    def _draw_two_point_polyline(self, page: Page) -> None:
        items = page.locator('.cvat-objects-sidebar-state-item:visible')
        before = items.count()
        page.locator('.cvat-draw-polyline-control').click()
        popover = page.locator('.cvat-draw-shape-popover:visible')
        expect(popover).to_be_visible()
        points = popover.locator('.cvat-draw-shape-popover-points-selector input')
        points.fill('2')
        popover.locator('.cvat-draw-polyline-shape-button').click()
        canvas = page.locator('.cvat-canvas-container')
        bounds = canvas.bounding_box()
        self.assertIsNotNone(bounds)
        assert bounds is not None
        canvas.click(position={'x': bounds['width'] * 0.4, 'y': bounds['height'] * 0.6})
        canvas.click(position={'x': bounds['width'] * 0.6, 'y': bounds['height'] * 0.4})
        expect(items).to_have_count(before + 1)

    def test_three_target_native_round_trip_correction_and_cache_readiness(self) -> None:
        page = self._new_page()
        self._login(page)
        expect(page.locator('#image-count')).to_have_text('50')
        expect(page.locator('#annotated-image-count')).to_have_text('50')
        initial_ids = self.receipt['initial_annotation_ids']
        repository = AnnotationRepository(Path(self.receipt['database_path']), point_task_definition())
        data = WorkspaceData(Path(self.receipt['root']) / 'workspace')
        runtime = RuntimeCache(Path(self.receipt['runtime_dir']))

        self._open_target(page, 'detect')
        detection = self._job_annotations(page)
        self.assertEqual(51, len(detection['shapes']))
        native_detection_ids = {shape['id'] for shape in detection['shapes']}
        self.assertEqual(51, len(native_detection_ids))
        detect_job = _acceptance_job(runtime, data, 'detect')
        self.assertIsNotNone(detect_job)
        assert detect_job is not None
        detect_bindings_before = repository.bindings(detect_job)
        self.assertEqual(native_detection_ids, {binding.object_id for binding in detect_bindings_before})
        self._complete_with_plugin(page)
        self.assertEqual(
            initial_ids['detect'], [str(record.id) for record in repository.annotations(step_key='detect')]
        )
        self.assertEqual(detect_bindings_before, repository.bindings(detect_job))

        self._open_target(page, 'classify')
        task_id, _ = self._job_identity(page)
        labels = page.evaluate(
            """async (taskId) => (await fetch(`/api/labels?task_id=${taskId}`, {
              credentials: 'same-origin'
            })).json()""",
            task_id,
        )
        self.assertEqual({'tag'}, {label['type'] for label in labels['results']})
        classification = self._job_annotations(page)
        self.assertEqual(51, len(classification['tags']))
        native_tag_ids = {tag['id'] for tag in classification['tags']}
        self.assertEqual(51, len(native_tag_ids))
        classify_job = _acceptance_job(runtime, data, 'classify')
        self.assertIsNotNone(classify_job)
        assert classify_job is not None
        classifications_before = repository.annotations(step_key='classify')
        self.assertEqual(initial_ids['classify'], [str(record.id) for record in classifications_before])
        classify_bindings_before = repository.bindings(classify_job.ref)
        initial_segments = repository.annotations(step_key='segment')
        self.assertEqual(native_tag_ids, {binding.object_id for binding in repository.bindings(classify_job.ref)})
        self._select_first_object_label(page, 'cc')
        self._complete_with_plugin(page, 'classify')
        classifications_after = repository.annotations(step_key='classify')
        changed_classifications = [
            (before, after)
            for before, after in zip(classifications_before, classifications_after, strict=True)
            if before != after
        ]
        self.assertEqual(1, len(changed_classifications))
        changed_before, changed_after = changed_classifications[0]
        self.assertEqual(changed_before.id, changed_after.id)
        self.assertEqual(changed_before.image_id, changed_after.image_id)
        self.assertEqual(changed_before.parent_id, changed_after.parent_id)
        self.assertEqual(classify_job.frames[0].parent_id, changed_after.parent_id)
        self.assertEqual('cc', changed_after.label)
        classify_bindings_after = repository.bindings(classify_job.ref)
        self.assertEqual(classify_bindings_before, classify_bindings_after)
        changed_binding = next(
            binding for binding in classify_bindings_after if binding.annotation_id == changed_after.id
        )
        self.assertIn(changed_binding.object_id, native_tag_ids)
        retained_segments = tuple(record for record in initial_segments if record.parent_id != changed_after.parent_id)
        removed_segment_ids = {record.id for record in initial_segments if record.parent_id == changed_after.parent_id}
        self.assertEqual(2, len(removed_segment_ids))
        self.assertEqual(50, len(retained_segments))
        self.assertEqual(retained_segments, repository.annotations(step_key='segment'))

        previous_classify_ref = classify_job.ref
        self._open_target(page, 'classify')
        classify_job = _acceptance_job(runtime, data, 'classify')
        self.assertIsNotNone(classify_job)
        assert classify_job is not None
        self.assertNotEqual(previous_classify_ref, classify_job.ref)
        classify_bindings_after = repository.bindings(classify_job.ref)
        invalid_classification = self._job_annotations(page)
        duplicate = dict(invalid_classification['tags'][0])
        duplicate.pop('id', None)
        self._create_malformed_annotations(page, {'tags': [duplicate], 'shapes': [], 'tracks': []})
        malformed_classification = self._job_annotations(page)
        self.assertEqual(52, len(malformed_classification['tags']))
        self.assertEqual(invalid_classification['tags'], malformed_classification['tags'][:-1])
        self._complete_malformed_via_api(page)
        expect(page.locator('#classify-error')).to_contain_text('只能保留一个分类标签')
        self.assertEqual(classifications_after, repository.annotations(step_key='classify'))
        self.assertEqual(classify_bindings_after, repository.bindings(classify_job.ref))
        correction = page.locator('#classify-correction')
        expect(correction).to_be_visible()
        with page.expect_navigation(wait_until='domcontentloaded'):
            correction.click()
        self._delete_last_object(page)
        self._complete_with_plugin(page, 'classify')
        expect(page.locator('#classify-annotated-count')).to_have_text('51')
        self.assertEqual(classifications_after, repository.annotations(step_key='classify'))
        self.assertEqual(classify_bindings_after, repository.bindings(classify_job.ref))

        self._open_target(page, 'segment')
        task_id, _ = self._job_identity(page)
        labels = page.evaluate(
            """async (taskId) => (await fetch(`/api/labels?task_id=${taskId}`, {
              credentials: 'same-origin'
            })).json()""",
            task_id,
        )
        self.assertEqual({'polyline'}, {label['type'] for label in labels['results']})
        segmentation = self._job_annotations(page)
        self.assertEqual(50, len(segmentation['shapes']))
        existing_shape_ids = {shape['id'] for shape in segmentation['shapes']}
        self.assertEqual(50, len(existing_shape_ids))
        segment_job = _acceptance_job(runtime, data, 'segment')
        self.assertIsNotNone(segment_job)
        assert segment_job is not None
        segment_records_before = repository.annotations(step_key='segment')
        self.assertEqual(retained_segments, segment_records_before)
        segment_bindings_before = repository.bindings(segment_job.ref)
        self.assertEqual(existing_shape_ids, {binding.object_id for binding in segment_bindings_before})
        self._draw_two_point_polyline(page)
        self._draw_two_point_polyline(page)
        self._complete_with_plugin(page, 'segment')
        segment_records = repository.annotations(step_key='segment')
        self.assertEqual(52, len(segment_records))
        self.assertTrue({record.id for record in retained_segments} < {record.id for record in segment_records})
        self.assertTrue(removed_segment_ids.isdisjoint(record.id for record in segment_records))
        new_segments = tuple(
            record for record in segment_records if record.id not in {item.id for item in segment_records_before}
        )
        self.assertEqual(2, len(new_segments))
        segment_bindings = repository.bindings(segment_job.ref)
        self.assertTrue(set(segment_bindings_before) < set(segment_bindings))
        for new_segment in new_segments:
            self.assertEqual(segment_job.frames[0].parent_id, new_segment.parent_id)
            self.assertEqual(segment_job.frames[0].image_id, new_segment.image_id)
            new_segment_binding = next(
                binding for binding in segment_bindings if binding.annotation_id == new_segment.id
            )
            self.assertNotIn(new_segment_binding.object_id, existing_shape_ids)

        previous_segment_ref = segment_job.ref
        self._open_target(page, 'segment')
        segment_job = _acceptance_job(runtime, data, 'segment')
        self.assertIsNotNone(segment_job)
        assert segment_job is not None
        self.assertNotEqual(previous_segment_ref, segment_job.ref)
        segment_bindings_before = repository.bindings(segment_job.ref)
        invalid_segment = self._job_annotations(page)
        invalid_line = dict(invalid_segment['shapes'][0])
        invalid_line.pop('id', None)
        invalid_line['points'] = [60, 140, 120, 100, 180, 60]
        self._create_malformed_annotations(page, {'tags': [], 'shapes': [invalid_line], 'tracks': []})
        malformed_segment = self._job_annotations(page)
        self.assertEqual(53, len(malformed_segment['shapes']))
        self.assertEqual(invalid_segment['shapes'], malformed_segment['shapes'][:-1])
        self._complete_malformed_via_api(page)
        expect(page.locator('#segment-error')).to_contain_text('必须恰好有两个点')
        self.assertEqual(segment_records, repository.annotations(step_key='segment'))
        self.assertEqual(segment_bindings_before, repository.bindings(segment_job.ref))
        correction = page.locator('#segment-correction')
        expect(correction).to_be_visible()
        with page.expect_navigation(wait_until='domcontentloaded'):
            correction.click()
        self._delete_last_object(page)
        self._draw_two_point_polyline(page)
        self._complete_with_plugin(page, 'segment')
        expect(page.locator('#segment-annotated-count')).to_have_text('51')
        corrected_segment_records = repository.annotations(step_key='segment')
        self.assertEqual(53, len(corrected_segment_records))
        self.assertTrue({record.id for record in segment_records} < {record.id for record in corrected_segment_records})
        self.assertTrue(set(segment_bindings_before) < set(repository.bindings(segment_job.ref)))

        records = repository.annotations()
        parents = {record.id: record.image_id for record in records if record.step_key == 'detect'}
        self.assertTrue(
            all(parents[record.parent_id] == record.image_id for record in records if record.step_key != 'detect')
        )
        self.assertTrue(all(frame.image_id == parents[frame.parent_id] for frame in segment_job.frames))
        page.locator('#classify-cache-action').click()
        expect(page.locator('#classify-cache-action')).to_have_text('训练缓存已生成')
        page.locator('#segment-cache-action').click()
        expect(page.locator('#segment-cache-action')).to_have_text('训练缓存已生成')
        for target in ('classify', 'segment'):
            fingerprint = data.target_fingerprint(target)
            publication = Path(self.receipt['runtime_dir']) / 'cache' / fingerprint
            self.assertEqual(
                {'fingerprint': fingerprint, 'target': target},
                json.loads((publication / 'manifest.json').read_text(encoding='utf-8')),
            )
            image_path = next(
                path for path in (publication / target).rglob('*') if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}
            )
            with Image.open(image_path) as image:
                image.verify()
        self._assert_immutable_inputs()


if __name__ == '__main__':
    unittest.main()
