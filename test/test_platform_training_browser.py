from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from xxtrain.platform.app import create_app
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import PlatformAccessError, WorkspaceView
from xxtrain.platform.training_contracts import ExecutionView, TrainingRun, TrainingRunView


class _Client:
    def __init__(self, app: Any) -> None:
        self.loop = asyncio.new_event_loop()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver')

    def get(self, path: str) -> httpx.Response:
        return self.loop.run_until_complete(self.client.get(path))

    def close(self) -> None:
        self.loop.run_until_complete(self.client.aclose())
        self.loop.close()


class _Annotations:
    def view(self, user_id: int) -> WorkspaceView:
        return WorkspaceView('line-3', '三号现场', 50, 50, 50, True, False)


class _Cvat:
    def current_user(self, cookie: str) -> int:
        if 'sessionid=active' not in cookie:
            raise PlatformAccessError('expired')
        return 17


class _Training:
    def __init__(self) -> None:
        run = TrainingRun(
            str(uuid4()),
            17,
            'line-3',
            '三号现场',
            'detect',
            'fingerprint',
            'cache/private',
            '2026-09-17T00:00:00+00:00',
            'attempt',
            'clearml-private',
        )
        self.view = TrainingRunView(run, ExecutionView('private', 'queued', True, None, None, 3.0, None, False, None))

    def workspace_view(self, user_id: int) -> dict[str, object]:
        return {'workspace': _Annotations().view(user_id), 'editable': False, 'training': {'detect': self.view}}

    def list_runs(self, user_id: int) -> tuple[TrainingRunView, ...]:
        return (self.view,)


class PlatformTrainingBrowserTest(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        config = WorkspaceConfig('line-3', '三号现场', 17, root, root / 'runtime', 'http://cvat.test')
        self.training = _Training()
        self.client = _Client(create_app(config, _Annotations(), _Cvat(), training_service=self.training))
        self.addCleanup(self.client.close)
        self.client.client.cookies.set('sessionid', 'active')

    def test_training_page_and_resources_are_separate_from_workspace(self) -> None:
        workspace = self.client.get('/platform/')
        page = self.client.get('/platform/training/')
        script = self.client.get('/platform/training.js')

        self.assertEqual(200, page.status_code)
        self.assertEqual(200, script.status_code)
        self.assertIn('我的训练任务', page.text)
        self.assertIn('src="/platform/training.js"', page.text)
        self.assertNotIn('id="training-list"', workspace.text)
        self.assertIn('现场工作区', workspace.text)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_workspace_submits_without_navigation_and_keeps_run_as_task_link(self) -> None:
        page = self.client.get('/platform/').text
        script = self.client.get('/platform/app.js').text
        ids = re.findall(r'id="([^"]+)"', page)
        run_id = self.training.view.run.id
        harness = f"""
(async () => {{
const ids = {json.dumps(ids)};
const elements = new Map();
function element(id) {{ return {{id, hidden:false, disabled:false, textContent:'', value:'', files:[], dataset:{{}},
  listeners:{{}}, classList:{{toggle(){{}}}}, addEventListener(n, f){{(this.listeners[n] ||= []).push(f)}},
  dispatchEvent(e){{for(const f of this.listeners[e.type] || []) f(e)}},
  setAttribute(){{}}, removeAttribute(){{}}, reset(){{}}}} }}
ids.forEach((id) => elements.set(id, element(id)));
let loaded; let assigned = null;
globalThis.document = {{cookie:'xxtrain_csrf=token', getElementById:(id)=>elements.get(id), querySelectorAll:()=>[],
  addEventListener:(name, fn)=>{{if(name==='DOMContentLoaded') loaded=fn}}}};
globalThis.location = {{search:'', assign:(url)=>{{assigned=url}}}};
globalThis.history = {{replaceState(){{}}}};
globalThis.sessionStorage = (()=>{{const values=new Map(); return {{getItem:(k)=>values.get(k)||null,
  setItem:(k,v)=>values.set(k,String(v)), removeItem:(k)=>values.delete(k)}}}})();
globalThis.crypto = {{randomUUID:()=> '11111111-1111-4111-8111-111111111111'}};
globalThis.setTimeout = ()=>1; globalThis.clearTimeout = ()=>{{}};
const baseTargets = [
  {{id:'detect', name:'检测', available:true, sample_count:50, annotated_sample_count:50,
    can_annotate:true, can_generate_cache:true, cache_ready:false}},
  {{id:'classify', name:'分类', available:true, sample_count:40, annotated_sample_count:40,
    can_annotate:true, can_generate_cache:true, cache_ready:false}},
  {{id:'segment', name:'分割', available:true, sample_count:40, annotated_sample_count:0,
    can_annotate:false, can_generate_cache:false, cache_ready:false}},
];
const execution = {{status:'queued', active:true, epoch:null, total_epochs:null, elapsed_seconds:3,
  metric:null, download_ready:false, detail:null}};
const run = {{id:{json.dumps(run_id)}, workspace_id:'line-3', workspace_name:'三号现场', target:'detect',
  submitted_at:'2026-09-17T00:00:00+00:00', execution}};
const workspace = {{workspace_id:'line-3', name:'三号现场', image_count:50,
  task:{{id:'point',name:'Point'}}, targets:baseTargets,
  editing_locked:true, training_enabled:true, training:{{detect:run, classify:null, segment:null}}}};
const calls=[];
globalThis.fetch = async (url, options={{}}) => {{ calls.push({{url, body:options.body||null}});
  const body = url.endsWith('/session') ? {{authenticated:true,user_id:17}}
    : url.endsWith('/train') ? {{run_id:{json.dumps(run_id)}, training_url:'/platform/training/?run={run_id}', run}}
    : workspace;
  return {{ok:true,status:200,json:async()=>body}};
}};
eval({json.dumps(script)}); await loaded();
const get=(id)=>elements.get(id); const detect=get('cache-action');
const queued=detect.textContent; detect.dispatchEvent(new Event('mouseenter')); const hovered=detect.textContent;
detect.dispatchEvent(new Event('mouseleave')); const restored=detect.textContent;
const annotationDisabled=get('primary-action').disabled;
await get('classify-cache-action').listeners.click[0]();
process.stdout.write(JSON.stringify({{queued,hovered,restored,annotationDisabled,classifyDisabled:get('classify-cache-action').disabled,
  notification:get('workspace-message').textContent,assigned,calls, trainingEnabled:workspace.training_enabled}}));
}})().catch((error)=>{{console.error(error);process.exitCode=1}});
"""
        completed = subprocess.run(['node', '-e', harness], text=True, encoding='utf-8', capture_output=True)
        self.assertEqual(0, completed.returncode, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual('排队中', result['queued'])
        self.assertEqual('查看训练任务', result['hovered'])
        self.assertEqual('排队中', result['restored'])
        self.assertTrue(result['annotationDisabled'])
        self.assertIn('trainingEnabled', result, result)
        self.assertTrue(result['trainingEnabled'])
        self.assertTrue(
            any(call['url'].endswith('/targets/classify/train') for call in result['calls']), result['calls']
        )
        self.assertEqual('已加入训练队列', result['notification'])
        self.assertIsNone(result['assigned'])


if __name__ == '__main__':
    unittest.main()
