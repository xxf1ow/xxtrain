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

from test.task_definitions import synthetic_task_definition
from xxtrain.platform.app import create_app
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import PlatformAccessError, TargetView, WorkspaceView
from xxtrain.platform.training_contracts import ExecutionView, TrainingRun, TrainingRunView


class _Client:
    def __init__(self, app: Any) -> None:
        self.loop = asyncio.new_event_loop()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url='http://testserver'
        )

    def get(self, path: str) -> httpx.Response:
        return self.loop.run_until_complete(self.client.get(path))

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.loop.run_until_complete(self.client.post(path, **kwargs))

    def close(self) -> None:
        self.loop.run_until_complete(self.client.aclose())
        self.loop.close()


class _Cvat:
    def current_user(self, cookie: str) -> int:
        if 'sessionid=active' not in cookie:
            raise PlatformAccessError('expired')
        return 17


class _Data:
    task = synthetic_task_definition()


class _Annotations:
    data = _Data()

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.workspace = WorkspaceView(
            'synthetic-workspace',
            'Synthetic workspace',
            3,
            (
                TargetView('regions', 3, 3, True, False, False, 'Regions', 'images'),
                TargetView('kind', 4, 2, True, True, False, 'Kind', 'crops'),
                TargetView('needles', 4, 0, True, False, False, 'Needles', 'crops'),
                TargetView('subregions', 4, 1, True, False, False, 'Subregions', 'crops'),
                TargetView('details', 1, 0, False, False, False, 'Details', 'crops'),
            ),
            'synthetic',
            'Synthetic task',
        )

    def view(self, user_id: int) -> WorkspaceView:
        return self.workspace

    def begin_target(self, user_id: int, target: str) -> str:
        self.calls.append(target)
        return '/tasks/41/jobs/73'


class _Training:
    def __init__(self, annotations: _Annotations) -> None:
        run = TrainingRun(
            str(uuid4()),
            17,
            'synthetic-workspace',
            'Synthetic workspace',
            'kind',
            'fingerprint',
            'cache/private',
            '2026-09-20T00:00:00+00:00',
            None,
            None,
            'execute',
            'test.task_definitions:synthetic_training_task_definition',
        )
        self.run = TrainingRunView(run, ExecutionView('', 'queued', True, None, None, None, None, False, None))
        self.annotations = annotations

    def workspace_view(self, user_id: int) -> dict[str, object]:
        return {
            'workspace': self.annotations.view(user_id),
            'can_upload': False,
            'target_editable': {'regions': False, 'kind': True, 'needles': False, 'subregions': True, 'details': True},
            'training': {'kind': self.run},
        }

    def list_runs(self, user_id: int) -> tuple[TrainingRunView, ...]:
        return (self.run,)


class TaskDefinitionHttpTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.config = WorkspaceConfig(
            'synthetic-workspace',
            'Synthetic workspace',
            17,
            root,
            root / 'runtime',
            'http://cvat.test',
            'test.task_definitions:synthetic_task_definition',
        )
        self.annotations = _Annotations()
        self.training = _Training(self.annotations)
        self.client = _Client(create_app(self.config, self.annotations, _Cvat(), training_service=self.training))
        self.addCleanup(self.client.close)
        self.client.client.cookies.set('sessionid', 'active')
        self.client.get('/platform/')
        token = self.client.client.cookies.get('xxtrain_csrf')
        self.headers = {'origin': 'http://testserver', 'x-xtrain-csrf': token}

    def test_workspace_uses_selected_task_order_metadata_permissions_and_training_summary(self) -> None:
        response = self.client.get('/platform/api/workspace')

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual('synthetic', payload['task']['id'])
        self.assertEqual('Synthetic task', payload['task']['name'])
        self.assertEqual(
            ['regions', 'kind', 'needles', 'subregions', 'details'], [row['id'] for row in payload['targets']]
        )
        self.assertEqual(('Kind', 'crops'), (payload['targets'][1]['name'], payload['targets'][1]['sample_unit']))
        self.assertFalse(payload['can_upload'])
        self.assertFalse(payload['targets'][0]['editable'])
        self.assertTrue(payload['targets'][1]['editable'])
        self.assertEqual(self.training.run.run.id, payload['targets'][1]['training']['id'])
        self.assertNotIn('editable', payload)
        self.assertNotIn('training', payload)

    def test_selected_definition_is_the_only_target_allowlist(self) -> None:
        unknown = self.client.post('/platform/api/targets/detect/start', json={}, headers=self.headers)
        accepted = self.client.post('/platform/api/targets/subregions/start', json={}, headers=self.headers)

        self.assertEqual(404, unknown.status_code)
        self.assertEqual(200, accepted.status_code)
        self.assertEqual(['subregions'], self.annotations.calls)

    def test_training_summary_uses_the_immutable_run_task_definition(self) -> None:
        response = self.client.get('/platform/api/training-runs')

        self.assertEqual(200, response.status_code)
        run = response.json()[0]
        self.assertEqual('Synthetic task', run['task']['name'])
        self.assertEqual('Kind', run['target_name'])
        self.assertEqual('Synthetic kind quality', run['metric_name'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_dynamic_workspace_keeps_unlocked_sibling_enabled_and_opens_training_history(self) -> None:
        page = self.client.get('/platform/').text
        script = self.client.get('/platform/app.js').text
        ids = re.findall(r'id="([^"]+)"', page)
        run_id = self.training.run.run.id
        payload = {
            'workspace_id': 'synthetic-workspace',
            'name': 'Synthetic workspace',
            'image_count': 3,
            'task': {'id': 'synthetic', 'name': 'Synthetic task'},
            'can_upload': False,
            'training_enabled': True,
            'targets': [
                {
                    'id': target.id,
                    'name': target.display_name,
                    'sample_unit': target.sample_unit,
                    'sample_count': target.sample_count,
                    'annotated_sample_count': target.annotated_sample_count,
                    'can_annotate': target.can_annotate,
                    'can_generate_cache': target.can_generate_cache,
                    'cache_ready': target.cache_ready,
                    'editable': target.id in {'kind', 'subregions', 'details'},
                    'training': None,
                }
                for target in self.annotations.workspace.targets
            ],
        }
        payload['targets'][1]['training'] = {
            'id': run_id,
            'target': 'kind',
            'execution': {'status': 'queued', 'active': True, 'download_ready': False},
        }
        payload['targets'][1]['editable'] = False
        harness = f"""
(async () => {{
class Element {{
  constructor(tag, id='') {{ this.tagName=tag; this.id=id; this.children=[]; this.listeners={{}}; this.attributes={{}};
    this.dataset={{}}; this.hidden=false; this.disabled=false; this.textContent=''; this.className=''; this.value='';
    this.files=[]; this.href=''; this.type=''; this.classList={{toggle(){{}},add(){{}},remove(){{}}}}; }}
  append(...nodes) {{ this.children.push(...nodes); }}
  replaceChildren(...nodes) {{ this.children=[...nodes]; }}
  addEventListener(name, fn) {{ (this.listeners[name] ||= []).push(fn); }}
  async fire(name) {{ for (const fn of this.listeners[name] || []) await fn({{preventDefault(){{}}}}); }}
  setAttribute(name, value) {{ this.attributes[name]=String(value); }}
  removeAttribute(name) {{ delete this.attributes[name]; }}
  reset() {{}}
}}
const elements=new Map({json.dumps(ids)}.map((id)=>[id,new Element('div',id)]));
let loaded; let assigned=null;
globalThis.document={{cookie:'xxtrain_csrf=token',getElementById:(id)=>elements.get(id),
  createElement:(tag)=>new Element(tag),querySelectorAll:()=>[],
  addEventListener:(name,fn)=>{{if(name==='DOMContentLoaded')loaded=fn}}}};
globalThis.location={{search:'',assign:(url)=>{{assigned=url}}}};
globalThis.history={{replaceState(){{}}}};
globalThis.sessionStorage={{getItem:()=>null,setItem(){{}},removeItem(){{}}}};
globalThis.FormData=class{{append(){{}}}};
globalThis.setTimeout=()=>1; globalThis.clearTimeout=()=>{{}};
const workspace={json.dumps(payload)};
globalThis.fetch=async(url)=>{{
  if(url.endsWith('/session'))return{{ok:true,status:200,json:async()=>({{authenticated:true,user_id:17}})}};
  if(url.endsWith('/workspace'))return{{ok:true,status:200,json:async()=>workspace}};
  throw new Error(`unexpected ${{url}}`);
}};
function all(node) {{return [node,...node.children.flatMap(all)]}}
eval({json.dumps(script)}); await loaded();
const rail=elements.get('target-rail');
const kind=all(rail).find((node)=>node.dataset.target==='kind');
const buttons=all(kind).filter((node)=>node.tagName==='button');
await buttons[1].fire('click');
process.stdout.write(JSON.stringify({{
  cardAriaDisabled:kind.attributes['aria-disabled'] ?? null,
  annotateDisabled:buttons[0].disabled,
  annotateAriaDisabled:buttons[0].attributes['aria-disabled'],
  trainDisabled:buttons[1].disabled,
  trainAriaDisabled:buttons[1].attributes['aria-disabled'],
  assigned,
}}));
}})().catch((error)=>{{console.error(error);process.exitCode=1}});
"""
        completed = subprocess.run(['node', '-e', harness], text=True, encoding='utf-8', capture_output=True)

        self.assertEqual(0, completed.returncode, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertIsNone(result['cardAriaDisabled'])
        self.assertTrue(result['annotateDisabled'])
        self.assertEqual('true', result['annotateAriaDisabled'])
        self.assertFalse(result['trainDisabled'])
        self.assertEqual('false', result['trainAriaDisabled'])
        self.assertEqual(f'/platform/training/?run={run_id}', result['assigned'])


if __name__ == '__main__':
    unittest.main()
