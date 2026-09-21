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
from xxtrain.platform.contracts import PlatformAccessError, TargetView, WorkspaceView
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
        return WorkspaceView(
            'line-3',
            '三号现场',
            50,
            (
                TargetView('detect', 50, 50, True, True, False, '检测', '张图片'),
                TargetView('classify', 40, 40, True, True, False, '分类', '张裁剪图'),
                TargetView('segment', 40, 0, True, False, False, '指针分割', '张裁剪图'),
            ),
        )


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
        return {
            'workspace': _Annotations().view(user_id),
            'can_upload': False,
            'target_editable': {'detect': False, 'classify': True, 'segment': True},
            'training': {'detect': self.view, 'classify': None, 'segment': None},
        }

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
function element(id, tag='div') {{ return {{id, tagName:tag, children:[], className:'', hidden:false,
  disabled:false, textContent:'', value:'', files:[], dataset:{{}},
  listeners:{{}}, classList:{{toggle(){{}}}}, addEventListener(n, f){{(this.listeners[n] ||= []).push(f)}},
  dispatchEvent(e){{for(const f of this.listeners[e.type] || []) f(e)}},
  append(...nodes){{this.children.push(...nodes)}}, replaceChildren(...nodes){{this.children=[...nodes]}},
  setAttribute(){{}}, removeAttribute(){{}}, reset(){{}}}} }}
ids.forEach((id) => elements.set(id, element(id)));
let loaded; let assigned = null;
globalThis.document = {{cookie:'xxtrain_csrf=token', getElementById:(id)=>elements.get(id),
  createElement:(tag)=>element('',tag), querySelectorAll:()=>[],
  addEventListener:(name, fn)=>{{if(name==='DOMContentLoaded') loaded=fn}}}};
globalThis.location = {{search:'?returned=1', assign:(url)=>{{assigned=url}}}};
globalThis.history = {{replaceState(_state,_title,url){{location.search='';location.pathname=url}}}};
const trainingStorageWrites=[];
globalThis.sessionStorage = (()=>{{
  const values=new Map([['xxtrain-return-target','detect']]); return {{getItem:(k)=>values.get(k)||null,
  setItem:(k,v)=>{{trainingStorageWrites.push(k);values.set(k,String(v))}}, removeItem:(k)=>values.delete(k)}}}})();
globalThis.FormData = class {{ append() {{}} }};
let cryptoCalls = 0;
globalThis.crypto = {{randomUUID:()=>{{cryptoCalls += 1; return 'unused'}}}};
globalThis.setTimeout = ()=>1; globalThis.clearTimeout = ()=>{{}};
const baseTargets = [
  {{id:'detect', name:'检测', sample_unit:'张图片', sample_count:50, annotated_sample_count:50,
    can_annotate:true, can_generate_cache:true, cache_ready:false, editable:false, training:null}},
  {{id:'classify', name:'分类', sample_unit:'张裁剪图', sample_count:40, annotated_sample_count:40,
    can_annotate:true, can_generate_cache:true, cache_ready:false, editable:true, training:null}},
  {{id:'segment', name:'指针分割', sample_unit:'张裁剪图', sample_count:40, annotated_sample_count:0,
    can_annotate:false, can_generate_cache:true, cache_ready:false, editable:false, training:null}},
];
const execution = {{status:'queued', active:true, epoch:null, total_epochs:null, elapsed_seconds:3,
  metric:null, download_ready:false, detail:null}};
const run = {{id:{json.dumps(run_id)}, workspace_id:'line-3', workspace_name:'三号现场', target:'detect',
  submitted_at:'2026-09-17T00:00:00+00:00', execution}};
baseTargets[0].training = run;
const workspace = {{workspace_id:'line-3', name:'三号现场', image_count:50,
  task:{{id:'point',name:'Point'}}, targets:baseTargets, can_upload:false, training_enabled:true}};
const calls=[];
let submittedRun = null;
let failWorkspace = false;
let failTrain = true;
globalThis.fetch = async (url, options={{}}) => {{ calls.push({{url, body:options.body||null}});
  if (url.endsWith('/train')) {{
    const target = url.includes('/classify/') ? 'classify' : 'segment';
    submittedRun = {{...run, id:target === 'classify' ? '33333333-3333-4333-8333-333333333333'
      : '44444444-4444-4444-8444-444444444444', target}};
    if (failTrain) return {{ok:false,status:502,json:async()=>({{detail:'提交结果未知'}})}};
  }}
  if (failWorkspace && url.endsWith('/workspace')) return {{ok:false,status:502,json:async()=>({{detail:'读取失败'}})}};
  const body = url.endsWith('/session') ? {{authenticated:true,user_id:17}}
    : url.endsWith('/train') ? {{run_id:submittedRun.id,
        training_url:`/platform/training/?run=${{submittedRun.id}}`, run:submittedRun}}
    : submittedRun
      ? {{...workspace, targets:workspace.targets.map((target)=>target.id === submittedRun.target
          ? {{...target, training:submittedRun}} : target)}}
      : workspace;
  return {{ok:true,status:200,json:async()=>body}};
}};
eval({json.dumps(script)}); await loaded();
function all(node){{return[node,...node.children.flatMap(all)]}}
function buttons(target){{const card=all(elements.get('target-rail')).find((node)=>node.dataset.target===target);
  return all(card).filter((node)=>node.tagName==='button')}}
const detect=buttons('detect')[1];
const queued=detect.textContent; detect.dispatchEvent(new Event('mouseenter')); const hovered=detect.textContent;
detect.dispatchEvent(new Event('mouseleave')); const restored=detect.textContent;
const annotationDisabled=buttons('detect')[0].disabled;
elements.get('image-files').files=[{{name:'duplicate.png'}}];
await elements.get('image-files').listeners.change[0]();
await buttons('classify')[1].listeners.click[0]();
const recoveredRunId=buttons('classify')[1].dataset.runId;
const recoveredText=buttons('classify')[1].textContent;
failWorkspace = true;
await buttons('segment')[1].listeners.click[0]();
const disabledAfterReadFailure=buttons('segment')[1].disabled;
failWorkspace = false;
failTrain = false;
await loaded();
const reconstructedRunId=buttons('segment')[1].dataset.runId;
const trainingBodies = calls.filter((call)=>call.url.endsWith('/train')).map((call)=>call.body);
process.stdout.write(JSON.stringify({{queued,hovered,restored,annotationDisabled,classifyDisabled:buttons('classify')[1].disabled,
  recoveredRunId,recoveredText,disabledAfterReadFailure,reconstructedRunId,assigned,calls,trainingBodies,
  cryptoCalls,trainingStorageWrites,trainingEnabled:workspace.training_enabled}}));
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
        self.assertTrue(any(call['url'].endswith('/targets/detect/sync') for call in result['calls']), result['calls'])
        self.assertFalse(any(call['url'].endswith('/images') for call in result['calls']), result['calls'])
        self.assertFalse(any(call['url'].endswith('/cache') for call in result['calls']), result['calls'])
        self.assertEqual([{}, {}], [json.loads(body) for body in result['trainingBodies']])
        self.assertEqual(0, result['cryptoCalls'])
        self.assertEqual([], result['trainingStorageWrites'])
        self.assertEqual('33333333-3333-4333-8333-333333333333', result['recoveredRunId'])
        self.assertEqual('排队中', result['recoveredText'])
        self.assertTrue(result['disabledAfterReadFailure'])
        self.assertEqual('44444444-4444-4444-8444-444444444444', result['reconstructedRunId'])
        self.assertIsNone(result['assigned'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_training_page_executes_actions_and_polling_lifecycle(self) -> None:
        script = self.client.get('/platform/training.js').text
        run_id = self.training.view.run.id
        completed_id = str(uuid4())
        missing_id = str(uuid4())
        cancelling_id = str(uuid4())
        unavailable_id = str(uuid4())
        harness = f"""
(async () => {{
class Element {{
  constructor(tag, id='') {{ this.tagName=tag; this.id=id; this.children=[]; this.listeners={{}}; this.attributes={{}};
    this.hidden=false; this.disabled=false; this.textContent=''; this.className=''; this.href=''; this.type=''; }}
  append(...nodes) {{ this.children.push(...nodes); }}
  replaceChildren(...nodes) {{ this.children=[...nodes]; }}
  addEventListener(name, fn) {{ (this.listeners[name] ||= []).push(fn); }}
  async fire(name, event={{preventDefault(){{}}}}) {{ for (const fn of this.listeners[name] || []) await fn(event); }}
  setAttribute(name, value) {{ this.attributes[name]=String(value); }}
}}
const elementIds=['training-list','training-detail','training-error','training-message'];
const elements=new Map(elementIds.map((id)=>[id,new Element('div',id)]));
const documentListeners={{}}; const windowListeners={{}};
globalThis.document={{cookie:'xxtrain_csrf=task-token',hidden:false,getElementById:(id)=>elements.get(id),
  createElement:(tag)=>new Element(tag),addEventListener:(name,fn)=>{{documentListeners[name]=fn}}}};
globalThis.window={{addEventListener:(name,fn)=>{{windowListeners[name]=fn}}}};
let assigned=null; globalThis.location={{search:'?run={missing_id}',assign:(url)=>{{assigned=url}}}};
globalThis.history={{replaceState(){{}}}};
const timers=new Map(); let timerCounter=0;
globalThis.setTimeout=(fn,ms)=>{{timers.set(++timerCounter,{{fn,ms}});return timerCounter}};
globalThis.clearTimeout=(id)=>timers.delete(id);
const missing={{id:'{missing_id}',workspace_id:'line-3',workspace_name:'三号现场',target:'segment',
  target_name:'指针分割模型',
  submitted_at:'2026-09-17T00:00:00+00:00',execution:null}};
const active={{id:'{run_id}',workspace_id:'line-3',workspace_name:'三号现场',target:'detect',target_name:'检测模型',
  submitted_at:'2026-09-17T00:01:00+00:00',execution:{{status:'unknown',active:true,epoch:null,
    total_epochs:null,elapsed_seconds:null,metric:null,download_ready:false,detail:'暂时不可用'}}}};
const completed={{id:'{completed_id}',workspace_id:'line-3',workspace_name:'三号现场',target:'classify',
  target_name:'分类模型',
  metric_name:'分类准确率：Top-1',submitted_at:'2026-09-17T00:02:00+00:00',cancellation_requested:true,execution:{{status:'completed',active:false,epoch:10,
    total_epochs:10,elapsed_seconds:30,metric:0.8,download_ready:true,detail:null}}}};
const cancelling={{...active,id:'{cancelling_id}',target:'segment',target_name:'指针分割模型',
  cancellation_requested:true,
  execution:{{...active.execution,status:'running'}}}};
const unavailable={{...completed,id:'{unavailable_id}',target:'detect',target_name:'检测模型',
  cancellation_requested:false,
  execution:{{...completed.execution,download_ready:false}}}};
let runs=[missing,active,completed,cancelling,unavailable]; let failList=false; let authFail=false; const calls=[];
globalThis.fetch=async(url,options={{}})=>{{calls.push({{url,method:options.method||'GET',body:options.body||null,csrf:options.headers?.['X-XTrain-CSRF']||null}});
  if(authFail&&url.endsWith('/training-runs')) return {{ok:false,status:401,json:async()=>({{detail:'expired'}})}};
  if(url.endsWith('/cancel')) return {{ok:true,status:200,json:async()=>active}};
  if(url.endsWith('/training-runs')&&failList) return {{ok:false,status:502,json:async()=>({{detail:'轮询失败'}})}};
  if(url.endsWith('/training-runs')) return {{ok:true,status:200,json:async()=>runs}};
  throw new Error(`unexpected ${{url}}`);
}};
function all(node) {{return [node,...node.children.flatMap(all)]}}
function byText(root,value) {{return all(root).find((node)=>node.textContent===value)}}
eval({json.dumps(script)}); await documentListeners.DOMContentLoaded();
const detail=elements.get('training-detail');
const missingState={{cancel:byText(detail,'取消训练').disabled,
  download:byText(detail,'下载部署产物').attributes['aria-disabled'],retry:Boolean(byText(detail,'重新训练'))}};
const taskButtons=elements.get('training-list').children;
await taskButtons.find((item)=>all(item).some((node)=>node.textContent==='指针分割模型')).fire('click');
const cancellationWaiting=all(detail).some((node)=>node.textContent==='取消请求已保存，等待停止');
const cancellationDisabled=byText(detail,'等待停止').disabled;
await taskButtons.filter((item)=>all(item).some((node)=>node.textContent==='检测模型'))[0].fire('click');
const outputUnavailable=all(detail).some((node)=>node.textContent==='训练已结束，部署产物不可用');
const unavailableDownload=byText(detail,'下载部署产物').attributes['aria-disabled'];
await taskButtons.find((item)=>all(item).some((node)=>node.textContent==='分类模型')).fire('click');
const downloadHref=byText(detail,'下载部署产物').href;
const metricText=Boolean(byText(detail,'本次验证集结果 · 分类准确率：Top-1：80.0%'));
failList=true; await documentListeners.visibilitychange(); await new Promise((resolve)=>setImmediate(resolve));
const retainedAfterFailure=all(detail).some((node)=>node.textContent==='分类模型'); failList=false;
await taskButtons.filter((item)=>all(item).some((node)=>node.textContent==='检测模型'))[1].fire('click');
const unknownState={{cancel:byText(detail,'取消训练').disabled,
  download:byText(detail,'下载部署产物').attributes['aria-disabled'],retry:Boolean(byText(detail,'重新训练'))}};
await byText(detail,'取消训练').fire('click');
const cancelCall=calls.find((call)=>call.url.endsWith('/cancel'));
const scheduledWhileAnotherActive=[...timers.values()].some((timer)=>timer.ms===5000);
document.hidden=true; await documentListeners.visibilitychange(); const paused=timers.size===0;
document.hidden=false; await documentListeners.visibilitychange(); await new Promise((resolve)=>setImmediate(resolve));
const resumed=[...timers.values()].some((timer)=>timer.ms===5000);
authFail=true; await documentListeners.visibilitychange(); await new Promise((resolve)=>setImmediate(resolve));
process.stdout.write(JSON.stringify({{missingState,unknownState,downloadHref,metricText,cancellationWaiting,cancellationDisabled,
  outputUnavailable,unavailableDownload,retainedAfterFailure,cancelCall,
  scheduledWhileAnotherActive,paused,resumed,assigned,retryCallCount:calls.filter((call)=>call.url.endsWith('/retry')).length}}));
}})().catch((error)=>{{console.error(error);process.exitCode=1}});
"""
        completed = subprocess.run(['node', '-e', harness], text=True, encoding='utf-8', capture_output=True)
        self.assertEqual(0, completed.returncode, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual({'cancel': True, 'retry': False, 'download': 'true'}, result['missingState'])
        self.assertEqual({'cancel': False, 'retry': False, 'download': 'true'}, result['unknownState'])
        self.assertEqual(f'/platform/api/training-runs/{completed_id}/download', result['downloadHref'])
        self.assertTrue(result['metricText'])
        self.assertTrue(result['cancellationWaiting'])
        self.assertTrue(result['cancellationDisabled'])
        self.assertTrue(result['outputUnavailable'])
        self.assertEqual('true', result['unavailableDownload'])
        self.assertTrue(result['retainedAfterFailure'])
        self.assertEqual('task-token', result['cancelCall']['csrf'])
        self.assertTrue(result['scheduledWhileAnotherActive'])
        self.assertTrue(result['paused'])
        self.assertTrue(result['resumed'])
        self.assertEqual(f'/platform/?return_run={run_id}', result['assigned'])
        self.assertEqual(0, result['retryCallCount'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the offline browser-script check')
    def test_workspace_submission_feedback_matches_observed_execution(self) -> None:
        page = self.client.get('/platform/').text
        script = self.client.get('/platform/app.js').text
        ids = re.findall(r'id="([^"]+)"', page)
        expectations = {
            'pending': '提交已保存，等待确认',
            'queued': '已加入训练队列',
            'running': '训练中',
            'completed': '训练已结束，部署产物不可用',
        }
        for execution_status, expected in expectations.items():
            with self.subTest(status=execution_status):
                harness = f"""
(async () => {{
const ids={json.dumps(ids)}; const elements=new Map();
function element(id,tag='div') {{return {{id,tagName:tag,children:[],className:'',hidden:false,
  disabled:false,textContent:'',value:'',files:[],dataset:{{}},
  listeners:{{}},
  classList:{{toggle(){{}}}},addEventListener(n,f){{(this.listeners[n]||=[]).push(f)}},setAttribute(){{}},
  append(...nodes){{this.children.push(...nodes)}},replaceChildren(...nodes){{this.children=[...nodes]}},
  removeAttribute(){{}},reset(){{}}}}}}
ids.forEach((id)=>elements.set(id,element(id))); let loaded;
globalThis.document={{cookie:'xxtrain_csrf=token',getElementById:(id)=>elements.get(id),createElement:(tag)=>element('',tag),querySelectorAll:()=>[],
  addEventListener:(name,fn)=>{{if(name==='DOMContentLoaded')loaded=fn}}}};
globalThis.location={{search:'',assign(){{}}}}; globalThis.history={{replaceState(){{}}}};
globalThis.sessionStorage={{getItem:()=>null,setItem(){{}},removeItem(){{}}}};
globalThis.FormData=class{{append(){{}}}}; globalThis.setTimeout=()=>1; globalThis.clearTimeout=()=>{{}};
const targets=['detect','classify','segment'].map((id)=>({{id,name:id,sample_unit:'样本',sample_count:50,
  annotated_sample_count:50,can_annotate:true,can_generate_cache:true,cache_ready:false,editable:true,training:null}}));
const execution={{status:{json.dumps(execution_status)},
  active:{str(execution_status in {'pending', 'queued', 'running'}).lower()},
  epoch:null,total_epochs:null,elapsed_seconds:null,metric:null,download_ready:false,detail:null}};
const run={{id:'55555555-5555-4555-8555-555555555555',workspace_id:'line-3',workspace_name:'三号现场',
  target:'detect',submitted_at:'2026-09-17T00:00:00+00:00',execution,cancellation_requested:false}};
const base={{workspace_id:'line-3',name:'三号现场',image_count:50,task:{{id:'point',name:'Point'}},targets,
  can_upload:true,training_enabled:true}};
let submitted=false;
globalThis.fetch=async(url)=>{{
  if(url.endsWith('/session'))return{{ok:true,status:200,json:async()=>({{authenticated:true,user_id:17}})}};
  if(url.endsWith('/train')){{submitted=true;return{{ok:true,status:200,json:async()=>({{run_id:run.id,run}})}}}}
  if(url.endsWith('/workspace'))return{{ok:true,status:200,json:async()=>submitted
    ?{{...base,targets:base.targets.map((target)=>target.id==='detect'?{{...target,training:run}}:target)}}:base}};
  throw new Error(`unexpected ${{url}}`);
}};
eval({json.dumps(script)}); await loaded();
function all(node){{return[node,...node.children.flatMap(all)]}}
const card=all(elements.get('target-rail')).find((node)=>node.dataset.target==='detect');
const train=all(card).filter((node)=>node.tagName==='button')[1];
await train.listeners.click[0]();
process.stdout.write(JSON.stringify({{message:elements.get('workspace-message').textContent}}));
}})().catch((error)=>{{console.error(error);process.exitCode=1}});
"""
                completed = subprocess.run(['node', '-e', harness], text=True, encoding='utf-8', capture_output=True)
                self.assertEqual(0, completed.returncode, completed.stderr)
                message = json.loads(completed.stdout)['message']
                self.assertEqual(expected, message)
                if execution_status != 'queued':
                    self.assertNotEqual('已加入训练队列', message)


if __name__ == '__main__':
    unittest.main()
