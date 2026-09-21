(() => {
  'use strict';

  const apiRoot = '/platform/api';
  const statusNames = {pending: '等待确认', queued: '排队中', running: '训练中', completed: '训练完成', failed: '训练失败', cancelled: '已取消', unknown: '状态查询失败'};
  const elements = {
    list: document.getElementById('training-list'), detail: document.getElementById('training-detail'),
    error: document.getElementById('training-error'), message: document.getElementById('training-message'),
  };
  let runs = [];
  let selectedId = new URLSearchParams(location.search).get('run');
  let timer = null;
  let loading = false;

  function cookie(name) {
    const prefix = `${name}=`;
    const value = document.cookie.split(';').map((part) => part.trim()).find((part) => part.startsWith(prefix));
    return value ? decodeURIComponent(value.slice(prefix.length)) : '';
  }

  async function request(path, options = {}) {
    const response = await fetch(`${apiRoot}${path}`, {credentials: 'same-origin', ...options});
    if (!response.ok) {
      if (response.status === 401) {
        const returnRun = selectedId && /^[0-9a-f-]{36}$/.test(selectedId) ? `?return_run=${encodeURIComponent(selectedId)}` : '';
        location.assign(`/platform/${returnRun}`);
        throw new Error('登录已失效，请重新登录。');
      }
      let detail = '操作未完成，请重试。';
      try { const body = await response.json(); if (typeof body.detail === 'string') detail = body.detail; } catch (_) {}
      const error = new Error(detail); error.status = response.status; throw error;
    }
    return response.status === 204 ? null : response.json();
  }

  function post(path, body) {
    return request(path, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-XTrain-CSRF': cookie('xxtrain_csrf')}, body: JSON.stringify(body)});
  }

  function text(tag, value, className = '') {
    const node = document.createElement(tag); node.textContent = value; if (className) node.className = className; return node;
  }

  function status(run) {
    if (run.execution?.active && run.cancellation_requested) return '取消请求已保存，等待停止';
    if (run.execution?.status === 'completed' && !run.execution.download_ready) return '训练已结束，部署产物不可用';
    return statusNames[run.execution?.status] || '等待状态';
  }

  function renderList() {
    elements.list.replaceChildren();
    if (!runs.length) { elements.list.append(text('p', '还没有训练任务。请从现场工作区提交已完成标注的模型。', 'empty-state')); return; }
    for (const run of [...runs].reverse()) {
      const button = document.createElement('button'); button.type = 'button'; button.className = 'training-list-item';
      button.setAttribute('aria-pressed', String(run.id === selectedId));
      button.append(text('strong', run.target_name || '训练任务'));
      button.append(text('span', `${run.workspace_name} · ${status(run)}`));
      button.addEventListener('click', () => { selectedId = run.id; history.replaceState({}, '', `/platform/training/?run=${run.id}`); render(); });
      elements.list.append(button);
    }
  }

  function renderDetail() {
    const run = runs.find((item) => item.id === selectedId);
    elements.detail.replaceChildren();
    if (!run) { elements.detail.append(text('p', '选择一项任务查看进度和结果。', 'empty-state')); return; }
    const execution = run.execution;
    elements.detail.append(text('p', run.workspace_name, 'section-code'));
    elements.detail.append(text('h2', run.target_name || '训练任务'));
    elements.detail.append(text('p', status(run), 'run-status'));
    elements.detail.append(text('p', `提交时间：${new Date(run.submitted_at).toLocaleString('zh-CN')}`));
    const progress = execution?.epoch == null || execution?.total_epochs == null ? '暂无' : `${execution.epoch} / ${execution.total_epochs} epoch`;
    elements.detail.append(text('p', `训练进度：${progress}`));
    const elapsed = execution?.elapsed_seconds == null ? '暂无' : `${Math.round(execution.elapsed_seconds)} 秒`;
    elements.detail.append(text('p', `已用时间：${elapsed}`));
    const metric = execution?.metric == null ? '暂无' : `${(execution.metric * 100).toFixed(1)}%`;
    elements.detail.append(text('p', `本次验证集结果 · ${run.metric_name || '训练指标'}：${metric}`));
    const actions = document.createElement('div'); actions.className = 'training-actions';
    const download = text('a', '下载部署产物', 'primary-button'); download.href = execution?.download_ready ? `${apiRoot}/training-runs/${run.id}/download` : '#';
    download.setAttribute('aria-disabled', String(!execution?.download_ready)); if (!execution?.download_ready) download.addEventListener('click', (event) => event.preventDefault());
    actions.append(download);
    const cancel = text('button', run.cancellation_requested && execution?.active ? '等待停止' : '取消训练', 'primary-button'); cancel.type = 'button'; cancel.disabled = !execution?.active || Boolean(run.cancellation_requested);
    cancel.addEventListener('click', async () => { cancel.disabled = true; cancel.textContent = '正在取消'; try { await post(`/training-runs/${run.id}/cancel`, {}); await load(); } catch (error) { showError(error.message); cancel.disabled = false; cancel.textContent = '取消训练'; } });
    actions.append(cancel);
    elements.detail.append(actions);
  }

  function render() { renderList(); renderDetail(); }
  function showError(message) { elements.error.textContent = message; elements.error.hidden = false; }

  function schedule() {
    clearTimeout(timer); timer = null;
    if (!document.hidden && runs.some((run) => run.execution?.active)) timer = setTimeout(load, 5000);
  }

  async function load() {
    if (loading) return;
    loading = true;
    try {
      const next = await request('/training-runs');
      runs = next;
      if (!selectedId && runs.length) selectedId = runs[runs.length - 1].id;
      elements.error.hidden = true;
      render();
    } catch (error) {
      showError(error.message);
    } finally { loading = false; schedule(); }
  }

  document.addEventListener('visibilitychange', () => { if (document.hidden) { clearTimeout(timer); timer = null; } else load(); });
  window.addEventListener('pagehide', () => clearTimeout(timer));
  document.addEventListener('DOMContentLoaded', load);
})();
