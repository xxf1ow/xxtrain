(() => {
  'use strict';

  const apiRoot = '/platform/api';
  const returnTargetKey = 'xxtrain-return-target';
  const trainingStatuses = {pending: '等待确认', queued: '排队中', running: '训练中', completed: '训练完成', failed: '训练失败', cancelled: '已取消', unknown: '状态查询失败'};
  const elements = {
    loginPanel: document.getElementById('login-panel'), loginForm: document.getElementById('login-form'),
    loginButton: document.getElementById('login-button'), loginError: document.getElementById('login-error'),
    username: document.getElementById('username'), password: document.getElementById('password'),
    workspacePanel: document.getElementById('workspace-panel'), workspaceName: document.getElementById('workspace-name'),
    taskName: document.getElementById('task-name'), imageCount: document.getElementById('image-count'),
    imageFiles: document.getElementById('image-files'), workspaceMessage: document.getElementById('workspace-message'),
    uploadError: document.getElementById('upload-error'), uploadProgress: document.getElementById('upload-progress'),
    sessionTools: document.getElementById('session-tools'), sessionUser: document.getElementById('session-user'),
    logoutButton: document.getElementById('logout-button'), targetRail: document.getElementById('target-rail'),
  };
  const targetElements = new Map();

  let workspace = null;
  let busy = false;
  let busyTarget = null;
  let notificationTimer;

  function makeElement(tag, className = '', content = '') {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content) node.textContent = content;
    return node;
  }

  function notify(message) {
    clearTimeout(notificationTimer);
    elements.workspaceMessage.textContent = message;
    elements.workspaceMessage.hidden = false;
    notificationTimer = setTimeout(() => { elements.workspaceMessage.hidden = true; }, 8000);
  }

  function cookie(name) {
    const prefix = `${name}=`;
    const value = document.cookie.split(';').map((part) => part.trim()).find((part) => part.startsWith(prefix));
    return value ? decodeURIComponent(value.slice(prefix.length)) : '';
  }

  async function request(path, options = {}) {
    const response = await fetch(`${apiRoot}${path}`, {credentials: 'same-origin', ...options});
    if (!response.ok) {
      let detail = '操作未完成，请重试。';
      let annotationUrl = null;
      try {
        const payload = await response.json();
        if (typeof payload.detail === 'string') detail = payload.detail;
        if (typeof payload.annotation_url === 'string' && /^\/tasks\/\d+\/jobs\/\d+(\?(defaultWorkspace=TAGS(&frame=\d+)?|frame=\d+))?$/.test(payload.annotation_url)) {
          annotationUrl = payload.annotation_url;
        }
      } catch (_) {
        // The public fallback is independent of upstream response content.
      }
      const error = new Error(detail);
      error.status = response.status;
      error.annotationUrl = annotationUrl;
      throw error;
    }
    return response.status === 204 ? null : response.json();
  }

  function platformPost(path, payload) {
    return request(path, {
      method: 'POST', headers: {'Content-Type': 'application/json', 'X-XTrain-CSRF': cookie('xxtrain_csrf')},
      body: JSON.stringify(payload),
    });
  }

  function showError(element, message) {
    element.textContent = message;
    element.hidden = false;
  }

  function clearError(element) {
    element.textContent = '';
    element.hidden = true;
  }

  function returnedAnnotationsPending() {
    return new URLSearchParams(location.search).get('returned') === '1';
  }

  function targetView(target) {
    return workspace?.targets?.find((item) => item.id === target) || null;
  }

  function storedReturnTarget() {
    const value = sessionStorage.getItem(returnTargetKey);
    if (value && targetView(value)) return value;
    return workspace?.targets?.[0]?.id || null;
  }

  function trainingLabel(run) {
    if (run.execution?.active && run.cancellation_requested) return '取消请求已保存，等待停止';
    if (run.execution?.status === 'completed' && !run.execution.download_ready) return '训练已结束，产物不可用';
    return trainingStatuses[run.execution?.status] || '查看训练任务';
  }

  function createTargetRow(target, index) {
    const card = makeElement('article', 'target');
    card.dataset.target = target.id;
    const targetIndex = makeElement('div', 'target-index', String(index + 1).padStart(2, '0'));
    targetIndex.setAttribute('aria-hidden', 'true');
    const summary = makeElement('div', 'target-summary');
    summary.append(makeElement('h2', '', target.name));
    const counts = makeElement('p');
    counts.append(makeElement('span', '', '标注进度：'));
    const annotated = makeElement('strong');
    const total = makeElement('span');
    counts.append(annotated, makeElement('span', '', ' / '), total, makeElement('span', '', ` ${target.sample_unit}`));
    summary.append(counts);
    const actions = makeElement('div', 'target-actions');
    const annotate = makeElement('button', 'primary-button', '开始标注');
    annotate.type = 'button';
    const cache = makeElement('button', 'primary-button', '开始训练');
    cache.type = 'button';
    actions.append(annotate, cache);
    const help = makeElement('p', 'target-help', '请按当前任务规则完成标注；服务端会在条件满足后开放训练。');
    const progress = makeElement('p', 'operation-progress');
    progress.setAttribute('role', 'status');
    progress.setAttribute('aria-live', 'polite');
    progress.hidden = true;
    const error = makeElement('p', 'error-message operation-error');
    error.setAttribute('role', 'alert');
    error.hidden = true;
    const correction = makeElement('a', 'correction-link operation-error', '返回问题帧修正');
    correction.hidden = true;
    card.append(targetIndex, summary, actions, help, progress, error, correction);
    annotate.addEventListener('click', () => beginTarget(target.id));
    cache.addEventListener('click', () => workspace?.training_enabled ? trainTarget(target.id) : generateCache(target.id));
    const showTaskLabel = () => {
      if (cache.dataset.runId) cache.textContent = '查看训练任务';
    };
    const restoreTaskLabel = () => {
      const run = targetView(target.id)?.training;
      if (run) cache.textContent = trainingLabel(run);
    };
    cache.addEventListener('mouseenter', showTaskLabel);
    cache.addEventListener('mouseleave', restoreTaskLabel);
    cache.addEventListener('focus', showTaskLabel);
    cache.addEventListener('blur', restoreTaskLabel);
    targetElements.set(target.id, {card, annotated, total, annotate, cache, progress, error, correction});
    return card;
  }

  function ensureTargetRows(targets) {
    const current = [...targetElements.keys()];
    if (current.length === targets.length && current.every((id, index) => id === targets[index].id)) return;
    targetElements.clear();
    elements.targetRail.replaceChildren(...targets.map(createTargetRow));
    elements.targetRail.setAttribute('aria-label', `${workspace.task.name} 模型目标`);
  }

  function setBusy(value, message = '', area = null) {
    busy = value;
    busyTarget = value ? area : null;
    const blocked = value || returnedAnnotationsPending();
    elements.imageFiles.disabled = blocked || !workspace?.can_upload;
    elements.loginButton.disabled = value;
    elements.logoutButton.disabled = value;
    for (const facts of workspace?.targets || []) {
      const row = targetElements.get(facts.id);
      if (!row) continue;
      const run = facts.training;
      row.annotate.disabled = blocked || !facts.editable || !facts.can_annotate;
      row.cache.disabled = blocked || (!run && (!facts.can_generate_cache || (!workspace.training_enabled && facts.cache_ready)));
      row.annotate.setAttribute('aria-disabled', String(row.annotate.disabled));
      row.cache.setAttribute('aria-disabled', String(row.cache.disabled));
      row.progress.textContent = value && facts.id === area ? message : '';
      row.progress.hidden = !row.progress.textContent;
      row.annotate.textContent = value && facts.id === area && message === '正在准备标注任务…' ? '正在准备…' : '开始标注';
    }
    elements.uploadProgress.textContent = value && area === 'upload' ? message : '';
    elements.uploadProgress.hidden = !elements.uploadProgress.textContent;
    if (value) {
      clearTimeout(notificationTimer);
      elements.workspaceMessage.hidden = true;
    }
  }

  function showLogin(message = '') {
    workspace = null;
    elements.loginPanel.hidden = false;
    elements.workspacePanel.hidden = true;
    elements.sessionTools.hidden = true;
    if (message) showError(elements.loginError, message);
    else clearError(elements.loginError);
  }

  function renderTargets(targets) {
    ensureTargetRows(targets);
    for (const facts of targets) {
      const row = targetElements.get(facts.id);
      row.card.classList.toggle('target-active', facts.can_annotate);
      row.annotated.textContent = String(facts.annotated_sample_count);
      row.total.textContent = String(facts.sample_count);
      const run = facts.training;
      row.cache.textContent = run ? trainingLabel(run)
        : facts.cache_ready && !workspace.training_enabled ? '训练缓存已生成' : '开始训练';
      row.cache.dataset.runId = run?.id || '';
    }
  }

  function submissionFeedback(run) {
    const execution = run.execution;
    if (!execution) return '已有历史任务，未保存新的训练请求';
    if (execution.status === 'pending' || execution.status === 'unknown') return '提交已保存，等待确认';
    if (execution.status === 'queued') return '已加入训练队列';
    if (execution.status === 'running') return '训练中';
    if (execution.status === 'completed') {
      return execution.download_ready ? '训练已完成，可下载部署产物' : '训练已结束，部署产物不可用';
    }
    if (execution.status === 'failed') return '训练已结束：训练失败';
    return '训练已取消';
  }

  function trainingReturnTarget() {
    const runId = new URLSearchParams(location.search).get('return_run');
    return runId && /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(runId)
      ? `/platform/training/?run=${runId}` : null;
  }

  async function trainTarget(target) {
    const facts = targetView(target);
    const run = facts?.training;
    if (run) {
      location.assign(`/platform/training/?run=${run.id}`);
      return;
    }
    if (busy || !workspace?.training_enabled || !facts?.can_generate_cache) return;
    const row = targetElements.get(target);
    clearError(row.error);
    setBusy(true, '正在加入训练队列…', target);
    let submitted = null;
    let submissionError = null;
    try {
      submitted = await platformPost(`/targets/${target}/train`, {});
    } catch (error) {
      submissionError = error;
    }
    try {
      renderWorkspace(await request('/workspace'));
      const observed = targetView(target)?.training;
      if (submissionError) showError(row.error, submissionError.message);
      else if (observed?.id !== submitted.run_id) showError(row.error, '训练任务状态尚未确认，请重试。');
      else notify(submissionFeedback(submitted.run));
    } catch (error) {
      workspace = null;
      showError(row.error, error.message);
    } finally {
      setBusy(false);
    }
  }

  function renderWorkspace(next) {
    workspace = next;
    elements.loginPanel.hidden = true;
    elements.workspacePanel.hidden = false;
    elements.sessionTools.hidden = false;
    elements.workspaceName.textContent = next.name;
    elements.taskName.textContent = next.task.name;
    elements.imageCount.textContent = String(next.image_count);
    renderTargets(next.targets);
    for (const row of targetElements.values()) {
      clearError(row.error);
      row.correction.hidden = true;
      row.correction.removeAttribute('href');
    }
    setBusy(busy, '', busyTarget);
  }

  async function syncAnnotations(target) {
    const row = targetElements.get(target);
    if (!row) return;
    clearError(row.error);
    row.correction.hidden = true;
    setBusy(true, '正在保存到平台…', target);
    try {
      renderWorkspace(await platformPost(`/targets/${target}/sync`, {}));
      sessionStorage.removeItem(returnTargetKey);
      history.replaceState({}, '', '/platform/');
      notify('标注已保存到平台，图片数量已更新。');
    } catch (error) {
      showError(row.error, error.annotationUrl ? error.message : `${error.message} 刷新页面可重新同步标注。`);
      if (error.annotationUrl) {
        row.correction.href = error.annotationUrl;
        row.correction.hidden = false;
      }
    } finally {
      setBusy(false);
    }
  }

  async function loadWorkspace() {
    const session = await request('/session');
    elements.sessionUser.textContent = `用户 ${session.user_id}`;
    renderWorkspace(await request('/workspace'));
    const returnedTarget = storedReturnTarget();
    if (returnedAnnotationsPending() && returnedTarget) await syncAnnotations(returnedTarget);
  }

  async function beginTarget(target) {
    const facts = targetView(target);
    if (busy || returnedAnnotationsPending() || !facts?.editable || !facts.can_annotate) return;
    const row = targetElements.get(target);
    clearError(row.error);
    setBusy(true, '正在准备标注任务…', target);
    try {
      const result = await platformPost(`/targets/${target}/start`, {});
      sessionStorage.setItem(returnTargetKey, target);
      location.assign(result.annotation_url);
    } catch (error) {
      showError(row.error, error.message);
      setBusy(false);
    }
  }

  async function generateCache(target) {
    const facts = targetView(target);
    if (busy || returnedAnnotationsPending() || !facts?.can_generate_cache || facts.cache_ready) return;
    const row = targetElements.get(target);
    clearError(row.error);
    setBusy(true, '正在生成训练缓存…', target);
    try {
      renderWorkspace(await platformPost(`/targets/${target}/cache`, {}));
      notify('训练缓存已生成。');
    } catch (error) {
      showError(row.error, error.message);
    } finally {
      setBusy(false);
    }
  }

  elements.loginForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (busy) return;
    clearError(elements.loginError);
    setBusy(true);
    try {
      await platformPost('/login', {username: elements.username.value, password: elements.password.value});
      elements.loginForm.reset();
      await loadWorkspace();
      const returnTarget = trainingReturnTarget();
      if (returnTarget) location.assign(returnTarget);
    } catch (error) {
      showLogin(error.message);
    } finally {
      setBusy(false);
    }
  });

  elements.imageFiles.addEventListener('change', async () => {
    if (busy || returnedAnnotationsPending() || !workspace?.can_upload || elements.imageFiles.files.length === 0) return;
    const files = new FormData();
    for (const file of elements.imageFiles.files) files.append('images', file);
    const selectedCount = elements.imageFiles.files.length;
    clearError(elements.uploadError);
    setBusy(true, `正在上传并去重 ${selectedCount} 张图片…`, 'upload');
    try {
      renderWorkspace(await request('/images', {
        method: 'POST', headers: {'X-XTrain-CSRF': cookie('xxtrain_csrf')}, body: files,
      }));
      elements.imageFiles.value = '';
      notify(`本次选择 ${selectedCount} 张图片，上传及去重完成。现场现有 ${workspace.image_count} 张有效图片。`);
    } catch (error) {
      showError(elements.uploadError, error.message);
    } finally {
      setBusy(false);
    }
  });

  elements.logoutButton.addEventListener('click', async () => {
    if (busy) return;
    setBusy(true);
    try {
      await platformPost('/logout', {});
      showLogin();
    } catch (error) {
      const first = targetElements.values().next().value;
      showError(first?.error || elements.uploadError, error.message);
    } finally {
      setBusy(false);
    }
  });

  document.addEventListener('DOMContentLoaded', async () => {
    setBusy(true, '正在读取…');
    try {
      await loadWorkspace();
    } catch (error) {
      if (error.status === 401) showLogin();
      else showLogin(error.message);
    } finally {
      setBusy(false);
    }
  });
})();
