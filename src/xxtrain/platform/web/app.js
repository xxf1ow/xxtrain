(() => {
  'use strict';

  const apiRoot = '/platform/api';
  const targetIds = ['detect', 'classify', 'segment'];
  const returnTargetKey = 'xxtrain-return-target';
  const trainingStatuses = {queued: '排队中', running: '训练中', completed: '训练完成', failed: '训练失败', cancelled: '已取消', unknown: '状态查询失败'};
  const elements = {
    loginPanel: document.getElementById('login-panel'), loginForm: document.getElementById('login-form'),
    loginButton: document.getElementById('login-button'), loginError: document.getElementById('login-error'),
    username: document.getElementById('username'), password: document.getElementById('password'),
    workspacePanel: document.getElementById('workspace-panel'), workspaceName: document.getElementById('workspace-name'),
    taskName: document.getElementById('task-name'), imageCount: document.getElementById('image-count'),
    imageFiles: document.getElementById('image-files'), workspaceMessage: document.getElementById('workspace-message'),
    uploadError: document.getElementById('upload-error'), sessionTools: document.getElementById('session-tools'),
    sessionUser: document.getElementById('session-user'), logoutButton: document.getElementById('logout-button'),
  };
  const targetElements = {
    detect: {
      annotated: document.getElementById('annotated-image-count'), total: document.getElementById('detect-image-total'),
      annotate: document.getElementById('primary-action'), cache: document.getElementById('cache-action'),
      progress: document.getElementById('detect-progress'), error: document.getElementById('workspace-error'),
      correction: document.getElementById('detect-correction'),
    },
    classify: {
      annotated: document.getElementById('classify-annotated-count'), total: document.getElementById('classify-image-total'),
      annotate: document.getElementById('classify-annotate-action'), cache: document.getElementById('classify-cache-action'),
      progress: document.getElementById('classify-progress'), error: document.getElementById('classify-error'),
      correction: document.getElementById('classify-correction'),
    },
    segment: {
      annotated: document.getElementById('segment-annotated-count'), total: document.getElementById('segment-image-total'),
      annotate: document.getElementById('segment-annotate-action'), cache: document.getElementById('segment-cache-action'),
      progress: document.getElementById('segment-progress'), error: document.getElementById('segment-error'),
      correction: document.getElementById('segment-correction'),
    },
  };

  let workspace = null;
  let busy = false;
  let busyTarget = null;
  let notificationTimer;

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

  function storedReturnTarget() {
    const value = sessionStorage.getItem(returnTargetKey);
    return targetIds.includes(value) ? value : 'detect';
  }

  function targetView(target) {
    return workspace?.targets?.find((item) => item.id === target) || null;
  }

  function setBusy(value, message = '', area = null) {
    busy = value;
    busyTarget = value ? area : null;
    const blocked = value || returnedAnnotationsPending();
    const editingLocked = Boolean(workspace?.editing_locked);
    elements.imageFiles.disabled = blocked || editingLocked;
    elements.loginButton.disabled = value;
    elements.logoutButton.disabled = value;
    for (const target of targetIds) {
      const facts = targetView(target);
      const row = targetElements[target];
      const run = workspace?.training?.[target];
      row.annotate.disabled = blocked || editingLocked || !facts?.can_annotate;
      row.cache.disabled = blocked || (!run && (!facts?.can_generate_cache || (!workspace?.training_enabled && facts?.cache_ready)));
      row.progress.textContent = value && target === area ? message : '';
      row.progress.hidden = !row.progress.textContent;
      row.annotate.textContent = value && target === area && message === '正在准备标注任务…' ? '正在准备…' : '开始标注';
    }
    const uploadProgress = document.getElementById('upload-progress');
    uploadProgress.textContent = value && area === 'upload' ? message : '';
    uploadProgress.hidden = !uploadProgress.textContent;
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
    const definitions = new Map(targets.map((target) => [target.id, target]));
    document.querySelectorAll('[data-target]').forEach((card) => {
      const facts = definitions.get(card.dataset.target);
      card.classList.toggle('target-active', Boolean(facts?.available));
      card.setAttribute('aria-disabled', facts?.can_annotate ? 'false' : 'true');
      const availability = card.querySelector('.availability');
      if (availability) availability.textContent = facts?.available ? '可用' : '未开放';
    });
    for (const target of targetIds) {
      const facts = definitions.get(target);
      const row = targetElements[target];
      row.annotated.textContent = String(facts?.annotated_sample_count ?? 0);
      row.total.textContent = String(facts?.sample_count ?? 0);
      const run = workspace?.training?.[target];
      row.cache.textContent = run ? (trainingStatuses[run.execution?.status] || '查看训练任务')
        : facts?.cache_ready && !workspace?.training_enabled ? '训练缓存已生成' : '开始训练';
      row.cache.dataset.runId = run?.id || '';
    }
  }

  function trainingReturnTarget() {
    const runId = new URLSearchParams(location.search).get('return_run');
    return runId && /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(runId)
      ? `/platform/training/?run=${runId}` : null;
  }

  async function trainTarget(target) {
    const run = workspace?.training?.[target];
    if (run) {
      location.assign(`/platform/training/?run=${run.id}`);
      return;
    }
    const facts = targetView(target);
    if (busy || !workspace?.training_enabled || !facts?.can_generate_cache) return;
    const row = targetElements[target];
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
      const next = await requestApiWorkspace();
      renderWorkspace(next);
      if (submissionError) showError(row.error, submissionError.message);
      else if (next.training?.[target]?.id !== submitted.run_id) showError(row.error, '训练任务状态尚未确认，请重试。');
      else if (submitted.run.execution?.status === 'queued') notify('已加入训练队列');
    } catch (error) {
      workspace = null;
      showError(row.error, error.message);
    } finally {
      setBusy(false);
    }
  }

  function requestApiWorkspace() {
    return request('/workspace');
  }

  function renderWorkspace(next) {
    workspace = next;
    elements.loginPanel.hidden = true;
    elements.workspacePanel.hidden = false;
    elements.sessionTools.hidden = false;
    elements.workspaceName.textContent = next.name;
    elements.taskName.textContent = next.task.name;
    elements.imageCount.textContent = String(next.image_count);
    for (const target of targetIds) {
      clearError(targetElements[target].error);
      targetElements[target].correction.hidden = true;
      targetElements[target].correction.removeAttribute('href');
    }
    renderTargets(next.targets);
    setBusy(busy, '', busyTarget);
  }

  async function syncAnnotations(target) {
    const row = targetElements[target];
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
    if (returnedAnnotationsPending()) await syncAnnotations(storedReturnTarget());
  }

  async function beginTarget(target) {
    if (busy || returnedAnnotationsPending() || !targetView(target)?.can_annotate) return;
    const row = targetElements[target];
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
    const row = targetElements[target];
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

  for (const target of targetIds) {
    targetElements[target].annotate.addEventListener('click', () => beginTarget(target));
    targetElements[target].cache.addEventListener('click', () => workspace?.training_enabled ? trainTarget(target) : generateCache(target));
    const showTaskLabel = () => {
      if (targetElements[target].cache.dataset.runId) targetElements[target].cache.textContent = '查看训练任务';
    };
    const restoreTaskLabel = () => {
      const run = workspace?.training?.[target];
      if (run) targetElements[target].cache.textContent = trainingStatuses[run.execution?.status] || '查看训练任务';
    };
    targetElements[target].cache.addEventListener('mouseenter', showTaskLabel);
    targetElements[target].cache.addEventListener('mouseleave', restoreTaskLabel);
    targetElements[target].cache.addEventListener('focus', showTaskLabel);
    targetElements[target].cache.addEventListener('blur', restoreTaskLabel);
  }

  elements.imageFiles.addEventListener('change', async () => {
    if (busy || returnedAnnotationsPending() || elements.imageFiles.files.length === 0) return;
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
      showError(targetElements.detect.error, error.message);
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
