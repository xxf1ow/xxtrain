(() => {
  'use strict';

  const apiRoot = '/platform/api';
  const elements = {
    loginPanel: document.getElementById('login-panel'),
    loginForm: document.getElementById('login-form'),
    loginButton: document.getElementById('login-button'),
    loginError: document.getElementById('login-error'),
    username: document.getElementById('username'),
    password: document.getElementById('password'),
    workspacePanel: document.getElementById('workspace-panel'),
    workspaceName: document.getElementById('workspace-name'),
    taskName: document.getElementById('task-name'),
    imageCount: document.getElementById('image-count'),
    annotatedImageCount: document.getElementById('annotated-image-count'),
    uploadForm: document.getElementById('upload-form'),
    imageFiles: document.getElementById('image-files'),
    uploadButton: document.getElementById('upload-button'),
    cacheAction: document.getElementById('cache-action'),
    workspaceMessage: document.getElementById('workspace-message'),
    workspaceError: document.getElementById('workspace-error'),
    primaryAction: document.getElementById('primary-action'),
    sessionTools: document.getElementById('session-tools'),
    sessionUser: document.getElementById('session-user'),
    logoutButton: document.getElementById('logout-button'),
  };

  let workspace = null;
  let busy = false;

  function cookie(name) {
    const prefix = `${name}=`;
    const value = document.cookie.split(';').map((part) => part.trim()).find((part) => part.startsWith(prefix));
    return value ? decodeURIComponent(value.slice(prefix.length)) : '';
  }

  async function request(path, options = {}) {
    const response = await fetch(`${apiRoot}${path}`, {credentials: 'same-origin', ...options});
    if (!response.ok) {
      let detail = '操作未完成，请重试。';
      try {
        const payload = await response.json();
        if (typeof payload.detail === 'string') detail = payload.detail;
      } catch (_) {
        // The public fallback is intentionally independent of upstream response content.
      }
      const error = new Error(detail);
      error.status = response.status;
      throw error;
    }
    return response.status === 204 ? null : response.json();
  }

  function platformPost(path, payload) {
    return request(path, {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-XTrain-CSRF': cookie('xxtrain_csrf')},
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

  function setBusy(value, message = '') {
    busy = value;
    const blocked = value || returnedAnnotationsPending();
    elements.primaryAction.disabled = blocked || !workspace || workspace.image_count === 0;
    elements.imageFiles.disabled = blocked;
    elements.uploadButton.disabled = blocked || elements.imageFiles.files.length === 0;
    elements.cacheAction.disabled = blocked || !workspace?.can_generate_detection_cache || workspace.detection_cache_ready;
    elements.loginButton.disabled = value;
    elements.logoutButton.disabled = value;
    elements.workspaceMessage.textContent = message;
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
      const target = definitions.get(card.dataset.target);
      const available = Boolean(target && target.available);
      card.classList.toggle('target-active', available);
      card.setAttribute('aria-disabled', available ? 'false' : 'true');
      const label = card.querySelector('.availability');
      if (label) label.textContent = available ? '可用' : '未开放';
    });
  }

  function renderWorkspace(next) {
    workspace = next;
    elements.loginPanel.hidden = true;
    elements.workspacePanel.hidden = false;
    elements.sessionTools.hidden = false;
    elements.workspaceName.textContent = next.name;
    elements.taskName.textContent = next.task.name;
    elements.imageCount.textContent = String(next.image_count);
    elements.annotatedImageCount.textContent = String(next.annotated_image_count);
    elements.cacheAction.textContent = next.detection_cache_ready ? '训练缓存已生成' : '生成训练缓存';
    clearError(elements.workspaceError);
    setBusy(busy);
    renderTargets(next.targets);
  }

  async function syncAnnotations() {
    clearError(elements.workspaceError);
    setBusy(true, '正在保存到平台…');
    try {
      renderWorkspace(await platformPost('/detection/sync', {}));
      history.replaceState({}, '', '/platform/');
    } catch (error) {
      showError(elements.workspaceError, `${error.message} 刷新页面可重新同步标注。`);
    } finally {
      setBusy(false);
    }
  }

  async function loadWorkspace() {
    const session = await request('/session');
    elements.sessionUser.textContent = `用户 ${session.user_id}`;
    renderWorkspace(await request('/workspace'));
    if (returnedAnnotationsPending()) await syncAnnotations();
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
    } catch (error) {
      showLogin(error.message);
    } finally {
      setBusy(false);
    }
  });

  elements.primaryAction.addEventListener('click', async () => {
    if (busy || returnedAnnotationsPending() || !workspace || workspace.image_count === 0) return;
    clearError(elements.workspaceError);
    setBusy(true, '正在准备标注任务…');
    try {
      const result = await platformPost('/detection/start', {});
      location.assign(result.annotation_url);
    } catch (error) {
      showError(elements.workspaceError, error.message);
      setBusy(false);
    }
  });

  elements.imageFiles.addEventListener('change', () => setBusy(busy));

  elements.uploadForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (busy || returnedAnnotationsPending() || elements.imageFiles.files.length === 0) return;
    const files = new FormData();
    for (const file of elements.imageFiles.files) files.append('images', file);
    clearError(elements.workspaceError);
    setBusy(true, '正在上传图片…');
    try {
      renderWorkspace(await request('/images', {
        method: 'POST',
        headers: {'X-XTrain-CSRF': cookie('xxtrain_csrf')},
        body: files,
      }));
      elements.uploadForm.reset();
    } catch (error) {
      showError(elements.workspaceError, error.message);
    } finally {
      setBusy(false);
    }
  });

  elements.cacheAction.addEventListener('click', async () => {
    if (busy || returnedAnnotationsPending() || !workspace?.can_generate_detection_cache || workspace.detection_cache_ready) return;
    clearError(elements.workspaceError);
    setBusy(true, '正在生成训练缓存…');
    try {
      renderWorkspace(await platformPost('/detection/cache', {}));
    } catch (error) {
      showError(elements.workspaceError, error.message);
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
      showError(elements.workspaceError, error.message);
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
