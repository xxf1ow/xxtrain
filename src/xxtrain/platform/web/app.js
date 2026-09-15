(() => {
  'use strict';

  const apiRoot = '/platform/api';
  const statusNames = {
    pending: '待标注',
    preparing: '正在准备标注任务',
    annotating: '标注中',
    sync_failed: '平台保存失败',
    saved: '已保存',
  };

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
    statusValue: document.getElementById('status-value'),
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

  function setBusy(value, message = '') {
    busy = value;
    elements.primaryAction.disabled = value;
    elements.loginButton.disabled = value;
    elements.logoutButton.disabled = value;
    if (message) elements.statusValue.textContent = message;
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
    elements.workspacePanel.dataset.status = next.status;
    elements.workspaceName.textContent = next.name;
    elements.taskName.textContent = next.task.name;
    elements.imageCount.textContent = String(next.image_count);
    elements.statusValue.textContent = statusNames[next.status] || '状态未知';
    elements.workspaceMessage.textContent = next.status === 'saved'
      ? '现场标注已写入平台文件。检测质量仍由标注人员确认。'
      : '检测、分类和分割共享这些图片与完整标注；当前只开放检测。';
    if (next.error) showError(elements.workspaceError, next.error);
    else clearError(elements.workspaceError);
    elements.primaryAction.textContent = next.status === 'sync_failed' ? '重试保存' : (next.status === 'pending' ? '开始标注' : '继续标注');
    elements.primaryAction.dataset.action = next.status === 'sync_failed' ? 'sync' : 'start';
    renderTargets(next.targets);
  }

  async function syncAnnotations() {
    clearError(elements.workspaceError);
    setBusy(true, '正在保存到平台…');
    try {
      renderWorkspace(await platformPost('/annotation/sync', {}));
      history.replaceState({}, '', '/platform/');
    } catch (error) {
      if (workspace) {
        workspace = {...workspace, status: 'sync_failed'};
        renderWorkspace(workspace);
      }
      showError(elements.workspaceError, error.message);
    } finally {
      setBusy(false);
    }
  }

  async function loadWorkspace() {
    const session = await request('/session');
    elements.sessionUser.textContent = `用户 ${session.user_id}`;
    renderWorkspace(await request('/workspace'));
    if (new URLSearchParams(location.search).get('returned') === '1') await syncAnnotations();
  }

  elements.loginForm.addEventListener('submit', async (event) => {
    event.preventDefault();
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
    if (busy) return;
    if (elements.primaryAction.dataset.action === 'sync') {
      await syncAnnotations();
      return;
    }
    clearError(elements.workspaceError);
    setBusy(true, '正在准备标注任务…');
    try {
      const result = await platformPost('/annotation/start', {});
      location.assign(result.annotation_url);
    } catch (error) {
      showError(elements.workspaceError, error.message);
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
    try {
      await loadWorkspace();
    } catch (error) {
      if (error.status === 401) showLogin();
      else showLogin(error.message);
    }
  });
})();
