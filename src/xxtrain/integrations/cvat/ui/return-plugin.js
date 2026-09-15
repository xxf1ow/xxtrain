(() => {
  'use strict';

  function register() {
    window.cvatUI.registerComponent(({store, dispatch, actionCreators, core}) => {
      let busy = false;
      const panel = document.createElement('aside');
      panel.style.cssText = 'position:relative;display:none;align-items:center;margin:0 3px';
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = '完成';
      button.title = '保存标注并返回平台';
      button.style.cssText = 'background:#007d8a;color:white;border:0;border-radius:3px;width:76px;height:42px;font:600 15px system-ui;cursor:pointer';
      const errorText = document.createElement('p');
      errorText.dataset.testid = 'xxtrain-return-error';
      errorText.setAttribute('role', 'alert');
      errorText.style.cssText = 'position:absolute;top:48px;right:0;z-index:1100;width:320px;padding:12px;background:white;border:1px solid #a23a36;color:#762722;white-space:pre-wrap;margin:0';
      errorText.hidden = true;
      panel.append(button, errorText);
      document.body.append(panel);

      function currentJob() {
        const match = location.pathname.match(/^\/tasks\/(\d+)\/jobs\/(\d+)\/?$/);
        const job = store.getState().annotation.job.instance;
        return match && job && job.id === Number(match[2]) && job.taskId === Number(match[1]) ? job : null;
      }

      function update() {
        const menu = document.querySelector('.cvat-annotation-header-menu-button');
        if (menu && panel.parentElement !== menu.parentElement) menu.before(panel);
        panel.style.display = currentJob() && menu ? 'flex' : 'none';
        button.disabled = busy || store.getState().annotation.annotations.saving.uploading;
      }

      button.onclick = async () => {
        const job = currentJob();
        if (!job || busy) return;
        errorText.hidden = true;
        const control = store.getState().annotation.canvas.activeControl;
        if (!['cursor', 'drag_canvas', 'zoom_canvas'].includes(control)) {
          errorText.textContent = '请先完成当前绘制或编辑，再保存返回。';
          errorText.hidden = false;
          return;
        }
        busy = true;
        button.textContent = '正在保存…';
        update();
        try {
          await job.frames.save();
          await job.annotations.save();
          await dispatch(actionCreators.updateJobAsync(job, {state: core.enums.JobState.COMPLETED}));
          const verified = await fetch(`/api/jobs/${job.id}`, {credentials: 'same-origin'});
          if (!verified.ok || (await verified.json()).state !== 'completed') throw new Error('completion not verified');
          location.assign('/platform/?returned=1');
        } catch (_) {
          errorText.textContent = 'CVAT 保存未完成，请留在本页重试。';
          errorText.hidden = false;
          busy = false;
          button.textContent = '完成';
          update();
        }
      };

      // CVAT can mount or replace the toolbar after a store notification or workspace change.
      const observer = new MutationObserver(update);
      observer.observe(document.getElementById('root'), {childList: true, subtree: true});
      update();
      return {
        name: 'xxtrain-return',
        destructor: () => {
          observer.disconnect();
          panel.remove();
        },
        globalStateDidUpdate: update,
      };
    });
  }

  if (window.cvatUI) register();
  else window.addEventListener('plugins.ready', register, {once: true});
})();
