# Deploy the xxtrain server

This tutorial starts with a clean Linux host and ends with a running xxtrain platform, CVAT and ClearML Server. It does not install or test a ClearML Agent or prove GPU training. Use a controlled host or LAN: these HTTP listeners do not provide public-network TLS.

## Prerequisites and published revision

Install and enable Docker with Compose v2, and provide an existing `uv`, Git, SSH access to `git@github.com:xxf1ow/xxtrain.git`, and permission to register a systemd unit through `sudo`. The operator must be able to run Docker. Reserve ports 8080, 8008, 8081 and 8082; ensure the host can pull the pinned images listed in [server versions](../../deploy/server/versions.md). The target directory is `/home/lxx/xxtest/xxtrain`; `.deployment/` beneath it owns all application configuration and persistent data. Docker image/container storage and `/etc/systemd/system/xxtrain-server.service` are the runtime exceptions.

Develop and test changes locally, commit them, and obtain human review and explicit approval before publishing the branch. Do not edit or commit source on the host. Choose a published `origin` branch or tag and record its full revision; `origin/master` is only an example, not a claim that it contains the latest reviewed change:

```sh
git clone git@github.com:xxf1ow/xxtrain.git /home/lxx/xxtest/xxtrain
cd /home/lxx/xxtest/xxtrain
git fetch origin
git switch --detach origin/master
git rev-parse HEAD
git status --porcelain
```

Replace `origin/master` with the reviewed, published `origin/<branch>` when deploying a feature branch. The status output must be empty before switching versions. Never use `git clean`, delete the checkout, or create a second checkout to switch revisions. In noninteractive SSH commands, `uv` may be installed at `/home/lxx/.local/bin/uv` without that directory on `PATH`. Give the command and its child processes a `PATH` containing `/home/lxx/.local/bin`; `serverctl install` invokes `uv sync` itself, so calling the outer `uv` by absolute path alone is insufficient. Keep uv's cache and managed Python installations in `.deployment/`, and do not source or edit shell startup files:

## Prepare configuration

Run installation from this checkout. It synchronizes locked dependencies with both extras, pulls Compose images, builds the custom CVAT UI, creates a private empty `.deployment/platform.env` only if absent, and registers/reloads the systemd unit on Linux. It does not start services or generate either JSON file. Repeating it preserves existing configuration and persistent data; an existing env file readable by group/others causes an error until its permissions are corrected.

```sh
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl install
chmod 600 .deployment/platform.env
```

On a new installation, bootstrap credentials before starting the platform. The checked-in Compose project can temporarily start its CVAT and ClearML services without invoking systemd; nginx serves ClearML while its platform route remains unavailable. Run from the checkout after `install`:

```sh
XXTRAIN_SITE_ROOT="$PWD/.deployment" docker compose -p xxtrain-server -f deploy/server/compose.yaml up -d cvat_server clearml_webserver nginx
XXTRAIN_SITE_ROOT="$PWD/.deployment" docker compose -p xxtrain-server -f deploy/server/compose.yaml exec cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Enter a new CVAT superuser name, email and password at the prompt. Wait until CVAT responds on `127.0.0.1:18080`; do not publish this loopback backend. The [official CVAT installation instructions](https://docs.cvat.ai/docs/administration/community/basics/installation/) specify `createsuperuser`, and its [Auth API](https://docs.cvat.ai/docs/api_sdk/sdk/reference/apis/auth-api/) documents `POST /api/auth/login` returning a `Token` with a `key`. The platform uses `Authorization: Token <key>`; CVAT Personal Access Tokens use `Bearer` and do not satisfy that interface. This interactive command obtains the login token without putting the password or token in shell history, appends it to the private env file, and prints the user ID needed in `workspace.json`:

```sh
python3 - <<'PY'
import getpass
import json
from pathlib import Path
from urllib.request import Request, urlopen

username = input('CVAT superuser name: ')
password = getpass.getpass('CVAT password: ')
root = 'http://127.0.0.1:18080'
payload = json.dumps({'username': username, 'password': password}).encode()
request = Request(root + '/api/auth/login', payload, {'Content-Type': 'application/json', 'Accept': 'application/vnd.cvat+json'})
with urlopen(request) as response:
    token = json.load(response)['key']
request = Request(root + '/api/users/self', headers={'Authorization': 'Token ' + token, 'Accept': 'application/vnd.cvat+json'})
with urlopen(request) as response:
    print('CVAT owner_user_id:', json.load(response)['id'])
with Path('.deployment/platform.env').open('a', encoding='utf-8') as env:
    env.write('XXTRAIN_CVAT_SERVICE_TOKEN=' + token + '\n')
PY
```

For a dedicated service identity, provision it in CVAT and repeat the same login-token operation for that account; the owner user ID must still identify the intended workspace owner. Avoid duplicate env assignments on repetition. Open `http://<host>:8082/settings/workspace-configuration` in a browser, select **Settings → Workspace → Create new credentials**, and copy the access and secret keys. These are the [official ClearML setup steps](https://clear.ml/docs/latest/docs/clearml_sdk/clearml_sdk_setup). Add the credentials and all three SDK service addresses to `.deployment/platform.env` with a private editor; do not commit or paste credentials into shell history. The platform process uses the host-loopback Compose mappings below, not the public nginx ports. Omitting any endpoint is a startup error because the ClearML SDK otherwise falls back to ClearML Cloud.

```dotenv
CLEARML_API_ACCESS_KEY=<access-key>
CLEARML_API_SECRET_KEY=<secret-key>
CLEARML_API_HOST=http://127.0.0.1:18083
CLEARML_WEB_HOST=http://127.0.0.1:18084
CLEARML_FILES_HOST=http://127.0.0.1:18082
```

Keep this file at `0600`. The unit reads it at startup; Compose receives `XXTRAIN_SITE_ROOT` from the unit and `serverctl`, not from this file. After credentials and endpoint addresses exist, stop only this temporary Compose project; `serverctl start` then takes over the full set:

```sh
XXTRAIN_SITE_ROOT="$PWD/.deployment" docker compose -p xxtrain-server -f deploy/server/compose.yaml stop
```

Create `.deployment/workspace.json` with the following exact keys (optional `task_entry` defaults to the Point definition). Replace the positive `owner_user_id` with the real CVAT user ID, and choose an identity and display name for the site. Relative paths resolve from `.deployment/`; keep every path used for application data beneath that directory.

```json
{
  "workspace_id": "site",
  "display_name": "Site",
  "owner_user_id": 1,
  "workspace_dir": "workspace",
  "runtime_dir": "runtime",
  "cvat_internal_url": "http://127.0.0.1:18080"
}
```

Replace the example `owner_user_id` with the ID printed during bootstrap. Create `.deployment/training.json` with its exact six keys. The `project` and `queue` are ClearML names you administer; `worker_script` is the deployed worker file, not a data directory. Verify that file exists at the selected revision. `runtime/cache` must lie under `shared_root`; `metadata_dir` must remain outside the disposable workspace runtime. Change these example paths together if your site layout differs.

```json
{
  "project": "xxtrain",
  "queue": "default",
  "shared_root": "runtime",
  "metadata_dir": "metadata",
  "worker_script": "../src/xxtrain/integrations/clearml/worker.py",
  "run_root": "runs"
}
```

Check `worker_script` against the actual checkout before starting; do not assume this example file path is valid for every revision. The configuration loaders require non-empty strings, a positive integer user ID, and exactly these keys. `start` requires all three files and rejects configuration symlinks resolving outside the checkout, but does not require the targets to remain within `.deployment/`; operators must keep every data path there. Protect the JSON files and back up `.deployment/` securely, including `workspace/`, `runtime/`, `metadata/`, `runs/`, `cvat/`, `clearml/`, the administrator file and credentials. SQLite and backend database directories are authoritative; runtime caches are reconstructible, but do not discard them during an upgrade. Coordinate a consistent backup with services stopped.

## Start and verify

```sh
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl start
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl status
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl verify
```

`start` enables `xxtrain-server.service` for boot and starts it; subsequent starts do not replace credentials or data. On its first invocation it creates `.deployment/administrator` as a plaintext random key with mode `0600`, only when absent. If you need to supply your own key, create that file privately before the first start. Existing content and permissions remain unchanged on later starts; restrict readers, securely back it up and never commit it. There is no administrator web page promised by this deployment.

`status` prints the full checkout SHA, clean/dirty working-tree state, unit enabled/active status, configured services and each container's state/health. Inspect those observations after each operation; a missing, stopped or unhealthy container causes a nonzero status. `verify` prints PASS/FAIL and observed HTTP codes for the platform (`/platform/`), unauthenticated CVAT denial, the loopback CVAT backend, and ClearML API, files and web listeners. It exits nonzero if any probe fails; initial backend readiness may take time. For startup failures inspect `sudo systemctl status xxtrain-server.service` and `sudo journalctl -u xxtrain-server.service -n 100 --no-pager`, then rerun status and verify. A passing HTTP probe does not establish authenticated usability.

Open `http://<host>:8080/platform/`, log in using a CVAT user and confirm that the user can open only CVAT objects granted by CVAT itself. In a separate unauthenticated browser, CVAT content must redirect to platform login or reject API requests; never expose the CVAT backend or xxtrain port directly. The nginx proxy is the external entry for the same-origin platform and CVAT; only host-loopback binds serve the backends. ClearML listeners are `http://<host>:8008/` (API), `:8081/` (files) and `:8082/` (web). Control LAN access to all four external ports. Authenticated multi-user CVAT authorization needs explicit site acceptance; `verify` does not test it.

## Reconnect, reboot and stop

SSH disconnect does not stop the systemd unit. Reconnect in the same checkout and run `status` and `verify`; after a host reboot do the same and inspect `systemctl is-enabled xxtrain-server.service` and `systemctl is-active xxtrain-server.service`. `start` enables boot startup. To stop and disable it, including when already stopped:

```sh
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl stop
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl status
```

The post-stop status reports an inactive/disabled unit and is expected to exit nonzero; `stop` never removes `.deployment/`. Compose containers have no independent boot restart policy. To resume, run `start`, `status` and `verify` again.

## Test a branch, upgrade or downgrade

Arrange downtime and a consistent `.deployment/` backup. Check the target version's database schema/migration compatibility and its recovery path before a downgrade: switching Git commits does not undo SQLite, CVAT or ClearML database migrations. Stop services, check `git status --porcelain` is empty (ignored `.deployment/` remains), fetch, choose the reviewed published revision and record its exact SHA. Do not force-switch a dirty checkout or assume a local branch tracks the intended published commit.

```sh
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl stop
git status --porcelain
git fetch origin
git switch --detach origin/<reviewed-branch>
git rev-parse HEAD
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv sync --locked --extra platform --extra clearml
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl install
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl start
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl status
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl verify
```

For rollback, stop and select the previously recorded, already fetched full SHA instead of `origin/<reviewed-branch>`, then repeat sync, install, start, status and verify. Restore the matching consistent data backup when schema compatibility requires it; never treat Git rollback alone as database rollback. Keep all configuration and data in the original `.deployment/` throughout. The same procedure tests a published feature branch in this one checkout. Server verification is separate from later ClearML Agent and GPU acceptance.
