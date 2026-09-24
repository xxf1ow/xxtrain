# Deploy the xxtrain server

This tutorial takes a clean Linux host to a running xxtrain platform, CVAT and ClearML Server using one Git checkout. It does not install or validate a ClearML Agent or GPU training. Use a controlled LAN: ClearML's external API, files and web ports do not provide a public-network security boundary.

## Prerequisites and published revision

Provide Git, SSH access to `git@github.com:xxf1ow/xxtrain.git`, Docker with Compose v2, an existing `uv`, and a user allowed to run Docker and the required `sudo` commands. If sudo needs a password, run `sudo -v` in the same interactive shell before `install`, `start` or `stop`; sudo may ask again if its authorization expires. The deployment process does not modify sudo policy. Reserve ports 8080, 8008, 8081 and 8082, and allow access to the pinned images in [server versions](../../deploy/server/versions.md).

The checkout is `/home/lxx/xxtest/xxtrain`; its ignored `.deployment/` directory is the only site configuration and application-data directory. The systemd unit and Docker's own image/container storage are runtime exceptions. Develop and test locally, commit, obtain human review and approval, and publish before deploying: the host only consumes a revision already present on `origin`. Choose the reviewed remote branch or tag; do not assume a fixed branch is current.

```sh
git clone git@github.com:xxf1ow/xxtrain.git /home/lxx/xxtest/xxtrain
cd /home/lxx/xxtest/xxtrain
git fetch origin
git switch --detach origin/<reviewed-branch-or-tag>
git rev-parse HEAD
git status --porcelain
```

Record the full SHA. The status must be empty before switching revisions. Never use `git clean`, remove the checkout, edit or commit source on the host, or create another checkout to switch versions. For noninteractive SSH, include the existing uv installation directory in `PATH` because `install` itself calls uv. Keep uv's cache and managed Python under `.deployment/`; do not source shell startup files.

## Prepare and start

The zero-start path is two commands from the checkout. `install` synchronizes locked dependencies, prepares the six writable bind directories, pulls pinned external images, builds the custom CVAT UI, creates missing private configuration and credentials, and registers/reloads the systemd unit. It does not start services. `start` enables and starts that unit; its bootstrap creates or synchronizes CVAT's `xxadmin` account, service token and first workspace configuration before the platform becomes ready.

```sh
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl install
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl start
```

On success, `install` exits zero without starting the unit, and `start` exits zero only after the service bootstrap succeeds. Repeating either command preserves existing credentials and JSON. `install` generates `.deployment/training.json`; the first successful `start` generates `.deployment/workspace.json` using CVAT's actual user ID. The latter cannot be prepared earlier because that ID comes from the running CVAT service.

The administrator is always `xxadmin`. Its password is stored as readable plaintext in `.deployment/.xxxxx`, mode `0600`; anyone who can read that file can log in as the administrator. Protect the host account and include the file in a restricted, consistent `.deployment/` backup. `.deployment/platform.env` holds ClearML keys and the CVAT service token and must also remain private (`0600`). ClearML's `secure.conf` is derived from the authoritative ClearML pair. Do not copy either secret to shell arguments, logs, tickets or source control.

To choose the administrator password, create or edit `.deployment/.xxxxx` privately before `start`, preserving mode `0600`; every `start`, including one while the unit is already active, synchronizes CVAT to the file's current contents. This also converges a manual edit on the next start. Repetition does not rotate machine credentials. If `.xxxxx` alone is lost while CVAT data remains, `start` creates a replacement and synchronizes it; loss of `.deployment/` or its databases requires restoring the consistent backup and is not repaired by generating a new password.

Back up all of `.deployment/` consistently before changes that could affect databases; stop services while taking a filesystem copy. Preserve SQLite, CVAT and ClearML databases and workspace data. Do not delete the directory or treat generated caches as a backup of authoritative data.

## Observe and verify

```sh
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl status
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl verify
```

`status` reports the full checkout SHA, worktree state, unit enabled/active state, and each Compose service's state and health. `verify` reports PASS/FAIL and HTTP observations for the platform entry, anonymous CVAT denial, loopback CVAT health and the three published ClearML listeners. Either exits nonzero on failure. A passing `verify` does not prove browser login, two-user CVAT isolation, Agent connectivity or GPU training.

Open `http://<host>:8080/platform/` and sign in with a CVAT user. Confirm that each user can access only the CVAT objects their CVAT permissions allow. CVAT pages and APIs pass through the same-origin proxy; do not expose the backend or xxtrain listener directly. ClearML's API, files and web interfaces are available at `http://<host>:8008/`, `http://<host>:8081/` and `http://<host>:8082/`. These ports are not protected by the CVAT session check; restrict them to a controlled LAN and never map them directly to the public internet.

If startup fails, inspect the specific unit and recent journal output, then compare `status` and `verify` observations:

```sh
sudo systemctl status xxtrain-server.service
sudo journalctl -u xxtrain-server.service -n 100 --no-pager
```

Correct the reported prerequisite or configuration problem and retry `start`. The service does not report ready when migrations, CVAT identity synchronization or authenticated ClearML bootstrap fails. Keep `.deployment/` and the selected checkout intact while diagnosing; do not start an alternate temporary Compose project.

## Reboot and stop

The unit starts on boot after `start` enables it. After reconnecting or rebooting, check `status` and `verify`; `systemctl is-enabled xxtrain-server.service` and `systemctl is-active xxtrain-server.service` expose the corresponding systemd states. To stop the service and disable boot startup:

```sh
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl stop
```

`stop` attempts to disable the unit even if stopping it fails. It does not delete `.deployment/`; containers have no independent boot-restart policy. A post-stop `status` is expected to report inactive/disabled and exit nonzero. Run `start` to resume.

## Test a published branch or upgrade

Arrange downtime and a consistent `.deployment/` backup first. Switching Git revisions does not reverse SQLite, CVAT or ClearML schema migrations; this procedure does not promise downgrade support. Stop the service, verify the worktree has no changes, fetch and select the reviewed published revision, then synchronize dependencies and run the same installation and start commands above. Inspect `status` and `verify` before returning the site to users.

```sh
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv run --locked --extra platform --extra clearml xxtrain serverctl stop
git status --porcelain
git fetch origin
git switch --detach origin/<reviewed-branch-or-tag>
git rev-parse HEAD
PATH="/home/lxx/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.deployment/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.deployment/uv-python" uv sync --locked --extra platform --extra clearml
```

Run the two commands in [Prepare and start](#prepare-and-start) from the same checkout after switching. Keep the original `.deployment/` throughout; do not force-switch a dirty worktree or infer the intended revision from a local branch name. The server procedure ends at server verification. Agent setup and GPU acceptance require their own procedure and evidence.
