# Server deployment

This procedure installs the server in one Linux checkout at `/home/lxx/xxtest/xxtrain`. It covers the platform, CVAT, ClearML Server, and their private-LAN proxy; it does not install a ClearML Agent or validate GPU training.

## Prerequisites

Provide a checkout at `/home/lxx/xxtest/xxtrain`, a locally assigned private-LAN IPv4 address, uv, Docker Engine with Compose, working repository/dependency/image access and any required repository authentication. Run as the checkout's ordinary user with permission to use Docker and sudo for unit registration and directory ownership. The deployment does not install tools, repair network access, or change sudo policy. If sudo needs authentication, run `sudo -v` in the same interactive session before each privileged `serverctl` action.

Select an audited revision from `origin`, not from a local branch or directory timestamp. On upgrades, stop the service, make a consistent backup of `.deployment/` including configuration, credentials, databases, and bound storage, and confirm a clean tracked worktree before switching commits. Keep the backup outside this checkout. A Git checkout change does not restore a database, and this procedure makes no downgrade or database-backward-compatibility promise.

```sh
cd /home/lxx/xxtest/xxtrain
git fetch origin
git rev-parse origin/<approved-branch>
git status --porcelain=v1
```

Review the full SHA printed by `git rev-parse` and use that exact value below. A first install has no service to stop; for an upgrade, stop it and back up `.deployment/` before checking out the revision:

```sh
uv run --locked --extra platform --extra clearml xxtrain serverctl stop
git status --porcelain=v1
git switch --detach <approved-full-SHA>
git rev-parse HEAD
```

Do not switch with tracked local changes. The checkout's ignored `.deployment/` persists across revisions; do not remove it during upgrades.

## Install and start

On the first install, provide the host's private-LAN address once. `install` saves it in `.deployment/listen-ip`, synchronizes locked dependencies, pulls and builds images, creates private configuration and credentials, and registers the systemd unit. It does not start services. Subsequent installs use the saved address and preserve data and credentials.

```sh
XXTRAIN_SERVER_LISTEN_IP=192.168.0.109 uv run --locked --extra platform --extra clearml xxtrain serverctl install
uv run --locked --extra platform --extra clearml xxtrain serverctl status
```

Replace the example IP with the actual local private-LAN address. Before `start`, edit `.deployment/.xxxxx` if a chosen `xxadmin` password is required; keep it nonempty, readable only by the owner (`chmod 600 .deployment/.xxxxx`), and include it in the private backup. Otherwise the installer-generated random password is used. The next start synchronizes this file with CVAT's `xxadmin` account. No interactive account, token, or configuration creation is required.

```sh
uv run --locked --extra platform --extra clearml xxtrain serverctl start
uv run --locked --extra platform --extra clearml xxtrain serverctl status
uv run --locked --extra platform --extra clearml xxtrain serverctl verify
```

`start` enables and restarts the systemd unit, waits for CVAT migrations and authenticated ClearML, synchronizes credentials, and starts the platform. `status` prints the exact Git commit, unit state, and Compose service states. `verify` checks the actual platform, CVAT identity, ClearML Web, API authentication, and file entries. Check `sudo systemctl status xxtrain-server.service` and `sudo journalctl -u xxtrain-server.service -n 100` if startup fails; correct prerequisites before retrying. Repeating `install` and `start` should preserve the existing `.deployment/` state.

## Inspect and stop

The platform is internal on `127.0.0.1:18001`; CVAT and ClearML backends bind loopback. The proxy alone serves the selected private-LAN address: platform and CVAT at `http://<LAN-IP>:8080/`, ClearML API at `:8008`, files at `:8081`, and Web at `:8082`. Do not expose these listeners directly to the public Internet. Check `ss -ltn` on the host and log into `/platform/` as `xxadmin` using `.deployment/.xxxxx`; the same CVAT browser session opens CVAT, where CVAT's own permissions restrict tasks. ClearML's three entries do not require a platform session, but its API service key is private.

```sh
uv run --locked --extra platform --extra clearml xxtrain serverctl stop
uv run --locked --extra platform --extra clearml xxtrain serverctl status
```

`stop` stops and disables the unit and its project containers without deleting `.deployment/`, images, or volumes. A stopped and disabled unit does not auto-start after boot; a started and enabled unit follows the same readiness path on unit restart or boot. Full-host boot behavior requires separate host-reboot verification.
