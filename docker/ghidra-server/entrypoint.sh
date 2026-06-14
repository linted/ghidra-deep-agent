#!/usr/bin/env bash
# Ghidra Server entrypoint: run the server (console mode) and ensure the service
# account exists.
#
# Ordering matters on a fresh volume: svrAdmin can only manage users once the
# server has initialized the repositories directory. On an empty /repositories
# the directory is not yet a valid Ghidra server directory, so a pre-start
# `svrAdmin -add` fails with "Invalid Ghidra server directory!" and the server
# comes up with zero users. So we start the server first, wait for it to
# initialize the repo dir, then add the user — the running server applies user
# commands asynchronously through its command queue (the "Command watcher").
set -uo pipefail

GHIDRA_HOME="${GHIDRA_HOME:-/opt/ghidra}"
SVR_DIR="${GHIDRA_HOME}/server"
REPO_DIR="${GHIDRA_REPOSITORIES_DIR:-/repositories}"

USER_SID="${GHIDRA_SERVER_USER:-agent}"
PASSWORD="${GHIDRA_SERVER_PASSWORD:-}"
DEFAULT_REPO="${GHIDRA_DEFAULT_REPOSITORY:-}"

echo "Ghidra Server: starting in console mode..."
"${SVR_DIR}/ghidraSvr" console &
server_pid=$!

# Binary import service (upload -> analyzeHeadless -> shared repo), co-located
# with the server. Stdlib-only; reachable by the web container on the docker
# network (8082), never published to the host. It connects to the server's RMI
# at import time, so it can start now regardless of provisioning progress.
#
# Supervise it in a restart loop rather than crash-coupling it to the server:
# this is the single persistent component every agent session depends on, so an
# importer hiccup must not bounce the server (unlike the per-session ghidra-mcp,
# where the engine and bridge are intentionally coupled).
echo "Ghidra Server: starting importer on :${IMPORTER_PORT:-8082}..."
supervise_importer() {
  while true; do
    python3 /opt/importer.py
    echo "Ghidra Server: importer exited (code $?); restarting in 2s..." >&2
    sleep 2
  done
}
supervise_importer &
importer_supervisor_pid=$!

# Forward termination cleanly: stop the supervisor (so it stops relaunching),
# kill any live importer, and shut down the server.
terminate() {
  kill "${importer_supervisor_pid}" 2>/dev/null || true
  pkill -f '/opt/importer.py' 2>/dev/null || true
  kill "${server_pid}" 2>/dev/null || true
}
trap terminate SIGTERM SIGINT

provision_user() {
  [ -n "${USER_SID}" ] || return 0

  # Wait for the server to initialize the repositories directory. It writes the
  # 'users' file on first start; until then svrAdmin cannot operate. Bounded so
  # we never hang forever, and bail out early if the server dies.
  local ready=""
  for _ in $(seq 1 60); do
    if [ -f "${REPO_DIR}/users" ]; then
      ready=1
      break
    fi
    kill -0 "${server_pid}" 2>/dev/null || {
      echo "Ghidra Server: exited before initializing ${REPO_DIR}; not provisioning user." >&2
      return 0
    }
    sleep 2
  done
  [ -n "${ready}" ] || {
    echo "Ghidra Server: timed out waiting for ${REPO_DIR} to initialize; not provisioning user." >&2
    return 0
  }

  # svrAdmin -users reads the user db directly; -add against the running server
  # is queued and applied by the command watcher (look for "User '<sid>' added").
  if "${SVR_DIR}/svrAdmin" -users 2>/dev/null | grep -qwF "${USER_SID}"; then
    echo "Ghidra Server: user '${USER_SID}' already exists, skipping add."
  elif [ -n "${PASSWORD}" ]; then
    echo "Ghidra Server: adding user '${USER_SID}'."
    printf '%s\n%s\n' "${PASSWORD}" "${PASSWORD}" \
      | "${SVR_DIR}/svrAdmin" -add "${USER_SID}" --p \
      || echo "Ghidra Server: WARNING — adding user '${USER_SID}' failed."
  else
    echo "Ghidra Server: GHIDRA_SERVER_PASSWORD is unset; adding '${USER_SID}' with default password 'changeme'." \
         "Set GHIDRA_SERVER_PASSWORD to provision a real password."
    "${SVR_DIR}/svrAdmin" -add "${USER_SID}" \
      || echo "Ghidra Server: WARNING — adding user '${USER_SID}' failed."
  fi

  seed_repository
}

# Seed a default shared repository via the Ghidra RMI client API (svrAdmin can
# only manage users, not create repositories — see CreateRepository.java). The
# helper authenticates as the service account, so it must run only after the
# user exists *and* the server's RMI ports are accepting logins; both can lag
# the -add command, so we wait for the user to materialize and then let the
# helper retry the connection. The helper is idempotent (no-op if repo exists).
seed_repository() {
  [ -n "${DEFAULT_REPO}" ] || return 0
  [ -f /opt/repo-tools/classpath ] || {
    echo "Ghidra Server: repo helper not built; skipping repository seed." >&2
    return 0
  }
  [ -n "${PASSWORD}" ] || {
    echo "Ghidra Server: GHIDRA_SERVER_PASSWORD unset; cannot seed repository '${DEFAULT_REPO}'." >&2
    return 0
  }

  # Wait for the service account to appear in the user db (the -add above is
  # applied asynchronously by the server's command watcher).
  for _ in $(seq 1 30); do
    "${SVR_DIR}/svrAdmin" -users 2>/dev/null | grep -qwF "${USER_SID}" && break
    kill -0 "${server_pid}" 2>/dev/null || return 0
    sleep 1
  done

  local cp
  cp="$(cat /opt/repo-tools/classpath)"
  echo "Ghidra Server: seeding default repository '${DEFAULT_REPO}'..."
  # The server runs with "Prompt for user ID: no", so it derives the login SID
  # from the client JVM's user.name and ignores the SID the authenticator sends.
  # This container runs as root, so without the override the helper logs in as
  # "root" and is denied. Force user.name to the service account (same fix the
  # ghidra-mcp engine applies for its own connection).
  for attempt in $(seq 1 10); do
    if java -Djava.awt.headless=true -Duser.name="${USER_SID}" \
         -cp "/opt/repo-tools:${cp}" CreateRepository; then
      return 0
    fi
    kill -0 "${server_pid}" 2>/dev/null || return 0
    echo "Ghidra Server: repository seed attempt ${attempt} did not succeed; retrying in 3s..."
    sleep 3
  done
  echo "Ghidra Server: WARNING — could not seed repository '${DEFAULT_REPO}' after retries." >&2
}

# Provision in the background so it can wait for init without blocking the server.
provision_user &

# The container's lifecycle tracks the server only: if the server dies the
# container exits (docker restarts it); the importer is supervised separately
# above and restarted in place without disrupting connected sessions.
wait "${server_pid}"
