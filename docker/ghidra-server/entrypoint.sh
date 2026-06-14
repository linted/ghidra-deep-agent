#!/usr/bin/env bash
# Ghidra Server entrypoint: ensure the service account exists, then run the
# server in the foreground (console mode) as PID 1.
set -euo pipefail

GHIDRA_HOME="${GHIDRA_HOME:-/opt/ghidra}"
SVR_DIR="${GHIDRA_HOME}/server"

USER_SID="${GHIDRA_SERVER_USER:-agent}"
PASSWORD="${GHIDRA_SERVER_PASSWORD:-}"

# Add the service account on first boot. svrAdmin operates on the user db while
# the server is stopped; the /repositories volume persists it across restarts,
# so we only add when the user is missing. With --p and no controlling tty,
# ServerAdmin reads the new password from stdin (System.console() is null) — so
# piping it sets the password directly and avoids the default "changeme"
# forced-change flow. Combined with the server's -e0 (no expiration), the
# service account can log in non-interactively.
if [ -n "${USER_SID}" ]; then
  if "${SVR_DIR}/svrAdmin" -users 2>/dev/null | grep -qwF "${USER_SID}"; then
    echo "Ghidra Server: user '${USER_SID}' already exists, skipping add."
  elif [ -n "${PASSWORD}" ]; then
    echo "Ghidra Server: adding user '${USER_SID}'."
    printf '%s\n%s\n' "${PASSWORD}" "${PASSWORD}" \
      | "${SVR_DIR}/svrAdmin" -add "${USER_SID}" --p \
      || echo "Ghidra Server: WARNING — adding user '${USER_SID}' failed (it may already exist)."
  else
    echo "Ghidra Server: GHIDRA_SERVER_PASSWORD is unset; adding '${USER_SID}' with default password 'changeme'." \
         "Set GHIDRA_SERVER_PASSWORD to provision a real password."
    "${SVR_DIR}/svrAdmin" -add "${USER_SID}" \
      || echo "Ghidra Server: WARNING — adding user '${USER_SID}' failed (it may already exist)."
  fi
fi

echo "Ghidra Server: starting in console mode..."
exec "${SVR_DIR}/ghidraSvr" console
