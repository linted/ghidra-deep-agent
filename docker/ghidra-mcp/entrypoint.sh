#!/usr/bin/env bash
# Combined entrypoint: launch the headless Ghidra REST engine (8089) and the
# Python MCP bridge (8081) in one container. If either process exits, tear the
# other down and exit so the orchestrator restarts the pair (no half-up state).
set -uo pipefail

GHIDRA_HOME="${GHIDRA_HOME:-/opt/ghidra}"
ENGINE_PORT="${GHIDRA_MCP_PORT:-8089}"
BRIDGE_PORT="${MCP_BRIDGE_PORT:-8081}"
BIND_ADDRESS="${GHIDRA_MCP_BIND_ADDRESS:-0.0.0.0}"
JAVA_OPTS="${JAVA_OPTS:--Xmx4g -XX:+UseG1GC}"

# Build the Ghidra runtime classpath (framework + features + processors) plus
# our headless jar. Unmatched globs stay literal and are filtered by the -f test.
CLASSPATH="/app/GhidraMCP.jar"
for category in Framework Features Processors; do
  for jar in "${GHIDRA_HOME}"/Ghidra/${category}/*/lib/*.jar; do
    [ -f "${jar}" ] && CLASSPATH="${CLASSPATH}:${jar}"
  done
done

# Optional service-account name to bypass Ghidra project ownership checks.
USER_OPT=""
[ -n "${GHIDRA_USER:-}" ] && USER_OPT="-Duser.name=${GHIDRA_USER}"

engine_pid=""
bridge_pid=""
terminate() {
  echo "Shutting down GhidraMCP container..."
  [ -n "${bridge_pid}" ] && kill "${bridge_pid}" 2>/dev/null || true
  [ -n "${engine_pid}" ] && kill "${engine_pid}" 2>/dev/null || true
}
trap terminate SIGTERM SIGINT

echo "Starting headless Ghidra engine on ${BIND_ADDRESS}:${ENGINE_PORT}..."
# Server connection (GHIDRA_SERVER_HOST/PORT/USER/PASSWORD) is read from the
# environment by GhidraServerManager at runtime when /server/connect is called.
# shellcheck disable=SC2086
java ${JAVA_OPTS} ${USER_OPT} \
  -Dghidra.home="${GHIDRA_HOME}" \
  -Dapplication.name=GhidraMCP \
  -classpath "${CLASSPATH}" \
  com.xebyte.headless.GhidraMCPHeadlessServer \
  --port "${ENGINE_PORT}" --bind "${BIND_ADDRESS}" "$@" &
engine_pid=$!

# Wait for the engine's REST API before starting the bridge (bounded retries).
echo "Waiting for the engine REST API on port ${ENGINE_PORT}..."
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${ENGINE_PORT}/check_connection" >/dev/null 2>&1; then
    echo "Engine is up."
    break
  fi
  if ! kill -0 "${engine_pid}" 2>/dev/null; then
    echo "Engine exited during startup; aborting." >&2
    exit 1
  fi
  sleep 2
done

echo "Starting MCP bridge on 0.0.0.0:${BRIDGE_PORT}..."
# Inside this container GHIDRA_MCP_URL is the *local engine's* REST endpoint —
# distinct from the deep-agent web service's GHIDRA_MCP_URL, which points at this
# bridge's /mcp. The bridge defaults to this value, but we set it explicitly.
GHIDRA_MCP_URL="http://127.0.0.1:${ENGINE_PORT}" \
  python3 /app/bridge_mcp_ghidra.py \
    --transport streamable-http \
    --mcp-host 0.0.0.0 \
    --mcp-port "${BRIDGE_PORT}" &
bridge_pid=$!

# Exit as soon as either background process exits, then clean up the other.
wait -n
exit_code=$?
echo "A process exited (code ${exit_code}); shutting down."
terminate
exit "${exit_code}"
