# syntax=docker/dockerfile:1
FROM python:3.12-slim

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cached layer) using the lockfile.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra web

# Now the source.
COPY . .
RUN uv sync --frozen --extra web

EXPOSE 8000

# Inside a container, stdio transport cannot reach a host Ghidra instance —
# set GHIDRA_MCP_TRANSPORT=http and GHIDRA_MCP_URL to a reachable MCP server.
CMD ["uv", "run", "ghidra-deep-agent-web"]
