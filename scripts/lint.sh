#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# --check reports problems without rewriting files; this is the CI gate.
if [[ "${1:-}" == "--check" ]]; then
  echo "==> ruff format --check"
  uv run ruff format --check .

  echo "==> ruff check"
  uv run ruff check .
  exit 0
fi

echo "==> ruff format"
uv run ruff format .

echo "==> ruff check --fix"
uv run ruff check --fix .
