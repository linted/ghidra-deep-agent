#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> ruff format"
uv run ruff format .

echo "==> ruff check --fix"
uv run ruff check --fix .
