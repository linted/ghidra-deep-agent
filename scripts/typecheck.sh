#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> mypy"
uv run mypy .
