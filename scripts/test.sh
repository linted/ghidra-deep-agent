#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Integration tests (test_knowledge.py) need a live MongoDB with vector search
# and an embedding model; run them explicitly with `uv run pytest -m integration`.
echo "==> pytest"
uv run pytest -m "not integration" "$@"
