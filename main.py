"""Convenience launcher: `uv run python main.py`.

The implementation lives in `ghidra_deep_agent.cli` so the installed
`ghidra-deep-agent` console script can reach it — a top-level `main` module is
not part of the wheel.
"""

from ghidra_deep_agent.cli import run

if __name__ == "__main__":
    run()
