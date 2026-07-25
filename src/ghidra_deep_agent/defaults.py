"""Fallback values and env parsing shared between the entry point and the TUI.

The constants have real defaults in ``cli.py`` (from ``RECURSION_LIMIT`` and
``MAX_CONTEXT_TOKENS``), but the TUI needs the same numbers when it mints its
own graph configs or renders the context gauge. They live here so the two can't
drift apart — which they did for a while, because ``cli.py`` kept restating the
literals instead of importing them.

``env_int`` / ``env_float`` live here for the same reason: several modules read
numeric knobs out of the environment, and a bare ``int(os.environ[...])`` turns a
typo in ``.env`` into a ``ValueError`` traceback at startup with no indication of
which variable was wrong.
"""

import os
import sys
from pathlib import Path

# LangGraph recursion limit for a deep analysis session. Overridden by
# RECURSION_LIMIT in cli.py; used here when a config carries no explicit value.
DEFAULT_RECURSION_LIMIT = 10000

# Context-window size assumed when the model exposes no profile. Overridden by
# MAX_CONTEXT_TOKENS in cli.py.
DEFAULT_MAX_CONTEXT_TOKENS = 200_000


def config_path(env_var: str, filename: str) -> Path:
    """Resolve an optional TOML config: ``env_var`` if set, else the repo root.

    ``parents[2]`` is the repo root for a source checkout
    (``<root>/src/ghidra_deep_agent/defaults.py``), but for an installed wheel it
    is whatever directory happens to contain ``site-packages`` — where no config
    lives. Since the project now ships a console script, fall back to the current
    working directory in that case, which is where someone running
    ``ghidra-deep-agent`` from their project would keep these files.
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / filename
    if candidate.exists():
        return candidate
    return Path.cwd() / filename


def _warn(var: str, raw: str, default: object, why: str) -> None:
    print(
        f"Warning: {var}={raw!r} is {why}; using {default}.",
        file=sys.stderr,
    )


def env_int(var: str, default: int, *, positive: bool = True) -> int:
    """Read an integer env var, falling back to ``default`` on anything invalid.

    ``positive`` rejects zero and negatives — every caller here is a count, a
    limit, or a TTL, where zero means "busy-loop" or "cache nothing" rather than
    anything the user intended.
    """
    raw = os.environ.get(var, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        _warn(var, raw, default, "not an integer")
        return default
    if positive and value <= 0:
        _warn(var, raw, default, "not positive")
        return default
    return value


def env_float(var: str, default: float, *, positive: bool = True) -> float:
    """Read a float env var, falling back to ``default`` on anything invalid."""
    raw = os.environ.get(var, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        _warn(var, raw, default, "not a number")
        return default
    if positive and value <= 0:
        _warn(var, raw, default, "not positive")
        return default
    return value
