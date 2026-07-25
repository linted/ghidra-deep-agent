"""Fallback values shared between the entry point and the TUI.

These have real defaults in ``main.py`` (from ``RECURSION_LIMIT`` and
``MAX_CONTEXT_TOKENS``), but the TUI needs the same numbers when it mints its
own graph configs or renders the context gauge. They live here so the two can't
drift apart.
"""

# LangGraph recursion limit for a deep analysis session. Overridden by
# RECURSION_LIMIT in main.py; used here when a config carries no explicit value.
DEFAULT_RECURSION_LIMIT = 10000

# Context-window size assumed when the model exposes no profile. Overridden by
# MAX_CONTEXT_TOKENS in main.py.
DEFAULT_MAX_CONTEXT_TOKENS = 200_000
