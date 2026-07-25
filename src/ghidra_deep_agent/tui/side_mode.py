"""State for the TUI's read-only side modes (``/plan`` and ``/ask``).

Both run a read-only coordinator on its own ephemeral checkpointer thread, minted
on entry and dropped on exit, and both are mutually exclusive with each other.
They used to be two parallel sets of attributes and three pairs of near-identical
methods, which is how the flags drifted: ``_run_agent`` cleared *both* seed flags
whenever *either* mode had seeded, correct only by the accident of exclusivity.

Modelling "the side mode currently active, if any" as one optional object makes
that state unrepresentable — there is only ever one seed flag to clear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Kind = Literal["plan", "ask"]


@dataclass
class SideMode:
    """The active side mode: which one, its thread, and its pending seed."""

    kind: Kind
    # Graph config for this mode's ephemeral thread. Never falls back to the main
    # config — that would run a side mode's turn on the real session thread.
    config: dict[str, Any]
    # True until the first turn seeds the thread with a marked summary of the main
    # session. Follow-up turns in the same mode don't re-seed.
    needs_seed: bool = True
    # Plan mode only: the timestamped markdown file the planner writes each turn.
    plan_path: str | None = None

    @property
    def is_plan(self) -> bool:
        return self.kind == "plan"

    @property
    def is_ask(self) -> bool:
        return self.kind == "ask"

    @property
    def label(self) -> str:
        """Human-facing name, for status messages."""
        return "Plan" if self.is_plan else "Ask"

    def decorate(self, query: str) -> str:
        """Prefix the user's text with the mode's standing instruction."""
        if self.is_plan and self.plan_path:
            return (
                f"[Plan mode — write/maintain the complete plan at "
                f"`{self.plan_path}`]\n\n{query}"
            )
        if self.is_ask:
            return (
                "[Ask mode — decompose the question(s), delegate investigation "
                "to the research sub-agent, and synthesize a grounded, cited "
                f"answer]\n\n{query}"
            )
        return query
