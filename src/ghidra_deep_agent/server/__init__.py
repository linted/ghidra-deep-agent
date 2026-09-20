"""HTTP server: many agents, one Ghidra.

Nothing in this package may import ``ghidra_deep_agent.tui`` — the server
runs without Textual, and importing any ``tui`` submodule loads the whole app.
"""
