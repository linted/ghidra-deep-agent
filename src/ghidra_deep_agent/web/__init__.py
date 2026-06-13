"""Web browser front-end for the Ghidra deep agent.

A FastAPI app that exposes the same agent runtime as the TUI over HTTP + a
WebSocket event stream, with a lightweight vanilla-JS client served from
``static/``. See [server.py](server.py) for the entry point.
"""
