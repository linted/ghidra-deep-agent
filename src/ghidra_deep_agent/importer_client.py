"""Client for the ghidra-mcp container's binary import service.

The web tier cannot import binaries itself (no Ghidra), so it forwards uploaded
bytes to the importer running next to the headless engine. This module isolates
that HTTP call — mirroring how :mod:`program_resolver` isolates MCP calls — so the
FastAPI route stays thin and can translate the importer's status codes faithfully.

Uses only the standard library: the blocking ``urllib`` request is run in a worker
thread so it does not stall the event loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

# The import does a JVM spin-up + commit, so allow a generous deadline.
_TIMEOUT_SECONDS = 600.0


@dataclass
class ImportResult:
    """Outcome of forwarding an upload to the importer.

    ``status_code`` is the importer's HTTP status (200 ok, 409 duplicate,
    400 bad name, 413 too large, 5xx failure); ``payload`` is its JSON body.
    """

    status_code: int
    payload: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.status_code == 200


def _importer_url() -> str:
    return os.environ.get("GHIDRA_IMPORTER_URL", "http://ghidra-mcp:8082/import")


def _decode(status: int, body: bytes) -> ImportResult:
    try:
        payload = json.loads(body.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        payload = {"error": body.decode("utf-8", "replace") or "non-JSON response"}
    if not isinstance(payload, dict):
        payload = {"result": payload}
    return ImportResult(status_code=status, payload=payload)


def _post(url: str, params: dict[str, str], data: bytes) -> ImportResult:
    req = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            return _decode(resp.status, resp.read())
    except urllib.error.HTTPError as exc:
        # 4xx/5xx from the importer carry a JSON error body we want to surface.
        return _decode(exc.code, exc.read())
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"import service unreachable: {exc}") from exc


async def import_binary(
    name: str, data: bytes, repo: str | None = None
) -> ImportResult:
    """Forward ``data`` to the importer as the program ``name``.

    Returns an :class:`ImportResult` carrying the importer's status and body.
    Raises ``RuntimeError`` only when the importer is unreachable, so the caller
    can distinguish "import service down" from a normal rejection (e.g. 409).
    """
    params: dict[str, str] = {"name": name}
    if repo:
        params["repo"] = repo
    return await asyncio.to_thread(_post, _importer_url(), params, data)
