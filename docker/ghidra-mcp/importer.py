#!/usr/bin/env python3
"""Binary import service for the ghidra-mcp container.

The web container has no Ghidra install, so it cannot run ``analyzeHeadless`` —
the only working way to import a *new* binary into the shared Ghidra Server repo
(the engine's ``/server/version_control/add`` endpoint is a stub that never
writes a program). This small stdlib HTTP service runs alongside the headless
engine and the MCP bridge and is the component that actually performs the import.

The web server forwards an uploaded file here as a raw ``application/octet-stream``
body with ``?name=<program-name>``; we stage it and run::

    analyzeHeadless ghidra://<server>/<repo> -import <file> -noanalysis \
        -max-cpu 1 -connect <user> -p -commit "..."

Deliberately conservative on resources (full auto-analysis previously exhausted
the host): ``-noanalysis``, a single CPU, a capped JVM heap, and at most one
import at a time (a global lock). Analysis is left for later, on demand.

Endpoints:
    GET  /healthz                  -> 200 "ok"
    POST /import?name=&repo=        -> 200 imported | 400 | 409 exists | 502
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# A program name becomes both an analyzeHeadless argument and a repository path,
# so keep it to an unsurprising allowlist (no separators, no shell metachars).
NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

GHIDRA_HOME = os.environ.get("GHIDRA_HOME", "/opt/ghidra")
ANALYZE_HEADLESS = f"{GHIDRA_HOME}/support/analyzeHeadless"
ENGINE_PORT = os.environ.get("GHIDRA_MCP_PORT", "8089")
IMPORTER_PORT = int(os.environ.get("IMPORTER_PORT", "8082"))

SERVER_HOST = os.environ.get("GHIDRA_SERVER_HOST", "ghidra-server")
SERVER_PORT = os.environ.get("GHIDRA_SERVER_PORT", "13100")
SERVER_USER = os.environ.get("GHIDRA_SERVER_USER", "agent")
SERVER_PASSWORD = os.environ.get("GHIDRA_SERVER_PASSWORD", "")
DEFAULT_REPO = os.environ.get("GHIDRA_DEFAULT_REPOSITORY", "agent-shared")

# Cap the request body we will read (defense in depth; the web tier also caps).
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "256"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Serialize imports: never run two analyzeHeadless JVMs at once on this host.
_IMPORT_LOCK = threading.Lock()


class ImportFailure(Exception):
    """Raised to map an import failure to an HTTP status code."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _repo_has_program(repo: str, name: str) -> bool | None:
    """Return whether ``name`` already exists in ``repo`` per the local engine.

    Best-effort: returns True/False when the engine answers, or None when it
    cannot be reached (so the caller can fall back to analyzeHeadless's own
    duplicate detection rather than failing the request).
    """
    base = f"http://127.0.0.1:{ENGINE_PORT}"
    try:
        # Ensure the engine is logged into the server before listing files.
        urllib.request.urlopen(
            urllib.request.Request(f"{base}/server/connect", method="POST"),
            timeout=30,
        ).read()
        query = urllib.parse.urlencode({"repo": repo, "folderPath": "/"})
        with urllib.request.urlopen(
            f"{base}/server/repository/files?{query}", timeout=30
        ) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None

    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, list):
        return False
    return any(isinstance(f, dict) and f.get("name") == name for f in files)


def _run_import(staged_path: str, repo: str, name: str) -> dict[str, object]:
    """Run analyzeHeadless to import ``staged_path`` into ``repo`` and commit.

    Raises ImportFailure with an HTTP status on failure.
    """
    if not SERVER_PASSWORD:
        raise ImportFailure(500, "GHIDRA_SERVER_PASSWORD is not set in the importer")

    project_url = f"ghidra://{SERVER_HOST}:{SERVER_PORT}/{repo}"
    cmd = [
        ANALYZE_HEADLESS,
        project_url,
        "-import",
        staged_path,
        "-noanalysis",
        "-max-cpu",
        "1",
        "-connect",
        SERVER_USER,
        "-p",
        "-commit",
        "uploaded via web",
    ]
    env = dict(os.environ)
    # The server derives the login SID from the JVM user.name (it runs
    # "Prompt for user ID: no"); this container runs as root, so force it to the
    # service account. MAXMEM caps the Ghidra launcher heap.
    env["JAVA_TOOL_OPTIONS"] = f"-Duser.name={SERVER_USER}"
    env["MAXMEM"] = "1G"

    with _IMPORT_LOCK:
        proc = subprocess.run(
            cmd,
            input=f"{SERVER_PASSWORD}\n",
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    out = f"{proc.stdout}\n{proc.stderr}"

    if "Added file to repository" in out or "Import succeeded" in out:
        return {
            "status": "imported",
            "repository": repo,
            "name": name,
            "analyzed": False,
        }
    # analyzeHeadless refuses an existing name without -overwrite; honor the
    # reject-on-duplicate policy even if the pre-check missed it.
    if re.search(r"already exists", out, re.IGNORECASE):
        raise ImportFailure(409, f"a program named '{name}' already exists in '{repo}'")
    tail = "\n".join(line for line in out.splitlines() if line.strip())[-1500:]
    raise ImportFailure(502, f"import failed: {tail}")


class Handler(BaseHTTPRequestHandler):
    server_version = "GhidraImporter/1.0"

    def _json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A002
        # Route access logs to stdout so docker captures them.
        print("importer: " + (fmt % args))

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/healthz":
            self._json(200, {"status": "ok"})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/import":
            self._json(404, {"error": "not found"})
            return

        params = urllib.parse.parse_qs(parsed.query)
        name = (params.get("name", [""])[0]).strip()
        repo = (params.get("repo", [DEFAULT_REPO])[0]).strip() or DEFAULT_REPO

        if not NAME_RE.match(name):
            self._json(
                400,
                {"error": "invalid program name (allowed: letters, digits, . _ -)"},
            )
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            self._json(400, {"error": "empty body"})
            return
        if length > MAX_UPLOAD_BYTES:
            self._json(413, {"error": f"file exceeds {MAX_UPLOAD_MB} MB"})
            return

        exists = _repo_has_program(repo, name)
        if exists is True:
            self._json(
                409,
                {"error": f"a program named '{name}' already exists in '{repo}'"},
            )
            return

        tmpdir = tempfile.mkdtemp(prefix="import-")
        try:
            staged = os.path.join(tmpdir, name)
            self._stream_to_file(staged, length)
            result = _run_import(staged, repo, name)
            self._json(200, result)
        except ImportFailure as exc:
            self._json(exc.status, {"error": exc.message})
        except subprocess.TimeoutExpired:
            self._json(504, {"error": "import timed out"})
        except OSError as exc:
            self._json(500, {"error": f"import error: {exc}"})
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _stream_to_file(self, dest: str, length: int) -> None:
        remaining = length
        with open(dest, "wb") as fh:
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)


def main() -> None:
    httpd = ThreadingHTTPServer(("0.0.0.0", IMPORTER_PORT), Handler)
    print(f"importer: listening on 0.0.0.0:{IMPORTER_PORT} (repo={DEFAULT_REPO})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
