#!/usr/bin/env python3
"""Binary import service, co-located with the Ghidra Server.

The web container has no Ghidra install, so it cannot run ``analyzeHeadless`` —
the only working way to import a *new* binary into the shared Ghidra Server repo
(the engine's ``/server/version_control/add`` endpoint is a stub that never
writes a program). This small stdlib HTTP service runs in the ghidra-server
container, the single persistent owner of the shared repo: importing a binary is
a global, session-agnostic operation, so it belongs here rather than on a
per-session ghidra-mcp engine (of which there may eventually be many).

The web server forwards an uploaded file here as a raw ``application/octet-stream``
body with ``?name=<program-name>``; we stage it and run::

    analyzeHeadless ghidra://<server>/<repo> -import <file> -noanalysis \
        -max-cpu 1 -connect <user> -p -commit "..."

Deliberately conservative on resources (full auto-analysis previously exhausted
the host): ``-noanalysis``, a single CPU, a capped JVM heap, and at most one
import at a time (a global lock). Analysis is left for later, on demand.

Raw/headerless binaries (e.g. firmware blobs) have no container Ghidra can
auto-detect, so the caller may supply import hints — ``loader``, ``processor``
(language ID), ``cspec`` (compiler spec) and ``base`` (image base) — which become
``-loader BinaryLoader -processor <id> -cspec <id> -loader-baseAddr <addr>``.
``processor``/``cspec`` are validated against the languages Ghidra actually has
installed (parsed from its ``.ldefs`` files, the same source ``DefaultLanguage\
Service`` reads); ``GET /languages`` serves that list to populate the UI picker.

Endpoints:
    GET  /healthz                  -> 200 "ok"
    GET  /languages                -> 200 {languages:[{id,compilerSpecs}], count}
    POST /import?name=&repo=&loader=&processor=&cspec=&base=
                                    -> 200 imported | 400 | 409 exists | 422 | 502
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.parse
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# A program name becomes both an analyzeHeadless argument and a repository path,
# so keep it to an unsurprising allowlist (no separators, no shell metachars).
NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
# Loader names are single CamelCase words (e.g. BinaryLoader, ElfLoader).
LOADER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,40}$")
# Image base is an address: optional 0x prefix then hex digits.
BASE_RE = re.compile(r"^(0x)?[0-9A-Fa-f]{1,16}$")

GHIDRA_HOME = os.environ.get("GHIDRA_HOME", "/opt/ghidra")
ANALYZE_HEADLESS = f"{GHIDRA_HOME}/support/analyzeHeadless"
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

# Installed languages: {languageID: [compilerSpecID, ...]}. Parsed once from the
# Ghidra .ldefs files (the same data DefaultLanguageService loads) and cached;
# the set is static for a given install, so there is no need for a JVM.
_LANG_CACHE: dict[str, list[str]] | None = None
_LANG_LOCK = threading.Lock()


def load_languages() -> dict[str, list[str]]:
    """Return ``{languageID: [compilerSpecID, ...]}`` for the installed Ghidra."""
    global _LANG_CACHE
    if _LANG_CACHE is not None:
        return _LANG_CACHE
    with _LANG_LOCK:
        if _LANG_CACHE is not None:
            return _LANG_CACHE
        langs: dict[str, list[str]] = {}
        pattern = os.path.join(
            GHIDRA_HOME, "Ghidra", "Processors", "*", "data", "languages", "*.ldefs"
        )
        for path in glob.glob(pattern):
            try:
                root = ET.parse(path).getroot()
            except ET.ParseError:
                continue
            for lang in root.iter("language"):
                lid = lang.get("id")
                if not lid:
                    continue
                cspecs: list[str] = []
                for compiler in lang.findall("compiler"):
                    cid = compiler.get("id")
                    if cid:
                        cspecs.append(cid)
                langs[lid] = cspecs
        _LANG_CACHE = langs
        return langs


class ImportFailure(Exception):
    """Raised to map an import failure to an HTTP status code."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _import_hint_args(
    loader: str | None, processor: str | None, cspec: str | None, base: str | None
) -> list[str]:
    """Build the analyzeHeadless loader/processor flags from validated hints.

    A ``processor`` implies a raw load, so the loader defaults to BinaryLoader
    when one is not named explicitly. ``base`` is BinaryLoader's image base
    (``-loader-baseAddr``) and only applies to that loader.
    """
    effective_loader = loader or ("BinaryLoader" if processor else None)
    args: list[str] = []
    if effective_loader:
        args += ["-loader", effective_loader]
    if processor:
        args += ["-processor", processor]
    if cspec:
        args += ["-cspec", cspec]
    if base and effective_loader == "BinaryLoader":
        args += ["-loader-baseAddr", base]
    return args


def _run_import(
    staged_path: str,
    repo: str,
    name: str,
    *,
    loader: str | None = None,
    processor: str | None = None,
    cspec: str | None = None,
    base: str | None = None,
) -> dict[str, object]:
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
        *_import_hint_args(loader, processor, cspec, base),
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
    # analyzeHeadless refuses an existing name without -overwrite; map that to
    # the reject-on-duplicate policy (409). On commit into a shared repo the
    # message is "Found conflicting program file in project"; a plain import
    # collision says "already exists".
    if re.search(r"already exists|conflicting program file", out, re.IGNORECASE):
        raise ImportFailure(409, f"a program named '{name}' already exists in '{repo}'")
    # A headerless/raw binary has no auto-detectable format; the caller must say
    # which processor (language ID) to use. Surface that as a client error.
    if "No load spec found" in out:
        raise ImportFailure(
            422,
            "Ghidra could not auto-detect a format for this file; specify a "
            "processor (language ID), e.g. ARM:LE:32:v8, to import a raw binary",
        )
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
        if path == "/languages":
            langs = load_languages()
            self._json(
                200,
                {
                    "languages": [
                        {"id": lid, "compilerSpecs": langs[lid]}
                        for lid in sorted(langs)
                    ],
                    "count": len(langs),
                },
            )
            return
        self._json(404, {"error": "not found"})

    def _import_hints(self, params: dict[str, list[str]]) -> dict[str, str | None]:
        """Validate optional loader/processor/cspec/base params or raise 400/422.

        ``processor``/``cspec`` are checked against the installed languages so an
        unknown value is rejected up front rather than after a JVM spin-up.
        """

        def one(key: str) -> str | None:
            val = (params.get(key, [""])[0]).strip()
            return val or None

        loader, processor, cspec, base = (
            one("loader"),
            one("processor"),
            one("cspec"),
            one("base"),
        )
        if loader and not LOADER_RE.match(loader):
            raise ImportFailure(400, "invalid loader name")
        if base and not BASE_RE.match(base):
            raise ImportFailure(400, "invalid image base (expected hex, e.g. 0x8000)")
        if cspec and not processor:
            raise ImportFailure(400, "cspec requires a processor")
        if processor:
            langs = load_languages()
            if processor not in langs:
                raise ImportFailure(400, f"unknown processor '{processor}'")
            if cspec and cspec not in langs[processor]:
                raise ImportFailure(
                    400, f"unknown cspec '{cspec}' for processor '{processor}'"
                )
        return {
            "loader": loader,
            "processor": processor,
            "cspec": cspec,
            "base": base,
        }

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

        try:
            hints = self._import_hints(params)
        except ImportFailure as exc:
            self._json(exc.status, {"error": exc.message})
            return

        # Duplicate detection is left to analyzeHeadless, which refuses an
        # existing name; _run_import maps that to 409. (There is no co-located
        # engine REST to pre-check against in the ghidra-server container.)
        tmpdir = tempfile.mkdtemp(prefix="import-")
        try:
            staged = os.path.join(tmpdir, name)
            self._stream_to_file(staged, length)
            result = _run_import(staged, repo, name, **hints)
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
