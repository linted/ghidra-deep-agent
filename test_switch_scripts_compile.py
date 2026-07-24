"""Compile-check the embedded Java GhidraScripts against the installed Ghidra.

The Java sources in ``*_script.py`` are compiled inside Ghidra at runtime, so an
API mismatch (e.g. a changed constructor signature) is invisible to the pure
Python tests and only surfaces as a "no JSON manifest found" failure in the
live agent. This test compiles each ``SCRIPT_SOURCE`` with ``javac`` against
the local Ghidra jars to catch that drift in CI on machines that have Ghidra.

Skips cleanly when Ghidra or a JDK is not installed.

Run:  uv run pytest test_switch_scripts_compile.py -v
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from ghidra_deep_agent import (
    apply_switch_override_script,
    find_unrecovered_switches_script,
    recover_prototypes_script,
)

_SCRIPT_MODULES = [
    apply_switch_override_script,
    find_unrecovered_switches_script,
    recover_prototypes_script,
]

_CLASS_RE = re.compile(r"public class (\w+)")

_CLASSPATH_JARS = [
    "Features/Base/lib/Base.jar",
    "Features/Decompiler/lib/Decompiler.jar",
    "Framework/SoftwareModeling/lib/SoftwareModeling.jar",
    "Framework/Generic/lib/Generic.jar",
    "Framework/Project/lib/Project.jar",
    "Framework/Utility/lib/Utility.jar",
    "Framework/Generic/lib/gson-*.jar",
]

_HOMEBREW_JAVAC = "/opt/homebrew/opt/openjdk@21/bin/javac"


def _find_ghidra_root() -> Path | None:
    env = os.environ.get("GHIDRA_INSTALL_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    hits = sorted(glob.glob("/opt/homebrew/Cellar/ghidra/*/libexec/Ghidra"))
    return Path(hits[-1]) if hits else None


def _javac_works(javac: str) -> bool:
    # macOS ships a /usr/bin/javac stub that fails when no system JDK is
    # installed, so "exists" is not enough — it has to actually run.
    try:
        return subprocess.run([javac, "-version"], capture_output=True).returncode == 0
    except OSError:
        return False


def _find_javac() -> str | None:
    # Prefer the Homebrew JDK (the one Ghidra itself depends on).
    candidates = [_HOMEBREW_JAVAC, shutil.which("javac")]
    for javac in candidates:
        if javac and Path(javac).is_file() and _javac_works(javac):
            return javac
    return None


def _build_classpath(ghidra_root: Path) -> str:
    jars: list[str] = []
    for rel in _CLASSPATH_JARS:
        matches = sorted(glob.glob(str(ghidra_root / rel)))
        if not matches:
            pytest.skip(f"Ghidra jar not found: {ghidra_root / rel}")
        jars.append(matches[-1])
    return os.pathsep.join(jars)


@pytest.mark.parametrize(
    "module", _SCRIPT_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1]
)
def test_embedded_script_compiles(module: ModuleType, tmp_path: Path) -> None:
    ghidra_root = _find_ghidra_root()
    if ghidra_root is None:
        pytest.skip("no Ghidra install found (set GHIDRA_INSTALL_DIR)")
    javac = _find_javac()
    if javac is None:
        pytest.skip("no javac found")

    source: str = module.SCRIPT_SOURCE
    match = _CLASS_RE.search(source)
    assert match, f"no public class in {module.__name__}"
    class_name = match.group(1)

    # javac requires the file name to match the public class name.
    java_file = tmp_path / f"{class_name}.java"
    java_file.write_text(source)

    result = subprocess.run(
        [
            javac,
            "-proc:none",
            "-nowarn",
            "-cp",
            _build_classpath(ghidra_root),
            "-d",
            str(tmp_path),
            str(java_file),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{class_name}.java failed to compile against {ghidra_root}:\n{result.stderr}"
    )
