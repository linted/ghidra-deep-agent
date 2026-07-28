"""Consistency checks between each embedded Java script and its Python wrapper.

Two facts are stated twice — once in Python, once inside the Java string literal —
with nothing keeping them in sync:

* the manifest markers, which the wrapper greps for and the script prints;
* the deployed file name, which must match the Java ``public class`` name because
  Ghidra's Java provider compiles ``<name>.java`` expecting ``class <name>``.

Editing one side alone breaks every run of that tool with an unhelpful "no JSON
manifest found". Unlike ``test_switch_scripts_compile.py`` these need no Ghidra
and no JDK, so they run everywhere — including CI.

Run:  uv run pytest tests/test_script_consistency.py -v
"""

from __future__ import annotations

import re

import pytest

from ghidra_deep_agent import (
    apply_switch_override_script,
    find_unrecovered_switches_script,
    ollvm_deobfuscate_script,
    recover_prototypes_script,
)
from ghidra_deep_agent.prototype_tools import _SCRIPT_NAME as PROTO_SCRIPT_NAME
from ghidra_deep_agent.switch_tools import (
    _APPLY_SCRIPT_NAME,
    _FIND_SCRIPT_NAME,
    _OLLVM_SCRIPT_NAME,
)

# (module, deployed file name) for every embedded GhidraScript.
SCRIPTS = [
    pytest.param(find_unrecovered_switches_script, _FIND_SCRIPT_NAME, id="find"),
    pytest.param(apply_switch_override_script, _APPLY_SCRIPT_NAME, id="apply"),
    pytest.param(ollvm_deobfuscate_script, _OLLVM_SCRIPT_NAME, id="ollvm"),
    pytest.param(recover_prototypes_script, PROTO_SCRIPT_NAME, id="recover"),
]


@pytest.mark.parametrize(("module", "script_name"), SCRIPTS)
def test_markers_match_the_java_source(module: object, script_name: str) -> None:
    """The marker the wrapper greps for must be the one the script prints.

    Asserted on the literal *value*, not a declaration shape: the scripts don't
    agree on a Java constant name (`MARK_START` vs `MANIFEST_START`) or on
    visibility, and none of that matters — only that the exact string the regex
    looks for is present in the source that emits it.
    """
    source = module.SCRIPT_SOURCE  # type: ignore[attr-defined]
    for const in ("MARK_START", "MARK_END"):
        value = getattr(module, const)
        assert f'"{value}"' in source, (
            f"{script_name}: Python {const}={value!r} appears nowhere in the Java "
            f"source. The wrapper would find no manifest at runtime."
        )


@pytest.mark.parametrize(("module", "script_name"), SCRIPTS)
def test_public_class_matches_the_deployed_name(
    module: object, script_name: str
) -> None:
    source = module.SCRIPT_SOURCE  # type: ignore[attr-defined]
    match = re.search(r"public class (\w+) extends GhidraScript", source)
    assert match is not None, f"{script_name}: no public GhidraScript class found"
    expected = script_name.removesuffix(".java")
    assert match.group(1) == expected, (
        f"{script_name}: deployed as {expected}.java but the class is "
        f"{match.group(1)} — Ghidra's Java provider will fail to compile it."
    )


@pytest.mark.parametrize(("module", "script_name"), SCRIPTS)
def test_markers_are_distinct(module: object, script_name: str) -> None:
    """A start marker that is a prefix of the end marker breaks the regex."""
    start = module.MARK_START  # type: ignore[attr-defined]
    end = module.MARK_END  # type: ignore[attr-defined]
    assert start != end
    assert start not in end, f"{script_name}: MARK_START is a substring of MARK_END"
