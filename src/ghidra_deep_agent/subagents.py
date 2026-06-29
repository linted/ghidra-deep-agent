"""Sub-agent definitions for delegation / context quarantine.

The main agent stays lean by delegating verbose or long-running work to
specialized sub-agents — only a compact summary returns to the main context,
not the dozens of tool calls (catalog dumps, decompiler blobs, slow threat
sweeps) that produced it.

Three sub-agents are defined:

- ``program-recon``  — read-only "what binary is this" reconnaissance.
- ``function-analyst`` — the full per-function loop: analyze, apply Ghidra
  changes (rename/retype/comment/prototype), and save findings.
- ``threat-hunter`` — isolates the heavy threat-analysis tools off the main
  critical path.

Tool allowlists are name-based and filtered against the live tool set, so a
renamed or absent Ghidra MCP tool is skipped (with a startup warning) rather
than crashing startup. ``SubAgent`` middleware does not inherit from the main
agent, so each sub-agent gets its own ``ArgumentValidationMiddleware``.
"""

import sys
from collections.abc import Sequence

from deepagents import SubAgent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ghidra_deep_agent.validation import create_argument_validation_middleware

# --- Tool allowlists (by name) -------------------------------------------------
# Names mirror the Ghidra MCP server's tools plus our knowledge tools. They are
# filtered against the live tool set at build time; misses are logged, not fatal.

_RECON_TOOLS = (
    # Ghidra read-only reconnaissance
    "get_current_program_info",
    "list_functions",
    "list_imports",
    "list_exports",
    "get_entry_points",
    "search_strings",
    # Knowledge base (read)
    "get_knowledge_summary",
    "query_knowledge",
    "query_by_address",
    "query_by_category",
    "query_by_tags",
    "list_analyzed_binaries",
)

_FUNCTION_ANALYST_TOOLS = (
    # Analysis (read)
    "decompile_function",
    "batch_decompile",
    "disassemble_function",
    "get_function_by_address",
    "get_function_callers",
    "get_function_callees",
    "get_function_xrefs",
    "get_xrefs_to",
    "get_xrefs_from",
    "read_memory",
    "search_instructions",
    "analyze_function_complete",
    "analyze_data_region",
    # Mutations (apply)
    "rename_function",
    "rename_variable",
    "set_local_variable_type",
    "set_decompiler_variable_type",
    "set_parameter_type",
    "set_function_prototype",
    "set_decompiler_comment",
    "set_disassembly_comment",
    "batch_rename_function_components",
    "batch_add_function_tags",
    # Knowledge base (read + write)
    "save_knowledge",
    "update_knowledge",
    "query_knowledge",
    "query_by_address",
)

_THREAT_HUNTER_TOOLS = (
    # Heavy threat analysis
    "find_anti_analysis_techniques",
    "detect_malware_behaviors",
    "extract_iocs_with_context",
    "detect_crypto_constants",
    "analyze_api_call_chains",
    # Supporting reads
    "decompile_function",
    "search_strings",
    "get_xrefs_to",
    "read_memory",
    "search_byte_patterns",
    # Knowledge base
    "save_knowledge",
    "query_knowledge",
)

# --- Sub-agent system prompts --------------------------------------------------

_RECON_PROMPT = (
    "You are a reconnaissance specialist for reverse engineering. Your job is to "
    "build a quick, high-level picture of an unfamiliar binary so the main agent "
    "can plan its analysis.\n"
    "\n"
    "Gather, in as few round-trips as possible (batch independent reads in one "
    "turn):\n"
    "- Program info: format, architecture, bit-width, base address, entry "
    "points.\n"
    "- The function inventory size and any notably named functions.\n"
    "- Imports and exports that hint at capability (networking, crypto, file "
    "I/O, process/thread, registry).\n"
    "- A scan of strings for telling artifacts (URLs, paths, commands, error "
    "messages, format strings).\n"
    "- Relevant prior knowledge: call `get_knowledge_summary` and "
    "`query_knowledge` to see what is already known about this binary.\n"
    "\n"
    "You are READ-ONLY: do not rename, retype, comment, or otherwise mutate the "
    "program. Return a single compact brief (a few short sections or bullets) "
    "covering format/arch/entry points, notable imports/exports/strings, counts, "
    "a one-line hypothesis about the binary's purpose, and anything already in "
    "the knowledge base. Do NOT paste full catalog listings or raw tool dumps — "
    "summarize."
)

_FUNCTION_ANALYST_PROMPT = (
    "You are an expert reverse engineer analyzing a single function (or a small "
    "set of related functions) in Ghidra. The assembly is ground truth: when it "
    "contradicts your assumptions, update your model — never dismiss what the "
    "assembly shows. Cite the instruction address or pattern behind every "
    "conclusion.\n"
    "\n"
    "For each function you are asked to analyze:\n"
    "1. Get the assembly and the decompiler output.\n"
    "2. Identify the calling convention and argument count from the prologue.\n"
    "3. Trace data flow through registers and the stack.\n"
    "4. Identify patterns — loops, conditionals, error checks, syscalls, API "
    "calls.\n"
    "5. Note cross-references: callers and callees.\n"
    "6. Query the knowledge base (`query_knowledge`, `query_by_address`) for "
    "prior findings before deciding anything.\n"
    "\n"
    "Then APPLY what the evidence supports, directly in Ghidra:\n"
    "- Rename variables/parameters to reflect purpose (e.g. `local_10` -> "
    "`file_size`).\n"
    "- Set correct types (e.g. `int` -> `FILE *`).\n"
    "- Rename the function and update its prototype to the real signature.\n"
    "- Add inline comments at non-obvious instructions.\n"
    "Use lowercase snake_case unless the binary uses another convention; prefix "
    "uncertain names with `maybe_`. Never rename or retype on speculation alone.\n"
    "\n"
    "Save every finding and decision to the knowledge base with `save_knowledge` "
    "(even partial or low-confidence ones). When done, return a COMPACT summary: "
    "what the function does, the key evidence, and the concrete changes you "
    "applied. Do NOT return the raw decompiler or disassembly output — the main "
    "agent only needs your conclusions."
)

_THREAT_HUNTER_PROMPT = (
    "You are a malware threat-analysis specialist. You run the heavy, slow "
    "threat-detection tools so the main agent doesn't pay their latency on its "
    "critical path.\n"
    "\n"
    "When asked to assess a binary or region, run the relevant heavy tools "
    "(`find_anti_analysis_techniques`, `detect_malware_behaviors`, "
    "`extract_iocs_with_context`, `detect_crypto_constants`, "
    "`analyze_api_call_chains`). These are independent — invoke them together in "
    "one turn so they overlap rather than running serially. Use the supporting "
    "reads (`decompile_function`, `search_strings`, `get_xrefs_to`) only to "
    "confirm or contextualize a hit.\n"
    "\n"
    "Persist concrete results — IOCs, detected behaviors, crypto constants, "
    "anti-analysis techniques — to the knowledge base with `save_knowledge`, "
    "tagged appropriately. Return a COMPACT threat summary: techniques found, "
    "behaviors, IOCs (deduplicated), and an overall risk read. Do NOT dump full "
    "raw tool output into your reply — summarize and rely on the knowledge base "
    "for detail."
)


def _select(
    by_name: dict[str, BaseTool], names: Sequence[str], *, subagent: str
) -> list[BaseTool]:
    """Return the tools whose names are in ``names``, skipping any not present.

    Tool names come from the Ghidra MCP server, which is the source of truth at
    runtime. A requested name that isn't available is reported and skipped so a
    renamed/removed tool can't crash agent startup.
    """
    selected: list[BaseTool] = []
    missing: list[str] = []
    for name in names:
        tool = by_name.get(name)
        if tool is None:
            missing.append(name)
        else:
            selected.append(tool)
    if missing:
        print(
            f"Warning: sub-agent '{subagent}' — {len(missing)} requested tool(s) "
            f"not available and skipped: {', '.join(sorted(missing))}",
            file=sys.stderr,
        )
    return selected


def build_subagents(
    all_tools: Sequence[BaseTool], subagent_model: str | BaseChatModel
) -> list[SubAgent]:
    """Build the program-recon, function-analyst, and threat-hunter sub-agents.

    Args:
        all_tools: The full tool set available to the main agent (knowledge
            tools + Ghidra MCP tools). Each sub-agent receives a name-filtered
            subset.
        subagent_model: Model the sub-agents run on (string or chat model).

    Returns:
        A list of ``SubAgent`` specs ready to pass to ``create_deep_agent``.
    """
    by_name = {tool.name: tool for tool in all_tools}

    recon: SubAgent = {
        "name": "program-recon",
        "description": (
            "Read-only reconnaissance. Delegate at the start of a session to get "
            "a compact brief on an unfamiliar binary (format, architecture, "
            "entry points, notable imports/exports/strings, prior knowledge)."
        ),
        "system_prompt": _RECON_PROMPT,
        "tools": _select(by_name, _RECON_TOOLS, subagent="program-recon"),
        "model": subagent_model,
        "middleware": [create_argument_validation_middleware()],
    }

    function_analyst: SubAgent = {
        "name": "function-analyst",
        "description": (
            "Deep analysis of one function (or a few related ones). It "
            "decompiles, traces data flow, applies the warranted renames/retypes/"
            "comments/prototype in Ghidra, saves findings to the knowledge base, "
            "and returns a compact summary. Delegate per function; several may "
            "run in parallel."
        ),
        "system_prompt": _FUNCTION_ANALYST_PROMPT,
        "tools": _select(by_name, _FUNCTION_ANALYST_TOOLS, subagent="function-analyst"),
        "model": subagent_model,
        "middleware": [create_argument_validation_middleware()],
    }

    threat_hunter: SubAgent = {
        "name": "threat-hunter",
        "description": (
            "Runs the heavy, slow threat-analysis tools (anti-analysis, malware "
            "behaviors, IOC extraction, crypto constants, API call chains) off "
            "the main critical path, writes findings to the knowledge base, and "
            "returns a compact threat summary."
        ),
        "system_prompt": _THREAT_HUNTER_PROMPT,
        "tools": _select(by_name, _THREAT_HUNTER_TOOLS, subagent="threat-hunter"),
        "model": subagent_model,
        "middleware": [create_argument_validation_middleware()],
    }

    return [recon, function_analyst, threat_hunter]
