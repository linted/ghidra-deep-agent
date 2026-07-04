import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.filesystem import FilesystemBackend
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo.errors import ServerSelectionTimeoutError

from ghidra_deep_agent.async_tasks import build_async_task_middleware
from ghidra_deep_agent.compaction import (
    auto_summarization_tuning_enabled,
    create_forced_summarization_tool_middleware,
    install_tuned_summarization,
)
from ghidra_deep_agent.ghidra_transport import get_mcp_config
from ghidra_deep_agent.knowledge import build_knowledge_tools
from ghidra_deep_agent.mcp_cache import build_mcp_cache_middleware
from ghidra_deep_agent.models import build_embeddings
from ghidra_deep_agent.program_resolver import resolve_binary_name
from ghidra_deep_agent.prompt import (
    ASK_MODE_SYSTEM_PROMPT,
    PLAN_MODE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    format_agent_memory,
)
from ghidra_deep_agent.resilience import (
    build_model_resilience_middleware,
    build_tool_retry_middleware,
)
from ghidra_deep_agent.sessions import build_session_store
from ghidra_deep_agent.subagents import (
    build_main_tools,
    build_plan_mode_main_tools,
    build_research_subagent,
    build_subagents,
    filter_withheld_tools,
    load_agent_config,
    make_model_resolver,
    resolve_model_spec,
)
from ghidra_deep_agent.tui import GhidraAgentApp
from ghidra_deep_agent.validation import create_argument_validation_middleware


async def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ghidra deep agent")
    parser.add_argument(
        "--session-id", default=None, help="Resume a previous session by ID"
    )
    parser.add_argument(
        "--binary-name",
        default=None,
        help="Binary name to use for knowledge isolation (overrides auto-detection)",
    )
    args = parser.parse_args()
    session_id = args.session_id or str(uuid.uuid4())

    mcp_config = get_mcp_config()
    # Fail fast on a bad config before connecting to anything.
    try:
        agent_config = load_agent_config()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    resolve_model = make_model_resolver(agent_config.default_model)

    agents_md_path = os.environ.get("AGENTS_MD", "")
    agents_md = ""
    if agents_md_path:
        resolved = Path(agents_md_path).expanduser()
        try:
            agents_md = resolved.read_text(encoding="utf-8")
            print(f"AGENTS.md memory loaded [{resolved}]")
        except OSError as exc:
            print(
                f"Warning: could not read AGENTS_MD file {resolved} ({exc})",
                file=sys.stderr,
            )

    transport_desc = mcp_config["ghidra"].get("transport", "http")
    url = mcp_config["ghidra"].get("url", "")
    print(f"Connecting to Ghidra MCP server [{transport_desc}: {url}]...")

    async def handle_mcp_errors(request: MCPToolCallRequest, handler: Any) -> Any:
        try:
            return await handler(request)
        except Exception as exc:
            return f"Tool '{request.name}' failed: {exc}"

    try:
        client = MultiServerMCPClient(mcp_config, tool_interceptors=[handle_mcp_errors])
        tools = await client.get_tools()
    except Exception as exc:
        print(f"Failed to connect to Ghidra MCP server: {exc}", file=sys.stderr)
        print(
            "Ensure Ghidra is running with the GhidrAssistMCP plugin enabled "
            "(MCP server on) and a program open, then set GHIDRA_MCP_TRANSPORT / "
            "GHIDRA_MCP_URL as needed.",
            file=sys.stderr,
        )
        sys.exit(1)

    # MCP server errors arrive as isError=True results, which langchain_mcp_adapters
    # converts to ToolException. Without handle_tool_error=True, ToolException bypasses
    # LangGraph's ToolNode default handler (which only catches ToolInvocationError) and
    # propagates all the way up through sub-agents to the TUI.
    for tool in tools:
        tool.handle_tool_error = True

    if not tools:
        print("Warning: no tools loaded from Ghidra MCP server.", file=sys.stderr)
    else:
        names = ", ".join(t.name for t in tools)
        print(f"Loaded {len(tools)} Ghidra tool(s): {names}")

    mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_db = os.environ.get("MONGODB_DB", "checkpointing_db")

    # EMBED_MODEL takes precedence; fall back to legacy OLLAMA_EMBED_MODEL.
    _ollama_fallback = (
        f"ollama:{os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')}"
    )
    embed_string = os.environ.get("EMBED_MODEL", _ollama_fallback)

    binary_name_override = args.binary_name or os.environ.get("BINARY_NAME")
    try:
        binary_name = await resolve_binary_name(tools, binary_name_override)
        print(f"Analyzing binary: {binary_name}")
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Registry of resumable sessions backing the TUI's /resume command. None when
    # MongoDB is unreachable — /resume then reports nothing to resume.
    session_store = build_session_store(mongodb_uri, mongodb_db)
    if session_store is not None:
        session_store.record_start(session_id, binary_name)

    try:
        embeddings = build_embeddings(embed_string)
        knowledge_tools = build_knowledge_tools(
            mongodb_uri, mongodb_db, embeddings, binary_name
        )
        print(f"Knowledge base ready  [embed: {embed_string}]")
    except Exception as exc:
        print(f"Warning: knowledge base unavailable ({exc})", file=sys.stderr)
        knowledge_tools = []

    # Resolve per-agent models and tool sets from the config. The coordinator
    # gets a restricted, high-level tool set; sub-agents are built from the full
    # tool list so their allowlists are unaffected by that restriction.
    all_tools = filter_withheld_tools(knowledge_tools + tools)
    built_model = resolve_model(agent_config.main_model)
    main_model_spec = resolve_model_spec(agent_config.main_model, agent_config)
    main_tools = build_main_tools(all_tools, agent_config)
    # Shared across the coordinator and sub-agents: one cache for the whole
    # session (same binary, same Mongo collection). None when disabled/unreachable.
    cache_mw = build_mcp_cache_middleware(mongodb_uri, mongodb_db, binary_name)
    # GhidrAssistMCP runs slow tools (e.g. get_code) as async tasks that return a
    # task_id stub; this middleware polls get_task_status so the agent sees the
    # resolved result. None when the server exposes no get_task_status tool.
    async_mw = build_async_task_middleware(tools)
    if async_mw is not None:
        print("Async task resolution enabled (polling get_task_status).")
    # The read-only `research` sub-agent is shared by both graphs: the normal
    # coordinator gets it (read-only deep investigation without applying changes)
    # and the plan-mode graph uses it as its only delegate.
    research_sub = build_research_subagent(
        all_tools,
        agent_config,
        resolve_model,
        cache_middleware=cache_mw,
        async_middleware=async_mw,
    )
    subagents = build_subagents(
        all_tools,
        agent_config,
        resolve_model,
        cache_middleware=cache_mw,
        async_middleware=async_mw,
    )
    subagents.append(research_sub)
    # Plan mode's delegates: the shared read-only `research` agent plus the
    # read-only, config-defined `prototype-auditor` (pulled out of the built
    # config sub-agents by name) so a planning session can also delegate
    # prototype/parameter-count audits. Both are read-only, safe for plan mode.
    plan_mode_subagents = [research_sub]
    proto_auditor_sub = next(
        (s for s in subagents if s.get("name") == "prototype-auditor"), None
    )
    if proto_auditor_sub is not None:
        plan_mode_subagents.append(proto_auditor_sub)
    else:
        print(
            "Warning: 'prototype-auditor' sub-agent not found in config; "
            "plan mode will run without it.",
            file=sys.stderr,
        )
    print(f"Main agent: {main_model_spec}  [{len(main_tools)} tool(s)]")
    for sub_cfg in agent_config.subagents:
        print(
            f"  sub-agent {sub_cfg.name}: "
            f"{resolve_model_spec(sub_cfg.model, agent_config)}"
        )

    recursion_limit = int(os.environ.get("RECURSION_LIMIT", "10000"))
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": recursion_limit,
    }

    output_dir = os.environ.get("AGENT_OUTPUT_DIR", "")
    backend: Any
    if output_dir:
        backend = FilesystemBackend(root_dir=output_dir, virtual_mode=True)
    else:
        backend = StateBackend()

    try:
        with MongoDBSaver.from_conn_string(
            mongodb_uri, db_name=mongodb_db
        ) as checkpointer:
            # SUMMARY_MODEL routes the (cheap, structured) summarization call to a
            # smaller/cheaper model; unset keeps the prior behavior of summarizing
            # with the main model.
            summary_spec = os.environ.get("SUMMARY_MODEL")
            summary_model = resolve_model(summary_spec) if summary_spec else built_model

            # Tune the auto-summarizer create_deep_agent wires internally (lower
            # trigger / cheaper summary model) when any compaction knob is set.
            # Routes the auto summary to SUMMARY_MODEL too, not just /compact.
            if auto_summarization_tuning_enabled():
                install_tuned_summarization(
                    resolve_model(summary_spec) if summary_spec else None
                )

            # Shared by both graphs (normal + plan mode). Built once so the two
            # agents carry identical middleware behavior.
            shared_middleware: list[Any] = [
                # Model-call resilience (outermost): provider fallback wraps
                # transient-error retry of the primary model.
                *build_model_resilience_middleware(resolve_model),
                # Tool calls: validate args (reject bad calls without retry),
                # serve immutable reads from cache, resolve async task stubs
                # (inside the cache so resolved results are what gets cached),
                # then retry transient I/O.
                create_argument_validation_middleware(),
                *([cache_mw] if cache_mw is not None else []),
                *([async_mw] if async_mw is not None else []),
                build_tool_retry_middleware(),
                create_forced_summarization_tool_middleware(summary_model, backend),
            ]

            agent_kwargs: dict[str, Any] = dict(
                model=built_model,
                tools=main_tools,
                system_prompt=SYSTEM_PROMPT + format_agent_memory(agents_md),
                checkpointer=checkpointer,
                middleware=shared_middleware,
                subagents=subagents,
                backend=backend,
            )

            agent = create_deep_agent(**agent_kwargs)

            # Plan-mode graph: read-only coordinator (no mutating tools) whose
            # delegates are the read-only `research` and `prototype-auditor`
            # sub-agents. Shares the checkpointer thread_id and backend with
            # `agent`, so conversation history and the plan file carry over when
            # the human approves.
            plan_agent = create_deep_agent(
                model=built_model,
                tools=build_plan_mode_main_tools(all_tools, agent_config),
                system_prompt=PLAN_MODE_SYSTEM_PROMPT + format_agent_memory(agents_md),
                checkpointer=checkpointer,
                middleware=shared_middleware,
                subagents=plan_mode_subagents,
                backend=backend,
            )

            # Ask-mode graph: read-only question-answering coordinator. Keeps the
            # full coordinator tool set (all knowledge tools + read-only
            # navigation/search — no Ghidra mutations) so it can record durable
            # findings, and delegates investigation to the shared read-only
            # `research` sub-agent plus the read-only, config-defined
            # `vuln-hunter` (pulled out of the built config sub-agents by name)
            # so exploitability questions can be routed to it. Both are
            # read-only, safe for ask mode. Shares the checkpointer/backend; runs
            # on its own ephemeral thread minted by the TUI.
            ask_mode_subagents = [research_sub]
            vuln_hunter_sub = next(
                (s for s in subagents if s.get("name") == "vuln-hunter"), None
            )
            if vuln_hunter_sub is not None:
                ask_mode_subagents.append(vuln_hunter_sub)
            else:
                print(
                    "Warning: 'vuln-hunter' sub-agent not found in config; "
                    "ask mode will run without it.",
                    file=sys.stderr,
                )
            ask_agent = create_deep_agent(
                model=built_model,
                tools=main_tools,
                system_prompt=ASK_MODE_SYSTEM_PROMPT + format_agent_memory(agents_md),
                checkpointer=checkpointer,
                middleware=shared_middleware,
                subagents=ask_mode_subagents,
                backend=backend,
            )

            profile = getattr(built_model, "profile", None) or {}
            ctx_max = profile.get("max_input_tokens") or int(
                os.environ.get("MAX_CONTEXT_TOKENS", "200000")
            )

            app = GhidraAgentApp(
                agent=agent,
                plan_agent=plan_agent,
                ask_agent=ask_agent,
                summary_model=summary_model,
                config=config,
                model=main_model_spec,
                session_id=session_id,
                mcp_ok=True,
                db_ok=True,
                max_context_tokens=ctx_max,
                session_store=session_store,
                binary_name=binary_name,
            )
            await app.run_async()
    except ServerSelectionTimeoutError as e:
        print(
            f"Error: could not connect to MongoDB — {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Session ID: {session_id}")


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
