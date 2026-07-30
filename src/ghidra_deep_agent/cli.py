"""Entry point: wires the Ghidra MCP tools, models, and middleware into the TUI.

``main()`` is the top-level flow; the ``_``-prefixed helpers below own one
startup concern each (MCP connection, tool assembly, storage backend, agent
graphs) so the flow stays readable.
"""

import argparse
import asyncio
import contextlib
import os
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

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
    build_tuned_summarization_middleware,
    create_forced_summarization_tool_middleware,
)
from ghidra_deep_agent.defaults import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_RECURSION_LIMIT,
    env_int,
)
from ghidra_deep_agent.ghidra_transport import get_mcp_config
from ghidra_deep_agent.knowledge import build_knowledge_tools
from ghidra_deep_agent.mcp_cache import build_mcp_cache_middleware
from ghidra_deep_agent.models import build_embeddings, ensure_chat_model
from ghidra_deep_agent.mongo_util import close_mongo_clients
from ghidra_deep_agent.program_resolver import resolve_binary_name
from ghidra_deep_agent.prompt import (
    ASK_MODE_SYSTEM_PROMPT,
    PLAN_MODE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    format_agent_memory,
    format_sandbox_guidance,
)
from ghidra_deep_agent.prototype_tools import build_prototype_tools
from ghidra_deep_agent.resilience import (
    build_model_resilience_middleware,
    build_tool_retry_middleware,
)
from ghidra_deep_agent.sandbox import (
    OPENSHELL_MODE,
    SANDBOX_WORKDIR,
    SUPPORTED_MODES,
    OpenShellSandboxError,
    open_sandbox_backend,
    sandbox_mode,
)
from ghidra_deep_agent.sandbox_sync import SandboxSyncMiddleware
from ghidra_deep_agent.sessions import build_session_store
from ghidra_deep_agent.subagents import (
    RESEARCH_SUBAGENT_NAME,
    build_main_tools,
    build_plan_mode_main_tools,
    build_subagents,
    filter_withheld_tools,
    load_agent_config,
    make_model_resolver,
    resolve_model_spec,
)
from ghidra_deep_agent.switch_tools import build_switch_tools
from ghidra_deep_agent.tui import GhidraAgentApp
from ghidra_deep_agent.validation import create_argument_validation_middleware


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ghidra deep agent")
    parser.add_argument(
        "--session-id", default=None, help="Resume a previous session by ID"
    )
    parser.add_argument(
        "--binary-name",
        default=None,
        help="Binary name to use for knowledge isolation (overrides auto-detection)",
    )
    return parser.parse_args()


def _load_agents_md() -> str:
    """Read the optional AGENTS_MD memory file; warn and continue if unreadable."""
    agents_md_path = os.environ.get("AGENTS_MD", "")
    if not agents_md_path:
        return ""
    resolved = Path(agents_md_path).expanduser()
    try:
        agents_md = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        print(
            f"Warning: could not read AGENTS_MD file {resolved} ({exc})",
            file=sys.stderr,
        )
        return ""
    print(f"AGENTS.md memory loaded [{resolved}]")
    return agents_md


async def _connect_mcp(mcp_config: dict[str, Any]) -> list[Any]:
    """Connect to the Ghidra MCP server and return its tools. Exits on failure."""
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
        tools: list[Any] = await client.get_tools()
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
    return tools


def _build_tools(
    mcp_tools: list[Any],
    mongodb_uri: str,
    mongodb_db: str,
    embed_string: str,
    binary_name: str,
) -> tuple[list[Any], bool]:
    """Merge the MCP tools with the locally-defined ones.

    Returns ``(tools, knowledge_ok)``. The full tool set is what per-agent
    allowlists are then drawn from — the coordinator's restricted tool set must
    not narrow what sub-agents can use. ``knowledge_ok`` feeds the TUI's `db`
    health indicator, which otherwise reports healthy unconditionally.
    """
    knowledge_ok = True
    try:
        embeddings = build_embeddings(embed_string)
        knowledge_tools = build_knowledge_tools(
            mongodb_uri, mongodb_db, embeddings, binary_name
        )
        print(f"Knowledge base ready  [embed: {embed_string}]")
    except Exception as exc:
        print(f"Warning: knowledge base unavailable ({exc})", file=sys.stderr)
        knowledge_tools = []
        knowledge_ok = False

    # Local `recover_prototypes` tool: drives a Ghidra-side prototype-recovery
    # script through the MCP `scripts` executor. Omitted (with a warning) when the
    # server's `scripts` tool is disabled.
    prototype_tools = build_prototype_tools(mcp_tools)

    # Local jump-table tools: `find_unrecovered_switches` (read-only detection)
    # and `apply_switch_override` (writes the decompiler jump-table override).
    # Both drive Ghidra-side scripts through the MCP `scripts` executor; omitted
    # (with a warning) when the server's `scripts` tool is disabled.
    switch_tools = build_switch_tools(mcp_tools)

    return (
        filter_withheld_tools(
            knowledge_tools + prototype_tools + switch_tools + mcp_tools
        ),
        knowledge_ok,
    )


def _read_only_delegates(
    subagents: list[Any],
) -> tuple[list[Any], list[Any]]:
    """Pick the sub-agents plan mode and ask mode may delegate to.

    Returns ``(plan_mode_subagents, ask_mode_subagents)``. Both modes are
    read-only, so only read-only sub-agents qualify — the other config
    sub-agents all mutate Ghidra (`prototype-fixer` included).

    `research` is a config `[[subagents]]` entry (`read_only = true`) shared by
    every graph: the normal coordinator gets it in its full set, and plan mode
    uses it as its ONLY delegate. Ask mode additionally gets the read-only
    `vuln-hunter` so exploitability questions can be routed to it.
    """
    research_sub = next(
        (s for s in subagents if s.get("name") == RESEARCH_SUBAGENT_NAME), None
    )
    if research_sub is None:
        raise ValueError(
            f"A read-only '{RESEARCH_SUBAGENT_NAME}' sub-agent is required "
            "(plan mode and ask mode depend on it); add it to the agent config "
            "with read_only = true."
        )

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
    return [research_sub], ask_mode_subagents


def _validate_sandbox_mode() -> None:
    """Reject an unknown SANDBOX value before anything expensive is created."""
    mode = sandbox_mode()
    if mode and mode not in SUPPORTED_MODES:
        print(
            f"Error: unsupported SANDBOX={mode!r}; supported values: "
            f"{', '.join(SUPPORTED_MODES)}",
            file=sys.stderr,
        )
        sys.exit(1)


class Storage(NamedTuple):
    """Where agent files live, and what that implies for prompts/middleware."""

    backend: Any
    # Syncs AGENT_OUTPUT_DIR to/from the sandbox each turn; None when not sandboxed.
    sync_middleware: Any
    # Appended to every agent prompt when sandboxed, so the model knows it has a
    # shell and where durable files belong. Empty otherwise.
    prompt_guidance: str


async def _open_backend(stack: contextlib.AsyncExitStack) -> Storage:
    """Open the agents' filesystem backend.

    An OpenShell sandbox is entered on ``stack``, so the caller's
    ``stack.aclose()`` tears it down after the TUI exits or crashes and it is
    never leaked.
    """
    mode = sandbox_mode()
    output_dir = os.environ.get("AGENT_OUTPUT_DIR", "")

    if mode != OPENSHELL_MODE:
        if output_dir:
            return Storage(
                FilesystemBackend(root_dir=output_dir, virtual_mode=True), None, ""
            )
        return Storage(StateBackend(), None, "")

    # Appended to every agent prompt when sandboxed, so the model knows it has a
    # shell and where to keep durable files.
    sandbox_guidance = format_sandbox_guidance(SANDBOX_WORKDIR, synced=bool(output_dir))
    try:
        backend = await stack.enter_async_context(open_sandbox_backend())
    except OpenShellSandboxError as exc:
        print(f"Failed to create OpenShell sandbox: {exc}", file=sys.stderr)
        print(
            "Check OPENSHELL_GATEWAY / OPENSHELL_GATEWAY_ENDPOINT and that "
            "the 'openshell' CLI is authenticated (see ~/.config/openshell/).",
            file=sys.stderr,
        )
        sys.exit(1)

    if not output_dir:
        print(
            "Sandbox: files live only inside the sandbox this session "
            "(set AGENT_OUTPUT_DIR to persist them locally)."
        )
        return Storage(backend, None, sandbox_guidance)

    # Files live in the sandbox; the middleware makes AGENT_OUTPUT_DIR the
    # durable local mirror, synced in before and out after a turn.
    return Storage(
        backend, SandboxSyncMiddleware(backend, Path(output_dir)), sandbox_guidance
    )


class MongoConfig(NamedTuple):
    """Connection details every Mongo-backed subsystem shares."""

    uri: str
    db: str
    embed_model: str


def _storage_config() -> MongoConfig:
    """Read the MongoDB/embedding settings from the environment."""
    # EMBED_MODEL takes precedence; fall back to legacy OLLAMA_EMBED_MODEL.
    ollama_fallback = (
        f"ollama:{os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')}"
    )
    return MongoConfig(
        uri=os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
        db=os.environ.get("MONGODB_DB", "checkpointing_db"),
        embed_model=os.environ.get("EMBED_MODEL", ollama_fallback),
    )


def _build_shared_middleware(
    *,
    storage: Storage,
    resolve_model: Any,
    cache_mw: Any,
    async_mw: Any,
    summary_model: Any,
    built_model: Any,
    summary_override: Any,
) -> list[Any]:
    """Middleware shared by all three graphs, in wrapping order.

    Built once so the normal, plan-mode, and ask-mode agents cannot drift apart
    in behaviour — they are meant to differ only in prompt, tools, and delegates.
    """
    return [
        # Sandbox file sync (first): its before_agent seeds the sandbox from
        # AGENT_OUTPUT_DIR and its after_agent syncs changed files back, so these
        # hooks bracket every other middleware. Absent when not sandboxed.
        *([storage.sync_middleware] if storage.sync_middleware else []),
        # Model-call resilience (outermost): provider fallback wraps
        # transient-error retry of the primary model.
        *build_model_resilience_middleware(resolve_model),
        # Tool calls: validate args (reject bad calls without retry), serve
        # immutable reads from cache, resolve async task stubs (inside the cache
        # so resolved results are what gets cached), then retry transient I/O.
        create_argument_validation_middleware(),
        *([cache_mw] if cache_mw is not None else []),
        *([async_mw] if async_mw is not None else []),
        build_tool_retry_middleware(),
        create_forced_summarization_tool_middleware(summary_model, storage.backend),
        # Auto-summarizer for the coordinator: stock thresholds (COMPACT_MAIN_*
        # overrides), summary routed per SUMMARY_MODEL. Replaces deepagents'
        # stock SummarizationMiddleware by name (0.7 replace-by-name).
        build_tuned_summarization_middleware(
            built_model, storage.backend, summary_model=summary_override, scope="main"
        ),
    ]


class Graphs(NamedTuple):
    """The three coordinator graphs the TUI switches between."""

    main: Any
    # Read-only planner: no mutating tools, delegates only to read-only sub-agents.
    plan: Any
    # Read-only question-answerer: full coordinator tool set minus Ghidra writes.
    ask: Any


def _build_graphs(
    *,
    built_model: Any,
    agents_md: str,
    storage: Storage,
    checkpointer: Any,
    middleware: list[Any],
    app_name: str,
    main_tools: Sequence[Any],
    plan_tools: Sequence[Any],
    subagents: Sequence[Any],
    plan_mode_subagents: Sequence[Any],
    ask_mode_subagents: Sequence[Any],
) -> Graphs:
    """Build the three graphs, which differ only in prompt, tools, and delegates.

    Everything else is held identical on purpose: they share one checkpointer
    thread and backend, so conversation history and the plan file carry over when
    the human approves a plan.
    """

    def build(
        system_prompt: str, tools: Sequence[Any], graph_subagents: Sequence[Any]
    ) -> Any:
        return create_deep_agent(
            model=built_model,
            tools=list(tools),
            system_prompt=system_prompt
            + format_agent_memory(agents_md)
            + storage.prompt_guidance,
            checkpointer=checkpointer,
            middleware=middleware,
            subagents=list(graph_subagents),
            backend=storage.backend,
            name=app_name,
        )

    return Graphs(
        main=build(SYSTEM_PROMPT, main_tools, subagents),
        plan=build(PLAN_MODE_SYSTEM_PROMPT, plan_tools, plan_mode_subagents),
        ask=build(ASK_MODE_SYSTEM_PROMPT, main_tools, ask_mode_subagents),
    )


async def main() -> None:
    load_dotenv()

    args = _parse_args()
    session_id = args.session_id or str(uuid.uuid4())

    mcp_config = get_mcp_config()
    # Fail fast on a bad config before connecting to anything.
    try:
        agent_config = load_agent_config()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    resolve_model = make_model_resolver(agent_config.default_model)

    agents_md = _load_agents_md()
    tools = await _connect_mcp(mcp_config)

    storage_cfg = _storage_config()
    mongodb_uri, mongodb_db = storage_cfg.uri, storage_cfg.db

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

    all_tools, knowledge_ok = _build_tools(
        tools, mongodb_uri, mongodb_db, storage_cfg.embed_model, binary_name
    )
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
    print(f"Main agent: {main_model_spec}  [{len(main_tools)} tool(s)]")
    for sub_cfg in agent_config.subagents:
        print(
            f"  sub-agent {sub_cfg.name}: "
            f"{resolve_model_spec(sub_cfg.model, agent_config)}"
        )

    recursion_limit = env_int("RECURSION_LIMIT", DEFAULT_RECURSION_LIMIT)
    # Name each top-level graph so LangSmith traces show the app instead of the
    # langgraph library default ("LangGraph").
    app_name = os.environ.get("APP_NAME", "ghidra-deep-agent")
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": recursion_limit,
    }

    _validate_sandbox_mode()

    stack = contextlib.AsyncExitStack()
    try:
        storage = await _open_backend(stack)

        with MongoDBSaver.from_conn_string(
            mongodb_uri, db_name=mongodb_db
        ) as checkpointer:
            # SUMMARY_MODEL routes the (cheap, structured) summarization call to a
            # smaller/cheaper model; unset keeps the prior behavior of summarizing
            # with the main model.
            summary_spec = os.environ.get("SUMMARY_MODEL")
            # None when unset, which the tuned-summarization builder reads as
            # "use the agent's own model".
            summary_override = resolve_model(summary_spec) if summary_spec else None
            # Resolved eagerly: the TUI calls `.ainvoke` on this to summarize prior
            # context when entering plan/ask mode, and `build_model` hands back a
            # bare string for any provider it doesn't special-case.
            summary_model = ensure_chat_model(summary_override or built_model)

            # Tuned auto-summarizers ride along as replace-by-name middleware:
            # sub-agents compact aggressively by default (they never reached
            # deepagents' 170k no-profile trigger); the main agent keeps stock
            # thresholds. COMPACT_* / COMPACT_MAIN_* env knobs override either
            # scope, and SUMMARY_MODEL routes the auto summary too, not just
            # /compact. Built here because they offload evicted history to the
            # session backend.
            subagents = build_subagents(
                all_tools,
                agent_config,
                resolve_model,
                storage.backend,
                cache_middleware=cache_mw,
                async_middleware=async_mw,
                summary_model=summary_override,
            )
            plan_mode_subagents, ask_mode_subagents = _read_only_delegates(subagents)

            shared_middleware = _build_shared_middleware(
                storage=storage,
                resolve_model=resolve_model,
                cache_mw=cache_mw,
                async_mw=async_mw,
                summary_model=summary_model,
                built_model=built_model,
                summary_override=summary_override,
            )

            graphs = _build_graphs(
                built_model=built_model,
                agents_md=agents_md,
                storage=storage,
                checkpointer=checkpointer,
                middleware=shared_middleware,
                app_name=app_name,
                main_tools=main_tools,
                plan_tools=build_plan_mode_main_tools(all_tools, agent_config),
                subagents=subagents,
                plan_mode_subagents=plan_mode_subagents,
                ask_mode_subagents=ask_mode_subagents,
            )

            # Probe the *resolved* model: `build_model` returns a bare string for
            # providers it doesn't special-case, and a string has no `.profile`,
            # which silently pinned the gauge to the fallback for those models.
            profile = getattr(ensure_chat_model(built_model), "profile", None) or {}
            ctx_max = profile.get("max_input_tokens") or env_int(
                "MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS
            )

            app = GhidraAgentApp(
                agent=graphs.main,
                plan_agent=graphs.plan,
                ask_agent=graphs.ask,
                summary_model=summary_model,
                config=config,
                model=main_model_spec,
                session_id=session_id,
                # `_connect_mcp` exits on a failed connection, so reaching here means
                # the server answered — but it can still answer with zero tools.
                mcp_ok=bool(tools),
                db_ok=knowledge_ok,
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
    finally:
        # Tears down the OpenShell sandbox if one was created; a no-op otherwise.
        await stack.aclose()
        # The knowledge base, session registry, and read cache share one client
        # per URI (mongo_util); close it so the connection pool doesn't outlive
        # the process's useful life. The checkpointer manages its own.
        close_mongo_clients()

    print(f"Session ID: {session_id}")


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
