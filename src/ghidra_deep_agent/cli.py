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
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.middleware.filesystem import FilesystemPermission
from deepagents.middleware.summarization import SummarizationToolMiddleware
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo.errors import ServerSelectionTimeoutError

from ghidra_deep_agent.agent_config import (
    build_worker,
    filter_withheld_tools,
    load_agent_config,
    make_model_resolver,
    read_only_tools,
    resolve_model_spec,
)
from ghidra_deep_agent.async_tasks import build_async_task_middleware
from ghidra_deep_agent.auto_mode import build_auto_mode_middleware
from ghidra_deep_agent.compaction import (
    build_tuned_summarization_middleware,
    create_manual_compaction_engine,
)
from ghidra_deep_agent.context_pruning import build_context_pruning_middleware
from ghidra_deep_agent.defaults import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_RECURSION_LIMIT,
    config_path,
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
from ghidra_deep_agent.script_file_tools import build_script_file_tools
from ghidra_deep_agent.sessions import build_session_store
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
    # script through the MCP `create_script`/`run_script` tools. Omitted (with a
    # warning) when those tools are disabled server-side.
    prototype_tools = build_prototype_tools(mcp_tools)

    # Local jump-table tools: `find_unrecovered_switches` (read-only detection)
    # and `apply_switch_override` (writes the decompiler jump-table override).
    # Both drive Ghidra-side scripts through the MCP `create_script`/`run_script`
    # tools; omitted (with a warning) when those are disabled server-side.
    switch_tools = build_switch_tools(mcp_tools)

    return (
        filter_withheld_tools(
            knowledge_tools + prototype_tools + switch_tools + mcp_tools
        ),
        knowledge_ok,
    )


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


# Where skills are mounted in the agents' virtual filesystem, and the source
# directories offered per graph. The repair skills (each needs write tools) are
# withheld from the read-only plan/ask graphs.
_SKILLS_ROUTE = "/skills/"
_ANALYSIS_SKILLS = "/skills/analysis/"
_REPAIR_SKILLS = "/skills/repair/"
MAIN_SKILL_SOURCES = [_ANALYSIS_SKILLS, _REPAIR_SKILLS]
READ_ONLY_SKILL_SOURCES = [_ANALYSIS_SKILLS]
# Deny the agents writing into the read-only skills mount (reads stay allowed).
SKILLS_PERMISSIONS = [
    FilesystemPermission(
        operations=["write"], paths=[f"{_SKILLS_ROUTE}**"], mode="deny"
    )
]


def _skills_routes() -> tuple[dict[str, Any], list[str]]:
    """Resolve the skills directory into a CompositeBackend route + skill names.

    Returns ``({"/skills/": FilesystemBackend(...)}, names)``, or ``({}, [])``
    with a warning if the directory is missing — the agents still run, just
    without any skills. ``names`` are the discovered ``<group>/<skill>`` dirs,
    printed at startup so a broken skills dir is visible early.
    """
    skills_dir = config_path("SKILLS_DIR", "skills")
    if not skills_dir.is_dir():
        print(
            f"Warning: skills directory not found at {skills_dir}; agents will run "
            "without skills. Set SKILLS_DIR to override.",
            file=sys.stderr,
        )
        return {}, []
    names = sorted(
        f"{md.parent.parent.name}/{md.parent.name}"
        for md in skills_dir.glob("*/*/SKILL.md")
    )
    route = FilesystemBackend(root_dir=str(skills_dir), virtual_mode=True)
    return {_SKILLS_ROUTE: route}, names


class Storage(NamedTuple):
    """Where agent files live, and what that implies for prompts/middleware."""

    backend: Any
    # Syncs AGENT_OUTPUT_DIR to/from the sandbox each turn; None when not sandboxed.
    sync_middleware: SandboxSyncMiddleware | None
    # Appended to every agent prompt when sandboxed, so the model knows it has a
    # shell and where durable files belong. Empty otherwise.
    prompt_guidance: str
    # The <group>/<skill> directories found under the skills mount (for display).
    skill_names: list[str]
    # The raw OpenShell backend when sandboxed, else None. Tools that push files
    # into the sandbox use it directly (the agent backend may be a composite).
    sandbox: SandboxBackendProtocol | None = None


async def _open_backend(stack: contextlib.AsyncExitStack) -> Storage:
    """Open the agents' filesystem backend.

    The agents' durable/working files live in the chosen backend (state,
    local dir, or sandbox); the read-only ``skills/`` tree is mounted alongside
    it via a ``CompositeBackend`` route so the skills middleware can read each
    ``SKILL.md`` through the same backend. ``SandboxSyncMiddleware`` keeps the raw
    sandbox backend, since it syncs the host mirror, not the skills mount.

    An OpenShell sandbox is entered on ``stack``, so the caller's
    ``stack.aclose()`` tears it down after the TUI exits or crashes and it is
    never leaked.
    """
    mode = sandbox_mode()
    output_dir = os.environ.get("AGENT_OUTPUT_DIR", "")
    routes, skill_names = _skills_routes()

    def wrap(backend: Any) -> Any:
        """Mount the skills route over ``backend`` (a no-op without skills)."""
        return CompositeBackend(backend, routes) if routes else backend

    if mode != OPENSHELL_MODE:
        if output_dir:
            raw: Any = FilesystemBackend(root_dir=output_dir, virtual_mode=True)
        else:
            raw = StateBackend()
        return Storage(wrap(raw), None, "", skill_names)

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
        return Storage(
            wrap(backend), None, sandbox_guidance, skill_names, sandbox=backend
        )

    # Files live in the sandbox; the middleware makes AGENT_OUTPUT_DIR the
    # durable local mirror, synced in before and out after a turn. Sync uses the
    # raw sandbox backend (the host mirror is not the skills mount).
    sync = SandboxSyncMiddleware(backend, Path(output_dir))
    # A last sync-back on the way out. Registered after the sandbox was entered,
    # so the stack's LIFO unwind runs it *before* teardown destroys the sandbox.
    # It rescues work from a turn that never reached the per-turn sync-back —
    # e.g. one cancelled, or cut short by quitting mid-run.
    stack.push_async_callback(_final_sync_back, sync)
    return Storage(wrap(backend), sync, sandbox_guidance, skill_names, sandbox=backend)


async def _final_sync_back(sync: SandboxSyncMiddleware) -> None:
    """Pull the sandbox's files home one last time before it is torn down."""
    try:
        await sync.sync_back()
    except Exception as exc:  # shutdown must continue to teardown regardless
        print(f"Warning: final sandbox sync-back failed: {exc}", file=sys.stderr)


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
    compaction_engine: Any,
    built_model: Any,
    summary_override: Any,
    prune_mw: Any = None,
    auto_mode_mw: Any = None,
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
        # The coordinator holds the full tool set and may apply small changes
        # itself; it delegates verbose/parallel work to the worker.
        create_argument_validation_middleware(),
        # Jev Auto Mode (AUTO_MODE=1): blocks risky calls to hard-to-undo tools.
        *([auto_mode_mw] if auto_mode_mw is not None else []),
        *([cache_mw] if cache_mw is not None else []),
        *([async_mw] if async_mw is not None else []),
        build_tool_retry_middleware(),
        # Jev relevance pruning (when TYPESAFE_API_KEY is set): blanks stale tool
        # results on each request only. Custom middleware lands inside the
        # summarizer's slot, so the summarizer still sees and offloads the raw
        # history; only the model sees the pruned copy.
        *([prune_mw] if prune_mw is not None else []),
        # The compact_conversation tool for the agent's own proactive use, on
        # the stock ~50% eligibility gate (stops premature self-compaction).
        # User-driven /compact no longer goes through it — the TUI drives the
        # same engine directly via compact_out_of_band.
        SummarizationToolMiddleware(compaction_engine),
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
    # Read-only planner: no mutating tools, read-only worker, analysis skills only.
    plan: Any
    # Read-only question-answerer: same tool set as plan mode (no Ghidra writes
    # and no knowledge-base writes).
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
    worker: Any,
    read_only_worker: Any,
) -> Graphs:
    """Build the three graphs, which differ only in prompt, tools, worker, skills.

    Everything else is held identical on purpose: they share one checkpointer
    thread and backend, so conversation history and the plan file carry over when
    the human approves a plan. The normal graph offers every skill and the
    write-capable worker; the read-only graphs offer only the analysis skills and
    the read-only worker.
    """

    def build(
        system_prompt: str,
        tools: Sequence[Any],
        graph_worker: Any,
        skills: Sequence[str],
    ) -> Any:
        return create_deep_agent(
            model=built_model,
            tools=list(tools),
            system_prompt=system_prompt
            + format_agent_memory(agents_md)
            + storage.prompt_guidance,
            checkpointer=checkpointer,
            middleware=middleware,
            subagents=[graph_worker],
            skills=list(skills),
            permissions=SKILLS_PERMISSIONS,
            backend=storage.backend,
            name=app_name,
        )

    return Graphs(
        main=build(SYSTEM_PROMPT, main_tools, worker, MAIN_SKILL_SOURCES),
        plan=build(
            PLAN_MODE_SYSTEM_PROMPT,
            plan_tools,
            read_only_worker,
            READ_ONLY_SKILL_SOURCES,
        ),
        # Ask mode gets `plan_tools`, not `main_tools`: it is a read-only mode, so
        # the knowledge-base write tools are withheld structurally rather than
        # left for the prompt to talk it out of using.
        ask=build(
            ASK_MODE_SYSTEM_PROMPT,
            plan_tools,
            read_only_worker,
            READ_ONLY_SKILL_SOURCES,
        ),
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
    resolve_model = make_model_resolver(
        agent_config.default_model,
        agent_config.default_max_tokens,
        agent_config.default_reasoning_effort,
    )

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
    built_model = resolve_model(
        agent_config.main_model,
        agent_config.main_max_tokens,
        agent_config.main_reasoning_effort,
    )
    main_model_spec = resolve_model_spec(agent_config.main_model, agent_config)
    # The coordinator holds the full (non-withheld) tool set; the read-only graphs
    # drop every mutating tool.
    main_tools = list(all_tools)
    plan_tools = read_only_tools(all_tools)
    # Shared across the coordinator and sub-agents: one cache for the whole
    # session (same binary, same Mongo collection). None when disabled/unreachable.
    cache_mw = build_mcp_cache_middleware(mongodb_uri, mongodb_db, binary_name)
    # GhidrAssistMCP runs slow tools (e.g. get_pseudocode) as async tasks that return a
    # task_id stub; this middleware polls get_task_status so the agent sees the
    # resolved result. None when the server exposes no get_task_status tool.
    async_mw = build_async_task_middleware(tools)
    if async_mw is not None:
        print("Async task resolution enabled (polling get_task_status).")
    # Jev context pruning, shared by the coordinator and every sub-agent so a
    # verdict on a tool result holds across graphs. None without a TypeSafe key.
    prune_mw = build_context_pruning_middleware(
        mongodb_uri, mongodb_db, session_id, binary_name
    )
    if prune_mw is not None:
        print("Jev context pruning enabled (savings log: MongoDB jev_prune_log).")
    # Jev Auto Mode, shared by the coordinator and every sub-agent. Opt-in.
    auto_mode_mw = build_auto_mode_middleware()
    if auto_mode_mw is not None:
        print(f"Jev Auto Mode enabled on: {', '.join(sorted(auto_mode_mw.guarded))}")

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
        # Built here, not in `_build_tools`: it needs the sandbox, which only
        # exists once the backend is open. It writes (it runs an arbitrary
        # script), so it joins the full tool set but never the read-only one.
        script_file_tools = build_script_file_tools(tools, storage.sandbox)
        all_tools = [*all_tools, *script_file_tools]
        main_tools = [*main_tools, *script_file_tools]
        print(f"Main agent: {main_model_spec}  [{len(main_tools)} tool(s)]")
        print(
            f"  worker: {resolve_model_spec(agent_config.worker_model, agent_config)}"
            f"  [{len(main_tools)} tool(s)]"
        )
        if storage.skill_names:
            print(
                f"  skills [{len(storage.skill_names)}]: "
                f"{', '.join(storage.skill_names)}"
            )

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
            # One summarization engine shared by the agent-facing
            # compact_conversation tool and the TUI's out-of-band /compact.
            # Built inside the backend context: it offloads evicted history
            # to the session backend.
            compaction_engine = create_manual_compaction_engine(
                summary_model, storage.backend
            )

            # The single worker, built twice: write-capable for the normal graph,
            # and a second read-only build (full MUTATION_TOOLS dropped) for the
            # plan/ask graphs. Building a separate read-only worker — rather than
            # trusting the prompt — is what makes those graphs read-only by
            # construction. The tuned sub-agent summarizer rides along inside each,
            # compacting aggressively; built here because it offloads evicted
            # history to the session backend.
            worker_kwargs: dict[str, Any] = dict(
                cache_middleware=cache_mw,
                async_middleware=async_mw,
                pruning_middleware=prune_mw,
                auto_mode_middleware=auto_mode_mw,
                summary_model=summary_override,
            )
            worker = build_worker(
                all_tools,
                agent_config,
                resolve_model,
                storage.backend,
                read_only=False,
                skills=MAIN_SKILL_SOURCES,
                **worker_kwargs,
            )
            read_only_worker = build_worker(
                all_tools,
                agent_config,
                resolve_model,
                storage.backend,
                read_only=True,
                skills=READ_ONLY_SKILL_SOURCES,
                **worker_kwargs,
            )

            shared_middleware = _build_shared_middleware(
                storage=storage,
                resolve_model=resolve_model,
                cache_mw=cache_mw,
                async_mw=async_mw,
                compaction_engine=compaction_engine,
                built_model=built_model,
                summary_override=summary_override,
                prune_mw=prune_mw,
                auto_mode_mw=auto_mode_mw,
            )

            graphs = _build_graphs(
                built_model=built_model,
                agents_md=agents_md,
                storage=storage,
                checkpointer=checkpointer,
                middleware=shared_middleware,
                app_name=app_name,
                main_tools=main_tools,
                plan_tools=plan_tools,
                worker=worker,
                read_only_worker=read_only_worker,
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
                compaction_engine=compaction_engine,
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
                sandbox_sync=storage.sync_middleware,
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
