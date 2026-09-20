"""Building agents, split into what is shared and what is per program.

The TUI runs one agent on one binary; the HTTP server runs many, one per
binary, in one process against one Ghidra. Both build their agents here so
they cannot drift apart:

- :class:`SharedRuntime` (:func:`open_shared_runtime`) holds what every agent
  in the process shares: the Ghidra MCP config, the agent config and model
  resolver, the MongoDB checkpointer and session registry, the summary model.
- :class:`Engine` (:func:`build_engine`) is one agent bound to one program: its
  own MCP client pinned to that program, knowledge tools and read cache scoped
  to it, its filesystem backend (and sandbox, when enabled), and the three
  coordinator graphs (normal, plan, ask).

Startup failures raise :class:`StartupError` instead of exiting the process:
the same code runs under a long-lived server, where a bad program name must
fail one request, not the service.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.summarization import SummarizationToolMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import (
    MCPToolCallRequest,
    ToolCallInterceptor,
)
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo.errors import ServerSelectionTimeoutError

from ghidra_deep_agent.async_tasks import build_async_task_middleware
from ghidra_deep_agent.compaction import (
    build_tuned_summarization_middleware,
    create_manual_compaction_engine,
)
from ghidra_deep_agent.context_pruning import build_context_pruning_middleware
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
from ghidra_deep_agent.pinning import (
    PinMismatchError,
    pin_program,
    verify_pinned,
)
from ghidra_deep_agent.program_resolver import ProgramRef
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
from ghidra_deep_agent.sessions import SessionStore, build_session_store
from ghidra_deep_agent.subagents import (
    ALL_WRITE_ACTIONS,
    DEFAULT_WRITE_POLICY,
    READ_ONLY_WRITE_POLICY,
    RESEARCH_SUBAGENT_NAME,
    AgentConfig,
    ModelResolver,
    build_main_tools,
    build_plan_mode_main_tools,
    build_subagents,
    filter_withheld_tools,
    load_agent_config,
    make_model_resolver,
    resolve_model_spec,
)
from ghidra_deep_agent.switch_tools import build_switch_tools
from ghidra_deep_agent.validation import create_argument_validation_middleware


class StartupError(RuntimeError):
    """A fatal configuration or connection problem, phrased for the user."""


# --- configuration ------------------------------------------------------------


class MongoConfig(NamedTuple):
    """Connection details every Mongo-backed subsystem shares."""

    uri: str
    db: str
    embed_model: str


def storage_config() -> MongoConfig:
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


def validate_sandbox_mode() -> None:
    """Reject an unknown SANDBOX value before anything expensive is created."""
    mode = sandbox_mode()
    if mode and mode not in SUPPORTED_MODES:
        raise StartupError(
            f"unsupported SANDBOX={mode!r}; supported values: "
            f"{', '.join(SUPPORTED_MODES)}"
        )


def load_agents_md() -> str:
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


# --- MCP ----------------------------------------------------------------------


async def handle_mcp_errors(request: MCPToolCallRequest, handler: Any) -> Any:
    """Turn a failed MCP call into a string result the model can act on."""
    try:
        return await handler(request)
    except Exception as exc:
        return f"Tool '{request.name}' failed: {exc}"


async def connect_mcp(
    mcp_config: dict[str, Any], *, interceptors: Sequence[ToolCallInterceptor]
) -> list[Any]:
    """Connect to the Ghidra MCP server and return its tools.

    Each call builds its own client: the client is stateless (a fresh session
    per tool call), so the cost is one ``list_tools`` round trip, and the
    interceptors — which is where a per-agent program pin lives — are bound
    per client.
    """
    try:
        client = MultiServerMCPClient(mcp_config, tool_interceptors=list(interceptors))
        tools: list[Any] = await client.get_tools()
    except Exception as exc:
        raise StartupError(
            f"failed to connect to Ghidra MCP server: {exc}\n"
            "Ensure Ghidra is running with the GhidrAssistMCP plugin enabled "
            "(MCP server on) and a program open, then set GHIDRA_MCP_TRANSPORT / "
            "GHIDRA_MCP_URL as needed."
        ) from exc

    # MCP server errors arrive as isError=True results, which langchain_mcp_adapters
    # converts to ToolException. Without handle_tool_error=True, ToolException bypasses
    # LangGraph's ToolNode default handler (which only catches ToolInvocationError) and
    # propagates all the way up through sub-agents to the caller.
    for tool in tools:
        tool.handle_tool_error = True
    return tools


# --- storage ------------------------------------------------------------------


class Storage(NamedTuple):
    """Where agent files live, and what that implies for prompts/middleware."""

    backend: Any
    # Syncs the output dir to/from the sandbox each turn; None when not sandboxed.
    sync_middleware: Any
    # Appended to every agent prompt when sandboxed, so the model knows it has a
    # shell and where durable files belong. Empty otherwise.
    prompt_guidance: str


async def open_storage(stack: contextlib.AsyncExitStack, *, output_dir: str) -> Storage:
    """Open an agent's filesystem backend.

    An OpenShell sandbox is entered on ``stack``, so the owner's
    ``stack.aclose()`` tears it down when the agent goes away and it is never
    leaked. Called once per engine: with several agents in one process each
    gets its own sandbox and its own local mirror.
    """
    mode = sandbox_mode()

    if mode != OPENSHELL_MODE:
        if output_dir:
            return Storage(
                FilesystemBackend(root_dir=output_dir, virtual_mode=True), None, ""
            )
        return Storage(StateBackend(), None, "")

    sandbox_guidance = format_sandbox_guidance(SANDBOX_WORKDIR, synced=bool(output_dir))
    try:
        backend = await stack.enter_async_context(open_sandbox_backend())
    except OpenShellSandboxError as exc:
        raise StartupError(
            f"failed to create OpenShell sandbox: {exc}\n"
            "Check OPENSHELL_GATEWAY / OPENSHELL_GATEWAY_ENDPOINT and that "
            "the 'openshell' CLI is authenticated (see ~/.config/openshell/)."
        ) from exc

    if not output_dir:
        print(
            "Sandbox: files live only inside the sandbox this session "
            "(set AGENT_OUTPUT_DIR to persist them locally)."
        )
        return Storage(backend, None, sandbox_guidance)

    # Files live in the sandbox; the middleware makes the output dir the
    # durable local mirror, synced in before and out after a turn.
    return Storage(
        backend, SandboxSyncMiddleware(backend, Path(output_dir)), sandbox_guidance
    )


# --- shared runtime -----------------------------------------------------------


@dataclass
class SharedRuntime:
    """Everything agents in this process share. See :func:`open_shared_runtime`."""

    mcp_config: dict[str, Any]
    agent_config: AgentConfig
    resolve_model: ModelResolver
    agents_md: str
    mongo: MongoConfig
    checkpointer: Any
    session_store: SessionStore | None
    built_model: Any
    main_model_spec: str
    # SUMMARY_MODEL resolved, or None to summarize with the agent's own model.
    summary_override: Any
    # Resolved eagerly: callers `.ainvoke` it to summarize prior context.
    summary_model: BaseChatModel
    recursion_limit: int
    app_name: str
    max_context_tokens: int
    # MCP tools with NO program pin, for the process's own housekeeping:
    # listing and opening programs. Never handed to an agent.
    probe_tools: list[Any]
    _stack: contextlib.ExitStack

    def config_for(self, thread_id: str) -> dict[str, Any]:
        """The graph config for one conversation thread."""
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.recursion_limit,
        }

    def close(self) -> None:
        """Close the checkpointer and the shared Mongo clients."""
        self._stack.close()
        # The knowledge base, session registry, and read cache share one client
        # per URI (mongo_util); close it so the connection pool doesn't outlive
        # the process's useful life. The checkpointer manages its own.
        close_mongo_clients()


async def open_shared_runtime() -> SharedRuntime:
    """Load config, connect to Ghidra and MongoDB, and resolve the models."""
    mcp_config = get_mcp_config()
    # Fail fast on a bad config before connecting to anything.
    try:
        agent_config = load_agent_config()
    except ValueError as exc:
        raise StartupError(str(exc)) from exc
    validate_sandbox_mode()
    resolve_model = make_model_resolver(
        agent_config.default_model, agent_config.default_max_tokens
    )
    agents_md = load_agents_md()

    transport_desc = mcp_config["ghidra"].get("transport", "http")
    url = mcp_config["ghidra"].get("url", "")
    print(f"Connecting to Ghidra MCP server [{transport_desc}: {url}]...")
    probe_tools = await connect_mcp(mcp_config, interceptors=[handle_mcp_errors])
    if not probe_tools:
        print("Warning: no tools loaded from Ghidra MCP server.", file=sys.stderr)
    else:
        names = ", ".join(t.name for t in probe_tools)
        print(f"Loaded {len(probe_tools)} Ghidra tool(s): {names}")

    mongo = storage_config()
    # Registry of resumable sessions. None when MongoDB is unreachable — resume
    # then reports nothing to resume.
    session_store = build_session_store(mongo.uri, mongo.db)

    built_model = resolve_model(agent_config.main_model, agent_config.main_max_tokens)
    main_model_spec = resolve_model_spec(agent_config.main_model, agent_config)
    print(f"Main agent: {main_model_spec}")

    stack = contextlib.ExitStack()
    try:
        checkpointer = stack.enter_context(
            MongoDBSaver.from_conn_string(mongo.uri, db_name=mongo.db)
        )
    except ServerSelectionTimeoutError as exc:
        stack.close()
        raise StartupError(f"could not connect to MongoDB — {exc}") from exc

    # SUMMARY_MODEL routes the (cheap, structured) summarization call to a
    # smaller/cheaper model; unset keeps the prior behavior of summarizing
    # with the main model.
    summary_spec = os.environ.get("SUMMARY_MODEL")
    summary_override = resolve_model(summary_spec) if summary_spec else None
    # `build_model` hands back a bare string for any provider it doesn't
    # special-case, so resolve to a chat model here, once.
    summary_model = ensure_chat_model(summary_override or built_model)

    # Probe the *resolved* model: a bare string has no `.profile`, which would
    # silently pin the context gauge to the fallback for those models.
    profile = getattr(ensure_chat_model(built_model), "profile", None) or {}
    ctx_max = profile.get("max_input_tokens") or env_int(
        "MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS
    )

    return SharedRuntime(
        mcp_config=mcp_config,
        agent_config=agent_config,
        resolve_model=resolve_model,
        agents_md=agents_md,
        mongo=mongo,
        checkpointer=checkpointer,
        session_store=session_store,
        built_model=built_model,
        main_model_spec=main_model_spec,
        summary_override=summary_override,
        summary_model=summary_model,
        recursion_limit=env_int("RECURSION_LIMIT", DEFAULT_RECURSION_LIMIT),
        # Name each top-level graph so LangSmith traces show the app instead
        # of the langgraph library default ("LangGraph").
        app_name=os.environ.get("APP_NAME", "ghidra-deep-agent"),
        max_context_tokens=ctx_max,
        probe_tools=probe_tools,
        _stack=stack,
    )


# --- per-program engine -------------------------------------------------------


class Graphs(NamedTuple):
    """The three coordinator graphs a caller switches between."""

    main: Any
    # Read-only planner: no mutating tools, delegates only to read-only sub-agents.
    plan: Any
    # Read-only question-answerer: same tool set as plan mode (no Ghidra writes
    # and no knowledge-base writes), with one more read-only delegate.
    ask: Any


@dataclass
class Engine:
    """One agent bound to one program. Close with :meth:`aclose`."""

    program: ProgramRef
    graphs: Graphs
    storage: Storage
    compaction_engine: Any
    knowledge_ok: bool
    tool_count: int
    # Local root of the agent's files (FilesystemBackend), or None.
    output_dir: str | None
    _stack: contextlib.AsyncExitStack

    async def aclose(self) -> None:
        """Tear down this engine's sandbox, if it has one."""
        await self._stack.aclose()


def build_tools(
    mcp_tools: list[Any],
    mongo: MongoConfig,
    binary_name: str,
) -> tuple[list[Any], bool]:
    """Merge the MCP tools with the locally-defined ones.

    Returns ``(tools, knowledge_ok)``. The full tool set is what per-agent
    allowlists are then drawn from — the coordinator's restricted tool set must
    not narrow what sub-agents can use. ``knowledge_ok`` feeds the `db` health
    indicator, which otherwise reports healthy unconditionally.
    """
    knowledge_ok = True
    try:
        embeddings = build_embeddings(mongo.embed_model)
        knowledge_tools = build_knowledge_tools(
            mongo.uri, mongo.db, embeddings, binary_name
        )
        print(f"Knowledge base ready  [embed: {mongo.embed_model}]")
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

    Returns ``(plan_mode_subagents, ask_mode_subagents)``. Pass the sub-agent
    list built with ``policy_override=READ_ONLY_WRITE_POLICY``: these two modes
    are read-only *by construction*, so what a config entry asks for doesn't
    matter here. Only investigation-shaped agents are offered — the fixers exist
    to mutate and have nothing to do once they cannot.

    `research` is a config `[[subagents]]` entry shared by every graph: the normal
    coordinator gets the annotating build of it, and plan mode uses the read-only
    build as its ONLY delegate. Ask mode additionally gets `vuln-hunter` so
    exploitability questions can be routed to it.
    """
    research_sub = next(
        (s for s in subagents if s.get("name") == RESEARCH_SUBAGENT_NAME), None
    )
    if research_sub is None:
        raise ValueError(
            f"A '{RESEARCH_SUBAGENT_NAME}' sub-agent is required "
            "(plan mode and ask mode depend on it); add it to the agent config."
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
) -> list[Any]:
    """Middleware shared by all three graphs, in wrapping order.

    Built once so the normal, plan-mode, and ask-mode agents cannot drift apart
    in behaviour — they are meant to differ only in prompt, tools, and delegates.
    """
    return [
        # Sandbox file sync (first): its before_agent seeds the sandbox from
        # the output dir and its after_agent syncs changed files back, so these
        # hooks bracket every other middleware. Absent when not sandboxed.
        *([storage.sync_middleware] if storage.sync_middleware else []),
        # Model-call resilience (outermost): provider fallback wraps
        # transient-error retry of the primary model.
        *build_model_resilience_middleware(resolve_model),
        # Tool calls: validate args (reject bad calls without retry), serve
        # immutable reads from cache, resolve async task stubs (inside the cache
        # so resolved results are what gets cached), then retry transient I/O.
        #
        # The coordinator never mutates the program itself — it delegates. Its
        # allowlist grants no write-only tool, but the dual read/write tools it
        # does need (`bookmarks`, to drain the pending-change queue) carry write
        # actions along with the reads, so every write action is blocked here.
        create_argument_validation_middleware(ALL_WRITE_ACTIONS),
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
        # Ask mode gets `plan_tools`, not `main_tools`: it is a read-only mode, so
        # the knowledge-base write tools are withheld structurally rather than
        # left for the prompt to talk it out of using.
        ask=build(ASK_MODE_SYSTEM_PROMPT, plan_tools, ask_mode_subagents),
    )


OnMismatch = Callable[[str, str], None]


async def build_engine(
    shared: SharedRuntime,
    program: ProgramRef,
    *,
    output_dir: str,
    session_id: str = "",
    verify_pin: bool = True,
    on_mismatch: OnMismatch | None = None,
) -> Engine:
    """Build one agent bound to ``program``.

    ``output_dir`` is where this agent's files live locally ("" for none).
    ``session_id`` only tags the Jev prune log; the TUI passes its session, the
    server (whose engines serve many sessions) passes the agent id.
    ``on_mismatch`` is told when a tool result shows Ghidra operated on some
    other program (the pin's last line of defense); the server uses it to stop
    scheduling runs on the instance.
    """
    stack = contextlib.AsyncExitStack()
    try:
        # The error wrapper is outermost so a pin mismatch reaches the model
        # as a failed tool call rather than unwinding the graph.
        tools = await connect_mcp(
            shared.mcp_config,
            interceptors=[
                handle_mcp_errors,
                pin_program(
                    program.project_path, verify=verify_pin, on_mismatch=on_mismatch
                ),
            ],
        )
        if verify_pin:
            try:
                await verify_pinned(tools, program)
            except PinMismatchError as exc:
                raise StartupError(
                    f"cannot pin to {program.project_path}: {exc}"
                ) from exc

        storage = await open_storage(stack, output_dir=output_dir)

        all_tools, knowledge_ok = build_tools(tools, shared.mongo, program.name)
        main_tools = build_main_tools(all_tools, shared.agent_config)
        # Shared across the coordinator and sub-agents: one cache for the whole
        # agent (same binary, same Mongo collection). None when disabled/unreachable.
        cache_mw = build_mcp_cache_middleware(
            shared.mongo.uri, shared.mongo.db, program.name
        )
        # GhidrAssistMCP runs slow tools (e.g. get_code) as async tasks that
        # return a task_id stub; this middleware polls get_task_status so the
        # agent sees the resolved result. None when the server exposes no
        # get_task_status tool.
        async_mw = build_async_task_middleware(tools)
        # Jev context pruning, shared by the coordinator and every sub-agent so
        # a verdict on a tool result holds across graphs. None without a
        # TypeSafe key. Verdicts are memoized by tool_call_id, which is unique
        # across sessions, so one instance per engine is safe.
        prune_mw = build_context_pruning_middleware(
            shared.mongo.uri, shared.mongo.db, session_id, program.name
        )
        if prune_mw is not None:
            print("Jev context pruning enabled (savings log: MongoDB jev_prune_log).")
        print(
            f"Agent for {program.project_path}: {shared.main_model_spec}  "
            f"[{len(main_tools)} tool(s)]"
        )
        for sub_cfg in shared.agent_config.subagents:
            # Show the write tier for anything but the default: how much of the
            # program an agent may change is worth seeing before a session starts.
            policy = (
                ""
                if sub_cfg.write_policy == DEFAULT_WRITE_POLICY
                else f"  [writes: {sub_cfg.write_policy}]"
            )
            print(
                f"  sub-agent {sub_cfg.name}: "
                f"{resolve_model_spec(sub_cfg.model, shared.agent_config)}{policy}"
            )

        # One summarization engine shared by the agent-facing
        # compact_conversation tool and the out-of-band /compact. Built inside
        # the storage context: it offloads evicted history to the backend.
        compaction_engine = create_manual_compaction_engine(
            shared.summary_model, storage.backend
        )

        # Tuned auto-summarizers ride along as replace-by-name middleware:
        # sub-agents compact aggressively by default (they never reached
        # deepagents' 170k no-profile trigger); the main agent keeps stock
        # thresholds. COMPACT_* / COMPACT_MAIN_* env knobs override either
        # scope, and SUMMARY_MODEL routes the auto summary too, not just
        # /compact. Built here because they offload evicted history to the
        # backend.
        subagents = build_subagents(
            all_tools,
            shared.agent_config,
            shared.resolve_model,
            storage.backend,
            cache_middleware=cache_mw,
            async_middleware=async_mw,
            pruning_middleware=prune_mw,
            summary_model=shared.summary_override,
        )
        # Plan mode and ask mode delegate to the SAME config entries, rebuilt
        # under a forced read-only policy. Building a second set (rather than
        # relying on `read_only = true` in the config) is what makes those
        # graphs read-only by construction: `research` can annotate in the
        # normal graph and still be structurally unable to write here.
        read_only_subagents = build_subagents(
            all_tools,
            shared.agent_config,
            shared.resolve_model,
            storage.backend,
            cache_middleware=cache_mw,
            async_middleware=async_mw,
            pruning_middleware=prune_mw,
            summary_model=shared.summary_override,
            policy_override=READ_ONLY_WRITE_POLICY,
        )
        try:
            plan_mode_subagents, ask_mode_subagents = _read_only_delegates(
                read_only_subagents
            )
        except ValueError as exc:
            raise StartupError(str(exc)) from exc

        shared_middleware = _build_shared_middleware(
            storage=storage,
            resolve_model=shared.resolve_model,
            cache_mw=cache_mw,
            async_mw=async_mw,
            compaction_engine=compaction_engine,
            built_model=shared.built_model,
            summary_override=shared.summary_override,
            prune_mw=prune_mw,
        )

        graphs = _build_graphs(
            built_model=shared.built_model,
            agents_md=shared.agents_md,
            storage=storage,
            checkpointer=shared.checkpointer,
            middleware=shared_middleware,
            app_name=shared.app_name,
            main_tools=main_tools,
            plan_tools=build_plan_mode_main_tools(all_tools, shared.agent_config),
            subagents=subagents,
            plan_mode_subagents=plan_mode_subagents,
            ask_mode_subagents=ask_mode_subagents,
        )
    except BaseException:
        # Anything entered so far (the sandbox) must not outlive a failed build.
        await stack.aclose()
        raise

    return Engine(
        program=program,
        graphs=graphs,
        storage=storage,
        compaction_engine=compaction_engine,
        knowledge_ok=knowledge_ok,
        tool_count=len(tools),
        output_dir=output_dir or None,
        _stack=stack,
    )
