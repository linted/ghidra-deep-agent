"""One run: stream the graph, translate, persist, fan out, finish.

Mirrors the TUI's ``_run_agent`` turn: the same input shapes (``None`` to
resume an interrupted turn), the same ``UsageLimitError`` → paused semantics,
and the same event rules via :func:`ghidra_deep_agent.stream.translate`.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from ghidra_deep_agent.prompt import ASK_MODE_TURN_PREFIX
from ghidra_deep_agent.resilience import UsageLimitError
from ghidra_deep_agent.seed import SeedError, marked_prior_context
from ghidra_deep_agent.server.events import (
    Cancelled,
    Failed,
    Final,
    Paused,
    RunWarning,
    Started,
)
from ghidra_deep_agent.server.registry import AgentInstance, AgentRegistry, RunRecord
from ghidra_deep_agent.stream import Reply, Token, Usage, translate
from ghidra_deep_agent.toasts import ToastRequest, toast_scope


def turn_input(
    run: RunRecord, *, background: str | None = None
) -> dict[str, Any] | None:
    """The graph input for this run (``None`` replays from the last checkpoint).

    ``background`` is the marked summary of the main session that seeds an ask
    thread's first turn, prepended as its own user message.
    """
    if run.resume:
        return None
    messages: list[dict[str, str]] = []
    if background:
        messages.append({"role": "user", "content": background})
    prompt = run.prompt or ""
    if run.mode == "ask":
        prompt = f"{ASK_MODE_TURN_PREFIX}\n\n{prompt}"
    messages.append({"role": "user", "content": prompt})
    return {"messages": messages}


async def ask_seed(
    run: RunRecord, inst: AgentInstance, registry: AgentRegistry, ask_graph: Any
) -> str | None:
    """The background block for an ask thread's FIRST turn, else ``None``.

    Same rule as the TUI: a fresh ask thread is seeded with a summary of the
    main session so the answerer knows what has been established. Follow-up
    turns on the same ask thread are not re-seeded.
    """
    if run.mode != "ask" or run.resume:
        return None
    engine = inst.engine
    assert engine is not None
    state = await ask_graph.aget_state(registry.shared.config_for(run.thread_id))
    if state.values.get("messages"):
        return None
    try:
        return await marked_prior_context(
            engine.graphs.main,
            registry.shared.config_for(run.session_id),
            registry.shared.summary_model,
        )
    except SeedError as exc:
        await registry.publish(run, RunWarning(f"Ask mode {exc}", "Agent", "warning"))
        return None


async def execute_run(
    run: RunRecord, inst: AgentInstance, registry: AgentRegistry
) -> None:
    engine = inst.engine
    assert engine is not None  # start_run checked readiness
    graph = engine.graphs.main if run.mode == "normal" else engine.graphs.ask
    config = registry.shared.config_for(run.thread_id)

    # Toasts are raised synchronously from inside middleware; queue them and
    # publish from this task so persistence stays ordered with the stream.
    warnings: asyncio.Queue[ToastRequest] = asyncio.Queue()

    async def flush_warnings() -> None:
        while not warnings.empty():
            toast = warnings.get_nowait()
            await registry.publish(
                run, RunWarning(toast.message, toast.title, toast.severity)
            )

    try:
        run.status = "running"
        run.started_at = datetime.now(UTC)
        await asyncio.to_thread(registry.store.upsert_run, run.to_dict())
        await registry.publish(
            run,
            Started(run.agent_id, run.session_id, run.thread_id, run.mode, run.resume),
        )
        background = await ask_seed(run, inst, registry, graph)
        with toast_scope(warnings.put_nowait):
            async for event in graph.astream_events(
                turn_input(run, background=background), config=config, version="v2"
            ):
                for ev in translate(event, run.run_state):
                    if isinstance(ev, Token):
                        continue  # token-level noise; `reply` carries the text
                    if isinstance(ev, Usage):
                        run.input_tokens += ev.input_tokens
                        run.output_tokens += ev.output_tokens
                    elif isinstance(ev, Reply):
                        run.reply = ev.text
                    await registry.publish(run, ev)
                await flush_warnings()
        await flush_warnings()
        run.status = "done"
        await registry.publish(
            run, Final(run.reply, run.input_tokens, run.output_tokens)
        )
    except UsageLimitError as exc:
        # Everything committed so far is durable in the checkpointer; the
        # caller resumes with continue=true on the same session.
        await flush_warnings()
        run.status = "paused"
        run.error = str(exc)
        await registry.publish(run, Paused(str(exc)))
    except asyncio.CancelledError:
        # Cancelled by /cancel or shutdown. The last checkpoint stands, so the
        # thread is resumable; swallow so the task ends cleanly.
        run.status = "cancelled"
        await registry.publish(run, Cancelled())
    except Exception as exc:
        await flush_warnings()
        run.status = "error"
        run.error = str(exc)
        await registry.publish(run, Failed(str(exc)))
    finally:
        # Persist the terminal record BEFORE signalling waiters, so anyone
        # released by `done` reads the final state from the store.
        registry.finish_run(run, inst)
        try:
            await asyncio.shield(
                asyncio.to_thread(registry.store.upsert_run, run.to_dict())
            )
        except BaseException:  # never mask the outcome with bookkeeping
            pass
        finally:
            run.done.set()
