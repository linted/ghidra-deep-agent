"""Relevance-based context pruning with TypeSafe's Jev.

Long research runs re-send the same decompiler dumps, xref listings, and
string-table hits on every model call long after the agent has finished with
them (``TODO.md`` records ~2.96M tokens per research invocation at an 84:1
prompt:completion ratio). The summarizer only fires at a token threshold, costs
a model call, and replaces history with prose; this middleware trims *every*
call for a fraction of a cent, and keeps what it keeps verbatim.

How it works, per model call (``wrap_model_call``):

1. Skip cheaply when the message history is under ``trigger_tokens``.
2. Find every completed tool exchange (a ``tool_calls`` entry on an
   ``AIMessage`` plus the ``ToolMessage`` answering it). The most recent
   ``keep_recent`` exchanges, tools in ``exclude_tools``, and results under
   ``min_tokens`` are never candidates.
3. Ask Jev — TypeSafe's classification model, which returns calibrated
   probabilities rather than text — one ``Noul`` question per candidate:
   *does the agent still need this result in working memory to finish the
   task?* Candidates are packed into batched requests under Jev's state budget
   (counted in *Jev's* tokens — it tokenizes disassembly at ~1.3 chars/token,
   three times denser than the model-side estimate) and judged concurrently.
   A batch Jev rejects as too large is split and retried; any other failure
   drops only that batch. Decisions are memoized per ``tool_call_id``:
   eviction is permanent, and a "keep" is re-asked only when the goal (first +
   latest human message, ignoring the truncation-recovery nudge) changes.
4. Replace each evicted ``ToolMessage``'s content with a short placeholder that
   names the tool so the model can re-run it (cheap where the MCP read cache
   serves it). The ``AIMessage`` and every identity field stay intact.

The rewrite is **per request only**: ``state["messages"]`` is never touched, so
the checkpoint, ``/compact``, and the summarizer (which sits outside this
middleware and still counts, summarizes, and offloads the raw history) are all
unaffected. Any Jev failure fails open — the affected results go out unpruned.

Every pass above the trigger is recorded to a MongoDB collection (one document
per model call, with per-exchange decisions) so the saving can be measured;
``python -m ghidra_deep_agent.context_pruning report`` aggregates it.

Configuration (env):
  TYPESAFE_API_KEY               enables pruning (unset: middleware not installed)
  JEV_PRUNE                      ``0`` disables pruning even with a key
  JEV_PRUNE_THRESHOLD            evict when P(keep) is below this (default 0.5)
  JEV_PRUNE_TRIGGER_TOKENS       history size that starts pruning (default 20000)
  JEV_PRUNE_KEEP_RECENT          most recent exchanges never judged (default 4)
  JEV_PRUNE_EXCLUDE_TOOLS        comma-separated tools never pruned (default: task)
  JEV_PRUNE_DEBUG                set to log each pass to stderr
  JEV_PRUNE_LOG                  ``0`` disables the MongoDB savings log
  MONGODB_PRUNE_LOG_COLLECTION   savings-log collection (default ``jev_prune_log``)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from pymongo.collection import Collection

from ghidra_deep_agent.defaults import env_float, env_int
from ghidra_deep_agent.mongo_util import get_mongo_client, mongo_write_with_retry
from ghidra_deep_agent.resilience import is_truncation_nudge

# Jev's P(keep) sits at 0.3–0.5 for results the agent has already acted on
# (mutation confirmations, notes it has read) and at 0.55+ for ones it is still
# working from, on the first measured session (2026-09-20: 33 of 57 verdicts in
# 0.4–0.5; a 0.2 threshold evicted 2). Evicting below 0.5 — "Jev does not lean
# keep" — separates those; a wrong eviction costs one re-run of the tool.
DEFAULT_THRESHOLD = 0.5
DEFAULT_TRIGGER_TOKENS = 20_000
DEFAULT_KEEP_RECENT = 4
DEFAULT_MIN_TOKENS = 500
DEFAULT_EXCLUDE_TOOLS: frozenset[str] = frozenset({"task"})
# Jev tokenizes decompiler and disassembly output at ~1.33 chars per token
# (measured), three times denser than the ~4 chars/token the model-side counter
# assumes. Every Jev-facing budget below is in Jev tokens via this ratio.
JEV_CHARS_PER_TOKEN = 1.3
# Jev's state budget is 32k of its tokens (plus the longest question); requests
# over it are rejected with a 400 ``max_tokens_exceeded``. The batch cap keeps
# the fixed fields, every excerpt, and JSON overhead well under it; a batch that
# still overflows is split and retried.
DEFAULT_BATCH_TOKENS = 20_000
DEFAULT_BATCH_QUESTIONS = 16
# Excerpt cap per result, in model tokens (~4 chars each): 1,500 is ~4.6k Jev
# tokens, and verdicts measured within ±0.05 of those from 6,000-token excerpts.
DEFAULT_RESULT_EXCERPT_TOKENS = 1_500
DEFAULT_TIMEOUT = 10.0
# Fixed per-batch overhead in Jev tokens: JSON structure plus one question's
# instructions and criteria for each member.
_BATCH_OVERHEAD_TOKENS = 200
_QUESTION_OVERHEAD_TOKENS = 250
_GOAL_MAX_CHARS = 2_000
_ARGS_MAX_CHARS = 1_000
_ACTIVITY_MAX_CHARS = 1_000
# TypeSafe bills Jev input at $0.042 per million tokens; output is free.
JEV_USD_PER_INPUT_TOKEN = 0.042 / 1_000_000

_QUESTION_INSTRUCTIONS = (
    "An autonomous reverse-engineering agent is working on the task in `task`; "
    "its most recent request from the user is `latest_request` and its latest "
    "reasoning is `recent_activity`. Earlier it called the tool named in "
    "`exchanges.{key}.tool` with `exchanges.{key}.arguments` and received "
    "`exchanges.{key}.result`. Does the agent still need that result in its "
    "working memory to finish the task? Answer yes only if it is likely to read "
    "or reference that result again; answer no if the result has been superseded "
    "by later results, has already been acted on, is unrelated to the task, or "
    "could simply be re-run if needed."
)
_CRITERIA_TRUE = (
    "The result holds information the agent still needs and has not yet used, "
    "or that its next steps will build on."
)
_CRITERIA_FALSE = (
    "The result is stale, superseded by a later result, already incorporated "
    "into later reasoning or outputs, unrelated to the task, or cheap to fetch "
    "again."
)


# --- Jev judge ------------------------------------------------------------------


@dataclass(frozen=True)
class Judgement:
    """One Jev request's answers: ``P(keep)`` per question key, plus usage."""

    keep_probabilities: dict[str, float]
    input_tokens: int | None = None


class Judge(Protocol):
    """What the middleware needs from Jev; injectable for tests."""

    def judge(self, state: dict[str, Any], keys: Sequence[str]) -> Judgement: ...

    async def ajudge(self, state: dict[str, Any], keys: Sequence[str]) -> Judgement: ...


def _questions(keys: Iterable[str]) -> dict[str, Any]:
    from langchain_typesafe import Noul, NoulCriteria

    criteria = NoulCriteria(true=_CRITERIA_TRUE, false=_CRITERIA_FALSE)
    return {
        key: Noul(
            instructions=_QUESTION_INSTRUCTIONS.format(key=key), criteria=criteria
        )
        for key in keys
    }


class JevJudge:
    """Ask Jev through ``TypeSafeClassifier``.

    Questions are constructor-bound in ``langchain-typesafe`` 0.0.1a2, so a
    classifier is built per batch; the ``httpx2`` clients are shared so the
    connection pool is not.
    """

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        import httpx2

        self._timeout = timeout
        self._client = httpx2.Client(timeout=timeout)
        self._async_client = httpx2.AsyncClient(timeout=timeout)

    def _classifier(self, keys: Sequence[str]) -> Any:
        from langchain_core._api.beta_decorator import LangChainBetaWarning
        from langchain_typesafe import TypeSafeClassifier

        # The classifier class is marked beta and warns on every construction;
        # once per batch would flood stderr.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LangChainBetaWarning)
            return TypeSafeClassifier(
                questions=_questions(keys),
                timeout=self._timeout,
                client=self._client,
                async_client=self._async_client,
            )

    @staticmethod
    def _judgement(response: Any, keys: Sequence[str]) -> Judgement:
        nouls = response.nouls
        # A key Jev did not answer is treated as "keep": never evict on silence.
        probs = {key: nouls[key].noul for key in keys if key in nouls}
        return Judgement(probs, response.usage.input_tokens)

    def judge(self, state: dict[str, Any], keys: Sequence[str]) -> Judgement:
        return self._judgement(self._classifier(keys).invoke(state), keys)

    async def ajudge(self, state: dict[str, Any], keys: Sequence[str]) -> Judgement:
        return self._judgement(await self._classifier(keys).ainvoke(state), keys)


def _is_too_large(exc: Exception) -> bool:
    """Jev's 400 for a state over its token budget (``max_tokens_exceeded``)."""
    body = getattr(exc, "body", None)
    detail = body.get("detail") if isinstance(body, dict) else None
    return (
        isinstance(detail, dict) and detail.get("error_type") == "max_tokens_exceeded"
    )


def _describe(exc: Exception) -> str:
    """``str(exc)`` plus the response body the TypeSafe SDK keeps out of it."""
    text = f"{type(exc).__name__}: {exc}"
    body = getattr(exc, "body", None)
    if body is not None:
        text += f" body={str(body)[:300]}"
    return text


# --- Savings log ------------------------------------------------------------------

_LOG_INDEX_NAME = "session_ts"


def _ensure_log_index(collection: Collection[dict[str, Any]]) -> None:
    if collection.index_information().get(_LOG_INDEX_NAME) is not None:
        return
    collection.create_index([("session_id", 1), ("ts", -1)], name=_LOG_INDEX_NAME)


class PruneLog:
    """Append-only record of pruning passes, one document per model call."""

    def __init__(self, collection: Collection[dict[str, Any]]) -> None:
        self._collection = collection

    def record(self, doc: dict[str, Any]) -> None:
        mongo_write_with_retry(lambda: self._collection.insert_one(doc))

    async def arecord(self, doc: dict[str, Any]) -> None:
        await asyncio.to_thread(self.record, doc)


def build_prune_log(mongodb_uri: str, mongodb_db: str) -> PruneLog | None:
    """Open the savings log, or ``None`` (with a warning) if Mongo is unreachable."""
    coll_name = os.environ.get("MONGODB_PRUNE_LOG_COLLECTION", "jev_prune_log")
    try:
        collection = get_mongo_client(mongodb_uri)[mongodb_db][coll_name]
        _ensure_log_index(collection)
    except Exception as exc:  # pragma: no cover - environmental
        print(f"Warning: Jev pruning log disabled ({exc})", file=sys.stderr)
        return None
    return PruneLog(collection)


# --- Message bookkeeping --------------------------------------------------------------


@dataclass(frozen=True)
class _Exchange:
    """One tool call and the ``ToolMessage`` answering it."""

    index: int  # position of the ToolMessage in the request
    tool_call_id: str
    tool: str
    args: Any
    tokens: int


@dataclass
class _Decision:
    p_keep: float
    evicted: bool
    goal_key: str


@dataclass
class _Pass:
    """Everything one pruning pass learned, for the debug line and the log."""

    tokens_before: int
    exchanges_total: int = 0
    candidates: int = 0
    judged: int = 0
    jev_requests: int = 0
    jev_failed_requests: int = 0
    jev_input_tokens: int = 0
    jev_latency_ms: int = 0
    decisions: list[dict[str, Any]] = field(default_factory=list)
    # tool_call_ids whose verdict was made in this pass (vs. memoized earlier).
    judged_ids: set[str] = field(default_factory=set)
    error: str | None = None  # first failure; the pass carries on without it
    tokens_after: int = 0

    @property
    def evicted(self) -> int:
        return sum(1 for d in self.decisions if d["evicted"])


def _is_summary(msg: AnyMessage) -> bool:
    return msg.additional_kwargs.get("lc_source") == "summarization"


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _excerpt(text: str, max_tokens: int) -> str:
    """Head-and-tail excerpt so one dump can't eat a Jev batch.

    ``max_tokens`` is in model tokens (the ~4 chars/token the counter assumes);
    the batch packer converts the excerpt's length to Jev tokens itself.
    """
    limit = max_tokens * 4
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n…[{omitted} characters omitted]…\n{text[-tail:]}"


def _jev_tokens(text: str) -> int:
    """Estimate of how many tokens Jev will count ``text`` as."""
    return int(len(text) / JEV_CHARS_PER_TOKEN) + 1


def _goal(messages: Sequence[AnyMessage]) -> tuple[str, str] | None:
    """(task, latest request): first and last real human messages.

    Summaries and the truncation-recovery nudge are skipped: the nudge is an
    automated "carry on" that changes nothing about the goal, and letting it
    change the goal key re-judged every kept result at once on the first run.
    """
    humans = [
        m
        for m in messages
        if isinstance(m, HumanMessage)
        and not _is_summary(m)
        and not is_truncation_nudge(m)
    ]
    if not humans:
        return None
    return _clip(humans[0].text, _GOAL_MAX_CHARS), _clip(
        humans[-1].text, _GOAL_MAX_CHARS
    )


def _recent_activity(messages: Sequence[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.text.strip():
            return _clip(msg.text, _ACTIVITY_MAX_CHARS)
    return ""


def _exchanges(messages: Sequence[AnyMessage]) -> list[_Exchange]:
    calls: dict[str, tuple[str, Any]] = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                calls[call["id"] or ""] = (call["name"], call["args"])
    found: list[_Exchange] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            continue
        origin = calls.get(msg.tool_call_id)
        if origin is None:
            continue
        name, args = origin
        found.append(
            _Exchange(
                index=i,
                tool_call_id=msg.tool_call_id,
                tool=msg.name or name,
                args=args,
                tokens=count_tokens_approximately([msg]),
            )
        )
    return found


def _placeholder(tool: str, p_keep: float) -> str:
    return (
        "[Tool result pruned from context: no longer relevant to the current "
        f"task (relevance {p_keep:.2f}). Re-run `{tool}` with the same arguments "
        "if you need it again.]"
    )


def _pruned_copy(msg: ToolMessage, tool: str, p_keep: float) -> ToolMessage:
    return ToolMessage(
        content=_placeholder(tool, p_keep),
        tool_call_id=msg.tool_call_id,
        name=msg.name,
        id=msg.id,
        status=msg.status,
    )


def _scope() -> tuple[str, str]:
    """("main" | "subagent", checkpoint namespace) for the running model call."""
    try:
        from langgraph.config import get_config

        ns = str(get_config().get("configurable", {}).get("checkpoint_ns", ""))
    except Exception:  # outside a graph (unit tests, direct calls): no namespace
        ns = ""
    # A sub-agent runs in a nested namespace, joined with "|" — the heuristic
    # the TUI's context gauge uses to ignore sub-agent usage.
    return ("subagent" if "|" in ns else "main"), ns


_Batch = tuple[dict[str, Any], list[tuple[str, _Exchange]]]
_Plan = tuple[_Pass, list[_Exchange], str, list[_Batch]]


# --- Middleware --------------------------------------------------------------


class JevContextPruningMiddleware(AgentMiddleware):
    """Blank tool results Jev judges irrelevant before each model call.

    See the module docstring for the algorithm. ``judge`` defaults to a live
    :class:`JevJudge`; tests inject a fake. ``log`` (when given) receives one
    document per pass above the trigger.
    """

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        trigger_tokens: int = DEFAULT_TRIGGER_TOKENS,
        keep_recent: int = DEFAULT_KEEP_RECENT,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        exclude_tools: frozenset[str] = DEFAULT_EXCLUDE_TOOLS,
        result_excerpt_tokens: int = DEFAULT_RESULT_EXCERPT_TOKENS,
        batch_tokens: int = DEFAULT_BATCH_TOKENS,
        batch_questions: int = DEFAULT_BATCH_QUESTIONS,
        timeout: float = DEFAULT_TIMEOUT,
        debug: bool = False,
        judge: Judge | None = None,
        log: PruneLog | None = None,
        session_id: str = "",
        binary_name: str = "",
    ) -> None:
        super().__init__()
        self._threshold = threshold
        self._trigger = trigger_tokens
        self._keep_recent = keep_recent
        self._min_tokens = min_tokens
        self._exclude = exclude_tools
        self._excerpt_tokens = result_excerpt_tokens
        self._batch_tokens = batch_tokens
        self._batch_questions = batch_questions
        self._debug = debug
        self._judge: Judge = judge if judge is not None else JevJudge(timeout=timeout)
        self._log = log
        self._session_id = session_id
        self._binary = binary_name
        # tool_call_id -> decision. Shared by the coordinator and every sub-agent
        # (ids are unique), so a result judged in one graph stays judged.
        self._decisions: dict[str, _Decision] = {}
        self._warned = False
        # Counters for tests and the debug line.
        self.judged = 0
        self.evicted = 0
        self.failures = 0

    # --- planning ----------------------------------------------------------------

    def _candidates(
        self, exchanges: Sequence[_Exchange], goal_key: str
    ) -> list[_Exchange]:
        """Exchanges that need a Jev verdict this pass."""
        eligible = exchanges[: -self._keep_recent] if self._keep_recent else exchanges
        pending: list[_Exchange] = []
        for ex in eligible:
            if ex.tool in self._exclude or ex.tokens < self._min_tokens:
                continue
            prior = self._decisions.get(ex.tool_call_id)
            if prior is not None and (prior.evicted or prior.goal_key == goal_key):
                continue
            pending.append(ex)
        return pending

    def _batches(
        self,
        messages: Sequence[AnyMessage],
        pending: Sequence[_Exchange],
        fixed_tokens: int,
    ) -> list[_Batch]:
        """Pack pending exchanges into Jev requests under the state budget.

        Costs are in Jev tokens: ``fixed_tokens`` covers the goal and activity
        fields every batch carries, and each member pays for its excerpt, its
        arguments, and one question's prompt.
        """
        batches: list[_Batch] = []
        current: dict[str, Any] = {}
        members: list[tuple[str, _Exchange]] = []
        base = fixed_tokens + _BATCH_OVERHEAD_TOKENS
        used = base
        for ex in pending:
            tool_msg = messages[ex.index]
            excerpt = _excerpt(tool_msg.text, self._excerpt_tokens)
            try:
                args_text = json.dumps(ex.args, default=str, sort_keys=True)
            except TypeError:
                args_text = str(ex.args)
            entry = {
                "tool": ex.tool,
                "arguments": _clip(args_text, _ARGS_MAX_CHARS),
                "result": excerpt,
            }
            cost = (
                _jev_tokens(excerpt)
                + _jev_tokens(entry["arguments"])
                + _jev_tokens(ex.tool)
                + _QUESTION_OVERHEAD_TOKENS
            )
            if members and (
                used + cost > self._batch_tokens
                or len(members) >= self._batch_questions
            ):
                batches.append((current, members))
                current, members, used = {}, [], base
            key = f"x{len(members)}"
            current[key] = entry
            members.append((key, ex))
            used += cost
        if members:
            batches.append((current, members))
        return batches

    @staticmethod
    def _halves(batch: _Batch) -> list[_Batch]:
        """Split a batch Jev rejected as too large into two smaller requests."""
        state, members = batch
        mid = len(members) // 2
        return [
            (
                {**state, "exchanges": {k: state["exchanges"][k] for k, _ in part}},
                list(part),
            )
            for part in (members[:mid], members[mid:])
        ]

    def _state(
        self, task: str, latest: str, activity: str, exchanges: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "task": task,
            "latest_request": latest,
            "recent_activity": activity,
            "exchanges": exchanges,
        }

    def _record_judgement(
        self,
        judgement: Judgement,
        members: Sequence[tuple[str, _Exchange]],
        goal_key: str,
        run: _Pass,
    ) -> None:
        run.jev_requests += 1
        run.jev_input_tokens += judgement.input_tokens or 0
        for key, ex in members:
            p_keep = judgement.keep_probabilities.get(key)
            if p_keep is None:
                continue  # unanswered: leave undecided, judged again next pass
            self._decisions[ex.tool_call_id] = _Decision(
                p_keep=p_keep, evicted=p_keep < self._threshold, goal_key=goal_key
            )
            run.judged_ids.add(ex.tool_call_id)
            run.judged += 1
            self.judged += 1

    # --- judging -----------------------------------------------------------------
    #
    # One batch failing must not sink the pass: a too-large batch is halved and
    # retried (down to single questions), and any other error fails open for
    # that batch alone while the rest are still judged.

    def _judge_batch(self, batch: _Batch, goal_key: str, run: _Pass) -> None:
        state, members = batch
        try:
            judgement = self._judge.judge(state, [key for key, _ in members])
        except Exception as exc:  # any Jev/network failure: never block the model
            if _is_too_large(exc) and len(members) > 1:
                run.jev_failed_requests += 1
                for half in self._halves(batch):
                    self._judge_batch(half, goal_key, run)
                return
            self._fail_open(run, exc)
            return
        self._record_judgement(judgement, members, goal_key, run)

    async def _ajudge_batch(self, batch: _Batch, goal_key: str, run: _Pass) -> None:
        state, members = batch
        try:
            judgement = await self._judge.ajudge(state, [key for key, _ in members])
        except Exception as exc:  # any Jev/network failure: never block the model
            if _is_too_large(exc) and len(members) > 1:
                run.jev_failed_requests += 1
                await asyncio.gather(
                    *(self._ajudge_batch(h, goal_key, run) for h in self._halves(batch))
                )
                return
            self._fail_open(run, exc)
            return
        self._record_judgement(judgement, members, goal_key, run)

    # --- applying ----------------------------------------------------------------

    def _apply(
        self, messages: Sequence[AnyMessage], exchanges: Sequence[_Exchange], run: _Pass
    ) -> list[AnyMessage]:
        pruned = list(messages)
        for ex in exchanges:
            decision = self._decisions.get(ex.tool_call_id)
            if decision is None:
                continue
            memoized = ex.tool_call_id not in run.judged_ids
            run.decisions.append(
                {
                    "tool": ex.tool,
                    "tool_call_id": ex.tool_call_id,
                    "tokens": ex.tokens,
                    "p_keep": decision.p_keep,
                    "evicted": decision.evicted,
                    "memoized": memoized,
                }
            )
            if decision.evicted:
                original = pruned[ex.index]
                assert isinstance(original, ToolMessage)
                pruned[ex.index] = _pruned_copy(original, ex.tool, decision.p_keep)
        return pruned

    def _finish(self, run: _Pass, pruned: Sequence[AnyMessage]) -> dict[str, Any]:
        run.tokens_after = count_tokens_approximately(pruned)
        self.evicted += sum(
            1 for d in run.decisions if d["evicted"] and not d["memoized"]
        )
        if self._debug:
            saved = run.tokens_before - run.tokens_after
            print(
                f"[jev-prune] evicted {run.evicted}/{run.exchanges_total} exchanges "
                f"(~{run.tokens_before // 1000}k → ~{run.tokens_after // 1000}k "
                f"tokens, saved ~{saved // 1000}k; judged {run.judged} in "
                f"{run.jev_requests} Jev request(s), {run.jev_latency_ms} ms)"
                + (f" [fail-open: {run.error}]" if run.error else ""),
                file=sys.stderr,
            )
        scope, ns = _scope()
        return {
            "session_id": self._session_id,
            "binary": self._binary,
            "ts": datetime.now(UTC),
            "scope": scope,
            "checkpoint_ns": ns,
            "tokens_before": run.tokens_before,
            "tokens_after": run.tokens_after,
            "tokens_saved": run.tokens_before - run.tokens_after,
            "exchanges_total": run.exchanges_total,
            "candidates": run.candidates,
            "judged": run.judged,
            "evicted": run.evicted,
            "jev_requests": run.jev_requests,
            "jev_failed_requests": run.jev_failed_requests,
            "jev_input_tokens": run.jev_input_tokens,
            "jev_latency_ms": run.jev_latency_ms,
            "decisions": run.decisions,
            "error": run.error,
        }

    def _fail_open(self, run: _Pass, exc: Exception) -> None:
        """Record a failed batch; its results go out unpruned this pass."""
        self.failures += 1
        run.jev_failed_requests += 1
        error = _describe(exc)
        if run.error is None:
            run.error = error
        if not self._warned:
            self._warned = True
            print(
                f"Warning: Jev context pruning failed ({error}); sending those "
                "results unpruned. Further failures are counted silently.",
                file=sys.stderr,
            )

    def _plan(self, messages: Sequence[AnyMessage]) -> _Plan | None:
        """Shared setup; ``None`` means pass the request through untouched."""
        tokens_before = count_tokens_approximately(messages)
        if tokens_before < self._trigger:
            return None
        goal = _goal(messages)
        if goal is None:
            return None
        task, latest = goal
        goal_key = hashlib.sha1(f"{task}\0{latest}".encode()).hexdigest()
        exchanges = _exchanges(messages)
        run = _Pass(tokens_before=tokens_before, exchanges_total=len(exchanges))
        pending = self._candidates(exchanges, goal_key)
        run.candidates = len(pending)
        activity = _recent_activity(messages)
        fixed = _jev_tokens(task) + _jev_tokens(latest) + _jev_tokens(activity)
        batches = [
            (self._state(task, latest, activity, state), members)
            for state, members in self._batches(messages, pending, fixed)
        ]
        return run, exchanges, goal_key, batches

    # --- hooks -------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        plan = self._plan(request.messages)
        if plan is None:
            return handler(request)
        run, exchanges, goal_key, batches = plan
        started = time.monotonic()
        for batch in batches:
            self._judge_batch(batch, goal_key, run)
        run.jev_latency_ms = int((time.monotonic() - started) * 1000)
        pruned = self._apply(request.messages, exchanges, run)
        doc = self._finish(run, pruned)
        if self._log is not None:
            try:
                self._log.record(doc)
            except Exception as exc:  # the log must never cost a model call
                print(f"Warning: Jev pruning log write failed ({exc})", file=sys.stderr)
        return handler(request.override(messages=pruned))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        plan = self._plan(request.messages)
        if plan is None:
            return await handler(request)
        run, exchanges, goal_key, batches = plan
        started = time.monotonic()
        await asyncio.gather(
            *(self._ajudge_batch(batch, goal_key, run) for batch in batches)
        )
        run.jev_latency_ms = int((time.monotonic() - started) * 1000)
        pruned = self._apply(request.messages, exchanges, run)
        doc = self._finish(run, pruned)
        if self._log is not None:
            try:
                await self._log.arecord(doc)
            except Exception as exc:  # the log must never cost a model call
                print(f"Warning: Jev pruning log write failed ({exc})", file=sys.stderr)
        return await handler(request.override(messages=pruned))


# --- Factory ------------------------------------------------------------------------


def _exclude_tools_from_env() -> frozenset[str]:
    raw = os.environ.get("JEV_PRUNE_EXCLUDE_TOOLS")
    if raw is None:
        return DEFAULT_EXCLUDE_TOOLS
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


def build_context_pruning_middleware(
    mongodb_uri: str, mongodb_db: str, session_id: str, binary_name: str
) -> JevContextPruningMiddleware | None:
    """Build the pruner from env, or ``None`` when it should not be installed.

    Pruning needs a ``TYPESAFE_API_KEY``; ``JEV_PRUNE=0`` turns it off with the
    key still set. The savings log rides along unless ``JEV_PRUNE_LOG=0``.
    """
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        return None
    if os.environ.get("JEV_PRUNE", "").strip() == "0":
        return None
    log = None
    if os.environ.get("JEV_PRUNE_LOG", "").strip() != "0":
        log = build_prune_log(mongodb_uri, mongodb_db)
    return JevContextPruningMiddleware(
        threshold=env_float("JEV_PRUNE_THRESHOLD", DEFAULT_THRESHOLD),
        trigger_tokens=env_int("JEV_PRUNE_TRIGGER_TOKENS", DEFAULT_TRIGGER_TOKENS),
        keep_recent=env_int(
            "JEV_PRUNE_KEEP_RECENT", DEFAULT_KEEP_RECENT, positive=False
        ),
        exclude_tools=_exclude_tools_from_env(),
        debug=bool(os.environ.get("JEV_PRUNE_DEBUG")),
        log=log,
        session_id=session_id,
        binary_name=binary_name,
    )


# --- Report ---------------------------------------------------------------------------


def _parse_since(text: str) -> timedelta:
    units = {"h": "hours", "d": "days", "w": "weeks", "m": "minutes"}
    unit = text[-1]
    if unit not in units or not text[:-1].isdigit():
        raise argparse.ArgumentTypeError(
            f"{text!r}: expected a number followed by m, h, d, or w (e.g. 7d)"
        )
    return timedelta(**{units[unit]: int(text[:-1])})


def report(
    collection: Collection[dict[str, Any]],
    *,
    session_id: str | None = None,
    since: timedelta | None = None,
) -> str:
    """Aggregate the savings log into a readable summary."""
    query: dict[str, Any] = {}
    if session_id:
        query["session_id"] = session_id
    if since is not None:
        query["ts"] = {"$gte": datetime.now(UTC) - since}
    docs = list(collection.find(query).sort("ts", 1))
    if not docs:
        return "No pruning passes recorded for that selection."

    per_session: dict[str, dict[str, int]] = defaultdict(
        lambda: {"passes": 0, "judged": 0, "evicted": 0, "saved": 0, "before": 0}
    )
    per_tool: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    histogram: Counter[int] = Counter()
    errors: Counter[str] = Counter()
    # Tokens that would have been saved (summed over passes, like the headline)
    # had the threshold been each of these instead — the tuning table.
    sweep_thresholds = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    sweep: Counter[float] = Counter()
    totals = {
        "passes": 0,
        "exchanges": 0,
        "judged": 0,
        "evicted": 0,
        "before": 0,
        "after": 0,
        "saved": 0,
        "jev_tokens": 0,
        "jev_requests": 0,
        "failures": 0,
    }
    for doc in docs:
        s = per_session[doc.get("session_id") or "?"]
        s["passes"] += 1
        s["judged"] += doc.get("judged", 0)
        s["evicted"] += doc.get("evicted", 0)
        s["saved"] += doc.get("tokens_saved", 0)
        s["before"] += doc.get("tokens_before", 0)
        totals["passes"] += 1
        totals["exchanges"] += doc.get("exchanges_total", 0)
        totals["judged"] += doc.get("judged", 0)
        totals["evicted"] += doc.get("evicted", 0)
        totals["before"] += doc.get("tokens_before", 0)
        totals["after"] += doc.get("tokens_after", 0)
        totals["saved"] += doc.get("tokens_saved", 0)
        totals["jev_tokens"] += doc.get("jev_input_tokens", 0)
        totals["jev_requests"] += doc.get("jev_requests", 0)
        if doc.get("error"):
            totals["failures"] += 1
            # "Type: endpoint: 400 Bad Request (request_id=…) body=…" -> kind
            errors[str(doc["error"]).split(" (request_id")[0][:100]] += 1
        for d in doc.get("decisions", []):
            p_keep = float(d.get("p_keep", 1.0))
            for t in sweep_thresholds:
                if p_keep < t:
                    sweep[t] += int(d.get("tokens", 0))
            if d.get("memoized"):
                continue  # count each verdict once, when it was made
            tally = per_tool[d.get("tool", "?")]
            tally[0] += 1
            tally[1] += 1 if d.get("evicted") else 0
            histogram[min(int(p_keep * 10), 9)] += 1

    def pct(num: int, den: int) -> str:
        return f"{100 * num / den:.0f}%" if den else "n/a"

    lines = [
        "Jev context pruning — savings report",
        f"  passes: {totals['passes']}   sessions: {len(per_session)}"
        f"   fail-open passes: {totals['failures']}",
        f"  verdicts made: {totals['judged']}   evictions in effect: "
        f"{totals['evicted']} summed over passes "
        f"({pct(totals['evicted'], totals['exchanges'])} of exchanges seen)",
        f"  input tokens sent: ~{totals['after']:,} instead of ~{totals['before']:,} "
        f"(saved ~{totals['saved']:,}, {pct(totals['saved'], totals['before'])})",
        f"  Jev: {totals['jev_requests']} requests, {totals['jev_tokens']:,} input "
        f"tokens ≈ ${totals['jev_tokens'] * JEV_USD_PER_INPUT_TOKEN:.4f}",
        "",
        "Per session (tokens saved / sent before pruning):",
    ]
    for sid, s in sorted(per_session.items(), key=lambda kv: -kv[1]["saved"]):
        lines.append(
            f"  {sid}: {s['passes']} passes, {s['evicted']} evicted, "
            f"saved ~{s['saved']:,} of ~{s['before']:,} "
            f"({pct(s['saved'], s['before'])})"
        )
    if per_tool:
        lines += ["", "Per tool (verdicts, evicted):"]
        for tool, (n, ev) in sorted(per_tool.items(), key=lambda kv: -kv[1][1]):
            lines.append(f"  {tool}: {n} judged, {ev} evicted ({pct(ev, n)})")
    if histogram:
        lines += ["", "P(keep) distribution of verdicts (0.1 buckets):"]
        total = sum(histogram.values())
        for bucket in range(10):
            n = histogram.get(bucket, 0)
            bar = "#" * int(40 * n / total) if total else ""
            lines.append(f"  {bucket / 10:.1f}–{(bucket + 1) / 10:.1f}  {n:5d}  {bar}")
        lines += ["", "Tokens saved had JEV_PRUNE_THRESHOLD been (summed over passes):"]
        for t in sweep_thresholds:
            lines.append(
                f"  {t:.1f}: ~{sweep[t]:,} ({pct(sweep[t], totals['before'])})"
            )
    if errors:
        lines += ["", "Fail-open causes (passes):"]
        for kind, n in errors.most_common():
            lines.append(f"  {n:5d}  {kind}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m ghidra_deep_agent.context_pruning report [...]``."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(prog="ghidra_deep_agent.context_pruning")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="summarize the Jev pruning savings log")
    rep.add_argument("--session", help="only this session id")
    rep.add_argument(
        "--since", type=_parse_since, help="only passes newer than e.g. 7d, 12h"
    )
    args = parser.parse_args(argv)
    # Same defaults as the CLI's storage config; kept inline so the report does
    # not import the TUI stack.
    uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    db = os.environ.get("MONGODB_DB", "checkpointing_db")
    coll_name = os.environ.get("MONGODB_PRUNE_LOG_COLLECTION", "jev_prune_log")
    collection = get_mongo_client(uri)[db][coll_name]
    print(report(collection, session_id=args.session, since=args.since))


if __name__ == "__main__":  # pragma: no cover
    main()
