"""Entry point: one agent, one binary, in the TUI.

The agent itself is built by :mod:`ghidra_deep_agent.runtime`, which the HTTP
server shares; this module only adds argument parsing, program selection with
a Textual picker, and the app.
"""

import argparse
import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv
from pymongo.errors import ServerSelectionTimeoutError

from ghidra_deep_agent.program_resolver import ProgramInfo, resolve_program
from ghidra_deep_agent.runtime import (
    StartupError,
    build_engine,
    open_shared_runtime,
)


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


async def _tui_chooser(programs: list[ProgramInfo]) -> ProgramInfo | None:
    """Ask the user which open program to analyze (several are open)."""
    # Imported here so the runtime modules never load Textual.
    from ghidra_deep_agent.tui import ProgramSelectApp

    by_path = {p.project_path: p for p in programs}
    selected = await ProgramSelectApp(list(by_path)).run_async()
    return by_path.get(selected or "")


async def _run(args: argparse.Namespace, session_id: str) -> None:
    shared = await open_shared_runtime()
    try:
        override = args.binary_name or os.environ.get("BINARY_NAME")
        try:
            program = await resolve_program(
                shared.probe_tools, override, choose=_tui_chooser
            )
        except RuntimeError as exc:
            raise StartupError(str(exc)) from exc
        print(f"Analyzing binary: {program.name}  [{program.project_path}]")
        if shared.session_store is not None:
            shared.session_store.record_start(session_id, program.name)

        engine = await build_engine(
            shared,
            program,
            output_dir=os.environ.get("AGENT_OUTPUT_DIR", ""),
            session_id=session_id,
        )
        try:
            from ghidra_deep_agent.tui import GhidraAgentApp

            app = GhidraAgentApp(
                agent=engine.graphs.main,
                plan_agent=engine.graphs.plan,
                ask_agent=engine.graphs.ask,
                summary_model=shared.summary_model,
                compaction_engine=engine.compaction_engine,
                config=shared.config_for(session_id),
                model=shared.main_model_spec,
                session_id=session_id,
                # `connect_mcp` raises on a failed connection, so reaching here
                # means the server answered — but it can still answer with zero
                # tools.
                mcp_ok=engine.tool_count > 0,
                db_ok=engine.knowledge_ok,
                max_context_tokens=shared.max_context_tokens,
                session_store=shared.session_store,
                binary_name=program.name,
            )
            await app.run_async()
        finally:
            # Tears down the OpenShell sandbox if one was created; a no-op otherwise.
            await engine.aclose()
    finally:
        shared.close()


async def main() -> None:
    load_dotenv()
    args = _parse_args()
    session_id = args.session_id or str(uuid.uuid4())
    try:
        await _run(args, session_id)
    except StartupError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ServerSelectionTimeoutError as exc:
        print(f"Error: could not connect to MongoDB — {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Session ID: {session_id}")


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
