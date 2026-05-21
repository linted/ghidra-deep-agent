from langchain_core.messages import AIMessage, ToolMessage

ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[36m"
ANSI_YELLOW = "\033[33m"
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_DIM = "\033[2m"


async def recover_from_tool_error(agent, config: dict, exc: Exception) -> None:
    state = await agent.aget_state(config)
    messages = state.values.get("messages", [])

    responded_ids = {msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage)}
    dangling = [
        tc
        for msg in messages
        if isinstance(msg, AIMessage)
        for tc in getattr(msg, "tool_calls", [])
        if tc["id"] not in responded_ids
    ]

    if not dangling:
        return

    await agent.aupdate_state(config, {
        "messages": [
            ToolMessage(content=str(exc), tool_call_id=tc["id"])
            for tc in dangling
        ]
    })

    print(f"{ANSI_DIM}[retrying after tool error]{ANSI_RESET}", flush=True)
    await stream_response(agent, None, config)


async def stream_response(agent, user_input: str | None, config: dict) -> None:
    """Stream the agent's response, showing tool calls and text tokens in real time."""
    print()

    in_text = False
    active_tool: str | None = None

    input_data = (
        {"messages": [{"role": "user", "content": user_input}]}
        if user_input is not None
        else None
    )

    async for event in agent.astream_events(
        input_data,
        config=config,
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_tool_start":
            name = event.get("name", "")
            internal = {"write_todos", "read_file", "write_file", "edit_file",
                        "ls", "glob", "grep", "task"}
            if name and name not in internal:
                if in_text:
                    print()
                    in_text = False
                active_tool = name
                print(f"{ANSI_YELLOW}⚙ {name}{ANSI_RESET}", end="  ", flush=True)

        elif kind == "on_tool_end":
            if active_tool:
                print(f"{ANSI_GREEN}✓{ANSI_RESET}", flush=True)
                active_tool = None

        elif kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk is None:
                continue

            content = chunk.content
            if isinstance(content, str) and content:
                if not in_text:
                    print()
                    in_text = True
                print(content, end="", flush=True)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            if not in_text:
                                print()
                                in_text = True
                            print(text, end="", flush=True)

    if in_text:
        print()
    print()
