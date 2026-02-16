import asyncio
from datetime import datetime

from fury import Agent, create_tool


def check_time():
    now = datetime.now().isoformat(timespec="seconds")
    return {"current_time": now}


def test_basic_tool_calling():
    tool = create_tool(
        "check_time",
        "Return the current local time as an ISO-8601 string.",
        check_time,
        "Checking the time...",
        {"type": "object", "properties": {}, "required": []},
        {"type": "object", "properties": {}, "required": []},
    )

    agent = Agent(
        model="unsloth/GLM-4.6V-Flash-GGUF:Q8_0",
        system_prompt=(
            "You are a helpful assistant. Use the correct tools to answer the user's question."
        ),
        tools=[tool],
    )

    async def run_chat():
        history = [
            {
                "role": "user",
                "content": "What time is it right now? Use the check_time tool. ",
            }
        ]

        buffer = []
        tool_called = False
        tool_result = None

        async for event in agent.chat(history, reasoning=False):
            if event.tool_call and event.tool_call.tool_name == "check_time":
                tool_called = True
                if event.tool_call.result is not None:
                    tool_result = event.tool_call.result
            if event.content:
                buffer.append(event.content)

        response = "".join(buffer).strip()
        return response, tool_called, tool_result

    response, tool_called, tool_result = asyncio.run(run_chat())
    assert tool_called
    assert tool_result
    assert response
