import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

import fury.agent as agent_module  # type: ignore[import-untyped]
from fury import Agent, create_tool  # type: ignore[import-untyped]


class FakeStream:
    def __init__(self, chunks: List[Any]):
        self._chunks = chunks
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        item = self._chunks[self._index]
        self._index += 1
        return item


class FakeCompletions:
    def __init__(self, scripted_responses: List[List[Any]]):
        self.scripted_responses = scripted_responses
        self.calls: List[Dict[str, Any]] = []
        self._call_index = 0

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._call_index >= len(self.scripted_responses):
            raise AssertionError("No scripted response available for create()")
        response = self.scripted_responses[self._call_index]
        self._call_index += 1
        return FakeStream(response)


class FakeAsyncOpenAI:
    scripted_responses: List[List[Any]] = []
    last_client = None

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.chat = SimpleNamespace(
            completions=FakeCompletions(self.__class__.scripted_responses)
        )
        self.__class__.last_client = self


def make_chunk(*, content=None, tool_calls=None, reasoning_content=None):
    delta = SimpleNamespace(
        content=content, tool_calls=tool_calls, reasoning_content=reasoning_content
    )
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


async def collect_chat_events(agent: Agent, history: List[Dict[str, Any]]):
    events = []
    async for event in agent.chat(history, reasoning=False):
        events.append(event)
    return events


def test_basic_chat_streams_text(monkeypatch):
    monkeypatch.setattr(agent_module.Agent, "_check_server_status", lambda self: None)
    monkeypatch.setattr(agent_module.Agent, "show_yourself", lambda self: None)
    monkeypatch.setattr(agent_module, "AsyncOpenAI", FakeAsyncOpenAI)

    FakeAsyncOpenAI.scripted_responses = [
        [
            make_chunk(content="Hello "),
            make_chunk(content="world!"),
        ]
    ]

    agent = Agent(model="test-model", system_prompt="You are test assistant.")
    history = [{"role": "user", "content": "Say hello"}]
    events = asyncio.run(collect_chat_events(agent, history))

    streamed_text = "".join(event.content for event in events if event.content)
    assert streamed_text == "Hello world!"
    assert all(event.reasoning is None for event in events)
    assert all(event.tool_call is None for event in events)

    first_call = FakeAsyncOpenAI.last_client.chat.completions.calls[0]
    assert first_call["messages"][0] == {
        "role": "system",
        "content": "You are test assistant.",
    }
    assert first_call["messages"][1] == {
        "role": "user",
        "content": "Say hello",
    }


def test_chat_executes_tool_call_and_returns_followup(monkeypatch):
    monkeypatch.setattr(agent_module.Agent, "_check_server_status", lambda self: None)
    monkeypatch.setattr(agent_module.Agent, "show_yourself", lambda self: None)
    monkeypatch.setattr(agent_module, "AsyncOpenAI", FakeAsyncOpenAI)

    tool_invocations: List[Dict[str, Any]] = []

    def add(a: int, b: int):
        tool_invocations.append({"a": a, "b": b})
        return a + b

    add_tool = create_tool(
        id="add",
        description="Add two numbers",
        execute=add,
        announcement_phrase="Adding numbers...",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        output_schema={"type": "integer"},
    )

    tool_call_chunk = SimpleNamespace(
        index=0,
        id="call_1",
        function=SimpleNamespace(name="add", arguments='{"a": 2, "b": 3, "extra": 99}'),
    )

    FakeAsyncOpenAI.scripted_responses = [
        [make_chunk(tool_calls=[tool_call_chunk])],
        [make_chunk(content="The result is 5.")],
    ]

    agent = Agent(
        model="test-model",
        system_prompt="You are test assistant.",
        tools=[add_tool],
    )
    history = [{"role": "user", "content": "What is 2+3?"}]
    events = asyncio.run(collect_chat_events(agent, history))

    tool_events = [event.tool_call for event in events if event.tool_call]
    assert tool_events[0].tool_name == "add"
    assert tool_events[0].arguments == {"a": 2, "b": 3}
    assert tool_events[0].announcement_phrase == "Using Adding numbers......"
    assert tool_events[1].tool_name == "add"
    assert tool_events[1].result == 5
    assert tool_invocations == [{"a": 2, "b": 3}]

    final_text = "".join(event.content for event in events if event.content)
    assert final_text == "The result is 5."

    second_call_messages = FakeAsyncOpenAI.last_client.chat.completions.calls[1][
        "messages"
    ]
    assert any(
        msg.get("role") == "tool"
        and msg.get("name") == "add"
        and msg.get("content") == "5"
        for msg in second_call_messages
    )
