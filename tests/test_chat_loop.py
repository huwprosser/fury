import asyncio

from fury import Agent


def test_basic_chat_loop():
    agent = Agent(
        model="unsloth/GLM-4.6V-Flash-GGUF:Q8_0",
        system_prompt="You are a helpful assistant.",
    )

    async def run_chat():
        history = [
            {"role": "user", "content": "Hello! Please reply with a short greeting."}
        ]
        buffer = []

        async for event in agent.chat(history, reasoning=False):
            if event.content:
                buffer.append(event.content)

        return "".join(buffer).strip()

    response = asyncio.run(run_chat())
    assert response
