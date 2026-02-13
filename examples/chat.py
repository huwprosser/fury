import asyncio
from fury import Agent


agent = Agent(
    model="unsloth/GLM-4.6V-Flash-GGUF:Q8_0",
    system_prompt="You are a helpful assistant.",
)


async def main():
    history = []

    while True:

        # Add the user's input to the history.
        history.append({"role": "user", "content": input("> ")})

        # history = agent.add_image_to_history(history, "image.jpg") Optionally add a image to the last message of the history.
        # history = agent.add_voice_message_to_history(history, base64_audio_bytes) Optionally add a voice message to the last message of the history.

        buffer = ""

        print()
        async for chunk, _, _ in agent.chat(history, reasoning=False):
            if chunk:
                buffer += chunk
                print(chunk, end="", flush=True)

        history.append({"role": "assistant", "content": buffer})


asyncio.run(main())
