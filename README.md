<p align="center">
  <img src="https://raw.githubusercontent.com/huwprosser/fury/a5f785da526e09af78d9522f1b275be421bbb5e8/fury.png" alt="Fury Logo" width="192" />
</p>

<h1 align="center">Fury</h1>

<p align="center">
  <a href="https://discord.gg/xC9Yd6VH2a">
    <img src="https://img.shields.io/discord/841085263266447400?logo=discord" alt="Discord">
  </a>
</p>

A flexible and powerful AI agent library for Python, designed to build agents with tool support, multimodal capabilities, and streaming responses.

## Features

- **Easy-to-use Agent API**: Simple interface to create agents with custom system prompts and models.
- **Tool Support**: Define and register custom tools (functions) that the agent can execute.
- **Parallel Tool Execution**: Built-in support for running multiple independent tools in parallel.
- **Multimodal Capabilities**: Support for image and voice inputs (using Whisper for STT).
- **Streaming Responses**: Real-time streaming of agent responses and reasoning.
- **OpenAI Compatible**: Built on top of `AsyncOpenAI`, making it compatible with OpenAI models and local inference servers (like vLLM, Ollama, etc.).

## Roadmap
- [ ] History manager with auto-compaction support.
- [ ] TTS support.
- [ ] E2E voice agent example.

## Installation

Install with uv:

```bash
uv add fury-sdk
```

Install with pip:

```bash
pip install fury-sdk
```

Install directly from github using:

```bash
uv add git+https://github.com/huwprosser/fury.git
```

or

```bash
pip install git+https://github.com/huwprosser/fury.git
```

### Examples

If you also want example dependencies:

```bash
uv add "git+https://github.com/huwprosser/fury.git[examples]"
```

Pip equivalent for examples:

```bash
pip install "git+https://github.com/huwprosser/fury.git[examples]"
```

For local development in this repository:

```bash
uv sync --all-extras
```

## Quick Start

Most basic usage:

```python
from fury import Agent

agent = Agent(
    model="your-model-name",  # e.g., "gpt-4o" or a local model
    system_prompt="You are a helpful assistant.",
    base_url="http://127.0.0.1:8080/v1",  # or https://openrouter.ai/api/v1, https://api.openai.com/v1
    api_key="your-api-key",
)

response = agent.ask("Hello!", history=[])
print(response)
```

Here is a simple example of how to create a chat agent:

```python
import asyncio
from fury import Agent

async def main():
    # Initialize the agent
    agent = Agent(
        model="your-model-name", # e.g., "gpt-4o" or a local model
        system_prompt="You are a helpful assistant.",
        base_url="http://127.0.0.1:8080/v1", # or https://openrouter.ai/api/v1, https://api.openai.com/v1
        api_key="your-api-key"
    )

    history = []

    # Simple chat loop
    while True:
        user_input = input("> ")
        history.append({"role": "user", "content": user_input})

        async for event in agent.chat(history):
            if event.content:
                print(event.content, end="", flush=True)

        print()

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration Options

```python
agent = Agent(
    model="your-model-name",
    system_prompt="You are a helpful assistant.",
    parallel_tool_calls=False,
    generation_params={
        "temperature": 0.2,
        "max_tokens": 512,
    },
)

# Disable reasoning stream content (default is False)
async for event in agent.chat(history, reasoning=False):
    ...

# Or for single-shot calls
response = agent.ask("Hello!", history=[], reasoning=False)
```

## Advanced Usage

### Defining Tools

You can give your agent tools to interact with the world. Tools are defined using the `create_tool` helper.

Input and output schemas help the model to correctly pass parameters through to the function. Fury will automatically prune any hallucinated parameters not defined in the input schema.

Learn more in the [OpenAI guide](https://developers.openai.com/api/docs/guides/function-calling/)

```python
from fury import Agent, create_tool

# Define the function
def add(a: int, b: int):
    return {"result": a + b}

# Create the tool
add_tool = create_tool(
    id="add",
    description="Add two numbers together",
    execute=add,
    announcement_phrase="Adding numbers...",
    input_schema={
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    },
    output_schema={
        "type": "object",
        "properties": {"result": {"type": "integer"}},
        "required": ["result"],
    },
)

# Pass to agent
agent = Agent(..., tools=[add_tool])
```

### Coding Assistant Example

Check out `examples/coding-assistant/coding_assistant.py` for a full-featured example that includes:

- File system operations (`read`, `write`, `edit`, `bash`).
- **Skills System**: Loading specialized capabilities from `SKILL.md` files.
- **Memory System**: Using `MEMORY.md` and `SOUL.md` for context.
- **History Compaction**: Summarizing long conversations to save context window.

## Running Examples

To run the provided examples, ensure you have the package installed.

**Basic Chat:**

```bash
uv run examples/chat.py
```

**Coding Assistant (Based on Pi.dev):**

```bash
uv run examples/coding-assistant/coding_assistant.py
```

## Project Structure

- `src/agent_lib/`: Core library code.
    - `agent.py`: Main `Agent` class and logic.
- `examples/`: Usage examples.
    - `chat.py`: Basic chat loop.
    - `coding-assistant/`: Advanced agent with file ops and memory.

# Run Tests

To run the pytest tests you will first need to install the additional test deps.
`uv sync --extra test`

Then run:
`uv run pytest -v`
