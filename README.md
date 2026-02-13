<p align="center">
  <img src="fury.png" alt="Fury Logo" width="192" />
</p>

<h1 align="center">Fury</h1>

<p align="center">
  <a href="https://discord.gg/xC9Yd6VH2a">
    <img src="https://img.shields.io/discord/841085263266447400?logo=discord" alt="Discord">
  </a>
</p>

Easy-to-use tooling for the worlds most powerful technology.

A flexible and powerful AI agent library for Python, designed to build agents with tool support, multimodal capabilities, and streaming responses.

## Features

- **Easy-to-use Agent API**: Simple interface to create agents with custom system prompts and models.
- **Tool Support**: Define and register custom tools (functions) that the agent can execute.
- **Parallel Tool Execution**: Built-in support for running multiple independent tools in parallel.
- **Multimodal Capabilities**: Support for image and voice inputs (using Whisper for STT).
- **Streaming Responses**: Real-time streaming of agent responses and reasoning.
- **OpenAI Compatible**: Built on top of `AsyncOpenAI`, making it compatible with OpenAI models and local inference servers (like vLLM, Ollama, etc.).

## Installation

Install directly from github using:

```bash
uv add git+https://github.com/huwprosser/fury.git
```

Pip equivalents:

```bash
pip install git+https://github.com/huwprosser/fury.git
```

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

        print()
        async for chunk, reasoning, tool_call in agent.chat(history):
            if chunk:
                print(chunk, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Usage

### Defining Tools

You can give your agent tools to interact with the world. Tools are defined using the `create_tool` helper.

```python
from agent_lib import Agent, create_tool
from pydantic import BaseModel

# Define input schema using Pydantic
class CalculatorInput(BaseModel):
    a: int
    b: int

# Define the function
def add(a: int, b: int):
    return a + b

# Create the tool
add_tool = create_tool(
    id="add",
    description="Add two numbers together",
    execute=add,
    announcement_phrase="Adding numbers...",
    input_schema=CalculatorInput.model_json_schema(),
    output_schema={"type": "integer"}
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

TO run the pytest tests you will first need to install the additional test deps.
`uv sync --extra test`

Then run:
`uv run pytest -v`
