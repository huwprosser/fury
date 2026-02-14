"""
A coding assistant that uses the Fury agent library to help with coding tasks.

This example is based on the Pi.dev coding assistant example.

This example is a semi-vibe-coded clone of the Pi.dev coding assistant example to use the Fury agent library.

It strips away most of Pi's features but does include:
- Auto-compaction of history.
- AgentSkills system.
- SOUL.md and MEMORY.md injection.
- A handful of useful tools for the agent to use.

It is a simple example of how to use the Fury agent library to create a coding assistant.
"""

import asyncio
import base64
import json
import mimetypes
import os
import tempfile
import subprocess
from dataclasses import dataclass
from termcolor import cprint
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from fury import Agent, create_tool

MAX_LINES = 2000
MAX_BYTES = 100 * 1024
IMAGE_MIME_PREFIXES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
CONTEXT_WINDOW = 32768
RESERVE_TOKENS = 8192
KEEP_RECENT_TOKENS = 8000
SUMMARY_PREFIX = "Summary of previous conversation:"
SKILLS_DOC_PATH = "docs/skills.md"


class ReadInput(BaseModel):
    path: str
    offset: Optional[int] = None
    limit: Optional[int] = None


class BashInput(BaseModel):
    command: str
    timeout: Optional[int] = None


class WriteInput(BaseModel):
    path: str
    content: str


class EditInput(BaseModel):
    path: str
    old_text: str
    new_text: str


@dataclass
class Skill:
    name: str
    description: str
    file_path: str
    base_dir: str
    disable_model_invocation: bool = False


# ------------------------------------------------------------------------------------------------ #
# TOOLS
# ------------------------------------------------------------------------------------------------ #


def resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def truncate_text(text: str) -> tuple[str, bool, int, int]:
    lines = text.splitlines()
    total_lines = len(lines)
    total_bytes = len(text.encode("utf-8"))
    truncated = False

    if total_lines > MAX_LINES:
        lines = lines[-MAX_LINES:]
        truncated = True

    truncated_text = "\n".join(lines)
    if len(truncated_text.encode("utf-8")) > MAX_BYTES:
        truncated = True
        truncated_text = truncated_text.encode("utf-8")[-MAX_BYTES:].decode(
            "utf-8", errors="replace"
        )

    return truncated_text, truncated, total_lines, total_bytes


def read_text_file(path: str, offset: Optional[int], limit: Optional[int]) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        if offset is not None or limit is not None:
            start_line = max(offset or 1, 1)
            end_line = start_line + (limit - 1) if limit else None
            selected = []
            for index, line in enumerate(handle, start=1):
                if index < start_line:
                    continue
                if end_line is not None and index > end_line:
                    break
                selected.append(line)
            return "".join(selected) or "(no content)"

        tail_lines: deque[str] = deque()
        total_lines = 0
        kept_bytes = 0
        for line in handle:
            total_lines += 1
            line_bytes = len(line.encode("utf-8"))
            tail_lines.append(line)
            kept_bytes += line_bytes
            while len(tail_lines) > MAX_LINES or kept_bytes > MAX_BYTES:
                removed = tail_lines.popleft()
                kept_bytes -= len(removed.encode("utf-8"))

        content = "".join(tail_lines) or "(empty file)"
        if total_lines > len(tail_lines):
            start_line = total_lines - len(tail_lines) + 1
            content += (
                f"\n\n[Showing lines {start_line}-{total_lines} of {total_lines}. "
                "Use offset/limit to view earlier lines.]"
            )
        return content


def read_tool(path: str, offset: Optional[int] = None, limit: Optional[int] = None):
    print()
    cprint(f"Reading {path}...", "green")
    resolved_path = resolve_path(path)
    if not os.path.exists(resolved_path):
        return f"Error: path not found: {resolved_path}"

    mime_type, _ = mimetypes.guess_type(resolved_path)
    if mime_type in IMAGE_MIME_PREFIXES:
        with open(resolved_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("utf-8")
        return {
            "description": f"Image read from {resolved_path}.",
            "image_base64": encoded,
        }

    return read_text_file(resolved_path, offset, limit)


def bash_tool(command: str, timeout: Optional[int] = None):
    try:
        print()
        cprint(f"Running command {command}...", "cyan")
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        output += f"\n\nCommand timed out after {timeout} seconds"
        return output

    output = output or "(no output)"
    truncated_text, truncated, total_lines, _ = truncate_text(output)
    if truncated:
        with tempfile.NamedTemporaryFile(
            delete=False, prefix="pi-bash-", suffix=".log", mode="w", encoding="utf-8"
        ) as handle:
            handle.write(output)
            full_path = handle.name
        truncated_text += (
            f"\n\n[Showing last {MAX_LINES} lines. Full output: {full_path}]"
        )

    if result.returncode != 0:
        truncated_text += f"\n\nCommand exited with code {result.returncode}"

    return truncated_text


def write_tool(path: str, content: str):
    print()
    cprint(f"Writing to {path}...", "magenta")
    resolved_path = resolve_path(path)
    os.makedirs(os.path.dirname(resolved_path) or ".", exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return f"Wrote {len(content)} characters to {resolved_path}"


def edit_tool(path: str, old_text: str, new_text: str):
    print()
    cprint(f"Editing {path}...", "red")
    resolved_path = resolve_path(path)
    if not os.path.exists(resolved_path):
        return f"Error: path not found: {resolved_path}"

    with open(resolved_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    if old_text not in content:
        return "Error: old_text not found in file."

    updated = content.replace(old_text, new_text)
    with open(resolved_path, "w", encoding="utf-8") as handle:
        handle.write(updated)

    return f"Replaced {content.count(old_text)} occurrence(s) in {resolved_path}"


# ------------------------------------------------------------------------------------------------ #
# CORE FUNCTIONS
# ------------------------------------------------------------------------------------------------ #


def parse_skill_frontmatter(content: str) -> Dict[str, Any]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    frontmatter: Dict[str, Any] = {}
    for index in range(1, len(lines)):
        line = lines[index]
        if line.strip() == "---":
            break
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in {"true", "false"}:
            frontmatter[key] = value.lower() == "true"
        else:
            frontmatter[key] = value

    return frontmatter


def discover_skill_files(root_dir: str) -> List[str]:
    files: List[str] = []
    if not os.path.exists(root_dir):
        return files

    try:
        entries = list(os.scandir(root_dir))
    except OSError:
        return files

    for entry in entries:
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        path = entry.path
        try:
            if entry.is_dir(follow_symlinks=False):
                files.extend(discover_skill_files_in_subdir(path))
            elif entry.is_file(follow_symlinks=False) and entry.name.endswith(".md"):
                files.append(path)
        except OSError:
            continue

    return files


def discover_skill_files_in_subdir(root_dir: str) -> List[str]:
    files: List[str] = []
    try:
        entries = list(os.scandir(root_dir))
    except OSError:
        return files

    for entry in entries:
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        path = entry.path
        try:
            if entry.is_dir(follow_symlinks=False):
                files.extend(discover_skill_files_in_subdir(path))
            elif entry.is_file(follow_symlinks=False) and entry.name == "SKILL.md":
                files.append(path)
        except OSError:
            continue

    return files


def load_skill_from_file(file_path: str) -> Optional[Skill]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except OSError:
        return None

    frontmatter = parse_skill_frontmatter(content)
    description = str(frontmatter.get("description", "")).strip()
    if not description:
        return None

    base_dir = os.path.dirname(file_path)
    name = str(frontmatter.get("name") or os.path.basename(base_dir)).strip()
    disable_model_invocation = frontmatter.get("disable-model-invocation") is True

    return Skill(
        name=name,
        description=description,
        file_path=file_path,
        base_dir=base_dir,
        disable_model_invocation=disable_model_invocation,
    )


def load_skills() -> List[Skill]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [os.path.join(base_dir, "skills")]

    skills_by_name: Dict[str, Skill] = {}
    seen_paths: set[str] = set()

    for search_dir in search_dirs:
        for file_path in discover_skill_files(search_dir):
            real_path = os.path.realpath(file_path)
            if real_path in seen_paths:
                continue
            skill = load_skill_from_file(file_path)
            if not skill:
                continue
            if skill.name in skills_by_name:
                continue
            skills_by_name[skill.name] = skill
            seen_paths.add(real_path)

    return list(skills_by_name.values())


def escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_skills_for_prompt(skills: List[Skill]) -> str:
    visible_skills = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible_skills:
        return ""

    lines = [
        "The following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "",
        "<available_skills>",
    ]

    for skill in visible_skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{escape_xml(skill.file_path)}</location>")
        lines.append("  </skill>")

    lines.append("</available_skills>")

    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ #
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------------------------ #


def format_token_count(count: int) -> str:
    if count < 1000:
        return str(count)
    if count < 10000:
        return f"{count / 1000:.1f}k"
    if count < 1000000:
        return f"{round(count / 1000)}k"
    if count < 10000000:
        return f"{count / 1000000:.1f}M"
    return f"{round(count / 1000000)}M"


def estimate_tokens_for_message(message: Dict[str, Any]) -> int:
    role = message.get("role")
    chars = 0

    if role in {"user", "system"}:
        content = message.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    chars += len(block.get("text", ""))
        elif isinstance(content, dict):
            chars += len(json.dumps(content, ensure_ascii=False))
        else:
            chars += len(str(content))
    elif role == "assistant":
        content = message.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    chars += len(block.get("text", ""))
                elif isinstance(block, dict) and block.get("type") == "toolCall":
                    chars += len(block.get("name", ""))
                    chars += len(
                        json.dumps(block.get("arguments", {}), ensure_ascii=False)
                    )
        else:
            chars += len(str(content))

        tool_calls = message.get("tool_calls")
        if tool_calls:
            chars += len(json.dumps(tool_calls, ensure_ascii=False))
    elif role == "tool":
        content = message.get("content", "")
        if isinstance(content, (dict, list)):
            chars += len(json.dumps(content, ensure_ascii=False))
        else:
            chars += len(str(content))
    else:
        content = message.get("content", "")
        chars += len(str(content))

    return (chars + 3) // 4


def get_context_usage(history: List[Dict[str, Any]]) -> tuple[int, float]:
    tokens = sum(estimate_tokens_for_message(msg) for msg in history)
    percent = (tokens / CONTEXT_WINDOW) * 100 if CONTEXT_WINDOW else 0.0
    return tokens, percent


def extract_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        calls.extend([call for call in tool_calls if isinstance(call, dict)])

    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "toolCall":
                calls.append(
                    {
                        "name": block.get("name"),
                        "arguments": block.get("arguments"),
                    }
                )

    return calls


def parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return {}


def collect_file_ops(messages: List[Dict[str, Any]]) -> tuple[set[str], set[str]]:
    read_files: set[str] = set()
    modified_files: set[str] = set()

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in extract_tool_calls(msg):
            name = call.get("name")
            args = parse_tool_arguments(call.get("arguments"))
            path = args.get("path")
            if not path:
                continue
            if name == "read":
                read_files.add(path)
            elif name in {"edit", "write"}:
                modified_files.add(path)

    return read_files, modified_files


def format_message_for_summary(message: Dict[str, Any]) -> str:
    role = message.get("role", "unknown")
    if role == "assistant":
        tool_calls = extract_tool_calls(message)
        if tool_calls:
            return f"{role} tool_calls: {json.dumps(tool_calls, ensure_ascii=False)}"

    content = message.get("content", "")
    if isinstance(content, (list, dict)):
        content = json.dumps(content, ensure_ascii=False)

    if role == "tool":
        return f"tool result: {content}"

    return f"{role}: {content}"


def should_compact(context_tokens: int) -> bool:
    return context_tokens > CONTEXT_WINDOW - RESERVE_TOKENS


def find_cut_index(messages: List[Dict[str, Any]]) -> int:
    accumulated = 0
    tentative_index = 0

    for i in range(len(messages) - 1, -1, -1):
        accumulated += estimate_tokens_for_message(messages[i])
        if accumulated >= KEEP_RECENT_TOKENS:
            tentative_index = i
            break
    else:
        return 0

    valid_indices = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in {"user", "assistant", "system"}:
            valid_indices.append(idx)

    candidates = [idx for idx in valid_indices if idx <= tentative_index]
    if not candidates:
        return 0
    return max(candidates)


async def compact_history(
    agent: Agent, history: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not history:
        return history

    existing_summary = None
    working_history = history

    if (
        history
        and history[0].get("role") == "system"
        and isinstance(history[0].get("content"), str)
        and history[0]["content"].startswith(SUMMARY_PREFIX)
    ):
        existing_summary = history[0]["content"][len(SUMMARY_PREFIX) :].strip()
        working_history = history[1:]

    context_tokens = sum(estimate_tokens_for_message(msg) for msg in working_history)
    if not should_compact(context_tokens):
        return history

    cprint("\nCompacting history...\n", "grey")

    cut_index = find_cut_index(working_history)
    if cut_index <= 0:
        return history

    to_summarize = working_history[:cut_index]
    tail = working_history[cut_index:]

    lines = []
    if existing_summary:
        lines.append("Existing summary:")
        lines.append(existing_summary)

    lines.append("Conversation to summarize:")
    for msg in to_summarize:
        lines.append(format_message_for_summary(msg))

    read_files, modified_files = collect_file_ops(to_summarize)
    if read_files or modified_files:
        lines.append("")
        lines.append("Known file operations (from tool calls in this segment):")
        if read_files:
            lines.append(f"Read files: {', '.join(sorted(read_files))}")
        if modified_files:
            lines.append(f"Modified files: {', '.join(sorted(modified_files))}")

    summary_prompt = "\n".join(lines).strip()
    if not summary_prompt:
        return history

    completion = await agent.client.chat.completions.create(
        model=agent.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize the conversation for future context using this format:\n"
                    "## Goal\n"
                    "[What the user is trying to accomplish]\n\n"
                    "## Constraints & Preferences\n"
                    "- [Requirements mentioned by user]\n\n"
                    "## Progress\n"
                    "### Done\n"
                    "- [x] [Completed tasks]\n\n"
                    "### In Progress\n"
                    "- [ ] [Current work]\n\n"
                    "### Blocked\n"
                    "- [Issues, if any]\n\n"
                    "## Key Decisions\n"
                    "- **[Decision]**: [Rationale]\n\n"
                    "## Next Steps\n"
                    "1. [What should happen next]\n\n"
                    "## Critical Context\n"
                    "- [Data needed to continue]\n\n"
                    "<read-files>\n"
                    "path/to/file1\n"
                    "</read-files>\n\n"
                    "<modified-files>\n"
                    "path/to/file2\n"
                    "</modified-files>\n\n"
                    "Be concise but include key decisions, filenames, commands, and TODOs."
                ),
            },
            {"role": "user", "content": summary_prompt},
        ],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    summary_text = completion.choices[0].message.content or "(summary unavailable)"
    summary_message = f"{SUMMARY_PREFIX}\n{summary_text.strip()}"

    return [{"role": "system", "content": summary_message}] + tail


def load_context_files() -> List[tuple[str, str]]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = ["SOUL.md", "MEMORY.md"]
    context_files: List[tuple[str, str]] = []

    for filename in candidates:
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                content = handle.read().strip()
        except OSError:
            continue
        if content:
            context_files.append((path, content))

    return context_files


def build_prompt() -> str:
    now = datetime.now().astimezone()
    date_time = (
        f"{now.strftime('%A, %B ')}"
        f"{now.day}"
        f"{now.strftime(', %Y at %I:%M:%S %p %Z')}"
    )
    cwd = os.getcwd()
    context_files = load_context_files()
    skills_section = format_skills_for_prompt(load_skills())

    prompt = f"""You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents
- bash: Execute bash commands (ls, grep, find, etc.)
- edit: Make surgical edits to files (find exact text and replace)
- write: Create or overwrite files
- multi_tool_use.parallel: Run multiple tool calls in parallel (use when independent tools can run together).  

Guidelines:
- Use bash for file operations like ls, rg, find
- Use read to examine files before editing. You must use this tool instead of cat or sed.
- Use edit for precise changes (old text must match exactly)
- Use write only for new files or complete rewrites
- When summarizing your actions, output plain text directly - do NOT use cat or bash to display what you did
- Be concise in your responses. Also sarcastic and witty.
- Show file paths clearly when working with files
- When a skill applies, read its SKILL.md and follow any linked docs before acting.
- Use multi_tool_use.parallel to batch independent tool calls instead of sequential calls.            
- Before creating or modifying a skill, ALWAYS read docs/skills.md and follow it exactly.

MEMORY.md: Access long-term facts, user preferences, and historical context from MEMORY.md when relevant to the conversation."""

    if context_files:
        prompt += "\n\n# Project Context\n\n"
        prompt += "Project-specific instructions and guidelines:\n\n"
        for path, content in context_files:
            prompt += f"## {path}\n\n{content}\n\n"

    if skills_section:
        prompt += skills_section

    prompt += (
        f"\n\nCurrent date and time: {date_time}\nCurrent working directory: {cwd}"
    )
    return prompt


async def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_KEY", "")

    agent = Agent(
        model="qwen/qwen3-coder-next",
        system_prompt=build_prompt(),
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0.7,
        tools=[
            create_tool(
                "read",
                "Read file contents (text or image). Supports offset/limit for text files.",
                read_tool,
                "Reading [path]...",
                input_schema=ReadInput,
            ),
            create_tool(
                "bash",
                "Execute a bash command in the current working directory.",
                bash_tool,
                "Running command...",
                input_schema=BashInput,
            ),
            create_tool(
                "write",
                "Write content to a file, creating parent directories if needed.",
                write_tool,
                "Writing to [path]...",
                input_schema=WriteInput,
            ),
            create_tool(
                "edit",
                "Replace exact text in a file with new text.",
                edit_tool,
                "Editing [path]...",
                input_schema=EditInput,
            ),
        ],
    )
    agent.show_yourself()

    history = []
    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        history.append({"role": "user", "content": user_input})

        buffer = []
        last_stream_kind = None
        async for event in agent.chat(history, True):
            if event.content:
                if last_stream_kind == "reasoning":
                    print()
                last_stream_kind = "chunk"
                buffer.append(event.content)
                print(event.content, end="", flush=True)

            if event.reasoning:
                if last_stream_kind == "chunk":
                    print()
                last_stream_kind = "reasoning"
                cprint(event.reasoning, "grey", end="", flush=True)

        print()
        history.append({"role": "assistant", "content": "".join(buffer)})
        history = await compact_history(agent, history)

        context_tokens, context_percent = get_context_usage(history)
        cprint(
            f"Context: {format_token_count(context_tokens)}/{format_token_count(CONTEXT_WINDOW)} "
            f"({context_percent:.1f}%)",
            "blue",
        )


if __name__ == "__main__":
    asyncio.run(main())
