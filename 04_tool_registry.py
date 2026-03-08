"""
Lesson 4: Structured Tool Execution

An interactive CLI agent with a tool registry.
Demonstrates clean separation between tool definitions and the generic agent loop.
"""

from pathlib import Path
import subprocess
from typing import Any, Callable

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

ToolHandler = Callable[..., str]
TOOL_HANDLERS: dict[str, ToolHandler] = {}
TOOL_SCHEMAS: list[dict[str, Any]] = []


def register_tool(
    *, name: str, description: str, input_schema: dict[str, Any]
) -> Callable[[ToolHandler], ToolHandler]:
    def decorator(func: ToolHandler) -> ToolHandler:
        TOOL_HANDLERS[name] = func
        TOOL_SCHEMAS.append(
            {
                "name": name,
                "description": description,
                "input_schema": input_schema,
            }
        )
        return func

    return decorator


@register_tool(
    name="read_file",
    description="Read a UTF-8 text file and return its contents.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read.",
            }
        },
        "required": ["path"],
    },
)
def read_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error reading file '{path}': {exc}"


def confirm_shell_command(command: str) -> bool:
    print("\nShell command requested:")
    print(f"  {command}")
    answer = input("Allow this command? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


@register_tool(
    name="run_shell",
    description="Run a shell command and return stdout/stderr and exit code.",
    input_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            }
        },
        "required": ["command"],
    },
)
def run_shell(command: str) -> str:
    if not confirm_shell_command(command):
        return f"Denied by user. Command not executed: {command}"

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after 30 seconds: {command}"
    except Exception as exc:
        return f"Error running command '{command}': {exc}"

    stdout = completed.stdout.strip() or "(empty)"
    stderr = completed.stderr.strip() or "(empty)"
    return (
        f"exit_code: {completed.returncode}\n"
        f"stdout:\n{stdout}\n\n"
        f"stderr:\n{stderr}"
    )


def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f"Unknown tool: {tool_name}"

    try:
        return handler(**tool_input)
    except Exception as exc:
        return f"Tool execution error for '{tool_name}': {exc}"


def run_agent_turn(messages: list[dict[str, Any]]) -> str:
    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            text_blocks = [block.text for block in response.content if block.type == "text"]
            return "\n".join(text_blocks).strip()

        tool_results = []
        for tool_use in tool_uses:
            result = execute_tool(tool_use.name, tool_use.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                }
            )

        messages.append({"role": "user", "content": tool_results})


def main() -> None:
    messages: list[dict[str, Any]] = []

    print("Tool Registry Agent (type 'quit' to exit)")
    print("-" * 40)
    print("Available tools:", ", ".join(sorted(TOOL_HANDLERS.keys())))

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})
        assistant_text = run_agent_turn(messages)
        print(f"\nClaude: {assistant_text}")


if __name__ == "__main__":
    main()
