"""
Lesson 6: HTTP API Calls

An interactive CLI agent with read, write, shell, and HTTP tools.
Demonstrates external API integration with graceful JSON/error handling.
"""

import json
from pathlib import Path
import subprocess
from typing import Any, Callable

import anthropic
from dotenv import load_dotenv
import requests

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


@register_tool(
    name="write_file",
    description="Write UTF-8 text content to a file, creating parent directories if needed.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Full file contents to write.",
            },
        },
        "required": ["path", "content"],
    },
)
def write_file(path: str, content: str) -> str:
    try:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to '{path}'."
    except Exception as exc:
        return f"Error writing file '{path}': {exc}"


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


@register_tool(
    name="http_request",
    description="Make an HTTP request and return status, headers, and body.",
    input_schema={
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": "HTTP method, e.g. GET, POST, PUT, PATCH, DELETE.",
            },
            "url": {
                "type": "string",
                "description": "Absolute URL to request.",
            },
            "headers": {
                "type": "object",
                "description": "Optional request headers as key/value strings.",
                "additionalProperties": {"type": "string"},
            },
            "body": {
                "type": "string",
                "description": "Optional request body as a string (often JSON).",
            },
        },
        "required": ["method", "url"],
    },
)
def http_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: str | None = None,
) -> str:
    method_upper = method.upper()
    request_headers = headers or {}
    request_kwargs: dict[str, Any] = {
        "method": method_upper,
        "url": url,
        "headers": request_headers,
        "timeout": 20,
    }

    if body is not None:
        content_type = request_headers.get("Content-Type", "").lower()
        if "application/json" in content_type:
            try:
                request_kwargs["json"] = json.loads(body)
            except json.JSONDecodeError:
                request_kwargs["data"] = body
        else:
            request_kwargs["data"] = body

    try:
        response = requests.request(**request_kwargs)
    except requests.RequestException as exc:
        return f"HTTP request error: {exc}"

    response_headers = dict(response.headers)
    content_type = response_headers.get("Content-Type", "").lower()

    if "application/json" in content_type:
        try:
            response_body = json.dumps(response.json(), indent=2)
        except ValueError:
            response_body = response.text
    else:
        response_body = response.text

    return (
        f"status_code: {response.status_code}\n"
        f"url: {response.url}\n"
        f"headers:\n{json.dumps(response_headers, indent=2)}\n\n"
        f"body:\n{response_body}"
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

    print("API Tool Agent (type 'quit' to exit)")
    print("-" * 40)
    print("Available tools:", ", ".join(sorted(TOOL_HANDLERS.keys())))
    print("Try: Fetch https://api.github.com and summarize the response.")

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
