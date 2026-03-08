"""
Lesson 9: Error Handling + Retries

An interactive CLI agent with read, write, shell, and HTTP tools.
Demonstrates retry/backoff, tool error propagation, and resilient execution.
"""

import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Callable

import anthropic
from dotenv import load_dotenv
import requests

load_dotenv()

client = anthropic.Anthropic()

ToolHandler = Callable[..., str]
TOOL_HANDLERS: dict[str, ToolHandler] = {}
TOOL_SCHEMAS: list[dict[str, Any]] = []

SYSTEM_PROMPT = """You are a practical coding assistant.
- Prefer concise, direct responses.
- Use tools when they are needed for factual verification or file/system actions.
- If a tool fails, explain the failure and propose the next best step.
"""

MAX_HISTORY_MESSAGES = 40
MIN_HISTORY_MESSAGES = 24
INPUT_TOKENS_SOFT_LIMIT = 20_000
SHELL_TIMEOUT_SECONDS = 30
HTTP_TIMEOUT_SECONDS = 20
API_MAX_RETRIES = 5
API_INITIAL_BACKOFF_SECONDS = 1.0
API_MAX_BACKOFF_SECONDS = 16.0


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
        return f"TOOL_ERROR: denied_by_user command={command}"

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"TOOL_ERROR: shell_timeout seconds={SHELL_TIMEOUT_SECONDS} command={command}"
    except Exception as exc:
        return f"TOOL_ERROR: shell_exec_failed command={command} error={exc}"

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
        "timeout": HTTP_TIMEOUT_SECONDS,
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
        return f"TOOL_ERROR: http_request_failed error={exc}"

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
        return f"TOOL_ERROR: unknown_tool name={tool_name}"

    try:
        return handler(**tool_input)
    except Exception as exc:
        return f"TOOL_ERROR: tool_execution_failed name={tool_name} error={exc}"


def truncate_history(messages: list[dict[str, Any]], keep_last: int) -> int:
    if len(messages) <= keep_last:
        return 0
    to_remove = len(messages) - keep_last
    del messages[:to_remove]
    return to_remove


def usage_stats(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    return input_tokens, output_tokens


def is_retryable_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 429 or 500 <= status_code < 600

    exc_name = exc.__class__.__name__.lower()
    retryable_names = {"ratelimiterror", "apiconnectionerror", "internalservererror"}
    return exc_name in retryable_names


def call_model_with_retries(messages: list[dict[str, Any]]) -> tuple[Any, list[str]]:
    backoff = API_INITIAL_BACKOFF_SECONDS
    last_exc: Exception | None = None

    for attempt in range(1, API_MAX_RETRIES + 1):
        streamed_text_parts: list[str] = []
        printed_any_text = False
        try:
            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    if not printed_any_text:
                        print("\nClaude: ", end="", flush=True)
                        printed_any_text = True
                    streamed_text_parts.append(text)
                    print(text, end="", flush=True)
                response = stream.get_final_message()

            if printed_any_text:
                print()
            return response, streamed_text_parts
        except Exception as exc:
            last_exc = exc
            if printed_any_text:
                print()
            retryable = is_retryable_api_error(exc)
            is_last_attempt = attempt == API_MAX_RETRIES
            if (not retryable) or is_last_attempt:
                raise

            sleep_seconds = min(backoff, API_MAX_BACKOFF_SECONDS) * random.uniform(0.9, 1.1)
            print(
                f"[api] Request failed ({exc.__class__.__name__}). "
                f"Retry {attempt}/{API_MAX_RETRIES} in {sleep_seconds:.1f}s."
            )
            time.sleep(sleep_seconds)
            backoff *= 2

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Model call failed without an exception.")


def run_agent_turn(messages: list[dict[str, Any]]) -> str:
    while True:
        removed = truncate_history(messages, MAX_HISTORY_MESSAGES)
        if removed:
            print(f"[context] Dropped {removed} old message(s) to keep history bounded.")

        try:
            response, streamed_text_parts = call_model_with_retries(messages)
        except Exception as exc:
            print(f"\n[api] Fatal API error: {exc}")
            return "API call failed."

        input_tokens, output_tokens = usage_stats(response)
        if input_tokens is not None and output_tokens is not None:
            print(f"[usage] input={input_tokens} output={output_tokens}")
            if input_tokens > INPUT_TOKENS_SOFT_LIMIT:
                removed_more = truncate_history(messages, MIN_HISTORY_MESSAGES)
                if removed_more:
                    print(
                        "[context] Input tokens exceeded soft limit; "
                        f"dropped {removed_more} additional old message(s)."
                    )

        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            streamed_text = "".join(streamed_text_parts).strip()
            if streamed_text:
                return streamed_text
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

    print("Robust Agent (type 'quit' to exit)")
    print("-" * 40)
    print("Available tools:", ", ".join(sorted(TOOL_HANDLERS.keys())))
    print("Retries, timeouts, usage stats, and context truncation are enabled.")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})
        run_agent_turn(messages)


if __name__ == "__main__":
    main()
