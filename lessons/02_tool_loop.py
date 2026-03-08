"""
Lesson 2: Tool Definitions + The Agent Loop

An interactive CLI agent that can call a `read_file` tool.
Demonstrates: tool schemas, tool_use/tool_result message types, and the loop.
"""

from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file and return its contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read.",
                }
            },
            "required": ["path"],
        },
    }
]


def read_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error reading file '{path}': {exc}"


def run_agent_turn(messages: list[dict]) -> str:
    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            text_blocks = [block.text for block in response.content if block.type == "text"]
            return "\n".join(text_blocks).strip()

        tool_results = []
        for tool_use in tool_uses:
            if tool_use.name == "read_file":
                result = read_file(tool_use.input["path"])
            else:
                result = f"Unknown tool: {tool_use.name}"

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                }
            )

        messages.append({"role": "user", "content": tool_results})


def main() -> None:
    messages: list[dict] = []

    print("Tool Agent (type 'quit' to exit)")
    print("-" * 40)
    print("Example: Read docs/PLAN.md and summarize lesson 2.")

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
