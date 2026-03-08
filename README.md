# Agent Loop Learning Project

This repository is a step-by-step learning project for building a coding agent in Python with the Anthropic SDK.

Each lesson is a standalone script that adds one major capability:
- `lessons/01_chat_loop.py` basic chat loop
- `lessons/02_tool_loop.py` first tool + tool loop
- `lessons/03_shell_tool.py` shell tool + confirmation
- `lessons/04_tool_registry.py` tool registry architecture
- `lessons/05_multi_step.py` write tool + multi-step workflows
- `lessons/06_api_tool.py` HTTP/API tool
- `lessons/07_streaming.py` streaming responses
- `lessons/08_context.py` system prompt + context management
- `lessons/09_robust.py` retries + resilience
- `lessons/10_secure.py` tool security controls
- `lessons/11_agent.py` polished CLI agent

The full curriculum and goals are in `docs/PLAN.md`.

## Quick Start

1. Install dependencies:
```bash
uv sync
```

2. Configure your API key:
```bash
cp .env.example .env
```
Then set `ANTHROPIC_API_KEY` in `.env`.

3. Run a lesson:
```bash
uv run python lessons/01_chat_loop.py
```
