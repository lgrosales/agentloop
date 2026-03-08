# Agent Loop Learning Project

This repository is a step-by-step learning project for building a coding agent in Python with the Anthropic SDK.

Each lesson is a standalone script that adds one major capability:
- `01_chat_loop.py` basic chat loop
- `02_tool_loop.py` first tool + tool loop
- `03_shell_tool.py` shell tool + confirmation
- `04_tool_registry.py` tool registry architecture
- `05_multi_step.py` write tool + multi-step workflows
- `06_api_tool.py` HTTP/API tool
- `07_streaming.py` streaming responses
- `08_context.py` system prompt + context management
- `09_robust.py` retries + resilience
- `10_secure.py` tool security controls
- `11_agent.py` polished CLI agent

The full curriculum and goals are in `PLAN.md`.

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

3. Run any lesson:
```bash
uv run python 01_chat_loop.py
```
or
```bash
uv run python 11_agent.py
```
