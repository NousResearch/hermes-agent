---
sidebar_position: 1
title: "Architecture"
description: "Hermes Agent internals — project structure, agent loop, key classes, and design patterns"
---

# Architecture

This guide covers the internal architecture of Hermes Agent for developers contributing to the project.

## Project Structure

```
hermes-agent/
├── run_agent.py              # AIAgent class — core conversation loop, tool dispatch
├── cli.py                    # HermesCLI class — interactive TUI, prompt_toolkit
├── model_tools.py            # Tool orchestration (thin layer over tools/registry.py)
├── toolsets.py               # Tool groupings and presets
├── hermes_state.py           # SQLite session database with FTS5 full-text search
├── batch_runner.py           # Parallel batch processing for trajectory generation
│
├── agent/                    # Agent internals (extracted modules)
│   ├── prompt_builder.py         # System prompt assembly (identity, skills, memory)
│   ├── context_compressor.py     # Auto-summarization when approaching context limits
│   ├── auxiliary_client.py       # Resolves auxiliary OpenAI clients (summarization, vision)
│   ├── display.py                # KawaiiSpinner, tool progress formatting
│   ├── model_metadata.py         # Model context lengths, token estimation
│   └── trajectory.py             # Trajectory saving helpers
│
├── hermes_cli/               # CLI command implementations
│   ├── main.py                   # Entry point, argument parsing, command dispatch
│   ├── config.py                 # Config management, migration, env var definitions
│   ├── setup.py                  # Interactive setup wizard
│   ├── auth.py                   # Provider resolution, OAuth, Nous Portal
│   ├── models.py                 # OpenRouter model selection lists
│   ├── banner.py                 # Welcome banner, ASCII art
│   ├── commands.py               # Slash command definitions + autocomplete
│   ├── callbacks.py              # Interactive callbacks (clarify, sudo, approval)
│   ├── doctor.py                 # Diagnostics
│   └── skills_hub.py             # Skills Hub CLI + /skills slash command handler
│
├── tools/                    # Tool implementations (self-registering)
│   ├── registry.py               # Central tool registry (schemas, handlers, dispatch)
│   ├── approval.py               # Dangerous command detection + per-session approval
│   ├── terminal_tool.py          # Terminal orchestration (sudo, env lifecycle, backends)
│   ├── file_operations.py        # read_file, write_file, search, patch
│   ├── web_tools.py              # web_search, web_extract
│   ├── vision_tools.py           # Image analysis via multimodal models
│   ├── delegate_tool.py          # Subagent spawning and parallel task execution
│   ├── code_execution_tool.py    # Sandboxed Python with RPC tool access
│   ├── session_search_tool.py    # Search past conversations
│   ├── cronjob_tools.py          # Scheduled task management
│   ├── skills_tool.py             # Skill search and load
│   ├── skill_manager_tool.py      # Skill management
│   └── environments/             # Terminal execution backends
│       ├── base.py                   # BaseEnvironment ABC
│       ├── local.py, docker.py, ssh.py, singularity.py, modal.py
│
├── gateway/                  # Messaging gateway
│   ├── run.py                    # GatewayRunner — platform lifecycle, message routing
│   ├── config.py                 # Platform configuration resolution
│   ├── session.py                # Session store, context prompts, reset policies
│   └── platforms/                # Platform adapters
│       ├── telegram.py, discord_adapter.py, slack.py, whatsapp.py
│
├── scripts/                  # Installer and bridge scripts
│   ├── install.sh                # Linux/macOS installer
│   ├── install.ps1               # Windows PowerShell installer
│   └── whatsapp-bridge/          # Node.js WhatsApp bridge (Baileys)
│
├── skills/                   # Bundled skills (copied to ~/.hermes/skills/)
├── environments/             # RL training environments (Atropos integration)
└── tests/                    # Test suite
```

## Core Loop

The main agent loop lives in `run_agent.py`:

```
User message → AIAgent._run_agent_loop()
  ├── Build system prompt (prompt_builder.py)
  ├── Build API kwargs (model, messages, tools, reasoning config)
  ├── Call LLM (OpenAI-compatible API)
  ├── If tool_calls in response:
  │     ├── Execute each tool via registry dispatch
  │     ├── Add tool results to conversation
  │     └── Loop back to LLM call
  ├── If text response:
  │     ├── Persist session to DB
  │     └── Return final_response
  └── Context compression if approaching token limit
```

```python
while turns < max_turns:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tool_schemas,
    )

    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = await execute_tool(tool_call)
            messages.append(tool_result_message(result))
        turns += 1
    else:
        return response.content
```

## AIAgent Class

```python
class AIAgent:
    def __init__(
        self,
        model: str = "anthropic/claude-opus-4.6",
        api_key: str = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_iterations: int = 60,
        enabled_toolsets: list = None,
        disabled_toolsets: list = None,
        verbose_logging: bool = False,
        quiet_mode: bool = False,
        tool_progress_callback: callable = None,
    ):
        ...

    def chat(self, message: str) -> str:
        # Main entry point - runs the agent loop
        ...
```

## File Dependency Chain

```
tools/registry.py  (no deps — imported by all tool files)
       ↑
tools/*.py  (each calls registry.register() at import time)
       ↑
model_tools.py  (imports tools/registry + triggers tool discovery)
       ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

Each tool file co-locates its schema, handler, and registration. `model_tools.py` is a thin orchestration layer.

## Key Design Patterns

### Self-Registering Tools

Each tool file calls `registry.register()` at import time. `model_tools.py` triggers discovery by importing all tool modules.

### Toolset Grouping

Tools are grouped into toolsets (`web`, `terminal`, `file`, `browser`, etc.) that can be enabled/disabled per platform.

### Session Persistence

All conversations are stored in SQLite (`hermes_state.py`) with full-text search. JSON logs go to `~/.hermes/sessions/`.

### Ephemeral Injection

System prompts and prefill messages are injected at API call time, never persisted to the database or logs.

### Provider Abstraction

The agent works with any OpenAI-compatible API. Provider resolution happens at init time (Nous Portal OAuth, OpenRouter API key, or custom endpoint).

### Conversation Format

Messages follow the OpenAI format:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Search for Python tutorials"},
    {"role": "assistant", "content": None, "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."},
    {"role": "assistant", "content": "Here's what I found..."},
]
```

## CLI Architecture

The interactive CLI (`cli.py`) uses:

- **Rich** — Welcome banner and styled panels
- **prompt_toolkit** — Fixed input area with history, `patch_stdout`, slash command autocomplete
- **KawaiiSpinner** — Animated kawaii faces during API calls; clean activity feed for tool results

Key UX behaviors:

- Thinking spinner shows animated kawaii face + verb (`(⌐■_■) deliberating...`)
- Tool execution results appear as `┊ {emoji} {verb} {detail} {duration}`
- Prompt shows `⚕ ❯` when working, `❯` when idle
- Pasting 5+ lines auto-saves to `~/.hermes/pastes/` and collapses

## Messaging Gateway Architecture

The gateway (`gateway/run.py`) uses `GatewayRunner` to:

1. Connect to all configured platforms
2. Route messages through per-chat session stores
3. Dispatch to AIAgent instances
4. Run the cron scheduler (ticks every 60s)
5. Handle interrupts and tool progress notifications

Each platform adapter conforms to `BasePlatformAdapter`.

## Configuration System

- `~/.hermes/config.yaml` — All settings
- `~/.hermes/.env` — API keys and secrets
- `_config_version` in `DEFAULT_CONFIG` — Bumped when required fields are added, triggers migration prompts
