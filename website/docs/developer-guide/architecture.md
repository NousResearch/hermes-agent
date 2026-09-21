---
sidebar_position: 1
title: "Architecture"
description: "Hermes Agent internals — runtime boundaries, execution paths, data flow, and where to read next"
---

# Architecture

Hermes is one agent runtime exposed through several interfaces. The Python core owns
conversation state, model calls, tool execution, and persistence; the CLI, TUI, Desktop,
dashboard, gateway, ACP adapter, and batch runner provide different ways to reach that core.

The filesystem is the source of truth for the complete tree. The map below intentionally lists
load-bearing directories and facades rather than every module.

## Runtime boundaries

```text
  classic CLI ───────────────┐
  messaging gateway ─────────┤
  ACP adapter ────────────────┤
  batch runner ───────────────┤
                                ▼
                         AIAgent runtime
                         (run_agent.py)
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
       prompt/context     provider runtime      tool registry
       agent/             agent/ + providers/   tools/ + toolsets.py
              │                 │                 │
              └─────────────────┼─────────────────┘
                                ▼
                    sessions, files, and backends
                    hermes_state.py / tools/environments/

  TUI ──────────────── ui-tui/ ────────┐
  Desktop ──────────── apps/desktop/ ──┼─ tui_gateway/
  Dashboard ────────── web/ ───────────┘  (RPC/backend boundary)
```

The interfaces do not each implement a second agent. They enter the same runtime with a
surface-specific transport, callback set, session source, and toolset selection.

| Surface | Main code | Boundary to the shared runtime |
| --- | --- | --- |
| Classic CLI | `hermes_cli/main.py`, `cli.py` | Direct Python calls into `run_agent.py` |
| TUI | `ui-tui/` | `tui_gateway/entry.py` over newline-delimited JSON-RPC on stdio |
| Desktop | `apps/desktop/`, `apps/shared/` | A headless `hermes serve` process exposing the `tui_gateway` API |
| Web dashboard | `web/`, `hermes_cli/web_server.py`, `hermes_cli/web_routers/` | FastAPI routes and the dashboard's chat transport |
| Messaging gateway | `gateway/`, `plugins/platforms/` | `GatewayRunner` creates and routes agent sessions |
| ACP | `acp_adapter/` | Editor-facing stdio/JSON-RPC adapter |
| Batch processing | `batch_runner.py` | Direct agent runs for trajectory generation and evaluation |

## Directory structure

```text
hermes-agent/
├── run_agent.py              # AIAgent facade; the turn loop lives in agent/turn_*.py
├── cli.py                    # Classic CLI facade and REPL integration
├── model_tools.py            # Tool discovery, schema collection, and dispatch
├── toolsets.py               # Tool groups, platform bundles, and surface selection
├── hermes_state.py           # Session/state database facade; siblings are hermes_state_*.py
├── hermes_constants.py       # Profile-aware Hermes home and path helpers
├── hermes_logging.py         # Profile-aware agent, gateway, and error logs
├── batch_runner.py           # Parallel batch processing
│
├── agent/                    # Agent loop, turn phases, providers, memory, and compression
│   ├── agent_init.py         # Agent construction and runtime wiring
│   ├── conversation_loop.py  # Conversation orchestration
│   ├── prompt_builder.py     # System prompt assembly
│   └── turn_*.py             # Focused turn-loop phases
│
├── hermes_cli/               # CLI commands, setup, config, plugins, updater, and dashboard backend
│   ├── main.py               # `hermes` entry point and top-level command dispatch
│   ├── cli_*_mixin.py        # Classic CLI topics split from cli.py
│   ├── web_server.py         # Dashboard server composition
│   └── web_routers/          # FastAPI routers grouped by dashboard surface
│
├── tools/                    # Model-facing tools and the registry that discovers them
│   ├── registry.py           # Registration, availability, and toolset metadata
│   ├── browser_tool.py       # Browser facade and browser_tool_*.py siblings
│   ├── mcp_tool.py           # MCP facade and mcp_tool_*.py siblings
│   └── environments/         # Local, Docker, SSH, Modal, Daytona, Singularity, Vercel
│
├── gateway/                  # Messaging gateway, sessions, delivery, and direct adapters
│   ├── run.py                # GatewayRunner facade and run_*.py phases
│   ├── platforms/            # Shared base plus built-in/direct adapters
│   └── builtin_hooks/        # Extension point for always-registered gateway hooks
├── plugins/                  # Bundled extension categories and plugin implementations
│   ├── platforms/            # Bundled messaging platform adapters
│   ├── model-providers/      # ProviderProfile implementations
│   ├── memory/               # Persistent-memory backends
│   ├── context_engine/       # Context compression engines
│   └── ...                   # Browser, web, image/video, cron, dashboard, and other plugins
├── providers/                # ProviderProfile registry and common provider contract
├── skills/                   # Bundled skills
├── optional-skills/          # Shipped skills that are installed explicitly
├── optional-mcps/            # Optional MCP server definitions
│
├── tui_gateway/              # Python JSON-RPC/WebSocket backend for TUI and Desktop
├── ui-tui/                   # React/Ink terminal UI
├── apps/
│   ├── desktop/              # Electron desktop application
│   ├── shared/               # Shared client transport and protocol code
│   └── bootstrap-installer/  # Desktop/bootstrap installation surface
├── web/                      # React dashboard frontend
│
├── acp_adapter/              # ACP server for VS Code, Zed, and JetBrains
├── cron/                     # Scheduled agent jobs and scheduler
├── evals/                    # Offline evaluation and benchmark harnesses
├── native/                   # Native helpers and extensions
├── scripts/                  # Test, release, CI, and maintenance scripts
├── website/                  # Docusaurus documentation site
├── tests/                    # Python tests, organized to mirror source areas
└── tests-js/                 # JavaScript/TypeScript tests and test utilities
```

Several root files and large historical modules are kept as public facades. New logic belongs in
a topical sibling where one exists. Current examples include:

- `run_agent.py` with `agent/agent_init.py`, `agent/conversation_loop.py`, and `agent/turn_*.py`
- `cli.py` with `hermes_cli/cli_*_mixin.py`
- `hermes_state.py` with `hermes_state_*.py`
- `gateway/run.py` with `gateway/run_*.py`
- `tools/browser_tool.py` and `tools/mcp_tool.py` with their sibling modules
- `hermes_cli/web_server.py` with the routers in `hermes_cli/web_routers/`

This facade/sibling layout keeps public imports stable while letting each topic evolve in a
focused module. For contributor rules and the routing table for area-specific instructions, see
the repository [AGENTS.md](https://github.com/NousResearch/hermes-agent/blob/main/AGENTS.md).

## Execution paths

### Classic CLI session

```text
User input
  → HermesCLI (`cli.py` and hermes_cli/cli_*_mixin.py)
  → AIAgent (`run_agent.py`)
  → prompt_builder + provider runtime
  → model API
  → model_tools.handle_function_call() for tool rounds
  → response callbacks and hermes_state.py persistence
```

### TUI and Desktop session

The TUI and Desktop have different client shells but share the Python gateway boundary:

```text
TUI:     ui-tui/ → python -m tui_gateway.entry → tui_gateway/ → AIAgent
Desktop: apps/desktop/ → hermes serve → tui_gateway/ → AIAgent
```

The TUI uses newline-delimited JSON-RPC over stdio. Desktop uses the headless server's
JSON-RPC/WebSocket API. Both retain the same agent, session, tool, and profile semantics as the
other surfaces.

### Web dashboard

```text
Browser → web/ SPA → hermes_cli/web_server.py
        → hermes_cli/web_routers/ and dashboard services
        → shared config, sessions, gateway controls, and agent runtime
```

The dashboard is an administration surface over the same profile and state directories. Its
chat path reuses the TUI/backend machinery where documented by the dashboard implementation.

### Messaging gateway

```text
Platform event
  → adapter in gateway/platforms/ or plugins/platforms/
  → GatewayRunner (`gateway/run.py`)
  → authorization and session-key resolution
  → AIAgent with the gateway's toolset and history
  → delivery back through the adapter
```

`gateway/platforms/` contains shared and direct adapters. Most bundled channels live under
`plugins/platforms/` and are discovered through the gateway's platform registry.

### Cron job

```text
Scheduler tick
  → load a due job from cron state
  → create a fresh AIAgent
  → inject attached skills/context
  → run the prompt
  → deliver to the configured target
  → persist the next run and job state
```

## Major subsystems

### Agent loop and prompt system

`run_agent.py` is the public agent facade. The orchestration is split across
`agent/conversation_loop.py` and `agent/turn_*.py`; prompt assembly lives in
`agent/prompt_builder.py` and `agent/system_prompt.py`. Compression, caching, provider adapters,
memory, callbacks, and persistence are separate topics under `agent/`.

Read [Agent Loop Internals](./agent-loop.md), [Prompt Assembly](./prompt-assembly.md), and
[Context Compression & Prompt Caching](./context-compression-and-caching.md) next.

### Provider resolution

Provider profiles are registered through `providers/` and implemented by bundled or user plugins
under `plugins/model-providers/`. Runtime resolution is shared by the CLI, gateway, cron, ACP,
Desktop/TUI, dashboard, and auxiliary calls. It maps the selected provider/model to the API mode,
credentials, endpoint, and provider-specific request behavior.

Read [Provider Runtime Resolution](./provider-runtime.md), [Adding Providers](./adding-providers.md),
and [Model Provider Plugins](./model-provider-plugin.md).

### Tool system

`tools/registry.py` is the central registry. Tool modules register metadata and handlers; the
registry exposes availability and toolset information, while `model_tools.py` selects schemas and
dispatches calls. `toolsets.py` defines shared bundles and surface-specific platform toolsets.

Terminal execution is implemented under `tools/environments/` and currently covers local, Docker,
SSH, Modal, Daytona, Singularity, and Vercel Sandbox environments. Browser, web, MCP, delegation,
file, code execution, and platform-specific capabilities follow the same registry/toolset path.

Read [Tools Runtime](./tools-runtime.md) and [Adding Tools](./adding-tools.md).

### Session persistence

`hermes_state.py` is the session/state facade, with topic-specific `hermes_state_*.py` siblings.
The database uses SQLite and FTS5 for session search, keeps platform and profile boundaries, and
tracks session lineage across compression and related operations. Gateway routing adds its own
session-key and delivery concerns in `gateway/`.

Read [Session Storage](./session-storage.md) and [Gateway Internals](./gateway-internals.md).

### Plugin and skill system

`plugins/` contains extension categories with their own contracts: messaging platforms, model
providers, memory, context engines, browser, web search, image/video generation, cron providers,
dashboard auth, and other integrations. General plugin discovery and lifecycle are documented in
the [Plugin Guide](/developer-guide/plugins). `skills/` contains bundled skills;
`optional-skills/` contains shipped skills that are activated explicitly.

### ACP and trajectories

`acp_adapter/` exposes Hermes to editor clients through ACP over stdio/JSON-RPC. `batch_runner.py`
and the trajectory utilities generate and process agent runs for offline evaluation and training
workflows.

Read [ACP Internals](./acp-internals.md) and [Trajectories & Training Format](./trajectory-format.md).

## Design principles

| Principle | What it means in practice |
| --- | --- |
| **Prompt stability** | The system prompt stays byte-stable for a conversation. Context compression is the lifecycle exception; prompt-affecting mutations are deferred or start a new session so prompt caching remains valid. |
| **One runtime, many surfaces** | CLI, TUI, Desktop, dashboard, gateway, ACP, and batch use the same agent and state model. Surface-specific behavior belongs at the boundary. |
| **Observable execution** | Tool calls and progress are surfaced through the callbacks and transports appropriate to the active interface. |
| **Interruptible work** | API calls and tool execution can be interrupted by user input, commands, or signals where the surface supports it. |
| **Narrow core, extensible edges** | Registries, toolsets, plugins, and skills carry optional capability without expanding the always-loaded core unnecessarily. |
| **Profile isolation** | Each profile has its own configuration, credentials, memory, skills, sessions, logs, and gateway process state. |

## Dependency chain

```text
tools/registry.py       # low-level registry
        ↑
tools/*.py              # tool modules register handlers and metadata
        ↑
model_tools.py          # discovery, schema selection, and dispatch
        ↑
run_agent.py / cli.py / batch_runner.py / gateway/ / tui_gateway/
```

Terminal environment implementations under `tools/environments/` are selected by the terminal
tool and do not form a second model-facing registry. The important boundary is that tool
registration and toolset selection happen before an agent turn is sent to the model, while the
surface chooses which toolsets are available for that session.

## Recommended reading order

If you are new to the codebase:

1. **This page** — runtime boundaries and repository map
2. **[Contributing](./contributing.md)** — development setup and repository rules
3. **[Agent Loop Internals](./agent-loop.md)** — conversation orchestration
4. **[Prompt Assembly](./prompt-assembly.md)** — system prompt construction
5. **[Provider Runtime Resolution](./provider-runtime.md)** — provider selection
6. **[Tools Runtime](./tools-runtime.md)** — registry, toolsets, and environments
7. **[Gateway Internals](./gateway-internals.md)** — messaging and session routing
8. **[Session Storage](./session-storage.md)** — SQLite schema and lineage
9. **[CLI Internals](./cli-internals.md)** — CLI decomposition and lifecycle rules
10. **[TUI & Desktop from Worktrees](./worktree-ui-dev.md)** — frontend development across worktrees
