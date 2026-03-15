---
sidebar_position: 1
title: "Architecture"
description: "Hermes Agent internals — major subsystems, execution paths, and where to read next"
---

# Architecture

This page is the top-level map of Hermes Agent internals. The project has grown beyond a single monolithic loop, so the best way to understand it is by subsystem.

## High-level structure

```text
hermes-agent/
├── run_agent.py              # AIAgent core loop
├── cli.py                    # interactive terminal UI
├── model_tools.py            # tool discovery/orchestration
├── toolsets.py               # tool groupings and presets
├── hermes_state.py           # SQLite session/state database
├── batch_runner.py           # batch trajectory generation
│
├── agent/                    # prompt building, compression, caching, metadata, trajectories
├── hermes_cli/               # command entrypoints, auth, setup, models, config, doctor
├── tools/                    # tool implementations and terminal environments
├── gateway/                  # messaging gateway, session routing, delivery, pairing, hooks
├── cron/                     # scheduled job storage and scheduler
├── honcho_integration/       # Honcho memory integration
├── acp_adapter/              # ACP editor integration server
├── acp_registry/             # ACP registry manifest + icon
├── environments/             # Hermes RL / benchmark environment framework
├── skills/                   # bundled skills
├── optional-skills/          # official optional skills
└── tests/                    # test suite
```

## Recommended reading order

If you are new to the codebase, read in this order:

1. this page
2. [Agent Loop Internals](./agent-loop.md)
3. [Prompt Assembly](./prompt-assembly.md)
4. [Provider Runtime Resolution](./provider-runtime.md)
5. [Adding Providers](./adding-providers.md)
6. [Tools Runtime](./tools-runtime.md)
7. [Session Storage](./session-storage.md)
8. [Gateway Internals](./gateway-internals.md)
9. [Context Compression & Prompt Caching](./context-compression-and-caching.md)
10. [ACP Internals](./acp-internals.md)
11. [Environments, Benchmarks & Data Generation](./environments.md)

## Major subsystems

### Agent loop

The core synchronous orchestration engine is `AIAgent` in `run_agent.py`.

It is responsible for:

- provider/API-mode selection
- prompt construction
- tool execution
- retries and fallback
- callbacks
- compression and persistence

See [Agent Loop Internals](./agent-loop.md).

### Prompt system

Prompt-building logic is split between:

- `run_agent.py`
- `agent/prompt_builder.py`
- `agent/prompt_caching.py`
- `agent/context_compressor.py`

See:

- [Prompt Assembly](./prompt-assembly.md)
- [Context Compression & Prompt Caching](./context-compression-and-caching.md)

### Provider/runtime resolution

Hermes has a shared runtime provider resolver used by CLI, gateway, cron, ACP, and auxiliary calls.

See [Provider Runtime Resolution](./provider-runtime.md).

### Tooling runtime

The tool registry, toolsets, terminal backends, process manager, and dispatch rules form a subsystem of their own.

See [Tools Runtime](./tools-runtime.md).

### Session persistence

Historical session state is stored primarily in SQLite, with lineage preserved across compression splits.

See [Session Storage](./session-storage.md).

### Messaging gateway

The gateway is a long-running orchestration layer for platform adapters, session routing, pairing, delivery, and cron ticking.

See [Gateway Internals](./gateway-internals.md).

### ACP integration

ACP exposes Hermes as an editor-native agent over stdio/JSON-RPC.

See:

- [ACP Editor Integration](../user-guide/features/acp.md)
- [ACP Internals](./acp-internals.md)

### Cron

Cron jobs are implemented as first-class agent tasks, not just shell tasks.

See [Cron Internals](./cron-internals.md).

### RL / environments / trajectories

Hermes ships a full environment framework for evaluation, RL integration, and SFT data generation.

See:

- [Environments, Benchmarks & Data Generation](./environments.md)
- [Trajectories & Training Format](./trajectory-format.md)

## Design themes

Several cross-cutting design themes appear throughout the codebase:

- prompt stability matters
- tool execution must be observable and interruptible
- session persistence must survive long-running use
- platform frontends should share one agent core
- optional subsystems should remain loosely coupled where possible

## Implementation notes

The older mental model of Hermes as “one OpenAI-compatible chat loop plus some tools” is no longer sufficient. Current Hermes includes:

- multiple API modes
- auxiliary model routing
- ACP editor integration
- gateway-specific session and delivery semantics
- RL environment infrastructure
- prompt-caching and compression logic with lineage-aware persistence

Use this page as the map, then dive into subsystem-specific docs for the real implementation details.
