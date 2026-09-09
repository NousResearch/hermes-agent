---
title: "Codebase Ownership Map"
description: "Which directories belong to which subsystem, and where the right docs entry point lives for each"
---

# Codebase Ownership Map

Hermes is a large repository, and most contributions touch exactly one subsystem. This page maps each subsystem to its source directories and the documentation entry point you should read before changing it. Use it to find the right starting doc, the right place for a change, and the right test directory (tests mirror source: code in `tools/` is tested in `tests/tools/`, plugins in `tests/plugins/<type>/`, and so on).

| Subsystem | Source directories | Docs entry point |
|-----------|-------------------|------------------|
| Agent core (loop, transports, compression) | `agent/`, `run_agent.py` | [Agent Loop](agent-loop.md), [Context Compression & Caching](context-compression-and-caching.md) |
| Prompt assembly | `agent/prompt_builder.py`, `agent/system_prompt.py` | [Prompt Assembly](prompt-assembly.md) |
| Model providers & transports | `agent/transports/`, `plugins/model-providers/`, `hermes_cli/models.py` | [Adding Providers](adding-providers.md), [Model Provider Plugins](model-provider-plugin.md), [Provider Runtime](provider-runtime.md) |
| Built-in tools | `tools/` | [Adding Tools](adding-tools.md), [Tools Runtime](tools-runtime.md) |
| Messaging gateway | `gateway/`, `plugins/platforms/` | [Gateway Internals](gateway-internals.md), [Adding Platform Adapters](adding-platform-adapters.md) |
| CLI | `hermes_cli/` | [Extending the CLI](extending-the-cli.md) |
| Plugins system | `plugins/` | [Build a Hermes Plugin](plugins/index.md) |
| Skills (bundled & optional) | `skills/`, `optional-skills/` | [Creating Skills](creating-skills.md) |
| Cron / scheduled jobs | `cron/` | [Cron Internals](cron-internals.md) |
| Session storage | `hermes_state.py`, `hermes_state_*.py` | [Session Storage](session-storage.md) |
| Browser stack | `tools/browser_tool.py`, `tools/browser_supervisor.py`, `tools/browser_cdp_tool.py` | [Browser Supervisor](browser-supervisor.md) |
| Egress firewall | `agent/proxy_sources/iron_proxy.py` | [Egress Internals](egress-internals.md) |
| ACP (IDE integration) | `acp_adapter/` | [ACP Internals](acp-internals.md) |
| Desktop app | `apps/desktop/` | [Desktop Plugin SDK](desktop-plugin-sdk.md), [Worktree UI Development](worktree-ui-dev.md) |
| TUI | `ui-tui/`, `tui_gateway/` | [Worktree UI Development](worktree-ui-dev.md) |
| Docs site | `website/` | [Contributing](contributing.md) |
| Tests | `tests/`, `tests-js/` | [Contributing → Before Submitting](contributing.md#before-submitting) |

A few conventions that fall out of this map:

- **Changes should stay inside their subsystem.** A plugin that needs to edit core files is a design smell — widen the generic plugin surface instead (see the contribution rubric in the repository's `AGENTS.md`).
- **Run the mirror test directory for every source directory you touch.** A change to `plugins/platforms/telegram/` needs `tests/plugins/platforms/` green, not just the test file you happened to think of.
- **When two subsystems are involved, the narrower one owns the change.** Prefer a fix in an adapter or plugin over a branch in the agent core; the core is a narrow waist, and every addition there is paid for on every API call.
