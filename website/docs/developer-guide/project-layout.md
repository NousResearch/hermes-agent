---
sidebar_position: 2
title: "Project Layout"
description: "Source entry points and profile-aware user state paths"
---

# Project Layout

## Project Structure

Counts shift constantly; the filesystem is canonical. Load-bearing entry points:

```
hermes-agent/
├── run_agent.py          # AIAgent facade; the turn loop lives in agent/turn_*.py
├── model_tools.py        # Tool orchestration, discover_builtin_tools(), handle_function_call()
├── toolsets.py           # TOOLSETS dict, _HERMES_CORE_TOOLS
├── cli.py                # HermesCLI (REPL, slash dispatch) + hermes_cli/cli_*_mixin.py
├── hermes_state.py       # SessionDB facade; hermes_state_*.py siblings
├── hermes_constants.py   # get_hermes_home(), display_hermes_home() — profile-aware paths
├── agent/                # turn loop phases, providers, memory, compression, prompt builder
├── hermes_cli/           # CLI subcommands, setup, config, plugins loader, updater, web_routers/
├── tools/                # Tool implementations (tools/registry.py) + environments/ backends
├── gateway/              # run.py facade + run_*.py phases + session*.py + platforms/
├── plugins/              # memory/, context_engine/, model-providers/, kanban/, image_gen/, ...
├── skills/               # Built-in skills (by category)   optional-skills/: shipped, not active
├── ui-tui/, tui_gateway/ # Ink terminal UI + its Python JSON-RPC backend (also serves Desktop)
├── apps/desktop/         # Electron desktop app (+ apps/shared)   web/: dashboard SPA
├── cron/                 # jobs.py + scheduler.py (+ scheduler_*.py)
├── pm/, hermes_platform/ # dependency/environment manager; machine facts + executable lookup
├── scripts/              # check, run_tests.sh, code_health/, ci/
├── website/              # Docusaurus docs (developer-guide/ holds the long-form area docs)
└── tests/                # Pytest suite, mirrors the source tree
```

**User state:** `~/.hermes/config.yaml` (settings), `.env` (secrets only), `logs/` (`hermes logs`);
all profile-aware via `get_hermes_home()`.
