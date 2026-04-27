---
title: "Code Mode"
sidebar_label: "Code Mode"
sidebar_position: 2
---

# Hermes Code Mode

Hermes Code Mode is the coding-focused surface for Hermes Agent. It adds a development console home screen, a HermesWeb Code Cockpit at `/code`, and quick CLI commands for workspaces, code sessions, approvals, and coding skills.

## What you get

- **Code Console home screen** — startup dashboard with provider, model, profile, workspace, branch, backend status, Code Cockpit URL, DB schema, active sessions, approvals, and tool/skill counts.
- **Code Cockpit** — browser UI at `http://localhost:3001/code` for coding sessions and workflow state.
- **REST-backed coding services** — workspaces, sessions, command execution, Git status/diff, diagnostics, model routing, multi-agent flows, coding skills, and approvals.
- **Safe degraded startup** — the CLI still renders when the backend is offline, the workspace is not a Git repo, or the state database is not initialized.

## Quick start

Start Hermes from your project workspace:

```bash
hermes
```

Open the cockpit:

```text
http://localhost:3001/code
```

Use `/web` inside the CLI to print the expected local URLs:

| Service | URL |
|---------|-----|
| Frontend | `http://localhost:3001` |
| Code Cockpit | `http://localhost:3001/code` |
| Backend API | `http://localhost:9119` |
| Backend status | `http://localhost:9119/api/status` |

## Code Mode slash commands

These commands are interactive-CLI helpers registered under the `Code Mode` command category.

| Command | Description |
|---------|-------------|
| `/code` | Show Code Mode help, cockpit URL, backend URL, and related commands. |
| `/web` | Show HermesWeb/backend URLs and local log paths. |
| `/workspace` | Show the current workspace path, Git branch, and cockpit URL. |
| `/session` | Show active code sessions from `GET /api/code/sessions`; gracefully handles offline/auth-required backends. |
| `/approvals` | Show pending approvals from `GET /api/approvals`; gracefully handles offline/auth-required backends. |
| `/skills-code` | List coding workflow skills. Alias: `/skillscode`. |

## Coding skills

`/skills-code` lists the coding-oriented workflow shortcuts:

- `fix_build`
- `review_diff`
- `stabilize_hanging_task`
- `fix_runtime_error`
- `implement_feature`
- `refactor_react_page`
- `benchmark_provider`

These are intended as fast entry points for common development tasks such as fixing builds, reviewing diffs, implementing features, stabilizing stuck tasks, and comparing providers.

## Architecture

```text
Hermes CLI
  ├─ Code Console banner
  ├─ Code Mode slash commands
  └─ normal agent loop, tools, skills, approvals

HermesWeb
  └─ /code Code Cockpit
       ├─ workspaces and sessions
       ├─ command logs and artifacts
       ├─ Git status/diff
       ├─ diagnostics/LSP
       ├─ provider/model controls
       ├─ multi-agent flows
       └─ approvals and skill runs

Backend API
  ├─ /api/code/*
  ├─ /api/providers*
  ├─ /api/approvals*
  └─ /ws realtime support
```

The backend exposes WebSocket support at `/ws`, but the current Code Cockpit integration is valid through REST polling. A dedicated frontend WebSocket client is a future optimization, not a requirement for daily use.

## CLI home screen fields

The Code Mode home screen shows:

- provider and model
- active profile
- workspace path
- Git branch when available
- session ID
- backend status and port
- Code Cockpit URL
- database schema version
- context length when available
- active code sessions and pending approvals
- tool and skill counts

All network calls use short timeouts and safe fallbacks so the CLI can start even when the backend is unavailable.

## API areas

Code Mode uses the backend in `hermes_cli/web_server.py` for:

- Workspaces: `/api/code/workspaces*`
- Sessions and events: `/api/code/sessions*`
- Commands: `/api/code/commands*`
- Git: `/api/code/workspaces/{workspace_id}/git/*`
- Diagnostics/LSP: `/api/code/workspaces/{workspace_id}/diagnostics*`, `/languages`, `/lsp/restart`
- Multi-agent flows: `/api/code/agent-flows*`
- Coding skills: `/api/code/skills*`, `/api/code/skill-runs*`
- Providers: `/api/providers*`
- Approvals: `/api/approvals*`
- Realtime hub: `/ws`

For the complete local developer reference, see `docs/HERMES_CODE_MODE.md` in the repository root.

## Verification

Targeted test command for Code Mode CLI/banner behavior:

```bash
source venv/bin/activate
python3 -m pytest \
  tests/hermes_cli/test_banner.py \
  tests/hermes_cli/test_code_mode_commands.py \
  --override-ini="addopts=" --tb=short
```

Website docs build, when dependencies are already installed:

```bash
cd website
npm run build
```

## Known limitations

- The frontend Code Cockpit currently relies on REST polling; backend WebSocket support exists but is not the primary UI path yet.
- Full end-to-end validation requires frontend, backend, auth/config, and local state to be running together.
- `/session` and `/approvals` intentionally degrade gracefully when the backend is offline or returns `401`/`403`.
