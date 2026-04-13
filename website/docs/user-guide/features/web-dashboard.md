---
sidebar_position: 15
title: "Web Dashboard"
description: "Browser-based dashboard for managing configuration, API keys, and monitoring sessions"
---

# Web Dashboard

The web dashboard is a browser-based UI for managing your Hermes Agent installation. Instead of editing YAML files or running CLI commands, you can configure settings, manage API keys, and monitor sessions from a clean web interface.

## Quick Start

```bash
hermes web
```

This starts a local web server and opens `http://127.0.0.1:9119` in your browser. The dashboard runs entirely on your machine — no data leaves localhost.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `9119` | Port to run the web server on |
| `--host` | `127.0.0.1` | Bind address |
| `--no-open` | — | Don't auto-open the browser |

```bash
# Custom port
hermes web --port 8080

# Bind to all interfaces (use with caution on shared networks)
hermes web --host 0.0.0.0

# Start without opening browser
hermes web --no-open
```

## Prerequisites

The web dashboard requires FastAPI and Uvicorn. Install them with:

```bash
pip install hermes-agent[web]
```

If you installed with `pip install hermes-agent[all]`, the web dependencies are already included.

When you run `hermes web` without the dependencies, it will tell you what to install. If the frontend hasn't been built yet and `npm` is available, it builds automatically on first launch.

## Pages

### Status

The landing page shows a live overview of your installation:

- **Agent version** and release date
- **Gateway status** — running/stopped, PID, connected platforms and their state
- **Active sessions** — count of sessions active in the last 5 minutes
- **Recent sessions** — list of the 20 most recent sessions with model, message count, token usage, and a preview of the conversation

The status page auto-refreshes every 5 seconds.

### Config

A form-based editor for `config.yaml`. All 150+ configuration fields are auto-discovered from `DEFAULT_CONFIG` and organized into tabbed categories:

- **model** — default model, provider, base URL, reasoning settings
- **terminal** — backend (local/docker/ssh/modal), timeout, shell preferences
- **display** — skin, tool progress, resume display, spinner settings
- **agent** — max iterations, gateway timeout, service tier
- **delegation** — subagent limits, reasoning effort
- **memory** — provider selection, context injection settings
- **approvals** — dangerous command approval mode (ask/yolo/deny)
- And more — every section of config.yaml has corresponding form fields

Fields with known valid values (terminal backend, skin, approval mode, etc.) render as dropdowns. Booleans render as toggles. Everything else is a text input.

**Actions:**

- **Save** — writes changes to `config.yaml` immediately
- **Reset to defaults** — reverts all fields to their default values (doesn't save until you click Save)
- **Export** — downloads the current config as JSON
- **Import** — uploads a JSON config file to replace the current values

:::tip
Config changes take effect on the next agent session or gateway restart. The web dashboard edits the same `config.yaml` file that `hermes config set` and the gateway read from.
:::

### API Keys

Manage the `.env` file where API keys and credentials are stored. Keys are grouped by category:

- **LLM Providers** — OpenRouter, Anthropic, OpenAI, DeepSeek, etc.
- **Tool API Keys** — Browserbase, Firecrawl, Tavily, ElevenLabs, etc.
- **Messaging Platforms** — Telegram, Discord, Slack bot tokens, etc.
- **Agent Settings** — non-secret env vars like `API_SERVER_ENABLED`

Each key shows:
- Whether it's currently set (with a redacted preview of the value)
- A description of what it's for
- A link to the provider's signup/key page
- An input field to set or update the value
- A delete button to remove it

Advanced/rarely-used keys are hidden by default behind a toggle.

:::warning Security
The web dashboard reads and writes your `.env` file, which contains API keys and secrets. It binds to `127.0.0.1` by default — only accessible from your local machine. If you bind to `0.0.0.0`, anyone on your network can view and modify your credentials. The dashboard has no authentication of its own.
:::

## `/reload` Slash Command

The dashboard PR also adds a `/reload` slash command to the interactive CLI. After changing API keys via the web dashboard (or by editing `.env` directly), use `/reload` in an active CLI session to pick up the changes without restarting:

```
You → /reload
  Reloaded .env (3 var(s) updated)
```

This re-reads `~/.hermes/.env` into the running process's environment. Useful when you've added a new provider key via the dashboard and want to use it immediately.

## REST API

The web dashboard exposes a REST API that the frontend consumes. You can also call these endpoints directly for automation:

### GET /api/status

Returns agent version, gateway status, platform states, and active session count.

### GET /api/sessions

Returns the 20 most recent sessions with metadata (model, token counts, timestamps, preview).

### GET /api/config

Returns the current `config.yaml` contents as JSON.

### GET /api/config/defaults

Returns the default configuration values.

### GET /api/config/schema

Returns a schema describing every config field — type, description, category, and select options where applicable. The frontend uses this to render the correct input widget for each field.

### PUT /api/config

Saves a new configuration. Body: `{"config": {...}}`.

### GET /api/env

Returns all known environment variables with their set/unset status, redacted values, descriptions, and categories.

### PUT /api/env

Sets an environment variable. Body: `{"key": "VAR_NAME", "value": "secret"}`.

### DELETE /api/env

Removes an environment variable. Body: `{"key": "VAR_NAME"}`.

## CORS

The web server restricts CORS to localhost origins only:

- `http://localhost:9119` / `http://127.0.0.1:9119` (production)
- `http://localhost:3000` / `http://127.0.0.1:3000`
- `http://localhost:5173` / `http://127.0.0.1:5173` (Vite dev server)

If you run the server on a custom port, that origin is added automatically.

## Development

If you're contributing to the web dashboard frontend:

```bash
# Terminal 1: start the backend API
hermes web --no-open

# Terminal 2: start the Vite dev server with HMR
cd web/
npm install
npm run dev
```

The Vite dev server at `http://localhost:5173` proxies `/api` requests to the FastAPI backend at `http://127.0.0.1:9119`.

The frontend is built with React 19, TypeScript, Tailwind CSS v4, and shadcn/ui-style components. Production builds output to `hermes_cli/web_dist/` which the FastAPI server serves as a static SPA.

## Automatic Build on Update

When you run `hermes update`, the web frontend is automatically rebuilt if `npm` is available. This keeps the dashboard in sync with code updates. If `npm` isn't installed, the update skips the frontend build and `hermes web` will build it on first launch.
