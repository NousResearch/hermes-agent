---
sidebar_position: 15
title: "Web Dashboard"
description: "Browser-based dashboard for managing configuration, API keys, sessions, logs, analytics, cron jobs, and skills"
---

# Web Dashboard

The web dashboard is a browser-based UI for managing your Hermes Agent installation. Instead of editing YAML files or running CLI commands, you can configure settings, manage API keys, and monitor sessions from a clean web interface.

## Quick Start

```bash
hermes dashboard
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
hermes dashboard --port 8080

# Bind to all interfaces (use with caution on shared networks)
hermes dashboard --host 0.0.0.0

# Start without opening browser
hermes dashboard --no-open
```

## Prerequisites

The default `hermes-agent` install does not ship the HTTP stack or PTY helper — those are optional extras. The **web dashboard** needs FastAPI and Uvicorn (`web` extra). The **Chat** tab also needs `ptyprocess` to spawn the embedded TUI behind a pseudo-terminal (`pty` extra on POSIX). Install both with:

```bash
pip install 'hermes-agent[web,pty]'
```

The `web` extra pulls in FastAPI/Uvicorn; `pty` pulls in `ptyprocess` (POSIX) or `pywinpty` (native Windows — note that the embedded TUI itself still requires WSL). `pip install hermes-agent[all]` includes both extras and is the easiest path if you also want messaging/voice/etc.

When you run `hermes dashboard` without the dependencies, it will tell you what to install. If the frontend hasn't been built yet and `npm` is available, it builds automatically on first launch.

## Pages

### Status

The landing page shows a live overview of your installation:

- **Agent version** and release date
- **Gateway status** — running/stopped, PID, connected platforms and their state
- **Active sessions** — count of sessions active in the last 5 minutes
- **Recent sessions** — list of the 20 most recent sessions with model, message count, token usage, and a preview of the conversation

The status page auto-refreshes every 5 seconds.

### Chat

The **Chat** tab embeds the full Hermes TUI (the same interface you get from `hermes --tui`) directly in the browser. Everything you can do in the terminal TUI — slash commands, model picker, tool-call cards, markdown streaming, clarify/sudo/approval prompts, skin theming — works identically here, because the dashboard is running the real TUI binary and rendering its ANSI output through [xterm.js](https://xtermjs.org/) with its WebGL renderer for pixel-perfect cell layout.

**How it works:**

- `/api/pty` opens a WebSocket authenticated with the dashboard's session token
- The server spawns `hermes --tui` behind a POSIX pseudo-terminal
- Keystrokes travel to the PTY; ANSI output streams back to the browser
- xterm.js's WebGL renderer paints each cell to an integer-pixel grid; mouse tracking (SGR 1006), wide characters (Unicode 11), and box-drawing glyphs all render natively
- Resizing the browser window resizes the TUI via the `@xterm/addon-fit` addon

**Resume an existing session:** from the **Sessions** tab, click the play icon (▶) next to any session. That jumps to `/chat?resume=<id>` and launches the TUI with `--resume`, loading the full history.

**Prerequisites:**

- Node.js (same requirement as `hermes --tui`; the TUI bundle is built on first launch)
- `ptyprocess` — installed by the `pty` extra (`pip install 'hermes-agent[web,pty]'`, or `[all]` covers both)
- POSIX kernel (Linux, macOS, or WSL). Native Windows Python is not supported — use WSL.

Close the browser tab and the PTY is reaped cleanly on the server. Re-opening spawns a fresh session.

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

### Sessions

Browse and inspect all agent sessions. Each row shows the session title, source platform icon (CLI, Telegram, Discord, Slack, cron), model name, message count, tool call count, and how long ago it was active. Live sessions are marked with a pulsing badge.

- **Search** — full-text search across all message content using FTS5. Results show highlighted snippets and auto-scroll to the first matching message when expanded.
- **Expand** — click a session to load its full message history. Messages are color-coded by role (user, assistant, system, tool) and rendered as Markdown with syntax highlighting.
- **Tool calls** — assistant messages with tool calls show collapsible blocks with the function name and JSON arguments.
- **Delete** — remove a session and its message history with the trash icon.

### Logs

View agent, gateway, and error log files with filtering and live tailing.

- **File** — switch between `agent`, `errors`, and `gateway` log files
- **Level** — filter by log level: ALL, DEBUG, INFO, WARNING, or ERROR
- **Component** — filter by source component: all, gateway, agent, tools, cli, or cron
- **Lines** — choose how many lines to display (50, 100, 200, or 500)
- **Auto-refresh** — toggle live tailing that polls for new log lines every 5 seconds
- **Color-coded** — log lines are colored by severity (red for errors, yellow for warnings, dim for debug)

### Analytics

Usage and cost analytics computed from session history. Select a time period (7, 30, or 90 days) to see:

- **Summary cards** — total tokens (input/output), cache hit percentage, total estimated or actual cost, and total session count with daily average
- **Daily token chart** — stacked bar chart showing input and output token usage per day, with hover tooltips showing breakdowns and cost
- **Daily breakdown table** — date, session count, input tokens, output tokens, cache hit rate, and cost for each day
- **Per-model breakdown** — table showing each model used, its session count, token usage, and estimated cost

### Cron

Create and manage scheduled cron jobs that run agent prompts on a recurring schedule.

- **Create** — fill in a name (optional), prompt, cron expression (e.g. `0 9 * * *`), and delivery target (local, Telegram, Discord, Slack, or email)
- **Job list** — each job shows its name, prompt preview, schedule expression, state badge (enabled/paused/error), delivery target, last run time, and next run time
- **Pause / Resume** — toggle a job between active and paused states
- **Trigger now** — immediately execute a job outside its normal schedule
- **Delete** — permanently remove a cron job

### Skills

Browse, search, and toggle skills and toolsets. Skills are loaded from `~/.hermes/skills/` and grouped by category.

- **Search** — filter skills and toolsets by name, description, or category
- **Category filter** — click category pills to narrow the list (e.g. MLOps, MCP, Red Teaming, AI)
- **Toggle** — enable or disable individual skills with a switch. Changes take effect on the next session.
- **Toolsets** — a separate section shows built-in toolsets (file operations, web browsing, etc.) with their active/inactive status, setup requirements, and list of included tools

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

### GET /api/sessions/\{session_id\}

Returns metadata for a single session.

### GET /api/sessions/\{session_id\}/messages

Returns the full message history for a session, including tool calls and timestamps.

### GET /api/sessions/search

Full-text search across message content. Query parameter: `q`. Returns matching session IDs with highlighted snippets.

### DELETE /api/sessions/\{session_id\}

Deletes a session and its message history.

### GET /api/logs

Returns log lines. Query parameters: `file` (agent/errors/gateway), `lines` (count), `level`, `component`.

### GET /api/analytics/usage

Returns token usage, cost, and session analytics. Query parameter: `days` (default 30). Response includes daily breakdowns and per-model aggregates.

### GET /api/cron/jobs

Returns all configured cron jobs with their state, schedule, and run history.

### POST /api/cron/jobs

Creates a new cron job. Body: `{"prompt": "...", "schedule": "0 9 * * *", "name": "...", "deliver": "local"}`.

### POST /api/cron/jobs/\{job_id\}/pause

Pauses a cron job.

### POST /api/cron/jobs/\{job_id\}/resume

Resumes a paused cron job.

### POST /api/cron/jobs/\{job_id\}/trigger

Immediately triggers a cron job outside its schedule.

### DELETE /api/cron/jobs/\{job_id\}

Deletes a cron job.

### GET /api/skills

Returns all skills with their name, description, category, and enabled status.

### PUT /api/skills/toggle

Enables or disables a skill. Body: `{"name": "skill-name", "enabled": true}`.

### GET /api/tools/toolsets

Returns all toolsets with their label, description, tools list, and active/configured status.

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
hermes dashboard --no-open

# Terminal 2: start the Vite dev server with HMR
cd web/
npm install
npm run dev
```

The Vite dev server at `http://localhost:5173` proxies `/api` requests to the FastAPI backend at `http://127.0.0.1:9119`.

The frontend is built with React 19, TypeScript, Tailwind CSS v4, and shadcn/ui-style components. Production builds output to `hermes_cli/web_dist/` which the FastAPI server serves as a static SPA.

## Automatic Build on Update

When you run `hermes update`, the web frontend is automatically rebuilt if `npm` is available. This keeps the dashboard in sync with code updates. If `npm` isn't installed, the update skips the frontend build and `hermes dashboard` will build it on first launch.

## Themes

Themes control the dashboard's visual presentation across three layers:

- **Palette** — colors (background, text, accents, warm glow, noise)
- **Typography** — font families, base size, line height, letter spacing
- **Layout** — corner radius and density (spacing multiplier)

Switch themes live from the header bar — click the palette icon next to the language switcher. Selection persists to `config.yaml` under `dashboard.theme` and is restored on page load.

### Built-in themes

Each built-in ships its own palette, typography, and layout — switching produces visible changes beyond color alone.

| Theme | Palette | Typography | Layout |
|-------|---------|------------|--------|
| **Hermes Teal** (`default`) | Dark teal + cream | System stack, 15px | 0.5rem radius, comfortable |
| **Midnight** (`midnight`) | Deep blue-violet | Inter + JetBrains Mono, 14px | 0.75rem radius, comfortable |
| **Ember** (`ember`) | Warm crimson / bronze | Spectral (serif) + IBM Plex Mono, 15px | 0.25rem radius, comfortable |
| **Mono** (`mono`) | Grayscale | IBM Plex Sans + IBM Plex Mono, 13px | 0 radius, compact |
| **Cyberpunk** (`cyberpunk`) | Neon green on black | Share Tech Mono everywhere, 14px | 0 radius, compact |
| **Rosé** (`rose`) | Pink and ivory | Fraunces (serif) + DM Mono, 16px | 1rem radius, spacious |

Themes that reference Google Fonts (everything except Hermes Teal) load the stylesheet on demand — the first time you switch to them, a `<link>` tag is injected into `<head>`.

### Custom themes

Drop a YAML file in `~/.hermes/dashboard-themes/` and it appears in the picker automatically. The file can be as minimal as a name plus the fields you want to override — every missing field inherits a sane default.

Minimal example (colors only, bare hex shorthand):

```yaml
# ~/.hermes/dashboard-themes/neon.yaml
name: neon
label: Neon
description: Pure magenta on black
colors:
  background: "#000000"
  midground: "#ff00ff"
```

Full example (every knob):

```yaml
# ~/.hermes/dashboard-themes/ocean.yaml
name: ocean
label: Ocean Deep
description: Deep sea blues with coral accents

palette:
  background:
    hex: "#0a1628"
    alpha: 1.0
  midground:
    hex: "#a8d0ff"
    alpha: 1.0
  foreground:
    hex: "#ffffff"
    alpha: 0.0
  warmGlow: "rgba(255, 107, 107, 0.35)"
  noiseOpacity: 0.7

typography:
  fontSans: "Poppins, system-ui, sans-serif"
  fontMono: "Fira Code, ui-monospace, monospace"
  fontDisplay: "Poppins, system-ui, sans-serif"   # optional, falls back to fontSans
  fontUrl: "https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&family=Fira+Code:wght@400;500&display=swap"
  baseSize: "15px"
  lineHeight: "1.6"
  letterSpacing: "-0.003em"

layout:
  radius: "0.75rem"      # 0 | 0.25rem | 0.5rem | 0.75rem | 1rem | any length
  density: comfortable   # compact | comfortable | spacious

# Optional — pin individual shadcn tokens that would otherwise derive from
# the palette. Any key listed here wins over the palette cascade.
colorOverrides:
  destructive: "#ff6b6b"
  ring: "#ff6b6b"
```

Refresh the dashboard after creating the file.

### Palette model

The palette is a 3-layer triplet — **background**, **midground**, **foreground** — plus a warm-glow rgba() string and a noise-opacity multiplier. Every shadcn token (card, muted, border, primary, popover, etc.) is derived from this triplet via CSS `color-mix()` in the dashboard's stylesheet, so overriding three colors cascades into the whole UI.

- `background` — deepest canvas color (typically near-black). The page background and card fill come from this.
- `midground` — primary text and accent. Most UI chrome reads this.
- `foreground` — top-layer highlight. In the default theme this is white at alpha 0 (invisible); themes that want a bright accent on top can raise its alpha.
- `warmGlow` — rgba() vignette color used by the ambient backdrop.
- `noiseOpacity` — 0–1.2 multiplier on the grain overlay. Lower = softer, higher = grittier.

Each layer accepts `{hex, alpha}` or a bare hex string (alpha defaults to 1.0).

### Typography model

| Key | Type | Description |
|-----|------|-------------|
| `fontSans` | string | CSS font-family stack for body copy (applied to `html`, `body`) |
| `fontMono` | string | CSS font-family stack for code blocks, `<code>`, `.font-mono` utilities, dense readouts |
| `fontDisplay` | string | Optional heading/display font stack. Falls back to `fontSans` |
| `fontUrl` | string | Optional external stylesheet URL. Injected as `<link rel="stylesheet">` in `<head>` on theme switch. Same URL is never injected twice. Works with Google Fonts, Bunny Fonts, self-hosted `@font-face` sheets, anything you can link |
| `baseSize` | string | Root font size — controls the rem scale for the whole dashboard. Example: `"14px"`, `"16px"` |
| `lineHeight` | string | Default line-height, e.g. `"1.5"`, `"1.65"` |
| `letterSpacing` | string | Default letter-spacing, e.g. `"0"`, `"0.01em"`, `"-0.01em"` |

### Layout model

| Key | Values | Description |
|-----|--------|-------------|
| `radius` | any CSS length | Corner-radius token. Cascades into `--radius-sm/md/lg/xl` so every rounded element shifts together. |
| `density` | `compact` \| `comfortable` \| `spacious` | Spacing multiplier. Compact = 0.85×, comfortable = 1.0× (default), spacious = 1.2×. Scales Tailwind's base spacing, so padding, gap, and space-between utilities all shift proportionally. |

### Color overrides (optional)

Most themes won't need this — the 3-layer palette derives every shadcn token. But if you want a specific accent that the derivation won't produce (a softer destructive red for a pastel theme, a specific success green for a brand), pin individual tokens here.

Supported keys: `card`, `cardForeground`, `popover`, `popoverForeground`, `primary`, `primaryForeground`, `secondary`, `secondaryForeground`, `muted`, `mutedForeground`, `accent`, `accentForeground`, `destructive`, `destructiveForeground`, `success`, `warning`, `border`, `input`, `ring`.

Any key set here overrides the derived value for the active theme only — switching to another theme clears the overrides.

### Layout variants

`layoutVariant` selects the overall shell layout. Defaults to `standard`.

| Variant | Behaviour |
|---------|-----------|
| `standard` | Single column, 1600px max-width (default) |
| `cockpit` | Left sidebar rail (260px) + main content. Populated by plugins via the `sidebar` slot |
| `tiled` | Drops the max-width clamp so pages can use the full viewport |

```yaml
layoutVariant: cockpit
```

The current variant is exposed as `document.documentElement.dataset.layoutVariant` so custom CSS can target it via `:root[data-layout-variant="cockpit"]`.

### Theme assets

Ship artwork URLs with a theme. Each named slot becomes a CSS var (`--theme-asset-<name>`) that plugins and the built-in shell read; the `bg` slot is automatically wired into the backdrop.

```yaml
assets:
  bg: "https://example.com/hero-bg.jpg"       # full-viewport background
  hero: "/my-images/strike-freedom.png"       # for plugin sidebars
  crest: "/my-images/crest.svg"               # for header slot plugins
  logo: "/my-images/logo.png"
  sidebar: "/my-images/rail.png"
  header: "/my-images/header-art.png"
  custom:
    scanLines: "/my-images/scanlines.png"     # → --theme-asset-custom-scanLines
```

Values accept bare URLs (wrapped in `url(...)` automatically), pre-wrapped `url(...)`/`linear-gradient(...)`/`radial-gradient(...)` expressions, and `none`.

### Component chrome overrides

Themes can restyle individual shell components without writing CSS selectors via the `componentStyles` block. Each bucket's entries become CSS vars (`--component-<bucket>-<kebab-property>`) that the shell's shared components read — so `card:` overrides apply to every `<Card>`, `header:` to the app bar, etc.

```yaml
componentStyles:
  card:
    clipPath: "polygon(12px 0, 100% 0, 100% calc(100% - 12px), calc(100% - 12px) 100%, 0 100%, 0 12px)"
    background: "linear-gradient(180deg, rgba(10, 22, 52, 0.85), rgba(5, 9, 26, 0.92))"
    boxShadow: "inset 0 0 0 1px rgba(64, 200, 255, 0.28)"
  header:
    background: "linear-gradient(180deg, rgba(16, 32, 72, 0.95), rgba(5, 9, 26, 0.9))"
  tab:
    clipPath: "polygon(6px 0, 100% 0, calc(100% - 6px) 100%, 0 100%)"
  sidebar: {...}
  backdrop: {...}
  footer: {...}
  progress: {...}
  badge: {...}
  page: {...}
```

Supported buckets: `card`, `header`, `footer`, `sidebar`, `tab`, `progress`, `badge`, `backdrop`, `page`. Property names use camelCase (`clipPath`) and are emitted as kebab (`clip-path`). Values are plain CSS strings — anything CSS accepts (`clip-path`, `border-image`, `background`, `box-shadow`, animations, etc.).

### Custom CSS

For selector-level chrome that doesn't fit `componentStyles` — pseudo-elements, animations, media queries, theme-scoped overrides — drop raw CSS into the `customCSS` field:

```yaml
customCSS: |
  :root[data-layout-variant="cockpit"] body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 100;
    background: repeating-linear-gradient(to bottom,
      transparent 0px, transparent 2px,
      rgba(64, 200, 255, 0.035) 3px, rgba(64, 200, 255, 0.035) 4px);
    mix-blend-mode: screen;
  }
```

The CSS is injected as a single scoped `<style data-hermes-theme-css>` tag on theme apply and cleaned up on theme switch. Capped at 32 KiB per theme.

## Dashboard plugins

Plugins live in `~/.hermes/plugins/<name>/dashboard/` (user) or repo `plugins/<name>/dashboard/` (bundled). Each ships a `manifest.json` plus a plain JS bundle that uses the plugin SDK exposed on `window.__HERMES_PLUGIN_SDK__`.

### Manifest

```json
{
  "name": "my-plugin",
  "label": "My Plugin",
  "icon": "Sparkles",
  "version": "1.0.0",
  "tab": {
    "path": "/my-plugin",
    "position": "after:skills",
    "override": "/",
    "hidden": false
  },
  "slots": ["sidebar", "header-left"],
  "entry": "dist/index.js",
  "css": "dist/index.css",
  "api": "api.py"
}
```

| Field | Description |
|-------|-------------|
| `tab.path` | Route path the plugin component renders at |
| `tab.position` | `end`, `after:<tab>`, or `before:<tab>` |
| `tab.override` | When set to a built-in path (`/`, `/sessions`, etc.), this plugin replaces that page instead of adding a new tab |
| `tab.hidden` | When true, register component + slots but skip the nav entry. Used by slot-only plugins |
| `slots` | Shell slots this plugin populates (documentation aid; actual registration happens from the JS bundle) |

### Shell slots

Plugins inject components into named shell locations by calling `window.__HERMES_PLUGINS__.registerSlot(pluginName, slotName, Component)`. Multiple plugins can populate the same slot — they render stacked in registration order.

| Slot | Location |
|------|----------|
| `backdrop` | Inside the backdrop layer stack |
| `header-left` | Before the Hermes brand in the top bar |
| `header-right` | Before the theme/language switchers |
| `header-banner` | Full-width strip below the nav |
| `sidebar` | Cockpit sidebar rail (only rendered when `layoutVariant === "cockpit"`) |
| `pre-main` | Above the route outlet |
| `post-main` | Below the route outlet |
| `footer-left` / `footer-right` | Footer cell content (replaces default) |
| `overlay` | Fixed-position layer above everything else |

### Plugin SDK

Exposed on `window.__HERMES_PLUGIN_SDK__`:

- `React` + `hooks` (useState, useEffect, useCallback, useMemo, useRef, useContext, createContext)
- `components` — Card, Badge, Button, Input, Label, Select, Separator, Tabs, **PluginSlot**
- `api` — Hermes API client, plus raw `fetchJSON`
- `utils` — `cn()`, `timeAgo()`, `isoTimeAgo()`
- `useI18n` — i18n hook for multi-language plugins

### Demo: Strike Freedom Cockpit

`plugins/strike-freedom-cockpit/` ships a complete skin demo showing every extension point — cockpit layout variant, theme-supplied hero/crest assets, notched card corners via `componentStyles`, scanlines via `customCSS`, and a slot-only plugin that populates the sidebar, header, and footer. Copy the theme YAML into `~/.hermes/dashboard-themes/` and the plugin directory into `~/.hermes/plugins/` to try it.

### Theme API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/themes` | GET | List available themes + active name. Built-ins return `{name, label, description}`; user themes also include a `definition` field with the full normalised theme object. |
| `/api/dashboard/theme` | PUT | Set active theme. Body: `{"name": "midnight"}` |
