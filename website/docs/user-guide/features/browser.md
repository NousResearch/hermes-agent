---
title: Browser Automation
description: Control browsers with multiple providers, local Chrome via CDP, or cloud browsers for web interaction, form filling, scraping, and more.
sidebar_label: Browser
sidebar_position: 5
---

# Browser Automation

Hermes Agent includes a full browser automation toolset with multiple backend options:

- **Browserbase cloud mode** via [Browserbase](https://browserbase.com) for managed cloud browsers and anti-bot tooling
- **Browser Use cloud mode** via [Browser Use](https://browser-use.com) as an alternative cloud browser provider
- **Firecrawl cloud mode** via [Firecrawl](https://firecrawl.dev) for cloud browsers with built-in scraping
- **Camofox local mode** via [Camofox](https://github.com/jo-inc/camofox-browser) for local anti-detection browsing (Firefox-based fingerprint spoofing)
- **Local Chrome via CDP** — connect browser tools to your own Chrome instance using `/browser connect`
- **Local browser mode** via the `agent-browser` CLI and a local Chromium installation

In all modes, the agent can navigate websites, interact with page elements, fill forms, and extract information.

## Overview

Pages are represented as **accessibility trees** (text-based snapshots), making them ideal for LLM agents. Interactive elements get ref IDs (like `@e1`, `@e2`) that the agent uses for clicking and typing.

Key capabilities:

- **Multi-provider cloud execution** — Browserbase, Browser Use, or Firecrawl — no local browser needed
- **Local Chrome integration** — attach to your running Chrome via CDP for hands-on browsing
- **Built-in stealth** — random fingerprints, CAPTCHA solving, residential proxies (Browserbase)
- **Session isolation** — each task gets its own browser session
- **Automatic cleanup** — inactive sessions are closed after a timeout
- **Vision analysis** — screenshot + AI analysis for visual understanding

## Setup

:::tip Nous Subscribers
If you have a paid [Nous Portal](https://portal.nousresearch.com) subscription, you can use browser automation through the **[Tool Gateway](tool-gateway.md)** without any separate API keys. Run `hermes model` or `hermes tools` to enable it.
:::

### Browserbase cloud mode

To use Browserbase-managed cloud browsers, add:

```bash
# Add to ~/.hermes/.env
BROWSERBASE_API_KEY=***
BROWSERBASE_PROJECT_ID=your-project-id-here
```

Get your credentials at [browserbase.com](https://browserbase.com).

### Browser Use cloud mode

To use Browser Use as your cloud browser provider, add:

```bash
# Add to ~/.hermes/.env
BROWSER_USE_API_KEY=***
```

Get your API key at [browser-use.com](https://browser-use.com). Browser Use provides a cloud browser via its REST API. If both Browserbase and Browser Use credentials are set, Browserbase takes priority.

### Firecrawl cloud mode

To use Firecrawl as your cloud browser provider, add:

```bash
# Add to ~/.hermes/.env
FIRECRAWL_API_KEY=fc-***
```

Get your API key at [firecrawl.dev](https://firecrawl.dev). Then select Firecrawl as your browser provider:

```bash
hermes setup tools
# → Browser Automation → Firecrawl
```

Optional settings:

```bash
# Self-hosted Firecrawl instance (default: https://api.firecrawl.dev)
FIRECRAWL_API_URL=http://localhost:3002

# Session TTL in seconds (default: 300)
FIRECRAWL_BROWSER_TTL=600
```

### Camofox local mode

[Camofox](https://github.com/jo-inc/camofox-browser) is a self-hosted Node.js server wrapping Camoufox (a Firefox fork with C++ fingerprint spoofing). It provides local anti-detection browsing without cloud dependencies.

```bash
# Install and run
git clone https://github.com/jo-inc/camofox-browser && cd camofox-browser
npm install && npm start   # downloads Camoufox (~300MB) on first run

# Or via Docker
docker run -d --network host -e CAMOFOX_PORT=9377 jo-inc/camofox-browser
```

Then set in `~/.hermes/.env`:

```bash
CAMOFOX_URL=http://localhost:9377
```

Or configure via `hermes tools` → Browser Automation → Camofox.

When `CAMOFOX_URL` is set, all browser tools automatically route through Camofox instead of Browserbase or agent-browser.

#### Persistent browser sessions

By default, each Camofox session gets a random identity — cookies and logins don't survive across agent restarts. To enable persistent browser sessions:

```yaml
# In ~/.hermes/config.yaml
browser:
  camofox:
    managed_persistence: true
```

When enabled, Hermes sends a stable profile-scoped `userId` to Camofox. The Camofox server automatically maps each `userId` to a dedicated persistent Firefox profile, so cookies, logins, and localStorage survive across restarts. Different Hermes profiles get different browser profiles (profile isolation).

#### VNC live view

When Camofox runs in headed mode (with a visible browser window), it exposes a VNC port in its health check response. Hermes automatically discovers this and includes the VNC URL in navigation responses, so the agent can share a link for you to watch the browser live.

### Local Chrome via CDP (`/browser connect`)

Instead of a cloud provider, you can attach Hermes browser tools to your own running Chrome instance via the Chrome DevTools Protocol (CDP). This is useful when you want to see what the agent is doing in real-time, interact with pages that require your own cookies/sessions, or avoid cloud browser costs.

In the CLI, use:

```
/browser connect              # Connect to Chrome at ws://localhost:9222
/browser connect ws://host:port  # Connect to a specific CDP endpoint
/browser status               # Check current connection
/browser disconnect            # Detach and return to cloud/local mode
```

If Chrome isn't already running with remote debugging, Hermes will attempt to auto-launch it with `--remote-debugging-port=9222`.

:::tip
To start Chrome manually with CDP enabled:
```bash
# Linux
google-chrome --remote-debugging-port=9222

# macOS
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9222
```
:::

When connected via CDP, all browser tools (`browser_navigate`, `browser_click`, etc.) operate on your live Chrome instance instead of spinning up a cloud session.

### Local browser mode

If you do **not** set any cloud credentials and don't use `/browser connect`, Hermes can still use the browser tools through a local Chromium install driven by `agent-browser`.

### Optional Environment Variables

```bash
# Residential proxies for better CAPTCHA solving (default: "true")
BROWSERBASE_PROXIES=true

# Advanced stealth with custom Chromium — requires Scale Plan (default: "false")
BROWSERBASE_ADVANCED_STEALTH=false

# Session reconnection after disconnects — requires paid plan (default: "true")
BROWSERBASE_KEEP_ALIVE=true

# Custom session timeout in milliseconds (default: project default)
# Examples: 600000 (10min), 1800000 (30min)
BROWSERBASE_SESSION_TIMEOUT=600000

# Inactivity timeout before auto-cleanup in seconds (default: 120)
BROWSER_INACTIVITY_TIMEOUT=120
```

### Install agent-browser CLI

```bash
npm install -g agent-browser
# Or install locally in the repo:
npm install
```

:::info
The `browser` toolset must be included in your config's `toolsets` list or enabled via `hermes config set toolsets '["hermes-cli", "browser"]'`.
:::

## Available Tools

### `browser_navigate`

Navigate to a URL. Must be called before any other browser tool. Initializes the Browserbase session.

```
Navigate to https://github.com/NousResearch
```

:::tip
For simple information retrieval, prefer `web_search` or `web_extract` — they are faster and cheaper. Use browser tools when you need to **interact** with a page (click buttons, fill forms, handle dynamic content).
:::

### `browser_snapshot`

Get a text-based snapshot of the current page's accessibility tree. Returns interactive elements with ref IDs like `@e1`, `@e2` for use with `browser_click` and `browser_type`.

- **`full=false`** (default): Compact view showing only interactive elements
- **`full=true`**: Complete page content

Snapshots over 8000 characters are automatically summarized by an LLM.

### `browser_click`

Click an element identified by its ref ID from the snapshot.

```
Click @e5 to press the "Sign In" button
```

### `browser_type`

Type text into an input field. Clears the field first, then types the new text.

```
Type "hermes agent" into the search field @e3
```

### `browser_scroll`

Scroll the page up or down to reveal more content.

```
Scroll down to see more results
```

### `browser_press`

Press a keyboard key. Useful for submitting forms or navigation.

```
Press Enter to submit the form
```

Supported keys: `Enter`, `Tab`, `Escape`, `ArrowDown`, `ArrowUp`, and more.

### `browser_back`

Navigate back to the previous page in browser history.

### `browser_get_images`

List all images on the current page with their URLs and alt text. Useful for finding images to analyze.

### `browser_vision`

Take a screenshot and analyze it with vision AI. Use this when text snapshots don't capture important visual information — especially useful for CAPTCHAs, complex layouts, or visual verification challenges.

The screenshot is saved persistently and the file path is returned alongside the AI analysis. On messaging platforms (Telegram, Discord, Slack, WhatsApp), you can ask the agent to share the screenshot — it will be sent as a native photo attachment via the `MEDIA:` mechanism.

```
What does the chart on this page show?
```

Screenshots are stored in `~/.hermes/cache/screenshots/` and automatically cleaned up after 24 hours.

### `browser_console`

Get browser console output (log/warn/error messages) and uncaught JavaScript exceptions from the current page. Essential for detecting silent JS errors that don't appear in the accessibility tree.

```
Check the browser console for any JavaScript errors
```

Use `clear=True` to clear the console after reading, so subsequent calls only show new messages.

## Practical Examples

### Filling Out a Web Form

```
User: Sign up for an account on example.com with my email john@example.com

Agent workflow:
1. browser_navigate("https://example.com/signup")
2. browser_snapshot()  → sees form fields with refs
3. browser_type(ref="@e3", text="john@example.com")
4. browser_type(ref="@e5", text="SecurePass123")
5. browser_click(ref="@e8")  → clicks "Create Account"
6. browser_snapshot()  → confirms success
```

### Researching Dynamic Content

```
User: What are the top trending repos on GitHub right now?

Agent workflow:
1. browser_navigate("https://github.com/trending")
2. browser_snapshot(full=true)  → reads trending repo list
3. Returns formatted results
```

## Session Recording

Automatically record browser sessions as WebM video files:

```yaml
browser:
  record_sessions: true  # default: false
```

When enabled, recording starts automatically on the first `browser_navigate` and saves to `~/.hermes/browser_recordings/` when the session closes. Works in both local and cloud (Browserbase) modes. Recordings older than 72 hours are automatically cleaned up.

## Stealth Features

Browserbase provides automatic stealth capabilities:

| Feature | Default | Notes |
|---------|---------|-------|
| Basic Stealth | Always on | Random fingerprints, viewport randomization, CAPTCHA solving |
| Residential Proxies | On | Routes through residential IPs for better access |
| Advanced Stealth | Off | Custom Chromium build, requires Scale Plan |
| Keep Alive | On | Session reconnection after network hiccups |

:::note
If paid features aren't available on your plan, Hermes automatically falls back — first disabling `keepAlive`, then proxies — so browsing still works on free plans.
:::

## Session Management

- Each task gets an isolated browser session via Browserbase
- Sessions are automatically cleaned up after inactivity (default: 2 minutes)
- A background thread checks every 30 seconds for stale sessions
- Emergency cleanup runs on process exit to prevent orphaned sessions
- Sessions are released via the Browserbase API (`REQUEST_RELEASE` status)

## Limitations

- **Text-based interaction** — relies on accessibility tree, not pixel coordinates
- **Snapshot size** — large pages may be truncated or LLM-summarized at 8000 characters
- **Session timeout** — cloud sessions expire based on your provider's plan settings
- **Cost** — cloud sessions consume provider credits; sessions are automatically cleaned up when the conversation ends or after inactivity. Use `/browser connect` for free local browsing.
- **No file downloads** — cannot download files from the browser
