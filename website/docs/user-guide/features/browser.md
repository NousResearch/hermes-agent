---
title: Browser Automation
description: Control cloud browsers with Browserbase integration for web interaction, form filling, scraping, and more.
sidebar_label: Browser
sidebar_position: 5
---

# Browser Automation

Hermes Agent includes a full browser automation toolset powered by [Browserbase](https://browserbase.com), enabling the agent to navigate websites, interact with page elements, fill forms, and extract information — all running in cloud-hosted browsers with built-in anti-bot stealth features.

## Overview

The browser tools use the `agent-browser` CLI with Browserbase cloud execution. Pages are represented as **accessibility trees** (text-based snapshots), making them ideal for LLM agents. Interactive elements get ref IDs (like `@e1`, `@e2`) that the agent uses for clicking and typing.

Key capabilities:

- **Cloud execution** — no local browser needed
- **Built-in stealth** — random fingerprints, CAPTCHA solving, residential proxies
- **Session isolation** — each task gets its own browser session
- **Automatic cleanup** — inactive sessions are closed after a timeout
- **Vision analysis** — screenshot + AI analysis for visual understanding

## Setup

### Required Environment Variables

```bash
# Add to ~/.hermes/.env
BROWSERBASE_API_KEY=your-api-key-here
BROWSERBASE_PROJECT_ID=your-project-id-here
```

Get your credentials at [browserbase.com](https://browserbase.com).

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

# Inactivity timeout before auto-cleanup in seconds (default: 300)
BROWSER_INACTIVITY_TIMEOUT=300
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

Screenshots are stored in `~/.hermes/browser_screenshots/` and automatically cleaned up after 24 hours.

### `browser_console`

Get browser console output (log/warn/error messages) and uncaught JavaScript exceptions from the current page. Essential for detecting silent JS errors that don't appear in the accessibility tree.

```
Check the browser console for any JavaScript errors
```

Use `clear=True` to clear the console after reading, so subsequent calls only show new messages.

### `browser_close`

Close the browser session and release resources. Call this when done to free up Browserbase session quota.

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
7. browser_close()
```

### Researching Dynamic Content

```
User: What are the top trending repos on GitHub right now?

Agent workflow:
1. browser_navigate("https://github.com/trending")
2. browser_snapshot(full=true)  → reads trending repo list
3. Returns formatted results
4. browser_close()
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
- Sessions are automatically cleaned up after inactivity (default: 5 minutes)
- A background thread checks every 30 seconds for stale sessions
- Emergency cleanup runs on process exit to prevent orphaned sessions
- Sessions are released via the Browserbase API (`REQUEST_RELEASE` status)

## Limitations

- **Requires Browserbase account** — no local browser fallback
- **Requires `agent-browser` CLI** — must be installed via npm
- **Text-based interaction** — relies on accessibility tree, not pixel coordinates
- **Snapshot size** — large pages may be truncated or LLM-summarized at 8000 characters
- **Session timeout** — sessions expire based on your Browserbase plan settings
- **Cost** — each session consumes Browserbase credits; use `browser_close` when done
- **No file downloads** — cannot download files from the browser
