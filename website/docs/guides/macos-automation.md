---
sidebar_position: 9
title: 'macOS Automation with Hermes'
description: 'Give Hermes full macOS desktop automation via AppleScript, JXA, screenshots, clipboard, and system controls using the macos-mcp Python MCP server'
---

# macOS Automation with Hermes

This guide shows how to give Hermes native macOS automation capabilities: run AppleScript,
control apps (Messages, Mail, Safari, Finder, Calendar, Xcode, and more), take screenshots,
manage the clipboard, and control system settings.

## How it works

[`macos-mcp`](https://github.com/YanSte/macos-mcp) is a Python MCP server that ports
[`@steipete/macos-automator-mcp`](https://github.com/steipete/macos-automator-mcp)
(TypeScript, MIT) to Python. It exposes 7 tools and a 498-script AppleScript/JXA knowledge base
over the MCP stdio protocol.

Hermes connects to it like any other MCP server — no code changes required.

## Installation

### 1. Install the server

```bash
pip install macos-mcp
# or with uv:
uv tool install macos-mcp
```

### 2. Enable MCP support in Hermes

```bash
pip install hermes-agent[mcp]
```

### 3. Add to `~/.hermes/config.yaml`

```yaml
mcp_servers:
  macos:
    command: python
    args: ['-m', 'macos_automator_mcp']
```

Or with `uvx` if you installed via `uv tool`:

```yaml
mcp_servers:
  macos:
    command: uvx
    args: ['macos-mcp']
```

### 4. Grant macOS permissions

The process running Hermes (usually Terminal or iTerm2) needs two permissions:

**System Settings → Privacy & Security → Automation**
Enable the parent terminal app. macOS also shows an automatic Allow/Deny prompt
the first time each target app (e.g. Messages, Safari) is scripted.

**System Settings → Privacy & Security → Screen Recording**
Enable the parent terminal app. Required only for `macos_screenshot`.

## Available tools

| Tool | What it does |
|---|---|
| `macos_run_script` | Run inline AppleScript or JXA; or run any of the 498 pre-built scripts by ID |
| `macos_scripting_tips` | Fuzzy-search the knowledge base: "send imessage", "safari screenshot", etc. |
| `macos_screenshot` | Take a screenshot — returns base64 PNG |
| `macos_open` | Open any app, file, or URL (`open` command) |
| `macos_clipboard` | Read or write the clipboard |
| `macos_notify` | Send a macOS system notification |
| `macos_system` | Volume, brightness, dark mode, lock screen, list/quit apps, TTS |

## What Hermes can do on macOS

**Messaging & email**
- Send and read iMessages
- Compose and send emails, count unread, search inbox
- Create group chats, send file attachments

**Browser control**
- Open URLs in Safari/Chrome/Firefox, execute JavaScript, take page screenshots
- Manage bookmarks, fill forms, inspect DOM

**Calendar & productivity**
- Create calendar events, list today's meetings
- Add reminders, search notes, look up contacts

**Developer tools**
- Build, run, and test Xcode projects
- Boot iOS Simulators, set location, install apps, send push notifications
- Run commands in Terminal/iTerm2/Ghostty, open VS Code/Cursor projects

**System**
- Control volume, brightness, dark mode
- Lock screen, sleep display, list/quit running apps
- Speak text (TTS), send notifications

## Usage examples

Once configured, talk to Hermes naturally:

> "Take a screenshot and show me what's on screen"

> "Send an iMessage to Mom: I'll be home by 7"

> "What do I have on my calendar today?"

> "Open Safari and go to github.com/NousResearch/hermes-agent"

> "Set the volume to 30%"

> "Build my Xcode project in ~/dev/MyApp"

For advanced use, reference script IDs directly:

> "Run the script `messages_get_chat_history` for my conversation with Alice"

Or discover available scripts:

> "What scripts do you know for controlling Spotify?"

## Security notes

- `macos_run_script` executes arbitrary AppleScript/JXA. The same trust model
  applies as any other code execution tool — Hermes's normal confirmation flow
  applies for sensitive operations.
- The server only runs on macOS; on other platforms all tools return an error.
- No network calls are made by the server itself — all operations are local.

## Source

- Server: https://github.com/YanSte/macos-mcp (MIT)
- Knowledge base: adapted from https://github.com/steipete/macos-automator-mcp (MIT)
