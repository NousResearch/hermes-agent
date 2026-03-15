---
sidebar_position: 2
title: "Slash Commands Reference"
description: "Complete reference for interactive CLI and messaging slash commands"
---

# Slash Commands Reference

Hermes has two slash-command surfaces:

- **Interactive CLI slash commands** — handled by `cli.py` / `hermes_cli/commands.py`
- **Messaging slash commands** — handled by `gateway/run.py`

Installed skills are also exposed as dynamic slash commands on both surfaces.

## Interactive CLI slash commands

Type `/` in the CLI to open the autocomplete menu. Built-in commands are case-insensitive.

### Session

| Command | Description |
|---------|-------------|
| `/new` | Start a new conversation (reset history) |
| `/reset` | Reset conversation only (keep screen) |
| `/clear` | Clear screen and reset conversation (fresh start) |
| `/history` | Show conversation history |
| `/save` | Save the current conversation |
| `/retry` | Retry the last message (resend to agent) |
| `/undo` | Remove the last user/assistant exchange |
| `/title` | Set a title for the current session (usage: /title My Session Name) |
| `/compress` | Manually compress conversation context (flush memories + summarize) |
| `/rollback` | List or restore filesystem checkpoints (usage: /rollback [number]) |
| `/background` | Run a prompt in the background (usage: /background &lt;prompt&gt;) |

### Configuration

| Command | Description |
|---------|-------------|
| `/config` | Show current configuration |
| `/model` | Show or change the current model |
| `/provider` | Show available providers and current provider |
| `/prompt` | View/set custom system prompt |
| `/personality` | Set a predefined personality |
| `/verbose` | Cycle tool progress display: off → new → all → verbose |
| `/reasoning` | Manage reasoning effort and display (usage: /reasoning [level\|show\|hide]) |
| `/skin` | Show or change the display skin/theme |
| `/voice [on\|off\|tts\|status]` | Toggle CLI voice mode and spoken playback. Recording uses `voice.record_key` (default: `Ctrl+B`). |

### Tools & Skills

| Command | Description |
|---------|-------------|
| `/tools` | List available tools |
| `/toolsets` | List available toolsets |
| `/skills` | Search, install, inspect, or manage skills from online registries |
| `/cron` | Manage scheduled tasks (list, add, remove) |
| `/reload-mcp` | Reload MCP servers from config.yaml |

### Info

| Command | Description |
|---------|-------------|
| `/help` | Show this help message |
| `/usage` | Show token usage for the current session |
| `/insights` | Show usage insights and analytics (last 30 days) |
| `/platforms` | Show gateway/messaging platform status |
| `/paste` | Check clipboard for an image and attach it |

### Exit

| Command | Description |
|---------|-------------|
| `/quit` | Exit the CLI (also: /exit, /q) |

### Dynamic CLI slash commands

| Command | Description |
|---------|-------------|
| `/<skill-name>` | Load any installed skill as an on-demand command. Example: `/gif-search`, `/github-pr-workflow`, `/excalidraw`. |
| `/skills ...` | Search, browse, inspect, install, audit, publish, and configure skills from registries and the official optional-skills catalog. |

### Quick commands

User-defined quick commands from `quick_commands` in `~/.hermes/config.yaml` are also available as slash commands. These are resolved at dispatch time, not shown in the built-in autocomplete/help tables.

## Messaging slash commands

The messaging gateway supports the following built-in commands inside Telegram, Discord, Slack, WhatsApp, Signal, Email, and Home Assistant chats:

| Command | Description |
|---------|-------------|
| `/new` | Start a new conversation. |
| `/reset` | Reset conversation history. |
| `/status` | Show session info. |
| `/stop` | Interrupt the running agent without queuing a follow-up prompt. |
| `/model [provider:model]` | Show or change the model, including provider switches. |
| `/provider` | Show provider availability and auth status. |
| `/personality [name]` | Set a personality overlay for the session. |
| `/retry` | Retry the last message. |
| `/undo` | Remove the last exchange. |
| `/sethome` | Mark the current chat as the platform home channel for deliveries. |
| `/compress` | Manually compress conversation context. |
| `/title [name]` | Set or show the session title. |
| `/resume [name]` | Resume a previously named session. |
| `/usage` | Show token usage for the current session. |
| `/insights [days]` | Show usage analytics. |
| `/reasoning [level\|show\|hide]` | Change reasoning effort or toggle reasoning display. |
| `/voice [on\|off\|tts\|join\|channel\|leave\|status]` | Control spoken replies in chat. `join`/`channel`/`leave` manage Discord voice-channel mode. |
| `/rollback [number]` | List or restore filesystem checkpoints. |
| `/background &lt;prompt&gt;` | Run a prompt in a separate background session. |
| `/reload-mcp` | Reload MCP servers from config. |
| `/update` | Update Hermes Agent to the latest version. |
| `/help` | Show messaging help. |
| `/<skill-name>` | Invoke any installed skill by name. |

## Notes

- `/skin`, `/tools`, `/toolsets`, `/config`, `/prompt`, `/cron`, `/skills`, `/platforms`, `/paste`, and `/verbose` are **CLI-only** commands.
- `/status`, `/stop`, `/sethome`, `/resume`, `/background`, and `/update` are **messaging-only** commands.
- `/voice`, `/reload-mcp`, and `/rollback` work in **both** the CLI and the messaging gateway.
- `/voice join`, `/voice channel`, and `/voice leave` are only meaningful on Discord.
