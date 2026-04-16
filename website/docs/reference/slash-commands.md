---
sidebar_position: 2
title: "Slash Commands Reference"
description: "Complete reference for interactive CLI and messaging slash commands"
---

# Slash Commands Reference

Hermes has two slash-command surfaces, both driven by a central `COMMAND_REGISTRY` in `hermes_cli/commands.py`:

- **Interactive CLI slash commands** — dispatched by `cli.py`, with autocomplete from the registry
- **Messaging slash commands** — dispatched by `gateway/run.py`, with help text and platform menus generated from the registry

Installed skills are also exposed as dynamic slash commands on both surfaces. That includes bundled skills like `/plan`, which opens plan mode and saves markdown plans under `.hermes/plans/` relative to the active workspace/backend working directory.

## Interactive CLI slash commands

Type `/` in the CLI to open the autocomplete menu. Built-in commands are case-insensitive.

### Session

| Command | Description |
|---------|-------------|
| `/new` (alias: `/reset`) | Start a new session (fresh session ID + history) |
| `/clear` | Clear screen and start a new session |
| `/history` | Show conversation history |
| `/save` | Save the current conversation |
| `/retry` | Retry the last message (resend to agent) |
| `/undo` | Remove the last user/assistant exchange |
| `/title` | Set a title for the current session (usage: /title My Session Name) |
| `/compress [focus topic]` | Manually compress conversation context (flush memories + summarize). Optional focus topic narrows what the summary preserves. |
| `/rollback` | List or restore filesystem checkpoints (usage: /rollback [number]) |
| `/snapshot [create\|restore <id>\|prune]` (alias: `/snap`) | Create or restore state snapshots of Hermes config/state. `create [label]` saves a snapshot, `restore <id>` reverts to it, `prune [N]` removes old snapshots, or list all with no args. |
| `/stop` | Kill all running background processes |
| `/queue <prompt>` (alias: `/q`) | Queue a prompt for the next turn (doesn't interrupt the current agent response). **Note:** `/q` is claimed by both `/queue` and `/quit`; the last registration wins, so `/q` resolves to `/quit` in practice. Use `/queue` explicitly. |
| `/resume [name]` | Resume a previously-named session |
| `/status` | Show session info |
| `/snapshot` (alias: `/snap`) | Create or restore state snapshots of Hermes config/state (usage: /snapshot [create\|restore \<id\>\|prune]) |
| `/background <prompt>` (alias: `/bg`) | Run a prompt in a separate background session. The agent processes your prompt independently — your current session stays free for other work. Results appear as a panel when the task finishes. See [CLI Background Sessions](/docs/user-guide/cli#background-sessions). |
| `/btw <question>` | Ephemeral side question using session context (no tools, not persisted). Useful for quick clarifications without affecting the conversation history. |
| `/plan [request]` | Load the bundled `plan` skill to write a markdown plan instead of executing the work. Plans are saved under `.hermes/plans/` relative to the active workspace/backend working directory. |
| `/branch [name]` (alias: `/fork`) | Branch the current session (explore a different path) |

### Configuration

| Command | Description |
|---------|-------------|
| `/config` | Show current configuration |
| `/model [model-name]` | Show or change the current model. Supports: `/model claude-sonnet-4`, `/model provider:model` (switch providers), `/model custom:model` (custom endpoint), `/model custom:name:model` (named custom provider), `/model custom` (auto-detect from endpoint). Use `--global` to persist the change to config.yaml. **Note:** `/model` can only switch between already-configured providers. To add a new provider, exit the session and run `hermes model` from your terminal. |
| `/provider` | Show available providers and current provider |
| `/personality` | Set a predefined personality |
| `/verbose` | Cycle tool progress display: off → new → all → verbose. Can be [enabled for messaging](#notes) via config. |
| `/fast` | Toggle fast mode — OpenAI Priority Processing / Anthropic Fast Mode (usage: /fast [normal\|fast\|status]) |
| `/reasoning` | Manage reasoning effort and display (usage: /reasoning [level\|show\|hide]) |
| `/fast [normal\|fast\|status]` | Toggle fast mode — OpenAI Priority Processing / Anthropic Fast Mode. Options: `normal`, `fast`, `status`, `on`, `off`. |
| `/skin` | Show or change the display skin/theme |
| `/statusbar` (alias: `/sb`) | Toggle the context/model status bar on or off |
| `/voice [on\|off\|tts\|status]` | Toggle CLI voice mode and spoken playback. Recording uses `voice.record_key` (default: `Ctrl+B`). |
| `/yolo` | Toggle YOLO mode — skip all dangerous command approval prompts. |

### Tools & Skills

| Command | Description |
|---------|-------------|
| `/tools [list\|disable\|enable] [name...]` | Manage tools: list available tools, or disable/enable specific tools for the current session. Disabling a tool removes it from the agent's toolset and triggers a session reset. |
| `/toolsets` | List available toolsets |
| `/browser [connect\|disconnect\|status]` | Manage local Chrome CDP connection. `connect` attaches browser tools to a running Chrome instance (default: `ws://localhost:9222`). `disconnect` detaches. `status` shows current connection. Auto-launches Chrome if no debugger is detected. |
| `/skills` | Search, install, inspect, or manage skills from online registries |
| `/cron` | Manage scheduled tasks (list, add/create, edit, pause, resume, run, remove) |
| `/reload-mcp` (alias: `/reload_mcp`) | Reload MCP servers from config.yaml |
| `/reload` | Reload `.env` variables into the running session (picks up new API keys without restarting) |
| `/plugins` | List installed plugins and their status |

### Info

| Command | Description |
|---------|-------------|
| `/help` | Show this help message |
| `/usage` | Show token usage, cost breakdown, and session duration |
| `/insights` | Show usage insights and analytics (last 30 days) |
| `/platforms` (alias: `/gateway`) | Show gateway/messaging platform status |
| `/paste` | Check clipboard for an image and attach it |
| `/image <path>` | Attach a local image file for your next prompt. |
| `/debug` | Upload debug report (system info + logs) and get shareable links. Also available in messaging. |
| `/profile` | Show active profile name and home directory |

### Exit

| Command | Description |
|---------|-------------|
| `/quit` | Exit the CLI (also: `/exit`). See note on `/q` under `/queue` above. |

### Dynamic CLI slash commands

| Command | Description |
|---------|-------------|
| `/<skill-name>` | Load any installed skill as an on-demand command. Example: `/gif-search`, `/github-pr-workflow`, `/excalidraw`. |
| `/skills ...` | Search, browse, inspect, install, audit, publish, and configure skills from registries and the official optional-skills catalog. |

### Quick Commands

User-defined quick commands map a short alias to a longer prompt. Configure them in `~/.hermes/config.yaml`:

```yaml
quick_commands:
  review: "Review my latest git diff and suggest improvements"
  deploy: "Run the deployment script at scripts/deploy.sh and verify the output"
  morning: "Check my calendar, unread emails, and summarize today's priorities"
```

Then type `/review`, `/deploy`, or `/morning` in the CLI. Quick commands are resolved at dispatch time and are not shown in the built-in autocomplete/help tables.

### Alias Resolution

Commands support prefix matching: typing `/h` resolves to `/help`, `/mod` resolves to `/model`. When a prefix is ambiguous (matches multiple commands), the first match in registry order wins. Full command names and registered aliases always take priority over prefix matches.

## Messaging slash commands

The messaging gateway supports the following built-in commands inside Telegram, Discord, Slack, WhatsApp, Signal, Email, and Home Assistant chats:

| Command | Description |
|---------|-------------|
| `/new` | Start a new conversation. |
| `/reset` | Reset conversation history. |
| `/status` | Show session info. |
| `/stop` | Kill all running background processes and interrupt the running agent. |
| `/model [provider:model]` | Show or change the model. Supports provider switches (`/model zai:glm-5`), custom endpoints (`/model custom:model`), named custom providers (`/model custom:local:qwen`), and auto-detect (`/model custom`). Use `--global` to persist the change to config.yaml. **Note:** `/model` can only switch between already-configured providers. To add a new provider or set up API keys, use `hermes model` from your terminal (outside the chat session). |
| `/provider` | Show provider availability and auth status. |
| `/personality [name]` | Set a personality overlay for the session. |
| `/fast [normal\|fast\|status]` | Toggle fast mode — OpenAI Priority Processing / Anthropic Fast Mode. |
| `/retry` | Retry the last message. |
| `/undo` | Remove the last exchange. |
| `/sethome` (alias: `/set-home`) | Mark the current chat as the platform home channel for deliveries. |
| `/compress [focus topic]` | Manually compress conversation context. Optional focus topic narrows what the summary preserves. |
| `/title [name]` | Set or show the session title. |
| `/resume [name]` | Resume a previously named session. |
| `/usage` | Show token usage, estimated cost breakdown (input/output), context window state, and session duration. |
| `/insights [days]` | Show usage analytics. |
| `/reasoning [level\|show\|hide]` | Change reasoning effort or toggle reasoning display. |
| `/voice [on\|off\|tts\|join\|channel\|leave\|status]` | Control spoken replies in chat. `join`/`channel`/`leave` manage Discord voice-channel mode. |
| `/rollback [number]` | List or restore filesystem checkpoints. |
| `/snapshot [create\|restore <id>\|prune]` (alias: `/snap`) | Create or restore state snapshots of Hermes config/state. |
| `/background <prompt>` | Run a prompt in a separate background session. Results are delivered back to the same chat when the task finishes. See [Messaging Background Sessions](/docs/user-guide/messaging/#background-sessions). |
| `/plan [request]` | Load the bundled `plan` skill to write a markdown plan instead of executing the work. Plans are saved under `.hermes/plans/` relative to the active workspace/backend working directory. |
| `/reload-mcp` (alias: `/reload_mcp`) | Reload MCP servers from config. |
| `/reload` | Reload `.env` variables into the running session. |
| `/yolo` | Toggle YOLO mode — skip all dangerous command approval prompts. |
| `/commands [page]` | Browse all commands and skills (paginated). |
| `/approve [session\|always]` | Approve and execute a pending dangerous command. `session` approves for this session only; `always` adds to permanent allowlist. |
| `/deny` | Reject a pending dangerous command. |
| `/update` | Update Hermes Agent to the latest version. |
| `/restart` | Gracefully restart the gateway after draining active runs. When the gateway comes back online, it sends a confirmation to the requester's chat/thread. |
| `/fast [normal\|fast\|status]` | Toggle fast mode — OpenAI Priority Processing / Anthropic Fast Mode. |
| `/debug` | Upload debug report (system info + logs) and get shareable links. |
| `/debug` | Upload debug report (system info + logs) and get shareable links. |
| `/help` | Show messaging help. |
| `/<skill-name>` | Invoke any installed skill by name. |

## Notes

- `/skin`, `/tools`, `/toolsets`, `/browser`, `/config`, `/cron`, `/skills`, `/platforms`, `/paste`, `/image`, `/statusbar`, and `/plugins` are **CLI-only** commands.
- `/verbose` is **CLI-only by default**, but can be enabled for messaging platforms by setting `display.tool_progress_command: true` in `config.yaml`. When enabled, it cycles the `display.tool_progress` mode and saves to config.
- `/sethome`, `/update`, `/restart`, `/approve`, `/deny`, and `/commands` are **messaging-only** commands.
- `/status`, `/background`, `/voice`, `/reload-mcp`, `/rollback`, `/snapshot`, `/debug`, `/fast`, and `/yolo` work in **both** the CLI and the messaging gateway.
- `/voice join`, `/voice channel`, and `/voice leave` are only meaningful on Discord.
