---
sidebar_position: 1
title: "CLI Commands Reference"
description: "Comprehensive reference for all hermes CLI commands and slash commands"
---

# CLI Commands Reference

## Terminal Commands

These are commands you run from your shell.

### Core Commands

| Command | Description |
|---------|-------------|
| `hermes` | Start interactive chat (default) |
| `hermes chat -q "Hello"` | Single query mode (non-interactive) |
| `hermes chat --continue` / `-c` | Resume the most recent session |
| `hermes chat --resume <id>` / `-r <id>` | Resume a specific session |
| `hermes chat --model <name>` | Use a specific model |
| `hermes chat --provider <name>` | Force a provider (`nous`, `openrouter`, `zai`, `kimi-coding`, `minimax`, `minimax-cn`) |
| `hermes chat --toolsets "web,terminal"` / `-t` | Use specific toolsets |
| `hermes chat --verbose` | Enable verbose/debug output |
| `hermes --worktree` / `-w` | Start in an isolated git worktree (for parallel agents) |

### Provider & Model Management

| Command | Description |
|---------|-------------|
| `hermes model` | Switch provider and model interactively |
| `hermes login` | OAuth login to a provider (use `--provider` to specify) |
| `hermes logout` | Clear provider authentication |

### Configuration

| Command | Description |
|---------|-------------|
| `hermes setup` | Full setup wizard (provider, terminal, messaging) |
| `hermes config` | View current configuration |
| `hermes config edit` | Open config.yaml in your editor |
| `hermes config set KEY VAL` | Set a specific value |
| `hermes config check` | Check for missing config (useful after updates) |
| `hermes config migrate` | Interactively add missing options |
| `hermes tools` | Interactive tool configuration per platform |
| `hermes status` | Show configuration status (including auth) |
| `hermes doctor` | Diagnose issues |

### Maintenance

| Command | Description |
|---------|-------------|
| `hermes update` | Update to latest version |
| `hermes uninstall` | Uninstall (can keep configs for later reinstall) |
| `hermes version` | Show version info |

### Gateway (Messaging + Cron)

| Command | Description |
|---------|-------------|
| `hermes gateway` | Run gateway in foreground |
| `hermes gateway setup` | Configure messaging platforms interactively |
| `hermes gateway install` | Install as system service (Linux/macOS) |
| `hermes gateway start` | Start the service |
| `hermes gateway stop` | Stop the service |
| `hermes gateway restart` | Restart the service |
| `hermes gateway status` | Check service status |
| `hermes gateway uninstall` | Uninstall the system service |
| `hermes whatsapp` | Pair WhatsApp via QR code |

### Skills

| Command | Description |
|---------|-------------|
| `hermes skills browse` | Browse all available skills with pagination (official first) |
| `hermes skills search <query>` | Search skill registries |
| `hermes skills install <identifier>` | Install a skill (with security scan) |
| `hermes skills inspect <identifier>` | Preview before installing |
| `hermes skills list` | List installed skills |
| `hermes skills list --source hub` | List hub-installed skills only |
| `hermes skills audit` | Re-scan all hub skills |
| `hermes skills uninstall <name>` | Remove a hub skill |
| `hermes skills publish <path> --to github --repo owner/repo` | Publish a skill |
| `hermes skills snapshot export <file>` | Export skill config |
| `hermes skills snapshot import <file>` | Import from snapshot |
| `hermes skills tap add <repo>` | Add a custom source |
| `hermes skills tap remove <repo>` | Remove a source |
| `hermes skills tap list` | List custom sources |

### Cron & Pairing

| Command | Description |
|---------|-------------|
| `hermes cron list` | View scheduled jobs |
| `hermes cron status` | Check if cron scheduler is running |
| `hermes cron tick` | Manually trigger a cron tick |
| `hermes pairing list` | View pending + approved users |
| `hermes pairing approve <platform> <code>` | Approve a pairing code |
| `hermes pairing revoke <platform> <user_id>` | Remove user access |
| `hermes pairing clear-pending` | Clear all pending pairing requests |

### Sessions

| Command | Description |
|---------|-------------|
| `hermes sessions list` | Browse past sessions |
| `hermes sessions export <id>` | Export a session |
| `hermes sessions delete <id>` | Delete a specific session |
| `hermes sessions prune` | Remove old sessions |
| `hermes sessions stats` | Show session statistics |

### Insights

| Command | Description |
|---------|-------------|
| `hermes insights` | Show usage analytics for the last 30 days |
| `hermes insights --days 7` | Analyze a custom time window |
| `hermes insights --source telegram` | Filter by platform |

---

## Slash Commands (Inside Chat)

Type `/` in the interactive CLI to see an autocomplete dropdown.

### Navigation & Control

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/quit` | Exit the CLI (aliases: `/exit`, `/q`) |
| `/clear` | Clear screen and reset conversation |
| `/new` | Start a new conversation |
| `/reset` | Reset conversation only (keep screen) |

### Tools & Configuration

| Command | Description |
|---------|-------------|
| `/tools` | List all available tools |
| `/toolsets` | List available toolsets |
| `/model [name]` | Show or change the current model |
| `/config` | Show current configuration |
| `/prompt [text]` | View/set custom system prompt |
| `/personality [name]` | Set a predefined personality |

### Conversation

| Command | Description |
|---------|-------------|
| `/history` | Show conversation history |
| `/retry` | Retry the last message |
| `/undo` | Remove the last user/assistant exchange |
| `/save` | Save the current conversation |
| `/compress` | Manually compress conversation context |
| `/usage` | Show token usage for this session |
| `/insights [--days N]` | Show usage insights and analytics (last 30 days) |

### Media & Input

| Command | Description |
|---------|-------------|
| `/paste` | Check clipboard for an image and attach it (see [Vision & Image Paste](/docs/user-guide/features/vision)) |

### Skills & Scheduling

| Command | Description |
|---------|-------------|
| `/cron` | Manage scheduled tasks |
| `/skills` | Browse, search, install, inspect, or manage skills |
| `/platforms` | Show gateway/messaging platform status |
| `/verbose` | Cycle tool progress: off → new → all → verbose |
| `/<skill-name>` | Invoke any installed skill |

### Gateway-Only Commands

These work in messaging platforms (Telegram, Discord, Slack, WhatsApp) but not the interactive CLI:

| Command | Description |
|---------|-------------|
| `/stop` | Stop the running agent (no follow-up message) |
| `/sethome` | Set this chat as the home channel |
| `/status` | Show session info |
| `/reload-mcp` | Reload MCP servers from config |
| `/update` | Update Hermes Agent to the latest version |

---

## Keybindings

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Alt+Enter` / `Ctrl+J` | New line (multi-line input) |
| `Alt+V` | Paste image from clipboard (see [Vision & Image Paste](/docs/user-guide/features/vision)) |
| `Ctrl+V` | Paste text + auto-check for clipboard image |
| `Ctrl+C` | Clear input/images, interrupt agent, or exit (contextual) |
| `Ctrl+D` | Exit |
| `Tab` | Autocomplete slash commands |

:::tip
Commands are case-insensitive — `/HELP` works the same as `/help`.
:::

:::info Image paste keybindings
`Alt+V` works in most terminals but **not** in VSCode's integrated terminal (VSCode intercepts Alt+key combos). `Ctrl+V` only triggers an image check when the clipboard also contains text (terminals don't send paste events for image-only clipboard). The `/paste` command is the universal fallback. See the [full compatibility table](/docs/user-guide/features/vision#platform-compatibility).
:::
