---
sidebar_position: 1
title: "CLI Interface"
description: "Master the Hermes Agent terminal interface — commands, keybindings, personalities, and more"
---

# CLI Interface

Hermes Agent's CLI is a full terminal user interface (TUI) — not a web UI. It features multiline editing, slash-command autocomplete, conversation history, interrupt-and-redirect, and streaming tool output. Built for people who live in the terminal.

## Running the CLI

```bash
# Start an interactive session (default)
hermes

# Single query mode (non-interactive)
hermes chat -q "Hello"

# With a specific model
hermes --model "anthropic/claude-sonnet-4"

# With a specific provider
hermes --provider nous        # Use Nous Portal
hermes --provider openrouter  # Force OpenRouter

# With specific toolsets
hermes --toolsets "web,terminal,skills"

# Resume previous sessions
hermes --continue             # Resume the most recent CLI session (-c)
hermes --resume <session_id>  # Resume a specific session by ID (-r)

# Verbose mode (debug output)
hermes --verbose
```

## Interface Layout

```text
┌─────────────────────────────────────────────────┐
│  HERMES-AGENT ASCII Logo                        │
│  ┌─────────────┐ ┌────────────────────────────┐ │
│  │  Caduceus   │ │ Model: claude-sonnet-4     │ │
│  │  ASCII Art  │ │ Terminal: local            │ │
│  │             │ │ Working Dir: /home/user    │ │
│  │             │ │ Available Tools: 19        │ │
│  │             │ │ Available Skills: 12       │ │
│  └─────────────┘ └────────────────────────────┘ │
├─────────────────────────────────────────────────┤
│ Conversation output scrolls here...             │
│                                                 │
│   (◕‿◕✿) 🧠 pondering... (2.3s)                │
│   ✧٩(ˊᗜˋ*)و✧ got it! (2.3s)                    │
│                                                 │
│ Assistant: Hello! How can I help you today?     │
├─────────────────────────────────────────────────┤
│ ❯ [Fixed input area at bottom]                  │
└─────────────────────────────────────────────────┘
```

The welcome banner shows your model, terminal backend, working directory, available tools, and installed skills at a glance.

## Keybindings

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Alt+Enter` or `Ctrl+J` | New line (multi-line input) |
| `Ctrl+C` | Interrupt agent (double-press within 2s to force exit) |
| `Ctrl+D` | Exit |
| `Tab` | Autocomplete slash commands |

## Slash Commands

Type `/` to see an autocomplete dropdown of all available commands.

### Navigation & Control

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/quit` | Exit the CLI (also: `/exit`, `/q`) |
| `/clear` | Clear screen and reset conversation |
| `/new` | Start a new conversation |
| `/reset` | Reset conversation only (keep screen) |

### Tools & Configuration

| Command | Description |
|---------|-------------|
| `/tools` | List all available tools grouped by toolset |
| `/toolsets` | List available toolsets with descriptions |
| `/model [name]` | Show or change the current model |
| `/config` | Show current configuration |
| `/prompt [text]` | View/set/clear custom system prompt |
| `/personality [name]` | Set a predefined personality |

### Conversation Management

| Command | Description |
|---------|-------------|
| `/history` | Show conversation history |
| `/retry` | Retry the last message |
| `/undo` | Remove the last user/assistant exchange |
| `/save` | Save the current conversation |
| `/compress` | Manually compress conversation context |
| `/usage` | Show token usage for this session |

### Skills & Scheduling

| Command | Description |
|---------|-------------|
| `/cron` | Manage scheduled tasks |
| `/skills` | Search, install, inspect, or manage skills |
| `/platforms` | Show gateway/messaging platform status |
| `/verbose` | Cycle tool progress display: off → new → all → verbose |
| `/<skill-name>` | Invoke any installed skill (e.g., `/axolotl`, `/gif-search`) |

:::tip
Commands are case-insensitive — `/HELP` works the same as `/help`. Most commands work mid-conversation.
:::

## Skill Slash Commands

Every installed skill in `~/.hermes/skills/` is automatically registered as a slash command. The skill name becomes the command:

```
/gif-search funny cats
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor

# Just the skill name loads it and lets the agent ask what you need:
/excalidraw
```

## Personalities

Set a predefined personality to change the agent's tone:

```
/personality pirate
/personality kawaii
/personality concise
```

Built-in personalities include: `helpful`, `concise`, `technical`, `creative`, `teacher`, `kawaii`, `catgirl`, `pirate`, `shakespeare`, `surfer`, `noir`, `uwu`, `philosopher`, `hype`.

You can also define custom personalities in `~/.hermes/config.yaml`:

```yaml
agent:
  personalities:
    helpful: "You are a helpful, friendly AI assistant."
    kawaii: "You are a kawaii assistant! Use cute expressions..."
    pirate: "Arrr! Ye be talkin' to Captain Hermes..."
    # Add your own!
```

## Multi-line Input

There are two ways to enter multi-line messages:

1. **`Alt+Enter` or `Ctrl+J`** — inserts a new line
2. **Backslash continuation** — end a line with `\` to continue:

```
❯ Write a function that:\
  1. Takes a list of numbers\
  2. Returns the sum
```

:::info
Pasting 5+ lines of text automatically saves to `~/.hermes/pastes/` and collapses to a reference, keeping your prompt clean.
:::

## Interrupting the Agent

You can interrupt the agent at any point:

- **Type a new message + Enter** while the agent is working — it interrupts and processes your new instructions
- **`Ctrl+C`** — interrupt the current operation (press twice within 2s to force exit)
- In-progress terminal commands are killed immediately (SIGTERM, then SIGKILL after 1s)
- Multiple messages typed during interrupt are combined into one prompt

## Tool Progress Display

The CLI shows animated feedback as the agent works:

**Thinking animation** (during API calls):
```
  ◜ (｡•́︿•̀｡) pondering... (1.2s)
  ◠ (⊙_⊙) contemplating... (2.4s)
  ✧٩(ˊᗜˋ*)و✧ got it! (3.1s)
```

**Tool execution feed:**
```
  ┊ 💻 terminal `ls -la` (0.3s)
  ┊ 🔍 web_search (1.2s)
  ┊ 📄 web_extract (2.1s)
```

Cycle through display modes with `/verbose`: `off → new → all → verbose`.

## Session Management

### Resuming Sessions

When you exit a CLI session, a resume command is printed:

```
Resume this session with:
  hermes --resume 20260225_143052_a1b2c3

Session:        20260225_143052_a1b2c3
Duration:       12m 34s
Messages:       28 (5 user, 18 tool calls)
```

Resume options:

```bash
hermes --continue                          # Resume the most recent CLI session
hermes -c                                  # Short form
hermes --resume 20260225_143052_a1b2c3     # Resume a specific session by ID
hermes -r 20260225_143052_a1b2c3           # Short form
```

Resuming restores the full conversation history from SQLite. The agent sees all previous messages, tool calls, and responses — just as if you never left.

Use `hermes sessions list` to browse past sessions.

### Session Logging

Sessions are automatically logged to `~/.hermes/sessions/`:

```
sessions/
├── session_20260201_143052_a1b2c3.json
├── session_20260201_150217_d4e5f6.json
└── ...
```

### Context Compression

Long conversations are automatically summarized when approaching context limits:

```yaml
# In ~/.hermes/config.yaml
compression:
  enabled: true
  threshold: 0.85    # Compress at 85% of context limit
```

When compression triggers, middle turns are summarized while the first 3 and last 4 turns are always preserved.

## Quiet Mode

By default, the CLI runs in quiet mode which:
- Suppresses verbose logging from tools
- Enables kawaii-style animated feedback
- Keeps output clean and user-friendly

For debug output:
```bash
hermes --verbose
```
