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
hermes chat --model "anthropic/claude-sonnet-4"

# With a specific provider
hermes chat --provider nous        # Use Nous Portal
hermes chat --provider openrouter  # Force OpenRouter

# With specific toolsets
hermes chat --toolsets "web,terminal,skills"

# Start with one or more skills preloaded
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -q "open a draft PR"

# Resume previous sessions
hermes --continue             # Resume the most recent CLI session (-c)
hermes --resume <session_id>  # Resume a specific session by ID (-r)

# Verbose mode (debug output)
hermes chat --verbose

# Isolated git worktree (for running multiple agents in parallel)
hermes -w                         # Interactive mode in worktree
hermes -w -q "Fix issue #123"     # Single query in worktree
```

## Interface Layout

<img className="docs-terminal-figure" src="/img/docs/cli-layout.svg" alt="Stylized preview of the Hermes CLI layout showing the banner, conversation area, and fixed input prompt." />
<p className="docs-figure-caption">The Hermes CLI banner, conversation stream, and fixed input prompt rendered as a stable docs figure instead of fragile text art.</p>

The welcome banner shows your model, terminal backend, working directory, available tools, and installed skills at a glance.

### Session Resume Display

When resuming a previous session (`hermes -c` or `hermes --resume <id>`), a "Previous Conversation" panel appears between the banner and the input prompt, showing a compact recap of the conversation history. See [Sessions — Conversation Recap on Resume](sessions.md#conversation-recap-on-resume) for details and configuration.

## Keybindings

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Alt+Enter` or `Ctrl+J` | New line (multi-line input) |
| `Alt+V` | Paste an image from the clipboard when supported by the terminal |
| `Ctrl+V` | Paste text and opportunistically attach clipboard images |
| `Ctrl+B` | Start/stop voice recording when voice mode is enabled (`voice.record_key`, default: `ctrl+b`) |
| `Ctrl+C` | Interrupt agent (double-press within 2s to force exit) |
| `Ctrl+D` | Exit |
| `Tab` | Autocomplete slash commands |

## Slash Commands

Type `/` to see the autocomplete dropdown. Hermes supports a large set of CLI slash commands, dynamic skill commands, and user-defined quick commands.

Common examples:

| Command | Description |
|---------|-------------|
| `/help` | Show command help |
| `/model` | Show or change the current model |
| `/tools` | List currently available tools |
| `/skills browse` | Browse the skills hub and official optional skills |
| `/background <prompt>` | Run a prompt in a separate background session |
| `/skin` | Show or switch the active CLI skin |
| `/voice on` | Enable CLI voice mode (press `Ctrl+B` to record) |
| `/voice tts` | Toggle spoken playback for Hermes replies |
| `/reasoning high` | Increase reasoning effort |
| `/title My Session` | Name the current session |

For the full built-in CLI and messaging lists, see [Slash Commands Reference](../reference/slash-commands.md).

For setup, providers, silence tuning, and messaging/Discord voice usage, see [Voice Mode](features/voice-mode.md).

:::tip
Commands are case-insensitive — `/HELP` works the same as `/help`. Installed skills also become slash commands automatically.
:::

## Quick Commands

You can define custom commands that run shell commands instantly without invoking the LLM. These work in both the CLI and messaging platforms (Telegram, Discord, etc.).

```yaml
# ~/.hermes/config.yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Then type `/status` or `/gpu` in any chat. See the [Configuration guide](/docs/user-guide/configuration#quick-commands) for more examples.

## Preloading Skills at Launch

If you already know which skills you want active for the session, pass them at launch time:

```bash
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -s github-auth
```

Hermes loads each named skill into the session prompt before the first turn. The same flag works in interactive mode and single-query mode.

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
Pasting multi-line text is supported — use `Alt+Enter` or `Ctrl+J` to insert newlines, or simply paste content directly.
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
hermes -c "my project"                     # Resume a named session (latest in lineage)
hermes --resume 20260225_143052_a1b2c3     # Resume a specific session by ID
hermes --resume "refactoring auth"         # Resume by title
hermes -r 20260225_143052_a1b2c3           # Short form
```

Resuming restores the full conversation history from SQLite. The agent sees all previous messages, tool calls, and responses — just as if you never left.

Use `/title My Session Name` inside a chat to name the current session, or `hermes sessions rename <id> <title>` from the command line. Use `hermes sessions list` to browse past sessions.

### Session Storage

CLI sessions are stored in Hermes's SQLite state database under `~/.hermes/state.db`. The database keeps:

- session metadata (ID, title, timestamps, token counters)
- message history
- lineage across compressed/resumed sessions
- full-text search indexes used by `session_search`

Some messaging adapters also keep per-platform transcript files alongside the database, but the CLI itself resumes from the SQLite session store.

### Context Compression

Long conversations are automatically summarized when approaching context limits:

```yaml
# In ~/.hermes/config.yaml
compression:
  enabled: true
  threshold: 0.50    # Compress at 50% of context limit by default
  summary_model: "google/gemini-3-flash-preview"  # Model used for summarization
```

When compression triggers, middle turns are summarized while the first 3 and last 4 turns are always preserved.

## Quiet Mode

By default, the CLI runs in quiet mode which:
- Suppresses verbose logging from tools
- Enables kawaii-style animated feedback
- Keeps output clean and user-friendly

For debug output:
```bash
hermes chat --verbose
```
