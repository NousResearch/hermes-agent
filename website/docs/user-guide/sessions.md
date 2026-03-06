---
sidebar_position: 7
title: "Sessions"
description: "Session persistence, resume, search, management, and per-platform session tracking"
---

# Sessions

Hermes Agent automatically saves every conversation as a session. Sessions enable conversation resume, cross-session search, and full conversation history management.

## How Sessions Work

Every conversation — whether from the CLI, Telegram, Discord, WhatsApp, or Slack — is stored as a session with full message history. Sessions are tracked in two complementary systems:

1. **SQLite database** (`~/.hermes/state.db`) — structured session metadata with FTS5 full-text search
2. **JSONL transcripts** (`~/.hermes/sessions/`) — raw conversation transcripts including tool calls (gateway)

The SQLite database stores:
- Session ID, source platform, user ID
- Model name and configuration
- System prompt snapshot
- Full message history (role, content, tool calls, tool results)
- Token counts (input/output)
- Timestamps (started_at, ended_at)
- Parent session ID (for compression-triggered session splitting)

### Session Sources

Each session is tagged with its source platform:

| Source | Description |
|--------|-------------|
| `cli` | Interactive CLI (`hermes` or `hermes chat`) |
| `telegram` | Telegram messenger |
| `discord` | Discord server/DM |
| `whatsapp` | WhatsApp messenger |
| `slack` | Slack workspace |

## CLI Session Resume

Resume previous conversations from the CLI using `--continue` or `--resume`:

### Continue Last Session

```bash
# Resume the most recent CLI session
hermes --continue
hermes -c

# Or with the chat subcommand
hermes chat --continue
hermes chat -c
```

This looks up the most recent `cli` session from the SQLite database and loads its full conversation history.

### Resume Specific Session

```bash
# Resume a specific session by ID
hermes --resume 20250305_091523_a1b2c3d4
hermes -r 20250305_091523_a1b2c3d4

# Or with the chat subcommand
hermes chat --resume 20250305_091523_a1b2c3d4
```

Session IDs are shown when you exit a CLI session, and can be found with `hermes sessions list`.

:::tip
Session IDs follow the format `YYYYMMDD_HHMMSS_<8-char-hex>`, e.g. `20250305_091523_a1b2c3d4`. You only need to provide enough of the ID to be unique.
:::

## Session Management Commands

Hermes provides a full set of session management commands via `hermes sessions`:

### List Sessions

```bash
# List recent sessions (default: last 20)
hermes sessions list

# Filter by platform
hermes sessions list --source telegram

# Show more sessions
hermes sessions list --limit 50
```

Output format:

```
ID                             Source       Model                          Messages Started
────────────────────────────────────────────────────────────────────────────────────────────────
20250305_091523_a1b2c3d4       cli          anthropic/claude-opus-4.6          24 2025-03-05 09:15
20250304_143022_e5f6g7h8       telegram     anthropic/claude-opus-4.6          12 2025-03-04 14:30 (ended)
```

### Export Sessions

```bash
# Export all sessions to a JSONL file
hermes sessions export backup.jsonl

# Export sessions from a specific platform
hermes sessions export telegram-history.jsonl --source telegram

# Export a single session
hermes sessions export session.jsonl --session-id 20250305_091523_a1b2c3d4
```

Exported files contain one JSON object per line with full session metadata and all messages.

### Delete a Session

```bash
# Delete a specific session (with confirmation)
hermes sessions delete 20250305_091523_a1b2c3d4

# Delete without confirmation
hermes sessions delete 20250305_091523_a1b2c3d4 --yes
```

### Prune Old Sessions

```bash
# Delete ended sessions older than 90 days (default)
hermes sessions prune

# Custom age threshold
hermes sessions prune --older-than 30

# Only prune sessions from a specific platform
hermes sessions prune --source telegram --older-than 60

# Skip confirmation
hermes sessions prune --older-than 30 --yes
```

:::info
Pruning only deletes **ended** sessions (sessions that have been explicitly ended or auto-reset). Active sessions are never pruned.
:::

### Session Statistics

```bash
hermes sessions stats
```

Output:

```
Total sessions: 142
Total messages: 3847
  cli: 89 sessions
  telegram: 38 sessions
  discord: 15 sessions
Database size: 12.4 MB
```

## Session Search Tool

The agent has a built-in `session_search` tool that performs full-text search across all past conversations using SQLite's FTS5 engine.

### How It Works

1. FTS5 searches matching messages ranked by relevance
2. Groups results by session, takes the top N unique sessions (default 3)
3. Loads each session's conversation, truncates to ~100K chars centered on matches
4. Sends to a fast summarization model for focused summaries
5. Returns per-session summaries with metadata and surrounding context

### FTS5 Query Syntax

The search supports standard FTS5 query syntax:

- Simple keywords: `docker deployment`
- Phrases: `"exact phrase"`
- Boolean: `docker OR kubernetes`, `python NOT java`
- Prefix: `deploy*`

### When It's Used

The agent is prompted to use session search automatically:

> *"When the user references something from a past conversation or you suspect relevant prior context exists, use session_search to recall it before asking them to repeat themselves."*

## Per-Platform Session Tracking

### Gateway Sessions

On messaging platforms, sessions are keyed by a deterministic session key built from the message source:

| Chat Type | Key Format | Example |
|-----------|-----------|---------|
| Telegram DM | `agent:main:telegram:dm` | One session per bot |
| Discord DM | `agent:main:discord:dm` | One session per bot |
| WhatsApp DM | `agent:main:whatsapp:dm:<chat_id>` | Per-user (multi-user) |
| Group chat | `agent:main:<platform>:group:<chat_id>` | Per-group |
| Channel | `agent:main:<platform>:channel:<chat_id>` | Per-channel |

:::info
WhatsApp DMs include the chat ID in the session key because multiple users can DM the bot. Other platforms use a single DM session since the bot is configured per-user via allowlists.
:::

### Session Reset Policies

Gateway sessions are automatically reset based on configurable policies:

- **idle** — reset after N minutes of inactivity
- **daily** — reset at a specific hour each day
- **both** — reset on whichever comes first (idle or daily)
- **none** — never auto-reset

Before a session is auto-reset, the agent is given a turn to save any important memories or skills from the conversation.

Sessions with **active background processes** are never auto-reset, regardless of policy.

## Storage Locations

| What | Path | Description |
|------|------|-------------|
| SQLite database | `~/.hermes/state.db` | All session metadata + messages with FTS5 |
| Gateway transcripts | `~/.hermes/sessions/` | JSONL transcripts per session + sessions.json index |
| Gateway index | `~/.hermes/sessions/sessions.json` | Maps session keys to active session IDs |

The SQLite database uses WAL mode for concurrent readers and a single writer, which suits the gateway's multi-platform architecture well.

### Database Schema

Key tables in `state.db`:

- **sessions** — session metadata (id, source, user_id, model, timestamps, token counts)
- **messages** — full message history (role, content, tool_calls, tool_name, token_count)
- **messages_fts** — FTS5 virtual table for full-text search across message content

## Session Expiry and Cleanup

### Automatic Cleanup

- Gateway sessions auto-reset based on the configured reset policy
- Before reset, the agent saves memories and skills from the expiring session
- Ended sessions remain in the database until pruned

### Manual Cleanup

```bash
# Prune sessions older than 90 days
hermes sessions prune

# Delete a specific session
hermes sessions delete <session_id>

# Export before pruning (backup)
hermes sessions export backup.jsonl
hermes sessions prune --older-than 30 --yes
```

:::tip
The database grows slowly (typical: 10-15 MB for hundreds of sessions). Pruning is mainly useful for removing old conversations you no longer need for search recall.
:::
