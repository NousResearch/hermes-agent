---
title: Honcho Memory
description: AI-native persistent memory for cross-session user modeling and personalization.
sidebar_label: Honcho Memory
sidebar_position: 8
---

# Honcho Memory

[Honcho](https://honcho.dev) is an AI-native memory system that gives Hermes Agent persistent, cross-session understanding of users. While Hermes has built-in memory (`MEMORY.md` and `USER.md` files), Honcho adds a deeper layer of **user modeling** — learning user preferences, goals, communication style, and context across conversations.

## How It Complements Built-in Memory

Hermes has two memory systems that work together:

| Feature | Built-in Memory | Honcho Memory |
|---------|----------------|---------------|
| Storage | Local files (`~/.hermes/memories/`) | Cloud-hosted Honcho API |
| Scope | Agent-level notes and user profile | Deep user modeling via dialectic reasoning |
| Persistence | Across sessions on same machine | Across sessions, machines, and platforms |
| Query | Injected into system prompt automatically | On-demand via `query_user_context` tool |
| Content | Manually curated by the agent | Automatically learned from conversations |

Honcho doesn't replace built-in memory — it **supplements** it with richer user understanding.

## Setup

### 1. Get a Honcho API Key

Sign up at [app.honcho.dev](https://app.honcho.dev) and get your API key.

### 2. Install the Client Library

```bash
pip install honcho-ai
```

### 3. Configure Honcho

Honcho reads its configuration from `~/.honcho/config.json` (the global Honcho config shared across all Honcho-enabled applications):

```json
{
  "apiKey": "your-honcho-api-key",
  "workspace": "hermes",
  "peerName": "your-name",
  "aiPeer": "hermes",
  "environment": "production",
  "saveMessages": true,
  "sessionStrategy": "per-directory",
  "enabled": true
}
```

Alternatively, set the API key as an environment variable:

```bash
# Add to ~/.hermes/.env
HONCHO_API_KEY=your-honcho-api-key
```

:::info
When an API key is present (either in `~/.honcho/config.json` or as `HONCHO_API_KEY`), Honcho auto-enables unless explicitly set to `"enabled": false` in the config.
:::

## Configuration Details

### Global Config (`~/.honcho/config.json`)

| Field | Default | Description |
|-------|---------|-------------|
| `apiKey` | — | Honcho API key (required) |
| `workspace` | `"hermes"` | Workspace identifier |
| `peerName` | *(derived)* | Your identity name for user modeling |
| `aiPeer` | `"hermes"` | AI assistant identity name |
| `environment` | `"production"` | Honcho environment |
| `saveMessages` | `true` | Whether to sync messages to Honcho |
| `sessionStrategy` | `"per-directory"` | How sessions are scoped |
| `sessionPeerPrefix` | `false` | Prefix session names with peer name |
| `contextTokens` | *(Honcho default)* | Max tokens for context prefetch |
| `sessions` | `{}` | Manual session name overrides per directory |

### Host-specific Configuration

You can configure per-host settings for multi-application setups:

```json
{
  "apiKey": "your-key",
  "hosts": {
    "hermes": {
      "workspace": "my-workspace",
      "aiPeer": "hermes-assistant",
      "linkedHosts": ["other-app"],
      "contextTokens": 2000
    }
  }
}
```

Host-specific fields override global fields. Resolution order:
1. Explicit host block fields
2. Global/flat fields from config root
3. Defaults (host name used as workspace/peer)

### Hermes Config (`~/.hermes/config.yaml`)

The `honcho` section in Hermes config is intentionally minimal — most configuration comes from the global `~/.honcho/config.json`:

```yaml
honcho: {}
```

## The `query_user_context` Tool

When Honcho is active, Hermes gains access to the `query_user_context` tool. This lets the agent proactively ask Honcho about the user during conversations:

**Tool schema:**
- **Name:** `query_user_context`
- **Parameter:** `query` (string) — a natural language question about the user
- **Toolset:** `honcho`

**Example queries the agent might make:**

```
"What are this user's main goals?"
"What communication style does this user prefer?"
"What topics has this user discussed recently?"
"What is this user's technical expertise level?"
```

The tool calls Honcho's dialectic chat API to retrieve relevant user context based on accumulated conversation history.

:::note
The `query_user_context` tool is only available when Honcho is active (API key configured and session context set). It registers in the `honcho` toolset and its availability is checked dynamically.
:::

## Session Management

Honcho sessions track conversation history for user modeling:

- **Session creation** — sessions are created or resumed automatically based on session keys (e.g., `telegram:123456` or CLI session IDs)
- **Message syncing** — new messages are synced to Honcho incrementally (only unsynced messages)
- **Peer configuration** — user messages are observed for learning; assistant messages are not
- **Context prefetch** — before responding, Hermes can prefetch user context (representation + peer card) in a single API call
- **Session rotation** — when sessions reset, old data is preserved in Honcho for continued user modeling

## Migration from Local Memory

When Honcho is activated on an instance that already has local conversation history:

1. **Conversation history** — prior messages can be uploaded to Honcho as a transcript file
2. **Memory files** — existing `MEMORY.md` and `USER.md` files can be uploaded for context

This ensures Honcho has the full picture even when activated mid-conversation.

## Use Cases

- **Personalized responses** — Honcho learns how each user prefers to communicate
- **Goal tracking** — remembers what users are working toward across sessions
- **Expertise adaptation** — adjusts technical depth based on user's background
- **Cross-platform memory** — same user understanding across CLI, Telegram, Discord, etc.
- **Multi-user support** — each user (via messaging platforms) gets their own user model
