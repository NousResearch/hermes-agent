---
title: Honcho Memory
description: AI-native persistent memory for cross-session user modeling and personalization.
sidebar_label: Honcho Memory
sidebar_position: 8
---

# Honcho Memory

[Honcho](https://honcho.dev) is an AI-native memory system that gives Hermes persistent, cross-session understanding of users. While Hermes has built-in memory (`MEMORY.md` and `USER.md`), Honcho adds a deeper layer of **user modeling** — learning preferences, goals, communication style, and context across conversations via a dual-peer architecture where both the user and the AI build representations over time.

## Works Alongside Built-in Memory

Hermes has two memory systems that can work together or be configured separately. In `hybrid` mode (the default), both run side by side — Honcho adds cross-session user modeling while local files handle agent-level notes.

| Feature | Built-in Memory | Honcho Memory |
|---------|----------------|---------------|
| Storage | Local files (`~/.hermes/memories/`) | Cloud-hosted Honcho API |
| Scope | Agent-level notes and user profile | Deep user modeling via dialectic reasoning |
| Persistence | Across sessions on same machine | Across sessions, machines, and platforms |
| Query | Injected into system prompt automatically | Prefetched + on-demand via tools |
| Content | Manually curated by the agent | Automatically learned from conversations |
| Write surface | `memory` tool (add/replace/remove) | `honcho_conclude` tool (persist facts) |

Set `memoryMode` to `honcho` to use Honcho exclusively. See [Memory Modes](#memory-modes) for per-peer configuration.


## Setup

### Interactive Setup

```bash
hermes honcho setup
```

The setup wizard walks through API key, peer names, workspace, memory mode, write frequency, recall mode, and session strategy. It offers to install `honcho-ai` if missing.

### Manual Setup

#### 1. Install the Client Library

```bash
pip install 'honcho-ai>=2.0.1'
```

#### 2. Get an API Key

Go to [app.honcho.dev](https://app.honcho.dev) > Settings > API Keys.

#### 3. Configure

Honcho reads from `~/.honcho/config.json` (shared across all Honcho-enabled applications):

```json
{
  "apiKey": "your-honcho-api-key",
  "hosts": {
    "hermes": {
      "workspace": "hermes",
      "peerName": "your-name",
      "aiPeer": "hermes",
      "memoryMode": "hybrid",
      "writeFrequency": "async",
      "recallMode": "hybrid",
      "sessionStrategy": "per-session",
      "enabled": true
    }
  }
}
```

`apiKey` lives at the root because it is a shared credential across all Honcho-enabled tools. All other settings are scoped under `hosts.hermes`. The `hermes honcho setup` wizard writes this structure automatically.

Or set the API key as an environment variable:

```bash
hermes config set HONCHO_API_KEY your-key
```

:::info
When an API key is present (either in `~/.honcho/config.json` or as `HONCHO_API_KEY`), Honcho auto-enables unless explicitly set to `"enabled": false`.
:::

## Configuration

### Global Config (`~/.honcho/config.json`)

Settings are scoped to `hosts.hermes` and fall back to root-level globals when the host field is absent. Root-level keys are managed by the user or the honcho CLI -- Hermes only writes to its own host block (except `apiKey`, which is a shared credential at root).

**Root-level (shared)**

| Field | Default | Description |
|-------|---------|-------------|
| `apiKey` | — | Honcho API key (required, shared across all hosts) |
| `sessions` | `{}` | Manual session name overrides per directory (shared) |

**Host-level (`hosts.hermes`)**

| Field | Default | Description |
|-------|---------|-------------|
| `workspace` | `"hermes"` | Workspace identifier |
| `peerName` | *(derived)* | Your identity name for user modeling |
| `aiPeer` | `"hermes"` | AI assistant identity name |
| `environment` | `"production"` | Honcho environment |
| `enabled` | *(auto)* | Auto-enables when API key is present |
| `saveMessages` | `true` | Whether to sync messages to Honcho |
| `memoryMode` | `"hybrid"` | Memory mode: `hybrid` or `honcho` |
| `writeFrequency` | `"async"` | When to write: `async`, `turn`, `session`, or integer N |
| `recallMode` | `"hybrid"` | Retrieval strategy: `hybrid`, `context`, or `tools` |
| `sessionStrategy` | `"per-session"` | How sessions are scoped |
| `sessionPeerPrefix` | `false` | Prefix session names with peer name |
| `contextTokens` | *(Honcho default)* | Max tokens for auto-injected context |
| `dialecticReasoningLevel` | `"low"` | Floor for dialectic reasoning: `minimal` / `low` / `medium` / `high` / `max` |
| `dialecticMaxChars` | `600` | Char cap on dialectic results injected into system prompt |
| `linkedHosts` | `[]` | Other host keys whose workspaces to cross-reference |

All host-level fields fall back to the equivalent root-level key if not set under `hosts.hermes`. Existing configs with settings at root level continue to work.

### Memory Modes

| Mode | Effect |
|------|--------|
| `hybrid` | Write to both Honcho and local files (default) |
| `honcho` | Honcho only — skip local file writes |

Memory mode can be set globally or per-peer (user, agent1, agent2, etc):

```json
{
  "memoryMode": {
    "default": "hybrid",
    "hermes": "honcho"
  }
}
```

To disable Honcho entirely, set `enabled: false` or remove the API key.

### Recall Modes

Controls how Honcho context reaches the agent:

| Mode | Behavior |
|------|----------|
| `hybrid` | Auto-injected context + Honcho tools available (default) |
| `context` | Auto-injected context only — Honcho tools hidden |
| `tools` | Honcho tools only — no auto-injected context |

### Write Frequency

| Setting | Behavior |
|---------|----------|
| `async` | Background thread writes (zero blocking, default) |
| `turn` | Synchronous write after each turn |
| `session` | Batched write at session end |
| *integer N* | Write every N turns |

### Session Strategies

| Strategy | Session key | Use case |
|----------|-------------|----------|
| `per-session` | Unique per run | Default. Fresh session every time. |
| `per-directory` | CWD basename | Each project gets its own session. |
| `per-repo` | Git repo root name | Groups subdirectories under one session. |
| `global` | Fixed `"global"` | Single cross-project session. |

Resolution order: manual map > session title > strategy-derived key > platform key.

### Multi-host Configuration

Multiple Honcho-enabled tools share `~/.honcho/config.json`. Each tool writes only to its own host block, reads its host block first, and falls back to root-level globals:

```json
{
  "apiKey": "your-key",
  "peerName": "eri",
  "hosts": {
    "hermes": {
      "workspace": "my-workspace",
      "aiPeer": "hermes-assistant",
      "memoryMode": "honcho",
      "linkedHosts": ["claude-code"],
      "contextTokens": 2000,
      "dialecticReasoningLevel": "medium"
    },
    "claude-code": {
      "workspace": "my-workspace",
      "aiPeer": "clawd"
    }
  }
}
```

Resolution: `hosts.<tool>` field > root-level field > default. In this example, both tools share the root `apiKey` and `peerName`, but each has its own `aiPeer` and workspace settings.

### Hermes Config (`~/.hermes/config.yaml`)

Intentionally minimal — most configuration comes from `~/.honcho/config.json`:

```yaml
honcho: {}
```

## How It Works

### Async Context Pipeline

Honcho context is fetched asynchronously to avoid blocking the response path:

```
Turn N:
  user message
    → consume cached context (from previous turn's background fetch)
    → inject into system prompt (user representation, AI representation, dialectic)
    → LLM call
    → response
    → fire background fetch for next turn
         → fetch context    ─┐
         → fetch dialectic  ─┴→ cache for Turn N+1
```

Turn 1 is a cold start (no cache). All subsequent turns consume cached results with zero HTTP latency on the response path. The system prompt on turn 1 uses only static context to preserve prefix cache hits at the LLM provider.

### Dual-Peer Architecture

Both the user and AI have peer representations in Honcho:

- **User peer** — observed from user messages. Honcho learns preferences, goals, communication style.
- **AI peer** — observed from assistant messages (`observe_me=True`). Honcho builds a representation of the agent's knowledge and behavior.

Both representations are injected into the system prompt when available.

### Dynamic Reasoning Level

Dialectic queries scale reasoning effort with message complexity:

| Message length | Reasoning level |
|----------------|-----------------|
| < 120 chars | Config default (typically `low`) |
| 120-400 chars | One level above default (cap: `high`) |
| > 400 chars | Two levels above default (cap: `high`) |

`max` is never selected automatically.

### Gateway Integration

The gateway creates short-lived `AIAgent` instances per request. Honcho managers are owned at the gateway session layer (`_honcho_managers` dict) so they persist across requests within the same session and flush at real session boundaries (reset, resume, expiry, server stop).

## Tools

When Honcho is active, four tools become available. Availability is gated dynamically — they are invisible when Honcho is disabled.

### `honcho_profile`

Fast peer card retrieval (no LLM). Returns a curated list of key facts about the user.

### `honcho_search`

Semantic search over memory (no LLM). Returns raw excerpts ranked by relevance. Cheaper and faster than `honcho_context` — good for factual lookups.

Parameters:
- `query` (string) — search query
- `max_tokens` (integer, optional) — result token budget

### `honcho_context`

Dialectic Q&A powered by Honcho's LLM. Synthesizes an answer from accumulated conversation history.

Parameters:
- `query` (string) — natural language question
- `peer` (string, optional) — `"user"` (default) or `"ai"`. Querying `"ai"` asks about the assistant's own history and identity.

Example queries the agent might make:

```
"What are this user's main goals?"
"What communication style does this user prefer?"
"What topics has this user discussed recently?"
"What is this user's technical expertise level?"
```

### `honcho_conclude`

Writes a fact to Honcho memory. Use when the user explicitly states a preference, correction, or project context worth remembering. Feeds into the user's peer card and representation.

Parameters:
- `conclusion` (string) — the fact to persist

## CLI Commands

```
hermes honcho setup                        # Interactive setup wizard
hermes honcho status                       # Show config and connection status
hermes honcho sessions                     # List directory → session name mappings
hermes honcho map <name>                   # Map current directory to a session name
hermes honcho peer                         # Show peer names and dialectic settings
hermes honcho peer --user NAME             # Set user peer name
hermes honcho peer --ai NAME               # Set AI peer name
hermes honcho peer --reasoning LEVEL       # Set dialectic reasoning level
hermes honcho mode                         # Show current memory mode
hermes honcho mode [hybrid|honcho]         # Set memory mode
hermes honcho tokens                       # Show token budget settings
hermes honcho tokens --context N           # Set context token cap
hermes honcho tokens --dialectic N         # Set dialectic char cap
hermes honcho identity                     # Show AI peer identity
hermes honcho identity <file>              # Seed AI peer identity from file (SOUL.md, etc.)
hermes honcho migrate                      # Migration guide: OpenClaw → Hermes + Honcho
```

### Doctor Integration

`hermes doctor` includes a Honcho section that validates config, API key, and connection status.

## Migration

### From Local Memory

When Honcho activates on an instance with existing local history, migration runs automatically:

1. **Conversation history** — prior messages are uploaded as an XML transcript file
2. **Memory files** — existing `MEMORY.md`, `USER.md`, and `SOUL.md` are uploaded for context

### From OpenClaw

```bash
hermes honcho migrate
```

Walks through converting an OpenClaw native Honcho setup to the shared `~/.honcho/config.json` format.

## AI Peer Identity

Honcho can build a representation of the AI assistant over time (via `observe_me=True`). You can also seed the AI peer explicitly:

```bash
hermes honcho identity ~/.hermes/SOUL.md
```

This uploads the file content through Honcho's observation pipeline. The AI peer representation is then injected into the system prompt alongside the user's, giving the agent awareness of its own accumulated identity.

```bash
hermes honcho identity --show
```

Shows the current AI peer representation from Honcho.

## Use Cases

- **Personalized responses** — Honcho learns how each user prefers to communicate
- **Goal tracking** — remembers what users are working toward across sessions
- **Expertise adaptation** — adjusts technical depth based on user's background
- **Cross-platform memory** — same user understanding across CLI, Telegram, Discord, etc.
- **Multi-user support** — each user (via messaging platforms) gets their own user model

:::tip
Honcho is fully opt-in — zero behavior change when disabled or unconfigured. All Honcho calls are non-fatal; if the service is unreachable, the agent continues normally.
:::
