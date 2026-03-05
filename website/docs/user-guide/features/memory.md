---
sidebar_position: 3
title: "Persistent Memory"
description: "How Hermes Agent remembers across sessions — MEMORY.md, USER.md, and session search"
---

# Persistent Memory

Hermes Agent has bounded, curated memory that persists across sessions. This lets it remember your preferences, your projects, your environment, and things it has learned.

## How It Works

Two files make up the agent's memory:

| File | Purpose | Token Budget |
|------|---------|-------------|
| **MEMORY.md** | Agent's personal notes — environment facts, conventions, things learned | ~800 tokens (~2200 chars) |
| **USER.md** | User profile — your preferences, communication style, expectations | ~500 tokens (~1375 chars) |

Both are stored in `~/.hermes/memories/` and are injected into the system prompt as a frozen snapshot at session start. The agent manages its own memory via the `memory` tool — it can add, replace, or remove entries.

:::info
Character limits keep memory focused. When memory is full, the agent consolidates or replaces entries to make room for new information.
:::

## Configuration

```yaml
# In ~/.hermes/config.yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
```

## Memory Tool Actions

The agent uses the `memory` tool with these actions:

- **add** — Add a new memory entry
- **replace** — Replace an existing entry with updated content
- **remove** — Remove an entry that's no longer relevant
- **read** — Read current memory contents

## Session Search

Beyond MEMORY.md and USER.md, the agent can search its past conversations using the `session_search` tool:

- All CLI and messaging sessions are stored in SQLite (`~/.hermes/state.db`) with FTS5 full-text search
- Search queries return relevant past conversations with Gemini Flash summarization
- The agent can find things it discussed weeks ago, even if they're not in its active memory

```bash
hermes sessions list    # Browse past sessions
```

## Honcho Integration (Cross-Session User Modeling)

For deeper, AI-generated user understanding that works across tools, you can optionally enable [Honcho](https://honcho.dev/) by Plastic Labs. Honcho runs alongside existing memory — USER.md stays as-is, and Honcho adds an additional layer of context.

When enabled:
- **Prefetch**: Each turn, Honcho's user representation is injected into the system prompt
- **Sync**: After each conversation, messages are synced to Honcho
- **Query tool**: The agent can actively query its understanding of you via `query_user_context`

**Setup:**

```bash
# 1. Install the optional dependency
uv pip install honcho-ai

# 2. Get an API key from https://app.honcho.dev

# 3. Create ~/.honcho/config.json
cat > ~/.honcho/config.json << 'EOF'
{
  "enabled": true,
  "apiKey": "your-honcho-api-key",
  "peerName": "your-name",
  "hosts": {
    "hermes": {
      "workspace": "hermes"
    }
  }
}
EOF
```

Or via environment variable:
```bash
hermes config set HONCHO_API_KEY your-key
```

:::tip
Honcho is fully opt-in — zero behavior change when disabled or unconfigured. All Honcho calls are non-fatal; if the service is unreachable, the agent continues normally.
:::
