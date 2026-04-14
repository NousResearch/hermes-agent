---
sidebar_position: 99
title: "Honcho Memory"
description: "AI-native persistent memory via Honcho — dialectic reasoning, multi-agent user modeling, and deep personalization"
---

# Honcho Memory

[Honcho](https://github.com/plastic-labs/honcho) is an AI-native memory backend that adds dialectic reasoning and deep user modeling on top of Hermes's built-in memory system. Instead of simple key-value storage, Honcho maintains a running model of who the user is — their preferences, communication style, goals, and patterns — by reasoning about conversations after they happen.

:::info Honcho is a Memory Provider Plugin
Honcho is integrated into the [Memory Providers](./memory-providers.md) system. All features below are available through the unified memory provider interface.
:::

## What Honcho Adds

| Capability | Built-in Memory | Honcho |
|-----------|----------------|--------|
| Cross-session persistence | ✔ File-based MEMORY.md/USER.md | ✔ Server-side with API |
| User profile | ✔ Manual agent curation | ✔ Automatic dialectic reasoning |
| Multi-agent isolation | — | ✔ Per-peer profile separation |
| Observation modes | — | ✔ Unified or directional observation |
| Conclusions (derived insights) | — | ✔ Server-side reasoning about patterns |
| Search across history | ✔ FTS5 session search | ✔ Semantic search over conclusions |

**Dialectic reasoning**: After each conversation, Honcho analyzes the exchange and derives "conclusions" — insights about the user's preferences, habits, and goals. These conclusions accumulate over time, giving the agent a deepening understanding that goes beyond what the user explicitly stated.

**Multi-agent profiles**: When multiple Hermes instances talk to the same user (e.g., a coding assistant and a personal assistant), Honcho maintains separate "peer" profiles. Each peer sees only its own observations and conclusions, preventing cross-contamination of context.

## Setup

```bash
hermes memory setup    # select "honcho" from the provider list
```

Or configure manually:

```yaml
# ~/.hermes/config.yaml
memory:
  provider: honcho
```

```bash
echo "HONCHO_API_KEY=your-key" >> ~/.hermes/.env
```

Get an API key at [honcho.dev](https://honcho.dev).

## Configuration Options

Honcho is configured in `~/.honcho/config.json` (global) or `$HERMES_HOME/honcho.json` (profile-local). The setup wizard handles this for you.

**Key settings:**

| Setting | Default | Description |
|---------|---------|-------------|
| `sessionStrategy` | `per-directory` | `per-directory`, `per-repo`, `per-session`, or `global` |
| `recallMode` | `hybrid` | `hybrid` (auto-inject + tools), `context` (inject only), `tools` (tools only) |
| `contextTokens` | uncapped | Token budget for auto-injected context per turn. Set to an integer (e.g. 1200) to cap |
| `dialecticReasoningLevel` | `low` | Base reasoning level: `minimal`, `low`, `medium`, `high`, `max` |
| `dialecticDynamic` | `true` | When `true`, model can override reasoning level per-call via tool param |
| `dialecticCadence` | `3` | Turns between Honcho LLM calls (higher = fewer calls) |
| `writeFrequency` | `async` | When to flush messages to Honcho: `async` (background thread), `turn` (sync each turn), `session` (flush on end), or integer N (every N turns) |
| `observation` | all on | Per-peer `observeMe`/`observeOthers` booleans |

**Session strategy** controls how Honcho sessions map to your work:
- `per-session` — each `hermes` run gets a fresh session. Clean starts, memory via tools. Recommended for new users.
- `per-directory` — one Honcho session per working directory. Context accumulates across runs.
- `per-repo` — one session per git repository.
- `global` — single session across all directories.

**Recall mode** controls how memory flows into conversations:
- `hybrid` — context auto-injected into system prompt AND tools available (model decides when to query).
- `context` — auto-injection only, tools hidden.
- `tools` — tools only, no auto-injection. Agent must explicitly call `honcho_reasoning`, `honcho_search`, etc.

**Dialectic cadence** controls cost. With default `3`, Honcho rebuilds the user model every 3 turns instead of every turn — ~66% fewer LLM calls without losing model fidelity.

**Settings per recall mode:**

| Setting | `hybrid` | `context` | `tools` |
|---------|----------|-----------|---------|
| `writeFrequency` | flushes messages | flushes messages | flushes messages |
| `dialecticCadence` | gates auto LLM calls | gates auto LLM calls | irrelevant — model calls explicitly |
| `contextTokens` | caps injection | caps injection | irrelevant — no injection |
| `dialecticDynamic` | gates model override | N/A (no tools) | gates model override |

In `tools` mode, the model is fully in control — it calls `honcho_reasoning` when it wants, at whatever `reasoning_level` it picks. `dialecticCadence` and `contextTokens` only apply to modes with auto-injection (`hybrid` and `context`).

## Tools

When Honcho is active as the memory provider, five tools become available:

| Tool | Purpose |
|------|---------|
| `honcho_profile` | Read or update peer card — pass `card` (list of facts) to update, omit to read |
| `honcho_search` | Semantic search over context — raw excerpts, no LLM synthesis |
| `honcho_context` | Full session context — summary, representation, card, recent messages |
| `honcho_reasoning` | Synthesized answer from Honcho's LLM — pass `reasoning_level` (minimal/low/medium/high/max) to control depth |
| `honcho_conclude` | Create or delete conclusions — pass `conclusion` to create, `delete_id` to remove (PII only) |

## CLI Commands

```bash
hermes honcho status          # Connection status, config, and key settings
hermes honcho setup           # Interactive setup wizard
hermes honcho strategy        # Show or set session strategy
hermes honcho peer            # Update peer names for multi-agent setups
hermes honcho mode            # Show or set recall mode
hermes honcho tokens          # Show or set context token budget
hermes honcho identity        # Show Honcho peer identity
hermes honcho sync            # Sync host blocks for all profiles
hermes honcho enable          # Enable Honcho
hermes honcho disable         # Disable Honcho
```

## Migrating from `hermes honcho`

If you previously used the standalone `hermes honcho setup`:

1. Your existing configuration (`honcho.json` or `~/.honcho/config.json`) is preserved
2. Your server-side data (memories, conclusions, user profiles) is intact
3. Set `memory.provider: honcho` in config.yaml to reactivate

No re-login or re-setup needed. Run `hermes memory setup` and select "honcho" — the wizard detects your existing config.

## Full Documentation

See [Memory Providers — Honcho](./memory-providers.md#honcho) for the complete reference.
