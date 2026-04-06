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

```yaml
# ~/.hermes/config.yaml
honcho:
  observation: directional    # "unified" (default for new installs) or "directional"
  peer_name: ""               # auto-detected from platform, or set manually
```

**Observation modes:**
- `unified` — All observations go into a single pool. Simpler, good for single-agent setups.
- `directional` — Observations are tagged with direction (user→agent, agent→user). Enables richer analysis of conversation dynamics.

## Tools

When Honcho is active as the memory provider, four additional tools become available:

| Tool | Purpose |
|------|---------|
| `honcho_conclude` | Trigger server-side dialectic reasoning on recent conversations |
| `honcho_context` | Retrieve relevant context from Honcho's memory for the current conversation |
| `honcho_profile` | View or update the user's Honcho profile |
| `honcho_search` | Semantic search across all stored conclusions and observations |

## CLI Commands

```bash
hermes honcho status          # Show connection status and config
hermes honcho peer            # Update peer names for multi-agent setups
```

## Migrating from `hermes honcho`

If you previously used the standalone `hermes honcho setup`:

1. Your existing configuration (`honcho.json` or `~/.honcho/config.json`) is preserved
2. Your server-side data (memories, conclusions, user profiles) is intact
3. Set `memory.provider: honcho` in config.yaml to reactivate

No re-login or re-setup needed. Run `hermes memory setup` and select "honcho" — the wizard detects your existing config.

## Full Documentation

See [Memory Providers — Honcho](./memory-providers.md#honcho) for the complete reference.
