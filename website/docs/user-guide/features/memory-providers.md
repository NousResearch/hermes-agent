---
sidebar_position: 4
title: "Memory Providers"
description: "External memory provider plugins — Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover"
---

# Memory Providers

Hermes Agent ships with 7 external memory provider plugins that give the agent persistent, cross-session knowledge beyond the built-in MEMORY.md and USER.md. Only **one** external provider can be active at a time — the built-in memory is always active alongside it.

## Quick Start

```bash
hermes memory setup      # interactive picker + configuration
hermes memory status     # check what's active
hermes memory off        # disable external provider
```

Or set manually in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: openviking   # or honcho, mem0, hindsight, holographic, retaindb, byterover
```

## How It Works

When a memory provider is active, Hermes automatically:

1. **Injects provider context** into the system prompt (what the provider knows)
2. **Prefetches relevant memories** before each turn (background, non-blocking)
3. **Syncs conversation turns** to the provider after each response
4. **Extracts memories on session end** (for providers that support it)
5. **Mirrors built-in memory writes** to the external provider
6. **Adds provider-specific tools** so the agent can search, store, and manage memories

The built-in memory (MEMORY.md / USER.md) continues to work exactly as before. The external provider is additive.

## Available Providers

### Honcho

AI-native cross-session user modeling with dialectic Q&A, semantic search, and persistent conclusions.

| | |
|---|---|
| **Best for** | Teams using Honcho's user modeling platform |
| **Requires** | `pip install honcho-ai` + API key |
| **Data storage** | Honcho Cloud |
| **Cost** | Honcho pricing |

**Tools:** `honcho_profile` (peer card), `honcho_search` (semantic search), `honcho_context` (LLM-synthesized), `honcho_conclude` (store facts)

**Setup:**
```bash
hermes memory setup    # select "honcho"
# Or manually:
hermes config set memory.provider honcho
echo "HONCHO_API_KEY=your-key" >> ~/.hermes/.env
```

**Config:** `$HERMES_HOME/honcho.json` — existing Honcho users' configuration and data are fully preserved.

:::tip Migrating from `hermes honcho`
If you previously used `hermes honcho setup`, your config and all server-side data are intact. Just set `memory.provider: honcho` to reactivate via the new system.
:::

---

### OpenViking

Context database by Volcengine (ByteDance) with filesystem-style knowledge hierarchy, tiered retrieval, and automatic memory extraction into 6 categories.

| | |
|---|---|
| **Best for** | Self-hosted knowledge management with structured browsing |
| **Requires** | `pip install openviking` + running server |
| **Data storage** | Self-hosted (local or cloud) |
| **Cost** | Free (open-source, AGPL-3.0) |

**Tools:** `viking_search` (semantic search), `viking_read` (tiered: abstract/overview/full), `viking_browse` (filesystem navigation), `viking_remember` (store facts), `viking_add_resource` (ingest URLs/docs)

**Setup:**
```bash
# Start the OpenViking server first
pip install openviking
openviking-server

# Then configure Hermes
hermes memory setup    # select "openviking"
# Or manually:
hermes config set memory.provider openviking
echo "OPENVIKING_ENDPOINT=http://localhost:1933" >> ~/.hermes/.env
```

**Key features:**
- Tiered context loading: L0 (~100 tokens) → L1 (~2k) → L2 (full)
- Automatic memory extraction on session commit (profile, preferences, entities, events, cases, patterns)
- `viking://` URI scheme for hierarchical knowledge browsing

---

### Mem0

Server-side LLM fact extraction with semantic search, reranking, and automatic deduplication.

| | |
|---|---|
| **Best for** | Hands-off memory management — Mem0 handles extraction automatically |
| **Requires** | `pip install mem0ai` + API key |
| **Data storage** | Mem0 Cloud |
| **Cost** | Mem0 pricing |

**Tools:** `mem0_profile` (all stored memories), `mem0_search` (semantic search + reranking), `mem0_conclude` (store verbatim facts)

**Setup:**
```bash
hermes memory setup    # select "mem0"
# Or manually:
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

**Config:** `$HERMES_HOME/mem0.json`

| Key | Default | Description |
|-----|---------|-------------|
| `user_id` | `hermes-user` | User identifier |
| `agent_id` | `hermes` | Agent identifier |

---

### Hindsight

Long-term memory with knowledge graph, entity resolution, and multi-strategy retrieval. The `hindsight_reflect` tool provides cross-memory synthesis that no other provider offers.

| | |
|---|---|
| **Best for** | Knowledge graph-based recall with entity relationships |
| **Requires** | Cloud: `pip install hindsight-client` + API key. Local: `pip install hindsight` + LLM key |
| **Data storage** | Hindsight Cloud or local embedded PostgreSQL |
| **Cost** | Hindsight pricing (cloud) or free (local) |

**Tools:** `hindsight_retain` (store with entity extraction), `hindsight_recall` (multi-strategy search), `hindsight_reflect` (cross-memory synthesis)

**Setup:**
```bash
hermes memory setup    # select "hindsight"
# Or manually:
hermes config set memory.provider hindsight
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
```

**Config:** `$HERMES_HOME/hindsight/config.json`

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud` or `local` |
| `bank_id` | `hermes` | Memory bank identifier |
| `budget` | `mid` | Recall thoroughness: `low` / `mid` / `high` |

---

### Holographic

Local SQLite fact store with FTS5 full-text search, trust scoring, and HRR (Holographic Reduced Representations) for compositional algebraic queries.

| | |
|---|---|
| **Best for** | Local-only memory with advanced retrieval, no external dependencies |
| **Requires** | Nothing (SQLite is always available). NumPy optional for HRR algebra. |
| **Data storage** | Local SQLite |
| **Cost** | Free |

**Tools:** `fact_store` (9 actions: add, search, probe, related, reason, contradict, update, remove, list), `fact_feedback` (helpful/unhelpful rating that trains trust scores)

**Setup:**
```bash
hermes memory setup    # select "holographic"
# Or manually:
hermes config set memory.provider holographic
```

**Config:** `config.yaml` under `plugins.hermes-memory-store`

| Key | Default | Description |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memory_store.db` | SQLite database path |
| `auto_extract` | `false` | Auto-extract facts at session end |
| `default_trust` | `0.5` | Default trust score (0.0–1.0) |

**Unique capabilities:**
- `probe` — entity-specific algebraic recall (all facts about a person/thing)
- `reason` — compositional AND queries across multiple entities
- `contradict` — automated detection of conflicting facts
- Trust scoring with asymmetric feedback (+0.05 helpful / -0.10 unhelpful)

---

### RetainDB

Cloud memory API with hybrid search (Vector + BM25 + Reranking), 7 memory types, and delta compression.

| | |
|---|---|
| **Best for** | Teams already using RetainDB's infrastructure |
| **Requires** | RetainDB account + API key |
| **Data storage** | RetainDB Cloud |
| **Cost** | $20/month |

**Tools:** `retaindb_profile` (user profile), `retaindb_search` (semantic search), `retaindb_context` (task-relevant context), `retaindb_remember` (store with type + importance), `retaindb_forget` (delete memories)

**Setup:**
```bash
hermes memory setup    # select "retaindb"
# Or manually:
hermes config set memory.provider retaindb
echo "RETAINDB_API_KEY=your-key" >> ~/.hermes/.env
```

---

### ByteRover

Persistent memory via the `brv` CLI — hierarchical knowledge tree with tiered retrieval (fuzzy text → LLM-driven search). Local-first with optional cloud sync.

| | |
|---|---|
| **Best for** | Developers who want portable, local-first memory with a CLI |
| **Requires** | ByteRover CLI (`npm install -g byterover-cli` or [install script](https://byterover.dev)) |
| **Data storage** | Local (default) or ByteRover Cloud (optional sync) |
| **Cost** | Free (local) or ByteRover pricing (cloud) |

**Tools:** `brv_query` (search knowledge tree), `brv_curate` (store facts/decisions/patterns), `brv_status` (CLI version + tree stats)

**Setup:**
```bash
# Install the CLI first
curl -fsSL https://byterover.dev/install.sh | sh

# Then configure Hermes
hermes memory setup    # select "byterover"
# Or manually:
hermes config set memory.provider byterover
```

**Key features:**
- Automatic pre-compression extraction (saves insights before context compression discards them)
- Knowledge tree stored at `$HERMES_HOME/byterover/` (profile-scoped)
- SOC2 Type II certified cloud sync (optional)

---

## Provider Comparison

| Provider | Storage | Cost | Tools | Dependencies | Unique Feature |
|----------|---------|------|-------|-------------|----------------|
| **Honcho** | Cloud | Paid | 4 | `honcho-ai` | Dialectic user modeling |
| **OpenViking** | Self-hosted | Free | 5 | `openviking` + server | Filesystem hierarchy + tiered loading |
| **Mem0** | Cloud | Paid | 3 | `mem0ai` | Server-side LLM extraction |
| **Hindsight** | Cloud/Local | Free/Paid | 3 | `hindsight-client` | Knowledge graph + reflect synthesis |
| **Holographic** | Local | Free | 2 | None | HRR algebra + trust scoring |
| **RetainDB** | Cloud | $20/mo | 5 | `requests` | Delta compression |
| **ByteRover** | Local/Cloud | Free/Paid | 3 | `brv` CLI | Pre-compression extraction |

## Profile Isolation

Each provider's data is isolated per [profile](/docs/user-guide/profiles):

- **Local storage providers** (Holographic, ByteRover) use `$HERMES_HOME/` paths which differ per profile
- **Config file providers** (Honcho, Mem0, Hindsight) store config in `$HERMES_HOME/` so each profile has its own credentials
- **Cloud providers** (RetainDB) auto-derive profile-scoped project names
- **Env var providers** (OpenViking) are configured via each profile's `.env` file

## Building a Memory Provider

See the [Developer Guide: Memory Provider Plugins](/docs/developer-guide/memory-provider-plugin) for how to create your own.
