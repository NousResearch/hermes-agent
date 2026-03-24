---
title: RetainDB Memory
description: Native RetainDB integration for cross-session memory in Hermes.
sidebar_label: RetainDB Memory
sidebar_position: 9
---

# RetainDB Memory

RetainDB adds an optional cross-session memory layer to Hermes. Hermes still keeps its local memory files (`MEMORY.md`, `USER.md`) and `state.db`; RetainDB adds deeper recall, background session ingestion, and explicit memory tools on top.

## What It Does

- Prefetches a compact memory overlay for the current turn
- Queues user and assistant turns for durable background ingestion
- Supports explicit memory writes and deletes through native Hermes tools
- Falls back cleanly to Hermes local memory when RetainDB is unavailable

## Setup

### Interactive setup

```bash
hermes retaindb setup
```

### Non-interactive setup

```bash
hermes retaindb setup --yes --api-key <key> --project <project>
```

The setup command writes:

- `~/.hermes/config.yaml` for RetainDB runtime settings
- `~/.hermes/.env` for `RETAINDB_API_KEY` and optional `RETAINDB_BASE_URL`

## Configuration

Hermes stores RetainDB settings under `retaindb:` in `~/.hermes/config.yaml`:

```yaml
retaindb:
  enabled: true
  base_url: https://api.retaindb.com
  project: my-project
  memory_mode: hybrid
  recall_mode: hybrid
  write_frequency: async
  context_tokens: 1200
  prefetch_timeout_ms: 1500
  flush_batch_size: 50
  disable_tool_exposure: false
  debug_recall_trace: false
  agent_id: hermes
```

### Memory modes

| Mode | Behavior |
|------|----------|
| `hybrid` | Keep Hermes local memory and RetainDB enabled together |
| `retaindb` | Disable local `MEMORY.md` / `USER.md` writes and rely on RetainDB |

### Recall modes

| Mode | Behavior |
|------|----------|
| `hybrid` | Inject RetainDB context and keep the RetainDB tools available |
| `context` | Inject RetainDB context, hide the RetainDB tools |
| `tools` | Expose RetainDB tools only, skip automatic context injection |

### Write frequency

| Setting | Behavior |
|---------|----------|
| `async` | Queue session ingestion in the background |
| `turn` | Flush after every turn |
| integer `N` | Flush every `N` turns |

## Tools

When RetainDB is active, Hermes can expose these tools:

- `retaindb_profile`
- `retaindb_search`
- `retaindb_context`
- `retaindb_remember`
- `retaindb_forget`

Use `retaindb_profile` and `retaindb_search` for raw recall. Use `retaindb_context` when you want a compact synthesized memory block for the current question.

## CLI Commands

```bash
hermes retaindb setup
hermes retaindb status
hermes retaindb test
hermes retaindb mode [hybrid|retaindb]
hermes retaindb tokens --context N
hermes retaindb identity [--session-id SESSION]
```

## How It Fits With Hermes Memory

RetainDB does not replace Hermes session storage. Hermes still uses its own session DB and local memory files unless you switch to `memory_mode: retaindb`.

The common default is:

- Hermes local memory for hot local context
- RetainDB for deeper cross-session recall
- async session ingestion so useful context can survive into later sessions
