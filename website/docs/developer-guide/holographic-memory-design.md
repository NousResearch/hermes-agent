---
title: "Holographic Memory Design"
description: "Design for Hermes Agent's local holographic memory provider."
---

# Holographic Memory Design

## Decision

Build and keep Holographic as an optional local memory provider, not a replacement for built-in memory or session search.

Rationale:
- Hermes already has bounded, high-signal built-in memory (`MEMORY.md` and `USER.md`) for always-on context.
- Hermes already has session search for full-transcript recall across past sessions.
- The remaining gap is local, structured, queryable fact recall that can answer entity-centric and compositional questions without relying on a cloud service.
- The existing provider plugin architecture can host this cleanly behind `memory.provider: holographic`, so no core memory rewrite is needed.

Do not make Holographic the default until it has targeted tests, setup/status verification, and a safe migration/import path from built-in memory.

## Goals

- Store durable facts in a local profile-scoped SQLite database.
- Retrieve facts by keyword, entity, related entity structure, and multi-entity composition.
- Preserve trust and freshness metadata so stale facts can sink rather than dominate recall.
- Keep built-in memory authoritative for compact always-injected facts.
- Keep all provider context fenced so recalled memory cannot leak into visible output or be recaptured as user text.

## Non-goals

- Replacing `MEMORY.md` or `USER.md`.
- Replacing `session_search` over session transcripts.
- Adding a remote service dependency.
- Using embeddings as the core representation for this provider.
- Automatically ingesting every conversation turn by default.

## User stories

1. As a user, I can enable `memory.provider: holographic` and get deeper recall while keeping normal built-in memory behavior.
2. As an agent, I can explicitly store a durable fact with a category and tags when it is too detailed for built-in memory.
3. As an agent, I can search facts with keywords before answering a question about prior work.
4. As an agent, I can probe an entity such as a person, project, tool, host, or repo and retrieve facts connected to it.
5. As an agent, I can reason over multiple entities, e.g. project + repo + deployment target, to find intersecting facts.
6. As a user, I can correct bad recall and have that fact's trust decrease or the fact removed.
7. As a privacy-conscious user, I can inspect and delete the local SQLite database and disable the provider without losing built-in memory.

## Existing Hermes memory landscape

### Built-in memory

Code paths:
- `tools/memory_tool.py`
- `run_agent.py` built-in memory initialization and prompt injection
- `$HERMES_HOME/memories/MEMORY.md`
- `$HERMES_HOME/memories/USER.md`

Behavior:
- Compact curated memory files are loaded into the system prompt when enabled.
- Character limits keep this always-on context bounded.
- The `memory` tool handles deliberate add/replace/remove actions.

Gap:
- Great for durable high-signal facts, but intentionally too small for deep recall.

### Session search

Code path:
- `tools/session_search_tool.py`
- `hermes_state.py` session storage and FTS-backed search

Behavior:
- Searches historical session transcripts and summarizes matching sessions.

Gap:
- Good for reconstructing past work, but not a structured fact graph and not designed as an explicit fact store with trust scoring.

### External memory providers

Code paths:
- `agent/memory_provider.py`
- `agent/memory_manager.py`
- `plugins/memory/*`
- `run_agent.py` provider initialization, tool injection, prompt block injection, prefetch, sync, and lifecycle hook calls

Behavior:
- At most one external provider runs alongside built-in memory.
- Providers may add prompt blocks, prefetched context, tools, turn sync, and lifecycle hooks.

Gap Holographic fills:
- Local-only structured facts with entity resolution, trust scoring, and HRR-style compositional retrieval.

## Target behavior

When enabled, Holographic should:
1. Initialize a local SQLite store under the active profile's `$HERMES_HOME` unless configured otherwise.
2. Add two tools:
   - `fact_store` for add/search/probe/related/reason/contradict/update/remove/list.
   - `fact_feedback` for helpful/unhelpful ratings.
3. Add a short provider prompt block saying the fact store is active and how many facts exist.
4. Prefetch a small set of relevant facts before a turn and inject them through the memory manager's fenced context path.
5. Mirror explicit built-in memory adds into Holographic as facts.
6. Optionally run conservative session-end extraction only when `auto_extract: true`.
7. Fail closed: provider errors should degrade to no extra recall, not break the agent loop.

## Data model

SQLite database: default `$HERMES_HOME/memory_store.db`.

Tables:
- `facts`
  - `fact_id` primary key
  - `content` unique fact text
  - `category` such as `user_pref`, `project`, `tool`, `general`
  - `tags` comma-separated lightweight labels
  - `trust_score` float in `[0.0, 1.0]`
  - `retrieval_count`
  - `helpful_count`
  - `created_at`, `updated_at`
  - `hrr_vector` optional serialized phase vector
- `entities`
  - `entity_id`
  - `name`
  - `entity_type`
  - `aliases`
  - `created_at`
- `fact_entities`
  - many-to-many fact/entity links
- `facts_fts`
  - FTS5 virtual table over fact content and tags
- `memory_banks`
  - bundled category vectors for HRR probing

Rules:
- Deduplicate by exact `content`.
- Keep schema profile-scoped by default.
- Treat HRR vectors as derived data: they can be regenerated from facts and entities.
- If NumPy is unavailable, keep FTS and trust scoring functional and disable HRR scoring paths.

## Retrieval behavior

Supported paths:
- `search`: FTS5 candidate retrieval, reranked by token overlap, trust, optional temporal decay, and optional HRR similarity.
- `probe`: entity-centric recall for facts structurally connected to one entity.
- `related`: adjacency-style recall for facts sharing structure with an entity.
- `reason`: multi-entity compositional recall for facts connected to all requested entities.
- `contradict`: candidate conflicting facts for memory hygiene.
- `list`: recent or filtered facts for inspection.

Ranking principles:
- Relevant facts with higher trust should outrank low-trust facts.
- Unhelpful feedback should penalize more strongly than helpful feedback rewards.
- Prefetch must stay small; the provider should return only a concise handful of facts.
- Raw provider context must be wrapped by `build_memory_context_block()` before model injection.

## Privacy and safety

- Local-first: default data stays in `$HERMES_HOME/memory_store.db`.
- Profile isolation: each profile has its own `$HERMES_HOME`, so local Holographic stores are separate by default.
- No secrets in config docs or issue comments; secret values still belong in `.env` if a future provider variant needs them.
- Explicit writes by default: the agent should add durable facts intentionally via tools.
- `auto_extract` default is `false` because automatic extraction can capture sensitive or low-signal content.
- Recalled memory is system context, not new user input; keep memory-manager fencing and streaming scrubbers in place.
- Deletion must be possible by fact id and by disabling/removing the local DB.
- Subagents and cron contexts should not silently pollute primary user memory; provider initialization receives `agent_context` for future guardrails.

## Migration path

Phase 0: no migration.
- Enabling Holographic starts with an empty store.
- Built-in memory remains authoritative and unchanged.

Phase 1: mirror new built-in memory writes.
- `on_memory_write(add, ...)` mirrors new `memory` tool additions into the fact store.
- Replacements/removals should eventually update or tombstone mirrored facts, but should not block initial usage.

Phase 2: explicit import command.
- Add an opt-in command or setup prompt to import existing `MEMORY.md` and `USER.md` entries into facts.
- Preserve source metadata (`memory` vs. `user`) and default categories.
- Dry-run before writing.

Phase 3: maintenance tools.
- Add dedupe, contradiction review, low-trust pruning, and rebuild-vector commands if real usage shows the store needs them.

## Impacted code paths

Provider interfaces and orchestration:
- `agent/memory_provider.py` defines lifecycle hooks and config contract.
- `agent/memory_manager.py` registers one external provider, fences prefetched context, routes provider tools, and calls lifecycle hooks.
- `run_agent.py` reads `memory.provider`, initializes the provider, injects provider tools, adds provider prompt blocks, and calls prefetch/sync hooks.

Holographic provider:
- `plugins/memory/holographic/__init__.py` implements `HolographicMemoryProvider`, tool schemas, tool dispatch, prompt block, prefetch, feedback, and optional extraction.
- `plugins/memory/holographic/store.py` owns SQLite schema, CRUD, entity extraction, trust updates, and memory bank maintenance.
- `plugins/memory/holographic/retrieval.py` owns hybrid scoring and HRR-aware retrieval modes.
- `plugins/memory/holographic/holographic.py` owns HRR phase-vector algebra.
- `plugins/memory/holographic/plugin.yaml` registers provider metadata.

CLI and docs:
- `hermes_cli/memory_setup.py` and `hermes_cli/main.py` expose setup/status/reset flows.
- `hermes_cli/config.py` defines defaults for `memory.memory_enabled`, `memory.user_profile_enabled`, and `memory.provider`.
- `website/docs/user-guide/features/memory.md` and `website/docs/user-guide/features/memory-providers.md` describe user behavior.
- `website/docs/developer-guide/memory-provider-plugin.md` describes provider authoring.

Tests to add or harden:
- Provider initialization and config expansion.
- Tool schema injection without duplicate tool names.
- SQLite CRUD and FTS search.
- Fallback behavior when NumPy is unavailable.
- Memory-manager context fencing for Holographic prefetch.
- Built-in memory write mirroring.

## Config keys

Core memory keys:
```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  provider: holographic
```

Holographic provider keys:
```yaml
plugins:
  hermes-memory-store:
    db_path: $HERMES_HOME/memory_store.db
    auto_extract: false
    default_trust: 0.5
    min_trust_threshold: 0.3
    temporal_decay_half_life: 0
    hrr_dim: 1024
    hrr_weight: 0.3
```

Recommended defaults:
- `auto_extract: false`
- `default_trust: 0.5`
- `min_trust_threshold: 0.3`
- `hrr_dim: 1024`
- `hrr_weight: 0.3` when NumPy is available, redistributed to text scores when it is not

## Build/buy/defer analysis

Build:
- Choose this for a local, inspectable, zero-service structured fact store.
- Fits the existing memory provider plugin architecture.
- Avoids cloud privacy concerns and recurring service cost.
- Enables provider-specific tools that do not belong in built-in memory.

Buy:
- Prefer Honcho, Mem0, Hindsight, RetainDB, ByteRover, or Supermemory when the user wants cloud sync, hosted user modeling, semantic embeddings, knowledge graphs, or managed memory infrastructure.
- These should remain alternatives under the same `memory.provider` selector.

Defer:
- Defer automatic ingestion-by-default, broad migrations, embedding/vector DB integration, multi-provider blending, and replacing built-in memory.
- Defer making Holographic default until test coverage and operational polish are adequate.

Decision: build the local provider as optional, buy/enable another provider for users who need managed or semantic cloud memory, and defer default activation plus automatic migration.

## Scoped implementation tickets

1. Holographic provider test coverage
   - Add targeted tests for `MemoryStore`, `FactRetriever`, provider tool dispatch, config expansion, and NumPy-unavailable fallback.
   - Verification: `python -m pytest tests/agent/test_memory_provider.py tests/plugins/test_holographic_memory.py -q -o 'addopts='`.

2. Built-in memory import and mirror parity
   - Add an explicit import path from `MEMORY.md` and `USER.md` to Holographic facts.
   - Extend mirroring so replace/remove operations do not leave stale mirrored facts.
   - Verification: tests create temporary built-in memories, import them, replace/remove them, and assert fact store parity.

3. Setup/status polish for Holographic
   - Ensure `hermes memory setup`, `hermes memory status`, and docs show the resolved DB path, fact count, NumPy/HRR availability, and auto-extract state.
   - Verification: CLI tests with temp `HERMES_HOME`; no real user memory files touched.

4. Safe auto-extraction experiment
   - Keep `auto_extract` opt-in.
   - Add extraction caps, filtering, and reviewable logs before storing facts from sessions.
   - Verification: fixture session produces only durable facts; trivial/sensitive examples are skipped.

## Open questions

- Should Holographic expose an import command under `hermes memory import holographic`, or should setup offer a one-time import prompt?
- Should mirrored built-in memory facts carry a stable source id to support exact replace/remove parity?
- Should `hrr_dim` be fixed after first database creation to avoid mixed-vector dimensions, or should vectors store their own dimensions per fact?
- Should category values remain a fixed enum or become free-form tags?
