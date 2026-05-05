---
sidebar_position: 5
title: "Knowledge Ledger"
description: "Cold, source-attributed recall for decisions, debug histories, skill evaluations, and project state"
---

# Knowledge Ledger

The knowledge ledger is an opt-in cold storage layer for durable, source-backed facts that are too large or too specific for `MEMORY.md` / `USER.md`.

It stores Markdown records under:

```text
$HERMES_HOME/knowledge/
├── inbox/
├── decisions/
├── debug/
├── skill-evals/
├── projects/
├── incidents/
├── entities/
└── sources/
```

## How It Differs from Memory

| Layer | Use for | Prompt cost |
|---|---|---|
| `MEMORY.md` / `USER.md` | Small hot facts that should always be visible | Always injected |
| `session_search` | Past conversation recall | On demand |
| Skills | Procedures and workflows | Loaded when relevant |
| Knowledge ledger | Source-backed decisions, debug findings, project state | On demand only |

Knowledge records are **not injected into the system prompt**. The agent must call `knowledge_search` and then `knowledge_get` when a task needs them.

## Tools

Enable the `knowledge` toolset with `hermes tools` or for a single run with `--toolsets knowledge,...`.

The toolset contains:

- `knowledge_capture` — create a Markdown record.
- `knowledge_search` — lexical search that returns short snippets, not a full dump.
- `knowledge_get` — read one record by id or relative path.

## Source Attribution Rule

Non-inbox records require at least one source. Examples:

- a URL
- a file path
- a command and output handle
- a ticket/PR/issue id
- a conversation reference
- a skill name/version

Unsourced or automatically extracted material must go to `kind="inbox"` as a `candidate` record for later review.

## Good Uses

- Why a technical decision was made.
- Confirmed root causes and rejected debugging paths.
- Skill evaluations: which skill was stale, what was patched, and why.
- Long-running project state that should be retrieved only when needed.
- Incidents and postmortems.

## Avoid

- General user preferences that belong in `USER.md`.
- Tiny environment facts that belong in `MEMORY.md`.
- Procedures that should be skills.
- Bulk conversation logs; use `session_search` for episodic recall.
- Always injecting knowledge records into prompts.

## Example

```json
{
  "kind": "decisions",
  "title": "MemKraft adoption boundary",
  "content": "Use a cold source-attributed ledger, not a hot memory replacement.",
  "sources": ["conversation:discord:Agent/#1/Ping"],
  "confidence": "observed",
  "tags": ["memory", "roi"]
}
```

Later retrieval:

```json
{"query": "MemKraft adoption boundary", "kind": "decisions"}
```

Then read the selected record with `knowledge_get`.
