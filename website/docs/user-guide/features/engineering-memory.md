---
title: "Cross-Engineering Memory"
sidebar_label: "Cross-Engineering Memory"
sidebar_position: 8
---

# Cross-Engineering Memory

Hermes can maintain a local shared engineering-memory ledger for reviewed diagnostic knowledge used by Hermes, Claude, and Codex. It is separate from profile-local `MEMORY.md` and `USER.md`: only repository behavior, diagnostics, repair evidence, verified environment facts, and reusable engineering references belong here.

## Safety model

The source ledger is structured Markdown in an explicitly configured vault. A deterministic curator validates schema and path containment, scans for prompt injection and secrets, records provenance, detects duplicates, and stages conflicts. Only explicit human review can approve a record. SQLite is a rebuildable index, not a second source of truth.

The index is diagnostic-only. It cannot authorize live trading, routing, broker operations, deployment, or submit actions. It does not mutate Hermes' stable system prompt, so per-conversation prompt caching remains intact. Hindsight, hosted memory services, and Neural Steering are not required.

## CLI contract

Configure a shared vault and materialized index explicitly, then use:

```text
hermes engineering-memory --vault /shared/engineering-memory --index /shared/engineering-memory/index.sqlite3 propose RECORD.md --json
hermes engineering-memory --vault /shared/engineering-memory --index /shared/engineering-memory/index.sqlite3 review RECORD_ID approve --reviewer HUMAN --reason "Evidence checked" --json
hermes engineering-memory --vault /shared/engineering-memory --index /shared/engineering-memory/index.sqlite3 rebuild --json
hermes engineering-memory --vault /shared/engineering-memory --index /shared/engineering-memory/index.sqlite3 search "runner timeout" --json
hermes engineering-memory --vault /shared/engineering-memory --index /shared/engineering-memory/index.sqlite3 verify --json
```

Every JSON response uses `hermes.engineering_memory.v1`. Search results are bounded and retain record ID, status, source label, verified head, conflicts, related records, and a reason code. Claude and Codex can consume this contract without importing Hermes modules.

## LAYA boundary

LAYA is an optional local decision sidecar. When explicitly downloaded, pinned, and calibrated, it may suggest relevance, evidence sufficiency, conflict likelihood, or a review queue. It cannot approve, rewrite, delete, supersede, compact context, or authorize an external action. Missing, uncalibrated, out-of-distribution, or low-confidence LAYA results fall back to deterministic rules.

The NVIDIA AI-Q research skill may use approved engineering records as diagnostic evidence. The memory index never becomes signal, risk, route, broker, or submit authority.
