# Kanban acceptance-evidence contract (v1)

This is the runtime-owned completion contract for a Kanban task. It makes a task's acceptance criteria machine-checkable without adding a new model tool: callers use the existing task creation/edit surfaces to declare the contract, `kanban_complete` to submit observations, and `kanban_show`/CLI/dashboard reads to retrieve the resulting task and run records.

The canonical JSON Schema is `docs/kanban/acceptance-evidence.schema.json`. Runtime validation implements the relational rules below in addition to JSON Schema validation.

## Persistent shape

`tasks.acceptance_evidence` is nullable JSON text. `NULL` is the legacy/default state and means no evidence gate is declared. The value is a v1 object:

```json
{
  "version": 1,
  "required": [
    {"id": "focused-tests", "kind": "check", "description": "Focused deterministic tests pass"},
    {"id": "canary", "kind": "canary", "description": "Spawned worker has a later heartbeat"},
    {"id": "pr", "kind": "pr", "description": "Required PR checks are clean", "applicable": false}
  ],
  "observed": [
    {
      "id": "focused-tests",
      "status": "passed",
      "observed_at": 1789400001,
      "source": "pytest",
      "payload": {"command": "pytest tests/... -q", "exit_code": 0}
    },
    {
      "id": "canary",
      "status": "passed",
      "observed_at": 1789400003,
      "source": "dispatcher",
      "payload": {"pid": 1234, "run_id": 44, "spawned_at": 1789400000, "heartbeat_at": 1789400002}
    }
  ]
}
```

The declaration (`required`) is task-owned and is changed only through an explicit task mutation. The observation snapshot submitted at completion is stored in the closing `task_runs.metadata.acceptance_evidence` and mirrored in an append-only `task_events` `acceptance_evidence` receipt. The task declaration is not overwritten by an attempt. This keeps retries auditable and prevents a later worker from silently weakening requirements.

`required[].id` is unique, lower-kebab-case, and names the proof. Every applicable required id must have exactly one final observation. An observation must reference a declared id; duplicate observations and `not_applicable` for an applicable requirement are invalid. `passed` is the only satisfier. `failed` and `pending` keep the card in-flight. `not_applicable` is allowed only when the requirement declaration set `applicable: false`.

A `canary` requirement is satisfied only when its passed payload contains a positive `pid`, `run_id`, `spawned_at`, and `heartbeat_at`, with `heartbeat_at > spawned_at`. The PID/run id must be read back from the active run and the heartbeat must be a later durable heartbeat for that same run; caller-supplied timestamps alone are insufficient.

A `pr` requirement delegates exact-head required-context verification to the existing `completion_contract`/`metadata.published_pr` receipt path. Its passed observation must retain that receipt's PR URL and current head SHA. A card with an applicable PR item cannot be accepted as `local-only`.

## Completion gate and blocker semantics

`complete_task` is the single terminal boundary for worker tools, CLI, review approval, dashboard `done`, and reconciliation-assisted closure. It must reject completion when a declared contract is invalid, any applicable item is unsatisfied, the canary readback is stale/non-matching, or a PR receipt is not clean. Rejection records an `acceptance_evidence_rejected` event containing only requirement ids, classifications, and safe recovery text; it does not mark the task done or clear its claim.

An explicit irreducible blocker is the only alternative terminal explanation for unsatisfied evidence. It is represented by the schema's `blocker` object and by the existing `block_task(kind=...)` transition. It must have `irreducible: true`, a typed kind, a concise reason, and a timestamp. A capability or needs-input blocker routes to `blocked`; a dependency blocker uses normal parent gating. A transient observation failure is not irreducible and must not be relabeled as one merely to bypass the completion gate.

## Validation and compatibility

1. Creation and task-edit paths validate JSON Schema and the relational rules before committing `tasks.acceptance_evidence`.
2. Completion re-validates the stored declaration and submitted snapshot under the existing SQLite write transaction, after the existing parent and exact-head PR checks.
3. Existing cards have `acceptance_evidence=NULL`; all current lifecycle behavior is unchanged. Migration adds a nullable column with no backfill. A malformed legacy value is treated as a rejected completion, never as an implicit pass.
4. Unknown top-level versions are rejected for mutation and completion. Unknown keys in v1 are rejected by the schema to prevent silently ignored proof requirements.
5. Existing `completion_contract` remains the PR-policy field; it is not renamed or overloaded.

## Consumer map

| Consumer | Read/write responsibility |
|---|---|
| `hermes_cli/kanban_db.py` | Owns column migration, declaration/observation validation, transactional completion gate, run metadata and event receipt. |
| `hermes_cli/kanban_db_dispatch.py` | Produces authoritative spawn PID and heartbeat readbacks; reconciliation uses unsatisfied evidence as a reason to retain/reclaim rather than silently complete. |
| `tools/kanban_tools.py` and `tools/kanban_tools_schemas.py` | Expose optional `acceptance_evidence` on create/complete and return safe rejection data through existing tool responses. No new core tool. |
| `hermes_cli/kanban.py`, parser, output | Accept JSON flags, show declaration plus latest receipt, and preserve legacy card output. |
| `plugins/kanban/dashboard/plugin_api.py` and dashboard UI | Include declaration and latest receipt in existing task/run endpoints; display statuses but never raw secrets. |
| Gateway/notifier/reconciliation | Consume event classification and safe summary only; do not treat a completion comment, local test, or PR URL as proof without the persisted receipt. |

## Security and redaction

Evidence is audit data, not a secret store. Before persistence, values pass the existing sensitive-text redactor recursively; secret-bearing keys (`token`, `secret`, `password`, `authorization`, credential variants) are rejected rather than retained. Payloads are capped (4 KiB per observation; 16 KiB serialized contract) and must contain receipts, identifiers, URLs, command names, exit classifications, and timestamps—not stdout dumps, environment values, stack traces, or unbounded tool output.

Dashboard and tool reads use the redacted persisted form. Events contain a minimal classification/ids summary; full redacted receipts remain in run metadata for authorized task reads. Attachments remain in the existing attachment table and must be referred to by attachment id/path metadata, never inlined into evidence JSON.

## Test obligations

The contract test suite must cover JSON serialization round trips, a nullable legacy card, malformed/unknown versions, duplicate and missing requirement observations, canary PID/run/strictly-later-heartbeat validation, clean and failing PR receipts, event redaction, and completion rejection without a state transition. The executable schema baseline is `tests/hermes_cli/test_kanban_acceptance_evidence_contract.py`.
