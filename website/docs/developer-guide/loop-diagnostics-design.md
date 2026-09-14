# Loop Diagnostics — Contract and Architecture (MVP)

Status: Draft for review · Owner: default · Board: ops
Sources: [arXiv:2607.06273 (AgentTether)](https://arxiv.org/abs/2607.06273) · [cobusgreyling/loop-engineering](https://github.com/cobusgreyling/loop-engineering) (structured loop patterns)
Related: `docs/observability/README.md` (observer hooks v1) · `hermes_cli/kanban_diagnostics.py` (existing task-level diagnostics) · `hermes_cli/kanban_db.py` (worker execution, retry, logging, failure paths)

This document is the implementation-ready contract for the loop-diagnostics MVP. It is deliberately **bounded to failure diagnosis** of Kanban worker runs — it is NOT a general observability rewrite. The graph recorder and the diagnosis engine are two independent components that agree on this contract and nothing else.

---

## 1. Problem statement

A Kanban worker run is a sequence of actions (tool calls, LLM turns, loop iterations) inside one task attempt. When the run fails — `blocked`, `crashed`, `timed_out`, `gave_up` — the current system answers **what** happened (run outcome, error string) but not **why** or **where** the failure originated.

The AgentTether paper (arXiv:2607.06273) shows that building a dependency graph of agent actions at runtime, then tracing error propagation through that graph, lets a diagnosis engine distinguish the *earliest actionable failure* from *downstream propagated errors* — and improves task success rate 18-34% across three benchmarks with targeted interventions (retry from safe checkpoint, alternative tool, trajectory repair).

The `loop-engineering` repository contributes the structured-loop framing: loops are first-class entities with an identity (`loop_id`), a bounded budget (max iterations / max turns), and a run log (`loop-run-log.md`). Its L1/L2 automation levels map to our MVP/non-goals split: the MVP is L1 — **record and diagnose**; intervention execution is L2 and explicitly out of scope.

Current gaps this MVP closes:

| Gap | Today | After MVP |
|---|---|---|
| Failure origin | Run outcome + one error string | Root-cause action id + propagation path |
| Repeated failure | `consecutive_failures` counter (opaque) | `loop_repeated` category with duplicate-hash evidence |
| Retry quality | Blind respawn on tick | `retry_exhausted` / `retry_from_checkpoint` recommendation |
| Downstream noise | Error string is the *latest* failure | Engine walks the graph to the *earliest* actionable failure |

---

## 2. Scope

### In scope (MVP)

1. **Action event schema** — a machine-readable trace event format for worker actions (see §4).
2. **Dependency-edge semantics** — directed edges between actions: `data`, `causal`, `loop`, `retry` (see §5).
3. **Loop/iteration identifiers** — stable `loop_id` + `iteration` so repeated or nested loops never conflate iterations (see §6).
4. **Failure categories** — canonical taxonomy the engine returns (see §7).
5. **Root-cause result format** — the diagnosis payload (see §8).
6. **Intervention recommendations** — ranked, never executed (see §9).
7. **Storage & retention** — per-board trace dir, JSONL, bounded by rotation (see §10).
8. **Sensitive-input redaction** — what is never stored, and the deterministic hash scheme (see §11).
9. **Extension points** — where later stages plug in without changing the contract (see §12).
10. **Graceful behavior on missing/malformed traces** — the engine degrades to `unknown` / `malformed_trace`, never crashes (see §13).
11. **Sample traces** — one success, one failure (see §15).

### Explicit non-goals

- **No intervention execution.** The engine recommends; a human or a later L2 component executes. `retry_from_checkpoint`, `alternative_tool`, `trajectory_repair` are **recommendations only** in the MVP.
- **No general observability rewrite.** The existing `hermes.observer.v1` hooks stay the telemetry surface; loop-diagnostics is a *consumer* of that contract where it exists, not a replacement.
- **No real-time streaming / UI.** Traces are written to disk for post-run diagnosis. Live dashboards are out of scope.
- **No cross-board trace federation.** Traces live per-board, mirroring worker logs.
- **No change to dispatcher retry semantics.** `dispatch_once`, `check_respawn_guard`, `_record_task_failure` are untouched. The diagnosis output is advisory.
- **No LLM-based diagnosis.** The engine is deterministic and bounded (see §8). An LLM summarizer is a possible later extension, not MVP.

---

## 3. Interfaces

Two components implement against this contract independently:

### 3.1 Graph recorder

```
Input:   observer hook events (pre_tool_call / post_tool_call / agent:step / subagent_*) when present,
         OR a worker-side instrumented wrapper around tool dispatch when hooks are absent
Output:  trace files under <board-root>/kanban/boards/<slug>/loop-traces/<task_id>/<run_id>.jsonl
Contract: every event conforms to hermes.loop_diagnostics.v1 (ActionStart / ActionEnd / DependencyEdge / RunHeader / RunFooter)
Behavior when disabled: zero import cost, zero write cost (see §13.4)
```

### 3.2 Diagnosis engine

```
Input:   task_id + run_id (+ optional failed_action_id)
Reads:   the run's trace file from the recorder's storage layout
Output:  DiagnosisResult (see §8) conforming to hermes.loop_diagnostics.v1 (diagnosis result branch)
Behavior on malformed input: returns status=malformed_trace or unknown, never raises (see §13)
Determinism: same trace file -> identical result (see §8.4)
```

**Boundary:** the recorder never reads a diagnosis; the engine never writes a trace. They share only the schema and the storage layout. This is what lets `t_8e6eb31d` (recorder) and `t_a2e356b5` (engine) proceed independently.

---

## 4. Action event schema

Canonical location: `hermes_cli/observability/schemas/hermes.loop_diagnostics.v1.schema.json`

The schema uses the repo's existing JSON Schema convention (see `hermes.shared_metrics.v2.schema.json`). It is a oneOf union of six record kinds:

| Kind | Emitted by | Purpose |
|---|---|---|
| `run_header` | recorder at run start | Task/run/attempt identity, goal_mode |
| `action_start` | recorder before action | Action node creation |
| `action_end` | recorder after action | Action status, duration, error, result hash |
| `edge` | recorder after action_end | Directed dependency edge |
| `run_footer` | recorder at run end | Outcome, error, event count |
| `diagnosis_result` | engine (separate file or appended) | Engine output |

### Core action fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `task_id` | string | yes | Kanban task id (`t_...`). |
| `run_id` | integer | yes | `task_runs.id` — the attempt. |
| `action_id` | string | yes | `"<run_id>:<seq>"` — globally stable per action, never reused across runs. |
| `parent_action_id` | string\|null | no | Action that spawned this one (subagent / nested loop). Null = top level. |
| `loop_id` | string\|null | no | Loop instance this action belongs to (see §6). |
| `iteration` | int\|null | no | 0-based ordinal within `loop_id` (see §6). |
| `ts` | int | yes | Unix epoch seconds (UTC). |
| `action_kind` | enum | yes | `tool_call` \| `llm_call` \| `loop_start` \| `loop_iteration` \| `checkpoint` \| `subagent` \| `other`. MVP emits `tool_call` and `loop_iteration`; the rest are reserved. |
| `tool_name` | string\|null | no | e.g. `terminal`, `web_search`, `kanban_comment`. |
| `summary` | string\|null | no | **Redacted** one-line summary (see §11). |
| `model` | string\|null | no | Model id for `llm_call`. |
| `turn_id` / `api_request_id` / `tool_call_id` | string\|null | no | Observer v1 correlation IDs when available. |

### Action end fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `status` | enum | yes | `ok` \| `error` \| `blocked` \| `cancelled` \| `unknown` (mirrors observer v1 status vocabulary; `unknown` = incomplete trace). |
| `duration_ms` | int | yes | End-to-end duration. |
| `summary` | string\|null | no | **Redacted** result summary. |
| `error_type` | string\|null | no | Exception class / error code, e.g. `ToolExecutionError`, `ExitCodeError`, `TimeoutError`. |
| `error_message` | string\|null | no | **Redacted** message (see §11). |
| `result_hash` | string\|null | no | SHA-256 of the **redacted** result text — detects repeated identical failures without storing bodies (see §11.3). |

---

## 5. Dependency-edge semantics

An `edge` record is `(from_action_id -> to_action_id)` with one of four kinds:

| Edge kind | Meaning | Direction | Example |
|---|---|---|---|
| `data` | `from`'s output is consumed by `to` | producer → consumer | `terminal` (clone repo) → `read_file` (in that repo) |
| `causal` | `to` ran *because* `from` completed | cause → effect | `web_search` → `web_extract` on a result URL |
| `loop` | `to` is the next iteration of the same `loop_id` as `from` | iteration N → N+1 | `search` iter 0 → `search` iter 1 |
| `retry` | `to` is a retry of `from` after a failure | failed action → retry action | `terminal` (exit 1) → `terminal` (retry) |

**Semantics rules:**

1. Every non-first action has at least one incoming edge; first actions have none (they are graph roots).
2. `data` edges are the primary propagation vector: an `error` status on `to` whose `data`-predecessor `from` also errored is *propagated*, not *root*.
3. `loop` and `retry` edges are the repeated-failure vector: N consecutive `error` statuses joined by `loop`/`retry` edges with identical `result_hash` are evidence of a repeated loop failure.
4. Edges are recorded **after** the consumer's `action_end`, so a cancelled action never emits a spurious edge.
5. Cycles are possible only via `loop` edges (a loop iteration links back into the loop). The engine must terminate on cycles — see §13.2.

**Recorder rule (simplification):** the MVP derives edges from a small, explicit dependency model rather than attempting full data-flow analysis:

- The tool registry knows which tools produce artifacts (files, notes, results) and which consume them, via a static allowlist map `{tool_name -> produces: bool, consumes: bool}`.
- Sequential actions within the same turn that share a producer/consumer pair get a `data` edge.
- `agent:step` iterations get `loop` edges with incrementing `iteration`.
- A retried action (same tool, consecutive, prior error) gets a `retry` edge.
- All other sequencing gets a `causal` edge.

This keeps the recorder deterministic and cheap. Precise data-flow inference is a **later extension** (§12).

---

## 6. Loop and iteration identifiers

**Loop instance** = one execution context that repeats. Identity is `loop_id`, a string:

- Retry loops: `"retry"` (or `"retry:<tool_name>"` when scoped).
- Goal-mode loops: `"goal"` (the `run_kanban_goal_loop` turn loop in `hermes_cli/goals.py`).
- Search/decomposition loops: `"<parent_action_id>:loop"` so nested loops never collide.
- A subagent's internal loop: `"<parent_action_id>:loop"` where `parent_action_id` is the `subagent` action.

**Iteration** = 0-based ordinal within a `loop_id` instance, monotonic, never reset mid-run.

**Conflation rule:** two actions are *the same loop iteration* iff `loop_id` AND `iteration` match; otherwise they are distinct nodes. This is what lets the graph represent:

- nested loops (outer `goal`, inner `search`),
- repeated identical calls in different iterations (same `tool_name`, same `result_hash`, different `iteration`),
- retries (same `loop_id`, incrementing `iteration`, `retry` edges).

**Stability:** `loop_id` is derived from a stable identity (the loop's origin action or the goal-mode env) — not from a random per-iteration token — so the engine can group iterations across the run.

---

## 7. Failure categories

The engine classifies every diagnosis into exactly one category:

| Category | Trigger | Example |
|---|---|---|
| `action_error` | A single action failed with no data-predecessor failure | `web_extract` 404 on a bad URL the agent chose |
| `input_invalid` | A `data` edge's producer failed or produced nothing, and the consumer failed on that input | `terminal` (clone failed) → `read_file` (path missing) |
| `loop_repeated` | ≥2 consecutive iterations in the same `loop_id` failed with identical `result_hash` | `web_search` returns same empty result 3× |
| `retry_exhausted` | A retry loop (`loop_id="retry"`) hit its budget with all attempts failed | `terminal` retried 5×, exit 1 each time |
| `timeout_propagation` | An action `timed_out` and downstream actions failed on the interrupted state | worker hit `TERMINAL_TIMEOUT`, next `read_file` fails |
| `cancellation_propagation` | An action was `cancelled` and downstream actions failed on the partial state | subagent cancelled mid-write → parent `read_file` fails |
| `malformed_trace` | The trace file cannot be parsed far enough to diagnose (see §13) | JSONL line corruption, missing run_header |
| `unknown` | No evidence for any category | Trace missing entirely |

**Priority order** (engine picks the highest applicable): `malformed_trace` → `retry_exhausted` → `loop_repeated` → `input_invalid` → `timeout_propagation` / `cancellation_propagation` → `action_error` → `unknown`.

---

## 8. Root-cause result format

The engine returns a `DiagnosisResult` (schema §`DiagnosisResult`):

| Field | Type | Meaning |
|---|---|---|
| `status` | enum | `root_cause_found` \| `candidates` \| `unknown` \| `malformed_trace` |
| `root_cause_action_ids` | string[] | Earliest actionable failure(s). Empty unless `root_cause_found`/`candidates`. |
| `confidence` | float [0,1] | ≥0.8 single deterministic path; 0.1-0.7 ranked candidates; 0.0 unknown/malformed. |
| `category` | enum | One of §7. |
| `propagation_path` | string[] | Ordered chain failed_action → … → root_cause_action. |
| `explanation` | string | Concise deterministic explanation (≤2048 chars). |
| `interventions` | Intervention[] | Ranked recommendations (see §9). |
| `evidence` | object | `failed_action_id`, `trace_event_count`, `missing_action_ends`, `malformed_lines`, `duplicate_hashes`. |

### 8.1 Engine algorithm (bounded, deterministic)

1. **Load + validate.** Parse the run's trace file; validate each line against the schema. Invalid lines are skipped and recorded in `evidence.malformed_lines`. A missing/empty file → `status=unknown`, category=`unknown`.
2. **Locate failure.** If `failed_action_id` is given, start there. Else find the last `action_end` with `status != ok` (prefer `error`, then `blocked`, then `timed_out`, then `cancelled`).
3. **Walk predecessors.** From the failed action, walk incoming `data`/`causal` edges. Stop at the *earliest actionable* node:
   - A node whose incoming `data` edge's producer also failed → continue walking (this is propagation).
   - A node with no failing predecessor → this is the root.
   - A node with ≥2 consecutive `loop`/`retry` failures with identical `result_hash` → `loop_repeated` root.
4. **Build propagation path** from the failed action back to the root.
5. **Classify** per §7 priority order.
6. **Recommend** per §9.
7. **Return** the `DiagnosisResult`.

### 8.2 Ambiguity handling

- If two independent failing branches exist with no causal link, the engine returns `status=candidates` with the top ranked by (distance from failed action, evidence strength) — it does **not** fabricate a single root cause.
- If the failure is a timeout/cancellation, the engine reports propagation even when the interrupted action's own status was `ok` (the interruption is the cause; downstream errors are effects).
- The engine never claims certainty it cannot support: ambiguous evidence ⇒ `candidates` or `unknown`.

### 8.3 Safety on cyclic input

- Graph traversal keeps a visited set and a depth cap (default 1000). Cycles via `loop` edges terminate by the visited set.
- Corrupt edges (self-loop, dangling action_id) are ignored with a `malformed_lines` note.
- Total per-call work is bounded; worst case is O(V + E) with V,E capped by the run's event count.

### 8.4 Determinism

- Iteration order is stable (insertion order of trace lines).
- Ties are broken by `(action_id, ts)` lexicographically.
- Equivalent traces (same redacted events, same order) produce byte-identical results. The redaction scheme in §11.3 ensures equivalent *sensitive* content hashes identically, so determinism survives redaction.

---

## 9. Intervention recommendations

The engine returns ranked `Intervention` objects. **MVP: recommendations only — never executed.**

| Intervention kind | When recommended | Payload |
|---|---|---|
| `retry_from_checkpoint` | Root is a transient failure after a verified checkpoint; a safe retry point exists | `{"checkpoint_action_id": "..."}` |
| `alternative_tool` | Root is a tool-specific failure (`action_error` on a tool whose goal another tool can meet) | `{"tool_name": "...", "goal": "..."}` |
| `trajectory_repair` | Root is a bad state that must be patched before retry (not just re-run) | `{"patch_hint": "..."}` |
| `escalate` | Category is `unknown` / `malformed_trace`, or failures are deterministic and non-transient | `{}` |
| `none` | Diagnosis is `root_cause_found` but the failure is expected/benign | `{}` |

Mapping guidance (deterministic):
- `retry_exhausted` → `escalate` (blind retry already failed; do not recommend more retries).
- `loop_repeated` → `escalate` or `alternative_tool`.
- `action_error` on a tool with an alternative in the tool registry → `alternative_tool`.
- `input_invalid` / `timeout_propagation` / `cancellation_propagation` → `retry_from_checkpoint` when a checkpoint action precedes the root, else `trajectory_repair`.
- `unknown` / `malformed_trace` → `escalate`.

---

## 10. Storage and retention

### Layout

```
<board-root>/kanban/boards/<slug>/loop-traces/<task_id>/<run_id>.jsonl
```

`default` board keeps the legacy root: `<kanban-home>/kanban/loop-traces/<task_id>/<run_id>.jsonl` — mirroring `worker_logs_dir` in `hermes_cli/kanban_db.py` so multi-board isolation is preserved.

The engine writes its `DiagnosisResult` to `<run_id>.diagnosis.json` in the same directory.

### Format

- Newline-delimited JSON (JSONL), one record per line, UTF-8.
- Records are append-only within a run; `run_footer` is the last record.
- A crashed/killed worker leaves a file with `action_start` records but no `run_footer` — the engine treats missing footer as truncation evidence (§13.3).

### Retention

Reuse the worker-log rotation pattern (`worker_log_rotation_config`, default 2 MiB / 1 backup) but with a **per-run cap** instead of per-file rotation:

- **Per-run cap:** `kanban.loop_diagnostics.max_events_per_run` (default 10,000 events). When exceeded, the recorder stops writing further `action_*`/`edge` records, writes a `run_footer` with `truncated=true`, and continues without error. The diagnosis engine sees the cap in the footer and reports `malformed_trace`/`unknown` with `evidence.malformed_lines` noting truncation.
- **On-disk retention:** `kanban.loop_diagnostics.retain_runs` (default 20 runs per task). Oldest run directories beyond the cap are pruned at run completion (best-effort, same spirit as `_prune_corrupt_backups`). Config keys mirror `kanban.*` naming in `config_defaults.py`.
- **Diagnosis retention:** diagnosis files follow the run directory; pruning a run prunes its diagnosis.

---

## 11. Sensitive-input redaction

### 11.1 Never stored

The recorder NEVER writes:

- Raw tool arguments (`args`).
- Raw tool results / file contents.
- User prompts / model outputs.
- Full terminal command strings (a one-line `summary` like `"ran git command"` is allowed; the command itself is not).
- Authorization headers, tokens, API keys, cookies, credentials.
- File paths that contain user-private identifiers (paths are summarized as `"<path>"` or basename only).
- Any URL query strings or fragments (mirrors the observer v1 subagent `tool_call_history` rule).

### 11.2 Redaction rules (applied at write time)

1. `summary` fields: truncate to 512 chars, strip secrets via the existing redaction filters (see `hermes_cli/config_defaults.py` `redact_secrets`, and the `tool_error` sanitizer path in `tools/registry.py`).
2. `error_message`: truncate to 1024 chars, apply the same secret strip. `error_type` is kept (class name only).
3. `tool_name`, `action_kind`, `status`, `ts`, `duration_ms`, `iteration`, `loop_id` are inherently safe.
4. Structured redaction by construction: the writer builds a brand-new dict with ONLY the schema keys — no passthrough of arbitrary args (same pattern as `hermes_cli/kanban_telemetry.py`).
5. The recorder is fail-open: if redaction is uncertain, it writes `null`/`"<redacted>"` rather than raw content.

### 11.3 Deterministic result hashing

- `result_hash = SHA-256(redacted_result_text)`.
- Hash input is the **redacted** text (after rule 11.2), so equivalent sensitive content hashes identically across runs — the engine can detect repeated identical failures without storing the bodies.
- Raw results are never persisted; only the hash is.

---

## 12. Extension points

These are seams where later stages plug in without changing the contract:

| Extension | Seam | Later stage |
|---|---|---|
| Precise data-flow edges | `edge` records already carry `edge_kind=data`; the recorder's static allowlist map is the only part to replace | A data-flow analyzer producing true producer/consumer edges |
| Real-time UI | Trace files are stable, schema-versioned JSONL; a watcher can tail them | Dashboard loop-trace view |
| Intervention execution | `Intervention.payload` is machine-readable; a new component consumes it | L2 `retry_from_checkpoint` executor |
| LLM summarizer | `DiagnosisResult` is complete without LLM; an LLM layer can add human prose on top | Operator-facing narrative |
| Cross-run correlation | `run_header`/`run_footer` carry `attempt`; engine can join runs on `task_id` | Retry-strategy tuning |

---

## 13. Graceful behavior on missing/malformed traces

### 13.1 Missing trace file

- Engine: `status=unknown`, `category=unknown`, `explanation="no trace recorded for run <run_id>"`, `interventions=[escalate]`.
- This is the expected path for pre-MVP runs and disabled-diagnostics runs — never an error.

### 13.2 Malformed lines / corrupt JSON

- Recorder: a line that fails schema validation is skipped and counted; the run continues. The recorder never crashes on its own write path (fail-open).
- Engine: invalid lines go to `evidence.malformed_lines`; diagnosis proceeds on the valid subset. If ≥50% of lines are invalid, return `status=malformed_trace`.

### 13.3 Truncated run (crash / kill / timeout)

- No `run_footer` ⇒ engine marks the run truncated.
- `action_start` records with no matching `action_end` ⇒ `evidence.missing_action_ends`; those actions are treated as `status=unknown`.
- If the truncated action is the failure site or a predecessor, category degrades toward `unknown`/`timeout_propagation`/`cancellation_propagation` as the evidence dictates.

### 13.4 Recorder disabled

- `kanban.loop_diagnostics.enabled: false` (default) ⇒ the recorder registers **zero hooks** and performs **zero writes**. No import-time side effects, no per-tool-call overhead. Behavior is byte-identical to today.
- The `has_hook(...)` gate in the observer contract (`docs/observability/README.md` §Performance) is the precedent: only build sanitized payloads when a consumer exists.

---

## 14. Failure-path integration (t_097b0a62)

The glue between the recorder + engine and the Kanban worker failure
lifecycle lives in `hermes_cli/observability/loop_diagnostics_integration.py`
and is invoked from the dispatcher-side failure paths in `kanban_db.py`
(`detect_crashed_workers` / `enforce_max_runtime` / `_record_task_failure`
(spawn-failure + gave_up) / `block_task` / `_block_task_locked`).

> **Operators:** the end-to-end usage guide — configuration, report fields,
> interpretation, limitations, retention/redaction, and a worked recovery
> example — lives in `docs/loop-diagnostics-ops-guide.md`. This section is
> the integration contract; the ops guide is how to use it.

### 14.1 Behavior

- On a **terminal attempt failure** (worker `crashed` / `timed_out` /
  `spawn_failed` / `gave_up` / `blocked` with an open run) the integration
  finalizes nothing process-local, runs the deterministic diagnosis engine
  on the run's trace, persists the `DiagnosisResult` to
  `<run_id>.diagnosis.json` next to the trace, appends a `diagnosis` task
  event (machine + human readable), and records one redacted metrics row
  (`~/.hermes/governance/telemetry/loop-diagnostics.jsonl`).
- **Success paths are unaffected.** No `diagnosis` event is emitted on
  `complete_task`; the only new side effect on failure is the event + file +
  metrics.
- **Never masks the worker error.** Every entry point is wrapped; a
  diagnosis failure (missing trace, engine hiccup, write error) degrades to
  a `diagnosis_failed` / `diagnosis_skipped` event with the original error
  preserved. The caller's error handling and retry/circuit-breaker accounting
  are untouched.

### 14.2 Configuration

- `kanban.loop_diagnostics.enabled` (default False) — master switch; the
  recorder plugin registers zero hooks when disabled.
- `kanban.loop_diagnostics.diagnose_on_failure` (default True) — independent
  gate for the failure-time diagnosis step. Set False to keep recording
  traces while disabling the diagnostic attach on failure.

### 14.3 Event payload (kind=`diagnosis`)

Machine-readable fields: `run_id`, `outcome`, `status`, `category`,
`confidence`, `root_cause_action_ids`, `propagation_path`, `interventions`
(kind / action_id / rationale / payload), `evidence`, `explanation`.
Human-readable one-liner: `summary` (e.g.
`diagnosis=root_cause_found category=input_invalid root=41:1 path=41:2,41:1 intervention=retry_from_checkpoint · detail=...`).
The original failure text is preserved as `original_error`; `trace_path`
points at the persisted `.diagnosis.json` for operators.

---

## 15. Sample traces

### 15.1 Success trace

```jsonl
{"schema_version":"hermes.loop_diagnostics.v1","kind":"run_header","task_id":"t_example","run_id":41,"attempt":1,"profile":"default","goal_mode":false,"ts":1785739000}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"action_start","task_id":"t_example","run_id":41,"action_id":"41:1","parent_action_id":null,"loop_id":null,"iteration":null,"ts":1785739001,"action_kind":"tool_call","tool_name":"terminal","summary":"clone repo"}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"action_end","task_id":"t_example","run_id":41,"action_id":"41:1","ts":1785739004,"status":"ok","duration_ms":3000,"summary":"repo cloned","result_hash":"a1b2c3d4e5f6..."}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"edge","task_id":"t_example","run_id":41,"from_action_id":"41:1","to_action_id":"41:2","edge_kind":"data","ts":1785739004}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"action_start","task_id":"t_example","run_id":41,"action_id":"41:2","parent_action_id":null,"loop_id":null,"iteration":null,"ts":1785739005,"action_kind":"tool_call","tool_name":"read_file","summary":"read design doc"}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"action_end","task_id":"t_example","run_id":41,"action_id":"41:2","ts":1785739007,"status":"ok","duration_ms":2000,"summary":"doc read","result_hash":"f6e5d4c3b2a1..."}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"run_footer","task_id":"t_example","run_id":41,"ts":1785739008,"outcome":"completed","event_count":7}
```

### 15.2 Failure trace (root cause = input_invalid)

```jsonl
{"schema_version":"hermes.loop_diagnostics.v1","kind":"run_header","task_id":"t_example","run_id":42,"attempt":2,"profile":"default","goal_mode":false,"ts":1785739100}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"action_start","task_id":"t_example","run_id":42,"action_id":"42:1","parent_action_id":null,"loop_id":null,"iteration":null,"ts":1785739101,"action_kind":"tool_call","tool_name":"terminal","summary":"clone repo"}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"action_end","task_id":"t_example","run_id":42,"action_id":"42:1","ts":1785739104,"status":"error","duration_ms":3000,"summary":"clone failed","error_type":"ExitCodeError","error_message":"remote: repository not found","result_hash":"deadbeef..."}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"edge","task_id":"t_example","run_id":42,"from_action_id":"42:1","to_action_id":"42:2","edge_kind":"data","ts":1785739104}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"action_start","task_id":"t_example","run_id":42,"action_id":"42:2","parent_action_id":null,"loop_id":null,"iteration":null,"ts":1785739105,"action_kind":"tool_call","tool_name":"read_file","summary":"read design doc"}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"action_end","task_id":"t_example","run_id":42,"action_id":"42:2","ts":1785739107,"status":"error","duration_ms":2000,"summary":"file not found","error_type":"FileNotFoundError","error_message":"path does not exist","result_hash":"c0ffee..."}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"run_footer","task_id":"t_example","run_id":42,"ts":1785739108,"outcome":"blocked","error":"read_file failed: path does not exist","event_count":7}
```

Engine result for run 42 (verbatim contract shape):

```json
{
  "schema_version": "hermes.loop_diagnostics.v1",
  "kind": "diagnosis_result",
  "diagnosis_version": "hermes.loop_diagnostics.diagnosis.v1",
  "task_id": "t_example",
  "run_id": 42,
  "status": "root_cause_found",
  "root_cause_action_ids": ["42:1"],
  "confidence": 0.9,
  "category": "input_invalid",
  "propagation_path": ["42:2", "42:1"],
  "explanation": "Action 42:2 (read_file) failed because its data predecessor 42:1 (terminal) failed with 'repository not found'. The root cause is the clone failure; the read failure is a downstream propagation.",
  "interventions": [
    {"kind": "retry_from_checkpoint", "action_id": "42:1", "rationale": "Clone failure is transient-adjacent (bad URL or auth); retry after correcting the remote URL.", "payload": {"checkpoint_action_id": "42:0"}}
  ],
  "evidence": {
    "failed_action_id": "42:2",
    "trace_event_count": 7,
    "missing_action_ends": [],
    "malformed_lines": [],
    "duplicate_hashes": []
  }
}
```

### 15.3 Failure trace (repeated loop failure)

```jsonl
{"schema_version":"hermes.loop_diagnostics.v1","kind":"run_header","task_id":"t_example","run_id":43,"attempt":3,"profile":"default","goal_mode":false,"ts":1785739200}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"action_start","task_id":"t_example","run_id":43,"action_id":"43:1","parent_action_id":null,"loop_id":"search","iteration":0,"ts":1785739201,"action_kind":"tool_call","tool_name":"web_search","summary":"search for source"}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"action_end","task_id":"t_example","run_id":43,"action_id":"43:1","ts":1785739203,"status":"error","duration_ms":2000,"summary":"search failed","error_type":"EmptyResults","error_message":"no results","result_hash":"abc123..."}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"edge","task_id":"t_example","run_id":43,"from_action_id":"43:1","to_action_id":"43:2","edge_kind":"loop","ts":1785739203}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"action_start","task_id":"t_example","run_id":43,"action_id":"43:2","parent_action_id":null,"loop_id":"search","iteration":1,"ts":1785739204,"action_kind":"tool_call","tool_name":"web_search","summary":"search for source"}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"action_end","task_id":"t_example","run_id":43,"action_id":"43:2","ts":1785739206,"status":"error","duration_ms":2000,"summary":"search failed","error_type":"EmptyResults","error_message":"no results","result_hash":"abc123..."}
{"schema_version":"hermes.loop_diagnostics.v1","kind":"run_footer","task_id":"t_example","run_id":43,"ts":1785739207,"outcome":"gave_up","error":"web_search returned no results","event_count":7}
```

Engine result: `category=loop_repeated`, `status=root_cause_found`, `root_cause_action_ids=["43:1"]`, `evidence.duplicate_hashes=["abc123..."]`, `interventions=[alternative_tool]`.

---

## 16. Acceptance criteria

A. The schema file exists at `hermes_cli/observability/schemas/hermes.loop_diagnostics.v1.schema.json` and validates all sample records in §15 with `jsonschema`.
B. The recorder (t_8e6eb31d) can emit all six record kinds and disable with zero behavioral delta.
C. The engine (t_a2e356b5) returns deterministic results for equivalent traces and safe termination on cyclic/corrupt input.
D. Sensitive inputs are never stored; redaction is by-construction.
E. Missing/malformed/truncated traces degrade to `unknown`/`malformed_trace` with evidence, never crash.
F. The design is bounded to failure diagnosis: no intervention execution, no general observability rewrite, no dispatcher semantics change.
G. The failure-path integration (t_097b0a62) attaches a machine + human readable `diagnosis` report to the task event stream on every terminal attempt failure (crashed / timed_out / spawn_failed / gave_up / blocked) while successful runs emit no `diagnosis` event; diagnosis failures never mask the original worker error.

---

## 17. Non-goals recap (explicit)

1. No intervention execution in MVP.
2. No real-time UI / streaming.
3. No cross-board federation.
4. No change to dispatcher retry/crash semantics.
5. No LLM-based diagnosis.
6. No general observability rewrite — the observer v1 contract remains the telemetry surface.
