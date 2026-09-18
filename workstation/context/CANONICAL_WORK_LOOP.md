# Canonical Work Loop — implementation evidence

Base audited on 2026-09-17: `main` and fetched `origin/main`
`edaa8cc8fd1180de05f92450caede853ec862daf`. Changes remain in the working tree.
This document does not declare the entire product program complete.

## Implemented contracts and integrations

- `contracts.py`: MessageOrigin/IntentAuthority/MessageEnvelope; TaskOutcome,
  OutcomeStatus, AcceptanceContract and AcceptanceEvaluator. Evidence acceptance
  requires a passing verifier linked to supplied evidence; required deliverables,
  pending work and uncertain mutation are checked. Advisory policy requires no
  artificial file. Policy is captured at task creation, never relaxed by a report.
- `kanban.py`: automatic promotion requires a matching authorized envelope.
  Legacy callers without provenance fail closed, including `force=True`.
  Incomplete/rejected reports block through canonical `needs_input`; no completion
  event is emitted. Summaries describe the result. Accepted verifier results go to
  the existing verification ledger. Repeatable verified reports can seed an
  unpromoted, sanitized procedure candidate in existing ProceduralMemory.
- `kanban_db.py`: completion of Workstation and Hybrid tasks requires acceptance
  metadata and re-evaluates the canonical contract against recorded verifier event IDs. Late reports cannot overwrite
  a closed timeout/failure when carrying run identity or Workstation metadata.
  Legacy manual completion for ordinary upstream tasks remains compatible.
- `events.py`: an uncorrelated critical event is an observation; no task creation.
- `runtime.py`: EvidenceClass distinguishes LIVE_HANDLE and DURABLE_PROOF.
  Live evidence requires an expiry; known legacy handle kinds get a 120-second
  default TTL when newly published. Unknown/expiry-free stored evidence does not
  prove running. Durable proof cannot promote running. WaitContract consumes
  correlated RuntimeEventBus events, with subscription before dispatch supported.
  Unaccepted `task.completed` messages mirror as progress.
- `durable_tasks.py`: interrupted/failed/cancelled/blocked parents block running
  children atomically; recovery reconciles stored running items against current
  live item IDs. A missing handle blocks rather than authorizing mutation retry.
- `tool_guardrails.py`, `batch_runner.py`, `task_compiler.py`: shared ExperimentKey
  detects three consecutive structurally identical failures across distinct items,
  stops fan-out and persists the circuit. Unexecuted items stay pending. Reopening
  the same circuit requires diagnosis; a corrected operation needs a fresh canary.
  Original canary and dispatch/checkpoint uncertain-mutation protections remain.
- `effects.py`: conservative capability introspection defaults, without changing
  provider schemas or registering a model tool. Unknown result schema remains
  unknown; positive support flags must be supplied by a capability owner.
- `artifacts.py`, `file_tools.py`: structured `artifact://` resolution for JSON
  (including `.data`), UTF-8 text and binary metadata; root/hash/size verification.
  Binary/oversized data never enters context automatically. Existing read_file
  resolves canonical profile artifact refs without exporter scripts.
- `browser_readiness.py`: extends original probes/wait API with path/entity,
  semantic targets, auth/hydration predicates and bounded diagnostics. Recipe
  preflight can consume this contract. No vendor-specific browser rules.
- `journal.py`: schema v2, sequence/hash chain, writer ID, explicit environment,
  build SHA/version, interprocess append lock, critical flush/fsync, located
  JOURNAL_DEGRADED errors. Legacy lines stay readable/unverified and are not
  rewritten. Hashes provide local integrity, not external authenticity.
- `verification_evidence.py`: additive outcome verifier events in the same DB,
  scoped by task/session, verifier/target/reference/time/environment.
- `cron/jobs.py`, scheduler/config: drift pauses scheduling with NEEDS_MIGRATION,
  preserving previous/current inference identities and reason/time. Explicit
  adoption, pinning or pause resolution uses the existing job owner. An ordinary
  explicit inference edit also resolves migration. Explicit follow-global opt-in
  bypasses the snapshot drift check; legacy defaults remain guarded.
- `hermes_state_search.py`: all compacted search paths suppress redundant identical
  session/role/content snapshots through a read projection; active messages and
  stored history survive. Duplicate rows/bytes can be measured locally.
- `memory.py`, `routines.py`: additive freshness metadata, failure demotion,
  candidate baseline metrics, fail-closed required pre/postconditions. Routine
  stop is an action, not canonical task completion. Existing validate/promote
  lifecycle remains the owner.
- `cockpit.py`, web_server: canonical read projection/API at
  `/api/workstation/tasks/{task_id}/cockpit` with token authentication. Resolves
  workplan canonical refs, browser refs, journal worker/process refs, human card,
  outcome, evidence and deliverables. Corrupt journal does not silently disappear.
- `trello_migration.py`: explicit provenance manifest, preview by default,
  idempotent non-destructive human board/list/card projection; no auto-delegation.
  Run `python -m workstation.trello_migration manifest.json [--board slug] [--apply]`.
- Workstation pytest collection/per-test isolation redirects both homes and the
  canonical Kanban DB. Existing upstream test isolation remains authoritative.
- EvaluationHarness exposes local AVCR and quality/cost/recovery/reuse projections,
  excluding non-production environments and retaining unknown usage as null.

## Additive migrations

`task_acceptance_contracts` lives in canonical Kanban SCHEMA_SQL. Existing tasks
without a contract use the safe evidence policy on Workstation completion.
`outcome_verification_events` lives in existing verification_evidence.db.
Artifact media/encoding and procedure freshness fields have safe legacy defaults.
Journal v2 writes extend legacy history without rewriting it. No existing tasks,
SessionDB messages, Trello imports or journals are deleted.

## Coverage and remaining implementation

`work100.py` catalogs all 30 required scenarios and launches real pytest and
Electron-platform regressions. All catalog cases now have replayable coverage;
some share an existing owner regression. The catalog gate passed with 35 Python
tests and 36 Electron-platform tests. This is not packaged-release E2E evidence.

Remaining program work: provenance at secondary connector/delegation ingress;
WorkIntent propagation to all browser/worker/risk consumers; operational
live-handle publisher parity and execution-environment identity; actual internal
publishers wired to the scoped compiler bus (external polling still exists); universal mutation
reconciliation; resume of a diagnosed existing circuit with a new verification
generation; capability-specific result schemas; live semantic browser observation
adapter beyond supplied preflight observations; full worker/process lineage;
automatic compatible candidate validation and replay savings; complete cockpit
live-handle/usage/next-action presentation; and native packaged release evidence.

Desktop typecheck, focused UI Vitest and Electron-platform contracts ran on
Windows. The platform tests exercise lifecycle owners with Electron bindings and
stubs; native renderer/packaged E2E remains a separate release gate. The existing
authorized red-team harness and profile isolation are preserved.

## Changed files

Validation completed on this working tree:

- Workstation full suite, after the continued ingress/outcome/handoff/Trello
  changes: **452 passed, 2 skips** (259.00s).
- Upstream owners (guardrails, verification evidence/FD leak, cron drift,
  session search/slow logging, Hybrid, review completion): **124 passed**.
- Post-suite integrated policy/provenance/compiler contracts: **98 passed**.
- Final canonical/continuity/Hybrid/report contracts after recorded-verifier
  admission and multiwriter journal proof: **49 passed**.
- Latest P0 and actual read_file/preflight integration proofs: **29 passed**.
- Continued Work100 seed: **35 Python + 36 Electron-platform passed**, exit **0**;
  all 30 catalog cases are linked to real owner regressions.
- Continued upstream/API/finalizer/plugin gate: **150 passed**; candidate
  acceptance integration: **27 passed**; focused UI Vitest: **29 passed**;
  Desktop typecheck: PASS.
- `git diff --check`: PASS. No red-team harness changes; no second task/session/
  browser/memory/approval/artifact/recipe store. Packaged/native renderer E2E was
  not run. The focused Electron platform gate, typecheck and UI Vitest ran.

Initial sandbox test attempt failed with WinError 5 before useful validation.
Native Python escalation with disposable homes passed. One intermediate cron
regression (`timezone` missing) was corrected. Four intermediate verification
fixture failures were caused by basetemp inside the checkout; using a fresh
external temp root made their real project-root contracts pass, without weakening
the tests or product classifier. Test counts above overlap and must not be summed.

Residual implementation risks: journal append currently verifies the full prior
chain, so large journals need measured incremental verification optimization;
legacy manual upstream completion remains outside the Workstation acceptance
gate; caller-supplied structured envelopes/verifier results require trusted
ingress/callbacks; read-file artifact access remains profile-scoped.
These are tracked boundaries, not claims of completed global hardening.

- `agent/tool_guardrails.py`
- `agent/verification_evidence.py`
- `cron/jobs.py`
- `cron/scheduler.py`
- `hermes_cli/config.py`
- `hermes_cli/hybrid_kanban.py`
- `hermes_cli/kanban_db.py`
- `hermes_cli/web_server.py`
- `hermes_state_search.py`
- `tests/hermes_cli/test_hybrid_kanban.py`
- `tools/effects.py`
- `tools/file_tools.py`
- `workstation/ROADMAP.md`
- `workstation/UPSTREAM_DELTA.md`
- `workstation/artifacts.py`
- `workstation/batch_runner.py`
- `workstation/benchmarks/durable_execution.py`
- `workstation/browser_readiness.py`
- `workstation/cockpit.py`
- `workstation/context/CANONICAL_WORK_LOOP.md`
- `workstation/context/CONSTRAINTS.md`
- `workstation/context/CURRENT_STATE.md`
- `workstation/context/DECISIONS.md`
- `workstation/context/HERMES_WORKSTATION_INTELLIGENCE.md`
- `workstation/context/KNOWN_ISSUES.md`
- `workstation/context/TESTING.md`
- `workstation/context/engineering-journal/CURRENT.md`
- `workstation/contracts.py`
- `workstation/durable_tasks.py`
- `workstation/evaluation.py`
- `workstation/events.py`
- `workstation/journal.py`
- `workstation/kanban.py`
- `workstation/memory.py`
- `workstation/routines.py`
- `workstation/runtime.py`
- `workstation/task_compiler.py`
- `workstation/tests/conftest.py`
- `workstation/tests/test_canonical_continuity.py`
- `workstation/tests/test_canonical_work_loop.py`
- `workstation/tests/test_events_pipeline.py`
- `workstation/tests/test_kanban_journal.py`
- `workstation/trello_migration.py`
- `workstation/work100.py`
- `workstation/work_intent.py`

## Adaptive execution correction — 2026-09-18

Real Workstation Browser dogfood added an important invariant to the canonical
loop: deterministic optimization must never become the only path by which a novel
safe workflow is allowed to make progress.

The canonical execution ladder is now:

~~~text
intent
  -> canonical work / run / leases
  -> choose execution policy
     -> adaptive reasoning for novelty/drift
     -> compiled execution for known repeated segments
     -> promoted deterministic routine for validated reuse
  -> evidence / acceptance / canonical commit
  -> learning
  -> future execution at the cheapest proven level
~~~

True homogeneous fan-out retains TaskCompiler, canary, checkpoints, idempotency
and uncertainty semantics. Adaptive work retains policy, approval, TaskRun
fencing, BrowserTask ownership, no-progress bounds and effect certainty.

A deterministic runner that encounters an unmodeled semantic state must emit a
compact NEEDS_REASONING/drift handoff with confirmed progress/evidence refs and
allow the agent to repair only the unresolved segment. Confirmed effects are not
replayed.

The operation — not the whole session — owns the compilation candidate. Browser
tool names are normalized to the canonical route before constraints are applied,
and tools.effects remains the sole effect taxonomy.

Canonical detailed specification:
[ADAPTIVE_EXECUTION_COMPILATION.md](ADAPTIVE_EXECUTION_COMPILATION.md).

