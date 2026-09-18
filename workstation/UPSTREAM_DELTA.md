# Hermes Workstation upstream delta


## Tracked upstream reliability intake — 2026-09-18 [PLANNED, NO PRODUCT DELTA YET]

Audit base: downstream `main@03e06cfd8c94e5a7627c288c8eddfd5d4c5c8033`.
Canonical plan:
`context/UPSTREAM_RELIABILITY_HARDENING_2026-09-18.md`.
Engineering evidence: H-065.

This entry records **planned selective adaptation**, not merged behavior. Upstream
PRs are field evidence and regression sources; they are not a cherry-pick queue.

Immediate candidates:

- #115068 / #115085: focused Desktop transcript recovery/resync patches are
  directly applicable to existing downstream files/tests;
- #111493: port the one-writer/read-only-owner invariant into downstream
  `active_sessions.py` and resume/admission paths;
- #114785 / #114793: adapt session provenance and heartbeat truth into the
  current canonical Kanban/TaskRun lineage;
- #114904: port only the topology-independent worker-exit evidence seam. Keep
  downstream protocol-violation streak, neutral rate-limit and breaker policy.
  This fork's one-shot exit path is in `cli.py`, not upstream's newer
  `quiet_single_query.py` topology;
- #114897: cap reconnect in the legacy/fallback `tools/browser_supervisor.py`
  after successful attach, with registry eviction.

Disposition-only references:

- #114964: BrowserTask keepalive semantics and native Electron evidence already
  exist downstream (H004/H013). Reuse/extend regression evidence; do not port
  upstream pane code into the Workstation runtime.
- #114986: watchlist only while downstream lacks its upstream `turnLeases`
  mechanism.
- #115056: P1 benchmark/quality input only; native Work already performs its
  principal snapshot inventory in one `webContents.executeJavaScript` call.

No source row in PATCH_MANIFEST is added until a production adaptation actually
lands. When it does, update this section from PLANNED to the exact downstream
files/commits and re-evaluate rebase conflict ownership.

## HW-022 — Verified procedure admission and planner context (2026-09-16)

Downstream base: `a977651539dd4b793e26f96c6116bce187677cc9`.
No new upstream comparison, wholesale file replacement or upstream merge.

Core adaptations: `agent/conversation_loop.py` projects authenticated durable state
on a copy immediately before provider sanitation/cache decoration. Automatic
`agent/conversation_compression.py` uses a deduplicated durable handoff when available;
manual force, commit fences and ordinary compression retain their existing paths.
`agent/context_compressor.py` is reused without modification for trusted references
and real-user turn identification. SessionDB history and cached system text remain.

`agent/tool_guardrails.py` scopes failures by structural ExperimentKey and verified
progress generation. Item values are excluded; selectors, commands, graph tool
identity/order/multiplicity and verifier expectations distinguish hypotheses.
Arbitrary tool output cannot declare progress. `agent/tool_result_classification.py`
uses canonical effect metadata instead of a second no-effect name whitelist.
`tools/registry.py` assigns first-party effects on registration; `tools/effects.py`
covers instructional reads and vision, with conservative unknown write capability.
`tools/workstation_work.py` exposes the bounded contract and recipe arguments through
the existing session-gated tool, without adding core tools or changing toolsets.

Canonical compiler/store/graph extensions and the seven-case Trello replay are
downstream-owned. TurnConstraintContext was already correctly scoped: real two-turn
tests prove reset and auxiliary route propagation; its lifecycle was not rewritten.
Workstation CI retains locked uv sync and adds the controlled replay and path filters
for affected core seams. See engineering journal for exact GitHub evidence.

Promotion gate follow-up extends Desktop-owned helpers: Windows secret-file mode
checks reject symlinks before the chmod no-op, and BrowserRuntime emits typed
human-control faults through the existing controller contract. First-launch E2E
checks visible setup/recovery or completed boot instead of arbitrary shell text;
stable UI selectors are additive. Windows qualification has an explicit bounded
900s timeout for the full suite (local Windows run measured 448s), Dashboard smoke
builds canonical web assets, and load evidence validates from repository cwd.
No gate, scope check, permission guard or uncertain replay protection is removed.

## HW-021 — Final durable graph/effect/turn hardening (2026-09-16)

Audit base: `17df394cfe3e913e6fce6f3130e3efc47e4592a2`, equal to fetched origin/main.
Focused upstream guardrails: `4716ec0ba4e212105f8f162c226f052b25f8a76b`.
No blind merge or downstream file replacement.

Core seams: `tools/effects.py` plus `tools/registry.py` add operator/plugin effect,
idempotency-key and provider-route metadata without enlarging LLM schemas.
`tools/mcp_tool.py` retains read hints on live/lazy-cache registrations while
preserving trust authorization. `agent/tool_executor.py` records successful
pre-compilation mutation artifacts; `run_agent.py` detects structural fan-out,
allows discovery beside rejected writes, carries adopted results, checks tool
routes and guards streaming/non-streaming provider calls. `agent/conversation_loop.py`
establishes TurnConstraintContext before provider selection; scoped ambient context
in `run_agent.py` and `agent/auxiliary_client.py` covers auxiliary resolver/cache
routes. `agent/chat_completion_helpers.py` prunes forbidden fallback entries before
client/credential resolution. ModelRouter shares the provider-route normalizer.

Guardrail adaptation centralizes upstream FAILURE_TOLERANT_TOOL_NAMES and
PROGRESS_RESET_TOOL_NAMES, and adds platform-aware non_interactive_hard_stop_enabled.
`agent/agent_init.py` uses that config instead of a separate environment/platform
override. Existing identical/cycle detection, downstream pollers and loop caps
remain. Unlike upstream blanket command-success reset, downstream requires trusted
delta/checkpoint or canonical landed file-write evidence.

Workstation additions extend existing compiler/runner/store: phase WorkItems,
bounded topological DAG, reference bindings, phase gates/checkpoints, graph ledger
and tool-owned connection closure. Linear plans remain compatible. Tests:
`test_durable_hardening.py` and controlled `benchmarks/trello_regression.py`.
CI uses the existing pinned setup-uv action/retry convention and uv.lock;
no individual missing dependencies are added manually.

## HW-020 — Durable execution boundary (2026-09-16)

Audited fork base: `3344e67fb67a3f9d2e89fa08707325807449d9b8`.
Restricted upstream comparison: `4e9d3c713a` (fetched upstream/main).
No merge/rebase or wholesale file replacement was performed.

Maintained core seams: `agent/conversation_loop.py` detects conservative repetitive
requests and records provider-reported usage for assistant persistence;
`run_agent.py` requires compiled mutations when the session has work_execute,
binds the existing scoped dispatcher, attaches trusted KanbanRun refs and forwards
token_count on flush. `agent/tool_executor.py` preserves pre-spill verifier input
and prevents private WorkItem messages from becoming unpaired transcript rows.
`toolsets.py`/`tools/workstation_work.py` add a session-selected desktop_ui capability,
never a new core tool. `tools/file_tools.py` adds hash/ref projection only for durable
reads, preserving legacy behavior/security/read stamps. `tools/tool_search.py` adds
Desktop session/profile/schema-addressed descriptions and explicit full reload.
`tools/browser_workstation.py` enforces constrained/durable internal routes before
probes or fallback.

Carefully adapted upstream guardrail behavior in `agent/tool_guardrails.py`:
period-2–4 identical cycles, identical-call halts, failure-tolerant distinct shell
commands and progress reset. Downstream tool names and poller exemptions remain.
`agent/agent_init.py` defaults unattended platforms to hard stops, respecting explicit
configuration and attended CLI/TUI/Desktop/subagent behavior. Verified checkpoint
progress is an explicit runtime signal; arbitrary result text cannot assert it.

Compressor comparison found substantial upstream refactoring; the fork's trusted
operational-ref boundary was preserved and reused. Storage comparison showed that
the scoped content-addressed downstream spillover extensions would disappear with
wholesale upstream replacement, so they were preserved. SessionDB already accounts
canonical provider usage: the missing per-message token_count came from the flush
payload, not a need to replace SessionDB. No new session/board/browser database.

Base: `NousResearch/hermes-agent@057dcdf236f8a6a26721c10fcc6ccb72726e272a`

| ID     | Area                                            | Downstream change                                                                                                                                                                                                                                                                                                    | Why core-level                                                                                                                                                                                                                   | Upstream candidate |
| ------ | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| HW-001 | Desktop main                                    | Load Workstation Browser runtime IPC                                                                                                                                                                                                                                                                                 | Browser is a first-class Desktop subsystem                                                                                                                                                                                       | Maybe              |
| HW-002 | Desktop preload                                 | Expose narrow Browser runtime bridge                                                                                                                                                                                                                                                                                 | Native view is owned by Electron main                                                                                                                                                                                            | Maybe              |
| HW-003 | Desktop routes/sidebar                          | Add `/browser` as core page/nav item                                                                                                                                                                                                                                                                                 | Workstation Browser is non-optional in this distribution                                                                                                                                                                         | No                 |
| HW-004 | Desktop surfaces                                | Render Browser page as built-in workspace page with host-aware native viewport geometry, native content clamping and resize/maximize/restore reconciliation, so stale Chat/Hub updates cannot move the single live `WebContentsView`                                                                                                                                          | Same reason as HW-003; host ownership and native window transitions must be enforced at the shared Desktop surface boundary                                                                                                                                                                  | No                 |
| HW-005 | Desktop Kanban plugin                           | Enable bundled Kanban by default                                                                                                                                                                                                                                                                                     | Kanban is Workstation task source of truth                                                                                                                                                                                       | Maybe              |
| HW-006 | global.d.ts                                     | Type the Browser IPC bridge                                                                                                                                                                                                                                                                                          | Required by HW-002                                                                                                                                                                                                               | Maybe              |
| HW-007 | Browser tool registry                           | Prefer Workstation Chromium before extension/legacy lanes                                                                                                                                                                                                                                                            | Browser must be a first-class agent capability                                                                                                                                                                                   | Yes                |
| HW-008 | CLI config example                              | Document `browser.workstation` defaults                                                                                                                                                                                                                                                                              | Makes routing/fail-closed behavior explicit                                                                                                                                                                                      | Yes                |
| HW-009 | Root `AGENTS.md`                                | Point coding agents at the downstream Workstation operational context/read-order                                                                                                                                                                                                                                     | Prevents downstream architecture changes from being made from upstream-only context                                                                                                                                              | No                 |
| HW-010 | GitHub Actions / Windows bootstrap                | Windows Workstation gate installs/tests the committed downstream tree and runs the strict doctor; the user-facing one-click bootstrap delegates to that same strict doctor before start, while the gate exposes focused Browser foundation lifecycle/session-state results and fails if install dirties the checkout | Prevents CI or the user-facing launcher from silently proceeding past a broken dependency/integration state while keeping scoped native-browser regressions visible beside known broad Windows failures | Maybe              |
| HW-011 | Tool schema assembly / registry                 | Preserve selected Workstation Browser schemas from session-scoped Desktop capability and fingerprint that capability in the tool-definition cache                                                                                                                                                                    | One gateway serves multiple client surfaces; transient controller reachability must not erase a valid Desktop schema or leak it across sessions                                                                                  | Yes                |
| HW-012 | Desktop Browser runtime / BrowserTask lifecycle | Add first-class BrowserTask metadata/lifecycle (`create`, `show`, `hide`, `park`, explicit `destroy`, crash recovery and safe logical restart restoration) around the existing `taskTabs` + `BrowserEntry.ownerTaskId` page ownership primitives, with focused runtime/persistence regression tests                  | BrowserTask semantics must be owned where Electron `WebContentsView` lifetime and task→page binding are enforced; implementing it only in downstream wrapper code would duplicate browser state or lose native lifecycle control | Maybe              |
| HW-013 | Desktop Browser runtime / BrowserSessionState   | Add one composite, versioned BrowserSessionState restart projection for ordinary/task logical tabs, order, active selection, sanitized restorable URL metadata and BrowserTask snapshot, including atomic-replacement recovery semantics that preserve the latest in-process intended projection after failed writes | Electron main owns the live page/runtime facts needed to derive and reconcile restart state; a wrapper-side store would duplicate page/task authority and cannot correctly preserve lazy task ownership                          | Maybe              |
| HW-014 | Desktop packaged/dev E2E harness                  | Inject Playwright's Electron loader when launching the packaged executable through a custom `executablePath`, isolate the Workstation Browser state/profile per sandbox, and resolve both workspace-local and Windows `.exe` Electron development binaries | Playwright only auto-injects its loader for the npm Electron binary; the real packaged executable otherwise starts without `__playwright_run()`, while workspace-local npm installs and Windows executable naming must also resolve to the same hidden dev runtime | Yes                |
| HW-015 | Workstation resource/event projection / client adapters | Expose the Electron-owned BrowserTask and Execution Journal projection through versioned loopback `/resources` and bounded `/events` endpoints, Desktop IPC, Dashboard REST and TUI JSON-RPC adapters, with a Dashboard recent-events view | Resource identity, permissions, lineage and operational events must be resolved from one browser/runtime authority across clients; keeping this adapter outside `workstation/` preserves upstream Hermes' narrow core and avoids a second task/journal store | Maybe              |
| HW-016 | Dashboard cross-engine E2E harness                 | Add a provider-free, isolated Playwright smoke for the built Dashboard served by the real Python backend in Chromium and Firefox, with Edge when the supported system browser is available | Cross-engine client compatibility is a Workstation acceptance gate, while the reusable harness/config belongs at the Desktop/Dashboard integration boundary rather than in the Python contract layer | No                 |
| HW-017 | Release qualification / bounded soak               | Add an evidence-producing release gate runner and bounded-duration soak metrics without mutating the checkout or inferring clean-machine results; execute the canonical four-session reconnect soak, the hidden-window native Electron Browser runtime reconnect soak, a real headless `hermes serve` multi-session/reconnect soak, and an integrated hidden-window Desktop/Browser load E2E in the Windows release workflow while retaining their JSON/durable evidence artifacts; let H013 select a bounded larger task/round/duration profile in candidate CI | Release promotion needs reproducible, reviewable evidence while clean-install and full agent/backend production-like Desktop/Browser load remain explicit acceptance boundaries; the backend probe and the integrated E2E extend evidence at the actual gateway/Chromium boundary without adding a second runtime owner, and explicit CI parameters distinguish a stronger release profile from the fast local regression | No                 |
| HW-018 | Desktop Windows portability closure                 | Make Desktop path, permission, Git, SSH, WSL, staging and deterministic-format contracts valid on Windows while preserving explicit POSIX and no-mux behavior | These are Desktop-owned runtime/test boundaries; fixing them at the integration edge removes the broad Windows gate debt without widening Hermes' core tool surface | Maybe              |
| HW-019 | Canonical Kanban Hybrid workspace                  | Add additive Hybrid board/column/card/activity tables to the existing per-board Kanban SQLite schema, a transaction-backed shared domain service, authenticated plugin routes, a service-gated agent tool and a Desktop projection that uses semantic moves rather than client ranks | Human and agent must mutate one canonical work domain without changing Agentic `tasks.status`; the existing Kanban DB, transaction and plugin transport are the only safe common seam | Maybe |

Operational context, engineering journal/probes, tests, migration helpers, and other files under `workstation/` are downstream-owned and do not add rows by themselves. Keep this list small. New edits outside `workstation/` require a row or an explicit expansion of an existing row.

HW-010 also covers the Windows workflow's fresh `RUNNER_TEMP` Workstation
home, persisted through `GITHUB_ENV` and shared by install and strict doctor;
the default user-facing `%LOCALAPPDATA%` path remains unchanged.

HW-017 also covers the release workflow's explicit isolated validation-dependency
step: the production Workstation install remains runtime-only, while the
candidate gate installs the pinned `[dev]` extra into `.venv` before invoking
the Python contract runner, so clean candidates do not fail merely because
`pytest` is absent from the runtime environment.

HW-017 also covers the production-only npm audit gate and the lockfile refresh
that closes the transitively reported `nanoid`, `sanitize-html` and `colord`
advisories without changing package manifests or enabling forced upgrades.

## Promoted browser-foundation evidence boundary

HW-012 was validated by the focused automated BrowserTask tests and the real Windows/Electron H004 lifecycle smoke recorded in `workstation/context/TESTING.md` and `workstation/context/engineering-journal/CURRENT.md`.

HW-013 is validated and promoted through PR #11. Exact head
`d5be442021ea0c744351622317eef5212219786d` passed static/focused gates and
the real Windows/Electron H010 clean/fault/abrupt restart probe; merge
`e0a99ef3aba6e6d2b65c30cf3c908ee1d49c4d29` is reachable from `main`.
The composite persistence adapter preserves HW-012 live ownership rather than
becoming a second BrowserTask/page store.

Neither promotion makes Browser Hub, Chat Browser View, Preview unification,
single-host transfer or complete Kanban/run integration part of the existing
delta. Those require their own V1 #1.5 rows/expansions when implemented.

## Canonical Work Loop working-tree seams (2026-09-17)

Additive canonical acceptance table/gates and Hybrid completion projection;
existing verification DB outcome events; shared ToolCallGuardrailController
WorkItem systemic failures; tool_contract introspection defaults; artifact refs
in read_file; cron drift migration metadata; SessionSearchMixin compacted dedup;
authenticated cockpit route in web_server. Provider schemas/system prompts,
SessionDB history and authorized red-team harness are unchanged. See
[implementation scope](context/CANONICAL_WORK_LOOP.md).
# Read-only durable preflight — 2026-09-17

Core seams: `run_agent.py` keeps discovery alive after refused compilation;
`agent/turn_constraints.py` parses mutation-scoped route restrictions without
changing global restrictions; `agent/conversation_loop.py` resets per-conversation
effect evidence; `tools/effects.py` separates method/channel capability evidence;
`tools/browser_tool.py` reuses structured extraction for Workstation inspection.
Canonical compiler/store checkpoints own persistence and restart behavior.
See `context/READONLY_DURABLE_PREFLIGHT.md` and `test_readonly_preflight.py`.
