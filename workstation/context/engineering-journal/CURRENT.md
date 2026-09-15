# CURRENT — Workstation Engineering Journal

Last updated: 2026-09-14
Active track: Workstation Knowledge Subsystem (Hermes Vault) & V3 Hardening
Repository: `kevynlucasprofissional-stack/hermes-agent`
Active feature branch: `main` plus current Workstation working tree
Journal entries:
- `v1-1-5-integrated-dogfood-mvp.md` (V1 #1.5 MVP verification)
- `v2-roadmap-completion.md` (Full roadmap completion: V1.1 and V2)
- `v3-runtime-hardening-closure.md` (V3.1–V3.4 contract-layer closure)
Status: 175/175 Workstation Pytest passing, Desktop typecheck passing.

## H-049 — Workstation Browser Automation Ergonomics: Input Hygiene, Proactive Human Handoff, Canvas Awareness, and Batch Extraction

Status: PLANNED & SPECIFIED — dogfood evidence recorded; roadmap promoted; implementation staged
Origin: workstation browser real-world dogfood session (session `20260914_223624_261d02`)
Date / ref: 2026-09-14 / `feat/workstation-hybrid-kanban`

### Claim and decision

Real-world dogfooding of the Hermes Workstation Browser across 5 diverse web paradigms (Python.org documentation, G1 news portal, Mercado Livre & Amazon e-commerce, GitHub issue tracker, and Google Maps SPA) demonstrated that while the Chromium runtime and agent task decomposition are robust, agentic web automation encounters four specific friction points:

1. **Input Hygiene (`browser_type`)**: Typing into inputs with pre-existing values (e.g. GitHub's issue search prefilled with `is:issue state:open `) resulted in duplicated/malformed queries (`is:issue state:open is:issue state:open label:...`) because CDP key events lacked explicit selection/clearing guarantees across platforms.
2. **Auth & Verification Wall Handoff**: When encountering account verification gates (Mercado Livre `/gz/account-verification`), the agent spent turns trying URL workarounds before abandoning the site. A proactive Human Handoff banner in the UI should invite the user to complete verification in the persistent profile once and yield control back to the agent.
3. **Canvas / WebGL SPA Settlement (Google Maps)**: Canvas-rendered applications yielded an initial accessibility snapshot with `total_text_chars: 1, element_count: 0`. The agent was forced to inject 10+ raw `browser_console` JS scripts to scroll and inspect feed cards. The browser runtime should detect canvas/sparse trees and perform adaptive DOM feed settling before returning snapshots.
4. **Batch Extraction Overhead**: Extracting product grids (Amazon) and opening hours tables (Maps) required 9–15 sequential tool calls and large token context budgets. Introducing a high-level `browser_extract_items` primitive allows one-step extraction of structured cards/tables without custom scripts.

### Observed evidence (Session `20260914_223624_261d02`)

- **Python.org**: Successfully extracted Python 3.14.7 and synthesized changelog in 2 turns.
- **G1 Tecnologia**: Successfully navigated, clicked lead article (`@e5`), and summarized deepfake investigation.
- **Mercado Livre**: Blocked by `/gz/account-verification` (trace ID `cff34157-597e-425d-818c-b3d1c2acfb6a`); agent pivoted to Amazon.
- **Amazon.com.br**: Extracted 47 items via raw console script; required 3 subsequent scans to map ASINs and filter 27" 144Hz monitors, demonstrating the need for structured extraction.
- **GitHub**: Search combobox `@e18` duplicated query strings upon `browser_type`.
- **Google Maps**: Initial snapshot was empty (0 elements); agent successfully used `browser_console` with DOM feed queries and `feed.scrollTop = feed.scrollHeight` to map 21 cafeterias and extract weekly schedules.

### Subsystem roadmap (V3.5)

- **Step 1: Input Clearing in `workstation-browser-runtime.ts`**: Make `typeRef` ensure complete input clearing (`Ctrl/Cmd+A` with explicit `windowsVirtualKeyCode: 65`, `select()`, and DOM value reset) before typing.
- **Step 2: Verification Wall Detection & Takeover Event**: Detect auth/challenge URL patterns and emit human-takeover prompt.
- **Step 3: Adaptive DOM Settlement for SPAs**: Heuristic wait for feed containers (`div[role="feed"]`, `main`, etc.) when canvas is detected and accessibility count is <= 2.
- **Step 4: `browser_extract_items` Tool**: First-class structured card/table extractor returning clean JSON with titles, prices, ratings, and links in one call.

## H-048 — Hybrid Kanban must extend canonical Kanban without inheriting Agentic status semantics

Status: EXPERIMENTAL VERTICAL SLICE IMPLEMENTED
Origin: human + agent shared workspaces
Date / ref: 2026-09-14 / `feat/workstation-hybrid-kanban`

### Claim and decision

The established Agentic Kanban is a dispatcher-owned execution state machine.
The requested Trello-like surface is instead a human/agent organization view.
The accepted seam is therefore additive Hybrid entities in the existing
per-board `hermes_cli.kanban_db` database, with one domain service shared by
the authenticated plugin API and agent tool. Hybrid columns have no mapping to
`tasks.status`, so a Hybrid “Done” move cannot complete an Agentic task.

### Experiment and observed evidence

- Added `hybrid_boards`, `hybrid_columns`, `hybrid_cards` and
  `hybrid_activity` through the normal idempotent Kanban schema initialization;
  no SessionDB, BrowserTask, Journal or Electron-local store was created.
- `hermes_cli.hybrid_kanban` performs all writes in `kanban_db.write_txn`.
  It uses semantic placement (`before_id`/`after_id`), a negative temporary
  rank namespace followed by dense canonical reindexing, and optional entity
  revision checks for stale client commands.
- The Kanban plugin API and `kanban_hybrid` agent tool call that same service.
  Activity records `human` or `agent` actor provenance plus available session
  and source; the Electron view is a cache/projection with post-write
  invalidation and bounded polling.
- `python -m pytest tests/hermes_cli/test_hybrid_kanban.py
  workstation/tests/test_kanban_journal.py -q` passed **9 tests**. It proves
  persistence after reopening SQLite, ordering repair, stale rejection,
  human/agent provenance, agent-tool/domain convergence, and that a Hybrid
  “Done” move leaves an Agentic task outside the terminal state.
- `npm run typecheck` in `apps/desktop` passed after restoring locked workspace
  dependencies.

### Deliberate boundary / next experiment

This is not yet a claim of multi-user realtime or a full Trello clone. The
next hardening slice should add visual activity/card drawer, destructive
archive policy, horizontal column drag, and Hybrid event invalidation through
the existing plugin websocket rather than a new realtime service.

## H-047 — Hermes Vault: Local-First Agentic PKM Subsystem (Obsidian-Compatible Knowledge Core)

Status: PLANNED & DESIGNED — architectural specification recorded; roadmap promoted; execution staged
Origin: workstation knowledge management & human-agent co-authoring
Date / ref: 2026-09-14 / `main`

### Claim

A personal AI agent's effectiveness multiplies when user and agent share a single, local-first, durable knowledge base. By introducing **Hermes Vault**—a native, Obsidian-compatible Personal Knowledge Management (PKM) subsystem inside Hermes Workstation—the user and agent can co-author plain Markdown files with bidirectional wikilinks (`[[Note]]`), backlinks indexing, YAML frontmatter, and interactive graph views directly in Hermes Desktop.

### Observed evidence & technical alignment

- Analysis of native Obsidian installation (`C:\Program Files\Obsidian`) confirms that Obsidian's foundation is Electron + Chromium + local Markdown files + CodeMirror editor + D3 force-directed graph.
- Hermes Desktop (`apps/desktop`) shares the identical Electron Chromium foundation and already includes a sophisticated force-directed simulation engine in `apps/desktop/src/app/starmap` (`d3-force`, canvas rendering, physics, zoom, viewport).
- Hermes Agent already owns rich file inspection, search, and editing capabilities (`read_file`, `write_to_file`, `replace_file_content`, `grep_search`).
- Creating a native `/vault` route and Vault indexer eliminates the boundary between external note-taking tools and the agent's memory/actions, enabling the agent to synthesize web research, log decisions, link concepts, and maintain maps of content (MOCs) locally with zero vendor lock-in.

### Subsystem architecture

1. **Vault Engine & Local Indexer (`workstation/vault.py` + Desktop IPC/service)**:
   - Root directory resolution (default `~/.hermes/vault` or user-configured external directory, e.g. an existing Obsidian vault).
   - Local filesystem watcher for `.md` files.
   - AST / regex parser for `[[wikilinks]]`, `#tags`, headings (`[[Note#Heading]]`), and YAML frontmatter properties.
   - In-memory bidirectional link cache (forward links and backlinks index).
2. **Desktop UI & Editor (`apps/desktop/src/app/vault`)**:
   - Master-detail file tree explorer + tags panel.
   - Markdown editor with live preview, syntax highlighting, and callout rendering.
   - Autocomplete triggers on typing `[[`.
   - Side panel showing incoming backlinks and metadata properties.
3. **Knowledge Graph View**:
   - Visualizing interconnected notes as an interactive graph using the `starmap` force-simulation primitives.
4. **Agent-Vault Bridge Tools**:
   - `vault_search(query)`: FTS5 / BM25 search across vault markdown notes.
   - `vault_read(note)`: Read note content and metadata.
   - `vault_write(note, content)`: Create or update notes with frontmatter.
   - `vault_backlinks(note)`: Query references and connected notes.

## H-054 — Corpus hardening audit: retain only gaps reproducible on current `main`

Status: ACTIVE — evidence and focused reproductions in progress
Origin: real Hermes conversation corpus supplied for the Workstation hardening
audit
Audit base: `main@1ca169ca15626071c21122f3ea6801065de8d407` (2026-09-14)

### Scope and evidence boundary

The supplied corpus describes 50 conversations with repeated tool payloads,
long polling waits, compactions and tool-call limits. It is workload evidence,
not proof that an historical implementation still has the same defect. The
named corpus Markdown/CSV attachments were not present in this checkout during
the initial audit, so their stated aggregate figures are retained as the task
input but no per-conversation causal claim is made without the attachments or a
current reproduction.

### Initial gap matrix on the audit base

| Area | Classification | Current evidence / disposition |
| --- | --- | --- |
| Large tool-result spillover | PARTIAL | `tools.tool_result_storage` persists only oversized results and gives a text/path preview; it has no content-addressed reference, scope-safe stable-result reuse, or `unchanged` contract. `read_file` is deliberately exempted from spillover. |
| Inline budget / large outputs | PARTIAL | Per-result and per-turn character budgets exist, but no normalized `verbosity`/`inline_budget`/artifact contract governs all results. |
| Persistent-worker result durability | CONFIRMED_GAP | `workstation/workers.py` persists worker records and pending messages, while `WorkerResultEnvelope`s live only in `PersistentWorker._results`; reconstructed workers start with an empty result list. |
| WorkItems / exception-driven execution | PARTIAL | Persistent workers, journal and events exist; no durable WorkItem lifecycle or generic validator-to-anomaly escalation was found. |
| Structured operational errors | CONFIRMED_GAP | The Electron controller serializes action exceptions as `{ success: false, error: message }`, rather than a stable remediation contract. |
| BrowserTask lifecycle/restart | ALREADY_SOLVED | BrowserTask/BrowserSessionState single-page and lazy restart contracts are implemented and covered by focused/runtime/native evidence. |
| Controller descriptor reconciliation | PARTIAL | Controller startup writes a descriptor with the Electron PID, but the audited route still needs a current lifecycle/reconciliation proof before any extraction or rewrite is considered. |
| Human takeover | PARTIAL | Browser actions are blocked while `controlOwner === 'human'`; the present owner is runtime-global, so cross-host/task scoping needs focused validation before changing semantics. |
| Process ownership guardrails | ALREADY_SOLVED | `tools.process_registry` only tree-kills tracked spawned PIDs and verifies recorded process start time before signalling; no wildcard Chrome-kill path was found in that owner. |
| Event-driven waits | PARTIAL | `RuntimeEventBus` and worker blocking waits exist, but reusable condition-specific await projections have not yet been established. |
| Context compaction | PARTIAL | Workstation session lifecycle records compaction markers and bounded memory compaction exists, but the core conversation compactor still needs a separate trace before a non-recursive operational-reference change can be justified. |
| Session `null` / KI-007 | NOT_REPRODUCED | KI-007 remains causally unproven; no SessionDB/Gateway change is authorized without the documented reproduction gate. |
| Session forensic manifest | PARTIAL | Session/worker/resource identities are individually projected; no single redacted manifest export was established in this audit. |
| Capability preflight | PARTIAL | Worker/host/toolset capability sources exist, but no single WorkPlan projection has yet been found. |
| Toolset/schema handles | PARTIAL | Tool-definition caching/fingerprinting exists, but repeated discovery results have no consumer-visible delta handle. |
| Skill identity / collisions | ALREADY_SOLVED | Skill hashes and explicit ambiguous-name rejection are present; no silent collision reproduction was found. |
| Skill preflight | PARTIAL | Existing guard and install validation cover substantial input validation; model-visible preflight constraints require a separate skill-manager trace. |
| Effective configuration semantics | NOT_REPRODUCED | The historical unknown-key symptom was not reproduced in the audit window; no config-owner change is justified yet. |
| Typed host/process operations | PARTIAL | The tracked ProcessRegistry and host adapters cover high-value operations; shell-wide command paths still need a focused safety/capability audit before adding API surface. |
| Provenance/validators | PARTIAL | Perception, journal and evidence metadata exist, but no generic WorkItem validator escalation was found. |
| Experience promotion | PARTIAL | Procedure promotion is explicit; the requested skill/guardrail/regression triage needs a separately scoped evidence lifecycle. |
| Observability projection | ALREADY_SOLVED | EvidenceState, RuntimeEventBus, resources and Desktop projections already expose the canonical operational plane. |
| Corpus benchmark | CONFIRMED_GAP | No versioned baseline-vs-hardened benchmark for the supplied workload shape was found. |

### First focused experiments

1. Reconstruct a completed persistent worker and prove whether an unread result
   survives exactly once.
2. Exercise the existing result-persistence boundary twice with stable content,
   then verify whether the second delivery can be represented by a safe handle
   instead of reinjecting the raw body.
3. Exercise controller error normalization at the authenticated action boundary
   without changing BrowserTask ownership or fallback routing.

No new SessionDB, Kanban, Memory store, BrowserTask store, browser runtime,
agent core, scheduler or event bus is permitted by this investigation.

### Observed result — 2026-09-15

- Confirmed the worker-result gap with the pre-change implementation: a
  reconstructed `PersistentWorker` had an empty `_results` list even when the
  previous executor had completed.
- `WorkerMessage.work_item_id`, a durable result envelope, explicit consumer
  ACK and durable in-flight claim now extend the existing worker record. The
  result is persisted before journal/event publication; persistence failure
  leaves recovery work and never publishes `COMPLETED`.
- Large results already selected for spillover now receive scope-isolated
  content-addressed `result_ref`, `content_hash`, `artifact_ref`, byte count,
  cache status and `inline_truncated` metadata. Repeated content in the same
  task scope reuses the immutable artifact; a distinct scope receives a
  distinct reference. This is a representation cache, not an unsafe
  argument-only execution cache.
- The Electron controller keeps the compatible `error` string and now adds a
  structured recovery contract for stale refs, unbound tabs, human control,
  timeout, invalid arguments and controller loss. BrowserTask ownership and
  fail-closed routing remain unchanged.
- Focused validation: `43 passed` across
  `tests/tools/test_tool_result_storage.py` and
  `workstation/tests/test_persistent_workers.py`; Python compile check passed.
  Electron typecheck/runtime validation remains blocked locally because the
  existing shared `node_modules` cannot materialize its locked dev dependencies
  (`ENOTEMPTY` during npm install). No dependency directory was removed or
  repaired by this investigation.

## H-046 — Chrome Web Store support is not yet an agentic capability boundary

Status: PARTIALLY RESOLVED — governed agentic install slice validated; discovery/catalog and dedicated UI remain open
Origin: agentic extensions / canonical Kanban continuation
Date / ref: 2026-09-14 / `main@9291584e6f`

### Claim

The V2.1 roadmap label "Completed" implies that a Desktop agent can safely
request, approve, install, load, verify and use a Chrome Web Store extension.

### Observed evidence

- `workstation/extensions.py` can download, unpack and register a CRX, but is
  not registered as an agent tool and records no policy or journal events;
- `WorkstationBrowserRuntime.loadInstalledExtensions()` scans directories and
  calls `session.loadExtension()` best-effort during startup, discarding both
  success and failure identity; it exposes no controller/IPC operation for
  immediate load, verification, removal or options navigation;
- `ScopedPolicyEngine` has no extension capability/risk rule, and no existing
  call site joins the manager to Hermes' approval gate;
- the canonical `WorkstationKanbanBridge` already creates parent/child cards
  and journals follow-up creation, but its child card body/metadata omit the
  supplied discovery evidence.

### Implemented result and evidence

- Implemented the stated Desktop-session-only slice in
  `tools/workstation_extensions.py`: four explicit tools call the canonical
  manager, policy and approval gate. Non-Desktop sessions do not receive their
  schemas, even if the process was launched by Electron.
- `ChromeExtensionManager` now inspects manifests before persistence, rejects
  unsafe ZIP paths, atomically writes its registry, and classifies permissions.
- The Electron controller now loads, verifies and removes an extension on
  demand, and permits a verified extension's options page only through the
  controlled action. Startup restoration uses the same verifier instead of a
  silent best-effort load.
- `26 passed` across focused Python contracts (extension manager, policy,
  Kanban follow-up dependency and Desktop-session schema surface); Electron
  runtime focused test passed `19/19`; Desktop typecheck passed.

### Remaining boundary

This result proves a supplied Web Store URL/ID through policy, approval,
install, Electron load and verification. It deliberately does not claim that
the agent can search a marketplace or that the Desktop has a human extension
management UI; those require a separately scoped catalog/UI capability.

## H-041 — Current Windows H010 rerun validates BrowserSessionState on the live working tree

Status: VALIDATED — LOCAL NATIVE EVIDENCE; CLEAN-CHECKOUT QUALIFICATION STILL OPEN
Origin: continuation of the V3 roadmap acceptance audit
Date / ref: 2026-09-12 / `main@d77901a6857cf90f9401a90377de3b6ee5254bef`

### Claim

The versioned H010 native probe still validates the BrowserSessionState and
BrowserTask restart boundary on the current Windows/Electron toolchain after
the latest Workstation changes.

### Observed evidence

- `node workstation/context/engineering-journal/probes/h010-native-browser-session-state-smoke.mjs` passed before any visible Desktop window was opened;
- Windows `10.0.26200`, Electron `40.10.2`, outer Node `v26.7.0`;
- clean restart, failed-write convergence, explicit-destroy cleanup and abrupt
  restart all emitted their pass markers;
- phase A and phase B used distinct Electron PIDs, and phase B restored the
  durable logical state before lazily recreating exactly one task page;
- final marker: `H010_CLASSIFICATION=VALIDATED`.

### Boundary

The probe resolved `HEAD` to the SHA above but imported product files from the
current working tree. Because the checkout contains uncommitted Workstation
changes, this is strong local native evidence rather than clean-machine or
clean-checkout release evidence. The Windows workflow's exact-SHA and
clean-install gates remain authoritative for promotion.

## H-042 — Hidden native Browser runtime reconnect soak passes

Status: VALIDATED — NATIVE RUNTIME BOUNDARY; FULL PRODUCTION LOAD STILL OPEN
Origin: continuation of the V3.2/V3.4 soak acceptance audit
Date / ref: 2026-09-12 / `main@d77901a6857cf90f9401a90377de3b6ee5254bef`

### Claim

The Electron Workstation Browser can repeatedly recreate its runtime across
processes while retaining canonical BrowserTask/session/resource identity and
the one-live-page-per-task invariant under navigation and host transitions.

### Observed evidence

- versioned probe: `workstation/context/engineering-journal/probes/h011-native-browser-runtime-soak.mjs`;
- 60-second local run on Windows `10.0.26200`, Electron `40.10.2`, Node
  `v26.7.0`;
- 39 hidden Electron process episodes, 936 task cycles, four task identities,
  repeated local Chromium navigation, `hub`/`chat` host changes, hide/park
  transitions and resource lineage assertions;
- every episode passed, and the final durable `browser-session.json` retained
  exactly four unique BrowserTask records;
- final marker: `H011_NATIVE_BROWSER_RUNTIME_CLASSIFICATION=VALIDATED`.

### Boundary

The probe uses a local deterministic HTTP page and direct runtime APIs, with
all BrowserWindow instances created as `show: false` and all child processes
spawned with `windowsHide: true`. It is stronger than mocked Electron tests,
but it does not prove the full Hermes agent/backend workload, clean-machine
installation, or a long-duration production deployment. The Windows workflow
now runs the same probe for 60 seconds and uploads its JSON/durable projection.

> Journal/probe/documentation commits after `1ac0e0a9...` do not change BrowserTask product behavior unless this file explicitly records a later code-bearing candidate. Always verify live `main` and compare product paths before carrying evidence into a later implementation.

## H-043 — Real headless backend multi-session reconnect soak passes

Status: VALIDATED — BACKEND/AGENT TRANSPORT BOUNDARY; FULL DESKTOP/BROWSER LOAD STILL OPEN
Origin: continuation of the V3.2/V3.4 production-like load evidence audit
Date / ref: 2026-09-12 / `main@d77901a6857cf90f9401a90377de3b6ee5254bef`

### Claim

The shipped headless `hermes serve` backend can sustain concurrent desktop-shaped
WebSocket sessions, streamed agent turns and durable session resumption across
real backend process restarts without relying on Electron or an external model
provider.

### Observed evidence

- versioned probe: `workstation/context/engineering-journal/probes/h012-headless-backend-reconnect-soak.py`;
- local workflow-shaped run used the real `hermes serve` child process, the
  authenticated `/api/ws` gateway and an isolated `HERMES_HOME`;
- 12 cycles, 48 turns, 44 reconnects, 2 backend restarts, 48 heartbeats and
  192 streamed events passed with zero errors and four concurrent turns;
- every session resumed from its persisted stored identity after process restart;
- final marker: `H012_HEADLESS_BACKEND_CLASSIFICATION=VALIDATED`.

### Boundary

The synthetic turn seam is deterministic and token-free, so this evidence is
about real backend process/transport/session behavior rather than provider or
model quality. It does not prove clean-machine installation, native Browser
composition, remote-client parity or a long-duration full Desktop/Browser
deployment. The Windows workflow runs the same probe with a 120-second budget,
requires two restarts and uploads the JSON/durable home evidence.

## H-044 — Integrated Desktop/Browser headless load gate is validated locally

Status: **VALIDATED — LOCAL HIDDEN-WINDOW RUNTIME GREEN**
Origin: continuation of the V3.2/V3.4 Desktop/Browser production-load audit
Date / ref: 2026-09-12 / `main@d77901a6857cf90f9401a90377de3b6ee5254bef`

### Claim

The Windows release workflow now includes a hidden-window integrated Desktop/
Browser E2E. It uses the real dev Electron shell and real `hermes serve`
backend, drives four deterministic local Chromium pages through the authenticated
Workstation controller, compares controller resource/event projections against
Desktop IPC, and exercises BrowserTask hide/park/destroy lifecycle cleanup.

### Evidence boundary

The spec typechecks, the Desktop build succeeds, and the local runtime passed
**2 tests in 36.6s** after the fixture resolved the workspace-local Windows
`electron.exe`. The base run exercised four native tasks, controller/IPC
identity, host-aware viewport geometry/transfer, native maximize/restore
reconciliation, events and hide/park/destroy cleanup. The sustained run added
eight concurrent tasks, three navigation/snapshot rounds and three real chat
turns through the `hermes serve` backend.
The test sets
`HERMES_DESKTOP_E2E_HEADLESS=1` and asserts that every Electron window remains
hidden, so it did not open a user-visible window. The Windows workflow still
executes H013 as the clean release-candidate gate.

## H-045 — Windows worker persistence tolerates transient atomic-replace locks

Status: **VALIDATED — WINDOWS CONTRACT REGRESSION GREEN**
Origin: continuation of the Windows portability gate exposed by the full
Workstation suite
Date / ref: 2026-09-12 / `main@d77901a6857cf90f9401a90377de3b6ee5254bef`

### Claim

`WorkerRegistry` persistence retains its atomic temp-file replacement contract
while tolerating a short Windows sharing lock from security/indexing software.
The bounded retry is only for `PermissionError`; a persistent failure still
raises to the caller.

### Observed evidence

- the previously exposed full-suite `WinError 5` at `Path.replace()` is covered
  by the bounded retry in `workstation/workers.py`;
- focused persistent-worker tests: **4 passed**;
- complete Workstation regression after the change: **158 passed in 12.94s**.

## H-046 — Shared Desktop viewport geometry is host-aware

Status: **VALIDATED — FOCUSED RUNTIME CONTRACT GREEN**
Origin: KI-004 hardening audit after the H013 integrated E2E
Date / ref: 2026-09-12 / current Workstation working tree

### Claim

Chat and Browser Hub share one Electron `BrowserWindow`; therefore the sender
window cannot by itself identify which pane produced a resize update. The
Desktop bridge now carries an optional expected host and the runtime ignores a
host-aware `setBounds` call from a non-owner pane while retaining the existing
backward-compatible unqualified call path.

### Evidence

The focused viewport tests cover stale-host rejection, zoom/clamp/invalid-bound
normalization, native resize reconciliation and rehoming one live
`WebContentsView` between two native window hosts. H013 additionally exercised
the host-aware bridge and native maximize/restore path against the real
Electron/Chromium view while all windows remained hidden. The full Workstation
suite remains green; broader compositor-race, long-duration load and
clean-machine Windows matrices are still explicit evidence gates.

## H-047 — H013 sustained Desktop/Browser load expansion

Status: **VALIDATED — LOCAL HIDDEN-WINDOW LOAD REGRESSION GREEN**
Origin: continuation of the V3.2/V3.4 Desktop/Browser production-load audit
Date / ref: 2026-09-12 / current Workstation working tree

### Claim

The integrated Desktop/Browser gate can sustain multiple concurrent native
BrowserTasks across repeated controller navigation/snapshot rounds while the
same real `hermes serve` backend handles chat turns, without duplicating task
pages or exposing a native window.

### Evidence

The expanded `workstation-headless-load.spec.ts` passed **2 tests in 36.6s**:
the original four-task controller/IPC/geometry scenario and a second scenario
with **8 tasks × 3 rounds**, native task selection across Hub/Chat, **3 real
chat turns**, resource/event parity and explicit hide/park/destroy cleanup. The
run used `HERMES_DESKTOP_E2E_HEADLESS=1`; broader long-duration production load,
clean-machine qualification and compositor-race coverage remain open.

## H-052 — H013 full resource/event projection parity

Status: **VALIDATED — FULL RESOURCE/EVENT PARITY GREEN**
Origin: KI-008 current-client parity evidence boundary
Date / ref: 2026-09-12 / current Workstation working tree

### Claim

The integrated H013 controller/IPC assertions can compare the complete
resource state and event payload—not only resource identity—while ignoring
only call-generated timestamps that are expected to differ between snapshots.

### Confirming evidence expected

- browser task lineage, permissions, lifecycle state and evidence URIs match
  across controller and Desktop IPC;
- bounded event task lineage and event payloads match across both surfaces;
- the four-task and scaled profiles remain hidden and clean after the stronger
  assertions.

### Refuting evidence expected

- controller and IPC disagree on any canonical resource/event field;
- the comparison hides a semantic field rather than only generated timestamps;
- the stronger parity check causes task/page duplication or cleanup failure.

### Result

H013 now compares complete controller/IPC resource projections, including
permissions, state, lineage and evidence URIs, plus the full bounded event
payload. Both the default suite (**2 tests in 35.4s**) and the exact scaled
profile (**2 tests in 3.1 minutes**) passed. The scaled report recorded **16
tasks**, **150 completed rounds**, `observed_duration_ms=120075`, **8 chat
turns** and `accepted=true`; the post-run process check reported `no_electron`.
Only generated snapshot timestamps are excluded from comparison.

## H-053 — Clean-machine install and doctor profile isolation

Status: **VALIDATED — ISOLATED INSTALL/DOCTOR PROFILE GREEN**
Origin: KI-008 clean-machine release evidence audit
Date / ref: 2026-09-12 / current Workstation working tree

### Claim

The Windows release job can prove that installation and doctor validation use
the same fresh Workstation home when the workflow allocates an explicit
`RUNNER_TEMP` directory and both scripts resolve `HERMES_WORKSTATION_HOME`.

### Confirming evidence expected

- install and doctor report the same isolated home and leave the source
  checkout clean;
- the clean-install evidence records the isolated home and candidate revision;
- existing bootstrap/doctor contracts remain green without changing the
  default user-facing `%LOCALAPPDATA%` behavior.

### Refuting evidence expected

- doctor still reports a different profile than install;
- the isolated home is not created or the workflow cannot carry it between
  steps;
- existing local/default path contracts regress.

### Result

Using a fresh `HERMES_WORKSTATION_HOME` under the Windows temp directory,
`workstation\install.cmd -SkipDependencies` and `workstation\doctor.cmd
-Strict` both passed. Install created `Runtime` and `Browser`, doctor reported
the exact same home and `Browser\User Data` path, and the checkout remained
unchanged. The workflow now persists a unique `RUNNER_TEMP` home through
`GITHUB_ENV` and includes it in candidate-matched clean-install evidence.

## H-054 — Clean-install evidence must name the isolated Workstation home

Status: **VALIDATED — SHARED REPORT VALIDATOR GREEN**
Origin: H-053 install/doctor profile isolation audit
Date / ref: 2026-09-12 / current Workstation working tree

### Claim

Requiring a non-empty `workstation_home` field in accepted clean-install
evidence prevents the release qualification runner from accepting a report
whose `accepted` flag and revision match but whose runtime profile is unknown.

### Confirming evidence expected

- the workflow report carries the isolated home and the runner requires it;
- existing accepted/rejected evidence tests remain precise and the workflow
  still passes its source contract;
- local/default behavior remains unchanged when no explicit home is set.

### Refuting evidence expected

- a report without `workstation_home` still passes the clean-install stage;
- the new field requirement rejects a valid report produced by the workflow;
- the runner starts inspecting or mutating the external machine path.

### Result

`ReleaseQualificationRunner._check_clean_install()` now requires a non-empty
`workstation_home`. The Windows workflow records the same fresh
`RUNNER_TEMP` home used by install and doctor. The focused isolation,
bootstrap and documentation suite passed **27 tests**, and the real local
`install.cmd -SkipDependencies` → `doctor.cmd -Strict` run passed with matching
`Runtime`, `Browser` and `Browser\User Data` paths while leaving the checkout
unchanged.

## H-055 — Clean-install home must be absolute and outside the candidate

Status: **VALIDATED — SHARED REPORT VALIDATOR GREEN**
Origin: H-054 clean-install evidence contract hardening
Date / ref: 2026-09-12 / current Workstation working tree

### Claim

Requiring an absolute `workstation_home` outside the candidate checkout makes
the accepted clean-install report prove profile separation rather than merely
recording an arbitrary non-empty string.

### Confirming evidence expected

- relative paths and paths inside the candidate root are rejected;
- the workflow's absolute `RUNNER_TEMP` path and an external absolute path are
  accepted without probing or mutating that path;
- existing release qualification behavior remains fail-closed.

### Refuting evidence expected

- `.` or a candidate subdirectory is accepted as an isolated home;
- valid external Windows/POSIX absolute paths are rejected by platform-neutral
  validation;
- the runner starts treating the evidence path as a writable local resource.

### Result

`workstation.release_qualification` now rejects missing, relative and
candidate-contained `workstation_home` values while accepting an external
absolute path without reading or mutating it. The qualification/bootstrap/
documentation suite passed **30 tests**, including both valid and invalid
evidence cases.

## H-056 — Full Workstation regression after qualification hardening

Status: **VALIDATED — FULL SUITE GREEN**
Origin: regression after H-053 through H-055 release-evidence changes
Date / ref: 2026-09-12 / current Workstation working tree

### Evidence

The complete `workstation/tests` suite passed **163 tests in 16.08s** after
the doctor profile, clean-install report and absolute-home validation changes.
The result includes the new qualification cases and all prior Workstation
contracts; the candidate-release workflow itself remains an external gate.

## H-057 — Shared validation for the H013 release evidence report

Status: **VALIDATED — SHARED REPORT VALIDATOR GREEN**
Origin: H-051/H-050 candidate-load report audit
Date / ref: 2026-09-12 / current Workstation working tree

### Claim

A read-only Python validator shared by the release workflow and contract tests
can enforce the complete H013 profile without relying on duplicated, weaker
PowerShell field checks.

### Confirming evidence expected

- accepted reports must meet minimum tasks, requested rounds, duration and
  chat-turn values and must reach either the round or duration boundary;
- malformed, partial or under-sized reports fail with a useful reason;
- the validator performs no writes and the workflow still uploads the original
  report for review.

### Refuting evidence expected

- a report below the configured profile is accepted;
- valid reports with a larger profile are rejected unnecessarily;
- moving validation to Python changes the hidden-window H013 behavior.

### Result

`workstation.desktop_load_evidence` now enforces minimum tasks, requested
rounds, duration, chat turns and a reached stopping boundary without writing
to the report. Its focused suite plus the qualification/bootstrap/documentation
contracts passed **41 tests**. It also accepted the actual 16-task H013 report
(`150` rounds, `120075` ms, `8` chat turns), and the workflow invokes the same
validator before allowing the H013 gate to succeed.

## H-058 — Full Workstation regression after shared evidence validation

Status: **VALIDATED — FULL SUITE GREEN**
Origin: regression after H-057 shared H013 report validation
Date / ref: 2026-09-12 / current Workstation working tree

### Evidence

The complete `workstation/tests` suite passed **174 tests in 10.02s** after
the shared desktop-load evidence validator and its contract coverage were
added. The candidate-release workflow remains an external clean-machine gate.

## H-059 — H013 validator numeric-boundary regression

Status: **VALIDATED — FULL SUITE GREEN**
Origin: malformed-duration audit of H-057 shared evidence validator
Date / ref: 2026-09-12 / current Workstation working tree

### Evidence

The H013 report validator now rejects non-positive task/round/chat counts and
negative duration values before applying the round-or-duration boundary rule.
The focused validator/bootstrap/isolation contracts passed **38 tests**, and
the complete `workstation/tests` suite passed **175 tests in 9.64s**.

## H-060 — Clean candidate validation dependency boundary

Status: **VALIDATED — LOCAL CLEAN CANDIDATE GREEN / OFFICIAL CI STILL REQUIRED**
Origin: release qualification stopped at `workstation_smoke` because the
runtime-only `.venv` intentionally did not contain `pytest`
Date / ref: 2026-09-12 / current Workstation working tree

### Claim

The Windows release workflow must install its pinned validation dependencies
explicitly before invoking the read-only release qualification runner, while
keeping the normal runtime installation free of test tooling.

### Result

The workflow now installs `.[dev]` into the isolated candidate `.venv` after
the production install/build and before release qualification. A temporary
clean candidate with the current tree passed real install, `npm ci`, Desktop
production build and strict doctor, then passed all **6/6** release
qualification stages. Its hidden H013 profile passed **2/2** in **3.2 minutes**
and the shared validator accepted the report (`16` tasks, `173` rounds,
`120262` ms, `8` chat turns). This validates the local candidate path only;
the official Windows candidate workflow remains the promotion authority.

## H-061 — Production dependency advisory closure

Status: **VALIDATED — LOCKFILE-ONLY SECURITY REFRESH GREEN**
Origin: clean-candidate `npm ci` reported three production advisories
Date / ref: 2026-09-12 / current Workstation working tree

### Claim

The production dependency gate can close the reported transitive advisories
without forced upgrades or package-manifest churn, while preserving a
reproducible Desktop install and build.

### Result

The lockfile now refreshes `colord` 2.9.3→2.10.0, `sanitize-html`
2.17.6→2.17.7 and both vulnerable nested `nanoid` 3.3.17 entries→3.3.19.
`npm ci` completed successfully, `npm audit --omit=dev
--audit-level=moderate` reported **0 vulnerabilities**, and the Desktop
production build passed. The Windows workflow now runs this audit as a
required gate; development-only advisories remain outside the production
gate and are not silently forced to upgrade.

## H-049 — Parametric H013 production-load profile

Status: **VALIDATED — LOCAL SCALED PROFILE GREEN / CI PROFILE CONFIGURED**
Origin: KI-008 full-duration/production-scale Desktop/Browser evidence gate
Date / ref: 2026-09-12 / current Workstation working tree

### Claim

The existing integrated H013 harness can provide a reproducible larger release
profile if task count, sustained rounds and an optional duration budget are
controlled by explicit CI environment inputs, while the local default remains
bounded and fast.

### Confirming evidence expected

- default local behavior remains the validated four-task plus eight-task,
  three-round hidden-window suite;
- the release workflow can select a larger bounded task/round profile without
  changing source or introducing a second browser/runtime owner;
- the scaled profile preserves controller/IPC identity, native geometry,
  backend chat turns and explicit cleanup.

### Refuting evidence expected

- environment parsing permits invalid or unbounded values;
- scaled execution exposes a window, duplicates task pages or leaks tasks after
  cleanup;
- the release workflow cannot distinguish the bounded local profile from the
  stronger candidate-release evidence.

### Result

The H013 harness now validates explicit bounded integer inputs for task count,
round count, duration and chat turns. The default local profile is unchanged.
A local scaled run with **12 tasks**, a **15-second** duration budget and **4
chat turns** passed **2 tests in 55.4s** with hidden Electron cleanup reporting
`no_electron`. The Windows workflow selects **16 tasks**, up to **120 seconds**
and **8 chat turns**; that candidate-release execution remains external
evidence and is not claimed from the local run.

## H-050 — H013 scaled evidence report and release assertion

Status: **VALIDATED — REPORT-BASED LOCAL PROFILE GREEN / CI ASSERTION CONFIGURED**
Origin: H-049 scaled-load gate hardening
Date / ref: 2026-09-12 / current Workstation working tree

### Claim

An explicit H013 JSON report, checked by the Windows workflow, can distinguish
a complete scaled candidate run from a Playwright process that merely exits
successfully after a partial or under-sized load.

### Confirming evidence expected

- the report records requested and completed rounds, task count, duration and
  real chat-turn count;
- the workflow requires the report to be accepted and to satisfy the selected
  16-task/120-second/8-turn profile, while allowing either the requested round
  count or the duration budget to be the stopping boundary;
- report generation happens only after explicit task cleanup succeeds.

### Refuting evidence expected

- a missing or partial report can still pass the release step;
- the report claims completion before hide/park/destroy cleanup;
- local default runs become dependent on a CI-only report path.

### Result

The H013 suite now writes an optional report only after all explicit
hide/park/destroy operations leave zero tasks. A local report-backed run with
**10 tasks**, a **5-second** duration budget and **2 chat turns** passed **2
tests in 39.0s** and produced `accepted=true`, **13 completed rounds**,
`observed_duration_ms=5165` and `no_electron` after cleanup. The Windows step
now rejects a missing/under-sized report and requires the configured 16-task,
120-second, 8-turn profile; the candidate workflow still supplies the required
external release evidence.

## H-051 — Full configured H013 candidate-profile local run

Status: **VALIDATED — FULL CONFIGURED PROFILE GREEN / HIDDEN CLEANUP**
Origin: H-050 report-based scaled-load gate
Date / ref: 2026-09-12 / current Workstation working tree

### Claim

The exact H013 profile selected by the Windows release workflow—16 concurrent
BrowserTasks, a 120-second sustained duration budget and 8 real backend chat
turns—completes with the same identity, geometry and cleanup invariants on the
current Windows toolchain while remaining hidden.

### Confirming evidence expected

- the report is accepted, reaches the configured duration or round boundary,
  and records at least 16 tasks and 8 chat turns;
- all controller/IPC/resource/event and native geometry assertions remain green;
- explicit cleanup leaves no tasks and no Electron process remains afterward.

### Refuting evidence expected

- the full profile times out, loses task identity, leaks a page/task or reveals
  a native window;
- report validation cannot prove the configured load boundary.

### Result

The exact configured profile passed **2 tests in 3.0 minutes**: **16 tasks**,
**170 completed rounds**, `requested_duration_ms=120000`,
`observed_duration_ms=120651`, **8 real backend chat turns** and
`accepted=true`. Controller/IPC/resource/event, native geometry and explicit
cleanup assertions remained green; the post-run process check reported
`no_electron`. This is strong local Windows/toolchain evidence for the scaled
profile, but it does not promote clean-machine or release-candidate evidence.

## H-048 — H013 formatted rerun and final process cleanup

Status: **VALIDATED — REPEAT GREEN / NO ELECTRON LEFT RUNNING**
Origin: final regression after the H013 fixture/test formatting pass
Date / ref: 2026-09-12 / current Workstation working tree

### Evidence

After the mechanical Prettier correction, Desktop typecheck passed with zero
errors and the same hidden-window H013 suite passed **2 tests in 38.8s**. The
rerun covered the four-task controller/IPC scenario plus **8 tasks × 3 rounds**,
three real backend chat turns, native maximize/restore reconciliation and
hide/park/destroy cleanup. The post-test process check reported `no_electron`;
the 36.6s run in H-047 remains valid prior-run evidence, while the duration
variation is expected for this local Windows environment. Clean-machine,
full-duration/production-scale and broader compositor-race evidence remain
open gates.

## WP-02 — Mainline Consolidation Gate

## H-013 — Roadmap V3.1/V3.2 gaps require extensions over existing owners

Status: ACTIVE — REGISTERED BEFORE PRODUCT CHANGE
Origin: User objective / roadmap audit
Date / ref: 2026-09-10 / `main@d77901a685`

### Claim

The remaining V3.1/V3.2 roadmap work can be implemented as a set of typed,
testable extensions over `ExecutionJournal`, `ProceduralMemory` and
`WorkerRegistry`, without introducing a second SessionDB, Kanban, Memory,
BrowserTask page store or agent core.

### Confirming evidence expected

- current code has no EvidenceState, independent supervisor, Recovery Plane,
  promoted routine runner, persistent worker lifecycle or typed resource bus;
- new modules can persist only projections/operational metadata and retain
  existing owners for journal, memory and lineage;
- focused tests can prove stale-state, recovery, drift and worker-control
  invariants without replacing existing integration paths.

### Refuting evidence expected

An existing owner already provides the missing semantics, or the proposed
extension requires duplicating canonical task/session/memory/browser state or
changing the model tool schema during a conversation.

### Current classification

**PARTIALLY VALIDATED — implementation and integration evidence pending.**

### Practical implication

Implement one foundation task at a time. Do not mark V3.1/V3.2 complete until
native, multi-process and soak boundaries named by the roadmap are exercised.

## H-014 — V3 runtime contracts remain compatible with canonical owners

Status: PARTIALLY VALIDATED — REGISTERED BEFORE NATIVE GATES
Origin: V3.1–V3.4 implementation cycle
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

Evidence, event, resource, supervisor, recovery, routine, worker, memory,
session, replay, protocol and evaluation behavior can be added without
replacing Hermes SessionDB/Kanban/Memory/BrowserTask ownership or changing the
model tool schema.

### Observed evidence

- `python -m pytest workstation/tests -q -p no:cacheprovider`: **141 passed**;
- focused tests cover independent child-process restart, multi-process session
  lease rejection, stale evidence reconciliation, bounded event queues,
  durable worker queue/reconstruction, canonical journal delivery, Recovery
  Plane CLI quarantine, memory snapshots, side-effect-free replay and
  model-independent regression budgets;
- `python -m compileall -q workstation`: passed;
- all new durable projections use caller/HERMES_HOME paths and atomic replace;
- no new SessionDB, Kanban database, browser page store, scheduler or core
  model tool was introduced.

### Evidence still required

The Python contracts do not prove clean-machine installation, Electron/Desktop
composition, cross-engine browser behavior, or long-duration multi-process
soak. Those claims remain open until the native gates below complete.

### Practical implication

Treat the V3 modules as implementation-ready contracts, but do not promote the
V3.1–V3.4 roadmap headings to complete solely from unit/contract evidence.

## H-015 — Existing broad Windows Desktop red is outside the V3 Python delta

Status: OBSERVED — NO CAUSAL REGRESSION FOUND
Origin: native validation ladder
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Observed result

- `workstation\doctor.ps1`: passed environment, committed integration, lock and
  license checks;
- `apps\desktop\npm run typecheck`: passed with 0 errors after running with
  approved build-file access;
- `apps\desktop\npm test`: **7,389 passed, 36 failed, 5 skipped** across 717
  files. Failures are existing Windows path/permission/mode/SSH/WSL/locale and
  unrelated UI expectations; no changed V3 Python module is imported by those
  failing Desktop specs;
- versioned H010 native BrowserSessionState probe: **H010_CLASSIFICATION=VALIDATED**
  on Windows 10.0.26200 / Electron 40.10.2, including clean restart, failed
  write convergence, explicit destroy cleanup and abrupt restart.

### Classification

The broad Desktop red remains a pre-existing portability/test debt and is not
converted into a green claim. The native H010 result validates the existing
browser boundary only; it does not claim that Python V3 runtime contracts are
Electron-integrated.

### Practical implication

Keep KI-006 open for a separate baseline-controlled portability fix. V3
roadmap evidence must be reported by contract/native/soak boundary rather than
by the broad Desktop aggregate.

## H-016 — V3 process-boundary contracts pass the Windows smoke

Status: VALIDATED — CONTRACT BOUNDARY
Origin: V3.1/V3.2 acceptance probe
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Observed evidence

The versioned probe
`workstation/context/engineering-journal/probes/v3-runtime-hardening-smoke.py`
emitted:

- `V3_SUPERVISOR_START_PASS`;
- `V3_SUPERVISOR_RECOVERY_PASS` after terminating the child process;
- `V3_SESSION_OWNERSHIP_PASS` from a real second Python process;
- `V3_EVIDENCE_RECONCILIATION_PASS`;
- `V3_RUNTIME_HARDENING_CLASSIFICATION=VALIDATED_CONTRACT_BOUNDARY`.

### Boundary

This validates the independent supervisor, cross-process lease and evidence
state contracts on Windows. It does not substitute for Desktop/client wiring,
clean-machine release qualification, cross-engine browser testing or soak;
those remain explicit roadmap evidence gates.

### Preconditions closed

- PR #11 was accepted at exact head `d5be442...` after H010 emitted
  `H010_CLASSIFICATION=VALIDATED`; it merged as `e0a99ef3...`.
- PR #12 then merged that promoted main without rewriting either history. Its
  reconciled head `39e5178...` passed Workstation CI (26 contracts), exact
  Windows install/checkout-clean/diff/typecheck, 46 focused browser tests and
  the native H010 step; it merged as `4b04f4c...`.
- The broad Windows aggregator still reported the classified KI-006 baseline:
  one missing UI route mock and 31 unrelated POSIX/path/mode/SSH/platform
  failures. No BrowserSessionState/BrowserTask scoped test failed.

### G-001 — repository-wide disposition audit

**Hypothesis:** after #11/#12 promotion, no accepted product delta remains only
on a lateral branch; the remaining divergent refs are temporary diagnostics,
superseded source snapshots or reproducible formatter output.

**Confirming evidence required:** compare every GitHub branch to exact audit-base
main, inspect unique file/commit scope, close all superseded open PRs, and leave
zero material `NEEDS INVESTIGATION`.

**Observed result:**

- all ancestor refs classify `ALREADY ON MAIN`;
- PR #4/#5 diagnostic lines contain historical Windows evidence and obsolete
  snapshots only;
- PR #6 source was finalized/promoted by #7; its unique workflow is temporary;
- PR #8 is workflow-only historical validation;
- `bot/js-autofix` is one broad mechanical formatter commit and is
  `REJECTED/DO NOT PROMOTE` wholesale;
- comments carrying these dispositions were added and PRs #4/#5/#6/#8 were
  closed;
- no source/evidence needed for V1 #1.5 remains branch-only.

**Classification:** `VALIDATED`. The complete branch/PR table and checklist are
canonical in `../MAINLINE_CONSOLIDATION.md`.

### G-002 — canonical-document convergence

**Hypothesis:** promotion is incomplete while canonical documents still say
BrowserSessionState is a candidate or V1 #1 is next.

**Confirming evidence required:** update current state, decisions, constraints,
architecture/delta, roadmap, testing, known issues, patch manifest, README and
journal; protect ordering/disposition with executable contracts.

**Observed candidate result at `872cbc57ce919ca505650da4a7593fc73f4276bb`:**

- integration anchors, component lock and license policy passed;
- 29/29 Workstation contracts passed;
- Windows install/checkout-clean/diff/typecheck passed;
- BrowserSessionState lint/format, 5 files / 46 focused tests and native H010
  passed;
- local diff, integration, Markdown formatting and direct document-contract
  checks passed; the local canonical Python runner correctly refused because
  its `.venv` lacks pytest, so GitHub CI supplies that evidence.

**Classification:** `VALIDATED / READY TO PROMOTE`. V1 #1.5 may branch only
after this gate candidate is merged into `main`.

## H-017 — The committed Workstation installer preserves the working tree

Status: VALIDATED — DEPENDENCY-SKIPPING INSTALLATION GATE
Origin: V3.3 release-qualification boundary
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The dependency-skipping installation path validates committed integration,
component-lock and license gates, checks the Node toolchain, prepares runtime
directories and leaves repository source plus untracked checkout state
unchanged.

### Confirming evidence expected

- `workstation\install.ps1 -SkipDependencies` completes successfully;
- the installer’s source-cleanliness assertion reports no checkout mutation;
- the existing `workstation\doctor.ps1` remains green afterward.

### Refuting evidence expected

The installer mutates source or creates an unignored artifact, or one of the
committed integration/lock/license/toolchain checks fails on this workstation.

### Observed result

`powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File
workstation\install.ps1 -SkipDependencies` completed successfully. Integration,
component-lock, license and Node checks passed; runtime directories were
prepared under `%LOCALAPPDATA%\HermesWorkstation`; the installer reported that
committed source was not modified. Existing tracked edits and the untracked
user icon artifact were preserved.

### Additional full-install evidence

The normal `workstation\install.ps1` path also completed successfully:
editable Python installation, `npm ci` from the committed lockfile,
`hermes_cli` import and final checkout-clean assertion all passed. npm reported
13 dependency audit findings and deprecation warnings; no lockfile or source
change was made because automatic audit remediation would be an unreviewed
dependency change.

### Classification

`VALIDATED` for the local dependency-skipping and full-install paths. A truly
fresh machine/release artifact qualification, including packaging and runtime
launch, remains a separate gate.

## H-018 — The installed Desktop source produces a clean production build

Status: VALIDATED — LOCAL PRODUCTION BUILD GATE
Origin: V3.3 release-qualification boundary
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

With dependencies installed from the committed lockfile, the Desktop package
can execute its production build (renderer, bundled Electron main process and
native dependency staging) without changing tracked source or requiring an
uncommitted fallback.

### Confirming evidence expected

- `npm run build` completes in `apps\desktop`;
- the post-build assertion succeeds;
- tracked source remains unchanged after the build.

### Refuting evidence expected

The build fails, its post-build assertion is bypassed, or it changes tracked
source/assets unexpectedly.

### Observed result

`npm run build` completed successfully after a clean TypeScript build: Vite
transformed 15,142 modules, the Electron main/preload bundles were produced,
native `node-pty`/`get-windows` dependencies were staged, and
`assert-dist-built.mjs` passed. The build emitted the expected dirty-checkout
warning and Vite/deprecation notices, but no tracked source change.

### Classification

`VALIDATED` for the local production build path. Installer freshness, signed
release artifacts and runtime launch on a clean machine remain separate gates.

## H-019 — The packaged Desktop artifact is structurally launchable

Status: VALIDATED — LOCAL UNPACKED PACKAGING GATE
Origin: V3.3 release-qualification boundary
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The Desktop packaging pipeline can assemble an unpacked Windows artifact from
the built source without publishing or signing, and its output contains the
expected executable/runtime payload.

### Confirming evidence expected

- `npm run pack` completes with `--dir --publish never`;
- electron-builder emits an unpacked artifact and required native/runtime files;
- tracked source remains unchanged.

### Refuting evidence expected

Packaging fails, the artifact is incomplete, or packaging mutates tracked
source/assets unexpectedly.

### Observed result

`npm run pack` completed with electron-builder 26.15.3 for Windows x64 and
Electron 40.10.2. The unpacked artifact was assembled at
`apps\desktop\release\win-unpacked`, Electron integrity was updated, the
Hermes executable was stamped with the configured icon/identity, and signing
was explicitly skipped by the requested `--publish never`/directory mode.

### Classification

`VALIDATED` for unsigned local unpacked artifact assembly. Signed publication,
clean-machine installation and actual end-user launch remain separate gates.

## H-020 — The Windows NSIS release target can be assembled locally

Status: VALIDATED — LOCAL NSIS PACKAGING GATE
Origin: V3.3 release-qualification boundary
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The supported Windows NSIS target can be built from the current installed
dependencies without silently using a missing packaged runtime or changing
tracked source.

### Confirming evidence expected

- `npm run dist:win:nsis` completes;
- electron-builder emits a Windows installer artifact;
- the output is structurally consistent with the thin-installer policy and
  tracked source remains unchanged.

### Refuting evidence expected

The NSIS target fails, ships a stale/fat Hermes payload, or mutates tracked
source/assets unexpectedly.

### Observed result

`npm run dist:win:nsis` completed for Windows x64 and Electron 40.10.2,
producing `apps\desktop\release\Hermes-0.17.0-win-x64.exe` plus its blockmap.
The artifact validator then passed without rebuilding: the unpacked app,
thin-installer negative payload assertion, install stamp, renderer payload and
three `node-pty` native binaries were all present. The build used the
repository's explicit unsigned/local packaging mode.

### Classification

`VALIDATED` for local NSIS assembly and structural artifact validation. A
clean-machine install/launch, signed publication and cross-engine/long-soak
acceptance remain separate roadmap gates.

## H-021 — The packaged Desktop client launches and renders in isolation

Status: VALIDATED — PACKAGED GUI E2E GATE
Origin: V3.2 typed-resource/client evidence boundary
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The actual packaged Windows executable opens through the supported Electron
client path in a credential-free sandbox, renders the root UI and survives the
boot/HUD geometry smoke without requiring a provider or mutating user state.

### Confirming evidence expected

- `e2e/launch-packaged-app.spec.ts` passes against `release\win-unpacked\Hermes.exe`;
- the fixture uses isolated `HERMES_HOME`/userData and cleans up after the run;
- tracked source remains unchanged.

### Refuting evidence expected

The packaged client fails to launch/render, leaks into the user's normal state,
or the smoke requires an unavailable provider/credential.

### Observed result

The first failing runs were traced to the Workstation controller eagerly
restoring a detached `WebContentsView` on app startup. Electron exposed that
view as a page target without an owning `BrowserWindow`, so Playwright waited
forever for its frame tree while the real Hermes window was already loaded.
The controller now starts lazily, and the E2E fixture assigns a temporary
`HERMES_WORKSTATION_HOME` so global browser state cannot enter the test. The
packaged fixture also injects Playwright's Electron loader for its custom
executable path.

The final isolated run completed `5 passed (26.4s)`, including the Hermes title,
renderer DOM, HUD containment, boot state and screenshot artifact. The HUD
assertion retains the geometry checks and allows only a 1 px subpixel tolerance
for native Windows scaling.

### Classification

`VALIDATED / PROMOTED` for the packaged GUI E2E gate on this workstation. The
clean-machine release qualification, broader client wiring, cross-engine
coverage and production soak gates remain separate and are not implied by this
single packaged smoke.

## H-022 — BrowserTask resources can be consumed by multiple client surfaces

Status: PARTIALLY VALIDATED — CLIENT CONTRACT WIRED
Origin: V3.2 typed-resource/client evidence boundary
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

Desktop IPC, the Dashboard REST surface and the TUI gateway can consume one
versioned Workstation resource envelope derived from Electron BrowserTask/page
ownership and the canonical Execution Journal, without creating a second task
or journal store.

### Observed result

- Electron exposes `/resources` on the authenticated loopback controller and a
  matching `hermes:workstation-browser:resources` IPC method, plus bounded
  `/events` and `hermes:workstation-browser:events` projections;
- `BrowserView` consumes the IPC projection for its resource count, while the
  Dashboard `/workstation` page polls `/api/workstation/resources`;
- `workstation.resources` is registered as a read-only, pooled TUI JSON-RPC
  adapter over the same Python transport client, with `workstation.events`
  providing the bounded journal event view;
- task resources preserve BrowserTask/session/Kanban/run lineage, advertise
  control only when the controller is ready, and fail closed to `read` when
  evidence is unavailable;
- journal details are bounded to the latest 200 events in the primary resource
  projection; the existing task-journal inspection path remains the deep-log
  surface;
- the resource helper/runtime tests passed, Dashboard typecheck/lint passed,
  Python client tests passed, and the Desktop/Dashboard production builds
  completed;
- the rebuilt packaged Electron E2E completed `6 passed (28.3s)`, including the
  live IPC-versus-loopback resource identity comparison.

### Classification

`VALIDATED` for the implemented client contract and build boundaries. A live
two-surface session proving identical resources against one running task and a
long-duration soak remain open gates; cross-engine browser coverage is recorded
separately below.

## H-023 — Dashboard smoke passes in Chromium and Firefox

Status: VALIDATED — CROSS-ENGINE CLIENT SMOKE
Origin: V3.4 evaluation gate
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The built Dashboard served by the real Python backend can load the Workstation
resource page in both Chromium and Firefox without page-level errors, using an
isolated provider-free Hermes home.

### Observed result

- the first run exposed a missing Playwright Chromium headless-shell and an
  ambiguous shell/page heading selector; both were harness defects and were
  corrected without changing product behavior;
- after installing the matching local Playwright Chromium and Firefox
  runtimes, `npx playwright test -c playwright.dashboard.config.ts` completed
  **2 passed (9.3s)** — one test in each browser project;
- the backend was started with `hermes dashboard --skip-build` against a
  PID-qualified temporary `HERMES_HOME`, and global setup/teardown isolated the
  fixture from user state;
- the smoke covers the `/workstation` route, BrowserTask empty state and the
  absence of browser page exceptions. It does not claim Electron packaged
  behavior or Safari/WebKit support.

### Classification

`VALIDATED` for the initial Chromium/Firefox Dashboard smoke. Additional
supported engines and long-duration production-like soak remain separate gates.

## H-024 — Release evidence and canonical reconnect soak are executable

Status: PARTIALLY VALIDATED — LOCAL EVIDENCE RUNNER
Origin: V3.3/V3.4 release and evaluation gates
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

Release qualification and reconnect soak evidence can be collected through
bounded, read-only runners without treating a local dirty checkout as a clean
release or adding another SessionDB/worker/journal owner.

### Observed result

- `python -m workstation.release_qualification` records ordered stage evidence,
  bounds subprocess output/time and stops at the first missing or failed gate;
- the clean-install stage requires an accepted JSON report carrying the exact
  candidate `HEAD` revision, so the local runner cannot infer clean-machine
  evidence;
- the Windows Workstation workflow now performs a fresh install and Desktop
  production build, verifies the checkout stays clean, emits that evidence and
  feeds it back into the release qualification gate;
- `python -m workstation.soak --duration 10 --iterations 3 --sessions 3`
  completed **3/3 fresh-process iterations, 3 sessions, 0 failures, 36 journal
  events, 9 memory snapshots, 6 worker reconstructions, 6 model changes and 6
  cold reloads**, exercising session lease, model changes, migration, cold-session
  reload, memory snapshot/restore and worker stop/reconstruct lineage;
- `python -m workstation.soak --duration 60 --iterations 1000 --sessions 3`
  completed **77 fresh-process iterations, 0 failures, 1,155 actions, 924
  journal events, 26,796 memory records, 231 snapshots, 228 worker
  reconstructions, 228 model changes and 228 cold reloads**; `timed_out=true`
  records that the requested duration budget was reached, not a failed child;
- the complete Workstation suite after the client-parity addition completed
  **151 passed in 10.37s**.

### Classification

`VALIDATED` for bounded local evidence collection and contract-level
session/worker reconnect behavior. Clean-machine promotion and long-duration
Desktop/Browser production soak remain open acceptance gates.

## H-025 — HW-018 closes the KI-006 Windows portability class

Status: VALIDATED — BROAD WINDOWS SUITES GREEN
Origin: follow-up to H-015 baseline evidence
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Changed boundary

The implementation fixed the actual Desktop-owned portability assumptions
identified by H-015: deterministic UI locale formatting, session-scoped
Preview promotion, injected POSIX/Windows path grammars, platform-correct
permission assertions, native Git physical-path comparison and non-repository
probe behavior, explicit SSH mux/no-mux test contracts, non-blocking WSL UNC
selection and executable-mode staging seams.

### Confirming evidence

- Desktop UI: **591 files / 5,669 tests passed**;
- Desktop platform/Electron: **126 files / 1,760 tests passed, 5 skipped**;
- Desktop TypeScript typecheck: **0 errors**;
- no coverage was deleted or disabled to obtain the result.

### Classification

`KI-006=RESOLVED_ON_CURRENT_WORKING_TREE`. The old H-015 counts and the
Implementation 4 baseline comparison remain historical causality evidence;
the current broad suites no longer reproduce that failure class. V3 clean-
machine, deeper client-parity and long-duration production soak gates remain
independent and open.

## H-026 — Dashboard and TUI adapters share one authenticated resource boundary

Status: VALIDATED — HIGH-LEVEL CLIENT PARITY
Origin: V3.2 typed-resource/client evidence boundary
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The Dashboard REST handler and the TUI JSON-RPC handler can consume the same
versioned resource envelope for one BrowserTask/session/journal projection
through the shared authenticated controller client, without introducing a
client-owned persistence store.

### Observed result

- a real authenticated loopback HTTP controller boundary served one task,
  session and journal resource set;
- the Dashboard route and `workstation.resources` JSON-RPC method each issued
  one controller read and returned the exact same normalized envelope;
- the identity assertion covered the browser resource, task resource and
  execution-journal resource, including task/session lineage;
- `python -m pytest workstation/tests/test_client.py -q -p no:cacheprovider`
  completed **4 passed**, and the complete Workstation suite completed **151
  passed**.

### Classification

`VALIDATED` for the high-level Python client adapters and their live HTTP
transport boundary. Electron packaged two-surface rendering, future remote
clients and production-like soak remain separate gates; this evidence does
not claim those boundaries complete.

## H-027 — Dashboard and TUI adapters share bounded journal events

Status: VALIDATED — EVENT PROJECTION PARITY
Origin: V3.2 event/resource client evidence boundary
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

Dashboard REST, TUI JSON-RPC and Desktop IPC expose bounded operational journal
events from the canonical Electron Workstation runtime without creating a
second event store or leaking an unknown task's journal.

### Observed result

- the Electron runtime provides versioned `/events` responses with a maximum
  of 200 events and task-scoped filtering against known BrowserTasks;
- Desktop preload/IPC, Dashboard `/api/workstation/events` and TUI
  `workstation.events` are wired to that projection;
- the Dashboard renders the bounded recent-event list while the existing task
  journal remains the deep inspection surface;
- the real authenticated loopback adapter test compared both high-level
  surfaces and passed **5/5** client tests; the Electron runtime event test
  passed as part of **11/11** focused runtime/resource tests;
- the complete Workstation suite passed **152/152**, Desktop and web
  typechecks passed, the Dashboard production build passed, and the broad
  Desktop platform suite passed **1,761 tests with 5 skipped**.

### Classification

`VALIDATED` for the current Desktop IPC, Dashboard REST and TUI JSON-RPC event
projection. Remote-client expansion, packaged two-surface rendering and
production-like soak remain separate evidence gates.

## H-028 — Extended multi-session reconnect soak remains failure-free

Status: VALIDATED — EXTENDED LOCAL SOAK
Origin: V3.2/V3.4 session, worker and evaluation evidence gate
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The canonical Workstation reconnect scenario remains stable across a longer
local budget with multiple sessions, process restarts, model changes, cold
reloads, memory snapshots and persistent worker reconstruction.

### Observed result

- `python -m workstation.soak --duration 180 --iterations 3000 --sessions 4`
  completed **221 fresh-process iterations with 0 failures**;
- the run recorded **4,420 actions, 3,536 journal events, 391,170 memory
  records, 884 snapshots and 880 reconstructed workers** across four sessions;
- all four sessions were migrated, with 880 model changes and 880 cold reloads;
- the duration budget ended with the explicit expected marker `timed_out=true`;
- this strengthens local contract evidence but does not claim the separate
  production-like Desktop/Browser soak gate, clean-machine qualification or
  packaged two-surface rendering.

### Classification

`VALIDATED` for the extended local session/memory/worker/reconnect scenario;
external production-like and clean-machine evidence boundaries remain open.

## H-029 — Dashboard smoke passes in Chromium, Firefox and Edge

Status: VALIDATED — THREE-ENGINE DASHBOARD SMOKE
Origin: V3.4 cross-engine WebUI evaluation gate
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The provider-free Dashboard Workstation route renders without page-level errors
across the supported Chromium and Firefox projects and an installed Microsoft
Edge system browser, using one isolated Python backend.

### Observed result

- the Playwright configuration detects a supported Edge executable and adds an
  `msedge` project only when that browser is present;
- `npx playwright test --config=playwright.dashboard.config.ts` completed
  **3/3 tests in 9.2 seconds** across Chromium, Firefox and Edge;
- the smoke verified the Workstation heading, Browser tasks section, Recent
  events section and provider-free empty state, with no `pageerror` events;
- environments without a supported Edge installation continue to run the
  Chromium/Firefox baseline instead of failing due to an unavailable browser.

### Classification

`VALIDATED` for the current three-engine Dashboard smoke on this Windows host;
packaged Electron two-surface rendering and production-like Desktop/Browser
evidence remain separate gates.

## H-030 — Packaged artifact preserves resource and event identity across IPC

Status: VALIDATED — PACKAGED PROJECTION BOUNDARY
Origin: V3.2 typed resources/event client acceptance boundary
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The exact Windows `win-unpacked` Desktop artifact renders its packaged GUI/HUD
surfaces and exposes the same Workstation resource and event projections through
Desktop IPC and its authenticated loopback controller, without requiring a
visible native window.

### Observed result

- the packaged E2E fixture now sets `HERMES_DESKTOP_E2E_HEADLESS=1`; the main
  window remains hidden while its renderer, preload and controller initialize;
- the identity test compared schema/runtime, resource identities and event
  envelopes from IPC and `/resources` + `/events`;
- `npx playwright test e2e/launch-packaged-app.spec.ts --config=playwright.config.ts
  --workers=1` passed **6/6 in 22.5 seconds**, including renderer boot, HUD
  composer containment and screenshot capture;
- the run used the explicit headless E2E flag, so it validates packaged DOM,
  IPC, controller and geometry behavior but does not infer visible-window
  behavior on a user's desktop session.

### Classification

`VALIDATED` for the packaged GUI/HUD and IPC/controller boundary in headless
E2E; visible desktop-session behavior, clean-machine qualification and
production-like soak remain open evidence gates.

## H-031 — Windows gate tracks E2E harness changes

Status: VALIDATED — CI TRIGGER COVERAGE
Origin: V3.3 release qualification and V3.4 cross-engine evidence governance
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

Changes to the Desktop E2E harness and Playwright configurations cannot bypass
the Windows Workstation gate through the workflow path filter.

### Observed result

- `.github/workflows/workstation-browser-windows.yml` now includes
  `apps/desktop/e2e/**` and `apps/desktop/playwright*.config.ts` in its pull
  request paths;
- the canonical source contract test asserts both paths alongside the existing
  clean-install, build, UI and platform gate requirements;
- the bootstrap/context regression run passed **11/11**, and `git diff --check`
  passed.

### Classification

`VALIDATED` for CI trigger coverage of the current Desktop/Workstation E2E
harness; the workflow's external clean-machine execution remains a separate
release evidence gate.

## H-032 — Windows workflow executes browser and packaged evidence gates

Status: VALIDATED — RELEASE WORKFLOW COVERAGE
Origin: V3.3 upstream ownership/release qualification gate
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The Windows Workstation workflow exercises the cross-engine Dashboard smoke and
the packaged Desktop GUI boundary, rather than leaving those checks as local
manual commands.

### Observed result

- after clean-install evidence and the production build, the workflow installs
  the Chromium/Firefox Playwright browsers and runs the provider-free
  cross-engine config (Edge is auto-detected when present);
- it builds the unpacked Desktop artifact, runs `launch-packaged-app.spec.ts`
  with the fixture's headless mode, and uploads both E2E result directories;
- each new step has an explicit outcome ID and the final preservation step
  fails if cross-engine, packaging or packaged GUI E2E is not successful;
- the workflow path filter already includes the E2E/config files, and the
  canonical bootstrap/context regression run passed **11/11** with typecheck
  and `git diff --check` clean.

### Classification

`VALIDATED` for source-level CI coverage of the browser/release evidence gates;
the actual clean-machine workflow result still requires execution by the
external Windows runner on a candidate release.

## H-033 — Windows I/O tests tolerate legitimate Git/PowerShell startup cost

Status: VALIDATED — PLATFORM REGRESSION STABILITY
Origin: HW-018 Windows portability closure
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The broad Desktop platform suite remains green under concurrent Windows load;
tests that create Git clones/worktrees or invoke PowerShell have enough timeout
headroom to measure behavior instead of failing on the Vitest five-second
default.

### Observed result

- isolated reproductions of the two timeout sites passed within the normal
  path, while the broad run exposed their sensitivity under suite load;
- the Git worktree remote-tracking test and Windows hand-off-marker test now
  use explicit 15-second test budgets, preserving their assertions and cleanup;
- the full platform suite then passed **126 files / 1,761 tests**, with 5
  intentional skips; no coverage was deleted or disabled.

### Classification

`VALIDATED` as a Windows test-harness stability fix; the underlying Git,
PowerShell and cleanup behavior remains covered by the same assertions.

## H-034 — Release workflow command parity and full contract rerun

Status: VALIDATED — LOCAL RELEASE-CHECK PARITY
Origin: V3 release-evidence follow-up
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The newly wired Windows release-workflow commands remain executable against the
current checkout, and the complete Workstation contract suite is still green
after the workflow and Windows timeout-hardening changes.

### Observed result

- the exact workflow packaging command
  `npm run builder --workspace apps/desktop -- --dir --publish never` completed
  successfully without launching a visible Electron process;
- the canonical bootstrap/context contracts passed **11/11**;
- the release-qualification/isolation contracts passed **10/10** when run with
  the required native temporary-directory access;
- the complete Workstation suite passed **152/152**;
- `git diff --check --no-ext-diff --unified=0` passed and a final process check
  found no running Electron instance;
- an initial sandboxed isolation run reached 6 passing cases but four
  `tmp_path` fixtures failed before test execution with `WinError 5`; the
  authorized rerun reproduced all ten passes, so this is classified as an
  environment ACL limitation rather than a product or test-contract failure.

### Classification

`VALIDATED` for local command parity and Workstation contract integrity. The
external Windows clean-machine workflow and production-like long-duration soak
remain separate acceptance gates and are not inferred from this run.

## H-035 — Windows workflow retains bounded reconnect-soak evidence

Status: VALIDATED — RELEASE WORKFLOW COVERAGE
Origin: V3.2 long-lived runtime evidence gate
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The Windows release workflow now executes the canonical multi-session,
multi-process reconnect scenario and preserves enough structured output to
audit failures without converting a bounded contract run into a production
soak claim.

### Observed result

- the workflow invokes `workstation.soak` for four sessions with a 180-second
  duration budget and a 3,000-iteration cap;
- it validates zero failures/errors and requires at least one completed fresh
  process iteration before reporting success;
- the JSON report and durable soak root are uploaded as a separate artifact;
- the final outcome-preservation step requires the soak outcome to be
  `success`, alongside the existing UI/platform/browser/package gates;
- the canonical workflow source contract passed after updating its intentional
  aggregator-message expectation.

### Classification

`VALIDATED` for reproducible CI coverage of the bounded Workstation reconnect
contract. Clean-machine promotion, future-client parity and long-duration
production-like Desktop/Browser soak remain explicit external gates.

## H-036 — Release qualification failure no longer hides downstream evidence

Status: VALIDATED — DIAGNOSTIC NON-MASKING
Origin: V3 release-workflow hardening
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The Windows workflow can collect the soak, cross-engine and packaged GUI
results even when release qualification itself fails, while the final job
still fails on the original release outcome.

### Observed result

- the release qualification step now has an explicit `release_qualification`
  outcome and `continue-on-error: true`;
- the final preservation step reads that original outcome and requires
  `success`, so continuation is diagnostic only and cannot turn a red release
  gate green;
- the workflow source contract covers both the release outcome and the
  downstream soak outcome, and the documentation continues to distinguish
  bounded CI evidence from production acceptance.

### Classification

`VALIDATED` as failure-observability hardening. Required gates remain strict;
the change only prevents an early release-gate failure from masking later
evidence.

## H-037 — Operational memory compaction bounds reconnect soak growth

Status: VALIDATED — MEMORY RETENTION CONTRACT
Origin: V3.2/V3.4 long-lived runtime acceptance
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The canonical memory owner can compact operational task-context records by an
explicit workspace/type policy, persist that deletion safely, and expose a
bounded live-memory metric in the reconnect soak without pruning procedures or
unrelated memory kinds.

### Observed result

- `ProceduralMemory.compact_memory()` retains the newest records within an
  explicit scope and carries deletion IDs through merge-on-write so compacted
  records are not resurrected by the persistence merge path;
- focused memory/soak tests passed **6/6**, and the complete Workstation suite
  passed **154/154**;
- the final 180-second, four-session soak completed **128 fresh-process
  iterations with 0 failures and 0 errors**, including 512 snapshots, 508
  worker reconstructions/model changes/cold reloads and 2,048 journal events;
- cumulative memory work produced 15,888 records, while live memory peaked at
  **32 records**, exactly eight per session, proving the operational bound;
- the soak still reports `timed_out=true` at the duration budget and remains
  contract evidence rather than a production-like Desktop/Browser claim.

### Classification

`VALIDATED` for explicit operational-memory retention and bounded reconnect
soak behavior. User-facing procedures/facts remain untouched unless a caller
explicitly selects them for compaction; clean-machine promotion, future-client
parity and production-like Desktop/Browser soak remain external gates.

## H-038 — Windows soak gate enforces the operational memory bound

Status: VALIDATED — RELEASE GATE INVARIANT
Origin: V3.2/V3.4 bounded-memory follow-up
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The Windows release workflow does not merely check that the reconnect soak had
no failures; it also rejects reports that omit or exceed the explicit
eight-record-per-session operational memory bound.

### Observed result

- the soak validation calculates `memoryBound = session_count * 8` from the
  report and requires `max_live_memory_records` to be present and no greater
  than that bound;
- the source contract asserts the report metric and bound check, while the
  final workflow aggregator still requires the soak step's original outcome
  to be `success`;
- the final 180-second soak report contained `max_live_memory_records=32` for
  four sessions, satisfying the same invariant that CI now enforces.

### Classification

`VALIDATED` as a fail-closed release-workflow invariant. The check bounds
operational task context only; it does not imply a production-like
Desktop/Browser soak or alter the separate clean-machine evidence boundary.

## H-039 — Strict doctor closes the bootstrap gate without opening Desktop

Status: VALIDATED — WINDOWS BOOTSTRAP GATE
Origin: V1 #14 clean-install/start acceptance
Date / ref: 2026-09-11 / working tree on `main@d77901a685`

### Claim

The Windows release workflow now exercises a real strict bootstrap diagnostic
after installation, and a failed required dependency/integration check cannot
be reported as accepted clean-install evidence.

### Observed result

- `workstation\\doctor.ps1 -Strict` preserves the default warning-oriented
  diagnostic mode but returns exit code 1 when required Git/Node/npm/Python,
  `.venv`, integration, lock or license checks fail;
- the local strict doctor passed on the current checkout with Python 3.13,
  Node 26, npm, `.venv`, integration anchors, lock and license policy;
- the Windows workflow invokes `workstation\\doctor.cmd -Strict`, carries its
  original outcome into clean-install evidence and requires that outcome in
  the final aggregator;
- the doctor source contract and workflow/documentation contracts pass, and no
  Electron process is launched by this validation.

### Classification

`VALIDATED` for strict bootstrap diagnostics and failure propagation. The
one-click launcher still owns the user-facing install → doctor → start flow;
external clean-machine and visible Desktop-session acceptance remain separate
evidence gates.

## H-040 — Root launcher changes trigger the Windows gate

Status: VALIDATED — CI PATH COVERAGE
Origin: V1 #14 one-click bootstrap hardening
Date / ref: 2026-09-12 / working tree on `main@d77901a685`

### Claim

A change to the repository-root one-click Workstation launcher cannot bypass
the Windows validation workflow merely because the launcher is outside the
`workstation/` directory.

### Observed result

- `.github/workflows/workstation-browser-windows.yml` now includes
  `START-HERMES-WORKSTATION.bat` in its pull-request path filter;
- the canonical source test checks that filter alongside the launcher’s strict
  doctor delegation;
- the launcher remains source-only in this validation, so no visible Electron
  process is started.

### Classification

`VALIDATED` for CI trigger coverage of the one-click bootstrap entrypoint.
Actual clean-machine launcher execution and visible Desktop-session behavior
remain environment-specific acceptance gates.

## Why this journal exists

This file is the Workstation project's durable anti-repeat memory for active engineering. It records not only experiments, but the **mistakes, rejected premises, evidence boundaries, and decision lineage** that a future agent must know before proposing another implementation.

The evidence hierarchy is:

1. current repository code/tests on the exact ref being changed;
2. versioned executable evidence tied to an exact SHA;
3. current GitHub CI/workflow results for that SHA;
4. canonical Workstation documents (`DECISIONS`, `CONSTRAINTS`, `CURRENT_STATE`, `TESTING`, `KNOWN_ISSUES`, `UPSTREAM_DELTA`, etc.);
5. historical session exports, manual observations, screenshots, and attached context;
6. model/agent narrative.

A lower layer may explain or motivate an investigation, but it must not override a higher layer. In particular, **an assistant saying that a tool/runtime was used is not evidence that it was actually used**; inspect the tool call/result, runtime boundary, code path, or executable marker.

Historical chats are useful because they preserve why a decision changed. They are not permission to resurrect an older architecture after a later decision became canonical.

## Durable project lineage — do not reopen settled branches casually

The project did not start with the current architecture. The early question was which external browser stack should be integrated with Hermes. `agent-browser`, Browser Use, BrowserOS, Browser4, VibeSurf, Sabrina, Hermes Browser Extension, Hermes WebUI, and other approaches were investigated at different times.

Several intermediate architectures were reasonable hypotheses at the time:

- `agent-browser` as the primary Hermes browser engine;
- BrowserOS as the dedicated persistent/logged-in browser;
- a separate `hermes-workstation` repository;
- plugin-first Workstation integration;
- Hermes + separate extension + separate WebUI + launcher;
- a Workstation monorepo wrapping an otherwise untouched Hermes.

Those were **decision stages, not current open choices**.

The settled product direction is now:

- this repository is a **thin downstream distribution/fork of Hermes**, with Workstation first-class;
- upstream Hermes remains authoritative for generic Hermes behavior;
- `workstation/` is the main downstream-owned architecture surface;
- core/upstream-owned changes are allowed only when deep integration requires them, and the delta must stay small, explicit, tested, and tracked;
- Hermes Sessions, Gateway, Kanban, Memory, approvals, profiles, tool registry, and routing remain the source of truth instead of being duplicated;
- the primary Workstation Browser is **Electron Chromium inside Hermes Desktop**, rendered with `WebContentsView`, with a dedicated persistent profile/session;
- `BrowserRuntime` remains an abstraction boundary so specialist/external runtimes can exist without redefining BrowserTask semantics;
- `BrowserTask` is the semantic identity/ownership unit for durable browser work, with at most one live task page per task in a process;
- Preview, Chat Browser View, and Browser Hub must ultimately become views/adapters over the same BrowserTask/runtime rather than independently navigated pages;
- external projects are reused selectively as code, protocol, benchmark, fallback, or design reference according to `SOURCE_MATRIX.md`; they are not automatically vendored into the product.

### Decisions that require new material evidence before reopening

| Old debate                             | Current settled direction                                  | What would justify reopening it                                                                                      |
| -------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| plugin/separate repo vs fork           | thin downstream fork; Workstation first-class              | a concrete upstream capability that removes the need for cross-cutting downstream integration, with migration proof  |
| external browser vs internal browser   | internal Electron Chromium is primary                      | measured inability of Electron runtime to satisfy a required invariant that a specialist runtime demonstrably solves |
| BrowserOS vs agent-browser as the core | neither is the primary Workstation runtime                 | changed product requirements or benchmark evidence, not familiarity/preference                                       |
| separate WebUI                         | official Hermes Desktop/Dashboard remain UI/state surfaces | a specific unmet product requirement that cannot be added without duplicating state/control planes                   |
| second Kanban/SessionDB/Memory         | reuse Hermes-owned systems                                 | only an explicit replacement architecture that supersedes the canonical decisions and includes migration/tests       |
| synchronize Preview and Browser by URL | one BrowserTask/live page, views/hosts                     | never as a cosmetic synchronization workaround; only a replacement of the ownership model with stronger proof        |
| browser availability from process env  | session/platform capability                                | only if Hermes changes its gateway/session architecture materially and tests prove the new identity model            |

The rule is not “never change architecture.” A replacement decision must identify the material new evidence, name the decision it supersedes, define migration, and prove the new behavior.

## Cross-track findings imported from prior investigations

### CT-001 — Preview is not proof of the Workstation Browser

A real Desktop smoke was once reported as “Workstation Browser interno” after the agent called `open_preview` and `read_preview`. That was a **false-positive identity claim**.

What that test actually proved:

- the Desktop Preview surface could navigate/render/read the page;
- Electron/Preview was functioning.

What it did **not** prove:

- `browser_navigate` existed in the session schema;
- `browser_snapshot` existed in the session schema;
- the Workstation `BrowserRuntime` handled the action;
- a BrowserTask was created/bound;
- Preview and Browser were the same page/runtime.

**Anti-repeat rule:** acceptance criteria that name a tool/runtime must be proven by the **exact tool invocation and execution boundary**, not by visually similar output. A successful substitute is evidence for the substitute only.

### CT-002 — Real Desktop tool exposure can fail before BrowserTask

A real-environment investigation established this before-state:

- the Capabilities UI showed **Browser Automation = ON**;
- `coding_context = auto`;
- `agent.disabled_toolsets = []`;
- persisted `platform_toolsets.cli` did **not** contain `browser`;
- effective `_get_platform_tools(..., "cli")` did **not** resolve `browser`;
- a genuinely new Desktop session did not receive `browser_navigate` or `browser_snapshot`.

This refuted the tempting hypothesis that `coding_context=focus` was simply stripping Browser in that environment.

The proven boundary was:

```text
Capabilities UI
  → toolset API/configuration
  → profile/config persistence
  → toolset resolution
  → Desktop session schema
  → browser_* execution
  → BrowserTask/runtime
```

**Anti-repeat rule:** do not modify BrowserTask lifecycle/persistence to fix a failure that occurs before the model receives `browser_*` tools.

### CT-003 — Automated lifecycle tests and native lifecycle evidence are different claims

Passing pure or mocked tests did not prove a real `WebContentsView`, native Windows process restart, or renderer identity behavior.

- pure tests prove lifecycle logic/serialization;
- mocked runtime tests prove adapter contracts;
- build/typecheck proves compilation;
- process launch proves launch;
- H-004 native probe proves the actual Windows/Electron/Chromium lifecycle contract.

**Anti-repeat rule:** never upgrade a lower-layer green result into a stronger claim than that layer can observe.

### CT-004 — CI must test committed source, not an auto-repaired checkout

An earlier Workstation install path could mutate/repair tracked source before validation, creating a false-green risk.

Durable policy:

- committed downstream source is canonical;
- normal install validates integration read-only;
- runtime state lives outside tracked source where appropriate;
- install must not dirty the checkout;
- migration/rebase helpers are separate from normal validation.

### CT-005 — A red broad Windows gate needs causality, not storytelling

Durable method:

1. reproduce exact base and candidate;
2. same OS/toolchain/dependency install/commands;
3. compare failure **signatures/classes**, not only counts;
4. keep scoped outcomes independently visible;
5. preserve final red when any required broad suite is red;
6. never delete/allowlist coverage to manufacture green.

### CT-006 — GUI/browser surface identity belongs to the session

Implementation 3 established:

- GUI/Desktop surface identity is session/platform state, not a process-global env proxy;
- controller reachability is execution state, not surface existence;
- tool-definition caching must include session-surface capability to avoid Desktop↔TUI leakage.

### CT-007 — Old BrowserClaw experiments are historical motivation, not the current architecture

Earlier BrowserClaw experiments inform requirements/failure modes; they are not permission to reintroduce BrowserClaw as the primary Workstation browser after the internal-runtime decision.

## Long-term product north star — governed by the roadmap

The broader direction is Hermes as a persistent execution/orchestration layer
for recurring work. V1 #1.5 now authorizes only the narrow, executable slices
listed in the roadmap; it is not permission to claim their later hardening scope
or to create parallel owners.

## WP-01 / BrowserSessionState objective — resolved and promoted

Complete the canonical BrowserSessionState foundation beyond BrowserTask-only
metadata while preserving the current live-page owners and avoiding a second
page store, SessionDB, Kanban, or control plane.

Operational base verification on 2026-08-30:

- the initially checked local `main` ref was stale at
  `ce78f120e8ed2974d6174e475cc7572afcfe41e0`;
- after fetching, `origin/main` resolved to the packet base
  `46a6ef9e257b4add01d6eb7f2a95a82bb433ee89`;
- branch `wp/codex/browser-session-state-core` was created directly from that
  verified remote ref without moving local `main`.

Scope boundary: ordinary logical tabs, order, active logical/generic tab, safe
URL/title metadata, BrowserTask relationship, available identity linkage, and
explicit recovery metadata/policy. Chat Browser View, Browser Hub, Preview
unification, host transfer, SessionDB/Gateway semantics, Kanban implementation,
LAN/Tailscale, Browser Memory, and KI-007 remain out of scope.

### Historical candidate acceptance status (superseded by final promotion)

- Implementer-focused validation on candidate
  `9ce769ee54ab6a02cad77a266c87cb05a8cad3f6` passed the scoped
  BrowserSessionState/BrowserTask tests, Desktop typecheck and Workstation
  contracts recorded below.
- Independent verification **blocked** that candidate because arbitrary raw
  page titles could cross the durable boundary and because the possible
  `new browserTasks + previous sanitized tab projection` crash snapshots had
  not been exercised through runtime restart/recovery.
- At that checkpoint the writer-branch work was a corrective candidate, not an
  accepted or promoted implementation.
- Independent re-verification and native Electron restart acceptance were still
  pending at that checkpoint; H010 and the final PR #11 promotion below
  supersede those temporary blockers.

### H-007 — BrowserSessionState can be a safe structural projection of existing authorities

Origin: WP-01 requires restart-safe logical browser state beyond the promoted
BrowserTask metadata, but forbids a second live-page owner.

Hypothesis:

- `entries`, `taskTabs`, and `activeTabId` remain authoritative for live
  process-local pages, task bindings, ordering, and active page;
- `BrowserTaskLifecycle` remains authoritative for logical task lifecycle and
  task identity/linkage metadata;
- the Chromium profile remains authoritative for browser-managed site/auth
  state;
- a versioned BrowserSessionState file can persist only a sanitized structural
  projection of those owners and restore logical intent without serializing or
  owning `WebContentsView`, `WebContents`, renderer heap, or process identity.

Experiment (registered before code inspection):

1. trace every mutation/read of `entries`, `taskTabs`, `activeTabId`, and
   `BrowserTaskLifecycle` on exact `origin/main`;
2. inspect current persistence, recovery, and BrowserTask regression tests;
3. classify each field as process-, BrowserTask-, BrowserSessionState-,
   Hermes-session-, or Chromium-profile-scoped;
4. derive the smallest persistence/migration seam that writes atomically and
   reconciles through the existing runtime owners;
5. define adversarial URL/title inputs whose credentials, secret markers, or
   sensitive content must fail closed before entering structural JSON.

Confirming evidence:

- no existing durable ordinary-tab owner exists;
- one snapshot can be built from current owners and restored through their
  existing mutation paths;
- task-owned tabs reconcile by `taskTabs`/`ownerTaskId` without duplicate pages;
- corrupt/unknown versions are ignored safely and atomic replacement preserves
  the last valid snapshot.

Refuting/reformulating evidence:

- another durable owner already exists for any proposed field;
- restoration would require a second `WebContentsView` map or independent task
  binding;
- raw URL/title persistence cannot be bounded by an explicit safe-metadata
  policy;
- any required change crosses SessionDB, Gateway, Kanban, Preview, or host/UI
  ownership.

Classification: **VALIDATED**.

Observed authority trace on exact base `46a6ef9e257b4add01d6eb7f2a95a82bb433ee89`:

- `WorkstationBrowserRuntime.entries` is the sole process-local live-page map.
  Its `Map` insertion order is the current tab order, and each `BrowserEntry`
  exclusively owns one `WebContentsView` plus live loading/crash state.
- `taskTabs` is the task id → live tab id index; `BrowserEntry.ownerTaskId` is
  the reciprocal ownership marker. `createTab(..., ownerTaskId)` and
  `rawEntryForTask()` reconcile stale/destroyed mappings through those two
  primitives and never allocate a second live page for the same task.
- `activeTabId` is the sole process-local physical active-tab pointer. Attach,
  navigation and UI state derive from it; it is cleared when its entry is
  discarded or destroyed.
- `BrowserTaskLifecycle.tasks` owns logical BrowserTask status, recovery and
  available linkage (`panelHost`, `controlHost`, `sessionHost`,
  `localConnection`, `leaseState`). Its version-1 `browser-tasks.json` is the
  only durable state on the base, and it deliberately contains no page URL,
  title, renderer object, or typed/page secret.
- `session.fromPath(workstationBrowserProfilePath(), { cache: true })` owns the
  dedicated Chromium profile: cookies, localStorage, IndexedDB, cache and
  compatible browser authentication. It is not BrowserSessionState.
- request `session_id` reaches the Desktop controller, but the base runtime does
  not currently mutate SessionDB/Gateway semantics with it. WP-01 therefore
  preserves available BrowserTask linkage without inventing missing
  SessionDB/run/Kanban authority.

Scope classification:

- process-scoped: `WebContentsView`, `WebContents`, renderer heap/process
  identity, `entries`, `taskTabs`, physical `activeTabId`, loading/crash flags,
  attach/bounds/control-server handles;
- BrowserTask-scoped: task id, lifecycle status/parked state, task recovery,
  task timestamps and available host/session/connection/lease linkage;
- BrowserSessionState-scoped: logical tab ids, ordinary/task relationship,
  structural order, active logical/generic tab, sanitized URL/title metadata,
  and explicit page-recreation policy/status;
- Hermes-session-scoped: the externally supplied Hermes session identity and
  its SessionDB/Gateway lineage remain owned outside this runtime; WP-01 only
  preserves an already-available linkage string inside BrowserTask metadata;
- Chromium-profile-scoped: site storage, cookies, cache, browser auth and other
  Chromium-managed data under the dedicated profile path.

Minimal persistence/migration conclusion:

1. introduce one versioned `browser-session.json` containing sanitized logical
   tab state plus the existing BrowserTask snapshot;
2. expose a BrowserTask persistence adapter over that same atomic file, so
   `BrowserTaskLifecycle` remains the in-memory task authority and no second
   BrowserTask JSON source remains active;
3. import valid version-1 `browser-tasks.json` only when
   `browser-session.json` is absent, atomically commit the composite state, then
   remove the legacy file; never fall back to stale legacy data when a new file
   exists but is corrupt or from an unknown version;
4. restore ordinary tabs by recreating new process objects in persisted order,
   while BrowserTask tabs remain metadata-only recovery hints until the promoted
   lifecycle lazily recreates exactly one page;
5. derive every subsequent snapshot from current runtime/lifecycle owners;
   temporary restart hints contain structural metadata only and never own a
   page object.

Security conclusion:

- URL metadata is safe only after an explicit allowlist/sanitization step:
  `about:blank` or HTTP(S), no userinfo, no query/fragment, bounded length, and
  rejection of credential/secret/session markers or opaque token-like path
  material;
- **Historical title conclusion — SUPERSEDED BY H-009 FOR DURABLE TITLES:**
  title metadata was considered safe after bounded normalization and rejection
  of controls, URLs/email-like data, secret markers, assignments and opaque
  token-like material;
- unsafe URL/title values become `null`/blank recovery metadata; raw values are
  never copied into structural JSON.

Current canonical durable-title rule: **Arbitrary page-controlled titles never
cross BrowserSessionState persistence; `safeTitle` is always `null`. Live
`WebContents` titles remain available in-process.**

Practical implication: H-007 supports the scoped implementation; it does not
justify changes to SessionDB, Gateway, Kanban, Preview, UI hosts, LAN, or KI-007.

### H-008 — Composite persistence and owner-driven reconciliation satisfy WP-01

Hypothesis:

- a standalone, pure BrowserSessionState serializer/persistence module plus a
  narrow runtime adapter can cover ordinary tab/order/active restoration,
  BrowserTask coexistence, safe metadata and restart reconciliation without
  changing any current live-page or lifecycle authority;
- adversarial serialization and runtime restart tests can prove that secret
  material and Electron process objects never enter the JSON and that task
  pages remain singular/lazy.

Confirming evidence: focused pure/runtime tests cover ordinary/order/active,
safe URL/title, unknown/corrupt version, atomic replacement failure, legacy
migration, BrowserTask coexistence/no duplicates, restart, secret exclusion and
stale/crashed reconciliation while all promoted BrowserTask regressions remain
green.

Refuting evidence: duplicate task pages, eager task-page resurrection, loss of
ordinary order/active state, raw secret markers in JSON, two independently
writable task files after migration, or any required cross-scope change.

Classification: **ACTIVE — REGISTERED BEFORE PRODUCT CHANGE**.

Product boundary marker: product edits are limited to the Electron runtime,
BrowserTask persistence seam, a dedicated BrowserSessionState module/tests, and
the branch-local engineering journal.

Historical policy qualifier: H-008 ran before the independent durable-title
blocker and H-009 correction. Its references to safe-title restoration or
preservation record the then-current candidate behavior, not the canonical
durable-title contract. The experimental outcomes below remain chronological
evidence and are not rewritten.

#### H-008 experiment 1 — focused pure/runtime gate

Command:

`npm run test:desktop:platforms --workspace apps/desktop -- electron/workstation-browser-session-state.test.ts electron/workstation-browser-task.test.ts electron/workstation-browser-runtime-task.test.ts`

Sandbox result: runner bootstrap failed before tests with `spawn EPERM`; this was
a sandbox subprocess restriction, not product evidence. The identical command
was rerun with approved process execution.

Executed result: **24 passed / 4 failed across 3 files**; the new pure
BrowserSessionState file and promoted pure BrowserTask file both passed. All
four failures were isolated to the runtime adapter:

1. three restored safe-title assertions under the historical pre-H-009 policy
   received `New Tab` because the fake WebContents emits `did-navigate` before
   any page title and `updateEntrySafeMetadata()` replaced the persisted safe
   title with `null`;
2. same-process crashed BrowserTask recovery reused the pending logical tab id,
   while the promoted regression requires a visibly new replacement tab/page id
   after a crash (`assert.notEqual(owned[0].id, firstTab.id)`).

Classification: **PARTIAL / CORRECTIVE**. H-008's composite persistence premise
is not refuted; the pure security/version/migration/atomic contracts passed.
The runtime reconciliation needs two narrow corrections.

Material correction registered before change:

- under the historical pre-H-009 policy, retain a previously safe title while
  navigation has no non-empty title, but still clear/reject it when a non-empty
  unsafe title arrives;
- distinguish restart recovery (`restored` hint may retain its logical tab id)
  from a same-process stale/crashed page (`stale` hint must allocate a new tab
  id while reusing only sanitized URL/title metadata and preserving order).

#### H-008 experiment 2 — focused gate after corrective changes

Identical approved command result: **3 test files passed / 28 tests passed / 0
failed**.

Covered outcomes:

- ordinary tab persistence, ordering and active logical tab restoration;
- historical pre-H-009 safe URL/title preservation and adversarial
  userinfo/query/fragment, secret-marker, opaque-token, URL/email-title and
  process-object exclusion;
- unknown/corrupt version fail-closed behavior;
- atomic replacement failure preserving the previous valid snapshot and
  cleaning the temp file;
- one-shot legacy BrowserTask migration into the composite file;
- BrowserTask/ordinary coexistence, lazy restart recovery and no duplicate task
  pages;
- unexpected stale ordinary-page reconciliation;
- all promoted pure/runtime BrowserTask regression tests.

Classification: **VALIDATED AT FOCUSED PURE/RUNTIME LAYER**. Next evidence
boundary: Desktop TypeScript typecheck/lint and the broader Electron platform
suite; no native Windows identity claim is added by this mocked runtime gate.

#### H-008 experiment 3 — Desktop typecheck, first pass

Command: `npm run typecheck --workspace apps/desktop`.

Result: **failed with one TypeScript error** at
`workstation-browser-session-state.ts`: the optional `browserTaskId` property
was correctly runtime-validated but TypeScript did not preserve narrowing across
a second property access before `.trim()` (`TS2339: Property 'trim' does not
exist on type 'unknown'`).

Classification: **STATIC NARROWING DEFECT**. Registered correction: capture the
unknown property once in a local, validate that local, then trim the narrowed
value. No runtime/persistence behavior changes.

First correction result: **still failed with the same TS2339**. Capturing the
property was insufficient because the `null | undefined | unknown` union was
reintroduced by the ternary expression. Materially changed correction: use an
explicit `if (rawBrowserTaskId === null) ... else { guard; trim }` branch so the
custom type predicate narrows inside one control-flow block.

Second correction result: the BrowserTask id occurrence narrowed successfully;
typecheck then reported the same TS2339 class at the sibling optional
`activeTabId` ternary. Registered correction: apply the same explicit guarded
branch to `activeTabId` before another full rerun.

Final typecheck result: **PASS / exit 0** for all three Desktop TypeScript
configs (`tsc -p .`, `tsconfig.electron.json`, and `tsconfig.e2e.json`, all
`--noEmit`).

Classification: **STATIC CONTRACT VALIDATED**.

Formatting check result: Prettier reported the four modified TS/test files plus
unchanged `workstation-browser-task.test.ts`. Classification: **EXPECTED LOCAL
FORMAT CORRECTION + PRE-EXISTING UNTOUCHED DRIFT**. Only modified files will be
formatted; the promoted untouched regression file will not receive unrelated
churn.

Focused ESLint first result: **423 problems (147 errors / 276 warnings)** across
the two modified legacy runtime files plus the two new files. The output shows
the legacy runtime/test already violate the current global `curly` and padding
rules throughout untouched lines; auto-fixing those files would create broad
out-of-scope churn. The new files also contain fixable import-order,
curly/padding and one `no-control-regex` issue.

Classification: **MIXED — PRE-EXISTING LEGACY LINT DEBT + NEW-FILE LINT**.
Registered action: make both new files independently ESLint-clean, then retain
typecheck/tests as the executable contract for the narrowly modified legacy
files rather than mass-rewriting them.

New-file lint/format result after scoped fixes:

- `npx eslint electron/workstation-browser-session-state.ts electron/workstation-browser-session-state.test.ts`: **PASS / 0 errors / 0 warnings**;
- `npx prettier --check` for the same files: **PASS**.

The `no-control-regex` occurrence was replaced with explicit code-point checks
covering C0 plus DEL/C1 controls; the security policy is unchanged.

#### H-008 experiment 4 — post-format focused/typecheck rerun

- focused Electron gate: **3 files passed / 28 tests passed / 0 failed**;
- Desktop typecheck: **PASS / exit 0** across renderer, Electron and E2E TS
  configs.

#### H-008 experiment 5 — full Electron/platform suite

Command: `npm run test:desktop:platforms --workspace apps/desktop`.

Final rerun after the active-logical-tab adversary: **12 files failed / 109
passed / 1 skipped; 33 tests failed / 1694 passed / 5 skipped (1732 total)**.

The three WP-01/BrowserTask files passed inside the full run. Every reported
failure is outside the changed subsystem and matches a documented KI-006
Windows class: POSIX permission-bit assertions, Darwin staging mode, Windows
8.3/realpath normalization, SSH ControlPath/Include assumptions, WSL probe
timeouts, symlink privileges, Git temp cleanup/timeouts, and PowerShell handoff
timing. No BrowserSessionState/BrowserTask failure appeared.

Classification: **KI-006 BROAD RED / WP-01 NON-REGRESSIVE AT SCOPED GATE**.
The broad command remains correctly reported as exit 1; no test was disabled,
weakened or allowlisted.

#### H-008 experiment 6 — Workstation Python contracts

Required wrapper: `scripts/run_tests.sh workstation/tests`.

Harness resolution evidence:

1. WSL Bash was denied inside the sandbox (`CreateInstance/E_ACCESSDENIED`);
2. approved WSL execution then found no pytest-capable WSL venv;
3. the Windows `.venv` also lacked pytest;
4. a system Windows Python had pytest 9.0.2, but passing it through WSL produced
   the invalid mixed path `C:\\mnt\\c\\...` before collection;
5. Git Bash provided the correct MSYS→Windows path translation; its first
   sandboxed launch was denied a signal pipe, then the identical approved
   wrapper run executed normally.

Final exact wrapper result: **4 files / 24 tests passed / 0 failed** in 5.9s.
This includes context-document contracts, bootstrap canonical-source checks,
Workstation contracts and browser routing/fail-closed tests.

Classification: **WORKSTATION CONTRACTS VALIDATED**. The failed preliminary
attempts were harness/environment failures before test collection, not product
failures.

#### H-008 experiment 7 — interleaved ordinary/task ordering adversary

The BrowserTask coexistence test was strengthened to persist order
`ordinary A → lazy task T → ordinary B` and assert `A → T → B` after T's lazy
page recreation.

First result: **27 passed / 1 failed**. T was recreated once with correct id,
URL, title and ownership, but appeared as `A → B → T`.

Root cause evidence: `restoreSessionTabs()` establishes the full restored order
before iterating, but creation of A called `reconcileRestoredEntryOrder()` while
the loop had not yet registered T as pending. Seeing zero pending hints, the
helper cleared the restored order prematurely.

Classification: **VALID ORDERING DEFECT / H-008 CORRECTIVE**. Registered minimal
fix: never clear the restored-order hint while the outer session restoration is
active; after the complete loop, retain it only while a lazy/stale structural
hint remains.

Corrective result: **3 files / 28 tests passed / 0 failed**. The strengthened
coexistence case now restores `A → T → B`, T is still created lazily through
`BrowserTaskLifecycle`, and its persisted logical id maps to exactly one live
page. A final forward-version adversary also proved that an existing
`BrowserSessionState.version > 1` is neither interpreted nor overwritten and
never triggers fallback to the stale legacy BrowserTask file; malformed/current
version state still starts from a fresh sanitized projection.

Classification: **H-008 VALIDATED**. The ordering hint is transient process
coordination only; `entries`, `taskTabs` and `BrowserEntry.ownerTaskId` remain
the live page/ownership authorities, while the composite file remains the only
restart projection.

#### H-008 experiment 8 — active logical BrowserTask without eager page resurrection

Hypothesis: reusing only process-authoritative `activeTabId` during restoration
would lose a persisted active BrowserTask because its physical page must remain
lazy. A task-only state would also need a generic live fallback page without
changing the structural active selection.

Experiment: make a task tab active, persist/restart, assert that no task page is
alive before `showTask()`, inspect the composite `activeTabId`, then show the
task twice and inspect id/ownership/page count.

Evidence: the first audit found that the physical fallback could overwrite the
structural selection. The corrective retains one transient
`restoredLogicalActiveTabId` only while its lazy tab hint exists. It is not a
page store: `activeTabId` still owns the physical active `entries` member, and
any ordinary activation or task materialization consumes the hint.

Final focused result: **3 files / 29 tests passed / 0 failed**. Before
`showTask()`, zero task pages exist and `browser-session.json.activeTabId`
retains the task's logical tab id. After `showTask()`, that same id maps to one
live page and becomes the physical `activeTabId`; the second show remains
idempotent.

Final static/contract results:

- full Desktop TypeScript typecheck (renderer + Electron + E2E): **passed**;
- ESLint for both new BrowserSessionState files: **0 errors / 0 warnings**;
- Prettier check for both new BrowserSessionState files: **passed**;
- final Workstation wrapper rerun: **4 files / 24 tests passed / 0 failed** in 4.9s.

Historical classification at that candidate: **IMPLEMENTER-FOCUSED VALIDATION
ONLY / INDEPENDENTLY BLOCKED**. These scoped results remain useful evidence, but they did not prove
the durable-title boundary or material composite-write interruption states.
Native Electron restart was still pending and promotion was not authorized at
that point; the later H010 final result supersedes this state.

### H-009 — Durable-title denial and executable composite recovery can correct the candidate narrowly

Origin: independent verification of candidate `9ce769ee...` demonstrated that
`safeTitleMetadata("Recovery code 482913")` returned the raw recovery code and
correctly identified that one BrowserTask operation can durably publish new
BrowserTask metadata before the final tabs/active projection.

Hypothesis:

- durable BrowserSessionState can set every page-controlled title to `null`
  without weakening the live `WebContents.getTitle()` surface;
- URL persistence can retain its conservative structural policy while failing
  closed for explicit recovery/verification/OTP/PIN/magic/one-time credential
  path semantics;
- injecting failure through the existing BrowserSessionState filesystem seam
  can capture the C1 create, C2 recreate/show and C3 destroy intermediate files
  through the real `WorkstationBrowserRuntime` integration;
- each possible `new browserTasks + previous sanitized tab projection`
  snapshot will restart deterministically, preserve at-most-one ownership,
  remain secret-free and converge to one canonical composite file.

Confirming evidence:

- every raw title, including harmless display text and numeric/customer labels,
  serializes as `safeTitle: null` while the live runtime still reports its real
  page title;
- credential-bearing URL query/fragment/userinfo/JWT/signed material and the
  explicit authentication path classes never appear in serialized or reloaded
  state;
- C1/C2/C3 restart assertions prove logical task uniqueness, lazy page
  recreation, orphan removal, deterministic active/order reconciliation and a
  canonical final snapshot.

Refuting evidence:

- any raw page title or forbidden credential material appears in JSON or after
  reload;
- a recovered task is duplicated, eagerly materialized, rebound to two pages,
  resurrected after destroy, or cannot converge after a faulted projection;
- proving recovery requires a second state owner or transaction framework.

Classification: **ACTIVE — REGISTERED BEFORE CORRECTIVE PRODUCT CHANGE**.

Product boundary marker: edits remain limited to the existing
BrowserSessionState sanitizer/persistence seam, its runtime injection point,
focused pure/runtime regressions and this journal. Native Electron restart is
still a later acceptance layer, not implied by the mocked Electron runtime
fault tests.

#### H-009 experiment 1 — security boundary and C1/C2/C3 focused gate

Command:

`npm run test:desktop:platforms --workspace apps/desktop -- electron/workstation-browser-session-state.test.ts electron/workstation-browser-task.test.ts electron/workstation-browser-runtime-task.test.ts`

The first sandboxed launch failed before config load with `spawn EPERM`; the
identical approved command then executed normally.

Result: **3 files / 33 tests passed / 0 failed**.

Observed executable evidence:

- every page-controlled title, including the five credential examples,
  harmless display text and a customer-number label, became `safeTitle: null`;
- live fake-WebContents titles remained visible before process teardown, while
  restored titles fell back to `New Tab`;
- access-token query, OAuth code, fragment token, userinfo/password, JWT and
  presigned credential material were absent from serialized and reloaded JSON;
- explicit recovery/verification/OTP/temporary-PIN/magic-login/one-time path
  classes failed closed, while `/customers/482913` remained restorable;
- C1 captured new BrowserTask T plus the previous sanitized ordinary-tab
  projection, then restored T parked/lazy and materialized exactly one page;
- C2 captured recreated/visible BrowserTask metadata plus the previous lazy
  sanitized tab/active projection, then restarted parked/lazy and converged to
  one task/page binding with deterministic order and active selection;
- C3 captured durable task removal before the final runtime projection, pruned
  the orphan relationship during normalization, refused later `showTask(T)` and
  reconciled the remaining ordinary tab as active.

Classification: **VALIDATED AT FOCUSED PURE/MOCKED-RUNTIME LAYER**. The
possible `new browserTasks + previous sanitized tab projection` intermediate is
explicitly accepted and proven recoverable; it is not described as impossible.
Next evidence boundary: formatting/static checks, full focused
`workstation-browser` tests, Desktop typecheck and Workstation contracts.

#### H-009 experiment 2 — required corrective validation matrix

Final scoped JavaScript results after formatting:

- BrowserSessionState focused: **1 file / 9 tests passed / 0 failed**;
- BrowserTask pure: **1 file / 10 tests passed / 0 failed**;
- Browser runtime-task: **1 file / 14 tests passed / 0 failed**;
- all focused `workstation-browser` tests together: **3 files / 33 tests
  passed / 0 failed**;
- targeted adversarial security regression: **1 passed / 8 skipped** under the
  title filter;
- targeted C1/C2/C3 regressions: **3 passed / 11 skipped** under the crash
  boundary filter.

Static results:

- complete Desktop typecheck (`tsc -p .`, Electron and E2E configs): **PASS /
  exit 0**;
- ESLint for the two dedicated BrowserSessionState files: **0 errors / 0
  warnings**;
- Prettier check for the two dedicated BrowserSessionState files: **PASS**.

Workstation Python contract result:

- the first exact wrapper attempt correctly refused to run because the local
  `.venv` lacks pytest;
- rerunning the same repository wrapper with `HERMES_PYTHON` pointing to the
  installed pytest-capable Python produced **4 files / 24 tests passed / 0
  failed** in 6.6s;
- context-document contracts passed with the corrected candidate/non-promotion
  terminology.

Historical classification at that candidate: **CORRECTIVE CANDIDATE VALIDATED
AT REQUESTED AUTOMATED LAYERS / NATIVE ELECTRON RESTART PENDING / PROMOTION NOT
AUTHORIZED**.
The delivery commit freezes the exact Git head for independent verification;
this journal does not treat that delivery action as promotion or native
acceptance.

### H-010 — The native profile-cookie failure is a probe lifetime defect

Origin: the `Workstation Browser Windows` run for PR #11 head
`9177d5a1ebab23d9ce3e5fe9664afbb7fdd43ec5` reached the H010 product path.
Phase A passed structural persistence, safe metadata and BrowserTask ordering,
but phase B failed with `phaseB Chromium profile cookie missing`.

The same exact-head run also exposed two independent static issues: Prettier
reported the two resilience test files, while the focused BrowserSessionState
suite still passed 5 files / 46 tests. Those formatting failures are mechanical
and are not evidence about runtime behavior.

Hypothesis registered before changing the probe:

- H010 creates `h010_profile_cookie` without `expirationDate`; Electron therefore
  treats it as a session cookie, which is not required to survive termination of
  the Chromium session;
- the dedicated `session.fromPath(...)` profile may be healthy even though the
  probe incorrectly asks a session cookie to prove durable profile persistence;
- setting an explicit future `expirationDate`, verifying the cookie immediately
  in phase A, flushing storage, and retaining the existing two-process phase-B
  assertion will test the intended persistent-profile boundary.

Confirming evidence:

- phase A observes the explicitly persistent cookie before shutdown;
- phase B observes the same cookie from the same Workstation profile path in a
  different Electron process;
- the cookie value remains absent from `browser-session.json`;
- all other H010 phases continue to pass on the exact corrected SHA.

Refuting/reformulating evidence:

- an explicitly persistent cookie is visible in phase A but missing in phase B;
- the two processes resolve different Workstation browser profile paths;
- a product shutdown/session-path defect, rather than cookie lifetime, is needed
  to explain the failure.

Local correction evidence after the change:

- the PR branch was first reconciled with `main` at
  `ce448d829e95112fd08b21535c7a8426ee866035`, preserving the one-click
  launcher before any promotion attempt;
- the five-file Browser foundation selection passed **5 files / 46 tests / 0
  failed**;
- complete Desktop typecheck passed;
- exact BrowserSessionState ESLint selection passed with **0 errors / 0
  warnings** after mechanical layout cleanup;
- exact BrowserSessionState Prettier selection passed;
- `git diff --check` and committed Workstation integration validation passed;
- this Linux environment cannot supply the repository's pytest-capable Python
  offline, so the unchanged Python contracts were not represented as a fresh
  local run. Their existing exact-branch evidence remains recorded above and
  GitHub CI must independently rerun them.

Classification: **LOCAL CORRECTION VALIDATED / EXACT-SHA WINDOWS H010 PENDING**.
Product boundary marker: this correction changes only the versioned native probe
and formatting-only test layout. It does not change BrowserSessionState or
BrowserTask runtime semantics. A new exact-SHA Windows run is mandatory because
the native probe itself changes.

Exact-SHA Windows result for `b5910874b27a4bfb4b45485a6438adbfd72cdcfb`:

- checkout identity, clean install, checkout-clean assertion, diff check,
  typecheck, exact ESLint/Prettier and the 5-file / 46-test focused suite passed;
- phase A observed the explicitly persistent cookie and completed;
- phase B, in a different Electron process, observed the same cookie from the
  same dedicated profile, retained it outside `browser-session.json`, restored
  ordinary/task ordering and lazily recreated exactly one task page;
- the original `phaseB Chromium profile cookie missing` failure is therefore
  **resolved**, validating the session-cookie lifetime diagnosis;
- H010 then advanced into its independent failed-write/destroy phase. Failed
  write convergence passed, but the immediate post-`destroyTask` assertion saw
  the old native `WebContents` before Electron emitted `destroyed`.

Reclassification: **COOKIE HYPOTHESIS VALIDATED / NEW NATIVE TEARDOWN TIMING
BOUNDARY ISOLATED AS H-011**.

### H-011 — Native `WebContents.close()` completion is asynchronous

Origin: H010 at exact head `b5910874...` failed with
`destroy fault mode prior WebContents survived` immediately after the runtime
had removed the BrowserTask and page entry in process.

Hypothesis registered before changing product or probe:

- `discardEntry()` synchronously detaches the view, clears `taskTabs`, removes
  the `entries` owner and calls Electron `webContents.close()`;
- real Electron completes `close()` asynchronously and emits `destroyed`, while
  the focused fake marks itself destroyed synchronously;
- the native probe's immediate `isDestroyed()` assertion therefore conflates
  synchronous logical-owner removal with asynchronous Chromium teardown;
- the correct native contract is: immediately no runtime/task owner remains,
  then the old `WebContents` reaches `destroyed` within a bounded wait before
  task-id reuse and fresh-page assertions continue.

Confirming evidence:

- immediately after the failed persistence write, `listTasks()` omits the task
  and `state().tabs` omits its old tab;
- the captured native `WebContents` emits/reaches `destroyed` within a short
  bounded wait;
- subsequent durable convergence and same-id recreation produce one fresh,
  blank page with a different tab id;
- the rest of H010, including abrupt-process restart, remains green.

Refuting/reformulating evidence:

- the runtime still exposes the old task/tab after `destroyTask` returns;
- native destruction does not complete within the bound;
- same-id recreation overlaps the old renderer or inherits its URL;
- a product-side lifecycle change is required to prevent continued page work.

Classification: **ACTIVE — NATIVE TIMING HYPOTHESIS REGISTERED BEFORE CHANGE**.
Product boundary marker: first correct the probe to observe Electron's
documented `destroyed` event/state with a bounded wait. Do not weaken immediate
assertions about logical task/tab ownership or fresh recreation.

Native result at exact head `ad905ddee902efc7d66db76220f44e963030843b`:

- immediate logical task/tab removal passed;
- bounded native `WebContents` teardown passed;
- durable convergence and same-id fresh recreation passed;
- H-011 is therefore **VALIDATED**;
- the probe then exposed a separate harness exit-code issue in abrupt phase A,
  isolated below as H-012.

### H-012 — Electron `process.exit()` return overwrote the abrupt exit code

Origin: H010 at exact head `ad905dde...` emitted
`H010_ABRUPT_PHASE1_DURABLE` and then unexpectedly emitted
`H010_MODE_PASS abrupt1`; the parent observed exit code `0` instead of the
allowed sentinel `17`.

Hypothesis registered before changing the probe:

- Electron's Windows main-process `process.exit(17)` shim initiated exit but
  returned control to the async ready callback;
- the common success tail then executed `app.exit(0)`, replacing the intended
  abrupt sentinel with a normal success code;
- persisted BrowserSessionState had already passed its pre-termination checks,
  so this is an orchestration/exit-code defect after the product boundary;
- selecting the final exit code exactly once through `app.exit(...)`, without
  calling `runtime.destroy()`, should preserve the abrupt-state experiment.

Confirming evidence:

- abrupt phase A exits with code `17` after its durable marker;
- abrupt phase B starts under a different PID, restores the ordinary logical
  tab and parked BrowserTask, lazily creates exactly one task page, and passes;
- full H010 reaches `H010_CLASSIFICATION=VALIDATED`.

Refuting/reformulating evidence:

- phase A still exits `0`, times out, or performs the runtime's clean destroy;
- phase B cannot restore the durable task/tab projection;
- a product persistence change is required before phase B can pass.

Final exact-SHA result at `d5be442021ea0c744351622317eef5212219786d`:

- abrupt phase A emitted `H010_ABRUPT_PHASE1_DURABLE` and exited with the
  allowed non-zero sentinel;
- abrupt phase B ran under a distinct PID and emitted
  `H010_ABRUPT_RESTART_PASS`;
- clean restart, profile separation, failed-write convergence and
  explicit-destroy cleanup remained green;
- the probe emitted `H010_CLASSIFICATION=VALIDATED`.

Classification: **VALIDATED / HARNESS DEFECT CORRECTED**. No
BrowserSessionState product change was required for H-012. PR #11 then promoted
the exact head as merge `e0a99ef3aba6e6d2b65c30cf3c908ee1d49c4d29`.

## Implementation 4 objective — closed

Required native lifecycle contract:

`create/navigate → hide or park → re-expose without replacement navigation → explicit destroy → real process restart → logical restore/recovery`

Final conclusion: **validated and promoted to `main`**.

## Evidence ledger

### H-001 — BrowserTask lifecycle caused the V9 native-smoke timeout

Classification: **REFUTED**.

Evidence: V9 printed `HARNESS_BOOT` but never `HARNESS_READY`; product runtime import had not occurred.

### H-002 — Electron 40.10.2 / Windows cannot reach `app.ready`

Classification: **REFUTED**.

Evidence: bare Electron readiness probe reached ready and created a BrowserWindow with exit code 0.

### H-003 — V9 readiness stall is caused by top-level `await app.whenReady()` in its ESM main path

Classification: **VALIDATED**.

Experiment: `probes/h003-esm-ready.mjs`.

Evidence:

- TLA case: boot → internal timeout; exit 3;
- `.then(...)` case: boot → ready → BrowserWindow → PASS; exit 0.

Conclusion: V9 was a harness bootstrap defect. Product runtime already uses non-blocking readiness registration.

### H-004 — Real BrowserTask lifecycle satisfies the Implementation 4 acceptance contract

Classification: **VALIDATED**.

Experiment: `probes/h004-native-browser-task-smoke.mjs` at `d8acc752133b125b9619cbc7fe09199f1283a22b`.

Live identity evidence:

- task `impl4-h004-live-task`;
- tab id stayed `a27236b9-4aaf-4adc-9556-7ee14f5c4274`;
- real `webContentsId` stayed `3`;
- owner page count stayed `1`;
- URL and renderer sentinel survived hide/show and park/show;
- hide produced logical `hidden` without destroying page;
- park produced logical `parked` without destroying page.

Explicit destroy evidence:

- `destroyTask` returned true;
- prior WebContents destroyed;
- task not listed;
- zero remaining task-owned tabs;
- no automatic replacement page.

Real restart evidence:

- process 1 PID `30968`: task persisted `parked` / `fresh`, structural state excluded page URL secret and renderer secret;
- process 2 PID `37440`: same task restored `parked` / `restored`, zero eager pages before show, first show created exactly one task page under same task id, recovery became `recreated`.

Practical conclusion: **Implementation 4 native lifecycle acceptance behavior is proven.**

### H-005 — Documentation-closure Workstation CI red represented a BrowserTask/product regression

Classification: **REFUTED AS PRODUCT REGRESSION / VALIDATED AS DOCUMENTATION CONTRACT REGRESSION**.

Evidence:

- `core-patch-dry-run` passed;
- 23/24 Workstation contract tests passed;
- sole failure was `test_context_separates_current_state_from_target_and_known_issues`;
- fingerprint was missing literal heading `## Not implemented yet` after a documentation rewrite combined tested sections.

Correction:

- separate canonical headings restored;
- test was not weakened.

### H-006 — Final-head broad Windows red introduced a new Implementation 4 failure class

Classification: **REFUTED BY CONTROLLED EQUIVALENCE**.

Controlled native-Windows A/B:

- baseline `ce78f120e8ed2974d6174e475cc7572afcfe41e0`;
- candidate `2ffee2335b6aba071e7b63457a047cd9334d4d92`;
- result `WINDOWS_BASELINE_COMPARISON=PASS_WITH_KI-006_RED`;
- candidate-specific BrowserTask tests: 2 files / 16 tests passed;
- candidate had fewer legacy failures than baseline and every remaining failure was identical to baseline or a variant of the same KI-006 causal class.

Final accepted head `75d10d35d4757496390debf8e4b4f9efb44c5432` differed from `2ffee...` only by contributor-attribution mapping and Workstation journal material. No BrowserTask product/runtime/probe/workflow/dependency code changed. On that head, committed integration, install, checkout-clean, typecheck and BrowserTask focused steps passed; only the broad aggregator remained red.

Classification used at promotion: `KI-006_ONLY_BY_CONTROLLED_EQUIVALENCE`.

## Experiment / failure ledger

| ID    | Attempt / fingerprint                                                | What happened                                                                                  | Classification                      | Anti-repeat lesson                                                                                                                               |
| ----- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| E-001 | PowerShell interpolation with `$code:` / `$ExpectedBranch:`          | ParserError before test                                                                        | Harness defect                      | Use `${name}:` when `:` follows an interpolated PowerShell variable.                                                                             |
| E-002 | Assume Electron/esbuild at root `.bin`                               | Dependency discovery failed despite `npm ci`                                                   | Harness defect                      | Inspect workspace ownership before hard-coding executable paths.                                                                                 |
| E-003 | PowerShell parameter `$Args`                                         | Arguments swallowed; tools printed usage                                                       | Harness defect                      | Never shadow automatic `$Args`.                                                                                                                  |
| E-004 | native stderr + `$ErrorActionPreference='Stop'`                      | Normal esbuild stderr became `NativeCommandError`                                              | Harness defect                      | stderr is not failure; gate on exit status.                                                                                                      |
| E-005 | `Start-Process` exit code on Windows PowerShell 5.1                  | successful run exposed unusable/null exit status                                               | Harness defect                      | Prefer Node `child_process` for native orchestration.                                                                                            |
| E-006 | arbitrary bundled `.mjs` as Electron target                          | launch did not prove valid app-entry semantics                                                 | Harness defect                      | Use a valid Electron app directory.                                                                                                              |
| E-007 | V9 + top-level `await app.whenReady()`                               | boot marker printed; ready marker never printed                                                | Harness defect                      | Never attribute pre-runtime timeout to BrowserTask.                                                                                              |
| E-008 | H-002 bare CommonJS readiness                                        | ready + BrowserWindow succeeded                                                                | Control evidence                    | General Electron/Windows startup is healthy.                                                                                                     |
| E-009 | H-003 ESM paired control                                             | TLA fails; `.then(...)` passes                                                                 | Root-cause evidence                 | Register readiness non-blockingly on this Windows/Electron path.                                                                                 |
| E-010 | H-004 real BrowserTask lifecycle                                     | live identity, explicit destroy, two-process restart, lazy recovery, secret isolation all pass | Acceptance evidence                 | Reuse the versioned probe; do not reconstruct ad hoc runners.                                                                                    |
| E-011 | Preview described as Workstation Browser validation                  | visually successful page was wrong lane                                                        | False-positive validation           | Exact tool/runtime boundary, not visual similarity, proves identity.                                                                             |
| E-012 | exact `browser_navigate` + `browser_snapshot` requested while absent | test correctly stopped rather than substituting                                                | Correct fail-closed behavior        | Capability absence is a result; do not substitute and call it passed.                                                                            |
| E-013 | coding focus blamed for Browser absence                              | environment showed `coding_context=auto`                                                       | Refuted hypothesis                  | Remove plausible explanations after direct contradiction.                                                                                        |
| E-014 | Capabilities UI ON but resolved CLI toolsets lacked `browser`        | new Desktop session lacked tools                                                               | Cross-layer discrepancy             | Trace UI → API → persistence → resolution → schema before changing runtime.                                                                      |
| E-015 | unit/adapter lifecycle tests treated as native smoke                 | mocks could not prove native restart/page identity                                             | Evidence-boundary error             | Match claim to observing layer; H-004 later supplied native proof.                                                                               |
| E-016 | installer repaired source before validation                          | mutated checkout could hide missing committed integration                                      | Resolved validation-design defect   | Test committed tree; keep repair/migration separate.                                                                                             |
| E-017 | broad Windows red interpreted without exact baseline                 | base/candidate shared failure classes                                                          | Causality lesson                    | Controlled A/B + signatures; baseline-equivalent red remains red.                                                                                |
| E-018 | GUI capability tied to process env/reachability/cache                | Desktop surface could disappear/leak across session types                                      | Resolved ownership-model defect     | Session-scoped surface identity; runtime reachability is execution state.                                                                        |
| E-019 | documentation rewrite removed tested canonical heading               | Workstation CI failed 1/24 although product code unchanged                                     | Documentation contract regression   | Inspect context-doc tests before restructuring canonical docs; documentation-only commits still require CI.                                      |
| E-020 | contributor email unmapped at final promotion                        | repository-wide attribution check failed                                                       | Process gate, corrected             | Merge hygiene is part of promotion; fix the mapping instead of dismissing/bypassing the gate.                                                    |
| E-021 | post-merge canonical docs still said Impl4 was pending               | code/GitHub and `CURRENT_STATE`/`ROADMAP` disagreed after merge                                | Post-promotion documentation defect | Promotion is not complete until canonical state documents reflect the new `main`; correct via a separate docs closure and run context contracts. |
| E-022 | Windows broad Desktop portability debt                               | Historical UI/platform failures reduced to 0 after contract-aware fixes and full reruns        | KI-006 resolved                      | Preserve platform-specific behavior explicitly; never hide a broad failure by deleting or weakening coverage.                                    |

## Stable anti-patterns / rules learned

1. Do not change product code because a validation harness failed before reaching the product boundary.
2. Instrument explicit markers at process boot, Electron ready, product import and each lifecycle milestone.
3. Prefer one orchestration layer with explicit timeout/exit semantics; on Windows use Node `child_process`, not stacked PowerShell wrappers.
4. Compare against a smallest known-good control and vary one material factor at a time.
5. Build success proves compilation only; process launch proves launch only; require behavior evidence.
6. Do not use top-level `await app.whenReady()` in native Workstation validation harnesses on the current Windows/Electron target.
7. Never repeat an experiment unless the materially changed input/assumption is recorded here first.
8. Never call `close-tab` or UI X equivalent to `destroyTask`; test explicit task destruction directly.
9. Do not use Task Manager PID disappearance as the canonical destroy invariant; use WebContents destruction + ownership/state evidence.
10. Once a reusable regression probe exists, improve that probe rather than creating an untracked replacement unless necessary.
11. Carry native smoke to a later documentation-only SHA only after Git proves relevant product/probe code unchanged.
12. Do not infer tool/runtime identity from screen location; inspect actual invocation/route/runtime identifiers.
13. Do not substitute Preview/web search/other browser lane when acceptance explicitly requires Workstation `browser_*`.
14. Never confuse Preview with Workstation Browser while separate lanes remain.
15. Locate the first broken boundary before fixing downstream components.
16. Label statements observed fact / hypothesis / inference / proven cause; reclassify when contradicted.
17. Typecheck, unit, mocked runtime, native smoke, and full E2E are distinct evidence classes.
18. A UI toggle is not proof backend configuration persisted.
19. Session, process, BrowserTask, profile, and renderer state have different owners/lifetimes; name owner before storing/caching/restoring.
20. Do not create parallel SessionDB/Kanban/Memory/browser state/control planes.
21. Do not synchronize duplicate browser pages by URL to imitate a shared BrowserTask; fix ownership.
22. Profile persistence and BrowserSessionState are different; renderer object identity never survives process restart.
23. Bound BrowserTasks fail closed; do not silently switch to a different stateful runtime.
24. Never allow install/CI to auto-heal tracked source before validation.
25. Broad pre-existing red requires controlled baseline/candidate evidence and remains red until fixed.
26. Keep diagnostic suites independent enough to collect evidence, while preserving final failure when required suites fail.
27. Prefer extending existing primitives over creating duplicate state owners.
28. Do not reopen superseded architecture debates without material new evidence.
29. One implementation/hypothesis cycle at a time: reproduce → isolate → smallest complete change → focused validation → correction → regressions → exact-SHA gates → docs → next.
30. In TDD corrective work, RED fails for intended reason; GREEN is smallest fix; REFACTOR only after green.
31. A checkpoint is memory, not a stop condition.
32. Before claiming a full Workstation browser path, separately prove configuration persistence, session schema exposure, exact tool execution, BrowserTask behavior, host/view unification, restart recovery, and profile persistence.
33. Canonical documentation is part of the tested product contract. Before renaming/removing required headings, fields, tables, or markers, inspect `workstation/tests` for structural assertions and run the relevant contracts after the edit.
34. After a promotion merge, audit state-bearing documents against the new `main`; pre-merge wording such as “candidate”, “pending”, or an old main SHA becomes a real consistency defect once the merge lands.

## Promotion closure

Implementation 4 promotion sequence is complete:

1. H-004 real Windows/Electron lifecycle accepted;
2. controlled Windows baseline comparison classified remaining broad red as KI-006;
3. final PR head frozen at `75d10d35d4757496390debf8e4b4f9efb44c5432` with no product/runtime/probe change after causal validation;
4. exact-final-head Workstation CI, Docker, contributor attribution, install/typecheck/focused BrowserTask gates observed;
5. PR #9 marked ready;
6. merge executed with expected head SHA;
7. PR verified merged and `main` verified at `fada723f43613e5e0f061cab24445573ac298998`;
8. merge parents verified: `ce78f120...` + `75d10d35...`;
9. canonical state/roadmap/known-issues/testing/journal closure is being reconciled in `docs/impl4-promotion-closure` because the pre-merge wording became stale only after promotion.

No WP-01 / BrowserSessionState candidate work belongs in this historical
Implementation 4 closure branch.

## Continuous update protocol

Before an experiment/change:

- register hypothesis/experiment ID when causal uncertainty exists;
- state confirming/refuting evidence;
- identify the product boundary marker;
- search this journal for the same fingerprint/premise/already-refuted hypothesis;
- inspect relevant executable contracts, including documentation contracts, before restructuring tested surfaces;
- state what material input changed if repeating a prior approach.

Immediately after:

- record exact output/error fingerprint;
- classify the hypothesis (`VALIDATED`, `PARTIAL`, `REFORMULATED`, `REFUTED`, `INCONCLUSIVE`);
- record practical implication;
- identify the next materially relevant action;
- promote stable product truth into canonical docs instead of leaving it only here.

No experiment is complete until this file is updated. A checkpoint is never permission to stop; it is memory for the next action.
