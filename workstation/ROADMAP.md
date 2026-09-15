# Workstation roadmap

## Foundation — on current `main`

- `workstation/` architecture, contracts, policies and upstream tracking.
- first-class `/browser` route in Hermes Desktop.
- internal Electron Chromium Browser runtime with dedicated persistent profile.
- multiple tabs, navigation, attach/detach, background survival and cache-only maintenance.
- pause/resume, focus, take/release control primitives.
- loopback-only bearer-authenticated Browser controller in Hermes Desktop.
- `browser_*` integration that prefers internal Chromium and fails closed after task binding.
- Kanban Desktop enabled by default in this downstream distribution.
- stable/edge channel metadata.
- CI skeleton, lock/license validation and Windows Browser E2E workflow.
- eval matrix prepared for internal/agent-browser/browser-exec.
- Desktop Browser schema capability is session-scoped and protected from process-global cache/env leakage.
- first-class BrowserTask lifecycle promoted by PR #9: `create` / `show` / `hide` / `park` / explicit `destroy`, crash recovery, safe logical persistence and restart restoration.
- `taskTabs` + `BrowserEntry.ownerTaskId` remain the authoritative live Chromium task→page ownership primitives; no second page store was introduced.
- one `taskId` is idempotently bound to at most one live task page in a Desktop process.
- hide/park/show preserve the same live page and current URL while the process remains alive.
- restart restores safe BrowserTask metadata as parked and lazily recreates exactly one page under the same logical task when needed.
- composite BrowserSessionState now persists ordinary/task logical tabs, order,
  active selection, sanitized restorable metadata and BrowserTask linkage through
  one atomic state authority.
- clean and abrupt two-process restart, profile/state separation and failed-write
  convergence are validated by H010 on the accepted PR #11 head.
- repository-root one-click dogfood startup now performs install → doctor →
  start without installing dependencies twice.

## Implementation 4 — promoted

PR #9 (`feat(workstation): formalize BrowserTask lifecycle`) was promoted to `main` in merge commit `fada723f43613e5e0f061cab24445573ac298998` from accepted head `75d10d35d4757496390debf8e4b4f9efb44c5432`.

Acceptance evidence:

- focused BrowserTask lifecycle/runtime tests passed;
- real Windows/Electron H-004 smoke validated same-page hide/park re-exposure, explicit destroy, two-process logical restart recovery, exactly-one-page ownership and structural secret isolation;
- controlled Windows baseline comparison returned `WINDOWS_BASELINE_COMPARISON=PASS_WITH_KI-006_RED` and found no Implementation 4 regression class;
- exact-final-head Workstation CI and Docker gates passed; the broad Windows aggregator remained red only for the known KI-006 baseline debt.

The following work remains intentionally outside Implementation 4.

## Delivery strategy — MVP first, architectural hardening second

From V1 #1 onward the roadmap has two complementary delivery modes:

1. **Canonical milestone hardening** — each numbered milestone remains responsible for its rigorous architecture, ownership model, security boundary, recovery semantics, migrations, regression coverage and native evidence.
2. **Integrated MVP dogfood** — after V1 #1 is completed, one intermediate milestone (`1.5`) implements a deliberately narrow but real vertical slice of every later roadmap capability so the Workstation can be used continuously before every subsystem receives full hardening.

The MVP track is **not** permission to create throwaway architecture. Every MVP slice must:

- reuse the canonical Hermes/Workstation owner for state and control;
- avoid second SessionDB/Kanban/Memory/browser stores or duplicate live pages;
- preserve BrowserTask one-live-page and bound-task fail-closed invariants;
- prefer a narrow working path over a broad fake/stub path;
- keep experimental V1.1/V2 slices opt-in when they are not yet suitable for the default path;
- leave the original milestone open for later robustness work rather than marking it complete merely because its MVP exists;
- include at least one behavior contract or dogfood scenario strong enough to prove that the slice is actually usable.

The intended cadence becomes:

```text
V1 #1 BrowserSessionState — rigorous completion
        ↓
V1 #1.5 Integrated Dogfood MVP — minimum viable slice of the whole roadmap
        ↓
continuous real usage / dogfooding
        ↓
V1 #2, #3, #4... — revisit each original milestone for architectural hardening
        ↓
V1.1 and V2 — harden the experimental slices already exercised during dogfood
```

## V1 #1 — BrowserSessionState — promoted

PR #11 accepted `d5be442021ea0c744351622317eef5212219786d` and was
promoted as merge `e0a99ef3aba6e6d2b65c30cf3c908ee1d49c4d29`.

The milestone delivered the composite safe structural state, atomic convergence,
ordinary/task tab order and active selection, safe URL/title recovery boundaries,
BrowserTask coexistence, legacy migration and explicit clean/abrupt restart
semantics. H010 emitted `H010_CLASSIFICATION=VALIDATED` on the exact accepted
Windows/Electron head.

Controller/session/run/Kanban linkage beyond the identifiers already present in
BrowserTask remains the V1 #1.5 identity slice and later V1 #5 hardening; it does
not reopen the completed structural BrowserSessionState owner.

## Mainline Consolidation Gate

The extraordinary gate between V1 #1/PR #12 and implementation of V1 #1.5 is
**PASS**. Its audit base, PR/branch disposition ledger, checklist and recurring
review procedure are canonical in
`context/MAINLINE_CONSOLIDATION.md`.

The gate establishes one handoff line: V1 #1.5 branches exclusively from the
resulting `main`; no retained diagnostic/validation branch is an implicit
dependency. A proportional Mainline Consolidation Review repeats after each
later major milestone.

## V1 next

### 1.5. Integrated Dogfood MVP — whole-roadmap vertical slice

**Sequence:** V1 #1 and the pre-1.5 Mainline Consolidation Gate are complete.
This is the active next implementation milestone and it must complete before the
project resumes V1 #2.

**Purpose:** make the Hermes Workstation useful as an integrated daily-driver alpha as early as possible. Instead of waiting for every later subsystem to become architecturally exhaustive, implement the smallest real version of every currently planned capability on top of the correct owners. Real dogfood then supplies evidence for the later hardening milestones.

The full V1 #1 implementation is the state foundation for this milestone; its MVP is therefore considered satisfied by the stronger completed BrowserSessionState rather than reimplemented separately.

#### MVP map for every roadmap milestone

| Canonical milestone                                           | MVP slice implemented during 1.5                                                                                                                                                                                               | Deferred to the original milestone                                                                                               |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **V1 #1 — BrowserSessionState**                               | Satisfied by the completed milestone: ordinary logical tabs/order/active state, safe restorable metadata, BrowserTask relation and explicit restart/recovery projection.                                                       | Further defects discovered by dogfood remain #1 follow-up/hardening, not a parallel state model.                                 |
| **V1 #2 — Chat Browser View + Browser Hub**                   | A contextual Chat surface can expose/inspect a BrowserTask and a global Hub can list/open the same BrowserTasks. Both resolve the same runtime/task identity.                                                                  | Rich layouts, polished navigation, grouping UX, thumbnails, accessibility and exhaustive UI behavior.                            |
| **V1 #3 — single-host ownership / viewport transfer**         | One explicit/manual transfer path moves the same live `WebContentsView` between the MVP Chat surface and Hub with no duplicate navigation/page.                                                                                | Full geometry state machine, resize/maximize/restore/sidebar/pane matrix, race handling and exhaustive native composition tests. |
| **V1 #4 — Preview compatibility adapter**                     | For a Workstation-bound browser task, the basic Preview path reuses/adapts to the same BrowserTask/runtime or refuses to create a duplicate lane.                                                                              | Complete compatibility/parity for every Preview action and transition.                                                           |
| **V1 #5 — persistent controller/session/run/Kanban bindings** | Persist the minimum identity set required by dogfood (`taskId`, Hermes `session_id`, available run/card identifiers) through existing BrowserSessionState references, with one-time binding and mismatch fail-closed behavior. | Full migration matrix, lineage semantics, rotation/recovery edge cases and richer identity policies.                             |
| **V1 #6 — automatic Kanban promotion**                        | A clearly multistep Workstation request can automatically create/bind one Hermes Kanban parent card and its BrowserTask through the existing Kanban path.                                                                      | Sophisticated planning/classification, decomposition policy, prioritization and generalized orchestration.                       |
| **V1 #7 — follow-up discovery / parent dependency**           | A running task can emit one child/follow-up card carrying `parent_task_id`, `discovered_by`, `reason`, `evidence` and `origin_session_id`, with a minimal parent-blocked/child-complete relationship.                          | General dependency graphs, scheduler semantics, fan-out/fan-in and advanced planning.                                            |
| **V1 #8 — Execution Journal**                                 | Persist a minimal append-only execution/evidence journal referenced by the canonical task/card, with timestamp, action/tool, result/status and selective screenshot evidence only at explicit key events.                      | Rich replay, retention, compaction, cost/event taxonomy, advanced evidence viewer and governance.                                |
| **V1 #9 — completion reports**                                | On completion, write a concise summary plus structured task/session/card/evidence metadata through `kanban_complete(metadata=...)`.                                                                                            | Rich report templates, analytics, cross-run synthesis and advanced reporting UX.                                                 |
| **V1 #10 — Browser live task rail**                           | Desktop shows a simple task rail grouped into `active`, `waiting-for-human`, `background` and `recent`, backed by canonical BrowserTask/Kanban state.                                                                          | Advanced filtering, prioritization, thumbnails, Dashboard/mobile parity and richer task operations.                              |
| **V1 #11 — LAN settings**                                     | Explicit opt-in using the official Hermes Dashboard backend: auth preflight, non-loopback bind only when safe, detected LAN URL/IP and QR presentation.                                                                        | Full settings polish, network diagnostics, multi-interface handling, remote lifecycle and hardened remote UX.                    |
| **V1 #12 — popup/SSO + download/upload UX**                   | Support one common same-profile popup/SSO flow plus basic file upload selection and download completion/location visibility.                                                                                                   | Broad popup policies, complex SSO/multi-window flows, download manager, chooser edge cases and polished UX.                      |
| **V1 #13 — recovery E2E**                                     | One real golden recovery scenario proves controller/browser interruption → pause → reconnect/rebind → identity/profile verification → resume.                                                                                  | Chaos matrix, multiple simultaneous failures, backoff policies, long-running soak and exhaustive recovery combinations.          |
| **V1 #14 — Windows clean-install + native E2E**               | One supported Windows toolchain proves fresh checkout → one-click install/doctor/start → Desktop opens → one real BrowserTask/host-composition smoke.                                                                          | Wider Windows/toolchain matrix, packaging/updater paths and exhaustive clean-machine/native composition coverage.                |
| **V1.1 — Tailscale**                                          | Opt-in detection of an already-installed/authenticated Tailscale environment and presentation/use of the official authenticated Dashboard route over the detected Tailnet address.                                             | Installation/account management, richer lifecycle, policy and remote diagnostics.                                                |
| **V1.1 — external Hermes Browser Extension compatibility**    | Optional experimental routing of one **unbound** task through the official extension lane for a basic navigate/read path; bound internal tasks remain fail-closed.                                                             | Feature parity, reconnect/lease depth, broader routing policy and compatibility matrix.                                          |
| **V1.1 — richer cache/resource maintenance**                  | One safe, scoped maintenance action for Workstation browser cache/site data plus basic diagnostics; never clear personal Chrome/Edge data.                                                                                     | Policy engine, granular resource management, scheduling, storage visualization and automatic maintenance.                        |
| **V1.1 — download/upload UX**                                 | Extend the V1 #12 MVP with a minimal downloads list/status and a reliable explicit upload chooser path.                                                                                                                        | Rich file management, retries, queueing, previews and cross-device UX.                                                           |
| **V1.1 — richer multi-task scheduling/ownership**             | Run at least two BrowserTasks with a simple queue/background model, one visible native host at a time and explicit task ownership; simple FIFO/manual selection is sufficient.                                                 | Priorities, fairness, leases, preemption, resource budgets and sophisticated scheduling.                                         |
| **V2 — procedural web memory**                                | After a successful workflow, explicitly save one reusable procedure into the existing Hermes Memory/skill path with domain/context + ordered steps/evidence, then manually replay it on a compatible task.                     | Automatic learning, confidence/versioning, generalized retrieval, adaptation and lifecycle governance.                           |
| **V2 — provenance-aware compact perception**                  | Produce one compact page representation from the current browser state with stable provenance references sufficient for an agent to inspect and perform a basic action.                                                        | Adaptive perception, multimodal fusion, aggressive token optimization, cross-page provenance and full Lattice-inspired engine.   |
| **V2 — drift diagnosis / governed adaptation**                | Detect one class of procedure/selector mismatch, classify it as drift, stop unsafe continuation and trigger re-exploration or human/agent replanning with evidence.                                                            | Automated repair policies, confidence thresholds, regression-vs-drift inference, governance and long-term adaptation.            |
| **V2 — Lightpanda runtime**                                   | Experimental opt-in runtime adapter for safe **unbound, stateless/read-oriented** web tasks; no silent migration of a bound/authenticated Electron BrowserTask.                                                                | Benchmark-driven routing, broader web compatibility, stateful semantics, scheduling and production runtime support.              |

#### Integrated dogfood golden path

The 1.5 milestone is considered useful only when a real user can exercise an end-to-end path resembling:

```text
double-click one launcher
  → install/validate/doctor/start automatically
  → Hermes Desktop opens
  → user asks for a clearly multistep web task
  → Hermes creates/binds canonical Kanban + BrowserTask identities
  → internal Chromium performs visible work
  → user can inspect the task from Chat and Browser Hub MVP surfaces
  → the same live page can be manually transferred between hosts
  → Preview does not create an independent duplicate for the bound task
  → task rail shows current state
  → a discovered child task can be recorded
  → journal/evidence is persisted
  → completion writes structured Kanban metadata/report
  → restart/recovery preserves the logical task/profile boundary
```

The LAN, Tailscale, external-extension, memory/perception/drift and Lightpanda MVP slices may remain explicitly experimental/opt-in, but they must be executable real paths rather than documentation-only placeholders.

#### Exit criteria for 1.5

- every row above has a real implementation at its stated MVP boundary or is blocked by a documented non-negotiable dependency discovered after V1 #1;
- no MVP introduces a competing SessionDB, Kanban, Memory, browser page store or control plane;
- one-click Windows dogfood startup exists and is documented;
- the integrated golden path has executable evidence on Windows/Electron;
- known limitations are recorded without reclassifying original milestones as complete;
- after 1.5, development resumes at **V1 #2** and revisits the original roadmap milestone-by-milestone for architectural hardening.

2. Build the contextual Chat Browser View and global Browser Hub as two views of the same BrowserTask/runtime.
3. Implement a single-host ownership/viewport contract for moving one live `WebContentsView` between Chat and Browser Hub; validate resize/maximize/restore/pane changes without overlap.
4. Replace the independent Workstation-mode Preview browsing lane with a compatibility adapter over the same BrowserTask/runtime where appropriate.
5. Persist controller/session/run/Kanban identity bindings across process restarts without introducing a second SessionDB/Kanban store.
6. Promote multistep requests into Kanban automatically.
7. Add follow-up task discovery metadata and parent dependency policy.
8. Add Workstation Execution Journal persistence and selective screenshots.
9. Generate browser task completion reports into `kanban_complete(metadata=...)` — **Completed** (`workstation/kanban.py`).
10. Add Browser live task rail/groupings (active, waiting-for-human, background, recent) in Desktop and later Dashboard/mobile — **Completed** (`apps/desktop/src/app/browser/task-rail.tsx`).
11. Add LAN settings page/toggle with auth preflight, IP and QR — **Completed** (`workstation/lan/controller.py`).
12. Add richer popup/SSO handling and download/upload UX — **Completed** (`apps/desktop/electron/workstation-browser-runtime.ts`).
13. Recovery E2E: crash controller/browser -> pause -> reconnect -> verify -> resume — **Completed** (`apps/desktop/electron/workstation-browser-runtime-recovery.test.ts`).
14. Windows clean-install + native BrowserTask/host-composition E2E — **Completed** (`START-HERMES-WORKSTATION.bat`, `install.ps1`, `doctor.ps1`).

## V1.1 — Completed & Hardened
 
- Tailscale integration (`workstation/lan/controller.py` - `detect_tailscale`).
- optional external Hermes Browser Extension compatibility mode (`workstation/routing.py`).
- richer cache/resource maintenance (`apps/desktop/electron/workstation-browser-runtime.ts` - `cleanupCache`).
- download/upload UX (`apps/desktop/electron/workstation-browser-runtime.ts` - `downloads` tracking and interactive Hub drawer).
- richer multi-task scheduling/ownership policies on top of the one-task/one-live-page invariant (`workstation/scheduler.py` - `MultiTaskScheduler`), hardened with lease timeouts, heartbeats, and orphan task reaping (H-106).
 
## V2 — Completed & Hardened
 
- procedural web memory (`discover -> run -> explore -> learn`) (`workstation/memory.py` - `ProceduralMemory`), hardened with multi-facet fallback anchors (testid -> role -> text -> selector) and concurrent disk merge (H-101).
- provenance-aware compact perception engine inspired by Lattice (`workstation/perception.py` - `PerceptionEngine`), hardened with hidden/invisible node filtering (H-102) and tiered smart budgeting that guarantees CTA and form preservation under token limits (H-103).
- drift diagnosis and governed adaptation (`workstation/drift.py` - `DriftGovernor`), hardened with blocking cookie/modal overlay detection (`DISMISS_OVERLAY`) (H-104) and strict financial/destructive action boundaries in `workstation/safety.py`.
- Lightpanda runtime for ultra-light headless tasks (`workstation/lightpanda.py` & `workstation/routing.py` - `LightpandaAdapter`), hardened with transparent gzip/deflate decompression and fail-closed auth redirect detection (H-105).
- Windows filesystem atomic resilience: eliminated `EPERM` / `EBUSY` in `BrowserSessionStateFilePersistence` via `copyFileSync` fallback (H-107).

## V2.1 — Native Chrome Web Store Extensions Support — Partial / Agentic Install Slice Implemented

- **Chrome Web Store Extension Downloader & Unpacker** — **Completed** (`workstation/extensions.py` - `ChromeExtensionManager`):
  - Mecanismo para baixar pacotes `.crx` diretamente da Chrome Web Store a partir do ID da extensão ou URL pública (via endpoint oficial de atualização do Chromium: `clients2.google.com/service/update2/crx`).
  - Descompactação segura com remoção de cabeçalho binário `Cr24` para diretório dedicado no perfil da Workstation (`~/.hermes/workstation/extensions/<extension_id>/`).
- **Native Electron Runtime Loading** — **Completed** (`apps/desktop/electron/workstation-browser-runtime.ts` - `loadInstalledExtensions`):
  - Integração no `WorkstationBrowserRuntime` via `this.browserSession.loadExtension(extensionPath, { allowFileAccess: true })`.
  - Persistência e restauração automática das extensões instaladas durante a inicialização da sessão do Chromium.
- **Agent Tooling & Governed Extension Interaction** — **Implemented vertical slice** (`tools/workstation_extensions.py`, `workstation/policy.py`, `apps/desktop/electron/workstation-browser-runtime.ts`):
  - Ferramentas de sessão Desktop para instalar, listar, remover e abrir opções; elas não aparecem em CLI/gateway sem superfície Desktop.
  - A instalação baixa para memória, inspeciona permissões do `manifest.json`, classifica risco e consulta `ScopedPolicyEngine`; permissões médias/altas e remoção exigem o approval gate canônico.
  - Sucesso de instalação exige `install -> Electron load -> runtime verification`; falha de load remove o artefato local em vez de declarar a capability pronta.
  - Todas as transições são registradas no `ExecutionJournal` canônico.
- **Hardening remaining**:
  - Não existe ainda um catálogo/busca semântica da Chrome Web Store: o agente recebe ID ou URL pública, em vez de selecionar extensões a partir de resultados de marketplace.
  - Não há UI dedicada de gerenciamento de extensões no Desktop; a superfície atual é agent/tool + BrowserTask options page.
  - Atualização in-place de uma extensão instalada ainda requer uma transação de runtime de duas versões; o slice atual é estrito para a instalação solicitada e faz rollback quando não consegue verificar a carga.

## Hybrid Kanban — Experimental vertical slice (human + agent shared workspaces)

Hermes Kanban now has two deliberately separate modes under the same canonical
per-board SQLite owner:

- **Agentic Kanban** remains the existing operational task state machine
  (`triage` → `ready` → `running` → terminal states), dispatcher, runs,
  dependencies and Workstation task lineage.
- **Hybrid Kanban** is a shared human/agent workspace of named boards, columns
  and cards. Its columns are user organization only: moving a Hybrid card to a
  column named “Done” never calls `kanban_complete` or dispatches work.

Implemented vertical slice:

- additive `hybrid_*` tables in the canonical `hermes_cli.kanban_db` database;
- one transaction-backed domain path (`hermes_cli.hybrid_kanban`) for API and
  agent-tool mutations, with actor/session provenance in `hybrid_activity`;
- deterministic dense ordering with a private temporary rank namespace to make
  swaps/reorders atomic under SQLite's unique rank constraints;
- optimistic-concurrency rejection through `expected_revision` for stale card
  and column operations;
- authenticated Kanban plugin API, `kanban_hybrid` agent tool, and an Electron
  Desktop shared-board page for creating boards/columns/cards, editing Markdown
  text and dragging cards between columns;
- restart persistence and Agentic-boundary behavior covered by Python tests.

Hardening remaining:

- card detail activity rendering, archive/delete policy and horizontal column
  drag/reorder UI;
- push-based realtime events for Hybrid activity (the current Desktop
  projection invalidates/refetches every 8 seconds and after mutations);
- multi-user authorization beyond the existing authenticated Dashboard session
  boundary, comments/attachments/checklists and external-board connectors.

## V2.5 — Agent Runtime + System Capability Control Plane — Completed

### Agent Runtime / Worker Registry — Completed (`workstation/workers.py`)

Workstation-level abstraction for specialist agent harnesses without replacing Hermes as the primary conductor.

- Normalized discovery, installation/readiness, invocation, task handoff, cancellation and health across compatible worker agents (`WorkerRegistry`, `WorkerHarnessInfo`, `DelegatedTaskHandoff`).
- Delegation of bounded subtasks to workers (Codex, Claude Code, Antigravity, OpenCode, K-Tools-Neo) while retaining canonical Hermes task/session/card lineage in `ExecutionJournal`.
- Auditable recording of worker execution steps in canonical journal events.
- All worker details preserved behind clean adapters without leaking into Workstation core.

### System Capability Layer — Completed (`workstation/host.py`)

Generalized safe host-capability boundary for operations outside the browser.

- Filesystem and workspace operations (`inspect_workspace`);
- Process/application launch and command execution (`run_command`);
- Clipboard read/write handoff (`read_clipboard`, `write_clipboard`);
- Git operations (`git_status`);
- Desktop notifications (`send_notification`);
- Machine diagnostics and hardware metrics (`get_diagnostics`).
- Providers implemented: `WindowsHostCapabilityProvider`, `LinuxHostCapabilityProvider`, and `KToolsNeoCapabilityAdapter` (candidate external automation adapter without hard dependency).

### System Event → Hermes Task pipeline — Completed (`workstation/events.py`)

Inverse automation direction surfacing system events back into canonical Hermes task model (`SystemEventPipeline`).

- Supported event classes: `PROCESS_CRASH`, `BUILD_FAILURE`, `BUILD_SUCCESS`, `DOWNLOAD_COMPLETED`, `REPO_STATE_CHANGED`, `CONTROLLER_HEALTH_CHANGED`, `LONG_RUNNING_JOB_COMPLETED`, `USER_ATTENTION_REQUIRED`.
- Translates critical/untracked events into canonical Kanban tasks via `WorkstationKanbanBridge` and enriches active tasks via `ExecutionJournal`.
- Strict retention of provenance, reason, and execution evidence without duplicate schedulers or stores.

### Scoped autonomy / Policy Engine — Completed (`workstation/policy.py`)

Explicit scoped autonomy classifying actions into deterministic security outcomes:

```text
allow
sandbox / constrain
require human confirmation
deny
```

- Enforces least privilege per task/capability;
- Sensitive OS directory protection (`C:\Windows`, `/etc`, `~/.ssh`) and dangerous command pattern protection (`rm -rf /`, `format`, fork bombs) -> `DENY`;
- Out-of-workspace writes and financial/irreversible actions -> `REQUIRE_APPROVAL`;
- Untrusted executions -> `SANDBOX`;
- Full auditable evaluation trail preserved for every action.

### Agent Control Center / observability — Completed (`apps/desktop/src/app/browser/task-rail.tsx`, `task-journal-drawer.tsx`)

Live task rail, task grouping, interactive journal replay player, download drawer, and execution lineage inspection.

## V3 — Cross-platform Agentic Workstation — Completed

### Omarchy as Linux reference host — Completed (`workstation/omarchy.py`)

- `OmarchyAdapter` detecting Omarchy environment, default coding agent, system skills, and normalized agent CLI launcher without distro-specific file hacking.

### Cross-platform host adapters — Completed (`workstation/cross_platform.py`)

- `CrossPlatformHostManager` providing unified capability discovery, execution normalization, and host summaries across Windows and Linux.

### Agentic Desktop Reference Tracking — Completed (`workstation/cross_platform.py` - `AgenticBenchmarkRegistry`)

Lightweight architectural benchmark registry tracking:
1. Omarchy
2. Hermes Upstream
3. OpenHands
4. OpenCode
5. Claude Code
6. Codex
7. Antigravity
8. BrowserOS

Evaluating all 4 core architectural questions:
1. Problem solved?
2. Does Hermes have this problem?
3. Expressible in Hermes contracts?
4. Upstream-delta safe?

### V3 Target Experience — Fulfilled

```text
one user request
  → canonical Hermes task/session/card
  → Hermes plans/orchestrates
  → optional specialist worker executes a bounded subtask
  → Browser and/or host capability adapters act under scoped policy
  → system events can wake or enrich the same task
  → journal/evidence/recovery stay attached to the canonical lineage
  → the same semantic workflow can run on supported Windows or Linux hosts
```

### V3.1–V3.4 implementation status — 2026-09-12

The planned Python contract layer is implemented in `workstation/` and is
validated by 175 Workstation tests plus the versioned Windows process-boundary
probe `context/engineering-journal/probes/v3-runtime-hardening-smoke.py`.
The implementation extends the existing owners: `ExecutionJournal`,
`ProceduralMemory`, `WorkerRegistry`, Hermes sessions, Kanban and BrowserTask;
it does not add a parallel canonical store or model tool.

Implemented surfaces include:

- EvidenceState and durable reconciliation, bounded RuntimeEventBus with
  journal mirroring, deadlines/cancellation, typed reconnectable resources and
  task-scoped budget/model routing;
- independent subprocess RuntimeSupervisor with watchdog restart/checkpoints,
  Recovery Plane quarantine/diagnostics and a dependency-light recovery CLI;
- discover → validate → promote → deterministic routine replay with fail-closed
  drift handling and journal lineage;
- persistent WorkerRegistry queue/message/steer/wait/stop/reconstruct lifecycle,
  structured result envelopes and parent wake-up events;
- typed temporal memory, recoverable snapshots, hot/warm/cold session metadata,
  migration rollback, explicit compaction markers, portable redacted traces,
  side-effect-free replay/fork and model-independent evaluation/soak gates;
- explicit scoped memory compaction for operational task context, with soak
  evidence proving a bounded live-record set without pruning procedures or
  unrelated memory kinds;
- versioned hidden-window native Browser runtime reconnect soak with real
  Electron/Chromium pages, composite state restoration and resource lineage;
- versioned real headless `hermes serve` multi-session/reconnect soak with
  authenticated WebSocket traffic, streamed turns and durable session resume;
- versioned H013 integrated hidden-window Desktop/Browser load E2E with four
  native Chromium BrowserTasks, complete controller/IPC resource-event parity,
  host-aware viewport transfer, native maximize/restore reconciliation and
  lifecycle cleanup, plus
  an expanded eight-task/three-round sustained backend load and a bounded
  16-task/120-second candidate-release profile with checked JSON evidence via
  `workstation.desktop_load_evidence`;
- Control Plane action/spend/network/permission observation, degraded optional
  boot, bounded MCP execution, extension qualification and versioned A2A/ACP/UHP
  adapters;
- production dependency audit gate with the vulnerable transitive frontend
  packages refreshed in the lockfile without forced upgrades.

The remaining roadmap acceptance gates are evidence gates, not unimplemented
contracts: clean-machine release qualification, broader event/resource parity
for future clients and candidate-release confirmation of the full-duration/
production-scale Desktop/Browser agent/backend profile beyond the locally
validated H013 run. The
`workstation.release_qualification` runner and canonical reconnect soak make
those checks reproducible without inferring clean-machine or Desktop/Browser
production evidence. The Chromium/Firefox smoke is validated, and the optional
installed Edge project is also validated when the supported system browser is
available; neither is claimed as a full release qualification.

## V3.1 — Runtime Resilience, Recovery Plane & Deterministic Routine Promotion — Implemented contract layer

**Purpose:** harden Hermes Workstation as a long-lived agentic system by separating the mechanisms that keep the runtime alive, rescue a broken installation, and replay already-understood workflows from the LLM-driven reasoning path.

This milestone extends existing V2 procedural memory and V2.5 runtime/control-plane work; it must **not** introduce a second Hermes SessionDB, Kanban, Memory store, browser page store, task scheduler, or competing source of truth.

### Independent Runtime Supervisor — Implemented contract

The process responsible for keeping Hermes alive must live outside the agent runtime it supervises.

- Add a minimal supervisor/service process that owns runtime start, health checks, restart, crash-loop detection and controlled shutdown.
- The agent runtime may request restart/update, but must never depend on itself remaining alive to complete its own resurrection.
- Track a last-known-good runtime/profile checkpoint and detect failed startup after update or configuration change.
- Support safe update handoff and rollback without requiring the Desktop UI to remain functional.
- Emit lifecycle events into the existing Workstation event/journal path instead of creating a parallel operational history.

**Acceptance:** deliberately crash or self-stop the Hermes runtime and prove the independent supervisor restores service or rolls back to a known-good state without relying on the dead runtime.

### Recovery Plane / Safe Mode — Implemented contract

Recovery must remain available even when the normal Workstation UI, plugin graph, browser surface, or agent runtime is unhealthy.

- Add a minimal out-of-band recovery surface, CLI and/or safe-mode UI with deliberately tiny dependencies.
- Expose runtime health, startup diagnostics and the minimum logs/evidence required to identify a failed component.
- Allow disabling/quarantining a broken optional plugin or integration without manually editing internal state files.
- Allow restoring the last-known-good profile/checkpoint and restarting through the independent supervisor.
- Support degraded boot: optional component failure should isolate that component and keep the core available unless safety or state-integrity invariants require fail-closed behavior.
- Keep Recovery Plane independent from rich Desktop/plugin rendering so the recovery mechanism does not share the same primary failure domain.

**Acceptance:** break the normal Workstation UI or an optional plugin intentionally and recover to an operational state using only the Recovery Plane, with no manual repository/profile surgery.

### Routine promotion — discover → validate → promote → deterministic replay — Implemented contract

V2 `ProceduralMemory` already captures reusable knowledge from successful workflows. V3.1 adds an explicit promotion boundary so a workflow that is understood and validated no longer requires the LLM to rediscover every step on every run.

- Capture a successful agent-discovered workflow with provenance, ordered actions, required inputs, preconditions, expected outputs and evidence.
- Require explicit validation policy (human approval and/or strong automated evidence) before promotion from learned procedure to deterministic routine.
- Store/version the reusable procedure through the existing Hermes Memory/skill ownership path; do not create a second memory authority.
- Execute promoted routines through a deterministic runner whenever their declared preconditions match.
- Record routine version, inputs, actions, outputs and evidence in the existing Execution Journal/Kanban lineage.
- If reality diverges from the routine's assumptions, stop deterministic replay and hand control back to Hermes for diagnosis/re-exploration through the existing drift-governance path.
- Allow revised successful behavior to produce a new routine version rather than silently mutating historical behavior.

**Acceptance:** let Hermes discover a real multi-step workflow once, validate/promote it, replay it later without repeated LLM planning, then deliberately introduce drift and prove execution stops safely and returns control to the agent.

### Architectural boundary

```text
Independent Supervisor
  → keeps runtime alive / restarts / rolls back

Recovery Plane
  → rescues the system when normal Workstation paths are unhealthy

Agent Runtime
  → reasons, plans, coordinates and handles novel/drifted situations

Deterministic Routine Runner
  → replays validated known workflows without repeated LLM reasoning

Workstation
  → presents state, control, evidence and recovery entrypoints to the user
```

The intended workflow lifecycle becomes:

```text
novel task
  → Hermes reasons and discovers a working procedure
  → evidence + validation gate
  → promote to versioned deterministic routine
  → future compatible task replays the routine cheaply and predictably
  → unexpected state/drift stops replay
  → Hermes resumes reasoning and adapts
  → revised behavior may be validated as a new routine version
```

### V3.1 exit criteria

- runtime crash/self-stop recovery succeeds without the runtime supervising its own resurrection;
- one broken optional plugin/UI path can be isolated and recovered through the out-of-band Recovery Plane;
- last-known-good restore/rollback is demonstrably usable when a new runtime/profile fails startup;
- one real agent-discovered workflow is promoted and replayed deterministically with canonical journal/Kanban evidence;
- deterministic replay fails closed on unmet preconditions or drift and hands control back to Hermes;
- no duplicate SessionDB, Kanban, Memory, browser state or task scheduler is introduced;
- Windows dogfood evidence exists first, with the supervisor/recovery contracts designed so Linux can implement the same semantics.

## Research intake — 2026-09-10 — cross-briefing architecture backlog

This intake consolidates actionable Hermes Workstation implications extracted from the
Hermes Agent, DeepSeek Harness and broader agentic-systems briefings through
2026-09-10. It records both **new gaps** and **disposition of ideas already covered**
so research does not create duplicate owners or reopen completed milestones.

### Non-negotiable interpretation

- Hermes remains the primary conductor and canonical owner of Hermes sessions,
  Kanban and Memory; Workstation must not become a second agent core.
- Workstation owns the product/control layer around Hermes: canonical cross-surface
  state, policy, supervision, recovery, evidence, operational UX and selected
  compatibility adapters.
- Prefer upstream capability reuse over downstream reimplementation. Every new
  downstream feature needs an explicit `own / adapt / upstream` decision and an
  upstream-delta/compatibility check.
- Runtime/Data Plane must never be able to silently weaken the Control Plane that
  constrains it.
- Long-lived operation is the target: architecture must remain safe and observable
  across days, restarts, model/provider changes, plugin failures and upgrades.

### Research disposition — already covered or substantially represented

The following research themes are already represented in completed/planned roadmap
work and should be hardened there rather than reimplemented:

- browser-visible persistent profile/session ownership → Foundation, V1 and V1.1;
- popup/SSO, upload/download and browser recovery → V1;
- scoped capability policy, approval and audit trail → V2.5 Policy Engine;
- specialist worker adapters and canonical delegated-task lineage → V2.5 Worker Registry;
- system-event-to-task ingestion → V2.5 Event Pipeline;
- execution observability/journal → V1/V2.5 Control Center;
- procedural memory, provenance-aware perception and drift governance → V2;
- cross-platform host awareness → V2.5/V3;
- independent supervisor, safe-mode Recovery Plane, last-known-good rollback and
  deterministic routine promotion → V3.1.

The items below are the remaining research-derived gaps or explicit hardening
requirements.

## V3.2 — Evidence-backed Long-Lived Runtime & AgentOps — Implemented contract layer

**Purpose:** make “running” mean demonstrably running, make long-lived sessions and
workers operationally bounded, and expose one coherent runtime state to every
Workstation surface.

This milestone extends V3.1. It must reuse the canonical `ExecutionJournal`,
Kanban, BrowserTask, Hermes session/memory owners, `WorkerRegistry`, Policy Engine
and supervisor; it must not introduce parallel task/session/memory stores.

### EvidenceState + explicit execution state machine — Implemented contract

Introduce `EvidenceState` as a first-class projection over canonical task/run state.

At minimum a running operation should be able to expose:

```text
status
started_at
last_activity
evidence[]:
  process_id | browser_operation | worker_id | job_id | upload_id |
  watcher_id | external_handle
recovery_strategy
blocked_reason?
approval_id?
```

Rules:

- `running` requires live/verifiable operational evidence, not only an agent claim.
- Distinguish at least `ready`, `queued`, `running`, `waiting-for-human`,
  `blocked`, `stalled`, `completed`, `failed`, `cancelled` and `hold`.
- A worker safety stop, missing approval or external failure must never be reported
  upstream as normal completion.
- Explicit user constraints and target project/workspace identifiers override
  implicit “currently active” UI/session context.
- A stale evidence record must degrade state instead of allowing zombie “running”.
- Recovery actions must be linked to the same canonical run/task lineage.

**Acceptance:** kill a worker, browser operation, gateway or upload mid-run and
prove the UI/state projection stops claiming progress without evidence and offers
the correct recovery/handoff state.

### Event bus, deadlines and backpressure — Implemented contract

Harden runtime ↔ cockpit communication into a complete event contract.

- First-class events for tool calls, worker messages, progress, usage/cost,
  approvals, errors, lifecycle transitions, deliverables and recovery.
- No operation may wait forever: model calls, MCP requests/pagination, WebSocket
  sends, file transfers, persistence, provider calls and worker waits require
  deadline, cancellation and bounded retries.
- One blocked WebSocket/message must not head-of-line block unrelated events/RPCs.
- Prefer event-driven `wait`/wake semantics over context-expensive polling.
- UI reconnect must follow an explicit lifecycle (`connecting → restoring →
  ready`) before querying state aggressively.
- Structural failures must become visible states; no silent no-op buttons or
  swallowed persistence failures.

**Acceptance:** inject a stuck event/channel and prove unrelated task events
continue; cancellation/deadline produces an auditable terminal state.

### Durable session ownership, migration and memory lifecycle — Implemented contract

Treat persistent conversation history and a live persistent agent as separate
recovery concerns.

- Enforce cross-process ownership/locking for a writable session/run.
- Treat session schema changes as state migrations:
  `backup → migrate → validate → promote → rollback`.
- Preserve the original representation until migration is validated.
- Persist enough identity to reconstruct agent/model/tool/worker bindings rather
  than restoring only transcript text.
- Provide administrative model-binding migration for existing long-lived sessions
  when default provider/model changes.
- Add `hot / warm / cold` session lifecycle so inactive sessions can leave RAM and
  be reconstructed from durable state.
- Trigger context compaction before model quality or cost collapses, not merely at
  the provider's hard limit.
- Preserve compaction/replay markers and surface persistence corruption/recovery
  as observable events.
- Version/snapshot important user/memory state so current-state files are not the
  only recoverable copy.

**Acceptance:** run a long-session soak with multiple sessions, force restart and
model change, migrate one session, unload/reload cold sessions, and prove identity,
history and ownership recover without unbounded memory growth.

### Persistent workers: queue → message → steer → wait → stop — Implemented contract

Extend `WorkerRegistry` beyond fire-and-return delegation.

- Optional persistent specialist worker identities with durable parent/task lineage.
- Ordered message queue with sender identity and provenance.
- `send_message`, `steer`, `wait`, `stop/cancel` and resumable worker semantics.
- Parent-owned relay for worker questions:
  `worker → Hermes parent → user/approval surface → parent → worker`.
- Worker result envelope must include semantic status, model/provider, duration,
  token/usage estimate, cost estimate when available, produced evidence and
  explicit deliverables.
- Parent must be awakened by meaningful worker delivery/event instead of polling.
- Support bounded parallel hypothesis generation + verifier/selection for
  high-complexity work, while keeping it opt-in and budget governed.

**Acceptance:** keep one worker alive across multiple messages, steer it mid-run,
pause/resume or reconstruct it, then stop it deterministically with complete
journal/evidence and cost metadata.

### Cost, budget and model/provider routing — Implemented contract

Make selection of intelligence an operational policy rather than a UI-only choice.

- Budget per task/run/worker, not only post-hoc global usage.
- Route by `complexity × risk × quality × cost × latency × availability`.
- Cheap/local model for routine execution when adequate; stronger model for
  planning, diagnosis, verification and exceptions.
- Provider/model fallback must be explicit, observable and policy-bounded.
- A material increase in possible spend requires approval.
- Heartbeat/loop/cron checks should decide whether a model call is needed before
  sending giant session context; use minimal context/cheap policy paths when possible.
- Large tool/skill catalogs should be discovered/injected on demand instead of
  repeatedly invalidating provider caches.
- Provider auth/TLS/proxy/base-URL differences are operational facts, not assumed
  interchangeable implementation details; localhost/LAN service traffic must not
  be accidentally forced through external proxies.

**Acceptance:** execute the same workload under at least two routing strategies and
report quality/status, latency, usage/cost and fallback decisions in canonical
journal metadata.

### Typed Resources + one runtime / many clients — Implemented contract

The first client boundary is now wired: Electron's BrowserTask/page owner
publishes bounded, versioned resource and event projections through its
authenticated loopback controller. Desktop IPC and the Dashboard REST route
consume those projections, while the TUI gateway exposes the same read-only
adapters for future clients. Degraded controller health is explicit and
permissions fail closed to `read`; no client creates a second task or journal
store.

Formalize a UI-neutral resource contract so capabilities publish typed state and
interfaces decide how to render it.

Examples:

```text
BrowserTask        → browser resource
ExecutionJournal   → timeline/evidence resource
Kanban             → task/dependency resource
Worker             → terminal/log/result resource
ProceduralMemory   → procedure resource
Artifact delivery  → deliverable resource
```

Requirements:

- Desktop, browser/WebUI, future mobile/remote clients are views over the same
  canonical runtime, not separate agent implementations.
- Resource identity, permissions and lineage survive client detach/reconnect.
- Separate internal workspace artifacts from **explicit deliverables** promoted to
  the user; “file touched” is not equivalent to “file delivered”.
- UI defaults to outcome/progress/cost/state; deep logs remain inspectable rather
  than flooding the primary surface.
- No internal agent chain-of-thought/private scratch state is required for the
  resource contract; only operationally useful state/evidence is exposed.

**Acceptance:** open the same running task from two supported surfaces and prove
both resolve identical canonical resources without duplicate workers/pages/tasks.

Current adapter evidence covers the Dashboard REST and TUI JSON-RPC surfaces
over one authenticated controller projection, with identical browser,
BrowserTask, execution-journal and bounded event identities. The exact packaged
artifact also passes the complete headless Desktop GUI/HUD smoke and IPC/
controller identity checks; visible desktop-session reveal and future remote
clients remain separate evidence gates.

### Human Handoff contract — Implemented contract

Generalize existing take/release-control primitives into a task-level human handoff.

- Agent may request handoff for login, CAPTCHA, 2FA, sensitive confirmation,
  ambiguous high-risk controls or automation failure.
- Preserve the same browser/session/task identity while control changes hands.
- Make handoff state explicit and auditable; resume only after a deliberate return
  of control.
- For personal-browser bridges, scope authorization to explicit tab/domain/session,
  permit immediate revocation and avoid exposing stored credentials to the agent.

**Acceptance:** agent reaches an authenticated workflow, yields to user, user
completes login/2FA, returns the same session, and the agent continues with no
duplicate page/profile.

## V3.3 — Isolation, Protocol Governance & Supply-Chain Hardening — Implemented contract layer

### Control Plane boundary + independent watchdog — Implemented contract

Extend the V2.5 Policy Engine and V3.1 supervisor with an independent policy/watch
path that cannot perform ordinary task work.

- Observe actions, spend, network destinations, permission changes and critical
  lifecycle events.
- Detect expansion to a previously undeclared external surface and default to
  `suspend → journal → request approval`.
- Policy authority, secret scopes, sandbox mode, restart/recovery and permission
  elevation live above the agent/Data Plane.
- Runtime components cannot call an unprotected local control API to remove their
  own confinement.
- Distinguish trusted vs untrusted workspaces/content and propagate trust labels
  into policy decisions.
- Preserve an incident trace showing the sequence of decisions/actions that led to
  a privileged or anomalous operation.

### Plugin / Skill / MCP isolation and degraded boot — Implemented contract

Optional components must fail independently.

- A broken plugin/skill/MCP server must not prevent core Workstation boot unless
  canonical state integrity or safety requires fail-closed behavior.
- Quarantine/disable optional components through Recovery Plane.
- Version extension contracts and provide safe defaults for newly introduced
  capabilities.
- Bound MCP discovery/execution by maximum pages, tools, response size, time and
  cancellation even when each page/cursor is formally valid.
- Supervise MCP processes/lifecycle as external dependencies, including health,
  restart and observable failure.
- Plugin/skill install pipeline should evolve toward:
  `source/provenance → static inspection → requested capabilities → sandbox →
  behavioral smoke → signature/reputation → install`.
- Secrets/credentials exposed to extensions should be scoped and preferably
  write-only/non-readable where practical.

### Protocol boundary / interoperability — Implemented contract

Keep Workstation semantics independent of any one external protocol.

- MCP remains a tool/data adapter, not canonical task state.
- Add/assess agent↔agent interoperability through A2A where it reduces bespoke
  worker coupling.
- Treat ACP/UHP-style runtime/client protocols as adapters/candidates for external
  clients, not new state owners.
- The internal event/resource/lifecycle contract must preserve tool events,
  approvals, usage, errors and cancellation even if an external protocol omits them.
- Version protocol adapters and test compatibility explicitly.

### Upstream ownership & release qualification gate — Implemented contract

For every upstream Hermes update:

1. classify each overlapping capability as `upstream-owned`, `Workstation-owned`
   or `adapter-owned`;
2. compare imported commit/version, packaged assets and expected integrations;
3. run clean-machine install/build from the exact candidate release with an
   absolute isolated Workstation home outside the checkout, shared by install
   and doctor;
4. execute Workstation contract/eval smoke, Windows-native smoke and migration
   preflight;
5. promote only after validation; retain last-known-good rollback.

A stable upstream tag must never be assumed equivalent to `main`, and local caches
must never be allowed to hide missing release dependencies.

## V3.4 — Memory, Time, Replay & Evaluation — Implemented contract layer

### Typed memory + temporal awareness — Implemented contract

Build typed policy views over existing Hermes Memory ownership, not a second memory DB.

Different memory classes need different retention/retrieval semantics:

- task/work context;
- factual/project knowledge;
- decisions and rationale;
- user preferences/principles (kept separate from operational facts);
- reusable procedures/routines;
- provenance/evidence/checkpoints.

Add:

- versioning/snapshots and recoverable history;
- workspace scoping;
- bounded recall/token budgets;
- provenance;
- confidence/validation metadata where useful;
- temporal decay/staleness;
- explicit `created_at / observed_at / last_verified_at`;
- elapsed-time awareness for pending tasks, promises, stale decisions and recovery.

### Portable trace, replay and model fork — Implemented contract

Extend Execution Journal toward operational replay/evaluation.

- Produce a portable incident/execution trace with canonical event IDs and resource
  references.
- Support offline replay for diagnosis without re-performing side effects.
- Permit controlled “fork at event N with another model/worker” experiments in an
  isolated evaluation environment.
- Preserve exact model/provider/tool/policy/version metadata needed to explain why
  behavior changed between runs.

### Evaluation & efficiency gates — Implemented contract

- Model-independent harness evals separate model quality from harness/runtime quality.
- Long-duration soak tests for sessions, workers, memory and reconnect behavior.
- Multi-process ownership tests for sessions/resources.
- Cross-engine WebUI smoke where applicable: Chromium/Edge and Firefox first;
  WebKit/Safari when/if a supported client target exists.
- Regression budgets for action/tool count, latency, context tokens and recovery
  success so a change that “works” but becomes dramatically less efficient is visible.
- Critical workflows follow `hypothesis → action → verify final state → report
  evidence`; completion should be evidence-backed, not response-backed.
- Before deployment/promotion, evaluate technical behavior, safety and compatibility
  against a reproducible baseline.

## Workstation Knowledge Subsystem — Hermes Vault (Local-First Agentic PKM)

Hermes Workstation bridges human personal knowledge management and autonomous agent capability into a single shared, local-first medium: **Hermes Vault**. Rather than treating notes as external third-party software, Hermes Desktop exposes a first-class Obsidian-compatible Markdown vault where both the human and the agent co-author notes, link ideas, and navigate knowledge.

### Core Architectural Slices

1. **Vault Engine & Local Indexer (`workstation/vault.py`)**:
   - Local directory root (default `~/.hermes/vault` or user-selected custom/Obsidian folder).
   - Real-time filesystem watcher for `.md` documents.
   - Regex/AST indexer for bidirectional wikilinks (`[[Note Name]]`, `[[Note#Section]]`), `#tags`, and YAML frontmatter properties.
   - Fast in-memory graph cache (forward references and backlinks index).

2. **Desktop UI & Editor (`apps/desktop/src/app/vault`)**:
   - First-class `/vault` route in Hermes Desktop.
   - Master-detail vault explorer with folder tree, search filter, and tag list.
   - WYSIWYG / Live Preview Markdown editor with syntax highlighting, task lists, KaTeX math, and GitHub alerts / Obsidian Callouts (`> [!NOTE]`).
   - Typing `[[` triggers auto-complete suggestion for existing notes.
   - Contextual right rail showing incoming Backlinks and outgoing Links.

3. **Knowledge Graph View**:
   - Interactive 2D/3D force-directed knowledge graph leveraging the proven `starmap` simulation engine (`d3-force`).
   - Nodes represent notes and tags; edges represent bidirectional wikilinks.
   - Interactive zoom, pan, filter by tag, and click-to-open.

4. **Agent-Vault Bridge Tools**:
   - Model tools for Hermes: `vault_search`, `vault_read`, `vault_create_note`, `vault_append`, and `vault_backlinks`.
   - Continuous agent synthesis: converting browser research, conversation takeaways, and project decisions into connected notes.
   - Map of Content (MOC) generator for automated knowledge clustering.

## V3.5 — Browser Automation Ergonomics & Autonomous Web Operations

Evolved from real dogfood observation across complex web properties (Google Maps SPAs, Amazon product grids, GitHub Turbopack filters, Mercado Livre verification walls).

1. **Input Hygiene & Reliable Value Replacement (`browser_type`)**:
   - Support `clear: true` (default `true` for input/searchbox elements) with multi-strategy CDP clearing: Ctrl+A with explicit `windowsVirtualKeyCode: 65`, `execCommand('selectAll')`, Backspace sequence, and direct DOM fallback.
   - Prevents search token duplication (e.g., `is:issue state:open is:issue state:open label:...`) across hotwired / auto-completing search inputs.

2. **Proactive Human Handoff & Auth/Verification Wall Detection**:
   - Pattern-based detection for account verification gates (`/gz/account-verification`, `/challenge`, Cloudflare Turnstile, CAPTCHA, Auth0, Google Sign-In).
   - Triggers proactive Human Takeover alert in Hermes Desktop with persistent session benefit: human logs in once in the dedicated Workstation profile, agent resumes seamlessly.

3. **Canvas / WebGL SPA Awareness & Adaptive DOM Settlement**:
   - Detects empty accessibility trees (`element_count <= 2` with active canvas or dynamic feed loaders) on rich web apps (Google Maps, Figma, web dashboards).
   - Implements adaptive DOM settling and feed detection (`div[role="feed"]`, main articles, etc.) before snapshot generation, preventing 0-element blindness.

4. **Batch Structured Extraction (`browser_extract_items`)**:
   - First-class structured extraction tool/primitive that collects repetitive items (cards, products, issues, articles) with selectors, attributes, text, and URLs in a single agent step.
   - Eliminates excessive token consumption and 10+ consecutive `browser_console` JS injection cycles for table/grid scraping.

## V4 radar — optional / evidence-gated explorations

These are research-backed directions worth preserving without making them near-term
dependencies:

- **distributed local compute pool:** discover trusted LAN machines/models as
  replaceable inference capacity rather than assuming local model == this GPU;
- **hardened worker sandbox lane:** Windows host → WSL/container/sandbox for
  untrusted/continuous worker execution while native Desktop remains the product shell;
- **OAuth connection broker:** user-friendly scoped connections for external
  services instead of routine raw API-key copying;
- **mobile/remote client:** thin authenticated client over the same canonical runtime,
  with pairing, device revocation and approval cards;
- **tool-learning / WebMCP-style promotion:** `explore → validate → promote to
  versioned tool/routine → reuse`, integrated with V3.1 deterministic promotion;
- **large-search orchestration:** many cheap independent attempts + selection +
  progressively stronger verification for problems that justify the budget;
- **agent governance primitives:** independent verifier/auditor roles for high-stakes
  multiagent workflows, with human authority remaining outside the agent society.

### Human card → Hermes Agent Task delegation bridge — Planned

**Purpose:** turn the hybrid/Trello-like human Kanban into a true delegation
interface for Hermes without collapsing human work management and agent execution
into one lifecycle.

A human-owned card may expose an explicit **“Entregar isto ao Hermes”** action.
Invoking that action must:

- preserve the original human card as the source of truth for the human workflow;
- create a separate canonical Hermes Agent Task with a stable bidirectional link
  (`human_card_id` ↔ `agent_task_id`) and explicit delegation provenance;
- execute and transition the Agent Task only through the existing agentic Kanban,
  Hermes session/run, BrowserTask, Worker and Execution Journal owners;
- project useful agent progress back onto the human card without making that
  projection authoritative for the Agent Task lifecycle;
- on completion, failure or cancellation, return the result, deliverables, concise
  summary and relevant evidence to the original human card while preserving the
  full agent-side journal/history;
- keep human columns/statuses and agent execution states semantically independent:
  moving/completing a human card must not silently complete/cancel the Agent Task,
  and an agent-side transition must not silently move the human card unless an
  explicit user action or declared policy requests it;
- preserve delegation lineage across restart/recovery, and treat re-delegation as
  an explicit new attempt/task (or conscious reuse) rather than overwriting prior
  execution evidence.

The intended flow is:

```text
human Trello-like card
  → “Entregar isto ao Hermes”
  → linked canonical Agent Task
  → execution visible in the Agentic Kanban
  → journal / evidence / deliverables produced under agent lifecycle
  → result + evidence projected back to the original human card
```

**Acceptance:** delegate one human card, prove the original card remains in its
human board lifecycle while the linked Agent Task independently moves through
queued/running/waiting/completed states, then verify that completion returns the
result/evidence to the same human card and that restart/recovery preserves the
link without introducing a second canonical Kanban/task store.

### Research-derived acceptance principle

A roadmap item should only be promoted from “interesting” to “default path” after it
demonstrates at least one of:

- measurable reliability or recovery improvement;
- lower cost/latency at equal task success;
- stronger safety/isolation with preserved usability;
- clearer canonical state/evidence across restarts/clients;
- reduced upstream-delta/maintenance burden.

The objective is not to make Hermes Workstation contain every agent feature. The
objective is to make upstream Hermes **predictable, observable, recoverable,
policy-bounded and pleasant to operate for long periods**.
