# Hermes Workstation foundation patch manifest

**Workstation version:** `0.1.0-dev.1`  
**Expected Hermes base:** `057dcdf236f8a6a26721c10fcc6ccb72726e272a`  
**Target:** Windows 11 first; portable architecture for Linux/macOS.

## How this downstream source is used

The integrated files are committed in this downstream fork. From the repository
root, use the one-click launcher:

```text
START-HERMES-WORKSTATION.bat
```

or run the phases individually:

```powershell
.\workstation\install.ps1
.\workstation\doctor.ps1
.\workstation\start.ps1
```

Normal `install.ps1` validates the committed integration in read-only mode,
prepares isolated dependencies/runtime directories, and fails if the checkout
changes. It does not apply or repair core patches. Migration helpers remain
explicit maintainer tools only. The complete downstream delta is documented in
`UPSTREAM_DELTA.md`.

## Functional in this foundation

- first-class `/browser` route and Browser item in Hermes Desktop navigation;
- embedded Chromium using Electron's own Chromium via `WebContentsView`;
- persistent dedicated browser profile outside the repository;
- multiple tabs, navigation chrome and in-app human control;
- browser tabs survive route changes and task-owned tabs are parked for background work;
- reduced frame rate for parked task tabs;
- loopback-only authenticated browser controller with a random bearer token;
- Hermes `browser_*` tools route to internal Chromium first;
- CDP-based click/type/scroll/key input so agent automation does not depend on window focus;
- configurable fallback routing to official extension/legacy lanes;
- fail-closed behavior after a task/session binds to internal Chromium;
- cloud metadata/IMDS security floor plus recognizable-secret URL/search blocking;
- top-level navigation/redirect protection for metadata targets inside Electron;
- cache maintenance that preserves cookies/localStorage/IndexedDB login identity;
- Pause, Resume, Stop, Focus Browser, Take Control and Release Control foundations;
- Kanban bundled plugin enabled by default for the Workstation distribution;
- stable/edge component pinning, upstream strategy, license policy and CI workflows;
- task/report/event schemas, routing/safety/health contracts, procedural-memory and perception interfaces;
- eval matrix scaffold for internal Chromium, extension, agent-browser and browser_exec lanes.
- first-class BrowserTask lifecycle with one-live-page ownership, explicit
  destroy, parking, crash recovery and logical restart restoration;
- composite BrowserSessionState for ordinary/task logical tabs, order, active
  state, sanitized metadata, atomic convergence and lazy task recovery;
- repository-root one-click install → doctor → start dogfood flow;
- pre-V1 #1.5 Mainline Consolidation Gate and recurring review contract.
- V3.1–V3.4 contract layer: EvidenceState/event/resource projections, deadlines,
  independent RuntimeSupervisor, Recovery Plane CLI, deterministic routine
  promotion/replay, persistent worker lifecycle, temporal memory/snapshots,
  session ownership/migration/compaction, portable replay/fork, Control Plane
  isolation, protocol adapters and model-independent evaluation gates;
- read-only `python -m workstation.release_qualification` runner with bounded
  stage evidence and candidate-matched clean-install evidence validation,
  including an explicit absolute isolated Workstation home outside the
  candidate checkout;
- read-only `python -m workstation.desktop_load_evidence` validator for the
  configured H013 candidate profile;
- bounded-duration `SoakRunner` mode with completed-iteration and timeout
  evidence plus the canonical session/lease/journal/worker reconnect scenario.
- versioned H011 hidden-window native Browser runtime reconnect soak covering
  real Electron/Chromium task pages, host transitions and composite restart
  identity;
- versioned H012 real headless `hermes serve` multi-session/reconnect soak
  covering authenticated WebSocket sessions, streamed turns and durable resume;
- versioned H013 integrated hidden-window Desktop/Browser load E2E covering four
  native Chromium BrowserTasks plus an eight-task/three-round sustained load,
  complete controller/IPC resource-event projection parity and
  hide/park/destroy cleanup;
- the Windows release workflow runs that reconnect scenario across four
  sessions for a 180-second budget, validates the report and uploads its JSON
  plus durable evidence root.
- the same workflow runs H012 across four concurrent sessions with two backend
  restarts and uploads its JSON plus isolated durable home evidence.
- the same workflow runs H013 with `HERMES_DESKTOP_E2E_HEADLESS=1`, selects a
  bounded 16-task/120-second release profile with eight backend chat turns and
  validates its bounded JSON evidence report and requires the integrated
  Desktop/Browser result in the final gate aggregator.
- the same workflow executes `workstation\\doctor.cmd -Strict` and requires its
  original outcome in the final gate aggregator.

## Remaining validation gates

The original foundation staging list is now implemented on the current line.
The following evidence boundaries remain intentionally explicit and are not
claimed complete from contract tests alone:

- clean-machine release qualification and last-known-good promotion;
- completing V3 resource/event parity across future client surfaces beyond the
  current Desktop IPC, Dashboard REST and TUI JSON-RPC adapters;
- extending cross-engine coverage beyond the validated Chromium/Firefox/Edge
  Dashboard smoke (and Electron/Chromium packaged boundary);
- candidate-release confirmation of full-duration/production-scale
  Desktop/Browser multi-session/worker/reconnect load beyond the locally
  validated H013 profile and the bounded Python/native-browser/headless-backend
  evidence;
- autonomous multi-agent swarm arbitration across remote physical hosts.

No second task/session/memory database is introduced: future work must extend Hermes' existing sources of truth.

## Validation performed before packaging

Latest promoted evidence:

- read-only core integration anchors, lockfile and license policy: PASS;
- Workstation Python contracts on the current V3 working tree: **175 passed**;
- production dependency audit after lockfile refresh: **0 vulnerabilities**;
- isolated clean candidate install/doctor/production build plus release
  qualification: **6/6 stages passed**; hidden H013 candidate profile:
  **2/2 passed** with an accepted 16-task report;
- normal Windows install kept the committed checkout clean;
- complete Desktop workspace typecheck: PASS;
- BrowserSessionState lint/format: PASS;
- focused Browser foundation: **5 files / 46 tests passed**;
- native Windows/Electron H010 on the current HEAD:
  `H010_CLASSIFICATION=VALIDATED`;
- native hidden-window Electron/Chromium H011 reconnect soak: **39 episodes /
  936 task cycles / 4 tasks / 60 seconds**, `H011_NATIVE_BROWSER_RUNTIME_CLASSIFICATION=VALIDATED`;
- real headless `hermes serve` H012 reconnect soak: **12 cycles / 48 turns /
  44 reconnects / 2 backend restarts / 48 heartbeats / 192 streamed events /
  0 errors / 4 concurrent turns**, `H012_HEADLESS_BACKEND_CLASSIFICATION=VALIDATED`;
- H013 integrated Desktop/Browser headless load E2E: **2 passed (36.6s)** with
  the four-task identity scenario plus an eight-task/three-round sustained
  load, real backend chat turns, complete controller/IPC resource-event parity,
  host-aware viewport geometry/transfer, native maximize/restore reconciliation and lifecycle
  cleanup; a local scaled 12-task/15-second profile also passed **2 tests
  (55.4s), and the exact 16-task/120-second/8-turn profile passed **2 tests in
  3.0 minutes** with 170 rounds; the Windows workflow remains the
  release-candidate gate;
- native V3 process-boundary smoke: `V3_RUNTIME_HARDENING_CLASSIFICATION=VALIDATED_CONTRACT_BOUNDARY`;
- local production build, unpacked packaging, NSIS assembly and artifact
  structure validation: PASS;
- packaged GUI/HUD Playwright smoke: **6 passed (22.5s)** in the headless E2E
  mode — H-021/KI-009 remain resolved with lazy Workstation Browser startup and
  per-sandbox state isolation; the run also compared the exact `win-unpacked`
  Desktop IPC resource/event envelopes with the authenticated loopback
  controller, without revealing a native window;
- Windows workflow source now installs the cross-engine browser dependencies,
  runs Dashboard cross-engine smoke, builds the unpacked package and runs the
  packaged GUI E2E, with each outcome required by the final gate-preservation
  step;
- shared Workstation resource/event projection: Desktop IPC, Dashboard REST and
  TUI JSON-RPC adapters are covered by focused resource/runtime checks;
- Dashboard REST and TUI JSON-RPC high-level adapters share one authenticated
  live HTTP resource/event projection: **5/5 client-boundary tests passed**;
- cross-engine Dashboard smoke: **3 passed (9.2s)** across Chromium, Firefox and
  the installed Edge browser against the real provider-free Python backend;
- sampled canonical reconnect soak: **3/3 fresh-process iterations, 3 sessions,
  0 failures, 36 journal events, 9 memory snapshots and 6 worker
  reconstructions, 6 model changes and 6 cold reloads**; long-duration
  Desktop/Browser soak remains an external gate;
- extended local reconnect soak: **77 fresh-process iterations, 0 failures,
  1,155 actions and 924 journal events** in the 60-second budget; this remains
  local contract evidence, not production-like Desktop/Browser evidence;
- latest extended local reconnect soak: **128 fresh-process iterations, 0
  failures, 2,560 actions, 2,048 journal events, 15,888 cumulative memory
  records, 512 snapshots and 508 worker reconstructions** across four sessions
  in the 180-second budget, with 508 model changes and 508 cold reloads; live
  memory peaked at **32 records** under the explicit eight-record-per-session
  bound; this remains local contract evidence, not production-like
  Desktop/Browser evidence;
- broad Desktop UI: **591 files / 5,669 tests passed**;
- broad Desktop platform/Electron: **126 files / 1,761 tests passed, 5 skipped**;
- the prior KI-006 Windows path/permission/SSH/WSL/locale failure class is closed
  by HW-018 without disabling or deleting coverage.
