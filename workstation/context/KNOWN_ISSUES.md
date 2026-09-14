# Workstation Known Issues

This file records observed/reproduced gaps and the evidence boundary around them. A listed symptom is **not** permission to assume a root cause; verify current `main` and any explicitly named candidate before changing code.

## KI-002 — Preview and Workstation Browser are separate browser lanes

**Observed:** Preview and the main Browser surface can show different independently navigated pages for what the user regards as one web task.

**Do not fix by:** synchronizing URLs between two pages.

**Target invariant:** Workstation-mode Preview compatibility and Browser Hub/Chat Browser View reference one BrowserTask/live page.

**Current status:** the MVP invariant is resolved. Chat Browser View, Browser
Hub and Workstation Preview reuse the same BrowserTask/runtime identity, and
the single-host transfer path is covered by
`apps/desktop/electron/workstation-browser-runtime-viewport.test.ts` plus the
packaged GUI smoke. Full Preview action parity, richer layouts and exhaustive
reconnect/compatibility coverage remain follow-up hardening rather than a
second browser lane.

## KI-004 — Native browser surface can overlap another Desktop pane

**Observed:** during bootstrap validation, independent Preview and Workstation Browser surfaces could visually overlap and resizing changed the overlap.

**Current evidence:** the Browser renderer already uses `ResizeObserver` and `getBoundingClientRect`, so “add another resize observer” is not an established fix.

**Target invariant:** one live `WebContentsView` host at a time plus an explicit host/viewport ownership contract; validate resize, maximize/restore, sidebar/pane changes, and host transfer.

**Current status:** the shared-surface overlap invariant is resolved.
`viewportHost` and `transferViewport()` move the same `WebContentsView` between
Hub and Chat, while host-aware `setBounds` rejects stale geometry from a
non-owner pane. The focused runtime tests cover host rejection, zoom/clamp
normalization and cross-window rehoming. The runtime also reconciles the active
view on native window resize/maximize/restore events, and H013 validates that
matrix against a real hidden Electron/Chromium view. More complex pane races,
compositor transitions and broader native composition evidence remain open
hardening work.

## KI-006 — Broad Windows Desktop suites contain pre-existing portability/test failures [RESOLVED]

**Observed:** after the canonical-source installer reached a clean install and passing Desktop typecheck, the broad UI and Electron/platform suites remained red on Windows.

**Causality status:** **not caused by the Workstation canonical-source / BrowserTask changes based on controlled baseline evidence.** A native-Windows side-by-side comparison used baseline `ce78f120e8ed2974d6174e475cc7572afcfe41e0` and Implementation 4 candidate `2ffee2335b6aba071e7b63457a047cd9334d4d92` under the same Windows/toolchain/install/test conditions.

Observed comparison:

- baseline UI legacy: 5 failed files / 11 failed tests;
- candidate UI legacy: 5 failed files / 9 failed tests;
- baseline platform/Electron legacy: 11 failed files / 33 failed tests;
- candidate platform/Electron legacy: 10 failed files / 28 failed tests;
- remaining candidate failures were identical baseline failures or variants of the same causal class around Windows path/permission assumptions, SSH ControlPath/Include assumptions, POSIX virtualenv layout, WSL/bridge timing, staging/file-mode assumptions, and locale-sensitive UI expectations;
- candidate-specific BrowserTask suite passed 2 files / 16 tests and candidate typecheck passed.

Controlled verdict: `WINDOWS_BASELINE_COMPARISON=PASS_WITH_KI-006_RED`.

The accepted PR head `75d10d35d4757496390debf8e4b4f9efb44c5432` changed no BrowserTask product/runtime/probe/workflow/dependency code after that controlled candidate; only contributor-attribution mapping and Workstation journal material were added. Its final Windows workflow again passed committed integration validation, install, checkout-clean, typecheck and focused BrowserTask diagnostics before the final broad aggregator remained red. No new Implementation 4 failure class was introduced.

On BrowserSessionState accepted head
`d5be442021ea0c744351622317eef5212219786d`, the scoped static checks,
5 focused files / 46 tests and H010 native probe all passed. The broad run had
one UI failure from a missing `BROWSER_ROUTE` test mock and 31 Electron failures
across unrelated POSIX path/mode/symlink/SSH/platform assumptions. This is the
current exact-head manifestation of the same KI-006 class, not a
BrowserSessionState failure.

**Resolution on the current working tree (HW-018, 2026-09-11):** the
underlying Windows contracts were fixed and validated without disabling or
deleting coverage:

- deterministic English locale formatting now keeps UI output stable across
  host locales, and the new-session Preview promotion preserves tabs when the
  stored session id arrives after the runtime session;
- POSIX fixture paths use an explicit path adapter, while Windows venv paths
  use Windows grammar;
- POSIX mode assertions now remain strict on POSIX and validate the Windows
  ACL-preserving/writable contract instead of requiring meaningless Node mode
  bits;
- Git path comparisons use native physical paths and non-repository probes
  avoid leaving Windows cwd handles locked;
- SSH ControlMaster tests opt into mux behavior explicitly, while no-mux
  behavior remains covered separately;
- WSL UNC selection no longer performs a blocking synchronous network share
  probe, and native staging tests verify the requested executable mode through
  an injected boundary.

Validation: Desktop UI **591 files / 5,669 tests passed**; Desktop
platform/Electron **126 files / 1,760 tests passed, 5 skipped**; Desktop
typecheck passed. The baseline comparison above remains historical evidence;
KI-006 is no longer an open failure on this working tree.

## KI-007 — `Session not found` / exported `session: null`

**Observed:** session export/logging from the browser validation cycle included a consistent message `session_id` while the exported session object was `null`, and logs previously included HTTP 404 `Session not found`.

**Causality status:** **unproven**. Preview/browser behavior has independent reproduced gaps, so this issue must not be used as their explanation without a causal trace.

**Required proof before any SessionDB/Gateway change:** reproduce on current `main` → identify endpoint/caller/session id → determine lineage/compression/rotation expectations → identify the responsible line/race → add regression test → only then change core.

## KI-008 — V3 product-level acceptance evidence is still open

**Observed:** the V3.1–V3.4 Python contract layer is implemented and validated,
including a Windows process-boundary smoke, and bounded native Browser plus
real headless-backend reconnect soaks now pass. The roadmap still requires
clean-machine release qualification, full event/resource parity beyond the
current clients and full-duration/production-scale Desktop/Browser load. The
Chromium/Firefox/Edge browser smoke is now validated separately when Edge is
available.

**Causality status:** **not a product failure; evidence boundary is explicit.**
The contract tests do not prove behavior that they do not exercise. The
historical KI-006 Desktop portability debt is resolved on the current working
tree, while the remaining V3 release evidence gates are still separate.

**Required action:** run the Windows clean-machine release workflow (which now
emits candidate-matched install/build evidence and the configured H013
hidden-window Desktop/Browser load gate, including its bounded 16-task/
120-second candidate profile (locally validated, but still unqualified on a
clean candidate) and the full-duration/production-scale
Desktop/Browser load gate on a candidate release, and extend client parity as new supported
surfaces are added, before promoting V3.1–V3.4 from “contract layer validated”
to fully accepted. Do not mark those gates green from unit-test coverage alone;
the bounded H011/H012 probes and local H013 integrated E2E are evidence, not a
substitute for the clean-machine/release-candidate gate.

## KI-009 — Packaged Desktop GUI E2E target initialization [RESOLVED]

**Original symptom:** the packaged Windows artifact loaded its real renderer,
but Playwright's Electron `beforeAll` waited for a detached Workstation
`WebContentsView` target and ended at the 90-second hook timeout.

**Resolution:** Workstation controller startup no longer creates a browser view
until the UI or a `browser_*` action needs one. The E2E fixture isolates
`HERMES_WORKSTATION_HOME`, injects the Electron loader for the custom packaged
executable and tolerates only 1 px of native subpixel geometry rounding.

**Evidence:** the original packaged GUI gate passed **5 passed (26.4s)**;
the latest run including the shared-resource identity check passed **6 passed
(28.3s)**.

## Resolved regression classes

### KI-003 — Complete logical BrowserSessionState did not survive restart

**Original symptom:** the persistent Chromium profile and BrowserTask-only
metadata existed, but ordinary logical tab ordering, active selection, safe
restorable metadata and one composite restart projection were incomplete.

**Resolved behavior:** PR #11 promoted one versioned atomic
BrowserSessionState containing ordinary/task logical tabs, order, active state,
sanitized URL/title metadata and the BrowserTask snapshot. It preserves the
latest intended composite across failed writes, migrates legacy task state once,
parks restored tasks and lazily recreates exactly one page.

Promotion:

- accepted head: `d5be442021ea0c744351622317eef5212219786d`;
- merge commit: `e0a99ef3aba6e6d2b65c30cf3c908ee1d49c4d29`;
- focused evidence: 5 files / 46 tests;
- native evidence: `H010_CLASSIFICATION=VALIDATED`, including clean and abrupt
  two-process restart, profile separation, failed-write convergence and
  explicit-destroy failure cleanup.

**Boundary retained:** a WebContentsView/renderer heap never survives process
restart. Controller/session/run/card identity expansion and host state belong to
V1 #1.5/V1 #5, not a second BrowserSessionState owner.

### KI-005 — BrowserTask lifecycle was implicit on pre-Implementation-4 `main`

**Original symptom:** `ownerTaskId`, `taskTabs`, parking, and attach/detach existed, but there was no complete first-class `show`/`hide`/`park`/`destroy` BrowserTask contract.

**Resolved behavior on current `main`:** PR #9 formalized BrowserTask around the existing `taskTabs`/`BrowserEntry.ownerTaskId` ownership primitives without introducing a second page store. `hide` and `park` preserve a live page, `show` re-exposes it, repeated task creation is idempotent, missing pages recover under the same logical task, and `destroyTask` is explicit.

Promotion:

- accepted PR head: `75d10d35d4757496390debf8e4b4f9efb44c5432`;
- merge commit on `main`: `fada723f43613e5e0f061cab24445573ac298998`.

**Automated regression coverage:**

- `apps/desktop/electron/workstation-browser-task.test.ts`;
- `apps/desktop/electron/workstation-browser-runtime-task.test.ts`;
- focused BrowserTask step in `Workstation Browser Windows`.

**Real native evidence:** `workstation/context/engineering-journal/probes/h004-native-browser-task-smoke.mjs` ran on Windows `10.0.26200`, Electron `40.10.2`, repository head `d8acc752133b125b9619cbc7fe09199f1283a22b` and emitted:

- `H004_LIVE_DESTROY_PASS`;
- `H004_RESTART_PASS`;
- `H004_CLASSIFICATION=VALIDATED`.

The native smoke proved same-page identity across hide/show and park/show, explicit destruction, two-process logical restart restoration/recreation, exactly-one-page ownership, and structural secret isolation.

**Carry-forward evidence:** Git comparison established that changes between the native-smoke/code-bearing evidence and the final accepted PR head did not alter the exercised BrowserTask product/runtime/probe behavior. Final-head Workstation CI and focused gates passed; broad Windows red retained only KI-006 baseline debt. KI-005 is therefore closed on promoted `main`.

### KI-001 — Desktop Browser capability could disappear from the model schema

**Resolved behavior:** Desktop/GUI Browser surface capability is derived from active Gateway session context instead of process-global Desktop identity or controller reachability. Selected Workstation-routed browser schemas remain present during controller startup/reachability races, while controller health/recovery governs execution. The tool-definition cache includes the session capability fingerprint so Desktop results cannot leak into TUI/CLI.

**Security/ownership boundary:** only the narrow set of actions actually routed through the embedded Workstation Browser is force-preserved, and only when those names are already selected by active toolsets. `browser_exec` is not promoted. TUI/CLI sessions do not gain Desktop capability merely from `HERMES_DESKTOP=1`.

**Regression coverage:** `tests/tui_gateway/test_workstation_browser_schema_capability.py`.

### RI-001 — Normal Workstation install rewrote tracked source before validation

**Resolved behavior:** the normal installer treats the committed downstream tree as canonical. It validates integration in read-only mode, installs isolated dependencies, prepares runtime state outside tracked source, and compares Git status before/after installation.

**Evidence:** canonical-source validation on Implementation 2 and subsequent Workstation CI/Windows install gates.

## Closing an issue here

When an issue is fixed, move it to the resolved section with the validated behavior and test/evidence boundary. Do not delete the historical symptom if it documents a regression class that future tests protect.
