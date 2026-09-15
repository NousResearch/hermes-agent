# PM audit remediation status

This is a historical audit record, not a current-head release status page.
Counts and native receipts below apply only to the revisions they name.
Subsequent work adds shared bundle assembly, signed-package E2E, the stable
release gate, Termux APT distribution, and Python 3.14. Those changes do not
retroactively extend earlier test receipts.

For current implementation contracts, use [Package management](../website/docs/reference/package-management.md),
[shared bundle builds](shared-bundle-builds.md), [stable releases](stable-releases.md),
and [bundled update acceptance](../tests/install/BUNDLED_UPDATES.md).

This change integrates repairs from the aggregate branch audit. It is not a
release certificate. The audit compared `49945b14029e09fef608db9ede899377cdb54e11`
with merge base `433f7196760e5d76f77ea6d5464ee7f1b602a4ee`.

## Runtime repairs after the documentation audit

The audit found a Python-layout mismatch in Electron, a fixed Python family in
Nix, contradictory cryptography requirements, and failures in clean Windows setup.
The follow-up changes the owning implementations:

- The bundle builder checks the completed payload and publishes its launch paths.
  Electron consumes that contract without payload probes, adoption, or creation.
- Nix selects Python’s major/minor from `pm/lock.json`. The uv2nix environment,
  package overrides, extra packages, and developer shell use that family.
  Real Nix evaluation and build acceptance remain CI gates.
- The direct cryptography requirement and security override match the patched
  version in `uv.lock`. The override still bypasses the vendor SDK’s stale cap.
- The bootstrap waits for uv to finish, then runs PM through Python directly.
  PM can replace its uv entry without the bootstrap holding its executable.
- PM receipt publication uses stdlib-only atomic JSON writes under a shared lock.
  Failure reporting no longer imports PyYAML through `utils`.
- The unused development-shell command is removed. Setup plus activation and
  `deactivate` is the single PM developer path.

A native Windows check started with an empty tool store in a disposable home.
Setup completed, PM published a dependency generation, and PowerShell activation
ran the store interpreter and source CLI. A separate PM installation passed
its real dependency check. These checks do not prove signed-package installation,
updates, or Nix builds. The broader PM plugin/config YAML coupling remains unchanged.
The [developer workflow](../website/docs/reference/package-management.md#developer-workflow)
distinguishes runtime activation from independent test and editor environments.

## Implemented repairs

| Findings | Implementation |
| --- | --- |
| C01, C11, C12, C24 | Remove the deleted TLS import; fix ACP file decoding, proxy path ownership, and the monitoring SDK call. |
| C02 | Restore code-execution schema, registration, and child lifecycle handling. |
| C03, C04, C17 | Keep update metadata commands read-only; synchronize plugin updates; compare feed identities correctly. |
| C05, C27, C28 | Include managed Python in the install closure; repair installer stage dispatch and runtime selection. |
| C06, C07, C09 | Remove the unsupported SCM surface. Retain Scheduled Task supervision and protect unrelated runtimes on desktop quit. |
| C08, C19 | Correct the App Installer checker and restore meaningful update states. |
| C10 | Repair uninstall dispatch and owned-link removal. |
| C13, C14 | Prepare one complete dependency environment before publication. Preserve declared constraints and explicit pins; permit compatible transitive upgrades. Infrastructure failures do not disable plugins. |
| C15, C18 | Retain recoverable tool-store entries on failed publication. Use atomic facts/config writes and context-local receipts. |
| C16, C29 | Migrate removed dependency helpers and restore missing-dependency hints. Run the embedded Hindsight service from its separate environment. |
| C20, C21 | Correct platform-test selection and make the removed-import guard inspect actual source. |
| C22, C23 | Publish immutable release artifacts before feeds and resolve FFmpeg artifacts by target. |
| C25 | Preserve paused downloads, resume handles, and single-worker ownership. |
| C26 | Wire the boot and plugin-cadence paths to their existing owners. |

Implementation does not mean that every platform acceptance test is complete.
The updater, backup, setup, voice-text helpers, and several plugins now use
one implementation per reconciled concern. Runtime paths have a shared owner
in `hermes_cli/runtime_paths.py`.

## Closure implementation

| Contract | Owner and proof |
| --- | --- |
| Interrupted publication | `hermes_cli/runtime_state.py` journals the old config and the proposed config hash. Startup recovers before dependency activation. Recovery refuses to overwrite unrelated edits. Real subprocess tests terminate before and after facts publication. |
| Live-generation collection | Startup holds a generation lease under the publication lock. `pm gc` removes only unselected, lease-managed generations without live readers. Tests keep a real reader alive while collection runs. |
| Receipt correlation | PM completions carry the invoking update ID. The updater embeds that completion, including failed steps and refusal reasons. Nested commands and copied contexts cannot finalize an enclosing receipt. |
| Warning surfaces | Doctor uses the update checker's local provenance rules. The desktop reads sync status and distinguishes a healthy no-op from an embedded failure. |
| Updater ownership | `electron/updater/checkout.ts` owns checkout checks and handoffs. `main.ts` supplies its dependencies. |
| Windows relaunch | A detached PowerShell waiter snapshots package identity before quit, waits for the original process to exit, then waits for the installed version to change. A failed registration produces a manual-reopen warning. |
| Test integrity | TCC behavior uses actual host markers. Installer source-grep assertions were removed. Runtime and installer behavior tests remain. |

## Verified execution

- The latest complete native-Windows Python run reported 44,557 passed, one
  failed, and 1,404 skipped tests across 3,746 files. Python source hashes stayed
  unchanged throughout the run. It also reported one pass-on-retry HTTP test.
  The install-ID race and HTTP test were then fixed. Their two files passed
  35 tests without retries. This is not a complete final-tree suite pass.
- Root `npm run check` completed with exit code zero, including MSIX packaging.
  Desktop UI: 7,231 passed. Electron: 2,311 passed and 23 skipped. TUI:
  1,719 passed and eight skipped. Dashboard: 291 passed. Root JS: 77 passed.
  Type checks and lint had no errors. Existing lint warnings remain. This
  packaging check used the primary build directory's existing payload, not
  the separately verified fresh audit payload.
- The actual core project resolved and built through the workspace helper.
  Imports resolved from the generated workspace and the source lock was unchanged.
- A fresh native ARM64 PM bundle completed its pinned-tool checks and all-extras
  environment build. Its manifest records source tree
  `60c9fb444c93e8a79ae22a01677c3291385343a6`.
- The rebuilt thin desktop started its real backend twice in an isolated home,
  answered HTTP 200, and exited cleanly. Home entries survived relaunch.
- A real isolated API-server messaging gateway retained its PID and birth time
  after ordinary desktop quit. The desktop backend stopped; the gateway did not.
- A thin MSIX built with the real packaging toolchain. Windows Sandbox installed
  version `0.17.0.0`, updated to `0.17.0.1`, and verified absence after uninstall.
  Test-certificate creation and trust were confined to the disposable guest.

- A separate fresh bundled MSIX was produced from payload tree `60c9fb44...`.
  The unpacked artifact's own CLI ran, imports resolved inside its payload,
  and `hermes serve` answered HTTP 200. The Sandbox deployment attempt failed
  on an incorrect unpacked path. It does not prove bundled installation.
- Plugin checks now run from the first housekeeping tick, with the configured
  interval gate controlling network checks. A real isolated gateway wrote two
  successful plugin-check receipts one tick apart. Auto-apply was disabled.

These receipts cover different layers. Thin-package deployment and unpacked
runtime startup do not prove bundled installation or App Installer-triggered
relaunch. No tests sent an LLM request as evidence of this acceptance pass.

## Merge-ready closeout

The branch includes upstream `5bd439d3ed4ae5f099857813383389dcd0ab4369`.
The following limits from the earlier audit are resolved:

- Context-only homes share one dependency root across journal recovery,
  selected runtime state, and plugin unions. The regression batch passed
  71 tests with eight host skips.
- `tests/tools/test_subagent_steer.py` produced the relative `MagicMock`
  databases. Its mock parents now explicitly have no database. The delegate
  verification batch passed without new debris. Earlier debris is archived
  outside the repository.
- Docker bootstrap includes the stdlib runtime-path and locking owners that
  PM imports before third-party dependencies exist. Both architecture builds
  passed after this fix.
- Installer path probes no longer read a missing lockfile. PowerShell 5.1 and
  7 installer tests passed in CI.
- Service runtime selection uses the shared selected-environment resolver.
  PM facts supply managed Node paths. Native fixtures use disposable homes.
- The Windows updater uses its registered App Installer source unless an
  explicit feed override is configured. It downloads the descriptor before
  teardown and opens the local file. It does not require the disabled
  `ms-appinstaller:` protocol.

## Non-E2E closeout

The full CI workflow on `a27cd5902a446eb4f1e457e09904d75edc616c2e`
completed successfully:

- Linux: 46,041 passed, zero failed, 492 skipped. No retry-only flakes.
- macOS: 94 passed, zero failed, 121 skipped. No retry-only flakes.
- Windows: 178 passed, zero failed, 174 skipped. One pipe-drain fixture
  passed only on retry because its cold child exceeded the idle window.

Commit `04182813cfa74111c6e4e10bedcc0f9de6510f55` limits Windows
nested-process concurrency without relaxing the fixture's assertions.
It also fixes compute-host stdin framing: a real Windows child received
its initial turn but did not receive subsequent control frames through the
text stream. The byte-stream reader passed the actual interrupt and
second-turn tests. Those tests now run in each native OS lane.

The other test repairs distinguish worker scheduling from the behavior
under test. Compression waits for engine entry, cron assertions observe
blocked work, orphan teardown asserts that the resume lock is free, and
hosted-room tests observe durable settlement after a concurrent snapshot.
Targeted parent verification passed 638 gateway/delegation tests, 41
cron/hygiene tests, four compression isolation tests, 53 hosted/compute
tests, and 19 real-child/compute protocol tests. These batches overlap;
they are not a whole-suite total.

The browser BOM regression failed before the read fix and passed after it.
Its file passed 79 tests with two host skips. The full Windows-footgun and
plugin-compat checks passed.

## Bundle acceptance (separate workstream)

The actual bundled Windows ARM64 MSIX built from
`07ee9299790e5635fd917da1769f969f14fc8030` installed in a disposable
Windows Sandbox as version `0.17.0.0`. Its registered app presented the UI,
About identified the embedded runtime, and its own packaged backend
answered HTTP 200. The unsigned build artifact's SHA-256 is
`31d3dfb53977e0e8fd662d0ef2475a7be92a31a49d6e514ceed812099867f304`.
Test signing and trust were confined to the guest.

The guest was terminated before update/relaunch verification completed.
There is no automatic-update acceptance claim. This artifact also predates
subsequent upstream integration and the compute-host fix. Bundle E2E work
belongs to the existing install/update test family, not a second local
Sandbox pipeline.

## Remaining gates

- Final-head native CI must confirm the latest fixes without retry-only failures.
- Maintainers must decide the `needs-decision` disposition and apply
  `ci-reviewed` after review. Manual CI skips PR-only review gates.
- Automatic package update/relaunch remains with the separate E2E workstream.

PR #95281 was closed as superseded by #102765, following the triage request
for one canonical PM PR. The duplicate label was removed. No approval
label was self-applied.

[Verified CI at a27cd5902a](https://github.com/NousResearch/hermes-agent/actions/runs/34059149746).
The exact-head validation uses the same workflow on an upstream validation
branch. Earlier GitHub graph failures reported
`resource_exhausted: gitmon refuses to schedule us: fail-fast:network`.

The external audit directory contains original findings, exact test selections,
per-run logs, source snapshots, and review adjudication. Host application,
certificate trust, and production services remained unchanged during that audit.
That audit did not publish a release; this is not a statement about later runs.
