# Source update completion ownership

## Phase seam

The command process owns admission, the update lock and output lifetime, pre-update
inventory, all-profile snapshots, gateway pause, Git selection/stash/restore and
syntax/HEAD guards, and the ZIP download/stage/dirty recheck/release graft/swap.
It imports the completion transport before swapping code. Once the final tree is
selected (including upstream merge), Git, already-current retry and ZIP all send
one versioned JSON request to `update_completion.py` **from that tree**. No cached
application module is evicted or reloaded in the command process.

The request carries canonical source/home, desktop product selection, interactive
and gateway mode, pre-update version, active and sibling snapshot identifiers,
serialized runtime plan, open receipt identity/data and paused-Windows token. It
contains data, never callables or pickles. stdin stays inherited for interactive
configuration prompts; gateway mode retains its non-interactive behavior. Child
output stays visible and is mirrored by the parent's update output stream.

## New-code owner

A stdlib-only entrypoint starts using the available Python with `-I -S`, so no
old site-packages or executable `.pth` files initialize. A private bytecode-cache
prefix fences stale cache files before any new-checkout imports. Its explicit
import path points at the new checkout. It calls the new PM interface to prepare the
recorded dependency union, then starts the selected Python with the new activation
environment. That interpreter also starts with site initialization disabled,
then the runtime owner leases and activates its selected generation before any
application imports. Only that interpreter imports application completion code. The same
receipt/correlation identity crosses this preparation boundary (including PM
results). Selected-Python completion owns launcher publication, builders, cache
invalidation, all-profile configuration/state/skills maintenance, process scans,
fleet restart, Windows resume, dashboard deduplication and verification.

The existing per-kind restart and abort-recovery algorithms remain; transient
supervisor/process failures are real even without mixed-generation imports. Only
the purge/reload workaround and independent retry/ZIP tail compositions disappear.
Gateway exit status is written before a restart can terminate the updater's cgroup.
Verification publishes the final receipt.

## After the commit point nothing fails the update

Once the tree has moved, `hermes update` exits 0 unless the code itself was rolled back.
Every completion step is independent: a failed launcher publish, product build (each of
TUI/web/desktop is attempted even when another failed), config migration,
bytecode sweep, gateway restart/verification, Windows resume or retired-channel adoption
prints a `⚠` line and is appended to the receipt's `followups` as `{step, reason}` while the
receipt's `outcome` stays `"success"`. The step's own obligation stays armed:
`source-completion-pending` for the tail (launchers, build, maintenance, config migration),
the host fleet-restart obligation for gateways (the CLI startup warning keeps naming it; an
owed restart records the pre-update gateways on the obligation, so a gateway that died at boot
stays owed until it serves the checkout instead of being settled by the gateway-less discharge),
the unstamped bytecode fingerprint for the sweep. The next launch or `hermes update` —
including the "Already up to date" path, which runs the same completion — retries it. A host
stamped "restarted" for this commit whose fleet is still off the checkout restarts again
instead of dead-ending. Exit 2 (refused / concurrent) and exit 1 (nothing committed, or
rolled back) keep their meaning.

Profile sync is best-effort: `_sync_profiles_after_update` prints a per-profile error and
carries on, so that error is not owed. Only a sync that escapes the step (for example with
`SystemExit`) becomes a `profile_sync` follow-up and keeps `source-completion-pending` armed.

A Windows gateway resume is attempted once after the commit point: by the completion child, or
by the parent when dependencies are owed. Its failure is the `windows_resume` follow-up. The
command's own exit path and its atexit net do not run it again, and a resume they still owe
(the child never answered) is reported the same way instead of raising. The historical takeover
completion (`update_finish`) also records the failure as a follow-up and keeps its exit status.

The receipt is durable while the run is open: `begin_update_receipt` writes it as
`outcome: "running"` to the run's own archive file and `latest.json`, and each stage
boundary refreshes it, so a killed update leaves its own record. The next update marks a
`running` record whose processes are gone `interrupted` (naming its last stage) and reports
it. Receipts and the `update.log` tee resolve to the root home
(`hermes_constants.get_default_hermes_root()`), never a sticky profile's; so do their readers
(`hermes logs update`, the debug bundle, the dashboard's update status, pm's sync receipts).

The completion bootstrap's dependency preparation (`ensure_tools_for_sync`, `pm.sync_venv`) also
runs after the tree moved: its failure is a `dependencies` follow-up ("dependencies not installed
yet — the next launch retries"), exit 0, with the tail obligation armed; a prepared child that
dies without a result is a `completion` follow-up. A Ctrl-C after the commit point closes the run
as `interrupted` (exit 130, never `failed`) and says the new code is in place with its remaining
steps owed.

Two more post-commit channels follow the same rule. The gateway `/update` marker
(`.update_exit_code`) keeps the committed result when a gateway restart fails (systemd unit,
abort recovery or Windows service resume): the restart debt is the `gateway_restart` follow-up,
the host obligation and the `⚠` lines in the forwarded output, never a "failed" notice. A receipt
store that refuses the terminal write prints `⚠ Update receipt not written` and the completion
child answers its parent with the correlated terminal record it finalized in memory, so the exit
status stays 0; only a user action (local changes left in the stash) still exits 1.

## Parent lifecycle and failures

The parent waits and propagates the child's exact nonzero result (a signal is
mapped to shell-style 128+signal); post-commit step failures never produce one. A child cannot succeed by merely exiting zero:
a terminal response with the matching receipt identity is required. The response
returns the mutated Windows token so the parent's registered emergency resume does
not repeat completed work. Normal parent completion performs no maintenance.

The parent retains its original receipt until acknowledged child finalization;
missing/failed child output leaves it available to the existing command-boundary
failure finalizer. The stdlib bootstrap returns correlated PM failure data even
when application imports are unavailable, and normalizes negative signal exits
at each process boundary. POSIX completion owns a new session/process group;
cancellation kills that group before releasing the lock (Windows uses the retained
child's `taskkill /T` tree). The parent records the pending fleet obligation before
starting the completion process, including when preparation cannot begin. The parent's emergency Windows resume remains a last-resort
lifecycle obligation when the child cannot execute or is killed. A failed child
never clears the pending fleet obligation. No automatic code rollback after
maintenance has begun (SQLite snapshots remain file-loss recovery, not rollback).

### Damaged recovery code

Launch-time checkout repair requires both the checkout lock and the child-custody
runner to be importable. If either module is torn, launch stops with
`Cannot safely repair` and retains the interrupted-update marker instead of
writing without exclusion. An updater that already imported healthy code can
still hold and write the checkout even when its on-disk modules are damaged.
Wait for any running update to finish.

The repair code lives in the tree a killed move tears (`hermes_bootstrap.py`,
`hermes_cli/__init__.py`, `_early_recovery.py`, `update_lock.py`,
`update_custody.py`). Before git writes, every tree move publishes those
recovery/lock/custody modules, as committed at the move's starting commit, to
`<git dir>/hermes-update-recovery/<pre>/`, outside the working tree and keyed
to the marker's `pre` (a full commit id, nothing else), with a `MANIFEST` of
each file's blob id; files and directory are fsynced before the rename. When
the checkout's own import fails while the marker exists, the minted launcher
(`.hermes/bin/hermes`) runs that published copy only if every file hashes to
its manifest id. A missing, torn or foreign copy (or none, from an older
updater) is rebuilt first from git's objects at `pre` (`--no-replace-objects`,
the recorded git or an absolute `PATH` entry, never the current directory) and
verified the same way. Then: stdlib plus the copy only, the same restore claim
and checkout lock (a live writer gets `Not repairing the checkout now` and exit
1, no traceback), then a relaunch from the restored tree. Limits: other entry
points (`python -m hermes_cli.main`, the `hermes-agent` hook) have no such
fallback; a launcher minted before this fallback existed has none, so a kill
during the first update onto this code (run under the old launcher) is not
covered; a `pre` whose tree lacks the checkout lock or custody runner is not
used.

### Behaviour changes on the git path

Intended differences from the pre-transactional updater, for release notes:

- The Windows ZIP fallback runs only from an untouched checkout. Once git has
  stashed, switched or half-moved the tree, the ZIP overlay (which replaces
  every top-level entry) would bury that state, so the run fails with the git
  error instead.
- The commit point arms the completion tail, the fleet restart and the marker
  before git writes. If any of them cannot be made durable, the update stops with
  the checkout unchanged: moving without them is the "tail never runs" state.
- Startup files (the launcher and `hermes_bootstrap` import closure) are
  compiled at the target commit before the move, by an interpreter the target's
  `requires-python` admits (`uv python find`, then `python3.N`). With none
  installed the refusal names the Python to install. The preflight is one
  `git cat-file --batch`.
- Every move names the resolved commit id, not a ref, and its marker is dropped
  only once HEAD is that commit. A branch that moves during the switch is
  reported as a failed switch. The restart is owed for the HEAD the switch
  actually landed on.
- The fork's upstream fast-forward is judged the same way before it moves. A
  broken upstream commit rolls the whole update back.
- A recovery marker whose target commit is gone (gc, re-clone) is retired only
  over a clean tracked tree at `pre`. Until then each launch warns.

## Historical surface

All names frozen from the complete reachable shipped updater history stay
resolvable. Historical dependency hooks retain the stdlib-only takeover bridge:
the old parent waits, carries receipt/recovery state and never resumes a retired
installer. Newly retired preparation and module-reload hooks explicitly marked
incomplete stop nonzero and request `hermes update` again; they cannot manufacture
a missing completion request. Current Git/current/ZIP callers use only the
canonical completion transport, not the historical takeover entrypoint.
Unfrozen branch-only retry compositions are deleted, not shimmed. ACP convenience
publication uses the launcher owner's `expose_cli`; the historical ACP entry is
only an adapter, never a second writer. The frozen set is never trimmed or replaced
with tag-only coverage. New current-path imports are unioned with that history.

## Verification

Use isolated homes, disposable Git repositories and fake dependency/build/service
adapters only. Exercise an old process with cached incompatible modules across a
real Git transition to new code, selected-Python execution, receipt identity and
snapshot transfer, nonzero/abrupt child exit, lock release and Windows-token
return. Focused existing tests cover dirty ZIP checks/grafts, snapshots, fleet
reconciliation, supervisor timing and historical imports. Native service restart
and Windows/macOS acceptance remain separate required lanes; no live user service
or user state is touched by this implementation's test runs.
