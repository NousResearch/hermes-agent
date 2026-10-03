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
Gateway exit status is written before a restart can terminate the updater's cgroup,
and is demoted on later failure. Verification publishes the final receipt.

## Parent lifecycle and failures

The parent waits and propagates the child's exact nonzero result (a signal is
mapped to shell-style 128+signal). A child cannot succeed by merely exiting zero:
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

## Crash-cell matrix

Each cell kills a real `hermes update` (or the Desktop hand-off script) at one point
of the update, then asserts what the user is owed: the next `hermes` launch is
runnable, the checkout is at the pre-update commit or the target, and nothing the
dead update left (a git lock, `.hermes-update-in-progress`) blocks the next update.
A cell whose fix is an open PR wraps only its final assertions in
`known_failure` (`tests/e2e/core/_pending_fixes.py`): it XFAILs on exactly that gap's
message, fails on anything else, and passes once the fix lands, whichever merges
first. Kill points are observed states (a git child in the process tree, HEAD read
from the ref files, the hand-off's update child plus its marker), never sleeps.

| Cell | Kill point | Test | Fixing lane |
|---|---|---|---|
| Windows `mid_git` | `taskkill /T /F` while the update's git child (fetch / merge / reset) runs | `tests/e2e/core/windows_update/test_crash_cells.py::test_update_killed_mid_git_leaves_a_runnable_install` | #132361 (stale `.git/index.lock`; launch-time interrupted-pull repair ran a bare `git` a machine with only the installer's Git does not have) |
| Windows `tree_moved` | right after the checkout reached the target, before the update finished | `tests/e2e/core/windows_update/test_crash_cells.py::test_update_killed_after_the_tree_moved_leaves_a_runnable_install` | green on main |
| Windows `desktop_handoff` | `scripts/desktop-update/windows.ps1` and its whole tree while its `hermes update` child runs | `tests/e2e/core/windows_update/test_crash_cells.py::test_desktop_handoff_killed_mid_run_leaves_a_runnable_install` | green on main |
| Windows `orphaned_update` | only `windows.ps1` (no `/T`); its `hermes update` keeps running and must finish with the marker LIVE until it exits, then gone | `tests/e2e/core/windows_update/test_crash_cells.py::test_desktop_handoff_script_killed_alone_keeps_the_marker_live_until_its_update_ends` | #132354 + #132365 (line-4 delegate) |
| POSIX commit points | per lane | `tests/e2e/core/upgrade/<area>/test_hostile_<lane>.py` | the lane that owns the file |

The Windows cells run in the Windows install + update journey
(`.github/workflows/windows-install-update-e2e.yml`); push a `wine2e-install/**`
branch to run them on demand. Both real-update suites are required on a pull
request whenever the change classifier's `e2e_upgrade` lane fires (any file on
the update path: `scripts/ci/classify_changes.py`), and the Desktop update suite
whenever `e2e_desktop_update` fires; they are skipped, and count as passing,
otherwise. Related: [macOS bundle updates](macos-bundle-updates.md).
