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

## Service cgroup ownership

On Linux, a mutating `hermes update` launched inside a systemd service starts
its actual updater in a transient user scope before creating the update receipt
or lock. A minimal observer in the old service only waits for the scope launcher;
it holds no update lock and runs no update pipeline. If launch fails before the
updater can publish a terminal receipt, this observer records the correlated
failure and propagates the nonzero exit without an unsafe fallback. It checks
only that action's terminal update archive, never a shared PM/latest pointer, and
cannot replace an already-published child terminal receipt with launcher status.
Once launch succeeds, the whole updater (including its completion parent, lock,
and output streams) is outside the launching service's control group. Restarting
that service may kill the observer, but not the actual updater or its terminal
receipt owner. Original interpreter arguments, working directory,
profile environment and standard streams remain intact. On systemd 254 or newer,
`--expand-environment=no` preserves literal dollar expressions in argv; confirmed
older versions retain their literal scope behavior without that unsupported flag.
An unknown version keeps the flag and never retries with environment expansion or
outside a scope. A new process session alone does not provide this isolation.

If a required scope is unavailable, the updater records a failed receipt and
refuses the update. Run the update from an external shell, or restore the
service user's systemd session bus. Ordinary external-shell updates and the
non-updating `--plan`, `--check`, `--list-venv-holders`, `--install-id` and
`--set-channel` modes do not require a restart-safe scope. Global `--version`
returns before update isolation even when an `update` subcommand is present.
Explicit `--no-gateway-restart` also bypasses mandatory scope availability: it
updates code without restarting service/cron runtimes. Operators must arrange
that later restart separately; completion reports code-only success, not a
fully updated fleet.

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
Gateway exit status is written only after the correlated terminal result is
published. Final success output and the completed-action marker follow fleet
verification and a successful terminal receipt. An explicitly deferred gateway
restart reports code-only completion, not a fully updated fleet. After a dashboard
restart loses its process registry, action status reads the matching terminal
update receipt; an old completion marker alone is not success. Dashboard admission
atomically replaces and syncs `logs/hermes-update-action.json` before spawning.
The record preserves the exact action id independently of bounded output tails,
so a crash before the updater's own banner cannot expose an older successful receipt.
Legacy timestamp banners are ordering barriers, not action ids; recovery requires
an exact completion-id/receipt match and refuses ambiguous ordering. An unrelated
PM receipt at `latest.json` is excluded even without an action id and does not
replace the exact update's archived receipt.

After process-registry loss, unfinished attempts remain `pending` or `unknown`.
The dashboard consumers keep polling and reject another attempt's terminal id.
They request status for the exact admitted `action_id`. If a newer admission
replaces it, the endpoint reads the requested attempt's archived terminal receipt.
A superseded attempt without terminal evidence stops polling with an unknown
outcome; it never adopts the newer attempt's process result or receipt.
After 20 minutes, automatic polling can stop as `abandoned` only when live process
and lock evidence is confirmed absent. Unreadable liveness evidence preserves
polling. Abandonment leaves the exit code unknown; it is not update failure.
Scope execution errors publish a correlated failed receipt and exit nonzero,
without retrying the update outside its required scope.

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
