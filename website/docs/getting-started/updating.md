---
sidebar_position: 3
title: "Updating & Uninstalling"
description: "How to update Hermes Agent to the latest version or uninstall it"
---

# Updating & Uninstalling

## Updating

Update to the latest version with a single command:

```bash
hermes update
```

This pulls the latest code from `main`, updates dependencies, and prompts you to configure any new options that were added since your last update.

:::tip
`hermes update` automatically detects new configuration options and prompts you to add them. If you skipped that prompt, you can manually run `hermes config check` to see missing options, then `hermes config migrate` to interactively add them.
:::

### Passive update notices

Pinned or noninteractive installations can disable passive CLI version and banner update checks:

```bash
hermes config set updates.check false
```

This suppresses both cached update notices and passive update-check network requests. The default is `true`. Explicit `hermes update --check` and `hermes update` still work; this setting does not control the Desktop application's updater.

### What happens during an update

When you run `hermes update`, the following steps occur:

1. **Pre-update snapshot** — a lightweight state snapshot is saved by default (covers pairing data, cron jobs, `config.yaml`, `.env`, `auth.json`, and other state files that get modified at runtime; individual files over 1 GiB are skipped so a large sessions DB never slows the update down). Because the code swap and gateway restarts touch every profile, the same snapshot is taken for **every profile** on the install — each into its own `state-snapshots/` directory — and the post-update cron-jobs safety net checks each profile against its own snapshot. Controlled by `updates.pre_update_backup` (`quick` by default, `full` for a zip of all of `HERMES_HOME`, `off` to disable). Recoverable via the snapshot restore flow described under [Snapshots and rollback](../user-guide/checkpoints-and-rollback.md). Quick snapshots are file-loss recovery, not code-rollback insurance — for a coherent point-in-time rollback use `--backup` (full mode). The snapshot is best-effort: if it fails, the update prints a `⚠ Pre-update snapshot FAILED` warning and continues, and the receipt records `pre_update_backup` as a failed step (a deliberate `off`/`--no-backup` lands in the receipt's skips with its reason instead).
2. **Git pull** — pulls the latest code from the `main` branch and updates submodules
3. **Post-pull syntax validation + auto-rollback** — after the pull, Hermes compiles the nine critical files every `hermes` invocation imports at startup. If any fails to parse (e.g. an orphan merge-conflict marker, an accidentally truncated file), Hermes runs `git reset --hard <pre-pull-sha>` to roll the install back so your shell stays bootable. Re-run `hermes update` once the upstream fix lands.
   After this point the updater re-executes itself on the freshly pulled code (`update.log` shows `=== hermes update continued on the pulled code ===`), so the remaining steps never mix old and new modules in one process. If you see two `hermes update` processes for a moment, that is the hand-off.
4. **Dependency install** — runs `uv pip install -e ".[all]"` to pick up new or changed dependencies. When the checkout is already current this step still runs if the venv is unhealthy (core imports fail) **or** if its installed `hermes-agent` distribution is from an older release than the checkout — the sign that a previous run's dependency install was refused or interrupted (`⚠ Checkout is current, but its dependencies were never synced after the last pull`), so `✓ Already up to date!` never hides a half-updated environment.
5. **Config migration** — detects new config options added since your version and prompts you to set them
6. **Desktop rebuild (stage-and-swap)** — if the Hermes Desktop app was built from this checkout, it is rebuilt so the GUI matches the new code. The rebuild packs into a temporary staging directory next to `apps/desktop/release/`, verifies the staged app, and only then renames it over the previous build (on Windows a real-time scanner briefly holding `release/win-unpacked` is ridden out with a few short retries). A rebuild that fails at any point — corrupt Electron download, missing dependency, disk full — leaves the previous app untouched and launchable; the update reports `⚠ Update partially complete` and `hermes desktop` retries the rebuild. On macOS the rebuilt bundle is then copied (with `ditto`, signature intact) over a stale `/Applications/Hermes.app` or `~/Applications/Hermes.app`, so the copy Finder and the Dock launch matches the backend; an installed copy that is currently running is left alone and the update tells you to quit it and run `hermes update` again.
7. **Gateway auto-restart**: running gateways are refreshed after the update completes. Service-managed gateways (systemd on Linux, launchd on macOS) restart through the service manager. Manual gateways are relaunched when Hermes can map their PID to a profile. Manually launched `hermes serve` / `hermes dashboard` backends are different: the updater leaves them running and asks their owner to restart them. See [Manual backend restart reminders](#manual-backend-restart-reminders). Backends owned by a running Desktop app remain the app's responsibility.
8. **Multiplex migration (multi-profile installs)** — once the fleet is verified on the new code, an install with two or more profiles that still run **one gateway per profile** is folded into a single multiplexed default gateway when nothing blocks it (same as `hermes gateway migrate --multiplex --yes`); if a blocker exists (a bot token shared by two profiles, a secondary profile binding a port with no `/p/<profile>/` ingress) the update prints the blockers with their fixes and changes nothing. Single-profile installs are never touched. See [Migrating from per-profile gateways](../user-guide/multi-profile-gateways.md#migrating-from-per-profile-gateways).

### Why the gateway restart can take a while

The restart is drain-first: the running gateway refuses new turns, then waits for in-flight work (chat turns, cron jobs, API runs) to finish before exiting, capped by `agent.restart_after_turn_timeout` (30 minutes by default) so a long-running job is never cut off mid-run. While that wait is in progress the updater prints, every 30 seconds, what the gateway is still holding for — for example:

```
  → hermes-gateway: draining (up to 1875s)...
  ⏳ still draining — 1560s left before the forced restart
     waiting on 1 active work unit(s):
       • cron job 6ba19dab68df (nightly-scout) in external worker pid 573597, running 6m40s
     finish or kill the work above to release the drain now; agent.restart_after_turn_timeout in config.yaml caps this wait
```

Chat turns show their session key, model and current tool; cron jobs show the job id, name and the process running them (an external restart-safe worker on systemd installs, otherwise the gateway itself). `hermes gateway status` lists the same units while the gateway is draining. To stop waiting, finish or kill the listed work, or lower `agent.restart_after_turn_timeout` in `config.yaml` (`0` enters the forced drain immediately).

Wedged work does not hold the restart: a chat turn idle past `agent.gateway_timeout`, or a cron run older than the scheduler's in-flight allowance (`max(2 × the job's interval, cron.inflight_max_minutes)`, 30 minutes by default), is excluded from the wait and interrupted by the restart instead.

### Missing Windows updater files

If the maintained updater script is missing (for example after antivirus quarantine), the legacy update forwarder fails instead of reporting a successful hand-off. Repair the installation and review the security software's quarantine report before retrying; do not disable antivirus protection. Before reporting success, the maintained updater checks the CLI import, Windows executable header, ASAR header and packaged main entry, readable renderer HTML with a local module entry, initial module files, and current build stamp. These are minimum artifact checks, not a full dependency audit or an application/backend launch test. Missing Python is reported before waiting for Desktop shutdown; dependency repair is still allowed to run as part of the update. Electron checks maintained handoff prerequisites before stopping backends when that layout is present; genuine legacy-flat updater layouts remain supported, so not every missing updater file is detected before backend shutdown.

On Windows, a Desktop reopened during packaging is stopped again immediately before the staged build is promoted. This cleanup is restricted to executables inside that checkout's Desktop release tree; unrelated installations are not stopped. A remaining lock still makes staged promotion fail rather than bypassing the rename error.

### Updating against a non-default branch: `--branch`

By default `hermes update` tracks `origin/main`. Pass `--branch <name>` to update against a different branch — useful for QA channels, feature branches, or release-candidate testing:

```bash
hermes update --branch release-candidate
hermes update --check --branch experimental   # preview behindness only
```

If your local checkout is on a different branch, Hermes auto-stashes any uncommitted work, switches HEAD to the target branch, and then pulls. Branches that don't exist locally are auto-tracked from `origin/<name>` (`git checkout -B <name> origin/<name>`). Branches that don't exist anywhere fail cleanly — your stashed changes are restored before exit so you're never stranded in a weird state. The `main`-only fork-upstream sync logic is automatically skipped on non-`main` branches.

### Checkout parked on a feature branch

If the source checkout was left sitting on a feature branch (by tooling, a worktree experiment, or a manual checkout), `hermes update` switches it back to the update target automatically whenever the working tree is clean:

- **Branch fully merged** (every commit already contained in `origin/main` — `git cherry` reports nothing unmerged): the update says so — `Checkout was parked on '<branch>' (fully merged) — switched back to main` — and stays on `main` afterwards.
- **Branch has unmerged commits** but the tree is clean: the update still switches to `main` so the update can proceed — this is what non-interactive callers (the desktop update button, gateway `/update`, cron) rely on, since they have no way to resolve a skip. Your commits are untouched: `git checkout` never discards committed work, and the update prints a loud notice naming the branch and commit count, plus the `git checkout <branch>` command to pick the work back up later.

If you *deliberately* run a custom branch (local patches maintained on top of main), set `updates.parked_branch_strategy: update_in_place` in `config.yaml`. The update then merges `origin/main` **into** your branch instead of switching away from it — the checkout never moves, your commits survive, and the running code advances. Fast-forward when possible; on divergence a true merge behind a `pre-update-<stamp>` safety tag, stopping cleanly (nothing changed) on conflict. `hermes update --switch-branch` overrides back to the switch path for one run — useful on a deep feature branch that must not accumulate update-driven merge commits.

When the parked branch has **uncommitted changes** (dirty tree), Hermes does **not** touch it. The code update is marked **SKIPPED** with a loud warning naming the branch, how far behind `origin/main` it is, and the exact commands to resolve — instead of pretending the update succeeded. The completion line always shows the actual branch and HEAD (`✓ Update complete! [main @ 30fcf9580]`) so drift is visible at a glance. Set `updates.auto_switch_parked_branch: false` in `config.yaml` to disable the auto-switch entirely (the skip warning still fires).

### Local changes on non-interactive updates

When you run `hermes update` in a terminal, Hermes stashes any uncommitted source-tree changes, pulls, then **asks** whether to restore them — exactly as it always has. Nothing changes for interactive updates.

The autostash only ever covers *source-tree* changes. On a **flat install** — where the git checkout root is also `$HERMES_HOME` (for example an install made with `HERMES_INSTALL_DIR=$HERMES_HOME`, or one created by an older installer) — the profile's runtime state (`state.db` and its WAL/SHM sidecars, `state-snapshots/`, `backups/`, `sessions/`, `cron/jobs.json`, the `cron/*.db` stores, `config.yaml`, `auth.json`, `memories/`, lock/pid files, …) lives inside the checkout as untracked files. Those paths are git-ignored, so the autostash never touches them and the running gateway keeps its database through the update. If you keep other untracked files in a flat install's root, move them out of the checkout or add them to `.git/info/exclude`; anything untracked and not ignored is swept into the autostash like a source edit.

When the update runs **without a terminal** — from the desktop/chat app's "Update" button or a gateway-triggered update — there's no prompt to answer. The `updates.non_interactive_local_changes` setting decides what happens to your stashed changes:

```yaml
# ~/.hermes/config.yaml
updates:
  non_interactive_local_changes: stash   # default: keep + auto-restore
  # non_interactive_local_changes: discard  # throw local source edits away
```

- `stash` (default) — auto-stash, pull, then auto-restore your changes on top of the updated code. Nothing is lost; if a restore hits conflicts they're preserved in a git stash for manual recovery.
- `discard` — auto-stash and drop the stash after the pull, so the update always lands on a clean tree. Use this only on machines where you never intend to keep local edits to the Hermes source. It stash-drops (not `git reset --hard` + `git clean -fd`), so ignored paths like `node_modules`, `venv`, and build outputs are never touched.

In the desktop app this is **Settings → Advanced → In-App Update Local Changes**.

**Desktop updates never auto-restore.** The desktop updater invokes `hermes update --keep-stash`: local source edits are still stashed so the update can proceed, but they are **not** re-applied afterward — they stay parked in `git stash` and the update log prints the exact `git stash apply <ref>` command to bring them back. This prevents local edits from silently riding along across desktop updates and breaking the freshly updated install. (`non_interactive_local_changes: discard` still wins if you've opted into discarding.) To restore parked changes manually:

```bash
cd ~/.hermes/hermes-agent   # or your install root
git stash list --format='%gd %H %s'   # find the hermes-update-autostash entry
git stash apply stash@{0}
```

You can pass `--keep-stash` to a terminal `hermes update` too if you want the same never-reapply behavior interactively.

### Preview-only: `hermes update --check`

Want to know if an update is available before pulling? Run `hermes update --check` — it fetches and compares commits against `origin/main`. No files are modified, no gateway is restarted. Useful in scripts and cron jobs that gate on "is there an update".

### Fleet preview: `hermes update --plan`

Before updating a machine that runs several profiles or services, run `hermes update --plan`. It prints the install kind, running Hermes services across profiles, their supervisors and running code versions, and the restart mechanism for each service. Manually launched `hermes serve` / `hermes dashboard` backends appear with their recorded bind endpoint, but restart is deferred to their owner; the updater does not stop or relaunch them. Image- or package-managed installs report the external update command instead. The plan is read-only and safe on a live fleet.

The same inventory is embedded in every real update's receipt (`~/.hermes/logs/update_receipts/`), so after an update you can compare what the updater saw against what it did.

### Update receipts and the fleet version check

Every `hermes update` run writes a machine-readable receipt to `~/.hermes/logs/update_receipts/` (last 20 kept, `latest.json` always points at the most recent): the pre-update fleet plan, each step taken, anything skipped and why, the gateway restart outcome, and the final fleet version matrix. The SQLite runtime repair is one of those steps (`sqlite_runtime_repair`): a failed repair records the actual reason (for example the `uv sync` error) and the SQLite version pair, a deferred or not-applicable repair lands in the skips with its reason. After the restart phase the updater compares each live gateway's running code against the freshly updated checkout and prints a per-profile matrix — a gateway still serving pre-update code is reported loudly with the exact restart command, and the update exits non-zero so automation never treats a mixed-version fleet as healthy. Both `--plan` and the fleet check ask each running gateway directly over its local control socket (`gateway.sock` in the profile's data directory, a named pipe on Windows) when available, so version and supervisor information comes from the gateway itself; gateways from older versions are still discovered through their state files as before.

A multiplexed default gateway is one process serving several profiles, so it appears once in the matrix and vouches for every profile in its `served_profiles` record. The same coverage clears the "A previous `hermes update` pulled new code but did not restart running gateways" hint: once that gateway (or, after a manual `git pull`, every gateway an update restarted) runs the current code, `hermes gateway restart` is enough — the hint no longer waits for the next `hermes update` to write a fresh receipt. The same is true of the restart obligation left by an update that died before recording which gateways it owed (or by an older updater that never recorded them): once every live gateway runs the current checkout, the obligation is retired and the hint stops. That obligation is recorded once per HOST, in the cross-profile rendezvous directory (`$HERMES_GATEWAY_LOCK_DIR`, else `$XDG_STATE_HOME/hermes/gateway-locks`) as `host-update-restart.json`, so every profile's CLI sees the same one: `hermes -p coder update` and `hermes -p writer update` restart the shared multiplexed gateway once between them, not once each. An obligation left behind by an older per-profile updater (`fleet_restart_pending` in one profile's Hermes home) is still read and cleared. An update whose pre-update plan found no gateway at all owes nothing and leaves no breadcrumb. A backend supervised by Desktop, systemd or launchd is restarted by its supervisor and never blocks this settlement; only a manual backend whose reminder could not be saved keeps the obligation open.

### Manual backend restart reminders

After updating, ask the owner of each manually launched backend to relaunch `hermes serve` or `hermes dashboard`; reconnect Desktop for an SSH backend. Restarting gateways alone does not refresh these processes.

For a verified live backend, the updater saves a reminder before recording its restart as `deferred`. This allows the update to complete if the remaining checks pass. An unknown process identity cannot qualify for fresh deferral.

Reminders live in `serve_restart_pending/` under the active Hermes home, separately from rotating update receipts. Each reminder identifies a process by PID and creation time. Startup warnings retain it while that process is alive or its liveness is unknown, and remove it only when that exact process is confirmed gone. A later update or a healthy gateway does not remove it.

If saving fails, Hermes warns and carries the unsaved obligation into later receipts for retry. Check storage permissions and free space, and restart the backend as instructed. If both reminder and receipt storage fail, durable retention cannot be guaranteed.

### Automated updates from inside the gateway: `--no-gateway-restart`

An update launched *by* the gateway (a cron job, the Desktop updater, any automation that is a
child of the gateway process) cannot survive its own fleet restart: the gateway drains on
`SIGUSR1` and systemd's `KillMode=mixed` then kills everything left in the cgroup, updater
included. `hermes update --no-gateway-restart` runs the full pipeline (pull, dependencies,
Node workspaces, web UI, maintenance) and skips only the restart and fleet verification. The
pending-restart marker is kept, so the next CLI start warns and the next normal `hermes update`
(or `hermes gateway restart`) catches the fleet up. Pair it with a separate restart step, for
example a timer 10–15 minutes after the update job. The receipt records the deferral; a stale
fleet caused only by the deferral does not make the update `partial`.

### Interrupted gateway restarts

If an earlier update pulled code but did not finish restarting the fleet, the next `hermes update` retries even when the checkout is already current. An empty process scan does not prove recovery: failed systemd units and installed launchd jobs may have no live PID. If discovery or restart fails, or a requested service cannot be verified active, the marker stays pending and the update exits nonzero. Recover the affected services with the printed commands and retry.

A failed historical receipt alone does not prove that gateways are still stale. Startup and gateway-status warnings, as well as update catch-up, check for a live successor on the current checkout for every gateway profile recorded in that receipt. A manual gateway restart can settle receipt-only advice without rewriting the historical outcome. Manual backend obligations must first transfer to their separate reminders.

A pending restart marker records its own pre-update runtime inventory and target commit. Recovery checks that inventory, never an older receipt's inventory. Settlement requires a nonempty verified current fleet, a successor at the target commit for every recorded gateway profile, and successful handling of any identified manual backend obligations. A manual backend that is still alive or whose liveness is unknown needs a saved reminder; a confirmed exited process needs no reminder. Missing gateways, unsupported runtimes, or failed reminder writes keep the marker pending.

Legacy markers without an inventory, and malformed or unsupported inventories, cannot be automatically cleared by startup or catch-up reconciliation. Restarting all visible gateways is not enough to establish what that update owed. The warning remains until a later update completes with a new marker containing its own inventory and settles successfully.

### Full pre-update backup: `--backup`

For high-value profiles (production gateways, shared team installs) you can opt into a full pre-pull backup of `HERMES_HOME` (config, auth, sessions, skills, pairing):

```bash
hermes update --backup
```

Or make it the default for every run:

```yaml
# ~/.hermes/config.yaml
updates:
  pre_update_backup: full
```

`updates.pre_update_backup` is a single knob with three modes: `quick` (default — the lightweight state snapshot described above), `full` (the quick snapshot plus a complete `HERMES_HOME` zip; can add minutes on large homes), and `off` (no pre-update backup at all — `--no-backup` does the same for a single run). Legacy boolean values still work: `true` means `full`, `false` means `off`.

:::tip Moving to a new machine instead?
Update backups protect an in-place update. If you're migrating your whole setup to different hardware, use `hermes backup` + `hermes import` instead — see [Exporting Hermes to another machine](../reference/faq.md#exporting-hermes-to-another-machine) and [`hermes backup` vs `hermes profile export`](../reference/faq.md#hermes-backup-vs-hermes-profile-export).
:::

### Windows: another `hermes.exe` is running

On Windows, `hermes update` will refuse to run if it detects another `hermes.exe` process holding the venv's entry-point executable open — most commonly the Hermes Desktop app's spawned backend, an open `hermes` REPL in another terminal, or a running gateway:

```
$ hermes update
✗ Another hermes.exe is running:
    PID 12345  hermes.exe

  Updating now would fail to overwrite ...\venv\Scripts\hermes.exe because
  Windows blocks REPLACE on a running executable.

  Close Hermes Desktop, exit any open `hermes` REPLs, and
  stop the gateway (`hermes gateway stop`) before retrying.
  Override with `hermes update --force` if you've already
  confirmed those processes will not write to the venv.
```

Close the listed processes and re-run. If you're sure the concurrent process won't interfere (rare — usually only useful when an antivirus shim is mis-attributed), pass `--force` to skip the check. In that case the updater will still retry the `.exe` rename with exponential backoff and, on stubborn locks, schedule the replacement for next reboot via `MoveFileEx(MOVEFILE_DELAY_UNTIL_REBOOT)` so the update can complete.

A second, separate guard refuses to touch the venv while any process is running from its Python interpreter (the Desktop app's backend, a gateway, a Python REPL). Those processes keep native extension files (`.pyd`) locked, and a dependency sync that dies partway on an access-denied error strands the install between versions. This guard is **not** bypassed by `--force`; if you're certain the detected holders are false positives, use the explicit `hermes update --force-venv`.

#### Scripted updates: `hermes update --list-venv-holders`

A scheduled `hermes update --yes` that keeps hitting the venv guard (typically because the
Desktop app relaunches its backend) can ask first instead of looping. `hermes update
--list-venv-holders` is read-only: it prints the processes the guard would refuse on as a
JSON list of `{pid, exe, argv, kind}` and exits `0` when the venv is free or `3` when holders
are present. `kind` is `gateway` (a pausable gateway the updater handles itself), `backend`
(a `hermes serve` / dashboard backend — the Desktop app's shape), `hermes:<subcommand>` for any
other Hermes process, or `python` for an unrelated interpreter. Automation can stop exactly
those PIDs (or quit the Desktop app) and retry; nothing is terminated by the flag itself. The
guard only exists on Windows, so the list is always `[]` elsewhere.

```
$ hermes update --list-venv-holders
[
  {"pid": 4242, "exe": "C:\\hermes\\venv\\Scripts\\python.exe",
   "argv": "...python.exe -m hermes_cli.main serve --port 8642", "kind": "backend"}
]
$ echo $LASTEXITCODE
3
```

Both guards, the Desktop update preflight, and the dependency repair steps look for the environment at `venv` first and then at the uv-default `.venv`, so a source checkout set up with `uv venv` / `uv sync` updates the same way an installer-created `venv` does. When both directories exist, `venv` is the one that gets updated.

#### Windows: the update finishes under the venv Python

`hermes.exe` itself cannot be replaced while it runs, so an update started from it stops after the code swap and prints `→ Windows: hermes.exe cannot replace itself while it runs; the update continues under the venv Python`. Your shell returns right away; a child interpreter finishes the dependency install and prints its own result. The child first waits (up to 30 s) for the process that started the update — the `hermes.exe` launcher when it can be identified, otherwise the interpreter it ran — to exit, then takes over the update lock; it alone restarts the gateways the update paused — the process that launched it never resumes them while it still holds the shim. If the launcher is still alive after that wait, the child says so and continues; a shim that is still locked at install time is reported as before and the install is deferred to the next `hermes` run.

#### Windows venv recreation is transactional

When the Windows installer must recreate an existing `venv`, it first moves the old directory to a unique `venv.stale.*` name, then creates and verifies the replacement. The old tree is deleted only after the dependency install completes and the baseline imports pass in the new tree — until then it is the rollback source (recorded in `venv.pending-backup`).

If the move cannot be completed, the installer stops and leaves the live `venv` untouched. If `uv` fails or reports success without creating the interpreter, any partial replacement is moved to `venv.failed.*` and the previous venv is restored. This keeps the health and blocker checks usable after a failed install.

A `venv.stale.*` or `venv.failed.*` directory can remain when another process still owns a file handle. Close Hermes Desktop, gateways, and Python processes using the install, then retry the install/update; parked directories are cleaned up best-effort after a successful recreation.

Expected output looks like:

```
$ hermes update
Updating Hermes Agent...
📥 Pulling latest code...
Already up to date.  (or: Updating abc1234..def5678)
📦 Updating dependencies...
✅ Dependencies updated
🔍 Checking for new config options...
✅ Config is up to date  (or: Found 2 new options — running migration...)
🔄 Restarting gateways...
✅ Gateway restarted
✅ Hermes Agent updated successfully!
```

### Recommended Post-Update Validation

`hermes update` handles the main update path, but a quick validation confirms everything landed cleanly:

1. `git status --short` — if the tree is unexpectedly dirty, inspect before continuing
2. `hermes doctor` — checks config, dependencies, and service health
3. `hermes --version` — confirm the version bumped as expected
4. If you use the gateway: `hermes gateway status`
5. If `doctor` reports npm audit issues: run `npm audit fix` in the flagged directory

:::warning Dirty working tree after update
If `git status --short` shows unexpected changes after `hermes update`, stop and inspect them before continuing. This usually means local modifications were reapplied on top of the updated code, or a dependency step refreshed lockfiles.
:::

### If your terminal disconnects mid-update

`hermes update` protects itself against accidental terminal loss:

- The update ignores `SIGHUP`, so closing your SSH session or terminal window no longer kills it mid-install. `pip` and `git` child processes inherit this protection, so the Python environment cannot be left half-installed by a dropped connection.
- All output is mirrored to `~/.hermes/logs/update.log` while the update runs. If your terminal disappears, reconnect and inspect the log to see whether the update finished and whether the gateway restart succeeded:

```bash
tail -f ~/.hermes/logs/update.log
```

- `Ctrl-C` (SIGINT) and system shutdown (SIGTERM) are still honored — those are deliberate cancellations, not accidents.

You no longer need to wrap `hermes update` in `screen` or `tmux` to survive a terminal drop.

### Checking your current version

```bash
hermes --version
```

Compare against the latest release at the [GitHub releases page](https://github.com/NousResearch/hermes-agent/releases).

### Updating from Messaging Platforms

You can also update directly from Telegram, Discord, Slack, WhatsApp, or Teams by sending:

```
/update
```

This pulls the latest code, updates dependencies, and restarts running gateways. The bot will briefly go offline during the restart (typically 5–15 seconds) and then resume.

### Manual Update

If you installed manually (not via the quick installer):

```bash
cd /path/to/hermes-agent
# Activate the venv you created during install (outside the source tree)
export VIRTUAL_ENV="$HOME/.hermes/venvs/hermes-dev"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Pull latest code
git pull origin main

# Reinstall (picks up new dependencies)
uv pip install -e ".[all]"

# Check for new config options
hermes config check
hermes config migrate   # Interactively add any missing options
```

### Rollback instructions

If an update introduces a problem, you can roll back to a previous version:

```bash
cd /path/to/hermes-agent

# List recent versions
git log --oneline -10

# Roll back to a specific commit
git checkout <commit-hash>
uv pip install -e ".[all]"

# Restart the gateway if running
hermes gateway restart
```

To roll back to a specific release tag (substitute your previous tag — e.g. a recent release like `v2026.5.16`, or any earlier tag from `git tag --sort=-version:refname`):

```bash
git checkout vX.Y.Z
uv pip install -e ".[all]"
```

:::warning
Rolling back may cause config incompatibilities if new options were added. Run `hermes config check` after rolling back and remove any unrecognized options from `config.yaml` if you encounter errors.
:::

### Image-managed installs (Docker): the provenance marker

Published Docker images bake a small read-only marker (`/etc/hermes/image-provenance.json`) that authoritatively identifies the filesystem as image-managed. `hermes update`, `hermes update --check`, and the dashboard's Update button all consult it before touching anything: on an image-managed install they refuse cleanly (exit code 2), print the actual update command (`docker pull nousresearch/hermes-agent:latest`), and write a `refused` receipt so fleet tooling can see the attempt happened. The marker wins even when a source checkout is bind-mounted into the container — the refusal is based on what the running filesystem *is*, not what it looks like. A damaged marker still refuses (fail-closed). Nix- and apt-managed installs refuse through the same gate using the existing detection.

### Note for Nix users

Nix is no longer an explicitly supported install path (best-effort only) — see [Nix Setup](./nix-setup.md). If you installed via Nix flake, updates are managed through the Nix package manager:

```bash
# Update the flake input
nix flake update hermes-agent

# Or rebuild with the latest
nix profile upgrade hermes-agent
```

Nix installations are immutable — rollback is handled by Nix's generation system:

```bash
nix profile rollback
```

See [Nix Setup](./nix-setup.md) for more details.

---

## Uninstalling

```bash
hermes uninstall
```

The uninstaller gives you the option to keep your configuration files (`~/.hermes/`) for a future reinstall.

:::tip Moving to a new machine rather than leaving?
Take your setup with you before removing anything: `hermes backup` captures the entire `~/.hermes` directory including credentials, while `hermes profile export` packs a single profile with credentials excluded by design (so an export alone is not a full backup). See [`hermes backup` vs `hermes profile export`](../reference/faq.md#hermes-backup-vs-hermes-profile-export).
:::

### Manual Uninstall

```bash
rm -f ~/.local/bin/hermes
rm -rf /path/to/hermes-agent
rm -rf ~/.hermes            # Optional — keep if you plan to reinstall
```

:::info
If you installed the gateway as a system service, stop and disable it first:
```bash
hermes gateway stop
# Linux: systemctl --user disable hermes-gateway
# macOS: launchctl remove ai.hermes.gateway
```
:::
