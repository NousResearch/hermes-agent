---
sidebar_position: 3
title: "Updating & Uninstalling"
description: "How to update Hermes Agent to the latest version or uninstall it"
---

# Updating & Uninstalling

## Updating

Choose the update method for the installation that is running:

| Installation | Update method |
|---|---|
| Managed source checkout | `hermes update`, or the source-built desktop's update handoff. |
| Windows sideload MSIX | The desktop Update control and Windows App Installer. |
| Microsoft Store package | Microsoft Store updates. |
| macOS bundled app | The desktop Update control, through `electron-updater`. |
| Docker image | Pull the chosen image and recreate the container with the same data mount. |
| Nix | Update the flake/profile and rebuild. |
| Termux APT | `pkg update`, then `pkg upgrade hermes-agent`. |

`hermes update` does not rewrite package-owned application files. PM manages
dependencies, not application distribution: `hermes pm update` is a maintainer
pin-update command, not an alternative application updater.

For a managed source installation:

```bash
hermes update
```

The default source channel tracks `main`. Configured stable and canary channels
track their published release commits. The update prepares dependencies through
PM and reports configuration changes and process-restart results.

### Bundled desktop updates

On Windows, the sideload app checks the registered App Installer source. Apply
downloads the `.appinstaller` descriptor before stopping app-owned backends,
opens that local file, and quits for Windows to replace the package.
A detached waiter attempts automatic relaunch after the package version changes.
A waiter-registration failure produces a manual-reopen warning.

On macOS, apply downloads the ZIP update and waits for Squirrel.Mac to accept
the signed app before stopping backends. Only that apply flow requests install
and relaunch. An ordinary quit does not install a pending download.
Download or verification failures leave backends running.

Store packages use the Store instead of either sideload feed. A broken or
unavailable update check must not be interpreted as "already up to date."
The application and its bundled base runtime update together; user data stays
outside the package. See [Desktop installation](../user-guide/desktop.md#install).

Stable and canary desktop bundles are separate applications. Installing canary
does not replace stable, including on Windows with MSIX. Each application keeps
its own desktop state and updates within the channel baked into its build;
changing channels means installing the other application, not changing a setting.
The canary icon has a yellow background (dark yellow in dark mode), and its CLI
command is `hermes-canary`.

One-off commit bundles are separate from both release channels and from other
commit bundles. Their red icons show the short build SHA, also used in the
`hermes-<short-sha>` CLI command. They do not check for or install updates,
including through `hermes update` or `hermes update --check`. They explain:

> This build doesn't get updates. Ask the developer who gave it to you for a new build.

Separate applications still share Hermes profiles, configuration, and sessions
under the same Hermes home. Running different builds against one profile is not
schema isolation: newer builds can change stored data that an older build cannot
read. Back up shared data before testing. The desktop and standalone CLI warn
when another live installation uses the same profile; this is advisory, not a
lock. Desktop post-update notices are scoped to the application, so launching
canary cannot consume stable's pending notice. The `hermes://` URL scheme remains
shared; the application that most recently registered it handles links.

### Source channels and install identity

```bash
hermes update --install-id
hermes update --set-channel stable
hermes update --channel stable --check
# Or track published canary commits in this source installation:
hermes update --set-channel canary
hermes update
```

`--install-id` prints the installation identity and path. `--set-channel`
changes only that installation's configuration, then exits without applying an
update. `--channel` is a one-run override. An explicit `--branch` takes precedence
for a source checkout.

Source `main` tracks the branch tip. Source `stable` and `canary` resolve the
published release for that channel and update the checkout to its exact Git
commit. Canary does not mean the current tip of `main`, and an unpublished tag
is not a release. Switching a source channel does not install a desktop package.
Per-install records live under `update.installs` in configuration, so one
checkout's choice does not change another installation's channel. The source-built
desktop uses that same selection for checks and update handoffs; it does not
replace a selected release channel with its default branch.

Packaged desktop feed channels derive from their build tag and package owner.
Changing a source channel is not an MSIX or Store channel switch. Canary builds
can advance stored data formats; switching back is not a schema rollback.
Back up data before changing release channels.

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

For an admitted source checkout, `hermes update` runs these phases:

1. **Pre-update snapshot** — Hermes saves selected state files for every profile in that profile's `state-snapshots/` directory. These include pairing data, cron jobs, `config.yaml`, `.env`, and `auth.json`. Automatic quick snapshots skip individual files larger than 1 GiB. `updates.pre_update_backup` selects `quick`, `full`, or `off`. Full archives use the [backup exclusions](/reference/faq#hermes-backup-vs-hermes-profile-export). Recovery uses [Snapshots and rollback](../user-guide/checkpoints-and-rollback.md). Quick snapshots recover state files, not application code.
2. **Code update** — applies the configured source branch or stable release tag and updates submodules.
3. **Post-pull syntax validation + auto-rollback** — after the pull, Hermes compiles the nine critical files every `hermes` invocation imports at startup. If any fails to parse (e.g. an orphan merge-conflict marker, an accidentally truncated file), Hermes runs `git reset --hard <pre-pull-sha>` to roll the install back so your shell stays bootable. Re-run `hermes update` once the upstream fix lands.
4. **Dependency preparation** — PM provisions required tools and prepares a complete Python environment from the new lock, existing extras, and enabled plugin requirements. It validates that environment before publishing its selection.
5. **Config migration** — detects new config options added since your version and prompts you to set them
6. **Desktop rebuild (stage-and-swap)** — if the Hermes Desktop app was built from this checkout, it is rebuilt so the GUI matches the new code. The rebuild packs into a temporary staging directory next to `apps/desktop/release/`, verifies the staged app, and only then renames it over the previous build. A rebuild that fails at any point — corrupt Electron download, missing dependency, disk full — leaves the previous app untouched and launchable; the update reports `⚠ Update partially complete` and `hermes desktop` retries the rebuild.
7. **Gateway auto-restart** — running gateways are refreshed after the update completes so the new code takes effect immediately. Service-managed gateways (systemd on Linux, launchd on macOS) are restarted through the service manager. Manual gateways are relaunched automatically when Hermes can map the running PID back to a profile. Manually-launched `hermes serve` / `hermes dashboard` backends (for example a network-bound serve powering a remote Desktop) are handled the same way: each backend records its bind address in the install's spawn ledger at startup, so the update stops it before the code swap and relaunches it afterward on the **same host and port** — a remote Desktop pointed at that endpoint reconnects instead of stranding. Backends owned by a running Desktop app are left to the app's own respawn.

### Missing Windows updater files

If the maintained updater script is missing (for example after antivirus quarantine), the legacy update forwarder fails instead of reporting a successful hand-off. Repair the installation and review the security software's quarantine report before retrying; do not disable antivirus protection. Before reporting success, the maintained updater checks the CLI import, Windows executable header, ASAR header and packaged main entry, readable renderer HTML with a local module entry, initial module files, and current build stamp. These are minimum artifact checks, not a full dependency audit or an application/backend launch test. Missing Python is reported before waiting for Desktop shutdown; dependency repair is still allowed to run as part of the update. Electron checks maintained handoff prerequisites before stopping backends when that layout is present; genuine legacy-flat updater layouts remain supported, so not every missing updater file is detected before backend shutdown.

On Windows, a Desktop reopened during packaging is stopped again immediately before the staged build is promoted. This cleanup is restricted to executables inside that checkout's Desktop release tree; unrelated installations are not stopped. A remaining lock still makes staged promotion fail rather than bypassing the rename error.

### Updating against a non-default branch: `--branch`

On the default source channel, `hermes update` tracks `origin/main`. Use
`--branch NAME` for a one-run branch override:

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

`hermes update --check` compares the checkout with its source-channel target
without applying code, installing dependencies, or restarting gateways. The
comparison can fetch Git metadata; it is not a promise of zero filesystem writes.
Package-owned installs report their external update method.

### Fleet preview: `hermes update --plan`

Before updating a machine that runs several profiles or services, `hermes update --plan` prints the full update plan without changing anything: the install kind (git checkout, Docker image, Nix/apt managed), every running Hermes service across all profiles with its supervisor (systemd, launchd, manual) and the code version it is actually serving, and the restart mechanism each one will get. Manually-launched `hermes serve` / `hermes dashboard` backends appear too (from the spawn ledger), with their recorded bind endpoint and a "stop before code swap, relaunch with recorded launch args" restart mechanism. On image- or package-managed installs the plan reports that the install is not updatable in place and names the right update command instead. Read-only and safe on a live fleet.

The same inventory is embedded in every real update's receipt (`~/.hermes/logs/update_receipts/`), so after an update you can compare what the updater saw against what it did.

### Update receipts and the fleet version check

Every `hermes update` run writes a machine-readable receipt to `~/.hermes/logs/update_receipts/` (last 20 kept, `latest.json` always points at the most recent): the pre-update fleet plan, each step taken, anything skipped and why, the gateway restart outcome, and the final fleet version matrix. After the restart phase the updater compares each live gateway's running code against the freshly updated checkout and prints a per-profile matrix — a gateway still serving pre-update code is reported loudly with the exact restart command, and the update exits non-zero so automation never treats a mixed-version fleet as healthy. Both `--plan` and the fleet check ask each running gateway directly over its local control socket (`gateway.sock` in the profile's data directory, a named pipe on Windows) when available, so version and supervisor information comes from the gateway itself; gateways from older versions are still discovered through their state files as before.

### Interrupted gateway restarts

If an earlier update pulled code but did not finish restarting the fleet, the next
`hermes update` retries even when the checkout is already current. An empty process
scan does not prove recovery: failed systemd units and installed launchd jobs may
have no live PID. The pending restart marker is retained if supervisor discovery
fails, a restart fails, or a requested service cannot be verified active. The update
exits nonzero and reports the affected services; recover them with the printed
commands and retry `hermes update`.

A failed historical receipt does not by itself prove that gateways are still stale. Startup and gateway-status warnings, as well as update catch-up, check the live fleet before acting on receipt-only restart obligations. Every recorded gateway profile must have a live successor on the current checkout; an unrelated current gateway cannot stand in for a missing, down, unknown-version, or non-gateway runtime. A manual gateway restart can therefore settle the warning without rewriting a failed update as successful. A separate pending marker remains authoritative because it can belong to a newer interrupted update whose inventory never reached the receipt.

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

`updates.pre_update_backup` has three modes:

- `quick` saves the selected state files described above. This is the default.
- `full` adds a zip archive with the [backup exclusions](/reference/faq#hermes-backup-vs-hermes-profile-export). Large data directories can take several minutes.
- `off` disables pre-update backups. `--no-backup` selects this mode for one run.

Legacy boolean values remain supported: `true` means `full`, and `false` means `off`.

:::tip Moving to a new machine instead?
Update backups protect an in-place update. If you're migrating your whole setup to different hardware, use `hermes backup` + `hermes import` instead — see [Exporting Hermes to another machine](/reference/faq#exporting-hermes-to-another-machine) and [`hermes backup` vs `hermes profile export`](/reference/faq#hermes-backup-vs-hermes-profile-export).
:::

### Windows process ownership and dependency changes

Windows can hold executable and native-extension files open while processes run.
PM therefore prepares dependency generations separately instead of replacing
imported libraries in place. A running process keeps its current imports until
it restarts.

Follow any blocker diagnostic for the specific process and installation. Do
not kill every process named Python or Hermes. Source updates and MSIX package
replacement have different owners and shutdown requirements.

The parser retains `--force` and `--force-venv` for Windows compatibility. They
are not normal update instructions or a guarantee that an OS file lock can be
bypassed. Inspect `hermes update --plan`, the update log, and `hermes pm status`
before retrying a failed update.

### Recommended Post-Update Validation

After a source update, run the following diagnostics. For a bundled app, use
its About page and backend health instead of Git commands inside the package.

1. `git status --short` — if the tree is unexpectedly dirty, inspect before continuing
2. `hermes doctor` — checks config, dependencies, and service health
3. `hermes --version` — confirm the version bumped as expected
4. If you use the gateway: `hermes gateway status`
5. `hermes pm status` — inspect dependency preparation and any failed steps.

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

### Manual source maintenance and rollback

Use the [source-install guide](../user-guide/switching-to-source.md) for an
independent development checkout. Commit or otherwise preserve local work
before changing its Git revision. Stop the services you own before replacing
the code that those services run.

After changing the source revision, run that revision's dependency/bootstrap
procedure. PM-managed checkouts use `python -m pm.cli install` from the selected
source environment. A bare pip install is not equivalent: it does not prepare
the managed tool store or PM's runtime selection.

A code checkout alone is not a data rollback. Newer releases can migrate
configuration or databases in ways older code cannot read. Use a coherent
backup from before the update if an older release requires the earlier data
format. Preserve the current data before attempting a restore.

Package-owned installations use their package manager's rollback or reinstall
procedure. Do not run Git or pip inside a signed app or an immutable image.

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

For source installs, review `hermes uninstall --dry-run` before removal.
The default removal can preserve configuration and user data. `--full` also
removes data. `--data` removes user data without deleting package-owned code.
These modes are destructive; make a backup first.

For MSIX/Store, use Windows Settings → Apps → Installed apps. For macOS,
quit Hermes and move its app bundle to Trash. Docker, Nix, and Termux use the
same manager that installed them. Their package files are not removed by the
source uninstaller. Data deletion is separate from package removal.

:::tip Moving to a new machine rather than leaving?
Run `hermes backup` before you remove the installation. The full archive includes
credentials but excludes downloaded runtimes, dependency environments, caches,
and browser profiles. Review its skipped-file report before you delete source data.
`hermes profile export` packs one profile without credentials.
See [`hermes backup` vs `hermes profile export`](/reference/faq#hermes-backup-vs-hermes-profile-export).
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
