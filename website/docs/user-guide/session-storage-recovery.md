---
title: "Session Storage Recovery"
description: "What to do when Hermes says the session database was replaced underneath it or another process holds an old copy of its write-ahead log, and what the files beside state.db are"
---

# Session storage recovery

Hermes keeps every conversation in one SQLite file per profile, `state.db`, with two
sidecar files SQLite manages itself: `state.db-wal` (the write-ahead log) and `state.db-shm`.
Several Hermes processes can share that file safely — the gateway, the Desktop app, the
dashboard, cron, and CLI commands all write through SQLite's own locking.

One thing is not safe: **rewriting the store while another process is writing to it**.
When that happens, the processes still holding the *old* copy of the log stop writing on
purpose and every turn answers with a message like:

> another Hermes process still holds an old copy of the session database's write-ahead log,
> so Hermes stopped writing to keep the file safe …

or, when the main file itself was swapped out from under a running Hermes:

> the session database file was replaced while Hermes was running …
> (`FATAL: state.db was replaced underneath the gateway` in the log)

This page is the guide both messages link to. Nothing is lost when you see either; the
refusal exists precisely so nothing gets lost. The three steps below fix the first message.
The second one has an extra step — finding what rewrote the file — covered in
[state.db was replaced](#state-db-was-replaced).

## The fix in three steps

1. **Quit every Hermes process on that profile.** Desktop app, gateway, dashboard, cron:

   ```bash
   hermes gateway stop          # add -p <profile> for a named profile
   ```

   then quit the Desktop app from its menu and stop any dashboard (`hermes dashboard --stop`)
   or custom service you run. Restarting *one* of them is not enough — a single process left
   holding the old log keeps every new one refusing.

2. **Ask doctor who is still holding the log.**

   ```bash
   hermes doctor                # add -p <profile> for a named profile
   ```

   While anything still holds the retired log, doctor prints each holder as
   `PID N (command)` with the same remedy, and skips its health probes and any `--fix` work
   so it cannot become another writer. Stop the listed processes and run it again until the
   line is gone.

3. **Start Hermes again** (one process first — the gateway or the Desktop app) and send your
   message once more. Your conversation resumes from where it stopped.

## Do not

- **Do not run `hermes doctor --fix` while the processes are running.** Doctor refuses the
  checkpoint while it can see a process holding the retired log, but on a host where it cannot
  inspect processes the fix path is exactly the second writer that caused the problem.
- **Do not delete `state.db-wal` or `state.db-shm`.** The log holds committed conversations
  that are not yet in `state.db`. Deleting it is the one action that turns a refusal into
  real data loss.
- **Do not copy `state.db` alone.** The three files are one image. Use a snapshot
  (`hermes backup`) or `hermes sessions recover`, never `cp state.db somewhere/`.
- **Do not ask the agent to fix it.** The agent's own session is in the same store; it will
  hit the same refusal.

## state.db was replaced {#state-db-was-replaced}

The second message means the *main* file changed identity under a running Hermes: the path
`state.db` now names a different file (or a rewritten copy) than the one the gateway opened.
Hermes refuses to keep writing because a write through the old handle would land in a file
nobody reads any more — or corrupt the new one. Your unsaved turns are diverted to
`sessions/<id>.jsonl` and the gateway's pending-messages spool, so nothing is lost, but
restarting alone will not help: whatever replaced the file once will do it again.

Find the replacer first. In practice it is one of these:

| What replaced the file | How to tell | What to do |
|---|---|---|
| **A file-sync client** (iCloud Drive, Dropbox, OneDrive, Google Drive, Syncthing, a synced or network home directory) mirroring `~/.hermes` — typically because you run Hermes on two machines, e.g. a desktop and a laptop, and want the same conversations on both. | The message appears on the machine that was *not* the last one writing; the `~/.hermes` folder (or a parent) sits inside the sync client's tree, or is a network share. | Move `~/.hermes` out of the synced folder on every machine (or exclude it from sync). One store belongs to one host. To use the same conversations from a second machine, run **one** gateway and connect to it from the other device — the Desktop app's [Gateways page](./multi-connection-desktop.md), the dashboard, or a messaging platform — instead of syncing the files. |
| **A backup or snapshot restored while Hermes was running** (`hermes backup restore`, `hermes sessions recover`, `state-snapshots/`, a Time Machine / rsync restore of the home folder). | You, a script, or a restore job touched `~/.hermes` within the last minutes. | Stop every Hermes process *before* restoring, then start again. Restores are safe with everything stopped. |
| **A manual copy over the file** (`cp other.db ~/.hermes/state.db`, `mv`, an editor that writes a temp file and renames it into place). | Same as above; `ls -li ~/.hermes/state.db` shows a new inode or a modification time you did not expect. | Never copy `state.db` alone (the three files are one image). Use `hermes backup` and `hermes sessions recover`, with Hermes stopped. |

Then recover:

1. Quit every Hermes process on the profile (`hermes gateway stop`, the Desktop app, dashboard, cron).
2. Stop the replacer: pause or exclude the sync client, or finish the restore.
3. Run `hermes doctor` (not `--fix`) and start Hermes again. The gateway reopens whichever
   file is now at `state.db`; if it is the old copy from before the replacement, the turns from
   the gap are still in `sessions/<id>.jsonl` and `hermes sessions recover --source ... --inspect-only`
   will tell you whether they can be merged.

Do not run `hermes doctor --fix` or `hermes sessions optimize` while the file is still being
swapped: an in-place repair of a file that is about to be overwritten again is how a refusal
turns into real loss.

## Maintenance commands refuse while someone is writing

`hermes sessions optimize`, `hermes sessions optimize-storage` and `hermes sessions prune`
rewrite the store (VACUUM, a full-text index rebuild, bulk deletes). Running one of them under
a live gateway is how a fleet of agents ends up in the refusal above, so they now check first
and refuse while another process holds the database:

```text
Refusing `hermes sessions optimize-storage`: another process is using ~/.hermes/state.db.
  PID 41230 (hermes gateway run): state.db, state.db-shm, state.db-wal
  PID 41355 (hermes serve --profile work): state.db-wal
Rewriting the database under a live writer is how every agent ends up refusing turns with the
retired state.db-wal error. Nothing is lost.
Stop them first (`hermes gateway stop`, quit the Desktop app, pause cron), then re-run.
Override with --force if you accept the risk.
```

`--dry-run` previews are never blocked. `--force` runs anyway — use it only when you know
the listed processes are idle (a reader you started yourself, for example). The same check
runs when you type `sessions optimize` in the Desktop console.

## Files you may find beside `state.db`

| File or directory | What it is | What to do |
|---|---|---|
| `state.db-wal`, `state.db-shm` | SQLite's live write-ahead log and its shared-memory index. A large `-wal` is normal while the gateway or Desktop is running. | Leave them alone. They shrink on their own at the next checkpoint. |
| `state.db.retired-wal-<timestamp>-<pid>/` | A capture Hermes made of the log copy a process was still holding when it refused to write, plus a `manifest.json` describing it. Forensic evidence, not a backup you restore blindly. | Keep it. If conversations from just before the incident are missing after recovery, attach the directory to a bug report; a maintainer can tell from `manifest.json` whether those frames belong on top of the current file. |
| `state.db.pre-update-emergency-<timestamp>.bak` | A snapshot the Desktop updater takes before it touches the store. | Keep it until you have used the updated app for a while. Restore only with every Hermes process stopped: `hermes sessions recover --source <file> --inspect-only` first. |
| `state.db.corrupt.<timestamp>.bak`, `*.malformed-backup` | Copies of a file Hermes found damaged before it repaired or quarantined it. | Do not restore them over `state.db` — they are the same damage. Keep for a report; safe to delete once you are back to normal. |
| `state-snapshots/` | Quick snapshots `hermes update` and `hermes backup` take. | Restore with every Hermes process stopped; see [`hermes backup`](../reference/cli-commands.md#hermes-backup). |

## When the three steps do not work

If every Hermes process is stopped, `hermes doctor` no longer lists a holder, and the
gateway still refuses to write when you start it, the file itself may be damaged. Stop
everything again and inspect without writing:

```bash
hermes sessions recover --source ~/.hermes/state.db --inspect-only
```

`--inspect-only` never modifies the file. If it reports the store as recoverable, follow the
command it prints, or restore the newest snapshot from `state-snapshots/`. The mechanics
behind all of this are in the developer guide:
[State DB recovery](../developer-guide/state-db-recovery.md) and
[Session storage](../developer-guide/session-storage.md).
