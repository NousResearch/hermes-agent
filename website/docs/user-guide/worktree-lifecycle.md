---
title: Worktree Lifecycle for Teams
sidebar_label: Worktree Lifecycle
---

# Worktree Lifecycle for Notebook and VPS teams

Hermes Worktree Lifecycle (WTL) gives every writable task its own Git worktree, branch, writer lease, and runtime namespace. It prevents one Codex/Claude/Grok task from switching the shared project folder out from under another task.

## Operating model

- Keep one canonical checkout per project for fetch, worktree administration, and release operations.
- Create one managed task worktree per writable task.
- Allow one writer machine per task. Reviewers inspect the same path and commit read-only.
- Move a task between Notebook and VPS with `handoff` then `accept`; never copy an active writer lease.
- Remove worktrees only with `cleanup`: six gates, a dry-run record, and 72-hour quarantine.

Default paths:

```text
Notebook: ~/Documents/Worktrees/<project>/<staff>/<task>-<slug>
VPS:      /home/linux-nat/.worktree/<project>/<staff>/<task>-<slug>
Branch:   task/<staff>/<task>-<slug>
```

For a team, point every connected machine at the VPS authority (the path is an example and must be access-controlled):

```bash
export HERMES_WORKTREE_REGISTRY='ssh://linux-nat@103.142.150.185/home/linux-nat/.hermes/worktrees/registry.json'
```

The Manager holds an OS file lock on the VPS for each transaction and writes the JSON atomically. A successful connected read/write refreshes a local cache. If the VPS is unreachable, only the machine already holding an unexpired lease may run:

```bash
hermes worktree status --task-id WTL-123 --machine-id notebook-nat --offline
```

Offline mode cannot open, transfer, close, import, abandon, or clean up tasks.

## Start and enter work

Preview first; the preview does not change Git or the registry:

```bash
hermes worktree open \
  --project-id hermes-agent --staff-id nat --task-id WTL-123 --slug fix-router \
  --repo /path/to/canonical \
  --machine-id notebook-nat
```

บน Notebook ถ้าไม่ระบุ `--root` ระบบจะใช้ `$HOME/Documents/Worktrees` โดยปริยาย หลังเจ้าของอนุมัติให้เรียกซ้ำพร้อม `--apply` ส่วนงานเดิมใช้:

```bash
hermes worktree status --task-id WTL-123
hermes worktree enter --task-id WTL-123
hermes worktree doctor
```

Do not run `git checkout` in a worktree to reuse it for another task.

## Notebook to VPS handoff

The source must be clean and fully pushed:

```bash
hermes worktree handoff --task-id WTL-123 --to-machine vps-linux-nat --apply
```

On the receiving machine, use its canonical repo and registered root:

```bash
hermes worktree accept --task-id WTL-123 \
  --machine-id vps-linux-nat \
  --repo /home/linux-nat/SynerryTools/hermes-agent/main \
  --root /home/linux-nat/.worktree \
  --worktree-path /home/linux-nat/.worktree/hermes-agent/nat/WTL-123-fix-router \
  --apply
```

The source lease is removed before the destination receives a new lease.

## Close and cleanup

Close records review or merge state; it does not delete:

```bash
hermes worktree close --task-id WTL-123 --merged --merge-sha <sha>
hermes worktree cleanup --task-id WTL-123
```

The first cleanup command is a dry-run. It must show 6/6:

1. clean worktree;
2. no unpushed commits;
3. no writer lease or registered process;
4. merged or explicitly abandoned by the owner;
5. recovery evidence exists;
6. cleanup dry-run is recorded.

`cleanup --apply` first enters 72-hour quarantine. A later `cleanup --apply`, after the quarantine and a second 6/6 check, uses `git worktree remove`. Direct filesystem deletion is not supported.

## Existing worktrees

Scan is read-only:

```bash
hermes worktree scan --repo /path/to/canonical
```

It classifies each worktree as `managed`, `unknown`, or `broken`, and reports code/dependency/cache/build bytes. Import requires stable project/task/staff/machine identifiers and owner approval; imported work starts `PAUSED` without a writer lease.

## Disk policy and PDCA

| Usage | Policy |
|---:|---|
| below 70% | normal |
| 70% | warning and cleanup planning |
| 85% | stop creating worktrees |
| 90% | recovery mode; stop work that increases disk usage |

Run a light report every 24 hours and a cleanup-review report every 168 hours:

```bash
hermes worktree report --record --json
hermes worktree report --record --cleanup-review --json
```

For a no-LLM scheduler, install the packaged entrypoint under the active Hermes home after this version is deployed, then register two script-only jobs:

```bash
cp scripts/worktree-lifecycle/pdca_report.py ~/.hermes/scripts/wtl-pdca-light.py
cp scripts/worktree-lifecycle/pdca_report.py ~/.hermes/scripts/wtl-pdca-cleanup.py
hermes cron create "every 24h" --name wtl-light-check --no-agent \
  --script wtl-pdca-light.py --deliver local
hermes cron create "every 168h" --name wtl-cleanup-review --no-agent \
  --script wtl-pdca-cleanup.py --deliver local
```

The entrypoint detects `cleanup` in its installed filename and records the 168-hour review. Verify with `hermes cron list`, `hermes cron run <job>`, and `hermes cron status`. Do not activate jobs from an unmerged source worktree.

These reports show state counts, blocked tasks, per-project bytes, and cleanup candidates. They never delete a worktree. In production, register these commands with Hermes cron only after the WTL-enabled Hermes version is installed on that machine; keep the scheduler output local or route it to an approved team channel.

The PDCA loop is continuous, not end-of-project only:

- Plan: create task identity, path, runtime budget, owner, and acceptance evidence.
- Do: work in the registered task worktree with one writer.
- Check: 24-hour health, 168-hour cleanup review, tests, Git and runtime evidence.
- Act: handoff, pause, fix drift, quarantine, or archive through the Manager.

## Legacy command compatibility

`scripts/hermes_newchat.py --task-id ...` delegates creation to WTL. `scripts/hermes_worktree_route.py --task-id ... --registry ...` resolves the exact registered task. When `HERMES_WORKTREE_REGISTRY` is configured, `scripts/hermes_write_permit.py` refuses a permit if task/path/branch/writer state does not match WTL.
