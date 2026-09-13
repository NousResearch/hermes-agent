---
title: "Multi Session Coordination — Coordinate concurrent sessions: claim shared resources"
sidebar_label: "Multi Session Coordination"
description: "Coordinate concurrent sessions: claim shared resources"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Multi Session Coordination

Coordinate concurrent sessions: claim shared resources.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/autonomous-ai-agents/multi-session-coordination` |
| Path | `optional-skills/autonomous-ai-agents/multi-session-coordination` |
| Version | `2.4.0` |
| Author | Tobias Musser (P2ppyJack), Hermes Agent |
| License | MIT |
| Platforms | linux, macos |
| Tags | `coordination`, `concurrency`, `sessions`, `locking`, `registry`, `co-worker` |
| Related skills | [`github`](/docs/user-guide/skills/bundled/software-development/software-development-github) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Multi-Session Coordination Skill

Use the dependency-free `session_coord.py` intention board to coordinate
concurrent sessions, subagents, bots, and scheduled jobs on one machine. The
board is advisory: actors follow the protocol; it does not intercept writes.
Stable board operation is manual/cooperative. Native automatic continuation is
an optional unreleased integration with separate capability checks.

## When to Use

- Two or more sessions, bots, subagents, or jobs can touch the same files,
  memory, skills, UI, remote machines, or scheduler state.
- This session is about to mutate a shared resource, even if no collision is
  currently visible.
- An agentic cron job should defer before loading a model.

**Do not use for:** one process excluding a second copy of itself; use a normal
single-flight lock for that. On a solo machine, use the board's `disable`
switch rather than removing data.

## Prerequisites

- Python 3.8+; SQLite is included in the standard library.
- A POSIX shell for `coord_guard.sh` and the shell selftests.
- Install from the standalone repository with
  `terminal(command="python3 install.py", timeout=600)`. Use
  `terminal(command="python3 install.py --check", timeout=120)` first when
  upgrading.
- After the full install, the usual CLI path is
  `~/.hermes/scripts/session_coord.py`. A skill-only install places it under
  the installed skill's `scripts/` directory instead.

`install.py` writes exact managed board enrollment to the default memory store,
existing bot `SOUL.md` carriers, and non-bot profile memory stores. It migrates
recognized shipped wire-v1 blocks, backs up changed carriers, and preserves
customized or malformed blocks as `ACTION NEEDED`. It never modifies the board
DB or cron manifest.

The installer copies the reconciliation watchdog but does not schedule it. It
also does not install a Hermes plugin, change Hermes configuration, start a
model, or restart a process.

## How to Run

Run the status check first, register once per task, claim the full resource set,
and release only when the task ends:

```python
terminal(command='python3 ~/.hermes/scripts/session_coord.py status', timeout=30)
terminal(command='python3 ~/.hermes/scripts/session_coord.py register --task "<task>" --surface desktop', timeout=30)
terminal(command='python3 ~/.hermes/scripts/session_coord.py claim --id <ID> --res "file:/absolute/path" --res "skill:<name>" --task "<task>"', timeout=30)
# Mutate only after CLAIMED.
terminal(command='python3 ~/.hermes/scripts/session_coord.py done --id <ID>', timeout=30)
```

If the claim says `HELD` or `QUEUED` (exit 75), do not mutate the requested
resources. Use `check`/`inbox`, ask the holder for an ETA, or let a long-lived
shell actor use one bounded `--wait` call. Do not use native `--yield` unless
`hermes_setup.py check` succeeds for the exact profile **and this receiver was
started after that configuration was written**; a check subprocess cannot prove
that an older resident process loaded the plugin.

## Quick Reference

| Command | Purpose |
|---|---|
| `status [--json]` | Show sessions, claims, queues, cron radar, enrollment audits |
| `register --task T [--surface S] [--parent P --slot a]` | Join the board |
| `claim --id ID --res K [--res K...]` | Request an atomic all-or-nothing set |
| `check --res K` / `wait --res K` | Inspect or wait for availability |
| `release --id ID [--res K]` / `done --id ID` | Release one/all and notify |
| `inbox --id ID` | Read release, expiry, and preemption notices |
| `prioritize --session ID --rank N` | Record a user-set priority |
| `preempt --id ID --res K` | Ask a lower-priority holder to checkpoint/pause |
| `pause --id ID --note TEXT` / `resume --id ID` | Manual cooperative pause/resume |
| `steal --id ID --res K --reason TEXT` | Break-glass release; user approval only |
| `cron-guard`, `cron-note`, `wait-for-cron` | Coordinate scheduled jobs |
| `switch`, `enable`, `disable` | Report/change the master switch |

Resource conventions:

| Key | Scope |
|---|---|
| `file:/absolute/path` | File or directory and descendants |
| `skill:<name>` | One skill while edited |
| `memory` | Machine-wide main memory store |
| `ui:desktop` | Foreground desktop control |
| `box:<host>` | Mutating work on a remote machine |
| `cron-store` | Scheduler registry mutation |
| `res:<name>` | Agreed custom resource |

Exit codes are `0` for success/free, `75` for held/queued, `1` for an
operational error, and `2` for invalid CLI arguments. Every command supports
`--json`.

## Procedure

### 1. Register and claim

1. Run `status` with `terminal`; completion criterion: the board path is shown.
2. Register once. Reuse that board id for the whole task.
3. Claim every required resource in one call. Completion criterion: output says
   `CLAIMED` for the complete set.
4. Hold claims until the task is finished; never release per file write.

Directory claims cover descendants after canonicalization. Do not claim broad
roots such as `file:~`.

### 2. Coordinate contention

- Exit 75 means another actor is ahead. Stop before mutation.
- Check `inbox` at natural pauses and before final reporting.
- Priorities come only from the user. Never self-rank or preempt based on an
  agent's own importance judgment.
- On a valid preemption request, finish the current atomic write, save progress,
  then `pause`. Chat negotiates an ETA; only `CLAIMED` authorizes mutation.
- `steal` requires explicit user approval and a recorded reason.

TTL expiry is not proof that the real resource is idle. Inspect resource state
before the first mutation after an expired holder. Use a longer `--ttl` for
multi-hour work and refresh the same idempotent claim after long interruptions.

### 3. Coordinate subagents and bots

Subagents inherit no reliable environment. Put the parent board id and each
child's disjoint resource set in its prompt. Each child registers its own id with
`--parent <ID> --slot <a|b|...>`, claims only its assigned resources, and calls
`done` on its own id. The parent must not pre-claim those same keys: parent-held
claims block children just like any other exclusive holder. Claim final merge or
publication resources only after children release their work keys.

Bot profiles use their `SOUL.md` managed block and register with `--surface
bot:<name>`. Non-bot profiles use their own memory carrier. `status` reports
persona profiles without the exact bot block as `UNENROLLED` and non-bot stores
without the exact board block as `UNWIRED`. A legacy marker substring is not
proof of current enrollment.

The only claim-free bot scope is that profile's internal memory, sessions, and
cron store. A file created in shared space remains shared.

Canonical board-only child and bot text:

- `examples/subagent-prompt.example.md`
- `templates/bot-soul-coordination.md`
- `examples/memory-entry.example.md`

### 4. Coordinate cron jobs

Declare each agentic job's complete footprint in
`~/.hermes/state/cron_resources.json`, including its `wait`/`skip` policy and
critical flag. Source `coord_guard.sh` as wrapper step zero, before single-flight
or model startup:

```bash
. "$HOME/.hermes/scripts/coord_guard.sh"
coord_guard <job-id> wait 900 90 || { [ $? -eq 75 ] && exit 0; }
```

Guard exit 75 is a polite deferral and should normally become wrapper exit 0.
A malformed/missing board fails open so it cannot block a backup. Keep the
manifest synchronized with the job's actual target set. Book every critical-job
pause with `cron-note`; never leave a critical deferral silent.

### 5. Finish

Run `done --id <ID>` after all work and verification. Completion criterion:
`status` shows none of this task's resources held and `inbox` has been checked.

## Optional Native Continuation (Unreleased)

Do not infer native support from a Hermes version, source file, or config key.
A separately supplied `session-coord-native` plugin and compatible Hermes host
must pass the real registration and joined-policy checks. The repository-root
helper is an operator utility and is not copied by a skill-only install.

From a repository checkout, run the read-only check for explicit profiles:

```python
terminal(command='python3 hermes_setup.py check --profile default --plugin-path /absolute/path/to/session-coord-native', timeout=120)
```

Explicit setup:

```python
terminal(command='python3 hermes_setup.py setup --profile default --profile research --plugin-path /absolute/path/to/session-coord-native', timeout=300)
```

Use repeated `--profile` or explicit `--all-profiles`; selection is never
implicit. The plugin path must be a clean Git worktree with committed bytes,
because sanctioned local plugin installation clones its `file://` URL at an
immutable commit.

The helper preflights every selected profile before mutation. It uses Hermes
Plugin Doctor, sanctioned plugin install/enable and config commands, exact JSON
Boolean readback of `delegation.wait_for_all`, and plugin-owned `native-check`.
Only after every profile passes does it write the separate managed native block.
Malformed JSON, unsupported host/policy, custom enrollment, or partial command
state returns not-ready and no misleading native instruction.

A successful result is **configured on disk; restart required** because the
plugin reports `activation=fresh_process_only`. The helper does not restart
Hermes or change model/provider settings.

A shared script-only watchdog is an additional explicit opt-in:

```python
terminal(command='python3 hermes_setup.py setup --profile default --plugin-path /absolute/path/to/session-coord-native --watchdog', timeout=300)
```

Exactly one job is owned through profile `default`. Check mode calls the
plugin's read-only `watchdog-setup --json --check`; uncertain creation is never
retried. Details and recovery invariants are in
`references/automatic-resume.md`.

## Pitfalls

- The board is advisory. A session that never loads the managed instruction can
  still collide; treat `UNENROLLED`/`UNWIRED` as real action items.
- Board-only use is stable; native automatic continuation is separate and
  unreleased. Never promise auto-resume after only `install.py`.
- A native receipt proves prompt admission, not task completion. Unknown
  delivery outcomes are not retried blindly.
- One-shot runs and children inherit no trustworthy target identity; never guess
  session/profile targets.
- Fail-open protects liveness but means a board outage is "flying blind". Report
  it before shared mutation.
- A `wait-for-cron` call while still holding the conflicting resource can
  deadlock against a wait-policy guard.
- `coord_guard.sh` and the shell suites require Bash, so the skill platform gate
  is Linux/macOS even though the Python engine itself is portable.
- Use the `github` skill for repository operations. Unreleased notes remain
  under `CHANGELOG.md` `[Unreleased]`; do not claim a release or upstream
  acceptance before publication and CI evidence exist.

## Verification

From a repository checkout, maintainers run all board suites through `terminal`;
every suite uses temporary DBs/stores. Installed skill bundles do not include the
repository test tree:

```python
terminal(command='bash skills/multi-session-coordination/scripts/selftest.sh', timeout=300)
terminal(command='bash skills/multi-session-coordination/scripts/selftest_priority.sh', timeout=300)
terminal(command='bash skills/multi-session-coordination/scripts/selftest_cron.sh', timeout=300)
terminal(command='bash skills/multi-session-coordination/scripts/selftest_toggle.sh', timeout=300)
terminal(command='python3 skills/multi-session-coordination/scripts/selftest_wakes.py', timeout=300)
terminal(command='python3 -m pytest -q tests', timeout=600)
```

Then run a scratch `register` → `claim --res res:verify` → `done` cycle and
confirm `status` shows no held resource. Verify enrollment by the managed begin
and end markers, not legacy marker counts.

External Hermes/plugin integration may skip when those separately supplied
components are absent. Such a skip verifies no native capability or live
activation. Maintainer/release procedure is in `references/publishing-and-ci.md`.
The standalone repository is MIT licensed and maintained by Tobias Musser
(P2ppyJack), with implementation assistance from Hermes Agent.
