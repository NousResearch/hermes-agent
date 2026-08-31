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
| Related skills | `github-repo-management`, `github-issues`, `github-code-review`, `github-issue-to-pr` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

<!-- EDIT HISTORY (newest first; max ~5 entries, older -> references/edit-history.md)
2026-08-31 | claude-fable-5 | anthropic | desktop session | v2.4.0 BOT ENROLLMENT hardening (Toby spotted the exemption-wording hazard): (1) blurb exemption tightened — "your OWN profile's memory" was blurrable into "things I made need no claim"; now names the EXACT profile-internal list, states self-created files in shared space still need claims, and disambiguates profile memory vs the machine's main memory store (template + example). (2) install.py step 7 wire_bots(): appends the blurb (marker "session-coord (bot-wire v1)") to every existing profiles/*/SOUL.md — default ON, --no-wire-bots/--profiles-dir; closes the each-new-bot-needs-a-manual-paste gap for existing bots. (3) status audits enrollment: persona-bearing profiles missing the marker -> UNENROLLED warning (+ unenrolled_bot_profiles in --json). (4) NON-BOT profiles (Toby spotted this gap too): a profile is a full agent instance whose own memory store the main rule never reaches — install.py step 8 wires the standing rule into each SOUL-less profile's memories/MEMORY.md (--no-wire-profiles), and status flags unwired ones (existing store without the rule; store-less = no evidence, never flagged). selftest_cron.sh 45->51 (133 total); CI wiring smoke step covers both legs.
2026-08-30 | deepseek-v4-flash | custom | desktop session | OFFICIAL-SKILL PACKAGING: repo restructured to the Hermes skill layout (skills/multi-session-coordination/ with SKILL.md + references/ + templates/ + scripts/ + examples/) so `hermes skills install` / tap / direct-URL installs work; frontmatter brought to house standards (author human-first, platforms audited to [linux, macos], description <= 60 chars); body re-ordered to When to Use / Prerequisites / How to Run / Quick Reference / Procedure / Pitfalls / Verification with Hermes-tool framing; install.py gains a skill-bundle install step (--skill-dest / --no-skill); tool location made path-robust (install.py vs skill-bundle installs).
2026-08-30 | claude-opus-4-8 | anthropic | defaults | Procedural quality-review pass (all custom skills): verified against live vendor docs/GitHub via browse-as-me + web, checked for supersession by official bundled skills, spot-checked cited paths/crons/config on-disk. Verdict: current.
2026-08-26 | deepseek-v4-flash | custom | desktop session | WIRE-IN DOCS: new "Wiring it in" section — the standing memory rule is the missing always-call carrier (install.py now writes it by default, marker 'session-coord (wire v1)', --no-wire-memory/--memory-file; canonical entry + delivery paths in repo examples/memory-entry.example.md). Laura-install gap: scripts+skill install ≠ enrollment. Toby's own memory now carries the rule (verified §-delimited, marker=1).
2026-08-26 | deepseek-v4-flash | custom | cli session | GitHub participation: related_skills frontmatter + repo-owner pointer block (bundled github-repo-management/github-issues/github-code-review/github-issue-to-pr) — session-coord is Toby's public repo, not just a port target
-->

# Multi-Session Coordination Skill

Cooperative coordination for concurrent AI-agent sessions, subagents, cron jobs, and
named bots sharing one machine. A SQLite "intention board" with a dependency-free CLI
(`session_coord.py`) and a zero-token cron guard (`coord_guard.sh`): actors announce
themselves, claim resources for the duration of a task, wait politely, and notify each
other on release. It never wraps syscalls — the board is advisory and the protocol makes
it effective. Framework-agnostic: anything that can run a CLI can participate.

## When to Use

- 2+ Hermes sessions may run at once, OR this session is about to mutate a shared
  resource — memory store, skills tree, `~/.hermes/scripts`, a cron store, a remote GPU
  box, the desktop UI.
- You are fanning out subagents or running Bot Mode bots that will touch the same files,
  skills, boxes, or state as other actors.
- An agentic cron job touches shared resources and should defer politely when a session
  is mid-task.

Hermes has **no built-in cross-session task collision detection**: WAL protects the
transcript DB; file locks exist only for auth/plugins state; nothing guards MEMORY.md,
skills, scripts, boxes, or the UI. This skill supplies that layer.

**Don't use for:** per-script mutual exclusion of one job with itself (that is
`singleflight.sh`); solo-operator machines where coordination is overhead (use the
master switch — `disable` — instead of uninstalling).

**Ethos (the co-worker rule):** never compete, never clobber, never silently duplicate
work. If a resource is held, `--wait` or do something else and tell the user; trust the
holder to finish and release promptly — you may be the one holding a co-worker up.

## Prerequisites

- Python >= 3.8 (stdlib only — SQLite is built in). A POSIX shell (`bash`) to run the
  selftests and the cron guard.
- **The CLI.** Two install paths:
  - **Full install (recommended):** `python3 install.py` from the repo
    (`P2ppyJack/session-coord`). Copies the CLI + guard + selftests to
    `~/.hermes/scripts/`, the skill bundle to `~/.hermes/skills/`, and wires the agent
    in (below). Idempotent; never touches an existing board DB or cron manifest.
  - **Skill-bundle install:** `hermes skills install P2ppyJack/session-coord/skills/multi-session-coordination`
    (or `hermes skills tap add P2ppyJack/session-coord`, then install the skill). The
    tool then lives in **this skill's own `scripts/`** directory — substitute
    `<skill-dir>/scripts/session_coord.py` wherever this document writes
    `~/.hermes/scripts/session_coord.py`.
- **Enrollment — the step that makes it a protocol.** The board is advisory: a session
  that never checks it gets no protection. Installing the files does not enroll
  sessions; the standing instruction does. Three carriers:
  - **Sessions:** the standing memory rule (marker `session-coord (wire v1)`) — injected
    into every session, every turn. `install.py` writes it automatically; manual copy in
    `examples/memory-entry.example.md` (canonical entry below).
  - **Bots:** the blurb in `templates/bot-soul-coordination.md`, pasted into the bot's
    SOUL.md (handoff runs inherit no env).
  - **Subagents:** the prompt text in `examples/subagent-prompt.example.md` (children
    inherit no env).
- **State.** The board DB (`~/.hermes/state/session_coordination.db`) and the cron
  manifest (`~/.hermes/state/cron_resources.json`) are created on first use; the
  installer never touches existing ones.

## How to Run

Canonical invocations, through the `terminal` tool (replace `SC` per the Prerequisites
tool-location note; `SC=~/.hermes/scripts/session_coord.py` after a full install):

```bash
# join the board once per task, then claim before mutating, done at the end
SC=~/.hermes/scripts/session_coord.py
ID=$(python3 $SC register --task "memory hygiene sweep" --surface desktop | head -1)
python3 $SC claim --id $ID --res memory --res "file:~/.hermes/skills" --wait --timeout 300
# ... do the whole task ...
python3 $SC done --id $ID
```

```bash
# who is doing what, right now (do this FIRST in any session)
python3 ~/.hermes/scripts/session_coord.py status
```

The CLI fails **open**: if the board DB is unavailable it warns on stderr and proceeds
(exit 0) — coordination must never strand real work. Exit codes: `0` ok/free, `75`
held/queued (same as `singleflight.sh`), `1` error. `--json` on every command.

## Quick Reference

| Verb | Purpose |
|---|---|
| `register --task '...' [--id X] [--surface s] [--parent P --slot a]` | Join the board; prints your id (keep it: `export HERMES_COORD_ID=$ID` for later calls) |
| `claim --id $ID --res <key> [--res ...] [--wait] [--ttl N] [--mode shared]` | Atomically claim everything the task touches; `--wait` polls politely; rc 75 = held |
| `check` / `wait` | See who holds what; block until a resource frees |
| `release --id $ID [--res <key>]` / `done --id $ID` | Free one resource / everything + deregister + notify waiters |
| `status` | Whole-board view: holders, tasks, ranks, waiters, 12 h cron radar |
| `inbox --id $ID` | Read-once notifications (released / EXPIRED / preempt requests) |
| `prioritize --session S --rank N` or `--order "a=1,b=2"` | **User-only**: set/clear ranks (never self-assign) |
| `preempt --id $ID --res <key>` | Ask a strictly-lower-ranked holder to checkpoint + pause + yield |
| `pause --id $ID --note "<checkpoint>"` / `resume --id $ID` | Yield with a reserved resume spot; resume re-acquires atomically |
| `steal --id $ID --res <key> --reason "..."` | Break-glass force release (loud, audited; user approval only) |
| `cron-guard --job <id>` | The guard's board query (used by `coord_guard.sh`) |
| `wait-for-cron --job <id> --timeout N` | Block until a scheduled job's fire is observed |
| `cron-note --job <id> --action paused/resumed --id $ID` | Book responsibility for a paused cron (done-time nag) |
| `switch` / `enable` / `disable` / `switch toggle` | Master switch: report, or turn the whole board OFF/ON (fail-open no-op while off) |

**Resource keys** (claim the narrowest real scope; never `file:~`):

| Key | Covers |
|---|---|
| `file:/abs/path` | File or directory (dir claim covers children; `~`/relative auto-normalizes) |
| `skill:<name>` | A skill being EDITED (read-only use = no claim, just `wait` if held) |
| `memory` | Hermes memory store (singleton — claim for hygiene/bulk-edit work) |
| `ui:desktop` | Desktop UI / foreground control |
| `box:<host-or-ip>` | Mutating work on a remote box |
| `cron-store` | Cron job registry sweeps |
| `res:<custom>` | Anything else — agree on the key in the task description |

## Procedure

**1. Register once** at task start (prints other active co-workers immediately):
```bash
ID=$(python3 ~/.hermes/scripts/session_coord.py register --task "..." --surface desktop | head -1)
export HERMES_COORD_ID=$ID
```

**2. Before writing files / editing a skill / mutating anything shared — check the board, then claim:**
```bash
python3 ~/.hermes/scripts/session_coord.py status
python3 ~/.hermes/scripts/session_coord.py claim --id $ID \
    --res memory --res "file:~/.hermes/skills" --task "..." --wait --timeout 300
```
- Claim **everything the task will touch, up front, in ONE call** — atomic
  all-or-nothing prevents deadlock (never acquire piecemeal).
- Claim granularity = the **task**, not each write. Hold until the task is done.
- If held (rc 75): `--wait` politely, work around it, or surface it to the user.
  **Never** proceed against a held resource; never steal without user approval or hard
  evidence the holder is dead.

**3. Check your inbox at natural pauses** (after long steps, before the final report):
```bash
python3 ~/.hermes/scripts/session_coord.py inbox --id $ID
```
Delivers "released by co-worker", "EXPIRED — holder may have died, VERIFY resource
state", "your claim was force-released (reason)".

**4. When the TASK is done** (one call — releases all claims, notifies waiters,
deregisters):
```bash
python3 ~/.hermes/scripts/session_coord.py done --id $ID
```

### User priority, preemption & pause

Priorities come from the USER, never self-assigned. When the user says "this session is
higher priority" / "do 1, then 2, then 3":
```bash
python3 ~/.hermes/scripts/session_coord.py prioritize --session $ID --rank 1
python3 ~/.hermes/scripts/session_coord.py prioritize --order "3f2a=1,8da2=2,55ae=3"
```
Preempt protocol: requester (must hold a user-set rank below the holder's) →
`preempt --id $ID --res <key>`; holder finishes the current atomic step, checkpoints
durably, then `pause --id $ID --note "<checkpoint>"`; requester claims, works, `done`;
holder `resume --id $ID` re-acquires everything atomically and gets its checkpoint note
back. Queue **fencing**: a free resource is refused (rc 75, `QUEUED`) when a
better-ranked (or equal-ranked, earlier-arrived) live waiter wants it — liveness-gated
so an abandoned waiter can't fence forever. Record the user's words in the claim/preempt
`--reason`.

### Subagents on the board

Children register with **lineage ranks** (parent rank 1 → children `1a`, `1b`; the
family sits between rank 1 and 2; re-ranking the parent re-ranks the family live). Put
IN THE CHILD'S PROMPT (they inherit no env):
```
Coordination: you are a subagent of coordination session <ID>. First run:
  CID=$(python3 ~/.hermes/scripts/session_coord.py register --task "<child task>" --surface subagent --parent <ID> --slot a | head -1)
Before mutating shared resources: claim --id $CID --res <keys> --wait. At the end: done --id $CID.
If your inbox shows a USER-PRIORITY/preempt request: checkpoint state to a file, run pause --id $CID --note "<file>", and report the checkpoint path in your summary.
```
Parents: distinct `--slot` per child in intended order (slot `a` = critical path);
children `done` their OWN id; if the parent is asked to pause, steer/stop children
first; pre-claim family-wide resources at the parent level.

### Bots (Bot Mode) on the board

A bot IS a Hermes profile running concurrently. One shared board — no per-bot scopes:
every actor sees every other actor's claims identically. Enrollment via SOUL.md blurb
(`templates/bot-soul-coordination.md`) — `install.py` appends it to every EXISTING
profile's SOUL.md automatically (marker `session-coord (bot-wire v1)`;
`--no-wire-bots` opts out), and `status` flags persona-bearing profiles missing the
marker as UNENROLLED so a bot created after install can't stay invisible until a
collision. NON-BOT profiles (no SOUL.md) are wired through their OWN memory store
instead — install.py appends the standing rule to `<profiles>/<name>/memories/MEMORY.md`
(`--no-wire-profiles` opts out), and `status` flags a SOUL-less profile whose existing
store lacks it as UNWIRED (store-less fresh profiles are never flagged — no evidence).
Bots register `--surface "bot:<name>"`. The blurb's claim-free exemption
is EXACTLY the bot's profile-internal stores (its own memory/sessions/cron) — files a
bot itself created in shared space still need claims, and profile memory is not the
machine's main memory store. Bot routines (profile cron stores) auto-surface in the
radar as `[bot:<name>]`, but a routine touching shared resources still needs a
**manifest entry** to get advisories/guarding — the merge makes it resolvable, the
manifest makes it declared. Ranks stay user-set; bot-to-bot chat is negotiation,
never a lock.

### Cron jobs on the board

Crons and sessions protect each other BOTH directions:
1. **Manifest** (`~/.hermes/state/cron_resources.json`): per job id, `resources`,
   `policy` (`wait` bounded poll then run, or `skip` instantly), `critical`. Keep it
   current — a job whose footprint changed gets wrong advisories.
2. **Cron-side guard** (`coord_guard.sh`, sourced as **step 0** of a wrapper, BEFORE
   singleflight and before any work):
   ```bash
   . "$HOME/.hermes/scripts/coord_guard.sh"
   coord_guard <job-id> wait 900 90 || { [ $? -eq 75 ] && exit 0; }
   ```
   Fail-open: a broken board never blocks a backup. Guard exit 75 = defer THIS tick
   silently (`exit 0`, never `exit 1` — that would page the user for a polite deferral).
3. **Session-side awareness**: claim/check print a CRON ADVISORY when a manifested job
   fires inside your claim's TTL window and its resources overlap; `status` shows the
   12 h radar. For CRITICAL jobs, never let one silently skip — pick finish-and-release,
   `wait-for-cron`, or pause-with-user-approval (`cron-note --action paused` books it so
   `done` nags if it stays paused).

### Turning coordination off (master switch)

```bash
python3 ~/.hermes/scripts/session_coord.py switch          # report state + what decides it
python3 ~/.hermes/scripts/session_coord.py disable         # OFF (persistent sentinel)
python3 ~/.hermes/scripts/session_coord.py enable          # ON
```
While OFF every verb is a fail-open no-op preserving its stdout contract (`register`
still prints a synthetic id, `cron-guard` emits empty stdout → unguarded). Precedence:
`HERMES_COORD_DISABLED` env > sentinel `~/.hermes/state/coordination_disabled` > default
ON. The shell guard honors it without spawning Python.

## Pitfalls

- **The registry is advisory.** A session that never checks it gets no protection. The
  standing memory rule (Prerequisites) + this skill are what make every session
  participate; after a reinstall or new-machine deploy, verify the entry exists
  (`search_files(pattern='session-coord (wire v1)', path='~/.hermes/memories/MEMORY.md')`
  → exactly 1) and `status` answers.
- **Explicit ids (v2.3.2+):** `register --id <memorable>` works and is the right way to
  pre-mint an id (or set `HERMES_COORD_ID` and omit `--id`). Claims under a
  never-registered id auto-create the session row, so no claim is ever orphaned.
- Don't claim broad dirs (`file:~`) — you'll block everyone. Narrowest real scope.
- `done` at the END of the task, not after each write — mid-task release is exactly the
  clobber window this system closes.
- One-shot `hermes chat -q`, subagents, and bot handoffs inherit no env — put the
  register/claim lines INTO their prompt. Children register their OWN id (`--parent
  $ID --slot a`) and `done` their own id; the parent never shares its $ID.
- Ranks are USER decisions. Never `prioritize` or `preempt` on your own judgment.
- On a preempt request: finish the current atomic step first (never pause mid-write),
  checkpoint durably, THEN `pause --note`. Pausing without a checkpoint note is a
  protocol violation.
- **Manifest drift:** `cron-guard` on a job with no manifest entry runs unguarded
  (fail-open, logged `no_manifest`). Update the manifest entry in the same edit that
  changes a cron script's footprint.
- **Never pause a critical cron without booking it** (`cron-note --action paused`) —
  an unbooked pause is invisible and WILL be forgotten.
- **`wait-for-cron` while still holding the conflicting claim** deadlocks against a
  `wait`-policy guard — release the overlap first, or accept the guard skipping that
  tick.
- Scheduled-fire ETAs come from the cron store's `next_run_at`; a sleeping machine
  pushes reality later — treat ETAs as earliest-case.
- Board unreadable (disk full, corrupt DB): tool fails open with a stderr WARNING —
  treat as "flying blind", tell the user, avoid shared-resource mutation until resolved.
- **Platforms:** the engine is cross-platform Python (CI-verified on Windows via Git
  Bash), but the shipped selftests and `coord_guard.sh` are POSIX shell — hence
  `platforms: [linux, macos]`. Windows users can still run the CLI directly.
- **CI green is the only proof, not a local sweep** — every suite + linter passing
  locally does NOT mean the repo's GitHub CI passes. Push, then poll the Actions run
  before reporting done. Detail: `references/publishing-and-ci.md`.
- Bare `bash` from Python on Windows is the WSL stub — the installer probes for a real
  bash (Git-for-Windows) and skips verification gracefully when none exists. Detail:
  `references/publishing-and-ci.md`.

## Verification

Four live selftest suites (133 checks total), all against scratch DBs — safe to run
anytime. From a full install the suites are at `~/.hermes/scripts/`; from a
skill-bundle install, in this skill's `scripts/`:

```bash
bash <scripts-dir>/selftest.sh          # 28: v1 core (race -> 1 winner, wait/notify, boundaries, TTL, steal) + id resolution + explicit-id/orphan-proofing
bash <scripts-dir>/selftest_priority.sh # 26: v2 ranks/preempt/pause/lineage/fencing + v1-schema migration
bash <scripts-dir>/selftest_cron.sh     # 51: cron leg + v2.2 bot leg (profile stores, [bot:] tagging, collision, broken-store inertness) + v2.4 enrollment audits (bot blurb + profile memory)
bash <scripts-dir>/selftest_toggle.sh   # 28: v2.3 master switch (OFF fail-open no-op, ON restores, env > sentinel, shell guard honors)
```

Then prove the deployment end-to-end:
1. `python3 ~/.hermes/scripts/session_coord.py status` answers with the board path.
2. `search_files(pattern='session-coord (wire v1)', path='~/.hermes/memories/MEMORY.md')`
   → count exactly 1 (or `grep -c` if you are in a shell).
3. Claim a scratch resource and release it: `register` → `claim --res res:verify` →
   `done`; then `status` shows no held resources.

**Repo:** the project is also the standalone public repo `P2ppyJack/session-coord`
(MIT). Changes to the live tool get ported there and the two stay in lock-step;
maintainer workflow, CI matrix, and publishing lessons live in
`references/publishing-and-ci.md` (read before any repo/publish work). Repo-owner
participation: bundled skills `github-repo-management` (tagged Releases, branch
protection), `github-issues` (triage incoming issues), `github-code-review` (community
PRs), `github-issue-to-pr` (fix a filed issue yourself).
