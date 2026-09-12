---
title: "Dynamic Workflow — Plan-in-code fan-outs, adversarial verification, waves"
sidebar_label: "Dynamic Workflow"
description: "Plan-in-code fan-outs, adversarial verification, waves"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Dynamic Workflow

Plan-in-code fan-outs, adversarial verification, waves.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/autonomous-ai-agents/dynamic-workflow` |
| Path | `optional-skills/autonomous-ai-agents/dynamic-workflow` |
| Version | `2.0.0` |
| Author | Teknium + Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `orchestration`, `fan-out`, `subagents`, `delegation`, `verification`, `migration`, `audit`, `research`, `campaign` |
| Related skills | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent), [`simplify-code`](/docs/user-guide/skills/bundled/software-development/software-development-simplify-code) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Dynamic Workflow Skill

Runs large fan-out work as a workflow: the plan, the loop and every intermediate
result live in a script and on disk, so the parent's context holds only verified
results. Covers one-shot fan-outs, adversarial convergence (attempts + refuters),
and multi-wave campaigns that integrate dozens of worker branches. It does not
make `delegate_task` durable across restarts; that is the kanban swarm's job.

## When to Use

Reach for it when the unit of work is clear (a file, an endpoint, a record) and
there are more units than one context can hold. Skip it for under ~10 units or
for serial chains. For a refactor or fix campaign on hermes-agent itself, load
`hermes-agent` (the dev workflow) alongside; this skill owns the fan-out shape.

## Prerequisites

- `delegate_task` available and `delegation.max_concurrent_children` sized for
  the wave (default 10; the runtime rejects a `tasks=[]` larger than that with a
  clear error rather than queueing). `delegation.max_spawn_depth >= 2` only if
  children must fan out themselves.
- A writable run directory resolved from the terminal environment's temp dir
  (`$TMPDIR`, else the platform temp dir). Never a literal `/tmp`: Termux has no
  `/tmp`, native Windows breaks on it. Use `<tmp>/wf_<name>_<uuid>/`, unique per
  run, so an interrupted earlier run cannot leave stale outputs to be misread.
- `execute_code` for the deterministic layer (only `web_search`, `web_extract`,
  `read_file`, `write_file`, `search_files`, `terminal`, `patch` exist inside it).

## How to Run

Two layers, split by a real capability boundary:

| | Layer A - `execute_code` script | Layer B - `delegate_task` batch |
|---|---|---|
| Use for | DETERMINISTIC work: fetch N URLs, parse N files, run N commands, template N outputs, build manifests, merge outputs | LLM-JUDGMENT work: classify, review, decide, write, refute, refactor one unit |
| Holds | loop, branching, intermediate variables | nothing; one call with `tasks=[...]`, each task its own isolated agent |
| Tools | the sandbox set above; it can NOT call `delegate_task` | the parent's toolsets, inherited unchanged (no per-task narrowing); children lose `delegate_task`, `clarify`, `memory`, `send_message`, `cronjob_manage` |
| Concurrency | yours (`ThreadPoolExecutor`, batches) | bounded by `delegation.max_concurrent_children` |
| Cost | tool calls only | one full agent tree per task; multiplies linearly |

Do the deterministic part in Layer A first, fan out only the irreducibly-LLM
step in Layer B, synthesize on the parent.

### Background-first: results re-enter as messages

A top-level `delegate_task` returns immediately with one handle per task; each
child's result re-enters the conversation as a new message when it finishes. You
cannot read `out_*.csv` on the line after the call. Finish whatever does not
depend on the children, give a one-line status, and END YOUR TURN; act on each
result message as it lands. An ordinary follow-up user message does not cancel
children; `/stop`, `/new` and process exit do. Only a delegation issued by an
orchestrator subagent (depth > 0) is synchronous.

## Quick Reference

- Unit must be answerable without sibling output, else it is serial.
- Manifest: one unit per line in `<run>/manifest.jsonl`; print count + run dir.
- Per child: ~8-12 mechanical edits, or ~2-3k lines of reading, or ~50-70 KB of
  corpus; size by the LARGEST unit. Structured output goes to files, never the
  `summary` field (it truncates under load); delimiter-separated lines over JSON.
- Parent verifies file count and per-run freshness before merging.
- A "stalled" child usually completed its write; check the filesystem first.
- Scoped slice first (one directory, 20 records), report token cost, then scale.

## Procedure

### One-shot fan-out

1. Decompose into independent units.
2. Layer A pre-pass writes the manifest.
3. Size chunks against the limits above; for more tasks than
   `max_concurrent_children`, issue bounded waves yourself.
4. Layer B: one `delegate_task(tasks=[...])`; each task reads its slice, writes
   `<run>/out_<i>.csv`, prints a status word, stops.
5. End the turn. As result messages arrive, read the files, verify, merge; the
   cross-cutting synthesis stays on the parent.

### Adversarial convergence (finding-quality work)

1. Independent attempts: the SAME question to N children (2-4) with DIFFERENT
   framings in each `context`, each writing one claim per line to
   `<run>/attempt_<i>.md`. Located, individually falsifiable claims only
   ("`POST /api/users/:id/role` in `src/routes/users.ts:142` has no role check");
   a refuter cannot break "the auth layer has problems".
2. Merge and dedupe on the parent; note the agreement count per claim.
3. Refuters: a second batch told to BREAK each claim with counter-evidence,
   emitting `claim_idx|survives|counter_evidence`. Give them the sources, not the
   attempts' reasoning.
4. Surface only survivors; drop refuted claims with a one-line reason.
5. Feed new claims from round 2 through one more refutation; stop when a round
   adds no survivors, cap at 3 rounds.

The same mechanic protects the parent from its own wrong premises: when you
hand children a heuristic ("every patch target on a facade is a dead seam"),
tell them to refute it with evidence before acting on it. Four squads doing so
turned a 647-site blanket rewrite into 59 real fixes and saved 130+ green tests.

### Campaign shape (dozens of workers, several waves, hours)

The one-shot recipe does not scale to a whole-codebase pass. What did:

1. Measure first (LOC, hotspots, dead symbols, oracle corpora) and write ONE
   shared `BRIEF.md` plus a per-cluster `task_<cluster>.md`. Every child reads
   both. When the fleet drifts (children shaving docstrings instead of cutting
   code), patch the brief once and steer; re-dispatched children inherit the fix.
2. Exclusive ownership: one cluster of files per worker, edits outside it are
   discarded at integration. Sub-fan-outs inside one file own line RANGES and
   define helpers inside their range so diffs merge cleanly.
3. Commit per verified step, locally, no push, no PR, no rebase from children.
   Committed state is the only handoff; every worker that died mid-campaign lost
   exactly its uncommitted tail. A worker sharing a worktree index commits with
   `git commit -- <paths>` only; a bare commit swept a sibling's staged hunks.
4. Fleet size 12-16 concurrent. Above ~40 processes on one OAuth grant the
   hourly token refresh stampedes into 401s and kills the wave. Queue the rest.
5. Parent liveness: a child reporting `completed` with a few dozen log lines, or
   with 0 commits on its branch, has not finished; look for its sub-branches or
   re-dispatch it with the predecessor's worktree and diff.
6. Integration per round: freeze a base SHA, rebase clean branches mechanically,
   give each conflicting branch its own rebase worker ("main's behaviour wins,
   re-applied inside the new structure"), merge onto one integration branch,
   run the FULL suite on the combined tree. Collisions that every branch passed
   alone appear only here. The next round branches from the integrated commit
   so its workers cannot conflict with each other.
7. Before declaring a round integrated: `git rev-list --count <integration>..<branch>`
   is 0 for EVERY branch. Workers keep committing after you merge their tip;
   168 commits across six slices were once left behind that way.
8. Test runs: exactly one runner on the box, behind a lock file, at high `-j`.
   Many parallel low-`-j` runners were slower AND killed each other's process
   groups. Red files are re-run on a bare `origin/main` worktree in the same
   venv; identical per-file failure sets are pre-existing, not yours.
9. Forward-port at the end, not per round: freeze main's SHA, fan out the
   conflicted files by directory to workers editing ONE merge worktree with
   no commits, then the parent commits the merge once. CI never runs on a
   conflicted PR, so re-merge main before every push.
10. Live QA is its own wave: one squad per surface, isolated `HERMES_HOME`,
    expectation written before the check, evidence on disk, report only, and a
    PR-vs-main difference is the only thing that counts as a regression. Green
    unit tests missed the one P0 (a logged-in code path no test exercised).
11. Reviewer claims get the same treatment as child claims: A/B against the
    base before "restoring" anything. Several confidently stated review deltas
    already behaved that way on base.
12. A parent restart needs a `HANDOFF.md`: why it died, which handles are dead,
    per-branch scorecard (LOC delta, import smoke, targeted tests), and the exact
    re-dispatch text. Snapshot every dirty worktree into a `wip:` commit first.

## Pitfalls

- Calling `delegate_task` inside an `execute_code` script: not in the sandbox.
- Synthesizing on the same turn as the fan-out call: the files do not exist yet.
- Promising background-durable-for-days from `delegate_task`: it is turn-scoped
  and dies with the process. Durable graph = kanban swarm; one-off = `cronjob`.
- Trusting `summary` for content, or `status=completed` for completion.
- Same framing in every "independent" attempt: they collapse to one answer.
- `git stash` anywhere in a worktree campaign: `refs/stash` is shared across
  worktrees and another worker will pop your edits. Compare via a temp worktree.
- Reporting a hit target when the honest number is lower. Say "16% so far, here
  is the path to 30%" and run the next round.

## Verification

- Manifest line count matches the expected unit count.
- Every `out_*.csv` exists and was written this run.
- Every dropped claim has recorded counter-evidence; every surfaced claim went
  through refutation.
- Campaign: every branch at 0 unmerged commits, full suite on the integrated
  tree with reds triaged against bare main, live QA report per surface, token
  cost reported on the scoped slice before the full run.
