---
sidebar_position: 13
title: "Operating Agent Loops with Hermes"
description: "Design recurring agent workflows with cron, skills, state, delegation, verification, and human gates."
---

# Operating Agent Loops with Hermes

An **agent loop** is a recurring workflow that discovers work, keeps state,
acts only when there is something useful to do, verifies its own output, and
hands off to a human when the risk or ambiguity is too high.

Hermes already has the primitives you need for this. You usually do **not** need
a new model tool or a custom daemon:

| Loop primitive | Hermes primitive |
| --- | --- |
| Schedule / trigger | [`cron`](/user-guide/features/cron), [webhooks](/user-guide/messaging/webhooks), GitHub Actions, or `/goal` for a focused in-session loop |
| Project context | Cron `workdir`, `AGENTS.md` / `CLAUDE.md`, skills, and context files |
| Durable state | A project `STATE.md`, a ticket/PR comment, cron output, or a small file/database updated by a script |
| Triage policy | A skill attached to the cron job, or loaded in the session |
| Maker/checker split | [`delegate_task`](/user-guide/features/delegation) for implementers and independent reviewers/verifiers |
| Safe isolation | Git worktrees, Hermes profiles, restricted toolsets, and checkpoints |
| Delivery / escalation | Cron `deliver`, `attach_to_session`, `[SILENT]`, and gateway threads |
| Cost control | No-agent prefilters, early exits, cadence selection, model pinning, and explicit token/run budgets |

This guide is intentionally conservative: start with report-only loops, prove
value, then add write actions and independent verification.

## Use a loop when it beats a reminder

Good loop candidates have a bounded watchlist and a clear definition of "worth
waking a human":

- CI is red on a watched branch.
- Open PRs are waiting on review, rebase, or failed checks.
- New issues need deduping, labels, or a concise triage summary.
- A dependency or security feed has actionable patch updates.
- A daily briefing should summarize only high-signal changes.

Avoid loops when the task is mostly judgment, broad design, or high-risk change.
Use a normal chat session or `/goal` instead.

## Readiness levels

| Level | Mode | What the loop can do | Minimum bar |
| --- | --- | --- | --- |
| L0 | Draft | Document intent | Purpose, scope, non-goals |
| L1 | Report-only | Read, summarize, update state, notify when useful | Schedule, state file, triage skill, `[SILENT]` for no-op runs |
| L2 | Assisted | Draft small changes or PRs for review | Worktree isolation, independent verifier, max attempts, human gates |
| L3 | Unattended | Act on a narrow allowlist without immediate human review | Budget, run log, denylist, rollback path, proven low false-positive rate |

Do not skip L1 on a production repo. A week of quiet, useful reports is better
evidence than one impressive demo.

## Minimal L1: report-only daily triage

Create a small state file in the project:

```md title="STATE.md"
# Project Loop State

Last run: never

## High Priority

## Watch List

## Resolved
```

Write a tight skill such as `project-triage` that defines what matters for this
repo. Then create a report-only cron job from chat or CLI:

```bash
hermes cron create "0 8 * * 1-5" \
  "Run project-triage. Read STATE.md. Update STATE.md with high-priority and watch items. If there is nothing new or actionable, respond with only [SILENT]." \
  --skill project-triage \
  --workdir /home/me/projects/acme \
  --deliver origin \
  --name acme-daily-triage
```

Key choices:

- `workdir` loads repo instructions and runs tools inside the project.
- `--skill` keeps the cron prompt short and reusable.
- `[SILENT]` prevents notification fatigue.
- To make reports replyable, set `cron.mirror_delivery: true` globally or ask
  Hermes to update the job with `attach_to_session=True` through the `cronjob`
  tool.

## L2: assisted fix with independent verification

Once the report-only loop is accurate, let it draft small changes but keep a
human gate:

```text
Every run:
1. Read STATE.md and prune closed/merged items.
2. Discover new CI failures on main and active PRs.
3. If nothing is actionable, respond with only [SILENT].
4. For one low-risk failure only, create a worktree and draft the smallest fix.
5. Spawn an independent verifier with delegate_task. The verifier must run the
   relevant test/lint command and reject unrelated file changes.
6. Do not merge. Report the branch/PR, test evidence, and any human decision
   needed.
7. If the same item has failed three automated attempts, stop and escalate.
```

A checker should receive different instructions from the implementer. For
example:

```python
delegate_task(
    goal="Verify the proposed CI fix before the cron job reports success.",
    context="""
Project: /home/me/projects/acme
Original failure: <job + log excerpt>
Proposed branch/worktree: <path>
Verifier rules:
- reject if unrelated files changed
- reject if denylisted paths changed
- run: npm test -- --runInBand
- report APPROVE, REJECT, or ESCALATE_HUMAN with command output
""",
    toolsets=["terminal", "file"],
)
```

The implementer does not grade its own homework.

## State and run logs

Every loop should have a state surface a human can inspect without reading chat
history. Common choices:

| State surface | Use when |
| --- | --- |
| `STATE.md` in the repo | The loop is project-specific and state should travel with the code |
| Pattern-specific files like `ci-sweeper-state.md` | Multiple loops operate in one repo |
| GitHub issue / PR comment | The loop is tied to one external artifact |
| Cron output + `context_from` | One job feeds another job, but state does not belong in the repo |
| Small database row | State is private, high-volume, or cross-repo |

At minimum, record:

```json
{
  "run_id": "2026-06-25T08:00:00Z",
  "items_found": 3,
  "actions_taken": 1,
  "escalations": 0,
  "outcome": "success"
}
```

Use append-only logs for trend analysis and postmortems. Keep secrets out of
state files that may be committed.

## Safety checklist

Before a loop edits anything, define these in the prompt or skill:

- **Scope:** repos, branches, labels, paths, and artifacts the loop may inspect.
- **Non-goals:** what the loop must not try to improve opportunistically.
- **Denylist:** secrets, auth, payments, billing, migrations, production infra,
  and any domain-specific risky paths.
- **Attempt cap:** usually three attempts per item before human escalation.
- **Notification rule:** ping only when action or review is needed.
- **Verification command:** exact test/lint/build command required before a
  proposed success.
- **Rollback:** how to pause the job and undo any generated branch/PR.

For unattended jobs, also pin the model/provider on the cron job so a global
model switch cannot silently change cost or behavior.

## Cost controls

Loop cost is dominated by cadence, context size, and subagents. Keep the cheap
path cheap:

1. Use a no-agent or script prefilter when the wake-up condition is mechanical.
2. Exit early when the watchlist is empty.
3. Spawn subagents only for actionable items.
4. Stagger high-frequency jobs.
5. Pin cheap-but-capable models for low-risk triage if quality is sufficient.
6. Keep the triage skill short and store durable project knowledge in skills,
   not in the cron prompt.

A 15-minute CI loop that fully analyzes logs and spawns implementer/verifier
agents every tick is expensive. A 15-minute loop that checks status cheaply and
only wakes the agent when CI is red is often reasonable.

## Pattern examples

### PR babysitter

Purpose: keep watched PRs moving without letting the bot merge risky changes.

- Schedule: every 10-15 minutes during working hours.
- State: `pr-babysitter-state.md` with watched PRs, last action, and blockers.
- Actions: summarize review comments, report red checks, draft minimal fix
  branches for low-risk comments.
- Human gate: merge, security/auth/payment changes, repeated failures.

### CI sweeper

Purpose: react to red CI with fast triage and cautious assisted fixes.

- Trigger: webhook or cron; prefer event-driven webhook for CI failures.
- State: failing commit/job, attempt count, branch/PR link, verifier result.
- Actions: classify flake vs regression vs infra; draft one minimal fix.
- Human gate: infra outages, secrets, flaky-test quarantine, max attempts.

### Changelog drafter

Purpose: convert merged PRs and commits into release notes for review.

- Schedule: daily or on release-prep tag.
- State: last processed commit/tag.
- Actions: draft categorized notes.
- Human gate: breaking changes, security wording, release publishing.

## When to pause or delete a loop

Pause immediately when:

- production incident or major migration is in progress;
- false positives are causing alert fatigue;
- the same item has consumed multiple attempts without progress;
- token budget is exceeded;
- a human reviewer is unavailable and the loop can make write actions.

Delete or retire the loop when the pattern is no longer valuable, or replace it
with an event-driven webhook if polling is the wrong shape.

## Related Hermes features

- [Scheduled Tasks (Cron)](/user-guide/features/cron)
- [Script-Only Cron Jobs](/guides/cron-script-only)
- [Subagent Delegation](/user-guide/features/delegation)
- [Goals](/user-guide/features/goals)
- [Git Worktrees](/user-guide/git-worktrees)
- [Webhooks](/user-guide/messaging/webhooks)
- [Kanban](/user-guide/features/kanban)
- [Automation Blueprints](/guides/automation-blueprints)
