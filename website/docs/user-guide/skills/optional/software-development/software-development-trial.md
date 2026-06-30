---
title: "Trial — Use before final responses to block unproven completion"
sidebar_label: "Trial"
description: "Use before final responses to block unproven completion"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Trial

Use before final responses to block unproven completion.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/software-development/trial` |
| Path | `optional-skills/software-development/trial` |
| Version | `0.5.1` |
| Author | Abdulrahman Qasem (Da7-Tech), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `verification`, `judging`, `false-done`, `quality-assurance`, `evidence`, `high-stakes` |
| Related skills | [`subagent-driven-development`](/docs/user-guide/skills/optional/software-development/software-development-subagent-driven-development), [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development), [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Trial Skill

Trial is a pre-delivery evidence gate: the final response remains a private draft
until every user-visible claim is accepted. It sends unsupported claims back to
proof or repair instead of telling the user that the agent almost lied.

## When to Use

- Before every final response about code, file changes, tests, or completion.
- When a green suite may not exercise the behavior the user requested.
- For high-stakes work involving auth, payments, permissions, user data,
  migrations, deletion, security, test edits, or repeated failure.
- Use the fast path for trivial work, but never skip the gate.

## Prerequisites

- No external dependency is required for the gate.
- Run proving commands with `terminal` and inspect related code with
  `search_files`.
- High-stakes work uses `delegate_task` for a fresh-eyes judgment when that tool
  is available; otherwise perform a separate adversarial self-review.

## How to Run

Apply Trial after implementation but before the final response. Hold the proposed
response as a private draft, extract every claim about what changed, what works,
what ran, and whether the task is complete, then bind each claim to a fresh
receipt. Release the response only after every visible claim is accepted.

## Quick Reference

A **receipt** is the exact command run for this task, its exit status, and the
decisive output lines. A named test that was not freshly executed is not a
receipt.

| Internal verdict | Required action |
|---|---|
| `ACCEPTED` | release only accepted claims with concise receipts |
| `NOT_PROVEN` / `NEEDS_EVIDENCE` | keep the draft internal and gather proof or remove the claim |
| `NEEDS_FIX` | keep the draft internal, repair, rerun, and judge again |
| `ESCALATE` | withhold completion until the required review or decision exists |

**Coverage beats green:** evidence counts only when the command would fail if
the user-facing acceptance criterion were false.

## Procedure

1. **Frame** — restate the request as testable acceptance criteria at the
   boundary where the user experiences the result.
2. **Build** — implement the work and use `search_files` to inspect every
   relevant copy, caller, and side effect.
3. **Prove** — use `terminal` to run a fresh proving command for every criterion.
   Add a meaningful check when none would fail on the old behavior.
4. **Draft privately** — prepare the proposed final response without sending it,
   then list every factual and completion claim it contains.
5. **Judge** — reject the draft unless every claim has a covering receipt and
   the receipts agree with the current tree and each other. For high-stakes work,
   use `delegate_task` with only the criteria, claims, diff, and receipts; any
   disagreement keeps the draft blocked.
6. **Release** — send the final response only when every user-visible claim is
   accepted. Negative verdicts remain internal and return the agent to work.

After a bounded repair effort, if proof is still impossible, send only a precise
incomplete status containing the blocker and remaining work. Do not expose
internal courtroom labels, accuse the agent, or soften uncertainty into implied
success.

## Pitfalls

- **A receipt is auditable evidence, not cryptographic proof.** A portable skill
  cannot physically restrain a model that ignores its instructions or fabricates
  tool output; host-captured traces and continuous integration are stronger.
- **Reject stale proof.** Output from another revision or a passing command
  contradicted by a relevant failure blocks release.
- **Gate all visible work claims.** Changed files, test counts, command results,
  current state, and completion are all claims, not just the word "done".
- **Keep trivial work cheap.** One proportionate proving check and a private
  claim scan are enough; no ceremony is required.

## Verification

The published controlled benchmark measured the version 0.4 receipt-and-coverage
rule: agents left a covering test 6/6 with Trial versus 4/6 without it, and the
sampled Trial reports contained no false verification claim. The current fixture,
behavioral mutation grader, and prompts can rerun the protocol, but the historical
run trees and complete reports were not retained, so the aggregate cannot be
independently re-scored.

Version 0.5 preserves that measured rule and adds the private-draft, fail-closed
delivery contract. That contract has deterministic policy, synchronization, and
packed-install coverage in the standalone repository, but has not yet been
re-measured on live agents: https://github.com/Da7-Tech/trial.
