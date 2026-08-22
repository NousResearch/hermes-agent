---
title: "Skill Behavior Validation — Replay a prior case to validate a skill's behavior change"
sidebar_label: "Skill Behavior Validation"
description: "Replay a prior case to validate a skill's behavior change"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Skill Behavior Validation

Replay a prior case to validate a skill's behavior change.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/skill-behavior-validation` |
| Version | `0.1.0` |
| Author | chelsealong, Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `skills`, `validation`, `review`, `replay`, `regression` |
| Related skills | [`hermes-agent-skill-authoring`](/docs/user-guide/skills/bundled/software-development/software-development-hermes-agent-skill-authoring), [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Skill Behavior Validation Skill

Validate a substantial behavioral edit to a SKILL.md by replaying a real prior
case through it, instead of trusting that the new prose reads correctly. This
is a workflow the agent follows itself when editing a skill; it is not a
runtime gate, a daemon, or a mutation lock on skill files.

## When to Use

Use after any edit that changes a skill's **priority ordering, safety/
classification rule, evidence requirement, routing decision, remediation
step, or completion criterion** — anything that could change what the agent
does the next time the skill fires.

**Skip for:** typo fixes, formatting/reflow, and purely factual reference
updates (a version number, a tool name, a URL) that cannot change a decision.
If you cannot describe a decision the edit could flip, it is nonbehavioral —
skip this workflow and say so.

## Prerequisites

- The skill file being edited, plus any skill it links to (`related_skills`,
  files under its `references/`) and any sibling skill covering overlapping
  ground.
- A real prior case the old skill got wrong or handled ambiguously — a past
  session transcript, an incident writeup, or a reported failure. Do not
  invent a synthetic case as a substitute; a fabricated case tends to exercise
  only the rule you already know you changed.
- `delegate_task` to spawn an independent reviewer (a fresh subagent, not the
  editing agent) for Step 6.

## How to Run

Follow this inline whenever a skill edit meets the trigger above. There is no
separate command — `read_file` the skill and the prior case, `search_files`
the rest of the instruction surface, edit with `patch`, and replay.

## Quick Reference

| Step | Action | Tool |
|---|---|---|
| 1 | State old failure + intended change | — |
| 2 | Search full instruction surface for conflicts | `search_files`, `read_file` |
| 3 | Replay the prior case chronologically | `read_file` |
| 4 | Run a negative control | `read_file` |
| 5 | Confirm the *decision* changed, not just the wording | — |
| 6 | Independent adversarial review: PASS / PARTIAL / FAIL | `delegate_task` |
| 7 | Patch and re-replay until clean | `patch` |
| 8 | Save a validation artifact | `write_file` |

## Procedure

### 1. State the failure and the intended change

Write one or two sentences: what did the old skill get wrong, and what
decision should the new wording produce instead. Keep this as the yardstick
for every later step — a step that doesn't move you toward it is scope creep.

### 2. Search the full instruction surface for conflicts

`search_files` the skill's own body, everything in `related_skills`, its
`references/*.md`, and any sibling skill covering the same ground (the same
survey `hermes-agent-skill-authoring` asks for before drafting). A new rule
that reads correctly in isolation can still contradict or shadow a rule
elsewhere. Note every conflict found; an edit that silently overrides a
sibling skill's rule is not done until that sibling is reconciled too.

### 3. Replay the prior case chronologically

Walk the real case step by step, revealing only the evidence that was
available at each original decision point — not the outcome, not later
context. At each point, ask what the *edited* skill would now direct the
agent to do. Do not skip to the end and check whether the final answer looks
right; a chronological replay catches cases where the new rule fires too late
or too early to have mattered.

### 4. Test a negative control

Pick or construct a case the *old* skill already handled correctly, and
confirm the new wording still produces the old, correct outcome. A rule
broadened to fix the target case can quietly swallow cases it was never meant
to touch. Skip this step only if no such case exists yet, and say so in the
artifact (Step 8) rather than silently omitting it.

### 5. Verify the decision changed, not just the words

The bar is behavioral: does the replay now choose a different action, a
different tool-call order, or a different stopping point than it did under
the old wording? A test or replay that only confirms new prose is present, or
that a helper function ran, proves nothing about behavior — see Pitfalls.

### 6. Get an independent adversarial review

Spawn a fresh reviewer with `delegate_task` that did not write the edit and
has not seen this validation's reasoning. Give it the old skill, the new
skill, the prior case, and the negative control, and ask it to try to break
the new rule — find a case it mishandles, a conflict it missed, an
overgeneralization. It returns one of:

- **PASS** — no material finding remains.
- **PARTIAL** — findings exist but are not yet resolved, or no reviewer was
  available at all. Never report the change as effective while status is
  `PARTIAL`.
- **FAIL** — the edit does not fix the original case, or breaks the negative
  control.

### 7. Patch and replay until clean

On `PARTIAL` or `FAIL`, patch the skill and repeat from Step 3. Stop only when
a review round returns `PASS` with no material finding left open.

### 8. Preserve a concise validation artifact

Record, next to the skill (its own PR description or a short note in
`references/`): the original failure, the case replayed, the negative
control, and the final review verdict. This is the evidence a later editor —
human or agent — checks before trusting the skill's current behavior; it is
not a runtime enforcement mechanism, and nothing outside this workflow reads
or blocks on it.

## Pitfalls

- **Tautological replay.** Asserting that the new sentence is present, or
  that a step "ran," is not the same as asserting the agent's decision
  changed. Assert the observable choice (which action, which order, where it
  stopped), not that a line executed.
- **Stale case.** Confirm the prior case's referenced files/behavior still
  match current `main` before replaying against them; a case built on code
  that has since changed proves nothing about the skill today.
- **Precedent that doesn't transfer.** A pattern copied from a sibling skill
  can be wrong here even if it reads naturally — check *why* the sibling
  needed it before reusing it.
- **Skipping the negative control.** Fixing the reported case by widening a
  rule is easy; proving the widened rule didn't break something the old rule
  got right is the part that's actually load-bearing.
- **Claiming effectiveness at `PARTIAL`.** No independent reviewer available
  is not license to call it `PASS` — record `PARTIAL`/unverified honestly.
- **Treating this as runtime enforcement.** This workflow runs when an agent
  is authoring or reviewing a skill edit. It is not a mutation gate, a
  filesystem watcher, or a check that blocks unrelated skill usage.

## Verification

- [ ] The edit meets the trigger (priority, safety, classification, evidence,
      routing, remediation, or completion-rule change) — or was correctly
      skipped as nonbehavioral.
- [ ] The full instruction surface (linked skills, references, overlapping
      siblings) was searched for conflicts.
- [ ] A real prior case was replayed chronologically, not reconstructed from
      its known outcome.
- [ ] A negative control was run, or its absence was noted in the artifact.
- [ ] The verified change is a changed decision/order/stopping point, not
      just changed wording.
- [ ] An independent reviewer (via `delegate_task`) returned a verdict; status
      is `PARTIAL` if none was available.
- [ ] No material finding remains open, and the overall verdict is `PASS`.
- [ ] A concise validation artifact was saved alongside the skill.
