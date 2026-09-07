---
title: "Grill Me — Adversarial plan interview before implementation"
sidebar_label: "Grill Me"
description: "Adversarial plan interview before implementation"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Grill Me

Adversarial plan interview before implementation.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/software-development/grill-me` |
| Path | `optional-skills/software-development\grill-me` |
| Version | `2.0.0` |
| Author | Rafael Zendron (rafaumeu) + Matt Pocock (mattpocock/skills, grilling) + Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `planning`, `adversarial`, `interview`, `decision-tree`, `pre-implementation`, `review`, `alignment` |
| Related skills | [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review), [`subagent-driven-development`](/docs/user-guide/skills/optional/software-development/software-development-subagent-driven-development), [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Grill Me

Stress-tests a plan through structured adversarial questioning before any
code is written. Models the plan as a **design tree** — every decision
branches into the decisions that hang off it — and interviews the user in
rounds until every branch is resolved and nothing is silently assumed.

Combines the phase discipline of the original with the frontier-rounds
mechanic from mattpocock/skills' `grilling`.

## When to Use

- User says "grill me", "interview my plan", "stress test this idea"
- Before complex work: auth flows, schema changes, migrations, payments
- A plan has unresolved decisions or seems vague
- Before `subagent-driven-development` decomposition

Do NOT use for existing code (use `requesting-code-review`) or simple one-off
tasks.

## Prerequisites

None. The skill works on any plan or raw idea.

## Core Mechanic: Frontier Rounds

Map the plan as a design tree. The **frontier** is every decision whose
prerequisites are already settled — the questions you can ask NOW without
guessing at answers you haven't heard yet.

Work in **rounds**: ask the whole current frontier in one message, numbered,
each question carrying your recommended answer. Then wait. A question whose
answer depends on another question still open in this round belongs to a
LATER round, not this one.

Format each round like so:

```
❓ Q1 — <question title>: <question body, options if relevant>
➡️ Recommendation: <your recommended answer + one-line why>

❓ Q2 — <question title>: <question body>
➡️ Recommendation: <...>
```

Each answer reshapes the tree: settled decisions push the frontier outward
and unblock dependent questions. Recompute the frontier and ask the next
round.

**Facts are your job; decisions are the user's.** When a frontier question
needs a fact from the environment (codebase, filesystem, config, docs), find
it yourself with `search_files` / `read_file` / `terminal` — or dispatch a
subagent via `delegate_task` for a heavy exploration. Never ask the user for
anything you could look up. Don't block on an exploration: only the questions
downstream of it wait; ask the rest of the frontier now.

## Question Coverage (work these branches into the tree)

**Understanding** — the real goal and boundaries:
- What is the ACTUAL objective? What is explicitly IN and OUT of scope?
- What are the constraints (time, tech, team, budget)? Who are the users?

**Technical decisions** — for each architectural choice:
- "Why this approach and not X?" / "What happens if Y fails?"
- "What's the worst case?" / "How would you roll back?"
- Cross-reference the existing codebase; if the project already has a
  pattern for this, call it out.

**Edge cases:**
- "What happens if the user does Z?" / "What if dependency X goes down?"
- "What if volume is 100x expected?" / "What are the security implications?"

## Synthesis (when the frontier is empty)

1. Summarize ALL decisions in bullet points
2. List anything left open, and what is explicitly OUT of scope
3. Ask: "Aligned? Should I start implementing, or adjust anything?"

Do not act on the plan until the user confirms shared understanding.

## Pitfalls

1. **Asking questions out of dependency order.** A question that depends on
   an unanswered question is a guess wearing a question mark. Keep it for a
   later round.
2. **Skipping the codebase.** Find facts in code with Hermes tools instead of
   asking the user.
3. **Accepting "I don't know" as final.** Suggest options, explain
   trade-offs, make a recommendation.
4. **Writing code during the interrogation.** Alignment only — code after the
   explicit green light.
5. **Being too agreeable.** Your job is to find problems. If everything looks
   fine, look harder.
6. **Not adapting to the user's language.** Interview in whatever language
   the user speaks.

## Verification

- [ ] Every question in a round had all its prerequisites already settled
- [ ] Provided a recommendation with each question
- [ ] Explored the codebase for facts instead of asking the user
- [ ] Frontier empty (no branch silently assumed) before synthesizing
- [ ] Produced a clear summary of all decisions and open items
- [ ] Confirmed user alignment before stopping
