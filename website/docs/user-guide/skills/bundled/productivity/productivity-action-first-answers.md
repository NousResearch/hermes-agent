---
title: "Action First Answers"
sidebar_label: "Action First Answers"
description: "Use when the user asks for ADHD-friendly, action-first, terse, no-preamble replies or wants communication optimized for starting friction"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Action First Answers

Use when the user asks for ADHD-friendly, action-first, terse, no-preamble replies or wants communication optimized for starting friction. Put the concrete action/result first, cap steps, include time estimates when useful, and end with one next action.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/action-first-answers` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `communication`, `focus`, `adhd`, `concise`, `action-first`, `productivity` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Action-First Answers

## Overview

Use this skill to make replies easier to act on when the user wants focus,
low-friction execution, or ADHD-friendly communication. The point is not to be
short for its own sake. The point is to remove the warm-up, make the first
move obvious, and keep working memory load low.

Action-first means the first line contains the answer, result, or concrete next
action. Context comes after, only when it changes what the user should do.

## When to Use

Load this skill when the user says or implies any of these:

- "action first", "no preamble", "bez wstępu", "krótko", "konkretnie"
- "ADHD", "focus mode", "help me start", "make it easy to act"
- They give terse commands and expect execution rather than explanation
- You are writing a status update, handoff, checklist, or implementation summary

Do not use it when the user asks for a tutorial, long-form essay, narrative
writing, legal/medical nuance, or a full design document. In those cases, use a
short executive summary first, then the fuller answer.

## Reply Shape

Default shape for normal answers:

1. **First line: action/result.** Say the thing the user can do or the outcome
   you achieved. Completion criterion: the user can read line one and know the
   point.
2. **Steps capped at five.** If there are more than five steps, group them into
   phases. Completion criterion: no list has more than five numbered items.
3. **Time estimates when useful.** Add realistic minute estimates for chores,
   setup, waiting, or manual review. Skip estimates for pure facts.
4. **One concrete next action.** End with the next button, command, file, or
   decision. Completion criterion: the final line is not generic encouragement.
5. **No closing fluff.** Do not add "hope this helps", "let me know", or a
   recap that repeats the answer.

## For Coding Tasks

Keep the repo discipline intact. Action-first is a presentation layer, not a
shortcut around engineering work.

1. **Do the work before summarizing.** Use tools to inspect, edit, and verify.
2. **Lead with what changed.** Mention files as `path:line` where possible.
3. **Report real verification.** Include the exact test/lint command and result.
4. **Keep caveats actionable.** If blocked, say what failed and the next command
   or input needed to unblock it.

Good final shape:

```text
Wdrożone: dodałem tryb X i testy przechodzą.

- Zmienione: src/foo.py:42, tests/test_foo.py:18
- Weryfikacja: scripts/run_tests.sh tests/test_foo.py → passed
- Ryzyko: brak migracji danych; zmiana dotyczy tylko walidacji wejścia

Następny ruch: uruchom pełne scripts/run_tests.sh przed merge.
```

## For Planning or Guidance

Use a five-line action card:

```text
Zrób najpierw: <najmniejszy sensowny krok> (~X min).

1. <krok>
2. <krok>
3. <krok>

Uważaj na: <jedno realne ryzyko>
Następny ruch: <konkretny command/file/click>
```

If there are trade-offs, use a tiny table instead of paragraphs.

## Language Rules

- Prefer verbs over labels: "Uruchom", "Sprawdź", "Zmień", "Usuń".
- Prefer concrete nouns over abstractions: file, command, test, URL, decision.
- Use short paragraphs, usually one to three sentences.
- Avoid apology loops. If something failed, state the failure and the next fix.
- Avoid motivational reassurance unless the user explicitly asks for coaching.

## Common Pitfalls

1. **Fake brevity.** A short answer that omits the needed command is worse than
   a longer useful one. Include the actionable detail.
2. **Too many bullets.** Ten bullets still overload working memory. Group into
   phases or pick the next five actions.
3. **Preamble with a new label.** "Quick summary:" is still preamble if it
   delays the result. Start with the result itself.
4. **Ending with an invitation.** "Let me know if..." gives no next action.
   End with the next concrete move.
5. **Skipping verification.** For coding work, action-first still requires real
   tests or a clear blocker.

## Verification Checklist

- [ ] First line contains the result, answer, or next action
- [ ] Numbered steps are capped at five or grouped into phases
- [ ] Any time estimate is realistic and useful
- [ ] Coding summaries include real tool/test output
- [ ] The final line gives one concrete next action or stops cleanly
- [ ] No preamble, filler, generic reassurance, or closing fluff
