---
title: "Open Steps — Report agent work in plain language for non-engineers"
sidebar_label: "Open Steps"
description: "Report agent work in plain language for non-engineers"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Open Steps

Report agent work in plain language for non-engineers.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/productivity/open-steps` |
| Path | `optional-skills/productivity/open-steps` |
| Version | `1.0.0` |
| Author | Pavlo Kharmanskyi (adapted by Nous Research) |
| License | MIT |
| Platforms | linux, macos |
| Tags | `plain-language`, `reporting`, `non-technical`, `status`, `handover`, `premortem`, `verification`, `productivity` |
| Related skills | [`decision-questionnaire`](../../optional/productivity/productivity-decision-questionnaire.md), [`weekly-review-planning`](../../bundled/productivity/productivity-weekly-review-planning.md), [`humanizer`](../../bundled/creative/creative-humanizer.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Open Steps Skill

Eight ways of talking to the person running the agent who does not read code:
the founder, product owner or operator who asked for the work and now needs to
know whether it is done, what they must do themselves, and what to pick next.
Same facts as the engineering version, one screen, bad news on its own row,
every "yes" naming its proof, "not checked" where nothing was checked.

It changes what the agent *says*; it never changes the work. It does not add
facts, drop warnings, or soften risks. It is not a writing-style humanizer
(that is `humanizer`) and not a code-review skill.

## When to Use

- Work wraps up and the user asks "are we done?", "what happened?", "status?"
- You need the user's hands: run a command, paste a secret, click, approve.
- You are about to ask a technical question or offer options.
- Something hard to undo is about to be agreed: a contract, purchase,
  migration, launch, price change.
- The user asks "what's next?", "what's left?", "what's blocked?"
- Another session or agent claims it finished and the user wants it checked.
- Any text - yours, a report, an error - reads like engineering.
- The user asks where the project as a whole stands.

## Prerequisites

None to be useful. `git` and `gh` let the modes measure instead of guess;
without them more of the output honestly says "not checked". `bash` for
`scripts/census.sh` (big-picture mode).

## Quick Reference - the routing table

Pick the mode, then `skill_view(name="open-steps", file_path="references/<mode>.md")`
and follow it. One mode at a time; the modes hand off to each other by name.

| Mode | Fires when | One-line contract |
|---|---|---|
| `done-or-not` | work wraps up; "are we done", "report", "status" | ten lines: lead, checkmark table, verdict (done / needed from you / new debt / safe to close); saves a handover report |
| `step-by-step` | you need the user to act; "walk me through it", "what do I do" | earn the ask first, then one action per step, commands labelled by what they touch, a way to check |
| `ask-simple` | before any technical question or option list | plain question + why it matters + what changes later + easy to undo; six-check table for structural choices; ONE marked recommendation |
| `what-could-go-wrong` | before anything hard to undo; "premortem", "poke holes" | decision brief → fresh `delegate_task` subagent attacks it → verdict first, nine areas accounted for |
| `whats-next` | "what now", "what's left", "what's blocked" | read report / local state / PRs / backlog; verify ready work; two lists; one recommended next task |
| `check-work` | another session says it is done; "check the others", "can we merge it" | treat the report as a claim; verify each at its source; `Claimed: … Measured: …`; closing block |
| `say-simple` | "say it simply", "wait, what?", "too long", any pasted text | restate for a non-coder: same facts, every warning kept, half the length; `N` = exactly N points |
| `big-picture` | "where are we", "what have we built", "what is stale"; after a report | keeps `BIG-PICTURE.md` between markers: features + stage + measured age/wiring + sourced backlog |

They work as a loop: `whats-next` picks the work, `step-by-step` walks the user
through their part, `done-or-not` reports the result, `check-work` accepts what
other sessions did, `ask-simple` handles the questions on the way,
`what-could-go-wrong` attacks anything hard to undo before it is agreed, and
`say-simple` rescues any text that still reads like engineering.

## Procedure

1. Detect the user's language from the conversation and write in it. Code,
   file names, commands and identifiers stay English.
2. Match the trigger to ONE mode in the table; load its reference.
3. Gather proof that takes seconds (`git status --porcelain`, `git log`,
   `gh pr checks`). Never re-run a test suite for a report; use results this
   session already produced. Not confirmable in seconds → "not checked".
4. Write to the mode's shape. Translate every template label into the user's
   language; keep the shape.
5. Save what the mode saves. Reports go under
   `${HERMES_HOME:-$HOME/.hermes}/open-steps/reports/<project-folder-name>/`
   (`latest.md` overwritten, `history/<YYYY-MM-DD-HHMM>.md` appended), never
   into the user's project. `BIG-PICTURE.md` is the one file written into
   the project, only between `<!-- open-steps:begin -->` / `<!-- open-steps:end -->`
   markers, and excluded via `.git/info/exclude`, never `.gitignore`.
6. Hand off by name when the moment changes: a needs-you item goes to
   `step-by-step`, a real fork to `ask-simple`, "is this true?" to `check-work`.

## Pack-wide invariants (these rules ARE the skill)

1. **One screen.** About ten lines, fifteen the ceiling. Twelve rows is a wall
   in a table costume.
2. **Lead with the outcome**, best thing first, never the chronology.
3. **Every "yes" names its proof; no proof → "not checked".** Silence reads as
   "nothing there", so say what you did not check.
4. **Bad news gets its own ⚠️ row**, never buried inside another line. An
   unhandled security risk or data loss is spelled out in full - the one
   thing never compressed.
5. **Add nothing, drop no bad news.** Every number, warning and caveat in the
   source survives a rewrite; numbers stay exact.
6. **Jargon → what the user would see.** Commit SHAs, branch names, build IDs
   stay out unless they name an action ("review PR #892" stays).
7. **A recommendation is required.** Never lay out options and stop; never
   end on a question mark when you hold the measurements.
8. **Never invent work; every backlog item names its source.**
9. **Irreversible actions wait for the user's explicit word** - merging,
   deleting, paying, creating tickets. Verified-ready is reported with proof
   and lands in the "On your word" row. (Deliberate divergence from upstream,
   which merges verified work automatically.)
10. **The agent that helped decide never attacks the decision.** Premortems
    run in a fresh `delegate_task` subagent, every time.

## Pitfalls

1. "New debt? - No" gets written both when there is none and when nobody
   looked. Say "no" only after checking.
2. A green check or a merge does not mean users have it: landed-but-unreached
   is a ⚠️ row naming what would ship it.
3. `printf …; IFS= read -rs VAR`, never `read -rsp`: in zsh `-p` reads from a
   coprocess and the secret file is written blank (macOS defaults to zsh).
4. A secret never travels through chat or a command argument; the templates in
   `step-by-step` prompt for it and confirm by byte count, never by content.
5. Quiet code is not dead code: quiet + still wired in is `stable`. Never
   propose retiring on age alone.
6. Restating is not editing: if you notice your original was wrong, say so
   plainly; never fix it silently inside the "simpler" version.
7. "I don't understand" about your own answer is feedback. Restate at a lower
   altitude; do not defend.

## Verification

- The report fits one screen and the verdict table has all four rows filled.
- Every ✅ row names its proof; every unverified claim says "not checked".
- Report file exists at `${HERMES_HOME:-$HOME/.hermes}/open-steps/reports/<project>/latest.md`
  with part two (technical detail) present when anything changed.
- `bash "${HERMES_SKILL_DIR}/scripts/census.sh" .` prints `MEASURED`, `AGE`
  and `PART` lines in a git repo and exits 1 with a message outside one.

## References

| File | Load when |
|---|---|
| `references/<mode>.md` (eight files, names in the table above) | that mode fires |
| `references/what-could-go-wrong-premortem-prompt.md` | dispatching the premortem subagent - pass it verbatim as `goal` |
| `references/what-could-go-wrong-why-these-rules.md` | you are tempted to skip the fresh agent or pad the risk list |
| `references/big-picture-rationale.md` | the map's rules feel excessive; the traps the script cannot catch |
| `references/sessions/<mode>-*.md` | before/after on real sessions; read one before your first report in a mode |

Upstream: kharmanskyi/open-steps (MIT). The upstream Stop hook, output style
and routing block are not ported: in Hermes this skill is chosen, not forced.
