---
title: "Elephant Goldfish — Draft specs with full context, review with fresh subagents"
sidebar_label: "Elephant Goldfish"
description: "Draft specs with full context, review with fresh subagents"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Elephant Goldfish

Draft specs with full context, review with fresh subagents.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/software-development/elephant-goldfish` |
| Path | `optional-skills/software-development/elephant-goldfish` |
| Version | `1.0.0` |
| Author | Google LLC (adapted by Nous Research) |
| License | Apache-2.0 |
| Platforms | linux, macos, windows |
| Tags | `design-docs`, `specs`, `review`, `subagents`, `planning`, `architecture` |
| Related skills | [`requesting-code-review`](../../bundled/software-development/software-development-requesting-code-review.md), [`grill-me`](../../optional/software-development/software-development-grill-me.md), [`subagent-driven-development`](../../optional/software-development/software-development-subagent-driven-development.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Elephant-Goldfish Skill

Design with an elephant, review with a goldfish. You (the elephant) hold the
project context, question the user, and draft a design document. Fresh
`delegate_task` subagents (the goldfish) then review only the document and the
source it references, which exposes the assumptions the elephant never wrote
down. The user resolves trade-offs and approves. This skill covers ideation,
drafting, multi-stage review, applying findings, contract READMEs, and
spec-vs-code drift. It is NOT for reviewing a code diff (use
`requesting-code-review`) and NOT for a quick implementation plan (use the `/plan` command). It does
not write production code.

## When to Use

- A feature needs design decisions before anyone implements it.
- The user asks to brainstorm, draft, or review a spec, RFC, or design doc.
- A finished design doc needs an adversarial review from readers without your context.
- Review findings exist and must be classified and applied.
- Checked-in design docs or READMEs may have drifted from the code.
- A codebase needs per-directory contract READMEs (interface, invariants, side effects).

Skip it for a small, clear bug fix; use the normal coding workflow.

## Prerequisites

- `delegate_task` is required for every goldfish review (brainstorm, review,
  review-loop) and for parallel bootstrap workers. If `delegate_task` is not
  available, STOP the review and tell the user: a review in the same context
  is not a goldfish review. Never fake one by "pretending to forget" in the
  current context. (Bootstrap alone may run its scout steps inline.)
- `clarify` for single-select gates. It is unavailable in cron and other
  headless runs: there, write the pending question into the document's open
  questions section (or the report) and stop.
- `read_file`, `search_files`, `write_file`, `patch`, `terminal` for the
  document and source work.

## Quick Reference

Load a workflow with `skill_view(name='elephant-goldfish', file_path='references/workflows/<x>.md')`.
The user can start one with `/elephant-goldfish <workflow> [args]`, for example
`/elephant-goldfish review docs/designs/webhooks.md`.

| Workflow | Use when | Reference |
| --- | --- | --- |
| brainstorm | Raw idea → stress-tested `concepts_brief.md` | `references/workflows/brainstorm.md` |
| draft | Brief or notes → section-by-section `design_draft.md` | `references/workflows/draft.md` |
| review | Five-stage goldfish review of a design doc | `references/workflows/review.md` |
| review-loop | Repeated rounds of a configured reviewer set until convergence | `references/workflows/review-loop.md` |
| resolve | Classify findings (PROPOSE FIX / CONTEST / ESCALATE / DECLINE) and apply approved edits | `references/workflows/resolve.md` |
| drift-report | Compare design docs or READMEs with current source | `references/workflows/drift-report.md` |
| bootstrap | Create or correct contract READMEs across a directory tree | `references/workflows/bootstrap.md` |

Typical sequence: brainstorm → draft → review (which calls review-loop and
resolve). Each workflow also runs on its own.

Reviewer personas (`references/reviewers/`), keyed by the persona id the
workflows use:

| Persona id | File | Lens |
| --- | --- | --- |
| `egm-zoom-out` | `zoom_out.md` | Simplicity, YAGNI, reuse |
| `egm-comprehension` | `comprehension.md` | Naive reader, undefined terms |
| `egm-design` | `design_critic.md` | Architecture, scale, risk |
| `egm-structural` | `structural.md` | Layout and template compliance |
| `egm-style` | `style.md` | Clarity and brevity |
| `egm-implementation-readiness` | `implementation_readiness.md` | Plan is actionable without decisions |
| `egm-codebase-details` | `codebase_details.md` | Doc matches code |
| `egm-synthesizer` | `synthesizer.md` | Merges and dedupes findings |
| `egm-bootstrap-orchestrator` | `bootstrap_orchestrator.md` | Bootstrap worker persona |

Document standards (`references/standards/`): `zoom-out-standard.md`,
`feature-spec-standard.md`, `implementation-readiness-standard.md`,
`re-implementation-standard.md`, `heilmeier-standard.md`, plus
`execution_instructions.md` (design-only scope, consensus gates, one question
at a time). Templates (`templates/`): `design_doc.md`,
`bootstrap-readme-templates.md`, `worker-return.schema.json`.

## The model

- **Elephant (you):** the main agent with the full conversation and project
  context. Explores the problem, drafts, and resolves findings.
- **Goldfish:** a fresh `delegate_task` child that sees only the document, the
  files it references, its persona, and a standard. Its confusion is the signal.
- **User:** owns the goals, makes the architectural trade-offs, approves.

## Spawning goldfish in Hermes

Upstream says "spawn a subagent"; in Hermes that is always `delegate_task`.

1. **One goldfish = one task.** `goal` names the lens and target, e.g.
   `"egm-design review of /abs/path/design_draft.md lines 1-140"`. `context`
   carries ONLY:
   - absolute path(s) of the document under review and the line range;
   - the referenced source paths (`GoldfishContext`);
   - the full persona text, read with
     `skill_view(name='elephant-goldfish', file_path='references/reviewers/<file>.md')`
     and pasted verbatim;
   - the applicable standard text from `references/standards/` (when the
     workflow assigns one);
   - `Mode: REVIEW` or `Mode: BRAINSTORM` (plus the anchored hypothesis for
     brainstorm);
   - the return format from the persona's Output Format section;
   - the line `Do not modify any files; report findings only.`
2. **Never** put conversation history, your reasoning, earlier drafts, or the
   user's discussion into a goldfish `context`. That is the entire method.
3. **Parallel reviewers** go in ONE `delegate_task` call with a `tasks` array,
   one entry per reviewer.
4. **Synthesizer:** a separate `delegate_task` after the reviewers return. Its
   context: the synthesizer persona, the document path, the line range, the
   raw reviewer outputs, each reviewer's status (`completed` / `not run`), and
   the report path to write (for example `goldfish_review_report.md`). Tell it
   the reviewers already ran, so it only merges, dedupes, and writes.
5. **Direct dispatch is the default.** Hermes delegation is flat unless
   `delegation.max_spawn_depth` is 2 or more, so a synthesizer child cannot
   spawn reviewers itself. Wherever a workflow says "spawn the synthesizer
   with Reviewers: [...]", spawn the reviewers yourself (step 3), then the
   synthesizer (step 4).
6. **Failed reviewers:** a child that errors, times out, or returns nothing
   did not review. Retry once; then record it as `not run`, never as zero
   findings.
7. **Gates:** an upstream JSON gate
   `{"questions": [{"question": ..., "options": [...], "is_multi_select": false}]}`
   maps to `clarify` with one entry in `questions`, `choices` = the options
   (max 4, single-select). Ask open-ended questions (file paths, problem
   drilling) in plain chat, not as `clarify` choices.

**Honest limits.** A child starts with a fresh conversation, but it inherits
the parent's toolset and may load the workspace's context files (such as
`AGENTS.md`). Hermes has no per-call tool filter, so "read-only reviewer" is an
instruction, not an enforced permission. Do not tell the user reviewers are
sandboxed.

## Procedure

1. Pick the workflow from the user's request (Quick Reference). If unclear,
   ask one question.
2. Load its reference with `skill_view`; workflows that say so also load
   `references/standards/execution_instructions.md`.
3. Collect missing inputs one at a time, as the workflow's intake step says.
4. Follow the workflow steps in order. Respect every gate: stop and wait for
   the user where the workflow says to yield.
5. Write every brief, draft, and report to a file with `write_file` and give
   the user the path. Do not paste large drafts into chat.
6. Spawn goldfish exactly as in "Spawning goldfish in Hermes".
7. Hand off to the next workflow only when the user picks it at the handoff gate.

## Pitfalls

1. **Context leakage.** Summarizing the conversation "for helpful background"
   turns a goldfish into a second elephant. Pass paths and persona text only.
2. **Reviewing in the same context.** If `delegate_task` fails or is missing,
   stop. A self-review is not a goldfish review.
3. **Counting a failed child as clean.** Status `not run` makes the round not clean.
4. **`clarify` marks the first choice as recommended.** Upstream forbids
   recommended defaults on gates, and the draft workflow forbids ranking
   architecture options. Put the upstream Option 0 first and say in chat that
   the marking is a UI default. For owner-decides architecture options, present
   them in plain chat so none is marked.
5. **More than one question per turn.** Execution instructions allow one.
   `clarify` gets exactly one entry in `questions`.
6. **Contesting without evidence.** In resolve, a CONTEST must cite something
   the reviewer could read; otherwise ESCALATE or PROPOSE FIX.
7. **Stopping after one clean round.** Review-loop converges only after two
   consecutive clean rounds (cap 5 rounds).
8. **Writing production code.** These workflows produce documents. Scratch
   code goes under `$TMPDIR` (Hermes scratch), never in the project tree.
9. **Headless runs.** Without `clarify`, record pending gates in the document
   and stop instead of choosing for the user.

## Verification

- Each produced file exists at the path you reported (`read_file` it).
- Every goldfish task's `context` holds only paths, persona, standard, mode,
  return format, and the no-modify line (re-read your `delegate_task` call).
- Review reports list a status for every requested reviewer.
- After resolve, the target document is still valid Markdown and the report
  starts with `Status: resolved`.
- Bootstrap: every created or modified contract file was re-read and every
  relative link resolves.

## Attribution

Adapted from Google's Elephant-Goldfish Model,
https://github.com/google/elephant-goldfish-model at commit
`403d8881d62022490637d740657b33f01e321ea8`, licensed Apache-2.0 (see
`LICENSE`). Original essay by Dave Rensin: "Elephants, Goldfish, and the New
Golden Age of Software Engineering". Workflow bodies, reviewer personas,
standards, and templates are kept 1:1 except for runtime references, which
are mapped to Hermes tools.
