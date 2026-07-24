---
title: "Simplify Code — Parallel 4-agent cleanup of recent code changes"
sidebar_label: "Simplify Code"
description: "Parallel 4-agent cleanup of recent code changes"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Simplify Code

Parallel 4-agent cleanup of recent code changes.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/simplify-code` |
| Version | `1.1.0` |
| Author | Hermes Agent (inspired by Claude Code /simplify) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `code-review`, `cleanup`, `refactor`, `delegation`, `subagent`, `parallel`, `simplify` |
| Related skills | [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review), [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development), [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Simplify Code — Parallel Review & Cleanup

Review your recent code changes with four focused reviewers running in
parallel, aggregate their findings, and apply the fixes worth applying.

**This is a cleanup pass, not a bug hunt.** You are improving the quality of
code that already works — removing duplication, flattening needless
complexity, cutting waste, and deepening band-aid fixes. Do not go hunting
for correctness bugs here; that's what `requesting-code-review` is for.

**Core principle:** Four narrow reviewers beat one broad reviewer. Each one
deeply searches the codebase for a single class of problem — reuse, quality,
efficiency, altitude — without diluting its attention across all four. They
run concurrently, so you pay the latency of one review, not four.

## When to Use

Trigger this skill when the user says any of:

- "simplify" / "simplify my changes" / "simplify these changes"
- "review my code" / "review my recent changes" / "clean up my changes"
- "/simplify" (if they're carrying the Claude Code habit over)

Optional modifiers the user may add — honor them:

- **Focus:** "simplify focus on efficiency" → run only the efficiency reviewer
  (or weight the aggregation toward it). Recognized focuses: `reuse`,
  `quality` (also accepts `simplification`), `efficiency`, `altitude`.
- **Dry run:** "simplify but don't change anything" / "just report" → run the
  four reviewers, present findings, apply NOTHING. Ask before applying.
- **Scope:** "simplify the last commit" / "simplify staged" / "simplify
  src/foo.py" → narrow the diff source accordingly (see Phase 1).

Do NOT auto-run this after every edit or tack it onto the end of unrelated
tasks. It costs four subagents' worth of tokens — invoke it only when the
user explicitly asks.

## The Process

### Phase 1 — Identify the changes

Capture the diff to review. Pick the source by what the user asked for, in
this default order:

```bash
# 1. Default: uncommitted working-tree changes (tracked files)
git diff

# 2. If that's empty, include staged changes
git diff HEAD

# 3. Scoped variants the user may request:
git diff --staged                 # "staged changes"
git diff HEAD~1                    # "the last commit"
git diff main...HEAD              # "this branch" / "my PR"
git diff -- src/foo.py            # specific file(s)
```

If `git diff` and `git diff HEAD` are both empty and there's no git repo or no
changes, fall back to the files the user explicitly named or that were
recently created/edited in this session. If you genuinely can't find any
changed code, say so and stop — there's nothing to simplify.

Capture the full diff text. Note its size: if it's very large (say >2000
changed lines), warn the user that four subagents each carrying the full diff
will be token-heavy, and offer to scope it down (per-directory, per-commit)
before proceeding.

### Phase 2 — Launch four reviewers in parallel

Use `delegate_task` **batch mode** — pass all four tasks in one `tasks`
array so they run concurrently. Four is the right fan-out for this pattern;
it's within the `delegation.max_concurrent_children` budget on any default
install.

**No delegation available?** If you can't call `delegate_task` in this
context (you're a leaf subagent, delegation is disabled, or the budget is
exhausted), do NOT skip the review or drop angles. Work through all four
reviewer angles yourself, sequentially, in this context — same search
standards, same finding format. Then say clearly in your final summary that
this was a single-pass inline review, not the parallel fan-out, so the user
knows what actually ran.

Give **every** reviewer the **complete diff** (not fragments — cross-file
issues hide in the gaps) plus the absolute repo path so they can search the
wider codebase. Each reviewer gets `terminal`, `file`, and `search`
toolsets (so they can `git`, `read_file`, and `search_files`/grep).

Tell each reviewer to:
- Search the existing codebase for evidence (don't reason from the diff alone).
- **Apply Chesterton's Fence:** before flagging anything for removal, run
  `git blame` on the line to understand why it exists. If you can't determine
  the original purpose, mark it `confidence: low` — don't guess.
- Report findings as structured output with the concrete cost, confidence,
  and risk:
  ```
  file:line → problem → cost (what's duplicated/wasted/harder to maintain) → suggested fix | confidence: high/medium/low | risk: SAFE/CAREFUL/RISKY
  ```
  The **cost** field forces each finding to justify itself — a finding that
  can't articulate what the problem actually costs is probably a nit.
  - **SAFE** = proven not to affect behavior (unused imports, commented-out
    code, pass-through wrappers). Auto-apply these.
  - **CAREFUL** = improves without changing semantics (rename local variable,
    flatten nested ternary, extract helper). Apply with test verification.
  - **RISKY** = may change behavior or breaks public contracts (N+1
    restructuring, public API rename, memory lifecycle change). Flag for
    human review — do NOT auto-apply.
- Skip nits and style-only churn. Only flag things that materially improve
  the code.

Pass these four goals (drop any the user's focus excludes):

**Reviewer 1 — Code Reuse**
> Review this diff for code that duplicates functionality already in the
> codebase. Search utility modules, shared helpers, and adjacent files
> (use search_files / grep) for existing functions, constants, or patterns
> the new code could call instead of reimplementing. Flag: new functions
> that duplicate existing ones; hand-rolled logic that an existing utility
> already does (manual string/path manipulation, custom env checks, ad-hoc
> type guards, re-implemented parsing). For each, name the existing thing to
> use and where it lives.

**Reviewer 2 — Code Quality**
> Review this diff for quality problems. Look for: redundant state (values
> that duplicate or could be derived from existing state; caches that don't
> need to exist); parameter sprawl (new params bolted on where the function
> should have been restructured); copy-paste-with-variation (near-duplicate
> blocks that should share an abstraction); leaky abstractions (exposing
> internals, breaking an existing encapsulation boundary); stringly-typed
> code (raw strings where a constant/enum/registry already exists — check the
> canonical registries before flagging); deeply nested conditionals (ternary
> chains, 3+-level if/else pyramids — flatten with guard clauses, early
> returns, or a lookup table); AI-generated slop patterns (extra
> comments restating obvious code like `// increment counter` above `count++`;
> unnecessary defensive null-checks on already-validated inputs; `as any`
> casts that bypass the type system; patterns inconsistent with the rest of
> the file). For each, give the concrete refactor.

**Reviewer 3 — Efficiency**
> Review this diff for efficiency problems. Look for: unnecessary work
> (redundant computation, repeated file reads, duplicate API calls, N+1
> access patterns); missed concurrency (independent ops run sequentially);
> hot-path bloat (heavy/blocking work on startup or per-request paths);
> TOCTOU anti-patterns (existence pre-checks before an op instead of doing
> the op and handling the error); memory issues (unbounded growth, missing
> cleanup, listener/handle leaks; long-lived callbacks or objects built as
> closures that capture the whole enclosing scope — everything captured
> stays alive as long as the object does, so prefer a small class or
> explicit-fields struct that copies only what it needs); overly broad reads
> (loading whole files when a slice would do); silent failures (empty catch
> blocks, ignored error returns, `except: pass`, `.catch(() => {})` with no
> handling, error propagation gaps — these hide bugs and should at minimum
> log before swallowing). For each, give the concrete fix and why it's
> faster or safer.

**Reviewer 4 — Altitude**
> Review this diff for changes implemented at the wrong depth — band-aids
> layered on top of shared infrastructure instead of fixes to the
> infrastructure itself. Signs of a too-shallow fix: a special case added to
> a generic code path to handle one caller (an `if (caller == X)` branch, a
> type check, a magic-value escape hatch); a symptom patched at the call
> site while sibling call sites keep the same flaw; a workaround stacked on
> an earlier workaround; a wrapper added to avoid touching the thing that
> actually needs changing; configuration or flags introduced to route around
> a broken default instead of fixing the default. For each, identify the
> underlying mechanism the change is dodging and describe the deeper fix —
> generalize the shared path, fix the root default, or fix the whole bug
> class — and honestly note when the deeper fix is large enough that it
> should be its own task rather than part of this cleanup. Read the
> surrounding code and `git blame` first: what looks like a band-aid is
> sometimes a deliberate boundary (compat shims, staged migrations,
> vendored-code isolation). Don't flag those.

### Phase 3 — Aggregate and apply

Wait for all four to return (batch mode returns them together).

1. **Merge** the findings into one list, deduping where reviewers overlap —
   when two findings target the same line or the same underlying mechanism,
   collapse them into one.
2. **Discard false positives** — you have the most context; you don't have to
   argue with a reviewer, just drop weak or wrong suggestions silently.
3. **Resolve conflicts.** Reviewers can disagree (Reviewer 1: "use existing
   util X"; Reviewer 3: "X is slow, inline it"). Default resolution order:
   **correctness > the user's stated focus > readability/reuse > micro-perf.**
   Don't apply a perf "fix" that hurts clarity unless the path is genuinely
   hot. When two suggestions are mutually exclusive and both defensible, pick
   the one that touches less code and note the alternative.
4. **Apply in risk-tier order:**
   - **SAFE first** (auto-apply): unused imports, commented-out code,
     pass-through wrappers, redundant type assertions. Run tests after.
   - **CAREFUL next** (apply with verification, one file at a time): rename
     locals, flatten ternaries, extract helpers, consolidate dupes. Run tests
     after each file. Revert any that break.
   - **RISKY last** (flag for review — do NOT auto-apply): N+1 restructuring,
     public API changes, concurrency fixes, error-handling changes. Present
     each with risk description and test coverage status. Altitude findings
     usually land here — deepening a fix means touching shared
     infrastructure, so present the deeper fix and let the user decide
     whether to do it now or as a follow-up.
   If the user opted for a dry run, present all three tiers and apply nothing.
5. **Verify** you didn't break anything: run the project's targeted tests for
   the touched files (not the full suite), and re-run any linter/type check the
   repo uses. If a fix breaks a test, revert that one fix and report it.
6. **Summarize** what you changed: a short list of applied fixes grouped by
   reviewer category and risk tier, plus any findings you deliberately skipped
   and why. If you ran inline (no delegation), say so here.

## Pitfalls

- **Don't fan out wider than 4.** More reviewers means more cost and more
  conflicting suggestions to reconcile, not better coverage. The four
  categories cover the space.
- **Give the WHOLE diff to each reviewer.** Splitting the diff across reviewers
  defeats the design — cross-file duplication and N+1s only show up with the
  full picture.
- **Reviewers search, they don't guess.** A reuse finding with no pointer to
  the existing utility ("there's probably a helper for this") is noise. Require
  `file:line` evidence; drop findings that lack it.
- **Apply ≠ rewrite.** This is cleanup of the user's recent changes, not a
  license to refactor the whole module. Keep edits scoped to what the diff
  touched plus the minimal surrounding change a fix requires. Altitude
  findings are the exception that proves the rule: when the right fix is
  deeper than the diff, FLAG it — don't unilaterally rebuild the shared
  mechanism inside a cleanup pass.
- **Don't drift into bug-hunting.** If a reviewer surfaces a genuine
  correctness bug, report it prominently — but as a separate "found a bug"
  note, not folded into cleanup fixes. Correctness review is a different
  pass with different verification standards.
- **Respect project conventions.** If the repo has AGENTS.md / CLAUDE.md /
  HERMES.md or a linter config, fold those rules into the reviewer prompts so
  suggestions match house style instead of fighting it.
- **Large diffs blow context.** If the diff is huge, scope it down before
  delegating — four subagents each carrying a 5000-line diff is expensive and
  may truncate.
- **Over-trusting dead code tools.** `knip`, `ts-prune`, and `depcheck` flag
  exports that ARE used dynamically (string-based imports, reflection). Always
  grep for the symbol name before removing — a clean tool report is not proof.
- **Renaming without checking public contracts.** Export names, API route
  paths, DB column names, and config keys are contracts — even if the name is
  bad, renaming breaks consumers. Tag public-contract changes as RISKY; never
  auto-rename them.
- **Removing "unnecessary" error handling.** An empty catch block or ignored
  error might be intentional — the error is expected and benign in that
  context. Flag it, don't remove it; let the human decide.
- **Not every special case is a band-aid.** Compat shims, staged migrations,
  and isolation layers around vendored code look like altitude violations but
  are deliberate design. Check `git blame` and surrounding comments before
  flagging; when the intent is unclear, mark `confidence: low`.

## Related

If your install has the `subagent-driven-development` skill (optional), it
covers the complementary case: parallel review *during* implementation, per
task. This skill is the standalone *after-the-fact* cleanup pass. Use
`requesting-code-review` for the pre-commit security/quality gate — that's
the bug hunt; this is the cleanup.
