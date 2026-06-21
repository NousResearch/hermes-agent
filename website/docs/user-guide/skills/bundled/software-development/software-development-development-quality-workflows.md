---
title: "Development Quality Workflows"
sidebar_label: "Development Quality Workflows"
description: "Software development quality umbrella: spike experiments, test-driven development, pre-commit review, code simplification, and exploratory dogfood QA"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Development Quality Workflows

Software development quality umbrella: spike experiments, test-driven development, pre-commit review, code simplification, and exploratory dogfood QA.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/development-quality-workflows` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Development Quality Workflows

Use this class-level skill when improving software through disciplined experiments, tests, review, simplification, or exploratory QA.

## Choose the right subsection

- **Spike** — validate an uncertain technical idea before committing to a full implementation.
- **Test-driven development** — write or update behavior tests before changing code, then make them pass and refactor.
- **Pre-commit verification** — inspect diffs for security, correctness, quality gates, and missing tests before shipping.
- **Code simplification** — reduce unnecessary complexity while preserving behavior.
- **Dogfood QA** — explore a real web app like a user, collect evidence, categorize issues, and report reproducible bugs.

## Universal loop

1. Define the behavior, risk, or question under test.
2. Choose the smallest artifact that can answer it: throwaway spike, failing test, diff review, simplified patch, or QA report.
3. Run real commands or browser flows; do not rely on visual inspection alone.
4. Preserve evidence: test output, screenshots, logs, diffs, URLs, or reproduction steps.
5. Stop when the question is answered or the quality gate is green; avoid speculative cleanup.

## Spike experiments

- Use for unknown libraries, architecture choices, API feasibility, performance risks, or integration uncertainty.
- Keep spike code isolated and disposable unless the user explicitly asks to productize it.
- Record what was proven, what failed, and whether to proceed.

## Test-driven development

- RED: write the narrow failing test that captures desired behavior or bug reproduction.
- GREEN: implement the smallest change that passes.
- REFACTOR: simplify while tests remain green.
- For bug fixes, preserve the reproduction as a regression test.

## Pre-commit verification

- Inspect `git diff` and staged changes.
- Scan for secrets, command injection, unsafe deserialization/eval, missing validation, error handling gaps, and broken tests.
- Run the project’s relevant tests/lint/type checks and report exact outputs.

## Code simplification

- Simplify only after behavior is covered by tests or clear verification.
- Prefer deleting abstraction, merging duplicate paths, clarifying names, and reducing state.
- Do not broaden scope into unrelated refactors.

## Dogfood QA

- Plan user flows and risk areas.
- Explore with browser/devtools, capture screenshots/console/network evidence, and write reproduction steps.
- Categorize findings by severity, user impact, and confidence.

## Verification checklist

- [ ] The chosen workflow matches the task risk.
- [ ] Evidence was collected from real execution.
- [ ] Changes are scoped and reviewable.
- [ ] Follow-up recommendations distinguish blockers from optional improvements.
## Support files

- `references/absorbed-skills.md` — list of original skill packages consolidated into this umbrella and where to recover full archived content.
