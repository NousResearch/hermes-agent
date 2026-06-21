---
title: "Cli Coding Agents"
sidebar_label: "Cli Coding Agents"
description: "Delegate coding to autonomous CLI agents such as Claude Code, Codex, and OpenCode; choose one-shot vs interactive sessions and verify their work"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Cli Coding Agents

Delegate coding to autonomous CLI agents such as Claude Code, Codex, and OpenCode; choose one-shot vs interactive sessions and verify their work.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/autonomous-ai-agents/cli-coding-agents` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# CLI Coding Agents

Use this umbrella when delegating coding, reviews, repo exploration, or implementation tasks to external autonomous coding CLIs.

## Agent choice

- **Claude Code**: strong for repository-wide implementation and multi-file reasoning when available.
- **Codex CLI**: strong for one-shot code edits, implementation slices, and review-style coding tasks.
- **OpenCode**: useful for interactive or long-running repo work when configured locally.

## Universal orchestration rules

1. Inspect the repo and define a narrow task before launching an agent.
2. Pass exact constraints: files to modify/avoid, tests to run, expected output, and whether it may commit.
3. Prefer non-interactive print/one-shot mode for bounded tasks; use PTY/background sessions for iterative or long-running work.
4. Require verifiable output: changed file list, command outputs, test results, or a diff.
5. Independently inspect the diff and run verification yourself before telling the user it worked.

## One-shot delegation pattern

- Prepare a concise prompt with goal, context, constraints, and verification commands.
- Run the CLI in the repository root.
- Capture stdout/stderr and exit status.
- Review `git diff` and run the promised tests.

## Interactive/background pattern

- Start the CLI in a tracked background/PTY session.
- Poll or wait for completion; do not let long-running work disappear silently.
- Send follow-up instructions only after reading its current output.

## Safety

- Do not give agents secrets unless the user explicitly authorizes that exact use.
- Do not trust self-reported success; verify artifacts and side effects directly.
- Keep delegation scoped. If an agent broadens the task, stop and re-scope.
## Support files

- `references/absorbed-skills.md` — list of original skill packages consolidated into this umbrella and where to recover full archived content.
