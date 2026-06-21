---
title: "Github Workflows"
sidebar_label: "Github Workflows"
description: "GitHub workflow umbrella: authentication, repository management, issues, pull requests, code review, CI triage, and codebase inspection using git/gh/API"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Github Workflows

GitHub workflow umbrella: authentication, repository management, issues, pull requests, code review, CI triage, and codebase inspection using git/gh/API.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/github/github-workflows` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# GitHub Workflows

Use this class-level skill for GitHub work across repositories, issues, pull requests, reviews, releases, and codebase reconnaissance.

## Operating rules

1. Discover repository context first: `git remote -v`, current branch, default branch, and `gh auth status` when `gh` is available.
2. Prefer `gh` for GitHub operations; fall back to GitHub REST/GraphQL via `curl` only when `gh` lacks the needed operation.
3. Read current state before mutating it. For issues/PRs, fetch the body, comments, labels, assignees, checks, and latest diff/status.
4. Make atomic changes: one branch/PR purpose, one issue update topic, one review comment per actionable finding.
5. Verify remote side effects by reading the created/updated issue, PR, release, branch, or review back.

## Authentication

- Check `gh auth status` and `git credential` setup before assuming credentials are absent.
- Use HTTPS tokens or SSH remotes according to existing repo configuration; avoid switching protocols unless necessary.
- Never print tokens. If token setup is needed, ask the user to provide or run the secret-bearing step.

## Repository management

- Clone/fork/create repos, manage remotes, tags, releases, and branch protection after reading owner/repo/default branch.
- For large repos, prefer shallow clone or sparse checkout only when the requested task does not need full history.

## Issues

- Search before creating duplicates.
- Use structured issue bodies for bug reports and feature requests: context, reproduction, expected/actual behavior, acceptance criteria.
- Preserve user labels/assignees unless asked to retag or triage.

## Pull request lifecycle

- Start from a clean branch when creating changes; include tests or verification evidence in PR bodies.
- PR body should include summary, test plan, risk/rollback, and linked issues.
- Watch CI/checks after opening or updating a PR; inspect failing logs before retrying.

## Code review

- Review diffs, not vibes: inspect changed files, tests, risk areas, and security implications.
- Comment only actionable findings with file/line context and suggested fixes where possible.
- Separate blocking issues from nits and approvals.

## Codebase inspection

- Use language/LOC/dependency inspection before estimating scope or review risk.
- Exclude vendored, generated, build, and lock files unless they are the target.
## Support files

- `references/absorbed-skills.md` — list of original skill packages consolidated into this umbrella and where to recover full archived content.
