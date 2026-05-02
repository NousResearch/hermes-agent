---
title: "Release Notes — Use when drafting release notes or changelog summaries from git history"
sidebar_label: "Release Notes"
description: "Use when drafting release notes or changelog summaries from git history"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Release Notes

Use when drafting release notes or changelog summaries from git history. Fix the range first, gather evidence from commits/diffs/PRs, and write grounded notes without inventing impact.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/release-notes` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Tags | `release-notes`, `changelog`, `releases`, `git`, `documentation`, `summaries` |
| Related skills | [`github-pr-workflow`](/docs/user-guide/skills/bundled/github/github-github-pr-workflow), [`github-repo-management`](/docs/user-guide/skills/bundled/github/github-github-repo-management), [`humanizer`](/docs/user-guide/skills/bundled/creative/creative-humanizer) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Release Notes Drafting

Draft release notes, changelog entries, and shipped-work summaries from repo history.

This skill is for turning a concrete change range into clear, audience-appropriate notes.
It assumes the source of truth is the repo itself: tags, commits, diffs, and optionally
merged PR metadata when `gh` is available.

## When to Use

- The user asks for release notes, a changelog, or a shipped summary
- The user wants notes for a tag-to-tag range, last release to `HEAD`, or a release branch
- The user wants a user-facing summary from raw git history
- The user wants a more structured draft before creating a GitHub release

Do not use this skill for:

- Creating the release object itself without drafting notes first
- Deep code review of the changes
- Broad product marketing copy detached from the actual diff
- Vague "summarize the repo" requests with no time or ref range

## Quick Reference

| Task | Command |
|---|---|
| Latest reachable tag | `git describe --tags --abbrev=0` |
| Recent tags | `git tag --sort=-version:refname | head -10` |
| Commit list for a range | `git log --oneline <from>..<to>` |
| File/change stats | `git diff --stat <from>..<to>` |
| Full patch for one file | `git diff <from>..<to> -- path/to/file` |
| Merged PRs if `gh` is available | `gh pr list --state merged --limit 50` |
| Release body template | `templates/release-notes.md` |

If the user says "latest release notes", resolve that into exact refs such as
`v1.4.0..HEAD` or `v1.4.0..v1.5.0` before drafting anything.

## Procedure

### 1. Lock the range and audience

Start by fixing the scope.

Common scopes:

- Last tag to `HEAD`
- Tag to tag
- Branch to branch
- Specific date window
- Single PR or release branch

Common audiences:

- **User-facing release notes**: only visible behavior, important fixes, breaking changes
- **Internal changelog**: broader engineering detail is acceptable
- **Maintainer summary**: includes tooling, infra, tests, and refactors when relevant

If the range is ambiguous, state the exact range you intend to use before continuing.

### 2. Gather raw evidence from git first

Use git as the baseline even if GitHub metadata is available.

```bash
# Pick the range
git describe --tags --abbrev=0
git tag --sort=-version:refname | head -10

# Summarize the range
git log --oneline <from>..<to>
git diff --stat <from>..<to>

# If needed, inspect specific subsystems
git diff --name-only <from>..<to>
git diff <from>..<to> -- path/to/file
```

If the range is large:

- Group commits by subsystem or directory
- Use `git diff --name-only` to find hotspots
- Read the touched files for high-impact areas before writing summary bullets

### 3. Enrich with PR metadata only when useful

If `gh` is installed and the repo uses pull requests consistently, use it to improve naming
and grouping. Do not make PR metadata the only source of truth.

```bash
gh pr list --state merged --limit 50
gh pr view <number>
gh pr diff <number> --name-only
```

Use PR titles to recover human-readable intent when commit messages are noisy.
Still verify that the PR actually matches the diff.

### 4. Classify changes before drafting

Sort changes into buckets before writing prose:

- New features
- Improvements
- Bug fixes
- Breaking changes
- Documentation or developer workflow changes
- Internal-only changes

For user-facing release notes:

- Include only changes a normal user or integrator would care about
- Collapse noisy implementation commits into one grounded bullet
- Exclude pure refactors, dependency churn, test-only work, and mechanical cleanup unless
  they changed behavior, stability, or upgrade risk

For internal changelogs:

- Keep operationally relevant tooling and infrastructure work
- Still avoid line-by-line commit narration

### 5. Draft from evidence, not from commit tone

Write short bullets that describe outcomes, not just commit subjects.

Good pattern:

- "Added per-profile session isolation for gateway runs."
- "Improved startup latency by lazy-loading heavy provider modules."
- "Fixed stale session rows that remained open after failed runs."

Bad pattern:

- "Refactor session finalizer path"
- "misc cleanup"
- "big improvements to reliability" when the evidence does not show the scope

If the benefit is not explicit in the diff or PR, use neutral wording:

- "Adjusted"
- "Updated"
- "Added support for"
- "Changed"

Do not inflate claims into performance, reliability, or UX wins unless the evidence clearly
supports that interpretation.

### 6. Use a stable output structure

Default structure for user-facing notes:

1. Title / version / range
2. One-paragraph summary
3. Highlights
4. Fixes and improvements
5. Breaking changes or upgrade notes
6. Optional acknowledgements or contributors

Default structure for internal notes:

1. Range
2. High-signal bullet list by subsystem
3. Risks / migrations
4. Follow-up items still open

Use `templates/release-notes.md` when the user wants a draft file or a predictable outline.

### 7. Verify every bullet before finalizing

For each bullet, confirm:

- It maps back to one or more commits, PRs, or file diffs
- It belongs inside the chosen range
- It fits the target audience
- It is not duplicated elsewhere in the draft

If a change may be breaking or require migration, surface that explicitly even if it makes
the notes less flattering.

## Pitfalls

- **No fixed range**: "latest changes" is not a range. Resolve exact refs first.
- **Raw commit-message dump**: commit subjects are inputs, not finished release notes.
- **Overclaiming impact**: avoid "faster", "safer", "easier", or "more reliable" unless the
  evidence supports that wording.
- **Including internal churn in user notes**: test renames, formatting, and refactors usually
  do not belong in public release notes.
- **Missing breaking changes**: config migrations, renamed commands, deleted flags, and changed
  defaults must be called out clearly.
- **Mixing shipped and unshipped work**: confirm whether `HEAD` contains unreleased commits.
- **Trusting PR titles blindly**: some PR titles are cleaner than the code they merged.

## Verification

- [ ] The release range is named explicitly
- [ ] Every bullet traces back to the diff, commits, or PR metadata
- [ ] User-facing notes exclude irrelevant internal churn
- [ ] Breaking changes and upgrade notes are called out separately
- [ ] The draft does not duplicate the same change in multiple sections
- [ ] Wording is neutral where evidence is incomplete
- [ ] Version, tag, branch, or date references match the actual repo state

## One-Shot Recipes

### Draft notes since the last tag

```bash
LAST_TAG=$(git describe --tags --abbrev=0)
git log --oneline "$LAST_TAG"..HEAD
git diff --stat "$LAST_TAG"..HEAD
```

Then draft notes for `"$LAST_TAG"..HEAD` using the user-facing structure above.

### Draft notes between two tags

```bash
git log --oneline v1.4.0..v1.5.0
git diff --stat v1.4.0..v1.5.0
```

Then group changes into features, fixes, and breaking changes.

### Draft a maintainer changelog from a noisy range

```bash
git diff --name-only <from>..<to>
git diff --stat <from>..<to>
git log --oneline <from>..<to>
```

Group by subsystem first, then write one concise bullet per meaningful change cluster.
