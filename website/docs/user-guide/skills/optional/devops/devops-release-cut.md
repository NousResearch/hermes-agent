---
title: "Release Cut — Semver bump, cut sequence, canary stages, rollback command"
sidebar_label: "Release Cut"
description: "Semver bump, cut sequence, canary stages, rollback command"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Release Cut

Semver bump, cut sequence, canary stages, rollback command.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/devops/release-cut` |
| Path | `optional-skills/devops/release-cut` |
| Version | `0.1.0` |
| Author | hope (rlaope), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `release`, `semver`, `changelog`, `tagging`, `canary`, `rollout`, `rollback`, `deployment` |
| Related skills | [`github`](../../bundled/software-development/software-development-github.md), [`iac-change`](../../optional/devops/devops-iac-change.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Release Cut Skill

Decide what a release contains and its version, cut it so the tag points at a tested tree, roll
it out in stages promoted by signals, and write the rollback trigger and literal rollback command
down before the tag is pushed. It does not cover app-store mobile releases or commanding an
active incident.

## When to Use

- "Cut a release", "what version is this?", "tag and publish vX.Y.Z".
- Planning a canary/staged rollout and its promotion and rollback criteria.
- Rolling back a bad release, or yanking a published package.

Don't use for: changing declared infrastructure (`iac-change`), drafting PR text (`github`).

## Prerequisites

- The project's own release mechanism, read first with `read_file`/`search_files`: release
  workflow (`.github/workflows/*release*`), `CHANGELOG`, every version surface (`pyproject.toml`,
  `package.json`, `Cargo.toml`, `version.py`, Helm `Chart.yaml`, docs).
- Who is allowed to approve publish and run rollback.

## How to Run

Read-only steps (listing changes, checking CI, comparing version surfaces) go through `terminal`
with `git` and `gh`. Tagging, publishing, deploying, approving and rolling back are user-approved
actions: present the exact command, run it only on approval, then observe its effect.

## Quick Reference

- Changes since last tag: `git log --oneline $(git describe --tags --abbrev=0)..HEAD`
- Release branch CI: `gh run list --branch <release-branch> --limit 5`
- Version bump rule: any breaking change → major; else any feature → minor; else patch
  (pre-1.0 projects state their own rule).
- Rollback by mechanism:

| Mechanism | Command shape | Cannot undo |
| --- | --- | --- |
| Redeploy previous artifact | `kubectl rollout undo deployment/<name>`, or redeploy the previous image tag | data the new version already wrote |
| Traffic shift | set the canary weight to 0 | nothing, while the old version still runs |
| Feature flag | turn the flag off | side effects already emitted (emails, webhooks, charges) |
| Revert and re-release | revert the merge, cut a patch | a published package: yank/deprecate it, never delete and reuse the version |

## Procedure

1. **Scope and version.** List changes since the last tag and sort them into breaking, features,
   fixes. Derive the version from those classes, not from how big it feels. Hold back what isn't
   ready behind a flag or out of the branch, and say which. Done when every change is classified
   and the version follows from the rule.
2. **Write the rollback before cutting.** Pushing a tag or publishing can itself deploy (a
   release-triggered deploy workflow), so this comes before either. Trigger = signal + threshold
   + window (e.g. "5xx rate above 1% for 5 minutes on the canary cohort"). Command = the literal line from the Quick
   Reference table, plus who may run it. A release crossing a schema migration needs that
   migration to be backward compatible (expand before contract) or the rollback is a restore; say
   which. Done when trigger, threshold, window, command and owner are all written down.
3. **Cut in this order** (each step prevents a named failure):

   | # | Step | Prevents |
   | --- | --- | --- |
   | 1 | Land the prep change: changelog entry, any version surface that gained a check | a bump that misses a surface and turns the release branch red |
   | 2 | Wait for the release branch's own CI on its last merge, then freeze it until the tag is pushed | a merge in the window breaking the atomic push of bump + tag |
   | 3 | Run the cut (release workflow or command): bump every version surface, run the suite on the bumped tree, push the bump commit and tag atomically (`git push --atomic origin <branch> <tag>`) | a tag pointing at a tree nobody tested |
   | 4 | Pass the approval gate (protected environment or manual approval) | an unreviewed publish |
   | 5 | Publish, then check every channel moved: package registry, container registry, installers, docs site | a release visible on one channel and missing on another |
   | 6 | Curate the notes on top of the generated change list | a list of titles nobody can act on |

   After the cut, compare every version surface against the tag. Done when all match.
4. **Staged rollout, promoted by signals:**

   | Stage | Traffic | Bake | Promote when |
   | --- | --- | --- | --- |
   | Canary | 1–5% or one instance | at least one traffic peak | error rate, latency and saturation stay inside the stated range vs baseline |
   | Partial | 25–50% | one more peak | same signals hold under more load |
   | Full | 100% | until the rollback window closes | previous artifact may be retired |

   A clock alone is not a promotion criterion. Done when each promotion cites the signal read.
5. **On trigger, roll back immediately** with the pre-written command, then observe the signal
   recover. Investigate after.

## Pitfalls

- A tag created before the suite ran on the bumped tree ships an untested tree.
- Deleting a published package version and re-publishing the same number breaks every cache and
  lockfile that saw it; yank/deprecate and bump instead.
- A feature-flag rollback leaves already-emitted side effects; name them.
- Rolling back code across a contract-phase migration fails at runtime; check migration order.
- Generated notes are a starting point, not release notes.

## Verification

- Every version surface equals the tag; every channel shows the new version.
- The rollback trigger, threshold, window, command and owner were recorded before the tag was pushed.
- Each promotion is backed by an observed signal reading, not elapsed time.

Ported from rlaope/oh-my-hermes (MIT), `skills/omh-release-cut`.
