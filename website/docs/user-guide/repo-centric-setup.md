---
sidebar_position: 12
title: "Repo-Centric Agent Setup"
description: "Share Hermes skills and project context from inside a repository so multiple agents and tools can collaborate on the same codebase."
---

# Repo-Centric Agent Setup

Most Hermes users start with a global home directory (`~/.hermes/`) and keep skills there. That works well for personal setups, but teams and multi-agent workflows often want the opposite pattern:

- project instructions live in the repo
- shared skills live in the repo
- multiple agent tools point at the same repo-owned skill directory
- skill changes can be reviewed in normal pull requests

This is the **repo-centric** pattern.

## When to use it

Use a repo-centric setup when:

- you want Hermes, Codex, Claude Code, or other agents to share the same project-specific skills
- your team wants reviewable skill changes in Git instead of private local drift
- you run multiple agents against one repository and want them to load the same instructions

Keep the default global setup when:

- your skills are mostly personal and cross-project
- the repository is public but your workflow notes are private
- you do not want skill changes tracked in Git

## Recommended layout

One common layout is:

```text
my-repo/
├── AGENTS.md
├── .claude/
│   └── skills/
│       ├── release-playbook/
│       │   └── SKILL.md
│       └── onboarding/
│           └── SKILL.md
└── src/
```

Hermes does **not** require the directory to be named `.claude/skills/`. Any folder works. What matters is that it contains standard skill folders with `SKILL.md`.

## Point Hermes at repo-owned skills

Add the repo path under `skills.external_dirs` in `~/.hermes/config.yaml`:

```yaml
skills:
  external_dirs:
    - ~/.agents/skills
    - /path/to/my-repo/.claude/skills
```

Windows paths work too:

```yaml
skills:
  external_dirs:
    - C:/Users/alice/code/my-repo/.claude/skills
```

You can also use environment variables:

```yaml
skills:
  external_dirs:
    - ${MY_REPO}/.claude/skills
```

## How Hermes resolves conflicts

Hermes scans both the local skill directory and any external dirs:

- local `~/.hermes/skills/` stays the primary read-write home
- external dirs are scanned read-only
- if the same skill name exists in both places, the local Hermes copy wins

That makes repo-owned skills safe for shared discovery while still letting you keep a personal override locally when needed.

## What belongs in the repo vs. outside it

Good repo-owned artifacts:

- `AGENTS.md`
- shared project skills
- templates, examples, and helper scripts that belong with the codebase

Keep these **outside** the repo unless you have a very deliberate reason:

- `~/.hermes/.env`
- OAuth tokens / auth stores
- private personal memory
- machine-specific credentials or secrets

In practice, the healthiest split is:

- **repo** for shared process and project knowledge
- **HERMES_HOME** for secrets, auth, and personal state

## Multi-agent coordination

Repo-centric setups work especially well when several agent tools point at the same directory:

- Hermes loads the shared skills through `skills.external_dirs`
- Codex or Claude Code can read the same repo-owned artifacts directly from the checkout
- normal PR review becomes the approval path for skill updates

That gives you one reviewable source of truth for project-specific automation knowledge instead of three separate local copies drifting apart.

## Suggested Git workflow

Treat shared skills like code:

1. keep one topic per PR
2. explain why the skill changed
3. include an example invocation when behavior changes
4. review skill edits with the same care as shell scripts or CI config

If a skill is only useful for one developer's machine, keep it in `~/.hermes/skills/` instead of checking it into the repo.

## Related docs

- [Skills System](/docs/user-guide/features/skills#external-skill-directories)
- [Context Files](/docs/user-guide/features/context-files)
