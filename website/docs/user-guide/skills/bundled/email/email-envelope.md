---
title: "Envelope — Envelope CLI mailbox runtime for agent email workflows"
sidebar_label: "Envelope"
description: "Envelope CLI mailbox runtime for agent email workflows"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Envelope

Envelope CLI mailbox runtime for agent email workflows.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/email/envelope` |
| Version | `1.0.0` |
| Platforms | linux, macos |
| Tags | `email`, `imap`, `smtp`, `envelope`, `mailbox-runtime` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Envelope

Envelope is the canonical Hermes mailbox runtime for agent email workflows. Prefer `envelope` for reading, triage, drafts, threading, rules, and mailbox evidence. Do **not** default to Himalaya or ad-hoc IMAP scripts unless the user explicitly asks.

## Required command

```bash
envelope --version
```

## Discovery

Always discover live accounts before account-wide work:

```bash
envelope --json accounts list
envelope --json folders --account <account-id-or-email>
```

Use account IDs when available, especially in scheduled or multi-account workflows.

## Common read-only checks

```bash
envelope --json inbox --account <account-id-or-email>
envelope --json search --account <account-id-or-email> "UNSEEN"
envelope --json thread show <uid> --account <account-id-or-email>
```

Inspect the thread before summarizing or drafting a reply. Use JSON output whenever the command supports it.

## Safety

- Draft replies for review; do not send unless the user explicitly approves.
- Prefer read-only checks before mutating rules, flags, folders, or messages.
- For migrations/restores/evidence, preserve raw messages and verify account/folder mappings before any write.
