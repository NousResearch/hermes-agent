---
title: "Slack Surfaces — Operate repo-defined Slack channels and canvases through the standalone slack-surfaces CLI"
sidebar_label: "Slack Surfaces"
description: "Operate repo-defined Slack channels and canvases through the standalone slack-surfaces CLI"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Slack Surfaces

Operate repo-defined Slack channels and canvases through the standalone slack-surfaces CLI.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/slack-surfaces` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Slack`, `Canvases`, `Channels`, `Operations`, `Documentation`, `Observability` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Slack Surfaces

Use this skill when Hermes needs to manage Slack channels or canvases from
repository-defined specs without embedding the implementation into Hermes core.

This skill assumes the standalone `slack-surfaces` repository is installed
locally and available on `PATH`.

## Setup

```bash
./setup-hermes.sh
```

If you store the Slack bot token in 1Password, set this in `.env`:

```bash
SLACK_BOT_TOKEN_OP_REF=op://Second Brain/Slack_hermes_admin_app/Slack_bot_token
```

## Workflow

```bash
scripts/slack-surfaces-run.sh validate --spec slack-surfaces/*.yaml
scripts/slack-surfaces-run.sh sync --spec slack-surfaces/*.yaml
scripts/slack-surfaces-run.sh sync --spec slack-surfaces/*.yaml --apply
```

1. Validate the spec files first.
2. Run a dry-run sync to inspect planned changes.
3. Run `--apply` only when `SLACK_BOT_TOKEN` is available and live updates are intended.
4. Keep environment-specific Slack specs in the repo that owns the operational workflow, such as an infra repo.

For `hermes-agent-infra`, prefer:

```bash
cd ~/code/hermes-agent-infra
./scripts/run-slack-surfaces.sh validate --json
./scripts/run-slack-surfaces.sh sync --apply --json
```

## Notes

- Specs are versioned in Git, so Slack structure changes are reviewable.
- Hermes should call the wrapper through the terminal tool rather than reimplementing the sync logic in Hermes itself.
- The standalone repository also contains example specs and a GitHub Actions template for push-driven Slack updates.
