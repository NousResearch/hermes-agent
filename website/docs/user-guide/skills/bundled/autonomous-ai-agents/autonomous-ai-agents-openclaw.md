---
title: "OpenClaw: Configure and recover OpenClaw agent installations"
sidebar_label: "OpenClaw"
description: "Configure and recover OpenClaw agent installations"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# OpenClaw

Configure and recover OpenClaw agent installations.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/autonomous-ai-agents/openclaw` |
| Version | `1.0.0` |
| Author | Joey Cera + Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `OpenClaw`, `configuration`, `agents`, `skills`, `gateway` |
| Related skills | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# OpenClaw Skill

Use this skill when Hermes needs to inspect, configure, restart, or recover an OpenClaw installation. Prefer OpenClaw's CLI and official documentation over direct file edits, and keep Hermes-owned state under `~/.hermes` separate from OpenClaw-owned state under `~/.openclaw`.

This skill is operational only. For full migration, defer to the official Hermes OpenClaw migration docs; do not improvise migration steps from this skill.

## When to Use

- The user asks Hermes to configure OpenClaw agents, skills, models, channels, gateway, or workspace paths.
- OpenClaw is stopped, sleeping, misconfigured, or needs a restart after config updates.
- A Hermes workflow needs to check whether OpenClaw can run, route messages, or load a skill.
- The user asks how Hermes and OpenClaw should coexist without overwriting each other's state.
- Do not use this for OpenClaw source-code changes unless the user explicitly asks to work in the OpenClaw repository.

## Prerequisites

- Use official OpenClaw docs only: start from `https://docs.openclaw.ai/llms.txt`, then fetch the specific docs page you need.
- Confirm the installed CLI before making changes:

```bash
openclaw --version
openclaw config file
```

- Default paths; verify the active install with `openclaw config file`:
  - Config: `~/.openclaw/openclaw.json`
  - Workspace: `~/.openclaw/workspace`
  - Workspace skills: `~/.openclaw/workspace/skills/<skill>/SKILL.md`
  - Shared skills: `~/.openclaw/skills/<skill>/SKILL.md`
- If `openclaw` is missing, use the official install or update docs. Do not invent install commands.

## How to Run

Start read-only. Capture the exact command output before proposing any change.

```bash
openclaw status
openclaw doctor
openclaw gateway status
openclaw config validate
openclaw agents list
openclaw skills list
```

Use `openclaw config get` for targeted reads:

```bash
openclaw config get agents.defaults.workspace
openclaw config get agents.defaults.skills
openclaw config get agents.list --json
```

For config mutations, prefer CLI writes with dry-run preview:

```bash
openclaw config set agents.defaults.heartbeat.every "2h" --dry-run
openclaw config patch --file ./openclaw.patch.json5 --dry-run
openclaw config validate --json
```

If `openclaw config --help` does not show `patch` on an older install, use the documented batch setter instead:

```bash
openclaw config set --batch-file ./openclaw-config-set.batch.json --dry-run
```

Apply only after the dry run matches the requested change:

```bash
openclaw config patch --file ./openclaw.patch.json5
openclaw gateway restart
openclaw status
```

## Quick Reference

| Task | Command |
| --- | --- |
| Show active config path | `openclaw config file` |
| Validate config | `openclaw config validate --json` |
| Run health checks | `openclaw doctor` |
| Check gateway | `openclaw gateway status` |
| Restart gateway | `openclaw gateway restart` |
| Inspect agents | `openclaw agents list` |
| Inspect skills | `openclaw skills list` |
| Check skill readiness | `openclaw skills check` |
| Install a skill | `openclaw skills install <slug-or-path>` |
| Migrate from OpenClaw | See the official Hermes OpenClaw migration docs |

## Procedure

1. Identify the OpenClaw home, config file, workspace, and gateway status.
2. Read the relevant official docs page when the requested config surface is unfamiliar or version-sensitive.
3. Inspect current values with `openclaw config get`, `openclaw agents list`, and `openclaw skills list`.
4. For agent and skill visibility, distinguish skill location from agent allowlists:
   - `agents.defaults.skills` sets the inherited default allowlist.
   - `agents.list[].skills` replaces the default for that agent.
   - Empty `skills: []` means no skills for that agent.
5. For config edits, create the smallest `openclaw.patch.json5`, batch `config set` file, or single `config set` command that changes only the requested keys.
6. Run a dry run, validate config, then apply.
7. Restart only the component that needs it. Prefer `openclaw gateway restart` after gateway, channel, agent, or skill-load changes.
8. Verify with `openclaw status`, `openclaw doctor`, and the narrow command that proves the requested state.

## Pitfalls

- Do not edit `~/.openclaw/openclaw.json` by hand when `openclaw config patch` or `openclaw config set` can make the same change with validation.
- Do not merge `agents.defaults.skills` with `agents.list[].skills`; a non-empty per-agent list is final.
- Do not treat a same-named skill in two locations as two active skills. OpenClaw loads by precedence, and the highest-precedence copy wins.
- Do not copy Hermes `config.yaml` or `.env` directly into OpenClaw. Use migration commands when moving state between systems.
- Do not print secret values. Use OpenClaw SecretRef config commands or local environment files.
- Do not add unverified command flags. If a command is not in official docs or `openclaw --help`, point to the docs instead of inventing syntax.

## Verification

After changes, report the exact commands used and the resulting state:

```bash
openclaw config validate --json
openclaw skills check
openclaw doctor
openclaw status
```

For cross-agent coexistence, also verify Hermes separately:

```bash
hermes status --all
hermes doctor
```
