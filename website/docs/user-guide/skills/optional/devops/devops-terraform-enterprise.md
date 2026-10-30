---
title: "Terraform Enterprise — Manage HCP Terraform and TFE workspaces, runs, state"
sidebar_label: "Terraform Enterprise"
description: "Manage HCP Terraform and TFE workspaces, runs, state"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Terraform Enterprise

Manage HCP Terraform and TFE workspaces, runs, state.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/devops/terraform-enterprise` |
| Path | `optional-skills/devops/terraform-enterprise` |
| Version | `1.0.0` |
| Author | dreinst (@dreinst), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Terraform`, `Terraform Enterprise`, `HCP Terraform`, `IaC`, `DevOps` |
| Related skills | [`docker-management`](../../optional/devops/devops-docker-management.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Terraform Enterprise Skill

Talk to the HCP Terraform / Terraform Enterprise API with the standard `TFE_*` environment
variables: list workspaces, read runs and plan logs, read state outputs, and queue, apply or
discard runs. It does not edit Terraform code or run `terraform` locally; for local CLI work it
explains how to hand the same token to the CLI.

## When to Use

- The user asks about TFE / HCP Terraform workspaces, runs, plans, or state outputs.
- The user wants a run queued, applied, discarded, or cancelled in a remote workspace.
- The user runs `terraform` locally against a TFE backend and needs the token wired in.

## Prerequisites

Stdlib Python only. Environment variables (the same names the `tfe` provider and `go-tfe` read):

| Variable | Required | Meaning |
|---|---|---|
| `TFE_TOKEN` | yes | User or team API token. Organization tokens cannot start or apply runs. |
| `TFE_ADDRESS` | no | TFE URL or hostname. Default `https://app.terraform.io`. `TFE_HOSTNAME` also works. |
| `TFE_ORGANIZATION` | no | Default organization for workspace commands. |

Hermes prompts for these on first load and stores them in `${HERMES_HOME:-~/.hermes}/.env`.
Because the skill declares them, they also pass through to Docker, Modal and other sandboxed
`terminal` backends.

## How to Run

Call the helper through the `terminal` tool. Output is JSON on stdout; errors go to stderr
with exit code 1.

```bash
python ~/.hermes/skills/devops/terraform-enterprise/scripts/tfe_api.py whoami
```

## Quick Reference

```bash
tfe_api.py whoami
tfe_api.py workspaces [--org ORG] [--search TEXT] [--limit N]
tfe_api.py workspace <name|ws-id> [--org ORG]
tfe_api.py runs <name|ws-id> [--org ORG] [--limit N]
tfe_api.py run <run-id>
tfe_api.py plan-log <run-id> [--tail N]
tfe_api.py outputs <name|ws-id> [--org ORG]
tfe_api.py start-run <name|ws-id> [--message TEXT] [--plan-only] [--destroy]
tfe_api.py apply <run-id> [--comment TEXT]
tfe_api.py discard <run-id> [--comment TEXT]
tfe_api.py cancel <run-id> [--comment TEXT]
```

## Procedure

1. Run `whoami` first. It confirms the token and shows which address it talks to.
2. Find the workspace with `workspaces --search <text>`, then `workspace <name>` for its
   settings and current run.
3. For a run, read `run <run-id>` (status plus add/change/destroy counts), then `plan-log` when
   the user wants the actual plan text. Summarize the log for the user instead of pasting all of
   it; `--tail` keeps the last N characters (default 20000).
4. Read state values with `outputs`. Sensitive outputs come back with `value: null`; say so
   instead of guessing.
5. Queue work with `start-run`. Use `--plan-only` when the user only wants to see a diff.
6. Before `apply`, `discard`, `cancel`, or `start-run --destroy`, show the user the run's
   counts and get an explicit yes. These change real infrastructure.

### Local `terraform` CLI

The `terraform` CLI does not read `TFE_TOKEN`. It reads `TF_TOKEN_<host>` with dots in the
hostname turned into underscores. Map it for a single command:

```bash
TF_TOKEN_app_terraform_io="$TFE_TOKEN" terraform plan
```

For a self-hosted instance at `tfe.example.com` the name is `TF_TOKEN_tfe_example_com`.

## Pitfalls

- An organization token can list workspaces but gets HTTP 403 on `start-run` and `apply`.
- `apply` only works on a run in `planned` or `cost_estimated`/`policy_checked` state, and never
  on a `--plan-only` run.
- A workspace with `auto-apply: true` applies on its own after `start-run`. Check `workspace`
  first and warn the user.
- `TFE_ADDRESS` with a path (for example `https://tfe.example.com/app`) is wrong; use the bare
  URL. The helper appends `/api/v2`.
- Workspace names are case-sensitive and scoped to an organization. Pass `--org` when the user
  works across several.

## Verification

`whoami` returns the expected username and `address`. `workspaces --limit 1` returns one entry
without an error. After `start-run`, `run <run-id>` moves from `pending` to `planning`.
