---
title: "Moemail — Use when an agent needs a disposable inbox to test sign-up, password-reset or OTP email flows"
sidebar_label: "Moemail"
description: "Use when an agent needs a disposable inbox to test sign-up, password-reset or OTP email flows"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Moemail

Use when an agent needs a disposable inbox to test sign-up, password-reset or OTP email flows.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/email/moemail` |
| Path | `optional-skills/email/moemail` |
| Version | `1.0.0` |
| Author | BeilunYang (beilunyang), MoeMail |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Email`, `Testing`, `Verification`, `MCP`, `MoeMail` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# MoeMail Skill

MoeMail gives the agent short-lived email addresses it can read. Use it when a
task needs a working inbox: testing the user's own sign-up or password-reset
flow, receiving a one-time code, or keeping the user's real address out of a
one-off form.

## When to Use

- The agent must receive an email (verification link, OTP, notification) to
  finish a task.
- The user wants an end-to-end check that their app's emails arrive and look
  right.

Do not use it for accounts the user will need to recover later; use their real
address for those.

## Setup

Connect MoeMail's remote MCP server in `~/.hermes/config.yaml`. Hermes runs the
OAuth flow on first connect: sign in to MoeMail and choose which tools the agent
may use (read, create, send, delete).

```yaml
mcp_servers:
  moemail:
    url: "https://moemail.app/mcp"
    auth: oauth
```

Headless (cron, gateway): use an API key instead. Put `MOEMAIL_API_KEY` in
`~/.hermes/.env`, then:

```yaml
mcp_servers:
  moemail:
    url: "https://moemail.app/mcp"
    headers:
      X-API-Key: "${MOEMAIL_API_KEY}"
```

For a self-hosted MoeMail, replace the host with your deployment. Without MCP,
see `references/api.md` for the same steps over the REST API.

## Workflow

1. `create_email` with `expiry: "1h"`. It returns the mailbox `id` and `address`.
2. Use the address in the form and submit it.
3. `wait_for_email` with the mailbox id and `since: "0"`. An email that already
   arrived while the form was being submitted is returned at once; otherwise it
   waits up to 90 seconds. On `status: "timeout"`, call it again with `since`
   set to the returned `cursor`; give up after about 5 minutes.
4. `read_message` with the mailbox id and the returned `messageId`, then extract
   the code or link.
5. `delete_email` when the task is done.

For a second email in the same mailbox (a resent code, say), pass the previous
result's `cursor` as `since` so the first email is not returned again.

## Rules

- Only use it for accounts and flows the user owns or is allowed to test.
- `send_email` needs send permission on the user's account; use it only when
  asked.
- Mailboxes expire on their own (`1h`, `24h`, `3d`); `permanent` needs a paid
  plan.
