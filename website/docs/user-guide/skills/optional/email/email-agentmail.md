---
title: "Agentmail — Give the agent its own dedicated email inbox via AgentMail"
sidebar_label: "Agentmail"
description: "Give the agent its own dedicated email inbox via AgentMail"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Agentmail

Give the agent its own dedicated email inbox via AgentMail. Send, receive, and manage email autonomously using agent-owned email addresses (e.g. hermes-agent@agentmail.to).

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/email/agentmail` |
| Path | `optional-skills/email/agentmail` |
| Version | `1.0.0` |
| Platforms | linux, macos, windows |
| Tags | `email`, `communication`, `agentmail`, `mcp` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# AgentMail — Agent-Owned Email Inboxes

## Requirements

- **AgentMail API key** (required) — sign up at https://console.agentmail.to (free tier: 3 inboxes, 3,000 emails/month; paid plans from $20/mo)
- Node.js 18+ (for the MCP server)

## When to Use
Use this skill when you need to:
- Give the agent its own dedicated email address
- Send emails autonomously on behalf of the agent
- Receive and read incoming emails
- Manage email threads and conversations
- Sign up for services or authenticate via email
- Communicate with other agents or humans via email

This is NOT for reading the user's personal email (use himalaya or Gmail for that).
AgentMail gives the agent its own identity and inbox.

## Setup

### 1. Get an API Key
- Go to https://console.agentmail.to
- Create an account and generate an API key (starts with `am_`)

### 2. Configure MCP Server
Add to `~/.hermes/config.yaml` (paste your actual key — MCP env vars are not expanded from .env):
```yaml
mcp_servers:
  agentmail:
    command: "npx"
    args: ["-y", "agentmail-mcp"]
    env:
      AGENTMAIL_API_KEY: "am_your_key_here"
```

### 3. Restart Hermes
```bash
hermes
```
All 11 AgentMail tools are now available automatically.

## Available Tools (via MCP)

| Tool | Description |
|------|-------------|
| `list_inboxes` | List all agent inboxes |
| `get_inbox` | Get details of a specific inbox |
| `create_inbox` | Create a new inbox (gets a real email address) |
| `delete_inbox` | Delete an inbox |
| `list_threads` | List email threads in an inbox |
| `get_thread` | Get a specific email thread |
| `send_message` | Send a new email |
| `reply_to_message` | Reply to an existing email |
| `forward_message` | Forward an email |
| `update_message` | Update message labels/status |
| `get_attachment` | Download an email attachment |

## Procedure

### Create an inbox and send an email
1. Create a dedicated inbox:
   - Use `create_inbox` with a username (e.g. `hermes-agent`)
   - The agent gets address: `hermes-agent@agentmail.to`
2. Send an email:
   - Use `send_message` with `inbox_id`, `to`, `subject`, `text`
3. Check for replies:
   - Use `list_threads` to see incoming conversations
   - Use `get_thread` to read a specific thread

### Check incoming email
1. Use `list_inboxes` to find your inbox ID
2. Use `list_threads` with the inbox ID to see conversations
3. Use `get_thread` to read a thread and its messages

### Reply to an email
1. Get the thread with `get_thread`
2. Use `reply_to_message` with the message ID and your reply text

## Example Workflows

**Sign up for a service:**
```
1. create_inbox (username: "signup-bot")
2. Use the inbox address to register on the service
3. list_threads to check for verification email
4. get_thread to read the verification code
```

**Agent-to-human outreach:**
```
1. create_inbox (username: "hermes-outreach")
2. send_message (to: user@example.com, subject: "Hello", text: "...")
3. list_threads to check for replies
```

## Pitfalls
- Free tier limited to 3 inboxes and 3,000 emails/month
- Emails come from `@agentmail.to` domain on free tier (custom domains on paid plans)
- Node.js (18+) is required for the MCP server (`npx -y agentmail-mcp`)
- The `mcp` Python package must be installed: `pip install mcp`
- Real-time inbound email (webhooks) requires a public server — use `list_threads` polling via cronjob instead for personal use

## Verification
After setup, test with:
```
hermes --toolsets mcp -q "Create an AgentMail inbox called test-agent and tell me its email address"
```
You should see the new inbox address returned.

## References
- AgentMail docs: https://docs.agentmail.to/
- AgentMail console: https://console.agentmail.to
- AgentMail MCP repo: https://github.com/agentmail-to/agentmail-mcp
- Pricing: https://www.agentmail.to/pricing
