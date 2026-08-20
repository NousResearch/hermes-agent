---
title: "E2A — Operate Hermes-owned email through a hosted MCP server"
sidebar_label: "E2A"
description: "Operate Hermes-owned email through a hosted MCP server"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# E2A

Operate Hermes-owned email through a hosted MCP server.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/email/e2a` |
| Path | `optional-skills/email/e2a` |
| Version | `1.0.0` |
| Author | Josh Zhang (jiashuoz) |
| License | Apache-2.0 |
| Platforms | linux, macos, windows |
| Tags | `email`, `communication`, `e2a`, `mcp`, `oauth` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# e2a Skill

Give Hermes a verified email identity and inbox through e2a's hosted MCP
server. Operate mail owned by the agent; do not use this skill to read a
user's personal mailbox.

## When to Use

Use this skill to:

- Create or select an e2a agent inbox.
- Read incoming agent mail and attachments.
- Send a new message or reply in an existing email thread.
- Configure a custom domain owned by the user.
- Direct a backend integration toward e2a's SDKs and webhooks.

## Prerequisites

Use the hosted e2a MCP server at `https://api.e2a.dev/mcp`.

For an interactive Hermes installation, add this entry to
`~/.hermes/config.yaml`:

```yaml
mcp_servers:
  e2a:
    url: "https://api.e2a.dev/mcp"
    auth: oauth
```

Complete authorization with `hermes mcp login e2a`. Hermes stores and refreshes
the OAuth credentials locally.

For a headless deployment, supply an e2a API key through a secret manager and
reference the environment variable:

```yaml
mcp_servers:
  e2a:
    url: "https://api.e2a.dev/mcp"
    headers:
      Authorization: "Bearer ${E2A_API_KEY}"
```

Prefer an agent-scoped key for a deployed Hermes instance. Never paste an API
key into chat or commit one to a configuration file.

## How to Run

Install the optional skill and authenticate the MCP server through the
`terminal` tool:

```bash
hermes skills install official/email/e2a
hermes mcp login e2a
hermes chat
```

After setup, ask Hermes to use e2a for inbox work. Hermes discovers the e2a
tools from the connected MCP server; do not infer tool signatures from this
skill when the server's `tools/list` result is available.

## Quick Reference

| Task | e2a tool | Notes |
|---|---|---|
| Identify credential scope | `whoami` | Call first |
| List account inboxes | `list_agents` | Account scope only |
| Create an inbox | `create_agent` | Account scope only |
| List inbound mail | `list_messages` | Defaults to unread inbound mail |
| Read a message | `get_message` | Returns body, headers, and attachment metadata |
| Download an attachment | `get_attachment` | Use the message's zero-based attachment index |
| Start a new thread | `send_message` | Do not use for a reply |
| Reply in-thread | `reply_to_message` | Preserves email threading headers |
| Start custom-domain setup | `register_domain` | Returns DNS records to publish |
| Verify custom-domain DNS | `verify_domain` | Call after DNS propagation |

## Procedure

1. **Confirm the connection.** Call `whoami` before the first operation. If
   the e2a tools are absent, configure the MCP server under Prerequisites and
   authenticate it before continuing.
2. **Select the inbox.** For agent scope, use the returned `agent_email`. For
   account scope, call `list_agents` and select the inbox explicitly. If no
   inbox exists, create one on the hosted shared domain; it requires no DNS
   setup.
3. **Read mail.** Call `list_messages`, then `get_message` for the complete
   body and sender-authentication evidence. Fetch attachment bytes only when
   needed with `get_attachment`.
4. **Continue an existing conversation.** Call `reply_to_message` with the
   original `message_id`. Keep the subject stable. Use `conversation_id` only
   for application correlation; it does not preserve an email-client thread.
5. **Start a new conversation.** Call `send_message` with the selected inbox,
   recipients, subject, and a complete plain-text body. Include equivalent
   HTML only when it improves structured content.
6. **Configure a custom domain only when requested.** Call `register_domain`,
   have the user publish every returned DNS record verbatim, wait for DNS
   propagation, then call `verify_domain`. Poll `get_domain` before promising
   branded outbound sending because inbound and outbound verification are
   separate.
7. **Use SDKs for backend integrations.** When a service—not Hermes itself—must
   receive or send mail, follow e2a's SDK and webhook documentation instead of
   treating MCP as a deployed webhook receiver.

## Pitfalls

- Do not guess an inbox for an account-scoped credential. Call `list_agents`
  and select one explicitly.
- Use `reply_to_message`, not a fresh send, when responding to existing mail.
- Treat `accepted`, `scheduled`, and `pending_review` as successful durable
  acceptance. Do not resend; inspect the message later or consume webhook
  events for the terminal outcome.
- A future `send_at` is a beta scheduled send. If an outbound message is held
  for review, its schedule is preserved as `scheduled_at` and re-armed when a
  reviewer approves it.
- For attachments, pass base64 returned by an attachment/file tool; do not
  invent or hand-edit encoded bytes. Use `get_attachment` when bytes are
  needed.
- Do not call `verify_domain` immediately after registration. DNS propagation
  is asynchronous.
- Do not treat an e2a inbox as the user's personal email account. The agent is
  the mailbox owner and the visible sender.
- Do not expose API keys in chat, logs, source files, or committed Hermes
  configuration.

## Verification

Run:

```bash
hermes --toolsets skills,mcp-e2a -q "Use the e2a skill to call whoami and report the credential scope without changing anything"
```

Verification succeeds when Hermes discovers the e2a MCP tools, `whoami`
returns the credential scope, and no mutating tool is called. For account
scope, a subsequent read-only `list_agents` call should return the available
inboxes.

## References

- Setup: https://e2a.dev/setup.md
- Authentication: https://e2a.dev/auth.md
- SDK and webhooks: https://e2a.dev/sdk.md
- Exact MCP tool signatures: use the connected server's `tools/list` result
- e2a MCP endpoint: https://api.e2a.dev/mcp
