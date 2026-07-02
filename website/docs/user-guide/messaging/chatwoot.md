# Chatwoot

Connect an [Chatwoot](https://www.chatwoot.com/) Agent Bot to Hermes so that a
customer messaging a connected inbox is talking to your agent. Each Chatwoot
conversation maps to its own agent session; the bot stays silent when a human
agent takes over.

## How it works

- **Inbound:** Chatwoot POSTs webhook events to the adapter's own HTTP listener.
  Only new inbound messages from the contact (`message_created`, `incoming`,
  non-private, conversation `status: pending`) are answered.
- **Outbound:** replies are posted to Chatwoot's Application API using the Agent
  Bot token in the `api_access_token` header.
- **Human handoff:** when a conversation is `open` (a human agent has taken
  over) the bot stays silent. Resolving or re-opening it as `pending` hands
  control back to the bot.

## Setup

1. In Chatwoot, create an **Agent Bot** (Super Admin → Agent Bots, or the
   platform API) and copy its **access token**.
2. Connect the bot to one or more **inboxes**.
3. Configure Hermes (env vars, or `hermes gateway setup` → Chatwoot):

   | Variable | Required | Purpose |
   |---|---|---|
   | `CHATWOOT_BASE_URL` | ✅ | Instance URL, e.g. `https://app.chatwoot.com` (trailing slash stripped). |
   | `CHATWOOT_TOKEN` | ✅ | Agent Bot token → customer-visible replies. |
   | `CHATWOOT_AGENT_TOKEN` | optional | User/agent token for **private notes** and **typing indicator** (if the bot token cannot post `private:true` or toggle typing). |
   | `CHATWOOT_ACCOUNT_ID` | optional | Default account id (otherwise derived per payload). |
   | `CHATWOOT_WEBHOOK_SECRET` | recommended when public | Shared secret validated as `?token=` on the webhook URL. |
   | `CHATWOOT_HOST` / `CHATWOOT_PORT` / `CHATWOOT_WEBHOOK_PATH` | optional | Listener bind (defaults `0.0.0.0` / `8647` / `/chatwoot/webhook`). |
   | `CHATWOOT_ALLOWED_USERS` / `CHATWOOT_ALLOW_ALL_USERS` | optional | Authorization by contact id. |
   | `CHATWOOT_PRIVATE_NOTE_TRACE` | optional (default off) | Post the agent's reasoning + tool/skill activity as private notes. |
   | `CHATWOOT_HOME_CHANNEL` | optional | `account:conversation` target for cron / proactive delivery. |

4. Set the Agent Bot's **Outgoing URL** to your listener, appending the secret:
   `http://<host>:<port>/chatwoot/webhook?token=<CHATWOOT_WEBHOOK_SECRET>`.
   Run `hermes gateway setup` → Chatwoot to print the exact URL.
5. Authorize contacts: set `CHATWOOT_ALLOWED_USERS` (comma-separated contact
   ids) or `CHATWOOT_ALLOW_ALL_USERS=true` for open access (dev only).

## Reasoning / tool activity as private notes

With `CHATWOOT_PRIVATE_NOTE_TRACE=true`, the agent's thinking and tool/skill
activity are posted as Chatwoot **private notes** (visible to human agents, not
the customer), while the customer only sees the final reply. Private notes are
attributed to an agent user, so they use `CHATWOOT_AGENT_TOKEN` when set. If the
trace is on but no agent token is configured and the bot token can't post
private notes, the note is skipped with a warning — your customer reply is never
affected.

## Scheduled / proactive delivery

Set `CHATWOOT_HOME_CHANNEL` to an `account:conversation` id (e.g. `1:42`) and use
`deliver=chatwoot` in a cron job to push a message into that conversation.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Webhook returns 404 | Adapter not running / creds missing. |
| Webhook returns 403 | `?token=` doesn't match `CHATWOOT_WEBHOOK_SECRET`. |
| Events arrive but always ignored | Outgoing/private message, empty body, or conversation `status: open` (handoff). |
| Received but never answered, no error | Authorization: allowlist unset (defaults to deny). |
| Reply fails 401/403 | Bad/expired bot token, or a user token used where the bot token is required. |
| Reply fails 404 | Wrong conversation id — the reply path uses the **display id**. |
| Private note fails 401/403 (reply works) | Set `CHATWOOT_AGENT_TOKEN`. |
| `No live adapter for platform 'chatwoot'` on cron | Gateway not running for out-of-process delivery. |
