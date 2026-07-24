---
title: "Synology Chat"
description: "Connect Hermes to Synology Chat on your NAS — bot DMs and channel webhooks, fully self-hosted"
---

# Synology Chat Setup

Connect Hermes to [Synology Chat](https://www.synology.com/en-global/dsm/feature/chat), the self-hosted chat application that ships with Synology NAS (DSM 7.x). Everything stays on your LAN: DSM talks to Hermes over plain HTTP webhooks, Hermes talks back to the DSM Chat API.

Two surfaces are supported:

- **Bot DMs** — a DSM Chat *bot* integration. Users DM the bot, Hermes replies through the bot's `method=chatbot` API.
- **Channels** — per-channel *outgoing* + *incoming* webhook pairs. DSM only fires the outgoing webhook for messages containing the **trigger word** you configure on the DSM side, so mention-gating is handled by DSM itself.

## How Hermes Behaves

- Each DM and each channel gets its own Hermes session (`dm:<user_id>` / `ch:<channel_id>` internally).
- The configured-channels table doubles as a **channel whitelist**: messages from channels you haven't configured are rejected with 401.
- Inbound authentication binds each DSM verification token to its source: a channel's token only authorizes messages claiming to come from *that* channel, and the bot token only authorizes DMs. This prevents cross-channel spoofing.
- User authorization is the standard Hermes gateway allowlist (`SYNOLOGY_CHAT_ALLOWED_USERS`) with the pairing flow for unknown users.
- Synology Chat renders plain text only: Hermes strips Markdown and converts links to DSM's `<URL|text>` syntax. Messages are chunked at 2000 characters.

## Step 1: Create the Bot (DMs)

1. In DSM Chat, open **User panel → Integration → Bots → Create**.
2. Name it (e.g. `Hermes`), save, and note:
   - the **token** (64-character string),
   - the **incoming URL** (contains `method=chatbot`),
   - the **outgoing URL** field — set it to `http://<hermes-host>:8645/`.

## Step 2: Create Channel Webhooks (optional, per channel)

Each channel needs **two** DSM integrations:

1. **Outgoing webhook** (DSM → Hermes): **Integration → Outgoing Webhooks → Create**, pick the channel, set a **trigger word** (e.g. your bot's name), point the URL at `http://<hermes-host>:8645/`, note its **token**.
2. **Incoming webhook** (Hermes → DSM): **Integration → Incoming Webhooks → Create**, pick the same channel, note its **URL** (contains `method=incoming`).

Find the channel ID in the DSM Chat channel URL, or just send a trigger-word message after configuring Hermes: the rejected attempt is logged as `rejected inbound (invalid token, channel='<id>')`, which tells you the exact channel ID to configure.

:::warning One outgoing integration per channel
A channel can carry several outgoing-webhook integrations (e.g. one left over from another bot). Hermes maps **one token per channel** — make sure the integration whose token you configured is the one pointing at Hermes, and disable or repoint the others. A perpetual 401 for a channel you *did* configure usually means a *different* integration on the same channel is doing the posting.
:::

## Step 3: Configure Hermes

### Option A: Interactive Setup

```bash
hermes gateway setup
```

Pick **Synology Chat** from the platform menu and follow the prompts.

### Option B: Manual Configuration

Add to `~/.hermes/.env`:

```bash
# Required — bot integration
SYNOLOGY_CHAT_TOKEN=your-64-char-bot-token
SYNOLOGY_CHAT_INCOMING_URL=https://nas.local:5001/webapi/entry.cgi?api=SYNO.Chat.External&method=chatbot&version=2&token=%22...%22

# Allowed users (DSM user IDs, comma-separated). Empty = deny + pairing flow.
SYNOLOGY_CHAT_ALLOWED_USERS=5,12
# Or allow everyone:
# SYNOLOGY_CHAT_ALLOW_ALL_USERS=true

# Per-channel webhook pairs (one pair per channel ID)
SYNOLOGY_CHANNEL_TOKEN_23=outgoing-webhook-token-of-channel-23
SYNOLOGY_CHANNEL_WEBHOOK_23=https://nas.local:5001/webapi/entry.cgi?api=SYNO.Chat.External&method=incoming&version=2&token=%22...%22

# Optional
# SYNOLOGY_CHAT_WEBHOOK_PORT=8645        # inbound listen port
# SYNOLOGY_CHAT_WEBHOOK_HOST=0.0.0.0     # bind interface
# SYNOLOGY_CHAT_HOME_CHANNEL=23           # channel for cron/notification delivery
# SYNOLOGY_CHAT_CA_BUNDLE=/path/to/nas-ca.pem   # recommended for self-signed TLS
# SYNOLOGY_CHAT_ALLOW_INSECURE_SSL=true  # escape hatch if you can't pin the CA
```

Channels can also be configured in `config.yaml`:

```yaml
synology_chat:
  channels:
    "23":
      token: outgoing-webhook-token
      incoming_url: https://nas.local:5001/webapi/entry.cgi?api=SYNO.Chat.External&method=incoming&version=2&token=%22...%22
```

### Start the Gateway

```bash
hermes gateway run
```

Look for `Synology Chat: webhook server listening on 0.0.0.0:8645` and `✓ synology_chat connected` in the logs, then DM your bot.

## Networking

- The DSM **outgoing URLs accept any host, port, path, and plain HTTP** — no TLS requirement. Hermes listens on one port (default `8645`) for all integrations.
- The Hermes host needs a **static IP/hostname**: DSM stores the outgoing URL statically, and a DHCP lease change silently breaks delivery.
- The webhook endpoint authenticates by token only (DSM has no per-message signature), so **restrict the port to your trusted LAN** (firewall rule) and never expose it to the internet.
- Replies go to the NAS Chat API over HTTPS. Synology's default certificate is self-signed: export your NAS CA and set `SYNOLOGY_CHAT_CA_BUNDLE` (preferred), or set `SYNOLOGY_CHAT_ALLOW_INSECURE_SSL=true` as a last resort (a warning is logged).

## Cron / Notification Delivery

Set `SYNOLOGY_CHAT_HOME_CHANNEL=<channel_id>` (or a user ID for DM delivery) and use `deliver=synology_chat` in cron jobs. Out-of-process delivery is supported — cron jobs work even when they run outside the gateway process.

## Quirks & Troubleshooting

- **DSM only accepts form-encoded payloads**: outbound messages are sent as `payload=<url-encoded JSON>` in an `application/x-www-form-urlencoded` body. A raw JSON body is silently ignored by DSM — if Hermes logs success but nothing appears in Chat, check you are not behind a proxy that rewrites the body.
- **Trigger word**: DSM strips nothing — Hermes removes the leading trigger word from channel messages before passing them to the agent.
- **DM payloads vs channel payloads**: DMs carry neither `channel_id` nor `trigger_word`. Some DSM versions omit `channel_id` and send only `channel_name` for channels — Hermes accepts either as the channel key. `post_id` presence varies by DSM version.
- **Rate limiting**: DSM throttles at roughly 2 messages/second (API error code `411`). Hermes paces sends at 500 ms and retries 411s with backoff. DSM has also been observed to return HTTP 200 and silently drop messages under sustained bursts.
- **DSM retries non-2xx webhooks** about once a minute — if your handler was briefly down, expect a delayed duplicate delivery of the last message.
- **User IDs**: the `user_id` in outgoing-webhook payloads is normally the same as the Chat API user ID used by `method=chatbot` — but on some installs integration-scoped IDs differ. If DM replies fail with DSM error `800` (*no target*), compare the webhook `user_id` with the ID shown in the DSM Chat user profile.
