---
sidebar_position: 7
title: "XMPP (Jabber)"
description: "Run a Hermes agent on any XMPP server — Prosody, ejabberd, public providers, or your own"
---

# XMPP Setup

Hermes connects to XMPP through the [`slixmpp`](https://lab.louiz.org/poezio/slixmpp) library. XMPP is an open, federated chat protocol — pick any server (run your own with [Prosody](https://prosody.im/) or [ejabberd](https://www.ejabberd.im/), or use a public provider like [disroot.org](https://disroot.org/) or [jabber.org](https://www.jabber.org/)) and the bot reaches you over standard 1:1 chats and MUC group rooms.

:::info Bundled plugin
XMPP ships as a bundled platform plugin (`plugins/platforms/xmpp/`) — it's auto-discovered, with no enablement step. The `slixmpp` dependency lazy-installs on first use; to pre-install it explicitly:

```bash
pip install 'hermes-agent[xmpp]'
```
:::

---

## Security model

The v1 adapter encrypts in transit (TLS to your server is mandatory — STARTTLS is forced) but **does not** implement end-to-end OMEMO encryption yet. Messages are visible to your server operator. If that's not acceptable, run your own server (Prosody on a trusted host is a 5-minute setup) or wait for the OMEMO follow-up.

---

## Prerequisites

- **An XMPP/Jabber account** — JID looks like `hermes@example.org` plus a password.
- **A server that supports modern XEPs** (almost any current Prosody/ejabberd does):
  - XEP-0030 (Service Discovery)
  - XEP-0045 (Multi-User Chat) — for group rooms
  - XEP-0085 (Chat State Notifications) — for typing indicators
  - XEP-0363 (HTTP File Upload) — for sending files

### Quickest path: Prosody on Linux

```bash
# Debian / Ubuntu
sudo apt install prosody

# Then create an account
sudo prosodyctl adduser hermes@your-server.local
```

If you don't already have a hostname, edit `/etc/prosody/prosody.cfg.lua` to add a VirtualHost matching your machine, enable the `http_file_share` and `muc` modules, and `sudo systemctl restart prosody`.

---

## Step 1: Configure the gateway

The interactive setup wizard handles env vars for you:

```bash
hermes gateway setup
```

Pick **XMPP (Jabber)** from the platform list. You'll be prompted for:

| Var | Required | Notes |
|-----|----------|-------|
| `XMPP_JID` | yes | Bot JID, e.g. `hermes@example.org` |
| `XMPP_PASSWORD` | yes | Sent over TLS only |
| `XMPP_HOST` | no | Override SRV lookup if your hostname differs from JID domain |
| `XMPP_PORT` | no | Defaults to 5222 (STARTTLS). Use 5223 for direct TLS. |
| `XMPP_MUC_ROOMS` | no | Comma-separated `room@conference.server[/nick]` entries |
| `XMPP_MUC_NICK` | no | Default nick when a MUC entry doesn't specify one |
| `XMPP_HOME_CHANNEL` | no | JID where cron jobs deliver results by default |
| `XMPP_ALLOWED_USERS` | yes (effectively) | Comma-separated **bare JIDs** allowed to DM the bot |
| `XMPP_ALLOW_ALL_USERS` | no | Set to `1` for dev only — bypasses both the DM allow-list and MUC checks |

:::info DM allow-list vs. MUC access
`XMPP_ALLOWED_USERS` only filters **direct messages**. MUC (group chat)
authorization is gated by room membership: if you joined the room via
`XMPP_MUC_ROOMS`, all messages in that room are accepted. The MUC server
itself controls who's in the room. See ADR-0003 in the `hermes-xmpp`
project repo for the full rationale.
:::

These all go into `~/.hermes/.env` if you used the wizard.

---

## Step 2: Start the gateway

```bash
hermes gateway start
```

Send a message from your normal XMPP client (Conversations on Android, Dino on Linux, Gajim on any desktop, Movim in a browser) to your bot's JID. The bot replies inline. In MUC rooms, address the bot by its nick to keep noise low — you can configure it via `XMPP_MUC_NICK`.

### Sending files

The agent can deliver files natively. When it emits `MEDIA:/absolute/path/to/file` in its response, the adapter:

1. Requests an upload slot from your server (XEP-0363).
2. PUTs the bytes over HTTPS.
3. Sends a chat message containing the resulting GET URL.

Modern XMPP clients render that URL inline as a file/image bubble. There's no separate "attachment" UI — the URL *is* the attachment in XEP-0363.

---

## Slash commands

All standard gateway slash commands work over XMPP:

| Command | What it does |
|---------|--------------|
| `/new`, `/reset` | Start a fresh conversation |
| `/model` | Change the underlying LLM |
| `/personality` | Switch personality |
| `/sethome` | Make this chat the cron-delivery default |
| `/status` | Show gateway + platform status |
| `/usage` | Token / cost summary |
| `/<skill-name>` | Invoke any skill the bot has access to |

---

## Troubleshooting

**`xmpp_auth_failed`** — JID or password is wrong, or the server requires SCRAM-SHA-256 with channel binding and your password store doesn't have the right format. Double-check by logging into the same account from a regular client first.

**`xmpp_connect_timeout`** — usually DNS / firewall. Set `XMPP_HOST` explicitly to bypass SRV lookup.

**`HTTP upload (XEP-0363) failed`** — your server's `http_file_share` (Prosody) or `mod_http_upload` (ejabberd) module isn't enabled, or the file exceeds the configured size limit. Check the server admin panel; defaults are usually 10–50 MB.

**Bot doesn't see MUC messages** — make sure the bot is configured in `XMPP_MUC_ROOMS` and the room allows non-member messages. For private rooms, invite the bot first.

**Allow-list rejects you (in a DM)** — `XMPP_ALLOWED_USERS` takes **bare JIDs** (no `/resource` suffix). For dev work, set `XMPP_ALLOW_ALL_USERS=1`.

**Bot doesn't reply in a MUC even though I'm in the allow-list** — MUC access is gated by *room membership*, not per-user. Add the room to `XMPP_MUC_ROOMS`. See ADR-0003 for why.

---

## Compared to other adapters

| Aspect | XMPP | Signal | Telegram |
|--------|------|--------|----------|
| Server you can self-host | Yes (Prosody, ejabberd) | No | No |
| End-to-end encryption (v1) | No | Yes | No (cloud chats) |
| Federation | Yes | No | No |
| Native file rendering in clients | Yes (XEP-0363) | Yes | Yes |
| Group chat | Yes (MUC) | Yes | Yes |

XMPP is the right choice when you want the bot reachable from any client, on any device, without a third-party gateway. Self-hosted Prosody on a LAN gives you a local-only chat with a Hermes agent that never crosses the internet.
