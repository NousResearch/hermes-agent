---
sidebar_position: 10
title: "Zulip"
description: "Set up Hermes Agent as a Zulip bot"
---

# Zulip Setup

Hermes Agent integrates with Zulip as a bot. Zulip conversations are organized
by streams and topics; Hermes treats the current Zulip topic as the thread, so
replies and tool progress stay where the conversation started.

## How Hermes Behaves

| Context | Behavior |
|---------|----------|
| **DMs** | Hermes responds to every message. No `@mention` needed. |
| **Streams/topics** | Hermes responds when you `@mention` the bot. After that, follow-up messages in the same topic do not need another mention while the gateway is running. |
| **Slash commands** | Hermes accepts `/help`, `/status`, `/new`, `/model`, and other gateway commands directly in Zulip topics. |
| **Approvals** | Dangerous-command approval prompts use Zulip reactions: check = once, clock = session, infinity = always, x = deny. |
| **Scheduled messages** | Optional home stream/topic settings are used for cron jobs, reminders, and proactive notifications. |

## Step 1: Create a Zulip Bot

1. Log in to Zulip as an administrator or a user allowed to create bots.
2. Open **Settings** > **Bots**.
3. Click **Add a new bot**.
4. Choose a generic bot and name it something like `Hermes`.
5. Copy the bot email and API key.

:::warning[Keep the API key secret]
The Zulip bot API key lets Hermes act as the bot. Store it in
`~/.hermes/.env`; never commit it to Git.
:::

Add the bot to any streams where you want it to respond.

## Step 2: Decide Who Can Use Hermes

Hermes supports two access-control styles:

- `ZULIP_ALLOWED_USERS` for quick setup. List Zulip user IDs or email
  addresses, comma-separated.
- `ZULIP_ALLOWED_GROUPS` for teams. List Zulip user group names or IDs,
  comma-separated. Manage membership in Zulip instead of editing Hermes config.

For a company workspace, the easiest pattern is to create a Zulip user group
such as `Hermes Users`, add approved people to that group, and set:

```bash
ZULIP_ALLOWED_GROUPS=Hermes Users
```

You can still keep `ZULIP_ALLOWED_USERS` for admins or break-glass access:

```bash
ZULIP_ALLOWED_USERS=123,alice@example.com
ZULIP_ALLOWED_GROUPS=Hermes Users,Infra
```

:::warning
Do not set `ZULIP_ALLOW_ALL_USERS=true` on a shared or production Zulip server
unless every Zulip user should be able to run Hermes.
:::

## Step 3: Configure Hermes

### Option A: Interactive Setup

Run:

```bash
hermes gateway setup
```

Select **Zulip** and provide:

- Zulip server URL, for example `https://chat.example.com`
- Bot email
- Bot API key
- Allowed users, optional
- Allowed user groups, optional
- Home stream, optional
- Home topic, optional

The home stream/topic prompts are only needed for cron jobs, reminders, and
notifications. Normal `@Hermes` conversations do not require them.

### Option B: Manual Configuration

Add this to `~/.hermes/.env`:

```bash
ZULIP_URL=https://chat.example.com
ZULIP_BOT_EMAIL=hermes-bot@example.com
ZULIP_API_KEY=***

# Choose at least one access-control option:
ZULIP_ALLOWED_USERS=123,alice@example.com
ZULIP_ALLOWED_GROUPS=Hermes Users,Infra

# Keep mention-gating enabled in shared streams:
ZULIP_REQUIRE_MENTION=true

# Optional, only for cron/proactive delivery:
ZULIP_HOME_CHANNEL=general
ZULIP_HOME_TOPIC=Hermes
```

For self-hosted Zulip behind an internal CA, also set:

```bash
ZULIP_CA_CERT=/path/to/internal-ca.crt
```

## Start the Gateway

```bash
hermes gateway
```

Then mention the bot in a Zulip topic:

```text
@Hermes summarize this incident thread
```

Hermes replies in that same topic. Follow-up messages in that topic can be
plain text without another `@Hermes`.

## Optional Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ZULIP_REQUIRE_MENTION` | `true` | Require `@Hermes` before responding in streams. |
| `ZULIP_FREE_RESPONSE_CHANNELS` | unset | Stream names where Hermes responds without an initial mention. |
| `ZULIP_ALLOWED_CHANNELS` | unset | If set, Hermes only responds in these stream names. |
| `ZULIP_APPROVAL_REQUIRE_SENDER` | `true` | Only the requester can resolve command approval reactions. |
| `ZULIP_APPROVAL_TIMEOUT_SECONDS` | `300` | Approval prompt timeout. |
| `ZULIP_HOME_CHANNEL` | unset | Stream for cron/proactive delivery. |
| `ZULIP_HOME_TOPIC` | `Hermes` | Topic for cron/proactive delivery. |
| `ZULIP_CA_CERT` | unset | CA bundle for self-hosted/internal TLS. |
