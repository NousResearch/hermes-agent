---
sidebar_position: 12
title: "Google Chat"
description: "Set up Hermes Agent as a Google Chat bot using Cloud Pub/Sub"
---

# Google Chat Setup

Connect Hermes Agent to Google Chat as a bot. The integration uses Cloud Pub/Sub
pull subscriptions for inbound events and the Chat REST API for outbound messages.
Equivalent ergonomics to Slack Socket Mode or Telegram long-polling: your Hermes
process does not need a public URL, a tunnel, or a TLS certificate. It connects,
authenticates, and listens on a subscription — the same way a Telegram bot listens
on a token.

:::note Workspace edition
Google Chat is part of Google Workspace. You can use this integration with a
personal Workspace (`@yourdomain.com` registered through Google) or a work
Workspace where you have the Admin rights to publish an app. Gmail-only accounts
cannot host Chat apps.
:::

## Overview

| Component | Value |
|-----------|-------|
| **Libraries** | `google-cloud-pubsub`, `google-api-python-client`, `google-auth` |
| **Inbound transport** | Cloud Pub/Sub pull subscription (no public endpoint) |
| **Outbound transport** | Chat REST API (`chat.googleapis.com`) |
| **Authentication** | Service Account JSON with `roles/pubsub.subscriber` on the subscription |
| **User identification** | Chat resource names (`users/{id}`) + email |

---

## Step 1: Create or pick a GCP project

You need a Google Cloud project to host the Pub/Sub topic. If you don't have one,
create it at [console.cloud.google.com](https://console.cloud.google.com) —
personal accounts get a free tier that easily covers bot traffic.

Note the project ID (e.g., `my-chat-bot-123`). You'll use it in every subsequent
step.

---

## Step 2: Enable two APIs

In the console, go to **APIs & Services → Library** and enable:

- **Google Chat API**
- **Cloud Pub/Sub API**

Both are free for the volumes a personal bot generates.

---

## Step 3: Create a Service Account

**IAM & Admin → Service Accounts → Create Service Account.**

- Name: `hermes-chat-bot`
- Skip the "Grant this service account access to project" step. IAM on the specific
  subscription is all you need — do **NOT** grant project-level Pub/Sub roles.

After creation, open the SA, go to **Keys → Add Key → Create new key → JSON** and
download the file. Save it somewhere only Hermes can read (e.g.,
`~/.hermes/google-chat-sa.json`, `chmod 600`).

:::caution There is NO "Chat Bot Caller" role
A common mistake is to search for a Chat-specific IAM role and grant it at the
project level. That role doesn't exist. Chat bot authority comes from being
installed in a space, not from IAM. All your SA needs is Pub/Sub subscriber on
the subscription you create in the next step.
:::

---

## Step 4: Create the Pub/Sub topic and subscription

**Pub/Sub → Topics → Create topic.**

- Topic ID: `hermes-chat-events`
- Leave the defaults for everything else.

After creation, the topic's detail page has a **Subscriptions** tab. Create one:

- Subscription ID: `hermes-chat-events-sub`
- Delivery type: **Pull**
- Message retention: **7 days** (so backlog survives a hermes restart)
- Leave the rest default.

---

## Step 5: IAM binding on the topic (critical)

On the **topic** (not the subscription), add an IAM principal:

- Principal: `chat-api-push@system.gserviceaccount.com`
- Role: `Pub/Sub Publisher`

Without this, Google Chat cannot publish events to your topic and your bot will
never receive anything.

---

## Step 6: IAM binding on the subscription

On the **subscription**, add your own Service Account as a principal:

- Principal: `hermes-chat-bot@<your-project>.iam.gserviceaccount.com`
- Role: `Pub/Sub Subscriber`

Also grant `Pub/Sub Viewer` on the same subscription — Hermes calls
`subscription.get()` at startup as a reachability check.

---

## Step 7: Configure the Chat app

Go to **APIs & Services → Google Chat API → Configuration**.

- **App name**: whatever you want users to see ("Hermes" is reasonable).
- **Avatar URL**: any public PNG (Google has some defaults).
- **Description**: a short sentence shown in the app directory.
- **Functionality**: enable **Receive 1:1 messages** and **Join spaces and group
  conversations**.
- **Connection settings**: select **Cloud Pub/Sub**, enter the topic name
  `projects/<your-project>/topics/hermes-chat-events`.
- **Visibility**: restrict to your workspace (or specific users) — do not publish
  to everyone while you're testing.

Save.

---

## Step 8: Install the bot in a test space

Open Google Chat in a browser. Start a DM with your app by searching for its name
in the **+ New Chat** menu. The first time you message it, Google sends an
`ADDED_TO_SPACE` event that Hermes uses to cache the bot's own `users/{id}` for
self-message filtering.

---

## Step 9: Configure Hermes

Add the Google Chat section to `~/.hermes/.env`:

```bash
# Required
GOOGLE_CHAT_PROJECT_ID=my-chat-bot-123
GOOGLE_CHAT_SUBSCRIPTION_NAME=projects/my-chat-bot-123/subscriptions/hermes-chat-events-sub
GOOGLE_CHAT_SERVICE_ACCOUNT_JSON=/home/you/.hermes/google-chat-sa.json

# Authorization — paste the emails of people allowed to talk to the bot
GOOGLE_CHAT_ALLOWED_USERS=you@yourdomain.com,coworker@yourdomain.com

# Optional
GOOGLE_CHAT_HOME_CHANNEL=spaces/AAAA...         # default delivery destination for cron jobs
GOOGLE_CHAT_MAX_MESSAGES=1                      # Pub/Sub FlowControl; 1 serializes commands per session
GOOGLE_CHAT_MAX_BYTES=16777216                  # 16 MiB — cap on in-flight message bytes
```

The project ID also falls back to `GOOGLE_CLOUD_PROJECT`, and the SA path falls
back to `GOOGLE_APPLICATION_CREDENTIALS` — use whichever convention you prefer.

Install Hermes with the optional dependencies:

```bash
pip install 'hermes-agent[google_chat]'
```

Start the gateway:

```bash
hermes gateway
```

You should see a log line like:

```
[GoogleChat] Connected; project=my-chat-bot-123, subscription=<redacted>,
             bot_user_id=users/XXXX, flow_control(msgs=1, bytes=16777216)
```

Send "hola" in the test DM. The bot posts a "Hermes is thinking…" marker, then
edits that same message in place with the real response — no "message deleted"
tombstones.

---

## Formatting and capabilities

Google Chat renders a limited markdown subset:

| Supported | Not supported |
|-----------|---------------|
| `*bold*`, `_italic_`, `~strike~`, `` `code` `` | Headings, lists |
| Inline images via URL | Interactive Card v2 buttons (v1 of this gateway) |
| Native file attachments (after `/setup-files` — see Step 10) | Native voice notes / circular video notes |

The agent's system prompt includes a Google Chat–specific hint so it knows these
limits and avoids formatting that won't render.

Message size limit: 4000 characters per message. Longer agent responses are
automatically split across multiple messages.

Thread support: when a user replies inside a thread, Hermes detects the
`thread.name` and posts its reply in the same thread, so each thread gets a
separate Hermes session.

---

## Step 10: Native attachment delivery (optional)

Out of the box the bot can post text, inline images via URL, and download cards
for audio/video/documents. To deliver **native** Chat attachments — the same
file widget you get when a human drags-and-drops a file — each user authorizes
the bot once via a per-user OAuth flow.

### Why a separate flow

Google Chat's `media.upload` endpoint hard-rejects service-account auth:

> This method doesn't support app authentication with a service account.
> Authenticate with a user account.

There's no IAM role or scope that fixes this. The endpoint only accepts user
credentials. So the bot has to act *as a user* whenever it uploads a file —
specifically, as the user who asked for the file.

### One-time host setup

1. Go to **APIs & Services → Credentials** in the same GCP project.
2. **Create credentials → OAuth client ID → Desktop app**.
3. Download the JSON. Move it onto the host that runs Hermes.
4. On the host, register the client with Hermes:

```bash
python -m gateway.platforms.google_chat_user_oauth \
    --client-secret /path/to/client_secret.json
```

That writes `~/.hermes/google_chat_user_client_secret.json`. This is shared
infrastructure — it identifies the OAuth *app*, not any individual user. One
file per host is enough no matter how many users authorize later.

### Per-user authorization (in chat)

Each user runs the flow once, in their own DM with the bot:

1. They send `/setup-files` to the bot. It replies with status and the next
   step.
2. They send `/setup-files start`. The bot replies with an OAuth URL.
3. They open the URL, click **Allow**, and watch the browser fail to load
   `http://localhost:1/?...&code=...`. That failure is expected — the auth
   code is in the URL bar.
4. They copy the failed URL (or just the `code=...` value) and paste it back
   into chat as `/setup-files <PASTED_URL>`. The bot exchanges it for a
   refresh token.

The token lands at `~/.hermes/google_chat_user_tokens/<sanitized_email>.json`.
Subsequent file requests in that user's DM use *their* token, so the bot
uploads as them and the message lands in their space.

To revoke later: `/setup-files revoke` deletes only that user's token. Other
users' tokens are untouched.

### Scope

The flow requests exactly one scope: `chat.messages.create`. That covers both
`media.upload` and the `messages.create` that references the uploaded
`attachmentDataRef`. No Drive, no broader Chat scopes — this is least-privilege
on purpose.

### Multi-user behavior

When the asker has no per-user token yet, the bot falls back to a legacy
single-user token at `~/.hermes/google_chat_user_token.json` (if present from
a pre-multi-user install). When neither is available, the bot posts a clear
text notice telling the asker to run `/setup-files`.

A user revoking only clears their own slot. A 401/403 from one user's token
evicts only that user's cache. Users don't disrupt each other.

---

## Troubleshooting

**Bot stays silent after sending "hola."**

1. Check the Pub/Sub subscription has undelivered messages in the console.
   If it does, Hermes isn't authenticated — verify `GOOGLE_CHAT_SERVICE_ACCOUNT_JSON`
   and that the SA is listed as `Pub/Sub Subscriber` on the subscription.
2. If the subscription has zero messages, Google Chat isn't publishing.
   Double-check the IAM binding on the **topic**:
   `chat-api-push@system.gserviceaccount.com` must have `Pub/Sub Publisher`.
3. Check `hermes gateway` logs for `[GoogleChat] Connected`. If you see
   `[GoogleChat] Config validation failed`, the error message tells you which
   env var to fix.

**Bot replies but an error message appears instead of the agent's answer.**

Check logs for `[GoogleChat] Pub/Sub stream died` — if these repeat, your SA
credentials may have been rotated or the subscription deleted. After 10 attempts
the adapter marks itself fatal.

**"403 Forbidden" on every outbound message.**

The bot was removed from the space, or you revoked it in the Chat API console.
Re-install it in the space (the next `ADDED_TO_SPACE` event will re-enable
messaging automatically).

**Too many "Rate limit hit" warnings.**

The Chat API's default quotas allow 60 messages per space per minute. If your
agent produces long streaming responses that exceed that, the adapter retries
with exponential backoff — but you'll still see user-visible latency. Consider
concise responses or raising the quota in the GCP console.

**Bot keeps posting the "/setup-files" notice instead of files.**

The asker has no per-user OAuth token and there's no legacy fallback. Run
`/setup-files` in their DM and follow Step 10. After the exchange completes
the next file request uploads natively without a gateway restart.

**`/setup-files start` says "No client credentials stored on the host."**

The one-time host setup wasn't done. From a terminal on the host that runs
Hermes:

```bash
python -m gateway.platforms.google_chat_user_oauth \
    --client-secret /path/to/client_secret.json
```

Then send `/setup-files start` again.

**`/setup-files <PASTED_URL>` says "Token exchange failed."**

The auth code is single-use and short-lived (typically a few minutes). Send
`/setup-files start` to get a fresh URL and retry.

---

## Security notes

- **Service Account scope**: the adapter requests `chat.bot` and `pubsub` scopes.
  IAM should be the actual enforcement — grant your SA the minimum
  (`roles/pubsub.subscriber` + `roles/pubsub.viewer` on the subscription), not
  project-level or org-level Pub/Sub roles.
- **Attachment download protection**: Hermes will only attach the SA bearer
  token to URLs whose host matches a short allowlist of Google-owned domains
  (`googleapis.com`, `drive.google.com`, `lh[3-6].googleusercontent.com`, and
  a few others). Any other host is rejected before the HTTP request, to
  protect against SSRF scenarios where a crafted event could redirect the
  bearer token to the GCE metadata service.
- **Redaction**: Service Account emails, subscription paths, and topic paths
  are stripped from log output by `agent/redact.py`. The debug envelope dump
  (`GOOGLE_CHAT_DEBUG_RAW=1`) routes through the same redaction filter and
  logs at DEBUG level.
- **Compliance**: if you plan to connect this bot to a regulated workspace
  (anything with a data-residency or AI-governance policy), get that approval
  before the first install.
- **User OAuth scope**: the per-user attachment flow requests *only*
  `chat.messages.create` — the minimum that covers `media.upload` plus the
  follow-up `messages.create`. Tokens are persisted as plain JSON at
  `~/.hermes/google_chat_user_tokens/<sanitized_email>.json` (filesystem
  permissions are the protection — same model as the SA key file). Each
  token is owned by exactly one user; revoke is scoped to that user.
