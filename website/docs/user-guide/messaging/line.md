---
sidebar_position: 17
title: "LINE"
description: "Set up Hermes Agent as a LINE Messaging API bot"
---

# LINE Setup

Run Hermes Agent as a [LINE](https://line.me/) bot via the official LINE Messaging API. The adapter lives as a bundled platform plugin under `plugins/platforms/line/` — no core edits, just enable it like any other platform.

LINE is the dominant messaging app in Japan, Taiwan, and Thailand. If your users live there, this is how they reach you.

## How the bot responds

| Context | Behavior |
|---------|----------|
| **1:1 chat** (`U` IDs) | Responds to every message |
| **Group chat** (`C` IDs) | Responds when the group is on the allowlist |
| **Multi-user room** (`R` IDs) | Responds when the room is on the allowlist |

Inbound text, images, audio, video, files, stickers, and locations are all handled. Outbound text uses the **free reply token first** (single-use, ~60s window) and falls back to the metered Push API when the token has expired.

---

## Step 1: Create a LINE Messaging API channel

1. Go to the [LINE Developers Console](https://developers.line.biz/console/).
2. Create a Provider, then under it a **Messaging API** channel.
3. From the channel's **Basic settings** tab, copy the **Channel secret**.
4. From the **Messaging API** tab, scroll to **Channel access token (long-lived)** and click **Issue**. Copy the token.
5. In the **Messaging API** tab, also disable **Auto-reply messages** and **Greeting messages** so they don't fight your bot's replies.

---

## Step 2: Expose the webhook port

LINE delivers webhooks over public HTTPS. The default port is `8646` — override with `LINE_PORT` if needed.

```bash
# Cloudflare Tunnel (recommended for production — fixed hostname)
cloudflared tunnel --url http://localhost:8646

# ngrok (good for dev)
ngrok http 8646

# devtunnel
devtunnel create hermes-line --allow-anonymous
devtunnel port create hermes-line -p 8646 --protocol https
devtunnel host hermes-line
```

Copy the `https://...` URL — you'll set it as the webhook URL below. **Leave the tunnel running** while testing. For production, set up a fixed Cloudflare named tunnel so the webhook URL doesn't change on restart.

---

## Step 3: Configure Hermes

Add to `~/.hermes/.env`:

```env
LINE_CHANNEL_ACCESS_TOKEN=YOUR_LONG_LIVED_TOKEN
LINE_CHANNEL_SECRET=YOUR_CHANNEL_SECRET

# Allowlist — at least one of these (or LINE_ALLOW_ALL_USERS=true for dev)
LINE_ALLOWED_USERS=U1234567890abcdef...           # comma-separated U-prefixed IDs
LINE_ALLOWED_GROUPS=C1234567890abcdef...          # optional group IDs
LINE_ALLOWED_ROOMS=R1234567890abcdef...           # optional room IDs

# Required for image / audio / video sends — the public HTTPS base URL
# the tunnel resolves to.  Without it, send_image/voice/video will refuse.
LINE_PUBLIC_URL=https://my-tunnel.example.com
```

Then in `~/.hermes/config.yaml`:

```yaml
gateway:
  platforms:
    line:
      enabled: true
```

That's enough — the bundled-plugin scan in `gateway/config.py` automatically picks up `plugins/platforms/line/`. No `Platform.LINE` enum edit, no `_create_adapter` registration.

---

## Step 4: Set the webhook URL

Back in the LINE console:

1. Open your channel → **Messaging API** tab.
2. Under **Webhook settings** → **Webhook URL**, paste `https://<your-tunnel>/line/webhook` (note the `/line/webhook` path — the adapter listens there).
3. Click **Verify**. LINE pings the URL; you should see a 200.
4. Toggle **Use webhook** to **On**.

---

## Step 5: Run the gateway

```bash
hermes gateway
```

The agent log shows:

```
LINE: webhook listening on 0.0.0.0:8646/line/webhook (public: https://my-tunnel.example.com)
```

Add the bot as a friend from the LINE app (scan the QR in the channel's **Messaging API** tab) and send it a message.

---

## Slow LLM responses

LINE's reply token is single-use and expires roughly 60 seconds after the inbound event. Slow LLMs can't reply in time, which would normally force a paid Push API call.

When the LLM is still running past `LINE_SLOW_RESPONSE_THRESHOLD` seconds (default `45`), the adapter consumes the original reply token to send a **Template Buttons** bubble:

> 🤔 Still thinking. Tap below to fetch the answer when it's ready.
>
> [ Get answer ]

The user taps **Get answer** when convenient — that postback delivers a *fresh* reply token, which the adapter uses to send the cached answer (still free).

State machine: `PENDING → READY → DELIVERED`, plus `ERROR` for cancelled runs (the orphan PENDING resolves to "Run was interrupted before completion." after `/stop` so the persistent button doesn't loop).

To disable the postback button and always Push-fallback instead:

```env
LINE_SLOW_RESPONSE_THRESHOLD=0
```

For the postback flow to fire reliably, suppress chatter that would consume the reply token before the threshold:

```yaml
# ~/.hermes/config.yaml
display:
  interim_assistant_messages: false
  platforms:
    line:
      tool_progress: off
```

---

## Cron / notification delivery

```env
LINE_HOME_CHANNEL=Uxxxxxxxxxxxxxxxxxxxx     # default delivery target
```

Cron jobs with `deliver: line` route to `LINE_HOME_CHANNEL`. The adapter ships a standalone Push-only sender so cron jobs work even when cron runs in a separate process from the gateway.

---

## Environment variable reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `LINE_CHANNEL_ACCESS_TOKEN` | yes | — | Long-lived channel access token |
| `LINE_CHANNEL_SECRET` | yes | — | Channel secret (HMAC-SHA256 webhook verification) |
| `LINE_HOST` | no | `0.0.0.0` | Webhook bind host |
| `LINE_PORT` | no | `8646` | Webhook bind port |
| `LINE_PUBLIC_URL` | for media | — | Public HTTPS base URL; required for image/voice/video sends |
| `LINE_ALLOWED_USERS` | one of | — | Comma-separated user IDs (U-prefixed) |
| `LINE_ALLOWED_GROUPS` | one of | — | Comma-separated group IDs (C-prefixed) |
| `LINE_ALLOWED_ROOMS` | one of | — | Comma-separated room IDs (R-prefixed) |
| `LINE_ALLOW_ALL_USERS` | dev only | `false` | Skip allowlist entirely |
| `LINE_HOME_CHANNEL` | no | — | Default cron / notification delivery target |
| `LINE_SLOW_RESPONSE_THRESHOLD` | no | `45` | Seconds before the postback button fires (`0` = disabled) |
| `LINE_PENDING_TEXT` | no | "🤔 Still thinking…" | Bubble text shown alongside the postback button |
| `LINE_BUTTON_LABEL` | no | "Get answer" | Button label |
| `LINE_DELIVERED_TEXT` | no | "Already replied ✅" | Reply when an already-delivered button is tapped again |
| `LINE_INTERRUPTED_TEXT` | no | "Run was interrupted before completion." | Reply when a `/stop` orphan button is tapped |

---

## Troubleshooting

**"invalid signature" on webhook verify.** The `Channel secret` was copied wrong, or your tunnel rewrote the request body. Verify with `curl -i https://<tunnel>/line/webhook/health` first — that should return `{"status":"ok","platform":"line"}`.

**Bot receives nothing in groups.** Check `LINE_ALLOWED_GROUPS` includes the `C...` group ID. To find a group ID, send a test message and grep `~/.hermes/logs/gateway.log` for `LINE: rejecting unauthorized source` — the rejected source dict has the IDs.

**`send_image` fails with "LINE_PUBLIC_URL must be set".** LINE's Messaging API does not accept binary uploads — images, audio, and video must be reachable HTTPS URLs. Set `LINE_PUBLIC_URL` to the tunnel's public hostname and the adapter will serve files from `/line/media/<token>/<filename>` automatically.

**Postback button never appears.** Either the LLM responded faster than `LINE_SLOW_RESPONSE_THRESHOLD`, or another bubble (tool-progress, streaming) consumed the reply token first. See the suppression block under "Slow LLM responses".

**"already in use by another profile".** The same channel access token is bound to another running Hermes profile. Stop the other gateway or use a separate channel.

---

## Limitations

* **Single bubble per chunk.** Each LINE text bubble is capped at 5000 characters, and at most 5 bubbles are sent per Reply/Push call. Longer responses are truncated with an ellipsis.
* **No native message editing.** LINE has no edit-message API — streaming responses always send fresh bubbles, never edit prior ones.
* **No Markdown rendering.** Bold (`**`), italics (`*`), code fences, and headings render as literal characters. The adapter strips them before sending; URLs are preserved (`[label](url)` becomes `label (url)`).
* **Loading indicator is DM-only.** LINE rejects the chat/loading API for groups and rooms, so the typing indicator only shows in 1:1 chats.
