---
sidebar_position: 6
title: "WhatsApp Business (Cloud API)"
description: "Set up Hermes Agent as a WhatsApp bot via Meta's official Business Cloud API"
---

# WhatsApp Business Cloud API Setup

Hermes can connect to WhatsApp through Meta's **official** WhatsApp Business Cloud API. This is the production-grade path: no Node.js bridge subprocess, no QR codes, no account-ban risk.

In exchange:

- You need a **Meta Business account** (not personal WhatsApp).
- The bot operates on a dedicated business phone number, not your personal number.
- The Hermes gateway needs a **public HTTPS URL** so Meta can deliver inbound messages via webhook.
- Replies more than 24 hours after the user's last message require a pre-approved **template** (this is Meta's "customer service window" rule, not a Hermes limit).

If those constraints don't work for your use case, the [Baileys bridge integration](./whatsapp.md) is the alternative — personal account, no public URL needed, but unofficial and ban-prone.

:::tip Which one should I use?
- **Cloud API (this guide)** — running a real business bot, want stability, fine with the Meta verification + template paperwork
- **[Baileys bridge](./whatsapp.md)** — personal projects, quick demos, single-user setups, willing to risk the bot phone number's account
:::

---

## Quick start

```bash
hermes whatsapp-cloud
```

The wizard walks you through every credential, validates each one as you paste it (catches the #1 setup trap — pasting a phone number into the Phone Number ID field), and prints exact follow-up instructions for the parts that need to happen outside the wizard (starting cloudflared, configuring Meta's webhook dashboard).

The rest of this page is the manual reference.

---

## Prerequisites

1. **A Meta Business account**.  Create one at [business.facebook.com](https://business.facebook.com/).
2. **A Meta app with WhatsApp enabled**.  See "Creating the Meta app" below.
3. **A way to expose a local port to the public internet** with HTTPS.  Cloudflare Tunnel (`cloudflared`) is recommended — free, no port forwarding, no domain required.  ngrok, your own domain with a reverse proxy + TLS, or a VPS with the gateway directly bound to a public IP all work too.
4. **Optional but recommended**: ffmpeg on `PATH` so outbound voice messages render as native WhatsApp voice-note bubbles (green waveform) instead of MP3 audio attachments. Hermes degrades gracefully if absent.

---

## Creating the Meta app

1. Go to [developers.facebook.com/apps](https://developers.facebook.com/apps) → **Create App**.
2. Choose use case: **"Connect with customers through WhatsApp"** → **Next**.
3. Pick or create a business portfolio.  Review the publishing requirements.  Confirm → **Create app**.
4. After creation you'll land on **Customize use case → Connect on WhatsApp → Quickstart**.  Click **Start using the API** → you're now on the **API Setup** page.
5. Make sure a WhatsApp Business Account (WABA) is linked.  If you created a new portfolio in step 3, one was auto-created.  Verify in the API Setup page.

You'll need these values from the dashboard — the wizard prompts for them in this order:

| Value | Where in dashboard | Field shape | Notes |
|---|---|---|---|
| **Phone Number ID** | App Dashboard → WhatsApp → API Setup → below the "From" dropdown | Numeric, 15-17 digits | **NOT** the phone number itself. The #1 setup mistake is pasting the actual phone number here. |
| **Access Token** | App Dashboard → WhatsApp → API Setup → "Generate access token" | Starts with `EAA`, 100+ chars | Temp tokens last 24h — see "Permanent token" below for production. |
| **App Secret** | App Dashboard → Settings → Basic → click "Show" next to App secret | 32-character lowercase hex | Used to verify incoming webhook signatures.  Without it, inbound delivery is refused with 503. |
| **App ID** (optional) | App Dashboard → Settings → Basic | Numeric, 15-16 digits | Not required for messaging, useful for analytics. |
| **WABA ID** (optional) | App Dashboard → WhatsApp → API Setup → near the top | Numeric, 15+ digits | Not required for messaging, useful for analytics. |

---

## Permanent token (production)

Temporary access tokens expire after **24 hours**, which means a token generated today stops working tomorrow.  For production deployments use a **System User permanent token**:

1. Go to [business.facebook.com/latest/settings](https://business.facebook.com/latest/settings) → **System users** (left sidebar).
2. **Add** → name (e.g. `hermes-bot`) → role: **Admin**.
3. Select the new user → **Assign Assets**:
   - Select your app → toggle **Manage app** under Full control.
   - Select your WhatsApp account → toggle **Manage WhatsApp Business Accounts** under Full control.
   - Click **Assign assets**.
4. **Generate token** with these permissions:
   - `business_management`
   - `whatsapp_business_messaging`
   - `whatsapp_business_management`
5. Set **token expiration: Never**.
6. Copy the token → update `WHATSAPP_CLOUD_ACCESS_TOKEN` in `~/.hermes/.env` → restart the gateway.

System User tokens don't expire unless you explicitly revoke them.

---

## Exposing Hermes to the internet

The Cloud API delivers inbound messages by HTTPS POST to your webhook URL — that means the Hermes gateway has to be reachable from Meta's servers.  Three common ways:

### Cloudflare Tunnel (recommended)

Free, no port forwarding, works on Windows / macOS / Linux.  Runs as a separate process alongside the gateway.

**Install:**

```bash
# Windows
winget install Cloudflare.cloudflared

# macOS
brew install cloudflared

# Linux
# Download the binary from https://github.com/cloudflare/cloudflared/releases
```

**Run a quick tunnel** (no Cloudflare account needed — gives you a `https://<random>.trycloudflare.com` URL):

```bash
cloudflared tunnel --url http://localhost:8090
```

Note the printed URL — that's what you'll give Meta.

:::warning Quick tunnels rotate
The free quick-tunnel URL changes every time you restart `cloudflared`.  For a stable URL, log in with `cloudflared tunnel login` and create a named tunnel.  Free Cloudflare accounts get unlimited named tunnels — see [Cloudflare's docs](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/) for the named-tunnel workflow.
:::

### ngrok

```bash
ngrok http 8090
```

Free tier shows a different URL on each restart.  Paid tier gives you a stable subdomain.

### Your own domain + reverse proxy

If you already have a server with a TLS cert (Caddy, nginx, etc.), point a route at `localhost:8090`.  This is the most stable option for production but requires existing infrastructure.

---

## Configuring the webhook on Meta's side

Once your tunnel is running:

1. Note the public URL printed by your tunnel — say `https://abc123.trycloudflare.com`.
2. Generate a **Verify Token** — the wizard does this for you with `secrets.token_urlsafe(32)`; if you're configuring manually, run:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
   Save it as `WHATSAPP_CLOUD_VERIFY_TOKEN` in `~/.hermes/.env`.
3. Start the Hermes gateway: `hermes gateway`.
4. In the Meta App Dashboard → **WhatsApp → Configuration** (or **Use cases → Customize → Configuration** depending on UI version) → click **Edit** on the Webhook section.
5. Fill in:
   - **Callback URL**: `https://abc123.trycloudflare.com/whatsapp/webhook`
   - **Verify Token**: the string from step 2 (must match exactly)
6. Click **Verify and save**.  Meta hits your URL with a GET request, the gateway echoes back the challenge, and Meta marks the webhook as verified.
7. Under **Webhook fields**, click **Manage** → subscribe to the **messages** field.  This is what tells Meta to actually deliver inbound messages to your webhook.

**To verify the loop manually** (from a third terminal):

```bash
TUNNEL="https://abc123.trycloudflare.com"
VERIFY="<your verify token>"

# Should print HTTP 200 with body "hello"
curl -i "$TUNNEL/whatsapp/webhook?hub.mode=subscribe&hub.verify_token=$VERIFY&hub.challenge=hello"

# Health endpoint — should show verify_token_configured: true and app_secret_configured: true
curl "$TUNNEL/health"
```

---

## Recipient whitelist (Meta-side)

In development mode (before your app goes through App Review), Meta restricts which numbers your bot can message:

1. App Dashboard → WhatsApp → API Setup → **To** dropdown.
2. Click **Manage phone number list**.
3. Add the phone numbers you want to message (yours, your team's, friendly testers).  Meta sends each one a 6-digit verification code via SMS or WhatsApp.

Up to 5 numbers in dev mode.  Going to App Review removes this limit.

---

## Allowlist (Hermes-side)

In addition to Meta's recipient whitelist, Hermes has its own per-platform allowlist that controls **which incoming messages the agent processes**.  Add to `~/.hermes/.env`:

```bash
# Comma-separated phone numbers, country code, no '+' / spaces / dashes
WHATSAPP_CLOUD_ALLOWED_USERS=15551234567,15557654321

# Or allow everyone (only safe in combination with Meta's recipient whitelist)
# WHATSAPP_CLOUD_ALLOW_ALL_USERS=true
```

The wizard sets this in step 6.  Without an allowlist, **every inbound message is denied** — this is intentional, so the bot can't be invoked by random numbers if the recipient whitelist is ever loosened.

---

## Polishing your bot's WhatsApp profile

WhatsApp displays a **name and profile picture** for your bot in the chat header and contact list. These can't be set via the Cloud API — they live in Meta's Business Manager.

Once your bot is working, head to **[business.facebook.com/wa/manage/phone-numbers](https://business.facebook.com/wa/manage/phone-numbers/)**, click your phone number, and you'll find:

| What | Where | Notes |
|---|---|---|
| **Display name** | Top of the phone-number page | Changes go through Meta's name-review process (~24–48 hours). |
| **Profile picture** | Top of the phone-number page | Square image, ≥640×640px recommended. Updates immediately. |
| **About / description / website / email / hours / category** | "Edit profile" button | These appear in the info pane when a user taps the bot's name. Cosmetic. |
| **Verified badge** (green checkmark) | Business Manager → Security Center → Start Verification | Requires Meta's separate business verification process. |

The `hermes whatsapp-cloud` wizard prints these links at the end of setup. None of this is required for the bot to work — it's pure polish for how your bot appears to users.

---

## Configuration reference

All settings live in `~/.hermes/.env`.  Required values are in **bold**.

| Variable | Default | Description |
|---|---|---|
| **`WHATSAPP_CLOUD_PHONE_NUMBER_ID`** | — | The 15-17 digit ID from API Setup.  **Not** the phone number. |
| **`WHATSAPP_CLOUD_ACCESS_TOKEN`** | — | Meta access token (starts with `EAA`).  Temp 24h or System User permanent. |
| **`WHATSAPP_CLOUD_APP_SECRET`** | — | 32-char hex from Settings → Basic.  Without it, inbound is refused with 503. |
| **`WHATSAPP_CLOUD_VERIFY_TOKEN`** | — | Shared secret for the GET handshake.  Auto-generated by the wizard. |
| **`WHATSAPP_CLOUD_ALLOWED_USERS`** | — | Comma-separated wa_ids allowed to message the bot. |
| `WHATSAPP_CLOUD_ALLOW_ALL_USERS` | `false` | Set to `true` to bypass the allowlist. |
| `WHATSAPP_CLOUD_APP_ID` | — | Optional, for future analytics integration. |
| `WHATSAPP_CLOUD_WABA_ID` | — | Optional, for future analytics integration. |
| `WHATSAPP_CLOUD_WEBHOOK_HOST` | `0.0.0.0` | Interface the webhook server binds to. |
| `WHATSAPP_CLOUD_WEBHOOK_PORT` | `8090` | Port the webhook server binds to.  Must match the port your tunnel forwards. |
| `WHATSAPP_CLOUD_WEBHOOK_PATH` | `/whatsapp/webhook` | URL path Meta posts to. |
| `WHATSAPP_CLOUD_API_VERSION` | `v20.0` | Meta Graph API version. Only override if a newer version is recommended in Meta's docs. |
| `WHATSAPP_CLOUD_HOME_CHANNEL` | — | wa_id to use as the bot's home channel (for cron jobs etc). |

You can have **both** the Baileys (`whatsapp`) and Cloud (`whatsapp_cloud`) adapters enabled simultaneously, targeting different phone numbers.

---

## Features

### Inbound

- **Text messages** — passed straight to the agent.
- **Images** — auto-downloaded and attached to the agent's input. Models with native vision (Claude, GPT-4o, Gemini, etc.) read the image directly; non-vision models receive an auto-generated text description.
- **Voice notes** — auto-downloaded as `.ogg`, transcribed via your configured STT provider (local faster-whisper, OpenAI/Nous, Groq, etc.), then handed to the agent as text.
- **Documents** — auto-downloaded. Small text-readable files (`.txt`, `.md`, `.json`, `.py`, `.csv`, etc.) up to 100KB get inlined into the agent's input so it can read them without a tool call. Larger files are cached locally for the agent's other tools to access.
- **Button taps** — when the user taps a button the bot sent earlier (clarify choice, command approval, slash-command confirm), the tap is routed directly to the right handler. Stale taps fall back to being treated as regular text input.
- **Reply context** — when the user replies to a previous bot message, the agent sees the original message as context.

### Outbound

- **Text** — markdown is auto-converted to WhatsApp's flavored syntax (`**bold**` → `*bold*`, `~~strike~~` → `~strike~`, headers → bold, `[link](url)` → `link (url)`). Long messages split at 4096 chars per chunk.
- **Images** — agent-generated images and local image files both supported, delivered as native photo attachments.
- **Voice messages** — text-to-speech output is converted via ffmpeg into the native WhatsApp voice-note bubble (green waveform). Without ffmpeg installed, falls back to an MP3 audio attachment. See "Voice messages" below.
- **Video / documents** — both supported, sent as native attachments.

### Interactive UX

When the agent invokes any of these flows, Hermes uses WhatsApp's native interactive messages — tap-to-answer buttons instead of "reply with the number" prompts:

- **`clarify` tool** — multi-choice questions render as quick-reply buttons (1–3 choices) or a tap-to-open list sheet (4+ choices). Picking "✏️ Other" lets the user type a free-form answer that the agent receives as the resolution.
- **Dangerous-command approvals** — when the agent's terminal/code execution hits a gated command, the user sees `✅ Approve` / `❌ Deny` buttons instead of needing to type `/approve` or `/deny`.
- **Slash-command confirmations** — privileged commands like `/reload-mcp` show `✅ Approve Once` / `🔒 Always` / `❌ Cancel` buttons.

All interactive prompts gracefully degrade to plain text if the buttons fail to render (e.g. on legacy WhatsApp clients).

### Read receipts and typing indicator

Hermes acknowledges inbound messages immediately:

- Your message shows **blue double-checkmarks** as soon as the gateway receives it.
- The bot's name in your WhatsApp chat shows **"typing…"** while the agent is preparing a reply.
- The typing indicator auto-dismisses when the bot's first response message arrives.

This makes it obvious when the bot has seen your message versus when it's still working on a response.

### Voice messages

WhatsApp distinguishes between a "voice note" (the green waveform bubble) and a generic audio file attachment. The difference is purely codec: voice notes need to be `audio/ogg` with `opus` encoding.

Hermes TTS produces MP3. Two paths:

- **With ffmpeg on PATH** (recommended) — outbound TTS is converted and arrives as a proper voice note. Install:
  - Windows: `winget install Gyan.FFmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: package manager
- **Without ffmpeg** — outbound TTS arrives as an MP3 audio attachment. Plays fine, just doesn't look like a voice note. A one-time warning fires in the gateway log so you know.

You can check whether the gateway found ffmpeg via the health endpoint:

```bash
curl http://localhost:8090/health
# look for "ffmpeg_present": true
```

---

## Known limitations

### 24-hour conversation window

Meta only allows **free-form messages** within a 24-hour window after the user's last inbound message. Outside that window, the only thing Meta's API accepts is a pre-approved **message template**.

**What this means in practice:**

- Reactive chat (user DMs → bot replies within 24h → user replies → ...) works forever. This covers >95% of normal bot use.
- **Cron jobs that deliver to WhatsApp** after a gap > 24h will fail with Graph error code `131047` ("Re-engagement message").
- **Long-running `delegate_task` async results** that take longer than 24h fail the same way.
- **Webhook subscribers** that route external events to WhatsApp fail when the user hasn't DM'd the bot recently.

Hermes warns the agent about this window in its system prompt, so the model knows to mention it when scheduling delayed messages.

Message-template support (the workaround for outside-window sends) is not yet implemented in Hermes. If you need it, please [open an issue](https://github.com/NousResearch/hermes-agent/issues) — it's planned but waiting on a clear demand signal.

### Group chats

The Cloud API has limited group support (capability-tier gated by Meta).  Hermes's `whatsapp_cloud` adapter currently handles **direct messages only** in v1.  If you need group chats, use the Baileys bridge.

### Outbound rate limit

Meta's default throughput is **80 messages/second per business phone number**, with upgrades available.  Hermes doesn't currently enforce this client-side — extremely high-volume sends could hit Meta's limit.

---

## Troubleshooting

### Setup verification fails ("URL couldn't be validated") in Meta dashboard

Almost always one of:

- **Tunnel URL is wrong or stale** — cloudflared quick tunnels rotate.  Get a fresh URL and update both `.env` and Meta's dashboard.
- **Verify token mismatch** — the token in `~/.hermes/.env`'s `WHATSAPP_CLOUD_VERIFY_TOKEN` must match exactly what you typed into Meta's dashboard.  Run the curl probe above to confirm the gateway's verify handshake works locally first.
- **Gateway not running** — check `hermes gateway` is up.
- **App Secret not set** — without it, Hermes refuses inbound POSTs with 503.  Meta interprets that as "can't validate."

### `graph error 100`: Object with ID '...' does not exist

You pasted your phone number (10-11 digits) into `WHATSAPP_CLOUD_PHONE_NUMBER_ID` instead of the Phone Number ID (Meta's 15-17 digit internal ID).  Re-check the API Setup page — the Phone Number ID is shown *below* the "From" dropdown.

The wizard catches this with a validator now, but it's worth knowing if you're configuring manually.

### `graph error 190`: Authentication Error

Your access token is invalid.  Subcodes:

- `subcode 463` — token expired.  Temp tokens last 24h.  Regenerate, or switch to a System User permanent token (see above).
- `subcode 467` — token invalidated (revoked or password changed).
- Other 190 — token didn't have the required permissions when generated.  Make sure all three (`business_management`, `whatsapp_business_messaging`, `whatsapp_business_management`) were selected.

### `graph error 131047`: Re-engagement message

The 24-hour conversation window expired (see "Known limitations").  Either:

- Ask the user to DM the bot first to reopen the window.
- Wait for template support to land in Hermes.

### Inbound message: `media metadata fetch failed (status=401)`

Same 401 root causes as outbound (`graph error 190`) — the access token is invalid or expired.  Fix the token.

### Bot replies appear as raw JSON / tool-call leakage

Common cause: the toolset configured for `whatsapp_cloud` is missing the tools the agent wants to call.  Check `hermes tools list` and verify the platform is using `hermes-whatsapp` (the default Cloud adapter toolset, same as Baileys).

If the model emits tool-call-shaped text instead of a structured call, it usually means the toolset was effectively empty.  See `hermes_cli/platforms.py` for the platform → default toolset mapping.

### STT (voice note transcription) returns empty / "could not transcribe"

The default `stt.provider: local` requires `pip install faster-whisper`.  If you're a Nous subscriber, you can route STT through Meta's managed audio gateway instead:

```bash
hermes config set stt.provider openai
hermes config set stt.use_gateway true
hermes gateway restart
```

This uses your Nous Portal access token instead of needing a separate OpenAI key.

---

## Security notes

- **Treat the App Secret like a password** — anyone with it can forge webhook payloads that Hermes will accept as authentic.
- **The verify token is a shared secret** — leaks are lower-stakes (worst case someone could re-subscribe Meta's webhook to a different URL of theirs), but still avoid committing it.
- **The access token is your bot's identity** — System User tokens are equivalent to long-lived API keys.  Rotate immediately if a deployment is compromised.
- **The webhook endpoint accepts only signed requests when `WHATSAPP_CLOUD_APP_SECRET` is set** — leave it set even in development.  Without it, the gateway refuses inbound delivery with HTTP 503.
- **The `/health` endpoint is unauthenticated** — it's safe to expose because it only reports config-presence booleans, not the values themselves.  But if you'd rather not surface it, restrict access at the reverse proxy / tunnel layer.

---

## Comparison to the Baileys bridge

| | Baileys (`hermes whatsapp`) | Cloud API (`hermes whatsapp-cloud`) |
|---|---|---|
| Account type | Personal | Business |
| Setup | QR code scan | Meta app + WABA + token |
| Dependencies | Node.js + npm | Pure Python (httpx + aiohttp) |
| Process | Managed Node subprocess | aiohttp webhook server |
| Public URL needed? | No | Yes |
| Account ban risk | Yes (unofficial API) | No (officially supported) |
| Inbound | Polling Node bridge | Webhook POST from Meta |
| Outbound | Local bridge → Baileys | HTTPS to graph.facebook.com |
| Groups | Full support | DMs only (v1) |
| 24h window | No restriction | Hard rule — templates required after |
| Voice notes (out) | Native | Native with ffmpeg, MP3 fallback otherwise |
| Read receipts | No | Yes (blue double-checkmarks) |
| Typing indicator | No | Yes (auto-dismisses on response) |
| Interactive buttons | Text fallback only | Native (clarify, approval, slash-confirm) |
| Production use | Risky (Meta can ban) | Designed for it |

Most users running Hermes for personal projects prefer Baileys. Most users running customer-facing bots prefer Cloud API.

---

## See also

- [Meta's official WhatsApp Business Cloud API docs](https://developers.facebook.com/documentation/business-messaging/whatsapp/) — authoritative reference for the underlying platform, pricing, App Review, and Meta-side rate limits.
- [WhatsApp (Baileys bridge) Setup](whatsapp.md) — the alternative integration for personal projects.
- [Messaging Platforms overview](index.md) — all messaging integrations at a glance.
