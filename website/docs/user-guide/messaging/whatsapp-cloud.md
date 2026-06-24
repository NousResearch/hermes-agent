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
4. **Optional for MP3/WAV providers**: ffmpeg on `PATH` lets Hermes convert non-Opus TTS output into native WhatsApp voice-note bubbles (green waveform). Command providers that emit Ogg/Opus directly do not need ffmpeg for outbound voice messages.

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
The free quick-tunnel URL changes every time you restart `cloudflared`.  For a stable URL, log in with `cloudflared tunnel login` and create a named tunnel.  Free Cloudflare accounts get unlimited named tunnels — see [Cloudflare's docs](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) for the named-tunnel workflow.
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
| `WHATSAPP_CLOUD_CALLING_SIDECAR_URL` | — | Optional loopback URL for an experimental WhatsApp Calling / WebRTC sidecar. |
| `WHATSAPP_CLOUD_CALLING_SIDECAR_TIMEOUT` | `10.0` | HTTP timeout in seconds for sidecar control-plane requests. |
| `WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND` | — | Optional command that writes raw `pcm_s16le` TTS audio to stdout for live calls. |
| `WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_TIMEOUT` | `180.0` | Timeout in seconds for the live-call TTS stream command. |

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
- **Voice messages** — text-to-speech output can be sent directly as Ogg/Opus for native WhatsApp voice-note bubbles (green waveform). MP3/WAV output uses ffmpeg conversion when available and otherwise falls back to a regular audio attachment. See "Voice messages" below.
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

Hermes can use either direct Ogg/Opus output or a conversion fallback:

- **Direct Ogg/Opus** (preferred) — command TTS providers can produce WhatsApp-ready `.ogg` output directly. Set `output_format: ogg` and `voice_compatible: true`; Hermes uploads the file as `audio/ogg; codecs=opus` without ffmpeg conversion.
- **With ffmpeg on PATH** — MP3/WAV outputs are converted and arrive as proper voice notes.
- **Without ffmpeg** — MP3/WAV outputs arrive as generic audio attachments. They play fine, but do not render as voice-note bubbles. A one-time warning fires in the gateway log so you know.

For a local `voice` / Kokoro command provider:

```yaml
tts:
  provider: kokoro
  providers:
    kokoro:
      type: command
      command: /home/you/.local/bin/voice say --format ogg-opus --input-file {input_path} --output {output_path} --voice {voice} --speed {speed}
      output_format: ogg
      voice_compatible: true
      voice: af_heart
      speed: 1.0
      timeout: 180
```

Validate a live Hermes home before restarting the gateway:

```bash
scripts/verify_voice_command_tts.py --use-existing-config --hermes-home ~/.hermes
```

A passing run proves the configured provider returns a `[[audio_as_voice]]`
media tag and real mono 48 kHz Ogg/Opus audio suitable for WhatsApp voice-note
delivery.

For a single local preflight that uses an isolated Hermes home and leaves the
live gateway untouched, run:

```bash
scripts/verify_voice_local_stack.py \
  --voice-bin /path/to/voice \
  --voice-repo /path/to/voice
```

That aggregate check starts the local Hermes CLI, verifies command-provider
Ogg/Opus output, runs the `voice` checkout's own WhatsApp contract verifier,
verifies that the local Baileys bridge keeps `.ogg` / `.opus` audio as a
native `audio/ogg; codecs=opus` voice note, verifies that the Cloud API adapter
uploads direct and converted Ogg/Opus with that same MIME while falling back to
MP3 only when conversion is unavailable, verifies the raw `voice stream` PCM
contract, and then runs the WhatsApp Calling control-plane smoke plus the
full-duplex sidecar smoke from the `voice` checkout. The voice contract step
proves `voice stream-contract`, direct Ogg/Opus voice-note output, extension
inference, misleading-extension rejection, raw PCM daemon streaming, and
streamed Ogg/Opus agree before Hermes adds its own routing. The control-plane
smoke is synthetic and local: it feeds Hermes a representative Meta `calls`
webhook and verifies the sidecar offer, Graph `pre_accept`, Graph `accept`,
drain startup, and terminate cleanup without contacting Meta. Pass
`--skip-whatsapp-bridge-media`, `--skip-whatsapp-cloud-voice`,
`--skip-calling-control-plane`, or `--skip-full-duplex` when you only want part
of the preflight.

After installing a local gateway service, verify the running process is using
the expected voice-native checkout:

```bash
scripts/verify_voice_live_gateway.py \
  --live-hermes-root /path/to/hermes-agent \
  --python-bin ~/.hermes/hermes-agent/venv/bin/python \
  --hermes-home ~/.hermes \
  --voice-bin /path/to/voice \
  --voice-repo /path/to/voice \
  --calling-sidecar-url http://127.0.0.1:8787 \
  --sidecar-service voice-webrtc-sidecar.service \
  --skip-bridge-health \
  --run-tts-smoke
```

When `--voice-bin` is passed with `--calling-sidecar-url`, the live verifier
also compares the running sidecar `/contract` with `voice stream-contract` so
PCM shape, endpoint paths, payload definitions, and advertised `voice`
surfaces cannot drift silently. When `--sidecar-service` is included, it also
checks the sidecar systemd user unit is active and points at the expected
`VOICE_BIN`, `WorkingDirectory`, `examples/webrtc-sidecar/sidecar.py`, and
sidecar URL host/port.

That live check inspects the systemd user service, verifies the running process
environment, confirms imports resolve from the expected checkout, validates the
configured Calling sidecar `/contract` and `/health`, and can run one
live-config TTS smoke that must produce mono 48 kHz Ogg/Opus. Omit
`--skip-bridge-health` when the Baileys WhatsApp bridge is also enabled and you
want the verifier to require the local bridge `/health` endpoint to be
connected.

You can check whether the gateway found ffmpeg via the health endpoint:

```bash
curl http://localhost:8090/health
# look for "ffmpeg_present": true
```

### WhatsApp Calling / WebRTC sidecar (experimental)

The Cloud API's live calling surface is a WebRTC media path, not a file-upload path. Hermes keeps that boundary separate from the normal Graph API adapter:

1. Meta sends a `calls` webhook with an SDP offer.
2. Hermes forwards the offer to a local sidecar at `WHATSAPP_CLOUD_CALLING_SIDECAR_URL`.
3. The sidecar returns an SDP answer and exposes a fixed PCM contract.
4. Hermes sends `pre_accept` and `accept` to Graph with that SDP answer.
5. The sidecar bridges WebRTC RTP audio to local 48 kHz, mono, 20 ms `pcm_s16le` frames.
6. Hermes drains inbound PCM from the sidecar, writes utterance WAV segments for the existing STT path, and sends outbound TTS frames back to the sidecar.

The sidecar contract is identified as `voice.webrtc_sidecar`. Hermes fetches `GET /contract` when available and uses its endpoint paths and audio fields; older sidecars can omit `/contract` and use the legacy `/offer`, `/calls/{call_id}/audio`, and `/calls/{call_id}/close` paths. When a newer sidecar advertises `clear_audio`, Hermes calls that endpoint at the start of a detected inbound speech segment so barge-in can discard stale queued outbound TTS without closing the call.

Minimal config:

```bash
WHATSAPP_CLOUD_CALLING_SIDECAR_URL=http://127.0.0.1:8787
```

For low-latency outbound speech, add a command that writes headerless `pcm_s16le` to stdout. With the `voice` daemon running, this uses `voice stream` directly instead of generating a file and decoding it with ffmpeg:

```bash
WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND='voice stream --quiet --sample-rate {sample_rate} --frame-ms {frame_ms} --raw-output - --input-file {input_path}'
WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_TIMEOUT=180
```

Hermes posts that stdout PCM to the sidecar on the negotiated `frame_ms`
cadence, so a fast TTS process does not build a large outbound playback queue.

For a local Linux user-service deployment, let Hermes generate the sidecar unit
and gateway drop-in first. The default is a dry run that prints the files and
config it would write:

```bash
scripts/install_voice_local_stack.py \
  --live-hermes-root /path/to/hermes-agent \
  --voice-repo /path/to/voice \
  --voice-bin /path/to/voice/target/release/voice \
  --webrtc-python-bin /tmp/voice-webrtc-venv/bin/python \
  --configure-tts \
  --configure-stt
```

When the JSON plan looks right, apply it and restart the gateway:

```bash
scripts/install_voice_local_stack.py \
  --apply \
  --configure-tts \
  --configure-stt \
  --restart-hermes \
  --live-hermes-root /path/to/hermes-agent \
  --voice-repo /path/to/voice \
  --voice-bin /path/to/voice/target/release/voice \
  --webrtc-python-bin /tmp/voice-webrtc-venv/bin/python
```

That writes `~/.config/systemd/user/voice-webrtc-sidecar.service`, writes
`~/.config/systemd/user/hermes-gateway.service.d/voice-stack.conf`, updates
`~/.hermes/config.yaml` so the Kokoro command provider emits Ogg/Opus through
`voice say --format ogg-opus`, reloads systemd, starts the sidecar, and restarts
`hermes-gateway.service`. With `--configure-stt`, it also sets
`stt.provider: voice` with a command provider that runs
`voice stream-transcribe --quiet {input_path}` for inbound voice messages and
live-call WAV segments.

The JSON plan also includes `verify_commands.local_stack`,
`verify_commands.live_gateway`, `verify_commands.live_gateway_cloud_only`, and
`verify_commands.live_gateway_cloud_ready`. Run the emitted `local_stack`
command before changing the live service; it includes an isolated Cloud webhook
smoke that starts the adapter locally, checks `/health`, verifies the
subscription challenge, accepts a signed status-only POST, and accepts a signed
synthetic audio webhook that caches an Opus/Ogg voice note for downstream STT
dispatch without contacting Meta. Then run `live_gateway` after restart. Use
the `cloud_only` variant when this host does not run the local Baileys bridge.
Use the `cloud_ready` variant before routing real Meta webhooks to the host; it
checks WhatsApp Cloud credential shape and recipient authorization, then probes
the running local Cloud `/health` endpoint and local Meta subscription challenge
handshake. It also sends a signed synthetic delivery-receipt POST to prove
inbound HMAC verification accepts Meta-shaped webhook delivery without
dispatching an agent message or printing secret values. If setup is incomplete,
it reports every missing or malformed Cloud setting it can detect in one
redacted failure.

The generated service files should look like this:

```ini
# ~/.config/systemd/user/voice-webrtc-sidecar.service
[Unit]
Description=Voice WebRTC Sidecar
After=network.target voice-daemon.service

[Service]
Type=simple
WorkingDirectory=/path/to/voice
Environment="VOICE_BIN=/path/to/voice/target/release/voice"
ExecStart=/tmp/voice-webrtc-venv/bin/python /path/to/voice/examples/webrtc-sidecar/sidecar.py --host 127.0.0.1 --port 8787 --rx-pcm /home/you/.hermes/voice-webrtc-sidecar/inbound.s16le --log-level INFO
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
```

```ini
# ~/.config/systemd/user/hermes-gateway.service.d/voice-stack.conf
[Service]
Environment="PYTHONPATH=/path/to/hermes-agent"
Environment="WHATSAPP_CLOUD_CALLING_SIDECAR_URL=http://127.0.0.1:8787"
Environment="WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND=/path/to/voice/target/release/voice stream --quiet --sample-rate {sample_rate} --frame-ms {frame_ms} --raw-output - --input-file {input_path} --voice af_heart --speed 1.0"
Environment="WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_TIMEOUT=180"
```

Validate the installed command shape before a live call:

```bash
scripts/verify_voice_stream_tts.py --voice-bin /path/to/voice
```

The verifier reads `/path/to/voice stream-contract`, validates the advertised
`streamed_voice_note`, `raw_outbound_pcm`, and `raw_inbound_pcm`
`voice_surfaces`, then runs the rendered stream command and checks stdout for
non-silent 48 kHz mono 20 ms `pcm_s16le` frames. It also runs
`voice stream --output ... --format ogg-opus` and verifies the streamed file is
real mono 48 kHz Ogg/Opus, so the low-latency daemon frame path and completed
voice-note path are both covered without a WAV intermediate.

Then validate the local WebRTC media bridge from a `voice` checkout before
involving Meta's Graph API:

```bash
cd /path/to/voice
python3 -m venv /tmp/voice-webrtc-venv
/tmp/voice-webrtc-venv/bin/pip install -r examples/webrtc-sidecar/requirements.txt pytest

cd /path/to/hermes-agent
VOICE_REPO=/path/to/voice \
VOICE_WEBRTC_PYTHON=/tmp/voice-webrtc-venv/bin/python \
  scripts/verify_voice_full_duplex_sidecar.py --voice-bin /path/to/voice
```

That wrapper runs `examples/webrtc-sidecar/full_duplex_loopback_smoke.py` from
the `voice` checkout. The full-duplex smoke keeps everything local: it sends
one `voice stream` utterance through a local WebRTC peer into the sidecar,
drains the decoded PCM through `voice stream-transcribe`, and simultaneously
queues outbound `voice stream` PCM back to the same peer. A passing run proves
the sidecar, PCM contract, inbound STT bridge, and outbound TTS bridge agree
before a real WhatsApp call is attempted. It also enforces a default one-second
outbound sidecar queue budget; adjust it with `--max-queued-tx-ms` on the
wrapper or `--full-duplex-max-queued-tx-ms` on the aggregate verifier. Current
sidecars report queue depth as both bytes and whole milliseconds
(`queued_tx_bytes`/`queued_tx_ms`, `queued_rx_bytes`/`queued_rx_ms`) so Hermes
can enforce latency budgets without duplicating PCM duration math. Hermes
accepts older sidecars that omit those fields, but rejects malformed telemetry
when a sidecar does report it.

To validate just the Cloud Calling control plane without sidecar media
dependencies, run:

```bash
scripts/verify_voice_whatsapp_calling_control_plane.py
```

That synthetic smoke proves Hermes turns a representative call offer into a
sidecar SDP request, Graph `pre_accept`, Graph `accept`, and local sidecar close
on termination. It does not contact Meta and does not prove RTP media flow; use
the full-duplex sidecar smoke above for the media path.

Supported stream-command placeholders:

| Placeholder | Meaning |
|---|---|
| `{input_path}` / `{text_path}` | Temp UTF-8 file containing the reply text |
| `{text}` | Reply text itself, shell-quoted for the command context |
| `{sample_rate}` | Sidecar PCM sample rate, currently `48000` |
| `{channels}` | Sidecar PCM channels, currently `1` |
| `{frame_ms}` | Sidecar frame duration, currently `20` |
| `{encoding}` | Sidecar PCM encoding, currently `pcm_s16le` |

The health endpoint reports `calling_sidecar_configured`, `calling_sidecar_contract_loaded`, and `calling_sidecar_tts_stream_configured` booleans so you can confirm the live-call path is wired.

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
| Voice notes (out) | Native with Ogg/Opus, ffmpeg fallback for MP3/WAV | Native with Ogg/Opus, ffmpeg fallback for MP3/WAV |
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
