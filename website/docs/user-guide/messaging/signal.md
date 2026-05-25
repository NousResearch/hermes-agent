---
sidebar_position: 6
title: "Signal"
description: "Set up Hermes Agent as a Signal messenger bot via signal-cli daemon"
---

# Signal Setup

Hermes connects to Signal through the [signal-cli](https://github.com/AsamK/signal-cli) daemon running in HTTP mode. The adapter streams messages in real-time via SSE (Server-Sent Events) and sends responses via JSON-RPC.

Signal is the most privacy-focused mainstream messenger — end-to-end encrypted by default, open-source protocol, minimal metadata collection. This makes it ideal for security-sensitive agent workflows.

:::info No New Python Dependencies
The Signal adapter uses `httpx` (already a core Hermes dependency) for all communication. No additional Python packages are required. You just need signal-cli installed externally.
:::

---

## Prerequisites

- **signal-cli** — Java-based Signal client ([GitHub](https://github.com/AsamK/signal-cli))
- **Java 25+** runtime — required by signal-cli 0.14.x (see [Java version](#java-version) below)
- **A phone number** with Signal installed (for linking as a secondary device)

:::danger Do not use `bbernhard/signal-cli-rest-api`
Most third-party Signal-on-Docker tutorials point at [`bbernhard/signal-cli-rest-api`](https://github.com/bbernhard/signal-cli-rest-api), which exposes `/v1/about`, `/v2/send`, `/v1/health`, etc. Hermes speaks to signal-cli's **native** daemon (`/api/v1/check`, `/api/v1/events` SSE, `/api/v1/rpc`) — the JSON-RPC body and SSE event schemas differ, so messages never deliver even when the health check looks green (symptom: `Signal SSE: connected` repeating every ~2 s in `gateway.log`). Install upstream `signal-cli` directly using the steps below.
:::

### Java version

Signal's servers reject all pre-0.14 clients (`StatusCode: 499 DeprecatedVersionException`), so signal-cli 0.14.x is the only supported line. 0.14.x is compiled with class-file version 69 and **requires Java 25 or newer** — running it on `openjdk-21-jre-headless` (Debian's current default) fails with:

```
UnsupportedClassVersionError: org/asamk/signal/Main has been compiled by a more recent
version of the Java Runtime (class file version 69.0), this version of the Java Runtime
only recognizes class file versions up to 65.0
```

Recipe for pinning Adoptium Temurin 25 (works on Debian/Ubuntu, including WSL2). signal-cli only needs a JRE; Adoptium ships the JRE under a directory named `jdk-25.<patch>+<build>-jre`, which the symlink glob below matches:

```bash
sudo mkdir -p /opt/java
curl -fsSL "https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jre/hotspot/normal/eclipse" \
  | sudo tar xz -C /opt/java
sudo ln -sf /opt/java/jdk-25*-jre /opt/java/current
export JAVA_HOME=/opt/java/current
export PATH="$JAVA_HOME/bin:$PATH"
```

If you run signal-cli under systemd, also pin `Environment=JAVA_HOME=/opt/java/current` in the unit file so the daemon does not fall back to the system JRE.

### Installing signal-cli

```bash
# macOS
brew install signal-cli

# Linux (download latest release)
VERSION=$(curl -Ls -o /dev/null -w %{url_effective} \
  https://github.com/AsamK/signal-cli/releases/latest | sed 's/^.*\/v//')
curl -L -O "https://github.com/AsamK/signal-cli/releases/download/v${VERSION}/signal-cli-${VERSION}.tar.gz"
sudo tar xf "signal-cli-${VERSION}.tar.gz" -C /opt
sudo ln -sf "/opt/signal-cli-${VERSION}/bin/signal-cli" /usr/local/bin/
```

:::caution
signal-cli is **not** in apt or snap repositories. The Linux install above downloads directly from [GitHub releases](https://github.com/AsamK/signal-cli/releases).
:::

---

## Step 1: Link Your Signal Account

Signal-cli works as a **linked device** — like WhatsApp Web, but for Signal. Your phone stays the primary device.

```bash
# Generate a linking URI (displays a QR code or link)
signal-cli link -n "HermesAgent"
```

1. Open **Signal** on your phone
2. Go to **Settings → Linked Devices**
3. Tap **Link New Device**
4. Scan the QR code or enter the URI

---

## Step 2: Start the signal-cli Daemon

```bash
# Replace +1234567890 with your Signal phone number (E.164 format)
signal-cli --account +1234567890 daemon --http 127.0.0.1:8080
```

:::tip
Keep this running in the background. You can use `systemd`, `tmux`, `screen`, or run it as a service.
:::

Verify it's running:

```bash
curl http://127.0.0.1:8080/api/v1/check
# Should return: {"versions":{"signal-cli":...}}
```

---

## Step 3: Configure Hermes

The easiest way:

```bash
hermes gateway setup
```

Select **Signal** from the platform menu. The wizard will:

1. Check if signal-cli is installed
2. Prompt for the HTTP URL (default: `http://127.0.0.1:8080`)
3. Test connectivity to the daemon
4. Ask for your account phone number
5. Configure allowed users and access policies

### Manual Configuration

Add to `~/.hermes/.env`:

```bash
# Required
SIGNAL_HTTP_URL=http://127.0.0.1:8080
SIGNAL_ACCOUNT=+1234567890

# Security (recommended) — see "Allowlist with Phone Number Privacy" below
SIGNAL_ALLOWED_USERS=+1234567890,+0987654321    # Comma-separated E.164 numbers or ACI UUIDs

# Optional
SIGNAL_GROUP_ALLOWED_USERS=groupId1,groupId2     # Enable groups (omit to disable, * for all)
SIGNAL_HOME_CHANNEL=+1234567890                  # Default delivery target for cron jobs
```

Then start the gateway:

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

---

## Access Control

### DM Access

DM access follows the same pattern as all other Hermes platforms:

1. **`SIGNAL_ALLOWED_USERS` set** → only those users can message
2. **No allowlist set** → unknown users get a DM pairing code (approve via `hermes pairing approve signal CODE`)
3. **`SIGNAL_ALLOW_ALL_USERS=true`** → anyone can message (use with caution)

### Allowlist with Phone Number Privacy

Signal has enabled [Phone Number Privacy](https://signal.org/blog/phone-number-privacy-usernames/) by default since 2023. With it on, inbound messages arrive with the sender's **ACI UUID** rather than their E.164 phone number, so an allowlist that only lists phone numbers will silently reject every real DM:

```
WARNING gateway.run: Unauthorized user: 00000000-aaaa-bbbb-cccc-000000000000 (DisplayName) on signal
```

The adapter compares the inbound identifier literally against `SIGNAL_ALLOWED_USERS`, so the fix is to put **both** identifiers in the list:

```bash
SIGNAL_ALLOWED_USERS=+15551234567,00000000-aaaa-bbbb-cccc-000000000000
```

The UUID for any user is logged in the `Unauthorized user:` warning the first time they message you — copy it from there into `SIGNAL_ALLOWED_USERS` and restart the gateway. The phone number form is still required for outbound delivery targets such as `SIGNAL_HOME_CHANNEL`.

### Group Access

Group access is controlled by the `SIGNAL_GROUP_ALLOWED_USERS` env var:

| Configuration | Behavior |
|---------------|----------|
| Not set (default) | All group messages are ignored. The bot only responds to DMs. |
| Set with group IDs | Only listed groups are monitored (e.g., `groupId1,groupId2`). |
| Set to `*` | The bot responds in any group it's a member of. |

---

## Features

### Attachments

The adapter supports sending and receiving media in both directions.

**Incoming** (user → agent):

- **Images** — PNG, JPEG, GIF, WebP (auto-detected via magic bytes)
- **Audio** — MP3, OGG, WAV, M4A (voice messages transcribed if Whisper is configured)
- **Documents** — PDF, ZIP, and other file types

**Outgoing** (agent → user):

The agent can send media files via `MEDIA:` tags in responses. The following delivery methods are supported:

- **Images** — `send_multiple_images` and `send_image_file` send PNG, JPEG, GIF, WebP as native Signal attachments
- **Voice** — `send_voice` sends audio files (OGG, MP3, WAV, M4A, AAC) as attachments
- **Video** — `send_video` sends MP4 video files
- **Documents** — `send_document` sends any file type (PDF, ZIP, etc.)

All outgoing media goes through Signal's standard attachment API. Unlike some platforms, Signal does not distinguish between voice messages and file attachments at the protocol level.

Attachment size limit: **100 MB** (both directions).
:::warning
**Signal servers will rate-limit attachment uploads**, the adapter uses a scheduler for multiple image sending that batches images in groups of 32 and throttles uploads to match the Signal server policy.
:::

### Native Formatting, Reply Quotes, and Reactions

Signal messages render with **native formatting** instead of literal markdown characters. The adapter converts markdown (`**bold**`, `*italic*`, `` `code` ``, `~~strike~~`, `||spoiler||`, headings) into Signal `bodyRanges` so the text shows up with real styling on the recipient's client rather than as visible `**` / `` ` `` characters.

**Reply quotes.** When Hermes replies to a specific message, it now posts a native reply that quotes the original — same UI affordance Signal users see when they use "Reply" themselves. This is automatic for replies generated in response to an inbound message.

**Reactions.** The agent can react to messages via the standard reaction API; reactions surface in Signal as emoji reactions on the referenced message rather than as extra text.

None of this requires additional config — it ships on by default in recent signal-cli builds. If your `signal-cli` version is too old, Hermes falls back to plaintext delivery and logs a one-time warning.

### Typing Indicators

The bot sends typing indicators while processing messages, refreshing every 8 seconds.

### Phone Number Redaction

All phone numbers are automatically redacted in logs:
- `+15551234567` → `+155****4567`
- This applies to both Hermes gateway logs and the global redaction system

### Note to Self (Single-Number Setup)

If you run signal-cli as a **linked secondary device** on your own phone number (rather than a separate bot number), you can interact with Hermes through Signal's "Note to Self" feature.

Just send a message to yourself from your phone — signal-cli picks it up and Hermes responds in the same conversation.

**How it works:**
- "Note to Self" messages arrive as `syncMessage.sentMessage` envelopes
- The adapter detects when these are addressed to the bot's own account and processes them as regular inbound messages
- Echo-back protection (sent-timestamp tracking) prevents infinite loops — the bot's own replies are filtered out automatically

**No extra configuration needed.** This works automatically as long as `SIGNAL_ACCOUNT` matches your phone number.

### Health Monitoring

The adapter monitors the SSE connection and automatically reconnects if:
- The connection drops (with exponential backoff: 2s → 60s)
- No activity is detected for 120 seconds (pings signal-cli to verify)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Cannot reach signal-cli"** during setup | Ensure signal-cli daemon is running: `signal-cli --account +YOUR_NUMBER daemon --http 127.0.0.1:8080` |
| **Messages not received** | Check that `SIGNAL_ALLOWED_USERS` includes the sender's **ACI UUID** (see [Allowlist with Phone Number Privacy](#allowlist-with-phone-number-privacy)) — phone-number-only allowlists silently reject real DMs |
| **`UnsupportedClassVersionError: ... class file version 69.0`** | signal-cli 0.14.x requires Java 25+; see [Java version](#java-version) — `openjdk-21-jre-headless` is too old |
| **`Verify error: StatusCode: 499 DeprecatedVersionException`** | `signal-cli < 0.14` is server-rejected — upgrade to 0.14.x and pin Java 25+ |
| **`Signal SSE: connected` repeating every ~2 s with no messages delivering** | You are pointed at `bbernhard/signal-cli-rest-api` instead of the native daemon. Switch to upstream `signal-cli --http` (see warning under [Prerequisites](#prerequisites)). |
| **`signal-cli register` returns 409 `AlreadyVerifiedException`** | See [Recovering from a stuck registration](#recovering-from-a-stuck-registration) below — `--reregister` and `verify` cannot fix this on their own |
| **"signal-cli not found on PATH"** | Install signal-cli and ensure it's in your PATH (third-party Docker wrappers are not compatible — see warning under [Prerequisites](#prerequisites)) |
| **Connection keeps dropping** | Check signal-cli logs for errors. Ensure Java 25+ is installed. |
| **Group messages ignored** | Configure `SIGNAL_GROUP_ALLOWED_USERS` with specific group IDs, or `*` to allow all groups. |
| **Bot responds to no one** | Configure `SIGNAL_ALLOWED_USERS`, use DM pairing, or explicitly allow all users through gateway policy if you want broader access. |
| **Duplicate messages** | Ensure only one signal-cli instance is listening on your phone number |

### Recovering from a stuck registration

A `409 AlreadyVerifiedException` from `signal-cli register` means Signal's server already has the number in a verified-but-uninitialized state — typically from a previous interrupted captcha/verify flow, or from a prior install via a third-party REST wrapper. From `signal-cli` alone this is unrecoverable: `--reregister` hits the same endpoint and returns the same 409, and `signal-cli verify <code>` returns `No registration verification session active`.

Confirmed working recovery:

1. Put the SIM in a phone, install the **Signal mobile app**, and complete a normal registration as the primary device.
2. In the app, go to **Settings → Account → Delete Account** to clear the server-side registration state.
3. Re-run `signal-cli register --captcha "<captcha>"` — it now succeeds.

This is the only reliable path; advice to "wait 24 h" or "try a different number" is unreliable and the session lifetime is undocumented.

---

## Security

:::warning
**Always configure access controls.** The bot has terminal access by default. Without `SIGNAL_ALLOWED_USERS` or DM pairing, the gateway denies all incoming messages as a safety measure.
:::

- Phone numbers are redacted in all log output
- Use DM pairing or explicit allowlists for safe onboarding of new users
- Keep groups disabled unless you specifically need group support, or allowlist only the groups you trust
- Signal's end-to-end encryption protects message content in transit
- The signal-cli session data in `~/.local/share/signal-cli/` contains account credentials — protect it like a password

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SIGNAL_HTTP_URL` | Yes | — | signal-cli HTTP endpoint |
| `SIGNAL_ACCOUNT` | Yes | — | Bot phone number (E.164) |
| `SIGNAL_ALLOWED_USERS` | No | — | Comma-separated E.164 numbers and/or ACI UUIDs (UUIDs required to match real DMs under Phone Number Privacy — see [allowlist guidance](#allowlist-with-phone-number-privacy)) |
| `SIGNAL_GROUP_ALLOWED_USERS` | No | — | Group IDs to monitor, or `*` for all (omit to disable groups) |
| `SIGNAL_ALLOW_ALL_USERS` | No | `false` | Allow any user to interact (skip allowlist) |
| `SIGNAL_HOME_CHANNEL` | No | — | Default delivery target for cron jobs |
