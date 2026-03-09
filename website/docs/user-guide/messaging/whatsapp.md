---
sidebar_position: 5
title: "WhatsApp"
description: "Set up Hermes Agent as a WhatsApp bot via the built-in Baileys bridge"
---

# WhatsApp Setup

Hermes connects to WhatsApp through a built-in bridge using [whatsapp-web.js](https://github.com/pedroslopez/whatsapp-web.js)
(Baileys-based). This works by emulating a WhatsApp Web session — **not** through the official
WhatsApp Business API. No Meta developer account or Business verification is required.

:::warning Unofficial API — Ban Risk
WhatsApp does **not** officially support third-party bots outside the Business API. Using
whatsapp-web.js carries a small risk of account restrictions. To minimize risk:
- **Use a dedicated phone number** for the bot (not your personal number)
- **Don't send bulk/spam messages** — keep usage conversational
- **Don't automate outbound messaging** to people who haven't messaged first
:::

:::warning WhatsApp Web Protocol Updates
WhatsApp periodically updates their Web protocol, which can temporarily break compatibility
with whatsapp-web.js. When this happens, Hermes will update the bridge dependency. If the
bot stops working after a WhatsApp update, pull the latest Hermes version and re-pair.
:::

## Two Modes

| Mode | How it works | Best for |
|------|-------------|----------|
| **Separate bot number** (recommended) | Dedicate a phone number to the bot. People message that number directly. | Clean UX, multiple users, lower ban risk |
| **Personal self-chat** | Use your own WhatsApp. You message yourself to talk to the agent. | Quick setup, single user, testing |

---

## Prerequisites

- **Node.js v18+** and **npm** — the WhatsApp bridge runs as a Node.js process
- **A phone with WhatsApp** installed (for scanning the QR code)

**On Linux headless servers**, you also need Chromium/Puppeteer dependencies:

```bash
# Debian / Ubuntu
sudo apt-get install -y \
  libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
  libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 \
  libpango-1.0-0 libcairo2 libasound2 libxshmfence1

# Fedora / RHEL
sudo dnf install -y \
  nss atk at-spi2-atk cups-libs libdrm libxkbcommon \
  libXcomposite libXdamage libXrandr mesa-libgbm \
  pango cairo alsa-lib
```

---

## Step 1: Run the Setup Wizard

```bash
hermes whatsapp
```

The wizard will:

1. Ask which mode you want (**bot** or **self-chat**)
2. Install bridge dependencies if needed
3. Display a **QR code** in your terminal
4. Wait for you to scan it

**To scan the QR code:**

1. Open WhatsApp on your phone
2. Go to **Settings → Linked Devices**
3. Tap **Link a Device**
4. Point your camera at the terminal QR code

Once paired, the wizard confirms the connection and exits. Your session is saved automatically.

:::tip
If the QR code looks garbled, make sure your terminal is at least 60 columns wide and supports
Unicode. You can also try a different terminal emulator.
:::

---

## Step 2: Getting a Second Phone Number (Bot Mode)

For bot mode, you need a phone number that isn't already registered with WhatsApp. Three options:

| Option | Cost | Notes |
|--------|------|-------|
| **Google Voice** | Free | US only. Get a number at [voice.google.com](https://voice.google.com). Verify WhatsApp via SMS through the Google Voice app. |
| **Prepaid SIM** | $5–15 one-time | Any carrier. Activate, verify WhatsApp, then the SIM can sit in a drawer. Number must stay active (make a call every 90 days). |
| **VoIP services** | Free–$5/month | TextNow, TextFree, or similar. Some VoIP numbers are blocked by WhatsApp — try a few if the first doesn't work. |

After getting the number:

1. Install WhatsApp on a phone (or use WhatsApp Business app with dual-SIM)
2. Register the new number with WhatsApp
3. Run `hermes whatsapp` and scan the QR code from that WhatsApp account

---

## Step 3: Configure Hermes

Add the following to your `~/.hermes/.env` file:

```bash
# Required
WHATSAPP_ENABLED=true
WHATSAPP_MODE=bot                          # "bot" or "self-chat"
WHATSAPP_ALLOWED_USERS=15551234567         # Comma-separated phone numbers (with country code, no +)

# Optional
WHATSAPP_HOME_CONTACT=15551234567          # Default contact for proactive/scheduled messages
```

Then start the gateway:

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a system service
```

The gateway starts the WhatsApp bridge automatically using the saved session.

---

## Session Persistence

The whatsapp-web.js `LocalAuth` strategy saves your session to the `.wwebjs_auth` folder inside
your Hermes data directory (`~/.hermes/`). This means:

- **Sessions survive restarts** — you don't need to re-scan the QR code every time
- The session data includes encryption keys and device credentials
- **Do not share or commit the `.wwebjs_auth` folder** — it grants full access to the WhatsApp account

---

## Re-pairing

If the session breaks (phone reset, WhatsApp update, manually unlinked), you'll see connection
errors in the gateway logs. To fix it:

```bash
hermes whatsapp
```

This generates a fresh QR code. Scan it again and the session is re-established. The gateway
handles **temporary** disconnections (network blips, phone going offline briefly) automatically
with reconnection logic.

---

## Voice Messages

Hermes supports voice on WhatsApp:

- **Incoming:** Voice messages (`.ogg` opus) are automatically transcribed using Whisper (requires `VOICE_TOOLS_OPENAI_KEY`)
- **Outgoing:** TTS responses are sent as MP3 audio file attachments
- Agent responses are prefixed with "⚕ **Hermes Agent**" for easy identification

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **QR code not scanning** | Ensure terminal is wide enough (60+ columns). Try a different terminal. Make sure you're scanning from the correct WhatsApp account (bot number, not personal). |
| **QR code expires** | QR codes refresh every ~20 seconds. If it times out, restart `hermes whatsapp`. |
| **Session not persisting** | Check that `~/.hermes/.wwebjs_auth/` exists and is writable. On Docker, mount this as a volume. |
| **Logged out unexpectedly** | WhatsApp unlinks devices after ~14 days of phone inactivity. Keep the phone on and connected to WiFi. Re-pair with `hermes whatsapp`. |
| **"Execution context was destroyed"** | Chromium crashed. Install the Puppeteer dependencies listed in Prerequisites. On low-RAM servers, add swap space. |
| **Bot stops working after WhatsApp update** | Update Hermes to get the latest bridge version, then re-pair. |
| **Messages not being received** | Verify `WHATSAPP_ALLOWED_USERS` includes the sender's number (with country code, no `+` or spaces). |

---

## Security

:::warning
**Always set `WHATSAPP_ALLOWED_USERS`** with phone numbers (including country code, without the `+`)
of authorized users. Without this setting, the gateway will **deny all incoming messages** as a
safety measure.
:::

- The `.wwebjs_auth` folder contains full session credentials — protect it like a password
- Set file permissions: `chmod 700 ~/.hermes/.wwebjs_auth`
- Use a **dedicated phone number** for the bot to isolate risk from your personal account
- If you suspect compromise, unlink the device from WhatsApp → Settings → Linked Devices
- Phone numbers in logs are partially redacted, but review your log retention policy
