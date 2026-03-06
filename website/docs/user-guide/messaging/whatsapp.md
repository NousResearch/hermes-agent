---
sidebar_position: 5
title: "WhatsApp"
description: "Set up Hermes Agent as a WhatsApp bot via the built-in Baileys bridge"
---

# WhatsApp Setup

WhatsApp doesn't have a simple bot API like Telegram or Discord. Hermes includes a built-in bridge using [Baileys](https://github.com/WhiskeySockets/Baileys) that connects via WhatsApp Web.

## Two Modes

| Mode | How it works | Best for |
|------|-------------|----------|
| **Separate bot number** (recommended) | Dedicate a phone number to the bot. People message that number directly. | Clean UX, multiple users |
| **Personal self-chat** | Use your own WhatsApp. You message yourself to talk to the agent. | Quick setup, single user |

## Setup

```bash
hermes whatsapp
```

The wizard will:

1. Ask which mode you want
2. For **bot mode**: guide you through getting a second number
3. Configure the allowlist
4. Install bridge dependencies (Node.js required)
5. Display a QR code — scan from WhatsApp → Settings → Linked Devices → Link a Device
6. Exit once paired

## Getting a Second Number (Bot Mode)

| Option | Cost | Notes |
|--------|------|-------|
| WhatsApp Business app + dual-SIM | Free (if you have dual-SIM) | Install alongside personal WhatsApp, no second phone needed |
| Google Voice | Free (US only) | voice.google.com, verify WhatsApp via the Google Voice app |
| Prepaid SIM | $3-10/month | Any carrier; verify once, phone can go in a drawer on WiFi |

## Starting the Gateway

```bash
hermes gateway            # Foreground
hermes gateway install    # Or install as a system service
```

The gateway starts the WhatsApp bridge automatically using the saved session.

## Environment Variables

```bash
WHATSAPP_ENABLED=true
WHATSAPP_MODE=bot                      # "bot" or "self-chat"
WHATSAPP_ALLOWED_USERS=15551234567     # Comma-separated phone numbers with country code
```

## Important Notes

- Agent responses are prefixed with "⚕ **Hermes Agent**" for easy identification
- WhatsApp Web sessions can disconnect if WhatsApp updates their protocol
- The gateway reconnects automatically
- If you see persistent failures, re-pair with `hermes whatsapp`

:::info Re-pairing
If WhatsApp Web sessions disconnect (protocol updates, phone reset), re-pair with `hermes whatsapp`. The gateway handles temporary disconnections automatically.
:::

## Voice Messages

Voice messages sent on WhatsApp are automatically transcribed (requires `VOICE_TOOLS_OPENAI_KEY`). TTS audio is sent as MP3 file attachments.

## Security

:::warning
Always set `WHATSAPP_ALLOWED_USERS` with phone numbers (including country code) to restrict who can use the bot.
:::
