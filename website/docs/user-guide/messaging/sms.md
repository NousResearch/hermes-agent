---
sidebar_position: 8
title: "SMS (Twilio)"
description: "Set up Hermes Agent as an SMS chatbot via Twilio"
---

# SMS Setup (Twilio)

Hermes connects to SMS through the [Twilio](https://www.twilio.com/) API. People text your Twilio phone number and get AI responses back — same conversational experience as Telegram or Discord, but over standard text messages.

:::info Shared Credentials
The SMS gateway shares credentials with the optional [telephony skill](/docs/reference/skills-catalog). If you've already set up Twilio for voice calls or one-off SMS, the gateway works with the same `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, and `TWILIO_PHONE_NUMBER`.
:::

---

## Prerequisites

- **Twilio account** — [Sign up at twilio.com](https://www.twilio.com/try-twilio) (free trial available)
- **A Twilio phone number** with SMS capability
- **A publicly accessible server** — Twilio sends webhooks to your server when SMS arrives
- **aiohttp** — `pip install 'hermes-agent[sms]'`

---

## Step 1: Get Your Twilio Credentials

1. Go to the [Twilio Console](https://console.twilio.com/)
2. Copy your **Account SID** and **Auth Token** from the dashboard
3. Go to **Phone Numbers → Manage → Active Numbers** — note your phone number in E.164 format (e.g., `+15551234567`)

---

## Step 2: Configure Hermes

### Interactive setup (recommended)

```bash
hermes gateway setup
```

Select **SMS (Twilio)** from the platform list. The wizard will prompt for your credentials.

### Manual setup

Add to `~/.hermes/.env`:

```bash
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+15551234567

# Security: restrict to specific phone numbers (recommended)
SMS_ALLOWED_USERS=+15559876543,+15551112222

# Optional: set a home channel for cron job delivery
SMS_HOME_CHANNEL=+15559876543
```

---

## Step 3: Configure Twilio Webhook

Twilio needs to know where to send incoming messages. In the [Twilio Console](https://console.twilio.com/):

1. Go to **Phone Numbers → Manage → Active Numbers**
2. Click your phone number
3. Under **Messaging → A MESSAGE COMES IN**, set:
   - **Webhook**: `https://your-server:8080/webhooks/twilio`
   - **HTTP Method**: `POST`

:::tip Exposing Your Webhook
If you're running Hermes locally, use a tunnel to expose the webhook:

```bash
# Using cloudflared
cloudflared tunnel --url http://localhost:8080

# Using ngrok
ngrok http 8080
```

Set the resulting public URL as your Twilio webhook.
:::

The webhook port defaults to `8080`. Override with:

```bash
SMS_WEBHOOK_PORT=3000
```

---

## Step 4: Start the Gateway

```bash
hermes gateway
```

You should see:

```
[sms] Twilio webhook server listening on port 8080, from: +1555***4567
```

Text your Twilio number — Hermes will respond via SMS.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TWILIO_ACCOUNT_SID` | Yes | Twilio Account SID (starts with `AC`) |
| `TWILIO_AUTH_TOKEN` | Yes | Twilio Auth Token |
| `TWILIO_PHONE_NUMBER` | Yes | Your Twilio phone number (E.164 format) |
| `SMS_WEBHOOK_PORT` | No | Webhook listener port (default: `8080`) |
| `SMS_ALLOWED_USERS` | No | Comma-separated E.164 phone numbers allowed to chat |
| `SMS_ALLOW_ALL_USERS` | No | Set to `true` to allow anyone (not recommended) |
| `SMS_HOME_CHANNEL` | No | Phone number for cron job / notification delivery |
| `SMS_HOME_CHANNEL_NAME` | No | Display name for the home channel (default: `Home`) |

---

## SMS-Specific Behavior

- **Plain text only** — Markdown is automatically stripped since SMS renders it as literal characters
- **1600 character limit** — Longer responses are split across multiple messages at natural boundaries (newlines, then spaces)
- **Echo prevention** — Messages from your own Twilio number are ignored to prevent loops
- **Phone number redaction** — Phone numbers are redacted in logs for privacy

---

## Security

**The gateway denies all users by default.** Configure an allowlist:

```bash
# Recommended: restrict to specific phone numbers
SMS_ALLOWED_USERS=+15559876543,+15551112222

# Or allow all (NOT recommended for bots with terminal access)
SMS_ALLOW_ALL_USERS=true
```

:::warning
SMS has no built-in encryption. Don't use SMS for sensitive operations unless you understand the security implications. For sensitive use cases, prefer Signal or Telegram.
:::

---

## Troubleshooting

### Messages not arriving

1. Check your Twilio webhook URL is correct and publicly accessible
2. Verify `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` are correct
3. Check the Twilio Console → **Monitor → Logs → Messaging** for delivery errors
4. Ensure your phone number is in `SMS_ALLOWED_USERS` (or `SMS_ALLOW_ALL_USERS=true`)

### Replies not sending

1. Check `TWILIO_PHONE_NUMBER` is set correctly (E.164 format with `+`)
2. Verify your Twilio account has SMS-capable numbers
3. Check Hermes gateway logs for Twilio API errors

### Webhook port conflicts

If port 8080 is already in use, change it:

```bash
SMS_WEBHOOK_PORT=3001
```

Update the webhook URL in Twilio Console to match.
