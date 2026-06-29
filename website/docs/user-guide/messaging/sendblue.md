---
sidebar_label: "Sendblue"
title: "Sendblue"
description: "Set up Hermes Agent as an iMessage/SMS/RCS chatbot via Sendblue"
---

# Sendblue Setup

Hermes can connect to [Sendblue](https://www.sendblue.com/) as a bundled
platform plugin. Users text your Sendblue line, Sendblue POSTs inbound receive
webhooks to Hermes, and Hermes replies through Sendblue's REST API.

This path is useful when you want a managed iMessage/SMS/RCS channel without
running your own Mac relay.

> Run `hermes gateway setup` and pick **Sendblue** for a guided walk-through.

## Prerequisites

- A Sendblue account with API credentials.
- At least one Sendblue phone line in E.164 format.
- A publicly reachable HTTPS URL for the receive webhook.
- A receive webhook secret configured in Sendblue.

The adapter uses `aiohttp`, which is included with the Hermes messaging extra.

## Configure Hermes

### Via setup wizard

```bash
hermes gateway setup
```

Select **Sendblue** and follow the prompts.

### Via environment variables

Add these to `~/.hermes/.env`:

```bash
SENDBLUE_API_KEY_ID=your_api_key_id
SENDBLUE_API_SECRET_KEY=your_api_secret
SENDBLUE_FROM_NUMBER=+15551234567
SENDBLUE_WEBHOOK_SECRET=your_receive_webhook_secret
SENDBLUE_ALLOWED_USERS=+15559876543
SENDBLUE_HOME_CHANNEL=+15559876543
```

If you have multiple Sendblue lines, use a pool:

```bash
SENDBLUE_FROM_NUMBERS=+15551234567,+15557654321
```

Hermes keeps a sticky recipient -> `from_number` map so each contact continues
to see the same Sendblue line after the first reply.

## Configure Sendblue

In the Sendblue dashboard or Webhooks API, create a **receive** webhook that
points to:

```text
https://your-public-host/sendblue/webhook
```

Set the webhook secret to the same value as `SENDBLUE_WEBHOOK_SECRET`. Hermes
rejects inbound webhooks that do not include the configured secret. For local
development only, you can bypass this with:

```bash
SENDBLUE_INSECURE_NO_SIGNATURE=true
```

Do not use that bypass on a public server.

## Start the gateway

```bash
hermes gateway start
```

Send an iMessage or SMS to your Sendblue number. Hermes will respond in the
same thread.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `SENDBLUE_API_KEY_ID` | Yes | Sendblue API key id, sent as `sb-api-key-id`. |
| `SENDBLUE_API_SECRET_KEY` | Yes | Sendblue API secret key, sent as `sb-api-secret-key`. |
| `SENDBLUE_FROM_NUMBER` | Yes | Default Sendblue line in E.164 format. |
| `SENDBLUE_FROM_NUMBERS` | Optional | Comma-separated line pool. Hermes chooses a stable line per recipient. |
| `SENDBLUE_API_BASE_URL` | Optional | Sendblue API base URL. Default: `https://api.sendblue.com`. |
| `SENDBLUE_WEBHOOK_SECRET` | Yes | Secret configured on the Sendblue receive webhook. |
| `SENDBLUE_WEBHOOK_HOST` | Optional | Webhook bind host. Default: `127.0.0.1`. |
| `SENDBLUE_WEBHOOK_PORT` | Optional | Webhook bind port. Default: `8650`. |
| `SENDBLUE_WEBHOOK_PATH` | Optional | Webhook path. Default: `/sendblue/webhook`. |
| `SENDBLUE_STATUS_CALLBACK` | Optional | Per-message Sendblue status callback URL for outbound delivery updates. |
| `SENDBLUE_SEAT_ID` | Optional | Sendblue seat id for message attribution. |
| `SENDBLUE_STICKY_STATE_PATH` | Optional | JSON file for recipient -> `from_number` stickiness. Default: `~/.hermes/sendblue_sticky_senders.json`. |
| `SENDBLUE_ALLOWED_USERS` | Recommended | Comma-separated E.164 phone numbers allowed to talk to the bot. |
| `SENDBLUE_ALLOW_ALL_USERS` | Dev only | Allow every sender. Not recommended for bots with terminal access. |
| `SENDBLUE_HOME_CHANNEL` | Optional | Default phone number or `group:<id>` for cron / notification delivery. |
| `SENDBLUE_HOME_CHANNEL_NAME` | Optional | Human label for the home channel. |
| `SENDBLUE_INSECURE_NO_SIGNATURE` | Dev only | Set `true` to disable webhook secret validation locally. |

## Cron delivery

Once `SENDBLUE_HOME_CHANNEL` is set, cron jobs can deliver through Sendblue:

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="sendblue",
    prompt="Send me a short status summary."
)
```

Or target a recipient explicitly:

```bash
hermes send sendblue:+15559876543 "Done."
```

Group replies use `group:<id>` chat ids. The adapter uses Sendblue's
`/api/send-group-message` endpoint for those targets.

## Sendblue-specific behavior

- **Inbound-first limits.** Sendblue's AI Agent plan is designed for contacts
  who text you first. Replies within the 24-hour inbound window are unlimited;
  follow-up messages after that window count against the plan's follow-up
  contact limit.
- **Sticky sender lines.** Sendblue requires `from_number` on every send and
  recommends continuing to use the same line for a contact. Hermes persists
  that mapping locally.
- **Webhook ACKs.** Hermes returns 200-level responses immediately after
  accepting inbound events so Sendblue does not retry the same webhook.
- **Rate limits.** Sendblue returns HTTP 429 when a line is throttled. Hermes
  marks those send failures as retryable and preserves `Retry-After` when
  present.
- **Attachments.** Outbound `media_url` is supported when provided as
  metadata by a caller. Local `MEDIA:/...` attachments from cron are reported
  as generated but are not uploaded automatically.

## Troubleshooting

**Gateway refuses to start without `SENDBLUE_WEBHOOK_SECRET`.** Configure a
secret on the Sendblue receive webhook and set the same value in
`~/.hermes/.env`.

**Hermes receives no messages.** Confirm your public URL routes to
`SENDBLUE_WEBHOOK_HOST:SENDBLUE_WEBHOOK_PORT` and the Sendblue webhook type is
`receive`.

**Outbound fails with HTTP 400.** Check that `SENDBLUE_FROM_NUMBER` is a line on
your Sendblue account and the target is E.164 formatted.

**Outbound fails with HTTP 429.** Slow down, wait for the Sendblue rate-limit
window, or spread traffic across more lines with `SENDBLUE_FROM_NUMBERS`.

## References

- [Sendblue sending messages](https://docs.sendblue.com/getting-started/sending-messages/)
- [Sendblue receiving messages](https://docs.sendblue.com/getting-started/receiving-messages/)
- [Sendblue webhooks](https://docs.sendblue.com/getting-started/webhooks/)
- [Sendblue limits](https://docs.sendblue.com/limits/)
