---
sidebar_position: 2
sidebar_label: "Inkbox"
title: "Inkbox Messaging"
description: "Use Hermes Agent with Inkbox email, SMS/MMS, and voice"
---

# Inkbox Messaging

Inkbox gives one Hermes identity a mailbox, a phone number, and an edge tunnel. The Hermes Inkbox adapter maps inbound email, SMS/MMS, and live voice calls into normal gateway sessions.

## Protocol Model

Inkbox is a multi-channel platform adapter, not just an SMS adapter:

- `message.received` webhooks become email turns.
- Inbound `text.received` webhooks are grouped into SMS/MMS turns.
- Other `text.*` webhooks are delivery/lifecycle callbacks and are logged without starting an agent turn.
- Incoming calls are answered by the adapter and bridged over `/phone/media/ws`.

Hermes uses the resolved Inkbox Contact as the primary `chat_id` when possible. That means email, SMS, and outbound voice can share one contact-scoped session. Unknown senders fall back to the raw email address or E.164 phone number.

## SMS Behavior

Inbound SMS can be buffered per contact before Hermes starts an agent turn. The quiet window is disabled by default so upstream deployments keep immediate SMS behavior; set `sms_text_batch_delay_seconds` or `INKBOX_SMS_TEXT_BATCH_DELAY_SECONDS` to a positive number when a deployment wants rapid human fragments, corrections, and follow-ups to arrive as one prompt instead of several competing prompts. A multi-message turn uses an `inkbox:sms_burst` routing marker and includes relative timestamps for each fragment.

Slash commands bypass SMS batching and routing markers, so a text such as `/approve` or `/deny` reaches the Hermes command parser as a command rather than tagged SMS body text. Carrier protocol words such as `START`, `STOP`, `HELP`, `YES`, `SUBSCRIBE`, `INFO`, and `UNSUBSCRIBE` are treated as SMS control traffic and are acknowledged at the webhook layer without starting an agent turn.

Outbound SMS is queued with `identity.send_text(to=..., text=...)`. The adapter returns a `SendResult` with the Inkbox text id and non-body metadata such as `delivery_status`.

Important SMS gates are enforced by Inkbox and carriers:

- New local numbers may need 10-15 minutes of carrier propagation before sending.
- Recipients must opt in by texting `START` to an org number.
- `STOP` opts a recipient out until they opt in again.
- Per-number SMS sending is rate limited.
- MMS media may appear on inbound text records; the gateway currently treats SMS/MMS turns as text-first.

Common structured send errors:

| Error code | Meaning | Retry behavior |
|------------|---------|----------------|
| `sender_sms_pending` | Sender number is still provisioning | Do not immediate-retry |
| `messaging_profile_disabled` | Sender messaging profile is disabled upstream/provider-side | Do not retry content variants; check provisioning |
| `recipient_not_opted_in` | Recipient has not opted in with `START` | Do not retry |
| `recipient_opted_out` | Recipient sent `STOP` | Do not retry |
| `rate_limited` / send-limit errors | Per-number or provider send limit | Wait for the provider window |
| HTTP 5xx / transient provider errors | Provider temporarily unavailable | Retry with backoff |

Hermes preserves these failures in `SendResult.raw_response` with `status_code`, `error_code`, `category`, and `retryable`. Provider-gated SMS failures disable the generic plain-text fallback because changing message formatting does not fix provisioning, consent, or carrier state.

## Webhook Safety

By default the adapter requires Inkbox webhook signatures:

```bash
INKBOX_REQUIRE_SIGNATURE=true
INKBOX_SIGNING_KEY=...
```

For local-only testing you can set `INKBOX_REQUIRE_SIGNATURE=false`, but production deployments should keep signature verification enabled.

The adapter deduplicates webhooks by `X-Inkbox-Request-Id` and also deduplicates inbound SMS by Inkbox text id. This protects against duplicate agent runs when webhook retries arrive with a new request id.

## Configuration

The installer configures Inkbox automatically. Manual deployments use:

```bash
INKBOX_API_KEY=...
INKBOX_IDENTITY=your-agent-handle
INKBOX_SIGNING_KEY=...
INKBOX_PUBLIC_URL=https://your-public-host.example   # optional when not using the SDK tunnel
INKBOX_LISTEN_PORT=8765
INKBOX_HOME_CHANNEL=contact-or-phone
```

Use `INKBOX_BASE_URL` only for staging or development environments.
