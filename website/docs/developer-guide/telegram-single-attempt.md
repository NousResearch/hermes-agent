---
title: Single-attempt Telegram notifications
---

# Single-attempt Telegram notifications

Native Python integrations can opt into a single content-bearing Telegram attempt:

```python
import json
from tools.send_message_tool import send_message_once

result = json.loads(send_message_once({
    "target": "telegram:CHAT_ID",
    "message": "Hermes needs your attention.",
}))
```

This is a Python integration entry point, not a new model tool, remote authorization
endpoint, or retry setting. The existing `send_message_tool` entry point, its schema,
other platforms, formatting, attachments and default retry policy remain unchanged.

The opt-in entry accepts only `target`, `message`, and optional `action="send"`.
Targets must use the Telegram platform. Messages must be nonempty plain text within
4,096 UTF-16 units; media markers, attachment/document requests, other actions and
additional fields are rejected before transport. Existing native target resolution,
profile configuration, egress authorization and session mirroring are reused. No bot
token is accepted in the request: the existing managed bot factory owns credentials.

A supported request makes at most one SDK `send_message` invocation. There is no
HTML parse fallback, topic fallback, chunking or content retry in this path. SDK
initialization can make a metadata request such as `getMe`; this is not a promise
of exactly one HTTP request of any kind. Managed proxy/request configuration is
preserved.

The JSON response distinguishes:

- `delivery_status="accepted"`: Telegram returned a message ID; not proof that the
  recipient read it or that a mobile notification appeared.
- `delivery_status="not_sent"`: the request was rejected before sending or the SDK
  reported `RetryAfter`. A recognized rate limit uses `TELEGRAM_RATE_LIMITED`, with
  an optional finite `retry_after` delay in seconds, bounded to one day.
- `delivery_status="unknown"`: the attempt could not be confirmed. SDK exception
  text is not returned. Do not infer non-delivery or automatically repeat it.

Existing routing/authorization errors keep their native error format and may not
have `delivery_status`. Native cron duplicate suppression can return `success=true,
skipped=true` without `delivery_status`: no notification was attempted by this call.
Treat that skip as not sent, not as an accepted delivery. Treat other missing or
unrecognized statuses as unknown, including errors from the native async runner.
Never infer acceptance from `success` alone. Any later
attempt requires the caller's own fresh authorization/state check; this API is not
an exactly-once protocol, scheduler or worker-lifecycle fence.

Tests exercise the real SDK against synthetic HTTP transport. They do not establish
live bot delivery, mobile notification behavior or deployment on an existing host.
