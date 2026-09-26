---
sidebar_position: 6.5
title: "Outbound Webhooks"
description: "Push signed, event-driven HTTP payloads to external endpoints from Hermes lifecycle events"
---

# Outbound Webhooks

Outbound webhooks are the push-side mirror of the [inbound webhook platform](../messaging/webhooks.md): inbound webhooks wake Hermes when the world changes; outbound webhooks tell the world when Hermes does something. Configure a list of HTTP endpoints and the lifecycle events they care about, and Hermes POSTs a signed JSON payload to each endpoint whenever a matching event fires — no polling on the receiving end.

Typical uses:

- Notify a CI system or dashboard when an agent turn finishes (`on_session_end`)
- Track subagent completions across a fleet (`subagent_stop`)
- Feed tool activity into external monitoring (`post_tool_call` with a `matcher`)
- Wake *another* Hermes instance: point the URL at that instance's inbound webhook

### Configuration

Add a `hooks.outbound:` list to `~/.hermes/config.yaml`:

```yaml
hooks:
  outbound:
    - name: ci-notify                       # optional label for logs
      url: https://ci.example.com/hermes-events
      events: [on_session_end, subagent_stop]
      secret_env: HERMES_OUTBOUND_WEBHOOK_SECRET   # env var holding the HMAC secret
      timeout: 10                           # per-attempt seconds (1–60)

    - name: tool-monitor
      url: https://metrics.example.com/hooks/hermes
      events: [post_tool_call]
      matcher: "terminal|delegate_task"     # regex, tool-scoped events only
```

Any event from the plugin-hook set is valid (`pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`, `on_session_start`, `on_session_end`, `subagent_start`, `subagent_stop`, ...). Malformed entries warn and are skipped — a broken webhook never crashes the agent. Changes take effect on the next CLI session / gateway restart.

Secrets: prefer `secret_env` (the name of an environment variable, typically set in `~/.hermes/.env`) over an inline `secret:` literal, so the config file stays free of credentials. Entries without a secret are delivered unsigned (flagged as `UNSIGNED` by `hermes hooks list`).

### Wire format

Each firing POSTs a JSON body with the same top-level shape as shell hooks' stdin, plus delivery metadata. `profile` names the Hermes profile that emitted the event (`"default"` outside profiles), so receivers behind a multiplexed gateway can tell profiles apart:

```json
{
  "hook_event_name": "on_session_end",
  "profile": "default",
  "tool_name": null,
  "tool_input": null,
  "session_id": "sess_abc123",
  "cwd": "/home/user/project",
  "extra": {"completed": true, "interrupted": false, "model": "...", "platform": "cli"},
  "delivery_id": "3f2c9a...",
  "timestamp": "2026-07-22T14:00:00Z"
}
```

Headers:

| Header | Value |
|--------|-------|
| `Content-Type` | `application/json` |
| `X-Hermes-Event` | The hook event name |
| `X-Hermes-Delivery` | Unique id per delivery — same value as `delivery_id` in the body |
| `X-Hermes-Signature-256` | `sha256=<hex>` — HMAC-SHA256 of the raw body, GitHub-style; only present when a secret is configured |

Verify the signature exactly as you would a GitHub webhook:

```python
import hashlib, hmac

def verify(body: bytes, header: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)
```

Because `delivery_id` and `timestamp` live **inside the signed body**, a verified receiver also gets replay protection for free:

- **Dedupe** on `delivery_id` (or the matching `X-Hermes-Delivery` header) — remember recently seen ids and skip duplicates. Hermes retries failed deliveries once, so the same id can legitimately arrive twice.
- **Reject stale events** by checking `timestamp` against your clock with a tolerance window (5 minutes is the common default). An attacker replaying a captured request can't forge a fresh timestamp without the secret.

### Delivery semantics

- **Fire-and-forget, off the hot path.** Events are serialized and queued instantly; a single background thread performs the HTTP POSTs. A slow or dead endpoint can never stall a tool call or an agent turn.
- **Notify-only.** Unlike shell hooks, outbound webhooks cannot block tool calls or inject context — the response body is ignored. They observe, never steer.
- **Bounded retries.** Connection errors and 5xx responses are retried once with backoff; 4xx responses are not retried (the receiver said the request itself is wrong). Failures are logged and dropped — delivery is best-effort, not guaranteed.
- **Redirects are never followed.** A 3xx response is treated as a misconfiguration and logged — following a redirected POST would silently drop the signed payload. Point the `url` at the final endpoint.
- **Bounded queue.** If the queue backs up (dead endpoint, event storm), new events are dropped with a warning rather than consuming unbounded memory.
- **No consent prompt.** Outbound targets execute no code on your machine — they receive data at a URL you configured. `HERMES_SAFE_MODE=1` still skips registration, same as plugins and shell hooks. Note that payloads include tool inputs and event metadata, so only point targets at endpoints you trust, and prefer `https://`.

`hermes hooks list` shows configured outbound targets alongside shell hooks, including whether each target is signed.
