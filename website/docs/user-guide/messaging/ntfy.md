# ntfy

[ntfy](https://ntfy.sh/) is a simple HTTP-based pub-sub notification service. It works with the free public server at `ntfy.sh` or any self-hosted instance, and supports any client that can make HTTP requests — phones, browsers, scripts, watches.

ntfy makes a great lightweight push channel for Hermes: subscribe to a topic from the [ntfy mobile app](https://ntfy.sh/docs/subscribe/phone/), send messages to the topic to talk to the agent, get the response back on your phone.

> Run `hermes gateway setup` and pick **ntfy** for a guided walk-through.

## Prerequisites

- A topic name (any unique string — `hermes-myname-2026` works fine)
- The [ntfy mobile app](https://ntfy.sh/docs/subscribe/phone/) installed and subscribed to that topic
- Optional: a self-hosted ntfy server, or an `ntfy.sh` account token for private/reserved topics

That's it. No SDK, no daemon, no Node.js. The adapter uses `httpx` which is already a Hermes dependency.

## Configure Hermes

### Via setup wizard

```bash
hermes gateway setup
```

Select **ntfy** and follow the prompts.

### Via environment variables

Add these to `~/.hermes/.env`:

```
NTFY_TOPIC=hermes-myname-2026
NTFY_ALLOWED_USERS=hermes-myname-2026
NTFY_HOME_CHANNEL=hermes-myname-2026
```

| Variable | Required | Description |
|---|---|---|
| `NTFY_TOPIC` | Yes | Topic to subscribe to (incoming messages) |
| `NTFY_SERVER_URL` | Optional | Server URL (default: `https://ntfy.sh`) — point to a self-hosted ntfy for privacy |
| `NTFY_TOKEN` | Optional | Bearer token (e.g. `tk_xyz`) or `user:pass` for Basic auth |
| `NTFY_PUBLISH_TOPIC` | Optional | Different topic for outgoing replies (defaults to `NTFY_TOPIC`) |
| `NTFY_MARKDOWN` | Optional | Set `true` to send replies with `X-Markdown: true` header |
| `NTFY_ALLOWED_USERS` | Recommended | Comma-separated topic names allowed (treated as user IDs; see below) |
| `NTFY_ALLOW_ALL_USERS` | Optional | Set `true` to allow every publisher — only safe for private topics with read tokens |
| `NTFY_HOME_CHANNEL` | Optional | Default topic for cron / notification delivery |
| `NTFY_HOME_CHANNEL_NAME` | Optional | Human label for the home channel |

## Identity model — read this before deploying

ntfy has no native authenticated user identity. The `title` field on a published message is **publisher-controlled** and can be anything the sender wants. The Hermes adapter does NOT use `title` for authorization — it would let any publisher who knows the topic spoof an allowed user.

Instead, **the topic name itself is the identity**. Every message published to the topic is treated as coming from the same logical user (the topic). `NTFY_ALLOWED_USERS` is therefore typically just the topic name itself — a single-entry allowlist that gates the whole channel.

This means **anyone who knows the topic can talk to the agent**. To make that a real trust boundary:

- **Self-host ntfy** and lock the topic down with [Access Control](https://docs.ntfy.sh/config/#access-control). Only authorized clients with the read/write token can publish.
- Or **use a private topic on ntfy.sh** ([reserved topics](https://docs.ntfy.sh/publish/#reserved-topics) require an account) and protect it with a `NTFY_TOKEN`.
- Or **pick a long, unguessable topic name** (`hermes-7d4f9c8b-2026`) and treat it as the shared secret. This is the lightest setup but the topic name leaks via any logs or screenshots.

In all cases, do not put sensitive data through ntfy unless the underlying topic is access-controlled.

## Quick start — talk to your agent from your phone

1. Pick a topic name: `hermes-myname-2026`
2. On your phone: install the [ntfy app](https://ntfy.sh/docs/subscribe/phone/), tap **+**, enter `hermes-myname-2026`
3. On the host:
   ```bash
   echo 'NTFY_TOPIC=hermes-myname-2026' >> ~/.hermes/.env
   echo 'NTFY_ALLOWED_USERS=hermes-myname-2026' >> ~/.hermes/.env
   hermes gateway restart
   ```
4. From the ntfy app, send a message to the topic. The agent's reply lands as a push notification.

## Using ntfy with cron jobs

Once `NTFY_HOME_CHANNEL` is set, cron jobs can deliver to ntfy:

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="ntfy",          # uses NTFY_HOME_CHANNEL
    prompt="Check for alerts and summarise."
)
```

Or target a specific topic explicitly:

```python
send_message(target="ntfy:alerts-channel", message="Done!")
```

This works even when the cron runs out-of-process from the gateway — the plugin registers a `standalone_sender_fn` that opens its own HTTP connection.

## Self-hosting ntfy

If you want full control:

```bash
# Docker
docker run -p 80:80 -it binwiederhier/ntfy serve

# Native
go install heckel.io/ntfy/v2@latest
ntfy serve
```

Then point Hermes at it:

```
NTFY_SERVER_URL=https://ntfy.mydomain.com
NTFY_TOPIC=hermes
NTFY_TOKEN=tk_abc123  # if you've set up access control
```

Self-hosting gives you topic access control, message persistence policies, attachments, and emoji tags. See the [ntfy server docs](https://docs.ntfy.sh/install/).

## Markdown formatting

ntfy clients render markdown when the publisher sets the `X-Markdown: true` header. To enable for outgoing Hermes replies:

```
NTFY_MARKDOWN=true
```

Or in `config.yaml`:

```yaml
platforms:
  ntfy:
    extra:
      markdown: true
```

The mobile app supports a subset of CommonMark — bold, italic, lists, links, fenced code blocks. See [ntfy's markdown docs](https://docs.ntfy.sh/publish/#markdown-formatting) for the exact set.

## Outgoing-only setup (notifications without inbound)

If you only want Hermes to *push* notifications to ntfy (cron summaries, alerts) and never accept messages back, set both `NTFY_TOPIC` and `NTFY_PUBLISH_TOPIC` to the same value and skip `NTFY_ALLOWED_USERS` entirely. With no allowlist, the agent never responds to inbound messages — your phone gets the pushes, but the conversation is one-way.

## Limits

- **Message size**: ntfy caps message bodies at 4096 chars. Hermes truncates with a warning when this is exceeded.
- **No typing indicators**: the protocol doesn't expose one; `send_typing` is a no-op.
- **No threads or attachments**: ntfy is plain push notifications. Long replies stay in the message body, no thread fanout.
- **No native user identity**: see the identity-model section above.

## Troubleshooting

**Auth failure / 401** — `NTFY_TOKEN` is wrong, or the token doesn't have publish/subscribe rights on this topic. The adapter halts its reconnect loop on 401 and the gateway runtime status will show `fatal: ntfy_unauthorized`. Fix the token and restart the gateway.

**Topic not found / 404** — `NTFY_TOPIC` doesn't exist on the configured server. For ntfy.sh, topics are auto-created on first publish, so a 404 means you're pointed at a self-hosted server that doesn't have the topic provisioned. The adapter halts its reconnect loop with `fatal: ntfy_topic_not_found`.

**Connected but no messages** — Check that `NTFY_ALLOWED_USERS` includes the topic name itself. With ntfy's identity model, the topic IS the user; leaving the allowlist empty rejects everything.

**Reconnects every 60s** — The stream keepalive default is 55s; ntfy may have intermittent network issues. The adapter applies exponential backoff (2 → 5 → 10 → 30 → 60s) and resets to 0 once a stream stays alive ≥60s.
