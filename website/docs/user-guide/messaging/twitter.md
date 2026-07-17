---
title: "Twitter / X"
description: "Connect Hermes to X mentions and direct messages with OAuth 2.0 PKCE"
---

# Twitter / X

The Twitter/X platform plugin lets Hermes respond to authorized mentions and direct messages, preserve public reply branches, send images, and deliver scheduled posts. It uses the X API v2 directly and adds no X SDK dependency.

## Create an X developer app

Create an application in the [X Developer Console](https://developer.x.com/) and enable OAuth 2.0 Authorization Code with PKCE. Register this exact loopback callback URL, or the loopback URL you intend to configure:

```text
http://127.0.0.1:8765/callback
```

The plugin requests only the scopes used by its implemented features:

```text
tweet.read tweet.write users.read offline.access
dm.read dm.write bookmark.read bookmark.write media.write
```

App-only bearer tokens and OAuth 1.0a are not supported.

## Run interactive setup

```bash
hermes gateway setup
```

Choose **Twitter / X**, enter the OAuth 2.0 client ID and registered loopback redirect URI, then complete authorization in the browser. Hermes validates the callback state and saves tokens under the active profile, not in `.env`.

Setup asks for numeric X user IDs. Access is fail-closed: if `allowed_users` is empty and `allow_all_users` is false, inbound posts and DMs do not reach the agent.

## Manual configuration

Interactive setup is recommended. The equivalent `~/.hermes/config.yaml` settings are:

```yaml
twitter:
  enabled: true
  client_id: "your-public-oauth-client-id"
  redirect_uri: "http://127.0.0.1:8765/callback"
  allowed_users:
    - "1234567890123456789"
  allow_all_users: false
  home_channel: timeline
  poll_interval_seconds: 30
  initial_backfill: 0
  conversation:
    max_depth: 8
    max_posts: 40
    siblings_per_parent: 5
  media:
    max_download_bytes: 10485760
    max_upload_bytes: 5242880
  queue:
    max_pending: 100
    max_wait_seconds: 900
```

Do not hand-edit the OAuth token file. Re-run setup to authorize another profile or account.

## Trigger and routing behavior

Hermes responds to an authorized public post only when X's structured data says the post mentions the authenticated account, directly replies to it, or quotes a bot-authored post. Nearby replies and text that merely contains the handle do not trigger unsolicited responses.

Public sessions use typed destinations:

```text
tweet:<conversation_id>:<participation_anchor_id>
```

The participation anchor keeps parallel branches of a popular conversation separate while allowing another independently authorized participant to continue a branch by replying to the bot. Direct messages use the actual X DM conversation ID:

```text
dm:<dm_conversation_id>
```

`timeline` creates a top-level public post and is the default cron destination. Bare numeric destinations are rejected because an X ID alone does not identify whether it is a post, user, or DM conversation.

## Backfill, context, and media

- `initial_backfill: 0` records the latest cursor without dispatching old events on first startup. A positive value dispatches at most that many newest events in chronological order.
- Public conversation search is best-effort and bounded. If it is unavailable, the triggering post is still delivered with whatever direct-parent context is available.
- Post text, profile fields, metrics, and media metadata are always labeled as untrusted user context.
- Inbound images are downloaded only after authorization and are subject to HTTPS, SSRF, MIME, count, timeout, and byte limits.
- Outbound delivery supports up to four JPG, PNG, or WEBP images on a post and one image in a DM. If any upload fails, Hermes does not silently send a text-only message.

## Rate limits and uncertain delivery

Requests are serialized through bounded per-endpoint queues. Explicit HTTP 429 responses may wait for `Retry-After` or `x-rate-limit-reset` and retry. A timeout or connection failure after a post or DM write begins is treated as an uncertain outcome and is not automatically retried, avoiding accidental duplicates.

The `twitter_bookmarks` and `twitter_post_metrics` tools appear only when the active Hermes profile has usable X OAuth credentials. Bookmarks are never injected into prompts automatically, and metrics are fetched only when the explicit tool is used.

Start and inspect the gateway with:

```bash
hermes gateway start
hermes gateway status
```
