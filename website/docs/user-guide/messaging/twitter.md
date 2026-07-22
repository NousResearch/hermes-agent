---
title: "Twitter / X"
description: "Connect Hermes to X mentions and direct messages with OAuth 2.0 PKCE"
---

# Twitter / X

The Twitter/X platform plugin lets Hermes respond to authorized mentions and direct messages, keep public reply branches separate, send images, and deliver scheduled posts. It uses the X API v2 directly; it does not scrape X or add an X SDK to Hermes core.

## Before setup

Create an app in the [X Developer Console](https://developer.x.com/) and enable OAuth 2.0 Authorization Code with PKCE. Hermes supports both X OAuth client types:

- **Public** clients use PKCE and have no client secret.
- **Confidential** clients use PKCE plus the client secret entered through Hermes's masked secret prompt. Setup stores the secret only in the active profile's credential `.env`, never in `config.yaml`.

Register the exact loopback callback URI you will configure, for example:

```text
http://127.0.0.1:8765/callback
```

Hermes requests and verifies these scopes:

```text
tweet.read tweet.write users.read offline.access
dm.read dm.write bookmark.read bookmark.write media.write
```

Before enabling automated replies, obtain any approval X requires for AI-generated replies, apply X's **Automated** account label, and identify the linked human-managed operator account. Hermes records your confirmation of those steps; it does not apply the label or obtain approval for you.

## Run setup

```bash
hermes gateway setup
```

Choose **Twitter / X**, select the client type, enter the client ID and registered callback URI, and complete authorization in the browser. The masked client-secret prompt appears only after you select **confidential**; setup writes that secret only to the active profile and uses it for the authorization in the same run. Public-client setup neither prompts for nor writes a secret. Setup saves access and refresh tokens in an owner-only file under the active Hermes profile. Do not hand-edit that file.

Setup asks whether every X user may invoke the agent. If not, it asks for numeric X user IDs; an empty allowlist then authorizes nobody. Setup also records the three automation-policy confirmations below, which default to `false`.

## Configuration

Setup writes the platform block to `~/.hermes/config.yaml`. These are the implemented settings and defaults:

```yaml
twitter:
  enabled: true
  client_id: "your-oauth-client-id"
  oauth_client_type: public # public or confidential
  redirect_uri: "http://127.0.0.1:8765/callback"
  allowed_users:
    - "1234567890123456789"
  allow_all_users: false
  home_channel: timeline
  policy:
    ai_reply_approval_confirmed: false
    automated_label_confirmed: false
    human_operator_account_confirmed: false
    opt_out_keywords:
      - stop
      - unsubscribe
      - opt out
  poll_interval_seconds: 30
  initial_backfill: 0
  conversation:
    max_depth: 8
    max_posts: 40
    siblings_per_parent: 5
    quote_posts_per_target: 5
  media:
    max_download_bytes: 10485760
    max_upload_bytes: 5242880
  queue:
    max_pending_per_bucket: 100
    max_wait_seconds: 900
```

Change all three `policy` confirmations to `true` only after the corresponding external requirements are true. Public replies, automated DM replies, and reply-routed scheduled delivery otherwise fail closed. `opt_out_keywords` must remain non-empty.

Non-secret behavior belongs in `config.yaml`. OAuth tokens and a confidential client secret remain profile-scoped credentials managed by setup.

## Triggers, quotes, and branches

An authorized public post triggers Hermes only when X's structured data says it:

1. mentions the authenticated account; or
2. directly replies to the authenticated account.

Handle substrings, keywords, nearby conversation posts, siblings, and quotes alone do not trigger a response. A quote is context only unless the quoting post independently contains a structured mention. Quote, conversation, profile, metric, and media data is bounded, best-effort, and labeled as untrusted user context.

Public sessions use:

```text
tweet:<conversation_id>:<participation_anchor_id>
```

The participation anchor separates parallel branches in one broad X conversation. After Hermes confirms a reply, it maps the new bot post back to that branch. A later direct reply can continue the branch, but every new author is checked against the allowlist independently. Each eligible inbound interaction can receive at most one automated reply.

## Direct messages and backfill

DM sessions use X's real conversation ID:

```text
dm:<dm_conversation_id>
```

Hermes sends only to a DM conversation first established by an authorized inbound DM. An exact, whitespace-normalized match for an `opt_out_keywords` entry records the conversation as opted out, creates no agent turn, and blocks later automated sends. Re-enabling it requires an explicit local operator action; the model cannot clear an opt-out.

`initial_backfill` applies to both mention and DM pollers and accepts `0` through `100`:

- `0` records the newest first-run boundaries without dispatching old events.
- A positive value dispatches at most that many newest first-run events in chronological order.
- After boundaries exist, restarts resume from persisted mention and DM event IDs and deduplicate overlap.

## Text and images

X receives one plain-text post. Hermes converts Markdown links to `label (URL)`, strips remaining Markdown markers, normalizes Unicode, and validates the result with X-compatible weighted counting for URLs, emoji, CJK, combining sequences, and reply mentions.

Over-limit text fails before an X write. Hermes does not split, truncate, or post a partial response.

Inbound images are downloaded only after authorization and are checked for HTTPS/SSRF safety, MIME, decoded format, dimensions, count, timeout, and byte limits. Outbound delivery accepts local JPG, PNG, or WEBP files: up to four images on a public post and one image in a DM. A validation or upload failure aborts the whole send; there is no silent text-only fallback. Video and animated GIF upload are not supported.

## Delivery safety, tools, and cron

Reads, writes, media, and tools use bounded endpoint-specific queues. A confirmed 429 may wait for X's reset and retry. A connection failure, timeout, or server failure after a write begins is ambiguous, so Hermes does not retry it automatically or claim success. Operator reconciliation may be required; exact-once delivery cannot be guaranteed after an ambiguous X result.

The plugin exposes two asynchronous tools only when the active profile has usable OAuth credentials:

- `twitter_bookmarks` lists, adds, or removes one bookmark explicitly.
- `twitter_post_metrics` reads public metrics, plus non-public metrics only when X returns them for eligible bot-authored posts.

Bookmarks are never inserted into prompts automatically, and Hermes does not continuously poll engagement metrics.

The default scheduled-delivery route is `timeline`:

```bash
hermes cron add "0 9 * * *" "Post one concise daily update" --deliver twitter
```

Timeline jobs reject unsolicited mentions. A typed public reply or DM schedule is accepted only when it carries the exact `tweet:...` or `dm:...` route and the persisted state proves the interaction or DM conversation is eligible; public replies also require the explicit inbound interaction ID. Bare numeric destinations are rejected.

Start and inspect the gateway with:

```bash
hermes gateway start
hermes gateway status
```

## Not supported

The plugin does not support OAuth 1.0a, app-only bearer authentication, scraping, browser automation, unofficial APIs, filtered-stream/webhook ingestion, automatic replies to every post in a conversation, quote-only replies, automatic multi-post thread splitting, video or animated GIF upload, durable replay of queued writes after restart, or exact-once guarantees after an ambiguous write.
