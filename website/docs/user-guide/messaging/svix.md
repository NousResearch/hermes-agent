---
sidebar_position: 14
title: "Svix"
description: "Receive webhook events through Svix polling endpoints no public ingress required"
---

# Svix

Consume webhook events from third-party services through [Svix](https://www.svix.com/) without exposing a public HTTP server. Instead of hosting an endpoint that the outside world POSTs to, the Svix adapter **polls** Svix's polling-endpoint API: Svix receives the upstream webhook, buffers it, and Hermes fetches the events at its own pace.

## Setup

### Via environment variables

```bash
# ~/.hermes/.env
SVIX_ENABLED=true
```

### Via config.yaml

```yaml
platforms:
  svix:
    enabled: true
    extra:
      poll_interval: 5    # seconds to wait after catching up (default: 5)
      poll_limit: 50      # messages per poll request (default: 50)
      routes:
        github-events:
          url: https://api.svix.com/api/v1/app/app_xxx/poller/poll_yyy/
          auth_token_env: SVIX_GITHUB_TOKEN   # endpoint-scoped token
          prompt: |
            GitHub event {__event_type__}:
            {__raw__}
          deliver: telegram
          deliver_extra:
            chat_id: "-1001234567890"
```

---

## Svix Ingest

[Svix Ingest](https://docs.svix.com/ingest/overview) lets you funnel webhooks from any third-party service through Svix. Instead of giving GitHub your server's IP, you give it a Svix Ingest URL Svix receives and validates the payload, then makes it available on a **polling endpoint** that Hermes consumes.

To set up a polling endpoint:

1. Open the [Svix dashboard](https://dashboard.svix.com)
2. Go to **Ingest** → **Sources** → **Add Source**
3. Select the source type (GitHub, Stripe, generic, etc.) and configure signature validation
4. Copy the generated **polling endpoint URL** it looks like `https://api.svix.com/api/v1/app/app_xxx/poller/poll_yyy/`
5. Copy the **endpoint-scoped token** (`sk_endp_…`) shown in the UI
6. Paste both into your route config:

```yaml
routes:
  stripe-events:
    url: https://api.svix.com/api/v1/app/app_xxx/poller/poll_yyy/
    auth_token_env: SVIX_STRIPE_TOKEN
    prompt: "Stripe event {__event_type__}: {__raw__}"
    deliver: slack
```

---

## Auth

Each polling endpoint requires its own scoped token (`sk_endp_…`). The adapter resolves the token per route in this order:

| Priority | Source | Config key |
|----------|--------|------------|
| 1st | Literal value in config.yaml | `auth_token: "sk_endp_..."` |
| 2nd | Environment variable name | `auth_token_env: MY_ENV_VAR` |

Each route must declare its own token — the adapter logs a warning and refuses to start if none is set.

Use `auth_token_env` to keep secrets out of config files:

```bash
# ~/.hermes/.env
SVIX_GITHUB_TOKEN=sk_endp_xxxxx.eu
SVIX_STRIPE_TOKEN=sk_endp_yyyyy.eu
```

```yaml
routes:
  github-events:
    url: https://...
    auth_token_env: SVIX_GITHUB_TOKEN
  stripe-events:
    url: https://...
    auth_token_env: SVIX_STRIPE_TOKEN
```

:::warning
If `auth_token_env` is set but the environment variable is empty, the adapter refuses to start it won't silently fall back to the global token, since a configured-but-empty env var almost always means a deployment mistake.
:::

---

## Svix CLI

The [Svix CLI](https://docs.svix.com/cli) is the easiest way to obtain your account token and explore polling endpoints:

```bash
# npm
npm i -g svix-cli

# macOS
brew install svix/svix/svix-cli

# Linux / Windows see https://github.com/svix/svix-webhooks/tree/main/svix-cli#installation
```

Log in to get your account token

```bash
svix login
```
List your applications and polling endpoints:

```bash
svix application list
svix message list --app-id app_xxx
```

---

## Route Configuration

Each route under `platforms.svix.extra.routes` supports one Svix-specific required field:

| Field | Required | Description |
|-------|----------|-------------|
| `url` | **Yes** | Full Svix polling endpoint URL: `https://api.svix.com/api/v1/app/<app_id>/poller/<sink_id>/` |
| `auth_token` | **Yes*** | Endpoint-scoped token literal (prefer `auth_token_env`) |
| `auth_token_env` | **Yes*** | Name of env var holding the endpoint-scoped token |

*Exactly one of these must be set per route. The adapter warns and refuses to start if neither is present.

All other route fields (`prompt`, `skills`, `deliver`, `deliver_extra`, `deliver_only`) work exactly as described on the [Webhooks page](webhooks.md#configuring-routes).


### Prompt templates

Same `{dot.notation}` and `{__raw__}` syntax as webhooks see [Prompt Templates](webhooks.md#configuring-routes). The payload is the Svix message's `payload` field (the original webhook body forwarded by Svix).

### Delivery

All delivery targets (`telegram`, `discord`, `slack`, `github_comment`, etc.) and `deliver_only` work the same as webhooks see [Delivery Options](webhooks.md#delivery-options) and [Direct Delivery Mode](webhooks.md#direct-delivery-mode).

---

## Full Example

```yaml
platforms:
  svix:
    enabled: true
    extra:
      poll_interval: 5
      poll_limit: 50
      routes:
        github-prs:
          url: https://api.svix.com/api/v1/app/app_2yRk/poller/poll_3xZm/
          auth_token_env: SVIX_GITHUB_TOKEN
          prompt: |
            Review this pull request:
            Repository: {repository.full_name}
            PR #{number}: {pull_request.title}
            Author: {pull_request.user.login}
            URL: {pull_request.html_url}
          skills: ["github-code-review"]
          deliver: github_comment
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"

        stripe-alerts:
          url: https://api.svix.com/api/v1/app/app_2yRk/poller/poll_9aKn/
          auth_token_env: SVIX_STRIPE_TOKEN
          prompt: "Payment failed: {__raw__}"
          deliver: telegram
          deliver_extra:
            chat_id: "-1001234567890"
          deliver_only: true
```

---

## Cursor Tracking

Svix tracks the read position server-side. When the adapter restarts it resumes from where it left off — no local state file is needed.

- `done == false` in a poll response means "more pages — keep fetching immediately"
- `done == true` means "caught up — wait `poll_interval` seconds before the next request"
- Failures retry with exponential backoff (1s → 60s max)

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SVIX_ENABLED` | Enable the Svix adapter | `false` |
| `SVIX_POLL_INTERVAL` | Seconds to wait after catching up | `5` |

---

## Troubleshooting

### Adapter not starting

- Check for `[svix]` errors in `~/.hermes/logs/gateway.log`
- Ensure each route has a valid `url` in the form `https://.../api/v1/app/<app_id>/poller/<sink_id>/`
- Ensure each route has a token (`auth_token` literal or `auth_token_env` pointing to a set env var)
- Run `hermes gateway` in the foreground to see startup errors immediately

### Events not arriving

- Verify the Svix Ingest source received the event (check the Svix dashboard → **Ingest** → **Sources** → your source → **Logs**)
- Check the `events` filter on the route if set, only listed `eventType` values are processed
- Confirm the polling endpoint URL and token match what the dashboard shows


### SDK not installed

The Svix Python SDK is installed automatically when the adapter first connects. If you see `svix SDK not installed`, run:

```bash
pip install svix
```
