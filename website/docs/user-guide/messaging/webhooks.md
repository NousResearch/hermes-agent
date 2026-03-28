---
sidebar_position: 13
title: "Webhooks"
description: "Receive events from GitHub, GitLab, and other services to trigger Hermes agent runs"
---

# Webhooks

Receive events from external services (GitHub, GitLab, JIRA, Stripe, etc.) and trigger Hermes agent runs automatically. The webhook adapter runs an HTTP server that accepts POST requests, validates HMAC signatures, transforms payloads into agent prompts, and routes responses back to the source or to another configured platform.

The agent processes the event and can respond by posting comments on PRs, sending messages to Telegram/Discord, or logging the result.

---

## Quick Start

1. Enable via `hermes gateway setup` or environment variables
2. Define webhook routes in `config.yaml`
3. Point your service at `http://your-server:8644/webhooks/<route-name>`

---

## Setup

There are two ways to enable the webhook adapter.

### Via setup wizard

```bash
hermes gateway setup
```

Follow the prompts to enable webhooks, set the port, and set a global HMAC secret.

### Via environment variables

Add to `~/.hermes/.env`:

```bash
WEBHOOK_ENABLED=true
WEBHOOK_PORT=8644        # default
WEBHOOK_SECRET=your-global-secret
```

### Verify the server

Once the gateway is running:

```bash
curl http://localhost:8644/health
```

Expected response:

```json
{"status": "ok", "platform": "webhook"}
```

---

## Configuring Routes {#configuring-routes}

Routes define how different webhook sources are handled. Each route is a named entry under `platforms.webhook.extra.routes` in your `config.yaml`.

### Route properties

| Property | Required | Description |
|----------|----------|-------------|
| `events` | No | List of event types to accept (e.g. `["pull_request"]`). If empty, all events are accepted. Event type is read from `X-GitHub-Event`, `X-GitLab-Event`, or `event_type` in the payload. |
| `secret` | **Yes** | HMAC secret for signature validation. Falls back to the global `secret` if not set on the route. Set to `"INSECURE_NO_AUTH"` for testing only (skips validation). |
| `prompt` | No | Template string with dot-notation payload access (e.g. `{pull_request.title}`). If omitted, the full JSON payload is dumped into the prompt. |
| `skills` | No | List of skill names to load for the agent run. |
| `deliver` | No | Where to send the response: `github_comment`, `telegram`, `discord`, `slack`, `signal`, `sms`, or `log` (default). |
| `deliver_extra` | No | Additional delivery config — keys depend on `deliver` type (e.g. `repo`, `pr_number`, `chat_id`). Values support the same `{dot.notation}` templates as `prompt`. |

### Full example

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644
      secret: "global-fallback-secret"
      routes:
        github-pr:
          events: ["pull_request"]
          secret: "github-webhook-secret"
          prompt: |
            Review this pull request:
            Repository: {repository.full_name}
            PR #{number}: {pull_request.title}
            Author: {pull_request.user.login}
            URL: {pull_request.html_url}
            Diff URL: {pull_request.diff_url}
            Action: {action}
          skills: ["github-code-review"]
          deliver: "github_comment"
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
        deploy-notify:
          events: ["push"]
          secret: "deploy-secret"
          prompt: "New push to {repository.full_name} branch {ref}: {head_commit.message}"
          deliver: "telegram"
```

### Prompt Templates

Prompts use dot-notation to access nested fields in the webhook payload:

- `{pull_request.title}` resolves to `payload["pull_request"]["title"]`
- `{repository.full_name}` resolves to `payload["repository"]["full_name"]`
- Missing keys are left as the literal `{key}` string (no error)
- Nested dicts and lists are JSON-serialized and truncated at 2000 characters

If no `prompt` template is configured for a route, the entire payload is dumped as indented JSON (truncated at 4000 characters).

The same dot-notation templates work in `deliver_extra` values.

---

## GitHub PR Review (Step by Step) {#github-pr-review}

This walkthrough sets up automatic code review on every pull request.

### 1. Create the webhook in GitHub

1. Go to your repository → **Settings** → **Webhooks** → **Add webhook**
2. Set **Payload URL** to `http://your-server:8644/webhooks/github-pr`
3. Set **Content type** to `application/json`
4. Set **Secret** to match your route config (e.g. `github-webhook-secret`)
5. Under **Which events?**, select **Let me select individual events** and check **Pull requests**
6. Click **Add webhook**

### 2. Add the route config

Add the `github-pr` route to your `~/.hermes/config.yaml` as shown in the example above.

### 3. Ensure `gh` CLI is authenticated

The `github_comment` delivery type uses the GitHub CLI to post comments:

```bash
gh auth login
```

### 4. Test it

Open a pull request on the repository. The webhook fires, Hermes processes the event, and posts a review comment on the PR.

---

## GitLab Webhook Setup {#gitlab-webhook-setup}

GitLab webhooks work similarly but use a different authentication mechanism. GitLab sends the secret as a plain `X-Gitlab-Token` header (exact string match, not HMAC).

### 1. Create the webhook in GitLab

1. Go to your project → **Settings** → **Webhooks**
2. Set the **URL** to `http://your-server:8644/webhooks/gitlab-mr`
3. Enter your **Secret token**
4. Select **Merge request events** (and any other events you want)
5. Click **Add webhook**

### 2. Add the route config

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        gitlab-mr:
          events: ["merge_request"]
          secret: "your-gitlab-secret-token"
          prompt: |
            Review this merge request:
            Project: {project.path_with_namespace}
            MR !{object_attributes.iid}: {object_attributes.title}
            Author: {object_attributes.last_commit.author.name}
            URL: {object_attributes.url}
            Action: {object_attributes.action}
          deliver: "log"
```

---

## Delivery Options {#delivery-options}

The `deliver` field controls where the agent's response goes after processing the webhook event.

| Deliver Type | Description |
|-------------|-------------|
| `log` | Logs the response to the gateway log output. This is the default and is useful for testing. |
| `github_comment` | Posts the response as a PR/issue comment via the `gh` CLI. Requires `deliver_extra.repo` and `deliver_extra.pr_number`. The `gh` CLI must be installed and authenticated on the gateway host (`gh auth login`). |
| `telegram` | Routes the response to Telegram. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `discord` | Routes the response to Discord. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `slack` | Routes the response to Slack. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `signal` | Routes the response to Signal. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `sms` | Routes the response to SMS via Twilio. Uses the home channel, or specify `chat_id` in `deliver_extra`. |

For cross-platform delivery (telegram, discord, slack, signal, sms), the target platform must also be enabled and connected in the gateway. If no `chat_id` is provided in `deliver_extra`, the response is sent to that platform's configured home channel.

---

## Security {#security}

The webhook adapter includes multiple layers of security:

### HMAC signature validation

The adapter validates incoming webhook signatures using the appropriate method for each source:

- **GitHub**: `X-Hub-Signature-256` header — HMAC-SHA256 hex digest prefixed with `sha256=`
- **GitLab**: `X-Gitlab-Token` header — plain secret string match
- **Generic**: `X-Webhook-Signature` header — raw HMAC-SHA256 hex digest

If a secret is configured but no recognized signature header is present, the request is rejected.

### Secret is required

Every route must have a secret — either set directly on the route or inherited from the global `secret`. Routes without a secret cause the adapter to fail at startup with an error. For development/testing only, you can set the secret to `"INSECURE_NO_AUTH"` to skip validation entirely.

### Rate limiting

Each route is rate-limited to **30 requests per minute** by default (fixed-window). Configure this globally:

```yaml
platforms:
  webhook:
    extra:
      rate_limit: 60  # requests per minute
```

Requests exceeding the limit receive a `429 Too Many Requests` response.

### Idempotency

Delivery IDs (from `X-GitHub-Delivery`, `X-Request-ID`, or a timestamp fallback) are cached for **1 hour**. Duplicate deliveries (e.g. webhook retries) are silently skipped with a `200` response, preventing duplicate agent runs.

### Body size limits

Payloads exceeding **1 MB** are rejected before the body is read. Configure this:

```yaml
platforms:
  webhook:
    extra:
      max_body_bytes: 2097152  # 2 MB
```

### Prompt injection risk

:::warning
Webhook payloads contain attacker-controlled data — PR titles, commit messages, issue descriptions, etc. can all contain malicious instructions. Run the gateway in a sandboxed environment (Docker, VM) when exposed to the internet. Consider using the Docker or SSH terminal backend for isolation.
:::

---

## Troubleshooting {#troubleshooting}

### Webhook not arriving

- Verify the port is exposed and accessible from the webhook source
- Check firewall rules — port `8644` (or your configured port) must be open
- Verify the URL path matches: `http://your-server:8644/webhooks/<route-name>`
- Use the `/health` endpoint to confirm the server is running

### Signature validation failing

- Ensure the secret in your route config exactly matches the secret configured in the webhook source
- For GitHub, the secret is HMAC-based — check `X-Hub-Signature-256`
- For GitLab, the secret is a plain token match — check `X-Gitlab-Token`
- Check gateway logs for `Invalid signature` warnings

### Event being ignored

- Check that the event type is in your route's `events` list
- GitHub events use values like `pull_request`, `push`, `issues` (the `X-GitHub-Event` header value)
- GitLab events use values like `merge_request`, `push` (the `X-GitLab-Event` header value)
- If `events` is empty or not set, all events are accepted

### Agent not responding

- Run the gateway in foreground to see logs: `hermes gateway run`
- Check that the prompt template is rendering correctly
- Verify the delivery target is configured and connected

### Duplicate responses

- The idempotency cache should prevent this — check that the webhook source is sending a delivery ID header (`X-GitHub-Delivery` or `X-Request-ID`)
- Delivery IDs are cached for 1 hour

### `gh` CLI errors (GitHub comment delivery)

- Run `gh auth login` on the gateway host
- Ensure the authenticated GitHub user has write access to the repository
- Check that `gh` is installed and on the PATH

---

## Environment Variables {#environment-variables}

| Variable | Description | Default |
|----------|-------------|---------|
| `WEBHOOK_ENABLED` | Enable the webhook platform adapter | `false` |
| `WEBHOOK_PORT` | HTTP server port for receiving webhooks | `8644` |
| `WEBHOOK_SECRET` | Global HMAC secret (used as fallback when routes don't specify their own) | _(none)_ |
