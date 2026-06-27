---
sidebar_position: 11
sidebar_label: "GitHub PR Reviews via Webhook"
title: "Automated GitHub PR Comments with Webhooks"
description: "Connect Hermes to GitHub so it automatically fetches PR diffs, reviews code changes, and posts comments — triggered by webhooks with no manual prompting"
---

# Automated GitHub PR Comments with Webhooks

This guide walks you through connecting Hermes Agent to GitHub so it automatically fetches a pull request's diff, analyzes the code changes, and posts a comment — triggered by a webhook event with no manual prompting.

When a PR is opened or updated, GitHub sends a webhook POST to your Hermes instance. Hermes runs the agent with a prompt that instructs it to retrieve the diff via the `gh` CLI, and the response is posted back to the PR thread.

:::tip Want a simpler setup without a public endpoint?
If you don't have a public URL or just want to get started quickly, check out [Build a GitHub PR Review Agent](./github-pr-review-agent.md) — uses cron jobs to poll for PRs on a schedule, works behind NAT and firewalls.
:::

:::info Reference docs
For the full webhook platform reference (all config options, delivery types, dynamic subscriptions, security model) see [Webhooks](/docs/user-guide/messaging/webhooks).
:::

:::warning Prompt injection risk
Webhook payloads contain attacker-controlled data — PR titles, commit messages, and descriptions can contain malicious instructions. When your webhook endpoint is exposed to the internet, run the gateway in a sandboxed environment (Docker, SSH backend). See the [security section](#security-notes) below.
:::

---

## Prerequisites

- Hermes Agent installed and running (`hermes gateway`)
- [`gh` CLI](https://cli.github.com/) installed and authenticated on the gateway host (`gh auth login`)
- A publicly reachable URL for your Hermes instance (see [Local testing with ngrok](#local-testing-with-ngrok) if running locally)
- Admin access to the GitHub repository (required to manage webhooks)

---

## Step 1 — Enable the webhook platform

Add the following to your `~/.hermes/config.yaml`:

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644          # default; change if another service occupies this port
      rate_limit: 30      # max requests per minute per route (not a global cap)

      routes:
        github-pr-review:
          secret: "your-webhook-secret-here"   # must match the GitHub webhook secret exactly
          events:
            - pull_request

          # The agent is instructed to fetch the actual diff before reviewing.
          # {number} and {repository.full_name} are resolved from the GitHub payload.
          prompt: |
            A pull request event was received (action: {action}).

            PR #{number}: {pull_request.title}
            Author: {pull_request.user.login}
            Branch: {pull_request.head.ref} → {pull_request.base.ref}
            Description: {pull_request.body}
            URL: {pull_request.html_url}

            If the action is "closed" or "labeled", stop here and do not post a comment.

            Otherwise:
            1. Run: gh pr diff {number} --repo {repository.full_name}
            2. Review the code changes for correctness, security issues, and clarity.
            3. Write a concise, actionable review comment and post it.

          deliver: github_comment
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
```

**Key fields:**

| Field | Description |
|---|---|
| `secret` (route-level) | HMAC secret for this route. Falls back to `extra.secret` global if omitted. |
| `events` | List of `X-GitHub-Event` header values to accept. Empty list = accept all. |
| `prompt` | Template; `{field}` and `{nested.field}` resolve from the GitHub payload. |
| `deliver` | `github_comment` posts via `gh pr comment`. `log` just writes to the gateway log. |
| `deliver_extra.repo` | Resolves to e.g. `org/repo` from the payload. |
| `deliver_extra.pr_number` | Resolves to the PR number from the payload. |

:::note The payload does not contain code
The GitHub webhook payload includes PR metadata (title, description, branch names, URLs) but **not the diff**. The prompt above instructs the agent to run `gh pr diff` to fetch the actual changes. The `terminal` tool is included in the default `hermes-webhook` toolset, so no extra configuration is needed.
:::

---

## Step 2 — Start the gateway

```bash
hermes gateway
```

You should see:

```
[webhook] Listening on 0.0.0.0:8644 — routes: github-pr-review
```

Verify it's running:

```bash
curl http://localhost:8644/health
# {"status": "ok", "platform": "webhook"}
```

---

## Step 3 — Register the webhook on GitHub

1. Go to your repository → **Settings** → **Webhooks** → **Add webhook**
2. Fill in:
   - **Payload URL:** `https://your-public-url.example.com/webhooks/github-pr-review`
   - **Content type:** `application/json`
   - **Secret:** the same value you set for `secret` in the route config
   - **Which events?** → Select individual events → check **Pull requests**
3. Click **Add webhook**

GitHub will immediately send a `ping` event to confirm the connection. It is safely ignored — `ping` is not in your `events` list — and returns `{"status": "ignored", "event": "ping"}`. It is only logged at DEBUG level, so it won't appear in the console at the default log level.

---

## Step 4 — Open a test PR

Create a branch, push a change, and open a PR. Within 30–90 seconds (depending on PR size and model), Hermes should post a review comment.

To follow the agent's progress in real time:

```bash
tail -f "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
```

---

## Local testing with ngrok

If Hermes is running on your laptop, use [ngrok](https://ngrok.com/) to expose it:

```bash
ngrok http 8644
```

Copy the `https://...ngrok-free.app` URL and use it as your GitHub Payload URL. On the free ngrok tier the URL changes each time ngrok restarts — update your GitHub webhook each session. Paid ngrok accounts get a static domain.

You can smoke-test a static route directly with `curl` — no GitHub account or real PR needed.

:::tip Use `deliver: log` when testing locally
Change `deliver: github_comment` to `deliver: log` in your config while testing. Otherwise the agent will attempt to post a comment to the fake `org/repo#99` repo in the test payload, which will fail. Switch back to `deliver: github_comment` once you're satisfied with the prompt output.
:::

```bash
SECRET="your-webhook-secret-here"
BODY='{"action":"opened","number":99,"pull_request":{"title":"Test PR","body":"Adds a feature.","user":{"login":"testuser"},"head":{"ref":"feat/x"},"base":{"ref":"main"},"html_url":"https://github.com/org/repo/pull/99"},"repository":{"full_name":"org/repo"}}'
SIG=$(printf '%s' "$BODY" | openssl dgst -sha256 -hmac "$SECRET" -hex | awk '{print "sha256="$2}')

curl -s -X POST http://localhost:8644/webhooks/github-pr-review \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: pull_request" \
  -H "X-Hub-Signature-256: $SIG" \
  -d "$BODY"
# Expected: {"status":"accepted","route":"github-pr-review","event":"pull_request","delivery_id":"..."}
```

Then watch the agent run:
```bash
tail -f "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
```

:::note
`hermes webhook test <name>` only works for **dynamic subscriptions** created with `hermes webhook subscribe`. It does not read routes from `config.yaml`.
:::

---

## Filtering to specific actions

GitHub sends `pull_request` events for many actions: `opened`, `synchronize`, `reopened`, `closed`, `labeled`, etc. The `events` list filters only by the `X-GitHub-Event` header value — it cannot filter by action sub-type at the routing level.

The prompt in Step 1 already handles this by instructing the agent to stop early for `closed` and `labeled` events.

:::warning The agent still runs and consumes tokens
The "stop here" instruction prevents a meaningful review, but the agent still runs to completion for every `pull_request` event regardless of action. GitHub webhooks can only filter by event type (`pull_request`, `push`, `issues`, etc.) — not by action sub-type (`opened`, `closed`, `labeled`). There is no routing-level filter for sub-actions. For high-volume repos, accept this cost or filter upstream with a GitHub Actions workflow that calls your webhook URL conditionally.
:::

> There is no Jinja2 or conditional template syntax. `{field}` and `{nested.field}` are the only substitutions supported. Anything else is passed verbatim to the agent.

---

## Using a skill for consistent review style

Load a [Hermes skill](/docs/user-guide/features/skills) to give the agent a consistent review persona. Add `skills` to your route inside `platforms.webhook.extra.routes` in `config.yaml`:

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        github-pr-review:
          secret: "your-webhook-secret-here"
          events: [pull_request]
          prompt: |
            A pull request event was received (action: {action}).
            PR #{number}: {pull_request.title} by {pull_request.user.login}
            URL: {pull_request.html_url}

            If the action is "closed" or "labeled", stop here and do not post a comment.

            Otherwise:
            1. Run: gh pr diff {number} --repo {repository.full_name}
            2. Review the diff using your review guidelines.
            3. Write a concise, actionable review comment and post it.
          skills:
            - review
          deliver: github_comment
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
```

> **Note:** Only the first skill in the list that is found is loaded. Hermes does not stack multiple skills — subsequent entries are ignored.

---

## Sending responses to Slack or Discord instead

Replace the `deliver` and `deliver_extra` fields inside your route with your target platform:

```yaml
# Inside platforms.webhook.extra.routes.<route-name>:

# Slack
deliver: slack
deliver_extra:
  chat_id: "C0123456789"   # Slack channel ID (omit to use the configured home channel)

# Discord
deliver: discord
deliver_extra:
  chat_id: "987654321012345678"  # Discord channel ID (omit to use home channel)
```

The target platform must also be enabled and connected in the gateway. If `chat_id` is omitted, the response is sent to that platform's configured home channel.

Valid `deliver` values: `log` · `github_comment` · `telegram` · `discord` · `slack` · `signal` · `sms`

---

## GitLab support

The same adapter works with GitLab. GitLab uses `X-Gitlab-Token` for authentication (plain string match, not HMAC) — Hermes handles both automatically.

For event filtering, GitLab sets `X-GitLab-Event` to values like `Merge Request Hook`, `Push Hook`, `Pipeline Hook`. Use the exact header value in `events`:

```yaml
events:
  - Merge Request Hook
```

GitLab payload fields differ from GitHub's — e.g. `{object_attributes.title}` for the MR title and `{object_attributes.iid}` for the MR number. The easiest way to discover the full payload structure is GitLab's **Test** button in your webhook settings, combined with the **Recent Deliveries** log. Alternatively, omit `prompt` from your route config — Hermes will then pass the full payload as formatted JSON directly to the agent, and the agent's response (visible in the gateway log with `deliver: log`) will describe its structure.

---

## Security notes

- **Never use `INSECURE_NO_AUTH`** in production — it disables signature validation entirely. It is only for local development.
- **Rotate your webhook secret** periodically and update it in both GitHub (webhook settings) and your `config.yaml`.
- **Rate limiting** is 30 req/min per route by default (configurable via `extra.rate_limit`). Exceeding it returns `429`.
- **Duplicate deliveries** (webhook retries) are deduplicated via a 1-hour idempotency cache. The cache key is `X-GitHub-Delivery` if present, then `X-Request-ID`, then a millisecond timestamp. When neither delivery ID header is set, retries are **not** deduplicated.
- **Prompt injection:** PR titles, descriptions, and commit messages are attacker-controlled. Malicious PRs could attempt to manipulate the agent's actions. Run the gateway in a sandboxed environment (Docker, VM) when exposed to the public internet.

---

## Troubleshooting

| Symptom | Check |
|---|---|
| `401 Invalid signature` | Secret in config.yaml doesn't match GitHub webhook secret |
| `404 Unknown route` | Route name in the URL doesn't match the key in `routes:` |
| `429 Rate limit exceeded` | 30 req/min per route exceeded — common when re-delivering test events from GitHub's UI; wait a minute or raise `extra.rate_limit` |
| No comment posted | `gh` not installed, not on PATH, or not authenticated (`gh auth login`) |
| Agent runs but no comment | Check the gateway log — if the agent output was empty or just "SKIP", delivery is still attempted |
| Port already in use | Change `extra.port` in config.yaml |
| Agent runs but reviews only the PR description | The prompt isn't including the `gh pr diff` instruction — the diff is not in the webhook payload |
| Can't see the ping event | Ignored events return `{"status":"ignored","event":"ping"}` at DEBUG log level only — check GitHub's delivery log (repo → Settings → Webhooks → your webhook → Recent Deliveries) |

**GitHub's Recent Deliveries tab** (repo → Settings → Webhooks → your webhook) shows the exact request headers, payload, HTTP status, and response body for every delivery. It is the fastest way to diagnose failures without touching your server logs.

---

## Full config reference

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      host: "0.0.0.0"         # bind address (default: 0.0.0.0)
      port: 8644               # listen port (default: 8644)
      secret: ""               # optional global fallback secret
      rate_limit: 30           # requests per minute per route
      max_body_bytes: 1048576  # payload size limit in bytes (default: 1 MB)

      routes:
        <route-name>:
          secret: "required-per-route"
          events: []            # [] = accept all; otherwise list X-GitHub-Event values
          prompt: ""            # {field} / {nested.field} resolved from payload
          skills: []            # first matching skill is loaded (only one)
          deliver: "log"        # log | github_comment | telegram | discord | slack | signal | sms
          deliver_extra: {}     # repo + pr_number for github_comment; chat_id for others
```

---

## What's Next?

- **[Cron-Based PR Reviews](./github-pr-review-agent.md)** — poll for PRs on a schedule, no public endpoint needed
- **[Webhook Reference](/docs/user-guide/messaging/webhooks)** — full config reference for the webhook platform
- **[Build a Plugin](/docs/guides/build-a-hermes-plugin)** — package review logic into a shareable plugin
- **[Profiles](/docs/user-guide/profiles)** — run a dedicated reviewer profile with its own memory and config
