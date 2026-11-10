---
sidebar_position: 13
title: "Webhooks"
description: "Receive events from GitHub, GitLab, and other services to trigger Hermes agent runs"
---

# Webhooks

Receive events from external services (GitHub, GitLab, JIRA, Stripe, etc.) and trigger Hermes agent runs automatically. The webhook adapter runs an HTTP server that accepts POST requests, validates HMAC signatures, transforms payloads into agent prompts, and routes responses back to the source or to another configured platform.

The agent processes the event and can respond by posting comments on PRs, sending messages to Telegram/Discord, or logging the result.

## Video Tutorial

<div style={{position: 'relative', width: '100%', aspectRatio: '16 / 9', marginBottom: '1.5rem'}}>
  <iframe
    src="https://www.youtube.com/embed/WNYe5mD4fY8"
    title="Hermes Agent — Webhooks Tutorial"
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', border: 0}}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowFullScreen
  />
</div>

---

## Quick Start

1. Enable via `hermes gateway setup` or environment variables
2. Define routes in `config.yaml` **or** create them dynamically with `hermes webhook subscribe`
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

Routes define how different webhook sources are handled. Each route is a named entry under `platforms.webhook.extra.routes` in your `config.yaml`. Adapter settings (`port`, `host`, `secret`, `routes`) may also be written directly under `platforms.webhook:` — both spellings reach the adapter; a value nested under `extra:` wins if the same key appears in both places.

### Route properties

| Property | Required | Description |
|----------|----------|-------------|
| `events` | No | List of event types to accept (e.g. `["pull_request"]`). If empty, all events are accepted. Event type is read from `X-GitHub-Event`, `X-GitLab-Event`, or `event_type` in the payload. |
| `secret` | **Yes** | HMAC secret for signature validation. Falls back to the global `secret` if not set on the route. Set to `"INSECURE_NO_AUTH"` for testing only (skips validation). |
| `profile` | No | Profile authorized to execute this route when `gateway.multiplex_profiles` is enabled. Omit it for a default-profile-only route; set a profile name (for example `coder`) to bind the route and its secret to `/p/coder/webhooks/<route>`. Dynamic subscriptions set it with `hermes webhook subscribe <name> --route-profile coder`. |
| `prompt` | No | Template string with dot-notation payload access (e.g. `{pull_request.title}`). If omitted, the full JSON payload is dumped into the prompt. Payload fields are untrusted — see [Authenticated does not mean trusted](#authenticated-does-not-mean-trusted). |
| `filters` | No | Declarative payload filters evaluated after auth/body/event filtering and before agent or direct delivery work. Non-matches return `{"status":"ignored","reason":"filter"}` with HTTP 200. |
| `script` | No | Filter/transform script under `~/.hermes/scripts/`. The webhook payload is passed as JSON on stdin. JSON object stdout replaces the payload before templating; text stdout is exposed as `script_output`; empty stdout, `[SILENT]`, or a nonzero exit code ignores the webhook. |
| `skills` | No | List of skill names to load for the agent run. |
| `toolsets` | No | List of toolset keys (e.g. `["terminal", "file", "web"]`) that **replaces** the platform-level webhook toolset for runs triggered by this route only. Manual config edit only — not settable via `hermes webhook subscribe`, so agent-created subscriptions cannot self-grant elevated tools. Names are validated the same way as `platform_toolsets` entries (unknown or platform-restricted names are dropped). See [Per-route toolsets](#per-route-toolsets). |
| `deliver` | No | Where to send the response: `github_comment`, `telegram`, `discord`, `slack`, `signal`, `sms`, `whatsapp`, `matrix`, `mattermost`, `homeassistant` (plugin), `email`, `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot`, or `log` (default). |
| `deliver_extra` | No | Additional delivery config — keys depend on `deliver` type (e.g. `repo`, `pr_number`, `chat_id`). Values support the same `{dot.notation}` templates as `prompt`. |
| `deliver_only` | No | If `true`, I skip the webhook-side agent and deliver the rendered `prompt` template. A receiving Bot Chat may still run its model and incur cost. See [Direct Delivery Mode](#direct-delivery-mode) for use cases. Requires `deliver` to be a real target (not `log`). |
| `cron_job` | No | Fire an existing cron job (by ID or name) on each event instead of starting a fresh webhook agent session. The rendered `prompt` becomes transient per-run context; the job's own prompt, skills, model, and delivery settings apply. Mutually exclusive with `deliver_only`. See [Event-Triggered Cron Jobs](#event-triggered-cron-jobs). |
| `coalesce` | No | Debounce rapid distinct events on the same logical entity into one agent run. Block with a required `key` (payload field or template identifying the entity, e.g. `pull_request.number`), optional `window_seconds` (quiet window, default 30) and `max_wait_seconds` (dispatch cap, default 300). See [Event Coalescing](#event-coalescing). Mutually exclusive with `deliver_only` and `cron_job`. |
| `mirror_to_session` | No | Default `false`. When `true`, after a successful delivery to a chat platform the delivered message is also written into that chat's session transcript (as a labelled user turn, the same way continuable cron briefs are), so when you reply in that chat the agent knows what it just sent you. See [Replying to a delivery](#replying-to-a-delivery). |

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
          filters:
            - field: "ref"
              equals: "refs/heads/main"
          deliver: "telegram"
```

### Payload Filters

Use `filters` when a provider sends a broad event stream but only some payloads should wake the agent or trigger `deliver_only` delivery. Filters run after signature validation, body parsing, and `events`, but before prompt rendering, idempotency, agent dispatch, or direct delivery.

```yaml
platforms:
  webhook:
    extra:
      routes:
        todoist:
          events: ["item:updated"]
          secret: "todoist-secret"
          filters:
            - field: "payload.labels"
              contains: "hermes"
            - any:
                - field: "payload.priority"
                  equals: 4
                - field: "payload.project_id"
                  in_file: "~/.hermes/data/todoist/watchlist.json"
          prompt: "Todoist task changed: {payload.content}"
```

Supported operators:

- `exists: true|false`
- `missing: true`
- `equals` / `not_equals`
- `contains` for strings, lists, and dict keys
- `in` for inline lists
- `in_file` for JSON arrays, JSON objects (keys are used), or newline-delimited text files
- `regex`
- `all`, `any`, and `not` groups

Field paths use dot notation. `payload.foo` reads from a top-level `payload` object when one exists, or from the root webhook body for flat payloads. `event` / `event_type` match the resolved event type, and `headers.<Name>` reads request headers.

### Event Coalescing {#event-coalescing}

Providers often fire several distinct events for the same logical entity in quick succession — five rapid pushes to one pull request, a burst of edits to one ticket, a flapping monitoring alert. Each event carries a fresh delivery ID, so the idempotency cache cannot suppress them and every event wakes a separate agent run.

Set `coalesce` on a route to debounce these into a single run per entity:

```yaml
platforms:
  webhook:
    extra:
      routes:
        github-pr:
          events: ["pull_request"]
          secret: "github-webhook-secret"
          coalesce:
            key: "{repository.full_name}#{pull_request.number}"
            window_seconds: 30      # quiet window (default 30)
            max_wait_seconds: 300   # dispatch cap (default 300)
          prompt: "Review PR #{pull_request.number}: {pull_request.title}"
          deliver: "github_comment"
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{pull_request.number}"
```

How it works:

- Events are grouped per route by the rendered `key`. A bare dotted field (`pull_request.number`) or a full template (`{repository.full_name}#{pull_request.number}`) both work.
- Each new event **replaces** the pending one and pushes the quiet-window timer back. When `window_seconds` pass with no new event, the group dispatches **one** agent run using the latest event's payload, prompt, and delivery templates.
- `max_wait_seconds` caps total buffering from the group's first event, so a steady event stream cannot postpone dispatch forever.
- When more than one event was coalesced, the prompt gets a short note telling the agent how many earlier events were superseded.
- If the `key` does not resolve for an event (the payload lacks the field — e.g. an `issue_comment` event on a route keyed by `pull_request.number`), that event is **dispatched immediately** instead of being coalesced, so unrelated entities never collapse into one group. Pick a key present on every event type the route accepts.
- Coalesced requests return HTTP 202 with `{"status": "coalesced"}`. Delivery-ID idempotency still runs first, so provider retries of the same delivery are dropped rather than counted.
- Pending groups are flushed (dispatched immediately) when the adapter disconnects — a gateway reconnect or `hermes gateway stop` — not dropped. Buffered events live in memory, so a hard process kill loses at most the current window's buffered burst.
- `coalesce` applies to agent-mode routes only; combining it with `deliver_only` or `cron_job` is rejected at startup.

### Script Filters and Transforms

Use `script` when declarative filters are not enough. Scripts must live under `~/.hermes/scripts/` for the active profile; relative paths resolve there, and path traversal outside that directory is blocked. `.sh` and `.bash` scripts run with bash, and all other extensions run with the current Python interpreter.

The route payload is sent to stdin as JSON:

```python
# ~/.hermes/scripts/todoist-hermes-label.py
import json
import sys

payload = json.load(sys.stdin)
labels = payload.get("payload", {}).get("labels", [])
if "hermes" not in labels:
    print("[SILENT]")
    raise SystemExit(0)

payload["body"] = payload["payload"]["content"]
print(json.dumps(payload))
```

Script outcomes:

- JSON object stdout replaces the payload used by `prompt` and `deliver_extra`.
- Non-JSON text stdout is added to the payload as `script_output`.
- Empty stdout, exact `[SILENT]`, `{"__hermes_ignore__": true}`, timeout, missing script, or nonzero exit code returns HTTP 200 with `{"status":"ignored","reason":"script"}`.

### Prompt Templates

Prompts use dot-notation to access nested fields in the webhook payload:

- `{pull_request.title}` resolves to `payload["pull_request"]["title"]`
- `{repository.full_name}` resolves to `payload["repository"]["full_name"]`
- `{__raw__}` — special token that dumps the **entire payload** as indented JSON (truncated at 4000 characters). Useful for monitoring alerts or generic webhooks where the agent needs the full context.
- Missing keys are left as the literal `{key}` string (no error)
- Nested dicts and lists are JSON-serialized and truncated at 2000 characters

You can mix `{__raw__}` with regular template variables:

```yaml
prompt: "PR #{pull_request.number} by {pull_request.user.login}: {__raw__}"
```

If no `prompt` template is configured for a route, the entire payload is dumped as indented JSON (truncated at 4000 characters).

The same dot-notation templates work in `deliver_extra` values.

### Forum Topic Delivery

When delivering webhook responses to Telegram, you can target a specific forum topic by including `message_thread_id` (or `thread_id`) in `deliver_extra`:

```yaml
webhooks:
  routes:
    alerts:
      events: ["alert"]
      prompt: "Alert: {__raw__}"
      deliver: "telegram"
      deliver_extra:
        chat_id: "-1001234567890"
        message_thread_id: "42"
```

If `chat_id` is not provided in `deliver_extra`, the delivery falls back to the home channel configured for the target platform.

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
| `bot-chat` | Sends the completed text final to the same effective profile's canonical Bot Chat as an automation turn. See [Bot Chat delivery](#bot-chat-delivery). |
| `github_comment` | Posts the response as a PR/issue comment via the `gh` CLI. Requires `deliver_extra.repo` and `deliver_extra.pr_number`. The `gh` CLI must be installed and authenticated on the gateway host (`gh auth login`). |
| `telegram` | Routes the response to Telegram. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `discord` | Routes the response to Discord. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `slack` | Routes the response to Slack. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `signal` | Routes the response to Signal. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `sms` | Routes the response to SMS via Twilio. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `whatsapp` | Routes the response to WhatsApp. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `matrix` | Routes the response to Matrix. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `mattermost` | Routes the response to Mattermost. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `homeassistant` | Routes the response to Home Assistant (requires the [`homeassistant` plugin](homeassistant.md)). Uses the home channel (`HASS_HOME_CHANNEL`), or specify `chat_id` in `deliver_extra`. |
| `email` | Routes the response to Email. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `dingtalk` | Routes the response to DingTalk. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `feishu` | Routes the response to Feishu/Lark. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `wecom` | Routes the response to WeCom. Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `weixin` | Routes the response to Weixin (WeChat). Uses the home channel, or specify `chat_id` in `deliver_extra`. |
| `bluebubbles` | Routes the response to BlueBubbles (iMessage). Uses the home channel, or specify `chat_id` in `deliver_extra`. |

For cross-platform delivery, the target platform must also be enabled and connected in the gateway. If no `chat_id` is provided in `deliver_extra`, the response is sent to that platform's configured home channel.

### Bot Chat delivery {#bot-chat-delivery}

```bash
hermes webhook subscribe build-review --route-profile coder \
  --deliver bot-chat:reviewer --prompt 'Review this build: {payload}'
```

I use `bot-chat` for the executing profile's canonical Bot Chat, or
`bot-chat:<existing-local-profile>` for another local profile. I choose only from
existing local profiles, not a path, session, thread, remote machine, or list.
`--route-profile coder` binds ingress and execution to `coder`; it does not choose
the destination. I POST to the reported `/p/coder/webhooks/build-review` URL.
Without an explicit route profile, the effective executing home owns bare
`bot-chat`, including standalone and profile-routed events.

I can select Bot Chat and its target profile in both the dashboard and Desktop.
YAML, dynamic CLI subscriptions, authenticated HTTP creators and MCP event
subscriptions use the same Bot Chat destination validator. MCP consent shows the
complete target. HTTP creation accepts `route_profile` separately from `deliver`;
the optional HTTP `profile` query scopes the subscription store, not the target.
List/readback preserves the route execution profile, complete destination and
canonical profile-prefixed URL. Changing an explicit execution profile requires a
new HMAC secret; omitting it on update preserves the existing binding.

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      secret: "my-hmac-secret"
      routes:
        notify:
          profile: coder
          deliver: bot-chat:reviewer
          deliver_only: true
          prompt: "Build result: {status}"
          mirror_to_session: true
```

I can use agent mode, the existing agent-only coalescer, direct rendered-template
delivery, and actual consumed-hook replies. Deliver only avoids the webhook-side
agent, not all inference: the receiving Bot Chat may run its model and incur cost.
Bot Chat already owns its canonical history; `mirror_to_session` is an accepted
no-op and never appends a duplicate. `deliver_extra` and arbitrary chat IDs remain
rejected. A cron-trigger route still fires the existing job, whose own destination
wins; the route's otherwise ignored `deliver` does not override the job. General
`deliver_only`/cron and coalescing mode conflicts remain rejected.

I receive webhook automation, not a forged human message. Forced secret redaction
applies before durable output is stored or handed off. Blank, whitespace and silent
results remain suppressed, including recovery. Source task outcome stays separate
from transport state. Selected attributable process/delegation completion finals
retain the original verified destination and pin; progress, running updates,
heartbeats and busy-turn summaries do not create new Bot Chat turns.

I retain complete textual Markdown, URLs, media and file references in the redacted
report. This is not binary attachment transfer or a promise that the receiving
profile can read a referenced file. I do not copy files across profile boundaries.

An HTTP `202` is acceptance, not completion. Direct and consumed-hook acknowledgements
include real durable origin/delivery IDs and their observed state. Agent results
retain the original admission-derived delivery ID. `queued` and `claimed` mean
native admission, while `settled` means the receiving turn completed. Neither is
proof that an external business action succeeded.

I persist delivery obligations and reconcile their same native IDs at startup,
hot-serve and bounded housekeeping passes while the producer remains running.
An offline owner can remain pending within the payload lifetime below. Retries revalidate
source authority, the current route and the pinned destination incarnation. They
never rerun source inference or a direct script just to retry transport. An
interrupted preparation can become `origin_unknown`, not permission to rerun an
unknown side effect. A changed/deleted target or revoked source is refused rather
than redirected. The pre-admission coalescing buffer remains in memory.

I distinguish an ended one-shot source from a deleted source. Deleting the source
removes its delivery authority: the next reconciliation refuses an outstanding
intent without sending or reconstructing history. A previously observed `settled`
outcome stays settled; a last observed queued/claimed receipt is not upgraded to
success after deletion. The source task's own outcome is not rewritten.

I compact producer message copies in the existing result-retirement maintenance
pass. Original/completion finals and their linked preparation/process copies share
the source result's existing seven-day horizon after deletion or pruning. A true
standalone direct/hook origin has no source result: its payload uses the same
seven-day horizon from preparation. At the first maintenance pass beyond that
horizon I discard the message and retained source snapshot, and refuse any still
outstanding delivery. Older standalone records without a creation timestamp age
from their first maintenance pass. IDs, digests, destination pins, observed receipt
identities, source outcomes and deduplication guards remain, so compaction never
permits another script/hook execution or transport replay. Secret redaction alone
does not remove ordinary private report content. This producer compaction neither
changes the recipient's native mailbox/history lifetime nor adds a retention daemon.

#### Inspect delivery states

```bash
hermes -p coder webhook deliveries
hermes -p coder webhook deliveries --cursor '<next_cursor>'
```

I can also read authenticated `GET /api/webhooks/deliveries?profile=coder` and
continue with its opaque `cursor`. Each read is a bounded page of at most 50
entries from `webhook.final.intent.v1` and `webhook.origin.v1` in that profile.
I see IDs, origin kind, source outcome, target profile, producer state, last
observed native status/admission ID and an allowlisted reason code. This is a
read-only ledger projection, not a fresh recipient probe or a delivery retry.
No raw message, home path, payload, credential or exception is exposed. Invalid
cursors fail without reading another profile. No unauthenticated diagnostics are
added. The existing MCP `list_event_types` result includes the same bounded
projection restricted to that server's local subscription routes; CLI/HTTP can
continue a returned cursor. I keep authorization on the existing operator surfaces.

### Replying to a delivery {#replying-to-a-delivery}

For Bot Chat, I already have canonical history and `mirror_to_session` does not append another copy. The following mirror behavior applies to other delivery targets.

By default a delivery is fire-and-forget: each webhook event runs in its own session, so if you reply to the delivered message in that chat, the agent there has no record of what was sent. Set `mirror_to_session: true` on the route (or pass `--mirror-to-session` to `hermes webhook subscribe`) and the delivered text is also appended to the target chat's session as `[Webhook delivery: <route>]` followed by the message, so a follow-up ("so he's out?") has the context.

- The mirror is best-effort: it never fails the delivery, and it is skipped when the chat has no gateway session yet (nobody has talked to the agent there).
- On a `/p/<profile>/` route it is written into that profile's session for the chat, never another profile's.
- The mirrored text enters the chat's history as if you had sent it. On a [`deliver_only`](#direct-delivery-mode) route it is the raw rendered payload, so only enable it for sources whose content you trust to sit in your conversation (your own services, not a public issue tracker).

---

## Direct Delivery Mode {#direct-delivery-mode}

By default, every webhook POST triggers an agent run — the payload becomes a prompt, the agent processes it, and the agent's response is delivered. This costs LLM tokens on every event.

For use cases where you just want to **push a plain notification** — no reasoning, no agent loop, just deliver the message — set `deliver_only: true` on the route. The rendered `prompt` template becomes the literal message body, and the adapter dispatches it directly to the configured delivery target.

### When to use direct delivery

- **External service push** — Supabase/Firebase webhook fires on a database change → notify a user in Telegram instantly
- **Monitoring alerts** — Datadog/Grafana alert webhook → push to a Discord channel
- **Inter-agent pings** — Agent A notifies Agent B's user that a long-running task finished
- **Background job completion** — Cron job finishes → post result to Slack

Benefits:

- **No webhook-side model tokens**: I skip the producer agent; a receiving Bot Chat may still run its model
- **No producer reasoning loop**: I render the template before transport; recipient processing may remain queued
- **Same security as agent mode** — HMAC auth, rate limits, idempotency, and body-size limits all still apply
- **Target-specific acknowledgement**: other targets retain their `200`/`502` transport contract. For Bot Chat I receive `202` with durable origin/delivery IDs, not a completion claim.

### Example: Telegram push from Supabase

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644
      secret: "global-secret"
      routes:
        antenna-matches:
          secret: "antenna-webhook-secret"
          deliver: "telegram"
          deliver_only: true
          prompt: "🎉 New match: {match.user_name} matched with you!"
          deliver_extra:
            chat_id: "{match.telegram_chat_id}"
```

Your Supabase edge function signs the payload with HMAC-SHA256 and POSTs to `https://your-server:8644/webhooks/antenna-matches`. The webhook adapter validates the signature, renders the template from the payload, delivers to Telegram, and returns `200 OK`.

### Example: Dynamic subscription via CLI

```bash
hermes webhook subscribe antenna-matches \
  --deliver telegram \
  --deliver-chat-id "123456789" \
  --deliver-only \
  --prompt "🎉 New match: {match.user_name} matched with you!" \
  --description "Antenna match notifications"
```

### Response codes

| Status | Meaning |
|--------|---------|
| `202 Accepted` (Bot Chat) | I receive durable `origin_id`, `origin_kind`, `delivery_id` and observed `status`. I inspect delivery states for later recipient settlement. |
| `200 OK` (other targets) | Delivered successfully. Body: `{"status": "delivered", "route": "...", "target": "...", "delivery_id": "..."}` |
| `200 OK` (status=duplicate) | Duplicate `X-GitHub-Delivery` ID within the idempotency TTL (1 hour). Not re-delivered. |
| `401 Unauthorized` | HMAC signature invalid or missing. |
| `400 Bad Request` | Malformed JSON body. |
| `404 Not Found` | Unknown route name. |
| `413 Payload Too Large` | Body exceeded `max_body_bytes`. |
| `429 Too Many Requests` | Route rate limit exceeded. |
| `502 Bad Gateway` | Target adapter rejected the message or raised. The error is logged server-side; the response body is a generic `Delivery failed` to avoid leaking adapter internals. |

### Configuration gotchas

- `deliver_only: true` requires `deliver` to be a real target. `deliver: log` (or omitting `deliver`) is rejected at startup — the adapter refuses to start if it finds a misconfigured route.
- The `skills` field is ignored in direct delivery mode (no agent runs, so there's nothing to inject skills into).
- Template rendering uses the same `{dot.notation}` syntax as agent mode, including the `{__raw__}` token.
- Idempotency uses the same `X-GitHub-Delivery` / `X-Request-ID` header — retries with the same ID return `status=duplicate` and do NOT re-deliver.

---

## Event-Triggered Cron Jobs {#event-triggered-cron-jobs}

Set `cron_job` on a route to fire an **existing cron job** whenever an event arrives — instead of polling on a fixed cadence or starting a fresh webhook agent session. This turns any scheduled job into an event-driven task: keep the schedule as a fallback sweep (or make it a rarely-firing one) and let the webhook fire it the moment something actually changes.

How it works:

1. The event passes the same HMAC auth, rate limiting, `events`/`filters`/`script` filtering, and idempotency as any other route.
2. The route's `prompt` template is rendered from the payload and injected into the job as **transient per-run context** (the same rail as `cronjob(action='run', prompt=...)` — the job's stored prompt is never mutated).
3. The job fires through the same at-most-once claim the scheduler uses, so a webhook burst cannot double-fire a job that is already running, and the job's own delivery target receives the output.

### Example: fire a PR-review job on review feedback

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        pr-feedback:
          events: ["pull_request_review"]
          secret: "github-webhook-secret"
          cron_job: "pr-review-sweeper"        # existing job ID or name
          prompt: |
            PR #{number} in {repository.full_name} received new review feedback
            from {review.user.login}: {review.body}
```

### Via the CLI

```bash
hermes webhook subscribe pr-feedback \
  --events "pull_request_review" \
  --cron-job "pr-review-sweeper" \
  --prompt "PR #{number} received feedback: {review.body}"
```

The job reference is validated when you create the subscription, so typos surface immediately.

### Notes

- `cron_job` and `deliver_only` are mutually exclusive (the adapter refuses to start if a route sets both). A cron job handles its own delivery.
- The route-level `deliver`, `deliver_extra`, and `skills` fields are ignored on `cron_job` routes — the job's own settings apply.
- Paused/disabled jobs are not fired; the event is logged and dropped.
- The POST returns `202 Accepted` immediately; the job runs in the background.

---

## Dynamic Subscriptions (CLI) {#dynamic-subscriptions}

In addition to static routes in `config.yaml`, you can create webhook subscriptions dynamically using the `hermes webhook` CLI command. This is especially useful when the agent itself needs to set up event-driven triggers.

### Create a subscription

```bash
hermes webhook subscribe github-issues \
  --events "issues" \
  --prompt "New issue #{issue.number}: {issue.title}\nBy: {issue.user.login}\n\n{issue.body}" \
  --deliver telegram \
  --deliver-chat-id "-100123456789" \
  --description "Triage new GitHub issues"
```

This returns the webhook URL and an auto-generated HMAC secret. Configure your service to POST to that URL.

### List subscriptions

```bash
hermes webhook list
```

### Remove a subscription

```bash
hermes webhook remove github-issues
```

### Test a subscription

```bash
hermes webhook test github-issues
hermes webhook test github-issues --payload '{"issue": {"number": 42, "title": "Test"}}'
```

### How dynamic subscriptions work

- Subscriptions are stored in `~/.hermes/webhook_subscriptions.json`
- The webhook adapter hot-reloads a coherent snapshot of this file on each incoming request (stat-gated (mtime/size/inode), lock-free)
- Static routes from `config.yaml` always take precedence over dynamic ones with the same name
- Dynamic subscriptions use the same route format and capabilities as static routes (events, prompt templates, skills, delivery)
- Create, update, enable/disable, and remove operations are serialized and atomically replace the store; a stale concurrent update is rejected rather than restoring a route another operation removed or disabled
- Rebinding an existing route with `--route-profile` rotates its HMAC secret automatically. A caller with the old profile's secret cannot authenticate the rebound route; pass a different `--secret` only when the receiving service requires an operator-chosen value
- No gateway restart required — subscribe and it's immediately live

### Agent-driven subscriptions

The agent can create subscriptions via the terminal tool when guided by the `webhook-subscriptions` skill. Ask the agent to "set up a webhook for GitHub issues" and it will run the appropriate `hermes webhook subscribe` command.

---

## Per-route toolsets {#per-route-toolsets}

Webhook agent runs default to a deliberately constrained toolset (`web_search`, `web_extract`, `vision_analyze`, `clarify`) because webhook payloads can carry untrusted third-party content — a public PR title or issue comment should never be able to prompt-inject its way into your terminal.

For **trusted** routes — a localhost monitoring daemon pushing system alerts, an internal CI system — you can grant a wider toolset to that route only, without widening every other webhook route:

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        oom-emergency:
          secret: "monitor-secret"
          prompt: "Memory emergency: {detail}. Diagnose with ps/free/py-spy and report."
          toolsets: ["terminal", "file", "code_execution", "web"]
          deliver: "telegram"
```

For dynamic subscriptions, add the `toolsets` key by editing `~/.hermes/webhook_subscriptions.json` directly:

```json
{
  "oom-emergency": {
    "secret": "...",
    "prompt": "...",
    "toolsets": ["terminal", "file", "web"],
    "deliver": "telegram"
  }
}
```

Behavior and safety properties:

- The route list **replaces** the platform-level webhook toolset resolution for that route's runs (it is not merged).
- Names are validated through the same path as `platform_toolsets` config — unknown names and platform-restricted toolsets are dropped.
- `hermes webhook subscribe` deliberately does **not** accept a toolsets flag. Granting elevated tools is a manual config-file edit, so an agent creating its own subscription at runtime cannot self-grant `terminal`.
- Only grant elevated toolsets to routes whose senders you fully control, with a real HMAC secret. Anyone who can POST a validly-signed payload to that route is effectively running an agent with those tools.

---

## Security {#security}

The webhook adapter includes multiple layers of security:

### HMAC signature validation

The adapter validates incoming webhook signatures using the appropriate method for each source:

- **GitHub**: `X-Hub-Signature-256` header — HMAC-SHA256 hex digest prefixed with `sha256=`
- **GitLab**: `X-Gitlab-Token` header — plain secret string match
- **Standard Webhooks**: `webhook-id`, `webhook-timestamp`, and `webhook-signature` headers — signed content is `{id}.{timestamp}.{raw_body}` with a `v1,<base64-hmac-sha256>` signature
- **Generic (V2, recommended)**: `X-Webhook-Signature-V2` + `X-Webhook-Timestamp` headers — HMAC-SHA256 hex digest of `<timestamp>.<body>`. The timestamp (Unix seconds) must be within ±300 seconds of the server clock, which prevents captured requests from being replayed later.
- **Generic (V1, legacy)**: `X-Webhook-Signature` header — raw HMAC-SHA256 hex digest of the body only. Still accepted for backward compatibility, but it has no replay protection (a captured request replays indefinitely); the gateway logs a deprecation warning once per route. Switch senders to V2.

If a secret is configured but no recognized signature header is present, the request is rejected.

### Secret is required

Every route must have a secret — either set directly on the route or inherited from the global `secret`. Routes without a secret cause the adapter to fail at startup with an error. For development/testing only, you can set the secret to `"INSECURE_NO_AUTH"` to skip validation entirely.

When multi-profile routing is enabled, the route's `profile` field also
binds that secret to one execution target. A route without `profile` is
default-profile-only. A request carrying a valid route signature is still
rejected if its `/p/<profile>/` prefix does not match the route binding.

`INSECURE_NO_AUTH` is only accepted when the gateway is bound to a loopback host (`127.0.0.1`, `localhost`, `::1`). If it is combined with a non-loopback bind such as `0.0.0.0` or a LAN IP, the adapter refuses to start — this prevents accidentally exposing an unauthenticated endpoint on a public interface.

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

Delivery IDs (from `X-GitHub-Delivery`, `svix-id`, `webhook-id`, `X-Request-ID`, or a random per-request ID) are cached for **1 hour**. Duplicate deliveries (e.g. webhook retries) are silently skipped with a `200` response, preventing duplicate agent runs.

### Body size limits

Payloads exceeding **1 MB** are rejected before the body is read. Configure this:

```yaml
platforms:
  webhook:
    extra:
      max_body_bytes: 2097152  # 2 MB
```

### Authenticated does not mean trusted

:::warning
**HMAC validation authenticates the _sender_, not the _content_.** A valid signature only proves the request came from a party holding the route's secret (e.g. GitHub). It says nothing about who wrote the _business fields_ inside the payload — PR titles, commit messages, issue descriptions, and any other upstream text are authored by arbitrary third parties and must be treated as untrusted.

This is the same trust model that applies to everything the agent reads: web pages, files, and tool output are all untrusted input. Hermes does not — and cannot reliably — sanitize untrusted text with a blocklist; phrasing, encoding, and translation make that trivially bypassable. **The trust boundary is the agent's capability surface, not the input channel.** Harden there:

- **Sandbox the runtime.** Run the gateway with the Docker or SSH terminal backend (or in a VM) when exposed to the internet, so a hijacked turn cannot touch the host.
- **Scope the toolset.** Disable `terminal`, `file`, and outbound-action tools on webhook-triggered sessions if the route only needs to read and summarize. Fewer capabilities means a smaller blast radius if a payload field carries injected instructions.
- **Keep approvals on** for any destructive or outbound operation, so an injected instruction cannot act unattended.
- **Template narrowly.** Prefer a specific `prompt` with named fields (`{pull_request.title}`) over `{__raw__}` or an empty template that dumps the whole payload, so only the fields you intend reach the prompt.
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

- The idempotency cache should prevent this — check that the webhook source is sending a delivery ID header (`X-GitHub-Delivery`, `svix-id`, `webhook-id`, or `X-Request-ID`)
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
