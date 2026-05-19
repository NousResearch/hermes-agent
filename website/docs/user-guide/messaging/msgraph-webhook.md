---
sidebar_position: 23
title: "Microsoft Graph Webhook Listener"
description: "Receive Microsoft Graph change notifications (meetings, calendar, chat, etc.) in Hermes"
---

# Microsoft Graph Webhook Listener

The `msgraph_webhook` gateway platform is an inbound event listener. It's how Hermes receives **change notifications** from Microsoft Graph — "a Teams meeting ended," "a new message landed in this chat," "this calendar event was updated." Different from the `teams` platform (which is a chat bot users type to) — this one is M365 telling Hermes something happened, not a person.

Right now the primary consumer is the Teams meeting summary pipeline: Graph notifies when a meeting produces a transcript, the pipeline fetches it, and Hermes posts a summary back into Teams. Other Graph resources (`/chats/.../messages`, `/users/.../events`) use the same listener — the pipeline consumers land with their own PRs.

## Prerequisites

- Microsoft Graph application credentials — [Register a Microsoft Graph Application](/docs/guides/microsoft-graph-app-registration)
- A **public HTTPS URL** that Microsoft Graph can reach (Graph does not call private endpoints). A dev tunnel works for testing; production needs a real domain with a valid certificate.
- A strong shared secret to use as the `clientState` value. Generate with `openssl rand -hex 32` and put it in `~/.hermes/.env` as `MSGRAPH_WEBHOOK_CLIENT_STATE`.

## Quick Start

Minimum `~/.hermes/config.yaml`:

```yaml
platforms:
  msgraph_webhook:
    enabled: true
    extra:
      port: 8646
      client_state: "replace-with-a-strong-secret"
      accepted_resources:
        - "communications/onlineMeetings"
```

Or via env vars in `~/.hermes/.env` (auto-merged on startup):

```bash
MSGRAPH_WEBHOOK_ENABLED=true
MSGRAPH_WEBHOOK_PORT=8646
MSGRAPH_WEBHOOK_CLIENT_STATE=<generate-with-openssl-rand-hex-32>
MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES=communications/onlineMeetings
```

Start the gateway: `hermes gateway run`. The listener exposes:

- `POST /msgraph/webhook` — change notifications from Graph
- `GET /msgraph/webhook?validationToken=...` — Graph subscription validation handshake
- `GET /health` — readiness probe with accepted/duplicate counters

Expose the listener publicly (reverse proxy, dev tunnel, ingress). Your notification URL for Graph subscriptions is your public HTTPS origin followed by `/msgraph/webhook`:

```
https://ops.example.com/msgraph/webhook
```

## Configuration

All settings go under `platforms.msgraph_webhook.extra`:

| Setting | Default | Description |
|---------|---------|-------------|
| `host` | `0.0.0.0` | Bind address for the HTTP listener. |
| `port` | `8646` | Bind port. |
| `webhook_path` | `/msgraph/webhook` | URL path Graph POSTs to. |
| `health_path` | `/health` | Readiness endpoint. |
| `client_state` | — | Shared secret Graph echoes in every notification. Compared with `hmac.compare_digest` — generate with `openssl rand -hex 32`. |
| `accepted_resources` | `[]` (accept all) | Allowlist of Graph resource paths/patterns. Trailing `*` acts as prefix match. Leading `/` is tolerated. Example: `["communications/onlineMeetings", "chats/*/messages"]`. |
| `max_seen_receipts` | `5000` | Dedupe cache size for notification IDs. Oldest entries evicted when the cap is hit. |
| `allowed_source_cidrs` | `[]` (allow all) | Optional source-IP allowlist. See below. |

Each setting also has an equivalent env var (`MSGRAPH_WEBHOOK_*`) that merges into the config at gateway startup — see the [environment variables reference](/docs/reference/environment-variables#microsoft-graph-teams-meetings).

## Security Hardening

### clientState is the primary auth check

Every Graph notification includes the `clientState` string your subscription registered with. The listener rejects any notification whose `clientState` doesn't match, using timing-safe comparison. This is Microsoft's documented mechanism — treat the value as a strong shared secret.

If `client_state` is unset, the listener accepts every well-formed POST. **Don't run without it in production.**

### Source-IP allowlisting (production deployments)

For production, restrict the listener to Microsoft's published Graph webhook source IP ranges. Microsoft documents the egress ranges under the [Office 365 IP Address and URL Web service](https://learn.microsoft.com/en-us/microsoft-365/enterprise/urls-and-ip-address-ranges). Configure them as:

```yaml
platforms:
  msgraph_webhook:
    enabled: true
    extra:
      client_state: "..."
      allowed_source_cidrs:
        - "52.96.0.0/14"
        - "52.104.0.0/14"
        # ...add the current Microsoft 365 "Common" + "Teams" category egress ranges
```

Or as an env var:

```bash
MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS="52.96.0.0/14,52.104.0.0/14"
```

Empty allowlist = accept from anywhere (default; preserves dev-tunnel workflows). Invalid CIDR strings log a warning and are ignored. **Review the Microsoft IP list quarterly** — it changes.

### HTTPS termination

The listener speaks plain HTTP. Terminate TLS at your reverse proxy (Caddy, Nginx, Cloudflare Tunnel, AWS ALB) and proxy to the listener over the local network. Graph refuses to deliver to non-HTTPS endpoints, so there's no path for unencrypted traffic to reach you from Graph itself.

### Response hygiene

On success the listener returns `202 Accepted` with an empty body — internal counters stay out of the wire response. Operators can observe counts via `/health`.

Status code table:

| Outcome | Status |
|---------|--------|
| Notification(s) accepted or deduped | 202 |
| Validation handshake (GET with `validationToken`) | 200 (echoes the token) |
| Every item in batch failed clientState | 403 |
| Malformed JSON / missing `value` array / unknown resource | 400 |
| Source IP not in allowlist | 403 |
| Bare GET without `validationToken` | 400 |

## Troubleshooting

| Problem | What to check |
|---------|---------------|
| Graph subscription validation fails | Public URL is reachable, `/msgraph/webhook` path matches, GET with `validationToken` echoes the token verbatim as `text/plain` within 10 seconds. |
| Notifications POST but nothing ingests | `client_state` matches what you registered the subscription with. Re-run `openssl rand -hex 32` and create a new subscription if the value drifted. Check `accepted_resources` includes the resource path Graph is sending. |
| Every notification 403s | `clientState` mismatch (forged, or subscription registered with a different value). Re-create the subscription with `hermes teams-pipeline subscribe --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE" ...` (ships with the pipeline runtime PR). |
| Listener starts but `curl http://localhost:8646/health` hangs | Port binding collision. Check `ss -tlnp \| grep 8646` and change `port:` if needed. |
| Real Graph requests from Microsoft get 403'd | Source IP allowlist is too narrow. Remove `allowed_source_cidrs` temporarily, confirm traffic flows, then widen the list to include the current Microsoft egress ranges. |

## Related Docs

- [Register a Microsoft Graph Application](/docs/guides/microsoft-graph-app-registration) — Azure app registration prereq
- [Environment Variables → Microsoft Graph](/docs/reference/environment-variables#microsoft-graph-teams-meetings) — full env var list
- [Microsoft Teams bot setup](/docs/user-guide/messaging/teams) — the different platform that lets users chat with Hermes in Teams
