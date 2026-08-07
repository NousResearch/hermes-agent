---
sidebar_position: 30
title: "Hermes Relay"
description: "Connect Hermes to messaging platforms through a relay connector that owns the platform credentials — enrollment, capabilities, config, and troubleshooting"
---

# Hermes Relay (Connector)

:::warning Experimental
Relay is **experimental**. The wire contract, auth scheme, and configuration
may change without a deprecation cycle while the system is being validated.
:::

Hermes Relay is not a chat platform itself — it is a **connector system** that
lets your gateway front one or more real messaging platforms (Discord,
Telegram, Slack, WhatsApp, …) **without holding any platform credentials**. A
separate service, the *connector*, owns the platform bot tokens and sockets.
Your gateway dials **out** to the connector over a single authenticated
WebSocket, receives a capability descriptor at handshake, and then exchanges
normalized message events (inbound) and actions (outbound) over that socket.

Key properties:

- **Outbound-only networking.** The gateway never opens an inbound port.
  Inbound messages ride back down the same WebSocket the gateway dialed, so
  relay works behind NAT and on hosts with no public IP.
- **No platform secrets on the gateway.** Bot tokens live on the connector.
  Auth-gated platform media URLs are re-hosted connector-side, so platform
  credentials never cross the wire.
- **Platform-agnostic.** The gateway learns what the fronted platform can do
  (message length limits, markdown dialect, edit/thread/streaming support, and
  the exact set of supported operations) from the handshake descriptor, not
  from hardcoded platform logic.

The formal gateway ⇄ connector interface lives in the repository at
`docs/relay-connector-contract.md`.

## When to use Relay

Relay is for deployments where a hosted or shared connector service manages
the platform side — for example multi-tenant hosting where one shared bot
fronts many users' agents, or setups where you don't want bot tokens on the
gateway machine. If you run your own bots directly, use the native platform
adapters ([Telegram](/user-guide/messaging/telegram),
[Discord](/user-guide/messaging/discord), etc.) instead.

## Enrollment

A self-hosted gateway authenticates to the connector with a per-gateway
secret. `hermes gateway enroll` redeems a **single-use enrollment token**
(minted by the connector when your tenant's route is provisioned and delivered
with your gateway config) for that secret:

```bash
hermes gateway enroll \
  --token <enrollment-token> \
  --connector-url wss://connector.example.com/relay
```

What it does:

1. Resolves a fresh Nous Portal access token from your existing login
   (`~/.hermes/auth.json`) — this proves which Nous org (tenant) you own. If
   `gateway.idp.token_url` is configured, a generic OAuth2 client-credentials
   token from your own IdP is used instead (the air-gapped / self-hosted-IdP
   path, no Nous Portal involved).
2. POSTs the enrollment token and a gateway id to the connector's
   `/relay/enroll` endpoint over TLS.
3. The connector verifies the token (signature, single-use, tenant match),
   mints a per-gateway secret plus a per-tenant delivery key, and returns them
   once.
4. Persists the credentials into `~/.hermes/.env`:
   `GATEWAY_RELAY_ID`, `GATEWAY_RELAY_SECRET`, `GATEWAY_RELAY_DELIVERY_KEY`
   (plus `GATEWAY_RELAY_URL` / `GATEWAY_RELAY_WAKE_URL` when supplied).

Restart the gateway afterwards to pick up the new environment.

Flags:

| Flag | Description |
|------|-------------|
| `--token` | The single-use enrollment token. Also settable via `GATEWAY_RELAY_ENROLL_TOKEN`. |
| `--connector-url` | Connector base or relay URL (`wss://…/relay` or `https://…`). Also settable via `GATEWAY_RELAY_URL` or `gateway.relay_url` in `config.yaml`. |
| `--gateway-id` | Stable id for this gateway instance (used for kill-switch granularity). Defaults to `gw-<hostname>`. |
| `--wake-url` | Optional reachable URL the connector pokes (payload-free GET) to wake this gateway when buffered work arrives while it is idle. Persisted as `GATEWAY_RELAY_WAKE_URL`. Without it the gateway still drains buffered messages whenever it next reconnects. |

:::note Managed installs
`hermes gateway enroll` refuses to run in managed/hosted installs — there the
hosting platform provisions the relay secret directly into the container
environment.
:::

## Configuration

Relay activates when a connector relay URL is configured — there is no
separate feature flag. Deployments that don't set it are unaffected.

| Setting | Where | Meaning |
|---------|-------|---------|
| `GATEWAY_RELAY_URL` | env (`~/.hermes/.env`) | Connector relay WebSocket URL. Presence enables the relay platform. |
| `gateway.relay_url` | `config.yaml` | Same as above, config-file form (env takes precedence). |
| `GATEWAY_RELAY_ID` | env | This gateway instance's id (written by `enroll`). |
| `GATEWAY_RELAY_SECRET` | env | Per-gateway secret authenticating the WebSocket upgrade (written by `enroll`). |
| `GATEWAY_RELAY_DELIVERY_KEY` | env | Per-tenant delivery key (written by `enroll`; retained for forward-compat). |
| `GATEWAY_RELAY_WAKE_URL` / `gateway.relay_wake_url` | env / `config.yaml` | Optional wake-poke target for idle/suspended gateways. |
| `GATEWAY_RELAY_PLATFORMS` | env | Comma-separated list of platforms this gateway fronts over one connection (e.g. `discord,telegram`). Usually stamped by the deployment/orchestrator. |
| `GATEWAY_RELAY_BOT_IDS` | env | JSON map of per-platform bot identities, e.g. `{"discord": {"botId": "…"}}`. Paired with `GATEWAY_RELAY_PLATFORMS`. |
| `gateway.idp.token_url` | `config.yaml` | When set, enrollment/provisioning authenticates via generic OAuth2 client-credentials against your own IdP instead of Nous Portal. |

## Supported capabilities

What actually works over a relay connection is negotiated at handshake: the
connector advertises a `supported_ops` list, and the gateway only uses an
operation the connector explicitly advertises (older connectors fall back to a
legacy `send`/`edit`/`typing`/`follow_up` set). Per-platform capability flags
(edit-based streaming, threads, draft streaming, markdown dialect, message
length limit) also come from the handshake descriptor. Subject to that
negotiation, the relay supports:

- **Text messages and streaming** — sends, replies, and progressive
  edit-based streaming when the fronted platform supports message editing;
  otherwise output degrades to one message per segment.
- **Media, both directions** — outbound images, voice, audio, video, and
  documents are uploaded to the connector (or referenced by public URL) and
  delivered through each platform's native upload lane, with captions.
  Inbound attachments are localized to files for the agent; auth-gated
  platform URLs are re-hosted connector-side so platform credentials never
  reach the gateway. Media re-hosts are capped at 25 MB and expire (~1 hour).
- **Native interactive prompts** — exec approvals, confirmations, and clarify
  questions render with **native platform controls** (Discord buttons,
  Telegram inline keyboards, Slack Block Kit actions, WhatsApp button/list
  messages) instead of numbered-text fallbacks. Button presses come back as
  authenticated prompt responses from the actual clicking user, so the
  gateway's authorization gates apply exactly as to a typed reply. Prompt
  expiry is enforced gateway-side.
- **Reaction ack lifecycle** — the bot's processing-status reactions
  (👀 while working, ✅/❌ on completion) work over the relay. Reactions are
  best-effort: a failed reaction never fails a turn.
- **Thread lifecycle** — creating handoff threads and renaming threads
  (including LLM-titled semantic renames) through platform-abstract
  `thread_create` / `thread_rename` operations, with a no-clobber guard so a
  human's manual rename wins. Availability depends on the platform (e.g.
  Slack threads can't be renamed; WhatsApp has no threads).
- **Typing indicators** — the gateway egresses typing (and stop-typing) through
  the connector while processing.
- **Chat metadata** — `get_chat_info` lookups are proxied to the connector
  when advertised.
- **Buffered delivery and wake** — when the gateway goes idle or disconnects,
  the connector buffers inbound messages durably and replays them in order on
  reconnect (ack-gated, no loss or duplication). If a wake URL is registered,
  the connector pokes it when buffered work arrives for a sleeping gateway.

Multi-platform fronting is supported: one gateway can front several platforms
(e.g. Discord *and* Telegram) over a single relay connection, with each
outbound message tagged for the platform it targets.

## Troubleshooting

**Enrollment fails with 401** — the connector could not verify your identity
token. Re-login with `hermes auth add nous` (or `hermes setup`) and retry.

**Enrollment fails with 403** — the enrollment token is invalid, expired,
already used, or belongs to a different tenant. Enrollment tokens are
single-use; request a fresh one from whoever provisioned your tenant route.

**"Could not reach the connector"** — check the connector URL. You can paste
either the `wss://…/relay` dial URL or the `https://…` base URL; the CLI maps
between them automatically.

**`enroll` refuses to run** — you are in a managed/hosted install, where the
relay secret is provisioned by the hosting platform. Self-enrollment is only
for self-hosted gateways.

**Relay platform shows as disabled after it previously worked** — a WebSocket
close with code 4401 *after* a successful handshake means the gateway's secret
was revoked (e.g. the instance was deprovisioned). The gateway deliberately
stops reconnecting and reports relay as disabled rather than retrying. A 4401
*before* any successful handshake is treated as a transient
not-yet-provisioned race and retried normally.

**Nothing changed after enrolling** — the gateway reads `GATEWAY_RELAY_*` at
startup. Restart it (`hermes gateway restart`).

**A feature (buttons, media, threads…) silently degrades to plain text** — the
connector for your platform did not advertise that operation in its handshake
`supported_ops`. The gateway intentionally falls back to the text behavior
rather than sending an op the connector can't handle.
