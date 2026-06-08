---
sidebar_position: 18
---

# Photon iMessage

Connect Hermes to **iMessage** through [Photon][photon], a managed
service that handles the Apple line allocation and abuse-prevention
layer so you don't have to run your own Mac relay.

The free tier uses Photon's shared iMessage line pool — different
recipients may see different sending numbers, but each conversation
stays stable. The paid Business tier gives every user the same
dedicated number; the plugin supports both, and the free tier is the
recommended starting point.

:::info Free to start
Photon's shared-line pool is free. No subscription is required to send
your first iMessage from Hermes — just a phone number we can bind to
your account.
:::

## Architecture

Inbound messages arrive as **signed webhooks**: Photon POSTs JSON with
an `X-Spectrum-Signature` header to a URL you register, and Hermes'
aiohttp listener verifies the HMAC-SHA256 signature before dispatching
the event into the agent.

Outbound replies go through a small supervised **Node sidecar** that
runs the `spectrum-ts` SDK on loopback. Photon does not currently
expose a public HTTP send-message endpoint — that's a roadmap item on
their side — so until then the sidecar is the only way to call
`Space.send(...)`. The Python plugin starts, supervises, and shuts
down the sidecar automatically. When Photon ships an HTTP send
endpoint we'll retire the sidecar in a follow-up release.

## Prerequisites

- A Photon account — sign up at [app.photon.codes][app]
- **Node.js 18.17 or newer** on PATH (`node --version`)
- A phone number that can receive iMessage (used to bind your account)
- A publicly reachable URL for the webhook receiver — Cloudflare
  Tunnel, ngrok, or your own gateway hostname all work

## First-time setup

Either run the unified gateway wizard and pick **Photon iMessage**:

```bash
hermes gateway setup
```

…or run the Photon setup directly (the wizard calls the same flow):

```bash
# Device-code login + project + user + sidecar deps, all in one
hermes photon setup --phone +15551234567
```

The setup:

1. Opens `https://app.photon.codes/` for device approval
2. Creates a Spectrum-enabled project under your account
3. Calls the Spectrum `create-user` endpoint with `type: shared` so
   Photon allocates an iMessage line from the free pool
4. Runs `npm install` inside the plugin's sidecar directory

Credentials are stored in `~/.hermes/auth.json` under
`credential_pool.photon` (bearer token) and
`credential_pool.photon_project` (project id + secret).

## Authorizing users

Photon uses the same authorization model as every other Hermes
channel. Choose one approach:

**DM pairing (default).** When an unknown number messages your Photon
line, Hermes replies with a pairing code. Approve it with:

```bash
hermes pairing approve photon <CODE>
```

Use `hermes pairing list` to see pending codes and approved users.

**Pre-authorize specific numbers** (in `~/.hermes/.env`):

```bash
PHOTON_ALLOWED_USERS=+15551234567,+15559876543
```

**Open access** (dev only, in `~/.hermes/.env`):

```bash
PHOTON_ALLOW_ALL_USERS=true
```

When `PHOTON_ALLOWED_USERS` is set, unknown senders are silently
ignored rather than offered a pairing code (the allowlist signals you
deliberately restricted access).

### Require mentions in group chats

By default Hermes responds to every authorized DM and group message.
To make group chats opt-in, enable mention gating (DMs still always
work):

```yaml
gateway:
  platforms:
    photon:
      enabled: true
      require_mention: true
```

With `require_mention: true`, group-chat messages are ignored unless
they match a wake-word pattern. The defaults match `Hermes` and
`@Hermes agent` variants. For a custom agent name, set regex patterns:

```yaml
gateway:
  platforms:
    photon:
      require_mention: true
      mention_patterns:
        - '(?<![\w@])@?amos\b[,:\-]?'
```

Both keys also accept env vars (`PHOTON_REQUIRE_MENTION`,
`PHOTON_MENTION_PATTERNS`). This is the same mention-gating model the
BlueBubbles iMessage channel uses.

## Registering the webhook

Photon needs a public URL it can POST to. Expose your local listener
(default port 8788, path `/photon/webhook`) via Cloudflare Tunnel or
ngrok, then:

```bash
hermes photon webhook register https://YOUR-PUBLIC-URL/photon/webhook
```

The response includes a `signingSecret` — **Photon only returns it
once.** Save it to `~/.hermes/.env`:

```bash
PHOTON_WEBHOOK_SECRET=v0_64-char-hex...
```

The plugin verifies every inbound `POST` against this secret and
rejects deliveries with a timestamp drift greater than 5 minutes.

## Start the gateway

```bash
hermes gateway start --platform photon
```

You'll see something like:

```
[photon] connected — webhook at 0.0.0.0:8788/photon/webhook, sidecar on 127.0.0.1:8789
```

Send an iMessage to your assigned number and Hermes will reply.

## Status & troubleshooting

```bash
hermes photon status
```

Prints:

```
Photon iMessage status
──────────────────────
  device token        : ✓ stored
  project id          : 3c90c3cc-0d44-4b50-...
  project key         : ✓ stored
  webhook key         : ✓ set
  node binary         : /usr/bin/node
  sidecar deps        : ✓ installed
```

Common issues:

- **`sidecar deps : ✗ run hermes photon install-sidecar`** — Node is
  installed but `spectrum-ts` isn't. Run the suggested command.
- **`webhook key : ⚠ unset — verification disabled`** — the
  plugin will accept ANY POST to the webhook URL, which is unsafe.
  Re-run `hermes photon webhook register` and store the secret.
- **`PHOTON_WEBHOOK_PORT` already in use** — set a different port via
  `~/.hermes/.env`.
- **Webhook reachable from localhost but Photon can't deliver** —
  Photon needs a public hostname. Cloudflare Tunnel is the easiest
  free option.

## Webhook management

```bash
hermes photon webhook list                  # show registered hooks
hermes photon webhook delete <webhook-id>   # remove one
```

## Limits today

- **Attachments are metadata-only.** Inbound webhooks carry the
  filename + MIME type but no download URL — Photon documents an
  attachment retrieval endpoint as roadmap.
- **Outbound attachments not wired yet.** Easy to add in the sidecar
  once the agent has reason to send them.
- **Photon's free quotas:** 5,000 messages per server per day,
  50 new-conversation initiations per shared line per day. Increases
  available — email `help@photon.codes`.

## Env vars

| Variable                  | Default            | Notes                                      |
|---------------------------|--------------------|--------------------------------------------|
| `PHOTON_PROJECT_ID`       | from `auth.json`   | Set by `hermes photon setup`               |
| `PHOTON_PROJECT_SECRET`   | from `auth.json`   | Set by `hermes photon setup`               |
| `PHOTON_WEBHOOK_SECRET`   | (unset)            | From `hermes photon webhook register`      |
| `PHOTON_WEBHOOK_PORT`     | `8788`             | Local port for the aiohttp listener        |
| `PHOTON_WEBHOOK_PATH`     | `/photon/webhook`  | Path under which the listener mounts       |
| `PHOTON_WEBHOOK_BIND`     | `0.0.0.0`          | Bind address for the listener              |
| `PHOTON_SIDECAR_PORT`     | `8789`             | Loopback port for sidecar control          |
| `PHOTON_SIDECAR_AUTOSTART`| `true`             | Whether the adapter spawns the sidecar     |
| `PHOTON_NODE_BIN`         | `which node`       | Override the Node binary path              |
| `PHOTON_HOME_CHANNEL`     | (unset)            | Default space ID for cron / notifications  |
| `PHOTON_HOME_CHANNEL_NAME`| (unset)            | Human label for the home channel           |
| `PHOTON_ALLOWED_USERS`    | (unset)            | Comma-separated E.164 allowlist            |
| `PHOTON_ALLOW_ALL_USERS`  | `false`            | Dev only — accept any sender               |
| `PHOTON_REQUIRE_MENTION`  | `false`            | Require a wake word before responding in groups |
| `PHOTON_MENTION_PATTERNS` | Hermes wake words  | JSON list / comma / newline regex patterns for group mentions |
| `PHOTON_API_HOST`         | `spectrum.photon.codes` | Override the Spectrum management API host |
| `PHOTON_DASHBOARD_HOST`   | `app.photon.codes` | Override the dashboard / device-login host |

[photon]: https://photon.codes/
[app]: https://app.photon.codes/
