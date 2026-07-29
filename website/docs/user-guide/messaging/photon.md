---
sidebar_position: 18
---

# Photon iMessage

Connect Hermes to **iMessage** through [Photon Spectrum][photon]. Use either
Photon's managed line service or Spectrum's local macOS provider with the
iMessage account already signed into Messages on your Mac.

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

Photon is a **persistent-connection** channel, like Discord or Slack —
**no webhook, no public URL, no signing secret to manage.**

Because the `spectrum-ts` SDK is TypeScript-only, Hermes runs it in a small
supervised **Node sidecar** and talks to it over loopback. Cloud mode uses a
long-lived gRPC stream to Photon; local mode reads and sends through the Mac's
Messages account:

- **Inbound** — the sidecar consumes the SDK's `app.messages` stream and
  forwards each message to the Python adapter over a loopback `GET /inbound`
  (NDJSON). The adapter dedupes and dispatches it to the agent, reconnecting
  automatically if the stream drops.
- **Outbound** — replies are loopback POSTs to the sidecar, which calls
  `space.send(...)` on the SDK.

The Python plugin normally starts, supervises, and shuts down the sidecar
automatically. Advanced local installations can supervise one shared sidecar
separately and connect multiple Hermes gateways to it.

## Prerequisites

- **Node.js 18.17 or newer** on PATH (`node --version`)
- For **Photon Cloud**: a [Photon account][app] and a phone number that can
  receive iMessage (used to bind your account)
- For **local mode**: macOS, Messages signed into the account Hermes should
  use, and Full Disk Access for the process that runs the Node sidecar so it
  can read `~/Library/Messages/chat.db`

That's it — there is no public URL or tunnel to set up.

## Photon Cloud setup

Either run the unified gateway wizard and pick **Photon iMessage**:

```bash
hermes gateway setup
```

…or run the Photon setup directly (the wizard calls the same flow):

```bash
# Device-code login + project + user + sidecar deps, all in one
hermes photon setup --phone +15551234567
```

The setup, in order:

1. **Device login** (`client_id=photon-cli`) — opens
   `https://app.photon.codes/` for approval and stores the bearer token.
2. **Finds or creates** the `Hermes Agent` project on your account.
3. **Enables Spectrum**, reads the project's Spectrum id, and rotates
   the project secret.
4. **Registers your phone number** as a Spectrum user — skipped if a
   user with that number already exists, so re-running is safe.
5. **Prints your assigned iMessage line** — the number you text to reach
   your agent.
6. **Runs `npm install`** inside the plugin's sidecar directory. On
   read-only / immutable install trees (hosted Docker images, Podman,
   Nix) the sidecar automatically falls back to a writable mirror under
   `~/.hermes/photon/sidecar`; set `PHOTON_SIDECAR_DIR` to pin an
   explicit location.

Runtime credentials are written to `~/.hermes/.env`
(`PHOTON_PROJECT_ID` = the Spectrum project id, `PHOTON_PROJECT_SECRET`),
the same place every other channel keeps its token. Management metadata
(device token, dashboard project id) lives in `~/.hermes/auth.json` under
`credential_pool.photon` / `credential_pool.photon_project`.

## Local macOS setup

Local mode uses Spectrum's separate `@spectrum-ts/imessage-local` provider and
does **not** require a Photon account, project id, project secret, or managed
line. Install the sidecar dependencies, then enable local mode in
`config.yaml`:

```bash
hermes photon install-sidecar
```

```yaml
platforms:
  photon:
    enabled: true
    extra:
      local: true
```

Restart the gateway after changing the environment or configuration. The Node
sidecar process needs Full Disk Access; macOS may also request Automation
permission when it sends through Messages. If you do not want to grant those
permissions to a general-purpose `node` executable, set `PHOTON_NODE_BIN` to a
dedicated signed runtime.

For an externally supervised sidecar, set `sidecar_url` and
`autostart_sidecar: false` under `platforms.photon.extra`, and use the same
`PHOTON_SIDECAR_TOKEN` in both processes. The endpoint should remain
loopback-only or otherwise be protected by transport security and the sidecar
token. An optional `allowed_chat_ids` list in the same `extra` mapping can hard
limit this Hermes instance to assigned local Messages chat GUIDs.

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

## Start the gateway

```bash
hermes gateway start
```

You'll see something like:

```
[photon] connected — sidecar on 127.0.0.1:8789, streaming inbound over gRPC
```

Send an iMessage to your assigned number and Hermes will reply.

## Status & troubleshooting

```bash
hermes photon status
```

Prints saved credentials, sidecar health, your registered number, and the
assigned iMessage line Hermes uses. When a Photon token and dashboard project
are available, `status` refreshes missing number rows from the dashboard
without provisioning new lines.

```
Photon iMessage status
──────────────────────
  device token        : ✓ stored
  dashboard project   : 3c90c3cc-0d44-4b50-...
  spectrum project id : sp-...
  project secret      : ✓ stored
  my number           : +15551234567
  assigned number     : +16282679185
  node binary         : /usr/bin/node
  sidecar deps        : ✓ installed
```

Common issues:

- **`sidecar deps : ✗ run hermes photon install-sidecar`** — Node is
  installed but `spectrum-ts` isn't. Run the suggested command.
- **`device token : ✗ missing`** — run `hermes photon setup` to log in.
- **`No iMessage line assigned yet`** — Spectrum is enabled but no line
  has been provisioned; re-run `hermes photon setup` or check the
  [dashboard][app].
- **Sidecar won't start** — confirm `node --version` is 18.17+ and that
  `hermes photon install-sidecar` completed without errors.

## Limits today

- **Inbound attachments are downloaded with a size cap.** The sidecar reads
  attachment bytes and base64-inlines them in the authenticated NDJSON stream;
  the adapter caches them so the agent can inspect images/files and transcribe
  voice notes. Items above `PHOTON_MAX_INLINE_ATTACHMENT_BYTES` (20 MB by
  default) fall back to a metadata marker.
- **Outbound attachments require a co-resident sidecar.** Hermes sends images,
  voice notes, video, and documents through spectrum-ts' `attachment()` /
  `voice()` content builders via the sidecar's `/send-attachment` endpoint.
  Captions arrive as a separate iMessage bubble after the media. Adapters using
  a configured remote sidecar reject attachments because a path on the Hermes
  host is not a path on the sidecar host; remote text delivery is unaffected.
- **Native polls are supported in cloud mode.** Hermes sends poll content
  through spectrum-ts' `poll()` builder via the sidecar's `/send-poll`
  endpoint.
- **Message effects are supported in cloud mode.** Hermes sends text with
  native iMessage bubble/screen effects through spectrum-ts' iMessage
  `effect()` builder via the sidecar's `/send-effect` endpoint.
- **Photon's free quotas:** 5,000 messages per server per day,
  50 new-conversation initiations per shared line per day. Increases
  available — email `help@photon.codes`.
- **Cron and standalone sends need the gateway running.** Out-of-process
  senders (cron jobs, `hermes send`, the dashboard) reuse the sidecar the
  gateway spawned — they read its port/token from
  `<hermes-home>/runtime/photon-sidecar.json`, written once the sidecar
  passes its health check and removed when it stops. If a standalone send
  reports the gateway appears to be down, start (or restart) the gateway
  first.
- **Shared/free-tier lines can't initiate conversations with new
  targets.** Photon-side policy: a shared line can only message a number
  after that number has texted the line first. A cron/standalone send to a
  brand-new recipient will be rejected by Photon even when Hermes is set
  up correctly — either have the recipient message the line once, or move
  to a dedicated line.

## Env vars

| Variable                  | Default            | Notes                                      |
|---------------------------|--------------------|--------------------------------------------|
| `PHOTON_PROJECT_ID`       | from `.env`        | Spectrum project id (the SDK's `projectId`); set by setup |
| `PHOTON_PROJECT_SECRET`   | from `.env`        | Project secret; set by setup               |
| `PHOTON_SIDECAR_PORT`     | `8789`             | Loopback port for the sidecar control + inbound channel |
| `PHOTON_SIDECAR_AUTOSTART`| `true`             | Whether the adapter spawns the sidecar     |
| `PHOTON_SIDECAR_TOKEN`    | generated          | Shared authentication token for the sidecar HTTP API |
| `PHOTON_NODE_BIN`         | `which node`       | Override the Node binary path              |
| `PHOTON_HOME_CHANNEL`     | (unset)            | Default space id for cron / notifications  |
| `PHOTON_HOME_CHANNEL_NAME`| (unset)            | Human label for the home channel           |
| `PHOTON_ALLOWED_USERS`    | (unset)            | Comma-separated E.164 allowlist            |
| `PHOTON_ALLOW_ALL_USERS`  | `false`            | Dev only — accept any sender               |
| `PHOTON_REQUIRE_MENTION`  | `false`            | Require a wake word before responding in groups |
| `PHOTON_MENTION_PATTERNS` | Hermes wake words  | JSON list / comma / newline regex patterns for group mentions |
| `PHOTON_DASHBOARD_HOST`   | `app.photon.codes` | Override the dashboard / device-login host |
| `PHOTON_SPECTRUM_HOST`    | `spectrum.photon.codes` | Override the Spectrum API host |

[photon]: https://photon.codes/
[app]: https://app.photon.codes/
