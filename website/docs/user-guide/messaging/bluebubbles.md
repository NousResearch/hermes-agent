# BlueBubbles (iMessage)

Connect Hermes to Apple iMessage via [BlueBubbles](https://bluebubbles.app/) — a free, open-source macOS server that bridges iMessage to any device.

## Prerequisites

- A **Mac** (always on) running [BlueBubbles Server](https://bluebubbles.app/)
- Apple ID signed into Messages.app on that Mac
- BlueBubbles Server v1.0.0+ (webhooks require this version)
- Network connectivity between Hermes and the BlueBubbles server

## Setup

### 1. Install BlueBubbles Server

Download and install from [bluebubbles.app](https://bluebubbles.app/). Complete the setup wizard — sign in with your Apple ID and configure a connection method (local network, Ngrok, Cloudflare, or Dynamic DNS).

### 2. Get your Server URL and Password

In BlueBubbles Server → **Settings → API**, note:
- **Server URL** (e.g., `http://192.168.1.10:1234`)
- **Server Password**

### 3. Configure Hermes

Run the setup wizard:

```bash
hermes gateway setup
```

Select **BlueBubbles (iMessage)** and enter your server URL and password.

Or set environment variables directly in `~/.hermes/.env`:

```bash
BLUEBUBBLES_SERVER_URL=http://192.168.1.10:1234
BLUEBUBBLES_PASSWORD=your-server-password
```

#### Optional: Require mentions in group chats

By default, Hermes responds to every authorized BlueBubbles/iMessage DM or group message. To make group chats opt-in, enable mention gating:

```yaml
platforms:
  bluebubbles:
    enabled: true
    extra:
      require_mention: true
```

With `require_mention: true`, DMs still work normally, but group-chat messages are ignored unless they match a mention pattern. If you do not configure custom patterns, Hermes uses conservative defaults for `Hermes` and `@Hermes agent` variants.

For a custom agent name, set regex patterns:

```yaml
platforms:
  bluebubbles:
    extra:
      require_mention: true
      mention_patterns:
        - '(?<![\w@])@?amos\b[,:\-]?'
```

### 4. Authorize Users

Choose one approach:

**DM Pairing (recommended):**
When someone messages your iMessage, Hermes automatically sends them a pairing code. Approve it with:
```bash
hermes pairing approve bluebubbles <CODE>
```
Use `hermes pairing list` to see pending codes and approved users.

**Pre-authorize specific users** (in `~/.hermes/.env`):
```bash
BLUEBUBBLES_ALLOWED_USERS=user@icloud.com,+15551234567
```

**Open access** (in `~/.hermes/.env`):
```bash
BLUEBUBBLES_ALLOW_ALL_USERS=true
```

### 5. Start the Gateway

```bash
hermes gateway run
```

Hermes will connect to your BlueBubbles server, register a webhook, and start listening for iMessage messages.

### 6. Verify the Setup

The setup wizard saves your credentials to `~/.hermes/.env` — it does **not** write `platforms.bluebubbles.enabled` into `~/.hermes/config.yaml`. With no explicit setting, present credentials are enough for the adapter to start; but an explicit `enabled: false` always wins over credentials, so if you previously disabled the adapter (for example, while using another iMessage bridge), the wizard will report success and the adapter will still never start.

Check the stored setting:

```bash
hermes config get platforms.bluebubbles.enabled
```

- `true` — explicitly enabled
- `Config key not set` — no explicit setting; the `.env` credentials drive enablement
- `false` — explicitly disabled; re-running setup will not flip this — set it to `true` yourself

Then check the gateway log for the two lines that prove the connection and the webhook registration:

```bash
hermes logs gateway
```

```text
[bluebubbles] connected to http://192.168.1.10:1234 (private_api=False, helper=False)
[bluebubbles] webhook registered with server: http://localhost:8645/bluebubbles-webhook?password=***
```

Finally, send yourself a test message from another device: a reply (or the pairing code for a new DM) is the end-to-end proof that inbound delivery works.

## How It Works

```
iMessage → Messages.app → BlueBubbles Server → Webhook → Hermes
Hermes → BlueBubbles REST API → Messages.app → iMessage
```

- **Inbound:** BlueBubbles sends webhook events to a local listener when new messages arrive. No polling — instant delivery.
- **Outbound:** Hermes sends messages via the BlueBubbles REST API.
- **Media:** Images, voice messages, videos, and documents are supported in both directions. Inbound attachments are downloaded and cached locally for the agent to process.

### Two URLs, opposite directions

The setup uses two URLs that point in opposite directions — don't confuse them:

- `BLUEBUBBLES_SERVER_URL` (e.g. `http://192.168.1.10:1234`) — Hermes **calls** your BlueBubbles server's API. This is the Server URL shown in BlueBubbles Server → Settings → API.
- The webhook (default `http://localhost:8645/bluebubbles-webhook`) — BlueBubbles **POSTs** new-message events to Hermes. Its host/port/path come from `BLUEBUBBLES_WEBHOOK_HOST` / `BLUEBUBBLES_WEBHOOK_PORT` / `BLUEBUBBLES_WEBHOOK_PATH`.

### How the webhook is registered

You do **not** need to create a webhook in the BlueBubbles UI. When the gateway connects, Hermes registers the webhook itself via the BlueBubbles REST API (`/api/v1/webhook`) for the `new-message` and `updated-message` events, and removes the registration again on clean shutdown.

Two details worth knowing:

- The registered URL carries the server password as a query parameter (`?password=…`) because the BlueBubbles webhook API does not support custom headers — this is how inbound events are authenticated.
- The webhook listener binds to `127.0.0.1` by default. That is fine when Hermes and BlueBubbles run on the same machine; if they are on different machines, set `BLUEBUBBLES_WEBHOOK_HOST` to an address the Mac running BlueBubbles can reach.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BLUEBUBBLES_SERVER_URL` | Yes | — | BlueBubbles server URL |
| `BLUEBUBBLES_PASSWORD` | Yes | — | Server password |
| `BLUEBUBBLES_WEBHOOK_HOST` | No | `127.0.0.1` | Webhook listener bind address |
| `BLUEBUBBLES_WEBHOOK_PORT` | No | `8645` | Webhook listener port |
| `BLUEBUBBLES_WEBHOOK_PATH` | No | `/bluebubbles-webhook` | Webhook URL path |
| `BLUEBUBBLES_HOME_CHANNEL` | No | — | Phone/email for cron delivery |
| `BLUEBUBBLES_ALLOWED_USERS` | No | — | Comma-separated authorized users |
| `BLUEBUBBLES_ALLOW_ALL_USERS` | No | `false` | Allow all users |
| `BLUEBUBBLES_REQUIRE_MENTION` | No | `false` | Require a mention pattern before responding in group chats |
| `BLUEBUBBLES_MENTION_PATTERNS` | No | Hermes wake words | JSON array, newline-separated, or comma-separated regex patterns for group mention matching |

Auto-marking messages as read is controlled by the `send_read_receipts` key under `platforms.bluebubbles.extra` in `~/.hermes/config.yaml` (default: `true`). There is no corresponding environment variable.

## Features

### Text Messaging
Send and receive iMessages. Markdown is automatically stripped for clean plain-text delivery.

### Rich Media
- **Images:** Photos appear natively in the iMessage conversation
- **Voice messages:** Audio files sent as iMessage voice messages
- **Videos:** Video attachments
- **Documents:** Files sent as iMessage attachments

### Tapback Reactions
Love, like, dislike, laugh, emphasize, and question reactions. Requires the BlueBubbles [Private API helper](https://docs.bluebubbles.app/helper-bundle/installation).

### Typing Indicators
Shows "typing..." in the iMessage conversation while the agent is processing. Requires Private API.

### Read Receipts
Automatically marks messages as read after processing. Requires Private API.

### Chat Addressing
You can address chats by email or phone number — Hermes resolves them to BlueBubbles chat GUIDs automatically. No need to use raw GUID format.

## Private API

Some features require the BlueBubbles [Private API helper](https://docs.bluebubbles.app/helper-bundle/installation):
- Tapback reactions
- Typing indicators
- Read receipts
- Creating new chats by address

Without the Private API, basic text messaging and media still work.

One caveat: "basic messaging works without Private API" assumes BlueBubbles can drive Messages.app in the foreground of a logged-in macOS user. If the Mac running BlueBubbles stays at the login screen or the server user is switched away from (Fast User Switching), AppleScript-based sending fails — see [BlueBubbles: multiple users on the same Mac](https://docs.bluebubbles.app/server/basic-guides/multiple-users-on-the-same-mac). In that setup the Private API is required for reliable sending, not just for the extras above.

## Troubleshooting

### "Cannot reach server"
- Verify the server URL is correct and the Mac is on
- Check that BlueBubbles Server is running
- Ensure network connectivity (firewall, port forwarding)

### Messages not arriving
- Check `hermes logs gateway` for webhook errors (or `hermes logs -f` to follow in real-time)
- Hermes registers the webhook itself on connect — only inspect BlueBubbles Server → Settings → API → Webhooks if the log shows a registration failure
- A webhook row in the BlueBubbles UI is not proof of delivery; the end-to-end proof is Hermes logging the message and replying
- If Hermes and BlueBubbles run on different machines, the default webhook bind address `127.0.0.1` is unreachable from the Mac — set `BLUEBUBBLES_WEBHOOK_HOST` to a reachable address and restart the gateway

### Setup succeeded, but the adapter never starts
- `hermes gateway setup` saves credentials to `~/.hermes/.env`; it does not set `platforms.bluebubbles.enabled: true`
- An explicit `enabled: false` in `~/.hermes/config.yaml` wins over credentials being present — check with `hermes config get platforms.bluebubbles.enabled`
- This commonly bites after switching iMessage bridges: if you used another iMessage bridge and disabled BlueBubbles at the time, re-running setup will not re-enable it. Set `enabled: true` (and disable the bridge you no longer use — two iMessage bridges will double-handle messages)

### Two BlueBubbles servers on one Mac (wrong Apple ID)
- Hermes uses `BLUEBUBBLES_SERVER_URL` from `~/.hermes/.env`, not the Server URL shown in the BlueBubbles UI (which can be stale after a DHCP change)
- Two macOS users on one Mac each run their own BlueBubbles server with its own API port and Apple ID — verify which one Hermes reaches: `curl "http://<server-url>/api/v1/server/info?password=<password>"` and compare the `computer_id`
- For the multi-user setup itself, follow [BlueBubbles: multiple users on the same Mac](https://docs.bluebubbles.app/server/basic-guides/multiple-users-on-the-same-mac) — one port per user, and don't log out the user running the server

### Duplicate replies
- Known issue: session handling can split one correspondent into two sessions (raw-GUID form vs. phone/email form) — tracked in [#30708](https://github.com/NousResearch/hermes-agent/issues/30708) and [#34372](https://github.com/NousResearch/hermes-agent/issues/34372)
- Not a documentation or configuration problem — follow those issues for fixes

### "♻️ Recovered reply" repeats, or sends hang for minutes
- If a `BLUEBUBBLES_HOME_CHANNEL` (or `platforms.bluebubbles.home_channel`) is configured, gateway restarts and platform reconnects notify the home channel and may re-deliver pending replies as "♻️ Recovered reply"
- If sends on the BlueBubbles side silently fail — typically AppleScript errors like `Not authorized to send Apple events to Messages. (-1743)` with long timeouts on a background macOS user — the retries pile up and can crowd out webhook delivery
- Leave the home channel empty until sending is proven reliable, and enable the Private API helper for background-user setups (see [Private API](#private-api))

### "Private API helper not connected"
- Install the Private API helper: [docs.bluebubbles.app](https://docs.bluebubbles.app/helper-bundle/installation)
- Basic messaging works without it — only reactions, typing, and read receipts require it

