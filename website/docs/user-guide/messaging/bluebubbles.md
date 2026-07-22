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
# Optional inbound mode. Default is webhook.
BLUEBUBBLES_TRANSPORT=webhook
```

You can also configure the inbound transport in `~/.hermes/config.yaml`:

```yaml
platforms:
  bluebubbles:
    enabled: true
    extra:
      server_url: http://192.168.1.10:1234
      # Prefer .env for passwords; shown here only to document the key.
      password: your-server-password
      transport: socketio
```

Unknown explicit transport values fail closed; use `webhook` or one of the Socket.IO aliases listed below. If lazy dependency installation is disabled, install the narrow Socket.IO extra first: `pip install 'hermes-agent[bluebubbles-socketio]'`.

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

Hermes will connect to your BlueBubbles server and start listening for iMessage messages. By default it registers a webhook. If `BLUEBUBBLES_TRANSPORT=socketio` is set, Hermes instead opens an outbound Socket.IO connection to the BlueBubbles server.

## How It Works

```
iMessage → Messages.app → BlueBubbles Server → Webhook or Socket.IO → Hermes
Hermes → BlueBubbles REST API → Messages.app → iMessage
```

- **Inbound:** By default, BlueBubbles sends webhook events to a local Hermes listener when new messages arrive. No polling — instant delivery.
- **Socket.IO inbound:** Set `BLUEBUBBLES_TRANSPORT=socketio` or `platforms.bluebubbles.extra.transport: socketio` when Hermes cannot expose a webhook listener that the Mac can reach. In this mode Hermes initiates the realtime connection outward to BlueBubbles. Accepted aliases are `socketio`, `socket.io`, `socket`, `websocket`, and `ws`. Duplicate events with the same iMessage GUID, including later `updated-message` events for edits/retractions, are acknowledged but not dispatched as new agent turns.
- **Outbound:** Hermes sends messages via the BlueBubbles REST API.
- **Media:** Images, voice messages, videos, and documents are supported in both directions. Inbound attachments are downloaded and cached locally for the agent to process.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BLUEBUBBLES_SERVER_URL` | Yes | — | BlueBubbles server URL |
| `BLUEBUBBLES_PASSWORD` | Yes | — | Server password |
| `BLUEBUBBLES_WEBHOOK_HOST` | No | `127.0.0.1` | Webhook listener bind address |
| `BLUEBUBBLES_WEBHOOK_PORT` | No | `8645` | Webhook listener port |
| `BLUEBUBBLES_WEBHOOK_PATH` | No | `/bluebubbles-webhook` | Webhook URL path |
| `BLUEBUBBLES_TRANSPORT` | No | `webhook` | Inbound transport: `webhook` or `socketio` |
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

## Troubleshooting

### "Cannot reach server"
- Verify the server URL is correct and the Mac is on
- Check that BlueBubbles Server is running
- Ensure network connectivity (firewall, port forwarding)

### Messages not arriving
- Check that the webhook is registered in BlueBubbles Server → Settings → API → Webhooks
- Verify the webhook URL is reachable from the Mac
- Check `hermes logs gateway` for webhook errors (or `hermes logs -f` to follow in real-time)

### "Private API helper not connected"
- Install the Private API helper: [docs.bluebubbles.app](https://docs.bluebubbles.app/helper-bundle/installation)
- Basic messaging works without it — only reactions, typing, and read receipts require it

