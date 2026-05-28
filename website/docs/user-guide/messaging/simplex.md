# SimpleX Chat

[SimpleX Chat](https://simplex.chat/) is a private, decentralised messaging platform where users own their contacts and groups. Unlike other platforms, SimpleX assigns no persistent user IDs — every contact is identified by an opaque internal ID generated at connection time, which makes it one of the most private messengers available.

> Run `hermes gateway setup` and pick **SimpleX** for a guided walk-through.

## Prerequisites

- The **simplex-chat** CLI installed and running as a daemon
- Python package **websockets** (`pip install websockets`)

## Install simplex-chat

Download the latest release from the [simplex-chat GitHub releases](https://github.com/simplex-chat/simplex-chat/releases) page:

```bash
# Linux / macOS binary
curl -L https://github.com/simplex-chat/simplex-chat/releases/latest/download/simplex-chat-ubuntu-22_04-x86-64 -o simplex-chat
chmod +x simplex-chat
```

The SimpleX Chat project does not publish a prebuilt Docker image for the chat client; to run it under Docker, build from source from the [simplex-chat repository](https://github.com/simplex-chat/simplex-chat).

## Start the daemon

```bash
simplex-chat -p 5225
```

The daemon listens on WebSocket at `ws://127.0.0.1:5225` by default.

## Configure Hermes

### Via setup wizard

```bash
hermes setup gateway
```

Select **SimpleX Chat** and follow the prompts.

### Via environment variables

Add these to `~/.hermes/.env`:

```
SIMPLEX_WS_URL=ws://127.0.0.1:5225
SIMPLEX_ALLOWED_USERS=<contact-id-1>,<contact-id-2>
SIMPLEX_HOME_CHANNEL=<contact-id>
```

| Variable | Required | Description |
|---|---|---|
| `SIMPLEX_WS_URL` | Yes | WebSocket URL of the simplex-chat daemon |
| `SIMPLEX_ALLOWED_USERS` | Recommended | Comma-separated contact IDs allowed to use the agent |
| `SIMPLEX_ALLOW_ALL_USERS` | Optional | Set `true` to allow every contact (use carefully) |
| `SIMPLEX_HOME_CHANNEL` | Optional | Default contact ID for cron job delivery |
| `SIMPLEX_HOME_CHANNEL_NAME` | Optional | Human label for the home channel |

## Find your contact ID

After starting the daemon, open a conversation with your agent contact. The contact ID will appear in session logs or via `hermes send_message action=list`.

## Authorization

By default **all contacts are denied**. You must either:

1. Set `SIMPLEX_ALLOWED_USERS` to a comma-separated list of contact IDs, or
2. Use **DM pairing** — send any message to the bot and it will reply with a pairing code. Enter that code via `hermes gateway pair`.

## Using SimpleX with cron jobs

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="simplex",          # uses SIMPLEX_HOME_CHANNEL
    prompt="Check for alerts and summarise."
)
```

Or target a specific contact:

```python
send_message(target="simplex:<contact-id>", message="Done!")
```

## Privacy notes

- SimpleX never reveals phone numbers or email addresses — contacts use opaque IDs
- The connection between Hermes and the daemon is local WebSocket (`ws://127.0.0.1:5225`) — no data leaves your machine
- Messages are end-to-end encrypted by the SimpleX protocol before reaching the daemon

## Troubleshooting

**"Cannot reach daemon"** — Ensure `simplex-chat -p 5225` is running and the port matches `SIMPLEX_WS_URL`.

**"websockets not installed"** — Run `pip install websockets`.

**Messages not received** — Check that the contact's ID is in `SIMPLEX_ALLOWED_USERS` or approve them via DM pairing.
