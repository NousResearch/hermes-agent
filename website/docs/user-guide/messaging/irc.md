# IRC

The IRC adapter connects Hermes to any IRC server and relays messages between an IRC channel (or direct messages) and the agent. It speaks the IRC protocol over Python's stdlib `asyncio` — **no external dependencies, no SDK, no daemon**. For agentic use, prefer a local or self-hosted ircd where you control the network policy.

IRC is plain text: there is no voice, image, file, thread, reaction, typing, or streaming support — replies are sent as `PRIVMSG` lines, with long messages split to fit the IRC line limit.

> Run `hermes gateway setup` and pick **IRC** for a guided walk-through.

## Prerequisites

- An IRC server to connect to (e.g. `localhost` for a self-hosted ircd)
- A channel to join (e.g. `#hermes`) — comma-separate to join several
- A nickname for the bot (default: `hermes-bot`)
- Optional: a registered nick + NickServ password if your network requires identification

## Configure Hermes

You can configure IRC two ways — environment variables (for a quick env-only setup) or the `gateway` block in `~/.hermes/gateway-config.yaml`.

### Option A — gateway-config.yaml

```yaml
gateway:
  platforms:
    irc:
      enabled: true
      extra:
        server: localhost
        port: 6697
        nickname: hermes-bot
        channel: "#hermes"
        use_tls: true
        server_password: ""       # optional server password
        nickserv_password: ""     # optional NickServ identification
        allowed_users: []         # empty = allow all, or list of nicks
        max_message_length: 450   # IRC line limit (safe default)
```

### Option B — environment variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `IRC_SERVER` | ✅ | IRC server hostname (e.g. `localhost` for a self-hosted ircd) |
| `IRC_CHANNEL` | ✅ | Channel(s) to join — comma-separate for multiple |
| `IRC_NICKNAME` | ✅ | Bot nickname (default: `hermes-bot`) |
| `IRC_PORT` | — | Server port (default: `6697` with TLS, `6667` without) |
| `IRC_USE_TLS` | — | Use TLS (`true`/`false`; default `true` on port 6697) |
| `IRC_SERVER_PASSWORD` | — | Server password for the `PASS` command |
| `IRC_NICKSERV_PASSWORD` | — | NickServ password for automatic IDENTIFY on connect |
| `IRC_ALLOWED_USERS` | — | Comma-separated nicks allowed to talk to the bot |
| `IRC_ALLOW_ALL_USERS` | — | Allow anyone in the channel to talk to the bot (dev only) |
| `IRC_HOME_CHANNEL` | — | Channel for cron / notification delivery (defaults to `IRC_CHANNEL`) |

## Access control

By default, only nicks listed in `allowed_users` (or `IRC_ALLOWED_USERS`) may talk to the bot. Leave the list empty **and** set `IRC_ALLOW_ALL_USERS=true` to let anyone in the channel chat with Hermes — useful for testing, but not recommended on public networks since IRC nicks are not authenticated unless the network enforces NickServ.

If your network registers nicks, set `IRC_NICKSERV_PASSWORD` (or `nickserv_password`) so the bot identifies to NickServ on connect and keeps its registered nick.

## Channels vs. DMs

- Messages in a joined channel are treated as a **group** conversation.
- Private messages to the bot are treated as **direct messages**.

Cron jobs and notifications are delivered to the **home channel** — `IRC_HOME_CHANNEL` if set, otherwise the first `IRC_CHANNEL`.

## Run the gateway

```bash
hermes gateway start
```

Check status with `hermes gateway status` — IRC connection state is reported there, including for env-only setups.

## Notes

- Long agent replies are automatically split into multiple `PRIVMSG` lines to stay within the IRC line limit (`max_message_length`, default 450 bytes after protocol overhead).
- The adapter acquires a scoped credential lock per server+nick, so two Hermes profiles won't fight over the same IRC identity.
