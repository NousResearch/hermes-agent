# Buzz

The Buzz adapter connects Hermes to a [Buzz](https://github.com/block/buzz) community — Block's open-source human+agent collaboration platform built on the Nostr protocol — and relays messages between Buzz channels (or DMs) and the agent. The adapter is pure Python stdlib: it shells out to the `buzz` CLI binary ("JSON in, JSON out") instead of speaking Nostr itself, so **no Python packages are required** — just the `buzz` binary.

Buzz renders markdown, so agent replies keep their formatting. Images are delivered as uploads (local files) or links (URLs). Replies can thread onto an existing message via its event id.

> Run `hermes gateway setup` and pick **Buzz** for a guided walk-through.

## Prerequisites

- The `buzz` CLI binary on your `PATH` (or point `BUZZ_CLI_PATH` at it) — build it from the [Buzz repo](https://github.com/block/buzz) with `cargo build --release -p buzz-cli`
- A Buzz community relay URL (e.g. `https://mycommunity.communities.buzz.xyz`)
- A Nostr private key (nsec or hex) whose identity is already a **member** of that community

## Configure Hermes

You can configure Buzz two ways — the `gateway` block in `config.yaml` (canonical) or environment variables (which override it). The private key is a **secret** and always belongs in `~/.hermes/.env`.

### Option A — config.yaml

```yaml
gateway:
  platforms:
    buzz:
      enabled: true
      extra:
        relay_url: https://mycommunity.communities.buzz.xyz
        channels:                  # channel UUIDs to watch (empty = all joined)
          - ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        home_channel: ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        poll_interval: 4           # seconds between inbound poll sweeps
        cli_path: ""               # buzz binary (default: PATH, then ~/bin/buzz)
        credentials_file: ""       # JSON file with the nsec (BUZZ_PRIVATE_KEY fallback)
        allowed_users: []          # empty = allow all; hex pubkeys or npubs
```

Plus, in `~/.hermes/.env`:

```
BUZZ_PRIVATE_KEY=nsec1...
```

### Option B — environment variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `BUZZ_RELAY_URL` | ✅ | Base URL of the community relay |
| `BUZZ_PRIVATE_KEY` | ✅ | Nostr private key (nsec or hex) — the only secret |
| `BUZZ_CHANNELS` | — | Comma-separated channel UUIDs to watch (default: all joined channels) |
| `BUZZ_HOME_CHANNEL` | — | Channel UUID for cron / notification delivery (defaults to the first watched channel) |
| `BUZZ_ALLOWED_USERS` | — | Comma-separated npubs or hex pubkeys allowed to talk to the agent |
| `BUZZ_ALLOW_ALL_USERS` | — | Allow any community member to talk to the agent |
| `BUZZ_POLL_INTERVAL` | — | Seconds between inbound poll sweeps (default: 4) |
| `BUZZ_CLI_PATH` | — | Path to the `buzz` binary (default: `buzz` on PATH, then `~/bin/buzz`) |
| `BUZZ_CREDENTIALS_FILE` | — | JSON credentials file holding the nsec, used when `BUZZ_PRIVATE_KEY` is unset |

## Mentions, channels, and DMs

- In shared channels the agent only responds when **addressed** — by `@name`, its npub, or its hex pubkey. Everything else is ignored.
- Direct messages always reach the agent, no mention needed.
- The agent's own messages are never dispatched back to it (self-echo suppression by pubkey), and every event is de-duplicated by event id against a per-channel high-water mark.

## Access control

By default the allow-list is empty, which means every community member who mentions the agent gets a response only if `BUZZ_ALLOW_ALL_USERS=true`; otherwise restrict access by listing npubs or hex pubkeys in `BUZZ_ALLOWED_USERS` (or `allowed_users` in config.yaml). Community membership itself is enforced by the relay — only members can post.

Cron jobs and notifications (`deliver=buzz`) are delivered to the **home channel** — `BUZZ_HOME_CHANNEL` if set, otherwise the first watched channel — and work even when cron runs outside the gateway process.

## Run the gateway

```bash
hermes gateway start
```

Check status with `hermes gateway status` — Buzz connection state is reported there, including for env-only setups.

## Notes and limitations

- **Inbound is polled, not streamed.** The `buzz` CLI is request/response, so the adapter polls `buzz messages get` per watched channel every `poll_interval` seconds (default 4). Expect up to one interval of latency on inbound messages. A future optimization is a websocket transport (the Buzz repo ships `buzz-ws-client` for true streaming).
- On (re)connect the adapter seeds its high-water mark from the newest events, so channel history is never replayed into the agent.
- New DM conversations are discovered automatically (every few poll sweeps).
- The private key is passed to the CLI via the subprocess environment — it never appears in argv or logs.
