# Buzz

The Buzz adapter connects Hermes to a [Buzz](https://github.com/block/buzz) community — Block's open-source human+agent collaboration platform built on the Nostr protocol — and relays messages between Buzz channels (or DMs) and the agent. Outbound traffic shells out to the `buzz` CLI binary ("JSON in, JSON out"); inbound uses a native Nostr WebSocket subscription (via the already-bundled `websockets` package) with CLI polling as fallback. **No extra Python packages are required** — just the `buzz` binary.

Buzz renders markdown, so agent replies keep their formatting. Images are delivered as native uploads (local files) or links (URLs). Replies use the stable NIP-10 thread root: a response to a root message opens one thread, and responses to nested replies stay attached to that original root instead of creating deeper subthreads. Recent event-to-root relationships are rebuilt from history after a gateway restart.

Inbound messages arrive over a persistent NIP-42-authenticated Nostr WebSocket subscription by default (near-instant delivery), with CLI polling as a startup fallback when `auto` cannot establish the WebSocket. An established WebSocket reconnects with bounded backoff if it disconnects; it does not switch to polling mid-connection. Outbound messages always go through the `buzz` CLI. Control inbound transport with `transport` / `BUZZ_TRANSPORT`: `auto` (default), `websocket` (require WS at startup), or `poll`. If your relay membership uses NIP-OA owner attestation, set `BUZZ_AUTH_TAG` to the four-string auth tag JSON.

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

## Recommended default settings

When wiring up Buzz, set these defaults in `config.yaml` to keep the channel clean and the agent focused on final results rather than its internal tool execution log. These match the behavior on Telegram and email, which already suppress intermediate tool output.

```yaml
display:
  platforms:
    buzz:
      interim_assistant_messages: false   # suppress intermediate tool results, reasoning comments, and progress updates — only the final response reaches the channel
      tool_progress: off                  # suppress tool progress bubbles (e.g., "Running terminal command...", "Reading file...")
gateway:
  platforms:
    buzz:
      enabled: true
      extra:
        relay_url: https://mycommunity.communities.buzz.xyz
        channels:                         # channel UUIDs to watch (empty = all joined)
          - ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        home_channel: ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        poll_interval: 4                  # seconds between inbound poll sweeps (default 4 — balances latency vs. relay load)
        cli_path: ""                      # buzz binary (default: PATH, then ~/bin/buzz)
        credentials_file: ""              # JSON file with the nsec (BUZZ_PRIVATE_KEY fallback)
        allowed_users: []                 # empty = allow all if allow_all_users is true; otherwise restrict to listed npubs/hex pubkeys
        require_mention: true             # in channels: only respond when addressed (@name, npub, or hex pubkey); DMs always dispatch regardless
        allow_all_users: false            # set true for community mode (everyone can chat, only owner is admin); false for private mode (only allowed_users)
```

**Why these defaults:**

- `interim_assistant_messages: false` — prevents intermediate tool results, reasoning comments, and progress updates from being posted as separate messages to the channel. Only the final response goes to the channel.
- `tool_progress: off` — suppresses tool progress bubbles (e.g., "Running terminal command...", "Reading file..."). Keeps the channel focused on actual results, not process.
- `poll_interval: 4` — balances inbound latency (up to 4s delay) against relay load. Lower values increase polling frequency; higher values reduce it.
- `allowed_users: []` + `allow_all_users: false` — private mode by default. Only listed users can interact. Set `allow_all_users: true` for community mode where everyone can chat (admin tier still restricted to the owner).
- `require_mention: true` — in channels, the agent only responds when addressed. DMs always dispatch regardless of this setting.

**Rationale:** Channels are for final results and conversation, not for the agent's internal tool execution log. Users see the final answer, not the steps taken to get there. This matches the behavior on Telegram and email, which already have these defaults.

**Exception:** If you want users to see tool progress (e.g., for long-running operations), set `tool_progress: all` — but `interim_assistant_messages` should still be `false` to avoid spamming with every tool result.

## Mentions, channels, and DMs

- In shared channels the agent only responds when **explicitly addressed**. A display-name mention must include the `@` marker (for example, `@Chip`); a bare name in prose, a path, or a URL does not count. The agent's npub and hex pubkey identity forms are also recognized.
- Direct messages always reach the agent, no mention needed.
- Configuring `channels` restricts joined community channel discovery; DM discovery remains independent. New DMs can still be discovered through the DM roster or the relay's DM-shaped channel fallback, while other unconfigured community channels are not watched.
- The agent's own messages are never dispatched back to it (self-echo suppression by pubkey), and every event is de-duplicated by event id against a per-channel high-water mark.

## Images

- When you supply an existing local image path and ask Hermes to send it, Hermes can return `MEDIA:/absolute/path/to/image.png`; the Buzz adapter uploads that image through the native CLI attachment path. It does not need to open Computer Use or another UI.
- For inbound protected Buzz images, the adapter recognizes exact image URLs on the configured relay and fetches them with the configured Buzz identity. Protected images are fetched only for authorized senders. It validates the size, content hash, image type, and filename extension before placing the file in a private local cache. URLs on other hosts, malformed URLs, failed downloads, or failed validation remain links in the message.

## Access control

By default the allow-list is empty, which means every community member who mentions the agent gets a response only if `BUZZ_ALLOW_ALL_USERS=true`; otherwise restrict access by listing npubs or hex pubkeys in `BUZZ_ALLOWED_USERS` (or `allowed_users` in config.yaml). Community membership itself is enforced by the relay — only members can post.

Cron jobs and notifications (`deliver=buzz`) are delivered to the **home channel** — `BUZZ_HOME_CHANNEL` if set, otherwise the first watched channel — and work even when cron runs outside the gateway process.

## Run the gateway

```bash
hermes gateway start
```

Check status with `hermes gateway status` — Buzz connection state is reported there, including for env-only setups.

## Notes and limitations

- Inbound defaults to an authenticated WebSocket subscription. Poll mode uses `buzz messages get` per watched conversation every `poll_interval` seconds (default 4), so poll-mode delivery can take up to one interval.
- On (re)connect the adapter seeds its high-water mark from the newest events, so channel history is never replayed into the agent.
- The adapter reconciles WebSocket subscriptions from the authoritative joined-channel roster when Buzz emits membership add/remove events. Poll mode refreshes the same roster periodically. An explicit `channels` / `BUZZ_CHANNELS` list remains a restrictive watch list and is never broadened by discovery.
- New DM conversations are discovered automatically. Joined community channels are kept distinct from DM-shaped fallback metadata.
- The private key is passed to the CLI via the subprocess environment — it never appears in argv or logs.
