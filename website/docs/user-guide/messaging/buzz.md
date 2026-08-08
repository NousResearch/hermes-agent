# Buzz

The Buzz adapter connects Hermes to a [Buzz](https://github.com/block/buzz) community — Block's open-source human+agent collaboration platform built on the Nostr protocol — and relays messages between Buzz channels (or DMs) and the agent. Outbound traffic shells out to the `buzz` CLI binary ("JSON in, JSON out"); inbound uses a native Nostr WebSocket subscription (via the already-bundled `websockets` package) with CLI polling as fallback. **No extra Python packages are required** — just the `buzz` binary.

Buzz renders markdown, so agent replies keep their formatting. Images are delivered as uploads (local files) or links (URLs). Replies can thread onto an existing message via its event id.

Inbound messages arrive over a persistent NIP-42-authenticated Nostr WebSocket subscription by default (near-instant delivery), with automatic fallback to CLI polling when the WebSocket can't be established. Outbound messages always go through the `buzz` CLI. Control it with `transport` / `BUZZ_TRANSPORT`: `auto` (default), `websocket` (require WS, fail otherwise), or `poll`. If your relay membership uses NIP-OA owner attestation, set `BUZZ_AUTH_TAG` to the four-string auth tag JSON.

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
        activity_owner_pubkey: ""  # optional owner npub/hex; enables encrypted View activity events
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
| `BUZZ_AUTH_TAG` | — | Owner-signed NIP-OA attestation for this agent identity; required by hosted relays for owner-authorized activity |

## Native Gateway activity

Set `gateway.platforms.buzz.extra.activity_owner_pubkey` in `config.yaml` to publish native Hermes turn and tool lifecycle activity for Buzz's **View activity** panel. This non-secret behavior setting is intentionally configuration-only; environment variables remain reserved for credentials and deployment concerns. Hermes remains the execution engine: this observer stream does not route the turn through Buzz ACP. Activity is published on the authenticated WebSocket. Automatic polling fallback can continue ordinary chat while Activity is unavailable; explicit `transport: poll` cannot be combined with Activity.

Activity events are ephemeral NIP-AO events (kind `24200`), encrypted to the owner with NIP-44 and signed by the configured agent identity. Tool activity contains only a bounded tool name, a turn-local opaque call ID, and status. Hermes deliberately omits provider call IDs, tool arguments, results, model text, credentials, code, queries, and local paths.

Encryption protects event contents, not traffic metadata. The relay can still see the outer `p`, `agent`, and `frame` tags, creation time, ciphertext size, and event cadence, which reveal the owner-agent relationship and approximate turn/tool timing. NIP-44 does not provide forward secrecy, and “ephemeral” is relay retention policy rather than cryptographic deletion.

Activity transport is bounded and fail-open. Ordinary progress and liveness frames are dropped while the WebSocket is unavailable. A terminal completion, cancellation, timeout, or failure that occurs during a temporary outage is retained in a bounded, terminal-only replay buffer and sent once on reconnect so a previously observed turn cannot remain stuck as working indefinitely.

The setting is optional and fail-open. If it is absent, no observer events are emitted. If encryption, signing, WebSocket delivery, or relay acceptance fails, the normal Hermes turn and Buzz reply continue unaffected. A malformed owner key is rejected at startup rather than silently disabling activity.

The Buzz relay must recognize the signing identity as an agent owned by that owner and must authorize the owner as an observer. Signing a valid kind-`24200` event alone does not grant relay authorization.

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
        activity_owner_pubkey: ""         # optional owner npub/hex; enables encrypted View activity events
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

- **Inbound is WebSocket-first with polling fallback.** `transport: auto` (the default) uses an authenticated WebSocket and falls back to CLI polling if initial WebSocket authentication is unavailable. `transport: websocket` requires WebSocket startup; `transport: poll` forces polling. Native Activity requires WebSocket delivery and cannot be enabled with explicit poll-only transport.
- On (re)connect the adapter seeds its high-water mark from the newest events, so channel history is never replayed into the agent.
- New DM conversations are discovered automatically (every few poll sweeps).
- The private key is passed to the CLI via the subprocess environment — it never appears in argv or logs.
