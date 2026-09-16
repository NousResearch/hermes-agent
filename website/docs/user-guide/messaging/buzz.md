# Buzz

The Buzz adapter connects Hermes to a [Buzz](https://github.com/block/buzz) community — Block's open-source human+agent collaboration platform built on the Nostr protocol — and relays messages between Buzz channels (or DMs) and the agent. Outbound traffic shells out to the `buzz` CLI binary ("JSON in, JSON out"); inbound uses a native Nostr WebSocket subscription (via the already-bundled `websockets` package) with CLI polling as fallback. **No extra Python packages are required** — just the `buzz` binary.

Buzz renders markdown, so agent replies keep their formatting. Images are delivered as uploads (local files) or links (URLs). Replies can thread onto an existing message via its event id. When progress or status messages are enabled, they inherit the triggering Buzz event as their reply anchor instead of appearing as unrelated top-level channel posts.

Files sent **to** the agent are fetched back off the relay with the agent's authenticated identity and cached locally, so tools receive a real file path rather than a `/media/…` URL that anonymous requests cannot read. Images, audio, video, and documents (PDFs and the like) are all handled.

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
        attachment_hosts: []         # additional exact HTTPS host[:port] origins for inbound files
        channels:                  # channel UUIDs to watch (empty = all joined)
          - ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        home_channel: ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        poll_interval: 4           # seconds between inbound poll sweeps
        cli_path: ""               # buzz binary (default: PATH, then ~/bin/buzz)
        credentials_file: ""       # JSON file with the nsec (BUZZ_PRIVATE_KEY fallback)
        activity_owner_pubkey: ""  # optional owner npub/hex; enables encrypted View activity events
        allowed_users: []          # empty = deny all unless allow_all_users is true
        allow_all_users: false
        require_mention: true
        thread_require_mention: true
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
| `BUZZ_REQUIRE_MENTION` | — | Require an explicit mention in top-level shared-channel messages (`true`/`false`) |
| `BUZZ_THREAD_REQUIRE_MENTION` | — | Independently require an explicit mention in structurally threaded replies (`true`/`false`) |
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
        attachment_hosts: []         # additional exact HTTPS host[:port] origins for inbound files
        channels:                         # channel UUIDs to watch (empty = all joined)
          - ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        home_channel: ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        poll_interval: 4                  # seconds between inbound poll sweeps (default 4 — balances latency vs. relay load)
        activity_owner_pubkey: ""         # optional owner npub/hex; enables encrypted View activity events
        cli_path: ""                      # buzz binary (default: PATH, then ~/bin/buzz)
        credentials_file: ""              # JSON file with the nsec (BUZZ_PRIVATE_KEY fallback)
        allowed_users: []                 # empty = allow all if allow_all_users is true; otherwise restrict to listed npubs/hex pubkeys
        require_mention: true             # in channels: only respond when addressed (@name, npub, or hex pubkey); DMs always dispatch regardless
        thread_require_mention: true      # apply the same default independently to marked and legacy NIP-10 thread replies
        allow_all_users: false            # set true for community mode (everyone can chat, only owner is admin); false for private mode (only allowed_users)
```

**Why these defaults:**

- `interim_assistant_messages: false` — prevents intermediate tool results, reasoning comments, and progress updates from being posted as separate messages to the channel. Only the final response goes to the channel.
- `tool_progress: off` — suppresses tool progress bubbles (e.g., "Running terminal command...", "Reading file..."). Keeps the channel focused on actual results, not process.
- `poll_interval: 4` — balances inbound latency (up to 4s delay) against relay load. Lower values increase polling frequency; higher values reduce it.
- `allowed_users: []` + `allow_all_users: false` — private mode by default. Only listed users can interact. Set `allow_all_users: true` for community mode where everyone can chat (admin tier still restricted to the owner).
- `require_mention: true` — in channels, the agent only responds when addressed. DMs always dispatch regardless of this setting.
- `thread_require_mention: true` — threaded replies have an independent mention policy. Hermes recognizes both marked and legacy positional NIP-10 thread structures.

**Rationale:** Channels are for final results and conversation, not for the agent's internal tool execution log. Users see the final answer, not the steps taken to get there. This matches the behavior on Telegram and email, which already have these defaults.

**Exception:** If you want users to see tool progress (e.g., for long-running operations), set `tool_progress: all` — but `interim_assistant_messages` should still be `false` to avoid spamming with every tool result.

## Mentions, channels, and DMs

- The watch set is the authoritative **joined** roster (`buzz channels list --member`). Joining or leaving a community while the Gateway runs adds or removes that channel's subscription live, on membership events over WebSocket and on the poll cadence; `channels` / `BUZZ_CHANNELS` only narrows the joined set, it cannot widen it. DMs are discovered independently.
- In shared channels the agent only responds when **addressed** — by `@name`, its npub, or its hex pubkey. Everything else is ignored.
- Direct messages always reach the agent, no mention needed.
- The agent's own messages are never dispatched back to it (self-echo suppression by pubkey), and every event is de-duplicated by event id against a per-channel high-water mark.

## Reply threading

Replies are threaded by default: the agent's answer (and any enabled progress/status messages) is anchored to the message that triggered it. Anchoring is NIP-10 aware — when the triggering message was already **inside** a thread, the agent replies to that thread's *root*, so the answer joins the existing thread instead of nesting a new one-message sub-thread under every turn.

To post replies flat at the channel level instead, set either of these (they are equivalent; `reply_in_thread` matches the key Slack uses):

```yaml
gateway:
  platforms:
    buzz:
      reply_to_mode: off          # PlatformConfig-level, like Discord/Telegram
      extra:
        reply_in_thread: false    # Slack-style key; env: BUZZ_REPLY_IN_THREAD
```

The opt-out applies to **all** send paths — final answers, streamed updates, interim commentary, tool-progress bubbles, and out-of-process cron delivery (`deliver=buzz`).

## Agent directory

The adapter publishes a replaceable kind-10100 agent-directory event to the relay: on every WebSocket connection, whenever the joined roster changes, and on the poll cadence for poll transport (over a short-lived authenticated socket). Buzz's agent picker reads it to show the agent's name, its joined community channels, and whether it answers anyone (`allow_all_users: true`) or only an allow-list. The projection is derived from the live effective policy and the joined roster, so it never advertises wider access than the Gateway actually grants; a deny-all policy is published as an empty allow-list, which the picker treats as ineligible. When `BUZZ_AUTH_TAG` is set, the tag's NIP-OA conditions are verified against the event before publication and attached to it. The event is signed with the adapter's own key, and publication waits for the relay's acknowledgement; a rejected event ends the connection like any other transport error.

## Access control

By default the Buzz-specific allow-list is empty. Buzz-specific access is granted by `BUZZ_ALLOW_ALL_USERS=true` or by listing npubs/hex pubkeys in `BUZZ_ALLOWED_USERS` (or `allowed_users` in config.yaml). These controls are not the complete Gateway authorization boundary: profile pairing approvals and global `GATEWAY_ALLOWED_USERS`, its `*` wildcard, or `GATEWAY_ALLOW_ALL_USERS` are additive and can authorize users beyond the Buzz-specific policy. Community membership itself is enforced by the relay — only members can post.

Policy changes to `allowed_users`, `allow_all_users`, `require_mention` and `thread_require_mention` in the profile's `config.yaml` or `.env` take effect on the next inbound event; no Gateway restart is required for these four fields. Precedence is explicit environment policy (`BUZZ_*`, managed values first), then the managed `config.yaml` overlay, then the profile's `config.yaml`, then restrictive defaults: no allowed users, allow-all off, mentions required in channels and threads.

The allow-list also gates **inbound attachments**: relay media is fetched with the agent's own Buzz credentials, so a download only happens for a sender the gateway explicitly authorizes. A denied, missing, or failed authorization leaves the message text untouched and makes no credentialed request.

Cron jobs and notifications (`deliver=buzz`) are delivered to the **home channel** — `BUZZ_HOME_CHANNEL` if set, otherwise the first watched channel — and work even when cron runs outside the gateway process.

## Inbound attachments

Buzz messages with native NIP-94 `imeta` tags can deliver images, audio,
video, and documents to the agent. Hermes downloads attachments only after
the message has passed self-echo, addressing, and sender authorization checks.
Each file must use HTTPS and declare an exact byte size and SHA-256 digest;
redirects, URL credentials, fragments, oversized payloads, and integrity
mismatches are rejected.

The relay's own HTTPS origin is trusted automatically. If a community stores
media on another public origin, add its exact `host` or `host:port` to
`attachment_hosts` under `gateway.platforms.buzz.extra`. Non-default ports
must be listed explicitly. Protected media that requires authenticated
retrieval through the Buzz CLI is not handled by this native public-URL path.

## Run the gateway

```bash
hermes gateway start
```

Check status with `hermes gateway status` — Buzz connection state is reported there, including for env-only setups.

## Notes and limitations

- **`BUZZ_*` env vars are available in terminal tool children for Buzz sessions** — the agent can invoke the `buzz` CLI directly (e.g. `buzz messages send ...`) because `BUZZ_PRIVATE_KEY`, `BUZZ_AUTH_TAG`, `BUZZ_RELAY_URL`, and the other `BUZZ_*` variables are passed through to terminal subprocesses when the session's platform is `buzz` or the process is a Buzz Desktop managed agent (`BUZZ_MANAGED_AGENT`). Non-Buzz sessions on the same host, `execute_code`, and other non-terminal spawns remain sealed.
- **Inbound streaming has a watchdog.** On the WebSocket transport a connection that goes quiet for five minutes, or whose socket the relay closed underneath us, is torn down and reconnected with backoff; while that happens the gateway health (`/health/detailed`, dashboard status) reports Buzz as `retrying`, not `connected`. On the `poll` transport the adapter polls `buzz messages get` per watched channel every `poll_interval` seconds (default 4), so expect up to one interval of latency.

- **Inbound is WebSocket-first with polling fallback.** `transport: auto` (the default) uses an authenticated WebSocket and falls back to CLI polling if initial WebSocket authentication is unavailable. `transport: websocket` requires WebSocket startup; `transport: poll` forces polling. Native Activity requires WebSocket delivery and cannot be enabled with explicit poll-only transport.
- On (re)connect the adapter seeds its high-water mark from the newest events, so channel history is never replayed into the agent.
- New DM conversations are discovered automatically (every few poll sweeps).
- The private key is passed to the CLI via the subprocess environment — it never appears in argv or logs.
