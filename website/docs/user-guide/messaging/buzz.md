# Buzz

The Buzz adapter connects Hermes to a [Buzz](https://github.com/block/buzz) community — Block's open-source human+agent collaboration platform built on the Nostr protocol — and relays messages between Buzz channels (or DMs) and the agent. Outbound traffic shells out to the `buzz` CLI binary ("JSON in, JSON out"); inbound uses a native Nostr WebSocket subscription (via the already-bundled `websockets` package) with CLI polling as fallback. **No extra Python packages are required** — just the `buzz` binary.

Buzz renders markdown, so agent replies keep their formatting. Images are delivered as uploads (local files) or links (URLs). Replies can thread onto an existing message via its event id. Final replies, progress/status messages, images, and files use the channel's effective reply-placement policy when they carry triggering-event metadata.

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
        allowed_users: []          # authorized sender npubs/hex keys; empty is private by default
        group_allow_admin_from:    # explicit admins allowed to change channel modes
          - npub1...
        channel_modes: {}          # sparse UUID-keyed overrides managed by /buzz
```

Plus, in `~/.hermes/.env`:

```
BUZZ_PRIVATE_KEY=nsec1...
```

### Option B — connection environment variables

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

Per-channel listening and reply modes do not add an environment variable. They
are behavioral configuration stored in the active profile's `config.yaml`.

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
        poll_interval: 4                  # seconds between fallback/poll-mode sweeps
        cli_path: ""                      # buzz binary (default: PATH, then ~/bin/buzz)
        credentials_file: ""              # JSON file with the nsec (BUZZ_PRIVATE_KEY fallback)
        allowed_users: []                 # empty = allow all if allow_all_users is true; otherwise restrict to listed npubs/hex pubkeys
        require_mention: true             # in channels: only respond when addressed (@name, npub, or hex pubkey); DMs always dispatch regardless
        allow_all_users: false            # true allows community members to chat; admins remain explicit below
        group_allow_admin_from:           # explicit channel-mode administrators
          - npub1...
```

**Why these defaults:**

- `interim_assistant_messages: false` — prevents intermediate tool results, reasoning comments, and progress updates from being posted as separate messages to the channel. Only the final response goes to the channel.
- `tool_progress: off` — suppresses tool progress bubbles (e.g., "Running terminal command...", "Reading file..."). Keeps the channel focused on actual results, not process.
- `poll_interval: 4` — controls fallback and explicit poll-mode frequency. Lower values reduce poll-mode latency but increase relay load; the default WebSocket path is near-instant.
- `allowed_users: []` + `allow_all_users: false` — private mode by default. Only listed users can interact. Set `allow_all_users: true` for community mode; channel-control administrators still require an explicit `group_allow_admin_from` entry.
- `require_mention: true` — in channels, the agent only responds when addressed. DMs always dispatch regardless of this setting.

**Rationale:** Channels are for final results and conversation, not for the agent's internal tool execution log. Users see the final answer, not the steps taken to get there. This matches the behavior on Telegram and email, which already have these defaults.

**Exception:** If you want users to see tool progress (e.g., for long-running operations), set `tool_progress: all` — but `interim_assistant_messages` should still be `false` to avoid spamming with every tool result.

## Mentions, channels, and DMs

- In shared channels the global default is to respond only when **addressed** — by `@name`, its npub, or its hex pubkey. A channel-specific `listen: always` override removes only this mention requirement; sender authorization still applies.
- A leading, standalone `/buzz` command token is treated as addressed so that channel status and controls work in mentions-only channels. Prose that merely contains `/buzz` does not bypass mention gating.
- Direct messages always reach the agent, no mention needed.
- The agent's own messages are never dispatched back to it (self-echo suppression by pubkey), and every event is de-duplicated by event id against a per-channel high-water mark.

## Per-channel controls

The native Buzz gateway exposes one channel-specific control surface:

| Command | Who can run it | Result |
|---|---|---|
| `/buzz status` | Any authorized sender | Show effective listening and reply modes, including whether each is inherited or explicit |
| `/buzz listen status` | Any authorized sender | Show the effective listening mode and source |
| `/buzz replies status` | Any authorized sender | Show the effective reply mode and source |
| `/buzz listen always` | Explicit channel admin | Process every otherwise-authorized channel message without requiring a mention |
| `/buzz listen mentions` | Explicit channel admin | Require a mention in this channel, even if the global setting is ambient |
| `/buzz listen reset` | Explicit channel admin | Remove only this channel's listening override |
| `/buzz replies flat` | Explicit channel admin | Post reply-capable output without a Buzz reply anchor |
| `/buzz replies threaded` | Explicit channel admin | Attach top-level input to a new thread and keep in-thread input on its stable NIP-10 root |
| `/buzz replies hybrid` | Explicit channel admin | Keep top-level output flat and in-thread output on the input's stable root |
| `/buzz replies reset` | Explicit channel admin | Remove only this channel's reply override |

Only the three exact status forms above are available to an ordinary authorized
user. Changing or resetting a mode requires an identity explicitly listed in
`group_allow_admin_from` for Buzz. Both lowercase hex public keys and their
equivalent `npub` encodings are accepted and normalized to the same identity.
Administrator status does not grant conversation access: the sender must also
pass the normal Buzz `allowed_users` / `allow_all_users` authorization gate.

The Buzz plugin also needs consent for its narrowly scoped platform action. If
an existing installation predates this capability, inspect and grant it
interactively:

```bash
hermes plugins capabilities buzz-platform
hermes plugins enable buzz-platform
```

The consent is recorded as `gateway.platform_actions` for the `buzz-platform`
plugin. A missing grant or missing explicit group administrator fails closed;
no configuration or live state is changed.

Channel overrides take precedence over the inherited global behavior:

1. `channel_modes.<channel UUID>.listen` or `.replies`, when present.
2. The existing global `require_mention` and `reply_to_mode` /
   `reply_in_thread` settings.
3. Buzz defaults: mention-gated listening and threaded replies.

`reset` removes only the selected property, so the channel immediately resumes
inheriting the current global setting. The stored mapping is intentionally
sparse and keyed by the immutable Buzz channel UUID:

```yaml
gateway:
  platforms:
    buzz:
      extra:
        group_allow_admin_from:
          - npub1...
        channel_modes:
          ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd:
            listen: always
            replies: hybrid
```

Successful commands atomically update the active profile's `config.yaml` and
apply to the currently connected adapter, so the next eligible message uses the
new mode without a restart. A replacement or restarted adapter loads the same
persisted mapping before accepting events. If persistence succeeds but the live
adapter update unexpectedly fails, the command says that the change was saved
and a restart is required; it does not claim immediate success.

One active gateway is the supported writer for each profile. `/buzz` mutations
serialize the complete read-merge-validate-write transaction across processes,
but concurrent manual edits of that profile's `config.yaml` while the gateway
is running are unsupported. Stop the gateway before making manual edits.

Channel policies do not apply to direct messages. `/buzz status` in a DM
explains the existing DM behavior; mutation and reset forms are no-ops and save
no override.

### Reply placement behavior

Replies are threaded by default. The effective per-channel mode applies
consistently to final and streamed text, enabled progress/status output, images,
and files that carry reply metadata:

| Mode | Top-level input | In-thread input |
|---|---|---|
| `flat` | No reply anchor | No reply anchor |
| `threaded` | Anchor to the triggering event, opening one thread | Anchor to the thread's stable NIP-10 root |
| `hybrid` | No reply anchor | Anchor to the thread's stable NIP-10 root |

If hybrid placement provenance is unavailable, Buzz falls back to flat output;
it does not invent a thread. Stable-root routing keeps a conversation on its
original root instead of nesting a new one-message sub-thread under every turn.

To change the inherited default for every channel without an explicit override,
set either of these equivalent global options (`reply_in_thread` matches the key
Slack uses):

```yaml
gateway:
  platforms:
    buzz:
      reply_to_mode: off          # PlatformConfig-level, like Discord/Telegram
      extra:
        reply_in_thread: false    # Slack-style compatibility key
```

The global opt-out also applies to out-of-process cron delivery
(`deliver=buzz`). A standalone delivery with an explicit thread target uses that
target in inherited/threaded or hybrid mode, and omits it in flat mode.

## Access control

By default the allow-list is empty and sender access is private. Set
`BUZZ_ALLOW_ALL_USERS=true` (or `allow_all_users: true` in `config.yaml`) to
accept any community member, or list authorized npubs/hex keys in
`BUZZ_ALLOWED_USERS` (or `allowed_users`). Community membership itself is
enforced by the relay — only members can post.

Sender authorization and channel-control authority are separate. `allowed_users`
or `allow_all_users` decides who may start an agent turn; the explicit
`group_allow_admin_from` list decides who among those authorized senders may
change `/buzz` channel policy. Never use the relay owner key or the ordinary
sender allow-list as an implicit administrator grant.

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

### Operator smoke test

On a gateway running this version, use a dedicated test channel and keep a
second channel as an isolation check:

1. Run `/buzz status` and record both inherited modes.
2. Run `/buzz listen always`, send one unmentioned message, and verify it starts
   exactly one agent turn.
3. Exercise `/buzz replies flat`, `/buzz replies threaded`, and
   `/buzz replies hybrid` with one top-level prompt and one in-thread follow-up
   each. Verify placement against the table above.
4. Restart the gateway, run `/buzz status`, and verify the explicit values and
   behavior survived.
5. Run `/buzz listen reset` and `/buzz replies reset`; verify status reports
   inherited values again.
6. Confirm the second channel's listening and reply behavior never changed.

## Notes and limitations

- **`BUZZ_*` env vars are available in terminal tool children for Buzz sessions** — the agent can invoke the `buzz` CLI directly (e.g. `buzz messages send ...`) because `BUZZ_PRIVATE_KEY`, `BUZZ_AUTH_TAG`, `BUZZ_RELAY_URL`, and the other `BUZZ_*` variables are passed through to terminal subprocesses when the session's platform is `buzz` or the process is a Buzz Desktop managed agent (`BUZZ_MANAGED_AGENT`). Non-Buzz sessions on the same host, `execute_code`, and other non-terminal spawns remain sealed.
- **Inbound prefers WebSocket with polling fallback.** In `auto` mode the adapter uses a persistent NIP-42-authenticated Nostr WebSocket subscription and falls back to `buzz messages get` polling when a WebSocket cannot be established. Explicit `websocket` and `poll` modes are also available.
- On (re)connect the adapter seeds its high-water mark from the newest events, so channel history is never replayed into the agent.
- New DM conversations are discovered automatically (every few poll sweeps).
- The private key is passed to the CLI via the subprocess environment — it never appears in argv or logs.
