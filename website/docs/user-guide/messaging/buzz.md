# Buzz

The Buzz adapter connects Hermes to a [Buzz](https://github.com/block/buzz) community — Block's open-source human+agent collaboration platform built on the Nostr protocol — and relays messages between Buzz channels (or DMs) and the agent. Outbound traffic shells out to the `buzz` CLI binary ("JSON in, JSON out"); inbound uses a native Nostr WebSocket subscription (via the already-bundled `websockets` package) with CLI polling as fallback. **No extra Python packages are required** — just the `buzz` binary.

Buzz renders markdown, so agent replies keep their formatting. Images are delivered as uploads (local files) or links (URLs). Replies can thread onto an existing message via its event id. When progress or status messages are enabled, they inherit the triggering Buzz event as their reply anchor instead of appearing as unrelated top-level channel posts.

Files sent **to** the agent are fetched back off the relay with the agent's authenticated identity and cached locally, so tools receive a real file path rather than a `/media/…` URL that anonymous requests cannot read. Images, audio, video, and documents (PDFs and the like) are all handled.

Inbound messages arrive over a persistent NIP-42-authenticated Nostr WebSocket subscription by default (near-instant delivery), with automatic fallback to CLI polling when the WebSocket can't be established. Outbound messages always go through the `buzz` CLI. Control it with `transport` / `BUZZ_TRANSPORT`: `auto` (default), `websocket` (require WS, fail otherwise), or `poll`. If your relay membership uses NIP-OA owner attestation, set `BUZZ_AUTH_TAG` to the four-string auth tag JSON.

> Run `hermes gateway setup` and pick **Buzz** for a guided walk-through.

## Recommended path (what actually works)

Path ③ keeps full Hermes (memory, skills, approvals, cron, multi-platform gateway). The chicken-and-egg is community membership: the agent key must already be allowed on the relay, and most community relays also want a NIP-OA auth tag.

**Practical order:**

1. **Mint identity in Buzz Desktop**

   Create the agent in Desktop so the community issues the Nostr keypair and NIP-OA auth tag. This is the easy membership step.

2. **Stop Desktop ACP on that key**

   Stop the worker and turn off start-on-launch. Do **not** leave Desktop ACP and the Hermes gateway running on the same agent key.

3. **Hand the secrets to Hermes**
   ```bash
   hermes gateway setup   # pick Buzz
   ```
   Paste:
   - relay URL (`https://your-community.communities.buzz.xyz`)
   - agent `nsec` from Desktop create
   - NIP-OA `BUZZ_AUTH_TAG` JSON from Desktop create
   - owner allowlist (your pubkey)
   - `require_mention=true` (recommended)

4. **Join channels + publish profile**

   Auth alone is not enough. If the agent is not a **channel member**, humans cannot DM or `@mention` it in Desktop search. The setup wizard can join all visible channels and set a display name. You can also:
   ```bash
   buzz users set-profile --name "Your Agent"
   buzz channels join --channel <uuid>
   ```

5. **Reload the Hermes gateway process** (from a terminal outside the running gateway), then smoke-test:
   - untagged DM → replies
   - channel without mention → silence (when `require_mention=true`)
   - `@Agent` in channel → replies

:::tip Field failure modes
- **Connected but unfindable in DM/@ search** → not a channel member yet (join channels, set profile).
- **Weird double replies / auth flaps** → Desktop ACP still running on the same key.
- **`relay_membership_required` / AUTH rejected** → missing or wrong `BUZZ_AUTH_TAG`, or identity never finished Desktop create.
:::

## Prerequisites

- The `buzz` CLI binary on your `PATH` (or point `BUZZ_CLI_PATH` at it) — build it from the [Buzz repo](https://github.com/block/buzz) with `cargo build --release -p buzz-cli`
- A Buzz community relay URL (e.g. `https://mycommunity.communities.buzz.xyz`)
- A Nostr private key (nsec or hex) whose identity is already a **member** of that community

### Hermes Cloud: install the CLI without Rust

The Buzz Desktop Debian package includes the self-contained `buzz` CLI. On an x86-64 Hermes Cloud instance, extract only that binary to the persistent volume. The commands below pin the current verified Buzz release, [v0.5.2](https://github.com/block/buzz/releases/tag/v0.5.2):

```bash
cd /tmp
curl -fsSL -o buzz.deb https://github.com/block/buzz/releases/download/v0.5.2/Buzz_0.5.2_amd64.deb
mkdir -p /opt/data/.local/bin
dpkg-deb --fsys-tarfile buzz.deb | tar -xO usr/bin/buzz > /opt/data/.local/bin/buzz
chmod +x /opt/data/.local/bin/buzz
rm buzz.deb
buzz --version
```

`/opt/data/.local/bin` is already on the Hermes Cloud image `PATH`, and `/opt/data` persists across restarts and image updates.

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
| `BUZZ_AUTH_TAG` | — | NIP-OA four-string auth tag JSON from Desktop create (required on many community relays) |
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
        attachment_hosts: []         # additional exact HTTPS host[:port] origins for inbound files
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

## Access control

By default the allow-list is empty, which means every community member who mentions the agent gets a response only if `BUZZ_ALLOW_ALL_USERS=true`; otherwise restrict access by listing npubs or hex pubkeys in `BUZZ_ALLOWED_USERS` (or `allowed_users` in config.yaml). Community membership itself is enforced by the relay — only members can post.

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

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Gateway skips Buzz | Missing `BUZZ_RELAY_URL` or key | Re-run setup; confirm `.env` |
| AUTH / membership errors | Missing Desktop mint or auth tag | Create agent in Desktop, copy tag, stop ACP |
| Connected, 0 channels | Not a channel member | `buzz channels join` or setup join-all |
| DM search: no matching users | No profile / not in channels | `set-profile` + join channels |
| Double replies | Desktop ACP + Hermes same key | Stop Desktop worker |
| CLI not found | `buzz` off PATH | Install CLI or set `BUZZ_CLI_PATH` |

## Notes and limitations

- **`BUZZ_*` env vars are available in terminal tool children for Buzz sessions** — the agent can invoke the `buzz` CLI directly (e.g. `buzz messages send ...`) because `BUZZ_PRIVATE_KEY`, `BUZZ_AUTH_TAG`, `BUZZ_RELAY_URL`, and the other `BUZZ_*` variables are passed through to terminal subprocesses when the session's platform is `buzz` or the process is a Buzz Desktop managed agent (`BUZZ_MANAGED_AGENT`). Non-Buzz sessions on the same host, `execute_code`, and other non-terminal spawns remain sealed.
- **Inbound is polled, not streamed.** The `buzz` CLI is request/response, so the adapter polls `buzz messages get` per watched channel every `poll_interval` seconds (default 4). Expect up to one interval of latency on inbound messages. A future optimization is a websocket transport (the Buzz repo ships `buzz-ws-client` for true streaming).
- On (re)connect the adapter seeds its high-water mark from the newest events, so channel history is never replayed into the agent.
- New DM conversations are discovered automatically (every few poll sweeps).
- The private key is passed to the CLI via the subprocess environment — it never appears in argv or logs.
