---
sidebar_position: 17
title: "Nostr"
description: "Connect Hermes to Nostr — decentralized NIP-17 encrypted DMs over relay WebSockets"
---

# Nostr 🔑

Connect Hermes to [Nostr](https://nostr.com/) — a decentralized, censorship-resistant messaging protocol. Hermes listens for [NIP-17](https://github.com/nostr-protocol/nips/blob/master/17.md) encrypted DMs on one or more relay WebSocket connections and replies using the same standard.

### Why Nostr?

Nostr gives you a private, sovereign channel to your AI assistant that no platform can take away:

- **End-to-end encrypted.** Messages are encrypted with NIP-44 (secp256k1 ECDH + ChaCha20) before they ever leave your device. Relays store and forward ciphertext — they never see your content.
- **You hold the keys.** Your bot's identity is a keypair you generate and control. There is no account to ban, no password to reset, no company that can lock you out.
- **Verified senders.** Every Nostr event carries a cryptographic signature. Hermes verifies it before processing — message senders cannot be spoofed or impersonated.
- **Censorship-resistant.** No single relay controls delivery. Configure two or more relays and your messages get through even if one goes down or decides to drop your traffic.
- **Permissionless.** No API keys to apply for, no terms of service approval, no rate-limit quotas handed down by a third party.

If you want a messaging channel where the infrastructure is yours, the encryption is end-to-end, and the protocol is open, Nostr is a strong choice.

## Prerequisites

- Python package: `pip install 'hermes-agent[nostr]'` (installs `nostr-sdk`)
- A Nostr keypair (see [Generate a keypair](#generate-a-keypair) below)
- One or more accessible Nostr relay URLs (WSS)

> **Platform support:** `nostr-sdk` is a Rust/PyO3 binding distributed as wheels only (no sdist). Wheels exist for macOS, Linux, and Windows on common Python versions. **Termux/Android is not supported.**

## Setup

### 1. Generate a keypair

Nostr uses secp256k1 keys. The easiest path is to let the setup wizard generate one for you (see step 3). You can also export an `nsec` from any Nostr client (Damus, Amethyst, Iris, Snort) and reuse it here.

To generate a keypair from the command line via Python:

```python
from nostr_sdk import Keys

keys = Keys.generate()
print(f"Private key (nsec): {keys.secret_key().to_bech32()}")
print(f"Public key  (npub): {keys.public_key().to_bech32()}")
```

Keep the `nsec` secret. Share the `npub` so users can find and message your bot.

### 2. Choose relays

Pick at least two reliable public relays for redundancy, or run your own:

- `wss://relay.damus.io`
- `wss://nos.lol`
- `wss://relay.nostr.band`
- `wss://relay.snort.social`

### 3. Configure Hermes

Run the interactive setup wizard:

```bash
hermes gateway setup
```

Select **Nostr** and enter your private key and relay URLs. Or set environment variables directly in `~/.hermes/.env`:

```bash
NOSTR_PRIVATE_KEY=nsec1...
NOSTR_RELAYS=wss://relay.damus.io,wss://nos.lol
```

### 4. Authorize users

`NOSTR_ALLOWED_NPUBS` controls who can message the bot. **The default is deny all** — no one can reach the bot until you configure this. Allowlist entries can be either `npub1…` bech32 or 64-char hex; both forms are accepted.

**Authorize specific npubs** (recommended):

```bash
NOSTR_ALLOWED_NPUBS=npub1abc...,npub1xyz...
```

**Open access** — allow anyone on Nostr (use with caution, especially if the bot has terminal access):

```bash
NOSTR_ALLOWED_NPUBS=*
```

**Unauthorized senders are silently ignored on Nostr** — the bot sends no reply at all. There is no DM pairing flow on this platform.

### 5. Start the gateway

```bash
hermes gateway run
```

Hermes opens a persistent WebSocket connection to each configured relay and subscribes to NIP-17 encrypted DM events addressed to the bot's public key.

## How It Works

```
Sender → NIP-17 encrypted DM → Nostr relay (WSS) → Hermes
Hermes → NIP-17 encrypted DM → Nostr relay (WSS) → Recipient
```

- **Inbound:** Hermes maintains a persistent WebSocket subscription to each relay and receives new DMs as push events. No polling.
- **Outbound:** Responses are published as NIP-17 sealed and giftwrapped DMs back to the sender via the same relays.
- **Encryption:** All DMs are end-to-end encrypted using NIP-17 (secp256k1 ECDH + NIP-44 encryption). The relay never sees plaintext message content.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NOSTR_PRIVATE_KEY` | Yes | — | Bot's private key — `nsec` bech32 string (recommended) or 64-char hex |
| `NOSTR_RELAYS` | Yes | — | Comma-separated relay WSS URLs |
| `NOSTR_ALLOWED_NPUBS` | No | deny all | Authorized npubs/hex pubkeys — `*` to allow everyone, comma-separated list for specific users, empty to deny all |
| `NOSTR_HOME_CHANNEL` | No | — | Default npub or hex pubkey for cron delivery |
| `NOSTR_BOT_NAME` | No | bot's npub | Display name published in the bot's kind-0 profile |
| `NOSTR_BOT_ABOUT` | No | — | Bio/description for the bot's kind-0 profile |
| `NOSTR_BOT_PICTURE` | No | — | Avatar URL for the bot's kind-0 profile |
| `NOSTR_NIP05` | No | — | NIP-05 verification identifier (e.g. `bot@yourdomain.com`) |
| `NOSTR_LUD16` | No | — | LUD-16 Lightning address for zaps (e.g. `bot@yourdomain.com`) |
| `NOSTR_BOT_WEBSITE` | No | — | Website URL for the bot's kind-0 profile |
| `NOSTR_EXPIRATION_MINUTES` | No | `10080` (7 days) | NIP-40 TTL for outbound messages, in minutes |
| `NOSTR_SEEN_MAX` | No | `1000` | Rolling window size for dedup cache persisted to disk |
| `NOSTR_LOOKBACK_MINUTES` | No | `2880` (48h) | Max age of gift wraps the subscription will replay on connect — see [Lookback window](#lookback-window) |
| `NOSTR_ALLOW_ALL_USERS` | No | `false` | Gateway-level authorization override — see [Authorization layers](#authorization-layers) |

### Authorization layers

Nostr has **two** authorization gates, evaluated independently:

1. **Adapter-level** (`NOSTR_ALLOWED_NPUBS`, `NOSTR_ALLOW_ALL_USERS`): consulted inside `NostrAdapter._process_event` before a DM is recorded in the dedup window. Parsed once at adapter init — **changes require a gateway restart**.
2. **Gateway-level** (`NOSTR_ALLOW_ALL_USERS` and the same `NOSTR_ALLOWED_NPUBS`): consulted by the gateway's `_is_user_authorized()` before a message is dispatched to the agent. Read from env at request time — changes apply on the next message.

For a sender to reach the agent, **both** gates must pass. The adapter gate is the strict security perimeter (silently drops disallowed DMs, never records them in the seen-list — so a flood from blocked senders cannot evict legitimate IDs). The gateway gate is the per-platform override layer used by other adapters (Discord, Slack) and is honored here for consistency.

`NOSTR_ALLOW_ALL_USERS=true` and `NOSTR_ALLOWED_NPUBS=*` are equivalent — either form opens both gates. The boolean exists to match the cross-platform `<PLATFORM>_ALLOW_ALL_USERS` convention used by Discord, Slack, Telegram, and others; use whichever reads better in your env file. Accepted truthy values are `true`, `1`, `yes` (case-insensitive).

## Lookback window

`NOSTR_LOOKBACK_MINUTES` caps how far back the relay subscription replays gift wraps when the gateway connects, preventing unbounded historical replay on a fresh install or restart.

The default is `2880` minutes (48 hours), which matches NIP-59's `created_at` randomization window. **Setting this lower than 2880 risks dropping legitimate inbound DMs** whose timestamps were randomized further back than your window.

## Health monitoring

The adapter supervises its own notification loop. `nostr-sdk`'s `Client.handle_notifications()` is opaque — it can return silently when the relay pool exhausts its internal retries — so a watchdog task polls observable health every 60 seconds:

- The notification task is alive (not done, not raised).
- At least one configured relay reports a connectable status (`CONNECTED`, `CONNECTING`, `PENDING`, `INITIALIZED`, or `SLEEPING`).

If either signal fails, the watchdog runs a soft-reconnect cycle: cancel the notification task, disconnect the client, settle for 1 second so the OS can recycle sockets, reconnect, re-subscribe to the gift-wrap filter (using a short 10-minute lookback rather than the full 48 hours — events that arrived during the outage will be backfilled, older events are already in the dedup window), and restart the notification task.

Recovery uses a fast initial ramp (5s, 15s, 45s) for the first three attempts to catch transient blips, then settles into a steady-state cadence of 90 seconds. **The watchdog retries indefinitely**; a long network outage will self-heal once relays are reachable again. Matching log lines to look for:

- `Nostr watchdog: notification loop unhealthy — attempting recovery`
- `Nostr watchdog: recovery attempt N (delay Xs)`
- `Nostr watchdog: recovery succeeded on attempt N`

Profile metadata (Kind 0) and the DM relay list (Kind 10050) are **not** republished during recovery — relays already hold those events and a reconnect storm should not fan out duplicate metadata.

## config.yaml Example

```yaml
platforms:
  nostr:
    enabled: true
    token: "${NOSTR_PRIVATE_KEY}"
    extra:
      relays: "wss://relay.damus.io,wss://nos.lol"
      name: "My Hermes Bot"
      about: "An AI assistant powered by Hermes Agent"
      nip05: "bot@yourdomain.com"
      lud16: "bot@yourdomain.com"
      expiration_minutes: 10080
      lookback_minutes: 2880
    home_channel:
      platform: nostr
      chat_id: "npub1..."
      name: "Owner"
```

## NIP-05 Verification

NIP-05 lets users discover your bot by a human-readable identifier like `bot@yourdomain.com`. To set it up:

1. Set `NOSTR_NIP05=bot@yourdomain.com` (or the `nip05` key in `config.yaml`)
2. Host a JSON file at `https://yourdomain.com/.well-known/nostr.json`:

```json
{
  "names": {
    "bot": "<your-bot-hex-pubkey>"
  }
}
```

3. The bot's kind-0 profile event will include the `nip05` field automatically.

## Allowlist Configuration

Access is controlled by `NOSTR_ALLOWED_NPUBS`. The default is **deny all** — set it before expecting any inbound DMs.

```bash
# Specific users only (recommended)
NOSTR_ALLOWED_NPUBS=npub1abc...,npub1xyz...

# Open access — anyone on Nostr can DM the bot
NOSTR_ALLOWED_NPUBS=*
```

Both formats are accepted and normalized internally.

## Home Channel

The home channel is used for cron job delivery and background task results. Set it to the npub or hex public key of the user who should receive these messages:

```bash
NOSTR_HOME_CHANNEL=npub1abc...
```

## Cron Delivery Example

Schedule a job to deliver results via Nostr:

```python
cronjob(
    action="create",
    prompt="Summarize today's AI news",
    schedule="0 9 * * *",
    deliver="nostr",
)
```

Or target a specific Nostr public key:

```python
cronjob(
    action="create",
    prompt="Daily briefing",
    schedule="0 8 * * *",
    deliver="nostr:npub1abc...",
)
```

## Troubleshooting

### "Cannot connect to relay"

- Verify the relay URL starts with `wss://` (not `ws://` for public relays)
- Test the relay directly: `websocat wss://relay.damus.io`
- Some relays require authentication or payment — check the relay's NIP-11 info document (`https://relay.damus.io` → `Accept: application/nostr+json`)
- Check `hermes logs gateway` for WebSocket error details

### Messages not arriving

- Confirm the bot's public key is correct — derive it from the private key using the Python snippet in [Generate a keypair](#generate-a-keypair)
- Verify the sending client is publishing to one of the same relays the bot subscribes to
- Check that the DM is formatted as NIP-17 (not the deprecated NIP-04)
- Watch the live WebSocket stream: `hermes logs -f`

### "Unauthorized user" rejections

- Add the user's npub to `NOSTR_ALLOWED_NPUBS` (either bech32 or hex form is accepted)
- For temporary open access during testing, set `NOSTR_ALLOWED_NPUBS=*`
- Note: Hermes' DM pairing flow does not apply on Nostr — unauthorized senders receive no reply at all

### Bot profile not visible

- The bot publishes a kind-0 profile event on startup. If it is not visible, check that the relay accepted it (some relays reject events without a valid NIP-05)
- Re-publish manually by restarting the gateway

### High relay latency

- Add a second relay — messages are published to all configured relays simultaneously
- Use geographically closer relays for lower round-trip times
