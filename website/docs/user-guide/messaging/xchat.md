# X Chat (encrypted X DMs)

[X Chat](https://docs.x.com/xchat/introduction) is X's end-to-end encrypted direct-message system. The Hermes adapter connects your agent to a bot X account's DMs: message bodies are encrypted and decrypted **locally** with the official Chat XDK — X only ever routes ciphertext, and every message is signed so recipients can verify the sender.

> Run `hermes xchat setup` for a guided walk-through, or pick **X Chat** in `hermes gateway setup`.

## Prerequisites

- An **X developer account** with an app configured for **OAuth 2.0 user context** ([Developer Console](https://developer.x.com/en/portal/dashboard)). X Chat endpoints require API access on your developer plan.
- A **user access token** for the bot account with scopes: `dm.read`, `dm.write`, `users.read`, `tweet.read` (add `offline.access` to receive a refresh token so Hermes can auto-renew the ~2-hour access token).
- Python 3.10+ (the `chatxdk` E2EE binding is lazy-installed at first use).

## Setup

```bash
hermes xchat setup
```

The wizard:

1. Stores the OAuth2 access token (and optional refresh token + client id) in `~/.hermes/.env`.
2. Derives the bot account's numeric user id via `GET /2/users/me`.
3. Generates the E2EE identity + signing keypairs with the Chat XDK, saves the private-key blob to `~/.hermes/xchat/private_keys.b64` (mode 600), and registers the public keys with the X API.

Key registration is **rate limited to a few writes per 24 hours** per account. The setup is resume-safe: the key blob and registration payload are persisted *before* any network call, so an interrupted or rate-limited run resumes the same identity instead of minting a new one.

Check state anytime:

```bash
hermes xchat status
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `XCHAT_ACCESS_TOKEN` | Yes | OAuth2 user access token (dm.read, dm.write, users.read, tweet.read) |
| `XCHAT_REFRESH_TOKEN` | Recommended | Refresh token — enables automatic access-token renewal (rotated on every refresh and re-persisted) |
| `XCHAT_CLIENT_ID` | With refresh | X app OAuth2 client id (required for token refresh) |
| `XCHAT_CLIENT_SECRET` | Optional | Only for confidential clients |
| `XCHAT_USER_ID` | Auto | Bot account's numeric user id (derived by setup) |
| `XCHAT_SIGNING_KEY_VERSION` | Auto | Registered public-key version (written by setup) |
| `XCHAT_PRIVATE_KEYS_B64` | Optional | Key blob override — takes precedence over the blob file |
| `XCHAT_ALLOWED_USERS` | Recommended | Comma-separated numeric X user ids allowed to talk to the bot |
| `XCHAT_ALLOW_ALL_USERS` | Optional | `true` allows every sender (dev only) |
| `XCHAT_CONVERSATION_IDS` | Optional | Pin specific conversation ids to poll; omit to auto-discover |
| `XCHAT_POLL_INTERVAL` | Optional | Seconds between event polls (default `10`, floor `2`) |
| `XCHAT_REQUIRE_MENTION` | Optional | In group conversations, only respond when a wake word matches (default `false`) |
| `XCHAT_MENTION_PATTERNS` | Optional | Custom wake-word regexes (JSON list or comma-separated) |
| `XCHAT_HOME_CHANNEL` | Optional | Default conversation/user id for cron delivery |
| `XCHAT_HOME_CHANNEL_NAME` | Optional | Human label for the home channel |

## How it works

- **Inbound** — the adapter polls each conversation's events endpoint. On first sight of a conversation it batch-decrypts the backlog (`decrypt_events`) to seed the SDK's verified conversation-key cache **without replying to old messages**, then decrypts new events individually. `KeyChange` events (conversation-key rotations) are verified and folded into the key cache automatically.
- **Outbound** — replies are encrypted and signed locally (`encrypt_message` with the session identity), then POSTed as ciphertext.
- **Senders** — each new sender's public keys are fetched once and pushed into the XDK's signing-key store so their message signatures verify.
- **Identity** — user ids are numeric X user ids; conversation ids look like `123-456` (1:1) or `g123…` (group).

## Authorization

By default all senders are denied. Either:

1. Set `XCHAT_ALLOWED_USERS` to a comma-separated list of numeric X user ids, or
2. Use **DM pairing** — an unknown sender gets a pairing code; approve with `hermes pairing approve xchat <CODE>`.

## Group conversations

Group chats (`g…` conversation ids) work out of the box. To keep the bot quiet unless addressed:

```
XCHAT_REQUIRE_MENTION=true
```

The default wake words are `hermes` / `hermes agent`; override with `XCHAT_MENTION_PATTERNS`. DMs are never gated.

## Using X Chat with cron jobs

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="xchat",            # uses XCHAT_HOME_CHANNEL
    prompt="Check for alerts and summarise."
)
```

Or target a conversation directly:

```bash
hermes send xchat:<conversation-id> "Done!"
```

Standalone delivery opens an ephemeral E2EE session, seeds the conversation key from the conversation's backlog, encrypts, and sends — no running gateway required.

## Limitations

- **Text only for now.** Encrypted media upload/download (the streaming-encrypt flow + `media_hash_key` endpoints) is not wired yet; inbound attachments surface as text-free events and are skipped.
- **Reply flows only.** The bot answers conversations that exist; initiating a brand-new conversation (which requires a conversation-key handshake) is not supported yet.
- **Polling latency.** Inbound uses REST polling (default 10s). Webhook / activity-stream delivery may come later.
- **Access tier.** X Chat API availability depends on your X developer plan.
