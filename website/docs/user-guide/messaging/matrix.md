---
sidebar_position: 6
title: "Matrix"
description: "Set up Hermes Agent as a Matrix bot on any homeserver"
---

# Matrix Setup

Hermes Agent integrates with the [Matrix](https://matrix.org/) protocol, letting you chat with your agent from Element, Cinny, FluffyChat, or any other Matrix client. The adapter connects to any Matrix homeserver — including self-hosted [Synapse](https://github.com/element-hq/synapse) or [Dendrite](https://github.com/matrix-org/dendrite) instances — using a simple access token. It auto-joins rooms on invite, supports text, images, files, and message editing, and delivers cron job results to a designated home room.

The integration is built on [mautrix-python](https://github.com/mautrix/python) and does **not** require end-to-end encryption (E2EE) to be set up — it works with standard unencrypted rooms out of the box. E2EE is fully supported and includes automatic cross-signing bootstrap so Element shows the bot as a verified device.

## Step 1: Register a Bot Account

Create a dedicated Matrix account for your Hermes bot. On a self-hosted Synapse server, use the `register_new_matrix_user` script:

```bash
register_new_matrix_user \
  -c /etc/matrix-synapse/homeserver.yaml \
  --no-admin \
  -u hermes \
  -p <password>
```

Adjust the path to `homeserver.yaml` for your deployment. If you're using a Docker or Kubernetes setup, run the command inside the Synapse container.

:::tip
If you're using a public homeserver (e.g., `matrix.org`), you can register the bot account through any Matrix client or the homeserver's registration API. However, a self-hosted server gives you full control over rate limits and access.
:::

:::warning
Do **not** give the bot admin rights on the homeserver. The `--no-admin` flag above is the safe default.
:::

## Step 2: Generate an Access Token and Device ID

Hermes authenticates with Matrix using an access token (not a password). Generate one by calling the login API directly:

```bash
curl -s -XPOST 'https://<homeserver>/_matrix/client/v3/login' \
  -H 'Content-Type: application/json' \
  -d '{"type":"m.login.password","user":"hermes","password":"<password>"}'
```

The response looks like:

```json
{
  "access_token": "syt_aGVybWVz_XXXXXXXXXXXXXXXXXXXX",
  "device_id": "ABCDEFGHIJ",
  "home_server": "matrix.example.org",
  "user_id": "@hermes:matrix.example.org"
}
```

**Save both values:**

- `access_token` (starts with `syt_`) → `MATRIX_ACCESS_TOKEN`
- `device_id` (e.g. `ABCDEFGHIJ`) → `MATRIX_DEVICE_ID`

The `device_id` is critical. Without it, Synapse creates a new device session every time the gateway restarts and re-delivers all recent messages from that room. Setting it tells Synapse to resume the existing session.

:::warning
Keep your access token secret. It grants full account access. If it leaks, log out of the device via `/_matrix/client/v3/logout` or your homeserver admin panel.
:::

## Step 3: Find Your Matrix User ID

Your Matrix user ID is in the format `@username:homeserver` (e.g., `@alice:matrix.org`). You'll need this to add yourself to the allowlist.

You can confirm it in Element: **Settings → General → your Matrix ID**.

## Step 4: Configure Hermes

### Option A: Interactive Setup (Recommended)

```bash
hermes gateway setup
```

Select **Matrix** when prompted. The wizard asks for:

1. Homeserver URL (e.g., `https://matrix.example.org`)
2. SSL verification — set `false` early if using self-signed certs so connectivity tests work
3. Bot Matrix user ID (e.g., `@hermes:matrix.example.org`)
4. Bot access token (the `syt_...` token from Step 2) — validated live against `/account/whoami`
5. Device ID (the `device_id` from Step 2) — keeps E2EE sessions stable across restarts
6. E2EE — whether to enable end-to-end encryption (requires libolm + mautrix[e2be])
7. Allowed Matrix users (comma-separated — your personal Matrix ID, not the bot's)
8. Home room ID (optional — for cron job delivery)

### Option B: Manual Configuration

Add the following to `~/.hermes/.env`:

```bash
# Required
MATRIX_HOMESERVER_URL=https://matrix.example.org
MATRIX_ACCESS_TOKEN=syt_aGVybWVz_XXXXXXXXXXXXXXXXXXXX
MATRIX_USER_ID=@hermes:matrix.example.org

# Strongly recommended — prevents message replay on restart
MATRIX_DEVICE_ID=ABCDEFGHIJ

# Optional — recommended
MATRIX_ALLOWED_USERS=@alice:matrix.org,@bob:example.org

# Optional — for cron job delivery
MATRIX_HOME_CHANNEL=!roomid:matrix.example.org
MATRIX_HOME_CHANNEL_NAME=Home

# Set to false for self-signed TLS certificates (common for self-hosted servers)
MATRIX_VERIFY_SSL=true
```

### Start the Gateway

```bash
hermes gateway run
```

The bot will connect and begin polling for new events within a few seconds.

## Step 5: Invite the Bot

Invite the bot to any Matrix room. It will **automatically accept** the invite — but only if the inviting user is on your `MATRIX_ALLOWED_USERS` list (or if `GATEWAY_ALLOW_ALL_USERS=true`).

In Element: open the room → **Invite** → search for `@hermes:matrix.example.org`.

## Home Room

The home room is where cron job results and background notifications are delivered. Set it by running `/sethome` in any Matrix room where the bot is present, or set it manually in `~/.hermes/.env`:

```bash
MATRIX_HOME_CHANNEL=!roomid:matrix.example.org
MATRIX_HOME_CHANNEL_NAME="My Notes"
```

To find a room's ID in Element: open the room → **Settings → Advanced → Internal Room ID**.

## Self-Hosted Servers with Self-Signed Certificates

If your homeserver uses a self-signed TLS certificate — which is the case for any self-hosted Synapse deployment not fronted by Let's Encrypt — set:

```bash
MATRIX_VERIFY_SSL=false
```

The connection is still TLS-encrypted; this only disables certificate chain validation. It is safe for private homeservers on a trusted network (Tailscale, LAN). Do not use it for public-facing deployments.

The setup wizard asks about this **before** collecting credentials, since a `false` value is needed for connectivity tests to succeed against self-signed servers.

:::tip
If you want to avoid disabling verification, add your homeserver's self-signed CA certificate to the system trust store instead. On Arch Linux:
```bash
sudo trust anchor --store ~/path/to/homeserver.crt
sudo update-ca-trust
```
:::

## End-to-End Encryption (E2EE)

E2EE is **optional and disabled by default**. Without it, rooms must be unencrypted for Hermes to read and send messages. With it enabled, Hermes can participate in encrypted rooms.

### Requirements

E2EE requires two things beyond the base install:

**1. libolm — a system C library:**
```bash
# Arch Linux
sudo pacman -S libolm

# Debian / Ubuntu
sudo apt install libolm-dev

# macOS
brew install libolm
```

**2. mautrix-python with the E2EE extra:**
```bash
pip install 'mautrix[e2be]' base58
```

### Enabling E2EE

The setup wizard (`hermes gateway setup matrix`) guides you through the E2EE setup and checks for both libolm and mautrix. To enable manually:

```bash
MATRIX_E2EE=true
```

When E2EE is enabled the adapter:
- Creates a persistent SQLite crypto store at `~/.hermes/matrix/crypto.db` (backed by
  `PgCryptoStore` via `aiosqlite`) so all Olm and Megolm sessions survive gateway restarts
- Uploads device keys to the homeserver on first connect
- Automatically bootstraps cross-signing keys so Element shows the bot as verified
- Participates in Megolm key sharing so it can decrypt room messages

The SQLite store is what prevents the "no session found" decrypt errors that occur when
session state is only held in memory. This mirrors the approach used by maubot and all
production mautrix bridges.

If `MATRIX_E2EE=true` is set but the required packages are not installed, the adapter
logs an error and refuses to connect. Install all dependencies before enabling E2EE:

```bash
pip install "mautrix[e2be]" asyncpg aiosqlite base58
```

### Without E2EE

Create rooms with encryption disabled. In Element: **New Room → disable the "Enable end-to-end encryption" toggle**. You cannot disable E2EE in an existing encrypted room.

For private homeservers on Tailscale or a VPN, E2EE is often unnecessary — the network transport is already encrypted end-to-end.

:::note
The `MATRIX_DEVICE_ID` value becomes especially important with E2EE. Without a stable device ID, Synapse treats each restart as a new device and other room members must re-verify the bot's keys each time.
:::

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Bot not connecting | Check that `MATRIX_HOMESERVER_URL`, `MATRIX_ACCESS_TOKEN`, and `MATRIX_USER_ID` are all set. Verify the access token is still valid. |
| Bot rejects invites | The inviting user is not in `MATRIX_ALLOWED_USERS`. Add their full Matrix ID (e.g., `@alice:matrix.org`). |
| TLS/SSL errors | Your homeserver uses a self-signed certificate. Set `MATRIX_VERIFY_SSL=false`. |
| Bot can't read messages | The room is end-to-end encrypted. Create an unencrypted room and invite the bot there. |
| Access token expired/invalid | Re-run Step 2 to generate a new token. Update `MATRIX_ACCESS_TOKEN` in `~/.hermes/.env`. |
| Bot not delivering cron results | Ensure `MATRIX_HOME_CHANNEL` is set to a valid room ID and the bot has joined that room. |

## Security

:::warning
Always set `MATRIX_ALLOWED_USERS` to restrict who can interact with your bot. Without it, the gateway denies all users by default as a safety measure.
:::

Matrix user IDs follow the format `@username:homeserver`. Make sure to include the full ID including the homeserver domain — `@alice` alone is not a valid identifier.

For more details, see the [Security documentation](/user-guide/security). You can also use [DM pairing](/user-guide/messaging#dm-pairing-alternative-to-allowlists) for a more dynamic approach to user authorization.
