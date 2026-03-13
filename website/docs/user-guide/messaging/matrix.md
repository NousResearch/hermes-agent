---
sidebar_position: 6
title: "Matrix"
description: "Set up Hermes Agent as a Matrix bot on any homeserver"
---

# Matrix Setup

Hermes Agent integrates with the [Matrix](https://matrix.org/) protocol, letting you chat with your agent from Element, Cinny, FluffyChat, or any other Matrix client. The adapter connects to any Matrix homeserver — including self-hosted [Synapse](https://github.com/element-hq/synapse) or [Dendrite](https://github.com/matrix-org/dendrite) instances — using a simple access token. It auto-joins rooms on invite, supports text, images, files, and message editing, and delivers cron job results to a designated home room.

The integration is built on [matrix-nio](https://github.com/poljar/matrix-nio) and does **not** require end-to-end encryption (E2EE) to be set up — it works with standard unencrypted rooms out of the box.

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

## Step 2: Generate an Access Token

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

Copy the `access_token` value (it starts with `syt_`). This is what goes into `MATRIX_ACCESS_TOKEN`.

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
2. Bot access token (the `syt_...` token from Step 2)
3. Bot Matrix user ID (e.g., `@hermes:matrix.example.org`)
4. Home room ID (optional — for cron job delivery)
5. Allowed Matrix users (comma-separated, e.g., `@alice:matrix.org`)
6. Verify SSL (set to `false` only for self-signed certificates)

### Option B: Manual Configuration

Add the following to `~/.hermes/.env`:

```bash
# Required
MATRIX_HOMESERVER_URL=https://matrix.example.org
MATRIX_ACCESS_TOKEN=syt_aGVybWVz_XXXXXXXXXXXXXXXXXXXX
MATRIX_USER_ID=@hermes:matrix.example.org

# Optional — recommended
MATRIX_ALLOWED_USERS=@alice:matrix.org,@bob:example.org

# Optional — for cron job delivery
MATRIX_HOME_CHANNEL=!roomid:matrix.example.org
MATRIX_HOME_CHANNEL_NAME=Home

# Optional — set to false for self-signed TLS certificates
MATRIX_VERIFY_SSL=true
```

### Start the Gateway

```bash
hermes gateway
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

If your homeserver uses a self-signed TLS certificate (common on LAN or private Kubernetes deployments), set:

```bash
MATRIX_VERIFY_SSL=false
```

This disables TLS certificate verification for the Matrix connection only. Use this only for private, trusted homeservers — not for public-facing deployments.

## Encryption (E2EE)

The current adapter operates without end-to-end encryption. Rooms must be **unencrypted** for Hermes to read and send messages. E2EE support is planned for a future release.

:::tip
To ensure a room is unencrypted in Element: create a new room and leave the "Enable end-to-end encryption" toggle off. You cannot disable E2EE in an existing encrypted room.
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
