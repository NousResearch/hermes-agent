---
sidebar_position: 18
title: "Messenger"
description: "Set up Hermes Agent as a Facebook Messenger Page bot"
---

# Facebook Messenger Setup

Run Hermes Agent as a Facebook Messenger bot for Page DMs through Meta's Graph API. The adapter is a bundled platform plugin under `plugins/platforms/messenger/`, so it is discovered at gateway startup without adding a core platform enum.

Messenger is webhook-based: Meta sends inbound Page messages to a public HTTPS URL, and Hermes replies through the Page Send API.

## Quick Setup

1. Create a Meta Developer account and app at [developers.facebook.com](https://developers.facebook.com/).
2. Add the Messenger product to your app and connect a Facebook Page.
3. Generate a Page Access Token for your Page.
4. Copy the App Secret from **App Settings > Basic**.
5. Generate a secure verify token, for example `openssl rand -hex 32`.
6. Add the three credential values to `~/.hermes/.env`.
7. Put Messenger behavior settings in `~/.hermes/config.yaml`.
8. Start the Hermes gateway.
9. Configure the Meta webhook callback URL as `https://<public-host>/messenger/webhook`.
10. Subscribe the Page to `messages` and `messaging_postbacks`.
11. DM the Page and approve the pairing code for the sender, or preconfigure `allow_from`.

## Requirements

- A Meta app with the Messenger product enabled.
- A Facebook Page connected to that app.
- A Page access token with `pages_messaging`.
- The app secret from **App Settings > Basic**.
- A webhook verify token that you choose.
- A public HTTPS route to the local Messenger webhook port.

## Configure Hermes

Add the default-account credentials to `~/.hermes/.env`:

```env
MESSENGER_PAGE_ACCESS_TOKEN=EA...
MESSENGER_APP_SECRET=...
MESSENGER_VERIFY_TOKEN=...
```

Put non-secret behavior settings in `~/.hermes/config.yaml`:

```yaml
gateway:
  platforms:
    messenger:
      enabled: true
      dm_policy: pairing
      extra:
        host: 0.0.0.0
        port: 8650
        webhook_path: /messenger/webhook
        api_version: v21.0
```

`dm_policy: pairing` is the safe default. Unknown senders receive a pairing code and cannot talk to the agent until you approve it.

For a private deployment, pre-allow the Messenger PSIDs that may talk to Hermes:

```yaml
gateway:
  platforms:
    messenger:
      enabled: true
      allow_from:
        - "12345678901234567"
        - "98765432109876543"
```

Use `allow_all_users: true` only for tightly controlled development Pages.

## Meta Developer Setup

Meta's developer dashboard requires several manual steps before webhooks are delivered.

### 1. Create a Meta Developer App

1. Open [developers.facebook.com](https://developers.facebook.com/).
2. Click **My Apps > Create App**.
3. Select a business-capable app type or the Messenger use case.
4. Fill in the app name and contact email, then create the app.

### 2. Add Messenger and Connect a Page

1. In the app dashboard, add or open the **Messenger** product.
2. Under **Access Tokens**, click **Add or Remove Pages**.
3. Connect the Facebook Page Hermes should answer from.
4. Generate a Page Access Token for that Page and save it as `MESSENGER_PAGE_ACCESS_TOKEN`.

### 3. Copy the App Secret

1. Go to **App Settings > Basic**.
2. Click **Show** next to **App Secret**.
3. Save that value as `MESSENGER_APP_SECRET`.

Hermes uses this secret to verify the `X-Hub-Signature-256` header on every inbound POST.

### 4. Expose the Webhook

Meta requires a public HTTPS callback URL. Use a fixed reverse proxy, Cloudflare Tunnel, ngrok, or another HTTPS tunnel for development.

Example ingress target:

```text
https://messenger.example.com/messenger/webhook -> http://localhost:8650/messenger/webhook
```

Start the gateway before verifying the callback:

```bash
hermes gateway
```

The gateway log should include:

```text
Messenger webhook listening on 0.0.0.0:8650 for 1 account(s)
```

You can test the public GET verification path with the same verify token configured in Meta:

```bash
curl "https://<public-host>/messenger/webhook?hub.mode=subscribe&hub.verify_token=<verify-token>&hub.challenge=ok"
```

It should return exactly:

```text
ok
```

### 5. Configure the Meta Webhook

1. In the Meta Developer Portal, open **Messenger > Settings > Webhooks**.
2. Click **Add Callback URL**.
3. Enter `https://<public-host>/messenger/webhook`.
4. Enter the exact `MESSENGER_VERIFY_TOKEN` value.
5. Click **Verify and Save**.

Meta sends a GET request with `hub.verify_token` and `hub.challenge`. Hermes must be running and reachable for this step to pass.

### 6. Subscribe to Webhook Fields

After verifying the callback, subscribe the Page to:

| Field | Required | Purpose |
|-------|----------|---------|
| `messages` | Yes | Receives text messages and media attachments from users. |
| `messaging_postbacks` | Recommended | Receives postback payloads when users tap structured message buttons. |
| `message_reads` | Optional | Read receipts; ignored by Hermes. |
| `message_deliveries` | Optional | Delivery confirmations; ignored by Hermes. |

### 7. Go Live

Development-mode apps only work for app admins, developers, and testers. To let arbitrary Facebook users message the Page, request App Review for `pages_messaging` and switch the app to Live mode after approval.

## Configuration Reference

Default-account credentials are secrets and belong in `~/.hermes/.env`:

| Variable | Description |
|----------|-------------|
| `MESSENGER_PAGE_ACCESS_TOKEN` | Facebook Page access token with Messenger permissions. |
| `MESSENGER_APP_SECRET` | Meta app secret used for webhook HMAC-SHA256 verification. |
| `MESSENGER_VERIFY_TOKEN` | Webhook verify token you configure in the Meta dashboard. |

Non-secret behavior belongs in `config.yaml`:

| Key | Description |
|-----|-------------|
| `gateway.platforms.messenger.extra.host` | Webhook bind host. Defaults to `0.0.0.0`. |
| `gateway.platforms.messenger.extra.port` | Webhook listen port. Defaults to `8650`. |
| `gateway.platforms.messenger.extra.webhook_path` | Webhook path. Defaults to `/messenger/webhook`. |
| `gateway.platforms.messenger.extra.api_version` | Meta Graph API version. Defaults to `v21.0`. |
| `gateway.platforms.messenger.allow_from` | Messenger PSIDs allowed to DM Hermes. |
| `gateway.platforms.messenger.allow_all_users` | Allow any Messenger sender. Use only for controlled development. |
| `gateway.platforms.messenger.home_channel.chat_id` | Default Messenger PSID for cron or notification delivery. |
| `gateway.platforms.messenger.dm_policy` | Unauthorized DM behavior. `pairing` sends pairing codes; `disabled` drops inbound DMs. |

Hermes can also read credentials from file-backed YAML references:

```yaml
gateway:
  platforms:
    messenger:
      enabled: true
      extra:
        token_file: "~/.hermes/secrets/messenger-page-token"
        secret_file: "~/.hermes/secrets/messenger-app-secret"
        verify_token_file: "~/.hermes/secrets/messenger-verify-token"
```

Environment variables take priority over YAML values for the default account.

## Multi-Account

Use `gateway.platforms.messenger.extra.accounts` for multiple Facebook Pages. Each account needs its own webhook path and matching Meta callback URL.

```yaml
gateway:
  platforms:
    messenger:
      enabled: true
      extra:
        accounts:
          page-a:
            token_file: "~/.hermes/secrets/page-a-token"
            secret_file: "~/.hermes/secrets/page-a-app-secret"
            verify_token_file: "~/.hermes/secrets/page-a-verify-token"
            webhook_path: "/messenger/webhook/page-a"
          page-b:
            token_file: "~/.hermes/secrets/page-b-token"
            secret_file: "~/.hermes/secrets/page-b-app-secret"
            verify_token_file: "~/.hermes/secrets/page-b-verify-token"
            webhook_path: "/messenger/webhook/page-b"
```

For named accounts, route cron or manual sends to `account-id:psid`, for example `page-a:12345678901234567`.

## Capabilities

| Feature | Support |
|---------|---------|
| Text DMs | ✅ |
| Postbacks | ✅ |
| Inbound images/video/audio/files | ✅ |
| Outbound text | ✅ |
| Outbound image URL | ✅ |
| Typing indicator | ✅ |
| Threads/groups | — |

Outbound text is chunked to Messenger's 2000-character limit. Messenger does not reliably render Markdown, so Hermes strips common Markdown markers and keeps bare URLs.

Messenger users can type text commands such as `/new`, `/reset`, `/status`, and `/model` as regular messages. The gateway handles them the same way it handles text commands on other messaging platforms.

## Troubleshooting

**Webhook verification returns 403**

- The verify token in Meta does not exactly match `MESSENGER_VERIFY_TOKEN`.
- Confirm you are verifying the same callback path Hermes is listening on.

**Webhook verification returns 502 or times out**

- The public tunnel or reverse proxy is not reaching the configured `gateway.platforms.messenger.extra.port`.
- The gateway must be running before clicking **Verify and Save**.
- Meta requires HTTPS for the public callback URL.

**POST returns 401**

- `MESSENGER_APP_SECRET` does not match the Meta app secret.
- A proxy may be modifying the raw request body before Hermes verifies the signature.

**No messages arrive**

- Confirm the Page is subscribed to `messages`.
- In Development mode, confirm the sender is an app admin, developer, or tester.
- In Live mode, confirm the app has `pages_messaging` approval.
- Check logs with `hermes logs --follow` and look for `messenger` entries.

**Token errors**

- Generate a long-lived Page token or System User token for production.
- Confirm the token belongs to the connected Page.
- Test the token with `curl -H "Authorization: Bearer $MESSENGER_PAGE_ACCESS_TOKEN" https://graph.facebook.com/v21.0/me`.

**Unknown sender gets a pairing code**

- Approve the code with `hermes pairing approve messenger <code>`.
- Or add the sender PSID to `gateway.platforms.messenger.allow_from`.

**Bot replies but the user cannot see messages**

- Confirm the Page Access Token has `pages_messaging`.
- Confirm the recipient is allowed by the app's Development/Live mode.
