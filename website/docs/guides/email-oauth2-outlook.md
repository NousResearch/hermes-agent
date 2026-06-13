---
title: "Email OAuth2 (Outlook/Hotmail)"
description: "Configure Microsoft OAuth2 (XOAUTH2) for Outlook and Hotmail email accounts in Hermes Agent"
---

# Email OAuth2 (Outlook / Hotmail / Microsoft 365)

Microsoft is [deprecating basic authentication](https://learn.microsoft.com/en-us/exchange/clients-and-mobile-in-exchange-online/deprecation-of-basic-authentication-exchange-online)
for IMAP and SMTP. Starting in 2025, Outlook, Hotmail, Live, and Microsoft 365
accounts must use OAuth2 (XOAUTH2) instead of a password.

Hermes Agent supports the **client-credentials flow** — tokens are acquired
automatically using an Azure AD app registration, with automatic refresh.

---

## How It Works

1. **Client-credentials flow** — Hermes authenticates directly to Microsoft's
   OAuth endpoint using a client ID and secret configured in your `.env`.
2. **Token persistence** — The access token is cached to
   `~/.hermes/oauth_tokens.json` with restrictive permissions (mode `0600`).
3. **Auto-refresh** — Tokens are refreshed automatically before expiry
   (120-second safety margin).
4. **XOAUTH2 SASL** — IMAP and SMTP connections authenticate using the
   standard XOAUTH2 SASL mechanism.

No browser interaction is required — authentication is fully automatic after
initial configuration.

---

## Prerequisites

### Step 1: Register an Application in Azure

You need an Azure AD application registration to obtain a **Client ID**.
Microsoft requires every OAuth2 client to have one.

1. Sign in to the [Azure Portal](https://portal.azure.com).
2. Navigate to **Microsoft Entra ID** → **App registrations** → **New registration**.
3. Fill in:
   - **Name:** `Hermes Email Agent` (or any name you'll recognize)
   - **Supported account types:** *Accounts in any organizational directory (Any Microsoft 365 tenant) and personal Microsoft accounts (e.g., Skype, Xbox)*
   - **Redirect URI:** leave blank (client-credentials flow does not need one)
4. Click **Register**.

You'll land on the app's overview page. Copy the **Application (client) ID**
value — this is your `MSGRAPH_CLIENT_ID`.

### Step 2: Create a Client Secret

1. In the left nav, open **Certificates & secrets** → **Client secrets**.
2. Click **New client secret**.
3. Add a description and expiration, then click **Add**.
4. Copy the **Value** immediately — it's only shown once. This is your
   `MSGRAPH_CLIENT_SECRET`.

### Step 3: Configure API Permissions

1. In the left nav, open **API permissions**.
2. Click **Add a permission** → **Microsoft Graph** → **Application permissions**.
3. Search for and add `https://outlook.office.com/.default` (Application permission).
4. Click **Add permissions**.
5. Click **Grant admin consent** — this step is required for client-credentials
   flows (an Entra admin must approve).

---

## Configuration

### Environment Variables

Add to `~/.hermes/.env`:

```bash
# Required — set to "outlook" to enable OAuth2
EMAIL_OAUTH_PROVIDER=outlook

# Required — from your Azure app registration
MSGRAPH_CLIENT_ID=12345678-1234-1234-1234-123456789abc
MSGRAPH_CLIENT_SECRET=your-client-secret-value
MSGRAPH_TENANT_ID=your-tenant-id-or-consumers

# Required — your Outlook/Hotmail/Live email address
EMAIL_ADDRESS=alice@outlook.com

# Optional — custom path for the token cache file
# EMAIL_OAUTH_TOKEN_PATH=~/.hermes/oauth_tokens.json
```

> When `EMAIL_OAUTH_PROVIDER=outlook` is set, `EMAIL_PASSWORD` is **not required**
> — the agent uses XOAUTH2 for IMAP and SMTP authentication instead.

### IMAP / SMTP Hosts

Outlook and Microsoft 365 use these endpoints:

| Setting | Value |
|---------|-------|
| `EMAIL_IMAP_HOST` | `outlook.office365.com` |
| `EMAIL_IMAP_PORT` | `993` |
| `EMAIL_SMTP_HOST` | `smtp.office365.com` |
| `EMAIL_SMTP_PORT` | `587` |

---

## How Tokens Are Stored

The token file at `~/.hermes/oauth_tokens.json` contains:

```json
{
  "email_oauth2": {
    "access_token": "eyJ...",
    "expires_at": 1700000000
  }
}
```

Security properties:

- File is created with **mode `0600`** (owner read/write only)
- Written **atomically** via `.tmp` + `os.replace()` — no partial writes
- `access_token` is **never logged**
- Token is refreshed automatically when less than 120 seconds from expiry

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Authentication failed"** with Outlook | Verify `MSGRAPH_TENANT_ID`, `MSGRAPH_CLIENT_ID`, `MSGRAPH_CLIENT_SECRET` are set correctly. |
| **Access denied** | Check that API permissions include `https://outlook.office.com/.default` as an **Application permission**, and that admin consent has been granted. |
| **Credentials not found** | Ensure all env vars are in `~/.hermes/.env` and the gateway was restarted after changes. |
| **Token expired** | Tokens refresh automatically. If the client secret was rotated, update `MSGRAPH_CLIENT_SECRET` and delete `~/.hermes/oauth_tokens.json`. |

---

## Security

- Use a **dedicated email account** for Hermes — never use your personal inbox
- Store the client secret in `~/.hermes/.env` — protect it with `chmod 600`
- Connections use **SSL/TLS**: IMAP on port 993, SMTP on port 587 (STARTTLS)
- The `MSGRAPH_CLIENT_SECRET` is **never stored** in the token cache
- The `access_token` is **never logged**
