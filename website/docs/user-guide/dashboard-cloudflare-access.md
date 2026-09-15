---
sidebar_position: 9
title: "Dashboard Authentication Behind a Cloudflare Tunnel"
description: "Use Cloudflare Access as the dashboard's OIDC identity provider for SSO login without a separate dashboard password"
---

# Dashboard Authentication Behind a Cloudflare Tunnel

This guide shows how to run the Hermes dashboard behind a Cloudflare tunnel and use **Cloudflare Access as the dashboard's identity provider**, so remote users (including the Hermes Desktop app) get a real single-sign-on flow — your existing IdP (Azure AD, Google, GitHub, Okta, email OTP, …) — with no separate dashboard password.

## The problem this solves

The dashboard's auth gate fail-closes on any non-loopback exposure: bind `127.0.0.1` behind a tunnel and stock Hermes serves the dashboard unauthenticated from the tunnel's perspective, while exposing it through any other ingress path on the host. Common workarounds have real downsides:

- **Basic auth** (`dashboard.basic_auth`): works, but it's a second password unrelated to your real identity provider.
- **Trusting "it came through the tunnel"**: unsafe. Access authenticates at Cloudflare's edge, not at your origin. By the time a request reaches Hermes, it is ordinary traffic from localhost unless something validates the identity. Any second ingress path (another tunnel, a reverse proxy, SSRF from a container on the same host) bypasses the edge entirely.

The clean answer is to make the origin authenticate too. Cloudflare Access can act as a **standard OpenID Connect identity provider** via a Zero Trust SaaS application, and Hermes ships a generic OIDC dashboard-auth provider (`dashboard.oauth.self_hosted`). No custom code is required — the two compose directly.

The result: Cloudflare Access remains the outer gate (network-level), and the OIDC provider is the inner gate (identity-level). A request must pass both.

## Prerequisites

- A Cloudflare zone on your account with Zero Trust enabled (free tier works)
- A running cloudflared tunnel routing your dashboard hostname (e.g. `hermes.example.com`) to the dashboard's loopback bind
- The `dashboard.public_url` set to the public hostname — this engages the auth gate even on a loopback bind

## Step 1 — Bind the dashboard to loopback

Keep the dashboard itself on `127.0.0.1`. The tunnel is the only ingress path.

```yaml
# ~/.hermes/config.yaml
dashboard:
  public_url: "https://hermes.example.com"
```

## Step 2 — Create a Cloudflare Access SaaS application

In the Cloudflare Zero Trust dashboard: **Access → Applications → Add → SaaS**, or via the API:

```bash
curl -X POST -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.cloudflare.com/client/v4/accounts/$ACCOUNT_ID/access/apps" \
  -d '{
    "name": "hermes-dashboard",
    "domain": "hermes.example.com",
    "type": "saas",
    "session_duration": "24h",
    "saas_app": {
      "auth_type": "oidc",
      "redirect_uris": [
        "https://hermes.example.com/auth/callback"
      ],
      "grant_types": ["authorization_code", "authorization_code_with_pkce", "refresh_tokens"],
      "scopes": ["openid", "profile", "email", "groups"],
      "token_lifetime": "24h",
      "refresh_token_options": {
        "rotation_type": "rotate",
        "lifetime": "24h",
        "expiration_inactivity_timeout": "24h"
      }
    }
  }'
```

Notes:

- The `redirect_uri` must be the **public** dashboard hostname plus `/auth/callback` — the dashboard's OIDC callback route.
- The response contains the SaaS app's `client_id` and `client_secret`. Save both; you'll need them in Step 4.
- If you use the Hermes Desktop app (or any native client), also add `http://127.0.0.1/callback` to `redirect_uris` — the desktop completes its native RFC 8252 flow through the dashboard's broker, which exchanges the code server-side.

### Gotchas

- **`allowed_idps` requires a full `PUT`.** `PATCH`ing the app with an `allowed_idps` change silently fails with "Method not allowed for this authentication scheme". Use `PUT` with the complete application body instead.
- **Refresh-token options are mandatory** when `refresh_tokens` is in `grant_types`; the API rejects the request without `refresh_token_options`.
- **Bot Fight Mode can break the token exchange.** If the server-side code-for-token POST gets a `1010` error, exempt the SSO host from Bot Fight Mode.

## Step 3 — Attach an Access policy and identity providers

Attach a policy to the SaaS app allowing your users (e.g. `email_domain: example.com`), and enable the identity providers you want on the login page (`allowed_idps` — again via full `PUT` if using the API). Users authenticate with your IdP at `https://<your-team>.cloudflareaccess.com`, then Cloudflare mints the OIDC tokens.

## Step 4 — Point Hermes at the Access OIDC issuer

Each SaaS app gets a client-scoped issuer URL:

```
https://<team>.cloudflareaccess.com/cdn-cgi/access/sso/oidc/<client_id>
```

Configure the `self-hosted` dashboard-auth provider via environment variables in `~/.hermes/.env` (secrets belong there, not in `config.yaml`):

```bash
# ~/.hermes/.env
HERMES_DASHBOARD_OIDC_ISSUER=https://<team>.cloudflareaccess.com/cdn-cgi/access/sso/oidc/<client_id>
HERMES_DASHBOARD_OIDC_CLIENT_ID=<client_id>
HERMES_DASHBOARD_OIDC_CLIENT_SECRET=<client_secret>
HERMES_DASHBOARD_OIDC_SCOPES="openid profile email groups"
```

or equivalently via `config.yaml`:

```yaml
dashboard:
  oauth:
    self_hosted:
      issuer: "https://<team>.cloudflareaccess.com/cdn-cgi/access/sso/oidc/<client_id>"
      client_id: "<client_id>"
      client_secret: "<client_secret>"
      scopes: "openid profile email groups"
```

Restart the dashboard. On startup you should see the provider register:

```
dashboard-auth-self-hosted: registered provider (issuer=..., client_id=..., scopes='openid profile email groups', confidential=True)
Dashboard binding to 127.0.0.1 with auth gate enabled. Providers: self-hosted
```

## Step 5 — Verify

1. Open `https://hermes.example.com` in a browser. You should be bounced to Cloudflare Access (edge), sign in with your IdP, then land on the dashboard signed in (origin OIDC session).
2. For the desktop app: **Settings → Gateways → Sign in** opens the system browser, runs the same OIDC round trip, and returns to the app. The connection test should pass the WebSocket-ticket mint (this is the step that fails when only the edge gate is configured and the origin has no session mechanism — the desktop cannot mint WebSocket tickets without a real origin session).
3. Unauthenticated requests to the public URL must redirect to Cloudflare Access; direct requests to the origin port from off-host must be unreachable (loopback bind).

## How the pieces fit

```
Browser/Desktop ──► Cloudflare Access (edge: network + IdP authentication)
                          │ tunnel
                          ▼
                 cloudflared ──► 127.0.0.1 dashboard
                                        │
                                 dashboard auth gate
                                 (self-hosted OIDC provider
                                  verifies Cloudflare-issued
                                  ID tokens, mints origin session,
                                  WebSocket tickets, refresh)
```

Cloudflare Access alone is a network gate; the OIDC provider is the identity gate at the origin. Running both is what makes the deployment correct: the tunnel cannot be bypassed into an unauthenticated dashboard, and no dashboard password needs to exist at all.

If you also want a local escape hatch (e.g. for cron/automation that can't do an OIDC dance), the `basic` password provider can be enabled alongside `self-hosted` — Hermes auto-selects the interactive (non-password) provider for native desktop sign-ins when both are configured.
