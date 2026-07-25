---
sidebar_position: 18
title: "Desktop Native Sign-In (RFC 8252)"
description: "How the Hermes Desktop app signs in to a gated gateway using your system browser and PKCE — no embedded webview, no session cookies"
---

# Desktop Native Sign-In (RFC 8252)

When the Hermes Desktop app connects to a **gated gateway** (a hosted or
self-hosted dashboard that sits behind an OAuth provider), it can sign in two
ways:

1. **Native sign-in (RFC 8252)** — the app opens your **real system browser**,
   you approve in the browser you already trust, and the app receives tokens it
   stores in your OS keychain. **No embedded webview, no browser session
   cookies.** This is the default whenever the gateway supports it.
2. **Embedded sign-in (legacy fallback)** — the app opens a small in-app
   browser window and captures the gateway's session cookie. Used automatically
   when the gateway is an older build that doesn't advertise native sign-in.

You don't choose between these — the app detects what the gateway supports and
picks the best one. This page explains what happens and why.

## Why native sign-in

Embedding a browser inside a native app for OAuth has well-known downsides:
the login page can't see your existing browser session (so you re-type
credentials and re-do MFA), password managers and passkeys often don't work,
and the app relies on reading a session cookie out of a private webview. RFC
8252 ("OAuth 2.0 for Native Apps") is the industry best practice that avoids
all of that: **do the authorization in the system browser and hand the app its
own tokens.**

For Hermes specifically, native sign-in means:

- **No embedded webview.** The authorization happens in Safari / Chrome /
  Firefox / Edge — whatever you use — with your logins, extensions, and
  passkeys intact.
- **No session cookies.** The app holds an OAuth **access token** (short-lived)
  and **refresh token**, encrypted at rest via your OS keychain (Electron
  `safeStorage`). REST calls and WebSocket tickets are authenticated with an
  `Authorization: Bearer` header, not a cookie jar.

## How it works

```
Desktop app                Gateway (/auth/native/*)          Nous Portal (IDP)
   │ 1. open loopback 127.0.0.1:<random port>
   │ 2. system browser ─►  /auth/native/authorize
   │    (PKCE challenge)    (starts the normal PKCE login) ─► /oauth/authorize
   │                        ◄──── code ──── /auth/callback ◄──┘
   │                        3. mint one-time gateway code
   │ ◄─ 302 127.0.0.1/cb?code=… ─┘
   │ 4. POST /auth/native/token (code + PKCE verifier)
   │ ◄─ 5. { access_token, refresh_token, expires_at } ───────┘
   │ 6. store in OS keychain; use Bearer for REST + WS tickets
```

The gateway **brokers** the flow: it is the authorization server *to the
desktop app* and an OAuth client *to the upstream identity provider* (Nous
Portal). This is required because the upstream `client_id` and permitted
redirect URIs are bound to the gateway's own origin — a desktop app can't be a
direct client of the Portal. The desktop still gets the full RFC 8252
experience: its own PKCE pair, its own loopback redirect, and tokens it owns.

**PKCE (RFC 7636)** protects the loopback hop: the one-time gateway code is
useless without the code verifier, which never leaves the app. The code is
single-use and short-lived.

## Capability detection & fallback

The desktop reads the gateway's public `/api/status` endpoint, which advertises
an `auth_flows` array:

| `auth_flows` value | Meaning |
|--------------------|---------|
| `["cookie", "native_pkce"]` | Gateway supports native sign-in → the app uses it |
| `["cookie"]` | Gateway supports only the legacy flow → the app uses the embedded webview |
| *(field absent)* | Older gateway → the app uses the embedded webview |

If native sign-in is advertised but fails for a local reason — e.g. a security
tool blocks the loopback listener, or you close the browser tab — the app
**falls back to the embedded flow automatically** so you can still sign in.

## Token lifecycle

- **Access token**: short-lived (minutes). Sent as `Authorization: Bearer` on
  every REST call and when minting a WebSocket ticket.
- **Refresh token**: longer-lived, rotating. When the access token is near
  expiry the app calls `/auth/native/refresh` to rotate both tokens, then
  updates the keychain.
- **Terminal expiry**: if the refresh token is dead (expired / revoked /
  reuse-detected), the app clears its stored tokens and prompts a fresh
  sign-in.
- **Sign out**: clears both the native tokens (keychain) and any legacy session
  cookie for that gateway.

## For gateway operators

Native sign-in is available automatically on any gated gateway that has a
brokerable OAuth provider registered (e.g. the bundled **Nous** provider). No
configuration is required — the `/auth/native/*` routes and the `auth_flows`
advertisement are part of the dashboard-auth subsystem. Password-only and
token-only providers do not advertise `native_pkce` (there is no upstream
redirect to broker), and those deployments continue to use their existing
login.

The relevant endpoints (all public, pre-auth bootstrap, same as the existing
`/auth/*` OAuth routes):

- `GET /auth/native/authorize` — starts the brokered PKCE login
- `POST /auth/native/token` — exchanges the loopback code + verifier for tokens
- `POST /auth/native/refresh` — rotates tokens from the app's refresh token

## See also

- [OAuth over SSH / Remote Hosts](./oauth-over-ssh.md) — the loopback-callback
  pattern for provider/MCP OAuth on remote machines.
- [Run Hermes with Nous Portal](./run-hermes-with-nous-portal.md)
