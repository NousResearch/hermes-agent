---
sidebar_position: 15
title: "MiniMax OAuth"
description: "Log into MiniMax via browser OAuth and use MiniMax-M2.7 models in Hermes Agent — no API key required"
---

# MiniMax OAuth

Hermes Agent supports **MiniMax** through a browser-based OAuth login flow, using the same credentials as the [MiniMax portal](https://www.minimax.io). No API key or credit card is required — log in once and Hermes automatically refreshes your session.

The transport reuses the `anthropic_messages` adapter (MiniMax exposes an Anthropic Messages-compatible endpoint at `/anthropic`), so all existing tool-calling, streaming, and context features work without any adapter changes.

## Overview

| Item | Value |
|------|-------|
| Provider ID | `minimax-oauth` |
| Display name | MiniMax (OAuth) |
| Auth type | Browser OAuth (PKCE redirect flow) |
| Transport | Anthropic Messages-compatible (`anthropic_messages`) |
| Models | `MiniMax-M2.7`, `MiniMax-M2.7-highspeed` |
| Global endpoint | `https://api.minimax.io/anthropic` |
| China endpoint | `https://api.minimaxi.com/anthropic` |
| Requires env var | No (`MINIMAX_API_KEY` is **not** used for this provider) |

## Prerequisites

- Python 3.9+
- Hermes Agent installed
- A MiniMax account at [minimax.io](https://www.minimax.io) (global) or [minimaxi.com](https://www.minimaxi.com) (China)
- A browser available on the local machine (or use `--no-browser` for remote sessions)

## Quick Start

```bash
# Launch the provider and model picker
hermes model
# → Select "MiniMax (OAuth)" from the provider list
# → Hermes opens your browser to the MiniMax authorization page
# → Approve access in the browser
# → Select a model (MiniMax-M2.7 or MiniMax-M2.7-highspeed)
# → Start chatting

hermes
```

After the first login, credentials are stored under `~/.hermes/auth.json` and are refreshed automatically before each session.

## Logging In Manually

You can trigger a login without going through the model picker:

```bash
hermes auth add minimax-oauth
```

### China region

If your account is on the China platform (`minimaxi.com`), use the API-key-based `minimax-cn` provider instead — `minimax-cn` is registered with `auth_type="api_key"` only (no OAuth flow). Configure `MINIMAX_CN_API_KEY` (and optionally `MINIMAX_CN_BASE_URL`) directly:

```bash
echo 'MINIMAX_CN_API_KEY=your-key' >> ~/.hermes/.env
```

### Remote / headless sessions

On servers or containers where no browser is available:

```bash
hermes auth add minimax-oauth --no-browser
```

Hermes will print the verification URL and user code — open the URL on any device and enter the code when prompted.

## The OAuth Flow

Hermes implements a PKCE browser OAuth flow against the MiniMax OAuth endpoints:

1. Hermes generates a PKCE verifier / challenge pair and a random state value.
2. It POSTs to `{base_url}/oauth/code` with the challenge and receives a `user_code` and `verification_uri`.
3. Your browser opens `verification_uri`. If prompted, enter the `user_code`.
4. Hermes polls `{base_url}/oauth/token` until the token arrives (or the deadline passes).
5. Tokens (`access_token`, `refresh_token`, expiry) are saved to `~/.hermes/auth.json` under the `minimax-oauth` key.

Token refresh (standard OAuth `refresh_token` grant) runs automatically at each session start when the access token is within 60 seconds of expiry.

## Checking Login Status

```bash
hermes doctor
```

The `◆ Auth Providers` section will show:

```
✓ MiniMax OAuth  (logged in, region=global)
```

or, if not logged in:

```
⚠ MiniMax OAuth  (not logged in)
```

## Switching Models

```bash
hermes model
# → Select "MiniMax (OAuth)"
# → Pick from the model list
```

Or set the model directly:

```bash
hermes config set model.default MiniMax-M2.7
hermes config set model.provider minimax-oauth
```

## Configuration Reference

After login, `~/.hermes/config.yaml` will contain entries similar to:

```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```

### Region endpoints

| Provider id | Portal | Inference endpoint |
|-------------|--------|-------------------|
| `minimax-oauth` (global) | `https://api.minimax.io` | `https://api.minimax.io/anthropic` |
| `minimax-cn` (China) | `https://api.minimaxi.com` | `https://api.minimaxi.com/anthropic` |

### Provider aliases

All of the following resolve to `minimax-oauth`:

```bash
hermes --provider minimax-oauth    # canonical
hermes --provider minimax-portal   # alias
hermes --provider minimax-global   # alias
hermes --provider minimax_oauth    # alias (underscore form)
```

## Environment Variables

The `minimax-oauth` provider does **not** use `MINIMAX_API_KEY` or `MINIMAX_BASE_URL`. Those variables are for the API-key-based `minimax` and `minimax-cn` providers only.

| Variable | Effect |
|----------|--------|
| `MINIMAX_API_KEY` | Used by `minimax` provider only — ignored for `minimax-oauth` |
| `MINIMAX_CN_API_KEY` | Used by `minimax-cn` provider only — ignored for `minimax-oauth` |

To use `minimax-oauth` as the active provider, set `model.provider: minimax-oauth` in `config.yaml` (use `hermes setup` for the guided flow), or pass `--provider minimax-oauth` for a single invocation:

```bash
hermes --provider minimax-oauth
```

## Models

| Model | Best for |
|-------|----------|
| `MiniMax-M2.7` | Long-context reasoning, complex tool-calling |
| `MiniMax-M2.7-highspeed` | Lower latency, lighter tasks, auxiliary calls |

Both models support up to 200,000 tokens of context.

`MiniMax-M2.7-highspeed` is also used automatically as the auxiliary model for vision and delegation tasks when `minimax-oauth` is the primary provider.

## Troubleshooting

### Token expired — not re-logging in automatically

Hermes refreshes the token on every session start if it is within 60 seconds of expiry. If the access token is already expired (for example, after a long offline period), the refresh happens automatically on the next request. If refresh fails with `refresh_token_reused` or `invalid_grant`, Hermes marks the session as requiring re-login.

When the refresh failure is terminal (HTTP 4xx, `invalid_grant`, revoked grant, etc.), Hermes marks the refresh token as dead and quarantines it locally so it doesn't keep replaying the doomed exchange. The agent surfaces a single "re-authentication required" message and stays out of the way until you log in again.

**Fix:** run `hermes auth add minimax-oauth` again to start a fresh login. The quarantine clears on the next successful exchange.

### Authorization timed out

The device-code flow has a finite expiry window. If you don't approve the login in time, Hermes raises a timeout error.

**Fix:** re-run `hermes auth add minimax-oauth` (or `hermes model`). The flow starts fresh.

### State mismatch (possible CSRF)

Hermes detected that the `state` value returned by the authorization server does not match what it sent.

**Fix:** re-run the login. If it persists, check for a proxy or redirect that is modifying the OAuth response.

### Logging in from a remote server

If `hermes` cannot open a browser window, use `--no-browser`:

```bash
hermes auth add minimax-oauth --no-browser
```

Hermes prints the URL and code. Open the URL on any device and complete the flow there.

### "Not logged into MiniMax OAuth" error at runtime

The auth store has no credentials for `minimax-oauth`. You have not logged in yet, or the credential file was deleted.

**Fix:** run `hermes model` and select MiniMax (OAuth), or run `hermes auth add minimax-oauth`.

## Logging Out

To remove stored MiniMax OAuth credentials:

```bash
hermes auth remove minimax-oauth
```

## See Also

- [AI Providers reference](../integrations/providers.md)
- [Environment Variables](../reference/environment-variables.md)
- [Configuration](../user-guide/configuration.md)
- [hermes doctor](../reference/cli-commands.md)
