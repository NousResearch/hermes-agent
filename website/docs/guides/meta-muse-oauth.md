---
sidebar_position: 16
title: "Meta Muse Subscription OAuth"
description: "Sign in with your Meta account to use Muse Spark models in Hermes Agent — no API key required"
---

# Meta Muse Subscription OAuth

Hermes Agent supports Meta's Muse models through a browser-based OAuth device-code login flow against [auth.meta.com](https://auth.meta.com). No `MODEL_API_KEY` is required — log in once with your Meta account and Hermes mints a short-lived Model API key for you, refreshing it automatically in the background.

The transport reuses the `codex_responses` adapter (Meta exposes a Responses-style endpoint at `api.meta.ai`), so reasoning, tool-calling, streaming, and prompt caching work without any adapter changes. Prompt caching is automatic: every request to `api.meta.ai` carries `prompt_cache_retention=24h`.

## Overview

| Item | Value |
|------|-------|
| Provider ID | `meta-oauth` |
| Display name | Meta (Muse subscription) |
| Auth type | Browser OAuth 2.0 device code |
| Transport | Meta Responses API (`codex_responses`) |
| Default model | `muse-spark-1.3` |
| Endpoint | `https://api.meta.ai/v1` |
| Auth server | `https://auth.meta.com` |
| Requires env var | No (`MODEL_API_KEY` is optional — see below) |
| Subscription | A Meta account with Muse access |

## Prerequisites

- Python 3.9+
- Hermes Agent installed
- A Meta account that can use Muse
- A browser available anywhere you can open the printed verification URL

## Quick Start

```bash
# Launch the provider and model picker
hermes model
# → Select "Meta (Muse subscription)" from the provider list
# → Hermes opens or prints an auth.meta.com verification URL
# → Enter the displayed code, then approve access in the browser
# → Pick a model (muse-spark-1.3 is at the top)
# → Start chatting

hermes
```

After the first login, credentials are stored under `~/.hermes/auth.json`. The minted Model API key lasts 24 hours; Hermes re-mints it automatically (starting 6 hours before expiry), so you stay signed in until you log out or revoke access.

## Logging In Manually

You can trigger a login without going through the model picker:

```bash
hermes auth add meta-oauth
```

### Remote / headless sessions

On servers, containers, browser-only consoles (Cloud Shell, Codespaces, EC2 Instance Connect), or SSH sessions where Hermes cannot open a browser locally, Hermes prints the Meta verification URL and user code. Open the URL in any browser on your laptop or in the cloud console, enter the code, and Hermes keeps polling until Meta approves the login. No SSH tunnel or local callback listener is required.

```bash
hermes auth add meta-oauth --no-browser
# Open the printed verification URL in your browser.
```

The same device-code flow applies when you sign in from the web dashboard or the desktop app: Hermes shows the verification URL and user code, then polls in the background until you approve access.

## How the Login Works

1. Hermes requests a device code from `auth.meta.com`.
2. You open the verification URL, sign in, enter the displayed code, and approve access.
3. Hermes polls Meta until approval, receiving an identity token.
4. Hermes exchanges the identity token for a 24-hour Model API key (`api.meta.ai/muse-code/key`) and saves both to `~/.hermes/auth.json`.
5. From then on, Hermes re-mints the API key in the background — you stay signed in until you `hermes auth logout meta-oauth` or revoke access from your Meta account settings.

## Checking Login Status

```bash
hermes doctor
```

The `◆ Auth Providers` section will show the current state of every provider, including `meta-oauth`.

## Switching Models

```bash
hermes model
# → Select "Meta (Muse subscription)"
# → Pick from the model list (muse-spark-1.3 is pinned to the top)
```

Or set the model directly:

```bash
hermes config set model.default muse-spark-1.3
hermes config set model.provider meta-oauth
```

## Configuration Reference

After login, `~/.hermes/config.yaml` will contain:

```yaml
model:
  default: muse-spark-1.3
  provider: meta-oauth
  base_url: https://api.meta.ai/v1
```

### Provider aliases

All of the following resolve to `meta-oauth`:

```bash
hermes --provider meta-oauth              # canonical
hermes --provider meta-subscription       # alias
hermes --provider muse-subscription       # alias
hermes --provider muse-code-subscription  # alias
```

The `meta` / `muse` / `muse-spark` shortcuts keep resolving to the API-key provider (`meta-ai`) — the subscription flow never clobbers them.

## Environment Variables

| Variable | Effect |
|----------|--------|
| `MODEL_API_KEY` | Explicit Meta Model API key (Meta's documented variable). Takes precedence over the subscription login when set. |
| `META_API_KEY` | Convenience alias for the same key. |
| `META_MODEL_API_KEY` | Convenience alias for the same key. |
| `META_BASE_URL` | Override the default `https://api.meta.ai/v1` endpoint (rarely needed). |

To select Meta as the active provider, set `model.provider: meta-oauth` in `config.yaml` (use `hermes setup` for the guided flow) or pass `--provider meta-oauth` for a single invocation.

## Troubleshooting

### Key expired — not re-logging in automatically

Hermes re-mints the key before each session (starting 6 hours before expiry) and again reactively on a 401. If re-minting fails because the session was revoked, Hermes quarantines the stored credential locally — subsequent calls skip the doomed re-mint instead of replaying the same failure. The agent surfaces a single "re-authentication required" message and stays out of the way until you log in again.

**Fix:** run `hermes auth add meta-oauth` again to start a fresh login. The quarantine clears on the next successful exchange.

### Authorization timed out

Device-code approval has a finite expiry window (Meta sets `expires_in` on the device-code response, typically on the order of tens of minutes). If you do not approve the login in time, Hermes raises a timeout error.

**Fix:** re-run `hermes auth add meta-oauth` (or `hermes model`). The flow starts fresh.

### Logging in from a remote server

On SSH or container sessions Hermes prints the verification URL and user code instead of opening a browser. Open that URL in a browser on your laptop or in a cloud console — no SSH port forward is needed for Meta Muse OAuth.

```bash
hermes auth add meta-oauth --no-browser
# Open the printed verification URL in your browser.
```

For loopback-redirect providers (Spotify, MCP servers), see [OAuth over SSH / Remote Hosts](./oauth-over-ssh.md).

### "No Meta credentials found" error at runtime

The auth store has no `meta-oauth` entry and no `MODEL_API_KEY`/`META_API_KEY`/`META_MODEL_API_KEY` is set. You haven't logged in yet, or the credential file was deleted.

**Fix:** run `hermes model` and pick the Meta (Muse subscription) provider, or run `hermes auth add meta-oauth`.

## Logging Out

To remove all stored Meta subscription credentials:

```bash
hermes auth logout meta-oauth
```

This clears both the singleton OAuth entry in `auth.json` and any credential-pool rows for `meta-oauth`. Use `hermes auth remove meta-oauth <index|id|label>` if you only want to drop a single pool entry (run `hermes auth list meta-oauth` to see them).

## See Also

- [OAuth over SSH / Remote Hosts](./oauth-over-ssh.md) — SSH tunnels for loopback-redirect providers (Spotify, MCP); Meta uses device code and does not need a tunnel
- [AI Providers reference](../integrations/providers.md)
- [Environment Variables](../reference/environment-variables.md)
- [Configuration](../user-guide/configuration.md)
