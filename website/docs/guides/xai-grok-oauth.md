---
sidebar_position: 16
title: "xAI Grok OAuth (SuperGrok / X Premium+)"
description: "Sign in with your SuperGrok or X Premium+ subscription to use Grok models in Hermes Agent — no API key required"
---

# xAI Grok OAuth (SuperGrok / X Premium+)

Hermes Agent supports xAI Grok through a browser-based OAuth device-code login flow against [accounts.x.ai](https://accounts.x.ai), using either a **SuperGrok subscription** ([grok.com](https://x.ai/grok)) or an **X Premium+ subscription** (linked X account). No `XAI_API_KEY` is required — log in once and Hermes automatically refreshes your session in the background.

When you sign in with an X account that has Premium+, xAI automatically links the subscription status to your xAI session, so the OAuth flow works the same as it does for direct SuperGrok subscribers.

The transport reuses the `codex_responses` adapter (xAI exposes a Responses-style endpoint), so reasoning, tool-calling, streaming, and prompt caching work without any adapter changes.

The same OAuth bearer token is also reused by every direct-to-xAI surface in Hermes — TTS, image generation, video generation, and transcription — so a single login covers all four.

## Overview

| Item | Value |
|------|-------|
| Provider ID | `xai-oauth` |
| Display name | xAI Grok OAuth (SuperGrok / X Premium+) |
| Auth type | Browser OAuth 2.0 device code |
| Transport | xAI Responses API (`codex_responses`) |
| Default model | `grok-build-0.1` |
| Endpoint | `https://api.x.ai/v1` |
| Auth server | `https://accounts.x.ai` |
| Requires env var | No (`XAI_API_KEY` is **not** used for this provider) |
| Subscription | [SuperGrok](https://x.ai/grok) or [X Premium+](https://x.com/i/premium_sign_up) — see note below |

## Prerequisites

- Python 3.9+
- Hermes Agent installed
- An active **SuperGrok** subscription on your xAI account, **or** an **X Premium+** subscription on the X account you sign in with (xAI links the subscription automatically)
- A browser available anywhere you can open the printed verification URL

:::warning xAI may restrict OAuth API access by tier
xAI's backend enforces its own allowlist on the OAuth API surface and has been seen to reject standard SuperGrok subscribers with `HTTP 403` (see issue [#26847](https://github.com/NousResearch/hermes-agent/issues/26847)) even though the in-app subscription is active. If OAuth login succeeds in the browser but inference returns 403, set `XAI_API_KEY` and switch to the API-key path (`provider: xai`) — that surface is not subject to the same gating today.
:::

## Quick Start

```bash
# Launch the provider and model picker
hermes model
# → Select "xAI Grok OAuth (SuperGrok / X Premium+)" from the provider list
# → Hermes opens or prints an accounts.x.ai verification URL
# → Enter the displayed code if prompted, then approve access in the browser
# → Pick a model (grok-build-0.1 is at the top)
# → Start chatting

hermes
```

After the first login, credentials are stored under `~/.hermes/auth.json` and refreshed automatically before they expire.

## Logging In Manually

You can trigger a login without going through the model picker:

```bash
hermes auth add xai-oauth
```

### Remote / headless sessions

On servers, containers, browser-only consoles (Cloud Shell, Codespaces, EC2 Instance Connect), or SSH sessions where Hermes cannot open a browser locally, Hermes prints the xAI verification URL and user code. Open the URL in any browser on your laptop or in the cloud console, enter the code if prompted, and Hermes will keep polling until xAI approves the login. No SSH tunnel or local callback listener is required.

```bash
hermes auth add xai-oauth --no-browser
# Open the printed verification URL in your browser.
```

The same device-code flow applies when you sign in from the web dashboard or the desktop app: Hermes shows the verification URL and user code, then polls in the background until you approve access.

## How the Login Works

1. Hermes requests a device code from `auth.x.ai`.
2. You open the verification URL, sign in, enter the displayed code if prompted, and approve access.
3. Hermes polls xAI until approval, then saves tokens to `~/.hermes/auth.json`.
4. From then on, Hermes refreshes the access token in the background — you stay signed in until you `hermes auth logout xai-oauth` or revoke access from your xAI account settings.

## Checking Login Status

```bash
hermes doctor
```

The `◆ Auth Providers` section will show the current state of every provider, including `xai-oauth`.

## Switching Models

```bash
hermes model
# → Select "xAI Grok OAuth (SuperGrok / X Premium+)"
# → Pick from the model list (grok-build-0.1 is pinned to the top)
```

Or set the model directly:

```bash
hermes config set model.default grok-build-0.1
hermes config set model.provider xai-oauth
```

## Configuration Reference

After login, `~/.hermes/config.yaml` will contain:

```yaml
model:
  default: grok-build-0.1
  provider: xai-oauth
  base_url: https://api.x.ai/v1
```

### Provider aliases

All of the following resolve to `xai-oauth`:

```bash
hermes --provider xai-oauth        # canonical
hermes --provider grok-oauth       # alias
hermes --provider x-ai-oauth       # alias
hermes --provider xai-grok-oauth   # alias
```

## Direct-to-xAI Tools (TTS / Image / Video / Transcription / X Search)

Once you're logged in via OAuth, every direct-to-xAI tool reuses the same bearer token automatically — there is **no separate setup** unless you'd rather use an API key.

To pick a backend for each tool:

```bash
hermes tools
# → Text-to-Speech       → "xAI TTS"
# → Image Generation     → "xAI Grok Imagine (image)"
# → Video Generation     → "xAI Grok Imagine"
# → X (Twitter) Search   → "xAI Grok OAuth (SuperGrok / X Premium+)"
```

If OAuth tokens are already stored, the picker confirms it and skips the credential prompt. If neither OAuth nor `XAI_API_KEY` is set, the picker offers a 3-choice menu: OAuth login, paste API key, or skip.

:::note Video generation is off by default
The `video_gen` toolset is disabled by default. Enable it in `hermes tools` → `🎬 Video Generation` (press space) before the agent can call `video_generate`. Otherwise the agent may fall back to the bundled ComfyUI skill, which is also tagged for video generation.
:::

:::note X search auto-enables when xAI credentials are present
The `x_search` toolset auto-enables whenever xAI credentials (a SuperGrok / X Premium+ OAuth token or `XAI_API_KEY`) are configured. Disable explicitly via `hermes tools` → `🐦 X (Twitter) Search` (press space) if you don't want this. The tool routes through xAI's built-in `x_search` Responses API — it works with **either** your SuperGrok / X Premium+ OAuth login or a paid `XAI_API_KEY`, and prefers OAuth when both are configured (uses your subscription quota instead of API spend). The tool schema is hidden from the model when no xAI credentials are configured, regardless of whether the toolset is enabled.
:::

### Models

| Tool | Model | Notes |
|------|-------|-------|
| Chat | `grok-build-0.1` | Default; auto-selected when you log in via OAuth |
| Chat | `grok-4.3` | Previous default |
| Chat | `grok-4.20-0309-reasoning` | Reasoning variant |
| Chat | `grok-4.20-0309-non-reasoning` | Non-reasoning variant |
| Chat | `grok-4.20-multi-agent-0309` | Multi-agent variant |
| Image | `grok-imagine-image` | Default; ~5–10 s |
| Image | `grok-imagine-image-quality` | Higher fidelity; ~10–20 s |
| Video | `grok-imagine-video` | Text-to-video |
| Video | `grok-imagine-video-1.5-preview` | Image-to-video; dated alias `grok-imagine-video-1.5-2026-05-30` |
| TTS | (default voice) | xAI `/v1/tts` endpoint |

The chat catalog is derived live from the on-disk `models.dev` cache; new xAI releases appear automatically once that cache refreshes. `grok-build-0.1` is always pinned to the top of the list.

## Environment Variables

| Variable | Effect |
|----------|--------|
| `XAI_BASE_URL` | Override the default `https://api.x.ai/v1` endpoint (rarely needed). |

To select xAI as the active provider, set `model.provider: xai-oauth` in `config.yaml` (use `hermes setup` for the guided flow) or pass `--provider xai-oauth` for a single invocation.

## Troubleshooting

### Token expired — not re-logging in automatically

Hermes refreshes the token before each session and again reactively on a 401. If refresh fails with `invalid_grant` (the refresh token was revoked, or the account was rotated), Hermes surfaces a typed re-auth message instead of crashing.

When the refresh failure is terminal (HTTP 4xx, `invalid_grant`, revoked grant, etc.), Hermes marks the refresh token as dead and quarantines it locally — subsequent calls skip the doomed refresh attempt instead of replaying the same 401 over and over. The agent surfaces a single "re-authentication required" message and stays out of the way until you log in again.

**Fix:** run `hermes auth add xai-oauth` again to start a fresh login. The quarantine clears on the next successful exchange.

### Authorization timed out

Device-code approval has a finite expiry window (xAI sets `expires_in` on the device-code response, typically on the order of tens of minutes). If you do not approve the login in time, Hermes raises a timeout error.

**Fix:** re-run `hermes auth add xai-oauth` (or `hermes model`). The flow starts fresh.

### Logging in from a remote server

On SSH or container sessions Hermes prints the verification URL and user code instead of opening a browser. Open that URL in a browser on your laptop or in a cloud console — no SSH port forward is needed for xAI Grok OAuth.

```bash
hermes auth add xai-oauth --no-browser
```

For loopback-redirect providers (Spotify, MCP servers), see [OAuth over SSH / Remote Hosts](./oauth-over-ssh.md).

### HTTP 403 after a successful login (tier / entitlement)

OAuth completed in the browser, tokens are saved, but inference or token refresh returns `HTTP 403` with a message similar to *"The caller does not have permission to execute the specified operation"*.

This is **not** a stale-token problem — re-running `hermes model` won't change it. xAI's backend has been seen to restrict OAuth API access to specific SuperGrok tiers despite the in-app subscription being active (issue [#26847](https://github.com/NousResearch/hermes-agent/issues/26847)).

**Fix:** set `XAI_API_KEY` and switch to the API-key path:

```bash
export XAI_API_KEY=xai-...
hermes config set model.provider xai
```

Or upgrade your subscription at [x.ai/grok](https://x.ai/grok) if the OAuth route is required.

### "No xAI credentials found" error at runtime

The auth store has no `xai-oauth` entry and no `XAI_API_KEY` is set. You haven't logged in yet, or the credential file was deleted.

**Fix:** run `hermes model` and pick the xAI Grok OAuth provider, or run `hermes auth add xai-oauth`.

## Logging Out

To remove all stored xAI Grok OAuth credentials:

```bash
hermes auth logout xai-oauth
```

This clears both the singleton OAuth entry in `auth.json` and any credential-pool rows for `xai-oauth`. Use `hermes auth remove xai-oauth <index|id|label>` if you only want to drop a single pool entry (run `hermes auth list xai-oauth` to see them).

## See Also

- [OAuth over SSH / Remote Hosts](./oauth-over-ssh.md) — SSH tunnels for loopback-redirect providers (Spotify, MCP); xAI uses device code and does not need a tunnel
- [AI Providers reference](../integrations/providers.md)
- [Environment Variables](../reference/environment-variables.md)
- [Configuration](../user-guide/configuration.md)
- [Voice & TTS](../user-guide/features/tts.md)
