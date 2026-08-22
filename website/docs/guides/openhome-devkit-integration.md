---
sidebar_position: 19
title: "Connect Hermes to an OpenHome DevKit"
description: "Expose your Hermes agent to an OpenHome ability so voice commands on the DevKit can talk directly to your own agent"
---

# Connect Hermes to an OpenHome DevKit

[OpenHome](https://openhome.com) DevKit abilities are Python capabilities that run in OpenHome's cloud and make outbound HTTP calls — there's no special OpenHome-side API required to reach Hermes. This guide covers the Hermes side: exposing the built-in [API Server](/user-guide/features/api-server) so an OpenHome ability can call it directly, with no local bridge process running on your machine at request time.

## What you're building

An OpenHome ability (`hermes-connector`, published in the [OpenHome community abilities repo](https://github.com/openhome-dev/abilities/tree/dev/community/hermes-connector)) that:

- Fires on a voice trigger like "talk to Hermes" or "ask my agent"
- POSTs the user's question to your Hermes agent's `/v1/chat/completions` endpoint
- Speaks Hermes's response back through the DevKit

This is the same OpenAI-compatible endpoint used by [Open WebUI](/user-guide/messaging/open-webui) and other frontends — OpenHome is just another caller.

## 1. Enable the API server

Add to `~/.hermes/.env`:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=<generate a long random token — this gates full tool access>
```

Restart the gateway:

```bash
hermes gateway restart
```

## 2. Restrict the toolset (recommended)

By default, the `api_server` platform gets Hermes's full toolset — including `terminal`. Since this endpoint will be reachable from OpenHome's cloud, scope it down in `~/.hermes/config.yaml` to whatever the ability actually needs:

```yaml
platform_toolsets:
  api_server:
    - web
    - memory
    - todo
```

Adjust the list to your own risk tolerance — the point is a leaked token shouldn't be equivalent to shell access on your machine. See [Security](/user-guide/security) for the full toolset list.

## 3. Give it a public URL

`API_SERVER_HOST` defaults to `127.0.0.1` — nothing outside your machine can reach it. OpenHome's ability runs in their cloud, so it needs a real public URL in front of your API server. Two options that don't require opening a port on your router:

**Tailscale Funnel** (stable URL, tied to your machine):

```bash
tailscale funnel --bg 8642
```

**Cloudflare quick tunnel** (no account, but rate-limited and the URL changes every run):

```bash
cloudflared tunnel --url http://localhost:8642
```

Either way, put TLS in front of the API server before exposing it — don't forward the raw port.

## 4. Configure the ability

Install `hermes-connector` from the [OpenHome abilities marketplace](https://github.com/openhome-dev/abilities/tree/dev/community/hermes-connector), then set its two secrets in the OpenHome dashboard (Abilities → Hermes Connector → API Keys):

- `hermes_api_url` — the public URL from step 3
- `hermes_api_key` — the `API_SERVER_KEY` from step 1

Full setup instructions (written for non-native English speakers) are in the ability's own README.

## Verifying it works

Test the endpoint directly before testing through the DevKit:

```bash
curl https://your-tunnel-url/v1/chat/completions \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "hermes-agent", "messages": [{"role": "user", "content": "reply with exactly: pong"}]}'
```

If that round-trips, saying "talk to Hermes" (or whichever trigger word you configured) on the DevKit should reach the same endpoint and speak back a real response.

## Limitations

- The API server is stateless request/response (see [API Server limitations](/user-guide/features/api-server#limitations)) — it can't push a background job's result to the DevKit proactively, only respond to a request the ability initiates.
- Each OpenHome user needs their own reachable Hermes instance and their own `API_SERVER_KEY` — there's no shared/hosted Hermes backing this integration.
