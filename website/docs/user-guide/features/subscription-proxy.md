---
sidebar_position: 15
title: "Subscription Proxy"
description: "Use your xAI OAuth login as an OpenAI-compatible endpoint for external apps"
---

# Subscription Proxy

The subscription proxy is a local HTTP server that lets external apps —
OpenViking, Karakeep, Open WebUI, anything that speaks OpenAI-compatible
chat completions — use your Rabbit-managed OAuth provider login as their
LLM endpoint. The proxy attaches the right credentials (refreshing them
automatically) so the app never needs a static API key.

This is different from the [API server](./api-server.md):

| | API server | Subscription proxy |
|---|---|---|
| What it serves | Your agent (full toolset, memory, skills) | Raw model inference |
| Use case | "Use Rabbit as a chat backend" | "Use my xAI login from another app" |
| Auth | Your `API_SERVER_KEY` | Any bearer (proxy attaches the real one) |
| Tool calls | Yes — the agent runs tools | No — passthrough only |

Use the API server when you want the **agent** as a backend. Use the
proxy when you just want **the model** through your OAuth login.

## Quick Start

### 1. Log into your provider (one-time)

```bash
rabbit auth add xai-oauth --type oauth
```

This runs the xAI OAuth flow. Rabbit stores the credentials in
`~/.rabbit/auth.json` — the same place all Rabbit provider logins live.

### 2. Start the proxy

```bash
rabbit proxy start
```

```
Starting Rabbit proxy for xAI Grok OAuth
  Listening on:  http://127.0.0.1:8645/v1
  Forwarding to: (resolved per-request from your OAuth credential)
  Use any bearer token in the client — the proxy attaches your real credential.
```

Leave this running in the foreground. Use `tmux`, `nohup`, or a systemd
unit if you want it to survive logout.

### 3. Point your app at it

Any OpenAI-compatible app config takes the same triple:

```
Base URL:   http://127.0.0.1:8645/v1
API key:    anything (e.g. "sk-unused")
Model:      grok-4    # or any model your xAI account serves
```

The proxy ignores the `Authorization` header from your app and attaches
your real xAI credential to the upstream request. Refreshes happen
automatically when the bearer approaches expiry.

## Available providers

```bash
rabbit proxy providers
```

Currently shipped: `xai` (xAI / Grok OAuth). More OAuth providers can be
added by implementing the `UpstreamAdapter` interface in
`rabbit_cli/proxy/adapters/`.

## Check status

```bash
rabbit proxy status
```

```
Rabbit proxy upstream adapters

  [xai     ] xAI Grok OAuth — ready
```

If you see `not logged in`, run `rabbit auth add xai-oauth --type oauth`.
If you see `credentials need attention`, your credential was revoked or
expired — re-run the same command.

## Allowed paths

The proxy only forwards paths the upstream actually serves. For xAI:

| Path | Purpose |
|------|---------|
| `/v1/chat/completions` | Chat completions (streaming + non-streaming) |
| `/v1/responses` | Responses API |
| `/v1/completions` | Legacy text completions |
| `/v1/embeddings` | Embeddings |
| `/v1/models` | Model list |

Other paths (`/v1/images/generations`, `/v1/audio/speech`, etc.) return
404 with a clear error pointing at the allowed paths. This keeps stray
clients from leaking weird requests to the upstream.

## Configuring OpenViking to use the proxy

[OpenViking](https://github.com/volcengine/OpenViking) is a context
database that needs an LLM provider for its VLM (vision/language model
used to extract memories) and embedding model. With the proxy, you can
point its `vlm.api_base` at your local proxy:

Edit `~/.openviking/ov.conf`:

```json
{
  "vlm": {
    "provider": "openai",
    "model": "grok-4",
    "api_base": "http://127.0.0.1:8645/v1",
    "api_key": "unused-proxy-attaches-real-creds"
  }
}
```

Then start your proxy in a terminal alongside `openviking-server`:

```bash
# Terminal 1
rabbit proxy start

# Terminal 2
openviking-server
```

OpenViking's VLM calls now flow through your xAI login. The embedding
model side still needs its own provider — the proxy does serve
`/v1/embeddings`, but the model selection depends on what your xAI
account supports.

## Configuring Karakeep (or any bookmark/summarizer app)

[Karakeep](https://karakeep.app/) takes an OpenAI-compatible API for
bookmark summarization. In its config:

```bash
# Karakeep .env
OPENAI_API_BASE_URL=http://127.0.0.1:8645/v1
OPENAI_API_KEY=any-non-empty-string
INFERENCE_TEXT_MODEL=grok-4
```

Same pattern works for Open WebUI, LobeChat, NextChat, or any other
OpenAI-compatible client.

## Exposing on LAN

By default the proxy binds `127.0.0.1` (localhost only). It refuses requests
whose `Host` header is not its own address (`localhost`, `127.0.0.1`, `[::1]`
or the bound IP) and any browser request from another site (an `Origin` other
than its own, or a `Sec-Fetch-Site` of `cross-site`/`same-site`), so a web page
open in your browser cannot use it. Clients such as SDKs and `curl` send
neither header and are unaffected. To let other machines on your
network use it:

```bash
rabbit proxy start --host 0.0.0.0 --port 8645
```

⚠ **Be aware:** anyone on your network can now use your xAI
login. A wildcard bind skips the `Host` check (any name may reach it),
so it refuses every request a web page makes (any `Origin`, or a
`Sec-Fetch-Site` other than `none`): browsers cannot use a wildcard-bound
proxy. The proxy has no auth of its own — it accepts any bearer.
Use a firewall, VPN, or reverse proxy with proper auth if you expose
this beyond your trusted network.

## Rate limits

Your xAI account's rate limits apply across the whole proxy. The
proxy doesn't fan out or pool — it's a single credential with your full
account quota.

## Architecture

The proxy is intentionally minimal. Per request:

1. Receive `POST /v1/chat/completions` from your app
2. Look up the adapter's current credential (refresh if expiring)
3. Forward the request body verbatim, with `Authorization: Bearer <credential>`
4. Stream the response back unchanged (SSE preserved)

No transformation. No logging of request bodies. No agent loop. The
proxy is a credential-attaching pass-through.

## Future: more OAuth providers

The adapter system is pluggable. Adding a new provider (e.g.
HuggingFace, GitHub Copilot's chat endpoint, Anthropic via OAuth)
requires implementing `UpstreamAdapter` in
`rabbit_cli/proxy/adapters/<provider>.py` and registering it in
`adapters/__init__.py`. Providers that aren't OpenAI-compatible at the
protocol level (Anthropic Messages API, for example) would need a
transformation layer, which is out of scope for the current shape.
