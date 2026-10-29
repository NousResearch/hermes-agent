---
sidebar_position: 10
title: "Run Hermes Through a Custom OpenAI-Compatible Router"
description: "Point a Hermes profile at a local router or gateway that speaks the OpenAI API — the profile shape, the streaming caveat that breaks naive setups, and how to verify the wiring end to end"
---

# Run Hermes Through a Custom OpenAI-Compatible Router

A *router* is a small server — on your own machine or your LAN — that accepts
OpenAI-compatible requests and forwards each one to a backend of its choosing:
a local model server, a pool of API providers, or a tier system of its own
("fast", "coding", "frontier"). LiteLLM is the best-known example; plenty of
people also run a hand-rolled gateway, and this guide's worked example is
[Omnirouter](https://github.com/Darren-dev141/omnirouter), a local router
developed alongside Hermes.

Hermes talks to a router through the built-in `custom` provider — no plugin
or code change needed. The setup is one profile file. The one non-obvious
part is streaming, covered below; skip it and a router that works fine from
`curl` can still fail every Hermes turn.

## The profile

Save this as `~/.hermes/profiles/myrouter/config.yaml` and run Hermes with
`hermes -p myrouter`:

```yaml
model:
  provider: custom
  base_url: http://127.0.0.1:8799/v1
  default: coding
  api_key: local-router
  streaming: false
```

Line by line:

- **`provider: custom`** — the generic OpenAI-compatible provider described
  in [Providers](../integrations/providers.md#custom--self-hosted-llm-providers).
- **`base_url`** — the router's OpenAI API root, including the `/v1` suffix.
  Use the host and port your router actually binds; `127.0.0.1` when it runs
  on the same machine as Hermes.
- **`default`** — the model name *the router* serves. With a router this is
  often a route or tier name (`coding` above), not an upstream model ID:
  Hermes asks the router for that name and the router decides what serves it.
  A `404 model not found` almost always means this name isn't one your
  router serves.
- **`api_key`** — whatever the router expects, if anything. If your router
  doesn't check keys, still put a non-empty placeholder: Hermes sends a key
  with custom-provider requests, and a missing one can surface as a `401`
  that has nothing to do with your router's own auth.
- **`streaming: false`** — see the next section. Whether you need it depends
  on how your router handles streamed requests.

Profiles are covered in depth in [Profiles](../user-guide/profiles.md); the
same block works as the `model:` section of a plain `~/.hermes/config.yaml`
if you want the router as your default rather than a named profile.

## The streaming caveat

Hermes streams by default. When it sends a request it sets `stream: true`
and expects the response to arrive as a server-sent-events (SSE) stream.

Some routers ignore the `stream` flag and answer with **one plain JSON
completion body** instead. The router served the call correctly — its own
logs show a success — but the client receives no stream chunks at all, so
the turn fails: on the main chat path Hermes reports

```
EmptyStreamError: Provider returned an empty stream with no finish_reason
(possible upstream error or malformed SSE response)
```

and auxiliary calls come back empty. Tellingly, the same endpoint answers a
non-streaming request perfectly.

If that is your router's behaviour, `streaming: false` in the profile is
the fix: Hermes then makes plain requests the router already handles, and
every call through that profile works. Routers that implement SSE properly
don't need the setting — but leaving it off is harmless for a router that
only ever returns complete bodies, so when in doubt, set it and remove it
later if you want streaming back.

This pairing — `custom` provider plus `streaming: false` against a router
that returns plain JSON — is field-verified: it is exactly the wiring the
Omnirouter project ships for Hermes after a test battery caught the
empty-stream failure against a live router.

## Verify the wiring

Three checks, in order:

1. **The router answers.** Ask it what it serves:

   ```bash
   curl http://127.0.0.1:8799/v1/models
   ```

   (or hit the router's own health endpoint if it has one). A connection
   refusal means the router isn't running or isn't bound where `base_url`
   points — fix that before touching Hermes config.

2. **One real completion through the profile.**

   ```bash
   hermes -p myrouter chat -q "Reply with exactly: OK"
   ```

   An `OK` reply proves the whole path: profile → router → backend. If
   `curl` works but this fails with the empty-stream error above, you're
   looking at the streaming mismatch — confirm `streaming: false` is in the
   profile that `hermes -p` is actually loading.

3. **The router saw the request you think it saw.** If your router keeps
   request logs or receipts, check that the call arrived and that the model
   name it served matches the `default` in your profile. That pair —
   requested name and served model — is what tells you the router routed
   the call where you expected, rather than silently falling back to some
   default of its own.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `EmptyStreamError` / empty responses, router logs show success | Router ignores `stream: true` → set `streaming: false` in the profile |
| `404 model not found` | `default:` names a model or tier the router doesn't serve |
| `401` from a router that has no auth | `api_key` missing — add a non-empty placeholder |
| Connection refused | Router not running, wrong port, or `base_url` missing the `/v1` suffix |
| Works in `curl`, fails in Hermes | Compare carefully: Hermes sends `stream: true` by default; your `curl` test probably didn't |
