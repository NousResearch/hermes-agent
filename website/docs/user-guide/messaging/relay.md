# Relay (Team Gateway) — Enterprise

> **Enterprise-only.** The relay lane applies when your Hermes gateway is
> fronted by a [Team Gateway connector](https://github.com/NousResearch/gateway-gateway)
> — the enterprise deployment model where the connector owns the platform
> credentials (e.g. one org Slack app) and Hermes speaks a relay protocol to
> it instead of connecting to the platform natively. Standalone/native
> installs can ignore this page; your platform's own page (e.g.
> [Slack](slack.md)) applies instead.

## How configuration works on the relay lane

The relay adapter is platform-neutral: the connector tells Hermes which
platform it fronts, and Hermes expresses per-turn decisions (threading,
status, placement) as frame metadata the connector executes mechanically.

A small set of **platform behavior controls** exist for the fronted platform.
They live under `platforms.relay.extra.<platform>` — a supported *subset* of
that platform's native options — NOT under the native platform block:

```yaml
platforms:
  slack:          # native adapter settings — ignored on the relay lane
    ...
  relay:
    extra:
      slack:      # relay-lane subset for fronted Slack
        reply_in_thread: true
```

Resolution order: the nested `extra.<platform>` object wins → a legacy flat
key on `extra` is honored as a fallback → the option's default.

## Supported controls — Slack

| Key | Default | Effect |
|---|---|---|
| `reply_in_thread` | `true` | `true`: thread-per-message — each top-level DM message opens its own thread carrying the entire turn (status, tool progress, approval cards, final reply), and each message runs as its own session so concurrent messages execute in parallel. `false`: flat rolling DM — everything posts at the DM root, one shared session, a second message steers the in-flight turn. |

Semantics match the native Slack adapter's `reply_in_thread`
([Slack docs](slack.md)); the relay subset exists so relay-fronted behavior
is configured explicitly rather than inherited from a native block that the
relay lane does not read.

In-progress "thinking…" statuses (with live per-tool phrases) are always on,
in whatever form the mode supports: the thread's replies footer in
thread-per-message mode, or anchored to the triggering message in flat mode.
They require only the `chat:write` bot scope — no Slack assistant surface.

Changing a control takes effect on gateway restart — `hermes config set
platforms.relay.extra.slack.reply_in_thread false` and restart; no connector
deploy is involved.

Other fronted platforms currently have no relay-lane controls; the set grows
as enterprise deployments need them (each addition is documented here).
