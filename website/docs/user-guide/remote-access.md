---
sidebar_position: 5
---

# Remote Access

Reach your Hermes backend from another machine — the desktop app's remote
gateway mode, the web dashboard from a phone, or any JSON-RPC/WS client —
without exposing more of the machine than the API itself.

The desktop's remote gateway settings expect "an already-running Hermes
backend on another machine or behind a trusted proxy". This page is that
setup: `hermes serve` (or `hermes dashboard`) stays bound to loopback, and
`hermes dashboard proxy` is the hardened surface you point a tunnel at.

## Why not tunnel straight to the backend?

Tunnelling the whole dashboard server over-shares in three ways:

1. **The SPA HTML embeds the dashboard session token.** Anyone who can load
   the page over the tunnel gets the token inlined in the document. The proxy
   forwards only `/api/*`; the SPA and static assets 404.
2. **Some API routes are machine-lifecycle operations.** A remote surface
   that can trigger `hermes update`, stop gateways, or download a full backup
   turns a leaked URL into a bricked machine or a one-request exfiltration of
   your entire `HERMES_HOME`. The proxy denies those routes by default.
3. **No attribution.** The proxy audit-logs every denied request to
   `~/.hermes/logs/remote-proxy-denied.log` so a surprise attempt has a
   timestamp and source.

Authentication is unchanged: every forwarded request still hits the backend's
own session-token or OAuth checks. The proxy reduces surface; it does not
replace auth, and it must not be your only line of defence.

## Setup

Run the backend and the proxy on the machine that hosts Hermes:

```bash
# 1. The backend, loopback-only (the default). `hermes dashboard` works too.
hermes serve --port 9119

# 2. The hardened remote surface.
hermes dashboard proxy --port 9123 --upstream http://127.0.0.1:9119
```

Then point your tunnel at the proxy, never at the backend directly:

```bash
# Cloudflare tunnel
cloudflared tunnel --url http://127.0.0.1:9123

# Tailscale (tailnet-only)
tailscale serve 9123

# Plain SSH reverse tunnel
ssh -N -R 9123:127.0.0.1:9123 you@your-vps
```

In the desktop app, set the remote gateway URL to the tunnel hostname and
sign in as usual — token and OAuth flows pass through the proxy untouched,
WebSockets included.

## Denied routes

By default the proxy blocks the routes whose blast radius is the machine, not
the conversation:

| Route | Why it is denied remotely |
|-------|---------------------------|
| `/api/hermes/update` | Spawns a self-update that restarts backends with nobody at the machine |
| `/api/gateway/start` `stop` `restart` `drain` | Lifecycle control of the machine's gateways |
| `/api/ops/backup`, `/api/ops/backup/download` | Creates, then serves, a zip of the entire `HERMES_HOME` |
| `/api/ops/import`, `/api/ops/import-upload` | Restores an uploaded archive over the live `HERMES_HOME` |
| `/api/ops/config-migrate` | Rewrites config on disk |

All of them remain available from localhost surfaces. Read-only companions
(for example `/api/hermes/update/check`) are not blocked.

To change the policy per invocation:

```bash
# Re-enable a route remotely (an explicit operator decision):
hermes dashboard proxy --allow-route /api/gateway/restart

# Block something extra:
hermes dashboard proxy --deny-route /api/sessions/import
```

Denied requests return `403` with a one-line body and are appended to
`~/.hermes/logs/remote-proxy-denied.log`:

```
2026-07-18T06:10:54+10:00 DENIED POST /api/hermes/update client=127.0.0.1
```

## Operational notes

- Keep both the backend and the proxy on `127.0.0.1`. The tunnel is the only
  thing that should reach the proxy; nothing should reach the backend but the
  proxy and local clients.
- The proxy is stateless: restart it freely, run it under launchd/systemd
  alongside the gateway, or only when travelling.
- WebSocket routes (`/api/ws`, `/api/events`, `/api/console`, `/api/pty`)
  pass through with the same deny-list applied. If you do not want a remote
  terminal at all, add `--deny-route /api/pty`.
- A denied route is exact-path. The deny-list is a safety net for the
  machine-lifecycle surface, not a substitute for keeping your tunnel
  authenticated (Cloudflare Access, Tailscale ACLs, or SSH).
