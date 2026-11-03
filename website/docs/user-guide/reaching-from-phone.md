---
sidebar_position: 6
title: "Reaching Your Agent From Your Phone"
description: "Ways to reach a self-hosted dashboard from a phone or another network when Tailscale is not an option — SSH tunnel, your own reverse proxy, messaging channels, Tailscale Serve/Funnel — plus the auth and WebSocket constraints every option shares"
---

# Reaching Your Agent From Your Phone

The [web dashboard](./features/web-dashboard.md) documents Tailscale as the default way to reach a machine you do not
physically control, and for a fleet of machines that is usually the right call. But a very common install is the one
Tailscale handles worst in practice: **one always-on box at home** — a Mac mini, a NUC, a Linux server behind a router
on carrier-grade NAT — **reached from a phone that spends its day moving between Wi-Fi, elevators and cellular.**

This page lists the paths that work without installing Tailscale on every device, what each one costs, and the
constraints that apply whichever you choose.

## Four things that are true of every option

1. **A non-loopback bind requires an auth provider.** The dashboard fails closed at startup otherwise. A loopback
   bind (`127.0.0.1`) is the only shape that serves without one, and it accepts loopback peers only — which is exactly
   why "tunnel in, then talk to localhost" stays the cheapest option below. See
   [remote dashboard setup](./features/web-dashboard.md#remote-dashboard-setup).
2. **`GET /api/status` answering says nothing about chat.** Chat lives on separate WebSockets (`/api/ws`, `/api/pty`)
   behind two extra checks. A WS closed with **4403** was rejected by the request guard (Host/peer mismatch); **4401**
   means the single-use ticket did not authenticate. Both are transport-independent, and both look like "the app
   cannot connect".
3. **The Host header must be the bound hostname or the configured public hostname**, or every request answers
   `400 Invalid Host header. Dashboard requests must use the bound hostname or the configured public hostname.` When
   you arrive through a proxy, a tunnel domain, or a MagicDNS name, set `dashboard.public_url`
   (`HERMES_DASHBOARD_PUBLIC_URL`) to the exact `scheme://host:port` clients use. The guard is an OR, so your existing
   IP/address keeps working. Note `public_url` also feeds OAuth callback and Origin checks — restart the dashboard
   after changing it. A report where the identical upgrade succeeded on the raw port and failed through a TLS
   terminator without any diagnostic anywhere is #138057; check by hitting the bound port directly before touching
   routing.
4. **Long-lived sockets are the fragile part, and mobile networks make them more fragile.** On a non-loopback bind
   uvicorn pings every `dashboard.ws_ping_interval` (default 20 s) and closes if the pong misses
   `ws_ping_timeout` (also 20 s) — raise both on high-latency links
   ([configuration reference](./configuration.md)); loopback binds never ping, because there is no middle layer to
   go half-open. Keepalive defeats *idle* timeouts only: anything that caps total connection lifetime (some tunnels
   close a socket after ~30 s regardless of traffic) still cuts it, and the client reattaches on its own. Three
   client-side behaviours are documented on the dashboard page and bite hardest on phones: automatic reconnects
   **pause while the tab is hidden** and resume when you return, **closing the tab reaps the PTY** server-side
   (reopening starts a fresh session), and keystrokes typed inside the reconnect window are discarded (#145511).
   One open bug matters more than any tuning knob: re-attach churn can end an **in-flight turn** outright
   (`end_reason=tui_shutdown`, #136713). So do not hold a long job open in a phone terminal — start it where it can
   finish and have the result delivered.

## Option A — SSH tunnel to a loopback bind

Nothing listens on the network, no auth provider needed, and it works from behind CGNAT because every connection is
outbound.

```bash
# on the home machine — keep the port off the wire
hermes dashboard --host 127.0.0.1 --port 9119 --no-open

# from the device you are actually holding (laptop, or Termux on Android: pkg install openssh)
ssh -N -L 9119:127.0.0.1:9119 -o ServerAliveInterval=30 you@your-home-endpoint
# then browse to http://localhost:9119
```

Because the dashboard sees a loopback peer, the peer-IP guard passes and the WebSocket never pings, so a laggy cell
path cannot manufacture 1006 disconnects from the ping timeout. What it does need:

- **A reachable SSH endpoint**: a router port-forward, or a hop through a cheap VPS
  (`-R 8443:localhost:9119` on the home box, or `-J bastion` from the client). That endpoint is your attack surface —
  key-only auth, fail2ban or similar.
- **A client that can hold the forward.** A browser cannot open the tunnel by itself: on Android use Termux
  (`pkg install openssh`), on iOS an SSH app that exposes a local SOCKS/port forward and survives backgrounding.
  Where the OS suspends background networking (aggressive battery managers do), expect to re-establish the tunnel
  rather than resume it.
- Plaintext HTTP, so sign in over the tunnel. The password travels in a POST body rather than a URL, and the tunnel
  encrypts the rest; still, prefer opening the panel only when you need it.

## Option B — your own public endpoint (VPS reverse proxy, frp, cloudflared)

Pick this when you want a stable bookmarkable URL (`https://agent.example.com`), HTTPS, or access for a second
person who will not tolerate VPN tooling. Terminate TLS at the edge and forward to the dashboard:

```caddyfile
agent.example.com {
    reverse_proxy 127.0.0.1:9119
}
```

```nginx
location / {
    proxy_pass http://127.0.0.1:9119;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 300s;
    proxy_buffering off;
}
```

Checklist specific to Hermes:

- Bind the dashboard to `127.0.0.1` and authenticate **at the proxy** (Caddy `basic_auth`, nginx `auth_basic`,
  Cloudflare Access, Teleport…), **or** bind `0.0.0.0` and let the dashboard's own provider handle it. Do not rely on
  "it is behind a proxy" while leaving a non-loopback bind with no provider — that combination refuses to start.
- In the second case, list the proxy in `dashboard.trusted_proxies` (bounded IPs or CIDRs; `*` and `/0` are refused,
  and forwarding headers from unlisted addresses are ignored). Forwarded-header handling is enabled exactly when the
  auth gate is engaged; without it `X-Forwarded-Proto` is dropped and cookie/redirect decisions see plain HTTP.
- Set `dashboard.public_url` to the public origin, per point 3 above.
- Idle timeout at the edge comfortably above `ws_ping_interval`; upstream's own default (20 s) exists because a
  Cloudflare-style tunnel goes idle around ~100 s. Total-lifetime caps are still fatal to the socket, they just cost
  a reattach.
- Each encapsulation layer you add is another buffer that has to drain. Stacking WebSocket ⊂ WireGuard/TLS ⊂ your
  proxy's TCP means a lossy radio makes the inner stream wait for the outer retransmission window — measured on one
  real tailnet path over cellular this came out at avg ≈3.9 s / max ≈5.9 s with ~40 % sampled loss on the relayed
  leg, versus ~7 ms direct on the same LAN. If a direct path is available, take it.

## Option C — skip the panel: let the agent message you

If the goal while out is *issuing a task and reading the result*, the dashboard is the wrong primitive: its live view
wants one socket held open across whatever your phone is doing. Messaging adapters
([Telegram](./messaging/telegram.md), [Discord](./messaging/discord.md), [Feishu](./messaging/feishu.md),
[WhatsApp](./messaging/index.md) …) run in the gateway process — separate from the dashboard — and deliver
asynchronously, so an elevator or a tower handoff costs you a delayed notification instead of a dead terminal. This
is also the only option that keeps working for a user who will not keep a tunnel app running.

Start the gateway alongside the dashboard if you rely on channels; the dashboard does not start it for you. For
long-running work, [`cron`](./features/cron.md) jobs and background sessions post their output to a channel, which
sidesteps #136713 entirely: nothing is attached when the network blinks, so nothing can be killed by a reattach.

## Option D — Tailscale Serve / Funnel

If the objection to Tailscale is "I do not want to expose a port or manage a proxy", Serve and Funnel answer it:
Serve keeps the listener inside the tailnet (ACL-enforced, no provider needed if you accept tailnet identity as the
credential), Funnel publishes HTTPS to the internet — in which case configure a real auth provider and set
`dashboard.public_url` to the Funnel origin, or the auth gate and the Host guard will both fight you. One known trap
on iOS: iCloud Private Relay rewrites the request chain and breaks sign-in through a Funnel served on the same
machine (#147887) — turning Private Relay off for the site resolves it.

## Which one to pick

| Situation | Path |
| --- | --- |
| You mostly want to fire tasks and read results while away | **C** — messaging channels (+ cron for long jobs) |
| One home box, you are the only user, SSH is already reachable | **A** — SSH tunnel, loopback bind |
| You want a permanent https URL, other people, or bookmarks on many devices | **B** — your own reverse proxy |
| Many devices that roam, no server you want to administer | **D** — Tailscale Serve/Funnel |

Whatever you pick, keep the rule from point 4: the phone is a viewer, not the place where a two-hour job lives.

## Related

- [Hermes Web Dashboard](./features/web-dashboard.md) — binding, auth providers, the auth gate, WS close codes
- [Connecting Desktop to Many Hermes Instances](./multi-connection-desktop.md) — Remote gateway and SSH gateway kinds
- [Messaging adapters](./messaging/index.md) and [cron](./features/cron.md) — delivery that survives a dropped socket
