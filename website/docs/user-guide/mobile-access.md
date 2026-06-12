---
sidebar_position: 8
title: "Mobile Access"
description: "Access your Hermes Agent from iPhone or Android over Tailscale — secure, no exposed ports, with an optional home screen app experience."
---

# Mobile Access

You can use Hermes Agent from your iPhone or Android device without building a native app. The web dashboard (or an alternative web UI like [Hermes Workspace](https://github.com/outsourc-e/hermes-workspace)) runs on your server and is reachable from your phone over a private [Tailscale](https://tailscale.com/) tunnel. Add it to your home screen and it launches like an app — no App Store, no port forwarding, no public exposure.

The core idea is simple: your phone connects directly to your server over WireGuard (Tailscale), so the agent is always reachable but never visible to the internet. All the heavy lifting — model inference, tool execution, cron jobs — stays on the server. Your phone is just the window.

## Why Tailscale?

| Approach | Security | Setup effort | Push notifications | Standalone app feel |
|----------|----------|-------------|-------------------|-------------------|
| **Tailscale + home screen** | Best — encrypted mesh, no open ports | ~5 min | No | Decent (Safari chrome visible) |
| Tailscale Funnel | Good — HTTPS via DERP relay | ~10 min | No | Full PWA (no browser chrome) |
| Caddy reverse proxy | Good — full HTTPS | ~30 min | No | Full PWA |
| Cloudflare Tunnel | Good — HTTPS, Cloudflare edge | ~15 min | No | Full PWA |
| Public internet + auth | Weakest — open port, needs auth hardening | ~15 min | No | Varies |

Tailscale's personal tier is free for up to 3 users and 100 devices, which is more than enough for a single-user Hermes deployment.

## Prerequisites

- A server running Hermes Agent (Linux, macOS, or WSL)
- [Tailscale](https://tailscale.com/download) installed on the server
- Tailscale installed on your phone ([iOS](https://apps.apple.com/us/app/tailscale/id1470499037), [Android](https://play.google.com/store/apps/details?id=com.tailscale.ipn))
- Both devices signed into the same Tailscale account

Verify your server's Tailscale IP:

```bash
tailscale ip -4
# Example output: 100.64.0.5
```

## Step 1: Start the web dashboard on the Tailscale IP

The dashboard must bind to your Tailscale IP (not `127.0.0.1`) so your phone can reach it through the tunnel:

```bash
# Start the dashboard bound to your Tailscale IP
hermes dashboard --host $(tailscale ip -4) --no-open
```

The `--no-open` flag skips opening a browser on the server (which isn't useful in this case).

To keep it running persistently:

```bash
# Option A: systemd user service (survives logout)
systemctl --user enable --now hermes-dashboard
# Then configure the bind address in the service override

# Option B: tmux session (simple, manual restart needed)
tmux new-session -d -s dashboard "hermes dashboard --host $(tailscale ip -4) --no-open"

# Option C: background process
nohup hermes dashboard --host $(tailscale ip -4) --no-open > /dev/null 2>&1 &
```

:::info Daemon bind address
When the dashboard is launched by the gateway (Dashboard as a sidebar pane), it binds to `127.0.0.1` by default. For mobile access you need a separate instance bound to your Tailscale IP, or configure the dashboard to bind to `0.0.0.0` and rely on Tailscale's ACLs for security.
:::

## Step 2: Connect from your phone

1. Open the Tailscale app on your phone and verify it's connected (green dot)
2. Open Safari (iPhone) or Chrome (Android)
3. Navigate to `http://<tailscale-ip>:9119` (e.g., `http://100.64.0.5:9119`)
4. You should see the Hermes dashboard

If you're using [Hermes Workspace](https://github.com/outsourc-e/hermes-workspace) instead of the built-in dashboard, the port is typically `8642` and it provides chat, file browsing, memory, skills, and terminal in one interface. The Tailscale access pattern is identical.

### Security note

The built-in dashboard at `http://<tailscale-ip>:9119` is accessible to anyone on your tailnet. For a single-user setup this is fine — no one else is on your tailnet. If you share your tailnet with others, enable [dashboard authentication](./features/web-dashboard.md#authentication-providers) or restrict access with [Tailscale ACLs](https://tailscale.com/kb/1018/acls).

## Step 3: Add to Home Screen

This is what makes it feel like an app:

**iPhone (Safari):**
1. Open the dashboard URL in Safari
2. Tap the Share button (square with arrow) at the bottom
3. Scroll down and tap **Add to Home Screen**
4. Give it a name (e.g., "Hermes") and tap **Add**

**Android (Chrome):**
1. Open the dashboard URL in Chrome
2. Tap the three-dot menu (⋮)
3. Tap **Add to Home Screen**
4. Give it a name and tap **Add**

You'll get an icon on your home screen that opens Hermes in a streamlined browser window. It won't have Safari/Chrome toolbar chrome (on Android) or will minimize it (on iPhone).

:::note Not a full PWA
Without HTTPS, iOS won't treat this as a full Progressive Web App — you'll still see the Safari navigation bar, and there are no push notifications or offline caching. For the full standalone app experience, set up HTTPS via one of the options below.
:::

## Optional: HTTPS for full PWA experience

Three paths to get `https://` so iOS treats your bookmark as a proper PWA (standalone window, no browser chrome, splash screen):

### Tailscale Funnel (easiest)

[Tailscale Funnel](https://tailscale.com/kb/1223/funnel) gives your dashboard a public `*.ts.net` domain with automatic Let's Encrypt certificates. One command:

```bash
tailscale funnel 9119
```

Then add `https://<your-machine>.<tailnet>.ts.net:9119` to your home screen. The traffic routes through Tailscale's DERP relay — slightly more latency than direct tailnet, but still encrypted end-to-end at the WireGuard layer.

:::caution Funnel exposes your dashboard to the internet
Anyone who guesses or discovers your `*.ts.net` URL can reach your dashboard. Always enable [dashboard authentication](./features/web-dashboard.md#authentication-providers) before using Funnel. The URL is unguessable (machine name + tailnet hash) but not a substitute for auth.
:::

### Caddy reverse proxy

If you own a domain, run [Caddy](https://caddyserver.com/) on your server to get HTTPS with automatic certificates. Point your domain's DNS at your Tailscale IP, then:

```bash
# /etc/caddy/Caddyfile
hermes.yourdomain.com {
    reverse_proxy 127.0.0.1:9119
}
```

Access `https://hermes.yourdomain.com` over Tailscale — Caddy handles the certs, Tailscale handles the networking.

### Cloudflare Tunnel

If you already use Cloudflare, `cloudflared tunnel` gives you HTTPS with Cloudflare's edge certificates. Similar caveat to Funnel — the endpoint is internet-reachable.

## Android-specific notes

- Android Chrome's "Add to Home Screen" creates a true WebAPK (with HTTPS) or a shortcut (without). Both work.
- Consider installing [Firefox for Android](https://play.google.com/store/apps/details?id=org.mozilla.firefox) if you prefer a different rendering engine.
- [Termux](https://hermes-agent.nousresearch.com/docs/getting-started/termux) can run Hermes Agent directly on Android — useful if you want the agent itself on your phone rather than reaching a remote server.

## Troubleshooting

**"Connection refused" on the phone**
- Confirm the dashboard is bound to the Tailscale IP: `ss -tlnp | grep 9119` should show `100.x.x.x:9119`, not `127.0.0.1:9119`
- Check that Tailscale is connected on both devices
- Verify you're using the Tailscale IP, not the server's LAN IP

**Dashboard loads but chat doesn't work**
- The Chat tab requires Node.js and the `pty` extra: `pip install 'hermes-agent[pty]'`
- On the first launch the frontend builds automatically if `npm` is available

**Dashboard works on LAN but not over Tailscale**
- Some VPS providers block non-standard ports even on private interfaces. Try `--port 8080` or `--port 443`
- Verify Tailscale ACLs aren't blocking the port: `tailscale status`

**"Not secure" warning on iPhone**
- Expected for `http://` URLs. Harmless within your tailnet — the WireGuard tunnel already encrypts all traffic. If the warning bothers you, set up HTTPS.

**Phone goes to sleep and loses connection**
- This is normal browser behavior. The home screen bookmark reconnects on next open. For persistent connectivity and notifications, consider the [messaging gateway](./messaging/index.md) with Telegram or Discord alongside the dashboard.
