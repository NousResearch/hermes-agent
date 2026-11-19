# Publishing Hermes Desktop to a browser

Publish the full Hermes Desktop workspace — sessions, files, terminal, Git, cron, Bots screen — in a normal browser, served by the gateway itself. This is the same renderer the native Desktop app ships, not the admin dashboard.

> Run `hermes gateway setup` (or `hermes setup`) and pick **🖥️ Hermes Desktop (browser)** for a guided walk-through (Linux + systemd) — the same menu where you configure Telegram, Discord, and the rest. Sign-in is your **Nous account** (Nous Portal OAuth); the setup keeps it Nous-only. On macOS/Windows, or without a domain, use `hermes webapp` locally or the Desktop app's **Settings → Web Access**.

This page covers the headless deployment path: a server/VM with no display, where the browser is the only surface.

## Requirements

- Linux with systemd (the setup flow configures Caddy and avahi). On macOS/Windows use `hermes webapp` locally, or the native Desktop app's **Settings → Web Access**.
- A signed-in Nous account (`hermes portal login`). Free-tier guest accounts cannot register sign-in.
- Node + network for the first renderer build (minutes; later runs reuse the bundle).
- **Local mode:** `avahi-daemon` + `avahi-utils` (the setup flow offers install commands).
- **Domain mode:** Caddy, a DNS A record to this machine, and ports **80/443** reachable (cloud security group *and* host firewall).

## The two access modes

### 1. Local network (`hermes-desktop.local`)

The gateway's webapp binds all interfaces (the sign-in gate engages), avahi publishes `hermes-desktop.local`, and Caddy serves `http://hermes-desktop.local` as a friendly redirect to the gateway's LAN origin.

Two constraints shape this mode — both are verified behaviors, not bugs:

- **Nous Portal OAuth accepts only IP or real-domain callbacks** (never `.local`). Sign-in therefore completes on the IP address (`http://<LAN-IP>:9120`); the `.local` name is the front door, not the OAuth origin.
- **Avahi registers one name per local interface address**, so the machine's mDNS identity is set to the name you choose (avahi `host-name=`). The setup flow always asks before changing it, and you can decline and use the IP URL.

Client caveats: macOS/iOS resolve `.local` natively; Windows 10 1703+ and Linux (avahi/nss-mdns) usually do; Android 12+ (Nov 2021 patch) can — type the full `http://hermes-desktop.local/` with the scheme. Older Android and VPN-connected devices should use the IP URL. mDNS does not cross routers/VLANs.

Plain HTTP on the LAN is **not a secure context**: clipboard reading, microphone (dictation/voice), system notifications and the Bot Screen viewer are disabled by the browser. For full features use domain mode (HTTPS), an SSH tunnel (`ssh -L 9120:127.0.0.1:9120 host` keeps `http://localhost`, a secure context), or Tailscale Serve (`tailscale serve --https=443 9120`).

### 2. Custom domain (HTTPS)

The setup flow shows this machine's public IP, you create a DNS **A record** (e.g. `desk.example.com → <IP>`), it verifies the record against public resolvers, then Caddy fronts the loopback webapp with automatic Let's Encrypt HTTPS. Sign-in is registered for `https://<your-host>/auth/callback`.

Notes:

- A VPS often cannot reach its own public IP (no hairpin NAT) — local self-tests may fail while issuance and external access work. Test from your own device, or `curl --resolve <host>:443:127.0.0.1 https://<host>/`.
- Certificate issuance starts on the first request after the gateway restart; give it a minute.
- No domain handy for a trial? `<ip>.nip.io` / `<ip>.sslip.io` resolve to the embedded IP automatically (shared services with Let's Encrypt rate limits — fine for testing, not a product default).

## How it works

- The **gateway owns the listener** (`webapp.enabled/host/port/public_url` in `config.yaml`, read at startup). There is no separate webapp service.
- The **sign-in gate** engages on a non-loopback bind or a declared `webapp.public_url`, and fails closed without a provider. The setup registers the webapp's own Portal client (`webapp/sign-in-registration.json`) and disables other interactive sign-in providers, so the login page offers **Nous Research only** (single-provider auto-SSO).
- Switching modes cleans up the previous mode's artifacts (its Caddy site block, and — with your confirmation — the mDNS identity change). Only blocks the setup flow itself wrote are ever removed.
- Webapp access is **root-equivalent**: a signed-in user can run commands and read the account's credentials. Keep the gate on, prefer HTTPS, and never publish the loopback port directly.

## Manual reference

```bash
hermes config set webapp.enabled true
hermes config set webapp.host 127.0.0.1        # loopback behind Caddy
hermes config set webapp.port 9120
hermes config set webapp.public_url https://desk.example.com
hermes webapp --build-only                     # renderer bundle
hermes gateway restart                         # webapp.* is startup-only
hermes webapp --status                         # serving?
hermes webapp --sign-out                       # end every browser session
```

## Troubleshooting

- **`hermes webapp --status` says off after setup** — the gateway has not restarted onto the new config yet.
- **Gate refuses to start** — a gated bind with no registered sign-in provider fails closed by design; re-run the setup (it registers the Nous client).
- **`Local name collision` in avahi logs** — something else already owns that mDNS name/address; pick a different name.
- **Certificate never issues** — check the A record (`dig +short <host> A`), ports 80/443 from the internet, and `journalctl -u caddy -n 50`. A default `:80` catch-all site can shadow the ACME challenge; remove it.
- **Password sign-in reappears** — running `hermes dashboard`'s password setup re-enables the basic provider; leave `dashboard.basic_auth` empty to keep Nous-only.