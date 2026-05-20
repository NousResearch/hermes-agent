# Sendblue (iMessage, no Mac)

Connect Hermes to Apple iMessage via [Sendblue](https://sendblue.com) — a cloud relay that delivers iMessages without requiring a Mac. The trade-off compared to [BlueBubbles](bluebubbles.md): Sendblue is a paid SaaS, but it works on any host (Linux VPS, etc.) and doesn't tie you to an always-on macOS machine.

## Prerequisites

- A [Sendblue](https://sendblue.com) account
- A phone number purchased through the Sendblue dashboard
- API keys from [app.sendblue.com/api-keys](https://app.sendblue.com/api-keys)
- A **publicly reachable HTTPS URL** that points at this machine — Sendblue's webhook delivery requires TLS, and Let's Encrypt won't issue certs for bare IP addresses. See [Public webhook URL](#public-webhook-url) below.
- A reverse proxy (Caddy, nginx, etc.) in front of the gateway port

## Setup

### 1. Sign up and get credentials

1. Create an account at [sendblue.com](https://sendblue.com)
2. Purchase a phone number from the dashboard
3. Generate API keys at [app.sendblue.com/api-keys](https://app.sendblue.com/api-keys)

### 2. Run the setup wizard

```bash
hermes gateway setup
```

Select **Sendblue (iMessage, no Mac)** from the platform list. The wizard will:

1. Prompt for your public webhook URL (with a guidance menu if you don't have one yet — see [below](#public-webhook-url))
2. Print copy-pasteable Caddy and nginx reverse-proxy snippets
3. Validate your API key/secret against Sendblue's API before saving
4. Generate a webhook signing secret if you don't supply one
5. Walk you through allowlist + home channel configuration

All credentials land in `~/.hermes/.env`. Nothing goes to `config.yaml`.

### 3. Register the webhook in the Sendblue dashboard

After the wizard finishes:

1. Open [app.sendblue.com/webhooks](https://app.sendblue.com/webhooks)
2. Create a new webhook pointing at the URL you configured (must end in `/sendblue-gateway/receive`)
3. Paste the signing secret the wizard generated
4. Save

The gateway will start receiving inbound iMessages on its next restart.

## Public webhook URL

Sendblue needs to POST inbound messages to a URL you control. Two constraints:

- The URL **must be HTTPS** with a valid TLS certificate
- Let's Encrypt — the only practical free certificate authority — **won't issue certificates for bare IP addresses**, only hostnames

So you need a hostname, but it doesn't have to be a domain you bought. Four options, easiest first:

### Cloudflare Tunnel {#cloudflare-tunnel}

Free. Gives you a `*.trycloudflare.com` hostname automatically. **No domain registration, no open ports, no static IP needed.** Works behind CGNAT / double-NAT — probably the lowest-friction option for residential ISPs.

```bash
# Install cloudflared (see https://github.com/cloudflare/cloudflared/releases)
cloudflared tunnel --url http://localhost:8665
```

The CLI prints a `https://<random>.trycloudflare.com` URL. Use that as your `SENDBLUE_WEBHOOK_PUBLIC_URL`, with `/sendblue-gateway/receive` appended. The tunnel must stay running for inbound messages to land — run it as a systemd service for production use.

For a stable hostname, [authenticate with Cloudflare](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) and create a named tunnel under a domain you own (free if you transfer the domain to Cloudflare).

### Tailscale Funnel {#tailscale-funnel}

Free for personal use. Gives you a `*.ts.net` hostname under your tailnet. Requires the Tailscale daemon running.

```bash
# After installing and authenticating Tailscale on this machine:
sudo tailscale funnel --bg --set-path /sendblue-gateway/receive \
  http://localhost:8665/sendblue-gateway/receive
```

Your URL is `https://<hostname>.<tailnet>.ts.net/sendblue-gateway/receive`. Funnel must be enabled at the tailnet level in the Tailscale admin console.

### DuckDNS + Caddy {#duckdns-caddy}

Free. Requires a stable public IP and an open port 443. Right path if your ISP gives you a static or rarely-changing IP and you can forward ports.

1. Sign up at [duckdns.org](https://www.duckdns.org), claim a subdomain (e.g. `myname.duckdns.org`), set the IP to your machine
2. Install Caddy and add a site block (the setup wizard prints the exact snippet)
3. Caddy fetches a Let's Encrypt cert automatically on first start

Webhook URL becomes `https://myname.duckdns.org/sendblue-gateway/receive`.

### ngrok {#ngrok}

Free tier rotates URLs on every restart, which breaks Sendblue's webhook registration (you'd have to re-register the URL with Sendblue every restart). **Use the paid plan for static URLs**:

```bash
ngrok http --domain=your-static-domain.ngrok.app 8665
```

## Reverse proxy

The gateway listens on `127.0.0.1:8665` by default. Your reverse proxy needs to forward `/sendblue-gateway/*` to it. The setup wizard prints these snippets, but for reference:

**Caddy:**
```
yourdomain.com {
    handle /sendblue-gateway/* {
        reverse_proxy 127.0.0.1:8665
    }
}
```

**nginx:**
```
location /sendblue-gateway/ {
    proxy_pass http://127.0.0.1:8665/sendblue-gateway/;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $remote_addr;
}
```

Reload your proxy after adding the snippet (`systemctl reload caddy` or `systemctl reload nginx`).

To change the gateway port, set `SENDBLUE_WEBHOOK_PORT` in `~/.hermes/.env`.

## Authorize users

Choose one approach:

**Pre-authorize specific phone numbers** (recommended for SMS — open access is risky):
```bash
SENDBLUE_ALLOWED_USERS=+15551234567,+15559876543
```

**DM Pairing:** Unknown senders are prompted for a pairing code; you approve via `hermes pairing approve sendblue <CODE>`.

**Open access** (anyone with your Sendblue number can message the bot — not recommended):
```bash
GATEWAY_ALLOW_ALL_USERS=true
```

## Home channel

Set the phone number that receives cron-job output and notifications:

```bash
SENDBLUE_HOME_CHANNEL=+15551234567
```

If you skip this during setup, you can configure it later by sending `/set-home` from your iMessage chat with the bot.

## Optional configuration

```bash
# Default send-style applied to all replies — confetti, balloons, fireworks, etc.
# Pokes and system messages stay unstyled.
SENDBLUE_DEFAULT_SEND_STYLE=confetti

# Override the local gateway port (default 8665)
SENDBLUE_WEBHOOK_PORT=8665
```

## Troubleshooting

**Setup says "Sendblue rejected the credentials (HTTP 401)"**
Your API key ID is wrong (Sendblue's read endpoints only authenticate against the key ID, so the wizard's "verified" step proves the ID is valid but cannot detect a wrong secret). Re-copy the key ID from [app.sendblue.com/api-keys](https://app.sendblue.com/api-keys). The setup wizard retries up to three times.

**Setup says credentials verified, but the gateway fails to register the webhook on first start**
The secret was wrong. Re-copy `SENDBLUE_API_SECRET` from [app.sendblue.com/api-keys](https://app.sendblue.com/api-keys) and restart the gateway — the secret is only validated when the adapter POSTs to Sendblue at connect time.

**Setup says "Could not reach Sendblue"**
Network/DNS issue between this machine and `api.sendblue.com`. Check that outbound HTTPS works. If you're behind a strict corporate firewall, you may need to allow `api.sendblue.com`.

**Gateway starts but webhooks never arrive**
- Confirm the webhook is registered in [app.sendblue.com/webhooks](https://app.sendblue.com/webhooks) pointing at the same URL the gateway expects
- Verify the URL is reachable from the public internet: `curl -i https://yourdomain.com/sendblue-gateway/receive` should return a non-error response (the gateway rejects unsigned GETs, but the connection itself should succeed)
- Check the reverse proxy's access log — the path must be `/sendblue-gateway/receive` exactly

**Signature verification failures in the gateway log**
The signing secret in `~/.hermes/.env` (`SENDBLUE_WEBHOOK_SECRET`) doesn't match the value configured in the Sendblue dashboard. Re-paste it.

**Group chat messages don't arrive**
Sendblue cloud-relay numbers don't reliably function as iMessage group participants when added from an Apple device — this is an Apple iMessage routing limitation, not a Hermes/Sendblue bug. Direct messages work as expected.

## See also

- [BlueBubbles](bluebubbles.md) — Mac-hosted alternative with no per-message fees
- [Messaging gateway overview](index.md)
