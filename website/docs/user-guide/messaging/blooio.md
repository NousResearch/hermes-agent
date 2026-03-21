---
sidebar_position: 7
title: "Blooio"
description: "Set up Hermes Agent as a Blooio messaging gateway with webhook-based delivery"
---

# Blooio Setup

Hermes connects to Blooio as a webhook-based messaging gateway. Once configured, Hermes can receive inbound messages, send replies, handle attachments, emit typing indicators, mark chats as read, and deliver scheduled results to a Blooio home chat.

Unlike Socket Mode platforms such as Slack, Blooio requires a **public HTTPS base URL** so Blooio can deliver webhook events back to your Hermes gateway and Hermes can temporarily host outgoing attachment URLs.

:::info Automatic Webhook Management
Hermes manages the Blooio webhook for you. You provide your API key and public base URL, and Hermes creates or updates the instance-specific webhook automatically.
:::

:::warning Public Hostname Required
Blooio does **not** work with a private-only Hermes instance. You must provide a **publicly reachable HTTPS hostname** that forwards to the Hermes gateway. A local-only `localhost` setup, a private LAN IP, or DNS without a real HTTPS service behind it is not enough.
:::

## Prerequisites

- A **Blooio API key**
- A **public HTTPS base URL** that can reach the machine running `hermes gateway`
- Hermes Gateway running somewhere reachable from Blooio

:::tip
Set `BLOOIO_PUBLIC_BASE_URL` to the base origin only, for example `https://hermes.example.com`. Hermes appends the instance-specific webhook path automatically.
:::

## Step 1: Get Your Blooio API Key

Create or copy a Blooio API key from your Blooio workspace or developer settings, then keep it ready for Hermes setup.

:::warning
Treat your Blooio API key like a password. Anyone with it can send messages and manage webhook behavior for that Blooio account.
:::

## Step 2: Decide on Your Public URL

Blooio needs to call back into Hermes over HTTPS. Common setups:

- A reverse proxy such as Nginx, Caddy, or Traefik in front of the Hermes gateway
- A cloud VM with a public DNS name
- A tunnel or ingress URL that exposes the gateway safely over HTTPS

Hermes listens locally and serves:

- `POST /webhooks/blooio/{instance_id}` for inbound Blooio events
- `GET /blooio/media/{instance_id}/...` for short-lived outgoing attachment URLs

You do **not** need to create these routes manually in Blooio. Hermes registers the webhook automatically.

### What `BLOOIO_PUBLIC_BASE_URL` actually means

Set `BLOOIO_PUBLIC_BASE_URL` to the **public HTTPS origin** that the outside world can reach.

Example:

```bash
BLOOIO_PUBLIC_BASE_URL=https://hermes.example.com
```

Hermes then derives URLs like:

- `https://hermes.example.com/webhooks/blooio/{instance_id}`
- `https://hermes.example.com/blooio/media/{instance_id}/...`

Blooio uses those URLs for two things:

1. **Inbound webhooks**: Blooio sends `message.received` and other events to Hermes
2. **Outgoing attachments**: Hermes gives Blooio a public file URL, and Blooio fetches it

Because of that, the URL must be:

- Publicly reachable from the internet
- HTTPS, not plain HTTP
- Routed to the Hermes gateway process

### DNS alone is not enough

Creating a DNS record such as `hermes.example.com -> your AWS host` is only the first step. You still need:

- A process listening for HTTPS traffic on that hostname
- TLS certificates for that hostname
- Routing from public HTTPS traffic to the local Hermes Blooio listener

In practice, that usually means one of:

- A reverse proxy such as **Caddy**, **Nginx**, or **Traefik**
- **Cloudflare Tunnel**
- **Tailscale Funnel**

If you only point DNS at a machine but do not run a proxy/tunnel/service on that host, Blooio has nowhere to send the webhook.

## Public URL Deployment Patterns

### Option A: Public server or VM with reverse proxy

This is the normal production setup.

1. Run `hermes gateway` on your server
2. Set `BLOOIO_WEBHOOK_PORT` to the local port Hermes should bind, for example `8081`
3. Put a reverse proxy in front of it on port `443`
4. Terminate TLS at the proxy
5. Forward requests for your hostname to Hermes

Typical flow:

```text
Internet
  -> https://hermes.example.com
  -> reverse proxy on :443
  -> http://127.0.0.1:8081
  -> hermes gateway
```

For AWS that usually means:

- Your DNS points at the EC2 instance, load balancer, or Cloudflare-managed hostname
- Security groups allow inbound `443`
- The reverse proxy forwards requests to Hermes on `BLOOIO_WEBHOOK_PORT`
- `BLOOIO_PUBLIC_BASE_URL` matches the public hostname exactly

### Option B: Cloudflare Tunnel

This is a strong option if you do **not** want to open inbound ports on the server.

Run `cloudflared` on the same machine as Hermes and publish a hostname that maps to the local Hermes listener. Cloudflare documents this as a public hostname that routes to a local service through an outbound tunnel.

Typical flow:

```text
Blooio
  -> https://hermes.example.com
  -> Cloudflare
  -> cloudflared tunnel
  -> http://127.0.0.1:8081
  -> hermes gateway
```

This works well for:

- AWS instances
- Home servers
- Developer machines

#### Cloudflare Tunnel quick walkthrough (recommended for testing)

If you already use Cloudflare, this is usually the fastest way to test Blooio.

1. Pick a hostname, for example `hermes-blooio.example.com`
2. Install `cloudflared` on the same machine as Hermes
3. Authenticate and create a tunnel:

```bash
cloudflared tunnel login
cloudflared tunnel create hermes-blooio
```

4. Route DNS for the hostname to the tunnel:

```bash
cloudflared tunnel route dns hermes-blooio hermes-blooio.example.com
```

5. Create a `cloudflared` config that forwards the public hostname to the local Hermes Blooio port:

```yaml
tunnel: <TUNNEL-UUID>
credentials-file: /home/your-user/.cloudflared/<TUNNEL-UUID>.json

ingress:
  - hostname: hermes-blooio.example.com
    service: http://127.0.0.1:8081
  - service: http_status:404
```

6. Set your Hermes Blooio config:

```bash
BLOOIO_PUBLIC_BASE_URL=https://hermes-blooio.example.com
BLOOIO_WEBHOOK_PORT=8081
BLOOIO_BIND_HOST=127.0.0.1
```

7. Start Hermes:

```bash
hermes gateway
```

8. Start the tunnel:

```bash
cloudflared tunnel run hermes-blooio
```

9. Verify the public URL reaches Hermes:

```bash
curl https://hermes-blooio.example.com/health
```

Expected response:

```json
{"status": "ok", "platform": "blooio"}
```

If `/health` does not work from the public URL, Blooio webhook delivery and outgoing attachment hosting will not work either.

### Option C: Tailscale Funnel

If Hermes is running on your local computer, plain Tailscale by itself is **not** enough, because Blooio is not inside your tailnet and needs a public HTTPS endpoint.

What can work is **Tailscale Funnel**, which exposes a local service to the public internet over HTTPS. Tailscale documents Funnel as public internet exposure, while `tailscale serve` is tailnet-only.

Typical flow:

```text
Blooio
  -> https://your-device.your-tailnet.ts.net
  -> Tailscale Funnel
  -> http://127.0.0.1:8081
  -> hermes gateway
```

If you use Funnel:

- `BLOOIO_PUBLIC_BASE_URL` should be the Funnel HTTPS origin
- Hermes should listen on the local port that Funnel proxies to
- Be aware Funnel is a Tailscale-managed public URL, not your own normal domain name

## Running Blooio from a Local Computer

Yes, a user can run Blooio from a laptop or desktop, but only if they expose Hermes through a **public HTTPS endpoint**.

Working approaches:

- **Cloudflare Tunnel** on the local machine
- **Tailscale Funnel** on the local machine
- Another public reverse tunnel that gives a stable HTTPS URL

What does **not** work by itself:

- Plain local `localhost`
- Plain LAN IPs like `192.168.x.x`
- Plain Tailscale private access via `tailscale serve`
- A DNS record pointing somewhere that does not actually terminate HTTPS and forward traffic to Hermes

### Local machine checklist

If you want Blooio on your own computer:

1. Start Hermes with a fixed Blooio port, for example `BLOOIO_WEBHOOK_PORT=8081`
2. Expose that local port through Cloudflare Tunnel or Tailscale Funnel
3. Set `BLOOIO_PUBLIC_BASE_URL` to the public HTTPS origin from that tunnel
4. Start `hermes gateway`
5. Confirm that the public URL can reach:
   - `GET /health`
   - `POST /webhooks/blooio/{instance_id}`

For a more general explanation of Hermes features that require public inbound connectivity, see [Webhooks](/docs/user-guide/messaging/webhooks).

:::warning
Avoid temporary or changing public URLs for normal use. If the public URL changes, Hermes will need to update the Blooio webhook registration and any in-flight media URLs may stop working. Stable hostnames are strongly preferred.
:::

## Step 3: Configure Hermes

### Option A: Interactive Setup (Recommended)

```bash
hermes gateway setup
```

Select **Blooio** when prompted. The wizard asks for:

- `BLOOIO_API_KEY`
- `BLOOIO_PUBLIC_BASE_URL`
- `BLOOIO_ALLOWED_USERS`
- `BLOOIO_HOME_CHANNEL` (optional)

### Option B: Manual Configuration

Add the following to `~/.hermes/.env`:

```bash
# Required
BLOOIO_API_KEY=your-blooio-api-key
BLOOIO_PUBLIC_BASE_URL=https://hermes.example.com

# Recommended
BLOOIO_ALLOWED_USERS=+15551234567,team@example.com

# Optional
BLOOIO_HOME_CHANNEL=+15551234567
BLOOIO_WEBHOOK_PORT=8081
BLOOIO_BIND_HOST=0.0.0.0
BLOOIO_FROM_NUMBER=+15557654321
BLOOIO_INSTANCE_ID=work-bot
```

Then start the gateway:

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

On startup, Hermes validates the API key, ensures the webhook exists, and starts the local Blooio listener.

### Example: AWS + Cloudflare DNS

If you already created a Cloudflare DNS record pointing to AWS, you still need one of these behind it:

- A reverse proxy on the AWS host, such as Caddy or Nginx
- A Cloudflare Tunnel running on the AWS host

Minimum checklist:

1. `hermes gateway` is running on the AWS box
2. Hermes is listening on `BLOOIO_WEBHOOK_PORT`
3. Something on the host accepts HTTPS for `hermes.example.com`
4. That HTTPS service forwards requests to Hermes
5. `BLOOIO_PUBLIC_BASE_URL=https://hermes.example.com`

Without steps 3 and 4, the DNS record exists but Blooio still cannot reach Hermes.

## Access Control

### DM Access

Blooio follows the same Hermes DM authorization model as Telegram, Signal, and the other gateway connectors:

1. **`BLOOIO_ALLOWED_USERS` set** → only those users can message
2. **No allowlist set** → unknown DM senders get a pairing code
3. **`BLOOIO_ALLOW_ALL_USERS=true`** → anyone can message

Allowed users can be sender identifiers such as phone numbers or email-style identifiers, depending on the Blooio channel type.

## Home Channel

Set `BLOOIO_HOME_CHANNEL` to the chat where Hermes should deliver scheduled task results, cron notifications, and other proactive messages.

Examples:

```bash
BLOOIO_HOME_CHANNEL=+15551234567
# or
BLOOIO_HOME_CHANNEL=ops@example.com
```

You can also set the current Blooio chat as the home channel from inside messaging with `/sethome`.

## Features

### Inbound Messaging

Hermes receives Blooio webhook events and turns `message.received` into normal gateway conversations:

- **1:1 chats** map to the sender as the chat ID
- **Group chats** use the Blooio group ID as the chat ID
- Blooio delivery/read/reaction events are tracked as control events and do not pollute the conversation transcript

### Attachments

The Blooio adapter supports both receiving and sending:

- **Images**
- **Audio / voice attachments**
- **Videos**
- **Documents**

Incoming attachments are downloaded into Hermes's local media caches so the usual vision, document, and speech-to-text flows work the same way they do on Telegram.

Outgoing attachments are staged under the active Hermes home and served from short-lived signed URLs. Blooio then fetches those URLs natively.

### Typing Indicators and Read Receipts

While Hermes is processing a Blooio message, it can:

- Start a typing indicator
- Stop the typing indicator when processing completes or is interrupted
- Mark the chat as read once the inbound message is accepted

## Multiple Hermes Instances on One Machine

Blooio is fully scoped to the resolved Hermes home, so multiple Hermes instances can coexist on one machine.

- State and staged media live under the active Hermes home, not a hardcoded `~/.hermes`
- If you launch Hermes with `HOME=/path/to/bot1` and `HERMES_HOME=/path/to/bot1`, Blooio state stays under that tree
- Each instance gets its own derived `instance_id` unless you explicitly set `BLOOIO_INSTANCE_ID`

For multi-instance setups:

- Give each instance a distinct `HERMES_HOME`
- Use a unique `BLOOIO_WEBHOOK_PORT` if they bind locally on the same host
- Use a unique public URL or unique instance path per instance

:::tip
If you run multiple Hermes gateways against the same Blooio account, Hermes only manages the webhook matching its own exact instance URL and keeps its webhook metadata under that instance's Hermes home.
:::

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Gateway says `BLOOIO_PUBLIC_BASE_URL not set` | Set a public HTTPS base URL. Blooio cannot work without it because inbound webhooks and outgoing attachment URLs both depend on it. |
| Blooio messages never arrive | Verify the public URL is reachable from the internet, the reverse proxy forwards requests to Hermes, and the gateway is running. |
| Startup fails with a webhook lock error | Another Hermes instance is already managing the same Blooio webhook URL. Give each instance a distinct `HERMES_HOME`, `BLOOIO_INSTANCE_ID`, and/or public route. |
| Attachments fail to send | Make sure `BLOOIO_PUBLIC_BASE_URL` is correct and publicly reachable so Blooio can fetch the staged media URL. |
| Responses go to the wrong chat | Set `BLOOIO_HOME_CHANNEL` explicitly or use `/sethome` from the correct Blooio conversation. |
| Unauthorized users get rejected | Add them to `BLOOIO_ALLOWED_USERS`, use DM pairing, or explicitly set `BLOOIO_ALLOW_ALL_USERS=true` if that is acceptable for your deployment. |
| Port already in use | Set a different `BLOOIO_WEBHOOK_PORT` for that Hermes instance. |

## Security

:::warning
Always configure `BLOOIO_ALLOWED_USERS`, DM pairing, or an explicit gateway policy before exposing Blooio publicly. Hermes has terminal and tool access, so the safe default is to deny unknown users.
:::

- Keep `BLOOIO_API_KEY` secret
- Protect the Hermes home directory because Blooio webhook metadata and staged media state are stored there
- Prefer explicit allowlists or DM pairing over `BLOOIO_ALLOW_ALL_USERS=true`
- If you reverse-proxy the gateway publicly, protect TLS termination and logs the same way you would for any authenticated webhook service
