# Cloudflare Tunnel exposure (`hermes tunnel`)

`hermes tunnel` exposes a local app, service, or API you built to the internet on a
per-user subdomain of your zone (e.g. `alice.noit2.com`) via a Cloudflare named tunnel.
The exposure is **ephemeral by default**: an idle-reset 30-minute dead-man's switch
closes the tunnel when traffic stops, so a forgotten test build can't leak to the
internet. For longer exposure, file a hold request and have an admin approve it.

The dashboard is just one possible origin — point a route at `127.0.0.1:9119` to expose it.

## Prerequisites

1. Install `cloudflared` and log in: `cloudflared login`.
2. Create a named tunnel: `cloudflared tunnel create <name>`. Note the credentials JSON path.
3. Route DNS for each subdomain: `cloudflared tunnel route dns <name> alice.noit2.com`.
4. Put the credentials path in config (below) or `HERMES_TUNNEL_CREDS`.

## Config

`~/.hermes/config.yaml` (or env overrides `HERMES_TUNNEL_*`):

```yaml
tunnel:
  zone: "noit2.com"
  tunnel_name: "alice"
  credentials_file: "/home/alice/.cloudflared/<uuid>.json"
  idle_timeout_seconds: 1800
  admin: ["alice"]            # who may approve/deny hold requests
  routes:
    - subdomain: alice
      host: 127.0.0.1
      port: 3000
```

## Commands

```bash
# Expose two origins (CLI origins override config routes):
hermes tunnel up --origin alice=127.0.0.1:3000 --origin alice-api=127.0.0.1:8080

# Start and immediately request a longer hold:
hermes tunnel up --origin alice=127.0.0.1:3000 --hold-request --reason "demo" --until 4h

# Mid-session, request more time:
hermes tunnel hold --reason "demo running long" --until 4h

# Admin side:
hermes tunnel requests
hermes tunnel approve <id> --until 6h
hermes tunnel deny <id> --reason "too long"

hermes tunnel status
hermes tunnel doctor
hermes tunnel down
```

## How the dead-man's switch works

Every `poll_interval_seconds` (default 5s) the supervisor reads cloudflared's request
counter from its `--metrics` endpoint. **Any increase resets the 30-minute idle clock.**
If 30 minutes pass with no increase, cloudflared is gracefully drained and killed; your
local origins keep running. An approved hold disables the idle timer until the approved
expiry, after which the idle timer resumes (the approval expiring does NOT hard-kill).

## Layout

`noit2.com` root is the Cloudflare Pages brand site. Each user's services live on their
own subdomains. Origins bind to `127.0.0.1` only — no port opens on your firewall; the
only public path is Cloudflare's edge on 443.
