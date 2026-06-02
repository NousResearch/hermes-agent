---
title: "Pinggy Tunnel — Zero-install localhost tunnels over SSH via Pinggy"
sidebar_label: "Pinggy Tunnel"
description: "Zero-install localhost tunnels over SSH via Pinggy"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pinggy Tunnel

Zero-install localhost tunnels over SSH via Pinggy.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/devops/pinggy-tunnel` |
| Path | `optional-skills/devops/pinggy-tunnel` |
| Version | `0.1.0` |
| Author | Teknium (teknium1), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Pinggy`, `Tunnel`, `Networking`, `SSH`, `Webhook`, `Localhost` |
| Related skills | `cloudflared-quick-tunnel`, [`webhook-subscriptions`](/docs/user-guide/skills/bundled/devops/devops-webhook-subscriptions) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Pinggy Tunnel Skill

Expose a local service (dev server, webhook receiver, MCP endpoint, demo) to the public internet using a Pinggy SSH reverse tunnel. No daemon to install — the user's stock SSH client connects to `a.pinggy.io:443` and Pinggy hands back a public HTTP/HTTPS URL.

Free tier: 60-minute tunnels, random subdomain, no signup. Pro tier ($3/mo) is an opt-in with a token.

## When to Use

- User asks to "expose this locally", "share my dev server", "make this URL public", "tunnel port N", "get a public URL for a webhook"
- Need to receive a webhook callback during a local task (Stripe, GitHub, Discord, AgentMail)
- Sharing a one-off HTTP demo (MCP server, Ollama/vLLM endpoint, dashboard) with a remote party
- The host has SSH but no `cloudflared` / `ngrok` binary, and installing one would be overkill

If the host already has `cloudflared` configured, prefer the `cloudflared-quick-tunnel` skill — Cloudflare quick tunnels don't expire after 60 minutes.

## Prerequisites

- `ssh` on PATH (`ssh -V`). Default on Linux, macOS, and Windows 10+. No other install.
- A local service listening on `127.0.0.1:<port>` before the tunnel starts. Pinggy will return URLs but they'll 502 until the local origin is up.

Optional:

- `PINGGY_TOKEN` env var for paid Pro features (persistent subdomain, custom domain, multiple tunnels, no 60-minute cap). Free tier needs no credentials.

## Quick Reference

```bash
# Plain HTTP/HTTPS tunnel for port 8000 (free tier)
ssh -p 443 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -R0:localhost:8000 free@a.pinggy.io

# TCP tunnel (databases, raw SSH, etc.)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:5432 tcp@a.pinggy.io

# TLS tunnel (Pinggy can't decrypt — bring your own certs at origin)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:443 tls@a.pinggy.io

# Basic auth gate (b:user:pass)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "b:admin:secret+free@a.pinggy.io"

# Bearer token gate (k:token)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "k:mysecrettoken+free@a.pinggy.io"

# IP whitelist (w:CIDR)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "w:203.0.113.0/24+free@a.pinggy.io"

# Enable CORS + force HTTPS redirect
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "co+x:https+free@a.pinggy.io"

# Pro tier (persistent URL, no 60-min cap)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 "$PINGGY_TOKEN+a.pinggy.io"
```

## Procedure — Start a Tunnel and Get the URL

The model SHOULD use the `terminal` tool. The tunnel must stay alive for the duration of the share, so run it as a background process and parse the public URL from stdout.

### 1. Confirm a local origin is up

```bash
curl -sI http://127.0.0.1:8000/ | head -1
# expect HTTP/1.x 200 (or any non-connection-refused response)
```

If nothing is listening yet, start it first (e.g. `python3 -m http.server 8000 --bind 127.0.0.1`). Pinggy will happily return a URL pointed at nothing — the user will see 502 until the origin comes up.

### 2. Launch the tunnel as a background process

Use `terminal(background=True)` and capture output to a logfile (Pinggy prints the URLs on stdout, then keeps the connection open):

```bash
LOG=/tmp/pinggy-8000.log
nohup ssh -p 443 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -R0:localhost:8000 free@a.pinggy.io \
    > "$LOG" 2>&1 &
echo $! > /tmp/pinggy-8000.pid
```

`StrictHostKeyChecking=no` + `UserKnownHostsFile=/dev/null` skips the first-run host-key prompt. `ServerAliveInterval=30` keeps the SSH session from getting torn down by an idle NAT.

### 3. Parse the URL out of the log

```bash
sleep 4
grep -oE 'https://[a-z0-9-]+\.[a-z]+\.pinggy\.link' /tmp/pinggy-8000.log | head -1
```

Expected output looks like:

```
You are not authenticated.
Your tunnel will expire in 60 minutes.
http://yqycl-98-162-69-48.a.free.pinggy.link
https://yqycl-98-162-69-48.a.free.pinggy.link
```

Hand the `https://...pinggy.link` URL to the user.

### 4. Verify

```bash
curl -sI https://<the-url>/ | head -3
# expect 200/302/whatever the local origin actually returns
```

If you get `502 Bad Gateway`, the SSH session is up but the local origin isn't listening — fix step 1 first.

### 5. Teardown

```bash
kill "$(cat /tmp/pinggy-8000.pid)"
# or, if the pid file got lost:
pkill -f 'ssh -p 443 .* free@a\.pinggy\.io'
```

If you have a session_id from `terminal(background=True)`, prefer `process(action='kill', session_id=...)`.

## Access Control via Username Keywords

Pinggy stacks control flags into the SSH username separated by `+`. Always quote the whole `user@host` argument when it contains a `+`:

| Keyword | Effect |
|---------|--------|
| `b:user:pass` | HTTP Basic auth gate |
| `k:token` | Bearer-token header gate (`Authorization: Bearer <token>`) |
| `w:CIDR` | IP whitelist (single IP or CIDR, repeatable) |
| `co` | Add `Access-Control-Allow-Origin: *` (CORS) |
| `x:https` | Force HTTPS — auto-redirect HTTP to HTTPS |
| `a:Name:Value` | Add request header |
| `u:Name:Value` | Update request header |
| `r:Name` | Remove request header |
| `qr` | Print a QR code of the URL to stdout (handy for mobile sharing) |

Combine freely: `"b:admin:secret+co+x:https+free@a.pinggy.io"`.

## Web Debugger (optional)

Pinggy can mirror the inbound traffic to `localhost:4300` for inspection. Add a local forward to the SSH command:

```bash
ssh -p 443 -L4300:localhost:4300 -R0:localhost:8000 free@a.pinggy.io
```

Then open `http://localhost:4300` in a browser to see live request/response pairs.

## Pitfalls

- **60-minute hard cap on the free tier.** The SSH session terminates at the 60-minute mark; the URL goes dead. For longer shares, either use `PINGGY_TOKEN` (Pro) or auto-restart with a shell loop (note that the URL changes on every restart for free-tier).
- **Free-tier URL is random and changes on restart.** Don't bookmark it, don't paste it into a config file. Re-parse from the log each time.
- **Concurrent free tunnels are limited to one per source IP.** Starting a second tunnel from the same machine usually kills the first. Pro tier lifts this.
- **`+` in usernames must be quoted.** Bare `ssh ... b:admin:secret+free@a.pinggy.io` works in bash but breaks under shells that treat `+` specially or when assembled programmatically. Always wrap in double quotes.
- **Don't tunnel anything sensitive without an access-control flag.** A bare HTTP tunnel is reachable by anyone with the URL. Use `b:`, `k:`, or `w:` for non-public services.
- **`process(action='log')` may miss SSH banner output.** Pinggy prints the URLs and then the SSH session goes interactive. Always redirect to a logfile and `grep` the file directly — same pattern as `cloudflared-quick-tunnel`.
- **Host-key prompt on first run.** Default OpenSSH config asks the user to accept Pinggy's host key. Always pass `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null` for unattended runs.
- **TCP and TLS tunnels return a `<subdomain>.a.pinggy.online:<port>` pair, not an https URL.** Parse with a different regex (`tcp://` and a port). Don't assume every Pinggy tunnel is HTTP.
- **Pro mode requires the token as the username, not a flag.** Use `"$PINGGY_TOKEN+a.pinggy.io"` (no `free@`). With a token you can also add `:persistent` for a stable subdomain — see `pinggy.io/docs/`.

## Recipes

Composite patterns combining a local origin with a Pinggy tunnel. Each recipe is self-contained — start the origin, start the tunnel, parse the URL, hand it back to the user.

### Recipe 1 — Receive a webhook callback

Use this when an external service (Stripe, GitHub, Discord, AgentMail, etc.) needs to POST to a publicly reachable URL during a local task.

```bash
# 1. Tiny capturing server: every request gets appended to /tmp/webhook-hits.log
cat >/tmp/webhook-server.py <<'PY'
import http.server, json, datetime, pathlib
LOG = pathlib.Path("/tmp/webhook-hits.log")
class H(http.server.BaseHTTPRequestHandler):
    def _capture(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n).decode("utf-8", "replace") if n else ""
        rec = {"t": datetime.datetime.utcnow().isoformat(), "path": self.path,
               "method": self.command, "headers": dict(self.headers), "body": body}
        with LOG.open("a") as f: f.write(json.dumps(rec) + "\n")
        self.send_response(200); self.send_header("content-type","application/json")
        self.end_headers(); self.wfile.write(b'{"ok":true}\n')
    def do_GET(self): self._capture()
    def do_POST(self): self._capture()
    def log_message(self,*a,**k): pass
http.server.HTTPServer(("127.0.0.1", 18080), H).serve_forever()
PY
nohup python3 /tmp/webhook-server.py >/tmp/webhook-server.log 2>&1 &
echo $! >/tmp/webhook-server.pid

# 2. Tunnel — bearer-token-gate so randos can't pollute the capture log
nohup ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:18080 "k:$(openssl rand -hex 12)+free@a.pinggy.io" \
    >/tmp/webhook-pinggy.log 2>&1 &
echo $! >/tmp/webhook-pinggy.pid
sleep 5
URL=$(grep -oE 'https://[a-z0-9-]+\.[a-z]+\.pinggy\.link' /tmp/webhook-pinggy.log | head -1)
echo "Webhook URL: $URL"

# 3. While the agent works, watch hits land
tail -f /tmp/webhook-hits.log
```

Hand `$URL` to the service that needs to call you. Teardown: `kill $(cat /tmp/webhook-server.pid) $(cat /tmp/webhook-pinggy.pid)`.

### Recipe 2 — Expose an MCP server over HTTP/SSE

Use when a remote MCP client (Claude Desktop on another machine, a teammate's editor, etc.) needs to reach an MCP server running on the local box. Only works for MCP servers that speak HTTP transport — stdio-mode servers can't be tunneled.

```bash
# 1. Start the MCP server in HTTP mode (example: a FastMCP server on port 8765)
nohup python3 my_mcp_server.py --transport http --port 8765 \
    >/tmp/mcp-server.log 2>&1 &
echo $! >/tmp/mcp-server.pid

# 2. Tunnel with a bearer token — MCP traffic should not be open to the internet
TOKEN=$(openssl rand -hex 16)
nohup ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:8765 "k:$TOKEN+free@a.pinggy.io" \
    >/tmp/mcp-pinggy.log 2>&1 &
echo $! >/tmp/mcp-pinggy.pid
sleep 5
URL=$(grep -oE 'https://[a-z0-9-]+\.[a-z]+\.pinggy\.link' /tmp/mcp-pinggy.log | head -1)
echo "MCP URL: $URL"
echo "Bearer token: $TOKEN"
```

The remote client connects to `$URL` with `Authorization: Bearer $TOKEN`. Hermes' own native MCP client config: `{"transport": "http", "url": "<URL>", "headers": {"Authorization": "Bearer <TOKEN>"}}`.

### Recipe 3 — Expose a local LLM endpoint (Ollama / vLLM / llama.cpp)

Share a local model with a remote caller (another agent, a phone, a teammate). Ollama listens on `:11434`, vLLM and llama.cpp typically on `:8000`.

```bash
# Pre-req: the model server is already running on 127.0.0.1:11434 (Ollama default)
TOKEN=$(openssl rand -hex 16)
nohup ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:11434 "k:$TOKEN+co+free@a.pinggy.io" \
    >/tmp/llm-pinggy.log 2>&1 &
echo $! >/tmp/llm-pinggy.pid
sleep 5
URL=$(grep -oE 'https://[a-z0-9-]+\.[a-z]+\.pinggy\.link' /tmp/llm-pinggy.log | head -1)
echo "Endpoint: $URL"
echo "Token:    $TOKEN"

# Verify
curl -s "$URL/api/tags" -H "Authorization: Bearer $TOKEN" | head
```

`co` enables CORS so a browser caller can hit the endpoint. Drop `co` for backend-only callers. For an OpenAI-compatible vLLM/llama.cpp endpoint, callers use base URL `$URL/v1` with `Authorization: Bearer $TOKEN` — but note Pinggy strips/replaces nothing in the body, so the model server itself sees Pinggy's token; the local server should be configured to ignore auth (it's already on `127.0.0.1`) and let Pinggy do the gating.

### Recipe 4 — Share a dev server with a one-shot password

The fastest "let a teammate poke at my running app" pattern. Random password, prints once, dies when you Ctrl-C.

```bash
PASS=$(openssl rand -base64 12 | tr -d '+/=' | head -c 12)
echo "Dev server password: $PASS"
ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:3000 "b:dev:$PASS+co+x:https+free@a.pinggy.io"
# URL prints to the terminal. Share URL + password. Ctrl-C to tear down.
```

`b:dev:$PASS` gates the URL with HTTP Basic auth. `x:https` forces TLS. `co` adds CORS for SPA frontends.

## Verification

```bash
# End-to-end: spin up a trivial origin, tunnel it, hit it, tear down
python3 -m http.server 18000 --bind 127.0.0.1 >/tmp/origin.log 2>&1 &
ORIGIN_PID=$!

nohup ssh -p 443 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -R0:localhost:18000 free@a.pinggy.io >/tmp/pinggy-verify.log 2>&1 &
SSH_PID=$!

sleep 5
URL=$(grep -oE 'https://[a-z0-9-]+\.[a-z]+\.pinggy\.link' /tmp/pinggy-verify.log | head -1)
echo "URL: $URL"
curl -sI "$URL/" | head -1

kill "$SSH_PID" "$ORIGIN_PID"
```

Expected: a `pinggy.link` URL and `HTTP/2 200` on the curl head.
