---
sidebar_position: 17
title: "OAuth over SSH / Remote Hosts"
description: "How to complete browser-based OAuth (Spotify, MCP servers) when Hermes runs on a remote machine, container, or behind a jump box"
---

# OAuth over SSH / Remote Hosts

Some Hermes providers — **Spotify** and **remote MCP servers** (Linear, Sentry, Atlassian, Asana, Figma, …) — use a *loopback redirect* OAuth flow. The auth server redirects your browser to `http://127.0.0.1:<port>/callback` so a tiny HTTP listener started by Hermes can grab the authorization code.

This works perfectly when Hermes and your browser are on the same machine. It breaks the moment they aren't: your laptop's browser tries to reach `127.0.0.1` on **your laptop**, but the listener is bound to `127.0.0.1` on **the remote server**.

The fix is a one-line SSH local-forward. For Spotify and MCP servers on an interactive terminal, you can often paste the redirect URL back instead (no tunnel).

**xAI Grok OAuth (`xai-oauth`) uses OAuth device code**, not a loopback callback — open the printed verification URL in any browser and Hermes polls until approval. No SSH tunnel is required. See [xAI Grok OAuth](./xai-grok-oauth.md).

## TL;DR

```bash
# On your local machine (laptop), in a separate terminal:
ssh -N -L 43827:127.0.0.1:43827 user@remote-host

# In your existing SSH session on the remote machine:
hermes auth add spotify --no-browser
# → Hermes prints an authorize URL. Open it in a browser on your laptop.
# → Your browser redirects to 127.0.0.1:43827/callback, the tunnel forwards
#   the request to the remote listener, login completes.
```

Hermes prints the exact port it bound to on the `Waiting for callback on ...` line — copy it from there. Spotify defaults to port `43827`.

## Which Providers Need This

| Provider | Loopback port | Tunnel needed? |
|----------|---------------|----------------|
| Spotify | `43827` (default) | Yes, when Hermes is remote (or paste redirect URL with `--manual-paste`) |
| MCP servers (`auth: oauth`) | auto-picked per server | Yes, when Hermes is remote (or paste redirect URL) |
| `xai-oauth` (Grok SuperGrok) | n/a | No — device code flow |
| `anthropic` (Claude Pro/Max) | n/a | No — paste-the-code flow |
| `openai-codex` (ChatGPT Plus/Pro) | n/a | No — device code flow |
| `minimax`, `nous-portal` | n/a | No — device code flow |

If your provider isn't in the table, you don't need a tunnel.

## Spotify

You have two ways to complete Spotify login from a remote host:

**Option 1 — paste the callback URL back (no tunnel).** Run `hermes auth spotify --manual-paste`. Hermes prints the authorize URL and skips the SSH-tunnel hint. Open the URL in a browser on your laptop and approve. The browser then tries to load `http://127.0.0.1:43827/spotify/callback`, which fails — that's expected, since the listener is on the remote host, not your laptop. Copy the **full URL from the browser's address bar** of that failed page and paste it at the `Callback URL:` prompt:

```
─── Manual callback paste ─────────────────────────────────────
After approving in your browser, your browser will try to load
  http://127.0.0.1:43827/spotify/callback
which fails (the loopback listener is on this remote machine,
not on your laptop) — that is expected. Copy the FULL URL
from your browser's address bar of that failed page and paste
it below. A bare '?code=...&state=...' fragment also works.
───────────────────────────────────────────────────────────────
Callback URL: http://127.0.0.1:43827/spotify/callback?code=abc&state=xyz
```

A bare `?code=...&state=...` fragment, or even a bare code value with no state at all, is also accepted.

**Option 2 — SSH port forward.** Use the tunnel steps under [Step-by-step: single SSH hop](#step-by-step-single-ssh-hop) below (no `--manual-paste` flag needed).

## MCP Servers

Remote MCP servers (Linear, Sentry, Atlassian, Asana, Figma, etc.) use the same loopback redirect flow. Hermes auto-picks a free port per server and prints the authorize URL when the OAuth flow kicks off — either at startup (when a new server appears in `mcp_servers:`) or when you run `hermes mcp login <server>`.

You have two ways to complete it from a remote host:

**Option 1 — paste the redirect URL back (no setup, works anywhere).** On an interactive terminal, Hermes prompts you to paste the redirect URL alongside running the local listener. After approving in your browser, the redirect to `http://127.0.0.1:<port>/callback` will show a connection error — that's expected. Copy the **full URL from the browser's address bar** and paste it at the Hermes prompt:

```
  MCP OAuth: authorization required.
  Open this URL in your browser:

    https://mcp.linear.app/authorize?response_type=code&...

  Or paste the redirect URL here (or the ?code=...&state=... portion) and press Enter:
> https://mcp.linear.app/callback?code=abc123&state=xyz
  Got authorization code from paste — completing flow.
```

A bare `?code=...&state=...` query string is accepted too. This works for any MCP server with `auth: oauth` and requires no SSH config changes.

**Option 2 — SSH port forward (same as Spotify).** Hermes prints the exact port it bound to in the SSH-session hint. Open a separate terminal on your laptop:

```bash
ssh -N -L <port>:127.0.0.1:<port> user@remote-host
```

Then open the authorize URL in your browser as normal; the redirect tunnels through and the listener picks it up. Use this when you need the flow to complete unattended (e.g. scripted re-auth where you can't paste interactively).

**Pitfall — the 30s config-reload race.** If you edit `~/.hermes/config.yaml` to add an OAuth MCP server from inside a running Hermes session, the CLI auto-reloads MCP connections with a 30s timeout. That's not enough time to complete an interactive OAuth flow, and the reload will give up. Use `hermes mcp login <server>` from a fresh terminal instead — it has no such cap and waits the full 5 min for you to paste back.

## Why the listener can't just bind 0.0.0.0

Spotify and most MCP OAuth servers validate the `redirect_uri` parameter against an allowlist. Both require the loopback form (`http://127.0.0.1:<exact-port>/callback`). Binding the listener to `0.0.0.0` or a different port would cause the auth server to reject the request as a redirect_uri mismatch. The SSH tunnel keeps the loopback URI intact end-to-end.

## Step-by-step: single SSH hop

### 1. Start the tunnel from your local machine

```bash
# Spotify (port 43827)
ssh -N -L 43827:127.0.0.1:43827 user@remote-host
```

`-N` means "don't open a remote shell, just hold the tunnel open." Keep this terminal running for the duration of the login.

### 2. In a separate SSH session, run the auth command

```bash
ssh user@remote-host
hermes auth add spotify --no-browser
```

Hermes detects the SSH session, skips the browser auto-open, and prints an authorize URL plus a `Waiting for callback on http://127.0.0.1:<port>/callback` line.

### 3. Open the URL in your local browser

Copy the authorize URL from the remote terminal and paste it into the browser on your laptop. Approve the consent screen. The auth server redirects to `http://127.0.0.1:<port>/callback`. Your browser hits the tunnel, the request is forwarded to the remote listener, and Hermes prints `Login successful!`.

You can tear down the tunnel (Ctrl+C in the first terminal) once you see the success line.

## Step-by-step: through a jump box

If you reach Hermes through a bastion / jump host, use SSH's built-in `-J` (ProxyJump):

```bash
ssh -N -L 43827:127.0.0.1:43827 -J jump-user@jump-host user@final-host
```

This chains a SSH connection through the jump host without putting the loopback port on the jump box itself. The local `127.0.0.1:43827` on your laptop tunnels straight through to `127.0.0.1:43827` on the final remote host.

For older OpenSSH that doesn't support `-J`, the long form is:

```bash
ssh -N \
    -o "ProxyCommand=ssh -W %h:%p jump-user@jump-host" \
    -L 43827:127.0.0.1:43827 \
    user@final-host
```

## Mosh, tmux, ssh ControlMaster

The tunnel is a property of the underlying SSH connection. If you're running Hermes inside `tmux` over a mosh session, the mosh roaming doesn't carry the `-L` forwarding. Open a *separate* plain SSH session **only** for the `-L` tunnel — that's the connection that has to stay alive during the auth flow. Your interactive mosh/tmux session can keep running Hermes normally.

If you use `ssh -o ControlMaster=auto`, port forwards on a multiplexed connection share the master's lifetime. Restart the master if the tunnel doesn't come up:

```bash
ssh -O exit user@remote-host
ssh -N -L 43827:127.0.0.1:43827 user@remote-host
```

## Troubleshooting

### `bind [127.0.0.1]:43827: Address already in use`

Something on your laptop is already using that port. Either the previous tunnel didn't shut down cleanly, or a local Hermes is also listening on it. Find and kill the offender:

```bash
# macOS / Linux
lsof -iTCP:43827 -sTCP:LISTEN
kill <PID>
```

Then retry the `ssh -L` command.

### Authorization timed out waiting for the local callback

The redirect never made it back to the remote listener. Check the tunnel is still alive (`ssh -N` doesn't show output, so look at the terminal you started it from), confirm you used the port from the latest `Waiting for callback on ...` line (Hermes may auto-bump if the preferred port is busy), restart the tunnel if needed, and re-run the auth command.

### Tokens land in the wrong `~/.hermes`

The tokens are written under the Linux user that ran `hermes auth add ...`. If your gateway / systemd service runs as a different user (e.g. `root` or a dedicated `hermes` user), authenticate as **that** user so the tokens land in their `~/.hermes/auth.json`. `sudo -u hermes -i` or equivalent.

## See Also

- [xAI Grok OAuth](./xai-grok-oauth.md) — device code; no SSH tunnel
- [Spotify (`Running over SSH`)](../user-guide/features/spotify.md#running-over-ssh--in-a-headless-environment)
- [Native MCP client (OAuth section)](../user-guide/features/mcp.md#oauth-authenticated-http-servers)
- [SSH `-J` / ProxyJump (man page)](https://man.openbsd.org/ssh#J)
