---
sidebar_position: 17
title: "OAuth over SSH / Remote Hosts"
description: "How to complete browser-based OAuth (xAI, Spotify) when Hermes runs on a remote machine, container, or behind a jump box"
---

# OAuth over SSH / Remote Hosts

Some Hermes providers — currently **xAI Grok OAuth** and **Spotify** — use a *loopback redirect* OAuth flow. The auth server (xAI, Spotify) redirects your browser to `http://127.0.0.1:<port>/callback` so a tiny HTTP listener started by the `hermes auth ...` command can grab the authorization code.

This works perfectly when Hermes and your browser are on the same machine. It breaks the moment they aren't: your laptop's browser tries to reach `127.0.0.1` on **your laptop**, but the listener is bound to `127.0.0.1` on **the remote server**.

The fix is a one-line SSH local-forward.

## TL;DR

```bash
# On your local machine (laptop), in a separate terminal:
ssh -N -L 56121:127.0.0.1:56121 user@remote-host

# In your existing SSH session on the remote machine:
hermes auth add xai-oauth --no-browser
# → Hermes prints an authorize URL. Open it in a browser on your laptop.
# → Your browser redirects to 127.0.0.1:56121/callback, the tunnel forwards
#   the request to the remote listener, login completes.
```

Port `56121` is what xAI OAuth uses. For Spotify, replace it with `43827`. Hermes prints the exact port it bound to on the `Waiting for callback on ...` line — copy it from there.

## Which Providers Need This

| Provider | Loopback port | Tunnel needed? |
|----------|---------------|----------------|
| `xai-oauth` (Grok SuperGrok) | `56121` | Yes, when Hermes is remote |
| Spotify | `43827` | Yes, when Hermes is remote |
| `anthropic` (Claude Pro/Max) | n/a | No — paste-the-code flow |
| `openai-codex` (ChatGPT Plus/Pro) | n/a | No — device code flow |
| `minimax`, `nous-portal` | n/a | No — device code flow |

If your provider isn't in the table, you don't need a tunnel.

## Why the listener can't just bind 0.0.0.0

xAI and Spotify both validate the `redirect_uri` parameter against an allowlist. Both require the loopback form (`http://127.0.0.1:<exact-port>/callback`). Binding the listener to `0.0.0.0` or a different port would cause the auth server to reject the request as a redirect_uri mismatch. The SSH tunnel keeps the loopback URI intact end-to-end.

## Step-by-step: single SSH hop

### 1. Start the tunnel from your local machine

```bash
# xAI Grok OAuth (port 56121)
ssh -N -L 56121:127.0.0.1:56121 user@remote-host

# Or for Spotify (port 43827)
ssh -N -L 43827:127.0.0.1:43827 user@remote-host
```

`-N` means "don't open a remote shell, just hold the tunnel open." Keep this terminal running for the duration of the login.

### 2. In a separate SSH session, run the auth command

```bash
ssh user@remote-host
hermes auth add xai-oauth --no-browser
# or for Spotify:
# hermes auth add spotify --no-browser
```

Hermes detects the SSH session, skips the browser auto-open, and prints an authorize URL plus a `Waiting for callback on http://127.0.0.1:<port>/callback` line.

### 3. Open the URL in your local browser

Copy the authorize URL from the remote terminal and paste it into the browser on your laptop. Approve the consent screen. The auth server redirects to `http://127.0.0.1:<port>/callback`. Your browser hits the tunnel, the request is forwarded to the remote listener, and Hermes prints `Login successful!`.

You can tear down the tunnel (Ctrl+C in the first terminal) once you see the success line.

## Step-by-step: through a jump box

If you reach Hermes through a bastion / jump host, use SSH's built-in `-J` (ProxyJump):

```bash
ssh -N -L 56121:127.0.0.1:56121 -J jump-user@jump-host user@final-host
```

This chains a SSH connection through the jump host without putting the loopback port on the jump box itself. The local `127.0.0.1:56121` on your laptop tunnels straight through to `127.0.0.1:56121` on the final remote host.

For older OpenSSH that doesn't support `-J`, the long form is:

```bash
ssh -N \
    -o "ProxyCommand=ssh -W %h:%p jump-user@jump-host" \
    -L 56121:127.0.0.1:56121 \
    user@final-host
```

## Mosh, tmux, ssh ControlMaster

The tunnel is a property of the underlying SSH connection. If you're running Hermes inside `tmux` over a mosh session, the mosh roaming doesn't carry the `-L` forwarding. Open a *separate* plain SSH session **only** for the `-L` tunnel — that's the connection that has to stay alive during the auth flow. Your interactive mosh/tmux session can keep running Hermes normally.

If you use `ssh -o ControlMaster=auto`, port forwards on a multiplexed connection share the master's lifetime. Restart the master if the tunnel doesn't come up:

```bash
ssh -O exit user@remote-host
ssh -N -L 56121:127.0.0.1:56121 user@remote-host
```

## Troubleshooting

### `bind [127.0.0.1]:56121: Address already in use`

Something on your laptop is already using that port. Either the previous tunnel didn't shut down cleanly, or a local Hermes is also listening on it. Find and kill the offender:

```bash
# macOS / Linux
lsof -iTCP:56121 -sTCP:LISTEN
kill <PID>
```

Then retry the `ssh -L` command.

### "Could not establish connection. We couldn't reach your app." (xAI)

xAI's authorize page shows this when its redirect to `127.0.0.1:<port>/callback` doesn't reach a listener. Either the tunnel isn't running, the port is wrong, or you're using the port Hermes printed in a previous run (the port can be auto-bumped if the preferred one is busy — always read the latest `Waiting for callback on ...` line).

### `xAI authorization timed out waiting for the local callback`

Same root cause as above — the redirect never made it back. Check the tunnel is still alive (`ssh -N` doesn't show output, so look at the terminal you started it from), restart it if needed, and re-run `hermes auth add xai-oauth --no-browser`.

### Tokens land in the wrong `~/.hermes`

The tokens are written under the Linux user that ran `hermes auth add ...`. If your gateway / systemd service runs as a different user (e.g. `root` or a dedicated `hermes` user), authenticate as **that** user so the tokens land in their `~/.hermes/auth.json`. `sudo -u hermes -i` or equivalent.

## See Also

- [xAI Grok OAuth](./xai-grok-oauth.md)
- [Spotify (`Running over SSH`)](../user-guide/features/spotify.md#running-over-ssh--in-a-headless-environment)
- [SSH `-J` / ProxyJump (man page)](https://man.openbsd.org/ssh#J)
