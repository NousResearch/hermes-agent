---
sidebar_position: 4
title: "Use Hermes from a browser"
description: "Run Hermes on a machine that has the disk space, then reach it from a laptop and phone browser and install it to the home screen"
---

# Use Hermes from a browser

Hermes needs a machine with real disk, a Python toolchain and a place to keep
`~/.hermes`. Your laptop and your phone do not both have to be that machine.
This page covers the split: **one host runs the agent, every other device is a
browser tab**.

That is the answer when the device in front of you cannot take the install —
a laptop with no room on `C:`, a work machine you cannot install to, an iPhone
or iPad, a Chromebook. Nothing is downloaded to those devices; the dashboard is
a web app served by the host, and it can be installed to the home screen so it
opens in its own window with no browser chrome.

:::info What you need first
A host that already runs Hermes. Any of these works: a cloud VM, a home server
or NAS, a Raspberry Pi, a spare desktop, or a second drive on the same laptop.
Set it up with the [Quickstart](./quickstart.md), or run the container from the
[Docker guide](../user-guide/docker.md) — Docker is usually the least work,
since the image carries the whole toolchain.
:::

## The shape of it

```
  phone browser  ─┐
                  ├──►  https://your-host  ──►  hermes dashboard  ──►  agent
  laptop browser ─┘         (login)              (the host machine)
```

The dashboard is the full product surface, not a cut-down remote: it embeds the
real `hermes --tui` over a terminal WebSocket, so chat, slash commands,
sessions, files, config and API keys all work exactly as they do locally. See
[Web Dashboard](../user-guide/features/web-dashboard.md) for the tour.

## 1. Bind the dashboard where other devices can reach it

On the host:

```sh
hermes dashboard --host 0.0.0.0 --port 9119 --no-open
```

Binding anything other than loopback turns on the **auth gate**, and the
dashboard refuses to start until you have configured a login. That refusal is
the feature — the dashboard reads and writes your API keys, and an
internet-facing one with no login is how people get their agent hijacked. Run
the command in a terminal and Hermes offers to set up a username and password
on the spot.

To configure it ahead of time instead, in `config.yaml`:

```yaml
dashboard:
  basic_auth:
    username: "you"
    password_hash: "<paste the hash>"
```

Generate the hash on the host:

```sh
python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('your-password'))"
```

Under Docker the same thing is environment variables — `HERMES_DASHBOARD=1`,
`HERMES_DASHBOARD_BASIC_AUTH_USERNAME` and
`HERMES_DASHBOARD_BASIC_AUTH_PASSWORD`. The
[dashboard section of the Docker guide](../user-guide/docker.md#running-the-dashboard)
has the full run command.

Keep the process alive across logout and reboot with systemd, `tmux`, or
`--restart unless-stopped` if you are running the container.

## 2. Decide how the other devices reach the host

Pick one. They are listed cheapest-first, not best-first.

| Reach | Good for | Cost |
|---|---|---|
| Same LAN, `http://<host-ip>:9119` | A host on your own network | Free, but no HTTPS |
| [Tailscale](https://tailscale.com/) or another VPN | Phone on mobile data, host at home | Free tier, HTTPS via Tailscale Serve |
| Reverse proxy with a real certificate | A host with a domain name | A domain and a proxy to run |

Username/password over plain HTTP sends that password in the clear, so treat
LAN-only as a trusted-network convenience and use a VPN or a certificate for
anything else. For a public deployment prefer OAuth over a password — see
[Authentication](../user-guide/features/web-dashboard.md#authentication-gated-mode).

If a reverse proxy terminates TLS, tell Hermes the address the browser actually
uses, so login redirects and asset URLs are built against it:

```yaml
dashboard:
  public_url: "https://hermes.example.com"
  trusted_proxies:
    - "10.0.0.7"   # the proxy's exact address, never a wildcard
```

The dashboard also honours `X-Forwarded-Prefix`, so mounting it at a subpath
like `https://example.com/hermes` works without rebuilding the frontend.

## 3. Install it to the home screen

Over HTTPS the dashboard is a progressive web app: it declares an app manifest
and registers a service worker, so a browser will offer to install it. Once
installed it launches from the home screen in its own window, with no address
bar, its own icon and its own task-switcher entry.

- **iPhone / iPad (Safari)** — Share → **Add to Home Screen**. Safari only
  offers this in Safari itself, not in Chrome or Firefox on iOS.
- **Android (Chrome)** — the **Install app** prompt, or ⋮ → **Add to Home
  screen**.
- **Desktop Chrome or Edge** — the install icon at the right of the address
  bar, or ⋮ → **Cast, save, and share** → **Install page as app**.

Long-pressing the installed icon on Android gives shortcuts straight to **Chat**
and **Sessions**.

:::note Installation requires a secure origin
Browsers only install a page served over HTTPS (or from `localhost`). On a
plain-HTTP LAN address the dashboard works normally in a tab, but no install
prompt appears. This is a browser rule, not a Hermes setting — put the
dashboard behind Tailscale Serve or a certificate to get the installable
version.
:::

The service worker caches the bundle's static assets so repeat loads over a
phone network are fast. It never caches `/api`, so no agent state, session
token or secret is written to browser storage — and it means the installed app
is **not** usable offline. Hermes is a live agent on the host; opening the app
with the host unreachable shows a short "dashboard unreachable" notice instead
of a stale screen.

## Using it on a phone

The dashboard is responsive, and the embedded terminal is wired for touch
keyboards — the layout resizes around the on-screen keyboard rather than being
covered by it. Installed to the home screen, the layout also insets itself
around the notch and the home indicator.

Two things are worth knowing:

- **The terminal needs a PTY**, so the host has to be Linux, macOS or WSL.
  Native Windows hosts serve the rest of the dashboard but not the embedded
  TUI.
- **Sessions live on the host**, so a conversation you start on the laptop is
  the same conversation when you pick up the phone. There is no syncing step.

## Troubleshooting

**The dashboard refuses to start.** Read the error — on a non-loopback bind it
names the exact missing setting. It is telling you no auth provider is
configured.

**The page loads but chat never connects.** The status probe hits a public
endpoint, so a loading page proves less than it looks. The terminal is a
separate WebSocket and needs the login to have succeeded and the `Host` header
to match what the dashboard bound to. See
[connecting to a remote backend](../user-guide/features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend).

**No install prompt.** Almost always a non-HTTPS origin. Confirm the address
bar shows `https://`, then reload once — the manifest and worker are picked up
on load.

**The installed app looks stale after upgrading Hermes.** Close and reopen it.
The worker only caches content-hashed asset files, so a new build changes their
names and the old ones are dropped; the page itself is never served from cache.
