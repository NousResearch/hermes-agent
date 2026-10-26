---
title: Browser Automation
description: Drive a real browser with browser_exec — Hermes-launched Chromium, cloud browsers, Lightpanda, or your own Chrome via CDP — for web interaction, form filling, scraping, and more.
sidebar_label: Browser
sidebar_position: 5
---

# Browser Automation

The `browser` toolset gives the agent a real web browser. One tool does the
driving — **`browser_exec`**, which runs model-written Python against the
browser over the Chrome DevTools Protocol (CDP) — with three companion tools
(`browser_cdp` and `browser_dialog` only while a CDP endpoint is configured, see
[Companion tools](#companion-tools)):

| Tool | Purpose |
|------|---------|
| `browser_exec` | Navigate, read, click, type, scrape and screenshot by running Python with pre-imported browser helpers. |
| `browser_cdp` | Send one raw CDP command (cookies, network control, iframe-scoped evaluation, anything the helpers don't cover). |
| `browser_dialog` | Accept or dismiss a native JavaScript dialog (`alert` / `confirm` / `prompt` / `beforeunload`) blocking the page. |
| `browser_vault` | Fill saved logins, cards, addresses and 2FA codes into the current page without the secrets entering the conversation. |

Where the browser comes from is a separate choice:

- **Local Chromium (default)** — Hermes launches its own packaged Chromium, headless, one browser per session. No API key, no setup.
- **Cloud browsers** — the [Nous subscription](tool-gateway.md), [Browserbase](https://browserbase.com), [Browser Use](https://browser-use.com) cloud, [Firecrawl](https://firecrawl.dev), or a browser-provider plugin.
- **[Lightpanda](https://lightpanda.io)** — a lightweight headless engine for text-first browsing on small machines.
- **Your own browser** — attach to a running Chrome, Brave, Chromium or Edge with `/browser connect`.
- **Your real profile** — a managed copy of your everyday browser profile, logins included (opt-in).

:::tip
For simple information retrieval, prefer `web_search` or `web_extract` — they are faster and cheaper. Use the browser when the agent needs to **interact** with a page (click buttons, fill forms, handle dynamic content).
:::

## How `browser_exec` works

Each call sends a block of Python (`code`) that runs with browser helpers
already imported; whatever the code prints comes back as the result. The
browser session persists across calls (open tabs, cookies, the current page),
Python variables do not — every call starts a fresh interpreter. A per-session
workspace directory (`$BH_AGENT_WORKSPACE`, also returned as `workspace`)
survives between calls, so long extractions can append batches to a file and
aggregate them at the end.

The helpers:

| Helper | Does |
|--------|------|
| `new_tab(url)` | Open a URL (use for the first navigation). |
| `goto_url(url)` | Navigate the current tab (later navigations). |
| `wait_for_load()` | Wait for the page to finish loading. |
| `page_info()` | Summarize the current page state. |
| `js(expr)` | Evaluate JavaScript and return the value (wrap function bodies as `js('(() => {...})()')`). |
| `fill_input(selector, text)` | Type into an input. |
| `click_at_xy(x, y)` | Click viewport coordinates. |
| `capture_screenshot()` | Save a screenshot and print its path. |
| `cdp('Domain.method', **kwargs)` | Raw CDP — e.g. `cdp('Accessibility.getFullAXTree')` for the accessibility tree, `cdp('DOM.getBoxModel', backendNodeId=n)` for click coordinates. |
| `ensure_real_tab()` | Recover from a stale or internal tab. |

The model batches a whole sub-procedure (navigate, wait, extract, act) into one
call instead of spending a call per action:

```python
# Searching GitHub trending for Python repos
new_tab("https://github.com/trending/python")
wait_for_load()
print(js("[...document.querySelectorAll('article h2 a')].slice(0, 10).map(a => a.textContent.trim())"))
```

**Screenshots.** On a model with native vision, an image saved by
`capture_screenshot()` is attached to the tool result and the model inspects it
directly. Text-only models are told to work text-first (`page_info()`, `js()`,
`fill_input()`); they can still hand a saved path to `vision_analyze`. On
messaging platforms you can ask the agent to send you a screenshot — it arrives
as a native photo attachment.

**Parameters.** Besides `code`, the tool takes `timeout_s` (default 300, max
1800) and `session`. Each `session` name gets its own driver process and its
own browser (a separate Chromium locally, a separate cloud browser on cloud
providers), so parallel subagents or simultaneous chats don't clobber one
shared browser. Omitting `session` uses the shared default session, which is
fine for one-at-a-time browsing. When [real-profile browsing](#real-profile-browsing-use-your-own-logins)
is on, the tool also exposes a `local` flag (see that section).

:::note
`browser_exec` executes model-written Python on your machine, so it is only
offered to sessions that also have the `terminal` toolset. A platform
configured without terminal access (for example a locked-down messaging
surface) gets no browser.
:::

The driver is [browser-harness](https://github.com/browser-use/browser-use), the
engine of the Browser Use CLI. It is a regular Python dependency of Hermes, so
every install — the Desktop app included — ships it with no separate download.

## Local Chromium (default)

With no cloud provider, `/browser connect` endpoint or real-profile toggle
configured, Hermes starts a Chromium for the session itself and points
`browser_exec` at it. Your installed Chrome is never touched — there is no
`chrome://inspect` toggle to enable and no "Allow remote debugging?" popup —
and it works on headless servers with no desktop browser at all.

- **One browser per session**, with a fresh, throwaway profile by default: the
  agent starts logged into nothing, and nothing persists after the session.
- **Headless by default.** Set `browser.headed: true` for a visible window (see
  [below](#headed-mode-visible-browser-window)).
- **Automatic cleanup.** The browser is closed after `browser.inactivity_timeout`
  seconds idle (default 120), when Hermes exits, and by an orphan sweep if a
  process crashed.

**Which Chromium.** Hermes picks the first that exists:

1. `browser.executable_path`, when set.
2. The Chromium (Chrome for Testing) installed by Hermes's package manager — the
   installers and `hermes update` install it by default.
3. `/etc/hermes/chromium-executable-path` in the official Docker image.
4. A system `chromium` / `chromium-browser` on `PATH` (Android/Termux, and
   platforms the package manager has no Chromium build for).

If none is found, the browser tool reports a clear hint instead of failing
silently. If you installed with `--skip-browser` / `-SkipBrowser`, or the
download failed, install it with:

```bash
hermes pm install chromium
```

This also undoes an earlier `--skip-browser` choice, so updates keep the
browser current. On Linux, Chromium may also need distribution system
libraries (`npx playwright install-deps chromium` installs them on
Debian/Ubuntu). On Android/Termux, run `pkg install chromium` where your
repository provides it; without a system Chromium, browsing is unavailable on
that device.

**Local options** (`~/.hermes/config.yaml`):

```yaml
browser:
  executable_path: ""     # Chromium binary to launch instead of the packaged one
  user_data_dir: ""       # persistent profile directory; "" = fresh profile per session
  chromium_args: []       # extra Chromium command-line flags, e.g. ["--lang=de-DE"]
  headed: false           # true = visible window
  command_timeout: 30     # Chromium launch timeout floor (seconds)
  inactivity_timeout: 120 # close an idle browser after N seconds
```

`browser.user_data_dir` keeps cookies and logins across sessions: every session
then shares that one profile directory. Leave it empty for isolated, throwaway
browsing.

### Headed mode (visible browser window)

```yaml
browser:
  headed: true  # default: false
```

Headed mode does two things:

1. **Launches Chromium with a visible window**, so you can watch the agent work
   and step in manually (sign-in challenges, CAPTCHAs).
2. **Keeps the browser open between turns.** Normally the browser is cleaned up
   after every agent reply; in headed mode the per-turn cleanup is skipped so
   login state stays warm across the conversation.

Idle browsers are still reaped after `browser.inactivity_timeout`, and all of
them close on shutdown. On a display-less host (servers, CI) the browser always
runs headless. Headed mode only affects local browsers — cloud sessions are
unaffected.

## Cloud browsers

:::tip Nous Subscribers
If you have a paid [Nous Portal](https://portal.nousresearch.com) subscription, you can use a cloud browser through the **[Tool Gateway](tool-gateway.md)** without any separate API keys. New installs can run `hermes setup --portal` to log in and turn on every gateway tool at once; existing installs can pick **Nous Subscription** as the browser provider via `hermes model` or `hermes tools`.
:::

Pick a provider in `hermes tools` → Browser Automation. `browser_exec` then
drives that provider's browser exactly as it drives local Chromium.

### Browser Use cloud

```bash
# Add to ~/.hermes/.env
BROWSER_USE_API_KEY=***
```

Get your API key at [browser-use.com](https://browser-use.com). Browser Use Cloud runs managed Chromium with [stealth](https://docs.browser-use.com/cloud/browser/stealth) and [residential proxies](https://docs.browser-use.com/cloud/browser/proxies) enabled by default, includes CAPTCHA solving, and supports [persistent profiles](https://docs.browser-use.com/cloud/guides/authentication) for cookies, localStorage, and saved passwords.

### Browserbase

```bash
# Add to ~/.hermes/.env
BROWSERBASE_API_KEY=***
BROWSERBASE_PROJECT_ID=your-project-id-here
```

Get your credentials at [browserbase.com](https://browserbase.com). Optional settings:

```bash
# Residential proxies for better CAPTCHA solving (default: "true")
BROWSERBASE_PROXIES=true

# Advanced stealth with custom Chromium — requires Scale Plan (default: "false")
BROWSERBASE_ADVANCED_STEALTH=false

# Session reconnection after disconnects — requires paid plan (default: "true")
BROWSERBASE_KEEP_ALIVE=true

# Custom session timeout in seconds (max 21600 = 6 hours) (default: project default)
BROWSERBASE_SESSION_TIMEOUT=1800
```

Browserbase stealth features:

| Feature | Default | Notes |
|---------|---------|-------|
| Basic Stealth | Always on | Random fingerprints, viewport randomization, CAPTCHA solving |
| Residential Proxies | On | Routes through residential IPs for better access |
| Advanced Stealth | Off | Custom Chromium build, requires Scale Plan |
| Keep Alive | On | Session reconnection after network hiccups |

If paid features aren't available on your plan, Hermes automatically falls back — first disabling `keepAlive`, then proxies — so browsing still works on free plans.

### Firecrawl

```bash
# Add to ~/.hermes/.env
FIRECRAWL_API_KEY=fc-***

# Optional: self-hosted Firecrawl instance (default: https://api.firecrawl.dev)
FIRECRAWL_API_URL=http://localhost:3002

# Optional: session TTL in seconds (default: 300)
FIRECRAWL_BROWSER_TTL=600
```

Get your API key at [firecrawl.dev](https://firecrawl.dev), then select **Firecrawl** in `hermes tools` → Browser Automation.

:::note Selecting the provider
The `.env` keys above supply **credentials only**. The active browser is chosen by the `browser.cloud_provider` selection written by `hermes tools` → Browser Automation (`local`, `browserbase`, `browser-use`, `firecrawl`, or `nous` for the Nous Subscription; provider plugins add their own names). Once a selection exists, adding or removing a key does not switch providers — and a selected provider with a missing key errors with guidance to run `hermes tools` instead of silently rerouting. Never-configured setups still autodetect from available credentials.
:::

Cloud sessions consume provider credits. Hermes releases them when the
conversation ends, after `browser.inactivity_timeout`, and on process exit.

### Hybrid routing: cloud for public URLs, local for LAN/localhost

When a cloud provider is configured, Hermes auto-spawns a **local Chromium sidecar**
for URLs that resolve to a private/loopback/LAN address (`localhost`, `127.0.0.1`,
`192.168.x.x`, `10.x.x.x`, `172.16-31.x.x`, `*.local`, `*.lan`, `*.internal`,
IPv6 loopback `::1`, link-local `169.254.x.x`). Public URLs continue to use the
cloud provider in the same conversation.

This solves the common "I'm developing locally but using Browserbase" workflow —
the agent can screenshot your dashboard at `http://localhost:3000` AND scrape
`https://github.com` without you switching providers or disabling the SSRF guard.
The cloud provider never sees the private URL.

The feature is **on by default**. To disable it (all URLs go to the configured
cloud provider):

```yaml
# ~/.hermes/config.yaml
browser:
  cloud_provider: browserbase
  auto_local_for_private_urls: false
```

With auto-routing disabled, private URLs are rejected with
`"Blocked: URL targets a private or internal address"` unless you also set
`browser.allow_private_urls: true` (which lets the cloud provider attempt them —
usually won't work since Browserbase etc. can't reach your LAN).

The sidecar is the same local Chromium described [above](#local-chromium-default),
so it needs that Chromium installed. Post-navigation redirects from a public URL
onto a private address are still blocked (you can't use a redirect-to-internal
trick to reach your LAN through the public path).

## Real profile browsing (use your own logins)

By default, local browsing runs in a clean, throwaway profile — the agent is
logged into nothing. Turn on **real profile browsing** to let the agent browse
as *you*, with your existing logins and cookies:

```yaml
# ~/.hermes/config.yaml
browser:
  use_real_profile: true
```

When enabled, Hermes copies your default browser's **active** profile — the one
you actually browse (`Local State → profile.last_used`), with its cookies, saved
logins, and preferences — into a managed snapshot under
`~/.hermes/browser-profile/<browser>/`, then launches your **real browser
binary** on that snapshot and attaches its browsing engine to it. Launching the
real binary (instead of a bundled Chromium with mock-keychain switches) is what
keeps OS-encrypted cookies decryptable — on macOS, Chrome cookies are encrypted
through the Keychain, and a mock-keychain launch would silently drop every one
of them, opening signed out. Your live browser profile is **never opened
directly**: the
snapshot is a separate directory, so it doesn't fight your running browser for
the profile lock and it sidesteps Chrome 136+'s block on remote-debugging the
default profile directory. The auth files (cookies/logins/preferences) are
re-synced from your real profile whenever a fresh session is launched, so logins
you do in your own browser show up in the agent's session. Only the active
profile is copied — other Chrome profiles are never snapshotted.

The snapshot browser runs **headless** — it drives your profile in the
background with no visible window and never steals focus, so you can keep
working while the agent tweets, fills forms, or scrapes on your behalf.
(Headless here uses Chrome's *new* headless mode, which reads your normal
cookie store, so your logins still load.) If you'd rather watch it work, the
same [headed-mode](#headed-mode-visible-browser-window) toggle applies —
`browser.headed: true` opens a visible window for real-profile browsing too. On a display-less host (servers, CI) it always runs
headless regardless.

If your browser has several profiles (say a work profile and a personal one)
and you don't want "whichever profile you touched last" deciding the agent's
identity, pin the snapshot source explicitly:

```yaml
# ~/.hermes/config.yaml
browser:
  use_real_profile: true
  real_profile_pin: "Profile 2"   # directory name under the browser's user-data dir
```

A pin naming a profile directory that doesn't exist never silently falls back to
the last-used profile: the agent gets a fresh browser without your logins and a
message naming the bad pin.

When you turn the toggle back off, Hermes deletes the snapshot store
(`~/.hermes/browser-profile/`) on the next browser use, so the copied
credentials don't linger after you revoke consent.

:::note When your profile can't be copied, the browser still works
Real-profile browsing never takes the browser away. If Hermes can't copy your
profile, the agent gets a fresh browser **without your logins** for that session,
and its first browser result says why (`real_profile_unavailable`), so it can tell
you instead of acting as if it were signed in. That happens when:

- **Windows, first use only:** a running Chrome/Edge/Brave holds its cookie
  database with an exclusive (deny-all) lock, so the first copy needs the browser
  **fully quit**, including any background/tray instance (Chrome's "continue
  running background apps when closed" keeps a `chrome.exe` alive after you close
  the window). After one successful copy, a later launch with the browser open
  reuses that copy instead: your logins are as of the last copy, and they refresh
  the next time Hermes starts its browser while yours is closed.
- **Your default browser isn't a supported Chromium** (Firefox, a Beta/Dev/Canary
  channel), or the pinned profile doesn't exist.
- **The cookie database is unreadable or corrupt**: the message names it and the
  SQLite error.

On macOS and Linux a running Chrome holds `Login Data`, `Login Data For Account`
and `Web Data` (saved passwords and autofill) with a write lock, but `Cookies`
still copies, and cookies are what keep you signed in to sites. Hermes copies the
cookies and skips the locked password/autofill files, so you can keep your browser
open. Each database is copied through SQLite's online backup (never a raw file
copy, which could lose recent logins).

Set `browser.real_profile_autoclose: true` to let Hermes **offer to close the
browser for you** when it's holding the profile lock (the Windows case). Even with
this on, Hermes never closes it automatically: the agent asks you first, and only on
your approval runs `hermes browser close-profile` (terminates the browser process
tree bound to that profile, losing unsaved tabs). A new browser session then picks
up your profile.
:::

- **Supported browsers:** Chrome, Edge, Brave, Brave Origin, Chromium (whichever is your OS
  default). A non-Chromium default (e.g. Firefox) gets a fresh browser and a clear
  message rather than a guess at which profile to use.
- **Works with any browser source.** Locally it's automatic once the toggle
  is on. Under a **cloud** browser provider, the agent can still open a
  real-profile local session on demand via the `browser_exec` tool's `local`
  argument (the tool only exposes that argument when this toggle is on) — the
  cloud provider keeps serving everything else.
- **Security framing:** this is a consent-gated convenience, not an isolation
  boundary. A page the agent visits runs with your real logins, so only enable
  it when you want the agent acting as you. Off by default.
- **Desktop:** toggle it in **Capabilities → Tools → Browser → Use My Real
  Browser Profile** (the switch sits above the backend options), or in
  Settings → Config under the `browser` section.

### Scheduled and unattended runs

A real-profile session lets a `cronjob` drive a site you're already signed into.
The snapshot runs headless by default, and the auth files (`Cookies`,
`Login Data`, …) are re-synced from your real profile on every fresh session —
an already-open session is reused as-is, so the sync happens when a new one is
launched. Without saved vault credentials, an expired session on that site
surfaces as a login page that the unattended tick reports rather than hanging on.

Three things to set up before scheduling one:

- **Turn the toggle on.** `browser.use_real_profile` defaults to `false`, and
  without it the job gets a clean, unauthenticated profile — see above.
- **Give the job the browser toolset:**
  `cronjob(action="create", enabled_toolsets=["browser", ...], ...)` — a per-job
  list wins over the cron-platform config
  ([details](./cron.md#toolsets-available-to-cron-jobs)).
- **Assume nothing can be prompted for.** A cron, webhook, API or
  `hermes chat -q` session has nobody to answer a prompt, so a site that is not
  usable on the synced cookies alone — a login form, a fresh 2FA challenge —
  needs its credentials saved ahead of time, authenticator key included. That is
  the [credential vault's](./credential-vault.md#headless-sessions) job, and it
  is what carries a run after your own session has expired.

**Windows prerequisite:** the browser has to be fully quit for the first snapshot.
After that, a tick that runs while it is open reuses the last snapshot (logins as of
that copy); with no snapshot yet, the tick runs on a fresh, signed-out browser. Hermes never closes it without asking you first
(see the note above).

The bundled `product-price-monitor` skill is a worked example of the recurring
shape — a JSON watch contract written during setup, then one scheduled tick that
re-reads it, compares, and alerts on change — though it covers the scheduling
half, not logins.

## Lightpanda local engine

[Lightpanda](https://lightpanda.io) is an open-source headless browser written from scratch. It starts instantly, runs 9x faster and uses 16x less memory than Chrome, which matters for agents that live on small VMs for long stretches.

Lightpanda is a **local engine** (a browser source, like local Chromium), not a cloud provider. Install the binary and put it on your `PATH` (see the [Lightpanda installation guide](https://lightpanda.io/docs/run-locally/installation/one-liner)), then pick **Lightpanda** in `hermes tools` → Browser Automation, or set:

```yaml
# Add to ~/.hermes/config.yaml
browser:
  cloud_provider: local
  engine: lightpanda   # auto (default, Chromium) | lightpanda
```

Hermes launches `lightpanda serve --host 127.0.0.1 --port <free>` itself — one process per `browser_exec` session name (or per task) — and points `browser_exec` at it. No Chromium or Node.js is needed. The process is reaped after `browser.inactivity_timeout`, on exit, and by the orphan sweep if Hermes crashes. All of these processes share one on-disk HTTP cache at `$HERMES_HOME/cache/browser-use/lightpanda/http-cache`, so repeat visits skip re-downloading assets. Hermes passes the cache flag only when the installed Lightpanda supports it (0.3.x+); older binaries simply run without a cache. To clear it, stop your Lightpanda sessions first, then delete that directory.

Lightpanda has no graphical renderer, so `capture_screenshot()` is unavailable and the tool description tells the model to work text-first; it also holds one page per session, so the model is told to call `new_tab()` once and `goto_url()` afterwards (tracked upstream in [lightpanda-io/browser#1962](https://github.com/lightpanda-io/browser/issues/1962)).

**When the engine is ignored.** `browser.engine` is the lowest-precedence browser setting: a cloud provider (including the Nous subscription browser — and on never-configured setups, any `BROWSERBASE_API_KEY` / `BROWSER_USE_API_KEY` in `~/.hermes/.env` auto-selects one), a `browser.cdp_url` / `/browser connect` override, or `browser.use_real_profile` all take precedence. Picking Lightpanda in `hermes tools` writes `cloud_provider: local` for you; `/browser status` and `hermes doctor` report when the engine is configured but shadowed, and by what.

## Your own browser via CDP (`/browser connect`)

Instead of the Chromium Hermes launches or a cloud provider, you can attach the browser tools to your own running Chrome, Brave, Chromium, or Edge instance via the Chrome DevTools Protocol (CDP). This is useful when you want to see what the agent is doing in real-time, interact with pages that require your own cookies/sessions, or avoid cloud browser costs.

:::note
`/browser connect` is an **interactive-CLI slash command** — it is not dispatched by the gateway. If you try to run it inside a WebUI, Telegram, Discord, or other gateway chat, the message will be sent to the agent as plain text and the command will not execute. Start Hermes from the terminal (`hermes` or `hermes chat`) and issue `/browser connect` there.
:::

In the CLI, use:

```
/browser connect                 # Auto-launch/connect to a local Chromium-family browser at http://127.0.0.1:9222
/browser connect ws://host:port  # Connect to a specific CDP endpoint
/browser status                  # Check current connection
/browser disconnect              # Detach and return to cloud/local mode
```

If a browser isn't already running with remote debugging, Hermes will attempt to auto-launch a supported Chromium-family browser with `--remote-debugging-port=9222`. Detection includes Brave, Brave Origin/Nightly, Google Chrome, Chromium, and Microsoft Edge, with common Linux install paths and binary names such as `brave-origin`, `brave-origin-nightly`, `/opt/brave.com/brave-origin/brave-origin`, `/opt/brave.com/brave-origin-nightly/brave-origin`, `/opt/brave-bin/brave`, and `/snap/bin/brave`.

:::tip
To start a Chromium-family browser manually with CDP enabled, use a dedicated user-data-dir so the debug port actually comes up even if the browser is already running with your normal profile:

```bash
# Linux — Brave
brave-browser \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.hermes/chrome-debug \
  --no-first-run \
  --no-default-browser-check &

# Linux — Google Chrome
google-chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.hermes/chrome-debug \
  --no-first-run \
  --no-default-browser-check &

# macOS — Brave
"/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.hermes/chrome-debug" \
  --no-first-run \
  --no-default-browser-check &

# macOS — Google Chrome
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.hermes/chrome-debug" \
  --no-first-run \
  --no-default-browser-check &
```

Then launch the Hermes CLI and run `/browser connect`.

**Why `--user-data-dir`?** Without it, launching a Chromium-family browser while a regular instance is already running typically opens a new window on the existing process — and that existing process was not started with `--remote-debugging-port`, so port 9222 never opens. A dedicated user-data-dir forces a fresh browser process where the debug port actually listens. `--no-first-run --no-default-browser-check` skips the first-launch wizard for the fresh profile.

**Chrome 136+ makes the dedicated profile mandatory.** Two separate mechanisms are in play, and neither applies once you pass a non-default `--user-data-dir`. First, since [Chrome 136](https://developer.chrome.com/blog/remote-debugging-port) `--remote-debugging-port` and `--remote-debugging-pipe` "will no longer be respected if attempting to debug the default Chrome data directory" — the flag is silently ignored, no dialog, and `/browser connect` (or `curl http://127.0.0.1:9222/json/version`) gets connection refused even from a cold start. Second, [Chrome 144+](https://developer.chrome.com/blog/chrome-devtools-mcp-debug-your-browser-session) adds an opt-in *approval* flow for debugging your real profile: you enable it under `chrome://inspect/#remote-debugging`, and Chrome then shows an **"Allow remote debugging?"** dialog for **every incoming connection** (not once per launch), with nothing listening until you press **Allow**. If you see that dialog, you are on the approval path, not the flag path. The fix is exactly the commands above: point `--user-data-dir` somewhere other than your default profile directory (e.g. `$HOME/.hermes/chrome-debug`), which needs neither the toggle nor the dialog. This applies to Chrome, Chromium, Edge, and Brave builds that have picked up the change.

A dedicated profile starts out signed out of everything. If you want the agent to browse with your existing logins *and* no approval dialog, use [`browser.use_real_profile`](#real-profile-browsing-use-your-own-logins) instead: it snapshots your active profile into a copy and drives that, which is a non-default user-data-dir and so never triggers either mechanism.
:::

When connected, `browser_exec`, `browser_cdp`, `browser_dialog` and `browser_vault` all drive your live browser instead of launching their own. The endpoint can also be set persistently with `browser.cdp_url` in `config.yaml` or the `BROWSER_CDP_URL` environment variable.

### WSL2 + Windows Chrome: prefer MCP over `/browser connect`

If Hermes runs inside WSL2 but the Chrome window you want to control runs on the Windows host, `/browser connect` is often not the best path.

Why:

- `/browser connect` expects Hermes itself to reach a usable CDP endpoint
- modern Chrome live-debugging sessions often expose a host-local endpoint that is not directly reachable from WSL the same way a classic `9222` port is
- even when Windows Chrome is debuggable, the cleanest integration is often to let a Windows-side browser MCP server attach to Chrome and let Hermes talk to that MCP server

For that setup, prefer `chrome-devtools-mcp` through Hermes MCP support.

See the MCP guide for the practical setup:

- [Use MCP with Hermes](../../guides/use-mcp-with-hermes.md#wsl2-bridge-hermes-in-wsl-to-windows-chrome)

## Bot Desktop screen

On a gateway with the [Bot Desktop screen](bot-screen.md) running, the bot's
browser and the dock's **Browser** icon are the same headed Chromium with one
persistent profile per bot, so you can take over, sign in to a site, and hand
the session back to the agent. See that page for the profile location and how
`browser.executable_path` / `browser.user_data_dir` apply there.

## Companion tools

`browser_vault` operates on the same browser session `browser_exec` drives —
local Chromium, a cloud browser, or the browser attached with `/browser connect`.
`browser_cdp` and `browser_dialog` are offered only while a CDP endpoint is
configured — `/browser connect`, `BROWSER_CDP_URL` or `browser.cdp_url` — and act
on that browser; a launched local or cloud session does not bring them.

### `browser_cdp`

Raw Chrome DevTools Protocol passthrough — the escape hatch for one-off
operations: native dialog handling, iframe-scoped evaluation, cookie/network
control, or any CDP verb the agent needs. (Inside `browser_exec`, the `cdp()`
helper covers the same ground for multi-step work.)

**CDP method reference:** https://chromedevtools.github.io/devtools-protocol/ — the agent can `web_extract` a specific method's page to look up parameters and return shape.

Common patterns:

```
# List tabs (browser-level, no target_id)
browser_cdp(method="Target.getTargets")

# Handle a native JS dialog on a tab
browser_cdp(method="Page.handleJavaScriptDialog",
            params={"accept": true, "promptText": ""},
            target_id="<tabId>")

# Evaluate JS in a specific tab
browser_cdp(method="Runtime.evaluate",
            params={"expression": "document.title", "returnByValue": true},
            target_id="<tabId>")

# Get all cookies
browser_cdp(method="Network.getAllCookies")
```

Browser-level methods (`Target.*`, `Browser.*`, `Storage.*`) omit `target_id`. Page-level methods (`Page.*`, `Runtime.*`, `DOM.*`, `Emulation.*`) require a `target_id` from `Target.getTargets`. Each stateless call is independent — sessions do not persist between calls.

**Cross-origin iframes:** pass the `frame_id` of an out-of-process (cross-origin) iframe to route the call through the CDP supervisor's live session for that frame. This is how `Runtime.evaluate` inside a cross-origin iframe works on Browserbase, where stateless CDP connections would hit signed-URL expiry. Same-origin iframes don't need `frame_id` — use `document.querySelector('iframe').contentDocument` from a top-level `Runtime.evaluate` instead.

### `browser_dialog`

Responds to a native JS dialog (`alert` / `confirm` / `prompt` / `beforeunload`). An open dialog blocks the page's JavaScript thread, so browser work on that page stalls until it is answered.

A persistent **CDP supervisor** — one WebSocket per task subscribed to Page/Runtime/Target events — detects dialogs as they open. The agent answers with `browser_dialog(action="accept")` or `browser_dialog(action="dismiss")`; for `prompt()` dialogs it passes `prompt_text="..."`, and when more than one dialog is queued it passes the `dialog_id` to pick one. The page's JS thread then resumes.

**How it works on Browserbase.** Browserbase's CDP proxy auto-dismisses real native dialogs server-side within ~10ms, so `Page.handleJavaScriptDialog` can't be used. The supervisor injects a small script via `Page.addScriptToEvaluateOnNewDocument` that overrides `window.alert`/`confirm`/`prompt` with a synchronous XHR, and intercepts those XHRs via `Fetch.enable` — the page's JS thread stays blocked on the XHR until the supervisor calls `Fetch.fulfillRequest` with the agent's response. `prompt()` return values round-trip back into page JS unchanged.

**Dialog policy** is configured in `config.yaml` under `browser.dialog_policy`:

| Policy | Behavior |
|--------|----------|
| `must_respond` (default) | Capture the dialog and wait for an explicit `browser_dialog()` call. Safety auto-dismiss after `browser.dialog_timeout_s` (default 300s) so a buggy agent can't stall forever. |
| `auto_dismiss` | Capture, dismiss immediately. The agent doesn't have to act. |
| `auto_accept` | Capture, accept immediately. Useful when navigating pages with aggressive `beforeunload` prompts. |

### `browser_vault`

Uses saved logins, passwords, cards, addresses and one-time codes on the
current page without the secrets ever entering the conversation. The agent
calls `browser_vault(action="list")` to see saved items as handles, then `fill`,
`save_login`, `enter_code` or `unlock` (1Password / Bitwarden). Fills are
refused unless the page origin matches the saved item. See the
[credential vault](credential-vault.md) page for setup, password-manager
backends, and headless use.

## Turning browsing off

There is no config switch for the browser — disable the `browser` toolset
instead. Run `hermes tools` and untick **Browser Automation** for the platforms
that shouldn't browse, or from a script:

```bash
hermes tools disable browser                      # CLI
hermes tools disable browser --platform telegram  # one gateway platform
```

`browser_exec` also disappears from any session without the `terminal` toolset.

## Camofox

Camofox moved to a plugin: `hermes plugins install camofox`. Existing Camofox
setups (`CAMOFOX_URL` set, or `browser.cloud_provider: camofox`) are migrated automatically on
update: the plugin is installed and the core `browser` toolset is turned off in that profile. A
profile that had disabled `browser` everywhere (`agent.disabled_toolsets`) gets `camofox` disabled
too, so browsing stays off.
A `browser.camofox` block with no `CAMOFOX_URL` is not a Camofox setup. A profile that sets up the
plugin after the update keeps `browser_exec` alongside it.

## Practical examples

### Filling out a web form

```
User: Sign up for an account on example.com with my email john@example.com

Agent: browser_exec(code="""
# Opening the signup form
new_tab("https://example.com/signup")
wait_for_load()
fill_input("input[name=email]", "john@example.com")
js("document.querySelector('button[type=submit]').click()")
wait_for_load()
print(page_info())
""")
```

For the password, the agent uses `browser_vault` (`save_login` prompts you in a
masked field) instead of typing a secret itself.

### Researching dynamic content

```
User: What are the top trending repos on GitHub right now?

Agent: browser_exec(code="""
# Reading GitHub trending
new_tab("https://github.com/trending")
wait_for_load()
print(js("[...document.querySelectorAll('article h2 a')].map(a => a.textContent.replace(/\\s+/g, ' ').trim())"))
""")
```

## Limitations

- **Session timeout** — cloud sessions expire based on your provider's plan settings.
- **Cost** — cloud sessions consume provider credits; sessions are cleaned up when the conversation ends or after inactivity. Local Chromium and `/browser connect` are free.
- **Lightpanda** — no screenshots and one page per session.
- **No terminal, no browser** — `browser_exec` runs Python, so sessions without the `terminal` toolset don't get it.
