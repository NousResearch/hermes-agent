---
title: Multiple CDP endpoints
description: Bind named Hermes sessions to extra Chrome DevTools Protocol endpoints without a Desktop Screen button, while keeping one stay-put default CDP.
sidebar_label: CDP endpoints
sidebar_position: 5.5
---

# Multiple CDP endpoints

Hermes already attaches to **one** Chrome via `browser.cdp_url` or `BROWSER_CDP_URL` (`/browser connect`). That unnamed default is unchanged.

`browser.cdp_endpoints` is a **name → URL map** so a named `browser_exec(session=<name>)` call, `/browser connect <name>`, or `BROWSER_CDP_ENDPOINT=<name>` can attach to a **sidecar** Chrome instead of the default. No Desktop Screen / Start button. No display spawn. Config and CLI only.

This is **not** Bot Screen (PR [#108914](https://github.com/NousResearch/hermes-agent/pull/108914)). It is also **not** per-profile isolation ([#49693](https://github.com/NousResearch/hermes-agent/issues/49693)) or parallel-agent tab locking ([#49691](https://github.com/NousResearch/hermes-agent/pull/49691)). Those issues stay independent. Today, `session=` without a map hit still means "own tab on the **same** Chrome."

## Resolve order

`_get_cdp_override_raw` (no network I/O):

1. **`BROWSER_CDP_URL`** — live `/browser connect` URL. Process-global. Always wins.
2. **Named map hit** — first match in `browser.cdp_endpoints` among:
   - explicit `endpoint=` / `browser_exec(session=<name>)`
   - `BROWSER_CDP_ENDPOINT`
   - exact `HERMES_SESSION_ID` / `HERMES_SESSION_KEY`, then a non-UUID last `:` `/` `.` segment
3. **`browser.cdp_url`** — unnamed default (CapSolver stay-put on ADA: `http://127.0.0.1:9222`)

Unknown names fall through to `cdp_url`. They do not error and they do not invent a second Chrome.

`/json/version` WebSocket discovery still runs only on the connect path (`_get_cdp_override`), never on `/browser status`.

## Config

```yaml
browser:
  backend: off
  cdp_url: http://127.0.0.1:9222   # default CapSolver stay-put
  cdp_endpoints:
    capsolver: http://127.0.0.1:9222
    lab2: http://127.0.0.1:9223
    lab3: http://127.0.0.1:9224
```

Leave `BROWSER_CDP_URL` unset for concurrent named sidecars. A process-global `/browser connect` URL would pin **every** session to that one Chrome.

Bind a sidecar without forking the default:

```bash
# named Browser Use session → lab2 Chrome
# (browser_exec session="lab2")

# or CLI / env, still no Screen button
export BROWSER_CDP_ENDPOINT=lab2
# /browser connect lab2
```

## ADA lab process map (do not bounce CapSolver)

| Role | Display | Chrome profile | CDP | RFB | noVNC | CapSolver extension |
|------|---------|----------------|-----|-----|-------|---------------------|
| **Stage A — primary** | Xvfb `:99` | `~/.hermes/chrome-profile` | `127.0.0.1:9222` | `127.0.0.1:5900` | Tailscale `:6080` | loaded |
| **Stage A.5 — sidecar** | Xvfb `:2` | `…/chrome-profile-2` | `127.0.0.1:9223` | `127.0.0.1:5902` | `:6082` | **no** |
| **Stage A.5 — sidecar** | Xvfb `:3` | `…/chrome-profile-3` | `127.0.0.1:9224` | `127.0.0.1:5903` | `:6083` | **no** |
| **Stage B — optional** | Bot Screen Xfce + TigerVNC ([#108914](https://github.com/NousResearch/hermes-agent/pull/108914)) | stronger metal only | n/a | Unix-socket RFB into Hermes Desktop | Desktop pane | do **not** move CapSolver Chrome here in v1 |

Hermes on ADA:

```yaml
browser:
  backend: off
  cdp_url: http://127.0.0.1:9222
```

**Never** bounce `:99` / `9222` / `5900` / `6080`. **Never** start a second CapSolver on `9222`. Point a named session at `9223`/`9224` via the map, `BROWSER_CDP_URL`, or `/browser connect` — do not relaunch the primary.

### What ADA 2012 metal can and cannot do

ADA-VPS class hosts (example: MacBookPro9,2 · 2c/4t · 15.5 GiB) can hold **one** stay-put CapSolver Chrome plus a couple of **light** sidecar Xvfb+Chrome processes. They cannot comfortably run N full Bot Screen seats (Xfce + TigerVNC + another Chrome per bot). Shared CapSolver CDP is the correct primary. Bot Screen ([#108914](https://github.com/NousResearch/hermes-agent/pull/108914)) is Stage B on stronger metal.

v1 fence vs Bot Screen: **do not move CapSolver Chrome onto the Bot Screen DISPLAY.** Lease-fence local/loopback CDP when a human holds Bot Screen; keep the stay-put Chrome on `:99` / `9222`.

## Playwright-on-CDP footgun

Playwright (and any second automation runtime) must **attach**, not **launch**, when CapSolver already owns the profile and port.

**Wrong** — second process on the same `user-data-dir` or the same `--remote-debugging-port=9222`:

```python
# Do not do this against the CapSolver profile or port.
playwright.chromium.launch_persistent_context(
    user_data_dir="~/.hermes/chrome-profile",  # lock fight / corrupt stay-put
    args=["--remote-debugging-port=9222"],     # port already taken
)
```

**Right** — connect over CDP to the Chrome that is already running:

```python
browser = playwright.chromium.connect_over_cdp("http://127.0.0.1:9222")
```

Sidecar lab windows use a **separate** profile and port (`chrome-profile-2` + `9223`). Do not load the CapSolver extension there. Do not point Playwright at `9222` if you meant `9223`.
