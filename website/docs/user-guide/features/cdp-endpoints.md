---
title: Multiple CDP endpoints
description: Bind named Hermes sessions to extra Chrome DevTools Protocol endpoints, and optionally fence a stay-put Chrome while a human holds Bot Screen.
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
3. **`browser.cdp_url`** — unnamed default

Unknown names fall through to `cdp_url`. They do not error and they do not invent a second Chrome.

`/json/version` WebSocket discovery still runs only on the connect path (`_get_cdp_override`), never on `/browser status`.

## Config

String values are still valid. Object values add optional `stay_put` provenance:

```yaml
browser:
  backend: off
  cdp_url: http://127.0.0.1:9222
  cdp_stay_put: true          # unnamed cdp_url is stay-put
  cdp_endpoints:
    primary:
      url: http://127.0.0.1:9222
      stay_put: true          # named session uses the same Chrome, also fenced
    lab2: http://127.0.0.1:9223   # string form: stay_put false
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

## Stay-put class (opt-in Bot Screen fence)

A **stay-put** CDP is a Chrome the agent actually uses as a long-lived shared cookie jar — often watched on a human TV — rather than a throwaway cloud browser. Bot Screen ([#108914](https://github.com/NousResearch/hermes-agent/pull/108914)) **unfences** user-supplied / cloud CDP by design (`run_fenced` skips anything without local provenance). That is correct for Browserbase and random `/browser connect` targets. It is wrong for a stay-put Chrome: a human holding the Bot Screen lease would still let the agent click the same jar.

Mark stay-put endpoints **opt-in**:

| Knob | Default | Effect |
|------|---------|--------|
| `browser.cdp_endpoints.<name>.stay_put` | `false` (and all string entries) | That named URL is fenced |
| `browser.cdp_stay_put` | `false` | The unnamed `cdp_url` is fenced |

When a marked URL is the selected CDP **and** a human holds the Bot Screen lease, browser actions refuse with `code: human_has_control` — the same shape as the local bot-desktop browser fence. Unmarked endpoints stay unfenced. No surprise breaks for cloud CDP.

The fence **soft-imports** `tools.bot_desktop.lease`. On `main` today that module is not present (Bot Screen is not merged): stay-put is recorded on the session, and the fence is a **no-op** (commands run). When #108914 lands, the same path lights up. Missing `bot_desktop` never crashes.

This does **not** spawn displays, add a Screen button, or move Chrome onto the Bot Screen `DISPLAY`. The stay-put Chrome keeps its own display and port; the lease is only an admission gate.

A CAPTCHA-solver extension sitting in that stay-put profile is one reason to mark it (shared cookies + a human watching the TV). The flag itself is generic — nothing in config or code requires a particular vendor.

## Example: stay-put lab Chrome plus sidecars

One long-lived Chrome on `:99` / `9222` plus light sidecar Chromes on other ports. Do not bounce the primary. Point named sessions at `9223` / `9224` via the map.

```yaml
browser:
  backend: off
  cdp_url: http://127.0.0.1:9222
  cdp_stay_put: true
  cdp_endpoints:
    primary:
      url: http://127.0.0.1:9222
      stay_put: true
    lab2: http://127.0.0.1:9223
    lab3: http://127.0.0.1:9224
```

Playwright (and any second automation runtime) must **attach**, not **launch**, when that stay-put Chrome already owns the profile and port.

**Wrong** — second process on the same `user-data-dir` or the same `--remote-debugging-port=9222`:

```python
playwright.chromium.launch_persistent_context(
    user_data_dir="~/.hermes/chrome-profile",
    args=["--remote-debugging-port=9222"],
)
```

**Right** — connect over CDP to the Chrome that is already running:

```python
browser = playwright.chromium.connect_over_cdp("http://127.0.0.1:9222")
```

Sidecar lab windows use a **separate** profile and port. Do not point Playwright at `9222` if you meant `9223`.
