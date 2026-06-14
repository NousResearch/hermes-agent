---
sidebar_position: 18
title: "Browser CDP Supervisor"
description: "How Hermes detects and responds to native JS dialogs and interacts with cross-origin iframes via a persistent CDP connection."
---

# Browser CDP Supervisor

The CDP supervisor closes two long-standing gaps in Hermes' browser tooling:

1. **Native JS dialogs** (`alert`/`confirm`/`prompt`/`beforeunload`) block the
   page's JS thread. Without supervision, the agent has no way to know a
   dialog is open — subsequent tool calls hang or throw opaque errors.
2. **Cross-origin iframes (OOPIFs)** are invisible to top-level
   `Runtime.evaluate`. The agent can see iframe nodes in the DOM snapshot but
   can't click, type, or eval inside them without a CDP session attached to
   the child target.

The supervisor solves both by holding a persistent WebSocket to the backend's
CDP endpoint per browser task, surfacing pending dialogs and frame structure
into `browser_snapshot`, and exposing a `browser_dialog` tool for explicit
responses.

## Backend support

| Backend | Dialog detect | Dialog respond | Frame tree | OOPIF `Runtime.evaluate` via `browser_cdp(frame_id=...)` |
|---|---|---|---|---|
| Local Chrome (`--remote-debugging-port`) / `/browser connect` | ✓ | ✓ full workflow | ✓ | ✓ |
| Browserbase | ✓ (via bridge) | ✓ full workflow (via bridge) | ✓ | ✓ |
| Camofox | ✗ no CDP (REST-only) | ✗ | partial via DOM snapshot | ✗ |

**Browserbase quirk.** Browserbase's CDP proxy uses Playwright internally and
auto-dismisses native dialogs within ~10ms, so `Page.handleJavaScriptDialog`
can't keep up. The supervisor injects a bridge script via
`Page.addScriptToEvaluateOnNewDocument` that overrides
`window.alert`/`confirm`/`prompt` with a synchronous XHR to a magic host
(`hermes-dialog-bridge.invalid`). `Fetch.enable` intercepts those XHRs before
they touch the network — the dialog becomes a `Fetch.requestPaused` event the
supervisor captures, and `respond_to_dialog` fulfills via
`Fetch.fulfillRequest` with a JSON body the injected script decodes.

From the page's perspective, `prompt()` still returns the agent-supplied
string. From the agent's perspective, it's the same `browser_dialog(action=...)`
API either way.

Camofox is unsupported — no CDP surface, REST-only.

## Architecture

### CDPSupervisor

One `asyncio.Task` running in a background daemon thread per Hermes `task_id`.
Holds a persistent WebSocket to the backend's CDP endpoint. Maintains:

- **Dialog queue** — `List[PendingDialog]` with `{id, type, message, default_prompt, session_id, opened_at}`
- **Frame tree** — `Dict[frame_id, FrameInfo]` with parent relationships, URL, origin, whether cross-origin child session
- **Session map** — `Dict[session_id, SessionInfo]` so interaction tools can route to the right attached session for OOPIF operations
- **Recent console errors** — ring buffer of the last 50 for diagnostics

Subscribes on attach:

- `Page.enable` — `javascriptDialogOpening`, `frameAttached`, `frameNavigated`, `frameDetached`
- `Runtime.enable` — `executionContextCreated`, `consoleAPICalled`, `exceptionThrown`
- `Target.setAutoAttach {autoAttach: true, flatten: true}` — surfaces child OOPIF targets; supervisor enables `Page`+`Runtime` on each

Thread-safe state access via a snapshot lock; tool handlers (sync) read the
frozen snapshot without awaiting.

### Lifecycle

- **Start:** `SupervisorRegistry.get_or_start(task_id, cdp_url)` — called by
  `browser_navigate`, Browserbase session create, `/browser connect`.
  Idempotent.
- **Stop:** session teardown or `/browser disconnect`. Cancels the asyncio
  task, closes the WebSocket, discards state.
- **Rebind:** if the CDP URL changes (user reconnects to a new Chrome), the
  old supervisor is stopped and a fresh one started — state is never reused
  across endpoints.

### Dialog policy

Configurable via `config.yaml` under `browser.dialog_policy`:

- **`must_respond`** (default) — capture, surface in `browser_snapshot`, wait
  for explicit `browser_dialog(action=...)` call. After a 300s safety timeout
  with no response, auto-dismiss and log. Prevents a buggy agent from stalling
  forever.
- `auto_dismiss` — record and dismiss immediately; agent sees it after the
  fact via `browser_state` inside `browser_snapshot`.
- `auto_accept` — record and accept (useful for `beforeunload` where the
  workflow wants to navigate away cleanly).

Policy is per-task; no per-dialog overrides.

## Agent surface

### `browser_dialog` tool

```
browser_dialog(action, prompt_text=None, dialog_id=None)
```

- `action="accept"` / `"dismiss"` → responds to the specified or sole pending dialog (required)
- `prompt_text=...` → text to supply to a `prompt()` dialog
- `dialog_id=...` → disambiguate when multiple dialogs are queued (rare)

Tool is response-only. The agent reads pending dialogs from `browser_snapshot`
output before calling.

### `browser_snapshot` extension

Adds three optional fields to the existing snapshot output when a supervisor
is attached:

```json
{
  "pending_dialogs": [
    {"id": "d-1", "type": "alert", "message": "Hello", "opened_at": 1650000000.0}
  ],
  "recent_dialogs": [
    {"id": "d-1", "type": "alert", "message": "...", "opened_at": 1650000000.0,
     "closed_at": 1650000000.1, "closed_by": "remote"}
  ],
  "frame_tree": {
    "top": {"frame_id": "FRAME_A", "url": "https://example.com/", "origin": "https://example.com"},
    "children": [
      {"frame_id": "FRAME_B", "url": "about:srcdoc", "is_oopif": false},
      {"frame_id": "FRAME_C", "url": "https://ads.example.net/", "is_oopif": true, "session_id": "SID_C"}
    ],
    "truncated": false
  }
}
```

- **`pending_dialogs`** — dialogs currently blocking the page's JS thread.
  The agent must call `browser_dialog(action=...)` to respond. Empty on
  Browserbase because their CDP proxy auto-dismisses within ~10ms.

- **`recent_dialogs`** — ring buffer of up to 20 recently-closed dialogs with
  a `closed_by` tag: `"agent"` (we responded), `"auto_policy"` (local
  auto_dismiss/auto_accept), `"watchdog"` (must_respond timeout hit), or
  `"remote"` (browser/backend closed it on us, e.g. Browserbase). This is
  how agents on Browserbase still get visibility into what happened.

- **`frame_tree`** — frame structure including cross-origin (OOPIF) children.
  Capped at 30 entries + OOPIF depth 2 to bound snapshot size on ad-heavy
  pages. `truncated: true` surfaces when limits were hit; agents needing
  the full tree can use `browser_cdp` with `Page.getFrameTree`.

No new tool schema surface for any of these — the agent reads the snapshot it
already requests.

### Availability gating

Both surfaces gate on `_browser_cdp_check` (supervisor can only run when a CDP
endpoint is reachable). On Camofox / no-backend sessions, the dialog tool is
hidden and the snapshot omits the new fields — no schema bloat.

## Cross-origin iframe interaction

`browser_cdp(frame_id=...)` routes CDP calls (notably `Runtime.evaluate`)
through the supervisor's already-connected WebSocket using the OOPIF's child
`sessionId`. Agents pick frame_ids out of
`browser_snapshot.frame_tree.children[]` where `is_oopif=true` and pass them
to `browser_cdp`. For same-origin iframes (no dedicated CDP session), the
agent uses `contentWindow`/`contentDocument` from a top-level
`Runtime.evaluate` instead — the supervisor surfaces an error pointing at that
fallback when `frame_id` belongs to a non-OOPIF.

On Browserbase, this is the only reliable path for iframe interaction —
stateless CDP connections (opened per `browser_cdp` call) hit signed-URL
expiry, while the supervisor's long-lived connection keeps a valid session.

## File layout

- `tools/browser_supervisor.py` — `CDPSupervisor`, `SupervisorRegistry`, `PendingDialog`, `FrameInfo`
- `tools/browser_dialog_tool.py` — `browser_dialog` tool handler
- `tools/browser_tool.py` — `browser_navigate` start-hook, `browser_snapshot` merge, `/browser connect` reattach, `_cleanup_browser_session` teardown
- `toolsets.py` — registers `browser_dialog` in `browser`, `hermes-acp`, `hermes-api-server`, and core toolsets (gated on CDP reachability)
- `hermes_cli/config.py` — `browser.dialog_policy` and `browser.dialog_timeout_s` defaults

## Non-goals

- Detection/interaction for Camofox (upstream gap; tracked separately)
- Streaming dialog/frame events live to the user (would require gateway hooks)
- Persisting dialog history across sessions (in-memory only)
- Per-iframe dialog policies (agent can express this via `dialog_id`)
- Replacing `browser_cdp` — it stays as the escape hatch for the long tail (cookies, viewport, network throttling)

## Testing

Unit tests (`tests/tools/test_browser_supervisor.py`) use an asyncio mock CDP
server that speaks enough of the protocol to exercise all state transitions:
attach, enable, navigate, dialog fire, dialog dismiss, frame attach/detach,
child target attach, session teardown. Real-backend E2E (Browserbase + local
Chromium-family browser) is manual — exercise via `/browser connect` to a
live Chromium-family browser and run the dialog/frame test cases described
above.
