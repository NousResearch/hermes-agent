# Browser CDP Supervisor — Design

**Status:** Shipped (PR 14540)
**Last updated:** 2026-04-23
**Author:** @teknium1

## Problem

Native JS dialogs (`alert`/`confirm`/`prompt`/`beforeunload`) and iframes are
the two biggest gaps in our browser tooling:

1. **Dialogs block the JS thread.** Any operation on the page stalls until the
   dialog is handled. Before this work, the agent had no way to know a dialog
   was open — subsequent tool calls would hang or throw opaque errors.
2. **Iframes are invisible.** The agent could see iframe nodes in the DOM
   snapshot but could not click, type, or eval inside them — especially
   cross-origin (OOPIF) iframes that live in separate Chromium processes.

[PR #12550](https://github.com/NousResearch/hermes-agent/pull/12550) proposed a
stateless `browser_dialog` wrapper. That doesn't solve detection — it's a
cleaner CDP call for when the agent already knows (via symptoms) that a dialog
is open. Closed as superseded.

## Backend capability matrix (verified live 2026-04-23)

Using throwaway probe scripts against a data-URL page that fires alerts in the
main frame and in a same-origin srcdoc iframe, plus a cross-origin
`https://example.com` iframe:

| Backend | Dialog detect | Dialog respond | Frame tree | OOPIF `Runtime.evaluate` via `browser_cdp(frame_id=...)` |
|---|---|---|---|---|
| Local Chrome (`--remote-debugging-port`) / `/browser connect` | ✓ | ✓ full workflow | ✓ | ✓ |
| Browserbase | ✓ (via bridge) | ✓ full workflow (via bridge) | ✓ | ✓ (`document.title = "Example Domain"` verified on real cross-origin iframe) |
| Camofox | ✗ no CDP (REST-only) | ✗ | partial via DOM snapshot | ✗ |

**How Browserbase respond works.** Browserbase's CDP proxy uses Playwright
internally and auto-dismisses native dialogs within ~10ms, so
`Page.handleJavaScriptDialog` can't keep up. To work around this, the
supervisor injects a bridge script via
`Page.addScriptToEvaluateOnNewDocument` that overrides
`window.alert`/`confirm`/`prompt` with a synchronous XHR to a magic host
(`hermes-dialog-bridge.invalid`). `Fetch.enable` intercepts those XHRs
before they touch the network — the dialog becomes a `Fetch.requestPaused`
event the supervisor captures, and `respond_to_dialog` fulfills via
`Fetch.fulfillRequest` with a JSON body the injected script decodes.

Net result: from the page's perspective, `prompt()` still returns the
agent-supplied string. From the agent's perspective, it's the same
`browser_dialog(action=...)` API either way. Tested end-to-end against
real Browserbase sessions — 4/4 (alert/prompt/confirm-accept/confirm-dismiss)
pass including value round-tripping back into page JS.

Camofox stays unsupported for this PR; follow-up upstream issue planned at
`jo-inc/camofox-browser` requesting a dialog polling endpoint.

## Architecture

### CDPSupervisor

One `asyncio.Task` running in a background daemon thread per Hermes `task_id`.
Holds a persistent WebSocket to the backend's CDP endpoint. Maintains:

- **Dialog queue** — `List[PendingDialog]` with `{id, type, message, default_prompt, session_id, opened_at}`
- **Frame tree** — `Dict[frame_id, FrameInfo]` with parent relationships, URL, origin, whether cross-origin child session
- **Session map** — `Dict[session_id, SessionInfo]` so interaction tools can route to the right attached session for OOPIF operations
- **Recent console errors** — ring buffer of the last 50 (for PR 2 diagnostics)

Subscribes on attach:
- `Page.enable` — `javascriptDialogOpening`, `frameAttached`, `frameNavigated`, `frameDetached`
- `Runtime.enable` — `executionContextCreated`, `consoleAPICalled`, `exceptionThrown`
- `Target.setAutoAttach {autoAttach: true, flatten: true}` — surfaces child OOPIF targets; supervisor enables `Page`+`Runtime` on each

Thread-safe state access via a snapshot lock; tool handlers (sync) read the
frozen snapshot without awaiting.

### Lifecycle

- **Start:** `SupervisorRegistry.get_or_start(task_id, cdp_url)` — called by
  `browser_navigate`, Browserbase session create, `/browser connect`. Idempotent.
- **Stop:** session teardown or `/browser disconnect`. Cancels the asyncio
  task, closes the WebSocket, discards state.
- **Rebind:** if the CDP URL changes (user reconnects to a new Chrome), stop
  the old supervisor and start fresh — never reuse state across endpoints.

### Dialog policy

Configurable via `config.yaml` under `browser.dialog_policy`:

- **`must_respond`** (default) — capture, surface in `browser_snapshot`, wait
  for explicit `browser_dialog(action=...)` call. After a 300s safety timeout
  with no response, auto-dismiss and log. Prevents a buggy agent from stalling
  forever.
- `auto_dismiss` — record and dismiss immediately; agent sees it after the
  fact via `browser_state` inside `browser_snapshot`.
- `auto_accept` — record and accept (useful for `beforeunload` where the user
  wants to navigate away cleanly).

Policy is per-task; no per-dialog overrides in v1.

## Agent surface (PR 1)

### One new tool

```
browser_dialog(action, prompt_text=None, dialog_id=None)
```

- `action="accept"` / `"dismiss"` → responds to the specified or sole pending dialog (required)
- `prompt_text=...` → text to supply to a `prompt()` dialog
- `dialog_id=...` → disambiguate when multiple dialogs queued (rare)

Tool is response-only. Agent reads pending dialogs from `browser_snapshot`
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

- **`pending_dialogs`**: dialogs currently blocking the page's JS thread.
  The agent must call `browser_dialog(action=...)` to respond. Empty on
  Browserbase because their CDP proxy auto-dismisses within ~10ms.

- **`recent_dialogs`**: ring buffer of up to 20 recently-closed dialogs with
  a `closed_by` tag — `"agent"` (we responded), `"auto_policy"` (local
  auto_dismiss/auto_accept), `"watchdog"` (must_respond timeout hit), or
  `"remote"` (browser/backend closed it on us, e.g. Browserbase). This is
  how agents on Browserbase still get visibility into what happened.

- **`frame_tree`**: frame structure including cross-origin (OOPIF) children.
  Capped at 30 entries + OOPIF depth 2 to bound snapshot size on ad-heavy
  pages. `truncated: true` surfaces when limits were hit; agents needing
  the full tree can use `browser_cdp` with `Page.getFrameTree`.

No new tool schema surface for any of these — the agent reads the snapshot
it already requests.

### Availability gating

Both surfaces gate on `_browser_cdp_check` (supervisor can only run when a CDP
endpoint is reachable). On Camofox / no-backend sessions, the dialog tool is
hidden and snapshot omits the new fields — no schema bloat.

## Cross-origin iframe interaction

Extending the dialog-detect work, `browser_cdp(frame_id=...)` routes CDP
calls (notably `Runtime.evaluate`) through the supervisor's already-connected
WebSocket using the OOPIF's child `sessionId`. Agents pick frame_ids out of
`browser_snapshot.frame_tree.children[]` where `is_oopif=true` and pass them
to `browser_cdp`. For same-origin iframes (no dedicated CDP session), the
agent uses `contentWindow`/`contentDocument` from a top-level
`Runtime.evaluate` instead — supervisor surfaces an error pointing at that
fallback when `frame_id` belongs to a non-OOPIF.

On Browserbase, this is the ONLY reliable path for iframe interaction —
stateless CDP connections (opened per `browser_cdp` call) hit signed-URL
expiry, while the supervisor's long-lived connection keeps a valid session.

## Camofox (follow-up)

Issue planned against `jo-inc/camofox-browser` adding:
- Playwright `page.on('dialog', handler)` per session
- `GET /tabs/:tabId/dialogs` polling endpoint
- `POST /tabs/:tabId/dialogs/:id` to accept/dismiss
- Frame-tree introspection endpoint

## Files touched (PR 1)

### New

- `tools/browser_supervisor.py` — `CDPSupervisor`, `SupervisorRegistry`, `PendingDialog`, `FrameInfo`
- `tools/browser_dialog_tool.py` — `browser_dialog` tool handler
- `tests/tools/test_browser_supervisor.py` — mock CDP WebSocket server + lifecycle/state tests
- `website/docs/developer-guide/browser-supervisor.md` — this file

### Modified

- `toolsets.py` — register `browser_dialog` in `browser`, `hermes-acp`, `hermes-api-server`, core toolsets (gated on CDP reachability)
- `tools/browser_tool.py`
  - `browser_navigate` start-hook: if CDP URL resolvable, `SupervisorRegistry.get_or_start(task_id, cdp_url)`
  - `browser_snapshot` (at ~line 1536): merge supervisor state into return payload
  - `/browser connect` handler: restart supervisor with new endpoint
  - Session teardown hooks in `_cleanup_browser_session`
- `hermes_cli/config.py` — add `browser.dialog_policy` and `browser.dialog_timeout_s` to `DEFAULT_CONFIG`
- Docs: `website/docs/user-guide/features/browser.md`, `website/docs/reference/tools-reference.md`, `website/docs/reference/toolsets-reference.md`

## Non-goals

- Detection/interaction for Camofox (upstream gap; tracked separately)
- Streaming dialog/frame events live to the user (would require gateway hooks)
- Persisting dialog history across sessions (in-memory only)
- Per-iframe dialog policies (agent can express this via `dialog_id`)
- Replacing `browser_cdp` — it stays as the escape hatch for the long tail (cookies, viewport, network throttling)

## Testing

Unit tests use an asyncio mock CDP server that speaks enough of the protocol
to exercise all state transitions: attach, enable, navigate, dialog fire,
dialog dismiss, frame attach/detach, child target attach, session teardown.
Real-backend E2E (Browserbase + local Chrome) is manual — exercise via
`/browser connect` to a live Chrome and run the dialog/frame test cases
described above.
