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
2. **Cross-origin iframes (OOPIFs)** need an attached CDP session for
   supervisor observation. Hermes exposes their structure in snapshots but
   deliberately does not expose a raw-CDP or JavaScript-evaluation path.

The supervisor holds a persistent WebSocket for any browser task with an
attached CDP URL and surfaces pending dialogs and frame structure into
`browser_snapshot`. `browser_dialog` and the read-only `browser_cdp` tool
register only for an explicit `/browser connect` or `browser.cdp_url` override;
that endpoint may be cloud-hosted. A provider-managed per-session CDP URL is not
automatically surfaced as that override.

## Backend support

| Backend | Dialog detect | Dialog respond | Frame tree | Raw-CDP / eval exposure |
|---|---|---|---|---|
| Explicit `/browser connect` or `browser.cdp_url` override (local or cloud-hosted) | ✓ | ✓ full workflow | ✓ | read-only browser inspection only |
| Provider-managed Browserbase, Browser Use, Firecrawl session CDP | ✓ when attached | ✗ not registered automatically | ✓ when attached | ✗ |
| Camofox / default local agent-browser | ✗ no CDP endpoint | ✗ | partial via DOM snapshot | ✗ |

The read-only direct inspection path is for explicitly configured CDP
transport only. Browser-controller protocols never negotiate or dispatch raw CDP
or arbitrary evaluation, including in Developer Mode.

**Cloud-session boundary.** A cloud provider's per-session CDP URL can attach a
supervisor and add snapshot fields, but it does not automatically register the
direct tools. Explicitly configure that URL through `/browser connect` or
`browser.cdp_url` to opt into the read-only inspection and dialog-response
tools. Camofox has no CDP endpoint.

## Architecture

### CDPSupervisor

One `asyncio.Task` running in a background daemon thread per Hermes `task_id`.
Holds a persistent WebSocket to the backend's CDP endpoint. Maintains:

- **Dialog queue** — `List[PendingDialog]` with `{id, type, message, default_prompt, session_id, opened_at}`
- **Frame tree** — `Dict[frame_id, FrameInfo]` with parent relationships, URL, origin, whether a cross-origin child session exists for supervisor observation
- **Session map** — `Dict[session_id, SessionInfo]` so interaction tools can route to the right attached session for OOPIF operations
- **Recent console errors** — ring buffer of the last 50 for diagnostics

Subscribes on attach:

- `Page.enable` — `javascriptDialogOpening`, `frameAttached`, `frameNavigated`, `frameDetached`
- `Runtime.enable` — `executionContextCreated`, `consoleAPICalled`, `exceptionThrown`
- `Target.setAutoAttach {autoAttach: true, flatten: true}` — surfaces child OOPIF targets; supervisor enables `Page`+`Runtime` on each

Thread-safe state access via a snapshot lock; tool handlers (sync) read the
frozen snapshot without awaiting.

### Lifecycle

- **Start:** `SupervisorRegistry.get_or_start(task_id, cdp_url)` — called when
  a task has an attached explicit override or provider session CDP URL.
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
  The agent must call `browser_dialog(action=...)` to respond.

- **`recent_dialogs`** — ring buffer of up to 20 recently-closed dialogs with
  a `closed_by` tag: `"agent"` (we responded), `"auto_policy"` (local
  auto_dismiss/auto_accept), `"watchdog"` (must_respond timeout hit), or
  `"remote"` (browser/backend closed it).

- **`frame_tree`** — frame structure including cross-origin (OOPIF) children.
  Capped at 30 entries + OOPIF depth 2 to bound snapshot size on ad-heavy
  pages. `truncated: true` surfaces when limits were hit; agents needing
  use snapshots and dedicated browser actions for further interaction.

No new tool schema surface for any of these — the agent reads the snapshot it
already requests.

### Availability gating

`browser_cdp` and `browser_dialog` gate on `_browser_cdp_check`: they register
only when an explicit CDP override was configured at session start via
`/browser connect` or `browser.cdp_url`, including a cloud-hosted endpoint. A
provider-managed session URL alone does not register them. Separately, any
attached CDP session can start the supervisor, so its snapshot fields may appear
for cloud sessions; Camofox and the default local agent-browser omit them.

## Cross-origin iframe boundary

The supervisor keeps child CDP sessions to observe dialogs and frame structure,
not to provide an evaluation transport. `browser_cdp` accepts only the narrow
read-only browser-level allowlist (`Browser.getVersion`, `Target.getTargets`)
and rejects targets and `frame_id`. `browser_console(expression=...)` is also
disabled. This prevents an attached or hostile page from using iframe routing
to execute code, access credentials, or reach internal services.

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
- Expanding `browser_cdp` beyond its read-only browser-level allowlist

## Testing

Unit tests (`tests/tools/test_browser_supervisor.py`) use an asyncio mock CDP
server that speaks enough of the protocol to exercise all state transitions:
attach, enable, navigate, dialog fire, dialog dismiss, frame attach/detach,
child target attach, session teardown. Manual end-to-end coverage uses
`/browser connect` with a live Chromium-family browser and the dialog/frame
test cases described above. Provider-managed cloud CDP is covered separately as
a snapshot-supervisor path; an explicitly configured cloud endpoint follows the
same direct-tool boundary as a local endpoint.
