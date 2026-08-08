---
sidebar_position: 7
title: "Live Glass Plugin API Findings"
description: "Verified implementation seams for the computer-use live glass plugin."
---

# Live Glass Plugin API Findings

This note records the code-level findings for the proposed computer-use "live glass" feature. It is intentionally scoped to the seams the implementation depends on, so follow-on work can build the plugin without re-litigating the discovery step.

## Feature shape

The desired product is a live window into `computer_use` activity with three event types:

- `frame` — near-live screenshot/capture updates from the controlled desktop.
- `log` — tool/action lifecycle entries with tool name, arguments, result status, and timing.
- `approval_request` — a user decision point with approve/deny controls that pauses execution until answered.

The architecture should stay transport-agnostic: one core event bus emits these events, and dashboard/chat surfaces render them differently.

## Verified dashboard plugin API

Dashboard extensions are supported without editing dashboard source. The current code and docs agree on this registration path:

- A plugin directory may contain `dashboard/manifest.json`, `dashboard/dist/index.js`, optional `dashboard/dist/style.css`, and optional `dashboard/plugin_api.py`.
- The dashboard scans user plugins, bundled plugins, and opt-in project plugins for `dashboard/manifest.json` in `hermes_cli/web_server.py` (`_discover_dashboard_plugins`).
- A new nav page is registered from the manifest `tab` block:
  - `tab.path` sets the route, defaulting to `/<name>`.
  - `tab.position` controls nav placement.
  - `tab.override` can replace a built-in page.
  - `tab.hidden` loads the plugin without adding a nav item.
- The frontend loads the manifest list from `GET /api/dashboard/plugins` and plugins register components through the browser globals described in `website/docs/user-guide/features/extending-the-dashboard.md`.
- Static plugin assets are served under `/dashboard-plugins/<name>/...`.
- Backend API routes are enabled by setting manifest `api` to a relative file inside `dashboard/`; that file must export `router = APIRouter()`.
- Plugin backend routes are mounted under `/api/plugins/<name>/`, so `@router.get("/events")` becomes `GET /api/plugins/<name>/events` or, for WebSockets, `WS /api/plugins/<name>/events`.

Important constraint: project plugins are discovered for UI, but project-plugin backend API routes are intentionally skipped for safety. `hermes_cli/web_server.py` blocks auto-importing Python APIs from `source == "project"`. For live glass, use a user-installed plugin under `~/.hermes/plugins/live-glass/` during dogfood, or a bundled plugin if upstreamed.

Important constraint: dashboard plugin API route mounting happens at dashboard startup. `GET /api/dashboard/plugins/rescan` refreshes plugin discovery, but a new/changed `plugin_api.py` route requires restarting `hermes dashboard`.

## Verified PluginManager hooks

`hermes_cli/plugins.py` is the correct plugin-loading seam for the transport-agnostic core:

- Directory plugins require `plugin.yaml` and `__init__.py` with `register(ctx)`.
- Plugin sources are bundled, user, opt-in project, and pip entry points (`hermes_agent.plugins`).
- Plugins are opt-in through `plugins.enabled`, with `plugins.disabled` as a deny list.
- The relevant lifecycle hooks are already listed in `VALID_HOOKS`:
  - `pre_tool_call`
  - `post_tool_call`
  - `pre_approval_request`
  - `post_approval_response`
- `post_tool_call` is observational and receives the tool name, args, result, task/session/tool-call/turn IDs, duration, status, and middleware trace from `model_tools.py`.
- `pre_tool_call` can block a tool call by returning a directive; live glass should generally observe rather than block.
- Approval hooks are explicitly observer-only. They cannot approve, deny, veto, or pre-answer a prompt.

Implementation consequence: the live-glass event bus can ship as a normal PluginManager plugin and register lifecycle hook callbacks for logs and approval observability.

## Verified gateway hook seam

`gateway/hooks.py` is a separate drop-in gateway event-hook system:

- Hooks live under `~/.hermes/hooks/<name>/` with `HOOK.yaml` and `handler.py`.
- Events include `gateway:startup`, `session:start`, `agent:start`, `agent:step`, `agent:end`, and `command:*`.
- `_register_builtin_hooks()` exists and is currently empty; it is the reserved slot for future always-on gateway hooks.
- Handler errors are caught and logged and must not block gateway execution.

Implementation consequence: do not start by editing `gateway/builtin_hooks/` or `gateway/hooks.py`. Build as a user plugin first. If dogfooding proves the gateway-side hook belongs upstream, propose a minimal built-in hook later.

## Verified approval flow

The dangerous-command approval flow is centralized in `tools/approval.py` for terminal/shell-style dangerous commands:

- `tools/approval.py` fires `pre_approval_request` and `post_approval_response` through `hermes_cli.plugins.invoke_hook`.
- The hook payload includes command, description, pattern key(s), session key, surface, turn ID, and tool-call ID.
- Gateway approvals block through an internal pending-entry/condition flow until a platform callback resolves or the request times out.
- `hermes_cli/callbacks.py` bridges CLI/TUI approval UI with a blocking queue and the choices `once`, `session`, `always`, and `deny`.

Correction to the original assumption: `computer_use` has its own approval gate in `tools/computer_use/tool.py` via `set_approval_callback()` / `_request_approval()`. That path does not currently fire `tools/approval.py`'s `pre_approval_request` and `post_approval_response` hooks.

Implementation consequence: the approval bridge ticket must either:

1. extend `computer_use` approval to emit equivalent PluginManager approval hooks, preserving its existing safety semantics, or
2. route `computer_use` approval through the shared approval observer path without changing who makes the decision.

The live-glass plugin must not invent a second approval authority. It should render the request and resolve through the existing approval callback/entry mechanism.

## Verified dashboard streaming transport

The dashboard already streams live agent events to the browser over WebSockets, not SSE:

- `hermes_cli/web_server.py` exposes `/api/pub` for the PTY-side gateway publisher and `/api/events` for browser subscribers.
- `tui_gateway/event_publisher.py` publishes newline-framed JSON over a best-effort WebSocket.
- `web/src/components/ChatSidebar.tsx` subscribes with `new WebSocket(.../api/events?...)` and renders tool/reasoning/status events.
- The stream is channel-scoped: `/api/pty` creates an opaque channel ID and `/api/pub` and `/api/events` must use the same channel.
- There is no global dashboard broadcast bus for arbitrary plugin clients today.

Implementation consequence: the dashboard live-glass endpoint should use WebSocket semantics to match the current dashboard stream. A plugin backend can expose its own `@router.websocket("/events")` under `/api/plugins/live-glass/events`; if reuse of `/api/pub` and `/api/events` is desired, that requires careful channel ownership rather than assuming a global bus.

## Verified computer-use screenshot/input seam

The active desktop-control tool is `tools/computer_use/`:

- `tools/computer_use_tool.py` registers the public `computer_use` tool.
- `tools/computer_use/schema.py` defines one `action`-discriminated function schema with `capture`, input actions, `capture_after`, and targeting fields.
- `tools/computer_use/tool.py` dispatches actions through a backend selected by `HERMES_COMPUTER_USE_BACKEND`, defaulting to `cua` / `cua-driver`.
- `capture` and `capture_after=True` return a multimodal tool result containing a `data:image/...;base64,...` screenshot plus text summary.
- `capture` supports `mode="som"`, `mode="vision"`, and `mode="ax"`; `som` includes numbered element overlays and an AX tree.
- Mutating actions (`click`, `drag`, `scroll`, `type`, `key`, `set_value`, `focus_app`) go through the separate computer-use approval callback.

Correction to wording: this repository's built-in tool is currently macOS/cua-driver based. The planned "computer-use-linux MCP" adapter may still be the target for Linux, but this codebase seam is the generic `computer_use` tool and its backend package, not a first-class `computer-use-linux` module in the repo.

Implementation consequence: the first frame emitter should observe `computer_use` tool results and extract/redact the latest screenshot from multimodal `capture` / `capture_after` results. A later poller can call the MCP/backend directly, but the safest first cut is to reuse already-produced captures and avoid extra desktop access.

## Gateway/platform rendering seams

`gateway/delivery.py` and `gateway/platforms/` already handle outbound delivery. Platform adapters differ in media and button capability:

- Telegram supports photo delivery and callback-style inline controls in its adapter stack.
- Discord and Slack have richer message/update APIs, but need adapter-specific implementation rather than a generic button abstraction assumed up front.
- BlueBubbles/iMessage should be treated as degraded: image messages plus reply-to-approve text, not inline buttons.

Implementation consequence: keep the core event bus ignorant of platforms. Build each chat adapter as a renderer subscribed to `frame`, `log`, and `approval_request` events, with capability flags per platform.

## CONTRIBUTING constraints that affect the plan

From `CONTRIBUTING.md`:

- Contribution priorities are, in order: bug fixes, cross-platform compatibility, security hardening, performance/robustness, new skills, new tools, documentation.
- The feature should be framed as robustness/observability and developed plugin-first; new core tools should be avoided unless necessary.
- Preferred full test command is `scripts/run_tests.sh`; targeted `pytest tests/...` is acceptable during development, but PR readiness expects the wrapper.
- New code touching OS behavior must account for Linux, macOS, and Windows/WSL2, and `scripts/check-windows-footguns.py` should run before PR.
- Commit messages use conventional-commit style (`feat(scope): ...`, `fix(scope): ...`, `docs(scope): ...`, `test(scope): ...`).
- Branch names should use conventional prefixes such as `feat/...`, `fix/...`, `docs/...`, or `test/...`.

The repository currently has a large pytest suite; follow-on implementation tickets should run targeted tests while iterating and record full-suite expectations in Linear before PR.

## Plan corrections

The Linear plan should be adjusted around these facts:

1. Primary dashboard surface remains correct, but implement the stream as WebSocket, not SSE.
2. Plugin-first remains correct, but backend API routes must be installed as user/bundled plugin routes, not project-plugin routes.
3. Approval reuse remains correct, but computer-use approval currently bypasses the shared approval hooks; add a bridge rather than assuming it already exists.
4. The initial frame source should be existing `computer_use` multimodal results. A separate screenshot poller should be a later enhancement with explicit throttling and safety controls.
5. Upstream proposal should ask maintainers whether the empty built-in gateway hook slot should host a small event publisher after the plugin proves useful.

## Suggested first implementation branch and scopes

- Branch for this note: `feat/live-glass-plugin-api-findings`.
- Follow-on implementation branch: `feat/live-glass-plugin`.
- Likely commit scopes:
  - `docs(live-glass)` for this findings note and proposal docs.
  - `feat(plugins)` for the PluginManager event-bus plugin.
  - `feat(dashboard)` for the live-view tab/backend WebSocket route.
  - `feat(gateway)` for chat renderers/adapters.
  - `test(live-glass)` for event bus, approval, adapter, and route tests.

## Files verified

- `hermes_cli/web_server.py`
- `web/src/plugins/registry.ts`
- `web/src/App.tsx`
- `web/src/components/ChatSidebar.tsx`
- `plugins/kanban/dashboard/manifest.json`
- `plugins/kanban/dashboard/plugin_api.py`
- `website/docs/user-guide/features/extending-the-dashboard.md`
- `hermes_cli/plugins.py`
- `gateway/hooks.py`
- `model_tools.py`
- `tools/approval.py`
- `hermes_cli/callbacks.py`
- `tools/computer_use_tool.py`
- `tools/computer_use/schema.py`
- `tools/computer_use/tool.py`
- `gateway/delivery.py`
- `gateway/platforms/`
- `CONTRIBUTING.md`
