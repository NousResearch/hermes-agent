---
title: Desktop Control Routing
sidebar_position: 9
---

# Desktop Control Routing

Hermes has several ways to inspect or operate user-facing software:

- `browser` for web pages and browser-backed dashboards.
- `vision` for screenshots and visual diagnosis.
- `terminal` and `file` for local code, config, logs, and filesystem state.
- `computer_use` for native desktop control. Today the production backend is macOS `cua-driver`.

The long-term shape should be a higher-level `desktop_control` abstraction that routes to the narrowest safe backend instead of forcing the model to choose a low-level tool first.

## Design goals

1. **Read-only before mutation.** Every native desktop backend starts with capture/tree/list operations only. Click/type/drag support comes later and remains approval-gated.
2. **No false capability claims.** Configuration can say a toolset is enabled while the runtime backend is unavailable. User-facing status must say both.
3. **No prompt-cache bloat.** Prefer CLI commands, skills, provider-gated tools, or a small routed tool surface. Do not add many always-present core tools.
4. **Cross-platform routing.** A desktop request should prefer browser/file/terminal/vision when those can solve the task more reliably than native UI automation.
5. **No foreground disruption on HQ.** Windows support must avoid focus stealing where possible and must not click security, payment, password, or 2FA surfaces without explicit approval.

## External references

- Microsoft UI Automation entry point: <https://learn.microsoft.com/en-us/windows/win32/winauto/entry-uiauto-win32>
- Microsoft UI Automation overview: <https://learn.microsoft.com/en-us/windows/win32/winauto/uiauto-uiautomationoverview>
- Microsoft UI Automation tree overview: <https://learn.microsoft.com/en-us/windows/win32/winauto/uiauto-treeoverview>

These references establish UI Automation as the native Windows accessibility/control API and describe the UIA element tree model that a read-only backend should expose.

## Phase 1: make availability visible

`hermes tools list` should distinguish:

- `✓ enabled`: configured and runtime requirements pass.
- `⚠ enabled but unavailable`: configured, but the tool registry `check_fn` would gate out the tool schema.
- `✗ disabled`: not configured.

For `computer_use` on Windows/Linux, the detail should say that `cua-driver` is macOS-only and route users to `browser`, `vision`, `terminal`, and `file` alternatives.

## Phase 2: Windows read-only GUI inspection backend

Add a Windows backend behind the existing computer-use backend boundary, but only expose read-only methods at first:

```text
Backend name: windows-uia
Initial operations:
- capture(mode="ax", app=None): return UIA tree as elements; screenshot optional.
- list_apps(): enumerate visible top-level windows/processes.
- focus_app(...): not enabled in read-only phase.
- click/type/key/scroll/drag/set_value: explicit NotImplemented / unavailable.
```

Implementation notes:

- Use UI Automation's control view as the default tree; expose raw view only for diagnostics because it can be noisy.
- Map UIA fields into existing `UIElement`: role/control type, name/automation id/value snippet, bounding rectangle, process id, window handle, enabled/offscreen/focusable flags.
- Prefer a dependency-light adapter. If a third-party Python package is chosen later, keep it behind the backend module and make import failure a clean availability reason.
- Add deterministic tests with fake UIA element objects before any live Windows smoke test.

Read-only output should be useful even for text-only models. Example:

```json
{
  "mode": "ax",
  "platform": "win32",
  "backend": "windows-uia",
  "window_title": "Settings",
  "elements": [
    {"index": 1, "role": "Button", "label": "Search", "bounds": [10, 12, 80, 30]}
  ]
}
```

## Phase 3: Windows UIA backend for computer_use

The existing internal backend selector already supports `HERMES_COMPUTER_USE_BACKEND` for tests. Long-term user-facing configuration should live in `config.yaml`; the env var can remain an internal/test override.

Suggested config shape:

```yaml
computer_use:
  backend: auto        # auto | cua | windows-uia | noop
  windows_uia:
    read_only: true    # default true until mutation support matures
    max_elements: 200
```

Selection policy:

```text
if backend == auto:
  darwin + cua-driver -> cua
  win32 + UIA available -> windows-uia read-only
  otherwise -> unavailable with route hints
```

Mutation support should be added only after read-only capture is stable:

1. `focus_app` with no raise/focus-steal guarantee if possible.
2. `set_value` for safe text/value controls.
3. `click` by element id only, not raw coordinates, with approval and post-capture verification.
4. `type`/`key` last, with the existing dangerous-pattern and hard-block rules mirrored on Windows.

## Phase 4: `desktop_control` top-level router

`desktop_control` should be an abstraction, not a pile of duplicated low-level actions. Proposed request shape:

```json
{
  "intent": "inspect | operate | diagnose",
  "target": "url | app | path | screenshot | unknown",
  "task": "human-readable goal",
  "safety": {"read_only": true},
  "preferred_surface": "auto | browser | native | vision | terminal | file"
}
```

Routing ladder:

1. URL or browser-detectable target -> `browser`.
2. File/config/log/repo task -> `file` + `terminal`.
3. Screenshot or visual-only evidence -> `vision`.
4. Native app read-only inspection -> platform backend (`cua` on macOS, `windows-uia` on Windows when available).
5. Native mutation -> approval-gated `computer_use` backend only when available and the action is not security/payment/secret/2FA-sensitive.

Return a route explanation with every result:

```json
{
  "route": "browser",
  "reason": "target is a URL and browser automation is cross-platform",
  "evidence": {...}
}
```

## Review checklist

- `hermes tools list` shows enabled-but-unavailable distinctly.
- `hermes computer-use status` remains platform-aware.
- Windows read-only backend never claims click/type support.
- User-facing docs do not instruct users to set new non-secret `HERMES_*` env vars as configuration.
- Mutating GUI actions remain approval-gated and post-verified.
