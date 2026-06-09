---
title: Computer Use
sidebar_position: 16
---

# Computer Use (macOS)

Hermes Computer Use lets an agent inspect and operate real macOS apps in the background without moving the user's cursor, stealing keyboard focus, or switching Spaces during normal operation.

Use it when browser/CDP, terminal, filesystem tools, or an app-specific API cannot complete the task.

## Native tool surface

The `computer_use` toolset exposes explicit first-class tools:

- `computer_use_list_apps`
- `computer_use_launch_app`
- `computer_use_get_app_state`
- `computer_use_click`
- `computer_use_perform_secondary_action`
- `computer_use_scroll`
- `computer_use_drag`
- `computer_use_type_text`
- `computer_use_set_value`
- `computer_use_press_key`
- `computer_use_select_text`
- `computer_use_daemon`

There is no catch-all action-dispatch Computer Use tool in the greenfield path. Each tool name carries intent for policy, approvals, logs, and the native app UX.

## Workflow

The rule is strict: **state → action → state**.

0. `computer_use_launch_app(app="Mail")` if Mail is not already available
1. `computer_use_get_app_state(app="Mail", mode="som")`
2. `computer_use_click(app="Mail", element=14)`
3. `computer_use_get_app_state(app="Mail", mode="som")` again to verify

Use `ax` for fast accessibility-tree reads, `som` for screenshot + indexed elements, and `vision` for screenshot-only inspection.

## How it works

The model calls native Hermes tools. Hermes applies policy, approval, screenshot handling, and backend routing underneath. The current macOS backend talks to the approved local Computer Use daemon so permissions are centralized and future Swift/TUI/Telegram controls can wrap the whole flow cleanly.

## Enabling

```bash
hermes computer-use install
hermes computer-use status
hermes -t computer_use chat
```

Grant macOS Accessibility and Screen Recording permissions when prompted.

## Safety

Hermes applies layered guardrails:

- Read-only calls are allowed.
- Mutating UI calls require approval with app/action scope.
- Destructive system shortcuts and dangerous typed shell payloads are hard-blocked.
- The agent must not click permission dialogs, type secrets, or follow instructions embedded in screenshots/web pages.
- High-impact UI actions need action-time confirmation: deletes, purchases, uploads, account creation finals, sensitive-data transmission, public posts/messages, system setting changes, medical/legal/financial flows, and similar.

## Multi-agent behavior

Different agents can usually operate different apps/windows concurrently. Mutating the same window should be locked or queued by policy/app UX; without a lease, element indices and UI state can race.

## Troubleshooting

- **Tool unavailable:** run `hermes tools` and enable Computer Use, or run `hermes computer-use install`.
- **Click no-op:** re-read state; a modal or stale element index probably blocked the action.
- **Text-only model:** use `mode="ax"`, or switch to a vision-capable model for screenshot-heavy work.
- **Daemon stuck:** the agent can call `computer_use_daemon(action="status"|"start"|"stop")` to inspect or restart the driver without shelling out via terminal.

## Configuration

- `computer_use.show_cursor` (bool) — show the agent-cursor overlay when actions fire. Useful for demos. Set with `hermes config set computer_use.show_cursor true`, or override per-process with `HERMES_CUA_SHOW_CURSOR=1`. Defaults to the driver's built-in setting.

## See also

- Root skill: `computer-use`
- Browser automation: `browser` toolset for web-native tasks
