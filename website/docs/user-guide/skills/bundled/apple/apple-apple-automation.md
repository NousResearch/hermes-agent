---
title: "Apple Automation"
sidebar_label: "Apple Automation"
description: "macOS Apple app automation umbrella: Notes, Reminders, Messages/iMessage, Find My, and GUI control via Accessibility/AppleScript"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Apple Automation

macOS Apple app automation umbrella: Notes, Reminders, Messages/iMessage, Find My, and GUI control via Accessibility/AppleScript. Use when managing local Apple apps or user data on macOS.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/apple/apple-automation` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Apple Automation

Use this umbrella whenever the task involves local macOS Apple applications, user-facing app data, or Apple ecosystem state. Prefer dedicated CLIs when available; use Accessibility/AppleScript and screenshots only when no structured CLI exists.

## Operating rules

1. Confirm the host is macOS before acting (`sw_vers`, `uname`, or tool availability).
2. Prefer non-interactive CLIs over GUI automation: `memo` for Notes, `remindctl` for Reminders, `imsg` for Messages, and Find My only through its app UI/screenshot path.
3. Treat Apple app data as user data: read before modifying, and avoid bulk destructive changes without explicit user scope.
4. When GUI automation is required, use Accessibility discovery first, then act by element role/name; verify with a fresh screenshot or state readback.

## Notes (`memo` CLI)

- Use for creating, searching, reading, and editing Apple Notes.
- Typical flow: discover note/list context, search existing notes before creating duplicates, perform the smallest edit, then read the note back.
- Do not use Notes for reminders or time-sensitive tasks; use Reminders/Calendar instead.

## Reminders (`remindctl` CLI)

- Use for listing reminder lists, adding reminders, completing items, and checking due reminders.
- Typical flow: list available lists, choose the requested/default list, create or update the reminder, then list/read it back to verify.
- Preserve due dates, notes, and list membership unless the user asked to change them.

## Messages / iMessage (`imsg` CLI)

- Use for reading recent chats, sending iMessages/SMS, and locating chat IDs.
- Typical flow: identify the recipient/chat, preview the exact message when ambiguity exists, send once, then verify the CLI returned a message/chat handle.
- Never invent a contact mapping; search/list chats when the user names a person or group ambiguously.

## Find My

- Use only for locating Apple devices/AirTags through the Find My app on macOS.
- Typical flow: open Find My, navigate to Devices/Items as needed, capture the window, interpret visible location/time/battery, and report uncertainty if the UI is stale or hidden.
- Do not claim live precision beyond what the visible UI states.

## macOS computer use

- Use Accessibility/screenshot control for any Apple or third-party GUI task where no CLI exists.
- Canonical loop: snapshot/accessibility tree → choose target → click/type/shortcut → verify with another snapshot/screenshot.
- Keep GUI work background-safe: avoid stealing foreground state unnecessarily and report when permissions block automation.
## Support files

- `references/absorbed-skills.md` — list of original skill packages consolidated into this umbrella and where to recover full archived content.
