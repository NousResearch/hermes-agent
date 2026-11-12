---
title: "Findmy — Locate Apple devices/AirTags via Find My (macOS)"
sidebar_label: "Findmy"
description: "Locate Apple devices/AirTags via Find My (macOS)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Findmy

Locate Apple devices/AirTags via Find My (macOS).

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/apple/findmy` |
| Version | `2.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | macos |
| Tags | `FindMy`, `AirTag`, `location`, `tracking`, `macOS`, `Apple` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Find My (Apple)

Find My has no CLI or API. Switch tabs through the app's **View menu** with
`osascript`, then read the list as text from the accessibility tree with
`computer_use(action="capture", mode="ax")`. No screenshots or vision needed.

## When to Use

- "Where is my [iPhone/Mac/AirPods/keys/bag]?"
- Checking AirTag or device locations
- Logging an owned item's location over time

## Prerequisites

- macOS with the Find My app, signed in to iCloud; devices/AirTags registered
- **Accessibility** permission for the app running `osascript` (the terminal,
  or Hermes). Error `-1719`/`-1728` "not allowed assistive access" means it
  is missing.
- `computer_use` needs cua-driver to have **Accessibility** and **Screen
  Recording**. Check with `hermes computer-use permissions status`
  (`hermes computer-use doctor` for detail).
  `hermes computer-use permissions grant` may fail to register CuaDriver in
  Screen Recording. The reliable fix: the user adds `CuaDriver.app` (path shown
  by `doctor`, under `~/.hermes/tools/cua-driver-*/CuaDriver.app`) with **+** in
  System Settings > Privacy & Security > Screen & System Audio Recording.
  **A human must do this. Never click permission dialogs yourself.**

## Procedure

Tell the user first: Find My comes to the front when opened and will briefly
take over the screen.

1. Open the app and go straight to Devices (or Items, for AirTags) **before
   any capture**. Find My reopens on the last tab used, often People:

   ```bash
   open -a FindMy; sleep 5
   osascript -e 'tell application "System Events" to tell process "FindMy" to click menu item "Devices" of menu 1 of menu bar item "View" of menu bar 1'
   sleep 3
   ```

   Use `"Items"` instead of `"Devices"` for AirTags. View menu items: People,
   Devices, Items, Map, Zoom In, Zoom Out, Map Mode, Share My Location, Enter
   Full Screen.

2. Read the list as text:

   ```
   computer_use(action="capture", app="FindMy", mode="ax")
   ```

   - The `AXHeading` names the current tab (Devices/Items). Confirm it before
     reading entries.
   - Entries are `AXStaticText` shaped
     `<name>, [Nearby, ]<place or address> · <last seen>[ · <battery>]`, e.g.
     `macbook, Home · 7 min. ago`,
     `AirPods Pro, Nearby, Home · 6 min. ago`,
     `Wallet, Nearby, Home · 23 min. ago · Quarter-Charged Battery`.
   - Other elements are map POI pins; ignore them.

3. Delete only the files that capture call returned: its `screenshot_path`
   (a `computer_use_*` file in `~/.hermes/cache/images`) and its
   `elements_file` (an `elements_*.json` file in
   `~/.hermes/cache/computer_use`). Never delete those directories or other
   files in them: `~/.hermes/cache/images` also holds images users sent.

## Don't

- Don't click the People/Devices/Items/Me tabs (an `AXTabGroup` of
  `AXRadioButton`s). Clicks via `computer_use` (element, background or
  foreground) only highlight the button; the list doesn't change. Use the View
  menu.
- Don't use `screencapture -w`: it waits for a human to click a window.

## Tracking Over Time

A cron job runs in the gateway, a different process from the interactive
terminal, and may lack these permissions. Say so to the user, and have the job
run `hermes computer-use permissions status` in that context first. Each run:
open, switch tab via the View menu, capture in `ax` mode, log the matching
entry's text, then delete only the files that capture call returned (its
`computer_use_*` screenshot in `~/.hermes/cache/images` and its
`elements_*.json` in `~/.hermes/cache/computer_use`), never the directories.

Unverified: older notes claim AirTags only update while their page is open.
Don't rely on it either way.

## Rules

1. **Privacy:** never capture before switching to Devices or Items. Don't read,
   store or repeat the People tab (other people's locations) unless the user
   explicitly asks about a person who shares location with them.
2. Only track devices/items the user owns.
3. Never grant permissions or click permission dialogs; ask the user.
4. Locations come from the Find My network and can be stale; report the
   "last seen" time with every location.
