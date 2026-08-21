---
title: "Omarchy — Customize Omarchy Linux desktops and system configuration"
sidebar_label: "Omarchy"
description: "Customize Omarchy Linux desktops and system configuration"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Omarchy

Customize Omarchy Linux desktops and system configuration.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/omarchy` |
| Version | `0.1.0` |
| Author | Community contributor, Hermes Agent |
| License | MIT |
| Platforms | linux |
| Tags | `Omarchy`, `Linux`, `Hyprland`, `Desktop`, `Configuration` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Omarchy Skill

Manage [Omarchy](https://omarchy.org/) Linux systems - a beautiful, modern,
opinionated Arch Linux distribution with Hyprland.

This skill is for end-user customization on installed systems. It is not for
contributing to Omarchy source code.

## When This Skill MUST Be Used

**ALWAYS invoke this skill for end-user requests involving ANY of these:**

- Editing ANY file in `~/.config/hypr/` (window rules, animations, keybindings, monitors, etc.)
- Editing `~/.config/omarchy/shell.json` (status bar layout, widgets)
- Editing terminal configs (alacritty, foot, kitty, ghostty)
- Editing ANY file in `~/.config/omarchy/`
- Window behavior, animations, opacity, blur, gaps, borders
- Layer rules, workspace settings, display/monitor configuration
- Themes, backgrounds, fonts, appearance changes
- User-facing `omarchy` commands (`omarchy theme ...`, `omarchy refresh ...`, `omarchy restart ...`, etc.)
- Screenshots, screen recording, reminders, night light, idle behavior, lock screen

**If you're about to edit a config file in `~/.config/` on this system, STOP and use this skill first.**

**Do NOT use this skill for Omarchy development tasks** (editing the Omarchy source tree, creating migrations, or running `omarchy dev ...` workflows).

## Topic Guides

Read the matching guide before starting:

- [`hyprland.md`](https://github.com/NousResearch/hermes-agent/blob/main/skills/omarchy/references/hyprland.md) - keybindings, monitors, window rules, and other Hyprland config
- [`plugins.md`](https://github.com/NousResearch/hermes-agent/blob/main/skills/omarchy/references/plugins.md) - the Omarchy shell: bar layout, widgets, plugins, idle behavior
- [`theming.md`](https://github.com/NousResearch/hermes-agent/blob/main/skills/omarchy/references/theming.md) - themes, backgrounds, and fonts
- [`hooks.md`](https://github.com/NousResearch/hermes-agent/blob/main/skills/omarchy/references/hooks.md) - automation hooks that run on system events
- [`capture.md`](https://github.com/NousResearch/hermes-agent/blob/main/skills/omarchy/references/capture.md) - screenshots, screen recordings, OCR text capture, and file sharing

## Critical Safety Rules

For privileged commands, use `sudo` when a terminal is available for the
password prompt, and `pkexec` when it is not. Do not wrap commands that
already manage privilege elevation themselves.

**For end-user customization tasks, NEVER modify anything in `/usr/share/omarchy/`** - but READING is safe and encouraged.

This directory is owned by the omarchy package and changes will be overwritten
on the next `omarchy update`.

<!-- ascii-guard-ignore -->
```
/usr/share/omarchy/     # READ-ONLY - NEVER EDIT (reading is OK)
├── bin/                # Command source (packaged binaries are on PATH)
├── config/             # Default config templates
├── themes/             # Stock themes
├── default/            # System defaults
├── shell/              # Omarchy shell source and defaults
├── migrations/         # Update migrations
└── install/            # Installation scripts
```
<!-- ascii-guard-ignore-end -->

Reading `/usr/share/omarchy/` is safe and useful for understanding commands,
checking stock themes, and referencing default Hyprland settings. Use
`read_file` for those reads.

**Always use these safe locations instead:**
- `~/.config/` - User configuration
- `~/.config/omarchy/themes/<custom-name>/` - Custom themes
- `~/.config/omarchy/hooks/` - Custom automation hooks

## Privilege Escalation

For interactive commands in a visible terminal, use `sudo` for privileged
work. Use `pkexec` only when the caller cannot interact with a terminal or
enter a password, such as a graphical background process. Do not replace
`sudo` with `pkexec` merely because a command changes system state.

## System Architecture

| Component | Purpose | Config Location |
|-----------|---------|-----------------|
| **Arch Linux** | Base OS | `/etc/`, `~/.config/` |
| **Hyprland** | Wayland compositor/WM | `~/.config/hypr/` |
| **Omarchy shell** | Status bar + notifications (Quickshell) | `~/.config/omarchy/shell.json` |
| **Launcher/menus** | Quickshell menu | `~/.config/omarchy/extensions/omarchy-menu.jsonc` |
| **Alacritty/Foot/Kitty/Ghostty** | Terminals | `~/.config/<terminal>/` |
| **Omarchy OSD** | On-screen display | Quickshell plugin |

## Command Discovery

Omarchy ships a single `omarchy` CLI that dispatches to all `omarchy-*`
binaries via `omarchy <group> <action>`. Always prefer this form.

```bash
omarchy commands --all
omarchy theme --help
omarchy refresh --help
omarchy restart --help
omarchy theme set --help
omarchy commands --json
```

Common groups include `refresh` (reset configs, backing up first),
`restart`, `toggle`, `theme`, `bar`, `plugin`, `hook`, `install`,
`launch`, `capture`, `reminder`, `pkg`, `setup`, and `update`.
Run `omarchy --help` for the full list.

## Configuration Locations

Hyprland config lives in `~/.config/hypr/`; the Omarchy shell is configured
in `~/.config/omarchy/shell.json`. Terminal configs are:

```
~/.config/alacritty/alacritty.toml
~/.config/foot/foot.ini
~/.config/kitty/kitty.conf
~/.config/ghostty/config
```

Restart terminals with `omarchy restart terminal`. Other user configs include
btop, fastfetch, lazygit, starship, and git under their usual `~/.config/`
locations.

## Safe Customization Patterns

For simple changes, use `read_file` to inspect current config, back it up,
then use `patch` to edit it. Hyprland changes MUST be validated with
`hyprctl reload` followed by `hyprctl configerrors`. Shell and menu changes
hot-reload; terminal changes apply with `omarchy restart terminal`.

### Reset to Defaults - ALWAYS SEEK USER CONFIRMATION BEFORE RUNNING

```bash
omarchy refresh shell
omarchy refresh hyprland
```

Refresh backs up current config with a timestamp, copies defaults from
`$OMARCHY_PATH/config/`, and restarts the affected component as needed.

## System Commands

```bash
omarchy update
omarchy version
omarchy debug --no-sudo --print
omarchy system lock
omarchy system shutdown
omarchy system reboot
```

Always run `omarchy debug` with `--no-sudo --print` to avoid interactive
sudo prompts that hang the terminal.

## Troubleshooting

```bash
omarchy debug --no-sudo --print
omarchy refresh <app>
omarchy refresh config <config-file>
omarchy reinstall
```

## Decision Framework

1. Stock Omarchy command? Use it directly.
2. Config edit? Edit in `~/.config/`, never `/usr/share/omarchy/`.
3. Theme customization? Follow [`theming.md`](https://github.com/NousResearch/hermes-agent/blob/main/skills/omarchy/references/theming.md) and create a NEW custom theme directory.
4. Automation? Follow [`hooks.md`](https://github.com/NousResearch/hermes-agent/blob/main/skills/omarchy/references/hooks.md), using `omarchy hook install`.
5. Package install? Use `omarchy pkg add <pkgs...>` or `omarchy pkg aur add <pkgs...>`.
6. Built-in shell/plugin code? Follow [`plugins.md`](https://github.com/NousResearch/hermes-agent/blob/main/skills/omarchy/references/plugins.md); clone it with `omarchy plugin clone`.
7. Unsure if a command exists? Run `omarchy commands` or `omarchy <group> --help`.

### Reminder Requests

Use `omarchy reminder <minutes> [message]` directly. Convert natural language
durations to minutes and title-case short reminder labels when appropriate.

```bash
omarchy reminder 15 "Pickup Jack"
omarchy reminder 60 "Check laundry"
omarchy reminder show
omarchy reminder clear
```

## Out of Scope

Do not use this skill for editing `/usr/share/omarchy/`, creating or editing
migrations, or running `omarchy dev ...` commands.

## Example Requests

- "Change my theme to catppuccin" -> `omarchy theme set catppuccin`
- "Configure my external monitor" -> Edit `~/.config/hypr/monitors.lua`
- "Make the window gaps smaller" -> Edit `~/.config/hypr/looknfeel.lua`
- "Turn on night light" -> `omarchy toggle nightlight`
- "Set a reminder to pickup jack in 15 minutes" -> `omarchy reminder 15 "Pickup Jack"`
- "Show my reminders" -> `omarchy reminder show`
- "Clear all reminders" -> `omarchy reminder clear`
- "Customize theme colors" -> Overlay `colors.toml` in `~/.config/omarchy/themes/<theme>/`, then re-apply the theme
- "Run a script every time I change themes" -> `omarchy hook install theme-set <script>`
- "Lock after ten minutes" -> Set `idle.lock` to `600` in `~/.config/omarchy/shell.json`
- "Record my screen" -> `omarchy screenrecord --fullscreen`, then `omarchy screenrecord --stop-recording` (see `capture.md`)
