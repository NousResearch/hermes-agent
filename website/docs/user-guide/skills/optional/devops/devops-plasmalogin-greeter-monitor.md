---
title: "Plasmalogin Greeter Monitor — Put the Plasma Login greeter on the right monitor"
sidebar_label: "Plasmalogin Greeter Monitor"
description: "Put the Plasma Login greeter on the right monitor"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Plasmalogin Greeter Monitor

Put the Plasma Login greeter on the right monitor.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/devops/plasmalogin-greeter-monitor` |
| Path | `optional-skills/devops/plasmalogin-greeter-monitor` |
| Version | `1.0.0` |
| Author | Dextroid17 (@Dextroid17) + Hermes Agent |
| License | MIT |
| Platforms | linux |
| Tags | `kde`, `plasma`, `plasmalogin`, `kwin`, `wayland`, `display`, `multi-monitor`, `login-screen` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Plasma Login Greeter Monitor Skill

Fixes a multi-monitor KDE Plasma 6 machine where the desktop layout is correct
but the login screen (greeter) appears on the wrong monitor. It does this by
giving the greeter's own KWin a copy of the user's display layout. It does not
change the user's desktop session and does not touch SDDM.

## When to Use

- The login screen appears on a secondary monitor after boot or logout.
- The desktop session's monitor layout and primary display are already correct.
- `display-manager.service` is `plasmalogin.service` (package `plasma-login-manager`).
- If the system runs SDDM instead, this skill does not apply.

## Prerequisites

- Linux with KDE Plasma 6 and Plasma Login Manager as the display manager.
- The user's desktop already has the intended primary monitor (System Settings →
  Display Configuration → "Primary").
- Root for one install command. The agent never handles a password: use
  `pkexec` (the user types into a GUI dialog) or give the user the command to run.
- Python 3 (stdlib only) for `scripts/greeter_outputs.py`.

## How to Run

All commands go through `terminal`. The helper path is relative to this skill's
directory.

```bash
python3 scripts/greeter_outputs.py scan
python3 scripts/greeter_outputs.py check ~/.config/kwinoutputconfig.json
python3 scripts/greeter_outputs.py stage ~/.config/kwinoutputconfig.json
```

## Quick Reference

- Greeter KWin user — `plasmalogin` (home `/var/lib/plasmalogin`)
- Greeter output config — `/var/lib/plasmalogin/.config/kwinoutputconfig.json`
- User output config — `~/.config/kwinoutputconfig.json`
- Primary flag — `setups[].outputs[].priority == 1`, with `outputIndex` pointing into `outputs[]` (`connectorName`)
- Required ownership — `plasmalogin:plasmalogin`, mode `0644`

## Procedure

1. Confirm the display manager: `systemctl status display-manager` should name
   `plasmalogin.service`.
2. Map connectors to physical monitors with `scan`. It reads each connector's EDID and
   prints the manufacturer code and model name, so "DP-2" becomes a real monitor.
   `kscreen-doctor -o` shows the live session's priorities as a cross-check.
3. Run `check` on the user's config. For the setup whose `connectors` list matches the
   monitors that are currently connected, `primary` must be the intended connector.
   If it isn't, fix the primary in System Settings first and re-check.
4. Run `stage` to copy the config into a private temp file. Note the printed
   `staged` path and `sha256`.
5. Install it for the greeter (replace `STAGED` with the printed path):

   ```bash
   pkexec install -D -m 0644 -o plasmalogin -g plasmalogin STAGED /var/lib/plasmalogin/.config/kwinoutputconfig.json
   ```

6. Tell the user the change takes effect the next time the greeter starts (logout or
   reboot). The current session is not affected.

## Pitfalls

- The greeter runs its own KWin as the `plasmalogin` user and reads its own config.
  Editing the user's desktop config alone never moves the login screen.
- With no greeter config, KWin uses defaults and often makes the first connector
  primary. That default is the usual cause of the problem.
- System Settings → Login Screen has no monitor option, so don't send the user there.
- Ownership must be `plasmalogin:plasmalogin`. If the file is root-owned, KWin can't
  rewrite it when the greeter's outputs change. `install -D` also creates the parent
  directory.
- Don't run `plasmalogin --help`. It starts the daemon-style binary and hangs.
- The package may ship no man pages. Don't spend calls looking for
  `plasmalogin.conf` documentation.
- Never ask for, type, or store the user's password. If `pkexec` has no GUI agent
  (for example over SSH), hand the command to the user instead.

## Verification

- `stat -c '%U:%G %a' /var/lib/plasmalogin/.config/kwinoutputconfig.json` reports
  `plasmalogin:plasmalogin 644`.
- `sha256sum` of the installed file matches the `sha256` printed by `stage`.
- Real proof: log out, and the greeter appears on the intended monitor.
- Still wrong: inspect `journalctl -b -u plasmalogin.service` for output or
  connector messages, and re-run `scan` in case connector names changed
  (for example after a GPU or cable change).
