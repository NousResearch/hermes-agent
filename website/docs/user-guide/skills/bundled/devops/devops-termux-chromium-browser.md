---
title: "Termux Chromium Browser — Native headless Chromium browser over CDP for Termux"
sidebar_label: "Termux Chromium Browser"
description: "Native headless Chromium browser over CDP for Termux"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Termux Chromium Browser

Native headless Chromium browser over CDP for Termux.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/devops/termux-chromium-browser` |
| Version | `0.1.0` |
| Author | Thamer (taljeri), @pjy010218, Hermes Agent |
| License | MIT |
| Platforms | linux, android |
| Tags | `Termux`, `Android`, `Chromium`, `Browser`, `CDP`, `Automation` |
| Related skills | [`sdlc-review`](/docs/user-guide/skills/bundled/devops/devops-sdlc-review), [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Termux Chromium Browser

Run a native headless Chromium browser on Android Termux and automate browsing over the Chrome DevTools Protocol (CDP) without requiring proot, root, or a graphical desktop.

Credits: Special thanks to **@pjy010218** for discovering and implementing the single-process Chromium flag combination that resolves the Android Termux `LD_PRELOAD` child execution conflict.

## When to Use

- "Browse web pages directly inside Android Termux"
- "Extract rendered text and JavaScript DOM content from a website on Termux"
- "Take web page screenshots on Android without proot or root"
- "Run headless Chromium over CDP on Android without Selenium or Playwright"
- "Interact with web forms, click buttons, or verify video playback on Termux"

Don't use for:
- Standard desktop/server systems with pre-installed Chrome or Playwright
- GPU-accelerated 3D WebGL games (single-process mode lacks full GPU acceleration on most mobile chipsets)

## The Android Technical Gotchas & Solution

Running Chromium natively inside Termux typically hits two major obstacles:
1. **ET_DYN ELF Executable**: The Android Linux kernel cannot exec Chromium directly without Termux's `LD_PRELOAD=libtermux-exec.so`.
2. **Child Process Breakage**: However, `LD_PRELOAD` breaks Chromium's internal zygote and network service child processes (`ERR_ABORTED`).

**The Solution**: Launch Chromium with `--no-zygote --single-process`. This forces Chromium to run entirely inside a single process tree, eliminating child process spawning and enabling seamless operation under Termux.

## Prerequisites

- Android device with Termux.
- Packages: `x11-repo`, `tur-repo`, `chromium`, `runit`, `python`.
- Python package: `websocket-client`.

## Installation

Run the setup commands in Termux:

```bash
# 1. Install packages
pkg install -y x11-repo tur-repo
pkg install -y chromium runit python
pip install websocket-client

# 2. Run service configuration script
bash skills/devops/termux-chromium-browser/scripts/setup_service.sh

# 3. Verify CDP liveness (listening on 127.0.0.1:9222)
python3 skills/devops/termux-chromium-browser/scripts/browser.py version
```

## How to Run

Execute commands via the `terminal` tool or shell:

```bash
# Open a new tab and navigate to a URL
python3 skills/devops/termux-chromium-browser/scripts/browser.py newtab "https://en.wikipedia.org"

# Read rendered visible text from the active page
python3 skills/devops/termux-chromium-browser/scripts/browser.py read 2000

# Search for specific text on the page
python3 skills/devops/termux-chromium-browser/scripts/browser.py find "Wikipedia"

# Capture a full-page screenshot
python3 skills/devops/termux-chromium-browser/scripts/browser.py shot page.png

# Evaluate JavaScript expression
python3 skills/devops/termux-chromium-browser/scripts/browser.py eval "document.title"

# List all open tabs
python3 skills/devops/termux-chromium-browser/scripts/browser.py tabs
```

## Quick Reference

| Task | Command |
|---|---|
| Open new tab | `python3 scripts/browser.py newtab <url>` |
| Navigate tab | `python3 scripts/browser.py navigate <url>` |
| Read page text | `python3 scripts/browser.py read [n]` |
| Screenshot | `python3 scripts/browser.py shot [out.png]` |
| Evaluate JS | `python3 scripts/browser.py eval "<expr>"` |
| Search text | `python3 scripts/browser.py find "<text>"` |
| Check version | `python3 scripts/browser.py version` |

## Pitfalls

- **ICU Data Permissions**: If Chromium fails with `Invalid file descriptor to ICU data received`, ensure read permissions on `$PREFIX/lib/chromium/` (`chmod -R 755 $PREFIX/lib/chromium`).
- **Memory Footprint**: Headless Chromium idles around 300–400 MB RSS. Close unused tabs periodically.
- **Renderer Idle**: Long-idle tabs in single-process mode may delay screenshot captures. Use `newtab` for fresh automation tasks.

## Verification

Verify CDP connection:
```bash
python3 skills/devops/termux-chromium-browser/scripts/browser.py version
```
