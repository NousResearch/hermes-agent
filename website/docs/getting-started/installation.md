---
sidebar_position: 2
title: "Installation"
description: "Install Hermes Agent on Linux, macOS, WSL2, or Android via Termux"
---

# Installation

Get Hermes Agent up and running in under two minutes with the one-line installer.

## Quick Install

### Linux / macOS / WSL2

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### Android / Termux

Hermes now ships a Termux-aware installer path too:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

The installer detects Termux automatically and switches to a tested Android flow:
- uses Termux `pkg` for hard requirements (`git`, `python`, certificates/curl, build tools as needed)
- creates the virtualenv with `python -m venv`
- exports `ANDROID_API_LEVEL` automatically for Android wheel builds
- installs `.[termux-all]` for the default install option, or `.[termux-minimal]` for `--install-option minimal` / `--install-option minimalTUI`
- skips optional Node/browser, WhatsApp, TUI/npm, voice/TTS, dashboard, and `ffmpeg` work unless the selected install option or `--with ...` requests them

If you want the fully explicit path, follow the dedicated [Termux guide](./termux.md).

:::warning Windows
Native Windows is **not supported**. Please install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and run Hermes Agent from there. The install command above works inside WSL2.
:::

The default installer uses the **default** install option: the full desktop/server feature set Hermes traditionally installed.

For a compact install, choose `minimal` or `minimalTUI` explicitly:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --install-option minimal
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --install-option minimalTUI
```

`minimal` installs the core Python CLI plus lightweight agent tools: skills, file editing, terminal/process, todo, memory, session search, clarify, and web search/extraction. `minimalTUI` adds the TUI dependencies. To opt into specific features during install, use `--with`:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --with terminal,file,web-search
```

### What the Installer Does

The installer handles the repo clone, virtual environment, global `hermes` command setup, and LLM provider configuration. The default install option installs the full Hermes feature set. Compact install options keep dependencies smaller: `minimal` sticks to the core Python CLI and lightweight agent tools, while `minimalTUI` adds TUI dependencies without pulling in every optional integration. Selected features add their own extras: for example `--with dashboard` installs local web UI/API dependencies, `--with browser` enables Node/browser setup, and `--with tts`/`--with voice` checks `ffmpeg`. By the end, you're ready to chat; install extra features only when you need them.

#### Install Layout

Where the installer puts things depends on whether you're installing as a normal user or as root:

| Installer | Code lives at | `hermes` binary | Data directory |
|---|---|---|---|
| Per-user (normal) | `~/.hermes/hermes-agent/` | `~/.local/bin/hermes` (symlink) | `~/.hermes/` |
| Root-mode (`sudo curl … \| sudo bash`) | `/usr/local/lib/hermes-agent/` | `/usr/local/bin/hermes` | `/root/.hermes/` (or `$HERMES_HOME`) |

The root-mode **FHS layout** (`/usr/local/lib/…`, `/usr/local/bin/hermes`) matches where other system-wide developer tools land on Linux. It's useful for shared-machine deployments where one system install should serve every user. Per-user config (auth, skills, sessions) still lives under each user's `~/.hermes/` or explicit `HERMES_HOME`.

### After Installation

Reload your shell and start chatting:

```bash
source ~/.bashrc   # or: source ~/.zshrc
hermes             # Start chatting!
```

To reconfigure individual settings later, use the dedicated commands:

```bash
hermes model          # Choose your LLM provider and model
hermes tools          # Configure which tools are enabled
hermes gateway setup  # Set up messaging platforms
hermes config set     # Set individual config values
hermes setup          # Or run the full setup wizard to configure everything at once
```

---

## Prerequisites

The hard prerequisites depend on the install option:

- **Default** install: Git, Python/uv, Node.js for frontend/browser/TUI features, and `ffmpeg` for media features are checked or installed as needed.
- **Minimal** install: Git is the only hard prerequisite; uv/Python are bootstrapped or managed on desktop platforms.
- **minimalTUI**: minimal plus the TUI dependency path.

Notes:
- **uv** (fast Python package manager) is bootstrapped if missing
- **Python 3.11** is managed via uv on desktop platforms
- **Node.js v22** is checked/installed for the default install option, browser/TUI features (`--with browser`, `--with tui`), or compact `minimalTUI`
- **ripgrep** is optional in minimal; file search falls back when it is absent
- **ffmpeg** is checked/installed for the default install option, TTS, or voice (`--with tts`, `--with voice`)

:::info
You do **not** need to install Python, Node.js, ripgrep, or ffmpeg manually for the minimal smoke path. Make sure `git` is available (`git --version`), run the installer with `--install-option minimal`, reload your shell, then start with `hermes`.
:::

:::tip Nix users
If you use Nix (on NixOS, macOS, or Linux), there's a dedicated setup path with a Nix flake, declarative NixOS module, and optional container mode. See the **[Nix & NixOS Setup](./nix-setup.md)** guide.
:::

---

## Manual / Developer Installation

If you want to clone the repo and install from source — for contributing, running from a specific branch, or having full control over the virtual environment — see the [Development Setup](../developer-guide/contributing.md#development-setup) section in the Contributing guide.

When you run `scripts/install.sh` from a local checkout, the installer uses that checkout's current tracked remote and branch by default. This keeps fork/feature-branch testing from silently updating `~/.hermes/hermes-agent` back to `NousResearch/main`. Override explicitly with `--repo URL --branch NAME` if needed.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `hermes: command not found` | Reload your shell (`source ~/.bashrc`) or check PATH |
| `API key not set` | Run `hermes model` to configure your provider, or `hermes config set OPENROUTER_API_KEY your_key` |
| Missing config after update | Run `hermes config check` then `hermes config migrate` |

For more diagnostics, run `hermes doctor` — it will tell you exactly what's missing and how to fix it.
