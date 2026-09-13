---
sidebar_position: 2
title: "Installation"
description: "Install Hermes Agent with desktop bundles, source installers, Docker, Nix, or the Termux APT package"
---

# Installation

Get Hermes Agent up and running in under two minutes!

:::tip Platform Support
For the full platform support matrix (which OSes, distribution methods, and
platform-gated features are supported), see **[Platform Support](./platform-support.md)**.
:::

## Quick Install
### Desktop packages on macOS or Windows

Download the package for your platform from the
[Hermes website](https://hermes-agent.nousresearch.com/).

- **Windows:** open the `.appinstaller` download with Windows App Installer.
  It installs the signed MSIX bundle and records its update source.
  Microsoft Store packages have separate Store ownership.
- **macOS:** open the DMG, then copy `Hermes.app` to Applications. The ZIP
  artifact carries the signed app used by the automatic updater.

Bundled packages contain the agent, Python, supported dependencies, and prebuilt
interfaces. First launch does not build that base runtime. Provider access and
optional integrations can still require network access.

A `Hermes-Setup` bootstrap installer is different: it downloads a source
installation and builds the desktop app. Light is a remote-only build variant,
not a bundled local runtime. See [Hermes Desktop](../user-guide/desktop.md).

### Without Hermes Desktop:
For a command-line only install without Hermes Desktop, run:

#### Linux / macOS / WSL2
```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

#### Windows (native)

Run in powershell:
```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1) 
```

If you want to install & run Hermes Desktop after a command-line only install, simply run
```bash
hermes desktop
```

### Android / Termux

Use the [Termux APT package](./termux.md) on aarch64 Android devices.
Configure its signed repository before running `pkg install hermes-agent`.
The desktop/server scripts are not the Termux installation path.

### What the source installer does

The scripts clone the source, bootstrap uv, and delegate dependency preparation
to PM. PM provides pinned Python, Node.js, npm, ripgrep, and FFmpeg. The source
installation selects the `all` Python extra, not every optional extra.
Browsers and other optional tools use their feature-specific installation paths.

The scripts create a launcher and prepare the data directory. Interactive runs
also invoke setup and gateway configuration. `--non-interactive` on POSIX, or
`-NonInteractive` on Windows, skips stages that need input. The optional
`--include-desktop` / `-IncludeDesktop` stage builds the desktop from source.

#### Install layout

| Method | Code | CLI entry point | Default user data |
|---|---|---|---|
| POSIX source script | `~/.hermes/hermes-agent/` | `~/.local/bin/hermes` wrapper | `~/.hermes/` |
| Windows source script | `%LOCALAPPDATA%\hermes\hermes-agent\` | `%LOCALAPPDATA%\hermes\bin\` | `%LOCALAPPDATA%\hermes\` |
| Desktop bundle | Inside the installed app package | Packaged launchers; Windows execution aliases | Platform default Hermes data directory |
| Docker | `/opt/hermes/` | Image entrypoint and `hermes` shim | Mounted `/opt/data/` |
| Termux APT | `$PREFIX/lib/hermes-agent/` | Symlinks in `$PREFIX/bin/` | `~/.hermes/` |

`HERMES_HOME` selects user data. The POSIX script's `--dir` selects its source
checkout independently. Windows provides `-HermesHome` and `-InstallDir`.
Running the POSIX script as root does not select an automatic FHS layout:
it uses root's home unless you provide an explicit source path.

PM's tool store and per-install Python generations have separate lifetimes.
See [Package management](../reference/package-management.md) for their locations.
Do not remove the data root to repair an application installation.

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
hermes config get     # Inspect individual config values
hermes setup          # Or run the full setup wizard to configure everything at once
```

:::tip Fastest path: Nous Portal
One subscription covers 300+ models plus the [Tool Gateway](/user-guide/features/tool-gateway) (web search, image generation, TTS, cloud browser). Skip the per-tool key juggling:

```bash
hermes setup --portal
```

That logs you in, sets Nous as your provider, and turns on the Tool Gateway in one command.
:::

:::tip Already running Hermes on another machine?
You don't need to rebuild your setup from scratch. Restore a full backup with `hermes import` (see [Exporting Hermes to another machine](/reference/faq#exporting-hermes-to-another-machine)), or bring over a single agent with `hermes profile import` (see [Moving a single profile to another machine](/reference/faq#moving-a-single-profile-to-another-machine)). Note that a profile export excludes credentials by design, so an export alone is not a full backup — [`hermes backup` vs `hermes profile export`](/reference/faq#hermes-backup-vs-hermes-profile-export) explains which to use.
:::

---

## Prerequisites

For the POSIX source script, provide Git, curl, tar, and SHA-256 utilities.
Windows can bootstrap its pinned Git for Windows archive when Git is absent.
An existing uv can bootstrap PM; otherwise the script downloads its verified pin.

Hermes requires **Python 3.14** (`>=3.14,<3.15`). PM selects the managed tool
versions from `pm/lock.json`; it does not adopt arbitrary system Node versions
as the installed runtime.

Source builds can require a native compiler and platform development libraries.
Building Electron from source adds Node native-module requirements. These
build prerequisites do not apply to installing a complete desktop package.
Linux Chromium also requires system libraries supplied by the distribution.

:::tip Nix users
Nix is **no longer an explicitly supported install path** (best-effort only). If you already use Nix (on NixOS, macOS, or Linux), there's a dedicated setup path with a Nix flake, declarative NixOS module, and optional container mode. See the **[Nix & NixOS Setup](./nix-setup.md)** guide.
:::

---

## Manual / Developer Installation

For a source checkout, start with the
[PM developer workflow](/reference/package-management#developer-workflow).
It covers activation, daily commands, dependency refresh, and current bootstrap limits.
[Development Setup](../developer-guide/contributing.md#development-setup) covers the separate test environment and checks.

---

## Non-Sudo / System Service User Installs

Run the source installer as the intended service user. Its home, tool store,
configuration, and launcher must belong to that user.

1. As an administrator, install the source-build prerequisites and any Linux
   libraries needed by the selected browser backend.
2. As the service user, run the regular installer:

   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
   ```

3. Add the actual launcher directory to the service user's shell environment:

   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   ```

4. Run `hermes doctor` from that account. Use the installed wrapper, not a
   hardcoded `venv/bin/hermes` path.
5. For a Linux user service that must survive logout, enable lingering as an administrator:

   ```bash
   sudo loginctl enable-linger SERVICE_USER
   ```

The current source installer does not run Playwright's `--with-deps` step or
provide a package-manager-specific sudo fallback. PM manages tool binaries;
the administrator supplies system libraries. See
[Browser automation](../user-guide/features/browser.md) and
[Messaging Gateway](../user-guide/messaging/index.md).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `hermes: command not found` | Reload your shell (`source ~/.bashrc`) or check PATH |
| `API key not set` | Run `hermes model` to configure your provider, or `hermes config set OPENROUTER_API_KEY your_key` |
| Missing config after update | Run `hermes config check` then `hermes config migrate` |

For more diagnostics, run `hermes doctor` — it will tell you exactly what's missing and how to fix it.

### Symlinked home directories and external storage

Hermes supports a symlinked `HERMES_HOME` and symlinked home subdirectories,
including `hooks`, `skills`, `sessions`, and `logs`. During home initialization,
existing directory links are preserved, and permissions on linked directories
(and descendants such as `logs/curator`) are left to their owner.

If a link target is missing, inaccessible, or not a directory, initialization
stops with a storage error naming the path and link target. Hermes does **not**
replace the link or create its missing target: doing so could write data onto
the local disk while an external or NAS volume is unmounted. Check the reported
link, restore the mount or correct its target, and verify access permissions
before retrying. For a deliberately new dotfiles target, create it yourself only
after confirming the intended storage is available.

`hermes doctor` reports these failures as storage problems, not invalid YAML.
Keep your existing `config.yaml`; running `hermes setup` is not the repair for an
unavailable directory. This is a directory-availability check, not a mount monitor:
an existing directory cannot establish that the intended volume is mounted.

## Install method auto-detection

The update owner depends on the running installation, not only its data home.
Source checkouts use the managed Git update path. Desktop bundles, Docker,
Nix, and Termux packages retain their package owner's update mechanism.
`hermes doctor` reports installation provenance. See
[Updating & Uninstalling](./updating.md) before changing package-owned files.
