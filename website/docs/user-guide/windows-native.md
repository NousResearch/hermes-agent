---
title: "Windows (Native) Guide"
description: "Run Hermes Agent natively on Windows 10 / 11 — install, feature matrix, UTF-8 console, Git Bash, gateway as a Scheduled Task, editor handling, PATH, uninstall, and common pitfalls"
sidebar_label: "Windows (Native)"
sidebar_position: 3
---

# Windows (Native) Guide

Hermes runs natively on Windows 10 and Windows 11 — no WSL, no Cygwin, no Docker. This page is the deep dive: what works natively, what's WSL-only, what the installer actually does, and the Windows-specific knobs you might need to touch.

If you just want to install, the one-liner on the [landing page](/) or [Installation page](../getting-started/installation#windows-native) is all you need. Come back here when something surprises you.

:::tip Want WSL instead?
If you prefer a POSIX environment for `fork` semantics or Linux-style file watchers, see the **[Windows (WSL2) Guide](./windows-wsl-quickstart.md)**. Both coexist cleanly: native data lives under `%LOCALAPPDATA%\hermes`, WSL data lives under `~/.hermes`.
:::

## Quick install

Open **PowerShell** (or Windows Terminal) and run:

```powershell
iex (irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1)
```

No admin rights required. The installer goes to `%LOCALAPPDATA%\hermes\` and adds `hermes` to your **User PATH** — open a new terminal after it finishes.

**Installer options** use a scriptblock:

```powershell
& ([scriptblock]::Create((irm https://hermes-agent.nousresearch.com/install.ps1))) -NonInteractive -Branch main
```

| Parameter | Purpose |
|---|---|
| `-Branch NAME` | Select the source branch; default `main`. |
| `-Commit SHA` | Select a commit after the branch checkout. |
| `-HermesHome PATH` | Select the data directory. |
| `-InstallDir PATH` | Select the source checkout directory. |
| `-NonInteractive` | Skip setup and gateway stages that need input. |
| `-IncludeDesktop` | Build the desktop app and create shortcuts. |
| `-ShowResolvedPaths` | Print resolved paths as JSON without installing. |
| `-Manifest` / `-ProtocolVersion` | Inspect the stage protocol used by the bootstrap GUI. |
| `-Stage NAME -Json` | Run one stage and emit its result frame. |

The current script does not accept `-NoVenv`, `-SkipSetup`, or `-Tag`.
To diagnose an unexpected short Windows path, use `-ShowResolvedPaths` first.

### MSIX / App Installer and Microsoft Store

The bundled desktop is separate from the source script. Its MSIX package
requires **Windows 11 22H2 or later**. Windows 10 source-script support does
not mean the MSIX package supports Windows 10.

Open the downloaded `.appinstaller` file with Windows App Installer. It installs
a signed universal bundle and records the update source. The package includes
Python, Node, supported dependencies, and prebuilt interfaces. It does not clone
a checkout or build the base runtime on first launch.

The MSIX execution aliases expose `hermes`, `hermes-agent`, and `hermes-acp`.
If another installation shadows an alias, inspect `Get-Command hermes -All`.
Windows Settings → Apps → Advanced app settings → App execution aliases
controls the aliases.

Sideload updates use the app's Update control and Windows App Installer.
Hermes downloads a local descriptor before teardown and registers automatic
relaunch. It does not require the `ms-appinstaller:` URL protocol. An unknown
update-check result is not a claim that the package is current.

The Microsoft Store variant uses its Partner Center package identity and Store
updates. It does not use the sideload feed. `hermes update` inside either
bundled runtime does not run Git against package files.

`Hermes-Setup.exe` is a different, bootstrap installer. It provisions a source
checkout through the scripts. Do not confuse it with the self-contained MSIX
package. See [Updating & Uninstalling](../getting-started/updating.md).

### Dependency bootstrap

PM owns managed tools. `hermes_cli/dep_ensure.py` maps feature requirements to
PM packages instead of invoking `install.ps1 -Ensure`. Already installed tools
come from PM facts; missing optional tools follow the lazy-install policy.

```powershell
hermes pm doctor
hermes pm install
```

## What the source installer does

1. Locate Git, or stage the verified Git for Windows pin when Git is absent.
2. Clone the selected repository branch and apply an optional commit pin.
3. Bootstrap uv and create the initial Python environment.
4. Run PM to provision Python 3.14, required tools, and the `all` Python extra.
5. Mint CLI launchers in the data home's `bin` directory and add it to User PATH.
6. Prepare configuration and invoke the interactive setup/gateway stages unless skipped.
7. If requested, build the desktop and create Start Menu/Desktop shortcuts.
8. Write the bootstrap-completion marker.

The runtime launcher executes PM's store Python and selects the dependency
environment before imports. PM can publish a new writable environment without
replacing libraries already loaded by a running process. There is no tiered
pip fallback to silently reduce the installed feature set.

:::tip Skip provider hunting on Windows
On Windows, per-tool API key setup (Firecrawl, FAL, Browser Use, OpenAI TTS) is the highest-friction part of getting a useful agent. A [Nous Portal](/user-guide/features/tool-gateway) subscription covers the model **and** all of those tools through one OAuth login. After the installer finishes, run `hermes setup --portal` to wire everything up.
:::

## Feature matrix

Windows support is feature- and architecture-specific. The base interfaces run
natively, but some optional SDKs are excluded from particular targets.

| Feature | Native Windows | WSL2 |
|---|---|---|
| CLI (`hermes chat`, `hermes setup`, `hermes gateway`, …) | ✓ | ✓ |
| Interactive TUI (`hermes --tui`) | ✓ | ✓ |
| Messaging gateway (Telegram, Discord, Slack, WhatsApp, 15+ platforms) | ✓ | ✓ |
| Cron scheduler | ✓ | ✓ |
| Browser tool (Chromium via Node) | ✓ | ✓ |
| MCP servers (stdio and HTTP) | ✓ | ✓ |
| Local Ollama / LM Studio / llama-server | ✓ | ✓ (via WSL networking) |
| Web dashboard (sessions, jobs, metrics, config) | ✓ | ✓ |
| Dashboard `/chat` embedded terminal pane | ConPTY through `pywinpty` | POSIX PTY |
| Auto-start at login | ✓ (schtasks) | ✓ (systemd) |

The dashboard uses its `pywinpty`/ConPTY bridge on Windows and `ptyprocess`
on POSIX. A missing or broken native dependency can make the terminal
unavailable; WSL is an alternative, not a requirement of the current design.

### Optional dependency limits

- Matrix's native encrypted adapter is Linux-only; use a supported proxy route
  or a Linux backend on Windows.
- Native Windows ARM64 excludes the `mem0` and `google-chat` SDK extras, and
  the openWakeWord engine. Sherpa supports native Windows ARM64 and is the
  automatic wake-word default on that target.
- Local Faster-Whisper STT is excluded on native Windows ARM64. Use a cloud
  or command-based STT provider. Porcupine remains a wake-engine alternative.

The platform markers in `pyproject.toml` define the packaged dependency set.
A general gateway or voice feature claim does not override those markers.

## How Hermes runs shell commands on Windows

Hermes's terminal tool runs commands through **Git Bash**, same strategy Claude Code uses. This sidesteps the POSIX-vs-Windows gap without rewriting every tool.

`pm.shell()` owns Bash resolution. It first checks the Git package recorded in
PM facts, then the provisioned `PATH`. If a PATH candidate belongs to a WindowsApps
package, the resolver prefers a conventional Git for Windows installation when available.

Packaged tools are not general-purpose host installations. An external Python
process can fail to start a WindowsApps payload executable with `WinError 5`.
Use the package's own launcher, or use conventional tools for a source checkout.
Do not disable Windows security controls to work around that boundary.

The current installer does not set `HERMES_GIT_BASH_PATH`. MinGit is not a
replacement for Git for Windows with Bash.

## UTF-8 console on Windows

Python's default stdio on Windows uses the console's active code page (usually cp1252 or cp437). Hermes's banner, slash-command list, tool feed, Rich panels, and skill descriptions all contain Unicode. Without intervention, any of that crashes with `UnicodeEncodeError: 'charmap' codec can't encode character…`.

The fix is in `hermes_cli/stdio.py::configure_windows_stdio()`, called early in every entry point (`cli.py::main`, `hermes_cli/main.py::main`, `gateway/run.py::main`). It:

1. Flips the console code page to CP_UTF8 (65001) via `kernel32.SetConsoleCP` / `SetConsoleOutputCP`.
2. Reconfigures `sys.stdout` / `sys.stderr` / `sys.stdin` to UTF-8 with `errors='replace'`.
3. Sets `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` (via `setdefault`, so explicit user values win) so child Python subprocesses inherit UTF-8.
4. Sets `EDITOR=notepad` if neither `EDITOR` nor `VISUAL` is set (see the Editor section below).

Idempotent. No-op on non-Windows.

**Opt out:** `HERMES_DISABLE_WINDOWS_UTF8=1` in the environment falls back to the legacy cp1252 stdio path. Useful for bisecting an encoding bug; unlikely to be the right setting in normal operation.

## The editor (`Ctrl-X Ctrl-E`, `/edit`)

Pre-#21561, pressing `Ctrl-X Ctrl-E` or typing `/edit` silently did nothing on Windows. prompt_toolkit has a hardcoded POSIX-absolute fallback list (`/usr/bin/nano`, `/usr/bin/pico`, `/usr/bin/vi`, …) that never resolves on Windows — even with full Git for Windows installed.

Hermes's Windows stdio shim now sets `EDITOR=notepad` as a default. Notepad ships with every Windows install and works as a blocking editor — `subprocess.call(["notepad", file])` blocks until the window closes.

**User overrides still win** (they're checked before the setdefault):

| Editor | PowerShell command |
|---|---|
| VS Code | `$env:EDITOR = "code --wait"` |
| Notepad++ | `$env:EDITOR = "'C:\Program Files\Notepad++\notepad++.exe' -multiInst -nosession"` |
| Neovim | `$env:EDITOR = "nvim"` |
| Helix | `$env:EDITOR = "hx"` |

The `--wait` flag on VS Code is critical — without it the editor returns immediately and Hermes gets a blank buffer back.

Set it permanently in your PowerShell profile:

```powershell
# In $PROFILE
$env:EDITOR = "code --wait"
```

Or as a User environment variable in System Settings so every new shell picks it up.

## `Ctrl+Enter` for newline in the CLI

Windows Terminal passes `Ctrl+Enter` through as a dedicated key sequence. Hermes binds it to "insert newline" so you can compose multi-line prompts in the CLI without falling back to `Esc`-then-`Enter`. Works in Windows Terminal, VS Code integrated terminal, and any modern Windows console host that honors VT escape sequences.

On legacy `cmd.exe` consoles `Ctrl+Enter` collapses to plain `Enter` — use `Esc Enter` instead, or upgrade to Windows Terminal (it's free and installed by default on Windows 11).

## Running the gateway at Windows login

`hermes gateway install` on Windows uses **Scheduled Tasks** with a Startup-folder fallback — no admin required.

### Install

```powershell
hermes gateway install
```

What happens under the hood:

1. `schtasks /Create /SC ONLOGON /RL LIMITED /TN Hermes_Gateway` — registers a task that runs at your login with standard (non-elevated) permissions. No UAC prompt.
2. If schtasks is blocked by group policy, falls back to writing a small `Hermes_Gateway.vbs` launcher (run hidden via `wscript.exe`) into `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`. Same effect, slightly cruder. A VBScript is used rather than a `cmd.exe` shortcut because a console allocated at logon can receive a close event that kills the gateway before it finishes starting.
3. Spawns the gateway **detached via `pythonw.exe`** — not `python.exe`. `pythonw.exe` has no console attached, which immunizes it against `CTRL_C_EVENT` broadcasts from sibling processes (a real issue that used to kill the gateway when you Ctrl+C'd anything in the same process group).

Flags used when spawning: `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB`.

### Manage

```powershell
hermes gateway status      # Merged view: schtasks + Startup folder + running PID
hermes gateway start       # Starts the scheduled task now
hermes gateway stop        # Graceful SIGTERM equivalent (TerminateProcess via psutil)
hermes gateway restart
hermes gateway uninstall   # Removes schtasks entry, Startup shortcut, pid file
```

`hermes gateway status` is idempotent — call it a thousand times in a row and it will never accidentally kill the gateway. (Pre-PR #21561 it silently did, via `os.kill(pid, 0)` colliding with `CTRL_C_EVENT` at the C level — see "process management internals" below if you care about the story.)

### Why not a Windows Service?

Services require admin rights to install and tie the gateway's lifecycle to machine boot, not user login. The typical Hermes user wants: log in → gateway available, log out → gateway gone. Scheduled Tasks do exactly that without elevation. If you genuinely want a service, use `nssm` or `sc create` manually — but you probably don't.

## Data layout

| Path | Contents |
|---|---|
| `%LOCALAPPDATA%\hermes\hermes-agent\` | Source checkout for the script installation; absent from an MSIX-only install. |
| `%LOCALAPPDATA%\hermes\tools\` | Writable managed-tool store. MSIX base tools remain inside the package. |
| `%LOCALAPPDATA%\hermes\installs\` | Per-install runtime selection, journals, and Python generations. |
| `%LOCALAPPDATA%\hermes\bin\` | Source-install CLI launchers. MSIX instead provides execution aliases. |
| `%LOCALAPPDATA%\hermes\` | User configuration, credentials, sessions, plugins, skills, and logs. |

These are default paths. `HERMES_HOME` and installer path arguments can change
them. A full deletion of `%LOCALAPPDATA%\hermes` also deletes user data and
can affect other installations that share it. Use the uninstall command or
Windows package removal instead of deleting that root to repair an app.

## Browser tool

Browser setup depends on the selected backend. PM supplies the pinned
`agent-browser` and Chromium packages for the built-in backend. Browser Use
has its own managed CLI installation through `hermes tools`. A self-contained
MSIX includes supported browser tools in its payload.

On Windows ARM64, the pinned Chromium and `agent-browser` binaries can use
Windows' x64 emulation. This differs from the native ARM64 Python runtime.
See [Browser automation](./features/browser.md) for backend selection.

## Running Hermes on Windows — practical notes

### PATH after install

The installer adds `%LOCALAPPDATA%\hermes\bin` to your **User PATH** via `[Environment]::SetEnvironmentVariable`. Existing terminals don't pick this up — open a new PowerShell window (or Windows Terminal tab) after installation. Close-and-reopen, don't `$env:PATH += …` by hand unless you know what you're doing.

Verify:

```powershell
Get-Command hermes        # should print C:\Users\<you>\AppData\Local\hermes\bin\hermes.exe
hermes --version
```

### Environment variables

Hermes honors both `$env:X` (process-scope) and User environment variables (permanent, set in System Properties → Environment Variables). Setting API keys in `%LOCALAPPDATA%\hermes\.env` (your `HERMES_HOME`) is the normal path — same as Linux:

```
OPENROUTER_API_KEY=sk-or-...
TELEGRAM_BOT_TOKEN=...
```

Don't put secrets in User environment variables unless you specifically want every Windows process to see them (it isn't what you want).

### Windows-specific env vars

These only affect native Windows installs:

| Variable | Effect |
|---|---|
| `HERMES_DISABLE_WINDOWS_UTF8` | Set to `1` to disable the UTF-8 stdio shim and fall back to the locale code page. Useful for bisecting an encoding bug. |
| `EDITOR` / `VISUAL` | Your editor for `/edit` and `Ctrl-X Ctrl-E`. Hermes defaults to `notepad` if both are unset. |

## Uninstall

From PowerShell:

```powershell
hermes uninstall
```

For source installs, the uninstaller removes owned launchers, service entries,
and application files. Review `hermes uninstall --dry-run` before removal.
`--full` also removes data; `--data` removes data without removing packaged code.
For MSIX or Store installations, remove the app through Windows Settings →
Apps → Installed apps. The CLI refuses to delete package-owned code.

:::caution User-data deletion
Before deleting data, stop every Hermes process that uses the selected `HERMES_HOME` and make a backup.
Review `hermes uninstall --dry-run` before choosing a data-removal mode.
Do not recursively delete the default data root to repair one application or profile.
A custom `HERMES_HOME` can be elsewhere, and package removal does not remove that data.
:::

The `hermes uninstall` CLI subcommand also handles the case where the schtasks entry was registered under a different task name (older installs) — it searches by install path rather than by hardcoded task name.

## Process management internals

This is background material — skip unless you're debugging an "it's killing itself" weirdness.

On Linux and macOS, the POSIX idiom `os.kill(pid, 0)` is a no-op permission check: "is this PID alive and can I signal it?" On Windows, Python's `os.kill` maps `sig=0` to `CTRL_C_EVENT` — they collide at integer value 0 — and routes it through `GenerateConsoleCtrlEvent(0, pid)`, which broadcasts Ctrl+C to the **entire console process group** containing the target PID. That's [bpo-14484](https://bugs.python.org/issue14484), open since 2012. It won't be fixed because changing it would break scripts that depend on the current behavior.

Consequence: any codepath that said "check if this PID is alive" via `os.kill(pid, 0)` on Windows was silently killing the target. Hermes migrated every such site (14 across 11 files) to `gateway.status._pid_exists()`, which uses `psutil.pid_exists()` (which in turn uses `OpenProcess + GetExitCodeProcess` on Windows — no signals). If you're writing a plugin or patch, use `psutil.pid_exists()` directly or `gateway.status._pid_exists()` — never `os.kill(pid, 0)`.

`scripts/check-windows-footguns.py` enforces this in CI: any new `os.kill(pid, 0)` call fails the `Windows footguns (blocking)` check unless the line carries a `# windows-footgun: ok — <reason>` marker.

## Common pitfalls

**`hermes: command not found` right after install.**
Open a new PowerShell window. The installer added `%LOCALAPPDATA%\hermes\bin` to User PATH, but existing shells need to be restarted to pick it up. In the meantime you can run `& "$env:LOCALAPPDATA\hermes\bin\hermes.exe"`.

**`WinError 193: %1 is not a valid Win32 application` when running a tool.**
You hit a shebang-script invocation that bypassed the `.cmd` shim. Hermes resolves commands through `shutil.which(cmd, path=local_bin)` so PATHEXT picks up `.CMD` — if you're invoking the tool via a hardcoded path instead, switch to the `.cmd` variant (e.g., `npx.cmd`, not `npx`).

**`[scriptblock]::Create(...)` fails with `The assignment expression is not valid`.**
Your download of `install.ps1` picked up a UTF-8 BOM. The `irm | iex` form strips BOMs automatically; `[scriptblock]::Create((irm ...))` does not. Re-run with the simple `irm | iex` form, or download the script manually and save it without a BOM via `[IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))`.

**Gateway won't stay running after restart.**
Check `hermes gateway status` — it merges the schtasks entry, the Startup-folder shortcut (if used), and the live PID. If schtasks is registered but not running, group policy may be blocking `ONLOGON` triggers. Run `schtasks /Query /TN Hermes_Gateway /V /FO LIST` (`Hermes_Gateway_<profile>` for a named profile) to see the task's failure reason. The Startup-folder fallback engages automatically only when `schtasks` itself fails to register the task; there is no environment variable or flag to force it.

**`/edit` still does nothing after setting `$env:EDITOR`.**
You set it in the current process only; close and reopen the shell, or set it at User scope in System Properties → Environment Variables. Verify with `echo $env:EDITOR` in a new PowerShell window.

**Browser tool launches but tools time out.**
Run `hermes doctor` and `hermes pm doctor`. Use `hermes tools` to inspect the
selected browser backend. Do not install an unrelated Playwright revision into
a signed app payload.

**`agent-browser` reports a Node version error.**
Run `hermes pm doctor` and inspect which Hermes launcher started the process.
PM supplies the managed Node version. Do not delete an unrelated system Node
installation to repair Hermes.

**Chinese / Japanese / Arabic characters show as `?` in the CLI.**
The UTF-8 stdio shim didn't activate. Check that `HERMES_DISABLE_WINDOWS_UTF8` is NOT set (`Get-ChildItem env:HERMES_DISABLE_WINDOWS_UTF8`). If it's empty and you still see `?`, the console host (very old `cmd.exe`) may not support UTF-8 at all — switch to Windows Terminal.

**Gateway can't send Telegram photos — "`BadRequest: payload contains invalid characters`".**
This is unrelated to Windows but sometimes surfaces first there. Usually it means your file path contains unescaped backslashes in a JSON body. Telegram should be receiving paths Hermes normalizes, not raw Windows paths — if you're seeing this inside a custom plugin, make sure you're passing the Hermes-provided path, not `str(Path(...))` from user input.

**"Works on my other machine" encoding weirdness after `git pull`.**
If you edited Hermes config or a skill on Windows using a non-UTF-8 editor (Notepad on older Windows versions, some Chinese IMEs), the file may have been saved with a BOM. Hermes tolerates `utf-8-sig` on most config reads, but a BOM inside a folded YAML scalar (`description: >`) silently breaks YAML parsing. Re-save the file as plain UTF-8 without BOM.

## Where to go next

- **[Installation](../getting-started/installation.md)** — the full install page, including Linux/macOS/WSL2.
- **[Windows (WSL2) Guide](./windows-wsl-quickstart.md)** — if you want POSIX semantics or the dashboard terminal pane.
- **[CLI Reference](../reference/cli-commands.md)** — every `hermes` subcommand.
- **[FAQ](../reference/faq.md)** — common non-Windows-specific questions.
- **[Messaging Gateway](./messaging/index.md)** — running Telegram/Discord/Slack on Windows.
