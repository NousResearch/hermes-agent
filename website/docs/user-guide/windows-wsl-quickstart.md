---
title: "Windows (WSL2) Guide"
description: "Run Hermes Agent on Windows via WSL2 — setup, filesystem access between Windows and Linux, networking, and common pitfalls"
sidebar_label: "Windows (WSL2)"
sidebar_position: 2
---

# Windows (WSL2) Guide

Hermes Agent now supports **both** native Windows and WSL2.  This page covers the WSL2 path; for the native PowerShell install see the dedicated **[Windows (Native) Guide](./windows-native.md)**.

**When to pick WSL2 over native:**
- You want to use the dashboard's embedded terminal (`/chat` tab) — that pane requires a POSIX PTY and is WSL2-only.
- You're doing POSIX-heavy development work and want your Hermes sessions to share the same filesystem / paths as your dev tools.
- You already have a WSL2 environment and don't want to maintain a second install.

**When native is fine (or better):**
- Interactive chat, gateway (Telegram/Discord/etc.), cron scheduler, browser tool, MCP servers, and most Hermes features all run natively on Windows.
- You don't want to think about crossing the WSL↔Windows boundary every time you reference a file or open a URL.

In WSL2 there are effectively two computers in play: your Windows host, and a Linux VM managed by WSL.  Most confusion comes from not being sure which one you're on at any moment.

This guide covers the parts of that split that specifically affect Hermes: installing WSL2, getting files back and forth between Windows and Linux, networking in both directions, and the pitfalls people actually hit.

:::info 简体中文
A Chinese-language walkthrough of the minimum install path is maintained on this same page — switch via the **language** menu (top right) and select **简体中文**.
:::

## Why WSL2 (vs. native Windows)

The native Windows install runs in Windows directly: your Windows terminal (PowerShell, Windows Terminal, etc.), Windows filesystem paths (`C:\Users\…`), and Windows processes.  Hermes uses Git Bash to run shell commands, which is how Claude Code and other agents handle Windows today — it sidesteps the POSIX-vs-Windows gap without a full rewrite.

WSL2 runs a real Linux kernel in a lightweight VM, so Hermes inside it is essentially identical to running on Ubuntu.  That's valuable when you want a real POSIX environment: `fork`, `/tmp`, UNIX sockets, signal semantics, PTY-backed terminals, shells like `bash`/`zsh`, and tools like `rg`, `git`, `ffmpeg` that behave the way they do on Linux.

Practical consequences of WSL2:

- The Hermes CLI, gateway, sessions, memory, skills, and tool runtimes all live inside the Linux VM.
- Windows programs (browsers, native apps, Chrome with your logged-in profile) live outside it.
- Every time you want the two to talk — share files, open URLs, control Chrome, hit a local model server, expose the Hermes gateway to your phone — you cross a boundary. Those boundaries are what this guide is about.

## Install WSL2

From an **Admin PowerShell** or Windows Terminal:

```powershell
wsl --install
```

On a fresh Windows 10 22H2+ or Windows 11 box this installs the WSL2 kernel, the Virtual Machine Platform feature, and a default Ubuntu distro. Reboot when prompted. After reboot Ubuntu will open and ask for a Linux username + password — this is a **new Linux user**, unrelated to your Windows account.

Verify you're actually on WSL2 (not legacy WSL1):

```powershell
wsl --list --verbose
```

You should see `VERSION  2`. If a distro shows `VERSION  1`, convert it:

```powershell
wsl --set-version Ubuntu 2
wsl --set-default-version 2
```

Hermes does not work reliably on WSL1 — WSL1 translates Linux syscalls on the fly and some behaviors (procfs, signals, network) diverge from real Linux.

### Distro choice

Ubuntu (LTS) is what we test against. Debian works. Arch and NixOS work for people who want them, but the one-line installer assumes a Debian-derived `apt` system — see the [Nix setup guide](/docs/getting-started/nix-setup) for that path.

### Enable systemd (recommended)

The hermes gateway (and anything else you want to keep running) is easier to manage with systemd. On modern WSL, enable it once inside your distro:

```bash
sudo tee /etc/wsl.conf >/dev/null <<'EOF'
[boot]
systemd=true

[interop]
enabled=true
appendWindowsPath=true

[automount]
options = "metadata,umask=22,fmask=11"
EOF
```

Then from PowerShell:

```powershell
wsl --shutdown
```

Reopen your WSL terminal. `ps -p 1 -o comm=` should print `systemd`.

The `metadata` mount option above is important — without it, files on `/mnt/c/...` can't store real Linux permission bits, which breaks things like `chmod +x` on scripts under Windows paths.

### Install Hermes inside WSL

Once you have a WSL2 shell open:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc
hermes
```

The installer treats WSL2 as plain Linux — nothing WSL-specific is needed. See [Installation](/docs/getting-started/installation) for the full layout.

## Filesystem: crossing the Windows ↔ WSL2 boundary

This is the part that trips up the most people. There are **two filesystems**, and where you put your files matters — for performance, correctness, and what tools can see.

### The two directions

| Direction | Path inside | Path you use |
|---|---|---|
| Windows disk, seen from WSL | `C:\Users\you\Documents` | `/mnt/c/Users/you/Documents` |
| WSL disk, seen from Windows | `/home/you/code` | `\\wsl$\Ubuntu\home\you\code` (or `\\wsl.localhost\Ubuntu\...` on newer builds) |

Both are real, both work, but they are **not the same filesystem** — they're bridged by a 9P network protocol under the hood. That has real performance and semantic consequences.

### Where to put Hermes and your projects

**Rule of thumb: keep everything Linux-ish inside the Linux filesystem.**

- Your Hermes install (`~/.hermes/`) — Linux side. The installer already does this.
- Your git repos that you work on from WSL — Linux side (`~/code/...`, `~/projects/...`).
- Your models, datasets, venvs — Linux side.

What you get by following this rule:

- **Fast I/O.** Operations on `/mnt/c/...` go through 9P and are 10–100× slower than native ext4. `git status` on a 10k-file repo that feels instant under `~/code` can take 15+ seconds under `/mnt/c`.
- **Correct permissions.** Linux permission bits are a best-effort emulation on `/mnt/c`. Things like `ssh` refusing a key with "bad permissions" or `chmod +x` silently failing are common.
- **Reliable file watchers.** inotify across 9P is flaky — file watchers (dev servers, test runners) routinely miss changes on `/mnt/c`.
- **No case-sensitivity surprises.** Windows paths are case-insensitive by default; Linux is case-sensitive. Projects with both `Readme.md` and `README.md` behave differently depending which side you're on.

Put things on `/mnt/c` only when you **need** a file to live on the Windows side — e.g., you want to open it from a Windows GUI app, or Windows Chrome's DevTools MCP needs the current directory to be a Windows-reachable path.

### Getting files back and forth

**From Windows → into WSL:** easiest is to open Explorer and type `\\wsl.localhost\Ubuntu` in the address bar. You can then drag-drop into `\home\<you>\...`. Or from PowerShell:

```powershell
wsl cp /mnt/c/Users/you/Downloads/file.pdf ~/incoming/
```

**From WSL → into Windows:** copy to `/mnt/c/Users/<you>/...` and it shows up in Windows Explorer immediately:

```bash
cp ~/reports/output.pdf /mnt/c/Users/you/Desktop/
```

**Open a WSL file in a Windows app** (GUI editor, browser, etc.): use `explorer.exe` or `wslview`:

```bash
sudo apt install wslu     # once — gives you wslview, wslpath, wslopen, etc.
wslview ~/reports/output.pdf    # opens with the Windows default handler
explorer.exe .                  # opens the current WSL dir in Windows Explorer
```

**Convert paths between the two universes:**

```bash
wslpath -w ~/code/project        # → \\wsl.localhost\Ubuntu\home\you\code\project
wslpath -u 'C:\Users\you'        # → /mnt/c/Users/you
```

### Line endings, BOMs, and git

If you edit files on the Windows side with a Windows editor, they may get `CRLF` line endings. When `bash` or Python on the Linux side reads them, shell scripts break with `bad interpreter: /bin/bash^M` and Python can fail on BOM'd `.env` files.

The fix is a sane git config inside WSL (not on Windows):

```bash
git config --global core.autocrlf input
git config --global core.eol lf
```

For files that already have CRLF:

```bash
sudo apt install dos2unix
dos2unix path/to/script.sh
```

### "Clone inside WSL or on `/mnt/c`?"

Clone inside WSL. Always, unless you have a specific reason not to. A typical Hermes workflow (`hermes chat`, tool calls that `rg`/`ripgrep` the repo, file watchers, background gateway) will be dramatically faster and more reliable against `~/code/myrepo` than `/mnt/c/Users/you/myrepo`.

One exception: **MCP bridges that launch Windows binaries.** If you're using `chrome-devtools-mcp` through `cmd.exe` (see [MCP guide: WSL → Windows Chrome](/docs/guides/use-mcp-with-hermes#wsl2-bridge-hermes-in-wsl-to-windows-chrome)), Windows may complain with a `UNC` warning if Hermes's current working directory is `~`. In that case, start Hermes from somewhere under `/mnt/c/` so the Windows process has a drive-letter cwd.

## Networking: WSL ↔ Windows

WSL2 runs in a lightweight VM with its own network stack. That means `localhost` inside WSL is **not the same as** `localhost` on Windows — they're two separate hosts from the network's point of view. You need to decide, for each service, which direction traffic flows and pick the right bridge.

Two cases come up constantly.

### Case 1 — Hermes in WSL talks to a service on Windows

Most common: you're running **Ollama, LM Studio, or a llama-server on Windows**, and Hermes (inside WSL) needs to hit it.

The canonical how-to for this lives in the providers guide: **[WSL2 Networking for Local Models →](/docs/integrations/providers#wsl2-networking-windows-users)**

Short version:

- **Windows 11 22H2+:** turn on mirrored networking mode (`networkingMode=mirrored` in `%USERPROFILE%\.wslconfig`, then `wsl --shutdown`). `localhost` then works in both directions.
- **Windows 10 or older builds:** use the Windows host IP (the default gateway of WSL's virtual network) and make sure the server on Windows binds to `0.0.0.0`, not just `127.0.0.1`. Windows Firewall usually also needs a rule for the port.

For the full table (Ollama / LM Studio / vLLM / SGLang bind addresses, firewall rule one-liners, dynamic IP helpers, Hyper-V firewall workaround), follow the link above — don't duplicate it.

### Case 2 — Something on Windows (or your LAN) talks to Hermes in WSL

This is the reverse direction and is less documented elsewhere, but it's what you need for:

- Using the Hermes **web dashboard** from a Windows browser.
- Using the **API server** (`hermes api`) from a Windows-side tool.
- Testing a **messaging gateway** (Telegram, Discord, etc.) where the platform pings a local webhook URL — usually you'd use `cloudflared`/`ngrok` rather than raw port forwarding.

#### Subcase 2a: from the Windows host itself

On **Windows 11 22H2+ with mirrored mode enabled**, there is nothing to do. A process in WSL that binds to `0.0.0.0:8080` (or even `127.0.0.1:8080`) is reachable from a Windows browser at `http://localhost:8080`. WSL publishes the bind back to the host automatically.

On **NAT mode** (Windows 10 / older Windows 11), the default "localhost forwarding" in WSL2 will generally forward Linux-side `127.0.0.1` binds to Windows `localhost`, so a Hermes service started with `--host 127.0.0.1` is usually reachable as `http://localhost:PORT` from Windows. If it isn't:

- Bind to `0.0.0.0` explicitly inside WSL.
- Find the WSL VM's IP with `ip -4 addr show eth0 | grep inet` and hit that from Windows.

#### Subcase 2b: from another device on your LAN (phone, tablet, another PC)

This is the real pain. Traffic flows **LAN device → Windows host → WSL VM**, and you have to set up both hops:

1. **Bind on all interfaces inside WSL.** A process listening on `127.0.0.1` will never be reachable from outside the VM. Use `0.0.0.0`.

2. **Port-forward Windows → WSL VM.** In mirrored mode this is automatic. In NAT mode you have to do it yourself, per port, in Admin PowerShell:

   ```powershell
   # Grab the WSL VM's current IP (it changes on every WSL restart under NAT)
   $wslIp = (wsl hostname -I).Trim().Split(' ')[0]

   # Forward Windows port 8080 → WSL:8080
   netsh interface portproxy add v4tov4 `
     listenaddress=0.0.0.0 listenport=8080 `
     connectaddress=$wslIp connectport=8080

   # Allow it through Windows Firewall
   New-NetFirewallRule -DisplayName "Hermes WSL 8080" `
     -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
   ```

   Remove later with `netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8080`.

3. **Point the LAN device at `http://<windows-lan-ip>:8080`.**

Because the WSL VM IP drifts on each restart in NAT mode, a one-shot rule survives only until the next `wsl --shutdown`. For anything persistent, either use mirrored mode or put the port-proxy step in a script that runs at Windows login.

For webhooks from cloud messaging providers (Telegram `setWebhook`, Slack events, etc.), don't fight port-forwarding — use `cloudflared` tunnels. See the [webhooks guide](/docs/user-guide/messaging/webhooks).

## Running Hermes services long-term on Windows

The Hermes [Tool Gateway](/docs/user-guide/features/tool-gateway) and the API server are long-lived processes. In WSL2 you have a few options for keeping them up.

### Inside WSL with systemd (recommended)

If you enabled systemd per the setup section above, `hermes gateway` and the API server work the way they do on any Linux machine. Use the gateway setup wizard:

```bash
hermes gateway setup
```

It will offer to install a systemd user unit so the gateway comes up automatically when WSL starts.

### Making WSL itself start on Windows login

WSL's VM only stays alive while something is using it. To keep your gateway reachable without a terminal window open, boot a WSL process at Windows login via Task Scheduler:

- **Trigger:** At log on (your user).
- **Action:** Start a program
  - Program: `C:\Windows\System32\wsl.exe`
  - Arguments: `-d Ubuntu --exec /bin/sh -c "sleep infinity"`

That keeps the VM alive so the systemd-managed gateway stays running. On Windows 11, the newer `wsl --install --no-launch` + auto-start flows also work; the `sleep infinity` trick is the portable version.

## GPU passthrough (local models)

WSL2 supports **NVIDIA** GPUs natively since WSL kernel 5.10.43+ — install the standard NVIDIA driver on Windows (do **not** install a Linux NVIDIA driver inside WSL), and `nvidia-smi` inside WSL will see the GPU. From there, CUDA toolkits, `torch`, `vllm`, `sglang`, and `llama-server` build against the real GPU as usual.

AMD ROCm and Intel Arc support inside WSL2 is still evolving and outside Hermes's test matrix — it may work with current drivers but we don't have a recipe to recommend.

If you're running a **Windows-native** local-model server (Ollama for Windows, LM Studio) that already uses your GPU through Windows drivers, you don't need WSL GPU passthrough at all — just follow Case 1 above and hit it over the network from WSL.

## Common pitfalls

**"Connection refused" to my Windows-hosted Ollama / LM Studio.**
See [WSL2 Networking](/docs/integrations/providers#wsl2-networking-windows-users). Ninety percent of the time the server is bound to `127.0.0.1` and needs `0.0.0.0` (Ollama: `OLLAMA_HOST=0.0.0.0`), or you're missing a firewall rule.

**Massive slowness on `git status` / `hermes chat` in a repo.**
You're probably working under `/mnt/c/...`. Move the repo to `~/code/...` (Linux side). Order-of-magnitude faster.

**`bad interpreter: /bin/bash^M` on scripts.**
CRLF line endings from a Windows editor. `dos2unix script.sh`, and set `core.autocrlf input` in your WSL git config.

**"UNC paths are not supported" warning from Windows binaries launched via MCP.**
Hermes's cwd is inside the Linux filesystem, and Windows `cmd.exe` doesn't know what to do with it. Start Hermes from `/mnt/c/...` for that session, or use a wrapper that `cd`s to a Windows-reachable path before invoking the Windows executable.

**Clock drift after sleep/hibernate.**
WSL2's clock can lag by minutes after the host resumes from sleep, which breaks anything cert-based (OAuth, HTTPS APIs). Fix it on demand:

```bash
sudo hwclock -s
```

Or install `ntpdate` and run it at login.

**DNS stops working after enabling mirrored mode, or when a VPN is connected.**
Mirrored mode proxies host network settings into WSL — if Windows DNS is funky (VPN split-tunnel, corporate resolver), WSL inherits that. Workaround: override `resolv.conf` manually (set `generateResolvConf=false` in `/etc/wsl.conf`, then write your own `/etc/resolv.conf` with `1.1.1.1` or your VPN's DNS).

**`hermes` not found after running the installer.**
The installer adds `~/.local/bin` to your shell's PATH via `~/.bashrc`. You need to `source ~/.bashrc` (or open a new terminal) for it to take effect in the current session.

**Windows Defender is slow on WSL files.**
Defender scans files via the 9P bridge when accessed from Windows, which magnifies the slowness of `/mnt/c`-style cross-boundary access. If you only touch WSL files from inside WSL, this doesn't matter. If you use Windows tools against `\\wsl$\...` frequently, consider excluding the WSL distro path from real-time scanning.

**Running out of disk.**
WSL2 stores its VM disk as a sparse VHDX under `%LOCALAPPDATA%\Packages\...`. It grows but doesn't auto-shrink when you delete files. To reclaim space: `wsl --shutdown`, then from an Admin PowerShell run `Optimize-VHD -Path <path-to-ext4.vhdx> -Mode Full` (requires Hyper-V tools) — or the simpler `diskpart` path documented on the WSL docs.

## Where to go next

- **[Installation](/docs/getting-started/installation)** — actual install steps (Linux/WSL2/Termux all use the same installer).
- **[Integrations → Providers → WSL2 Networking](/docs/integrations/providers#wsl2-networking-windows-users)** — the canonical networking deep-dive for local model servers.
- **[MCP guide → WSL → Windows Chrome](/docs/guides/use-mcp-with-hermes#wsl2-bridge-hermes-in-wsl-to-windows-chrome)** — controlling your signed-in Windows Chrome from Hermes in WSL.
- **[Tool Gateway](/docs/user-guide/features/tool-gateway)** and **[Web Dashboard](/docs/user-guide/features/web-dashboard)** — the long-lived services you'll most often want to expose from WSL to the rest of your network.
