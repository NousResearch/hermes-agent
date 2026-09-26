---
sidebar_position: 2.5
title: "Platform Support"
description: "Which operating systems, distribution methods, and features Hermes Agent supports."
---

# Platform Support

Hermes Agent maintains support for many platforms and distribution methods, but we can't support every possible install method.

---

## Tier 1

We strive to never break installations and updates for these. Issues & regressions in Tier 1 are our first priority and take precedence over other platforms.

| OS / Architecture                                                             | Installation methods                                                                                                           | Notes                                                                                                                                                     |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **macOS** (Apple Silicon)                                                     | [Hermes Desktop](https://hermes-agent.nousresearch.com/), [`install.sh`](./installation.md) |
| [**Windows 10 / 11**](../user-guide/windows-native.md) (x86_64, aarch64) | [`install.ps1`](./installation.md), [MSIX desktop](../user-guide/windows-native.md) | The MSIX package requires Windows 11 22H2 or later. Optional dependencies have [architecture limits](../user-guide/windows-native.md). |
| **Linux / [WSL2](../user-guide/windows-wsl-quickstart.md)** (x86_64, aarch64) | [`install.sh`](./installation.md)                                                           | We test on the latest Ubuntu and WSL2. If your distro has glibc, systemd, and follows the Filesystem Hierarchy Standard, it's likely to work pretty well. |
| [**Docker Container**](../user-guide/docker.md) (x86_64, aarch64) | [`docker pull`](../user-guide/docker.md)                                                                           | Docker installs do not support `hermes update`. Updating is done by running a new image.                                                                  |

---

## Windows Native Notes

Native Windows (not WSL2) has a few platform-specific behaviors worth knowing:

### Encoding

Windows consoles default to a legacy code page (cp1252 on Western locales, cp936/GBK on Chinese locales). Hermes reconfigures its own stdio to UTF-8 at startup, so interactive output is safe. But:

- **Subprocess text-mode reads** default to the locale encoding unless the caller passes `encoding='utf-8'` explicitly. The repo enforces this via `scripts/check-windows-footguns.py`.
- **Scheduled Tasks / CI** run with no console at all; `hermes update` and other interactive prompts detect this and take safe defaults rather than blocking.

### File Paths

Windows forbids `:` in path segments. Session-scoped sandbox directory names (e.g. `session:<key>`) are sanitized automatically before use as directory names. Symlink loops are avoided during directory walks.

### Process Management

`signal.SIGKILL` does not exist on Windows; use `gateway.status.terminate_pid(pid, force=True)` which routes to `taskkill /T /F`. `os.kill(pid, 0)` is **not** a liveness probe on Windows — it delivers `CTRL_C_EVENT` and can kill the target. Use `psutil.pid_exists()` or `gateway.status._pid_exists()` instead.

### Linting

Before pushing changes that touch file I/O, subprocesses, or signals, run:

```
python scripts/check-windows-footguns.py --all
```

This catches common cross-platform mistakes (missing `encoding=`, unguarded `signal.SIGKILL`, POSIX-only imports, and more).

### Single-GPU VRAM Contention

On a single-GPU rig running a local model (Ollama, llama.cpp) alongside GPU-bound tools (ComfyUI, Stable Diffusion), the agent's model and the tool compete for VRAM. The degradation is **silent**: `ollama ps` reports `100% GPU` throughout because every layer is still assigned, but free VRAM drops below ~1.5 GB and the WDDM driver demotes compute/KV buffers to system RAM without logging anything. Measured impact on an RTX 3090:

| State | Effect |
|-------|--------|
| Model loaded while ComfyUI renders | SDXL render 562s vs. 3s unloaded (187x slower) |
| ComfyUI rendering while model answers | Prompt eval 24 tok/s vs. 1105 tok/s (45x slower) |

The practical rule: **don't take agent turns while a GPU-bound tool is mid-render on the same card.** Use a GPU-mode switch script (unload model → render → reload) or schedule render checks to avoid agent turns until the render completes.

---

## Tier 2

These platforms are maintained in-tree only as a best effort.
Releases may break them, and we can't promise we'll fix them promptly when they break.

PRs will be accepted to fix issues with them, but they will take precedence below fixing issues with Tier 1 platforms.

| OS / Architecture              | Installation methods                                                 | Notes                                                                        |
| ------------------------------ | -------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Nix** (macOS, Linux, NixOS) | [Nix flake and modules](./nix-setup.md) | Nix owns runtime installation and updates. |
| **Android / [Termux](./termux.md)** (aarch64) | [Signed APT repository](./termux.md), then `pkg install hermes-agent` | Prerelease package with Python, Node, and TUI. Run the gateway in a Termux session; Android can terminate background processes. |

### Build targets and support priority

The native bundle pipeline includes Intel macOS (`x64`) as well as Apple Silicon.
It also defines signed-package update acceptance for both architectures.
That coverage does not change the Tier 1 priority assigned to Apple Silicon.
Linux desktop packaging is disabled in the release workflow, although local
AppImage builds and native Linux PM bundle checks exist.

## Unsupported

These platforms and distribution methods are **not** supported.
We suggest that you migrate to a supported distribution method or platform.
They may be broken right now, they may break more in the future.
PRs to fix them will _not_ be accepted, and any code that keeps compatibility with them may be removed at any point.

- Android / Termux on non-aarch64 devices (aarch64 is [supported](./termux.md) via our APT package)
- installs via the AUR (we might upstream patches if it helps out &lt;3)
- 32-bit x86 macOS. Intel x86_64 has native bundle build and package-update acceptance lanes; this does not change the Tier 1 priority for Apple Silicon.
- installs via `pypi` (e.g. `uv tool install hermes-agent`, `pip install hermes-agent`, etc.)
- installs via `brew` (`brew install hermes-agent`)

If you are using an unsupported distribution method, please read the [the installation guide](./installation.md) to learn how to switch to a supported one.
