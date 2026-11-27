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
| **macOS** (Apple Silicon)                                                     | [Hermes Desktop](https://hermes-agent.nousresearch.com/), [`install.sh`](./installation.md) | The macOS desktop bundle (`Hermes Agent.app`) requires **macOS 13 Ventura or later** on Apple Silicon (`arm64`). Intel Macs and macOS 12 or earlier use [`install.sh`](./installation.md#linux--macos--wsl2). |
| [**Windows 10 / 11**](../user-guide/windows-native.md) (x86_64, aarch64) | [Microsoft Store / MSIX desktop](../user-guide/windows-native.md), [`install.ps1`](./installation.md) | Desktop bundles (Microsoft Store and `.appinstaller` MSIX, x64 or ARM64) require **Windows 11 22H2 or later (`10.0.22621.0`)**. Windows 10 and Windows 11 earlier than 22H2 use [`install.ps1`](./installation.md#windows-native) or [WSL2](../user-guide/windows-wsl-quickstart.md). Optional dependencies have [architecture limits](../user-guide/windows-native.md). |
| **Linux / [WSL2](../user-guide/windows-wsl-quickstart.md)** (x86_64, aarch64) | [`install.sh`](./installation.md)                                                           | We test on the latest Ubuntu and WSL2. If your distro has glibc, systemd, and follows the Filesystem Hierarchy Standard, it's likely to work pretty well. |
| [**Docker Container**](../user-guide/docker.md) (x86_64, aarch64) | [`docker pull`](../user-guide/docker.md)                                                                           | Docker installs do not support `hermes update`. Updating is done by running a new image.                                                                  |

---

## Tier 2

These platforms are maintained in-tree only as a best effort.
Releases may break them, and we can't promise we'll fix them promptly when they break.

PRs will be accepted to fix issues with them, but they will take precedence below fixing issues with Tier 1 platforms.

| OS / Architecture              | Installation methods                                                 | Notes                                                                        |
| ------------------------------ | -------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Nix** (macOS, Linux, NixOS) | [Nix flake and modules](./nix-setup.md) | Nix owns runtime installation and updates. |
| **Android / [Termux](./termux.md)** (aarch64) | [Signed APT repository](./termux.md), then `pkg install hermes-agent` | Prerelease package with Python, Node, and TUI. Not built by default: only releases cut with `--with termux` (and `windows` and `macos-arm64`) update the APT channel. Run the gateway in a Termux session; Android can terminate background processes. |

### Build targets and support priority

Releases build the macOS desktop bundle (`Hermes Agent.app`) for **Apple Silicon
(`arm64`) on macOS 13 Ventura or later** (`LSMinimumSystemVersion`
`13.0`, matching Electron 44's OS floor); no release or CI workflow builds or
tests an Intel macOS (`x64`) bundle. The macOS update feed is arm64-only, so
Intel Macs installed from an older desktop bundle get no update. Intel Macs and
macOS 12 Monterey or earlier should use the
[CLI / source installer (`install.sh`)](./installation.md#linux--macos--wsl2).

Windows desktop bundles (both the Microsoft Store package and the signed
`.appinstaller` MSIX bundle, x64 and ARM64) require **Windows 11 22H2 or later
(`10.0.22621.0`)**. Windows 10 and Windows 11 earlier than 22H2 are supported
through the [Windows source installer (`install.ps1`)](./installation.md#windows-native)
or [WSL2](../user-guide/windows-wsl-quickstart.md).

The Termux APT package is **not built by default**: a release includes it only
when it is cut with `release.py release --with termux`. A release without Termux
leaves the Termux APT channel on its previous release.
The legacy `Hermes-Setup.dmg` bootstrap installer is built for Apple Silicon
only, so on an Intel Mac it reports "not supported on this Mac".
Linux desktop AppImages are built on canary tags as diagnostic artifacts that no
release offers yet; local AppImage builds and native Linux PM bundle checks exist.

## Unsupported

These platforms and distribution methods are **not** supported.
We suggest that you migrate to a supported distribution method or platform.
They may be broken right now, they may break more in the future.
PRs to fix them will _not_ be accepted, and any code that keeps compatibility with them may be removed at any point.

- Android / Termux on non-aarch64 devices (aarch64 is [supported](./termux.md) via our APT package)
- installs via the AUR (we might upstream patches if it helps out &lt;3)
- 32-bit x86 macOS, Intel x86_64 macOS desktop bundles (none are built), and desktop bundles on macOS 12 Monterey or earlier (use [`install.sh`](./installation.md#linux--macos--wsl2) instead)
- Windows desktop bundles (Microsoft Store or `.appinstaller` MSIX) on Windows 10 or Windows 11 earlier than 22H2 (use [`install.ps1`](./installation.md#windows-native) or [WSL2](../user-guide/windows-wsl-quickstart.md) instead)
- installs via `pypi` (e.g. `uv tool install hermes-agent`, `pip install hermes-agent`, etc.)
- installs via `brew` (`brew install hermes-agent`)

If you are using an unsupported distribution method, please read the [the installation guide](./installation.md) to learn how to switch to a supported one.
