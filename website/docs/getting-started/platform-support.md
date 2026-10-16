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

## Tier 2

These platforms are maintained in-tree only as a best effort.
Releases may break them, and we can't promise we'll fix them promptly when they break.

PRs will be accepted to fix issues with them, but they will take precedence below fixing issues with Tier 1 platforms.

| OS / Architecture              | Installation methods                                                 | Notes                                                                        |
| ------------------------------ | -------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Nix** (macOS, Linux, NixOS) | [Nix flake and modules](./nix-setup.md) | Nix owns runtime installation and updates. |
| **Android / [Termux](./termux.md)** (aarch64) | [Signed APT repository](./termux.md), then `pkg install hermes-agent` | Prerelease package with Python, Node, and TUI. Not built by default: only releases cut with `--with termux` (and `windows` and `macos-arm64`) update the APT channel. Run the gateway in a Termux session; Android can terminate background processes. |

### Build targets and support priority

Stable releases build the macOS desktop bundle for Apple Silicon (`arm64`) only.
The Intel macOS (`x64`) bundle and the Termux APT package are **not built by
default**: a release includes them only when it is cut with
`release.py release --with macos-intel` or `--with termux`. A release
without macOS Intel replaces the one shared macOS stable update feed with an
arm64-only feed, so Intel Macs installed from a desktop bundle get no update
until a release is cut with `--with macos-intel`. A release without Termux
leaves the Termux APT channel on its previous release.
The `Hermes-Setup.dmg` bootstrap installer is built for Apple Silicon only, so
on an Intel Mac it reports "not supported on this Mac". Intel Macs can install
the [CLI](./installation.md#linux--macos--wsl2--android-termux) and run
`hermes desktop`, or use a `darwin-x64` bundle from a release that included it.
Linux desktop packaging is disabled in the release workflow, although local
AppImage builds and native Linux PM bundle checks exist.

## Unsupported

These platforms and distribution methods are **not** supported.
We suggest that you migrate to a supported distribution method or platform.
They may be broken right now, they may break more in the future.
PRs to fix them will _not_ be accepted, and any code that keeps compatibility with them may be removed at any point.

- Android / Termux on non-aarch64 devices (aarch64 is [supported](./termux.md) via our APT package)
- installs via the AUR (we might upstream patches if it helps out &lt;3)
- 32-bit x86 macOS. Intel x86_64 desktop bundles are built only for releases cut with `--with macos-intel`.
- installs via `pypi` (e.g. `uv tool install hermes-agent`, `pip install hermes-agent`, etc.)
- installs via `brew` (`brew install hermes-agent`)

If you are using an unsupported distribution method, please read the [the installation guide](./installation.md) to learn how to switch to a supported one.
