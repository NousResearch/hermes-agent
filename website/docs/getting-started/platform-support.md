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
| **macOS** (Apple Silicon)                                                     | [Hermes Desktop](https://hermes-agent.nousresearch.com/), [`install.sh`](./installation.md#linux--macos--wsl2--android-termux) |
| [**Windows 10 / 11**](../user-guide/windows-native.md) (x86_64, aarch64)      | [Hermes Desktop](https://hermes-agent.nousresearch.com/), [`install.ps1`](./installation.md#windows-native)                    | A few features are [not available](../user-guide/windows-native.md#feature-matrix).                                                                       |
| **Linux / [WSL2](../user-guide/windows-wsl-quickstart.md)** (x86_64, aarch64) | [`install.sh`](./installation.md#linux--macos--wsl2--android-termux)                                                           | We test on the latest Ubuntu and WSL2. If your distro has glibc, systemd, and follows the Filesystem Hierarchy Standard, it's likely to work pretty well. |
| [**Docker Container**](../user-guide/docker.md#quick-start) (x86_64, aarch64) | [`docker pull`](../user-guide/docker.md#quick-start)                                                                           | Docker installs do not support `hermes update`. Updating is done by running a new image.                                                                  |

---

## Tier 2

These platforms are maintained in-tree only as a best effort.
Releases may break them, and we can't promise we'll fix them promptly when they break.

PRs will be accepted to fix issues with them, but they will take precedence below fixing issues with Tier 1 platforms.

| OS / Architecture              | Installation methods                                                 | Notes                                                                        |
| ------------------------------ | -------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Android (Termux)** (aarch64) | [`install.sh`](./installation.md#linux--macos--wsl2--android-termux) | A few features are [not available](./termux.md#known-limitations-on-phones). |
| **Nix** (MacOS, Linux, NixOS)  | [`install.sh`](./nix-setup.md)                                       | Breaks often due to node.js packaging woes. Best of luck~! &lt;3             |

## Unsupported

These platforms and distribution methods are **not** supported.
We suggest that you migrate to a supported distribution method or platform.
They may be broken right now, they may break more in the future.
PRs to fix them will _not_ be accepted, and any code that keeps compatibility with them may be removed at any point.

- installs via the AUR (we might upstream patches if it helps out &lt;3)
- macOS on x86 (Intel) processors
- installs via `pypi` (e.g. `uv tool install hermes-agent`, `pip install hermes-agent`, etc.)
- installs via `brew` (`brew install hermes-agent`)

If you are using an unsupported distribution method, please read the [the installation guide](./installation.md) to learn how to switch to a supported one.
