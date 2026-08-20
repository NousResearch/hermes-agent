---
sidebar_position: 2.5
title: "平台支持"
description: "Hermes Agent 支持的操作系统、分发方式和功能。"
---

# 平台支持

Hermes Agent 支持多种平台和分发方式，但我们无法支持所有可能的安装方式。

---

## 一级平台

我们致力于不会破坏这些平台的安装和更新。一级平台上的问题和回归修复是最高优先级，优先于其他平台。

| 操作系统 / 架构 | 安装方式 | 说明 |
| --- | --- | --- |
| **macOS**（Apple Silicon） | [Hermes Desktop](https://hermes-agent.nousresearch.com/)、[`install.sh`](./installation.md#linux--macos--wsl2--android-termux) |
| [**Windows 10 / 11**](../user-guide/windows-native.md)（x86_64、aarch64） | [Hermes Desktop](https://hermes-agent.nousresearch.com/)、[`install.ps1`](./installation.md#windows-native) | 少数功能[不可用](../user-guide/windows-native.md#feature-matrix)。 |
| **Linux / [WSL2](../user-guide/windows-wsl-quickstart.md)**（x86_64、aarch64） | [`install.sh`](./installation.md#linux--macos--wsl2--android-termux) | 我们在最新的 Ubuntu 和 WSL2 上进行测试。如果你的发行版有 glibc、systemd 并遵循文件系统层级标准，大概率能运行良好。 |
| [**Docker 容器**](../user-guide/docker.md#quick-start)（x86_64、aarch64） | [`docker pull`](../user-guide/docker.md#quick-start) | Docker 安装不支持 `hermes update`。更新需要运行新镜像。 |

---

## 二级平台

这些平台仅以尽力维护的方式在代码仓库中维护。
版本更新可能会破坏它们，我们无法保证会及时修复。

接受修复这些问题的 PR，但它们的优先级低于一级平台。

| 操作系统 / 架构 | 安装方式 | 说明 |
| --- | --- | --- |
| **Android（Termux）**（aarch64） | [`install.sh`](./installation.md#linux--macos--wsl2--android-termux) | 少数功能[不可用](./termux.md#known-limitations-on-phones)。 |
| **Nix**（macOS、Linux、NixOS） | [`install.sh`](./nix-setup.md) | 因为 node.js 打包问题经常挂掉。祝你好运~！❤️ |

## 不支持

以下平台和分发方式**不**受支持。
我们建议你切换到受支持的分发方式或平台。
它们可能现在就有问题，未来可能会出更多问题。
修复它们的 PR **不会**被接受，任何与它们保持兼容的代码可能会随时被移除。

- 通过 AUR 安装（如果有助于修复，我们可能会上游补丁 ❤️）
- macOS x86（Intel）处理器
- 通过 `pypi` 安装（如 `uv tool install hermes-agent`、`pip install hermes-agent` 等）
- 通过 `brew` 安装（`brew install hermes-agent`）

如果你正在使用不受支持的分发方式，请阅读[安装指南](./installation.md)了解如何切换到受支持的方式。
