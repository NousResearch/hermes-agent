---
title: "Windows (WSL2) Quick Start"
description: "Run Hermes Agent on Windows using WSL2 — supported path for CLI and Tool Gateway"
sidebar_label: "Windows (WSL2)"
sidebar_position: 2
---

# Windows (WSL2) Quick Start

Hermes Agent is developed and tested on **Linux** and **macOS**. On Windows, the supported setup is **WSL2** (Windows Subsystem for Linux), not legacy native Windows shells.

:::info Full guide in Chinese
The detailed checklist (WSL2, `uv`, repo clone, gateway tips) is maintained in **简体中文**. Use the **language** menu (top right) and select **简体中文**, then open this same page again.
:::

## Minimum path

1. Install [WSL2](https://learn.microsoft.com/windows/wsl/install) and a recent Ubuntu (or another supported distro).
2. Open your WSL terminal and follow [Installation](/getting-started/installation) inside that environment.
3. Run `hermes model` / `hermes tools` from WSL so paths, process isolation, and the Tool Gateway match upstream expectations.

For Tool Gateway and image tooling behavior, see [Tool Gateway](/user-guide/features/tool-gateway) and [Image Generation](/user-guide/features/image-generation).
