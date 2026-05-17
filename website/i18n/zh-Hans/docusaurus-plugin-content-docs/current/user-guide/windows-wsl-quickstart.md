---
title: "Windows 用户快速上手（WSL2）"
description: "在 Windows 上通过 WSL2 安装 uv、Hermes 与 Tool Gateway 的推荐路径与常见坑"
sidebar_label: "Windows（WSL2）"
sidebar_position: 2
---

# Windows 用户快速上手（WSL2）

上游开发与 CI 以 **Linux / macOS** 为主；在 Windows 上，**官方推荐路径是 WSL2**，而不是在「旧版原生 CMD/PowerShell」里直接跑完整 Hermes 栈。本页给出从 0 到可跑 `hermes` + Tool Gateway 的最短闭环。

## 1. 安装 WSL2 与发行版

1. 以管理员打开 PowerShell，安装 WSL 与默认 Ubuntu（具体命令以 [微软文档](https://learn.microsoft.com/zh-cn/windows/wsl/install) 为准）：
   ```powershell
   wsl --install
   ```
2. 重启后完成 Ubuntu 首次用户名/密码设置。
3. 在 Microsoft Store 或 `wsl --list --online` 中可选用较新 Ubuntu LTS，便于获得较新的 `glibc` 与 Python 工具链。

:::caution 关于「原生 Windows」
若你只在 PowerShell 里装 Python/uv，可能遇到路径、子进程、网关单例与 Token 缓存等与上游假设不一致的问题。**请优先在 WSL 终端内**完成安装与日常使用。
:::

## 2. 在 WSL 内安装 `uv`

在 **WSL 的 Bash** 中执行（勿混用 Windows 路径）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

将 `uv` 加入当前 shell 的 `PATH`（安装脚本结尾会提示），然后：

```bash
uv --version
```

## 3. 获取 Hermes Agent

在 WSL 里 clone 本仓库（或你的 fork），进入目录后按 [安装说明](/getting-started/installation) 使用 `uv sync` / 文档中的推荐命令安装依赖。

:::tip 路径与权限
Hermes 默认配置目录为 `~/.hermes/`（在 WSL 内即 Linux 家目录）。请勿把 WSL 项目放在会被 Windows 杀毒实时深度扫描的极慢盘符上；推荐放在 WSL 文件系统（例如 `~/projects/...`）而非 `/mnt/c/...` 下的重度 IO 路径。
:::

## 4. 模型与 Tool Gateway

1. 在 WSL 内运行 `hermes model`，按提示绑定 **Nous Portal**（或其他提供商）。  
2. 付费订阅用户可启用 **[Tool Gateway](/user-guide/features/tool-gateway)**，用于网页搜索、文生图、TTS、浏览器自动化等，而无需单独配置 `FAL_KEY` / Firecrawl 等（详见该页）。  
3. 文生图模型列表与计费说明见 **[文生图](/user-guide/features/image-generation)**。

## 5. 常见故障速查

| 现象 | 建议 |
|------|------|
| 网关相关进程重复 / 端口占用 | 确认是否同时在 Windows 侧与 WSL 侧各启动了一份 agent；同一机器上只保留**一个**常驻会话。 |
| `hermes` 找不到 | 确认 `uv run hermes` 或按安装文档将 CLI 暴露到 `PATH`；命令应在 **WSL** 内执行。 |
| 图像工具 4xx | 可能是 Portal 尚未代理该 FAL 模型；可换模型或配置直连 `FAL_KEY`（见文生图文档）。 |

## 6. 下一步

- 英文摘要页（默认语言）：仍保留轻量说明，便于非中文读者理解 WSL2 要求。  
- 深入 CLI：见 [CLI 界面](/user-guide/cli)。  
- 全局配置项：见 [配置说明](/user-guide/configuration)。  
