---
sidebar_position: 2
title: "安装"
description: "在 Linux、macOS、WSL2 或通过 Termux 在 Android 上安装 Hermes Agent"
---

# 安装

使用一行安装程序，在两分钟内启动并运行 Hermes Agent。

## 快速安装

### Linux / macOS / WSL2

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### Android / Termux

Hermes 现在也提供了支持 Termux 的安装路径：

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

安装程序会自动检测 Termux 并切换到经过测试的 Android 流程：
- 使用 Termux `pkg` 安装系统依赖（`git`、`python`、`nodejs`、`ripgrep`、`ffmpeg`、构建工具）
- 使用 `python -m venv` 创建虚拟环境
- 自动导出 `ANDROID_API_LEVEL` 以构建 Android wheel
- 使用 `pip` 安装精选的 `.[termux]` 额外依赖
- 默认跳过未经测试的浏览器 / WhatsApp 引导

如果你想要完全明确的路径，请参考专用的 [Termux 指南](./termux.md)。

:::warning Windows
原生 Windows **不受支持**。请安装 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) 并从中运行 Hermes Agent。上述安装命令可在 WSL2 内部运行。
:::

### 安装程序的工作原理

安装程序自动处理所有事务 —— 依赖项（Python、Node.js、ripgrep、ffmpeg）、仓库克隆、虚拟环境、全局 `hermes` 命令设置，以及 LLM 提供商配置。完成后，你就可以开始聊天了。

#### 安装布局

安装程序将文件放在哪里取决于你是以普通用户还是 root 身份安装：

| 安装程序 | 代码位置 | `hermes` 二进制文件 | 数据目录 |
|---|---|---|---|
| 每用户（普通） | `~/.hermes/hermes-agent/` | `~/.local/bin/hermes`（符号链接） | `~/.hermes/` |
| Root 模式（`sudo curl … | sudo bash`） | `/usr/local/lib/hermes-agent/` | `/usr/local/bin/hermes` | `/root/.hermes/`（或 `$HERMES_HOME`） |

Root 模式 **FHS 布局**（`/usr/local/lib/…`、`/usr/local/bin/hermes`）与其他系统级开发工具在 Linux 上的位置一致。对于共享机器部署非常有用，一个系统安装可以服务所有用户。每用户配置（认证、技能、会话）仍然位于每个用户的 `~/.hermes/` 或明确的 `HERMES_HOME` 下。

### 安装后

重新加载你的 shell 并开始聊天：

```bash
source ~/.bashrc   # 或者：source ~/.zshrc
hermes             # 开始聊天！
```

以后要重新配置各个设置，请使用专用命令：

```bash
hermes model          # 选择你的 LLM 提供商和模型
hermes tools          # 配置启用哪些工具
hermes gateway setup  # 设置消息平台
hermes config set     # 设置单个配置值
hermes setup          # 或者运行完整设置向导一次性配置所有内容
```

---

## 前置要求

唯一的前提条件是 **Git**。安装程序自动处理其他所有内容：

- **uv**（快速的 Python 包管理器）
- **Python 3.11**（通过 uv，无需 sudo）
- **Node.js v22**（用于浏览器自动化和 WhatsApp 桥接）
- **ripgrep**（快速文件搜索）
- **ffmpeg**（用于 TTS 的音频格式转换）

:::info
你**不需要**手动安装 Python、Node.js、ripgrep 或 ffmpeg。安装程序会检测缺失的内容并自动为你安装。只需确保 `git` 可用（`git --version`）。
:::

:::tip Nix 用户
如果你使用 Nix（在 NixOS、macOS 或 Linux 上），有一条专用设置路径，包含 Nix flake、声明式 NixOS 模块和可选的容器模式。参见 **[Nix 和 NixOS 设置](./nix-setup.md)** 指南。
:::

---

## 手动 / 开发者安装

如果你想克隆仓库并从源代码安装 —— 用于贡献、运行特定分支或完全控制虚拟环境 —— 请参阅贡献指南中的[开发设置](../developer-guide/contributing.md#development-setup)部分。

---

## 故障排除

| 问题 | 解决方案 |
|---------|----------|
| `hermes: command not found` | 重新加载 shell（`source ~/.bashrc`）或检查 PATH |
| `API key not set` | 运行 `hermes model` 配置你的提供商，或 `hermes config set OPENROUTER_API_KEY your_key` |
| 更新后配置缺失 | 运行 `hermes config check` 然后 `hermes config migrate` |

要进行更多诊断，请运行 `hermes doctor` —— 它会准确告诉你缺少什么以及如何修复。
