---
sidebar_position: 0
title: "在 Hermes Agent 中免费运行 Nemotron 3 Ultra"
description: "在 Nous Portal 上试用 NVIDIA Nemotron 3 Ultra —— 6 月 4-18 日免费 —— Hermes Agent 同日支持"
---

# 在 Hermes Agent 中免费运行 Nemotron 3 Ultra

Nous Research 已入选 **Nemotron Coalition**，这是与 **NVIDIA** 合作推进开源前沿基础模型的顶级 AI 实验室联盟。为此，我们与 **Nebius** 合作，在 [Nous Portal](https://portal.nousresearch.com) 上提供两周（**6 月 4 日 - 6 月 18 日**）的免费 **Nemotron 3 Ultra**。请按照以下说明在你的 Hermes Agent 中试用这个模型。

:::info 限时优惠
`nvidia/nemotron-3-ultra:free` 层级在 **6 月 4 日至 6 月 18 日**期间可用。`:free` 标签是保持其在免费计划上的关键 —— 请确保选择这个具体变体。
:::

选择适合你的安装方式。**桌面应用**最简单 —— 无需终端。如果你习惯用终端，**命令行**安装就在下方。

## 方式 A —— 桌面应用（推荐）

最简单的路径：一键安装器加引导式、点击即完成的设置。无需终端。

### 1. 下载并安装

[下载 Hermes Desktop 安装程序](https://hermes-agent.nousresearch.com/)（支持 macOS 或 Windows），然后打开它。首次启动时会自动完成设置（通常不到一分钟）。

### 2. 连接 Nous Portal

应用打开后，你会看到 "Let's get you set up" 页面。点击 **Nous Portal**（标记为 **Recommended**）。浏览器会打开 —— 创建一个 [Nous Portal](https://portal.nousresearch.com) 账号（或登录），选择 **Free** 计划，并授权 Hermes。应用会自动连接。

### 3. 选择免费的 Nemotron 3 Ultra 模型

连接后，应用会显示 **Default model** 卡片。点击 **Change**，搜索 **nemotron 3 ultra**，选择标记为 **Free tier** 的变体：

```
nvidia/nemotron-3-ultra:free
```

`:free` 标签是保持其在免费层级上的关键 —— 请确保选择这个变体。

### 4. 开始聊天

点击 **Start chatting**。就这样 —— 你已经在免费使用 Nemotron 3 Ultra 了。

## 方式 B —— 命令行

更习惯用终端？

### 1. 安装 Hermes Agent

在 macOS/Linux/WSL2/Android 上，运行

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

在 Windows 上，运行

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

想先审查一下？下载 [`install.sh`](https://hermes-agent.nousresearch.com/install.sh)，检查后运行。

安装完成后，重新加载你的 shell：

```bash
source ~/.bashrc   # or source ~/.zshrc
```

### 2. 运行快速设置

```bash
hermes setup
```

选择 **Quick Setup**。Hermes 会打开一个浏览器标签页，等待你完成后续步骤。

### 3. 创建 Nous Portal 账号

在浏览器中，创建一个 [Nous Portal](https://portal.nousresearch.com) 账号（或登录）并选择 **Free** 计划。

### 4. 连接你的账号

当提示将账号连接到 Hermes Agent 时，点击 **Connect**。连接成功后会看到确认信息。

### 5. 选择免费的 Nemotron 3 Ultra 模型

回到终端。从模型列表中选择：

```
nvidia/nemotron-3-ultra:free
```

`:free` 后缀是保持在免费层级上的关键，所以请确保选择这个变体。

### 6. 开始聊天

完成剩余的快速设置提示，然后运行：

```bash
hermes
```

就这样 —— 你已经在免费使用 Nemotron 3 Ultra 了。

## 后续切换

已经用其他模型设置好了？

- **桌面应用：** 打开模型选择器，搜索 **nemotron 3 ultra**，选择 **Free tier** 变体。
- **CLI / TUI：** 在会话中随时使用 `/model nvidia/nemotron-3-ultra:free` 切换，或运行 `/model` 打开选择器从列表中选择。

## 故障排除

- **在列表中看不到模型？** 确保你完成了 Nous Portal 连接，并且使用的是 **Free** 计划。在 CLI 中，`hermes portal info` 可以确认你已登录并通过 Nous 路由。
- **选错了变体？** 重新选择 `nvidia/nemotron-3-ultra:free` —— 需要 `:free` 后缀才能保持在免费层级上。
- **浏览器没有打开 / 你在远程主机上（CLI）？** 参见 [通过 SSH / 远程主机使用 OAuth](/guides/oauth-over-ssh) 了解端口转发方案。

## 另请参阅

- **[桌面应用](/user-guide/desktop)** —— 原生一键应用（macOS、Windows、Linux）
- **[使用 Nous Portal 运行 Hermes Agent](/guides/run-hermes-with-nous-portal)** —— 完整 Portal 使用指南：模型、Tool Gateway 和验证
- **[Nous Portal 集成](/integrations/nous-portal)** —— 订阅内容介绍
- **[快速开始](/getting-started/quickstart)** —— 5 分钟内完成安装到聊天
