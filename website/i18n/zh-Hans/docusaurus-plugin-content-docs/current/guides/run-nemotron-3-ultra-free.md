---
sidebar_position: 0
title: "在 Hermes Agent 中免费体验 Nemotron 3 Ultra"
sidebar_label: "免费体验 Nemotron 3 Ultra"
description: "在 Nous Portal 上尝试 NVIDIA Nemotron 3 Ultra——限时免费（6 月 4 日至 6 月 18 日）——Hermes Agent 提供第一时间支持"
---

# 在 Hermes Agent 中免费体验 Nemotron 3 Ultra

Nous Research 已加入 **Nemotron Coalition**，与 **NVIDIA** 一起推进开放前沿基础模型的发展。借此机会，我们与 **Nebius** 合作，在 [Nous Portal](https://portal.nousresearch.com) 上限时免费提供 **Nemotron 3 Ultra**（**6 月 4 日至 6 月 18 日**）。按照以下说明，今天就可以在 Hermes Agent 中体验这个模型。

:::info 限时活动
`nvidia/nemotron-3-ultra:free` 层级将在 **6 月 4 日至 6 月 18 日**期间可用。`:free` 后缀是免付费计划生效的关键——请务必选择带该后缀的版本。
:::

你可以选择适合自己习惯的安装方式。**桌面应用**最简单，无需使用终端；如果你习惯在终端里操作，下面也有对应的 **命令行**方式。

## 方案 A——桌面应用（推荐）

最简单的路径：使用一键安装程序，全程引导式操作，无需终端。

### 1. 下载并安装

[下载 Hermes Desktop 安装程序](https://hermes-agent.nousresearch.com/desktop)（支持 macOS 或 Windows），然后打开。首次启动时，程序会自动完成初始化（通常不到一分钟）。

### 2. 连接 Nous Portal

打开应用后，你会看到 "Let's get you set up" 界面。点击标记为 **Recommended** 的 **Nous Portal**。浏览器会自动打开——注册或登录 [Nous Portal](https://portal.nousresearch.com) 账户，选择 **Free** 计划，并授权 Hermes。应用随后会自动完成连接。

### 3. 选择免费 Nemotron 3 Ultra 模型

连接完成后，应用会显示 **Default model** 卡片。点击 **Change**，搜索 **nemotron 3 ultra**，并选择带 **Free tier** 标记的版本：

```
nvidia/nemotron-3-ultra:free
```

带 `:free` 后缀的版本才是免费层级——请务必选择该版本。

### 4. 开始对话

点击 **Start chatting**。这样你就可以免费和 Nemotron 3 Ultra 对话了。

## 方案 B——命令行

更习惯终端？

### 1. 安装 Hermes Agent

在 macOS/Linux/WSL2/Android 上运行：

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

在 Windows 上运行：

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

如果想先查看脚本内容，可先下载 [`install.sh`](https://hermes-agent.nousresearch.com/install.sh)，确认后再运行。

安装完成后，重新加载终端配置：

```bash
source ~/.bashrc   # 或 source ~/.zshrc
```

### 2. 运行快速设置

```bash
hermes setup
```

选择 **Quick Setup**。Hermes 会打开浏览器标签页，并等待你完成后续步骤。

### 3. 创建 Nous Portal 账户

在浏览器中注册或登录 [Nous Portal](https://portal.nousresearch.com) 账户，并选择 **Free** 计划。

### 4. 连接账户

当提示将账户连接到 Hermes Agent 时，点击 **Connect**。连接成功后，会显示确认信息。

### 5. 选择免费 Nemotron 3 Ultra 模型

回到终端，在模型列表中选择：

```
nvidia/nemotron-3-ultra:free
```

带 `:free` 后缀的版本才是免费层级，请务必选择该版本。

### 6. 开始对话

完成剩余的 Quick Setup 提示后，运行：

```bash
hermes
```

这样你就可以免费和 Nemotron 3 Ultra 对话了。

## 之后如何切换到该模型

如果你已经配置了其他模型：

- **桌面应用**：打开模型选择器，搜索 **nemotron 3 ultra**，选择带 **Free tier** 标记的版本。
- **CLI / TUI**：随时可在会话中输入 `/model nvidia/nemotron-3-ultra:free` 切换，或输入 `/model` 打开选择器后手动选择。

## 常见问题排查

- **在列表里找不到该模型？** 请确认你已完成 Nous Portal 连接，并处于 **Free** 计划。在 CLI 中，可以运行 `hermes portal info` 来确认登录状态及路由情况。
- **选错了版本？** 请重新选择 `nvidia/nemotron-3-ultra:free`——必须使用 `:free` 后缀，才能保持在免费层级。
- **浏览器没有打开，或者你在远程主机上使用 CLI？** 可参阅 [OAuth over SSH / Remote Hosts](/guides/oauth-over-ssh)，其中包含端口转发和手动粘贴的解决方案。

## 另请参阅

- **[桌面应用](/user-guide/desktop)** —— 原生一键安装应用（macOS、Windows、Linux）
- **[使用 Nous Portal 运行 Hermes Agent](/guides/run-hermes-with-nous-portal)** —— Portal 完整使用指南：模型、Tool Gateway 和验证
- **[Nous Portal 集成](/integrations/nous-portal)** —— 订阅包含哪些内容
- **[快速开始](/getting-started/quickstart)** —— 5 分钟内从安装到对话
