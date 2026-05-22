---
sidebar_position: 4
title: "团队 Telegram 助手"
description: "一步一步搭建一个给整个团队使用的 Telegram 机器人，用于代码帮助、研究、系统管理等"
---

# 搭建一个团队 Telegram 助手

本教程会带你搭建一个由 Hermes Agent 驱动的 Telegram bot，让多个团队成员都能使用。完成后，你的团队就会拥有一个可以直接发消息求助的共享 AI 助手，支持代码帮助、研究、系统管理等功能，并且通过用户授权确保安全。

## 我们要做什么

一个 Telegram bot，具备：

- **任何被授权的团队成员都能私聊使用**，例如代码审查、研究、shell 命令、调试
- **运行在你自己的服务器上**，具备完整工具访问：终端、文件编辑、网页搜索、代码执行
- **每个用户独立会话**，每个人都有自己的上下文
- **默认安全**，只允许被批准的用户访问，且有两种授权方式
- **计划任务**，可以把每日站会、健康检查和提醒发到团队频道

## 前置条件

开始前，请确认：

- 你已经在服务器或 VPS 上安装了 **Hermes Agent**（不是笔记本，因为 bot 需要持续运行）。如果还没有，请先看 [安装指南](/getting-started/installation)。
- 你有一个 **Telegram 账号**（bot 所有人）
- 你已经配置好 **LLM provider**，至少在 `~/.hermes/.env` 中配置了 OpenAI、Anthropic 或其他支持的 provider 的 API key

:::tip
一个每月 5 美元的 VPS 已经足够运行 gateway。Hermes 本身很轻量，真正花钱的是 LLM API 调用，而且这些调用是远程完成的。
:::

## 第 1 步：创建 Telegram bot

每个 Telegram bot 都从 **@BotFather** 开始——这是 Telegram 官方的 bot 创建工具。

1. 在 Telegram 中搜索 `@BotFather`，或者直接访问 [t.me/BotFather](https://t.me/BotFather)
2. 发送 `/newbot`，BotFather 会问你两个问题：显示名称和用户名
3. 复制 bot token，保存好，下一步要用
4. 可选地设置 description
5. 可选地设置 bot commands，给用户一个命令菜单

:::warning
bot token 必须保密。任何拿到 token 的人都能控制这个 bot。如果泄露了，请在 BotFather 里用 `/revoke` 重新生成。
:::

## 第 2 步：配置 gateway

你有两个选择：交互式向导（推荐）或手动配置。

### 方案 A：交互式设置

```bash
hermes gateway setup
```

它会带你一步步完成。选择 **Telegram**，粘贴 bot token，然后按提示输入你的用户 ID。

### 方案 B：手动配置

把这些写入 `~/.hermes/.env`：

```bash
TELEGRAM_BOT_TOKEN=7123456789:AAH1bGciOiJSUzI1NiIsInR5cCI6Ikp...
TELEGRAM_ALLOWED_USERS=123456789
```

### 找到你的用户 ID

你的 Telegram user ID 是数字，不是用户名。查找方法：

1. 在 Telegram 里给 [@userinfobot](https://t.me/userinfobot) 发消息
2. 它会返回你的数字 user ID
3. 把这个数字写进 `TELEGRAM_ALLOWED_USERS`

## 第 3 步：启动 gateway

### 先做快速测试

先前台运行，确认一切正常：

```bash
hermes gateway
```

你应该会看到类似输出：

```text
[Gateway] Starting Hermes Gateway...
[Gateway] Telegram adapter connected
[Gateway] Cron scheduler started (tick every 60s)
```

然后打开 Telegram，找到你的 bot，发一条消息。如果它回复了，就说明成功了。按 `Ctrl+C` 停止。

### 生产环境：安装为服务

如果想让它重启后继续运行：

```bash
hermes gateway install
sudo hermes gateway install --system   # 仅 Linux：开机系统服务
```

这会创建后台服务：Linux 默认创建用户级 systemd 服务，macOS 默认创建 launchd 服务；如果带 `--system`，Linux 会创建开机系统服务。

## 第 4 步：配置团队访问

现在给团队成员开权限。有两种方法。

### 方法 A：静态 allowlist

收集每个成员的 Telegram user ID，把它们写成逗号分隔：

```bash
TELEGRAM_ALLOWED_USERS=123456789,987654321,555555555
```

修改后重启 gateway：

```bash
hermes gateway stop
hermes gateway start
```

### 方法 B：DM pairing（团队更推荐）

这更灵活，不需要提前收集所有 user ID。

1. 团队成员先私聊 bot
2. bot 返回一次性配对码
3. 他们把这个码发给你
4. 你在服务器上批准它
5. 通过后 bot 会立刻开始响应

```bash
hermes pairing approve telegram XKGH5N7P
```

查看和管理配对用户：

```bash
hermes pairing list
hermes pairing revoke telegram 987654321
hermes pairing clear-pending
```

:::tip
DM pairing 很适合团队，因为新增成员时不需要重启 gateway，批准立即生效。
:::

### 安全注意事项

- 不要在有终端访问权限的 bot 上设置 `GATEWAY_ALLOW_ALL_USERS=true`
- 配对码 1 小时后过期，并使用密码学随机数生成
- 有速率限制，防止暴力尝试
- 多次失败会触发锁定
- 所有配对数据以 `chmod 0600` 权限存储

## 第 5 步：配置 bot

### 设置 home channel

home channel 是 bot 投递 cron 结果和主动消息的位置。没有它，定时任务就没有地方发输出。
