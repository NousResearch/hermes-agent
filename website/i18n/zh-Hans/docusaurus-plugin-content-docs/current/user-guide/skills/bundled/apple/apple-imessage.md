---
title: "Imessage — 通过 imsg CLI 在 macOS 上发送和接收 iMessages/SMS"
sidebar_label: "Imessage"
description: "通过 imsg CLI 在 macOS 上发送和接收 iMessages/SMS"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Imessage

通过 imsg CLI 在 macOS 上发送和接收 iMessages/SMS。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/apple/imessage` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | macos |
| 标签 | `iMessage`、`SMS`、`messaging`、`macOS`、`Apple` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时触发的完整技能定义。这是代理激活技能时看到的指令。
:::

# iMessage

使用 `imsg` 通过 macOS Messages.app 读取和发送 iMessage/SMS。

## 前置条件

- **macOS** 带有已登录的 Messages.app
- 安装：`brew install steipete/tap/imsg`
- 授予终端完全磁盘访问权限（系统设置 → 隐私与安全性 → 完全磁盘访问）
- 出现提示时授予 Messages.app 的自动化权限

## 使用场景

- 用户要求发送 iMessage 或短信
- 读取 iMessage 对话历史
- 检查最近的 Messages.app 聊天
- 发送到电话号码或 Apple ID

## 不使用场景

- Telegram/Discord/Slack/WhatsApp 消息 → 使用相应的网关频道
- 群聊管理（添加/移除成员）→ 不支持
- 批量/群发消息 → 总是先与用户确认

## 快速参考

### 列出聊天

```bash
imsg chats --limit 10 --json
```

### 查看历史

```bash
# 按聊天 ID
imsg history --chat-id 1 --limit 20 --json

# 带附件信息
imsg history --chat-id 1 --limit 20 --attachments --json
```

### 发送消息

```bash
# 仅文本
imsg send --to "+14155551212" --text "Hello!"

# 带附件
imsg send --to "+14155551212" --text "Check this out" --file /path/to/image.jpg

# 强制 iMessage 或 SMS
imsg send --to "+14155551212" --text "Hi" --service imessage
imsg send --to "+14155551212" --text "Hi" --service sms
```

### 监听新消息

```bash
imsg watch --chat-id 1 --attachments
```

## 服务选项

- `--service imessage` — 强制 iMessage（需要收件人有 iMessage）
- `--service sms` — 强制 SMS（绿色气泡）
- `--service auto` — 让 Messages.app 决定（默认）

## 规则

1. **发送前始终确认收件人和消息内容**
2. **未经用户明确批准，不向未知号码发送**
3. **附加前验证文件路径**存在
4. **不要刷屏** — 对自己限速

## 示例工作流程

用户："给妈妈发消息说我要晚了"

```bash
# 1. 找到妈妈的聊天
imsg chats --limit 20 --json | jq '.[] | select(.displayName | contains("Mom"))'

# 2. 与用户确认："找到妈妈的号码 +1555123456。发送'我要晚了'通过 iMessage？"

# 3. 确认后发送
imsg send --to "+1555123456" --text "I'll be late"
```
