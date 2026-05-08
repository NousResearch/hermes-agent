---
title: "Agentmail — 通过 AgentMail 给予代理自己的专用电子邮件收件箱"
sidebar_label: "Agentmail"
description: "通过 AgentMail 给予代理自己的专用电子邮件收件箱"
---

{/* 此页面由 website/scripts/generate-skill-docs.py 根据 skill 的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Agentmail

通过 AgentMail 给予代理自己的专用电子邮件收件箱。使用代理拥有的电子邮件地址（例如 hermes-agent@agentmail.to）自主发送、接收和管理电子邮件。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/email/agentmail` 安装 |
| 路径 | `optional-skills/email/agentmail` |
| 版本 | `1.0.0` |
| 标签 | `email`、`communication`、`agentmail`、`mcp` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# AgentMail — 代理自有电子邮件收件箱

## 要求

- **AgentMail API 密钥**（必需）— 在 https://console.agentmail.to 注册（免费层级：3 个收件箱，3,000 封邮件/月；付费计划起价 $20/月）
- Node.js 18+（用于 MCP 服务器）

## 使用场景

在需要以下情况时使用此技能：
- 给予代理自己的专用电子邮件地址
- 代表代理自主发送电子邮件
- 接收和阅读传入的电子邮件
- 管理电子邮件线程和对话
- 通过电子邮件注册服务或进行身份验证
- 通过电子邮件与其他代理或人类通信

这不是用于阅读用户个人电子邮件的（为此使用 himalaya 或 Gmail）。
AgentMail 给予代理自己的身份和收件箱。

## 设置

### 1. 获取 API 密钥
- 前往 https://console.agentmail.to
- 创建账户并生成 API 密钥（以 `am_` 开头）

### 2. 配置 MCP 服务器
添加到 `~/.hermes/config.yaml`（粘贴您的实际密钥 — MCP 环境变量不会从 .env 展开）：
```yaml
mcp_servers:
  agentmail:
    command: "npx"
    args: ["-y", "agentmail-mcp"]
    env:
      AGENTMAIL_API_KEY: "am_your_key_here"
```

### 3. 重启 Hermes
```bash
hermes
```
现在 11 个 AgentMail 工具都自动可用了。

## 可用工具（通过 MCP）

| Tool | Description |
|------|-------------|
| `list_inboxes` | 列出所有代理收件箱 |
| `get_inbox` | 获取特定收件箱的详情 |
| `create_inbox` | 创建新收件箱（获取真实电子邮件地址） |
| `delete_inbox` | 删除收件箱 |
| `list_threads` | 列出收件箱中的电子邮件线程 |
| `get_thread` | 获取特定电子邮件线程 |
| `send_message` | 发送新电子邮件 |
| `reply_to_message` | 回复现有电子邮件 |
| `forward_message` | 转发电子邮件 |
| `update_message` | 更新消息标签/状态 |
| `get_attachment` | 下载电子邮件附件 |

## 步骤

### 创建收件箱并发送电子邮件
1. 创建专用收件箱：
   - 使用用户名（例如 `hermes-agent`）调用 `create_inbox`
   - 代理获得地址：`hermes-agent@agentmail.to`
2. 发送电子邮件：
   - 使用 `inbox_id`、`to`、`subject`、`text` 调用 `send_message`
3. 检查回复：
   - 使用 `list_threads` 查看传入对话
   - 使用 `get_thread` 读取特定线程

### 检查传入电子邮件
1. 使用 `list_inboxes` 找到您的收件箱 ID
2. 使用收件箱 ID 调用 `list_threads` 查看对话
3. 使用 `get_thread` 读取线程及其消息

### 回复电子邮件
1. 使用 `get_thread` 获取线程
2. 使用消息 ID 和回复文本调用 `reply_to_message`

## 示例工作流

**注册服务：**
```
1. create_inbox (username: "signup-bot")
2. 使用收件箱地址在服务上注册
3. list_threads 检查验证电子邮件
4. get_thread 读取验证码
```

**代理到人类外联：**
```
1. create_inbox (username: "hermes-outreach")
2. send_message (to: user@example.com, subject: "Hello", text: "...")
3. list_threads 检查回复
```

## 陷阱

- 免费层级限制为 3 个收件箱和 3,000 封邮件/月
- 免费层级电子邮件来自 `@agentmail.to` 域名（付费计划可自定义域名）
- MCP 服务器需要 Node.js（18+）（`npx -y agentmail-mcp`）
- 必须安装 `mcp` Python 包：`pip install mcp`
- 实时入站电子邮件（webhooks）需要公共服务器 — 对于个人使用，通过 cronjob 使用 `list_threads` 轮询代替

## 验证

设置后，用以下方式测试：
```
hermes --toolsets mcp -q "Create an AgentMail inbox called test-agent and tell me its email address"
```
您应该看到返回的新收件箱地址。

## 参考

- AgentMail 文档：https://docs.agentmail.to/
- AgentMail 控制台：https://console.agentmail.to
- AgentMail MCP 仓库：https://github.com/agentmail-to/agentmail-mcp
- 定价：https://www.agentmail.to/pricing
