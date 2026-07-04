---
title: "Agentmail — 通过 AgentMail 为 Agent 提供专属电子邮件收件箱"
sidebar_label: "Agentmail"
description: "通过 AgentMail 为 Agent 提供专属电子邮件收件箱"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Agentmail

通过 AgentMail 为 Agent 提供专属电子邮件收件箱。使用 Agent 专属电子邮件地址（例如 hermes-agent@agentmail.to）自主发送、接收和管理电子邮件。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/email/agentmail` 安装 |
| 路径 | `optional-skills/email/agentmail` |
| 版本 | `1.0.0` |
| 平台 | linux, macos, windows |
| 标签 | `email`, `communication`, `agentmail`, `mcp` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 Agent 所看到的指令内容。
:::

# AgentMail — Agent 专属电子邮件收件箱

## 前置要求

- **AgentMail API 密钥**（必需）— 在 https://console.agentmail.to 注册（免费套餐：3 个收件箱，每月 3,000 封邮件；付费套餐起价 $20/月）
- Node.js 18+（用于 MCP 服务器）

## 使用场景
在以下情况下使用此 skill：
- 为 Agent 提供专属电子邮件地址
- 代表 Agent 自主发送电子邮件
- 接收并读取传入邮件
- 管理邮件线程和对话
- 通过电子邮件注册服务或进行身份验证
- 通过电子邮件与其他 Agent 或人类进行通信

此 skill **不适用于**读取用户的个人邮件（请使用 himalaya 或 Gmail）。
AgentMail 为 Agent 提供独立的身份和收件箱。

## 配置

### 1. 获取 API 密钥
- 访问 https://console.agentmail.to
- 创建账户并生成 API 密钥（以 `am_` 开头）

### 2. 配置 MCP 服务器
添加至 `~/.hermes/config.yaml`（粘贴实际密钥 — MCP 环境变量不会从 .env 展开）：
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
所有 11 个 AgentMail 工具现已自动可用。

## 可用工具（通过 MCP）

| 工具 | 描述 |
|------|-------------|
| `list_inboxes` | 列出所有 Agent 收件箱 |
| `get_inbox` | 获取特定收件箱的详细信息 |
| `create_inbox` | 创建新收件箱（获得真实电子邮件地址） |
| `delete_inbox` | 删除收件箱 |
| `list_threads` | 列出收件箱中的邮件线程 |
| `get_thread` | 获取特定邮件线程 |
| `send_message` | 发送新邮件 |
| `reply_to_message` | 回复已有邮件 |
| `forward_message` | 转发邮件 |
| `update_message` | 更新邮件标签/状态 |
| `get_attachment` | 下载邮件附件 |

## 操作流程

### 创建收件箱并发送邮件
1. 创建专属收件箱：
   - 使用 `create_inbox` 并指定用户名（例如 `hermes-agent`）
   - Agent 获得地址：`hermes-agent@agentmail.to`
2. 发送邮件：
   - 使用 `send_message`，传入 `inbox_id`、`to`、`subject`、`text`
3. 检查回复：
   - 使用 `list_threads` 查看传入对话
   - 使用 `get_thread` 读取特定线程

### 检查传入邮件
1. 使用 `list_inboxes` 查找收件箱 ID
2. 使用 `list_threads` 并传入收件箱 ID 查看对话
3. 使用 `get_thread` 读取线程及其消息

### 回复邮件
1. 使用 `get_thread` 获取线程
2. 使用 `reply_to_message`，传入消息 ID 和回复内容

## 示例工作流

**注册服务：**
```
1. create_inbox (username: "signup-bot")
2. 使用该收件箱地址在服务上注册
3. list_threads 检查验证邮件
4. get_thread 读取验证码
```

**Agent 对人类的外发联系：**
```
1. create_inbox (username: "hermes-outreach")
2. send_message (to: user@example.com, subject: "Hello", text: "...")
3. list_threads 检查回复
```

## 注意事项
- 免费套餐限制为 3 个收件箱，每月 3,000 封邮件
- 免费套餐邮件来自 `@agentmail.to` 域名（付费套餐支持自定义域名）
- MCP 服务器需要 Node.js（18+）（`npx -y agentmail-mcp`）
- 必须安装 `mcp` Python 包：`pip install mcp`
- 实时入站邮件（webhook）需要公网服务器 — 个人使用时建议改用 `list_threads` 轮询配合 cronjob
- **xAI Grok 丢弃可选的多行字符串参数：** 在 `provider: xai` / `xai-oauth` 与 `grok-4.3` 下，值包含换行符的可选字符串参数（例如 `send_message` 的 `subject` / `text`，`reply_to_message` 的 `text`）会被静默剥离，导致 `send_message` 发出空白邮件且无错误。参见 [xAI Grok OAuth 指南](https://hermes-agent.nousresearch.com/docs/guides/xai-grok-oauth) 故障排查部分了解变通方案（将字段标记为 `required`，或修补 MCP 服务器的 zod schema 去除这些字段的 `.optional()`）。

## 验证
配置完成后，使用以下命令测试：
```
hermes --toolsets mcp -q "Create an AgentMail inbox called test-agent and tell me its email address"
```
应返回新收件箱的地址。

## 参考资料
- AgentMail 文档：https://docs.agentmail.to/
- AgentMail 控制台：https://console.agentmail.to
- AgentMail MCP 仓库：https://github.com/agentmail-to/agentmail-mcp
- 定价：https://www.agentmail.to/pricing