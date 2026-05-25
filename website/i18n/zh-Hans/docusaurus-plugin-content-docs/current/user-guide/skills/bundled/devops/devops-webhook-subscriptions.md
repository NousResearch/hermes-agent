---
title: "Webhook Subscriptions — Webhook subscriptions: event-driven agent runs"
sidebar_label: "Webhook Subscriptions"
description: "Webhook subscriptions：事件驱动的 agent 运行"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Webhook Subscriptions

Webhook subscriptions：事件驱动的 agent 运行。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/devops/webhook-subscriptions` |
| 版本 | `1.1.0` |
| 平台 | linux, macos, windows |
| 标签 | `webhook`, `events`, `automation`, `integrations`, `notifications`, `push` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Webhook Subscriptions

创建动态 webhook 订阅，使外部服务（GitHub、GitLab、Stripe、CI/CD、IoT 传感器、监控工具）能够通过向 URL 发送 POST 请求来触发 Hermes agent 运行。

## 设置（必须先完成）

在创建订阅之前，必须先启用 webhook 平台。检查方式：
```bash
hermes webhook list
```

如果提示"Webhook platform is not enabled"，请进行设置：

### 选项 1：设置向导
```bash
hermes gateway setup
```
按照提示启用 webhook、设置端口并配置全局 HMAC 密钥。

### 选项 2：手动配置
在 `~/.hermes/config.yaml` 中添加：
```yaml
platforms:
  webhook:
    enabled: true
    extra:
      host: "0.0.0.0"
      port: 8644
      secret: "generate-a-strong-secret-here"
```

### 选项 3：环境变量
在 `~/.hermes/.env` 中添加：
```bash
WEBHOOK_ENABLED=true
WEBHOOK_PORT=8644
WEBHOOK_SECRET=generate-a-strong-secret-here
```

配置完成后，启动（或重启）gateway：
```bash
hermes gateway run
# 如果使用 systemd：
systemctl --user restart hermes-gateway
```

验证是否正在运行：
```bash
curl http://localhost:8644/health
```

## 命令

所有管理操作均通过 `hermes webhook` CLI 命令完成：

### 创建订阅
```bash
hermes webhook subscribe <name> \
  --prompt "Prompt template with {payload.fields}" \
  --events "event1,event2" \
  --description "What this does" \
  --skills "skill1,skill2" \
  --deliver telegram \
  --deliver-chat-id "12345" \
  --secret "optional-custom-secret"
```

返回 webhook URL 和 HMAC 密钥。用户将其服务配置为向该 URL 发送 POST 请求。

### 列出订阅
```bash
hermes webhook list
```

### 删除订阅
```bash
hermes webhook remove <name>
```

### 测试订阅
```bash
hermes webhook test <name>
hermes webhook test <name> --payload '{"key": "value"}'
```

## Prompt 模板

Prompt（提示词）支持使用 `{dot.notation}` 访问嵌套的 payload 字段：

- `{issue.title}` — GitHub issue 标题
- `{pull_request.user.login}` — PR 作者
- `{data.object.amount}` — Stripe 支付金额
- `{sensor.temperature}` — IoT 传感器读数

如果未指定 prompt，完整的 JSON payload 将直接传入 agent prompt。

## 常见模式

### GitHub：新 issue
```bash
hermes webhook subscribe github-issues \
  --events "issues" \
  --prompt "New GitHub issue #{issue.number}: {issue.title}\n\nAction: {action}\nAuthor: {issue.user.login}\nBody:\n{issue.body}\n\nPlease triage this issue." \
  --deliver telegram \
  --deliver-chat-id "-100123456789"
```

然后在 GitHub 仓库的 Settings → Webhooks → Add webhook 中：
- Payload URL：返回的 webhook_url
- Content type：application/json
- Secret：返回的 secret
- Events："Issues"

### GitHub：PR 审查
```bash
hermes webhook subscribe github-prs \
  --events "pull_request" \
  --prompt "PR #{pull_request.number} {action}: {pull_request.title}\nBy: {pull_request.user.login}\nBranch: {pull_request.head.ref}\n\n{pull_request.body}" \
  --skills "github-code-review" \
  --deliver github_comment
```

### Stripe：支付事件
```bash
hermes webhook subscribe stripe-payments \
  --events "payment_intent.succeeded,payment_intent.payment_failed" \
  --prompt "Payment {data.object.status}: {data.object.amount} cents from {data.object.receipt_email}" \
  --deliver telegram \
  --deliver-chat-id "-100123456789"
```

### CI/CD：构建通知
```bash
hermes webhook subscribe ci-builds \
  --events "pipeline" \
  --prompt "Build {object_attributes.status} on {project.name} branch {object_attributes.ref}\nCommit: {commit.message}" \
  --deliver discord \
  --deliver-chat-id "1234567890"
```

### 通用监控告警
```bash
hermes webhook subscribe alerts \
  --prompt "Alert: {alert.name}\nSeverity: {alert.severity}\nMessage: {alert.message}\n\nPlease investigate and suggest remediation." \
  --deliver origin
```

### 直接投递（无 agent，零 LLM 成本）

适用于只需将通知推送给用户聊天的场景——无需推理，无需 agent 循环——添加 `--deliver-only`。渲染后的 `--prompt` 模板将作为字面消息体直接分发到目标适配器。

适用场景：
- 外部服务推送通知（Supabase/Firebase webhooks → Telegram）
- 应原样转发的监控告警
- 一个 agent 向另一个 agent 的用户发送消息的 agent 间通信
- 任何 LLM 往返调用属于浪费的 webhook 场景

```bash
hermes webhook subscribe antenna-matches \
  --deliver telegram \
  --deliver-chat-id "123456789" \
  --deliver-only \
  --prompt "🎉 New match: {match.user_name} matched with you!" \
  --description "Antenna match notifications"
```

投递成功时 POST 返回 `200 OK`，目标失败时返回 `502`——以便上游服务能够智能重试。HMAC 认证、速率限制和幂等性仍然适用。

要求 `--deliver` 为真实目标（telegram、discord、slack、github_comment 等）——`--deliver log` 会被拒绝，因为仅记录日志的直接投递毫无意义。

## 安全性

- 每个订阅自动生成 HMAC-SHA256 密钥（也可通过 `--secret` 自行提供）
- webhook 适配器对每个传入的 POST 请求验证签名
- `config.yaml` 中的静态路由不会被动态订阅覆盖
- 订阅持久化保存至 `~/.hermes/webhook_subscriptions.json`

## 工作原理

1. `hermes webhook subscribe` 写入 `~/.hermes/webhook_subscriptions.json`
2. webhook 适配器在每次收到请求时热重载该文件（基于 mtime 检测，开销可忽略不计）
3. 当匹配路由的 POST 请求到达时，适配器格式化 prompt 并触发 agent 运行
4. agent 的响应被投递到已配置的目标（Telegram、Discord、GitHub comment 等）

## 故障排查

如果 webhook 无法正常工作：

1. **gateway 是否在运行？** 通过 `systemctl --user status hermes-gateway` 或 `ps aux | grep gateway` 检查
2. **webhook 服务器是否在监听？** `curl http://localhost:8644/health` 应返回 `{"status": "ok"}`
3. **查看 gateway 日志：** `grep webhook ~/.hermes/logs/gateway.log | tail -20`
4. **签名不匹配？** 验证服务中的 secret 与 `hermes webhook list` 返回的一致。GitHub 发送 `X-Hub-Signature-256`，GitLab 发送 `X-Gitlab-Token`。
5. **防火墙/NAT？** webhook URL 必须能从该服务访问到。本地开发时，请使用隧道工具（ngrok、cloudflared）。
6. **事件类型错误？** 检查 `--events` 过滤器是否与服务发送的事件匹配。使用 `hermes webhook test <name>` 验证路由是否正常工作。