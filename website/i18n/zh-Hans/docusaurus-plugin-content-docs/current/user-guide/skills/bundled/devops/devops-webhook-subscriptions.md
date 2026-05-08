---
title: "Webhook Subscriptions — Webhook 订阅：事件驱动的 agent 运行"
sidebar_label: "Webhook Subscriptions"
description: "Webhook 订阅：事件驱动的 agent 运行"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Webhook Subscriptions

Webhook 订阅：事件驱动的 agent 运行。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/devops/webhook-subscriptions` |
| 版本 | `1.1.0` |
| 标签 | `webhook`, `events`, `automation`, `integrations`, `notifications`, `push` |

## 参考：完整 SKILL.md

:::info
以下是 Hermites 加载此技能时的完整技能定义。这是技能激活时 agent 看到的指令。
:::

# Webhook 订阅

创建动态 webhook 订阅，以便外部服务（GitHub、GitLab、Stripe、CI/CD、IoT 传感器、监控工具）可以通过 POST 事件到 URL 来触发 Hermites agent 运行。

## 设置（首先需要）

必须启用 webhook 平台才能创建订阅。使用以下命令检查：
```bash
hermes webhook list
```

如果显示"Webhook 平台未启用"，请按以下方式设置：

### 选项 1：设置向导
```bash
hermes gateway setup
```
按照提示启用 webhooks、设置端口和设置全局 HMAC 密钥。

### 选项 2：手动配置
添加到 `~/.hermes/config.yaml`：
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
添加到 `~/.hermes/.env`：
```bash
WEBHOOK_ENABLED=true
WEBHOOK_PORT=8644
WEBHOOK_SECRET=generate-a-strong-secret-here
```

配置后，启动（或重启）gateway：
```bash
hermes gateway run
# 或者如果使用 systemd：
systemctl --user restart hermes-gateway
```

验证正在运行：
```bash
curl http://localhost:8644/health
```

## 命令

所有管理都通过 `hermes webhook` CLI 命令：

### 创建订阅
```bash
hermes webhook subscribe <name> \
  --prompt "带有 {payload.fields} 的提示模板" \
  --events "event1,event2" \
  --description "这是做什么的" \
  --skills "skill1,skill2" \
  --deliver telegram \
  --deliver-chat-id "12345" \
  --secret "optional-custom-secret"
```

返回 webhook URL 和 HMAC 密钥。用户配置他们的服务 POST 到该 URL。

### 列出订阅
```bash
hermes webhook list
```

### 移除订阅
```bash
hermes webhook remove <name>
```

### 测试订阅
```bash
hermes webhook test <name>
hermes webhook test <name> --payload '{"key": "value"}'
```

## 提示模板

提示支持 `{dot.notation}` 访问嵌套的有效载荷字段：

- `{issue.title}` — GitHub issue 标题
- `{pull_request.user.login}` — PR 作者
- `{data.object.amount}` — Stripe 支付金额
- `{sensor.temperature}` — IoT 传感器读数

如果未指定提示，则将完整 JSON 有效载荷转储到 agent 提示中。

## 常见模式

### GitHub：新 issues
```bash
hermes webhook subscribe github-issues \
  --events "issues" \
  --prompt "新的 GitHub issue #{issue.number}: {issue.title}\n\n操作: {action}\n作者: {issue.user.login}\n正文:\n{issue.body}\n\n请对这个问题进行分类。" \
  --deliver telegram \
  --deliver-chat-id "-100123456789"
```

然后在 GitHub 仓库设置 → Webhooks → 添加 webhook：
- Payload URL：返回的 webhook_url
- Content type: application/json
- Secret: 返回的密钥
- Events: "Issues"

### GitHub：PR 审阅
```bash
hermes webhook subscribe github-prs \
  --events "pull_request" \
  --prompt "PR #{pull_request.number} {action}: {pull_request.title}\n作者: {pull_request.user.login}\n分支: {pull_request.head.ref}\n\n{pull_request.body}" \
  --skills "github-code-review" \
  --deliver github_comment
```

### Stripe：支付事件
```bash
hermes webhook subscribe stripe-payments \
  --events "payment_intent.succeeded,payment_intent.payment_failed" \
  --prompt "支付 {data.object.status}: {data.object.amount} 美分来自 {data.object.receipt_email}" \
  --deliver telegram \
  --deliver-chat-id "-100123456789"
```

### CI/CD：构建通知
```bash
hermes webhook subscribe ci-builds \
  --events "pipeline" \
  --prompt "在 {project.name} 分支 {object_attributes.ref} 上的构建 {object_attributes.status}\n提交: {commit.message}" \
  --deliver discord \
  --deliver-chat-id "1234567890"
```

### 通用监控告警
```bash
hermes webhook subscribe alerts \
  --prompt "告警: {alert.name}\n严重性: {alert.severity}\n消息: {alert.message}\n\n请调查并建议补救措施。" \
  --deliver origin
```

### 直接传递（无 agent，零 LLM 成本）

对于您只想将通知推送到用户聊天的用例 — 无推理，无 agent 循环 — 添加 `--deliver-only`。渲染的 `--prompt` 模板成为字面消息正文，直接分派到目标适配器。

用于：
- 外部服务推送通知（Supabase/Firebase webhooks → Telegram）
- 应该转发原文的监控告警
- agent 间 ping，其中一个 agent 告诉另一个 agent 的用户某些内容
- 任何 LLM 往返都是浪费的 webhook

```bash
hermes webhook subscribe antenna-matches \
  --deliver telegram \
  --deliver-chat-id "123456789" \
  --deliver-only \
  --prompt "🎉 新匹配: {match.user_name} 与你匹配了！" \
  --description "Antenna 匹配通知"
```

POST 在成功传递时返回 `200 OK`，在目标失败时返回 `502` — 因此上游服务可以智能重试。HMAC 认证、速率限制和幂等性仍然适用。

需要 `--deliver` 是一个真实目标（telegram、discord、slack、github_comment 等）— `--deliver log` 被拒绝，因为仅日志直接传递毫无意义。

## 安全

- 每个订阅获取一个自动生成的 HMAC-SHA256 密钥（或者使用 `--secret` 提供您自己的）
- webhook 适配器在每个传入 POST 上验证签名
- config.yaml 中的静态路由不能被动态订阅覆盖
- 订阅持久化到 `~/.hermes/webhook_subscriptions.json`

## 工作原理

1. `hermes webhook subscribe` 写入 `~/.hermes/webhook_subscriptions.json`
2. webhook 适配器在每个传入请求上热重载此文件（基于 mtime 门控，开销可忽略不计）
3. 当匹配的 POST 到达时，适配器格式化提示并触发 agent 运行
4. agent 的响应传递到配置的目标（Telegram、Discord、GitHub 评论等）

## 故障排除

如果 webhook 不工作：

1. **gateway 正在运行吗？** 使用 `systemctl --user status hermes-gateway` 或 `ps aux | grep gateway` 检查
2. **webhook 服务器正在监听吗？** `curl http://localhost:8644/health` 应返回 `{"status": "ok"}`
3. **检查 gateway 日志：** `grep webhook ~/.hermes/logs/gateway.log | tail -20`
4. **签名不匹配？** 验证您服务中的密钥与 `hermes webhook list` 返回的匹配。GitHub 发送 `X-Hub-Signature-256`，GitLab 发送 `X-Gitlab-Token`。
5. **防火墙/NAT？** webhook URL 必须可从服务访问。对于本地开发，使用隧道（ngrok、cloudflared）。
6. **错误的事件类型？** 检查 `--events` 过滤器与发送的内容匹配。使用 `hermes webhook test <name>` 验证路由工作。
