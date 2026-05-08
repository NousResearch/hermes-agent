---
sidebar_position: 13
title: "Webhooks"
description: "接收 GitHub、GitLab 和其他服务的事件，自动触发 Hermes 智能体运行"
---

# Webhooks

接收外部服务（GitHub、GitLab、JIRA、Stripe 等）的事件，自动触发 Hermes 智能体运行。Webhook 适配器运行一个 HTTP 服务器，接收 POST 请求，验证 HMAC 签名，将载荷转换为智能体提示，并将响应路由回源或其他配置的平台。

智能体处理事件，可以通过在 PR 上发布评论、发送消息到 Telegram/Discord 或记录结果来响应。

## 快速开始

1. 通过 `hermes gateway setup` 或环境变量启用
2. 在 `config.yaml` 中定义路由 **或** 用 `hermes webhook subscribe` 动态创建
3. 将你的服务指向 `http://your-server:8644/webhooks/<route-name>`

## 设置

有两种方式启用 webhook 适配器。

### 通过设置向导

```bash
hermes gateway setup
```

按照提示启用 webhooks，设置端口，设置全局 HMAC secret。

### 通过环境变量

添加到 `~/.hermes/.env`：

```bash
WEBHOOK_ENABLED=true
WEBHOOK_PORT=8644        # 默认
WEBHOOK_SECRET=your-global-secret
```

### 验证服务器

```bash
curl http://localhost:8644/health
```

预期响应：`{"status": "ok", "platform": "webhook"}`

---

## 配置路由

路由定义如何处理不同的 webhook 来源。每个路由是你 `config.yaml` 中 `platforms.webhook.extra.routes` 下的命名条目。

### 路由属性

| 属性 | 必需 | 说明 |
|----------|----------|-------------|
| `events` | 否 | 要接受的事件类型列表。如果为空，接受所有事件。 |
| `secret` | **是** | HMAC secret 用于签名验证。如果未设置，回退到全局 `secret`。设为 `"INSECURE_NO_AUTH"` 仅用于测试（跳过验证）。 |
| `prompt` | 否 | 带点号表示法的模板字符串（例如 `{pull_request.title}`）。如果省略，整个 JSON 载荷被转储到提示中。 |
| `skills` | 否 | 为智能体运行加载的 skill 名称列表。 |
| `deliver` | 否 | 响应发送到哪里：`github_comment`、`telegram`、`discord`、`slack`、`signal`、`sms`、`whatsapp`、`matrix`、`mattermost`、`homeassistant`、`email`、`dingtalk`、`feishu`、`wecom`、`weixin`、`bluebubbles`、`qqbot` 或 `log`（默认）。 |
| `deliver_extra` | 否 | 附加投递配置 — 键取决于 `deliver` 类型（例如 `repo`、`pr_number`、`chat_id`）。值支持与 `prompt` 相同的 `{dot.notation}` 模板。 |

### 完整示例

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644
      secret: "global-fallback-secret"
      routes:
        github-pr:
          events: ["pull_request"]
          secret: "github-webhook-secret"
          prompt: |
            审查此 pull request：
            仓库：{repository.full_name}
            PR #{number}: {pull_request.title}
            作者：{pull_request.user.login}
            URL：{pull_request.html_url}
          skills: ["github-code-review"]
          deliver: "github_comment"
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
```

### 提示模板

提示使用点号表示法访问 webhook 载荷中的嵌套字段：

- `{pull_request.title}` 解析为 `payload["pull_request"]["title"]`
- `{__raw__}` — 特殊标记，将**整个载荷**转储为缩进的 JSON（截断到 4000 字符）

---

## GitHub PR 审查（分步）

### 1. 在 GitHub 中创建 webhook

1. 进入你的仓库 → **Settings** → **Webhooks** → **Add webhook**
2. 设置 **Payload URL** 为 `http://your-server:8644/webhooks/github-pr`
3. 设置 **Content type** 为 `application/json`
4. 设置 **Secret** 与你的路由配置匹配
5. 在 **Which events?** 下，选择 **Let me select individual events** 并勾选 **Pull requests**

### 2. 添加路由配置

将 `github-pr` 路由添加到你的 `~/.hermes/config.yaml`。

### 3. 确保 `gh` CLI 已认证

```bash
gh auth login
```

### 4. 测试

在仓库上打开 pull request。Webhook 触发，Hermes 处理事件，并在 PR 上发布审查评论。

---

## GitLab Webhook 设置

GitLab webhook 工作方式类似，但使用不同的认证机制。GitLab 将 secret 作为纯 `X-Gitlab-Token` 头发送（精确字符串匹配，非 HMAC）。

### 配置示例

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        gitlab-mr:
          events: ["merge_request"]
          secret: "your-gitlab-secret-token"
          prompt: |
            审查此 merge request：
            项目：{project.path_with_namespace}
            MR !{object_attributes.iid}: {object_attributes.title}
            作者：{object_attributes.last_commit.author.name}
          deliver: "log"
```

---

## 投递选项

| 投递类型 | 说明 |
|-------------|-------------|
| `log` | 将响应记录到 gateway 日志输出。默认。 |
| `github_comment` | 通过 `gh` CLI 将响应发布为 PR/issue 评论。需要 `deliver_extra.repo` 和 `deliver_extra.pr_number`。 |
| `telegram` | 将响应路由到 Telegram。使用主频道或在 `deliver_extra` 中指定 `chat_id`。 |
| `discord` | 将响应路由到 Discord。 |
| `slack` | 将响应路由到 Slack。 |
| `signal` | 将响应路由到 Signal。 |
| `sms` | 通过 Twilio 将响应路由到 SMS。 |
| `whatsapp` | 将响应路由到 WhatsApp。 |
| `matrix` | 将响应路由到 Matrix。 |
| `email` | 将响应路由到 Email。 |

---

## 直接投递模式

默认，每个 webhook POST 触发智能体运行 — 载荷变为提示，智能体处理它，智能体响应被投递。

对于只想**推送纯通知**的用例 — 无推理、无智能体循环、只投递消息 — 在路由上设置 `deliver_only: true`。渲染的 `prompt` 模板成为字面消息体，适配器直接分派到配置的目标。

### 何时使用直接投递

- **外部服务推送** — Supabase/Firebase webhook 在数据库更改时触发 → 立即在 Telegram 中通知用户
- **监控告警** — Datadog/Grafana 告警 webhook → 推送到 Discord 频道
- **智能体间 ping** — 智能体 A 通知智能体 B 的用户长时间运行的任务完成

### 好处

- **零 LLM tokens** — 智能体从不调用
- **亚秒投递** — 单一适配器调用，无推理循环
- **与智能体模式相同的安全** — HMAC 认证、速率限制、幂等性和主体大小限制都适用

---

## 动态订阅（CLI）

除了 `config.yaml` 中的静态路由，你还可以用 `hermes webhook` CLI 命令动态创建 webhook 订阅。

### 创建订阅

```bash
hermes webhook subscribe github-issues \
  --events "issues" \
  --prompt "新 issue #{issue.number}: {issue.title}\nBy: {issue.user.login}" \
  --deliver telegram \
  --deliver-chat-id "-100123456789" \
  --description "分流新 GitHub issues"
```

### 列出订阅

```bash
hermes webhook list
```

### 移除订阅

```bash
hermes webhook remove github-issues
```

---

## 安全

### HMAC 签名验证

适配器使用适合每个来源的方法验证传入 webhook 签名：

- **GitHub**: `X-Hub-Signature-256` 头 — HMAC-SHA256 十六进制摘要，前缀为 `sha256=`
- **GitLab**: `X-Gitlab-Token` 头 — 纯 secret 字符串匹配
- **通用**: `X-Webhook-Signature` 头 — 原始 HMAC-SHA256 十六进制摘要

### 速率限制

每个路由默认限制为 **每分钟 30 请求**（固定窗口）。超过限制的请求收到 `429 Too Many Requests` 响应。

### 幂等性

投递 ID（来自 `X-GitHub-Delivery`、`X-Request-ID` 或时间戳回退）缓存 **1 小时**。重复投递静默跳过。

### 提示注入风险

:::warning
Webhook 载荷包含攻击者控制的数据 — PR 标题、提交消息、issue 描述等都可能包含恶意指令。暴露到互联网时，在沙箱环境（Docker、VM）中运行 gateway。
:::

---

## 故障排除

| 问题 | 解决方案 |
|---------|----------|
| Webhook 未到达 | 验证端口已暴露且可从 webhook 来源访问。检查防火墙规则。 |
| 签名验证失败 | 确保路由配置中的 secret 与 webhook 来源配置的 secret 完全匹配。检查 gateway 日志中的 `Invalid signature` 警告。 |
| 事件被忽略 | 检查事件类型是否在路由的 `events` 列表中。 |
| 智能体不响应 | 前台运行 gateway 查看日志。检查提示模板是否正确渲染。 |

---

## 环境变量

| 变量 | 说明 | 默认 |
|----------|-------------|---------|
| `WEBHOOK_ENABLED` | 启用 webhook 平台适配器 | `false` |
| `WEBHOOK_PORT` | 接收 webhooks 的 HTTP 服务器端口 | `8644` |
| `WEBHOOK_SECRET` | 全局 HMAC secret | _(无)_ |
