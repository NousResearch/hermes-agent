---
sidebar_position: 13
title: "Webhooks"
description: "接收来自 GitHub、GitLab 等服务的事件以触发 Hermes agent 运行"
---

# Webhooks

接收来自外部服务（GitHub、GitLab、JIRA、Stripe 等）的事件，并自动触发 Hermes agent 运行。Webhook 适配器运行一个 HTTP 服务器，接受 POST 请求、验证 HMAC 签名、将 payload（载荷）转换为 agent prompt（提示词），并将响应路由回来源或其他已配置的平台。

agent 处理事件后，可通过在 PR 上发布评论、向 Telegram/Discord 发送消息或记录结果来响应。

## 视频教程

<div style={{position: 'relative', width: '100%', aspectRatio: '16 / 9', marginBottom: '1.5rem'}}>
  <iframe
    src="https://www.youtube.com/embed/WNYe5mD4fY8"
    title="Hermes Agent — Webhooks Tutorial"
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', border: 0}}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowFullScreen
  />
</div>

---

## 快速开始

1. 通过 `hermes gateway setup` 或环境变量启用
2. 在 `config.yaml` 中定义路由，**或**使用 `hermes webhook subscribe` 动态创建
3. 将你的服务指向 `http://your-server:8644/webhooks/<route-name>`

---

## 设置

有两种方式启用 webhook 适配器。

### 通过设置向导

```bash
hermes gateway setup
```

按照提示启用 webhooks、设置端口和全局 HMAC secret。

### 通过环境变量

添加到 `~/.hermes/.env`：

```bash
WEBHOOK_ENABLED=true
WEBHOOK_PORT=8644        # default
WEBHOOK_SECRET=your-global-secret
```

### 验证服务器

gateway 运行后：

```bash
curl http://localhost:8644/health
```

预期响应：

```json
{"status": "ok", "platform": "webhook"}
```

---

## 配置路由 {#configuring-routes}

路由定义了不同 webhook 来源的处理方式。每个路由是 `config.yaml` 中 `platforms.webhook.extra.routes` 下的一个命名条目。

### 路由属性

| 属性 | 是否必填 | 描述 |
|----------|----------|-------------|
| `events` | 否 | 要接受的事件类型列表（例如 `["pull_request"]`）。若为空，则接受所有事件。事件类型从 `X-GitHub-Event`、`X-GitLab-Event` 或 payload 中的 `event_type` 读取。 |
| `secret` | **是** | 用于签名验证的 HMAC secret。若路由未设置，则回退到全局 `secret`。仅用于测试时可设为 `"INSECURE_NO_AUTH"`（跳过验证）。 |
| `prompt` | 否 | 使用点号表示法访问 payload 字段的模板字符串（例如 `{pull_request.title}`）。若省略，则将完整 JSON payload 转储到 prompt 中。 |
| `skills` | 否 | agent 运行时加载的 skill 名称列表。 |
| `deliver` | 否 | 响应发送目标：`github_comment`、`telegram`、`discord`、`slack`、`signal`、`sms`、`whatsapp`、`matrix`、`mattermost`、`homeassistant`、`email`、`dingtalk`、`feishu`、`wecom`、`weixin`、`bluebubbles`、`qqbot`，或 `log`（默认）。 |
| `deliver_extra` | 否 | 额外的投递配置——键取决于 `deliver` 类型（例如 `repo`、`pr_number`、`chat_id`）。值支持与 `prompt` 相同的 `{dot.notation}` 模板语法。 |
| `deliver_only` | 否 | 若为 `true`，完全跳过 agent——渲染后的 `prompt` 模板直接作为消息体投递。零 LLM token 消耗，亚秒级投递。参见[直接投递模式](#direct-delivery-mode)了解使用场景。要求 `deliver` 为真实目标（非 `log`）。 |

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
            Review this pull request:
            Repository: {repository.full_name}
            PR #{number}: {pull_request.title}
            Author: {pull_request.user.login}
            URL: {pull_request.html_url}
            Diff URL: {pull_request.diff_url}
            Action: {action}
          skills: ["github-code-review"]
          deliver: "github_comment"
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
        deploy-notify:
          events: ["push"]
          secret: "deploy-secret"
          prompt: "New push to {repository.full_name} branch {ref}: {head_commit.message}"
          deliver: "telegram"
```

### Prompt 模板

Prompt 使用点号表示法访问 webhook payload 中的嵌套字段：

- `{pull_request.title}` 解析为 `payload["pull_request"]["title"]`
- `{repository.full_name}` 解析为 `payload["repository"]["full_name"]`
- `{__raw__}` — 特殊 token，将**整个 payload** 以缩进 JSON 格式转储（截断至 4000 个字符）。适用于监控告警或通用 webhook，agent 需要完整上下文时使用。
- 缺失的键保留为字面量 `{key}` 字符串（不报错）
- 嵌套的 dict 和 list 会被 JSON 序列化并截断至 2000 个字符

可以将 `{__raw__}` 与常规模板变量混合使用：

```yaml
prompt: "PR #{pull_request.number} by {pull_request.user.login}: {__raw__}"
```

若路由未配置 `prompt` 模板，则将整个 payload 以缩进 JSON 格式转储（截断至 4000 个字符）。

`deliver_extra` 的值中同样支持点号表示法模板。

### 论坛话题投递

向 Telegram 投递 webhook 响应时，可通过在 `deliver_extra` 中包含 `message_thread_id`（或 `thread_id`）来指定特定论坛话题：

```yaml
webhooks:
  routes:
    alerts:
      events: ["alert"]
      prompt: "Alert: {__raw__}"
      deliver: "telegram"
      deliver_extra:
        chat_id: "-1001234567890"
        message_thread_id: "42"
```

若 `deliver_extra` 中未提供 `chat_id`，则回退到目标平台配置的主频道。

---

## GitHub PR 审查（分步说明） {#github-pr-review}

本演练将为每个 pull request 设置自动代码审查。

### 1. 在 GitHub 中创建 webhook

1. 进入你的仓库 → **Settings** → **Webhooks** → **Add webhook**
2. 将 **Payload URL** 设为 `http://your-server:8644/webhooks/github-pr`
3. 将 **Content type** 设为 `application/json`
4. 将 **Secret** 设为与路由配置匹配的值（例如 `github-webhook-secret`）
5. 在 **Which events?** 下，选择 **Let me select individual events** 并勾选 **Pull requests**
6. 点击 **Add webhook**

### 2. 添加路由配置

按照上方示例，将 `github-pr` 路由添加到 `~/.hermes/config.yaml`。

### 3. 确保 `gh` CLI 已认证

`github_comment` 投递类型使用 GitHub CLI 发布评论：

```bash
gh auth login
```

### 4. 测试

在仓库中打开一个 pull request。webhook 触发后，Hermes 处理事件并在 PR 上发布审查评论。

---

## GitLab Webhook 设置 {#gitlab-webhook-setup}

GitLab webhook 的工作方式类似，但使用不同的认证机制。GitLab 通过 `X-Gitlab-Token` 请求头以明文字符串匹配（非 HMAC）发送 secret。

### 1. 在 GitLab 中创建 webhook

1. 进入你的项目 → **Settings** → **Webhooks**
2. 将 **URL** 设为 `http://your-server:8644/webhooks/gitlab-mr`
3. 输入你的 **Secret token**
4. 选择 **Merge request events**（以及其他你需要的事件）
5. 点击 **Add webhook**

### 2. 添加路由配置

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
            Review this merge request:
            Project: {project.path_with_namespace}
            MR !{object_attributes.iid}: {object_attributes.title}
            Author: {object_attributes.last_commit.author.name}
            URL: {object_attributes.url}
            Action: {object_attributes.action}
          deliver: "log"
```

---

## 投递选项 {#delivery-options}

`deliver` 字段控制 agent 处理 webhook 事件后响应的发送目标。

| 投递类型 | 描述 |
|-------------|-------------|
| `log` | 将响应记录到 gateway 日志输出。这是默认值，适合测试使用。 |
| `github_comment` | 通过 `gh` CLI 将响应作为 PR/issue 评论发布。需要 `deliver_extra.repo` 和 `deliver_extra.pr_number`。`gh` CLI 必须安装并在 gateway 主机上完成认证（`gh auth login`）。 |
| `telegram` | 将响应路由到 Telegram。使用主频道，或在 `deliver_extra` 中指定 `chat_id`。 |
| `discord` | 将响应路由到 Discord。使用主频道，或在 `deliver_extra` 中指定 `chat_id`。 |
| `slack` | 将响应路由到 Slack。使用主频道，或在 `deliver_extra` 中指定 `chat_id`。 |
| `signal` | 将响应路由到 Signal。使用主频道，或在 `deliver_extra` 中指定 `chat_id`。 |
| `sms` | 通过 Twilio 将响应路由到 SMS。使用主频道，或在 `deliver_extra` 中指定 `chat_id`。 |
| `whatsapp` | 将响应路由到 WhatsApp。使用主频道，或在 `deliver_extra` 中指定 `chat_id`。 |
| `matrix` | 将响应路由到 Matrix。使用主频道，或在 `deliver_extra` 中指定 `chat_id`。 |
| `mattermost` | 将响应路由到 Mattermost。使用主频道，或在 `deliver_extra` 中指定 `chat_id`。 |
| `homeassistant` | 将响应路由到 Home Assistant。使用主频道，或在 `deliver_extra` 中指定 `chat_id`。 |
| `email` | 将响应路由到 Email。使用主频道，或在 `deliver_extra` 中指定 `chat_id`。 |
| `dingtalk` | 将响应路由到 DingTalk。使用主频道，或在 `deliver_extra` 中指定 `chat_id`。 |
| `feishu` | 将响应路由到 Feishu/Lark。使用主频道，或在 `deliver_extra` 中指定 `chat_id`。 |
| `wecom` | 将响应路由到 WeCom。使用主频道，或在 `deliver_extra` 中指定 `chat_id`。 |
| `weixin` | 将响应路由到 Weixin（微信）。使用主频道，或在 `deliver_extra` 中指定 `chat_id`。 |
| `bluebubbles` | 将响应路由到 BlueBubbles（iMessage）。使用主频道，或在 `deliver_extra` 中指定 `chat_id`。 |

跨平台投递时，目标平台也必须在 gateway 中启用并连接。若 `deliver_extra` 中未提供 `chat_id`，响应将发送到该平台配置的主频道。

---

## 直接投递模式 {#direct-delivery-mode}

默认情况下，每次 webhook POST 都会触发一次 agent 运行——payload 成为 prompt，agent 处理后投递响应。这会在每次事件时消耗 LLM token。

对于只需**推送纯文本通知**的场景——无需推理、无需 agent 循环，只需投递消息——可在路由上设置 `deliver_only: true`。渲染后的 `prompt` 模板直接作为消息体，适配器将其直接分发到配置的投递目标。

### 何时使用直接投递

- **外部服务推送** — Supabase/Firebase webhook 在数据库变更时触发 → 即时通知 Telegram 用户
- **监控告警** — Datadog/Grafana 告警 webhook → 推送到 Discord 频道
- **agent 间通知** — Agent A 通知 Agent B 的用户某个长时任务已完成
- **后台任务完成** — Cron 任务完成 → 将结果发布到 Slack

优势：

- **零 LLM token** — agent 从不被调用
- **亚秒级投递** — 单次适配器调用，无推理循环
- **与 agent 模式相同的安全性** — HMAC 认证、速率限制、幂等性和请求体大小限制均正常生效
- **同步响应** — 投递成功后 POST 返回 `200 OK`，若目标拒绝则返回 `502`，便于上游服务智能重试

### 示例：从 Supabase 推送到 Telegram

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644
      secret: "global-secret"
      routes:
        antenna-matches:
          secret: "antenna-webhook-secret"
          deliver: "telegram"
          deliver_only: true
          prompt: "🎉 New match: {match.user_name} matched with you!"
          deliver_extra:
            chat_id: "{match.telegram_chat_id}"
```

你的 Supabase edge function 使用 HMAC-SHA256 对 payload 签名并 POST 到 `https://your-server:8644/webhooks/antenna-matches`。webhook 适配器验证签名、从 payload 渲染模板、投递到 Telegram，并返回 `200 OK`。

### 示例：通过 CLI 动态订阅

```bash
hermes webhook subscribe antenna-matches \
  --deliver telegram \
  --deliver-chat-id "123456789" \
  --deliver-only \
  --prompt "🎉 New match: {match.user_name} matched with you!" \
  --description "Antenna match notifications"
```

### 响应状态码

| 状态码 | 含义 |
|--------|---------|
| `200 OK` | 投递成功。响应体：`{"status": "delivered", "route": "...", "target": "...", "delivery_id": "..."}` |
| `200 OK`（status=duplicate） | 在幂等性 TTL（1 小时）内重复的 `X-GitHub-Delivery` ID。不重复投递。 |
| `401 Unauthorized` | HMAC 签名无效或缺失。 |
| `400 Bad Request` | JSON 请求体格式错误。 |
| `404 Not Found` | 未知路由名称。 |
| `413 Payload Too Large` | 请求体超过 `max_body_bytes`。 |
| `429 Too Many Requests` | 路由速率限制已超出。 |
| `502 Bad Gateway` | 目标适配器拒绝消息或抛出异常。错误记录在服务端日志中；响应体为通用的 `Delivery failed`，避免泄露适配器内部信息。 |

### 配置注意事项

- `deliver_only: true` 要求 `deliver` 为真实目标。`deliver: log`（或省略 `deliver`）在启动时会被拒绝——适配器发现路由配置错误时拒绝启动。
- 直接投递模式下 `skills` 字段被忽略（不运行 agent，无处注入 skill）。
- 模板渲染使用与 agent 模式相同的 `{dot.notation}` 语法，包括 `{__raw__}` token。
- 幂等性使用相同的 `X-GitHub-Delivery` / `X-Request-ID` 请求头——携带相同 ID 的重试返回 `status=duplicate` 且**不**重复投递。

---

## 动态订阅（CLI） {#dynamic-subscriptions}

除了 `config.yaml` 中的静态路由，还可以使用 `hermes webhook` CLI 命令动态创建 webhook 订阅。当 agent 本身需要设置事件驱动触发器时，这尤为有用。

### 创建订阅

```bash
hermes webhook subscribe github-issues \
  --events "issues" \
  --prompt "New issue #{issue.number}: {issue.title}\nBy: {issue.user.login}\n\n{issue.body}" \
  --deliver telegram \
  --deliver-chat-id "-100123456789" \
  --description "Triage new GitHub issues"
```

此命令返回 webhook URL 和自动生成的 HMAC secret。将你的服务配置为 POST 到该 URL。

### 列出订阅

```bash
hermes webhook list
```

### 删除订阅

```bash
hermes webhook remove github-issues
```

### 测试订阅

```bash
hermes webhook test github-issues
hermes webhook test github-issues --payload '{"issue": {"number": 42, "title": "Test"}}'
```

### 动态订阅的工作原理

- 订阅存储在 `~/.hermes/webhook_subscriptions.json`
- webhook 适配器在每次收到请求时热重载该文件（基于 mtime 检测，开销可忽略不计）
- `config.yaml` 中的静态路由始终优先于同名的动态订阅
- 动态订阅与静态路由使用相同的格式和功能（events、prompt 模板、skills、delivery）
- 无需重启 gateway——订阅后立即生效

### agent 驱动的订阅

agent 可通过 terminal 工具在 `webhook-subscriptions` skill 的引导下创建订阅。向 agent 请求"为 GitHub issues 设置 webhook"，它将运行相应的 `hermes webhook subscribe` 命令。

---

## 安全性 {#security}

webhook 适配器包含多层安全机制：

### HMAC 签名验证

适配器使用适合各来源的方式验证传入的 webhook 签名：

- **GitHub**：`X-Hub-Signature-256` 请求头——以 `sha256=` 为前缀的 HMAC-SHA256 十六进制摘要
- **GitLab**：`X-Gitlab-Token` 请求头——明文 secret 字符串匹配
- **通用**：`X-Webhook-Signature` 请求头——原始 HMAC-SHA256 十六进制摘要

若已配置 secret 但请求中不存在已识别的签名请求头，则请求被拒绝。

### Secret 为必填项

每个路由必须有 secret——直接设置在路由上或从全局 `secret` 继承。没有 secret 的路由会导致适配器在启动时报错退出。仅用于开发/测试时，可将 secret 设为 `"INSECURE_NO_AUTH"` 以完全跳过验证。

`INSECURE_NO_AUTH` 仅在 gateway 绑定到回环地址（`127.0.0.1`、`localhost`、`::1`）时被接受。若与非回环绑定（如 `0.0.0.0` 或局域网 IP）组合使用，适配器拒绝启动——这可防止在公共接口上意外暴露未认证的端点。

### 速率限制

每个路由默认限制为**每分钟 30 次请求**（固定窗口）。可全局配置：

```yaml
platforms:
  webhook:
    extra:
      rate_limit: 60  # requests per minute
```

超出限制的请求收到 `429 Too Many Requests` 响应。

### 幂等性

投递 ID（来自 `X-GitHub-Delivery`、`X-Request-ID` 或时间戳回退）缓存 **1 小时**。重复投递（例如 webhook 重试）会被静默跳过并返回 `200` 响应，防止重复触发 agent 运行。

### 请求体大小限制

超过 **1 MB** 的 payload 在读取请求体之前即被拒绝。可配置：

```yaml
platforms:
  webhook:
    extra:
      max_body_bytes: 2097152  # 2 MB
```

### Prompt 注入风险

:::warning
Webhook payload 包含攻击者可控的数据——PR 标题、commit 消息、issue 描述等均可能包含恶意指令。在暴露于互联网时，请在沙箱环境（Docker、VM）中运行 gateway。考虑使用 Docker 或 SSH terminal 后端进行隔离。
:::

---

## 故障排查 {#troubleshooting}

### Webhook 未到达

- 验证端口已暴露且可从 webhook 来源访问
- 检查防火墙规则——端口 `8644`（或你配置的端口）必须开放
- 验证 URL 路径是否匹配：`http://your-server:8644/webhooks/<route-name>`
- 使用 `/health` 端点确认服务器正在运行

### 签名验证失败

- 确保路由配置中的 secret 与 webhook 来源中配置的 secret 完全一致
- 对于 GitHub，secret 基于 HMAC——检查 `X-Hub-Signature-256`
- 对于 GitLab，secret 为明文 token 匹配——检查 `X-Gitlab-Token`
- 检查 gateway 日志中的 `Invalid signature` 警告

### 事件被忽略

- 检查事件类型是否在路由的 `events` 列表中
- GitHub 事件使用如 `pull_request`、`push`、`issues` 等值（`X-GitHub-Event` 请求头的值）
- GitLab 事件使用如 `merge_request`、`push` 等值（`X-GitLab-Event` 请求头的值）
- 若 `events` 为空或未设置，则接受所有事件

### Agent 未响应

- 在前台运行 gateway 以查看日志：`hermes gateway run`
- 检查 prompt 模板是否正确渲染
- 验证投递目标已配置并连接

### 重复响应

- 幂等性缓存应能防止此问题——检查 webhook 来源是否发送了投递 ID 请求头（`X-GitHub-Delivery` 或 `X-Request-ID`）
- 投递 ID 缓存 1 小时

### `gh` CLI 错误（GitHub 评论投递）

- 在 gateway 主机上运行 `gh auth login`
- 确保已认证的 GitHub 用户对该仓库有写权限
- 检查 `gh` 是否已安装并在 PATH 中

---

## 环境变量 {#environment-variables}

| 变量 | 描述 | 默认值 |
|----------|-------------|---------|
| `WEBHOOK_ENABLED` | 启用 webhook 平台适配器 | `false` |
| `WEBHOOK_PORT` | 接收 webhook 的 HTTP 服务器端口 | `8644` |
| `WEBHOOK_SECRET` | 全局 HMAC secret（路由未指定自身 secret 时作为回退） | _（无）_ |