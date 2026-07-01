---
sidebar_position: 11
sidebar_label: "通过 Webhook 进行 GitHub PR 审查"
title: "使用 Webhook 自动发布 GitHub PR 评论"
description: "将 Hermes 连接到 GitHub，使其自动获取 PR diff、审查代码变更并发布评论——由 webhook 触发，无需手动提示"
---

# 使用 Webhook 自动发布 GitHub PR 评论

本指南介绍如何将 Hermes Agent 连接到 GitHub，使其自动获取 pull request 的 diff、分析代码变更并发布评论——由 webhook 事件触发，无需手动 prompt（提示词）。

当 PR 被打开或更新时，GitHub 会向你的 Hermes 实例发送一个 webhook POST 请求。Hermes 使用一个 prompt 运行 agent，该 prompt 指示其通过 `gh` CLI 获取 diff，并将响应发布回 PR 线程。

:::tip 想要无需公网端点的更简单配置？
如果你没有公网 URL，或只是想快速上手，请查看 [构建 GitHub PR 审查 Agent](./github-pr-review-agent.md) —— 使用 cron 作业按计划轮询 PR，可在 NAT 和防火墙后运行。
:::

:::info 参考文档
完整的 webhook 平台参考（所有配置选项、投递类型、动态订阅、安全模型），请参阅 [Webhooks](/user-guide/messaging/webhooks)。
:::

:::warning Prompt 注入风险
Webhook payload 包含攻击者可控的数据——PR 标题、commit 消息和描述中可能包含恶意指令。当你的 webhook 端点暴露在公网时，请在沙箱环境（Docker、SSH 后端）中运行 gateway。请参阅下方的[安全说明](#security-notes)。
:::

---

## 前提条件

- Hermes Agent 已安装并运行（`hermes gateway`）
- [`gh` CLI](https://cli.github.com/) 已安装并在 gateway 主机上完成认证（`gh auth login`）
- 你的 Hermes 实例有一个可公网访问的 URL（如果在本地运行，请参阅[使用 ngrok 进行本地测试](#local-testing-with-ngrok)）
- 对 GitHub 仓库的管理员权限（管理 webhook 所需）

---

## 第一步——启用 webhook 平台

在你的 `~/.hermes/config.yaml` 中添加以下内容：

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644          # 默认值；如果该端口被其他服务占用，请修改
      rate_limit: 30      # 每条路由每分钟最大请求数（非全局上限）

      routes:
        github-pr-review:
          secret: "your-webhook-secret-here"   # 必须与 GitHub webhook secret 完全一致
          events:
            - pull_request

          # agent 被指示在审查前先获取实际的 diff。
          # {number} 和 {repository.full_name} 从 GitHub payload 中解析。
          prompt: |
            A pull request event was received (action: {action}).

            PR #{number}: {pull_request.title}
            Author: {pull_request.user.login}
            Branch: {pull_request.head.ref} → {pull_request.base.ref}
            Description: {pull_request.body}
            URL: {pull_request.html_url}

            If the action is "closed" or "labeled", stop here and do not post a comment.

            Otherwise:
            1. Run: gh pr diff {number} --repo {repository.full_name}
            2. Review the code changes for correctness, security issues, and clarity.
            3. Write a concise, actionable review comment and post it.

          deliver: github_comment
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
```

**关键字段：**

| 字段 | 说明 |
|---|---|
| `secret`（路由级别） | 该路由的 HMAC secret。如果省略，则回退到 `extra.secret` 全局配置。 |
| `events` | 要接受的 `X-GitHub-Event` 请求头值列表。空列表 = 接受所有。 |
| `prompt` | 模板；`{field}` 和 `{nested.field}` 从 GitHub payload 中解析。 |
| `deliver` | `github_comment` 通过 `gh pr comment` 发布。`log` 仅写入 gateway 日志。 |
| `deliver_extra.repo` | 从 payload 中解析为例如 `org/repo`。 |
| `deliver_extra.pr_number` | 从 payload 中解析为 PR 编号。 |

:::note Payload 中不包含代码
GitHub webhook payload 包含 PR 元数据（标题、描述、分支名、URL），但**不包含 diff**。上方的 prompt 指示 agent 运行 `gh pr diff` 来获取实际变更。`terminal` 工具已包含在默认的 `hermes-webhook` 工具集中，无需额外配置。
:::

---

## 第二步——启动 gateway

```bash
hermes gateway
```

你应该看到：

```
[webhook] Listening on 0.0.0.0:8644 — routes: github-pr-review
```

验证其是否正在运行：

```bash
curl http://localhost:8644/health
# {"status": "ok", "platform": "webhook"}
```

---

## 第三步——在 GitHub 上注册 webhook

1. 进入你的仓库 → **Settings** → **Webhooks** → **Add webhook**
2. 填写：
   - **Payload URL：** `https://your-public-url.example.com/webhooks/github-pr-review`
   - **Content type：** `application/json`
   - **Secret：** 与路由配置中 `secret` 设置的值相同
   - **Which events?** → 选择单个事件 → 勾选 **Pull requests**
3. 点击 **Add webhook**

GitHub 会立即发送一个 `ping` 事件以确认连接。该事件会被安全忽略——`ping` 不在你的 `events` 列表中——并返回 `{"status": "ignored", "event": "ping"}`。它仅在 DEBUG 级别记录日志，因此不会在默认日志级别的控制台中显示。

---

## 第四步——打开一个测试 PR

创建一个分支，推送一个变更，并打开一个 PR。在 30–90 秒内（取决于 PR 大小和模型），Hermes 应该会发布一条审查评论。

要实时跟踪 agent 的进度：

```bash
tail -f "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
```

---

## 使用 ngrok 进行本地测试

如果 Hermes 在你的笔记本上运行，使用 [ngrok](https://ngrok.com/) 将其暴露到公网：

```bash
ngrok http 8644
```

复制 `https://...ngrok-free.app` URL 并将其用作你的 GitHub Payload URL。在 ngrok 免费版中，每次 ngrok 重启后 URL 都会变化——每次会话都需要更新你的 GitHub webhook。付费 ngrok 账户可获得静态域名。

你可以直接用 `curl` 对静态路由进行冒烟测试——无需 GitHub 账户或真实 PR。

:::tip 本地测试时使用 `deliver: log`
在测试时，将配置中的 `deliver: github_comment` 改为 `deliver: log`。否则 agent 将尝试向测试 payload 中的假 `org/repo#99` 仓库发布评论，这将会失败。对 prompt 输出满意后，再切换回 `deliver: github_comment`。
:::

```bash
SECRET="your-webhook-secret-here"
BODY='{"action":"opened","number":99,"pull_request":{"title":"Test PR","body":"Adds a feature.","user":{"login":"testuser"},"head":{"ref":"feat/x"},"base":{"ref":"main"},"html_url":"https://github.com/org/repo/pull/99"},"repository":{"full_name":"org/repo"}}'
SIG=$(printf '%s' "$BODY" | openssl dgst -sha256 -hmac "$SECRET" -hex | awk '{print "sha256="$2}')

curl -s -X POST http://localhost:8644/webhooks/github-pr-review \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: pull_request" \
  -H "X-Hub-Signature-256: $SIG" \
  -d "$BODY"
# Expected: {"status":"accepted","route":"github-pr-review","event":"pull_request","delivery_id":"..."}
```

然后观察 agent 运行：
```bash
tail -f "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
```

:::note
`hermes webhook test <name>` 仅适用于通过 `hermes webhook subscribe` 创建的**动态订阅**。它不读取 `config.yaml` 中的路由。
:::

---

## 过滤特定 action

GitHub 会针对多种 action 发送 `pull_request` 事件：`opened`、`synchronize`、`reopened`、`closed`、`labeled` 等。`events` 列表仅按 `X-GitHub-Event` 请求头值过滤——无法在路由级别按 action 子类型过滤。

第一步中的 prompt 已通过指示 agent 对 `closed` 和 `labeled` 事件提前停止来处理这一问题。

:::warning Agent 仍会运行并消耗 token（令牌）
"stop here" 指令会阻止有意义的审查，但无论 action 如何，agent 仍会对每个 `pull_request` 事件运行至完成。GitHub webhook 只能按事件类型（`pull_request`、`push`、`issues` 等）过滤——无法按 action 子类型（`opened`、`closed`、`labeled`）过滤。路由级别没有针对子 action 的过滤器。对于高流量仓库，请接受这一成本，或通过 GitHub Actions workflow 在上游进行过滤，有条件地调用你的 webhook URL。
:::

> 不支持 Jinja2 或条件模板语法。`{field}` 和 `{nested.field}` 是唯一支持的替换方式。其他内容会原样传递给 agent。

---

## 使用 skill 保持一致的审查风格

加载一个 [Hermes skill](/user-guide/features/skills) 以赋予 agent 一致的审查风格。在 `config.yaml` 的 `platforms.webhook.extra.routes` 中，向你的路由添加 `skills`：

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        github-pr-review:
          secret: "your-webhook-secret-here"
          events: [pull_request]
          prompt: |
            A pull request event was received (action: {action}).
            PR #{number}: {pull_request.title} by {pull_request.user.login}
            URL: {pull_request.html_url}

            If the action is "closed" or "labeled", stop here and do not post a comment.

            Otherwise:
            1. Run: gh pr diff {number} --repo {repository.full_name}
            2. Review the diff using your review guidelines.
            3. Write a concise, actionable review comment and post it.
          skills:
            - review
          deliver: github_comment
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
```

> **注意：** 列表中只有第一个找到的 skill 会被加载。Hermes 不会叠加多个 skill——后续条目会被忽略。

---

## 将响应发送到 Slack 或 Discord

将路由中的 `deliver` 和 `deliver_extra` 字段替换为你的目标平台：

```yaml
# 在 platforms.webhook.extra.routes.<route-name> 内部：

# Slack
deliver: slack
deliver_extra:
  chat_id: "C0123456789"   # Slack 频道 ID（省略则使用配置的默认频道）

# Discord
deliver: discord
deliver_extra:
  chat_id: "987654321012345678"  # Discord 频道 ID（省略则使用默认频道）
```

目标平台也必须在 gateway 中启用并连接。如果省略 `chat_id`，响应将发送到该平台配置的默认频道。

有效的 `deliver` 值：`log` · `github_comment` · `telegram` · `discord` · `slack` · `signal` · `sms`

---

## GitLab 支持

同一适配器也适用于 GitLab。GitLab 使用 `X-Gitlab-Token` 进行认证（纯字符串匹配，非 HMAC）——Hermes 会自动处理两者。

对于事件过滤，GitLab 将 `X-GitLab-Event` 设置为 `Merge Request Hook`、`Push Hook`、`Pipeline Hook` 等值。在 `events` 中使用精确的请求头值：

```yaml
events:
  - Merge Request Hook
```

GitLab 的 payload 字段与 GitHub 不同——例如，MR 标题使用 `{object_attributes.title}`，MR 编号使用 `{object_attributes.iid}`。发现完整 payload 结构最简单的方式是使用 GitLab webhook 设置中的 **Test** 按钮，结合 **Recent Deliveries** 日志。或者，在路由配置中省略 `prompt`——Hermes 将把完整 payload 作为格式化 JSON 直接传递给 agent，agent 的响应（在 gateway 日志中通过 `deliver: log` 可见）将描述其结构。

---

## 安全说明

- **永远不要在生产环境中使用 `INSECURE_NO_AUTH`**——它会完全禁用签名验证。仅用于本地开发。
- **定期轮换你的 webhook secret**，并在 GitHub（webhook 设置）和你的 `config.yaml` 中同步更新。
- **速率限制**默认为每条路由每分钟 30 次请求（可通过 `extra.rate_limit` 配置）。超出限制返回 `429`。
- **重复投递**（webhook 重试）通过 1 小时的幂等性缓存进行去重。缓存键依次为 `X-GitHub-Delivery`（如果存在）、`X-Request-ID`、毫秒级时间戳。当两个投递 ID 请求头都未设置时，重试**不会**去重。
- **Prompt 注入：** PR 标题、描述和 commit 消息均为攻击者可控内容。恶意 PR 可能尝试操纵 agent 的行为。当暴露在公网时，请在沙箱环境（Docker、VM）中运行 gateway。

---

## 故障排查

| 现象 | 检查项 |
|---|---|
| `401 Invalid signature` | config.yaml 中的 secret 与 GitHub webhook secret 不匹配 |
| `404 Unknown route` | URL 中的路由名称与 `routes:` 中的键不匹配 |
| `429 Rate limit exceeded` | 每条路由每分钟 30 次请求已超出——在 GitHub UI 中重新投递测试事件时常见；等待一分钟或提高 `extra.rate_limit` |
| 未发布评论 | `gh` 未安装、不在 PATH 中，或未完成认证（`gh auth login`） |
| Agent 运行但无评论 | 检查 gateway 日志——如果 agent 输出为空或仅为"SKIP"，投递仍会被尝试 |
| 端口已被占用 | 在 config.yaml 中修改 `extra.port` |
| Agent 运行但仅审查了 PR 描述 | prompt 中未包含 `gh pr diff` 指令——diff 不在 webhook payload 中 |
| 看不到 ping 事件 | 被忽略的事件仅在 DEBUG 日志级别返回 `{"status":"ignored","event":"ping"}`——检查 GitHub 的投递日志（仓库 → Settings → Webhooks → 你的 webhook → Recent Deliveries） |

**GitHub 的 Recent Deliveries 标签页**（仓库 → Settings → Webhooks → 你的 webhook）显示每次投递的精确请求头、payload、HTTP 状态和响应体。这是无需查看服务器日志即可诊断故障的最快方式。

---

## 完整配置参考

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      host: "0.0.0.0"         # 绑定地址（默认：0.0.0.0）
      port: 8644               # 监听端口（默认：8644）
      secret: ""               # 可选的全局回退 secret
      rate_limit: 30           # 每条路由每分钟请求数
      max_body_bytes: 1048576  # payload 大小限制，单位字节（默认：1 MB）

      routes:
        <route-name>:
          secret: "required-per-route"
          events: []            # [] = 接受所有；否则列出 X-GitHub-Event 值
          prompt: ""            # {field} / {nested.field} 从 payload 中解析
          skills: []            # 加载第一个匹配的 skill（仅一个）
          deliver: "log"        # log | github_comment | telegram | discord | slack | signal | sms
          deliver_extra: {}     # github_comment 需要 repo + pr_number；其他平台需要 chat_id
```

---

## 下一步

- **[基于 Cron 的 PR 审查](./github-pr-review-agent.md)** —— 按计划轮询 PR，无需公网端点
- **[Webhook 参考](/user-guide/messaging/webhooks)** —— webhook 平台的完整配置参考
- **[构建 Plugin](/guides/build-a-hermes-plugin)** —— 将审查逻辑打包为可共享的 plugin
- **[Profiles](/user-guide/profiles)** —— 运行一个拥有独立内存和配置的专属审查者 profile