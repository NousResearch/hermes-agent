---
sidebar_position: 11
sidebar_label: "通过 Webhook 的 GitHub PR 审查"
title: "使用 Webhooks 自动化 GitHub PR 评论"
description: "将 Hermes 连接到 GitHub，以便它自动获取 PR 差异、审查代码更改并发布评论——由 webhook 触发，无需手动提示"
---

# 使用 Webhooks 自动化 GitHub PR 评论

本指南引导你将 Hermes 代理连接到 GitHub，以便它自动获取拉取请求的差异、分析代码更改并发布评论——由 webhook 事件触发，无需手动提示。

当 PR 被打开或更新时，GitHub 向你的 Hermes 实例发送一个 webhook POST。Hermes 使用提示运行代理，指示它通过 `gh` CLI 检索差异，响应被发布回 PR 线程。

:::tip 想要更简单的设置而无需公共端点？
如果你没有公共 URL 或只是想快速开始，请查看[构建 GitHub PR 审查代理](./github-pr-review-agent.md)——使用 cron 作业按计划轮询 PR，可在 NAT 和防火墙后面工作。
:::

:::info 参考文档
有关完整的 webhook 平台参考（所有配置选项、交付类型、动态订阅、安全模型），请参阅 [Webhooks](/docs/user-guide/messaging/webhooks)。
:::

:::warning 提示注入风险
Webhook 有效载荷包含攻击者控制的数据——PR 标题、提交消息和描述可能包含恶意指令。当你的 webhook 端点暴露在互联网上时，在沙箱环境（Docker、SSH 后端）中运行网关。请参阅下面的[安全部分](#security-notes)。
:::

---

## 先决条件

- 已安装并运行 Hermes 代理（`hermes gateway`）
- 网关主机上已安装并认证 [`gh` CLI](https://cli.github.com/)（`gh auth login`）
- 你的 Hermes 实例的公共可达 URL（如果在本地运行，请参阅[使用 ngrok 进行本地测试](#local-testing-with-ngrok)）
- GitHub 仓库的管理员访问权限（需要管理 webhooks）

---

## 步骤 1 — 启用 webhook 平台

将以下内容添加到你的 `~/.hermes/config.yaml`：

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644          # 默认；如果另一个服务占用此端口则更改
      rate_limit: 30      # 每个路由每分钟最大请求数（不是全局上限）

      routes:
        github-pr-review:
          secret: "your-webhook-secret-here"   # 必须与 GitHub webhook 密钥完全匹配
          events:
            - pull_request

          # 代理被指示在审查之前获取实际差异。
          # {number} 和 {repository.full_name} 从 GitHub 有效载荷解析。
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

| 字段 | 描述 |
|---|---|
| `secret`（路由级别） | 此路由的 HMAC 密钥。如果省略，回退到 `extra.secret` 全局。 |
| `events` | 要接受的 `X-GitHub-Event` 头值列表。空列表 = 接受所有。 |
| `prompt` | 模板；`{field}` 和 `{nested.field}` 从 GitHub 有效载荷解析。 |
| `deliver` | `github_comment` 通过 `gh pr comment` 发布。`log` 仅写入网关日志。 |
| `deliver_extra.repo` | 从有效载荷解析为，例如 `org/repo`。 |
| `deliver_extra.pr_number` | 从有效载荷解析为 PR 编号。 |

:::note 有效载荷不包含代码
GitHub webhook 有效载荷包含 PR 元数据（标题、描述、分支名称、URL）但**不是差异**。上面的提示指示代理运行 `gh pr diff` 来获取实际更改。`terminal` 工具包含在默认的 `hermes-webhook` 工具集中，因此不需要额外配置。
:::

---

## 步骤 2 — 启动网关

```bash
hermes gateway
```

你应该看到：

```
[webhook] Listening on 0.0.0.0:8644 — routes: github-pr-review
```

验证它正在运行：

```bash
curl http://localhost:8644/health
# {"status": "ok", "platform": "webhook"}
```

---

## 步骤 3 — 在 GitHub 上注册 webhook

1. 进入你的仓库 → **Settings** → **Webhooks** → **Add webhook**
2. 填写：
   - **Payload URL:** `https://your-public-url.example.com/webhooks/github-pr-review`
   - **Content type:** `application/json`
   - **Secret:** 与你在路由配置中为 `secret` 设置的值相同
   - **Which events?** → 选择单个事件 → 检查 **Pull requests**
3. 点击 **Add webhook**

GitHub 将立即发送一个 `ping` 事件来确认连接。它被安全地忽略——`ping` 不在你的 `events` 列表中——并返回 `{"status": "ignored", "event": "ping"}`。它仅在 DEBUG 级别记录，因此不会在默认日志级别出现在控制台中。

---

## 步骤 4 — 打开测试 PR

创建一个分支，推送更改，并打开一个 PR。在 30-90 秒内（取决于 PR 大小和模型），Hermes 应该发布一个审查评论。

要实时跟踪代理的进度：

```bash
tail -f "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
```

---

## 使用 ngrok 进行本地测试

如果 Hermes 在你的笔记本电脑上运行，使用 [ngrok](https://ngrok.com/) 公开它：

```bash
ngrok http 8644
```

复制 `https://...ngrok-free.app` URL 并将其用作你的 GitHub Payload URL。在免费 ngrok 层，URL 每次重启 ngrok 时都会更改——在每个会话中更新你的 GitHub webhook。付费 ngrok 账户获得静态域名。

你可以用 `curl` 直接对静态路由进行冒烟测试——不需要 GitHub 账户或真实 PR。

:::tip 在本地测试时使用 `deliver: log`
在测试时将配置中的 `deliver: github_comment` 更改为 `deliver: log`。否则代理会尝试将评论发布到测试有效载荷中的虚假 `org/repo#99` 仓库，这会失败。一旦你对提示输出满意，就切换回 `deliver: github_comment`。
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

然后观察代理运行：
```bash
tail -f "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
```

:::note
`hermes webhook test <name>` 仅适用于使用 `hermes webhook subscribe` 创建的**动态订阅**。它不从 `config.yaml` 读取路由。
:::

---

## 过滤到特定操作

GitHub 为许多操作发送 `pull_request` 事件：`opened`、`synchronize`、`reopened`、`closed`、`labeled` 等。`events` 列表仅按 `X-GitHub-Event` 头值过滤——它不能在路由级别按操作子类型过滤。

步骤 1 中的提示已经通过指示代理对 `closed` 和 `labeled` 事件提前停止来处理这个问题。

:::warning 代理仍然运行并消耗令牌
"stop here" 指令阻止有意义的审查，但代理仍然为每个 `pull_request` 事件运行到完成，无论操作如何。GitHub webhook 只能按事件类型过滤（`pull_request`、`push`、`issues` 等）——不能按操作子类型（`opened`、`closed`、`labeled`）过滤。没有用于子操作的路由级过滤器。对于高流量仓库，接受此成本或在 upstream 使用 GitHub Actions 工作流进行条件过滤。
:::

> 没有 Jinja2 或条件模板语法。`{field}` 和 `{nested.field}` 是唯一支持的替换。其他任何内容都按原样传递给代理。

---

## 使用技能获得一致的审查风格

加载 [Hermes 技能](/docs/user-guide/features/skills) 以给代理一致的审查角色。将 `skills` 添加到 `config.yaml` 中 `platforms.webhook.extra.routes` 内的路由中：

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

> **注意：** 列表中只有找到的第一个技能被加载。Hermes 不会堆叠多个技能——后续条目被忽略。

---

## 改为发送到 Slack 或 Discord

将路由内的 `deliver` 和 `deliver_extra` 字段替换为你目标平台：

```yaml
# 在 platforms.webhook.extra.routes.<route-name> 内：

# Slack
deliver: slack
deliver_extra:
  chat_id: "C0123456789"   # Slack 频道 ID（省略以使用配置的主频道）

# Discord
deliver: discord
deliver_extra:
  chat_id: "987654321012345678"  # Discord 频道 ID（省略以使用主频道）
```

目标平台也必须在网关中启用并连接。如果 `chat_id` 被省略，响应被发送到该平台配置的主频道。

有效的 `deliver` 值：`log` · `github_comment` · `telegram` · `discord` · `slack` · `signal` · `sms`

---

## GitLab 支持

相同的适配器适用于 GitLab。GitLab 使用 `X-Gitlab-Token` 进行身份验证（纯字符串匹配，不是 HMAC）——Hermes 自动处理两者。

对于事件过滤，GitLab 将 `X-GitLab-Event` 设置为 `Merge Request Hook`、`Push Hook`、`Pipeline Hook` 等值。在 `events` 中使用确切的 header 值：

```yaml
events:
  - Merge Request Hook
```

GitLab 有效载荷字段与 GitHub 的不同——例如，MR 标题的 `{object_attributes.title}` 和 MR 编号的 `{object_attributes.iid}`。发现完整有效载荷结构的最简单方法是 GitLab webhook 设置中的 **Test** 按钮，结合 **Recent Deliveries** 日志。或者，从你的路由配置中省略 `prompt`——Hermes 然后会将完整有效载荷作为格式化 JSON 直接传递给代理，代理的响应（在 `deliver: log` 的网关日志中可见）将描述其结构。

---

## 安全说明

- **永远不要在生产中使用 `INSECURE_NO_AUTH`**——它完全禁用签名验证。它仅用于本地开发。
- **定期轮换你的 webhook 密钥**并在 GitHub（webhook 设置）和你的 `config.yaml` 中更新它。
- **速率限制**默认为每个路由 30 req/min（可通过 `extra.rate_limit` 配置）。超过返回 `429`。
- **重复交付**（webhook 重试）通过 1 小时幂等缓存去重。缓存键是 `X-GitHub-Delivery`（如果存在），然后是 `X-Request-ID`，然后是毫秒时间戳。当两个交付 ID 头都未设置时，重复不会被去重。
- **提示注入：** PR 标题、描述和提交消息是攻击者控制的。恶意 PR 可能试图操纵代理的行为。当暴露在公共互联网上时，在沙箱环境（Docker、VM）中运行网关。

---

## 故障排除

| 症状 | 检查 |
|---|---|
| `401 Invalid signature` | config.yaml 中的密钥与 GitHub webhook 密钥不匹配 |
| `404 Unknown route` | URL 中的路由名称与 `routes:` 中的键不匹配 |
| `429 Rate limit exceeded` | 每个路由 30 req/min 被超出——从 GitHub UI 重新交付测试事件时常见；等待一分钟或提高 `extra.rate_limit` |
| 没有评论发布 | `gh` 未安装、不在 PATH 中或未认证（`gh auth login`） |
| 代理运行但没有评论 | 检查网关日志——如果代理输出为空或只是"SKIP"，仍然尝试交付 |
| 端口已被占用 | 在 config.yaml 中更改 `extra.port` |
| 代理运行但仅审查 PR 描述 | 提示未包含 `gh pr diff` 指令——差异不在 webhook 有效载荷中 |
| 看不到 ping 事件 | 被忽略的事件在 DEBUG 日志级别返回 `{"status":"ignored","event":"ping"}`——检查 GitHub 的交付日志（仓库 → Settings → Webhooks → 你的 webhook → Recent Deliveries） |

**GitHub 的 Recent Deliveries 标签**（仓库 → Settings → Webhooks → 你的 webhook）显示每次交付的确切请求头、有效载荷、HTTP 状态和响应正文。这是诊断故障的最快方法，无需触碰你的服务器日志。

---

## 完整配置参考

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      host: "0.0.0.0"         # 绑定地址（默认：0.0.0.0）
      port: 8644               # 监听端口（默认：8644）
      secret: ""               # 可选的全局回退密钥
      rate_limit: 30           # 每个路由每分钟请求数
      max_body_bytes: 1048576  # 有效载荷大小限制（字节）（默认：1 MB）

      routes:
        <route-name>:
          secret: "required-per-route"
          events: []            # [] = 接受所有；否则列出 X-GitHub-Event 值
          prompt: ""            # {field} / {nested.field} 从有效载荷解析
          skills: []            # 第一个匹配的技能被加载（仅一个）
          deliver: "log"        # log | github_comment | telegram | discord | slack | signal | sms
          deliver_extra: {}     # github_comment 的 repo + pr_number；其他的是 chat_id
```

---

## 接下来做什么？

- **[基于 Cron 的 PR 审查](./github-pr-review-agent.md)** — 按计划轮询 PR，无需公共端点
- **[Webhook 参考](/docs/user-guide/messaging/webhooks)** — webhook 平台的完整配置参考
- **[构建插件](/docs/guides/build-a-hermes-plugin)** — 将审查逻辑打包成可共享的插件
- **[配置文件](/docs/user-guide/profiles)** — 运行带有自己记忆和配置的专用审查器配置文件
