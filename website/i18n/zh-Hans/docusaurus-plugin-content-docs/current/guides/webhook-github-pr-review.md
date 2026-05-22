---
sidebar_position: 11
sidebar_label: "通过 Webhook 做 GitHub PR 审查"
title: "使用 Webhook 自动发送 GitHub PR 评论"
description: "让 Hermes 连接 GitHub，自动获取 PR diff、审查代码变更并发送评论，由 webhook 触发，无需手动提示"
---

# 使用 Webhook 自动发送 GitHub PR 评论

本指南会教你把 Hermes Agent 接到 GitHub 上，让它在收到 pull request webhook 事件时自动抓取 diff、分析变更，并把评论发回 PR 线程。

当 PR 被打开或更新时，GitHub 会向你的 Hermes 实例发送 webhook POST。Hermes 会根据提示通过 `gh` CLI 获取 diff，然后把审查结果发回 PR。

:::tip 如果你想先快速上手
如果你还没有公网地址，或者只是想快速试试，可以先看 [Build a GitHub PR Review Agent](/guides/github-pr-review-agent) —— 它通过定时任务轮询 PR，适合在 NAT 或防火墙后面运行。
:::

:::info 参考文档
关于 webhook 平台的完整配置项、投递类型、动态订阅和安全模型，请参见 [Webhooks](/user-guide/messaging/webhooks)。
:::

:::warning 提示注入风险
Webhook payload 里包含攻击者可控的数据——PR 标题、commit message 和描述都可能带有恶意指令。只要 webhook 暴露到公网，就应在沙箱环境中运行 gateway（Docker、SSH backend）。安全说明见下文的安全部分。
:::

## 前置条件

- Hermes Agent 已安装并运行（`hermes gateway`）
- 已安装并认证 `gh` CLI
- 你有一个可公网访问的 Hermes URL
- 你有 GitHub 仓库管理员权限，能管理 webhooks

## 第 1 步：启用 webhook 平台

在 `~/.hermes/config.yaml` 中添加：

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644
      rate_limit: 30

      routes:
        github-pr-review:
          secret: "your-webhook-secret-here"
          events:
            - pull_request

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

## 第 2 步：启动 gateway

```bash
hermes gateway
```

你应该看到：

```text
[webhook] Listening on 0.0.0.0:8644 — routes: github-pr-review
```

验证运行状态：

```bash
curl http://localhost:8644/health
```

## 第 3 步：在 GitHub 上注册 webhook

1. 打开仓库 → Settings → Webhooks → Add webhook
2. 填入 payload URL、content type、secret，以及 Pull requests 事件
3. 点击 Add webhook

GitHub 会立刻发一个 `ping` 事件确认连接。它会被安全忽略。

## 第 4 步：打开一个测试 PR

创建分支、推送变更、打开 PR。通常 30–90 秒内，Hermes 就会在 PR 下发出 review comment。

## 本地测试

如果 Hermes 跑在本机，可以用 ngrok 暴露：

```bash
ngrok http 8644
```

测试时建议把 `deliver: github_comment` 改成 `deliver: log`，这样不会真的往仓库发评论。

## 过滤特定 action

GitHub 的 `pull_request` 事件会覆盖很多 action。`events` 只能按事件类型过滤，不能按 action 子类型过滤。

提示中的 `closed` / `labeled` 停止逻辑是通过 prompt 实现的。

## 使用 skill 保持审查风格一致

你可以给这个 route 加 skill，让审查风格更统一：

```yaml
skills:
  - review
```

## 改发到 Slack 或 Discord

把 `deliver` 和 `deliver_extra` 改成目标平台即可。
