---
sidebar_position: 10
title: "教程：GitHub PR 审查代理"
description: "构建一个自动化 AI 代码审查器，持续监控仓库、审查 pull request 并发送反馈"
---

# 教程：构建 GitHub PR 审查代理

**问题：** 团队开 PR 的速度比你能 review 的速度更快。PR 一拖就是几天。新人会在没人检查的情况下合并 bug。你早上都在补 diffs，而不是做新东西。

**解决方案：** 做一个 AI agent，24 小时监控仓库，自动审查新的 PR，检查 bug、安全问题和代码质量，并把摘要发给你——这样你只需要亲自处理真正需要人工判断的 PR。

**你会构建什么：**

```text
Cron Timer  ->  Hermes Agent  ->  GitHub API  ->  Review delivery
(每 2 小时)      + gh CLI          (PR diffs)       (Telegram,
                 + skill                            Discord,
                 + memory                           local)
```

本指南使用 **cron job** 按计划轮询 PR，不需要服务器公网上线，也不需要公共 endpoint。可以在 NAT 或防火墙后运行。

:::tip 想要实时审查？
如果你有公网 endpoint，可以看 [使用 Webhook 自动发送 GitHub PR 评论](/guides/webhook-github-pr-review) —— GitHub 在 PR 打开或更新时直接把事件推给 Hermes。
:::

## 前置条件

- 已安装 Hermes Agent
- gateway 已运行
- 已安装并认证 GitHub CLI `gh`
- 已配置好消息平台（可选）

## 第 1 步：验证配置

确认 Hermes 能访问 GitHub。启动会话：

```bash
hermes
```

测试命令：

```text
Run: gh pr list --repo NousResearch/hermes-agent --state open --limit 3
```

如果能看到 open PR 列表，说明准备好了。

## 第 2 步：先手动 review 一次

让 Hermes 先审查一个真实 PR：

```text
Review this pull request. Read the diff, check for bugs, security issues,
and code quality. Be specific about line numbers and quote problematic code.

Run: gh pr diff 3888 --repo NousResearch/hermes-agent
```

Hermes 会：
1. 执行 `gh pr diff`
2. 读完整个 diff
3. 生成带具体发现的结构化 review

## 第 3 步：创建 review skill

一个 skill 可以让 Hermes 在多个会话和 cron 运行里保持一致的审查标准。

```bash
mkdir -p ~/.hermes/skills/code-review
```

在 `~/.hermes/skills/code-review/SKILL.md` 里写入审查准则。

## 第 4 步：教它你的团队规范

给 Hermes 灌输你们团队的约定，例如后端用 FastAPI、前端用 TypeScript + React、不要直接写 raw SQL 等。

这些记忆会持续保留。

## 第 5 步：创建自动 cron job

现在把它们串起来，每 2 小时运行一次：

```bash
hermes cron create "0 */2 * * *" \
  "Check for new open PRs and review them." \
  --name "pr-review" \
  --deliver telegram \
  --skill code-review
```

验证任务：

```bash
hermes cron list
```

## 第 6 步：按需运行

```bash
hermes cron run pr-review
```

或者在聊天里：

```text
/cron run pr-review
```

## 继续扩展

### 直接把 review 发到 GitHub

可以把交付目标改成 GitHub 评论或 review。

### 周度 PR 看板

可以创建每周一上午的 dashboard，汇总所有仓库的 open PR 状态。
