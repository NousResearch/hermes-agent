---
sidebar_position: 10
title: "教程：GitHub PR 审查 Agent"
description: "构建一个自动化 AI 代码审查器，监控你的仓库、审查 Pull Request 并自动发送反馈——全程无需人工干预"
---

# 教程：构建 GitHub PR 审查 Agent

**问题所在：** 团队提交 PR 的速度比你审查的速度还快。PR 等待数天无人问津。初级开发者因为没人检查而合并了有 bug 的代码。你每天早上都在追赶 diff，而不是在写新功能。

**解决方案：** 一个全天候监控你的仓库的 AI agent，对每个新 PR 进行 bug、安全问题和代码质量审查，并向你发送摘要——这样你只需把时间花在真正需要人工判断的 PR 上。

**你将构建的内容：**

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   Cron Timer  ──▶  Hermes Agent  ──▶  GitHub API  ──▶  Review     │
│   (every 2h)       + gh CLI           (PR diffs)       delivery   │
│                    + skill                             (Telegram, │
│                    + memory                            Discord,   │
│                                                        local)     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

本指南使用 **cron 任务**按计划轮询 PR——无需服务器或公开端点，在 NAT 和防火墙后面同样可用。

:::tip 想要实时审查？
如果你有可用的公开端点，请查看[使用 Webhook 自动化 GitHub PR 评论](./webhook-github-pr-review.md)——GitHub 会在 PR 被打开或更新时立即向 Hermes 推送事件。
:::

---

## 前提条件

- **已安装 Hermes Agent** — 参见[安装指南](/getting-started/installation)
- **Gateway 已运行**（用于 cron 任务）：
  ```bash
  hermes gateway install   # Install as a service
  # or
  hermes gateway           # Run in foreground
  ```
- **已安装并认证 GitHub CLI（`gh`）**：
  ```bash
  # Install
  brew install gh        # macOS
  sudo apt install gh    # Ubuntu/Debian

  # Authenticate
  gh auth login
  ```
- **已配置消息通知**（可选）— [Telegram](/user-guide/messaging/telegram) 或 [Discord](/user-guide/messaging/discord)

:::tip 没有消息通知？没关系
使用 `deliver: "local"` 将审查结果保存到 `~/.hermes/cron/output/`。在接入通知之前用于测试非常方便。
:::

---

## 第一步：验证配置

确保 Hermes 可以访问 GitHub。启动对话：

```bash
hermes
```

用一个简单命令测试：

```
Run: gh pr list --repo NousResearch/hermes-agent --state open --limit 3
```

你应该能看到一个开放 PR 的列表。如果成功，就可以继续了。

---

## 第二步：手动试审一个 PR

仍在对话中，让 Hermes 审查一个真实的 PR：

```
Review this pull request. Read the diff, check for bugs, security issues,
and code quality. Be specific about line numbers and quote problematic code.

Run: gh pr diff 3888 --repo NousResearch/hermes-agent
```

Hermes 将会：
1. 执行 `gh pr diff` 获取代码变更
2. 通读整个 diff
3. 生成包含具体发现的结构化审查报告

如果你对审查质量满意，就可以开始自动化了。

---

## 第三步：创建审查 Skill

Skill 为 Hermes 提供一致的审查准则，在会话和 cron 运行之间持久保存。没有 skill，审查质量会参差不齐。

```bash
mkdir -p ~/.hermes/skills/code-review
```

创建 `~/.hermes/skills/code-review/SKILL.md`：

```markdown
---
name: code-review
description: Review pull requests for bugs, security issues, and code quality
---

# Code Review Guidelines

When reviewing a pull request:

## What to Check
1. **Bugs** — Logic errors, off-by-one, null/undefined handling
2. **Security** — Injection, auth bypass, secrets in code, SSRF
3. **Performance** — N+1 queries, unbounded loops, memory leaks
4. **Style** — Naming conventions, dead code, missing error handling
5. **Tests** — Are changes tested? Do tests cover edge cases?

## Output Format
For each finding:
- **File:Line** — exact location
- **Severity** — Critical / Warning / Suggestion
- **What's wrong** — one sentence
- **Fix** — how to fix it

## Rules
- Be specific. Quote the problematic code.
- Don't flag style nitpicks unless they affect readability.
- If the PR looks good, say so. Don't invent problems.
- End with: APPROVE / REQUEST_CHANGES / COMMENT
```

验证是否已加载——启动 `hermes`，你应该能在启动时的 skill 列表中看到 `code-review`。

---

## 第四步：教会它你的团队规范

这才是让审查器真正有用的关键。启动一个会话，向 Hermes 传授你的团队标准：

```
Remember: In our backend repo, we use Python with FastAPI.
All endpoints must have type annotations and Pydantic models.
We don't allow raw SQL — only SQLAlchemy ORM.
Test files go in tests/ and must use pytest fixtures.
```

```
Remember: In our frontend repo, we use TypeScript with React.
No `any` types allowed. All components must have props interfaces.
We use React Query for data fetching, never useEffect for API calls.
```

这些记忆会永久保存——审查器无需每次提醒就会自动执行你的规范。

---

## 第五步：创建自动化 Cron 任务

现在把所有内容串联起来。创建一个每 2 小时运行一次的 cron 任务：

```bash
hermes cron create "0 */2 * * *" \
  "Check for new open PRs and review them.

Repos to monitor:
- myorg/backend-api
- myorg/frontend-app

Steps:
1. Run: gh pr list --repo REPO --state open --limit 5 --json number,title,author,createdAt
2. For each PR created or updated in the last 4 hours:
   - Run: gh pr diff NUMBER --repo REPO
   - Review the diff using the code-review guidelines
3. Format output as:

## PR Reviews — today

### [repo] #[number]: [title]
**Author:** [name] | **Verdict:** APPROVE/REQUEST_CHANGES/COMMENT
[findings]

If no new PRs found, say: No new PRs to review." \
  --name "pr-review" \
  --deliver telegram \
  --skill code-review
```

验证任务已调度：

```bash
hermes cron list
```

### 其他常用调度计划

| 计划 | 触发时机 |
|------|----------|
| `0 */2 * * *` | 每 2 小时 |
| `0 9,13,17 * * 1-5` | 工作日每天三次 |
| `0 9 * * 1` | 每周一早上汇总 |
| `30m` | 每 30 分钟（高流量仓库） |

---

## 第六步：按需手动触发

不想等待调度？手动触发：

```bash
hermes cron run pr-review
```

或在对话会话中：

```
/cron run pr-review
```

---

## 进阶用法

### 直接在 GitHub 上发布审查评论

不将结果发送到 Telegram，而是让 agent 直接在 PR 上评论：

在你的 cron prompt（提示词）中添加：

```
After reviewing, post your review:
- For issues: gh pr review NUMBER --repo REPO --comment --body "YOUR_REVIEW"
- For critical issues: gh pr review NUMBER --repo REPO --request-changes --body "YOUR_REVIEW"
- For clean PRs: gh pr review NUMBER --repo REPO --approve --body "Looks good"
```

:::caution
确保 `gh` 使用的 token 具有 `repo` 权限范围。审查评论将以 `gh` 当前认证的用户身份发布。
:::

### 每周 PR 看板

创建一个每周一早上的仓库概览：

```bash
hermes cron create "0 9 * * 1" \
  "Generate a weekly PR dashboard:
- myorg/backend-api
- myorg/frontend-app
- myorg/infra

For each repo show:
1. Open PR count and oldest PR age
2. PRs merged this week
3. Stale PRs (older than 5 days)
4. PRs with no reviewer assigned

Format as a clean summary." \
  --name "weekly-dashboard" \
  --deliver telegram
```

### 多仓库监控

在 prompt 中添加更多仓库即可扩展规模。Agent 会按顺序处理它们——无需额外配置。

---

## 故障排查

### "gh: command not found"
Gateway 在精简环境中运行。请确保 `gh` 在系统 PATH 中，然后重启 gateway。

### 审查结果过于泛泛
1. 添加 `code-review` skill（第三步）
2. 通过 memory（记忆）向 Hermes 传授你的团队规范（第四步）
3. 它对你的技术栈了解越多，审查质量越好

### Cron 任务未运行
```bash
hermes gateway status    # Is the gateway running?
hermes cron list         # Is the job enabled?
```

### 速率限制
GitHub 对已认证用户每小时允许 5,000 次 API 请求。每次 PR 审查约消耗 3-5 次请求（列表 + diff + 可选评论）。即使每天审查 100 个 PR，也远低于限制。

---

## 下一步

- **[基于 Webhook 的 PR 审查](./webhook-github-pr-review.md)** — 在 PR 被打开时立即获得审查（需要公开端点）
- **[每日简报 Bot](/guides/daily-briefing-bot)** — 将 PR 审查与你的晨间资讯摘要结合
- **[构建 Plugin](/guides/build-a-hermes-plugin)** — 将审查逻辑封装为可共享的 plugin
- **[Profiles](/user-guide/profiles)** — 运行一个专属审查器 profile，拥有独立的 memory 和配置
- **[Fallback Providers](/user-guide/features/fallback-providers)** — 确保在某个 provider 不可用时审查任务仍能正常运行