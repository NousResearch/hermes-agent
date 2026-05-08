---
sidebar_position: 10
title: "教程：GitHub PR 审查代理"
description: "构建一个自动化 AI 代码审查器，监控你的仓库、审查拉取请求并交付反馈——无需手动操作"
---

# 教程：构建 GitHub PR 审查代理

**问题：** 你的团队打开 PR 的速度比你能审查的速度快。PR 等待数天才能获得关注。初级开发人员合并了 bug，因为没人有时间检查。你早上花时间追赶差异而不是构建。

**解决方案：** 一个 AI 代理，全天候监控你的仓库，审查每个新 PR 的 bug、安全问题和代码质量，并发送摘要给你——这样你只需要花时间在真正需要人工判断的 PR 上。

**你将构建：**

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   Cron 定时器  ──▶  Hermes 代理  ──▶  GitHub API  ──▶  审查     │
│   (每 2 小时)       + gh CLI           (PR 差异)       交付   │
│                    + skill                             (Telegram, │
│                    + memory                            Discord,   │
│                                                       本地)     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

本指南使用** cron 作业**按计划轮询 PR——不需要服务器或公共端点。可以在 NAT 和防火墙后面工作。

:::tip 想要实时审查？
如果你有可用的公共端点，请查看[使用 Webhooks 自动化 GitHub PR 评论](./webhook-github-pr-review.md)——当 PR 被打开或更新时，GitHub 立即将事件推送到 Hermes。
:::

---

## 先决条件

- **已安装 Hermes 代理**——请参阅[安装指南](/docs/getting-started/installation)
- **网关运行中**以使用 cron 作业：
  ```bash
  hermes gateway install   # 安装为服务
  # 或
  hermes gateway           # 在前台运行
  ```
- **已安装并认证 GitHub CLI (`gh`)**：
  ```bash
  # 安装
  brew install gh        # macOS
  sudo apt install gh    # Ubuntu/Debian

  # 认证
  gh auth login
  ```
- **消息传递已配置**（可选）——[Telegram](/docs/user-guide/messaging/telegram) 或 [Discord](/docs/user-guide/messaging/discord)

:::tip 没有消息传递？没问题
使用 `deliver: "local"` 将审查保存到 `~/.hermes/cron/output/`。在连接通知之前测试非常有用。
:::

---

## 步骤 1：验证设置

确保 Hermes 可以访问 GitHub。开始聊天：

```bash
hermes
```

使用简单命令测试：

```
运行：gh pr list --repo NousResearch/hermes-agent --state open --limit 3
```

你应该看到打开的 PR 列表。如果这能工作，你就准备好了。

---

## 步骤 2：尝试手动审查

仍在聊天中，要求 Hermes 审查一个真实的 PR：

```
审查这个拉取请求。读取差异，检查 bug、安全问题
和代码质量。对行号要具体，并引用有问题的代码。

运行：gh pr diff 3888 --repo NousResearch/hermes-agent
```

Hermes 将：
1. 执行 `gh pr diff` 以获取代码更改
2. 通读整个差异
3. 生成带有具体发现的结构化审查

如果你对质量满意，就可以自动化它了。

---

## 步骤 3：创建审查技能

技能为 Hermes 提供跨会话和 cron 运行的一致审查指南。没有它，审查质量会有所不同。

```bash
mkdir -p ~/.hermes/skills/code-review
```

创建 `~/.hermes/skills/code-review/SKILL.md`：

```markdown
---
name: code-review
description: 审查拉取请求的 bug、安全问题和代码质量
---

# 代码审查指南

审查拉取请求时：

## 检查内容
1. **Bug**——逻辑错误、差一错误、空/未定义处理
2. **安全性**——注入、身份验证绕过、代码中的秘密、SSRF
3. **性能**——N+1 查询、无限循环、内存泄漏
4. **样式**——命名约定、死代码、缺少错误处理
5. **测试**——更改是否经过测试？测试是否覆盖边缘情况？

## 输出格式
对于每个发现：
- **文件:行**——精确位置
- **严重性**——严重 / 警告 / 建议
- **问题**——一句话说明
- **修复**——如何修复

## 规则
- 要具体。引用有问题的代码。
- 除非影响可读性，否则不要标记样式挑剔。
- 如果 PR 看起来不错，请说出来。不要捏造问题。
- 以以下结尾：APPROVE / REQUEST_CHANGES / COMMENT
```

验证它已加载——启动 `hermes`，你应该在启动时的技能列表中看到 `code-review`。

---

## 步骤 4：教导它你的约定

这就是让审查器真正有用的方法。开始一个会话并教导 Hermes 你团队的标准：

```
记住：在我们的后端仓库中，我们使用 Python 和 FastAPI。
所有端点必须有类型注释和 Pydantic 模型。
我们不允许原始 SQL——只能使用 SQLAlchemy ORM。
测试文件放在 tests/ 中，必须使用 pytest fixtures。
```

```
记住：在我们的前端仓库中，我们使用 TypeScript 和 React。
不允许 `any` 类型。所有组件必须有 props 接口。
我们使用 React Query 进行数据获取，从不对 API 调用使用 useEffect。
```

这些记忆永久保留——审查器将强制执行你的约定，而无需每次都告知。

---

## 步骤 5：创建自动化 Cron 作业

现在将它们连接在一起。创建一个每 2 小时运行的 cron 作业：

```bash
hermes cron create "0 */2 * * *" \
  "检查新的打开 PR 并审查它们。

要监控的仓库：
- myorg/backend-api
- myorg/frontend-app

步骤：
1. 运行：gh pr list --repo 仓库 --state open --limit 5 --json number,title,author,createdAt
2. 对于在过去 4 小时内创建或更新的每个 PR：
   - 运行：gh pr diff 编号 --repo 仓库
   - 使用代码审查指南审查差异
3. 格式化为：

## PR 审查——今天

### [仓库] #[编号]：[标题]
**作者：** [姓名] | **结论：** APPROVE/REQUEST_CHANGES/COMMENT
[发现]

如果没有找到新 PR，请说：没有要审查的新 PR。" \
  --name "pr-review" \
  --deliver telegram \
  --skill code-review
```

验证它已安排：

```bash
hermes cron list
```

### 其他有用的计划

| 计划 | 时间 |
|----------|------|
| `0 */2 * * *` | 每 2 小时 |
| `0 9,13,17 * * 1-5` | 每天三次，仅工作日 |
| `0 9 * * 1` | 每周一早晨汇总 |
| `30m` | 每 30 分钟（高流量仓库） |

---

## 步骤 6：按需运行

不想等待计划？手动触发：

```bash
hermes cron run pr-review
```

或从聊天会话中：

```
/cron run pr-review
```

---

## 进一步操作

### 直接将审查发布到 GitHub

不是交付到 Telegram，而是让代理在 PR 本身上评论：

将此添加到你的 cron 提示中：

```
审查后，发布你的审查：
- 对于有问题的 PR：gh pr review 编号 --repo 仓库 --comment --body "你的审查"
- 对于严重问题：gh pr review 编号 --repo 仓库 --request-changes --body "你的审查"
- 对于干净的 PR：gh pr review 编号 --repo 仓库 --approve --body "看起来不错"
```

:::caution
确保 `gh` 具有带有 `repo` 范围的令牌。审查以 `gh` 认证的身份发布。
:::

### 每周 PR 仪表板

创建所有仓库的周一早晨概览：

```bash
hermes cron create "0 9 * * 1" \
  "生成每周 PR 仪表板：
- myorg/backend-api
- myorg/frontend-app
- myorg/infra

对于每个仓库显示：
1. 打开的 PR 数量和最旧 PR 的年龄
2. 本周合并的 PR
3. 陈旧 PR（超过 5 天）
4. 没有分配审查者的 PR

格式化为干净的摘要。" \
  --name "weekly-dashboard" \
  --deliver telegram
```

### 多仓库监控

通过在提示中添加更多仓库来扩展。代理按顺序处理它们——不需要额外的设置。

---

## 故障排除

### "gh：未找到命令"
网关在最小环境中运行。确保 `gh` 在系统 PATH 中并重新启动网关。

### 审查太笼统
1. 添加 `code-review` 技能（步骤 3）
2. 通过记忆教导 Hermes 你的约定（步骤 4）
3. 它对你的技术栈有越多的上下文，审查就越好

### Cron 作业不运行
```bash
hermes gateway status    # 网关正在运行吗？
hermes cron list         # 作业已启用吗？
```

### 速率限制
GitHub 允许已认证用户每小时 5,000 个 API 请求。每个 PR 审查使用约 3-5 个请求（列表 + 差异 + 可选评论）。即使每天审查 100 个 PR 也远低于限制。

---

## 接下来做什么？

- **[基于 Webhook 的 PR 审查](./webhook-github-pr-review.md)**——在 PR 打开时获取即时审查（需要公共端点）
- **[每日简报机器人](/docs/guides/daily-briefing-bot)**——将 PR 审查与你的晨间新闻摘要结合
- **[构建插件](/docs/guides/build-a-hermes-plugin)**——将审查逻辑包装成可共享的插件
- **[配置文件](/docs/user-guide/profiles)**——运行带有自己记忆和配置的专用审查器配置文件
- **[后备提供商](/docs/user-guide/features/fallback-providers)**——即使一个提供商关闭也能确保审查运行
