---
sidebar_position: 3
title: "教程: 每日简报机器人"
description: "用 Cron + Web 搜索 + 汇总能力，自动在每天早上推送简报到 Telegram 或 Discord。"
---

# 教程: 每日简报机器人

本教程会带你创建一个每天自动运行的简报任务: 搜索最新信息、生成摘要并推送到消息平台。

## 前置条件

- 已安装 Hermes: [安装指南](/getting-started/installation)
- 已启动网关: `hermes gateway start` 或服务模式
- 已配置 Web 搜索后端（如 `FIRECRAWL_API_KEY`）
- 已配置消息平台（可选，或用 `local`）

## 先手动验证提示词

先在 `hermes` 会话中测试提示词，确认输出格式符合预期。

示例提示词:

```text
搜索过去 24 小时 AI agent 和开源 LLM 的最新动态，输出最重要的 3 条新闻。
每条包含: 标题、2 句摘要、来源链接。
```

## 创建定时任务

方式一（自然语言）: 直接在会话中说“每天 8 点执行以上简报并发送到 telegram”。

方式二（命令）:

```text
/cron add "0 8 * * *" "搜索过去 24 小时 AI agent 和开源 LLM 新闻，输出 3 条摘要并附链接，语气简洁专业。"
```

## 管理任务

```bash
hermes cron list
hermes cron run <job_id>
hermes cron remove <job_id>
```

## 进阶

- 工作日执行: `0 8 * * 1-5`
- 多主题并行: 在提示词里要求“按主题并行调研后合并输出”
- 晚间回顾: 再建一条 `0 18 * * *` 任务

## 相关文档

- [定时任务（Cron）](/user-guide/features/cron)
- [委派与并行工作](/guides/delegation-patterns)
- [提示技巧](/guides/tips)
