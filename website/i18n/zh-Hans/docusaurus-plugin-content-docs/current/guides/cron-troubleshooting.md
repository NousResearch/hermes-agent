---
sidebar_position: 12
title: "Cron 故障排查"
description: "快速定位 Cron 任务不触发、投递失败、技能加载异常和性能问题。"
---

# Cron 故障排查

遇到 Cron 异常时，按下面顺序排查: 任务状态 -> 调度器 -> 投递 -> 脚本/技能。

## 1. 任务是否存在且激活

```bash
hermes cron list
```

确认状态为 `active`。若是 `paused`，先恢复。若是一性任务（one-shot），执行后消失属于正常。

## 2. 调度器是否在运行

Cron 由 gateway 后台 ticker 驱动，单独打开 CLI 聊天并不会自动触发。

```bash
hermes cron status
hermes gateway start
```

## 3. 计划表达式是否正确

常见写法:

- `0 9 * * *`: 每天 9:00
- `0 9 * * 1`: 每周一 9:00
- `every 2h`: 每 2 小时
- `30m`: 30 分钟后执行一次

## 4. 投递目标是否可用

检查 `deliver` 目标和平台凭证是否配置。

- Telegram: `TELEGRAM_BOT_TOKEN`
- Discord: `DISCORD_BOT_TOKEN`
- Slack: `SLACK_BOT_TOKEN`
- local: 写入 `~/.hermes/cron/output/`

## 5. 脚本与技能问题

- 脚本应放在 `~/.hermes/scripts/`
- 脚本要有执行权限
- 技能需先安装，名称要与 `hermes skills list` 一致

## 6. 查看日志

```bash
hermes logs
```

重点看 `agent.log` 与 `errors.log`。

## 相关文档

- [定时任务（Cron）](/user-guide/features/cron)
- [仅脚本 Cron 任务](/guides/cron-script-only)
- [自动化 Cron 指南](/guides/automate-with-cron)
