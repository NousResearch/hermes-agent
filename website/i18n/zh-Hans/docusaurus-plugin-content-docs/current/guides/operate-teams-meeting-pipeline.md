---
title: "运维 Teams 会议流水线"
description: "Microsoft Teams meeting pipeline 的运行手册、巡检清单与故障排查要点。"
---

# 运维 Teams 会议流水线

本页面向已完成基础配置的运维人员，重点覆盖日常巡检、订阅续期和故障排查。

前置配置请先看 [Teams Meetings](/user-guide/messaging/teams-meetings)。

## 核心命令

```bash
hermes teams-pipeline validate
hermes teams-pipeline token-health
hermes teams-pipeline subscriptions
hermes teams-pipeline maintain-subscriptions --dry-run
```

## 生产环境关键点

Microsoft Graph 订阅最长约 72 小时到期。如果不定期续期，事件会在几天后静默中断。

建议至少每 12 小时执行一次:

```bash
hermes teams-pipeline maintain-subscriptions
```

可以通过 Hermes Cron、systemd timer 或系统 crontab 自动化。

## 日常巡检

- 检查 token 健康
- 检查即将过期订阅
- 查看失败任务列表
- 确认 Teams 投递目标未变更

## 故障排查

1. 没有新任务: 检查 webhook URL、client_state、订阅是否过期
2. 任务失败: 检查转录/录音权限、Graph 认证、`ffmpeg`
3. 已产出摘要但未投递: 检查 Teams 平台配置与目标 chat/channel

## 相关文档

- [Teams Meetings](/user-guide/messaging/teams-meetings)
- [Microsoft Teams](/user-guide/messaging/teams)
- [定时任务（Cron）](/user-guide/features/cron)
