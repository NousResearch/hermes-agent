---
sidebar_position: 12
title: "Cron Troubleshooting"
description: "诊断和修复常见的 Hermes cron 问题 — 任务未触发、交付失败、技能加载错误和性能问题"
---

# Cron 故障排除

当 cron 任务未按预期运行时，按顺序检查以下内容。大多数问题属于四类之一：计时、交付、权限或技能加载。

---

## 任务未触发

### 检查 1：验证任务存在且处于活动状态

```bash
hermes cron list
```

查找任务并确认其状态为 `[active]`（不是 `[paused]` 或 `[completed]`）。如果显示 `[completed]`，重复次数可能已用尽 — 编辑任务以重置它。

### 检查 2：确认调度正确

格式错误的调度会静默默认为单次或被完全拒绝。测试你的表达式：

| 你的表达式 | 应计算为 |
|----------------|-------------------|
| `0 9 * * *` | 每天上午 9:00 |
| `0 9 * * 1` | 每周一上午 9:00 |
| `every 2h` | 从现在起每 2 小时 |
| `30m` | 从现在起 30 分钟 |
| `2025-06-01T09:00:00` | 2025 年 6 月 1 日上午 9:00 UTC |

如果任务触发一次然后从列表中消失，则它是单次调度（`30m`、`1d` 或 ISO 时间戳）— 预期行为。

### 检查 3：gateway 是否在运行？

Cron 任务由 gateway 的后台 ticker 线程触发，该线程每 60 秒 tick 一次。普通的 CLI 聊天会话**不会**自动触发 cron 任务。

如果你期望任务自动触发，你需要一个运行中的 gateway（`hermes gateway` 或 `hermes serve`）。对于一次性调试，你可以使用 `hermes cron tick` 手动触发一个 tick。

### 检查 4：检查系统时钟和时区

任务使用本地时区。如果你的机器时钟错误或与你预期的时区不同，任务将在错误的时间触发。验证：

```bash
date
hermes cron list   # Compare next_run times with local time
```

---

## 交付失败

### 检查 1：验证交付目标正确

交付目标区分大小写，并且需要配置正确的平台。配置错误的目标会静默丢弃响应。

| 目标 | 需要 |
|--------|----------|
| `telegram` | `~/.hermes/.env` 中的 `TELEGRAM_BOT_TOKEN` |
| `discord` | `~/.hermes/.env` 中的 `DISCORD_BOT_TOKEN` |
| `slack` | `~/.hermes/.env` 中的 `SLACK_BOT_TOKEN` |
| `whatsapp` | WhatsApp gateway 已配置 |
| `signal` | Signal gateway 已配置 |
| `matrix` | Matrix homeserver 已配置 |
| `email` | `config.yaml` 中已配置 SMTP |
| `sms` | SMS 提供商已配置 |
| `local` | 对 `~/.hermes/cron/output/` 的写入权限 |
| `origin` | 交付到创建任务的聊天 |

其他受支持的平台包括 `mattermost`、`homeassistant`、`dingtalk`、`feishu`、`wecom`、`weixin`、`bluebubbles`、`qqbot` 和 `webhook`。你还可以使用 `platform:chat_id` 语法定位特定聊天（例如，`telegram:-1001234567890`）。

如果交付失败，任务仍然运行 — 只是不会发送到任何地方。检查 `hermes cron list` 中的更新 `last_error` 字段（如果可用）。

### 检查 2：检查 `[SILENT]` 使用

如果你的 cron 任务没有产生输出或 agent 响应了 `[SILENT]`，则交付被抑制。这对于监控任务是有意的 — 但请确保你的提示词没有意外地抑制所有内容。

一个说 "respond with [SILENT] if nothing changed" 的提示词也会静默地吞掉非空响应。检查你的条件逻辑。

### 检查 3：平台 token 权限

每个消息平台机器人需要特定权限才能接收消息。如果交付静默失败：

- **Telegram**：机器人必须是目标群组/频道的管理员
- **Discord**：机器人必须在目标频道中有发送权限
- **Slack**：机器人必须已添加到工作区并具有 `chat:write` scope

### 检查 4：响应包装

默认情况下，cron 响应用页眉和页脚包装（`config.yaml` 中的 `cron.wrap_response: true`）。某些平台或集成可能无法很好地处理此问题。要禁用：

```yaml
cron:
  wrap_response: false
```

---

## 技能加载失败

### 检查 1：验证技能已安装

```bash
hermes skills list
```

技能必须先安装才能附加到 cron 任务。如果技能缺失，请先使用 `hermes skills install <skill-name>` 或通过 CLI 中的 `/skills` 安装。

### 检查 2：检查技能名称 vs. 技能文件夹名称

技能名称区分大小写，必须与已安装技能的文件夹名称匹配。如果你的任务指定了 `ai-funding-daily-report` 但技能文件夹是 `ai-funding-daily-report`，请从 `hermes skills list` 确认确切名称。

### 检查 3：需要交互工具的技能

Cron 任务运行时 `cronjob`、`messaging` 和 `clarify` 工具集被禁用。这可以防止递归 cron 创建、直接消息发送（交付由调度器处理）和交互式提示。如果技能依赖这些工具集，它在 cron 上下文中不会工作。

检查技能的文档以确认它在非交互（无头）模式下工作。

### 检查 4：多技能顺序

使用多个技能时，它们按顺序加载。如果技能 A 依赖技能 B 的上下文，请确保 B 先加载：

```bash
/cron add "0 9 * * *" "..." --skill context-skill --skill target-skill
```

在此示例中，`context-skill` 在 `target-skill` 之前加载。

---

## 任务错误和失败

### 检查 1：查看最近的任务输出

如果任务运行并失败，你可能会在以下位置看到错误上下文：

1. 任务交付到的聊天（如果交付成功）
2. `~/.hermes/logs/agent.log` 中的调度器消息（或 `errors.log` 中的警告）
3. 通过 `hermes cron list` 查看任务的 `last_run` 元数据

### 检查 2：常见错误模式

**脚本的 "No such file or directory"**
`script` 路径必须是绝对路径（或相对于 Hermes 配置目录）。验证：
```bash
ls ~/.hermes/scripts/your-script.py   # Must exist
hermes cron edit <job_id> --script ~/.hermes/scripts/your-script.py
```

**任务执行时的 "Skill not found"**
技能必须安装在运行调度器的机器上。如果你在机器之间移动，技能不会自动同步 — 使用 `hermes skills install <skill-name>` 重新安装。

**任务运行但未交付**
可能是交付目标问题（参见上面的交付失败）或静默抑制的响应（`[SILENT]`）。

**任务挂起或超时**
调度器使用基于不活动的超时（默认 600s，可通过 `HERMES_CRON_TIMEOUT` 环境变量配置，`0` 表示无限制）。只要 agent 在积极调用工具，它就可以运行 — 计时器仅在持续不活动后才触发。长时间运行的任务应使用脚本处理数据收集并仅交付结果。

### 检查 3：锁争用

调度器使用基于文件的锁定来防止重叠的 tick。如果两个 gateway 实例正在运行（或 CLI 会话与 gateway 冲突），任务可能会延迟或跳过。

杀死重复的 gateway 进程：
```bash
ps aux | grep hermes
# Kill duplicate processes, keep only one
```

### 检查 4：jobs.json 的权限

任务存储在 `~/.hermes/cron/jobs.json` 中。如果此文件对你的用户不可读/写，调度器将静默失败：

```bash
ls -la ~/.hermes/cron/jobs.json
chmod 600 ~/.hermes/cron/jobs.json   # Your user should own it
```

---

## 性能问题

### 任务启动慢

每个 cron 任务创建一个全新的 AIAgent 会话，这可能涉及提供商身份验证和模型加载。对于时间敏感的调度，请添加缓冲时间（例如，`0 8 * * *` 而不是 `0 9 * * *`）。

### 太多重叠任务

调度器在每个 tick 内按顺序执行任务。如果多个任务同时到期，它们将一个接一个地运行。考虑错开调度（例如，`0 9 * * *` 和 `5 9 * * *` 而不是都在 `0 9 * * *`）以避免延迟。

### 大脚本输出

转储兆字节输出的脚本会减慢 agent 速度并可能达到 token 限制。在脚本级别过滤/总结 — 仅发出 agent 需要推理的内容。

---

## 诊断命令

```bash
hermes cron list                    # Show all jobs, states, next_run times
hermes cron run <job_id>            # Schedule for next tick (for testing)
hermes cron edit <job_id>           # Fix configuration issues
hermes logs                         # View recent Hermes logs
hermes skills list                  # Verify installed skills
```

---

## 获取更多帮助

如果你已经完成本指南但问题仍然存在：

1. 使用 `hermes cron run <job_id>`（在下一个 gateway tick 时触发）运行任务并观察聊天输出中的错误
2. 检查 `~/.hermes/logs/agent.log` 中的调度器消息和 `~/.hermes/logs/errors.log` 中的警告
3. 在 [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) 上提出 issue，包含：
   - 任务 ID 和调度
   - 交付目标
   - 你期望的 vs. 实际发生的
   - 日志中的相关错误消息

---

*有关完整的 cron 参考，请参阅[使用 Cron 实现任何自动化](/docs/guides/automate-with-cron)和[定时任务 (Cron)](/docs/user-guide/features/cron)。*
