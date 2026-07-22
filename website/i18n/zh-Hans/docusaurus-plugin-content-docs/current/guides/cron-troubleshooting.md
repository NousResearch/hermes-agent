---
sidebar_position: 12
title: "Cron 故障排查"
description: "诊断并修复常见的 Hermes cron 问题——任务未触发、投递失败、skill 加载错误及性能问题"
---

# Cron 故障排查

当 cron 任务行为异常时，请按顺序逐项检查。大多数问题属于以下四类之一：时序、投递、权限或 skill 加载。

---

## 任务未触发

### 检查 1：确认任务存在且处于活跃状态

```bash
hermes cron list
```

找到该任务并确认其状态为 `[active]`（而非 `[paused]` 或 `[completed]`）。若显示 `[completed]`，可能是重复次数已耗尽——编辑该任务以重置。

### 检查 2：确认调度表达式正确

格式错误的调度表达式会静默降级为单次执行，或被直接拒绝。测试你的表达式：

| 你的表达式 | 应解析为 |
|----------------|-------------------|
| `0 9 * * *` | 每天上午 9:00 |
| `0 9 * * 1` | 每周一上午 9:00 |
| `every 2h` | 从现在起每 2 小时 |
| `30m` | 从现在起 30 分钟后 |
| `2025-06-01T09:00:00` | 2025 年 6 月 1 日 09:00 UTC |

若任务触发一次后从列表中消失，说明这是单次调度（`30m`、`1d` 或 ISO 时间戳）——属于预期行为。

### 检查 3：gateway 是否正在运行？

Cron 任务由 gateway 的后台 ticker 线程触发，该线程每 60 秒 tick 一次。普通的 CLI 聊天会话**不会**自动触发 cron 任务。

如果你期望任务自动触发，需要运行一个 gateway（前台运行用 `hermes gateway`，安装为服务用 `hermes gateway start`）。如需单次调试，可手动触发一次 tick：`hermes cron tick`。

### 检查 4：检查系统时钟和时区

任务使用本地时区。若机器时钟有误或时区与预期不符，任务将在错误的时间触发。验证方法：

```bash
date
hermes cron list   # 将 next_run 时间与本地时间对比
```

---

## 投递失败

### 检查 1：确认投递目标正确

投递目标区分大小写，且要求对应平台已正确配置。目标配置错误会静默丢弃响应。

| 目标 | 所需配置 |
|--------|----------|
| `telegram` | `~/.hermes/.env` 中的 `TELEGRAM_BOT_TOKEN` |
| `discord` | `~/.hermes/.env` 中的 `DISCORD_BOT_TOKEN` |
| `slack` | `~/.hermes/.env` 中的 `SLACK_BOT_TOKEN` |
| `whatsapp` | 已配置 WhatsApp gateway |
| `signal` | 已配置 Signal gateway |
| `matrix` | 已配置 Matrix homeserver |
| `email` | `config.yaml` 中已配置 SMTP |
| `sms` | 已配置 SMS 提供商 |
| `local` | 对 `~/.hermes/cron/output/` 有写权限 |
| `origin` | 投递到创建该任务的聊天会话 |

其他支持的平台包括 `mattermost`、`homeassistant`、`dingtalk`、`feishu`、`wecom`、`weixin`、`bluebubbles`、`qqbot` 和 `webhook`。你也可以使用 `platform:chat_id` 语法指定特定聊天（例如 `telegram:-1001234567890`）。

若投递失败，任务仍会执行——只是不会发送到任何地方。检查 `hermes cron list` 中的 `last_error` 字段（如有）。

### 检查 2：检查 `[SILENT]` 的使用

若你的 cron 任务没有输出，或 agent 响应为 `[SILENT]`，投递会被抑制。这对监控类任务是预期行为——但请确认你的 prompt（提示词）没有意外地抑制所有输出。

若 prompt 中写有"如果没有变化则回复 [SILENT]"，非空响应也可能被静默吞掉。请检查你的条件逻辑。

### 检查 3：平台 token 权限

每个消息平台的 bot 需要特定权限才能发送消息。若投递静默失败：

- **Telegram**：Bot 必须是目标群组/频道的管理员
- **Discord**：Bot 必须有目标频道的发送权限
- **Slack**：Bot 必须已加入工作区并拥有 `chat:write` scope

### 检查 4：响应包装

默认情况下，cron 响应会添加页眉和页脚（`config.yaml` 中的 `cron.wrap_response: true`）。某些平台或集成可能无法正常处理。如需禁用：

```yaml
cron:
  wrap_response: false
```

---

## Skill 加载失败

### 检查 1：确认 skill 已安装

```bash
hermes skills list
```

Skill 必须先安装才能附加到 cron 任务。若 skill 缺失，先用 `hermes skills install <skill-name>` 安装，或在 CLI 中通过 `/skills` 安装。

### 检查 2：检查 skill 名称与 skill 文件夹名称

Skill 名称区分大小写，必须与已安装 skill 的文件夹名称完全匹配。若任务指定的是 `ai-funding-report`，但 skill 文件夹是 `ai-funding-daily-report`，请从 `hermes skills list` 确认确切名称。

### 检查 3：依赖交互式工具的 skill

Cron 任务运行时，`cronjob`、`messaging` 和 `clarify` 工具集均被禁用。这可防止递归创建 cron、直接发送消息（投递由调度器处理）以及交互式提示。若某 skill 依赖这些工具集，它将无法在 cron 上下文中运行。

请查阅该 skill 的文档，确认其支持非交互式（headless）模式。

### 检查 4：多 skill 加载顺序

使用多个 skill 时，它们按顺序加载。若 Skill A 依赖 Skill B 的上下文，请确保 B 先加载：

```bash
/cron add "0 9 * * *" "..." --skill context-skill --skill target-skill
```

在此示例中，`context-skill` 先于 `target-skill` 加载。

---

## 任务错误与失败

### 检查 1：查看近期任务输出

若任务运行后失败，可在以下位置查看错误上下文：

1. 任务投递的聊天会话（若投递成功）
2. `~/.hermes/logs/agent.log`（调度器消息）或 `errors.log`（警告信息）
3. 通过 `hermes cron list` 查看任务的 `last_run` 元数据

### 检查 2：常见错误模式

**脚本报 "No such file or directory"**
`script` 路径必须为绝对路径（或相对于 Hermes 配置目录的路径）。验证：
```bash
ls ~/.hermes/scripts/your-script.py   # 必须存在
hermes cron edit <job_id> --script ~/.hermes/scripts/your-script.py
```

**任务执行时报 "Skill not found"**
Skill 必须安装在运行调度器的机器上。若你在不同机器间切换，skill 不会自动同步——请用 `hermes skills install <skill-name>` 重新安装。

**任务运行但没有投递任何内容**
可能是投递目标问题（见上方"投递失败"部分）或响应被静默抑制（`[SILENT]`）。

**任务挂起或超时**
调度器使用基于不活跃时间的超时机制（默认 600 秒，可通过 `HERMES_CRON_TIMEOUT` 环境变量配置，`0` 表示无限制）。只要 agent 持续调用工具，就可以一直运行——计时器仅在持续不活跃后触发。长时间运行的任务应使用脚本处理数据采集，仅将结果投递出去。

### 检查 3：锁竞争

调度器使用基于文件的锁来防止 tick 重叠。若同时运行了两个 gateway 实例（或 CLI 会话与 gateway 冲突），任务可能被延迟或跳过。

终止重复的 gateway 进程：
```bash
ps aux | grep hermes
# 终止重复进程，只保留一个
```

### 检查 4：jobs.json 的权限

任务存储在 `~/.hermes/cron/jobs.json`。若该文件对当前用户不可读写，调度器将静默失败：

```bash
ls -la ~/.hermes/cron/jobs.json
chmod 600 ~/.hermes/cron/jobs.json   # 应由你的用户拥有
```

---

## 性能问题

### 任务启动缓慢

每个 cron 任务都会创建一个全新的 AIAgent 会话，可能涉及提供商认证和模型加载。对于时间敏感的调度，请预留缓冲时间（例如用 `0 8 * * *` 代替 `0 9 * * *`）。

### 过多任务重叠

调度器在每次 tick 内顺序执行任务。若多个任务同时到期，它们将依次运行。考虑错开调度时间（例如用 `0 9 * * *` 和 `5 9 * * *` 代替两者都设为 `0 9 * * *`）以避免延迟。

### 脚本输出过大

输出数兆字节数据的脚本会拖慢 agent，并可能触及 token 限制。请在脚本层面进行过滤/摘要——只输出 agent 需要推理的内容。

---

## 诊断命令

```bash
hermes cron list                    # 显示所有任务、状态、next_run 时间
hermes cron run <job_id>            # 安排在下次 tick 执行（用于测试）
hermes cron edit <job_id>           # 修复配置问题
hermes logs                         # 查看近期 Hermes 日志
hermes skills list                  # 确认已安装的 skill
```

---

## 获取更多帮助

若你已按本指南逐项排查，问题仍未解决：

1. 使用 `hermes cron run <job_id>` 运行任务（在下次 gateway tick 时触发），观察聊天输出中的错误
2. 查看 `~/.hermes/logs/agent.log` 中的调度器消息和 `~/.hermes/logs/errors.log` 中的警告
3. 在 [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) 提交 issue，并附上：
   - 任务 ID 和调度表达式
   - 投递目标
   - 预期行为与实际行为
   - 日志中的相关错误信息

---

*完整的 cron 参考文档，请参阅 [用 Cron 自动化一切](/guides/automate-with-cron) 和 [定时任务（Cron）](/user-guide/features/cron)。*
