---
sidebar_position: 11
title: "Cron 内部原理"
description: "Hermes 如何存储、调度、编辑、暂停、技能加载和投递 cron 作业"
---

# Cron 内部原理

cron 子系统提供调度任务执行 — 从简单的一次性延迟到带技能注入和跨平台投递的循环 cron-expression 作业。

## 关键文件

| 文件 | 用途 |
|------|------|
| `cron/jobs.py` | 作业模型、存储、对 `jobs.json` 的原子读/写 |
| `cron/scheduler.py` | 调度器循环 — 到期作业检测、执行、重复跟踪 |
| `tools/cronjob_tools.py` | 模型面向的 `cronjob` 工具注册和处理器 |
| `gateway/run.py` | 网关集成 — 长时间运行循环中的 cron 滴答 |
| `hermes_cli/cron.py` | CLI `hermes cron` 子命令 |

## 调度模型

支持四种调度格式：

| 格式 | 示例 | 行为 |
|--------|---------|----------|
| **相对延迟** | `30m`、`2h`、`1d` | 一次性的，在指定持续时间后触发 |
| **间隔** | `every 2h`、`every 30m` | 循环的，以规则间隔触发 |
| **Cron 表达式** | `0 9 * * *` | 标准 5 字段 cron 语法（分钟、小时、日、月、周日） |
| **ISO 时间戳** | `2025-01-15T09:00:00` | 一次性的，在确切时间触发 |

模型面向的表面是一个带有 action-style 操作的 `cronjob` 工具：`create`、`list`、`update`、`pause`、`resume`、`run`、`remove`。

## 作业存储

作业存储在 `~/.hermes/cron/jobs.json` 中，具有原子写入语义（写入临时文件，然后重命名）。每个作业记录包含：

```json
{
  "id": "a1b2c3d4e5f6",
  "name": "Daily briefing",
  "prompt": "Summarize today's AI news and funding rounds",
  "schedule": {
    "kind": "cron",
    "expr": "0 9 * * *",
    "display": "0 9 * * *"
  },
  "skills": ["ai-funding-daily-report"],
  "deliver": "telegram:-1001234567890",
  "repeat": {
    "times": null,
    "completed": 42
  },
  "state": "scheduled",
  "enabled": true,
  "next_run_at": "2025-01-16T09:00:00Z",
  "last_run_at": "2025-01-15T09:00:00Z",
  "last_status": "ok",
  "created_at": "2025-01-01T00:00:00Z",
  "model": null,
  "provider": null,
  "script": null
}
```

### 作业生命周期状态

| 状态 | 含义 |
|-------|-------|
| `scheduled` | 活动的，将在下次计划时间触发 |
| `paused` | 暂停的 — 除非恢复否则不会触发 |
| `completed` | 重复次数用尽或已触发的一次性作业 |
| `running` | 当前正在执行（瞬态状态） |

### 向后兼容

较旧的作业可能具有单个 `skill` 字段而不是 `skills` 数组。调度器在加载时规范化它 — 单个 `skill` 被提升为 `skills: [skill]`。

## 调度器运行时

### 滴答周期

调度器按周期滴答运行（默认：每 60 秒）：

```text
tick()
  1. Acquire scheduler lock (prevents overlapping ticks)
  2. Load all jobs from jobs.json
  3. Filter to due jobs (next_run <= now AND state == "scheduled")
  4. For each due job:
     a. Set state to "running"
     b. Create fresh AIAgent session (no conversation history)
     c. Load attached skills in order (injected as user messages)
     d. Run the job prompt through the agent
     e. Deliver the response to the configured target
     f. Update run_count, compute next_run
     g. If repeat count exhausted → state = "completed"
     h. Otherwise → state = "scheduled"
  5. Write updated jobs back to jobs.json
  6. Release scheduler lock
```

### 网关集成

在网关模式下，调度器在专用后台线程中运行（`gateway/run.py` 中的 `_start_cron_ticker`），每 60 秒调用 `scheduler.tick()` 以及消息处理。

在 CLI 模式下，cron 作业仅在运行 `hermes cron` 命令或活动 CLI 会话期间触发。

### 全新会话隔离

每个 cron 作业在完全全新的代理会话中运行：

- 无上次运行的对话历史
- 无上次 cron 执行的历史记录（除非持久化到内存/文件）
- 提示词必须是自包含的 — cron 作业不能提出澄清问题
- `cronjob` 工具集被禁用（递归保护）

## 基于技能的作业

cron 作业可以通过 `skills` 字段附加一个或多个技能。在执行时：

1. 技能按指定顺序加载
2. 每个技能的 SKILL.md 内容被注入为上下文
3. 作业的提示词作为任务指令附加
4. 代理处理组合的技能上下文 + 提示词

这支持可重用、测试过的工作流，而无需将完整指令粘贴到 cron 提示词中。例如：

```
Create a daily funding report → attach "ai-funding-daily-report" skill
```

### 基于脚本的作业

作业也可以通过 `script` 字段附加 Python 脚本。脚本在每个代理轮次**之前**运行，其 stdout 作为上下文注入到提示词中。这支持数据收集和变化检测模式：

```python
# ~/.hermes/scripts/check_competitors.py
import requests, json
# Fetch competitor release notes, diff against last run
# Print summary to stdout — agent analyzes and reports
```

脚本超时默认为 120 秒。`_get_script_timeout()` 通过三层链解析限制：

1. **模块级覆盖** — `_SCRIPT_TIMEOUT`（用于测试/monkeypatch）。仅在不同于默认值时使用。
2. **环境变量** — `HERMES_CRON_SCRIPT_TIMEOUT`
3. **配置** — `config.yaml` 中的 `cron.script_timeout_seconds`（通过 `load_config()` 读取）
4. **默认值** — 120 秒

### Provider 恢复

`run_job()` 将用户配置的回退 provider 和凭据池传递到 `AIAgent` 实例：

- **回退 provider** — 从 `config.yaml` 读取 `fallback_providers`（列表）或 `fallback_model`（旧版字典），匹配网关的 `_load_fallback_model()` 模式。作为 `fallback_model=` 传递给 `AIAgent.__init__`，它将两种格式规范化为回退链。
- **凭据池** — 通过 `agent.credential_pool` 中的 `load_pool(provider)` 使用解析的运行时 provider 名称加载。仅在池有凭据（`pool.has_credentials()`）时传递。在 429/速率限制错误时启用同一 provider 密钥轮换。

这镜像网关的行为 — 没有它，cron 代理会在不尝试恢复的情况下因速率限制而失败。

## 投递模型

cron 作业结果可以投递到任何支持的平台：

| 目标 | 语法 | 示例 |
|--------|--------|---------|
| 原始聊天 | `origin` | 投递到创建作业的聊天 |
| 本地文件 | `local` | 保存到 `~/.hermes/cron/output/` |
| Telegram | `telegram` 或 `telegram:<chat_id>` | `telegram:-1001234567890` |
| Discord | `discord` 或 `discord:#channel` | `discord:#engineering` |
| Slack | `slack` | 投递到 Slack 主频道 |
| WhatsApp | `whatsapp` | 投递到 WhatsApp 主设备 |
| Signal | `signal` | 投递到 Signal |
| Matrix | `matrix` | 投递到 Matrix 主房间 |
| Mattermost | `mattermost` | 投递到 Mattermost 主设备 |
| Email | `email` | 通过邮件投递 |
| SMS | `sms` | 通过短信投递 |
| Home Assistant | `homeassistant` | 投递到 HA 对话 |
| DingTalk | `dingtalk` | 投递到 DingTalk |
| Feishu | `feishu` | 投递到 Feishu |
| WeCom | `wecom` | 投递到 WeCom |
| Weixin | `weixin` | 投递到 Weixin（WeChat） |
| BlueBubbles | `bluebubbles` | 通过 BlueBubbles 投递到 iMessage |
| QQ Bot | `qqbot` | 通过官方 API v2 投递到 QQ（腾讯） |

对于 Telegram 话题，使用格式 `telegram:<chat_id>:<thread_id>`（例如 `telegram:-1001234567890:17585`）。

### 响应包装

默认情况下（`cron.wrap_response: true`），cron 投递被包装：
- 标识 cron 作业名称和任务的标题
- 关于代理在对话中看不到已投递消息的脚注

cron 响应中的 `[SILENT]` 前缀完全压制投递 — 用于仅需要写入文件或执行副作用的作业。

### 会话隔离

Cron 投递**不会**镜像到网关会话对话历史中。它们仅存在于 cron 作业自己的会话中。这防止目标聊天的对话中出现消息交替违规。

## 递归保护

Cron 运行的会话禁用了 `cronjob` 工具集。这防止：
- 调度作业创建新 cron 作业
- 可能爆炸 token 使用量的递归调度
- 从作业内部意外改变作业调度

## 锁定

调度器使用跨进程基于文件的锁定（Unix 上的 `fcntl.flock`，Windows 上的 `msvcrt.locking`）来防止重叠的滴答执行相同的到期作业批次两次 — 即使在网关的进程内滴答和独立 `hermes cron` / 手动 `tick()` 调用之间。如果无法获取锁定，`tick()` 立即返回 0。

## CLI 接口

`hermes cron` CLI 提供直接的作业管理：

```bash
hermes cron list                    # Show all jobs
hermes cron create                  # Interactive job creation (alias: add)
hermes cron edit <job_id>           # Edit job configuration
hermes cron pause <job_id>          # Pause a running job
hermes cron resume <job_id>         # Resume a paused job
hermes cron run <job_id>            # Trigger immediate execution
hermes cron remove <job_id>         # Delete a job
```

## 相关文档

- [Cron 功能指南](/docs/user-guide/features/cron)
- [网关内部原理](./gateway-internals.md)
- [Agent 循环内部原理](./agent-loop.md)
