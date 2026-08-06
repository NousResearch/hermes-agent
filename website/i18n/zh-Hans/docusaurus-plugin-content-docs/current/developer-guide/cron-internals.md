---
sidebar_position: 11
title: "Cron 内部机制"
description: "Hermes 如何存储、调度、编辑、暂停、加载技能以及投递 cron 任务"
---

# Cron 内部机制

cron 子系统提供定时任务执行能力——从简单的单次延迟到带技能注入和跨平台投递的周期性 cron 表达式任务。

## 关键文件

| 文件 | 用途 |
|------|---------|
| `cron/jobs.py` | 任务模型、存储、对 `jobs.json` 的原子读写 |
| `cron/scheduler.py` | 调度器循环——到期任务检测、执行、重复计数跟踪 |
| `tools/cronjob_tools.py` | 面向模型的 `cronjob` 工具注册与处理器 |
| `gateway/run.py` | Gateway 集成——在长运行循环中触发 cron tick |
| `hermes_cli/cron.py` | CLI `hermes cron` 子命令 |

## 调度模型

支持四种调度格式：

| 格式 | 示例 | 行为 |
|--------|---------|----------|
| **相对延迟** | `30m`、`2h`、`1d` | 单次触发，在指定时长后执行 |
| **间隔** | `every 2h`、`every 30m` | 周期触发，按固定间隔执行 |
| **Cron 表达式** | `0 9 * * *` | 标准 5 字段 cron 语法（分钟、小时、日、月、星期） |
| **ISO 时间戳** | `2025-01-15T09:00:00` | 单次触发，在精确时间点执行 |

面向模型的接口是单个 `cronjob` 工具，支持以下操作：`create`、`list`、`update`、`pause`、`resume`、`run`、`remove`。

## 任务存储

任务存储在 `~/.hermes/cron/jobs.json` 中，采用原子写入语义（先写入临时文件，再重命名）。每条任务记录包含：

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

### 任务生命周期状态

| 状态 | 含义 |
|-------|---------|
| `scheduled` | 活跃，将在下次计划时间触发 |
| `paused` | 已暂停——恢复前不会触发 |
| `completed` | 重复次数已耗尽，或单次任务已执行 |
| `running` | 正在执行（瞬态状态） |

### 向后兼容性

旧版任务可能使用单个 `skill` 字段而非 `skills` 数组。调度器在加载时会对此进行规范化——单个 `skill` 会被提升为 `skills: [skill]`。

## 调度器运行时

### Tick 周期

调度器按周期性 tick 运行（默认：每 60 秒）：

```text
tick()
  1. 获取调度器锁（防止 tick 重叠）
  2. 从 jobs.json 加载所有任务
  3. 筛选到期任务（next_run <= now 且 state == "scheduled"）
  4. 对每个到期任务：
     a. 将状态设为 "running"
     b. 创建全新的 AIAgent 会话（无对话历史）
     c. 按顺序加载附加技能（以用户消息形式注入）
     d. 通过 agent 执行任务 prompt（提示词）
     e. 将响应投递到配置的目标
     f. 更新 run_count，计算下次运行时间
     g. 若重复次数耗尽 → state = "completed"
     h. 否则 → state = "scheduled"
  5. 将更新后的任务写回 jobs.json
  6. 释放调度器锁
```

### Gateway 集成

在 gateway 模式下，调度器运行在专用后台线程中（`gateway/run.py` 中的 `_start_cron_ticker`），每 60 秒调用一次 `scheduler.tick()`，与消息处理并行运行。

在 CLI 模式下，cron 任务仅在运行 `hermes cron` 命令或活跃 CLI 会话期间触发。

### 全新会话隔离

每个 cron 任务在完全全新的 agent 会话中运行：

- 无前次运行的对话历史
- 无前次 cron 执行的记忆（除非已持久化到内存/文件）
- prompt 必须自包含——cron 任务无法提出澄清性问题
- `cronjob` 工具集已禁用（递归防护）

## 技能支持的任务

cron 任务可通过 `skills` 字段附加一个或多个技能。执行时：

1. 按指定顺序加载技能
2. 每个技能的 SKILL.md 内容作为上下文注入
3. 任务的 prompt 作为任务指令追加
4. Agent 处理技能上下文与 prompt 的组合内容

这使得可复用、经过测试的工作流无需将完整指令粘贴到 cron prompt 中。例如：

```
创建每日融资报告 → 附加 "ai-funding-daily-report" 技能
```

### 脚本支持的任务

任务还可通过 `script` 字段附加 Python 脚本。该脚本在每次 agent 轮次*之前*运行，其 stdout 作为上下文注入到 prompt 中。这支持数据采集和变更检测模式：

```python
# ~/.hermes/scripts/check_competitors.py
import requests, json
# 获取竞争对手发布说明，与上次运行结果进行差异比对
# 将摘要打印到 stdout——agent 进行分析并报告
```

脚本超时默认为 120 秒。`_get_script_timeout()` 通过三层链路解析限制：

1. **模块级覆盖** — `_SCRIPT_TIMEOUT`（用于测试/monkeypatching）。仅在与默认值不同时使用。
2. **环境变量** — `HERMES_CRON_SCRIPT_TIMEOUT`
3. **配置** — `config.yaml` 中的 `cron.script_timeout_seconds`（通过 `load_config()` 读取）
4. **默认值** — 120 秒

### Provider 恢复

`run_job()` 将用户配置的备用 provider 和凭证池传入 `AIAgent` 实例：

- **备用 provider** — 从 `config.yaml` 读取 `fallback_providers`（列表）或 `fallback_model`（旧版字典），与 gateway 的 `_load_fallback_model()` 模式一致。以 `fallback_model=` 形式传入 `AIAgent.__init__`，后者将两种格式规范化为备用链。
- **凭证池** — 通过 `agent.credential_pool` 中的 `load_pool(provider)` 使用解析后的运行时 provider 名称加载。仅在池中有凭证时传入（`pool.has_credentials()`）。在遭遇 429/限速错误时启用同 provider 的密钥轮换。

这与 gateway 的行为保持一致——否则 cron agent 在遭遇限速时将直接失败而不尝试恢复。

## 投递模型

Cron 任务结果可投递到任何受支持的平台。

裸平台名（`slack`、`telegram` 等）会投递到该平台配置的**主频道**。若要投递到**特定**目标，请在冒号后追加目标：`platform:<target>`。目标在任务触发时解析（而非创建时），因此任务可以指定一个尚未连接的平台目标，待其上线后即开始投递。

大多数平台还支持以第三段指定可选的话题/线程：`platform:<chat_id>:<thread_id>`。

| 目标 | 语法 | 示例 |
|--------|--------|---------|
| 来源聊天 | `origin` | 投递到创建该任务的聊天 |
| 本地文件 | `local` | 保存到 `~/.hermes/cron/output/` |
| Telegram | `telegram`、`telegram:<chat_id>`、`telegram:<chat_id>:<thread_id>`、`telegram:@username` | `telegram:-1001234567890:17585` |
| Discord | `discord`、`discord:#channel`、`discord:<channel_id>`、`discord:<channel_id>:<thread_id>` | `discord:#engineering` |
| Slack | `slack`、`slack:#channel`、`slack:<channel_id>`、`slack:<channel_id>:<thread_ts>` | `slack:#engineering` |
| Matrix | `matrix`、`matrix:<!room_id:server>`、`matrix:<@user:server>` | `matrix:!abc123:example.org` |
| Feishu | `feishu`、`feishu:<chat_id>`、`feishu:<chat_id>:<thread_id>` | `feishu:oc_abc123def` |
| WhatsApp | `whatsapp`、`whatsapp:<jid>`、`whatsapp:+<E.164>` | `whatsapp:123456@g.us` |
| Signal | `signal`、`signal:group:<id>`、`signal:+<E.164>` | `signal:group:aBcD==` |
| SMS | `sms`、`sms:+<E.164>` | `sms:+<E.164 号码>` |
| Email | `email`、`email:<address>` | `email:alerts@example.com` |
| Weixin | `weixin`、`weixin:<wxid>` | `weixin:wxid_abc123` |
| Mattermost | `mattermost` 或 `mattermost:<channel_id>` | 裸名投递到 Mattermost 主频道 |
| Home Assistant | `homeassistant` 或 `homeassistant:<conversation>` | 裸名投递到 HA 对话 |
| DingTalk | `dingtalk` 或 `dingtalk:<chat_id>` | 裸名投递到钉钉 |
| WeCom | `wecom` 或 `wecom:<chat_id>` | 裸名投递到企业微信 |
| BlueBubbles | `bluebubbles` 或 `bluebubbles:<chat_guid>` | 裸名通过 BlueBubbles 投递到 iMessage |
| QQ Bot | `qqbot` 或 `qqbot:<chat_id>` | 裸名通过官方 API v2 投递到 QQ（腾讯） |

第一组平台具有显式、经校验的目标语法——具名频道（`#channel`）、话题/线程、房间/用户 ID、群组 ID 或电话号码。其余平台接受通用的 `platform:<chat_id>` 形式（冒号后的值原样用作目标 ID）；裸平台名始终投递到主频道。

**具名频道**（`slack:#engineering`、`discord:#engineering`，或像 `slack:engineering` 这样的友好名称）会根据 gateway 从已连接适配器构建的频道目录进行解析，因此 gateway 必须已发现该频道，名称解析才能成功；原始 ID（`slack:C0123ABCD45`）则始终可用。

对于 **Telegram 话题**，使用 `telegram:<chat_id>:<thread_id>`（例如 `telegram:-1001234567890:17585`）。对于 **Slack 线程**，第三段是父消息的 `thread_ts`（例如 `slack:C0123ABCD45:1700000000.000100`），因此仅在回复某条已有消息下方时适用。

### 响应包装

默认情况下（`cron.wrap_response: true`），cron 投递内容会被包装：
- 头部标识 cron 任务名称和任务内容
- 尾部说明 agent 无法在对话中看到已投递的消息

cron 响应中的 `[SILENT]` 前缀会完全抑制投递——适用于只需写入文件或执行副作用的任务。

### 会话隔离

Cron 投递**不会**镜像到 gateway 会话的对话历史中。它们仅存在于 cron 任务自身的会话中。这可防止目标聊天对话中出现消息交替违规。

## 递归防护

Cron 运行的会话已禁用 `cronjob` 工具集。这可防止：
- 定时任务创建新的 cron 任务
- 可能导致 token 用量爆炸的递归调度
- 在任务内部意外修改任务调度

## 锁机制

调度器使用跨进程文件锁（Unix 上的 `fcntl.flock`，Windows 上的 `msvcrt.locking`）防止重叠的 tick 对同一批到期任务执行两次——即使在 gateway 的进程内 ticker 与独立的 `hermes cron` / 手动 `tick()` 调用之间也如此。若无法获取锁，`tick()` 立即返回 0。

## CLI 接口

`hermes cron` CLI 提供直接的任务管理功能：

```bash
hermes cron list                    # 显示所有任务
hermes cron create                  # 交互式创建任务（别名：add）
hermes cron edit <job_id>           # 编辑任务配置
hermes cron pause <job_id>          # 暂停运行中的任务
hermes cron resume <job_id>         # 恢复已暂停的任务
hermes cron run <job_id>            # 触发立即执行
hermes cron remove <job_id>         # 删除任务
```

## 相关文档

- [Cron 功能指南](/user-guide/features/cron)
- [Gateway 内部机制](./gateway-internals.md)
- [Agent 循环内部机制](./agent-loop.md)