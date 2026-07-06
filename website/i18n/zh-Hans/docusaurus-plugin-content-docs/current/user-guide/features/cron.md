---
sidebar_position: 5
title: "定时任务（Cron）"
description: "用自然语言调度自动化任务，通过单一 cron 工具管理，并附加一个或多个 skill"
---

# 定时任务（Cron）

使用自然语言或 cron 表达式调度自动运行的任务。Hermes 通过单一 `cronjob` 工具暴露 cron 管理能力，采用动作式操作，而非分散的 schedule/list/remove 工具。

## Cron 当前能做什么

Cron 任务可以：

- 调度一次性或周期性任务
- 暂停、恢复、编辑、触发和删除任务
- 为任务附加零个、一个或多个 skill
- 将结果回传到来源会话、本地文件或已配置的平台目标
- 在全新的 agent 会话中运行，使用正常的静态工具列表
- 以**无 agent 模式**运行——按计划执行脚本，其 stdout 原样投递，零 LLM 参与（参见下方[无 agent 模式](#no-agent-mode-script-only-jobs)章节）

所有这些功能均可通过 `cronjob` 工具由 Hermes 自身使用，因此你可以用自然语言创建、暂停、编辑和删除任务——无需 CLI。

:::warning
Cron 运行的会话不能递归创建更多 cron 任务。Hermes 在 cron 执行内部禁用了 cron 管理工具，以防止失控的调度循环。
:::

## 创建定时任务

### 在聊天中使用 `/cron`

```bash
/cron add 30m "Remind me to check the build"
/cron add "every 2h" "Check server status"
/cron add "every 1h" "Summarize new feed items" --skill blogwatcher
/cron add "every 1h" "Use both skills and combine the result" --skill blogwatcher --skill maps
```

### 从独立 CLI

```bash
hermes cron create "every 2h" "Check server status"
hermes cron create "every 1h" "Summarize new feed items" --skill blogwatcher
hermes cron create "every 1h" "Use both skills and combine the result" \
  --skill blogwatcher \
  --skill maps \
  --name "Skill combo"
```

### 通过自然对话

直接向 Hermes 描述：

```text
Every morning at 9am, check Hacker News for AI news and send me a summary on Telegram.
```

Hermes 会在内部使用统一的 `cronjob` 工具。

## 附带 skill 的 cron 任务

Cron 任务可以在运行 prompt（提示词）之前加载一个或多个 skill。

### 单个 skill

```python
cronjob(
    action="create",
    skill="blogwatcher",
    prompt="Check the configured feeds and summarize anything new.",
    schedule="0 9 * * *",
    name="Morning feeds",
)
```

### 多个 skill

Skill 按顺序加载。Prompt 作为任务指令叠加在这些 skill 之上。

```python
cronjob(
    action="create",
    skills=["blogwatcher", "maps"],
    prompt="Look for new local events and interesting nearby places, then combine them into one short brief.",
    schedule="every 6h",
    name="Local brief",
)
```

当你希望定时 agent 继承可复用的工作流，而不必将完整的 skill 文本塞入 cron prompt 本身时，这非常有用。

## 在指定项目目录中运行任务

Cron 任务默认与任何代码仓库脱离运行——不加载 `AGENTS.md`、`CLAUDE.md` 或 `.cursorrules`，终端/文件/代码执行工具从 gateway 启动时的工作目录运行。传入 `--workdir`（CLI）或 `workdir=`（工具调用）可更改此行为：

```bash
# 独立 CLI（schedule 和 prompt 为位置参数）
hermes cron create "every 1d at 09:00" \
  "Audit open PRs, summarize CI health, and post to #eng" \
  --workdir /home/me/projects/acme
```

```python
# 在聊天中，通过 cronjob 工具
cronjob(
    action="create",
    schedule="every 1d at 09:00",
    workdir="/home/me/projects/acme",
    prompt="Audit open PRs, summarize CI health, and post to #eng",
)
```

设置 `workdir` 后：

- 该目录中的 `AGENTS.md`、`CLAUDE.md` 和 `.cursorrules` 会被注入系统 prompt（发现顺序与交互式 CLI 相同）
- `terminal`、`read_file`、`write_file`、`patch`、`search_files` 和 `execute_code` 均以该目录为工作目录（通过 `TERMINAL_CWD`）
- 路径必须是已存在的绝对目录——相对路径和不存在的目录在创建/更新时会被拒绝
- 编辑时传入 `--workdir ""`（或工具中的 `workdir=""`）可清除该设置并恢复原有行为

:::note 串行化
设置了 `workdir` 的任务在调度器 tick 时串行运行，而非在并行池中运行。这是有意为之——`TERMINAL_CWD` 是进程全局变量，两个 workdir 任务同时运行会互相破坏各自的 cwd。无 workdir 的任务仍像以前一样并行运行。
:::

## 在指定 profile 中运行 cron 任务

默认情况下，cron 任务继承创建它的 gateway/CLI 所属的 Hermes profile。传入 `--profile <name>`（CLI）或 `profile=`（cronjob 工具）可将任务重定向到不同的 profile——调度器会解析该 profile 的 `HERMES_HOME`，在运行期间临时切换到该 profile，加载其 `.env` 和 `config.yaml`，并在其中执行任务：

```bash
# 将任务固定到 `night-ops` profile，无论在哪里调度
hermes cron create "every 1d at 03:00" \
  "Tail the security log and flag anomalies" \
  --profile night-ops
```

```python
# 在聊天中，通过 cronjob 工具
cronjob(
    action="create",
    schedule="every 1d at 03:00",
    prompt="Tail the security log and flag anomalies",
    profile="night-ops",
)
```

使用 `--profile default` 可显式固定到根 Hermes profile。指定的 profile 必须已存在；调度器不会动态创建 profile。在 `cron edit` 时清除 profile 固定，传入空字符串（`--profile ""` 或 `profile=""`）——任务将恢复在调度器当前所在的 profile 中运行。

如果固定的 profile 后来被删除，调度器会记录警告并回退到在当前 profile 中运行该任务，而不是崩溃——因此过期的 `profile` 引用不会卡住任务。

:::note 串行化
设置了 `profile` 的任务也串行运行，原因与 `workdir` 固定任务相同：切换 `HERMES_HOME` 是进程全局变更，两个 profile 固定任务并行运行会产生竞争。未固定的任务仍在正常并行池中运行。
:::

## 编辑任务

无需删除并重建任务来修改它们。

:::tip 任务引用
下方（以及[生命周期操作](#lifecycle-actions)中）的 `<job_id>` 占位符也接受任务名称（不区分大小写）——当你记得 `morning-digest` 但不记得十六进制 ID 时很方便。精确的任务 ID 优先于名称匹配；如果引用不是 ID 且名称匹配到多个任务，命令会拒绝执行并打印候选 ID 供你消歧义。
:::

### 聊天

```bash
/cron edit <job_id> --schedule "every 4h"
/cron edit <job_id> --prompt "Use the revised task"
/cron edit <job_id> --skill blogwatcher --skill maps
/cron edit <job_id> --remove-skill blogwatcher
/cron edit <job_id> --clear-skills
```

### 独立 CLI

```bash
hermes cron edit <job_id> --schedule "every 4h"
hermes cron edit <job_id> --prompt "Use the revised task"
hermes cron edit <job_id> --skill blogwatcher --skill maps
hermes cron edit <job_id> --add-skill maps
hermes cron edit <job_id> --remove-skill blogwatcher
hermes cron edit <job_id> --clear-skills
```

注意：

- 重复使用 `--skill` 会替换任务已附加的 skill 列表
- `--add-skill` 追加到现有列表，不替换
- `--remove-skill` 删除指定的已附加 skill
- `--clear-skills` 删除所有已附加的 skill

## 生命周期操作

Cron 任务现在拥有比创建/删除更完整的生命周期。

### 聊天

```bash
/cron list
/cron pause <job_id>
/cron resume <job_id>
/cron run <job_id>
/cron remove <job_id>
```

### 独立 CLI

```bash
hermes cron list
hermes cron pause <job_id>
hermes cron resume <job_id>
hermes cron run <job_id>
hermes cron remove <job_id>
hermes cron status
hermes cron tick
```

各操作说明：

- `pause` — 保留任务但停止调度
- `resume` — 重新启用任务并计算下次运行时间
- `run` — 在下次调度器 tick 时触发任务
- `remove` — 彻底删除任务

## 工作原理

**Cron 执行由 gateway 守护进程处理。** Gateway 每 60 秒 tick 一次调度器，在隔离的 agent 会话中运行到期的任务。

```bash
hermes gateway install     # 安装为用户服务
sudo hermes gateway install --system   # Linux：服务器开机启动的系统服务
hermes gateway             # 或在前台运行

hermes cron list
hermes cron status
```

### Gateway 调度器行为

每次 tick 时，Hermes：

1. 从 `~/.hermes/cron/jobs.json` 加载任务
2. 对照当前时间检查 `next_run_at`
3. 为每个到期任务启动全新的 `AIAgent` 会话
4. 可选地将一个或多个已附加的 skill 注入该新会话
5. 将 prompt 运行至完成
6. 投递最终响应
7. 更新运行元数据和下次调度时间

`~/.hermes/cron/.tick.lock` 处的文件锁防止重叠的调度器 tick 重复运行同一批任务。

## 投递选项

调度任务时，你可以指定输出的去向：

| 选项 | 说明 | 示例 |
|--------|-------------|---------|
| `"origin"` | 回传到任务创建的来源 | 消息平台上的默认值 |
| `"local"` | 仅保存到本地文件（`~/.hermes/cron/output/`） | CLI 上的默认值 |
| `"telegram"` | Telegram 主频道 | 使用 `TELEGRAM_HOME_CHANNEL` |
| `"telegram:123456"` | 按 ID 指定的 Telegram 会话 | 直接投递 |
| `"telegram:-100123:17585"` | 指定 Telegram 话题 | `chat_id:thread_id` 格式 |
| `"discord"` | Discord 主频道 | 使用 `DISCORD_HOME_CHANNEL` |
| `"discord:#engineering"` | 按频道名指定的 Discord 频道 | 按频道名 |
| `"slack"` | Slack 主频道 | |
| `"whatsapp"` | WhatsApp 主账号 | |
| `"signal"` | Signal | |
| `"matrix"` | Matrix 主房间 | |
| `"mattermost"` | Mattermost 主频道 | |
| `"email"` | 邮件 | |
| `"sms"` | 通过 Twilio 发送 SMS | |
| `"homeassistant"` | Home Assistant | |
| `"dingtalk"` | 钉钉 | |
| `"feishu"` | 飞书/Lark | |
| `"wecom"` | 企业微信 | |
| `"weixin"` | 微信（WeChat） | |
| `"bluebubbles"` | BlueBubbles（iMessage） | |
| `"qqbot"` | QQ Bot（腾讯 QQ） | |
| `"all"` | 扇出到所有已连接的主频道 | 触发时解析 |
| `"telegram,discord"` | 扇出到指定的一组频道 | 逗号分隔列表 |
| `"origin,all"` | 投递到来源**加上**所有其他已连接频道 | 可组合任意 token |

Agent 的最终响应会自动投递，无需在 cron prompt 中调用 `send_message`。

### 路由意图（`all`）

`all` 让你将一个 cron 任务发送到所有已配置的消息频道，无需逐一列举名称。它在**触发时解析**，因此在你配置 `TELEGRAM_HOME_CHANNEL` 之前创建的任务，会在下次 tick 时自动纳入 Telegram。

语义：`all` 展开为所有已配置主频道的平台。零个也没问题；任务只是没有投递目标，并在上游记录为投递失败。

`all` 可与显式目标组合。`origin,all` 投递到来源会话**加上**所有其他已连接的主频道，按 `(platform, chat_id, thread_id)` 去重。

### Telegram cron 话题（`TELEGRAM_CRON_THREAD_ID`）

启用 Telegram 话题模式后，根 DM 被保留为系统大厅——发送到那里的回复会被拒绝并附带大厅提示，`reply_to_message_id` 会被丢弃，因此你无法回复落在主聊天中的 cron 消息。

将 cron 指向专用的论坛话题：

1. 在 Telegram 中打开机器人 DM，创建一个名为 `Cron` 的话题。长按话题标题 → **复制链接**；末尾的整数即为该话题的 `message_thread_id`。
2. 在 `.env` 中设置 `TELEGRAM_CRON_THREAD_ID=<该 id>`。

这仅适用于 cron 投递。`TELEGRAM_HOME_CHANNEL_THREAD_ID`（用于其他地方，如重启通知）不受影响。显式的 `deliver="telegram:chat_id:thread_id"` 目标仍优先于环境变量。对 cron 消息的回复现在会进入已有的话题会话，你可以直接在其中操作。

### 响应包装

默认情况下，投递的 cron 输出会带有页眉和页脚，以便接收方知道这来自定时任务：

```
Cronjob Response: Morning feeds
-------------

<agent output here>

Note: The agent cannot see this message, and therefore cannot respond to it.
```

若要投递不带包装的原始 agent 输出，将 `cron.wrap_response` 设为 `false`：

```yaml
# ~/.hermes/config.yaml
cron:
  wrap_response: false
```

### 静默抑制

如果 agent 的最终响应以 `[SILENT]` 开头，投递将被完全抑制。输出仍会保存到本地以供审计（位于 `~/.hermes/cron/output/`），但不会向投递目标发送任何消息。

这对于只在出现问题时才需要上报的监控任务很有用：

```text
Check if nginx is running. If everything is healthy, respond with only [SILENT].
Otherwise, report the issue.
```

失败的任务无论 `[SILENT]` 标记如何都会投递——只有成功的运行才能被静默。

## 脚本超时

预运行脚本（通过 `script` 参数附加）的默认超时为 120 秒。如果你的脚本需要更长时间——例如，包含随机延迟以避免类机器人的时序模式——可以增加此值：

```yaml
# ~/.hermes/config.yaml
cron:
  script_timeout_seconds: 300   # 5 分钟
```

或设置 `HERMES_CRON_SCRIPT_TIMEOUT` 环境变量。解析顺序为：环境变量 → config.yaml → 默认 120 秒。

## 无 agent 模式（纯脚本任务）

对于不需要 LLM 推理的周期性任务——经典的看门狗、磁盘/内存告警、心跳、CI ping——在创建时传入 `no_agent=True`。调度器按计划运行你的脚本，并直接投递其 stdout，完全跳过 agent：

```bash
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"
```

语义：

- 脚本 stdout（去除首尾空白）→ 原样作为消息投递。
- **stdout 为空 → 静默 tick**，不投递。这是看门狗模式："只在出现问题时才说话"。
- 非零退出或超时 → 投递错误告警，确保损坏的看门狗不会静默失败。
- 最后一行输出 `{"wakeAgent": false}` → 静默 tick（与 LLM 任务使用相同的门控）。
- 无 token、无模型、无 provider 回退——任务永远不会触及推理层。

`.sh`/`.bash` 文件在 `/bin/bash` 下运行；其他文件在当前 Python 解释器（`sys.executable`）下运行。脚本必须位于 `~/.hermes/scripts/`（与预运行脚本门控相同的沙箱规则）。

### Agent 为你设置这些

`cronjob` 工具的 schema 直接向 Hermes 暴露了 `no_agent`，因此你可以在聊天中描述一个看门狗，让 agent 来配置它：

```text
Ping me on Telegram if RAM is over 85%, every 5 minutes.
```

Hermes 会通过 `write_file` 将检查脚本写入 `~/.hermes/scripts/`，然后调用：

```python
cronjob(action="create", schedule="every 5m",
        script="memory-watchdog.sh", no_agent=True,
        deliver="telegram", name="memory-watchdog")
```

当消息内容完全由脚本决定时（看门狗、阈值告警、心跳），它会自动选择 `no_agent=True`。同一工具也让 agent 可以暂停、恢复、编辑和删除任务——整个生命周期都通过聊天驱动，无需任何人接触 CLI。

参见[纯脚本 Cron 任务指南](/guides/cron-script-only)获取实际示例。

## 通过 `context_from` 串联任务

Cron 任务在隔离的会话中运行，不保留之前运行的记忆。但有时一个任务的输出恰好是下一个任务所需的输入。`context_from` 参数自动建立这种连接——任务 B 的 prompt 在运行时会将任务 A 的最新输出作为上下文前置。

```python
# 任务 1：收集原始数据
cronjob(
    action="create",
    prompt="Fetch the top 10 AI/ML stories from Hacker News. Save them to ~/.hermes/data/briefs/raw.md in markdown format with title, URL, and score.",
    schedule="0 7 * * *",
    name="AI News Collector",
)

# 任务 2：分类——接收任务 1 的输出作为上下文
# 从 cronjob(action="list") 获取任务 1 的 ID
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/raw.md. Score each story 1–10 for engagement potential and novelty. Output the top 5 to ~/.hermes/data/briefs/ranked.md.",
    schedule="30 7 * * *",
    context_from="<job1_id>",
    name="AI News Triage",
)

# 任务 3：发布——接收任务 2 的输出作为上下文
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/ranked.md. Write 3 tweet drafts (hook + body + hashtags). Deliver to telegram:7976161601.",
    schedule="0 8 * * *",
    context_from="<job2_id>",
    name="AI News Brief",
)
```

**工作原理：**

- 任务 2 触发时，Hermes 从 `~/.hermes/cron/output/{job1_id}/*.md` 读取任务 1 的最新输出
- 该输出自动前置到任务 2 的 prompt
- 任务 2 无需硬编码"读取此文件"——它以上下文形式接收内容
- 链可以是任意长度：任务 1 → 任务 2 → 任务 3 → …

**`context_from` 接受的格式：**

| 格式 | 示例 |
|--------|---------|
| 单个任务 ID（字符串） | `context_from="a1b2c3d4"` |
| 多个任务 ID（列表） | `context_from=["job_a", "job_b"]` |

输出按列表顺序拼接。

**适用场景：**

- 多阶段流水线（收集 → 过滤 → 格式化 → 投递）
- 步骤 N 依赖步骤 N−1 输出的依赖任务
- 一个任务聚合多个其他任务结果的扇入模式

## Provider 恢复

Cron 任务继承你配置的回退 provider 和凭证池轮换。如果主 API key 被限速或 provider 返回错误，cron agent 可以：

- **回退到备用 provider**，前提是你在 `config.yaml` 中配置了 `fallback_providers`（或旧版 `fallback_model`）
- **轮换到下一个凭证**，即同一 provider 的[凭证池](/user-guide/configuration#credential-pool-strategies)中的下一个

这意味着高频运行或在高峰时段运行的 cron 任务更具弹性——单个被限速的 key 不会导致整次运行失败。

## 调度格式

Agent 的最终响应会自动投递——你**无需**在 cron prompt 中为同一目标包含 `send_message`。如果 cron 运行调用了 `send_message` 且目标与调度器已投递的目标完全相同，Hermes 会跳过该重复发送，并告知模型将面向用户的内容放在最终响应中。仅对额外或不同的目标使用 `send_message`。

### 相对延迟（一次性）

```text
30m     → 30 分钟后运行一次
2h      → 2 小时后运行一次
1d      → 1 天后运行一次
```

### 间隔（周期性）

```text
every 30m    → 每 30 分钟
every 2h     → 每 2 小时
every 1d     → 每天
```

### Cron 表达式

```text
0 9 * * *       → 每天上午 9:00
0 9 * * 1-5     → 工作日上午 9:00
0 */6 * * *     → 每 6 小时
30 8 1 * *      → 每月 1 日上午 8:30
0 0 * * 0       → 每周日午夜
```

### ISO 时间戳

```text
2026-03-15T09:00:00    → 2026 年 3 月 15 日上午 9:00 一次性运行
```

## 重复行为

| 调度类型 | 默认重复次数 | 行为 |
|--------------|----------------|----------|
| 一次性（`30m`、时间戳） | 1 | 运行一次 |
| 间隔（`every 2h`） | 永久 | 运行直到删除 |
| Cron 表达式 | 永久 | 运行直到删除 |

可以覆盖：

```python
cronjob(
    action="create",
    prompt="...",
    schedule="every 2h",
    repeat=5,
)
```

## 以编程方式管理任务

面向 agent 的 API 是单一工具：

```python
cronjob(action="create", ...)
cronjob(action="list")
cronjob(action="update", job_id="...")
cronjob(action="pause", job_id="...")
cronjob(action="resume", job_id="...")
cronjob(action="run", job_id="...")
cronjob(action="remove", job_id="...")
```

对于 `update`，传入 `skills=[]` 可删除所有已附加的 skill。

## Cron 任务可用的工具集

Cron 在全新的 agent 会话中运行每个任务，不附加任何聊天平台。默认情况下，cron agent 获得**你在 `hermes tools` 中为 `cron` 平台配置的工具集**——不是 CLI 默认值，也不是所有工具。

```bash
hermes tools
# → 在 curses UI 中选择 "cron" 平台
# → 像 Telegram/Discord 等平台一样切换工具集开关
```

通过 `cronjob.create`（或通过 `cronjob.update` 对现有任务）上的 `enabled_toolsets` 字段可进行更精细的单任务控制：

```text
cronjob(action="create", name="weekly-news-summary",
        schedule="every sunday 9am",
        enabled_toolsets=["web", "file"],      # 仅 web + file，无 terminal/browser 等
        prompt="Summarize this week's AI news: ...")
```

当任务上设置了 `enabled_toolsets` 时，它优先生效；否则 `hermes tools` 的 cron 平台配置生效；否则 Hermes 回退到内置默认值。这对成本控制很重要：在每个小型"获取新闻"任务中携带 `moa`、`browser`、`delegation` 会在每次 LLM 调用时膨胀工具 schema prompt。

### 完全跳过 agent：`wakeAgent`

如果你的 cron 任务附加了预检脚本（通过 `script=`），脚本可以在运行时决定 Hermes 是否应该调用 agent。在 stdout 最后一行输出如下格式：

```text
{"wakeAgent": false}
```

……cron 将完全跳过本次 tick 的 agent 运行。适用于高频轮询（每 1–5 分钟），只在状态实际发生变化时才需要唤醒 LLM——否则你会为一遍遍的零内容 agent 轮次付费。

```python
# 预检脚本
import json, sys
latest = fetch_latest_issue_count()
prev = read_state("issue_count")
if latest == prev:
    print(json.dumps({"wakeAgent": False}))   # 跳过本次 tick
    sys.exit(0)
write_state("issue_count", latest)
print(json.dumps({"wakeAgent": True, "context": {"new_issues": latest - prev}}))
```

省略 `wakeAgent` 时，默认为 `true`（照常唤醒 agent）。

#### 实用方案：低成本预运行门控

`wakeAgent` 门控提供了一种零成本的方式，用于决定定时任务是否应该消耗任何 LLM token。三种模式覆盖了大多数使用场景。

**文件变更门控**——仅在被监视文件自上次成功 tick 以来有新内容时运行。调度器记录每个任务的 `last_run_at`；将其与文件的 mtime 比较。

```bash
#!/bin/bash
# ~/.hermes/scripts/feed-changed.sh
FEED="$HOME/data/feed.json"
STATE="$HOME/.hermes/scripts/.feed-changed.last"
test -f "$FEED" || { echo '{"wakeAgent": false}'; exit 0; }
mtime=$(stat -c %Y "$FEED")
last=$(cat "$STATE" 2>/dev/null || echo 0)
if [ "$mtime" -le "$last" ]; then
  echo '{"wakeAgent": false}'
else
  echo "$mtime" > "$STATE"
  echo '{"wakeAgent": true}'
fi
```

```text
cronjob(action="create", name="process-feed",
        schedule="every 30m",
        script="feed-changed.sh",
        prompt="A new ~/data/feed.json has landed. Summarize what changed.")
```

**外部标志门控**——仅在其他进程发出就绪信号时运行（例如，部署 hook 落下一个文件，CI 任务在状态存储中设置一个值）。

```bash
#!/bin/bash
# ~/.hermes/scripts/flag-ready.sh
if test -f /tmp/new-data-ready; then
  rm -f /tmp/new-data-ready
  echo '{"wakeAgent": true}'
else
  echo '{"wakeAgent": false}'
fi
```

```text
cronjob(action="create", name="nightly-analysis",
        schedule="0 9 * * *",
        script="flag-ready.sh",
        prompt="Run the nightly analysis over today's batch.")
```

**SQL 计数门控**——仅在你自己的数据库中有新行需要处理时运行。脚本还可以通过 `context` 将计数传递给 agent，让 agent 无需重新查询就知道数据量。

```python
#!/usr/bin/env python
# ~/.hermes/scripts/new-rows.py
import json, sqlite3
conn = sqlite3.connect("/home/me/data/app.db")
n = conn.execute(
    "SELECT COUNT(*) FROM messages WHERE ts > strftime('%s','now','-2 hours')"
).fetchone()[0]
if n < 1:
    print(json.dumps({"wakeAgent": False}))
else:
    print(json.dumps({"wakeAgent": True, "context": {"new_rows": n}}))
```

```text
cronjob(action="create", name="summarize-new-msgs",
        schedule="every 2h",
        script="new-rows.py",
        prompt="Summarize the new messages from the last 2 hours.")
```

同样的模式适用于任何可以从脚本查询的数据源——Postgres、HTTP API、你自己的状态存储——无需将 SQL 求值器内置到 cron 子系统中。

:::tip
Hermes 自身的 `~/.hermes/state.db` 是内部 schema，会在版本间变更。不要从预运行门控中查询它——指向你自己的数据库或 feed。
:::

致谢：此方案集由 @iankar8 在 [#2654](https://github.com/NousResearch/hermes-agent/pull/2654) 中的探索所启发，该 PR 提议将 sql/file/command 触发器作为并行机制添加。`script` + `wakeAgent` 门控已以零成本覆盖了所有三种情况，因此该工作以文档形式落地。

### 串联任务：`context_from`

Cron 任务可以通过在 `context_from` 中列出其他任务的名称（或 ID）来消费这些任务最近一次成功运行的输出：

```text
cronjob(action="create", name="daily-digest",
        schedule="every day 7am",
        context_from=["ai-news-fetch", "github-prs-fetch"],
        prompt="Write the daily digest using the outputs above.")
```

被引用任务最近一次完成的输出会作为上下文注入到本次运行的 prompt 之上。每个上游条目必须是有效的任务 ID 或名称（参见 `cronjob action="list"`）。注意：串联读取的是*最近一次完成*的输出——它不会等待同一 tick 中正在运行的上游任务。

## 任务存储

任务存储在 `~/.hermes/cron/jobs.json`。任务运行的输出保存到 `~/.hermes/cron/output/{job_id}/{timestamp}.md`。

任务可能将 `model` 和 `provider` 存储为 `null`。省略这些字段时，Hermes 在执行时从全局配置中解析它们。只有设置了单任务覆盖时，这些字段才会出现在任务记录中。

存储使用原子文件写入，因此中断的写入不会留下部分写入的任务文件。

## 自包含的 prompt 仍然重要

:::warning 重要
Cron 任务在完全全新的 agent 会话中运行。Prompt 必须包含 agent 所需的一切，除非已由附加的 skill 提供。
:::

**错误：** `"Check on that server issue"`

**正确：** `"SSH into server 192.168.1.100 as user 'deploy', check if nginx is running with 'systemctl status nginx', and verify https://example.com returns HTTP 200."`

## 安全性

定时任务的 prompt 在创建和更新时会扫描 prompt 注入和凭证外泄模式。包含不可见 Unicode 技巧、SSH 后门尝试或明显的密钥外泄载荷的 prompt 会被拦截。