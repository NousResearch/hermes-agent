---
sidebar_position: 5
title: "定时任务（Cron）"
description: "使用自然语言安排自动化任务，通过一个 cron 工具管理它们，并附加一个或多个技能"
---

# 定时任务（Cron）

使用自然语言或 cron 表达式安排任务自动运行。Hermes 通过单一的 `cronjob` 工具暴露 cron 管理功能，采用操作式而非分离的 schedule/list/remove 工具。

## cron 能做什么

Cron 任务可以：

- 安排一次性或循环任务
- 暂停、恢复、编辑、触发和删除任务
- 附加零个、一个或多个技能到任务
- 将结果传回到发起聊天的源、本地文件或配置的平台目标
- 在带有正常静态工具列表的新鲜 agent 会话中运行
- 以 **无 agent 模式** 运行——按计划运行的脚本，逐字传递 stdout，零 LLM 调用（见下方[无 agent 模式](#no-agent-mode-script-only-jobs)）

所有这些功能都通过 `cronjob` 工具对 Hermes 本身可用，因此您可以用纯语言创建、暂停、编辑和删除任务——无需 CLI。

:::warning
Cron 运行的会话不能递归创建更多 cron 任务。Hermes 在 cron 执行内部禁用 cron 管理工具，防止失控的调度循环。
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

正常询问 Hermes：

```text
Every morning at 9am, check Hacker News for AI news and send me a summary on Telegram.
```

Hermes 内部使用统一的 `cronjob` 工具。

## 技能支撑的 Cron 任务

Cron 任务可以在运行提示之前加载一个或多个技能。

### 单个技能

```python
cronjob(
    action="create",
    skill="blogwatcher",
    prompt="Check the configured feeds and summarize anything new.",
    schedule="0 9 * * *",
    name="Morning feeds",
)
```

### 多个技能

技能按顺序加载。提示成为叠加在这些技能之上的任务指令。

```python
cronjob(
    action="create",
    skills=["blogwatcher", "maps"],
    prompt="Look for new local events and interesting nearby places, then combine them into one short brief.",
    schedule="every 6h",
    name="Local brief",
)
```

当您希望调度 agent 继承可复用工作流而不将完整技能文本塞入 cron 提示时，这很有用。

## 在项目目录中运行任务

Cron 任务默认与任何 repo 分离运行——不加载 `AGENTS.md`、`CLAUDE.md` 或 `.cursorrules`，终端/文件/code-exec 工具从网关启动时的工作目录运行。通过 `--workdir`（CLI）或 `workdir=`（工具调用）来更改：

```bash
# 独立 CLI（计划和提示是位置参数）
hermes cron create "every 1d at 09:00" \
  "Audit open PRs, summarize CI health, and post to #eng" \
  --workdir /home/me/projects/acme
```

```python
# 从聊天，通过 cronjob 工具
cronjob(
    action="create",
    schedule="every 1d at 09:00",
    workdir="/home/me/projects/acme",
    prompt="Audit open PRs, summarize CI health, and post to #eng",
)
```

设置 `workdir` 时：

- 该目录的 `AGENTS.md`、`CLAUDE.md` 和 `.cursorrules` 被注入到系统提示中（与交互式 CLI 相同的发现顺序）
- `terminal`、`read_file`、`write_file`、`patch`、`search_files` 和 `execute_code` 都使用该目录作为工作目录（通过 `TERMINAL_CWD`）
- 路径必须是存在的绝对目录——相对路径和缺失目录在创建/更新时被拒绝
- 在编辑时传递 `--workdir ""`（或通过工具的 `workdir=""`）以清除并恢复旧行为

:::note 序列化
带有 `workdir` 的任务在调度器 tick 时按顺序运行，不在并行池中。这是故意的——`TERMINAL_CWD` 是进程全局的，因此两个 workdir 任务同时运行会损坏彼此的 cwd。没有 workdir 的任务仍像以前一样并行运行。
:::

## 编辑任务

您不需要删除并重新创建任务来更改它们。

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
hermes cron add-skill maps
hermes cron edit <job_id> --remove-skill blogwatcher
hermes cron edit <job_id> --clear-skills
```

注意：

- 重复的 `--skill` 替换任务的附加技能列表
- `--add-skill` 追加到现有列表而不替换
- `--remove-skill` 移除特定附加技能
- `--clear-skills` 移除所有附加技能

## 生命周期操作

Cron 任务现在有比 create/remove 更完整的生命周期。

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
- `resume` — 重新启用任务并计算下一次运行时间
- `run` — 在下一个调度器 tick 触发任务
- `remove` — 完全删除

## 工作原理

**Cron 执行由网关守护进程处理。** 网关每 60 秒 tick 一次调度器，在隔离的 agent 会话中运行任何到期任务。

```bash
hermes gateway install     # 安装为用户服务
sudo hermes gateway install --system   # Linux：开机自启系统服务
hermes gateway             # 或在前台运行

hermes cron list
hermes cron status
```

### 网关调度器行为

每次 tick 时 Hermes：

1. 从 `~/.hermes/cron/jobs.json` 加载任务
2. 检查 `next_run_at` 与当前时间
3. 为每个到期任务启动一个新的 `AIAgent` 会话
4. 可选地将一个或多个附加技能注入该新会话
5. 运行提示直到完成
6. 传递最终响应
7. 更新运行元数据和下一个调度时间

`~/.hermes/cron/.tick.lock` 上的文件锁防止重叠的调度器 tick 双重运行同一批次任务。

## 传递选项

调度任务时，指定输出位置：

| 选项 | 描述 | 示例 |
|--------|-------------|---------|
| `"origin"` | 返回到任务创建的地方 | 消息平台的默认设置 |
| `"local"` | 仅保存到本地文件（`~/.hermes/cron/output/`） | CLI 的默认设置 |
| `"telegram"` | Telegram 主页频道 | 使用 `TELEGRAM_HOME_CHANNEL` |
| `"telegram:123456"` | 按 ID 的特定 Telegram 聊天 | 直接传递 |
| `"telegram:-100123:17585"` | 特定 Telegram 主题 | `chat_id:thread_id` 格式 |
| `"discord"` | Discord 主页频道 | 使用 `DISCORD_HOME_CHANNEL` |
| `"discord:#engineering"` | 特定 Discord 频道 | 按频道名称 |
| `"slack"` | Slack 主页频道 | |
| `"whatsapp"` | WhatsApp 主页 | |
| `"signal"` | Signal | |
| `"matrix"` | Matrix 主页房间 | |
| `"mattermost"` | Mattermost 主页频道 | |
| `"email"` | 电子邮件 | |
| `"sms"` | 通过 Twilio 发送 SMS | |
| `"homeassistant"` | Home Assistant | |
| `"dingtalk"` | DingTalk | |
| `"feishu"` | Feishu/Lark | |
| `"wecom"` | WeCom | |
| `"weixin"` | Weixin | |
| `"bluebubbles"` | BlueBubbles（iMessage） | |
| `"qqbot"` | QQ 机器人 | |

Agent 的最终响应会自动传递。您不需要在 cron 提示中调用 `send_message`。

### 响应包装

默认情况下，传递的 cron 输出用页眉和页脚包装，以便接收方知道它来自调度任务：

```
Cronjob Response: Morning feeds
-------------

<agent output here>

Note: The agent cannot see this message, and therefore cannot respond to it.
```

要传递不含包装的原始 agent 输出，设置 `cron.wrap_response` 为 `false`：

```yaml
# ~/.hermes/config.yaml
cron:
  wrap_response: false
```

### 静默抑制

如果 agent 的最终响应以 `[SILENT]` 开头，则完全抑制传递。输出仍保存到本地供审计（`~/.hermes/cron/output/`），但不发送到传递目标。

这对于只应在出错时报告的监控任务很有用：

```text
Check if nginx is running. If everything is healthy, respond with only [SILENT].
Otherwise, report the issue.
```

失败的任务总是会传递，无论 `[SILENT]` 标记如何——只有成功的运行可以被静默。

## 脚本超时

通过 `script` 参数附加的预运行脚本默认超时为 120 秒。如果您的脚本需要更长时间——例如，包含避免 bot 式时间模式的随机延迟——可以增加：

```yaml
# ~/.hermes/config.yaml
cron:
  script_timeout_seconds: 300   # 5 分钟
```

或设置 `HERMES_CRON_SCRIPT_TIMEOUT` 环境变量。解析顺序：环境变量 → config.yaml → 120秒默认值。

## 无 Agent 模式（纯脚本任务）

对于不需要 LLM 推理的循环任务——经典的看门狗、磁盘/内存警报、心跳、CI ping——在创建时传递 `no_agent=True`。调度器按计划运行脚本并直接传递其 stdout，完全跳过 agent：

```bash
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"
```

语义：

- 脚本 stdout（修剪后）→ 逐字作为消息传递。
- **空 stdout → 静默 tick**，无传递。这就是看门狗模式："只在出错时说些什么"。
- 非零退出或超时 → 传递错误警报，因此损坏的看门狗不会静默失败。
- 最后一行 `{"wakeAgent": false}` → 静默 tick（与 LLM 任务使用相同的门控）。
- 无令牌、无模型、无提供商回退——任务从不接触推理层。

`.sh` / `.bash` 文件在 `/bin/bash` 下运行；其他在当前 Python 解释器（`sys.executable`）下运行。脚本必须位于 `~/.hermes/scripts/`（与预运行脚本门控相同的沙箱规则）。

### Agent 为您设置这些

`cronjob` 工具的 schema 向 Hermes 直接暴露 `no_agent`，因此您可以在聊天中描述看门狗，让 agent 连接它：

```text
Ping me on Telegram if RAM is over 85%, every 5 minutes.
```

Hermes 会通过 `write_file` 将检查脚本写入 `~/.hermes/scripts/`，然后调用：

```python
cronjob(action="create", schedule="every 5m",
        script="memory-watchdog.sh", no_agent=True,
        deliver="telegram", name="memory-watchdog")
```

当消息内容完全由脚本决定时（看门狗、阈值警报、心跳），它自动选择 `no_agent=True`。同一工具还让 agent 暂停、恢复、编辑和删除任务——因此整个生命周期由聊天驱动，无需任何人接触 CLI。

参见[纯脚本 Cron 任务指南](/docs/guides/cron-script-only)中的完整示例。

## 使用 `context_from` 链接任务

Cron 任务在隔离会话中运行，没有先前运行的记忆。但有时一个任务的输出正是下一个任务需要的。`context_from` 参数自动连接——Job B 的提示在运行时获得 Job A 的最新输出作为前置上下文。

```python
# 任务 1：收集原始数据
cronjob(
    action="create",
    prompt="Fetch the top 10 AI/ML stories from Hacker News. Save them to ~/.hermes/data/briefs/raw.md in markdown format with title, URL, and score.",
    schedule="0 7 * * *",
    name="AI News Collector",
)

# 任务 2：分类 — 接收任务 1 的输出作为上下文
# 从 cronjob(action="list") 获取任务 1 的 ID
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/raw.md. Score each story 1–10 for engagement potential and novelty. Output the top 5 to ~/.hermes/data/briefs/ranked.md.",
    schedule="30 7 * * *",
    context_from="<job1_id>",
    name="AI News Triage",
)

# 任务 3：发布 — 接收任务 2 的输出作为上下文
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/ranked.md. Write 3 tweet drafts (hook + body + hashtags). Deliver to telegram:7976161601.",
    schedule="0 8 * * *",
    context_from="<job2_id>",
    name="AI News Brief",
)
```

**工作原理：**

- 当任务 2 触发时，Hermes 从 `~/.hermes/cron/output/{job1_id}/*.md` 读取任务 1 的最新输出
- 该输出自动前置到任务 2 的提示
- 任务 2 不需要硬编码"读取此文件"——它将内容作为上下文接收
- 链接可以是任意长度：任务 1 → 任务 2 → 任务 3 → ...

**`context_from` 接受的格式：**

| 格式 | 示例 |
|--------|---------|
| 单个任务 ID（字符串） | `context_from="a1b2c3d4"` |
| 多个任务 ID（列表） | `context_from=["job_a", "job_b"]` |

输出按列出顺序连接。

**何时使用：**

- 多阶段流水线（收集 → 过滤 → 格式化 → 传递）
- 步骤 N 的工作依赖于步骤 N-1 输出的相关任务
- Fan-out/fan-in 模式，其中一个任务聚合多个其他任务的结果

## 提供商恢复

Cron 任务继承您配置的备用提供商和凭证池轮换。如果主 API 密钥被限流或提供商返回错误，cron agent 可以：

- 如果您在 `config.yaml` 中配置了 `fallback_providers`（或旧的 `fallback_model`），**回退到备用提供商**
- **轮换到同一提供商**的[凭证池](./credential-pools.md)中下一个凭证

这意味着高频运行或在高峰时段运行的任务更有弹性——单个限流密钥不会导致整个运行失败。

## 计划格式

Agent 的最终响应自动传递——您**不需要**在 cron 提示中包含 `send_message` 来发送到同一目标。如果 cron 运行调用 `send_message` 到调度器已经要传递的确切目标，Hermes 跳过该重复发送并告诉模型将面向用户的内容放在最终响应中。只有在其他或不同目标时才使用 `send_message`。

### 相对延迟（一次性）

```text
30m     → 30 分钟后运行一次
2h      → 2 小时后运行一次
1d      → 1 天后运行一次
```

### 间隔（循环）

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

| 计划类型 | 默认重复次数 | 行为 |
|--------------|----------------|----------|
| 一次性（`30m`、时间戳） | 1 | 运行一次 |
| 间隔（`every 2h`） | 永久 | 运行直到被删除 |
| Cron 表达式 | 永久 | 运行直到被删除 |

您可以覆盖：

```python
cronjob(
    action="create",
    prompt="...",
    schedule="every 2h",
    repeat=5,
)
```

## 编程管理任务

面向 agent 的 API 是一个工具：

```python
cronjob(action="create", ...)
cronjob(action="list")
cronjob(action="update", job_id="...")
cronjob(action="pause", job_id="...")
cronjob(action="resume", job_id="...")
cronjob(action="run", job_id="...")
cronjob(action="remove", job_id="...")
```

对于 `update`，传递 `skills=[]` 以移除所有附加技能。

## 任务可用的工具集

Cron 在带有聊天平台附加的新鲜 agent 会话中运行每个任务。默认情况下，cron agent 获得**您在 `hermes tools` 中为 `cron` 平台配置的工具集**——不是 CLI 默认，不是所有可能的工具。

```bash
hermes tools
# → 在 curses UI 中选择 "cron" 平台
# → 像为 Telegram/Discord/etc 配置一样切换工具集开/关
```

通过 `cronjob.create`（或通过 `cronjob.update` 在现有任务上）的 `enabled_toolsets` 字段可以获得更精细的按任务控制：

```text
cronjob(action="create", name="weekly-news-summary",
        schedule="every sunday 9am",
        enabled_toolsets=["web", "file"],      # 只有 web + file，没有 terminal/browser/etc
        prompt="Summarize this week's AI news: ...")
```

当任务上设置了 `enabled_toolsets` 时，它优先；否则使用 `hermes tools` cron 平台配置；否则 Hermes 回退到内置默认值。这对成本控制很重要：将 `moa`、`browser`、`delegation` 带入每个小的"获取新闻"任务会增加每个 LLM 调用的工具 schema 提示。

### 完全跳过 agent：`wakeAgent`

如果您的 cron 任务附加了预检查脚本（通过 `script=`），脚本可以在运行时决定 Hermes 是否应该调用 agent。输出一行最终 stdout 格式：

```text
{"wakeAgent": false}
```

...cron 完全跳过本次 tick 的 agent 运行。对频繁轮询（每 1-5 分钟）很有用，这些轮询只在状态实际改变时才需要唤醒 LLM——否则会反复为零内容 agent 调用付费。

```python
# 预检查脚本
import json, sys
latest = fetch_latest_issue_count()
prev = read_state("issue_count")
if latest == prev:
    print(json.dumps({"wakeAgent": False}))   # 跳过本次 tick
    sys.exit(0)
write_state("issue_count", latest)
print(json.dumps({"wakeAgent": True, "context": {"new_issues": latest - prev}}))
```

省略 `wakeAgent` 时，默认值为 `true`（像往常一样唤醒 agent）。

### 链接任务：`context_from`

cron 任务可以通过在 `context_from` 中列出其他任务的名称（或 ID）来消费一个或多个其他任务最近成功的输出：

```text
cronjob(action="create", name="daily-digest",
        schedule="every day 7am",
        context_from=["ai-news-fetch", "github-prs-fetch"],
        prompt="Write the daily digest using the outputs above.")
```

被引用任务的最近完成输出被注入到本次运行的提示上方作为上下文。每个上游条目必须是有效的任务 ID 或名称（参见 `cronjob action="list"`）。注意：链接读取*最近完成*的输出——它不等待在同一 tick 中运行的上游任务。

## 任务存储

任务存储在 `~/.hermes/cron/jobs.json`。任务运行的输出保存到 `~/.hermes/cron/output/{job_id}/{timestamp}.md`。

任务可以存储 `model` 和 `provider` 为 `null`。当这些字段被省略时，Hermes 在执行时从全局配置解析。它们只有在设置了按任务覆盖时才会出现在任务记录中。

存储使用原子文件写入，这样中断的写入不会留下部分写入的任务文件。

## 自包含提示仍然重要

:::warning 重要
Cron 任务在完全新鲜的 agent 会话中运行。提示必须包含 agent 所需的一切，而不是附加技能已经提供的内容。
:::

**不好：** `"Check on that server issue"`

**好：** `"SSH into server 192.168.1.100 as user 'deploy', check if nginx is running with 'systemctl status nginx', and verify https://example.com returns HTTP 200."`

## 安全性

调度任务提示在创建和更新时会被扫描是否有提示注入和凭证泄露模式。包含不可见 Unicode 技巧、SSH 后门尝试或明显 secret 泄露负载的提示会被阻止。
