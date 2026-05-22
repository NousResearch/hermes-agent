---
sidebar_position: 5
title: "定时任务 (Cron)"
description: "使用自然语言或 cron 表达式安排自动化任务，用一个 cron 工具管理它们，并附加一个或多个技能"
---

# 定时任务 (Cron)

使用自然语言或 cron 表达式安排任务自动运行。Hermes 通过单个 `cronjob` 工具公开 cron 管理，采用 action 风格操作，而非单独的 schedule/list/remove 工具。

## Cron 现在能做什么

Cron 作业可以：

- 安排一次性或重复任务
- 暂停、恢复、编辑、触发和移除作业
- 为零个、一个或多个技能附加到作业
- 将结果交付回原始聊天、本地文件或配置的平台目标
- 在具有正常静态工具列表的全新智能体会话中运行
- 以 **no-agent 模式**运行 —— 按计划运行脚本，其 stdout 逐字交付，零 LLM 参与（请参阅下面的 [no-agent 模式](#no-agent-模式仅脚本作业) 部分）

所有这些都可通过 `cronjob` 工具供 Hermes 本身使用，因此您可以用 plain language 创建、暂停、编辑和移除作业 —— 无需 CLI。

:::warning
Cron 运行会话不能递归创建更多 cron 作业。Hermes 在 cron 执行中禁用 cron 管理工具，以防止失控的调度循环。
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
每天早上 9 点，查看 Hacker News 上的 AI 新闻，并在 Telegram 上给我发送摘要。
```

Hermes 将在内部使用统一的 `cronjob` 工具。

## 技能支持的 Cron 作业

Cron 作业可以在运行提示之前加载一个或多个技能。

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

当您希望定时智能体继承可重用工作流而无需将完整技能文本塞入 cron 提示本身时，这很有用。

## 在项目目录中运行作业

Cron 作业默认与任何仓库分离运行 —— 不加载 `AGENTS.md`、`CLAUDE.md` 或 `.cursorrules`，终端 / 文件 / 代码执行工具从网关启动时的工作目录运行。传递 `--workdir`（CLI）或 `workdir=`（工具调用）来更改：

```bash
# 独立 CLI（schedule 和 prompt 是位置参数）
hermes cron create "every 1d at 09:00" \
  "Audit open PRs, summarize CI health, and post to #eng" \
  --workdir /home/me/projects/acme
```

```python
# 从聊天中，通过 cronjob 工具
cronjob(
    action="create",
    schedule="every 1d at 09:00",
    workdir="/home/me/projects/acme",
    prompt="Audit open PRs, summarize CI health, and post to #eng",
)
```

当设置 `workdir` 时：

- 该目录中的 `AGENTS.md`、`CLAUDE.md` 和 `.cursorrules` 被注入系统提示（与交互式 CLI 相同的发现顺序）
- `terminal`、`read_file`、`write_file`、`patch`、`search_files` 和 `execute_code` 都使用该目录作为工作目录（通过 `TERMINAL_CWD`）
- 路径必须是存在的绝对目录 —— 相对路径和缺失的目录在创建 / 更新时被拒绝
- 在编辑时传递 `--workdir ""`（或通过工具的 `workdir=""`）以清除它并恢复旧行为

:::note 序列化
带有 `workdir` 的作业在调度器 tick 上顺序运行，而非并行池中。这是故意的 —— `TERMINAL_CWD` 是进程全局的，因此两个同时运行的 workdir 作业会互相破坏 cwd。无 workdir 的作业仍像以前一样并行运行。
:::

## 编辑作业

您无需删除并重新创建作业来更改它们。

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

- 重复的 `--skill` 替换作业的附加技能列表
- `--add-skill` 附加到现有列表而不替换它
- `--remove-skill` 移除特定附加技能
- `--clear-skills` 移除所有附加技能

## 生命周期操作

Cron 作业现在拥有比仅创建/移除更完整的生命周期。

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

它们的作用：

- `pause` —— 保留作业但停止调度它
- `resume` —— 重新启用作业并计算下一次未来运行
- `run` —— 在下一次调度器 tick 上触发作业
- `remove` —— 完全删除它

## 工作原理

**Cron 执行由网关守护进程处理。** 网关每 60 秒 tick 一次调度器，在隔离的智能体会话中运行任何到期作业。

```bash
hermes gateway install     # 安装为用户服务
sudo hermes gateway install --system   # Linux：服务器开机时系统服务
hermes gateway             # 或前台运行

hermes cron list
hermes cron status
```

### 网关调度器行为

每次 tick Hermes：

1. 从 `~/.hermes/cron/jobs.json` 加载作业
2. 将 `next_run_at` 与当前时间对比
3. 为每个到期作业启动全新的 `AIAgent` 会话
4. 可选地将一个或多个附加技能注入该全新会话
5. 将提示运行到完成
6. 交付最终响应
7. 更新运行元数据和下一次计划时间

`~/.hermes/cron/.tick.lock` 上的文件锁防止重叠的调度器 tick 双运行同一批作业。

## 交付选项

安排作业时，您指定输出去向：

| 选项 | 描述 | 示例 |
|--------|-------------|---------|
| `"origin"` | 回到作业创建的地方 | 消息平台上的默认 |
| `"local"` | 仅保存到本地文件（`~/.hermes/cron/output/`） | CLI 上的默认 |
| `"telegram"` | Telegram 主频道 | 使用 `TELEGRAM_HOME_CHANNEL` |
| `"telegram:123456"` | 按 ID 的特定 Telegram 聊天 | 直接交付 |
| `"telegram:-100123:17585"` | 特定 Telegram 主题 | `chat_id:thread_id` 格式 |
| `"discord"` | Discord 主频道 | 使用 `DISCORD_HOME_CHANNEL` |
| `"discord:#engineering"` | 特定 Discord 频道 | 按频道名称 |
| `"slack"` | Slack 主频道 | |
| `"whatsapp"` | WhatsApp 主页 | |
| `"signal"` | Signal | |
| `"matrix"` | Matrix 主房间 | |
| `"mattermost"` | Mattermost 主频道 | |
| `"email"` | 电子邮件 | |
| `"sms"` | 通过 Twilio 的 SMS | |
| `"homeassistant"` | Home Assistant | |
| `"dingtalk"` | 钉钉 | |
| `"feishu"` | 飞书/Lark | |
| `"wecom"` | 企业微信 | |
| `"weixin"` | 微信 | |
| `"bluebubbles"` | BlueBubbles (iMessage) | |
| `"qqbot"` | QQ 机器人 (腾讯 QQ) | |
| `"all"` | 扇出到每个连接的主频道 | 触发时解析 |
| `"telegram,discord"` | 扇出到特定频道集 | 逗号分隔列表 |
| `"origin,all"` | 交付到原始**加上**每个其他连接频道 | 组合任何 token |

智能体的最终响应自动交付。您无需在 cron 提示中调用 `send_message`。

### 路由意图 (`all`)

`all` 让您将单个 cron 作业发送到您配置的每个消息频道，无需按名称枚举它们。它在**触发时解析**，因此您在设置 `TELEGRAM_HOME_CHANNEL` 之前创建的作业将在设置后的下一次 tick 上接收到 Telegram。

语义：`all` 展开为每个配置了主频道的平台。零也可以；作业只是不产生交付目标，并在上游记录为交付失败。

`all` 与显式目标组合。`origin,all` 交付到原始聊天*加上*每个其他连接的主频道，按 `(platform, chat_id, thread_id)` 去重。

### 响应包装

默认情况下，交付的 cron 输出包装有页眉和页脚，以便收件人知道它来自定时任务：

```
Cronjob Response: Morning feeds
-------------

<agent output here>

Note: The agent cannot see this message, and therefore cannot respond to it.
```

要不带包装器交付原始智能体输出，将 `cron.wrap_response` 设置为 `false`：

```yaml
# ~/.hermes/config.yaml
cron:
  wrap_response: false
```

### 静默抑制

如果智能体的最终响应以 `[SILENT]` 开头，交付将被完全抑制。输出仍本地保存以供审计（在 `~/.hermes/cron/output/` 中），但没有消息发送到交付目标。

这适用于仅应在出现问题时报告的监控作业：

```text
Check if nginx is running. If everything is healthy, respond with only [SILENT].
Otherwise, report the issue.
```

失败的作业无论 `[SILENT]` 标记如何都会交付 —— 只有成功的运行可以被静默。

## 脚本超时

预运行脚本（通过 `script` 参数附加）的默认超时为 120 秒。如果您的脚本需要更长时间 —— 例如，包含避免机器人般定时模式的随机延迟 —— 您可以增加：

```yaml
# ~/.hermes/config.yaml
cron:
  script_timeout_seconds: 300   # 5 分钟
```

或设置 `HERMES_CRON_SCRIPT_TIMEOUT` 环境变量。解析顺序为：环境变量 → config.yaml → 120 秒默认值。

## No-agent 模式（仅脚本作业）

对于不需要 LLM 推理的重复作业 —— 经典看门狗、磁盘/内存警报、心跳、CI ping —— 在创建时传递 `no_agent=True`。调度器按计划运行您的脚本并直接交付其 stdout，完全跳过智能体：

```bash
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"
```

语义：

- 脚本 stdout（修剪后）→ 逐字交付为消息。
- **空 stdout → 静默 tick**，无交付。这是看门狗模式："仅在有问题时说些什么"。
- 非零退出或超时 → 交付错误警报，因此损坏的看门狗不能静默失败。
- 最后一行上的 `{"wakeAgent": false}` → 静默 tick（与 LLM 作业使用相同的门）。
- 无 token、无模型、无提供商回退 —— 作业从不接触推理层。

`.sh` / `.bash` 文件在 `/bin/bash` 下运行；其他任何东西在当前 Python 解释器（`sys.executable`）下运行。脚本必须位于 `~/.hermes/scripts/`（与预运行脚本门相同的沙盒规则）。

### 智能体为您设置这些

`cronjob` 工具的 schema 直接向 Hermes 公开 `no_agent`，因此您可以在聊天中描述一个看门狗，让智能体连接它：

```text
如果 RAM 超过 85%，每 5 分钟在 Telegram 上 ping 我。
```

Hermes 将通过 `write_file` 将检查脚本写入 `~/.hermes/scripts/`，然后调用：

```python
cronjob(action="create", schedule="every 5m",
        script="memory-watchdog.sh", no_agent=True,
        deliver="telegram", name="memory-watchdog")
```

当消息内容完全由脚本确定时（看门狗、阈值警报、心跳），它自动选择 `no_agent=True`。相同的工具还让智能体暂停、恢复、编辑和移除作业 —— 因此整个生命周期都是聊天驱动的，无需任何人接触 CLI。

请参阅 [仅脚本 Cron 作业指南](/guides/cron-script-only) 以获取工作示例。

## 使用 `context_from` 链式作业

Cron 作业在隔离会话中运行，没有先前运行的记忆。但有时一个作业的输出正是下一个作业需要的。`context_from` 参数自动连接 —— 作业 B 的提示在运行时将作业 A 的最新输出作为上下文前置。

```python
# 作业 1：收集原始数据
cronjob(
    action="create",
    prompt="Fetch the top 10 AI/ML stories from Hacker News. Save them to ~/.hermes/data/briefs/raw.md in markdown format with title, URL, and score.",
    schedule="0 7 * * *",
    name="AI News Collector",
)

# 作业 2：筛选 —— 接收作业 1 的输出作为上下文
# 从以下获取作业 1 的 ID：cronjob(action="list")
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/raw.md. Score each story 1–10 for engagement potential and novelty. Output the top 5 to ~/.hermes/data/briefs/ranked.md.",
    schedule="30 7 * * *",
    context_from="<job1_id>",
    name="AI News Triage",
)

# 作业 3：发布 —— 接收作业 2 的输出作为上下文
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/ranked.md. Write 3 tweet drafts (hook + body + hashtags). Deliver to telegram:7976161601.",
    schedule="0 8 * * *",
    context_from="<job2_id>",
    name="AI News Brief",
)
```

**工作原理：**

- 当作业 2 触发时，Hermes 从 `~/.hermes/cron/output/{job1_id}/*.md` 读取作业 1 的最新输出
- 该输出自动前置到作业 2 的提示
- 作业 2 不需要硬编码 "read this file" —— 它接收内容为上下文
- 链可以是任意长度：作业 1 → 作业 2 → 作业 3 → ...

**`context_from` 接受什么：**

| 格式 | 示例 |
|--------|---------|
| 单个作业 ID（字符串） | `context_from="a1b2c3d4"` |
| 多个作业 ID（列表） | `context_from=["job_a", "job_b"]` |

输出按列出的顺序连接。

**何时使用它：**

- 多阶段流水线（收集 → 筛选 → 格式化 → 交付）
- 依赖任务，其中步骤 N 的工作依赖于步骤 N−1 的输出
- 扇出/扇入模式，其中一个作业聚合来自其他几个作业的结果

## 提供商恢复

Cron 作业继承您配置的 fallback 提供商和凭证池轮换。如果主 API 密钥被速率限制或提供商返回错误，cron 智能体可以：

- **回退到备用提供商**，如果您在 `config.yaml` 中配置了 `fallback_providers`（或旧版 `fallback_model`）
- **轮换到凭证池中的下一个凭证**，用于同一提供商

这意味着高频运行或高峰时段的 cron 作业更具弹性 —— 单个速率限制密钥不会导致整个运行失败。

## 计划格式

智能体的最终响应自动交付 —— 您**无需**在 cron 提示中包含 `send_message` 来实现相同目标。如果 cron 运行调用 `send_message` 到调度器将交付的确切目标，Hermes 跳过该重复发送，并告诉模型将用户面向的内容放入最终响应中。仅对额外或不同的目标使用 `send_message`。

### 相对延迟（一次性）

```text
30m     → 30 分钟后运行一次
2h      → 2 小时后运行一次
1d      → 1 天后运行一次
```

### 间隔（重复）

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
30 8 1 * *      → 每月第一天上午 8:30
0 0 * * 0       → 每周日午夜
```

### ISO 时间戳

```text
2026-03-15T09:00:00    → 2026 年 3 月 15 日上午 9:00 一次性
```

## 重复行为

| 计划类型 | 默认重复 | 行为 |
|--------------|----------------|----------|
| 一次性（`30m`、时间戳） | 1 | 运行一次 |
| 间隔（`every 2h`） | forever | 直到移除前持续运行 |
| Cron 表达式 | forever | 直到移除前持续运行 |

您可以覆盖它：

```python
cronjob(
    action="create",
    prompt="...",
    schedule="every 2h",
    repeat=5,
)
```

## 以编程方式管理作业

面向智能体的 API 是一个工具：

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

## Cron 作业可用的工具集

Cron 在每个作业的全新智能体会话中运行，没有聊天平台附加。默认情况下，cron 智能体获得**您在 `hermes tools` 中为 `cron` 平台配置的工具集** —— 不是 CLI 默认值，不是一切。

```bash
hermes tools
# → 在 curses UI 中选择 "cron" 平台
# → 像为 Telegram/Discord 等切换工具集一样切换开/关
```

更严格的每作业控制可通过 `cronjob.create` 上的 `enabled_toolsets` 字段（或通过 `cronjob.update` 在现有作业上）：

```text
cronjob(action="create", name="weekly-news-summary",
        schedule="every sunday 9am",
        enabled_toolsets=["web", "file"],      # 仅 web + file，无 terminal/browser 等
        prompt="Summarize this week's AI news: ...")
```

当在作业上设置 `enabled_toolsets` 时，它胜出；否则 `hermes tools` cron 平台配置胜出；否则 Hermes 回退到内置默认值。这很重要，用于成本控制：将 `moa`、`browser`、`delegation` 带入每个微小的 "fetch news" 作业会在每次 LLM 调用上膨胀工具 schema 提示。

## 完全跳过智能体：`wakeAgent`

如果您的 cron 作业附加了预检查脚本（通过 `script=`），脚本可以在运行时决定 Hermes 是否甚至应该调用智能体。发出最终 stdout 行，格式为：

```text
{"wakeAgent": false}
```

…然后 cron 完全跳过此 tick 的智能体运行。适用于频繁轮询（每 1-5 分钟），仅在状态实际更改时才需要唤醒 LLM —— 否则您会一遍又一遍地为零内容智能体轮次付费。

```python
# 预检查脚本
import json, sys
latest = fetch_latest_issue_count()
prev = read_state("issue_count")
if latest == prev:
    print(json.dumps({"wakeAgent": False}))   # 跳过此 tick
    sys.exit(0)
write_state("issue_count", latest)
print(json.dumps({"wakeAgent": True, "context": {"new_issues": latest - prev}}))
```

当省略 `wakeAgent` 时，默认值为 `true`（照常唤醒智能体）。

### 链式作业：`context_from`

Cron 作业可以通过在 `context_from` 中列出它们的名称（或 ID）来消费一个或多个其他作业的最新成功输出：

```text
cronjob(action="create", name="daily-digest",
        schedule="every day 7am",
        context_from=["ai-news-fetch", "github-prs-fetch"],
        prompt="Write the daily digest using the outputs above.")
```

引用作业的最新完成输出作为此运行的上下文注入到提示上方。每个上游条目必须是有效的作业 ID 或名称（请参阅 `cronjob action="list"`）。注意：链接读取*最近完成的*输出 —— 它不会等待在同一 tick 中运行的上游作业。

## 作业存储

作业存储在 `~/.hermes/cron/jobs.json` 中。作业运行的输出保存到 `~/.hermes/cron/output/{job_id}/{timestamp}.md`。

作业可能将 `model` 和 `provider` 存储为 `null`。当这些字段被省略时，Hermes 在执行时从全局配置解析它们。它们仅在设置每作业覆盖时出现在作业记录中。

存储使用原子文件写入，因此中断的写入不会在部分写入的作业文件后留下。

## 自包含提示仍然重要

:::warning 重要
Cron 作业在完全全新的智能体会话中运行。提示必须包含智能体需要的所有内容，除非已由附加技能提供。
:::

**不好：** `"Check on that server issue"`

**好：** `"SSH into server 192.168.1.100 as user 'deploy', check if nginx is running with 'systemctl status nginx', and verify https://example.com returns HTTP 200."`

## 安全

定时任务提示在创建和更新时被扫描提示注入和凭证外泄模式。包含不可见 Unicode 技巧、SSH 后门尝试或明显秘密外泄有效负载的提示被阻止。
