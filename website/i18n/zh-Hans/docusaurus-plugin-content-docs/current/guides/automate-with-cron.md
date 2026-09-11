---
sidebar_position: 11
title: "用 Cron 自动化一切"
description: "使用 Hermes cron 的真实自动化模式——监控、报告、数据管道与多技能工作流"
---

# 用 Cron 自动化一切

[每日简报机器人教程](/guides/daily-briefing-bot)涵盖了基础内容。本指南更进一步——五种真实的自动化模式，可直接改造用于你自己的工作流。

完整功能参考请见 [定时任务（Cron）](/user-guide/features/cron)。

:::info 核心概念
Cron 任务在全新的 agent 会话中运行，不保留当前对话的任何记忆。Prompt（提示词）必须**完全自包含**——把 agent 需要知道的一切都写进去。
:::

:::tip 不需要 LLM？你有两种零 token 方案。
- **循环看门狗**：脚本本身已能生成精确消息（内存告警、磁盘告警、心跳）时，使用 [纯脚本 cron 任务](/guides/cron-script-only)。相同的调度器，无需 LLM。你可以在对话中让 Hermes 帮你设置——`cronjob` 工具知道何时选择 `no_agent=True` 并为你编写脚本。
- **已在运行的脚本发起的一次性通知**（CI 步骤、post-commit hook、部署脚本、外部调度的监控）：使用 [`hermes send`](/guides/pipe-script-output) 将 stdout 或文件直接推送到 Telegram / Discord / Slack 等，无需设置 cron 条目。
:::

---

## 模式一：网站变更监控

监视某个 URL 的变化，仅在内容发生变化时发送通知。

`script` 参数是这里的秘密武器。每次执行前会先运行一个 Python 脚本，其 stdout 作为上下文传给 agent。脚本负责机械性工作（抓取、对比差异）；agent 负责推理（这个变化是否值得关注？）。

创建监控脚本：

```bash
mkdir -p ~/.hermes/scripts
```

```python title="~/.hermes/scripts/watch-site.py"
import hashlib, json, os, urllib.request

URL = "https://example.com/pricing"
STATE_FILE = os.path.expanduser("~/.hermes/scripts/.watch-site-state.json")

# Fetch current content
req = urllib.request.Request(URL, headers={"User-Agent": "Hermes-Monitor/1.0"})
content = urllib.request.urlopen(req, timeout=30).read().decode()
current_hash = hashlib.sha256(content.encode()).hexdigest()

# Load previous state
prev_hash = None
if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        prev_hash = json.load(f).get("hash")

# Save current state
with open(STATE_FILE, "w") as f:
    json.dump({"hash": current_hash, "url": URL}, f)

# Output for the agent
if prev_hash and prev_hash != current_hash:
    print(f"CHANGE DETECTED on {URL}")
    print(f"Previous hash: {prev_hash}")
    print(f"Current hash: {current_hash}")
    print(f"\nCurrent content (first 2000 chars):\n{content[:2000]}")
else:
    print("NO_CHANGE")
```

设置 cron 任务：

```bash
/cron add "every 1h" "If the script output says CHANGE DETECTED, summarize what changed on the page and why it might matter. If it says NO_CHANGE, respond with just [SILENT]." --script ~/.hermes/scripts/watch-site.py --name "Pricing monitor" --deliver telegram
```

:::tip `[SILENT]` 技巧
当 agent 的最终响应包含 `[SILENT]` 时，投递会被抑制。这意味着只有在真正发生变化时你才会收到通知——安静时段不会产生垃圾消息。
:::

---

## 模式二：每周报告

从多个来源汇总信息，生成格式化摘要。每周运行一次，投递到你的主频道。

```bash
/cron add "0 9 * * 1" "Generate a weekly report covering:

1. Search the web for the top 5 AI news stories from the past week
2. Search GitHub for trending repositories in the 'machine-learning' topic
3. Check Hacker News for the most discussed AI/ML posts

Format as a clean summary with sections for each source. Include links.
Keep it under 500 words — highlight only what matters." --name "Weekly AI digest" --deliver telegram
```

通过 CLI：

```bash
hermes cron create "0 9 * * 1" \
  "Generate a weekly report covering the top AI news, trending ML GitHub repos, and most-discussed HN posts. Format with sections, include links, keep under 500 words." \
  --name "Weekly AI digest" \
  --deliver telegram
```

`0 9 * * 1` 是标准 cron 表达式：每周一上午 9:00。

---

## 模式三：GitHub 仓库监控

监控某个仓库的新 issue、PR 或 release。

```bash
/cron add "every 6h" "Check the GitHub repository NousResearch/hermes-agent for:
- New issues opened in the last 6 hours
- New PRs opened or merged in the last 6 hours
- Any new releases

Use the terminal to run gh commands:
  gh issue list --repo NousResearch/hermes-agent --state open --json number,title,author,createdAt --limit 10
  gh pr list --repo NousResearch/hermes-agent --state all --json number,title,author,createdAt,mergedAt --limit 10

Filter to only items from the last 6 hours. If nothing new, respond with [SILENT].
Otherwise, provide a concise summary of the activity." --name "Repo watcher" --deliver discord
```

:::warning 自包含的 Prompt
注意 prompt 中包含了精确的 `gh` 命令。cron agent 不记得之前的运行记录或你的偏好——把所有内容都明确写出来。
:::

---

## 模式四：数据采集管道

定期抓取数据、保存到文件，并随时间检测趋势。此模式将脚本（用于采集）与 agent（用于分析）结合使用。

```python title="~/.hermes/scripts/collect-prices.py"
import json, os, urllib.request
from datetime import datetime

DATA_DIR = os.path.expanduser("~/.hermes/data/prices")
os.makedirs(DATA_DIR, exist_ok=True)

# Fetch current data (example: crypto prices)
url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd"
data = json.loads(urllib.request.urlopen(url, timeout=30).read())

# Append to history file
entry = {"timestamp": datetime.now().isoformat(), "prices": data}
history_file = os.path.join(DATA_DIR, "history.jsonl")
with open(history_file, "a") as f:
    f.write(json.dumps(entry) + "\n")

# Load recent history for analysis
lines = open(history_file).readlines()
recent = [json.loads(l) for l in lines[-24:]]  # Last 24 data points

# Output for the agent
print(f"Current: BTC=${data['bitcoin']['usd']}, ETH=${data['ethereum']['usd']}")
print(f"Data points collected: {len(lines)} total, showing last {len(recent)}")
print(f"\nRecent history:")
for r in recent[-6:]:
    print(f"  {r['timestamp']}: BTC=${r['prices']['bitcoin']['usd']}, ETH=${r['prices']['ethereum']['usd']}")
```

```bash
/cron add "every 1h" "Analyze the price data from the script output. Report:
1. Current prices
2. Trend direction over the last 6 data points (up/down/flat)
3. Any notable movements (>5% change)

If prices are flat and nothing notable, respond with [SILENT].
If there's a significant move, explain what happened." \
  --script ~/.hermes/scripts/collect-prices.py \
  --name "Price tracker" \
  --deliver telegram
```

脚本负责机械性的数据采集；agent 在此之上添加推理层。

---

## 模式五：多技能工作流

将多个 skill（技能）串联起来，完成复杂的定时任务。Skill 按顺序加载，然后执行 prompt。

```bash
# 使用 arxiv skill 查找论文，再用 obsidian skill 保存笔记
/cron add "0 8 * * *" "Search arXiv for the 3 most interesting papers on 'language model reasoning' from the past day. For each paper, create an Obsidian note with the title, authors, abstract summary, and key contribution." \
  --skill arxiv \
  --skill obsidian \
  --name "Paper digest"
```

直接通过工具调用：

```python
cronjob(
    action="create",
    skills=["arxiv", "obsidian"],
    prompt="Search arXiv for papers on 'language model reasoning' from the past day. Save the top 3 as Obsidian notes.",
    schedule="0 8 * * *",
    name="Paper digest",
    deliver="local"
)
```

Skill 按顺序加载——先加载 `arxiv`（教 agent 如何搜索论文），再加载 `obsidian`（教 agent 如何写笔记）。Prompt 将二者串联起来。

---

## 管理你的任务

```bash
# 列出所有活跃任务
/cron list

# 立即触发某个任务（用于测试）
/cron run <job_id>

# 暂停任务而不删除
/cron pause <job_id>

# 编辑运行中任务的调度或 prompt
/cron edit <job_id> --schedule "every 4h"
/cron edit <job_id> --prompt "Updated task description"

# 为现有任务添加或移除 skill
/cron edit <job_id> --skill arxiv --skill obsidian
/cron edit <job_id> --clear-skills

# 永久删除任务
/cron remove <job_id>
```

---

## 投递目标

`--deliver` 标志控制结果发送到哪里：

| 目标 | 示例 | 使用场景 |
|--------|---------|----------|
| `origin` | `--deliver origin` | 创建该任务的对话（默认） |
| `local` | `--deliver local` | 仅保存到本地文件 |
| `telegram` | `--deliver telegram` | 你的 Telegram 主频道 |
| `discord` | `--deliver discord` | 你的 Discord 主频道 |
| `slack` | `--deliver slack` | 你的 Slack 主频道 |
| 指定对话 | `--deliver telegram:-1001234567890` | 特定 Telegram 群组 |
| 线程投递 | `--deliver telegram:-1001234567890:17585` | 特定 Telegram 话题线程 |

---

## 使用技巧

**让 prompt 完全自包含。** Cron 任务中的 agent 不记得你的任何对话。把 URL、仓库名、格式偏好和投递说明直接写进 prompt。

**大量使用 `[SILENT]`。** 对于监控类任务，始终加上类似"如果没有变化，回复 `[SILENT]`"的指令，防止通知噪音。

**用脚本做数据采集。** `script` 参数让 Python 脚本处理枯燥的部分（HTTP 请求、文件 I/O、状态追踪）。Agent 只看到脚本的 stdout，并对其进行推理。这比让 agent 自己抓取更省钱、更可靠。

**用 `/cron run` 测试。** 不要等调度触发，使用 `/cron run <job_id>` 立即执行，验证输出是否符合预期。

**调度表达式。** 支持的格式：相对延迟（`30m`）、间隔（`every 2h`）、标准 cron 表达式（`0 9 * * *`）、ISO 时间戳（`2025-06-15T09:00:00`）。不支持自然语言如 `daily at 9am`——请改用 `0 9 * * *`。

---

*完整的 cron 参考——所有参数、边界情况和内部机制——请见 [定时任务（Cron）](/user-guide/features/cron)。*