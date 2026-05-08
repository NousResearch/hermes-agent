---
sidebar_position: 11
title: "Automate Anything with Cron"
description: "使用 Hermes cron 的实际自动化模式 — 监控、报告、流水线和多技能工作流"
---

# 使用 Cron 实现任何自动化

[每日简报机器人教程](/docs/guides/daily-briefing-bot)涵盖了基础知识。本指南更进一步 — 介绍五个你可以根据自己的工作流调整的实际自动化模式。

有关完整的功能参考，请参阅[定时任务 (Cron)](/docs/user-guide/features/cron)。

:::info 核心概念
Cron 任务在全新的 agent 会话中运行，不保留当前聊天的记忆。提示词必须**完全自包含** — 包含 agent 需要知道的所有内容。
:::

:::tip 不需要 LLM？使用 no-agent 模式。
对于定期看门狗任务（脚本已经生成你想要发送的确切消息，如内存警报、磁盘警报、CI ping、心跳），可以通过[纯脚本 cron 任务](/docs/guides/cron-script-only)完全跳过 LLM。零 token 消耗，相同的调度器。你可以在聊天中让 Hermes 为你设置 — `cronjob` 工具知道何时选择 `no_agent=True` 并为你编写脚本。
:::

---

## 模式 1：网站变更监控

监控 URL 的变更，仅在有差异时接收通知。

`script` 参数是这里的秘密武器。Python 脚本在每次执行前运行，其 stdout 成为 agent 的上下文。脚本处理机械性工作（获取、差异对比）；agent 处理推理（这个变更是否有意义？）。

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

:::tip [SILENT] 技巧
当 agent 的最终响应包含 `[SILENT]` 时，交付会被抑制。这意味着你只在实际发生某些事情时收到通知 — 安静时段不会收到垃圾通知。
:::

---

## 模式 2：周报

从多个来源编译信息，生成格式化的摘要。这每周运行一次，发送到你的主页频道。

```bash
/cron add "0 9 * * 1" "Generate a weekly report covering:

1. Search the web for the top 5 AI news stories from the past week
2. Search GitHub for trending repositories in the 'machine-learning' topic
3. Check Hacker News for the most discussed AI/ML posts

Format as a clean summary with sections for each source. Include links.
Keep it under 500 words — highlight only what matters." --name "Weekly AI digest" --deliver telegram
```

从 CLI 执行：

```bash
hermes cron create "0 9 * * 1" \
  "Generate a weekly report covering the top AI news, trending ML GitHub repos, and most-discussed HN posts. Format with sections, include links, keep under 500 words." \
  --name "Weekly AI digest" \
  --deliver telegram
```

`0 9 * * 1` 是标准的 cron 表达式：每周一上午 9:00。

---

## 模式 3：GitHub 仓库监视器

监控仓库的新 issue、PR 或 release。

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

:::warning 自包含的提示词
注意提示词如何包含确切的 `gh` 命令。cron agent 没有先前运行或你偏好的记忆 — 把所有内容都写清楚。
:::

---

## 模式 4：数据收集管道

定期抓取数据，保存到文件，并检测随时间变化的趋势。此模式结合了脚本（用于收集）和 agent（用于分析）。

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

脚本处理机械性收集；agent 添加推理层。

---

## 模式 5：多技能工作流

将技能链接在一起，用于复杂的定时任务。技能在提示词执行之前按顺序加载。

```bash
# 使用 arxiv 技能查找论文，然后使用 obsidian 技能保存笔记
/cron add "0 8 * * *" "Search arXiv for the 3 most interesting papers on 'language model reasoning' from the past day. For each paper, create an Obsidian note with the title, authors, abstract summary, and key contribution." \
  --skill arxiv \
  --skill obsidian \
  --name "Paper digest"
```

直接从工具调用：

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

技能按顺序加载 — 先 `arxiv`（教会 agent 如何搜索论文），然后 `obsidian`（教会如何写入笔记）。提示词将它们联系在一起。

---

## 管理你的任务

```bash
# 列出所有活动任务
/cron list

# 立即触发任务（用于测试）
/cron run <job_id>

# 暂停任务而不删除
/cron pause <job_id>

# 编辑运行任务的调度或提示词
/cron edit <job_id> --schedule "every 4h"
/cron edit <job_id> --prompt "Updated task description"

# 添加或移除现有任务的技能
/cron edit <job_id> --skill arxiv --skill obsidian
/cron edit <job_id> --clear-skills

# 永久移除任务
/cron remove <job_id>
```

---

## 交付目标

`--deliver` 标志控制结果的去向：

| 目标 | 示例 | 使用场景 |
|--------|---------|----------|
| `origin` | `--deliver origin` | 创建任务的同一聊天（默认） |
| `local` | `--deliver local` | 仅保存到本地文件 |
| `telegram` | `--deliver telegram` | 你的 Telegram 主页频道 |
| `discord` | `--deliver discord` | 你的 Discord 主页频道 |
| `slack` | `--deliver slack` | 你的 Slack 主页频道 |
| 特定聊天 | `--deliver telegram:-1001234567890` | 特定的 Telegram 群组 |
| 线程化 | `--deliver telegram:-1001234567890:17585` | 特定的 Telegram 主题线程 |

---

## 提示

**使提示词自包含。** Cron 任务中的 agent 没有你的对话记忆。在提示词中直接包含 URL、仓库名称、格式偏好和交付说明。

**大量使用 `[SILENT]`.** 对于监控任务，始终包含类似 "if nothing changed, respond with `[SILENT]`" 的说明。这可以防止通知噪音。

**使用脚本进行数据收集。** `script` 参数让 Python 脚本处理枯燥的部分（HTTP 请求、文件 I/O、状态跟踪）。agent 只看到脚本的 stdout 并对其进行推理。这比让 agent 自己获取更便宜、更可靠。

**使用 `/cron run` 测试。** 在等待调度触发之前，使用 `/cron run <job_id>` 立即执行并验证输出是否正确。

**调度表达式。** 支持的格式：相对延迟（`30m`）、间隔（`every 2h`）、标准 cron 表达式（`0 9 * * *`）和 ISO 时间戳（`2025-06-15T09:00:00`）。不支持像 `daily at 9am` 这样的自然语言 — 请使用 `0 9 * * *`。

---

*有关完整的 cron 参考 — 所有参数、边界情况和内部机制 — 请参阅[定时任务 (Cron)](/docs/user-guide/features/cron)。*
