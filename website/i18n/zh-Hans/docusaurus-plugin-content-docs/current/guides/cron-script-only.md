---
sidebar_position: 13
title: "纯脚本 Cron 任务（无 LLM）"
description: "跳过 LLM 的经典看门狗 cron 任务 — 脚本按计划运行，其 stdout 被发送到你的消息平台。内存警报、磁盘警报、CI pings、定期健康检查。"
---

# 纯脚本 Cron 任务

有时你已经确切知道你想要发送什么消息。你不需要 agent 来推理 — 你只需要一个脚本按计时器运行，并且其输出（如果有）到达 Telegram / Discord / Slack / Signal。

Hermes 称之为**无 agent 模式**。它是没有 LLM 的 cron 系统。

```
 ┌──────────────────┐          ┌──────────────────┐
 │ scheduler tick   │  every   │ run script       │
 │ (every N minutes)│ ──────▶ │ (bash or python) │
 └──────────────────┘          └──────────────────┘
                                          │
                                          │ stdout
                                          ▼
                                 ┌──────────────────┐
                                 │ delivery router  │
                                 │ (telegram/disc…) │
                                 └──────────────────┘
```

- **无 LLM 调用。** 零 token，零 agent 循环，零模型费用。
- **脚本就是任务。** 脚本决定是否警报。发出输出 → 消息被发送。发出无内容 → 静默 tick。
- **Bash 或 Python。**`.sh` / `.bash` 文件在 `/bin/bash` 下运行；任何其他扩展名在当前 Python 解释器下运行。`~/.hermes/scripts/` 中的任何内容都被接受。
- **相同的调度器。** 与 LLM 任务一起存在于 `cronjob` 中 — 暂停、恢复、列表、日志和交付目标都以相同方式工作。

## 何时使用

在以下情况使用无 agent 模式：

- **内存 / 磁盘 / GPU 看门狗。** 每 5 分钟运行一次，仅在超过阈值时警报。
- **CI 钩子。** 部署完成 → 发布 commit SHA。构建失败 → 发送日志的最后 100 行。
- **定期指标。** "每天上午 9 点 Stripe 收入" 作为简单的 API 调用 + 美化打印。
- **外部事件轮询器。** 检查 API，在状态变更时警报。
- **心跳。** 每 N 分钟 ping 一次仪表板以证明主机存活。

当你需要 agent **决定**说什么时 — 总结长文档、从 feed 中挑选有趣的项目、起草人性化消息 — 请使用普通的（LLM 驱动的）cron 任务。无 agent 路径适用于脚本的 stdout 已经是消息的情况。

## 从聊天创建

无 agent 模式的真正优势在于 agent 本身可以为你设置看门狗 — 无需编辑器、无需 shell、无需记住 CLI 标志。你描述你想要什么，Hermes 编写脚本、安排它，并告诉它何时触发。

### 示例对话记录

> **你：** ping me on telegram if RAM is over 85% every 5 minutes
>
> **Hermes:** *(writes `~/.hermes/scripts/memory-watchdog.sh`, then calls `cronjob(...)` with `no_agent=true`)*
>
> Set up. Runs every 5 min, alerts Telegram only when RAM is over 85%. Script: `memory-watchdog.sh`. Job ID: `abc123`.

在底层，agent 进行两个工具调用：

```python
# 1. Write the check script
write_file(
    path="~/.hermes/scripts/memory-watchdog.sh",
    content='''#!/usr/bin/env bash
ram_pct=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$ram_pct" -ge 85 ]; then
  echo "RAM ${ram_pct}% on $(hostname)"
fi
# Empty stdout = silent tick; no message sent.
''',
)

# 2. Schedule it — no_agent=True skips the LLM on every tick
cronjob(
    action="create",
    schedule="every 5m",
    script="memory-watchdog.sh",
    no_agent=True,
    deliver="telegram",
    name="memory-watchdog",
)
```

从那时起，每个 tick 都是免费的：调度器运行脚本，如果非空则将其 stdout 管道传输到 Telegram，并且从不接触模型。

### Agent 为你决定什么

当你表述像 "alert me when X" 或 "every N minutes check Y and tell me if Z" 这样的请求时，Hermes 的 `cronjob` 工具描述告诉它在消息内容完全由脚本确定时使用 `no_agent=True`。当请求需要推理时（*"summarize the new issues"*、*"pick the most interesting headlines"*、*"draft a friendly reminder"*），它会回退到正常的 LLM 驱动路径。

你不需要自己指定 `--no-agent`。只需描述行为。

### 从聊天管理看门狗

Agent 可以暂停、恢复、编辑和移除任务，就像它创建它们一样：

> **你：** stop the memory watchdog for tonight
>
> **Hermes:** *(calls `cronjob(action='pause', job_id='abc123')`)*
>
> Paused. Resume with "turn it back on" or via `hermes cron resume abc123`.

> **你：** change it to every 15 minutes
>
> **Hermes:** *(calls `cronjob(action='update', job_id='abc123', schedule='every 15m')`)*

完整的生命周期（创建 / 列表 / 更新 / 暂停 / 恢复 / 立即运行 / 移除）对 agent 可用，无需你学习任何 CLI 命令。

## 从 CLI 创建

更喜欢 shell？CLI 路径用三个命令给你相同的结果：

```bash
# 1. Write your script
cat > ~/.hermes/scripts/memory-watchdog.sh <<'EOF'
#!/usr/bin/env bash
# Alert when RAM usage is over 85%. Silent otherwise.
RAM_PCT=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$RAM_PCT" -ge 85 ]; then
  echo "⚠ RAM ${RAM_PCT}% on $(hostname)"
fi
# Empty stdout = silent run; no message sent.
EOF
chmod +x ~/.hermes/scripts/memory-watchdog.sh

# 2. Schedule it
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"

# 3. Verify
hermes cron list
hermes cron run <job_id>    # fire it once to test
```

这就是全部。无提示词，无技能，无模型。

## 脚本输出如何映射到交付

| 脚本行为 | 结果 |
|-----------------|--------|
| 退出 0，非空 stdout | stdout 按原样交付 |
| 退出 0，空 stdout | 静默 tick — 无交付 |
| 退出 0，stdout 在最后一行包含 `{"wakeAgent": false}` | 静默 tick（与 LLM 任务共享网关） |
| 非零退出代码 | 交付错误警报（因此损坏的看门狗不会静默失败） |
| 脚本超时 | 交付错误警报 |

"空时静默"行为是经典看门狗模式的关键：脚本可以自由运行每一分钟，但频道仅在实际需要关注时才会看到消息。

## 脚本规则

脚本必须位于 `~/.hermes/scripts/`。这在任务创建时和运行时的 enforced — 绝对路径、`~/` 扩展和路径遍历模式（`../`）被拒绝。同一目录与 LLM 任务使用的前检查脚本网关共享。

解释器选择按文件扩展名：

| 扩展名 | 解释器 |
|-----------|-------------|
| `.sh`, `.bash` | `/bin/bash` |
| 其他任何 | `sys.executable`（当前 Python） |

我们有意不遵守 `#!/...` shebang — 保持解释器设置明确且小可以减小组度器信任的表面。

## 调度语法

与所有其他 cron 任务相同：

```bash
hermes cron create "every 5m"        # interval
hermes cron create "every 2h"
hermes cron create "0 9 * * *"       # standard cron: 9am daily
hermes cron create "30m"             # one-shot: run once in 30 minutes
```

有关完整语法，请参阅 [cron 功能参考](/docs/user-guide/features/cron)。

## 交付目标

`--deliver` 接受 gateway 知道的所有内容。一些常见形式：

```bash
--deliver telegram                       # platform home channel
--deliver telegram:-1001234567890        # specific chat
--deliver telegram:-1001234567890:17585  # specific Telegram forum topic
--deliver discord:#ops
--deliver slack:#engineering
--deliver signal:+15551234567
--deliver local                          # just save to ~/.hermes/cron/output/
```

对于 bot-token 平台（Telegram、Discord、Slack、Signal、SMS、WhatsApp），脚本运行时不需要运行 gateway — 工具使用 `~/.hermes/.env` / `~/.hermes/config.yaml` 中已有的凭据直接调用每个平台的 REST 端点。

## 编辑和生命周期

```bash
hermes cron list                                    # see all jobs
hermes cron pause <job_id>                          # stop firing, keep definition
hermes cron resume <job_id>
hermes cron edit <job_id> --schedule "every 10m"    # adjust cadence
hermes cron edit <job_id> --agent                   # flip to LLM mode
hermes cron edit <job_id> --no-agent --script …     # flip back
hermes cron remove <job_id>                         # delete it
```

在 LLM 任务上工作的所有内容（暂停、恢复、手动触发器、交付目标更改）在无 agent 任务上也工作。

## 工作示例：磁盘空间警报

```bash
cat > ~/.hermes/scripts/disk-alert.sh <<'EOF'
#!/usr/bin/env bash
# Alert when / or /home is over 90% full.
THRESHOLD=90
df -h / /home 2>/dev/null | awk -v t="$THRESHOLD" '
  NR > 1 && $5+0 >= t {
    printf "⚠ Disk %s full on %s\n", $5, $6
  }
'
EOF
chmod +x ~/.hermes/scripts/disk-alert.sh

hermes cron create "*/15 * * * *" \
  --no-agent \
  --script disk-alert.sh \
  --deliver telegram \
  --name "disk-alert"
```

当两个文件系统都低于 90% 时静默；当一个填满时，每个超过阈值的文件系统恰好触发一行。

## 与其他模式比较

| 方法 | 运行什么 | 何时使用 |
|----------|-----------|-------------|
| `hermes send`（单次） | 任何管道到它的 shell 命令 | 临时交付或作为外部调度器（systemd、launchd）的动作 |
| `cronjob --no-agent`（本页） | Hermes 调度器上的你的脚本 | 不需要推理的定期看门狗 / 警报 / 指标 |
| `cronjob`（默认，LLM） | 带有可选前检查脚本的 Agent | 消息内容需要针对数据进行推理时 |
| OS cron + `hermes send` | OS 调度器上的你的脚本 | 当 Hermes 可能不健康（你正在监控的事物）时 |

对于必须*即使 gateway 关闭时也触发*的关键系统健康看门狗，请继续使用 OS 级 cron + 普通的 `curl` 或 `hermes send` 调用 — 这些作为独立的 OS 进程运行，不依赖于 Hermes 启动。当被监控的事物是外部的时候，网关内调度器是正确的选择。

## 相关

- [使用 Cron 实现任何自动化](/docs/guides/automate-with-cron) — LLM 驱动的 cron 模式。
- [定时任务 (Cron) 参考](/docs/user-guide/features/cron) — 完整调度语法、生命周期、交付路由。
- [使用 `hermes send` 管道脚本输出](/docs/guides/pipe-script-output) — 临时脚本的单次对应项。
- [Gateway 内部](/docs/developer-guide/gateway-internals) — 交付路由器内部。
