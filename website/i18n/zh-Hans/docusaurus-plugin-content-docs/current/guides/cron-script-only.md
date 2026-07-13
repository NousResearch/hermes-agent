---
sidebar_position: 13
title: "纯脚本 Cron 任务（无 LLM）"
description: "完全跳过 LLM 的经典看门狗 cron 任务——脚本按计划运行，其 stdout 输出直接投递到你的消息平台。内存告警、磁盘告警、CI 通知、定期健康检查。"
---

# 纯脚本 Cron 任务

有时你已经清楚地知道要发送什么消息。你不需要 agent 来推理——你只需要一个脚本按计时器运行，并将其输出（如有）发送到 Telegram / Discord / Slack / Signal。

Hermes 将此称为**无 agent 模式**。这是去掉 LLM 的 cron 系统。

<!-- ascii-guard-ignore -->
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
<!-- ascii-guard-ignore-end -->

- **无 LLM 调用。** 零 token，零 agent 循环，零模型费用。
- **脚本即任务。** 由脚本决定是否告警。有输出 → 发送消息；无输出 → 静默执行。
- **Bash 或 Python。** `.sh` / `.bash` 文件在 `/bin/bash` 下运行；其他扩展名在当前 Python 解释器下运行。`~/.hermes/scripts/` 中的任何文件均可接受。
- **同一调度器。** 与 LLM 任务共存于 `cronjob` 中——暂停、恢复、列出、日志和投递目标的操作方式完全相同。

## 适用场景

以下情况使用无 agent 模式：

- **内存 / 磁盘 / GPU 看门狗。** 每 5 分钟运行一次，仅在超过阈值时告警。
- **CI hook（钩子）。** 部署完成 → 发送 commit SHA；构建失败 → 发送最后 100 行日志。
- **定期指标。** "每天上午 9 点的 Stripe 收入"——一次简单的 API 调用加格式化输出。
- **外部事件轮询。** 检查 API，在状态变化时告警。
- **心跳。** 每 N 分钟 ping 一次仪表板，证明主机存活。

当你需要 agent **决定**说什么时——总结长文档、从 feed 中挑选有趣条目、起草友好提醒——请使用普通的（LLM 驱动的）cron 任务。无 agent 路径适用于脚本的 stdout 本身就是消息内容的场景。

## 通过聊天创建

无 agent 模式的真正优势在于：agent 本身可以为你设置看门狗——无需编辑器、无需 shell、无需记忆 CLI 参数。你描述需求，Hermes 编写脚本、安排计划，并告知你何时触发。

### 示例对话

> **你：** 每 5 分钟检查一次，如果内存超过 85% 就在 telegram 通知我
>
> **Hermes：** *（写入 `~/.hermes/scripts/memory-watchdog.sh`，然后以 `no_agent=true` 调用 `cronjob(...)`）*
>
> 已设置。每 5 分钟运行一次，仅在内存超过 85% 时告警 Telegram。脚本：`memory-watchdog.sh`。任务 ID：`abc123`。

在底层，agent 进行了两次工具调用：

```python
# 1. 写入检查脚本
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

# 2. 调度任务 — no_agent=True 在每次触发时跳过 LLM
cronjob(
    action="create",
    schedule="every 5m",
    script="memory-watchdog.sh",
    no_agent=True,
    deliver="telegram",
    name="memory-watchdog",
)
```

此后每次触发都是免费的：调度器运行脚本，若 stdout 非空则将其发送到 Telegram，且从不调用模型。

### agent 为你决定的内容

当你的请求类似"当 X 时告警我"或"每 N 分钟检查 Y，如果 Z 则通知我"，Hermes 的 `cronjob` 工具描述会指示它在消息内容完全由脚本决定时使用 `no_agent=True`。当请求需要推理时（*"总结新 issue"*、*"挑选最有趣的头条"*、*"起草一条友好提醒"*），则回退到普通的 LLM 驱动路径。

你不必自己指定 `--no-agent`。只需描述你想要的行为即可。

### 通过聊天管理看门狗

agent 可以用与创建任务相同的方式暂停、恢复、编辑和删除任务：

> **你：** 今晚先停掉内存看门狗
>
> **Hermes：** *（调用 `cronjob(action='pause', job_id='abc123')`）*
>
> 已暂停。说"重新开启"或执行 `hermes cron resume abc123` 可恢复。

> **你：** 改成每 15 分钟一次
>
> **Hermes：** *（调用 `cronjob(action='update', job_id='abc123', schedule='every 15m')`）*

完整生命周期（创建 / 列出 / 更新 / 暂停 / 恢复 / 立即运行 / 删除）均可由 agent 操作，无需你学习任何 CLI 命令。

## 通过 CLI 创建

偏好 shell？CLI 路径用三条命令即可达到相同效果：

```bash
# 1. 编写脚本
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

# 2. 调度任务
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"

# 3. 验证
hermes cron list
hermes cron run <job_id>    # 触发一次以测试
```

就这些。无 prompt（提示词），无技能，无模型。


## 脚本输出与投递的映射关系

| 脚本行为 | 结果 |
|-----------------|--------|
| 退出码 0，stdout 非空 | stdout 原样投递 |
| 退出码 0，stdout 为空 | 静默执行——不投递 |
| 退出码 0，stdout 最后一行包含 `{"wakeAgent": false}` | 静默执行（与 LLM 任务共用的门控） |
| 非零退出码 | 投递错误告警（确保损坏的看门狗不会静默失败） |
| 脚本超时 | 投递错误告警 |

"空则静默"的行为是经典看门狗模式的关键：脚本可以每分钟运行一次，但只有在真正需要关注时，频道才会收到消息。

## 脚本规则

脚本必须位于 `~/.hermes/scripts/`。这在任务创建时和运行时均会强制检查——绝对路径、`~/` 展开以及路径穿越模式（`../`）均会被拒绝。该目录与 LLM 任务使用的预检脚本门控共享。

解释器由文件扩展名决定：

| 扩展名 | 解释器 |
|-----------|-------------|
| `.sh`、`.bash` | `/bin/bash` |
| 其他任意扩展名 | `sys.executable`（当前 Python） |

我们有意**不**遵循 `#!/...` shebang——保持解释器集合明确且精简，可减少调度器信任的攻击面。

## 计划语法

与所有其他 cron 任务相同：

```bash
hermes cron create "every 5m"        # 间隔
hermes cron create "every 2h"
hermes cron create "0 9 * * *"       # 标准 cron：每天上午 9 点
hermes cron create "30m"             # 单次：30 分钟后运行一次
```

完整语法请参阅 [cron 功能参考](/user-guide/features/cron)。

## 投递目标

`--deliver` 接受 gateway 已知的所有目标。常见形式：

```bash
--deliver telegram                       # 平台默认频道
--deliver telegram:-1001234567890        # 指定聊天
--deliver telegram:-1001234567890:17585  # 指定 Telegram 论坛话题
--deliver discord:#ops
--deliver slack:#engineering
--deliver signal:+15551234567
--deliver local                          # 仅保存到 ~/.hermes/cron/output/
```

对于使用 bot token 的平台（Telegram、Discord、Slack、Signal、SMS、WhatsApp），脚本运行时无需运行中的 gateway——工具直接使用 `~/.hermes/.env` / `~/.hermes/config.yaml` 中已有的凭据调用各平台的 REST 端点。

## 编辑与生命周期

```bash
hermes cron list                                    # 查看所有任务
hermes cron pause <job_id>                          # 停止触发，保留定义
hermes cron resume <job_id>
hermes cron edit <job_id> --schedule "every 10m"    # 调整频率
hermes cron edit <job_id> --agent                   # 切换为 LLM 模式
hermes cron edit <job_id> --no-agent --script …     # 切换回无 agent 模式
hermes cron remove <job_id>                         # 删除任务
```

所有适用于 LLM 任务的操作（暂停、恢复、手动触发、投递目标变更）同样适用于无 agent 任务。

## 实战示例：磁盘空间告警

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

当两个文件系统均低于 90% 时静默；当某个文件系统超出阈值时，每个超限文件系统触发一行告警。

## 与其他模式的对比

| 方式 | 运行内容 | 适用场景 |
|----------|-----------|-------------|
| `cronjob --no-agent`（本页） | 你的脚本，由 Hermes 调度 | 不需要推理的周期性看门狗 / 告警 / 指标 |
| `cronjob`（默认，LLM） | 带可选预检脚本的 agent | 消息内容需要对数据进行推理时 |
| OS cron + `curl` 到 [webhook 订阅](/user-guide/messaging/webhooks) | 你的脚本，由 OS 调度 | 当 Hermes 本身可能不健康时（即被监控对象） |

对于必须在 **gateway 宕机时也能触发**的关键系统健康看门狗，请使用 OS 级 cron 配合 `curl` 调用 Hermes webhook 订阅（或任何外部告警端点）——这些作为独立 OS 进程运行，不依赖 Hermes 是否在线。当被监控对象是外部系统时，in-gateway 调度器才是正确选择。

## 相关文档

- [用 Cron 自动化一切](/guides/automate-with-cron) — LLM 驱动的 cron 模式。
- [定时任务（Cron）参考](/user-guide/features/cron) — 完整计划语法、生命周期、投递路由。
- [Webhook 订阅](/user-guide/messaging/webhooks) — 供外部调度器使用的即发即忘 HTTP 入口。
- [Gateway 内部机制](/developer-guide/gateway-internals) — 投递路由器内部实现。