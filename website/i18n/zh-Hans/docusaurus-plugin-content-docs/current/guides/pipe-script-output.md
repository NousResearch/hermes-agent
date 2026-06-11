---
sidebar_position: 12
title: "将脚本输出推送到消息平台"
description: "使用 `hermes send` 将任意 shell 脚本、cron 任务、CI hook 或监控守护进程的文本发送到 Telegram、Discord、Slack、Signal 等平台。"
---

# 将脚本输出推送到消息平台

`hermes send` 是一个轻量、可脚本化的 CLI，能将消息推送到 Hermes 已配置的任意消息平台。可以把它理解为跨平台的通知专用 `curl`——无需运行中的 gateway，无需 LLM，也无需在每个脚本里重复粘贴 bot token。

适用场景：

- 系统监控（内存、磁盘、GPU 温度、长时任务完成通知）
- CI/CD 通知（部署完成、测试失败）
- 需要将结果推送给你的 cron 脚本
- 从终端发送一次性消息
- 将任意工具的输出管道到任意平台（`make | hermes send --to slack:#builds`）

该命令复用 `hermes gateway` 已有的凭据和平台适配器，无需维护第二套配置。

---

## 快速开始

```bash
# 向某平台的默认频道发送纯文本
hermes send --to telegram "deploy finished"

# 将任意命令的 stdout 通过管道传入
echo "RAM 92%" | hermes send --to telegram:-1001234567890

# 发送文件
hermes send --to discord:#ops --file /tmp/report.md

# 附加主题/标题行
hermes send --to slack:#eng --subject "[CI] build.log" --file build.log

# 指定线程目标（Telegram 话题、Discord 线程）
hermes send --to telegram:-1001234567890:17585 "threaded reply"

# 列出所有已配置的目标
hermes send --list

# 按平台过滤
hermes send --list telegram
```

---

## 参数参考

| 标志 | 说明 |
|------|-------------|
| `-t, --to TARGET` | 目标地址。参见[目标格式](#target-formats)。 |
| `message`（位置参数） | 消息文本。省略时从 `--file` 或 stdin 读取。 |
| `-f, --file PATH` | 从文件读取消息体。`--file -` 强制从 stdin 读取。 |
| `-s, --subject LINE` | 在消息体前添加标题/主题行。 |
| `-l, --list` | 列出可用目标。可选位置参数用于按平台过滤。 |
| `-q, --quiet` | 成功时不输出到 stdout（仅返回退出码——适合脚本使用）。 |
| `--json` | 输出发送结果的原始 JSON。 |
| `-h, --help` | 显示内置帮助文本。 |

### 目标格式 {#target-formats}

| 格式 | 示例 | 含义 |
|--------|---------|---------|
| `platform` | `telegram` | 发送到该平台配置的默认频道 |
| `platform:chat_id` | `telegram:-1001234567890` | 指定数字 chat / 群组 / 用户 |
| `platform:chat_id:thread_id` | `telegram:-1001234567890:17585` | 指定线程或 Telegram 论坛话题 |
| `platform:#channel` | `discord:#ops` | 易读的频道名称（通过频道目录解析） |
| `platform:+E164` | `signal:+15551234567` | 以电话号码寻址的平台：Signal、SMS、WhatsApp |

Hermes 附带适配器的所有平台均可作为目标：
`telegram`、`discord`、`slack`、`signal`、`sms`、`whatsapp`、`matrix`、
`mattermost`、`feishu`、`dingtalk`、`wecom`、`weixin`、`email` 等。

### 退出码

| 码 | 含义 |
|------|---------|
| `0` | 发送（或列出）成功 |
| `1` | 平台层面投递失败（认证、权限、网络） |
| `2` | 用法 / 参数 / 配置错误 |

退出码遵循标准 Unix 惯例，脚本可以像处理 `curl` 或 `grep` 一样对其进行分支判断。

---

## 消息体解析顺序

`hermes send` 按以下顺序解析消息体：

1. **位置参数** — `hermes send --to telegram "hi"`
2. **`--file PATH`** — `hermes send --to telegram --file msg.txt`
3. **管道 stdin** — `echo hi | hermes send --to telegram`

当 stdin 是 TTY（无管道）时，Hermes **不会**等待输入——你会收到明确的用法错误提示。这可以防止脚本在意外省略消息体时挂起。

---

## 实际使用示例

### 监控：内存 / 磁盘告警

用一行简洁的代码替换 watchdog 脚本中的 `curl https://api.telegram.org/...` 调用：

```bash
#!/usr/bin/env bash
ram_pct=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$ram_pct" -ge 85 ]; then
  hermes send --to telegram --subject "⚠ MEMORY WARNING" \
    "RAM ${ram_pct}% on $(hostname)"
fi
```

由于 `hermes send` 复用你的 Hermes 配置，同一脚本可在任何安装了 Hermes 的主机上运行——无需手动将 bot token 导出到每台机器的环境变量中。

:::tip 不要用 gateway 监控自身
对于可能在 gateway 本身出现问题时触发的 watchdog（OOM 告警、磁盘满告警），请继续使用最简单的 `curl` 调用，而非 `hermes send`。如果 Python 解释器因机器抖动无法加载，你仍然希望告警能发出去。
:::

### CI / CD：构建与测试结果

```bash
# 在 .github/workflows/deploy.yml 或任意 CI 脚本中
if ./scripts/deploy.sh; then
  hermes send --to slack:#deploys "✅ ${CI_COMMIT_SHA:0:7} deployed"
else
  tail -n 100 deploy.log | hermes send \
    --to slack:#deploys --subject "❌ deploy failed"
  exit 1
fi
```

### Cron：每日报告

```bash
# Crontab 条目
0 9 * * * /usr/local/bin/generate-metrics.sh \
  | /home/me/.hermes/bin/hermes send \
      --to telegram --subject "Daily metrics $(date +%Y-%m-%d)"
```

### 长时任务：完成后推送通知

```bash
./train.py --epochs 200 && \
  hermes send --to telegram "training done" || \
  hermes send --to telegram "training failed (exit $?)"
```

### 脚本中使用 `--json` 与 `--quiet`

```bash
# 投递失败时让脚本硬失败；成功时不污染日志
hermes send --to telegram --quiet "keepalive" || {
  echo "Telegram delivery failed" >&2
  exit 1
}

# 捕获消息 ID 以便后续编辑 / 回复线程
msg_id=$(hermes send --to discord:#ops --json "build started" \
  | jq -r .message_id)
```

---

## `hermes send` 需要 gateway 运行吗？

**通常不需要。** 对于所有基于 bot token 的平台——Telegram、Discord、Slack、Signal、SMS、WhatsApp Cloud API 等——`hermes send` 直接使用 `~/.hermes/.env` 和 `~/.hermes/config.yaml` 中的凭据调用平台的 REST 接口。它是一个独立的子进程，消息投递完成后即退出。

只有依赖持久适配器连接的**插件平台**才需要运行中的 gateway（例如，某个保持长连接 WebSocket 的自定义插件）。此时你会收到明确的错误提示，指引你启动 gateway；执行 `hermes gateway start` 后重试即可。

---

## 列出与发现目标

在向特定频道发送消息之前，可以查看可用目标：

```bash
# 列出所有已配置平台的所有目标
hermes send --list

# 仅列出 Telegram 目标
hermes send --list telegram

# 机器可读格式
hermes send --list --json
```

列表数据来源于 `~/.hermes/channel_directory.json`，gateway 运行期间每隔几分钟刷新一次。如果看到"尚未发现频道"，请先启动一次 gateway（`hermes gateway start`）以填充缓存。

易读名称（`discord:#ops`、`slack:#engineering`）在发送时通过该缓存解析，无需记忆数字 ID。

---

## 与其他方案的对比

| 方案 | 多平台 | 复用 Hermes 凭据 | 需要 gateway | 最适合 |
|----------|----------------|---------------------|---------------|----------|
| `hermes send` | ✅ | ✅ | 否（bot token） | 以下所有场景 |
| 对各平台直接 `curl` | 各自单独编写 | 手动管理 | 否 | 关键 watchdog |
| 带 `--deliver` 的 `cron` 任务 | ✅ | ✅ | 否 | 定时 agent 任务 |
| `send_message` agent 工具 | ✅ | ✅ | 否 | agent 循环内部 |

`hermes send` 有意保持最简接口。如果需要 agent 决定说什么，请在对话或 cron 任务中使用 `send_message` 工具。如果需要定时运行并生成 LLM 内容，请使用带 `deliver='telegram:...'` 的 `cronjob(action='create', prompt=...)`。如果只需要管道传输原始字符串，直接用 `hermes send`。

---

## 相关文档

- [用 Cron 自动化一切](/guides/automate-with-cron) —
  输出自动投递到任意平台的定时任务。
- [Gateway 内部机制](/developer-guide/gateway-internals) —
  `hermes send` 与 cron 投递共享的投递路由器。
- [消息平台配置](/user-guide/messaging/) —
  各平台的一次性配置说明。