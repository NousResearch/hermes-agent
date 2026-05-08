---
sidebar_position: 1
title: "消息网关"
description: "通过 Telegram、Discord、Slack、WhatsApp、Signal、SMS、Email、Home Assistant、Mattermost、Matrix、DingTalk、Feishu/Lark、WeCom、Weixin、BlueBubbles (iMessage)、QQ、Yuanbao、Microsoft Teams、Webhooks 或通过 API 服务器的任何 OpenAI 兼容前端与 Hermes 聊天 — 架构和设置概述"
---

# 消息网关

通过 Telegram、Discord、Slack、WhatsApp、Signal、SMS、Email、Home Assistant、Mattermost、Matrix、DingTalk、Feishu/Lark、WeCom、Weixin、BlueBubbles (iMessage)、QQ、Yuanbao、Microsoft Teams 或你的浏览器与 Hermes 聊天。gateway 是一个单一的后台进程，连接到你配置的所有平台，处理会话、运行 cron 作业并递送语音消息。

有关完整的语音功能集 —— 包括 CLI 麦克风模式、消息中的语音回复和 Discord 语音频道对话 —— 请参阅 [Voice Mode](/docs/user-guide/features/voice-mode) 和 [Use Voice Mode with Hermes](/docs/guides/use-voice-mode-with-hermes)。

## 平台对比

| 平台 | 语音 | 图片 | 文件 | 主题 | 反应 | 输入中 | 流式 |
|----------|:-----:|:------:|:-----:|:-------:|:---------:|:------:|:---------:|
| Telegram | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| Discord | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Slack | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Google Chat | — | ✅ | ✅ | ✅ | — | ✅ | — |
| WhatsApp | — | ✅ | ✅ | — | — | ✅ | ✅ |
| Signal | — | ✅ | ✅ | — | — | ✅ | ✅ |
| SMS | — | — | — | — | — | — | — |
| Email | — | ✅ | ✅ | ✅ | — | — | — |
| Home Assistant | — | — | — | — | — | — | — |
| Mattermost | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| Matrix | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| DingTalk | — | ✅ | ✅ | — | ✅ | — | ✅ |
| Feishu/Lark | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| WeCom | ✅ | ✅ | ✅ | — | — | ✅ | ✅ |
| WeCom Callback | — | — | — | — | — | — | — |
| Weixin | ✅ | ✅ | ✅ | — | — | ✅ | ✅ |
| BlueBubbles | — | ✅ | ✅ | — | ✅ | ✅ | — |
| QQ | ✅ | ✅ | ✅ | — | — | ✅ | — |
| Yuanbao | ✅ | ✅ | ✅ | — | — | ✅ | ✅ |
| Microsoft Teams | — | ✅ | — | ✅ | — | ✅ | — |

**语音** = TTS 音频回复和/或语音消息转录。**图片** = 发送/接收图片。**文件** = 发送/接收文件附件。**主题** = 线程对话。**反应** = 消息上的表情反应。**输入中** = 处理时显示输入指示器。**流式** = 通过编辑进行渐进式消息更新。

## 架构

```mermaid
flowchart TB
    subgraph Gateway["Hermes Gateway"]
        subgraph Adapters["Platform adapters"]
            tg[Telegram]
            dc[Discord]
            wa[WhatsApp]
            sl[Slack]
            gc[Google Chat]
            sig[Signal]
            sms[SMS]
            em[Email]
            ha[Home Assistant]
            mm[Mattermost]
            mx[Matrix]
            dt[DingTalk]
    fs[Feishu/Lark]
    wc[WeCom]
    wcb[WeCom Callback]
    wx[Weixin]
    bb[BlueBubbles]
    qq[QQ]
    yb[Yuanbao]
    ms[Microsoft Teams]
    api["API Server<br/>(OpenAI-compatible)"]
    wh[Webhooks]
        end

        store["Session store<br/>per chat"]
        agent["AIAgent<br/>run_agent.py"]
        cron["Cron scheduler<br/>ticks every 60s"]
    end

    tg --> store
    dc --> store
    wa --> store
    sl --> store
    gc --> store
    sig --> store
    sms --> store
    em --> store
    ha --> store
    mm --> store
    mx --> store
    dt --> store
    fs --> store
    wc --> store
    wcb --> store
    wx --> store
    bb --> store
    qq --> store
    yb --> store
    ms --> store
    api --> store
    wh --> store
    store --> agent
    cron --> store
```

每个平台适配器接收消息，通过每个聊天的会话存储路由它们，并将它们分派给 AIAgent 进行处理。gateway 还运行 cron 调度器，每 60 秒触发一次以执行任何到期的作业。

## 快速设置

配置消息平台最简单的方法是交互式向导：

```bash
hermes gateway setup        # 交互式设置所有消息平台
```

这将引导你使用方向键选择配置每个平台，显示哪些平台已经配置，并在完成时提供启动/重启 gateway 的选项。

## Gateway 命令

```bash
hermes gateway              # 在前台运行
hermes gateway setup        # 交互式配置消息平台
hermes gateway install      # 安装为用户服务（Linux）/launchd 服务（macOS）
sudo hermes gateway install --system   # 仅 Linux：安装为启动时系统服务
hermes gateway start        # 启动默认服务
hermes gateway stop         # 停止默认服务
hermes gateway status       # 检查默认服务状态
hermes gateway status --system         # 仅 Linux：显式检查系统服务状态
```

## 聊天命令（在消息中）

| 命令 | 描述 |
|---------|-------------|
| `/new` 或 `/reset` | 开始新的对话 |
| `/model [provider:model]` | 显示或更改模型（支持 `provider:model` 语法） |
| `/personality [name]` | 设置人格 |
| `/retry` | 重试上一条消息 |
| `/undo` | 删除上一次交换 |
| `/status` | 显示会话信息 |
| `/stop` | 停止正在运行的 agent |
| `/approve` | 批准待处理危险命令 |
| `/deny` | 拒绝待处理危险命令 |
| `/sethome` | 将此聊天设置为主频道 |
| `/compress` | 手动压缩对话上下文 |
| `/title [name]` | 设置或显示会话标题 |
| `/resume [name]` | 恢复之前命名的会话 |
| `/usage` | 显示此会话的 token 使用量 |
| `/insights [days]` | 显示使用洞察和分析 |
| `/reasoning [level\|show\|hide]` | 更改推理努力或切换推理显示 |
| `/voice [on\|off\|tts\|join\|leave\|status]` | 控制消息语音回复和 Discord 语音频道行为 |
| `/rollback [number]` | 列出或恢复文件系统检查点 |
| `/background <prompt>` | 在单独的后台会话中运行提示 |
| `/reload-mcp` | 从配置重新加载 MCP 服务器 |
| `/update` | 将 Hermes Agent 更新到最新版本 |
| `/help` | 显示可用命令 |
| `/<skill-name>` | 调用任何已安装的技能 |

## 会话管理

### 会话持久性

会话在重置之前跨消息持久化。agent 记住你的对话上下文。

### 重置策略

会话根据可配置的策略重置：

| 策略 | 默认 | 描述 |
|--------|---------|-------------|
| 每日 | 4:00 AM | 每天在特定小时重置 |
| 空闲 | 1440 分钟 | N 分钟不活动后重置 |
| 两者 | （组合） | 以先触发者为准 |

在 `~/.hermes/gateway.json` 中配置每个平台的覆盖：

```json
{
  "reset_by_platform": {
    "telegram": { "mode": "idle", "idle_minutes": 240 },
    "discord": { "mode": "idle", "idle_minutes": 60 }
  }
}
```

## 安全

**默认情况下，gateway 拒绝所有不在白名单中或未通过 DM 配对的用户。** 这对于具有终端访问权限的机器人来说是安全默认值。

```bash
# 限制为特定用户（推荐）：
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=123456789012345678
SIGNAL_ALLOWED_USERS=+155****4567,+155****6543
SMS_ALLOWED_USERS=+155****4567,+155****6543
EMAIL_ALLOWED_USERS=trusted@example.com,colleague@work.com
MATTERMOST_ALLOWED_USERS=3uo8dkh1p7g1mfk49ear5fzs5c
MATRIX_ALLOWED_USERS=@alice:matrix.org
DINGTALK_ALLOWED_USERS=user-id-1
FEISHU_ALLOWED_USERS=ou_xxxxxxxx,ou_yyyyyyyy
WECOM_ALLOWED_USERS=user-id-1,user-id-2
WECOM_CALLBACK_ALLOWED_USERS=user-id-1,user-id-2
TEAMS_ALLOWED_USERS=aad-object-id-1,aad-object-id-2

# 或者允许
GATEWAY_ALLOWED_USERS=123456789,987654321

# 或者明确允许所有用户（对于具有终端访问权限的机器人不推荐）：
GATEWAY_ALLOW_ALL_USERS=true
```

### DM 配对（白名单的替代方案）

无需手动配置用户 ID，未知用户在 DM 机器人时会收到一次性配对码：

```bash
# 用户看到："配对码：XKGH5N7P"
# 你用以下命令批准他们：
hermes pairing approve telegram XKGH5N7P

# 其他配对命令：
hermes pairing list          # 查看待处理和已批准的用户
hermes pairing revoke telegram 123456789  # 移除访问权限
```

配对码在 1 小时后过期，受速率限制，并使用加密随机性。

## 中断 Agent

在 agent 正在工作时发送任何消息以中断它。关键行为：

- **运行中的终端命令会立即被终止**（SIGTERM，然后在 1 秒后 SIGKILL）
- **工具调用被取消** — 只有当前正在执行的工具会运行，其余的会被跳过
- **多条消息被合并** — 在中断期间发送的消息被连接成一个提示
- **`/stop` 命令** — 中断而不排队后续消息

### 队列 vs 中断 vs 引导（忙碌输入模式）

默认情况下，向忙碌的 agent 发送消息会中断它。另有两种模式可用：

- `queue` — 后续消息等待，并在当前任务完成后作为下一轮运行。
- `steer` — 后续消息通过 `/steer` 注入当前运行，在下一个工具调用后到达 agent。没有中断，没有新轮。如果 agent 尚未开始，则回退到 `queue` 行为。

```yaml
display:
  busy_input_mode: steer   # 或 queue，或 interrupt（默认）
  busy_ack_enabled: true   # 设置为 false 以完全禁止 ⚡/⏳/⏩ 聊天回复
```

第一次在任何平台上向忙碌的 agent 发送消息时，Hermes 会追加一行 busy-ack 说明该旋钮的提醒（`"💡 首次提示 — …"`）。提醒每次安装触发一次 —— `onboarding.seen.busy_input_prompt` 下的标志闩锁它。删除该键可以再次看到提示。

如果你觉得 busy-ack 很嘈杂 —— 特别是使用语音输入或快速连续的消息 —— 设置 `display.busy_ack_enabled: false`。你的输入仍然正常排队/引导/中断，只有聊天回复被静音。

## 工具进度通知

控制在 `~/.hermes/config.yaml` 中显示多少工具活动：

```yaml
display:
  tool_progress: all    # off | new | all | verbose
  tool_progress_command: false  # 设置为 true 以在消息中启用 /verbose
```

启用后，机器人在工作时发送状态消息：

```text
💻 `ls -la`...
🔍 web_search...
📄 web_extract...
🐍 execute_code...
```

## 后台会话

在单独的后台会话中运行提示，以便 agent 独立工作，而你的主聊天保持响应：

```
/background 检查集群中的所有服务器并报告任何已关闭的服务器
```

Hermes 立即确认：

```
🔄 后台任务已启动："检查集群中的所有服务器..."
   任务 ID：bg_143022_a1b2c3
```

### 工作原理

每个 `/background` 提示生成一个**单独的 agent 实例**，异步运行：

- **隔离会话** — 后台 agent 有自己的会话，有自己的对话历史。它不知道你当前聊天上下文，只接收你提供的提示。
- **相同配置** — 从当前 gateway 设置继承你的模型、提供商、工具集、推理设置和提供商路由。
- **非阻塞** — 你的主聊天保持完全交互。在它工作时发送消息、运行其他命令或启动更多后台任务。
- **结果递送** — 当任务完成时，结果被发送回**发出命令的同一聊天或频道**，前面带有"✅ 后台任务完成"。如果失败，你会看到"❌ 后台任务失败"以及错误。

### 后台进程通知

当运行后台会话的 agent 使用 `terminal(background=true)` 启动长时间运行的进程（服务器、构建等）时，gateway 可以将状态更新推送到你的聊天。使用 `~/.hermes/config.yaml` 中的 `display.background_process_notifications` 进行控制：

```yaml
display:
  background_process_notifications: all    # all | result | error | off
```

| 模式 | 你收到什么 |
|------|-----------------|
| `all` | 运行输出更新**以及**最终完成消息（默认） |
| `result` | 仅最终完成消息（无论退出代码如何） |
| `error` | 仅在退出代码非零时的最终消息 |
| `off` | 完全没有进程监视器消息 |

你也可以通过环境变量设置：

```bash
HERMES_BACKGROUND_NOTIFICATIONS=result
```

### 用例

- **服务器监控** — "/background 检查所有服务的健康状况，如果有任何问题请提醒我"
- **长时间构建** — "/background 构建并部署暂存环境"，而你继续聊天
- **研究任务** — "/background 研究竞争对手的定价并以表格形式总结"
- **文件操作** — "/background 将 ~/Downloads 中的照片按日期组织到文件夹中"

:::tip
消息平台上的后台任务是"触发后忘记" —— 你不需要等待或检查它们。任务完成时结果会自动到达同一聊天。
:::

## 服务管理

### Linux（systemd）

```bash
hermes gateway install               # 安装为用户服务
hermes gateway start                 # 启动服务
hermes gateway stop                  # 停止服务
hermes gateway status                # 检查状态
journalctl --user -u hermes-gateway -f  # 查看日志

# 启用驻留（退出后保持运行）
sudo loginctl enable-linger $USER

# 或者安装仍作为你的用户运行的启动时系统服务
sudo hermes gateway install --system
sudo hermes gateway start --system
sudo hermes gateway status --system
journalctl -u hermes-gateway -f
```

在笔记本电脑和开发机器上使用用户服务。在 VPS 或无头主机上使用系统服务，这些主机应该在启动时恢复，而不依赖 systemd linger。

除非你真的打算，否则不要同时保留用户和系统 gateway 单元。Hermes 会在检测到两者时发出警告，因为 start/stop/status 行为会变得模糊。

:::info 多个安装
如果你在同一台机器上运行多个 Hermes 安装（使用不同的 `HERMES_HOME` 目录），每个都有自己的 systemd 服务名称。默认的 `~/.hermes` 使用 `hermes-gateway`；其他安装使用 `hermes-gateway-<hash>`。`hermes gateway` 命令会自动针对你当前 `HERMES_HOME` 的正确服务。
:::

### macOS（launchd）

```bash
hermes gateway install               # 安装为 launchd agent
hermes gateway start                 # 启动服务
hermes gateway stop                  # 停止服务
hermes gateway status                # 检查状态
tail -f ~/.hermes/logs/gateway.log   # 查看日志
```

生成的 plist 位于 `~/Library/LaunchAgents/ai.hermes.gateway.plist`。它包含三个环境变量：

- **PATH** — 安装时你的完整 shell PATH，预先附加了 venv `bin/` 和 `node_modules/.bin`。这确保用户安装的工具（Node.js、ffmpeg 等）可用于 gateway 子进程，如 WhatsApp 桥接器。
- **VIRTUAL_ENV** — 指向 Python 虚拟环境，以便工具可以正确解析包。
- **HERMES_HOME** — 将 gateway 限定到你的 Hermes 安装。

:::tip 安装后 PATH 更改
launchd plist 是静态的 —— 如果在设置 gateway 后安装了新工具（例如通过 nvm 安装的新 Node.js 版本，或通过 Homebrew 安装的 ffmpeg），请再次运行 `hermes gateway install` 以捕获更新的 PATH。gateway 会检测到过时的 plist 并自动重新加载。
:::

:::info 多个安装
与 Linux systemd 服务一样，每个 `HERMES_HOME` 目录都有自己的 launchd 标签。默认的 `~/.hermes` 使用 `ai.hermes.gateway`；其他安装使用 `ai.hermes.gateway-<suffix>`。
:::

## 平台特定工具集

每个平台都有自己的工具集：

| 平台 | 工具集 | 功能 |
|----------|---------|--------------|
| CLI | `hermes-cli` | 完全访问 |
| Telegram | `hermes-telegram` | 包括终端的完整工具 |
| Discord | `hermes-discord` | 包括终端的完整工具 |
| WhatsApp | `hermes-whatsapp` | 包括终端的完整工具 |
| Slack | `hermes-slack` | 包括终端的完整工具 |
| Google Chat | `hermes-google-chat` | 包括终端的完整工具 |
| Signal | `hermes-signal` | 包括终端的完整工具 |
| SMS | `hermes-sms` | 包括终端的完整工具 |
| Email | `hermes-email` | 包括终端的完整工具 |
| Home Assistant | `hermes-homeassistant` | 完整工具 + HA 设备控制（ha_list_entities、ha_get_state、ha_call_service、ha_list_services） |
| Mattermost | `hermes-mattermost` | 包括终端的完整工具 |
| Matrix | `hermes-matrix` | 包括终端的完整工具 |
| DingTalk | `hermes-dingtalk` | 包括终端的完整工具 |
| Feishu/Lark | `hermes-feishu` | 包括终端的完整工具 |
| WeCom | `hermes-wecom` | 包括终端的完整工具 |
| WeCom Callback | `hermes-wecom-callback` | 包括终端的完整工具 |
| Weixin | `hermes-weixin` | 包括终端的完整工具 |
| BlueBubbles | `hermes-bluebubbles` | 包括终端的完整工具 |
| QQBot | `hermes-qqbot` | 包括终端的完整工具 |
| Yuanbao | `hermes-yuanbao` | 包括终端的完整工具 |
| Microsoft Teams | `hermes-teams` | 包括终端的完整工具 |
| API Server | `hermes`（默认） | 包括终端的完整工具 |
| Webhooks | `hermes-webhook` | 包括终端的完整工具 |

## 下一步

- [Telegram 设置](telegram.md)
- [Discord 设置](discord.md)
- [Slack 设置](slack.md)
- [Google Chat 设置](google_chat.md)
- [WhatsApp 设置](whatsapp.md)
- [Signal 设置](signal.md)
- [SMS 设置（Twilio）](sms.md)
- [Email 设置](email.md)
- [Home Assistant 集成](homeassistant.md)
- [Mattermost 设置](mattermost.md)
- [Matrix 设置](matrix.md)
- [DingTalk 设置](dingtalk.md)
- [Feishu/Lark 设置](feishu.md)
- [WeCom 设置](wecom.md)
- [WeCom Callback 设置](wecom-callback.md)
- [Weixin 设置](weixin.md)
- [BlueBubbles 设置（iMessage）](bluebubbles.md)
- [QQBot 设置](qqbot.md)
- [Yuanbao 设置](yuanbao.md)
- [Microsoft Teams 设置](teams.md)
- [Open WebUI + API Server](open-webui.md)
- [Webhooks](webhooks.md)
