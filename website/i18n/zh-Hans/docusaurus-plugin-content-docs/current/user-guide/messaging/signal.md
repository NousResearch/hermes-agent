---
sidebar_position: 6
title: "Signal"
description: "通过 signal-cli 守护进程将 Hermes Agent 设置为 Signal 消息机器人"
---

# Signal 设置

Hermes 通过以 HTTP 模式运行的 [signal-cli](https://github.com/AsamK/signal-cli) 守护进程连接到 Signal。适配器通过 SSE（服务器发送事件）实时流式传输消息，并通过 JSON-RPC 发送响应。

Signal 是最受关注的主流消息应用 —— 默认端到端加密、开源协议、最小化元数据收集。这使其成为安全敏感 agent 工作流的理想选择。

:::info 无新的 Python 依赖
Signal 适配器使用 `httpx`（已经是核心 Hermes 依赖）进行所有通信。不需要额外的 Python 包。你只需要在外部安装 signal-cli。
:::

---

## 前提条件

- **signal-cli** — 基于 Java 的 Signal 客户端（[GitHub](https://github.com/AsamK/signal-cli)）
- **Java 17+** 运行时 — signal-cli 需要
- **已安装 Signal 的电话号码**（用于作为辅助设备链接）

### 安装 signal-cli

```bash
# macOS
brew install signal-cli

# Linux（下载最新版本）
VERSION=$(curl -Ls -o /dev/null -w %{url_effective} \
  https://github.com/AsamK/signal-cli/releases/latest | sed 's/^.*\/v//')
curl -L -O "https://github.com/AsamK/signal-cli/releases/download/v${VERSION}/signal-cli-${VERSION}.tar.gz"
sudo tar xf "signal-cli-${VERSION}.tar.gz" -C /opt
sudo ln -sf "/opt/signal-cli-${VERSION}/bin/signal-cli" /usr/local/bin/
```

:::caution
signal-cli **不在** apt 或 snap 仓库中。上面 Linux 的安装直接从 [GitHub releases](https://github.com/AsamK/signal-cli/releases) 下载。
:::

---

## 步骤 1：链接你的 Signal 账号

signal-cli 作为**链接设备**工作 —— 像 WhatsApp Web，但用于 Signal。你的手机是主设备。

```bash
# 生成链接 URI（显示二维码或链接）
signal-cli link -n "HermesAgent"
```

1. 在你的手机上打开 **Signal**
2. 转到 **Settings → Linked Devices**
3. 点击 **Link New Device**
4. 扫描二维码或输入 URI

---

## 步骤 2：启动 signal-cli 守护进程

```bash
# 将 +1234567890 替换为你的 Signal 电话号码（E.164 格式）
signal-cli --account +1234567890 daemon --http 127.0.0.1:8080
```

:::tip
在后台保持运行。你可以使用 `systemd`、`tmux`、`screen`，或将其作为服务运行。
:::

验证它正在运行：

```bash
curl http://127.0.0.1:8080/api/v1/check
# 应返回：{"versions":{"signal-cli":...}}
```

---

## 步骤 3：配置 Hermes

最简单的方法：

```bash
hermes gateway setup
```

从平台菜单中选择 **Signal**。向导将：

1. 检查 signal-cli 是否已安装
2. 提示输入 HTTP URL（默认：`http://127.0.0.1:8080`）
3. 测试到守护进程的连接
4. 询问你的账号电话号码
5. 配置允许的用户和访问策略

### 手动配置

添加到 `~/.hermes/.env`：

```bash
# 必需
SIGNAL_HTTP_URL=http://127.0.0.1:8080
SIGNAL_ACCOUNT=+1234567890

# 安全（推荐）
SIGNAL_ALLOWED_USERS=+1234567890,+0987654321    # 逗号分隔的 E.164 号码或 UUID

# 可选
SIGNAL_GROUP_ALLOWED_USERS=groupId1,groupId2     # 启用群组（省略以禁用，* 表示所有）
SIGNAL_HOME_CHANNEL=+1234567890                  # cron 作业的默认递送目标
```

然后启动 gateway：

```bash
hermes gateway              # 前台
hermes gateway install      # 安装为用户服务
sudo hermes gateway install --system   # 仅 Linux：启动时系统服务
```

---

## 访问控制

### DM 访问

DM 访问遵循与所有其他 Hermes 平台相同的模式：

1. **设置了 `SIGNAL_ALLOWED_USERS`** → 只有这些用户可以发消息
2. **未设置白名单** → 未知用户获得 DM 配对码（通过 `hermes pairing approve signal CODE` 批准）
3. **`SIGNAL_ALLOW_ALL_USERS=true`** → 任何人都可以发消息（谨慎使用）

### 群组访问

群组访问由 `SIGNAL_GROUP_ALLOWED_USERS` 环境变量控制：

| 配置 | 行为 |
|---------------|----------|
| 未设置（默认） | 所有群组消息被忽略。机器人只回复 DM。 |
| 设置了群组 ID | 仅监视列出的群组（例如 `groupId1,groupId2`）。 |
| 设置为 `*` | 机器人在它所属的任何群组中回复。 |

---

## 功能

### 附件

适配器支持双向发送和接收媒体。

**入站**（用户 → agent）：

- **图片** — PNG、JPEG、GIF、WebP（通过魔数自动检测）
- **音频** — MP3、OGG、WAV、M4A（如果配置了 Whisper，则转录语音消息）
- **文档** — PDF、ZIP 和其他文件类型

**出站**（agent → 用户）：

agent 可以通过响应中的 `MEDIA:` 标签发送媒体文件。支持以下递送方法：

- **图片** — `send_multiple_images` 和 `send_image_file` 将 PNG、JPEG、GIF、WebP 作为原生 Signal 附件发送
- **语音** — `send_voice` 将音频文件（OGG、MP3、WAV、M4A、AAC）作为附件发送
- **视频** — `send_video` 发送 MP4 视频文件
- **文档** — `send_document` 发送任何文件类型（PDF、ZIP 等）

所有出站媒体通过 Signal 的标准附件 API 发送。与某些平台不同，Signal 在协议级别不区分语音消息和文件附件。

附件大小限制：**100 MB**（双向）。

:::warning
**Signal 服务器会对附件上传进行速率限制**，适配器使用调度器进行多次图片发送，将图片分批为 32 个一组，并限制上传以匹配 Signal 服务器策略。
:::

### 原生格式、回复引用和反应

Signal 消息使用**原生格式**渲染，而不是字面 markdown 字符。适配器将 markdown（`**bold**`、`*italic*`、`` `code` ``、`~~strike~~`、`||spoiler||`、标题）转换为 Signal `bodyRanges`，以便文本在收件人的客户端上显示为真实样式，而不是可见的 `**` / `` ` `` 字符。

**回复引用。** 当 Hermes 回复特定消息时，它现在发布一个引用原始消息的原生回复 —— 与 Signal 用户使用"回复"时看到的相同 UI 功能。对于作为对入站消息的响应而生成的回复，这是自动的。

**反应。** agent 可以通过标准反应 API 对消息做出反应；反应在 Signal 中作为所引用消息上的 emoji 反应显示，而不是作为额外文本。

所有这些都不需要额外配置 —— 它在最近的 signal-cli 构建中默认随附。如果你的 `signal-cli` 版本太旧，Hermes 会回退到纯文本递送并记录一次性警告。

### 输入指示器

机器人在处理消息时发送输入指示器，每 8 秒刷新一次。

### 电话号码编辑

所有电话号码在日志中自动编辑：
- `+15551234567` → `+155****4567`
- 这适用于 Hermes gateway 日志和全局编辑系统

### 给自己留言（单号码设置）

如果你将 signal-cli 作为自己的电话号码上的**链接辅助设备**运行（而不是单独的机器人号码），你可以通过 Signal 的"给自己留言"功能与 Hermes 交互。

只需从你的手机给自己发消息 —— signal-cli 会收到它，Hermes 会在同一对话中回复。

**工作原理：**
- "给自己留言"消息作为 `syncMessage.sentMessage` 信封到达
- 适配器检测这些何时发送到机器人自己的账号，并将它们作为常规入站消息处理
- 回声保护（发送时间戳跟踪）防止无限循环 —— 机器人自己的回复会自动过滤

**无需额外配置。** 只要 `SIGNAL_ACCOUNT` 匹配你的电话号码，这就会自动工作。

### 健康监控

适配器监控 SSE 连接并在以下情况下自动重连：
- 连接断开（指数退避：2s → 60s）
- 120 秒内未检测到活动（ping signal-cli 以验证）

---

## 故障排除

| 问题 | 解决方案 |
|---------|----------|
| **设置期间"无法到达 signal-cli"** | 确保 signal-cli 守护进程正在运行：`signal-cli --account +YOUR_NUMBER daemon --http 127.0.0.1:8080` |
| **未收到消息** | 检查 `SIGNAL_ALLOWED_USERS` 包含发送者的号码，格式为 E.164（带 `+` 前缀） |
| **"signal-cli 未在 PATH 上找到"** | 安装 signal-cli 并确保它在你的 PATH 中，或使用 Docker |
| **连接不断断开** | 检查 signal-cli 日志中的错误。确保安装了 Java 17+。 |
| **群组消息被忽略** | 使用特定群组 ID 配置 `SIGNAL_GROUP_ALLOWED_USERS`，或 `*` 以允许所有群组。 |
| **机器人无人响应** | 配置 `SIGNAL_ALLOWED_USERS`、使用 DM 配对，或如果你想要更广泛的访问权限则通过 gateway 策略明确允许所有用户。 |
| **重复消息** | 确保只有一个 signal-cli 实例在你的电话号码上监听 |

---

## 安全

:::warning
**始终配置访问控制。** 机器人默认具有终端访问权限。如果没有 `SIGNAL_ALLOWED_USERS` 或 DM 配对，gateway 作为安全措施会默认拒绝所有入站消息。
:::

- 电话号码在所有日志输出中被编辑
- 使用 DM 配对或明确白名单以安全地加入新用户
- 除非你特别需要群组支持，否则保持群组禁用，或者只白名单你信任的群组
- Signal 的端到端加密保护传输中的消息内容
- `~/.local/share/signal-cli/` 中的 signal-cli 会话数据包含账号凭证 —— 像密码一样保护它

---

## 环境变量参考

| 变量 | 必需 | 默认 | 描述 |
|----------|----------|---------|-------------|
| `SIGNAL_HTTP_URL` | 是 | — | signal-cli HTTP 端点 |
| `SIGNAL_ACCOUNT` | 是 | — | 机器人电话号码（E.164） |
| `SIGNAL_ALLOWED_USERS` | 否 | — | 逗号分隔的电话号码/UUID |
| `SIGNAL_GROUP_ALLOWED_USERS` | 否 | — | 要监视的群组 ID，或 `*` 表示所有（省略以禁用群组） |
| `SIGNAL_ALLOW_ALL_USERS` | 否 | `false` | 允许任何用户交互（跳过白名单） |
| `SIGNAL_HOME_CHANNEL` | 否 | — | cron 作业/通知递送的默认目标 |
