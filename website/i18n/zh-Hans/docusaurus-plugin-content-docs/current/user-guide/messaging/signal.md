---
sidebar_position: 6
title: "Signal"
description: "通过 signal-cli 守护进程将 Hermes Agent 设置为 Signal 机器人"
---

# Signal 配置

Hermes 通过 [signal-cli](https://github.com/AsamK/signal-cli) 连接到 Signal，通常经由 [bbernhard/signal-cli-rest-api](https://github.com/bbernhard/signal-cli-rest-api) 容器（`MODE=json-rpc`）。适配器通过 WebSocket（`/v1/receive/<number>`）实时接收消息，并通过 REST API（`/v2/send`、表情回应、附件及相关路由）发送响应。

Signal 是隐私保护最完善的主流即时通讯工具——默认端对端加密、开源协议、极少的元数据收集。这使其非常适合对安全性要求较高的 Agent 工作流。

:::info Python 依赖
Signal 适配器使用 `httpx`（已是 Hermes 的核心依赖）发送出站 REST 请求，并使用 `aiohttp`（包含在 `[messaging]` extra 中）运行入站 WebSocket 监听器。你需要在 signal-cli-rest-api v0.99+ 之后运行 signal-cli ≥ **0.14.5**。
:::

---

## 前提条件

- **signal-cli** — 基于 Java 的 Signal 客户端（[GitHub](https://github.com/AsamK/signal-cli)）
- **Java 17+** 运行时 — signal-cli 所需
- **一个已安装 Signal 的手机号**（用于作为辅助设备关联）

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
signal-cli **不在** apt 或 snap 仓库中。上述 Linux 安装方式直接从 [GitHub releases](https://github.com/AsamK/signal-cli/releases) 下载。
:::

---

## 第一步：关联你的 Signal 账号

signal-cli 作为**关联设备**运行——类似 WhatsApp Web，但用于 Signal。你的手机仍是主设备。

```bash
# 生成关联 URI（显示二维码或链接）
signal-cli link -n "HermesAgent"
```

1. 在手机上打开 **Signal**
2. 进入 **设置 → 关联设备**
3. 点击 **关联新设备**
4. 扫描二维码或输入 URI

---

## 第二步：启动 signal-cli-rest-api

推荐使用 `MODE=json-rpc` 的 Docker 镜像：

```bash
docker run -d --name signal-api \
  -p 127.0.0.1:8080:8080 \
  -e MODE=json-rpc \
  -v $HOME/.local/share/signal-cli:/home/.local/share/signal-cli \
  bbernhard/signal-cli-rest-api:latest
```

:::caution signal-cli 版本
请使用 **signal-cli 0.14.5 或更新版本**。较旧的版本（包括 0.14.3）可能无法解密入站消息（`getServerGuid must not be null`）。在 Apple Silicon 上，由于原生 Linux arm64 0.14.5 构建并不总是可用，你可能需要设置 `platform: linux/amd64`。
:::

在较旧的环境中也可以运行原生守护进程（`signal-cli --account +1234567890 daemon --http 127.0.0.1:8080`），但当前 Hermes 面向 signal-cli-rest-api v0.99+。

验证是否正在运行：

```bash
curl http://127.0.0.1:8080/v1/about
# 应返回 signal-cli-rest-api 和 signal-cli 的版本信息
```

---

## 第三步：配置 Hermes

最简单的方式：

```bash
hermes gateway setup
```

从平台菜单中选择 **Signal**。向导将：

1. 检查 signal-cli 是否已安装
2. 提示输入 HTTP URL（默认：`http://127.0.0.1:8080`）
3. 测试与守护进程的连通性
4. 询问你的账号手机号
5. 配置允许的用户和访问策略

### 手动配置

在 `~/.hermes/.env` 中添加：

```bash
# 必填
SIGNAL_HTTP_URL=http://127.0.0.1:8080
SIGNAL_ACCOUNT=+1234567890

# 安全设置（推荐）
SIGNAL_ALLOWED_USERS=+1234567890,+0987654321    # 逗号分隔的 E.164 号码或 UUID

# 可选
SIGNAL_GROUP_ALLOWED_USERS=groupId1,groupId2     # 启用群组（省略则禁用，* 表示全部）
SIGNAL_HOME_CHANNEL=+1234567890                  # cron 任务的默认投递目标
```

然后启动 gateway：

```bash
hermes gateway              # 前台运行
hermes gateway install      # 安装为用户服务
sudo hermes gateway install --system   # 仅 Linux：开机自启系统服务
```

---

## 访问控制

### 私信访问

私信访问遵循与其他 Hermes 平台相同的模式：

1. **已设置 `SIGNAL_ALLOWED_USERS`** → 仅允许这些用户发送消息
2. **未设置白名单** → 未知用户会收到私信配对码（通过 `hermes pairing approve signal CODE` 审批）
3. **`SIGNAL_ALLOW_ALL_USERS=true`** → 任何人均可发送消息（谨慎使用）

### 群组访问

群组访问由 `SIGNAL_GROUP_ALLOWED_USERS` 环境变量控制：

| 配置 | 行为 |
|------|------|
| 未设置（默认） | 忽略所有群组消息，机器人仅响应私信。 |
| 设置群组 ID | 仅监听列出的群组（如 `groupId1,groupId2`）。 |
| 设置为 `*` | 机器人在其所在的任意群组中均会响应。 |

### 群组中的 @提及门控

设置 `SIGNAL_REQUIRE_MENTION=true`（或在平台配置中设置 `require_mention: true`）后，机器人只回应 @提及了其账号的群组消息。检查同时接受提及元数据中的电话号码和 Signal UUID（ACI）——现代 Signal 客户端往往只发送 UUID 而省略号码。

**关联设备账号：** 当 Hermes 以关联设备方式接入你自己的 Signal 账号时，机器人与你共用同一个电话号码，没有独立身份，因此无法被真正 @提及。此时可以设置一个纯文本触发词：

```bash
SIGNAL_TRIGGER_ALIAS=hermes    # 不区分大小写；任何包含该词的群组消息都会通过提及门控
```

触发词与真正的 @提及可以同时使用——满足其一即可。

---

## 功能特性

### 附件

适配器支持双向收发媒体文件。

**接收**（用户 → Agent）：

- **图片** — PNG、JPEG、GIF、WebP（通过魔数自动检测）
- **音频** — MP3、OGG、WAV、M4A（若已配置 Whisper，语音消息将自动转录）
- **文档** — PDF、ZIP 及其他文件类型

**发送**（Agent → 用户）：

Agent 可通过响应中的 `MEDIA:` 标签发送媒体文件，支持以下投递方式：

- **图片** — `send_multiple_images` 和 `send_image_file` 将 PNG、JPEG、GIF、WebP 作为原生 Signal 附件发送
- **语音** — `send_voice` 将音频文件（OGG、MP3、WAV、M4A、AAC）作为附件发送
- **视频** — `send_video` 发送 MP4 视频文件
- **文档** — `send_document` 发送任意文件类型（PDF、ZIP 等）

所有外发媒体均通过 Signal 标准附件 API 处理。与某些平台不同，Signal 在协议层面不区分语音消息和文件附件。

附件大小限制：**100 MB**（双向）。

:::warning
**Signal 服务器会对附件上传进行速率限制**，适配器使用调度器批量发送多张图片，每批最多 32 张，并按照 Signal 服务器策略限速上传。
:::

### 原生格式、引用回复与表情回应

Signal 消息以**原生格式**渲染，而非显示原始 markdown 字符。适配器将 markdown（`**粗体**`、`*斜体*`、`` `代码` ``、`~~删除线~~`、`||剧透||`、标题）转换为 Signal `bodyRanges`，使文本在接收方客户端以真实样式显示，而非可见的 `**` 或 `` ` `` 字符。

**引用回复。** 当 Hermes 回复某条特定消息时，会发送原生引用回复——与 Signal 用户使用"回复"功能时看到的 UI 效果相同。对于响应入站消息而生成的回复，此功能自动生效。

**表情回应。** Agent 可通过标准 reaction API 对消息添加表情回应；回应会以 emoji 形式显示在被引用消息上，而非额外的文字。

以上功能无需额外配置——在近期的 signal-cli 版本中默认启用。若你的 `signal-cli` 版本过旧，Hermes 会回退到纯文本投递，并记录一次性警告日志。

### 长消息

Signal 单条消息上限为 **8,000 字符**。Hermes 会自动将更长的回复拆分为带编号的分段（`(1/3)`、`(2/3)`……）发送，而不是截断内容。此行为覆盖所有投递路径——实时对话回复、cron 任务投递、`hermes send` 以及 MCP `send_message` 调用——且原生格式（加粗、斜体、代码、剧透）在分段边界处保持不变。

### 正在输入指示器

机器人在处理消息时会发送正在输入指示器，每 8 秒刷新一次。

### 手机号脱敏

所有手机号在日志中自动脱敏：
- `+15551234567` → `+155****4567`
- 适用于 Hermes gateway 日志和全局脱敏系统

### 给自己发消息（单号码配置）

如果你将 signal-cli 作为自己手机号的**关联辅助设备**运行（而非单独的机器人号码），可以通过 Signal 的"给自己发消息"功能与 Hermes 交互。

只需从手机向自己发送消息——signal-cli 会接收到该消息，Hermes 在同一会话中响应。

**工作原理：**
- "给自己发消息"以 `syncMessage.sentMessage` 信封形式到达
- 适配器检测到这些消息是发给机器人自身账号的，并将其作为普通入站消息处理
- 回声保护（已发时间戳追踪）防止无限循环——机器人自身的回复会被自动过滤

**无需额外配置。** 只要 `SIGNAL_ACCOUNT` 与你的手机号匹配，此功能自动生效。

### 健康监控

传输层的恢复由 WebSocket 协议自身处理：适配器每 30 秒发送一次 WS ping，若未收到 pong 则断开连接并触发重连（指数退避：2s → 60s）。

另外，一个存活监控器每 30 秒观察一次接收路径，并在 300 秒内没有任何应用层活动时记录日志。仅有应用层空闲**不会**触发重连——安静的会话本来就可能长时间无消息，而心跳机制已经能检测到失效的传输连接。监控器还会记录最近一条已分发入站消息的时间戳（在 `gateway_state.json` 中以 `last_inbound_message_at` 暴露），供外部看门狗检测"已连接但静默不投递"的守护进程。

---

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| 配置时提示 **"Cannot reach signal-cli"** | 确保 signal-cli 守护进程正在运行：`signal-cli --account +YOUR_NUMBER daemon --http 127.0.0.1:8080` |
| **消息未收到** | 检查 `SIGNAL_ALLOWED_USERS` 是否包含发送方号码（E.164 格式，带 `+` 前缀） |
| **"signal-cli not found on PATH"** | 安装 signal-cli 并确保其在 PATH 中，或使用 Docker |
| **连接持续断开** | 检查 signal-cli 日志中的错误信息，确保已安装 Java 17+。 |
| **群组消息被忽略** | 使用具体群组 ID 配置 `SIGNAL_GROUP_ALLOWED_USERS`，或设为 `*` 允许所有群组。 |
| **机器人对所有人无响应** | 配置 `SIGNAL_ALLOWED_USERS`，使用私信配对，或通过 gateway 策略显式允许所有用户（如需更广泛的访问权限）。 |
| **消息重复** | 确保只有一个 signal-cli 实例在监听你的手机号 |

---

## 安全

:::warning
**务必配置访问控制。** 机器人默认具有终端访问权限。若未设置 `SIGNAL_ALLOWED_USERS` 或私信配对，gateway 会拒绝所有入站消息作为安全措施。
:::

- 手机号在所有日志输出中均已脱敏
- 使用私信配对或显式白名单安全地引导新用户
- 除非明确需要群组支持，否则保持群组禁用状态，或仅将受信任的群组加入白名单
- Signal 的端对端加密保护传输中的消息内容
- `~/.local/share/signal-cli/` 中的 signal-cli 会话数据包含账号凭据——请像保护密码一样保护它

---

## 环境变量参考

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `SIGNAL_HTTP_URL` | 是 | — | signal-cli HTTP 端点 |
| `SIGNAL_ACCOUNT` | 是 | — | 机器人手机号（E.164） |
| `SIGNAL_ALLOWED_USERS` | 否 | — | 逗号分隔的手机号/UUID |
| `SIGNAL_GROUP_ALLOWED_USERS` | 否 | — | 要监听的群组 ID，或 `*` 表示全部（省略则禁用群组） |
| `SIGNAL_ALLOW_ALL_USERS` | 否 | `false` | 允许任意用户交互（跳过白名单） |
| `SIGNAL_HOME_CHANNEL` | 否 | — | cron 任务的默认投递目标 |