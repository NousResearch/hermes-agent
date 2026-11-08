# BlueBubbles（iMessage）

通过 [BlueBubbles](https://bluebubbles.app/) 将 Hermes 连接至 Apple iMessage——这是一款免费、开源的 macOS 服务端，可将 iMessage 桥接至任意设备。

## 前提条件

- 一台**始终开机的 Mac**，运行 [BlueBubbles Server](https://bluebubbles.app/)
- 该 Mac 上的 Messages.app 已登录 Apple ID
- BlueBubbles Server v1.0.0+（webhook 需要此版本）
- Hermes 与 BlueBubbles 服务端之间的网络连通性

## 配置步骤

### 1. 安装 BlueBubbles Server

从 [bluebubbles.app](https://bluebubbles.app/) 下载并安装。完成设置向导——使用 Apple ID 登录，并配置连接方式（本地网络、Ngrok、Cloudflare 或动态 DNS）。

### 2. 获取服务端 URL 和密码

在 BlueBubbles Server → **Settings → API** 中，记录：
- **Server URL**（例如 `http://192.168.1.10:1234`）
- **Server Password**

### 3. 配置 Hermes

运行设置向导：

```bash
hermes gateway setup
```

选择 **BlueBubbles (iMessage)** 并输入服务端 URL 和密码。

或直接在 `~/.hermes/.env` 中设置环境变量：

```bash
BLUEBUBBLES_SERVER_URL=http://192.168.1.10:1234
BLUEBUBBLES_PASSWORD=your-server-password
```

### 4. 授权用户

选择以下任一方式：

**DM 配对（推荐）：**
当有人向你的 iMessage 发送消息时，Hermes 会自动向其发送配对码。使用以下命令批准：
```bash
hermes pairing approve bluebubbles <CODE>
```
使用 `hermes pairing list` 查看待处理的配对码和已授权用户。

**预授权特定用户**（在 `~/.hermes/.env` 中）：
```bash
BLUEBUBBLES_ALLOWED_USERS=user@icloud.com,+15551234567
```

**开放访问**（在 `~/.hermes/.env` 中）：
```bash
BLUEBUBBLES_ALLOW_ALL_USERS=true
```

### 5. 启动 Gateway

```bash
hermes gateway run
```

Hermes 将连接至你的 BlueBubbles 服务端，注册 webhook，并开始监听 iMessage 消息。

## 工作原理

```
iMessage → Messages.app → BlueBubbles Server → Webhook → Hermes
Hermes → BlueBubbles REST API → Messages.app → iMessage
```

- **入站：** 新消息到达时，BlueBubbles 向本地监听器发送 webhook 事件。无需轮询——即时送达。
- **出站：** Hermes 通过 BlueBubbles REST API 发送消息。
- **媒体：** 双向支持图片、语音消息、视频和文档。入站附件会被下载并在本地缓存，供 Agent 处理。

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|----------|----------|---------|-------------|
| `BLUEBUBBLES_SERVER_URL` | 是 | — | BlueBubbles 服务端 URL |
| `BLUEBUBBLES_PASSWORD` | 是 | — | 服务端密码 |
| `BLUEBUBBLES_WEBHOOK_HOST` | 否 | `127.0.0.1` | Webhook 监听器绑定地址 |
| `BLUEBUBBLES_WEBHOOK_PORT` | 否 | `8645` | Webhook 监听器端口 |
| `BLUEBUBBLES_WEBHOOK_PATH` | 否 | `/bluebubbles-webhook` | Webhook URL 路径 |
| `BLUEBUBBLES_HOME_CHANNEL` | 否 | — | cron 投递使用的手机号/邮箱 |
| `BLUEBUBBLES_ALLOWED_USERS` | 否 | — | 逗号分隔的授权用户列表 |
| `BLUEBUBBLES_ALLOW_ALL_USERS` | 否 | `false` | 允许所有用户 |

自动将消息标记为已读由 `~/.hermes/config.yaml` 中 `platforms.bluebubbles.extra` 下的 `send_read_receipts` 键控制（默认值：`true`）。该选项没有对应的环境变量。

## 功能特性

### 文字消息
发送和接收 iMessage。Markdown 会自动去除，以确保纯文本的整洁呈现。

### 富媒体
- **图片：** 照片在 iMessage 对话中原生显示
- **语音消息：** 音频文件以 iMessage 语音消息形式发送
- **视频：** 视频附件
- **文档：** 文件以 iMessage 附件形式发送

### Tapback 反应
支持喜爱、点赞、踩、大笑、强调和疑问等反应。需要 BlueBubbles [Private API helper](https://docs.bluebubbles.app/helper-bundle/installation)。

### 正在输入指示器
Agent 处理消息期间，iMessage 对话中会显示"正在输入……"。需要 Private API。

### 已读回执
处理消息后自动标记为已读。需要 Private API。

### 聊天寻址
你可以通过邮箱或手机号寻址聊天——Hermes 会自动将其解析为 BlueBubbles 聊天 GUID，无需使用原始 GUID 格式。

## Private API

部分功能需要 BlueBubbles [Private API helper](https://docs.bluebubbles.app/helper-bundle/installation)：
- Tapback 反应
- 正在输入指示器
- 已读回执
- 通过地址创建新聊天

不使用 Private API 时，基本文字消息和媒体功能仍可正常使用。

## 故障排查

### "Cannot reach server"
- 确认服务端 URL 正确且 Mac 已开机
- 检查 BlueBubbles Server 是否正在运行
- 确保网络连通（防火墙、端口转发）

### 消息未送达
- 检查 webhook 是否已在 BlueBubbles Server → Settings → API → Webhooks 中注册
- 确认 webhook URL 可从 Mac 访问
- 查看 `hermes logs gateway` 中的 webhook 错误（或使用 `hermes logs -f` 实时跟踪）

### "Private API helper not connected"
- 安装 Private API helper：[docs.bluebubbles.app](https://docs.bluebubbles.app/helper-bundle/installation)
- 不安装也可使用基本消息功能——仅反应、正在输入和已读回执需要它