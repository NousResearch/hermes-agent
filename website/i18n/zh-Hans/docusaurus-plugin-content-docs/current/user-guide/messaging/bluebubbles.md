# BlueBubbles (iMessage)

通过 [BlueBubbles](https://bluebubbles.app/) 将 Hermes 连接到 Apple iMessage —— 这是一个免费的、开源的 macOS 服务器，可以将 iMessage 桥接到任何设备。

## 前提条件

- 一台始终运行的 **Mac**，运行 [BlueBubbles Server](https://bluebubbles.app/)
- 在该 Mac 的 Messages.app 中登录的 Apple ID
- BlueBubbles Server v1.0.0+（webhook 需要此版本）
- Hermes 和 BlueBubbles 服务器之间的网络连接

## 设置

### 1. 安装 BlueBubbles Server

从 [bluebubbles.app](https://bluebubbles.app/) 下载并安装。完成设置向导 —— 使用你的 Apple ID 登录并配置连接方法（本地网络、Ngrok、Cloudflare 或动态 DNS）。

### 2. 获取服务器 URL 和密码

在 BlueBubbles Server → **Settings → API** 中，记录：
- **服务器 URL**（例如 `http://192.168.1.10:1234`）
- **服务器密码**

### 3. 配置 Hermes

运行设置向导：

```bash
hermes gateway setup
```

选择 **BlueBubbles (iMessage)** 并输入你的服务器 URL 和密码。

或者直接设置 `~/.hermes/.env` 中的环境变量：

```bash
BLUEBUBBLES_SERVER_URL=http://192.168.1.10:1234
BLUEBUBBLES_PASSWORD=your-server-password
```

### 4. 授权用户

选择一种方式：

**DM 配对（推荐）：**
当有人向你的 iMessage 发送消息时，Hermes 会自动向他们发送配对码。使用以下命令批准：

```bash
hermes pairing approve bluebubbles <CODE>
```
使用 `hermes pairing list` 查看待处理的配对码和已批准的用户。

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

Hermes 将连接到你的 BlueBubbles 服务器，注册一个 webhook，并开始监听 iMessage 消息。

## 工作原理

```
iMessage → Messages.app → BlueBubbles Server → Webhook → Hermes
Hermes → BlueBubbles REST API → Messages.app → iMessage
```

- **入站：** BlueBubbles 在新消息到达时向本地监听器发送 webhook 事件。无需轮询 —— 即时送达。
- **出站：** Hermes 通过 BlueBubbles REST API 发送消息。
- **媒体：** 双向支持图片、语音消息、视频和文档。入站附件会被下载并在本地缓存，供 agent 处理。

## 环境变量

| 变量 | 必需 | 默认值 | 描述 |
|----------|----------|---------|-------------|
| `BLUEBUBBLES_SERVER_URL` | 是 | — | BlueBubbles 服务器 URL |
| `BLUEBUBBLES_PASSWORD` | 是 | — | 服务器密码 |
| `BLUEBUBBLES_WEBHOOK_HOST` | 否 | `127.0.0.1` | Webhook 监听器绑定地址 |
| `BLUEBUBBLES_WEBHOOK_PORT` | 否 | `8645` | Webhook 监听器端口 |
| `BLUEBUBBLES_WEBHOOK_PATH` | 否 | `/bluebubbles-webhook` | Webhook URL 路径 |
| `BLUEBUBBLES_HOME_CHANNEL` | 否 | — | 用于 cron 送达的电话/邮箱 |
| `BLUEBUBBLES_ALLOWED_USERS` | 否 | — | 逗号分隔的授权用户列表 |
| `BLUEBUBBLES_ALLOW_ALL_USERS` | 否 | `false` | 允许所有用户 |

自动标记消息为已读由 `~/.hermes/config.yaml` 中 `platforms.bluebubbles.extra` 下的 `send_read_receipts` 键控制（默认值：`true`）。没有对应的环境变量。

## 功能

### 文本消息

发送和接收 iMessage。Markdown 会自动去除，以干净的纯文本形式送达。

### 富媒体

- **图片：** 照片在 iMessage 对话中原生显示
- **语音消息：** 音频文件作为 iMessage 语音消息发送
- **视频：** 视频附件
- **文档：** 文件作为 iMessage 附件发送

### Tapback 反应

点赞、喜欢、不喜欢、大笑、强调和疑问反应。需要 BlueBubbles [Private API helper](https://docs.bluebubbles.app/helper-bundle/installation)。

### 输入指示器

在 agent 处理过程中，iMessage 对话中显示"正在输入..."。需要 Private API。

### 已读回执

处理完成后自动将消息标记为已读。需要 Private API。

### 聊天地址

你可以通过邮箱或电话号码来寻址聊天 —— Hermes 会自动将它们解析为 BlueBubbles 聊天 GUID。无需使用原始 GUID 格式。

## Private API

某些功能需要 BlueBubbles [Private API helper](https://docs.bluebubbles.app/helper-bundle/installation)：

- Tapback 反应
- 输入指示器
- 已读回执
- 通过地址创建新聊天

如果没有 Private API，基本的文本消息和媒体仍然可以工作。

## 故障排除

### "无法连接到服务器"

- 验证服务器 URL 是否正确，Mac 是否开机
- 检查 BlueBubbles Server 是否正在运行
- 确保网络连接正常（防火墙、端口转发）

### 消息未送达

- 检查 BlueBubbles Server → Settings → API → Webhooks 中是否已注册 webhook
- 验证 webhook URL 是否可以从 Mac 访问
- 检查 `hermes logs gateway` 中的 webhook 错误（或使用 `hermes logs -f` 实时跟踪）

### "Private API helper 未连接"

- 安装 Private API helper：[docs.bluebubbles.app](https://docs.bluebubbles.app/helper-bundle/installation)
- 基本消息功能无需它也能工作 —— 只有反应、输入指示器和已读回执需要它
