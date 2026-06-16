# SimpleX Chat

[SimpleX Chat](https://simplex.chat/) 是一个私密的去中心化即时通讯平台，用户完全掌控自己的联系人和群组。与其他平台不同，SimpleX 不分配任何持久用户 ID——每个联系人在建立连接时由系统生成一个不透明的内部 ID，这使其成为目前隐私性最强的即时通讯工具之一。

## 前提条件

- 已安装并以守护进程方式运行的 **simplex-chat** CLI
- Python 包 **websockets**（`pip install websockets`）

## 安装 simplex-chat

从 [simplex-chat GitHub releases](https://github.com/simplex-chat/simplex-chat/releases) 页面下载最新版本：

```bash
# Linux / macOS binary
curl -L https://github.com/simplex-chat/simplex-chat/releases/latest/download/simplex-chat-ubuntu-22_04-x86-64 -o simplex-chat
chmod +x simplex-chat
```

SimpleX Chat 项目未发布聊天客户端的预构建 Docker 镜像；如需在 Docker 下运行，请从 [simplex-chat 仓库](https://github.com/simplex-chat/simplex-chat) 源码构建。

## 启动守护进程

```bash
simplex-chat -p 5225
```

守护进程默认在 `ws://127.0.0.1:5225` 上监听 WebSocket 连接。

## 配置 Hermes

### 通过设置向导

```bash
hermes gateway setup
```

选择 **SimpleX Chat** 并按提示操作。

### 通过环境变量

将以下内容添加到 `~/.hermes/.env`：

```
SIMPLEX_WS_URL=ws://127.0.0.1:5225
SIMPLEX_ALLOWED_USERS=<contact-id-1>,<contact-id-2>
SIMPLEX_HOME_CHANNEL=<contact-id>
```

| 变量 | 是否必填 | 说明 |
|---|---|---|
| `SIMPLEX_WS_URL` | 是 | simplex-chat 守护进程的 WebSocket URL |
| `SIMPLEX_ALLOWED_USERS` | 建议填写 | 允许使用 Agent 的联系人 ID，以逗号分隔 |
| `SIMPLEX_ALLOW_ALL_USERS` | 可选 | 设为 `true` 以允许所有联系人（请谨慎使用） |
| `SIMPLEX_HOME_CHANNEL` | 可选 | cron 任务投递的默认联系人 ID |
| `SIMPLEX_HOME_CHANNEL_NAME` | 可选 | 主频道的可读标签 |

## 查找联系人 ID

启动守护进程后，与你的 Agent 联系人开启一段对话。联系人 ID 将出现在会话日志中，或通过 `hermes send_message action=list` 查看。

## 授权

默认情况下**所有联系人均被拒绝访问**。你必须选择以下方式之一：

1. 将 `SIMPLEX_ALLOWED_USERS` 设置为以逗号分隔的联系人 ID 列表，或
2. 使用 **DM 配对**——向 Bot 发送任意消息，Bot 将回复一个配对码。通过 `hermes pairing approve simplex <CODE>` 输入该配对码。

## 在 cron 任务中使用 SimpleX

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="simplex",          # uses SIMPLEX_HOME_CHANNEL
    prompt="Check for alerts and summarise."
)
```

或指定特定联系人：

```python
send_message(target="simplex:<contact-id>", message="Done!")
```

## 隐私说明

- SimpleX 从不暴露手机号或电子邮件地址——联系人使用不透明 ID 标识
- Hermes 与守护进程之间的连接为本地 WebSocket（`ws://127.0.0.1:5225`）——数据不会离开你的机器
- 消息在到达守护进程之前已由 SimpleX 协议进行端到端加密

## 故障排查

**"Cannot reach daemon"** — 确保 `simplex-chat -p 5225` 正在运行，且端口与 `SIMPLEX_WS_URL` 一致。

**"websockets not installed"** — 运行 `pip install websockets`。

**消息未收到** — 检查该联系人的 ID 是否已加入 `SIMPLEX_ALLOWED_USERS`，或通过 DM 配对方式批准该联系人。