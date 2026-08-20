# IRC

IRC 适配器将 Hermes 连接到任何 IRC 服务器，在 IRC 频道（或私信）和 Agent 之间中继消息。它通过 Python 标准库的 `asyncio` 实现 IRC 协议 —— **无外部依赖、无 SDK、无守护进程**。它支持 [Libera.Chat](https://libera.chat/) 等公共网络和任何自托管 ircd。

IRC 是纯文本的：不支持语音、图片、文件、线程、反应、输入中或流式传输 —— 回复以 `PRIVMSG` 行发送，长消息会拆分以适应 IRC 行限制。

> 运行 `hermes gateway setup` 并选择 **IRC** 进行引导式设置。

## 前置条件

- 一个要连接的 IRC 服务器（例如 `irc.libera.chat`）
- 一个要加入的频道（例如 `#hermes`）—— 逗号分隔可加入多个频道
- 机器人的昵称（默认：`hermes-bot`）
- 可选：已注册的昵称 + NickServ 密码（如果你的网络需要身份验证）

## 配置 Hermes

你可以通过两种方式配置 IRC —— 环境变量（快速仅环境配置）或 `~/.hermes/gateway-config.yaml` 中的 `gateway` 块。

### 方式 A —— gateway-config.yaml

```yaml
gateway:
  platforms:
    irc:
      enabled: true
      extra:
        server: irc.libera.chat
        port: 6697
        nickname: hermes-bot
        channel: "#hermes"
        use_tls: true
        server_password: ""       # optional server password
        nickserv_password: ""     # optional NickServ identification
        allowed_users: []         # empty = allow all, or list of nicks
        max_message_length: 450   # IRC line limit (safe default)
```

### 方式 B —— 环境变量

| 变量 | 必需 | 说明 |
|----------|:--------:|-------------|
| `IRC_SERVER` | ✅ | IRC 服务器主机名（例如 `irc.libera.chat`） |
| `IRC_CHANNEL` | ✅ | 要加入的频道 —— 逗号分隔可加入多个 |
| `IRC_NICKNAME` | ✅ | 机器人昵称（默认：`hermes-bot`） |
| `IRC_PORT` | — | 服务器端口（默认：带 TLS 为 `6697`，不带为 `6667`） |
| `IRC_USE_TLS` | — | 使用 TLS（`true`/`false`；端口 6697 默认 `true`） |
| `IRC_SERVER_PASSWORD` | — | `PASS` 命令的服务器密码 |
| `IRC_NICKSERV_PASSWORD` | — | 连接时自动 IDENTIFY 的 NickServ 密码 |
| `IRC_ALLOWED_USERS` | — | 逗号分隔的允许与机器人对话的昵称 |
| `IRC_ALLOW_ALL_USERS` | — | 允许频道中任何人与机器人对话（仅限开发） |
| `IRC_HOME_CHANNEL` | — | 定时任务/通知投放的频道（默认为 `IRC_CHANNEL`） |

## 访问控制

默认情况下，只有 `allowed_users`（或 `IRC_ALLOWED_USERS`）中列出的昵称可以与机器人对话。将列表留空**且**设置 `IRC_ALLOW_ALL_USERS=true` 可以让频道中任何人与 Hermes 聊天 —— 适合测试，但在公共网络上不推荐，因为 IRC 昵称在没有网络强制 NickServ 的情况下无法验证。

如果你的网络注册昵称，请设置 `IRC_NICKSERV_PASSWORD`（或 `nickserv_password`），使机器人连接时向 NickServ 身份验证并保持其注册昵称。

## 频道 vs. 私信

- 加入频道中的消息被视为**群组**对话。
- 发给机器人的私信被视为**直接消息**。

定时任务和通知投放到**主频道** —— 如果设置了 `IRC_HOME_CHANNEL` 就使用它，否则使用第一个 `IRC_CHANNEL`。

## 运行网关

```bash
hermes gateway start
```

使用 `hermes gateway status` 检查状态 —— IRC 连接状态在那里报告，包括仅环境配置的情况。

## 注意事项

- 长 Agent 回复会自动拆分为多条 `PRIVMSG` 行以保持在 IRC 行限制内（`max_message_length`，默认 450 字节，扣除协议开销后）。
- 适配器为每个服务器+昵称获取一个作用域凭据锁，因此两个 Hermes 配置文件不会争夺同一个 IRC 身份。
