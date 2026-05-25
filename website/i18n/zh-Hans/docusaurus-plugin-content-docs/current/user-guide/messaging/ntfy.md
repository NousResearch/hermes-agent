# ntfy

[ntfy](https://ntfy.sh/) 是一个简单的基于 HTTP 的发布-订阅通知服务。它可与 `ntfy.sh` 上的免费公共服务器或任何自托管实例配合使用，支持任何能发起 HTTP 请求的客户端——手机、浏览器、脚本、手表。

ntfy 是 Hermes 的轻量级推送渠道的理想选择：通过 [ntfy 移动应用](https://ntfy.sh/docs/subscribe/phone/) 订阅一个 topic（主题），向该 topic 发送消息与 agent 对话，然后在手机上收到回复。

## 前提条件

- 一个 topic 名称（任意唯一字符串——`hermes-myname-2026` 即可）
- 已安装 [ntfy 移动应用](https://ntfy.sh/docs/subscribe/phone/) 并订阅该 topic
- 可选：自托管的 ntfy 服务器，或用于私有/保留 topic 的 `ntfy.sh` 账户 token

仅此而已。无需 SDK、无需守护进程、无需 Node.js。适配器使用 `httpx`，该库已是 Hermes 的依赖项。

## 配置 Hermes

### 通过设置向导

```bash
hermes setup gateway
```

选择 **ntfy** 并按提示操作。

### 通过环境变量

将以下内容添加到 `~/.hermes/.env`：

```
NTFY_TOPIC=hermes-myname-2026
NTFY_ALLOWED_USERS=hermes-myname-2026
NTFY_HOME_CHANNEL=hermes-myname-2026
```

| 变量 | 是否必填 | 说明 |
|---|---|---|
| `NTFY_TOPIC` | 是 | 要订阅的 topic（接收消息） |
| `NTFY_SERVER_URL` | 可选 | 服务器 URL（默认：`https://ntfy.sh`）——指向自托管 ntfy 以保护隐私 |
| `NTFY_TOKEN` | 可选 | Bearer token（如 `tk_xyz`）或用于 Basic 认证的 `user:pass` |
| `NTFY_PUBLISH_TOPIC` | 可选 | 用于发送回复的不同 topic（默认与 `NTFY_TOPIC` 相同） |
| `NTFY_MARKDOWN` | 可选 | 设为 `true` 以使用 `X-Markdown: true` 请求头发送回复 |
| `NTFY_ALLOWED_USERS` | 推荐 | 允许的 topic 名称（逗号分隔，视为用户 ID；见下文） |
| `NTFY_ALLOW_ALL_USERS` | 可选 | 设为 `true` 以允许所有发布者——仅在具有读取 token 的私有 topic 下安全 |
| `NTFY_HOME_CHANNEL` | 可选 | cron 任务/通知投递的默认 topic |
| `NTFY_HOME_CHANNEL_NAME` | 可选 | 主渠道的可读标签 |

## 身份模型——部署前请阅读

ntfy 没有原生的已认证用户身份。已发布消息中的 `title` 字段由**发布者控制**，可以是发布者想要的任何内容。Hermes 适配器**不**使用 `title` 进行授权——否则任何知道 topic 的发布者都可以伪造允许的用户。

相反，**topic 名称本身即为身份**。发布到该 topic 的每条消息都被视为来自同一个逻辑用户（即该 topic）。因此 `NTFY_ALLOWED_USERS` 通常就是 topic 名称本身——一个控制整个渠道访问的单条目白名单。

这意味着**任何知道 topic 的人都可以与 agent 对话**。要将其变为真正的信任边界：

- **自托管 ntfy** 并通过[访问控制](https://docs.ntfy.sh/config/#access-control)锁定 topic。只有持有读/写 token 的授权客户端才能发布。
- 或**在 ntfy.sh 上使用私有 topic**（[保留 topic](https://docs.ntfy.sh/publish/#reserved-topics) 需要账户），并通过 `NTFY_TOKEN` 保护。
- 或**选择一个长且难以猜测的 topic 名称**（`hermes-7d4f9c8b-2026`），将其视为共享密钥。这是最轻量的方案，但 topic 名称可能通过日志或截图泄露。

在任何情况下，除非底层 topic 已启用访问控制，否则不要通过 ntfy 传输敏感数据。

## 快速开始——从手机与 agent 对话

1. 选择一个 topic 名称：`hermes-myname-2026`
2. 在手机上：安装 [ntfy 应用](https://ntfy.sh/docs/subscribe/phone/)，点击 **+**，输入 `hermes-myname-2026`
3. 在主机上：
   ```bash
   echo 'NTFY_TOPIC=hermes-myname-2026' >> ~/.hermes/.env
   echo 'NTFY_ALLOWED_USERS=hermes-myname-2026' >> ~/.hermes/.env
   hermes gateway restart
   ```
4. 从 ntfy 应用向该 topic 发送一条消息。agent 的回复将以推送通知的形式送达。

## 在 cron 任务中使用 ntfy

设置 `NTFY_HOME_CHANNEL` 后，cron 任务即可投递到 ntfy：

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="ntfy",          # uses NTFY_HOME_CHANNEL
    prompt="Check for alerts and summarise."
)
```

或显式指定目标 topic：

```python
send_message(target="ntfy:alerts-channel", message="Done!")
```

即使 cron 在 gateway 进程外运行，此功能也有效——插件注册了一个 `standalone_sender_fn`，会自行建立 HTTP 连接。

## 自托管 ntfy

如需完全掌控：

```bash
# Docker
docker run -p 80:80 -it binwiederhier/ntfy serve

# Native
go install heckel.io/ntfy/v2@latest
ntfy serve
```

然后将 Hermes 指向该实例：

```
NTFY_SERVER_URL=https://ntfy.mydomain.com
NTFY_TOPIC=hermes
NTFY_TOKEN=tk_abc123  # if you've set up access control
```

自托管可提供 topic 访问控制、消息持久化策略、附件和 emoji 标签。参见 [ntfy 服务器文档](https://docs.ntfy.sh/install/)。

## Markdown 格式化

当发布者设置 `X-Markdown: true` 请求头时，ntfy 客户端会渲染 Markdown。要为 Hermes 的出站回复启用此功能：

```
NTFY_MARKDOWN=true
```

或在 `config.yaml` 中配置：

```yaml
platforms:
  ntfy:
    extra:
      markdown: true
```

移动应用支持 CommonMark 的子集——粗体、斜体、列表、链接、围栏代码块。确切支持范围参见 [ntfy 的 Markdown 文档](https://docs.ntfy.sh/publish/#markdown-formatting)。

## 仅出站设置（只推送通知，不接收消息）

如果只希望 Hermes *推送*通知到 ntfy（cron 摘要、告警），而不接受任何回复消息，可将 `NTFY_TOPIC` 和 `NTFY_PUBLISH_TOPIC` 设为相同值，并完全省略 `NTFY_ALLOWED_USERS`。没有白名单时，agent 不会响应任何入站消息——手机可收到推送，但对话是单向的。

## 限制

- **消息大小**：ntfy 将消息体上限设为 4096 个字符。超出时 Hermes 会截断并发出警告。
- **无输入状态指示**：协议不支持此功能；`send_typing` 为空操作。
- **无线程或附件**：ntfy 是纯推送通知。长回复保留在消息体中，不会分线程展开。
- **无原生用户身份**：参见上文的身份模型章节。

## 故障排查

**认证失败 / 401** — `NTFY_TOKEN` 有误，或该 token 对此 topic 没有发布/订阅权限。适配器在收到 401 时会停止重连循环，gateway 运行时状态将显示 `fatal: ntfy_unauthorized`。修正 token 后重启 gateway。

**Topic 未找到 / 404** — `NTFY_TOPIC` 在所配置的服务器上不存在。对于 ntfy.sh，topic 在首次发布时自动创建，因此 404 意味着你指向的自托管服务器尚未创建该 topic。适配器会停止重连循环并显示 `fatal: ntfy_topic_not_found`。

**已连接但收不到消息** — 检查 `NTFY_ALLOWED_USERS` 是否包含 topic 名称本身。在 ntfy 的身份模型中，topic 即用户；白名单为空时所有消息都会被拒绝。

**每 60 秒重连一次** — 流式 keepalive 默认为 55 秒；ntfy 可能存在间歇性网络问题。适配器采用指数退避（2 → 5 → 10 → 30 → 60 秒），一旦流保持存活 ≥60 秒则重置为 0。