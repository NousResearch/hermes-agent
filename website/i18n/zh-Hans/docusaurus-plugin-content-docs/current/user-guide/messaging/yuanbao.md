---
sidebar_position: 16
title: "Yuanbao"
description: "通过 WebSocket gateway 将 Hermes Agent 连接到元宝企业消息平台"
---

# Yuanbao

将 Hermes 连接到腾讯企业消息平台 [元宝（Yuanbao）](https://yuanbao.tencent.com/)。该适配器使用 WebSocket gateway 实现实时消息传递，支持单聊（C2C）和群聊两种会话模式。

:::info
元宝是一个企业消息平台，主要用于腾讯内部及企业环境。它使用 WebSocket 进行实时通信，采用基于 HMAC 的认证方式，支持图片、文件和语音消息等富媒体内容。
:::

## 前提条件

- 拥有机器人创建权限的元宝账号
- 元宝 APP_ID 和 APP_SECRET（由平台管理员提供）
- Python 包：`websockets` 和 `httpx`
- 媒体支持需要：`aiofiles`

安装所需依赖：

```bash
pip install websockets httpx aiofiles
```

## 配置

### 1. 在元宝中创建机器人

1. 从 [https://yuanbao.tencent.com/](https://yuanbao.tencent.com/) 下载元宝应用
2. 在应用中进入 **PAI → 我的机器人**，创建一个新机器人
3. 机器人创建完成后，复制 **APP_ID** 和 **APP_SECRET**

### 2. 运行配置向导

配置元宝最简便的方式是通过交互式向导：

```bash
hermes gateway setup
```

在提示时选择 **Yuanbao**。向导将：

1. 询问你的 APP_ID
2. 询问你的 APP_SECRET
3. 自动保存配置

:::tip
WebSocket URL 和 API Domain 均内置了合理的默认值。只需提供 APP_ID 和 APP_SECRET 即可开始使用。
:::

### 3. 配置环境变量

初始配置完成后，在 `~/.hermes/.env` 中验证以下变量：

```bash
# 必填
YUANBAO_APP_ID=your-app-id
YUANBAO_APP_SECRET=your-app-secret
YUANBAO_WS_URL=wss://api.yuanbao.example.com/ws
YUANBAO_API_DOMAIN=https://api.yuanbao.example.com

# 可选：机器人账号 ID（通常从 sign-token 自动获取）
# YUANBAO_BOT_ID=your-bot-id

# 可选：内部路由环境（如 test/staging/production）
# YUANBAO_ROUTE_ENV=production

# 可选：cron/通知的主频道（格式：direct:<account> 或 group:<group_code>）
YUANBAO_HOME_CHANNEL=direct:bot_account_id
YUANBAO_HOME_CHANNEL_NAME="Bot Notifications"

# 可选：限制访问（旧版，细粒度策略请参见下方访问控制）
YUANBAO_ALLOWED_USERS=user_account_1,user_account_2
```

### 4. 启动 Gateway

```bash
hermes gateway
```

适配器将连接到元宝 WebSocket gateway，使用 HMAC 签名进行认证，并开始处理消息。

## 功能特性

- **WebSocket gateway** — 实时双向通信
- **HMAC 认证** — 使用 APP_ID/APP_SECRET 进行安全请求签名
- **C2C 消息** — 用户与机器人的单聊会话
- **群聊消息** — 群组聊天中的会话
- **媒体支持** — 通过 COS（云对象存储）支持图片、文件和语音消息
- **Markdown 格式化** — 消息自动分块以适应元宝的大小限制
- **消息去重** — 防止同一消息被重复处理
- **心跳/保活** — 维持 WebSocket 连接稳定性
- **输入状态指示** — 在 agent 处理期间显示"正在输入…"状态
- **自动重连** — 以指数退避方式处理 WebSocket 断线
- **群组信息查询** — 获取群组详情和成员列表
- **表情/Emoji 支持** — 在会话中发送 TIMFaceElem 表情和 emoji
- **自动设置主频道** — 第一个向机器人发消息的用户自动成为主频道所有者
- **慢响应通知** — 当 agent 处理时间超出预期时发送等待提示

## 配置选项

### 聊天 ID 格式

元宝根据会话类型使用带前缀的标识符：

| 聊天类型 | 格式 | 示例 |
|----------|------|------|
| 单聊（C2C） | `direct:<account>` | `direct:user123` |
| 群聊 | `group:<group_code>` | `group:grp456` |

### 媒体上传

元宝适配器通过 COS（腾讯云对象存储）自动处理媒体上传：

- **图片**：支持 JPEG、PNG、GIF、WebP
- **文件**：支持所有常见文档类型
- **语音**：支持 WAV、MP3、OGG

媒体 URL 在上传前会自动验证并下载，以防止 SSRF 攻击。

## 主频道

在任意元宝聊天（单聊或群聊）中使用 `/sethome` 命令，将其指定为**主频道**。定时任务（cron job）的结果将发送到该频道。

:::tip 自动设置主频道
如果未配置主频道，第一个向机器人发消息的用户将自动成为主频道所有者。如果当前主频道是群聊，第一条单聊消息将把主频道升级为直接频道。
:::

也可以在 `~/.hermes/.env` 中手动设置：

```bash
YUANBAO_HOME_CHANNEL=direct:user_account_id
# 或者设置为群组：
# YUANBAO_HOME_CHANNEL=group:group_code
YUANBAO_HOME_CHANNEL_NAME="My Bot Updates"
```

### 示例：设置主频道

1. 在元宝中与机器人开始对话
2. 发送命令：`/sethome`
3. 机器人回复："Home channel set to [chat_name] with ID [chat_id]. Cron jobs will deliver to this location."
4. 后续 cron job 和通知将发送到该频道

### 示例：Cron Job 投递

创建一个 cron job：

```bash
/cron "0 9 * * *" Check server status
```

定时输出将在每天上午 9 点发送到你的元宝主频道。

## 使用技巧

### 开始对话

在元宝中向机器人发送任意消息：

```
hello
```

机器人将在同一会话线程中回复。

### 可用命令

所有标准 Hermes 命令均可在元宝上使用：

| 命令 | 描述 |
|------|------|
| `/new` | 开始新对话 |
| `/model [provider:model]` | 查看或切换模型 |
| `/sethome` | 将当前聊天设为主频道 |
| `/status` | 显示会话信息 |
| `/help` | 显示可用命令 |

### 发送文件

在元宝聊天中直接附加文件即可发送给机器人。机器人将自动下载并处理附件。

也可以在附件中附带消息：

```
Please analyze this document
```

### 接收文件

当你要求机器人创建或导出文件时，它会直接将文件发送到你的元宝聊天中。

## 故障排查

### 机器人在线但不响应消息

**原因**：WebSocket 握手期间认证失败。

**解决方法**：
1. 验证 APP_ID 和 APP_SECRET 是否正确
2. 检查 WebSocket URL 是否可访问
3. 确保机器人账号拥有适当权限
4. 查看 gateway 日志：`tail -f ~/.hermes/logs/gateway.log`

### "Connection refused" 错误

**原因**：WebSocket URL 不可达或不正确。

**解决方法**：
1. 验证 WebSocket URL 格式（应以 `wss://` 开头）
2. 检查到元宝 API 域名的网络连通性
3. 确认防火墙允许 WebSocket 连接
4. 使用以下命令测试 URL：`curl -I https://[YUANBAO_API_DOMAIN]`

### 媒体上传失败

**原因**：COS 凭证无效或媒体服务器不可达。

**解决方法**：
1. 验证 API_DOMAIN 是否正确
2. 检查机器人是否已启用媒体上传权限
3. 确保媒体文件可访问且未损坏
4. 联系平台管理员检查 COS bucket 配置

### 消息未投递到主频道

**原因**：主频道 ID 格式不正确或 cron job 尚未触发。

**解决方法**：
1. 验证 YUANBAO_HOME_CHANNEL 格式是否正确
2. 使用 `/sethome` 命令自动检测正确格式
3. 使用 `/status` 检查 cron job 计划
4. 验证机器人在目标聊天中是否有发送权限

### 频繁断线

**原因**：WebSocket 连接不稳定或网络不可靠。

**解决方法**：
1. 检查 gateway 日志中的错误模式
2. 在连接设置中增加心跳超时时间
3. 确保到元宝 API 的网络连接稳定
4. 考虑启用详细日志：`HERMES_LOG_LEVEL=debug`

## 访问控制

元宝支持对单聊和群聊进行细粒度访问控制：

```bash
# 单聊策略：open（默认）| allowlist | disabled
YUANBAO_DM_POLICY=open
# 允许单聊机器人的用户 ID，逗号分隔（仅在 DM_POLICY=allowlist 时生效）
YUANBAO_DM_ALLOW_FROM=user_id_1,user_id_2

# 群聊策略：open（默认）| allowlist | disabled
YUANBAO_GROUP_POLICY=open
# 允许的群组代码，逗号分隔（仅在 GROUP_POLICY=allowlist 时生效）
YUANBAO_GROUP_ALLOW_FROM=group_code_1,group_code_2
```

也可以在 `config.yaml` 中设置：

```yaml
platforms:
  yuanbao:
    extra:
      dm_policy: allowlist
      dm_allow_from: "user1,user2"
      group_policy: open
      group_allow_from: ""
```

## 高级配置

### 消息分块

元宝有最大消息大小限制。Hermes 自动对大响应进行分块，采用 Markdown 感知拆分（遵守代码围栏、表格和段落边界）。

### 连接参数

以下连接参数内置于适配器中，具有合理的默认值：

| 参数 | 默认值 | 描述 |
|------|--------|------|
| WebSocket 连接超时 | 15 秒 | 等待 WS 握手的时间 |
| 心跳间隔 | 30 秒 | 保持连接活跃的 ping 频率 |
| 最大重连次数 | 100 | 最大重连尝试次数 |
| 重连退避 | 1s → 60s（指数） | 重连尝试之间的等待时间 |
| 回复心跳间隔 | 2 秒 | RUNNING 状态发送频率 |
| 发送超时 | 30 秒 | 出站 WS 消息的超时时间 |

:::note
这些值目前无法通过环境变量配置，已针对典型元宝部署场景进行优化。
:::

### 详细日志

启用 debug 日志以排查连接问题：

```bash
HERMES_LOG_LEVEL=debug hermes gateway
```

## 与其他功能集成

### Cron Job

在元宝上调度定时任务：

```
/cron "0 */4 * * *" Report system health
```

结果将投递到你的主频道。

### 后台任务

在不阻塞会话的情况下运行长时间操作：

```
/background Analyze all files in the archive
```

### 跨平台消息

从 CLI 向元宝发送消息：

```bash
hermes chat -q "Send 'Hello from CLI' to yuanbao:group:group_code"
```

## 相关文档

- [消息 Gateway 概览](./index.md)
- [斜杠命令参考](/reference/slash-commands)
- [Cron Job](/user-guide/features/cron)
- [后台会话](/user-guide/cli#background-sessions)