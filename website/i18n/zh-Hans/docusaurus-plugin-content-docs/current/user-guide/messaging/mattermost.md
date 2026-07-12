---
sidebar_position: 8
title: "Mattermost"
description: "将 Hermes Agent 配置为 Mattermost 机器人"
---

# Mattermost 配置

Hermes Agent 以机器人身份集成到 Mattermost，让你可以通过私信或团队频道与 AI 助手对话。Mattermost 是一个自托管的开源 Slack 替代品——运行在你自己的基础设施上，完全掌控数据。机器人通过 Mattermost 的 REST API（v4）和 WebSocket 连接以接收实时事件，将消息通过 Hermes Agent 管道（包括工具调用、记忆和推理）处理后实时响应。支持文本、文件附件、图片和斜杠命令。

无需额外的 Mattermost 库——适配器使用 `aiohttp`，该库已作为 Hermes 的依赖项包含在内。

在开始配置之前，先了解大多数人最关心的部分：Hermes 进入你的 Mattermost 实例后的行为方式。

## Hermes 的行为方式

| 场景 | 行为 |
|---------|----------|
| **私信（DM）** | Hermes 响应每一条消息，无需 `@提及`。每个私信有独立的会话。 |
| **公开/私有频道** | Hermes 仅在被 `@提及` 时响应。未被提及时，Hermes 忽略消息。 |
| **线程（Thread）** | 若设置 `MATTERMOST_REPLY_MODE=thread`，Hermes 在你的消息下方以线程形式回复。线程上下文与父频道隔离。 |
| **多用户共享频道** | 默认情况下，Hermes 在频道内按用户隔离会话历史。同一频道中的两个人不会共享同一份对话记录，除非你明确禁用该设置。 |

:::tip
如果你希望 Hermes 以线程对话方式回复（嵌套在原始消息下方），请设置 `MATTERMOST_REPLY_MODE=thread`。默认值为 `off`，即在频道中发送普通消息。
:::

### Mattermost 中的会话模型

默认情况下：

- 每个私信有独立的会话
- 每个线程有独立的会话命名空间
- 共享频道中的每个用户在该频道内有独立的会话

这由 `config.yaml` 控制：

```yaml
group_sessions_per_user: true
```

仅当你明确希望整个频道共享一个对话时，才将其设为 `false`：

```yaml
group_sessions_per_user: false
```

共享会话在协作频道中可能有用，但也意味着：

- 用户共享上下文增长和 token 消耗
- 一个人的长时间重度工具调用任务会使所有人的上下文膨胀
- 一个人正在进行的任务可能会打断同一频道中另一个人的后续操作

本指南将带你完成完整的配置流程——从在 Mattermost 上创建机器人到发送第一条消息。

## 第一步：启用机器人账户

在创建机器人账户之前，必须先在 Mattermost 服务器上启用该功能。

1. 以**系统管理员**身份登录 Mattermost。
2. 前往**系统控制台** → **集成** → **机器人账户**。
3. 将**启用机器人账户创建**设置为 **true**。
4. 点击**保存**。

:::info
如果你没有系统管理员权限，请联系 Mattermost 管理员启用机器人账户并为你创建一个。
:::

## 第二步：创建机器人账户

1. 在 Mattermost 中，点击左上角的 **☰** 菜单 → **集成** → **机器人账户**。
2. 点击**添加机器人账户**。
3. 填写详细信息：
   - **用户名**：例如 `hermes`
   - **显示名称**：例如 `Hermes Agent`
   - **描述**：可选
   - **角色**：`Member` 即可
4. 点击**创建机器人账户**。
5. Mattermost 将显示**机器人 token**。**立即复制。**

:::warning[Token 仅显示一次]
机器人 token 仅在创建机器人账户时显示一次。如果丢失，需要在机器人账户设置中重新生成。切勿公开分享你的 token 或将其提交到 Git——任何持有此 token 的人都能完全控制该机器人。
:::

将 token 保存在安全的地方（例如密码管理器）。第五步中会用到它。

:::tip
你也可以使用**个人访问 token** 代替机器人账户。前往**个人资料** → **安全** → **个人访问 Token** → **创建 Token**。如果你希望 Hermes 以你自己的用户身份发帖而非独立的机器人用户，这种方式很有用。
:::

## 第三步：将机器人添加到频道

机器人需要成为你希望它响应的频道的成员：

1. 打开你希望添加机器人的频道。
2. 点击频道名称 → **添加成员**。
3. 搜索你的机器人用户名（例如 `hermes`）并添加。

对于私信，直接与机器人开启私信即可——它将立即能够响应。

## 第四步：查找你的 Mattermost 用户 ID

Hermes Agent 使用你的 Mattermost 用户 ID 来控制谁可以与机器人交互。查找方式：

1. 点击左上角的**头像** → **个人资料**。
2. 用户 ID 显示在个人资料对话框中——点击即可复制。

你的用户 ID 是一个 26 位字母数字字符串，例如 `3uo8dkh1p7g1mfk49ear5fzs5c`。

:::warning
你的用户 ID **不是**你的用户名。用户名是 `@` 后面显示的内容（例如 `@alice`）。用户 ID 是 Mattermost 内部使用的长字母数字标识符。
:::

**替代方法**：你也可以通过 API 获取用户 ID：

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-mattermost-server/api/v4/users/me | jq .id
```

:::tip
要获取**频道 ID**：点击频道名称 → **查看信息**。频道 ID 显示在信息面板中。如果你想手动设置主频道，需要用到它。
:::

## 第五步：配置 Hermes Agent

### 方式 A：交互式配置（推荐）

运行引导式配置命令：

```bash
hermes gateway setup
```

在提示时选择 **Mattermost**，然后按提示粘贴你的服务器 URL、机器人 token 和用户 ID。

### 方式 B：手动配置

在你的 `~/.hermes/.env` 文件中添加以下内容：

```bash
# 必填
MATTERMOST_URL=https://mm.example.com
MATTERMOST_TOKEN=***
MATTERMOST_ALLOWED_USERS=3uo8dkh1p7g1mfk49ear5fzs5c

# 多个允许的用户（逗号分隔）
# MATTERMOST_ALLOWED_USERS=3uo8dkh1p7g1mfk49ear5fzs5c,8fk2jd9s0a7bncm1xqw4tp6r3e

# 可选：回复模式（thread 或 off，默认：off）
# MATTERMOST_REPLY_MODE=thread

# 可选：无需 @提及 即可响应（默认：true = 需要提及）
# MATTERMOST_REQUIRE_MENTION=false

# 可选：机器人无需 @提及 即可响应的频道（逗号分隔的频道 ID）
# MATTERMOST_FREE_RESPONSE_CHANNELS=channel_id_1,channel_id_2
```

`~/.hermes/config.yaml` 中的可选行为设置：

```yaml
group_sessions_per_user: true
```

- `group_sessions_per_user: true` 使每个参与者在共享频道和线程中的上下文保持隔离

### 启动 Gateway

配置完成后，启动 Mattermost gateway：

```bash
hermes gateway
```

机器人应在几秒内连接到你的 Mattermost 服务器。发送一条消息——私信或在已添加机器人的频道中——进行测试。

:::tip
你可以在后台运行 `hermes gateway`，或将其配置为 systemd 服务以持续运行。详情参见部署文档。
:::

## 主频道

你可以指定一个"主频道"，机器人将在此频道发送主动消息（例如 cron 任务输出、提醒和通知）。有两种设置方式：

### 使用斜杠命令

在机器人所在的任意 Mattermost 频道中输入 `/sethome`。该频道即成为主频道。

### 手动配置

在你的 `~/.hermes/.env` 中添加：

```bash
MATTERMOST_HOME_CHANNEL=abc123def456ghi789jkl012mn
```

将 ID 替换为实际的频道 ID（点击频道名称 → 查看信息 → 复制 ID）。

## 回复模式

`MATTERMOST_REPLY_MODE` 设置控制 Hermes 发布响应的方式：

| 模式 | 行为 |
|------|----------|
| `off`（默认） | Hermes 在频道中发送普通消息，与普通用户一样。 |
| `thread` | Hermes 在你的原始消息下方以线程形式回复。在大量来回交流时保持频道整洁。 |

在你的 `~/.hermes/.env` 中设置：

```bash
MATTERMOST_REPLY_MODE=thread
```

## 提及行为

默认情况下，机器人仅在频道中被 `@提及` 时响应。你可以更改此行为：

| 变量 | 默认值 | 描述 |
|----------|---------|-------------|
| `MATTERMOST_REQUIRE_MENTION` | `true` | 设为 `false` 可响应频道中的所有消息（私信始终有效）。 |
| `MATTERMOST_FREE_RESPONSE_CHANNELS` | _（无）_ | 逗号分隔的频道 ID，机器人在这些频道中无需 `@提及` 即可响应，即使 require_mention 为 true。 |

在 Mattermost 中查找频道 ID：打开频道，点击频道名称标题，在 URL 或频道详情中查找 ID。

当机器人被 `@提及` 时，提及内容会在处理前自动从消息中去除。

## 频道白名单（`allowed_channels`）

将机器人限制在固定的 Mattermost 频道集合中。设置后，机器人**仅**在 ID 出现在列表中的频道响应——来自其他频道的消息将被静默忽略，即使机器人被 `@提及`。

**私信不受此过滤器限制**，因此授权用户始终可以通过私信联系机器人。

```yaml
mattermost:
  allowed_channels:
    - "abc123def456ghi789jkl012mno"   # #ops
    - "xyz987uvw654rst321opq098nml"   # #incident-response
```

或通过环境变量设置（逗号分隔）：

```bash
MATTERMOST_ALLOWED_CHANNELS="abc123def456ghi789jkl012mno,xyz987uvw654rst321opq098nml"
```

行为说明：

- 空值/未设置 → 无限制（完全向后兼容）。
- 非空值 → 频道 ID 必须在列表中，否则消息在任何其他门控（提及要求、`MATTERMOST_FREE_RESPONSE_CHANNELS` 等）运行之前即被丢弃。
- 通过 Mattermost UI → 频道标题 → "查看信息"查找频道 ID，或从频道 URL 中读取。

另请参阅：[管理员/用户斜杠命令分离](../../reference/slash-commands.md#permissions-and-adminuser-split)。

## 故障排查

### 机器人不响应消息

**原因**：机器人不是该频道的成员，或 `MATTERMOST_ALLOWED_USERS` 中未包含你的用户 ID。

**解决方法**：将机器人添加到频道（频道名称 → 添加成员 → 搜索机器人）。确认你的用户 ID 在 `MATTERMOST_ALLOWED_USERS` 中。重启 gateway。

### 403 Forbidden 错误

**原因**：机器人 token 无效，或机器人没有在该频道发帖的权限。

**解决方法**：检查 `.env` 文件中的 `MATTERMOST_TOKEN` 是否正确。确认机器人账户未被停用。确认机器人已被添加到频道。如果使用个人访问 token，确保你的账户具有所需权限。

### WebSocket 断开连接/重连循环

**原因**：网络不稳定、Mattermost 服务器重启，或防火墙/代理对 WebSocket 连接的干扰。

**解决方法**：适配器会以指数退避方式（2s → 60s）自动重连。检查服务器的 WebSocket 配置——反向代理（nginx、Apache）需要配置 WebSocket 升级头。确认没有防火墙阻止 Mattermost 服务器上的 WebSocket 连接。

对于 nginx，确保你的配置包含：

```nginx
location /api/v4/websocket {
    proxy_pass http://mattermost-backend;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 600s;
}
```

### 启动时出现"Failed to authenticate"

**原因**：token 或服务器 URL 不正确。

**解决方法**：确认 `MATTERMOST_URL` 指向你的 Mattermost 服务器（包含 `https://`，末尾无斜杠）。检查 `MATTERMOST_TOKEN` 是否有效——用 curl 测试：

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-server/api/v4/users/me
```

如果返回机器人的用户信息，则 token 有效。如果返回错误，请重新生成 token。

### 机器人离线

**原因**：Hermes gateway 未运行，或连接失败。

**解决方法**：检查 `hermes gateway` 是否正在运行。查看终端输出中的错误信息。常见问题：URL 错误、token 过期、Mattermost 服务器无法访问。

### "User not allowed"/机器人忽略你

**原因**：你的用户 ID 不在 `MATTERMOST_ALLOWED_USERS` 中。

**解决方法**：将你的用户 ID 添加到 `~/.hermes/.env` 中的 `MATTERMOST_ALLOWED_USERS`，然后重启 gateway。注意：用户 ID 是 26 位字母数字字符串，不是你的 `@用户名`。

## 按频道设置 Prompt

为特定 Mattermost 频道分配临时系统 prompt（提示词）。该 prompt 在每次对话轮次中于运行时注入——从不持久化到对话记录——因此更改立即生效。

```yaml
mattermost:
  channel_prompts:
    "channel_id_abc123": |
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "channel_id_def456": |
      Code review mode. Be precise about edge cases and
      performance implications.
```

键为 Mattermost 频道 ID（在频道 URL 或通过 API 查找）。匹配频道中的所有消息都会将该 prompt 作为临时系统指令注入。

## 安全

:::warning
务必设置 `MATTERMOST_ALLOWED_USERS` 以限制谁可以与机器人交互。若未设置，gateway 默认拒绝所有用户作为安全措施。仅添加你信任的人的用户 ID——授权用户对 agent 的所有功能拥有完整访问权限，包括工具调用和系统访问。
:::

有关保护 Hermes Agent 部署的更多信息，请参阅[安全指南](../security.md)。

## 说明

- **自托管友好**：适用于任何自托管的 Mattermost 实例。无需 Mattermost Cloud 账户或订阅。
- **无额外依赖**：适配器使用 `aiohttp` 处理 HTTP 和 WebSocket，该库已包含在 Hermes Agent 中。
- **兼容团队版**：同时支持 Mattermost 团队版（免费）和企业版。