---
sidebar_position: 8
title: "Mattermost"
description: "将 Hermes Agent 设置为 Mattermost 机器人"
---

# Mattermost 设置

Hermes Agent 与 Mattermost 集成为机器人，让你可以通过私信或团队频道与 AI 助手聊天。Mattermost 是一个自托管的开源 Slack 替代方案 —— 你在自己的基础设施上运行它，保持对数据的完全控制。机器人通过 Mattermost 的 REST API（v4）和 WebSocket 连接以获取实时事件，通过 Hermes Agent 管道（包括工具使用、记忆和推理）处理消息，并实时回复。它支持文本、文件附件、图片和斜杠命令。

不需要外部 Mattermost 库 —— 适配器使用 `aiohttp`，它是 Hermes 的已有依赖。

在设置之前，这里是大多数人想了解的内容：Hermes 在你的 Mattermost 实例中后是如何行为的。

## Hermes 的行为方式

| 上下文 | 行为 |
|---------|----------|
| **私信** | Hermes 回复每条消息。不需要 `@mention`。每个私信都有自己的会话。 |
| **公开/私有频道** | Hermes 在你 `@mention` 它时回复。如果没有提及，Hermes 会忽略该消息。 |
| **主题** | 如果 `MATTERMOST_REPLY_MODE=thread`，Hermes 在你的消息下方的线程中回复。主题上下文与父频道保持隔离。 |
| **多用户共享频道** | 默认情况下，Hermes 在频道内为每个用户隔离会话历史。同一频道中的两个人不会共享一个对话记录，除非你明确禁用该选项。 |

:::tip
如果你希望 Hermes 以线程对话回复（嵌套在你的原始消息下），请设置 `MATTERMOST_REPLY_MODE=thread`。默认是 `off`，它在频道中发送扁平消息。
:::

### Mattermost 中的会话模型

默认情况下：

- 每个私信都有自己的会话
- 每个主题都有自己的会话命名空间
- 共享频道中的每个用户在该频道内都有自己的会话

这可以通过 `config.yaml` 控制：

```yaml
group_sessions_per_user: true
```

仅当你明确希望整个频道共享一个对话时，才将其设置为 `false`：

```yaml
group_sessions_per_user: false
```

共享会话对于协作频道可能很有用，但它们也意味着：

- 用户共享上下文增长和 token 成本
- 一个人的长时间工具密集型任务可能会使其他人的上下文膨胀
- 一个人的运行中任务可能会中断同一频道中另一个人的后续操作

本指南将引导你完成完整的设置过程 —— 从在 Mattermost 上创建你的机器人到发送你的第一条消息。

## 步骤 1：启用机器人账号

在创建机器人之前，必须在你的 Mattermost 服务器上启用机器人账号。

1. 以**系统管理员**身份登录 Mattermost。
2. 转到 **System Console** → **Integrations** → **Bot Accounts**。
3. 将 **Enable Bot Account Creation** 设置为 **true**。
4. 点击 **Save**。

:::info
如果你没有系统管理员访问权限，请让你的 Mattermost 管理员启用机器人账号并为你创建一个。
:::

## 步骤 2：创建机器人账号

1. 在 Mattermost 中，点击 **☰** 菜单（左上角）→ **Integrations** → **Bot Accounts**。
2. 点击 **Add Bot Account**。
3. 填写详细信息：
   - **Username**：例如 `hermes`
   - **Display Name**：例如 `Hermes Agent`
   - **Description**：可选
   - **Role**：`Member` 就足够了
4. 点击 **Create Bot Account**。
5. Mattermost 会显示**机器人令牌**。**立即复制它。**

:::warning[令牌仅显示一次]
机器人的令牌仅在创建机器人账号时显示一次。如果丢失，你需要从机器人账号设置中重新生成它。永远不要公开分享你的令牌或将其提交到 Git —— 任何拥有此令牌的人都可以完全控制机器人。
:::

将令牌存储在安全的地方（例如密码管理器）。你将在步骤 5 中需要它。

:::tip
你也可以使用**个人访问令牌**而不是机器人账号。转到**个人资料** → **安全** → **个人访问令牌** → **创建令牌**。如果你希望 Hermes 以你自己的用户发帖而不是单独的机器人用户，这很有用。
:::

## 步骤 3：将机器人添加到频道

机器人需要是你希望它回复的任何频道的成员：

1. 打开你希望机器人所在的频道。
2. 点击频道名称 → **Add Members**。
3. 搜索你的机器人用户名（例如 `hermes`）并添加它。

对于私信，只需打开与机器人的直接消息 —— 它将能够立即回复。

## 步骤 4：找到你的 Mattermost 用户 ID

Hermes Agent 使用你的 Mattermost 用户 ID 来控制谁可以与机器人交互。要找到它：

1. 点击你的**头像**（左上角）→ **Profile**。
2. 你的用户 ID 显示在个人资料对话框中 —— 点击它以复制。

你的用户 ID 是一个 26 个字符的字母数字字符串，如 `3uo8dkh1p7g1mfk49ear5fzs5c`。

:::warning
你的用户 ID **不是**你的用户名。用户名是 `@` 后出现的内容（例如 `@alice`）。用户 ID 是 Mattermost 在内部使用的长字母数字标识符。
:::

**替代方案**：你也可以通过 API 获取你的用户 ID：

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-mattermost-server/api/v4/users/me | jq .id
```

:::tip
要获取**频道 ID**：点击频道名称 → **View Info**。频道 ID 显示在信息面板中。如果你想手动设置主页频道，你需要这个。
:::

## 步骤 5：配置 Hermes Agent

### 选项 A：交互式设置（推荐）

运行引导式设置命令：

```bash
hermes gateway setup
```

当提示时选择 **Mattermost**，然后在询问时粘贴你的服务器 URL、机器人令牌和用户 ID。

### 选项 B：手动配置

将以下内容添加到你的 `~/.hermes/.env` 文件：

```bash
# 必需
MATTERMOST_URL=https://mm.example.com
MATTERMOST_TOKEN=***
MATTERMOST_ALLOWED_USERS=3uo8dkh1p7g1mfk49ear5fzs5c

# 多个允许的用户（逗号分隔）
# MATTERMOST_ALLOWED_USERS=3uo8dkh1p7g1mfk49ear5fzs5c,8fk2jd9s0a7bncm1xqw4tp6r3e

# 可选：回复模式（thread 或 off，默认：off）
# MATTERMOST_REPLY_MODE=thread

# 可选：无需 @mention 即可回复（默认：true = 需要提及）
# MATTERMOST_REQUIRE_MENTION=false

# 可选：无需 @mention 即可回复的频道（逗号分隔的频道 ID）
# MATTERMOST_FREE_RESPONSE_CHANNELS=channel_id_1,channel_id_2
```

`~/.hermes/config.yaml` 中的可选行为设置：

```yaml
group_sessions_per_user: true
```

- `group_sessions_per_user: true` 保持每个参与者的上下文在共享频道和主题中隔离

### 启动 Gateway

配置完成后，启动 Mattermost gateway：

```bash
hermes gateway
```

机器人应在几秒钟内连接到你的 Mattermost 服务器。向它发送一条消息 —— 私信或它已被添加的频道 —— 进行测试。

:::tip
你可以在后台运行 `hermes gateway` 或作为 systemd 服务以实现持久运行。详见部署文档。
:::

## 主页频道

你可以指定一个"主页频道"，机器人可以在那里发送主动消息（如 cron 作业输出、提醒和通知）。有两种设置方法：

### 使用斜杠命令

在任何机器人所在的 Mattermost 频道中输入 `/sethome`。该频道成为主页频道。

### 手动配置

将以下内容添加到你的 `~/.hermes/.env`：

```bash
MATTERMOST_HOME_CHANNEL=abc123def456ghi789jkl012mn
```

将 ID 替换为实际的频道 ID（点击频道名称 → View Info → 复制 ID）。

## 回复模式

`MATTERMOST_REPLY_MODE` 设置控制 Hermes 如何发布回复：

| 模式 | 行为 |
|------|----------|
| `off`（默认） | Hermes 在频道中发布扁平消息，像普通用户一样。 |
| `thread` | Hermes 在你的原始消息下方的线程中回复。当有很多来回时，保持频道清洁。 |

在你的 `~/.hermes/.env` 中设置它：

```bash
MATTERMOST_REPLY_MODE=thread
```

## 提及行为

默认情况下，机器人只在被 @mention 时在频道中回复。你可以更改这个：

| 变量 | 默认 | 描述 |
|----------|---------|-------------|
| `MATTERMOST_REQUIRE_MENTION` | `true` | 设置为 `false` 以回复频道中的所有消息（私信始终有效）。 |
| `MATTERMOST_FREE_RESPONSE_CHANNELS` | _(无)_ | 逗号分隔的频道 ID，机器人在这些频道中即使 require_mention 为 true 也无需 @mention 即可回复。 |

在 Mattermost 中找到频道 ID：打开频道，点击频道名称标题，在 URL 或频道详情中查找 ID。

当机器人被 @mention 时，提及会在处理之前自动从消息中去除。

## 故障排除

### 机器人未响应消息

**原因**：机器人不是频道的成员，或 `MATTERMOST_ALLOWED_USERS` 不包含你的用户 ID。

**修复**：将机器人添加到频道（频道名称 → Add Members → 搜索机器人）。验证你的用户 ID 在 `MATTERMOST_ALLOWED_USERS` 中。重启 gateway。

### 403 禁止错误

**原因**：机器人令牌无效，或者机器人没有在频道中发布的权限。

**修复**：检查 `.env` 文件中的 `MATTERMOST_TOKEN` 是否正确。确保机器人账号未被停用。验证机器人已被添加到频道。如果使用个人访问令牌，确保你的账号有所需的权限。

### WebSocket 断开连接 / 重连循环

**原因**：网络不稳定、Mattermost 服务器重启或 WebSocket 连接的防火墙/代理问题。

**修复**：适配器会自动以指数退避重新连接（2s → 60s）。检查服务器的 WebSocket 配置 —— 反向代理（nginx、Apache）需要配置 WebSocket 升级标头。验证没有防火墙阻止你的 Mattermost 服务器上的 WebSocket 连接。

对于 nginx，确保你的配置包括：

```nginx
location /api/v4/websocket {
    proxy_pass http://mattermost-backend;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 600s;
}
```

### 启动时出现"Failed to authenticate"

**原因**：令牌或服务器 URL 不正确。

**修复**：验证 `MATTERMOST_URL` 指向你的 Mattermost 服务器（包含 `https://`，没有尾部斜杠）。检查 `MATTERMOST_TOKEN` 是否有效 —— 用 curl 尝试它：

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-server/api/v4/users/me
```

如果这返回你的机器人的用户信息，令牌是有效的。如果返回错误，请重新生成令牌。

### 机器人离线

**原因**：Hermes gateway 未运行，或连接失败。

**修复**：检查 `hermes gateway` 是否正在运行。查看终端输出中的错误消息。常见问题：错误的 URL、过期的令牌、Mattermost 服务器不可达。

### "User not allowed" / 机器人忽略你

**原因**：你的用户 ID 不在 `MATTERMOST_ALLOWED_USERS` 中。

**修复**：将你的用户 ID 添加到 `~/.hermes/.env` 中的 `MATTERMOST_ALLOWED_USERS` 并重启 gateway。记住：用户 ID 是一个 26 个字符的字母数字字符串，不是你的 `@username`。

## 每频道提示

为特定的 Mattermost 频道分配临时系统提示。提示在运行时注入每轮 —— 从不持久化到记录历史 —— 因此更改会立即生效。

```yaml
mattermost:
  channel_prompts:
    "channel_id_abc123": |
      你是一个研究助手。专注于学术来源、
      引用和简洁的综合。
    "channel_id_def456": |
      代码审查模式。对边缘情况和
      性能影响要精确。
```

键是 Mattermost 频道 ID（在频道 URL 中或通过 API 找到它们）。匹配频道中的所有消息都会注入提示作为临时系统指令。

## 安全

:::warning
始终设置 `MATTERMOST_ALLOWED_USERS` 以限制谁可以与机器人交互。如果没有它，gateway 默认拒绝所有用户，作为一种安全措施。只添加你信任的人的用户 ID —— 授权用户拥有对 agent 功能的完全访问权限，包括工具使用和系统访问。
:::

有关保护你的 Hermes Agent 部署的更多信息，请参阅 [安全指南](../security.md)。

## 注意事项

- **自托管友好**：可与任何自托管 Mattermost 实例配合使用。不需要 Mattermost Cloud 账号或订阅。
- **无额外依赖**：适配器使用 `aiohttp` 进行 HTTP 和 WebSocket，它已包含在 Hermes Agent 中。
- **Team Edition 兼容**：适用于 Mattermost Team Edition（免费）和 Enterprise Edition。
