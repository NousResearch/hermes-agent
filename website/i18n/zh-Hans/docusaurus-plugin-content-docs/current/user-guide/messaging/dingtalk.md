---
sidebar_position: 10
title: "DingTalk"
description: "将 Hermes Agent 设置为钉钉聊天机器人"
---

# 钉钉设置

Hermes Agent 集成了钉钉作为聊天机器人，让你可以通过私信或群聊与 AI 助手对话。机器人通过钉钉的 Stream Mode 连接 —— 一种长连接的 WebSocket，无需公网 URL 或 webhook 服务器 —— 并通过钉钉的会话 webhook API 使用 markdown 格式的消息进行回复。

在设置之前，这里是大多数人想了解的内容：Hermes 在你的钉钉工作空间中是如何行为的。

## Hermes 的行为方式

| 上下文 | 行为 |
|---------|----------|
| **私信（1:1 聊天）** | Hermes 回复每条消息。不需要 `@mention`。每个私信都有自己的会话。 |
| **群聊** | Hermes 在你 `@mention` 它时回复。如果没有提及，Hermes 会忽略该消息。 |
| **多用户共享群** | 默认情况下，Hermes 在群内为每个用户隔离会话历史。同一群里的两个人不会共享一个对话记录，除非你明确禁用该选项。 |

### 钉钉中的会话模型

默认情况下：

- 每个私信都有自己的会话
- 共享群聊中的每个用户在该群内都有自己的会话

这可以通过 `config.yaml` 控制：

```yaml
group_sessions_per_user: true
```

仅当你明确希望整个群共享一个对话时，才将其设置为 `false`：

```yaml
group_sessions_per_user: false
```

本指南将引导你完成完整的设置过程 —— 从创建你的钉钉机器人到发送你的第一条消息。

## 前提条件

安装所需的 Python 包：

```bash
pip install "hermes-agent[dingtalk]"
```

或单独安装：

```bash
pip install dingtalk-stream httpx alibabacloud-dingtalk
```

- `dingtalk-stream` — 钉钉官方 Stream Mode SDK（基于 WebSocket 的实时消息）
- `httpx` — 用于通过会话 webhook 发送回复的异步 HTTP 客户端
- `alibabacloud-dingtalk` — 用于 AI Cards、表情反应和媒体下载的钉钉 OpenAPI SDK

## 步骤 1：创建钉钉应用

1. 前往 [钉钉开发者控制台](https://open-dev.dingtalk.com/)。
2. 使用你的钉钉管理员账号登录。
3. 点击 **应用开发** → **自建应用** → **创建应用 via H5微应用**（或根据你的控制台版本选择 **机器人**）。
4. 填写：
   - **应用名称**：例如 `Hermes Agent`
   - **描述**：可选
5. 创建后，导航到 **凭证与基础信息** 找到你的 **Client ID**（AppKey）和 **Client Secret**（AppSecret）。复制这两个值。

:::warning[凭证仅显示一次]
Client Secret 仅在创建应用时显示一次。如果丢失，你需要重新生成。切勿公开分享这些凭证或将其提交到 Git。
:::

## 步骤 2：启用机器人能力

1. 在应用的设置页面，转到 **添加应用能力** → **机器人**。
2. 启用机器人能力。
3. 在 **消息接收模式** 下，选择 **Stream Mode**（推荐 —— 无需公网 URL）。

:::tip
Stream Mode 是推荐的配置方式。它使用从你的机器发起的长连接 WebSocket，因此你不需要公网 IP、域名或 webhook 端点。它可以在 NAT、防火墙和本地机器上工作。
:::

## 步骤 3：找到你的钉钉用户 ID

Hermes Agent 使用你的钉钉用户 ID 来控制谁可以与机器人交互。钉钉用户 ID 是由你的组织管理员设置的字母数字字符串。

要找到你的 ID：

1. 询问你的钉钉组织管理员 —— 用户 ID 在钉钉管理员控制台的 **通讯录** → **成员** 中配置。
2. 或者，机器人会记录每条 incoming 消息的 `sender_id`。启动 gateway，向机器人发送一条消息，然后检查日志中的你的 ID。

## 步骤 4：配置 Hermes Agent

### 选项 A：交互式设置（推荐）

运行引导式设置命令：

```bash
hermes gateway setup
```

当提示时选择 **DingTalk**。设置向导可以通过两种路径之一进行授权：

- **二维码设备流程（推荐）。** 使用钉钉手机 App 扫描终端中打印的二维码 —— 你的 Client ID 和 Client Secret 会自动返回并写入 `~/.hermes/.env`。无需访问开发者控制台。
- **手动粘贴。** 如果你已有凭证（或扫码不方便），在提示时粘贴你的 Client ID、Client Secret 和允许的用户 ID。

:::note openClaw 品牌披露
由于钉钉的 `verification_uri_complete` 在 API 层硬编码为 openClaw 身份，目前二维码授权使用 `openClaw` 源字符串，直到阿里巴巴/钉钉-Real-AI 在服务器端注册 Hermes 特定的模板。这纯粹是钉钉呈现同意屏幕的方式 —— 你创建的是完全属于你并且对你的租户私有的机器人。
:::

### 选项 B：手动配置

将以下内容添加到你的 `~/.hermes/.env` 文件：

```bash
# 必需
DINGTALK_CLIENT_ID=your-app-key
DINGTALK_CLIENT_SECRET=your-app-secret

# 安全：限制谁可以与机器人交互
DINGTALK_ALLOWED_USERS=user-id-1

# 多个允许的用户（逗号分隔）
# DINGTALK_ALLOWED_USERS=user-id-1,user-id-2

# 可选：群聊门控（镜像 Slack/Telegram/Discord/WhatsApp）
# DINGTALK_REQUIRE_MENTION=true
# DINGTALK_FREE_RESPONSE_CHATS=cidABC==,cidDEF==
# DINGTALK_MENTION_PATTERNS=^小马
# DINGTALK_HOME_CHANNEL=cidXXXX==
# DINGTALK_ALLOW_ALL_USERS=true
```

`~/.hermes/config.yaml` 中的可选行为设置：

```yaml
group_sessions_per_user: true

gateway:
  platforms:
    dingtalk:
      extra:
        # 在群中需要 @mention 后机器人才回复（与 Slack/Telegram/Discord 对等）。
        # 私信忽略此设置 —— 机器人始终在 1:1 聊天中回复。
        require_mention: true

        # 按平台许可名单。设置时，只有这些钉钉用户 ID 可以与机器人交互
        #（与 DINGTALK_ALLOWED_USERS 语义相同，但在此处限定范围而不是在 .env 中）。
        allowed_users:
          - user-id-1
          - user-id-2
```

- `group_sessions_per_user: true` 保持每个参与者的上下文在共享群聊中隔离
- `require_mention: true` 防止机器人回复每条群消息 —— 它只在有人 @提及它时才回答
- `allowed_users` 在 `dingtalk.extra` 下是 `DINGTALK_ALLOWED_USERS` 的替代方案；如果两者都设置，它们会被合并

### 启动 Gateway

配置完成后，启动钉钉 gateway：

```bash
hermes gateway
```

机器人应在几秒钟内连接到钉钉的 Stream Mode。向它发送一条消息 —— 私信或它已被添加的群中 —— 进行测试。

:::tip
你可以在后台运行 `hermes gateway` 或作为 systemd 服务以实现持久运行。详见部署文档。
:::

## 功能

### AI Cards

Hermes 可以使用钉钉 AI Cards 而不是纯 markdown 消息进行回复。卡片提供更丰富、更结构化的显示，并在 agent 生成回复时支持流式更新。

要启用 AI Cards，在 `config.yaml` 中配置卡片模板 ID：

```yaml
platforms:
  dingtalk:
    enabled: true
    extra:
      card_template_id: "your-card-template-id"
```

你可以在钉钉开发者控制台中你的应用的 AI Card 设置下找到你的卡片模板 ID。启用 AI Cards 后，所有回复都作为带有流式文本更新的卡片发送。

### 表情反应

Hermes 会自动为你的消息添加表情反应以显示处理状态：

- 🤔思考中 — 当机器人开始处理你的消息时添加
- 🥳完成 — 当回复完成时添加（替换思考中反应）

这些反应在私信和群聊中都有效。

### 显示设置

你可以独立于其他平台自定义钉钉的显示行为：

```yaml
display:
  platforms:
    dingtalk:
      show_reasoning: false   # 在回复中显示模型推理/思考
      streaming: true         # 启用流式回复（与 AI Cards 配合使用）
      tool_progress: all      # 显示工具执行进度（all/new/off）
      interim_assistant_messages: true  # 显示中间评论消息
```

要禁用工具进度和中间消息以获得更简洁的体验：

```yaml
display:
  platforms:
    dingtalk:
      tool_progress: off
      interim_assistant_messages: false
```

## 故障排除

### 机器人未响应消息

**原因**：机器人能力未启用，或 `DINGTALK_ALLOWED_USERS` 不包含你的用户 ID。

**修复**：验证机器人能力在应用设置中已启用，并且选择了 Stream Mode。检查你的用户 ID 是否在 `DINGTALK_ALLOWED_USERS` 中。重启 gateway。

### "dingtalk-stream not installed" 错误

**原因**：未安装 `dingtalk-stream` Python 包。

**修复**：安装它：

```bash
pip install dingtalk-stream httpx
```

### "DINGTALK_CLIENT_ID and DINGTALK_CLIENT_SECRET required"

**原因**：凭证未在环境或 `.env` 文件中设置。

**修复**：验证 `DINGTALK_CLIENT_ID` 和 `DINGTALK_CLIENT_SECRET` 在 `~/.hermes/.env` 中正确设置。Client ID 是你的 AppKey，Client Secret 是你的 AppSecret，来自钉钉开发者控制台。

### 流断开连接 / 重连循环

**原因**：网络不稳定、钉钉平台维护或凭证问题。

**修复**：适配器会自动以指数退避重新连接（2s → 5s → 10s → 30s → 60s）。检查你的凭证是否有效，以及你的应用未被停用。验证你的网络允许出站 WebSocket 连接。

### 机器人离线

**原因**：Hermes gateway 未运行，或连接失败。

**修复**：检查 `hermes gateway` 是否正在运行。查看终端输出中的错误消息。常见问题：凭证错误、应用被停用、`dingtalk-stream` 或 `httpx` 未安装。

### "No session_webhook available"

**原因**：机器人尝试回复但没有会话 webhook URL。这通常发生在 webhook 过期或在接收消息和发送回复之间机器人重启时。

**修复**：向机器人发送一条新消息 —— 每条 incoming 消息都提供一个新的会话 webhook 用于回复。这是正常的钉钉限制；机器人只能回复它最近收到的消息。

## 安全

:::warning
始终设置 `DINGTALK_ALLOWED_USERS` 以限制谁可以与机器人交互。如果没有它，gateway 默认拒绝所有用户，作为一种安全措施。只添加你信任的人的用户 ID —— 授权用户拥有对 agent 功能的完全访问权限，包括工具使用和系统访问。
:::

有关保护你的 Hermes Agent 部署的更多信息，请参阅 [安全指南](../security.md)。

## 注意事项

- **Stream Mode**：无需公网 URL、域名或 webhook 服务器。连接从你的机器通过 WebSocket 发起，因此它可以工作在 NAT 和防火墙后面。
- **AI Cards**：可选择使用富 AI Cards 而不是纯 markdown 进行回复。通过 `card_template_id` 配置。
- **表情反应**：自动 🤔思考中/🥳完成反应用于显示处理状态。
- **Markdown 回复**：回复格式化为钉钉的 markdown 格式，用于富文本显示。
- **媒体支持**：incoming 消息中的图片和文件会自动解析，并可由视觉工具处理。
- **消息去重**：适配器在 5 分钟窗口内对消息进行去重，以防止重复处理同一条消息。
- **自动重连**：如果流连接断开，适配器会自动以指数退避重新连接。
- **消息长度限制**：回复每条消息限制为 20,000 个字符。较长的回复会被截断。
