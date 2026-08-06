---
sidebar_position: 10
title: "DingTalk"
description: "将 Hermes Agent 设置为钉钉聊天机器人"
---

# 钉钉设置

Hermes Agent 可作为聊天机器人集成到钉钉（DingTalk），让你通过单聊或群聊与 AI 助手对话。机器人通过钉钉的 Stream Mode（流模式）连接——一种长连接 WebSocket，无需公网 URL 或 webhook 服务器——并通过钉钉的 session webhook API 以 markdown 格式回复消息。

在开始设置之前，先了解大多数人最关心的内容：Hermes 进入你的钉钉工作空间后的行为方式。

## Hermes 的行为方式

| 场景 | 行为 |
|---------|----------|
| **单聊（1:1 对话）** | Hermes 响应每条消息，无需 `@提及`，每个单聊有独立会话。 |
| **群聊** | Hermes 仅在被 `@提及` 时响应，未被提及则忽略消息。 |
| **多用户共享群聊** | 默认情况下，Hermes 在群内按用户隔离会话历史。同一群中的两个用户不共享同一对话记录，除非你明确禁用该功能。 |

### 钉钉中的会话模型

默认情况下：

- 每个单聊有独立会话
- 共享群聊中的每个用户在该群内有独立会话

通过 `config.yaml` 控制：

```yaml
group_sessions_per_user: true
```

仅当你明确希望整个群共享一个对话时，才将其设为 `false`：

```yaml
group_sessions_per_user: false
```

本指南将带你完成完整的设置流程——从创建钉钉机器人到发送第一条消息。

## 前置条件

安装所需的 Python 包：

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[dingtalk]"
```

或单独安装：

```bash
pip install dingtalk-stream httpx alibabacloud-dingtalk
```

- `dingtalk-stream` — 钉钉官方 Stream Mode SDK（基于 WebSocket 的实时消息）
- `httpx` — 异步 HTTP 客户端，用于通过 session webhook 发送回复
- `alibabacloud-dingtalk` — 钉钉 OpenAPI SDK，用于 AI 卡片、emoji 反应和媒体下载

## 第一步：创建钉钉应用

1. 前往[钉钉开发者控制台](https://open-dev.dingtalk.com/)。
2. 使用钉钉管理员账号登录。
3. 点击**应用开发** → **自建应用** → **创建 H5 微应用**（或根据控制台版本选择**机器人**）。
4. 填写：
   - **应用名称**：例如 `Hermes Agent`
   - **描述**：可选
5. 创建完成后，进入**凭证与基础信息**，找到你的 **Client ID**（AppKey）和 **Client Secret**（AppSecret），复制两者。

:::warning[凭证仅显示一次]
Client Secret 仅在创建应用时显示一次。如果丢失，需要重新生成。切勿公开分享这些凭证或将其提交到 Git。
:::

## 第二步：启用机器人能力

1. 在应用设置页面，进入**添加能力** → **机器人**。
2. 启用机器人能力。
3. 在**消息接收模式**下，选择 **Stream Mode**（推荐——无需公网 URL）。

:::tip
Stream Mode 是推荐的设置方式。它使用从你的机器发起的长连接 WebSocket，无需公网 IP、域名或 webhook 端点，可在 NAT、防火墙及本地机器后正常工作。
:::

## 第三步：找到你的钉钉用户 ID

Hermes Agent 使用你的钉钉用户 ID 来控制谁可以与机器人交互。钉钉用户 ID 是由组织管理员设置的字母数字字符串。

查找方式：

1. 询问你的钉钉组织管理员——用户 ID 在钉钉管理后台的**通讯录** → **成员**中配置。
2. 或者，机器人会在日志中记录每条传入消息的 `sender_id`。启动 gateway，向机器人发送一条消息，然后在日志中查找你的 ID。

## 第四步：配置 Hermes Agent

### 方式 A：交互式设置（推荐）

运行引导式设置命令：

```bash
hermes gateway setup
```

在提示时选择 **DingTalk**。设置向导支持两种授权路径：

- **二维码设备流（推荐）。** 用钉钉手机 App 扫描终端中打印的二维码——Client ID 和 Client Secret 将自动返回并写入 `~/.hermes/.env`，无需前往开发者控制台。
- **手动粘贴。** 如果你已有凭证（或扫码不方便），在提示时粘贴你的 Client ID、Client Secret 和允许的用户 ID。

:::note openClaw 品牌披露
由于钉钉的 `verification_uri_complete` 在 API 层硬编码为 openClaw 身份，在 Alibaba / DingTalk-Real-AI 在服务端注册 Hermes 专属模板之前，二维码目前以 `openClaw` 来源字符串进行授权。这仅是钉钉呈现授权界面的方式——你创建的机器人完全属于你，且对你的租户私有。
:::

### 方式 B：手动配置

在 `~/.hermes/.env` 文件中添加以下内容：

```bash
# 必填
DINGTALK_CLIENT_ID=your-app-key
DINGTALK_CLIENT_SECRET=your-app-secret

# 安全：限制可与机器人交互的用户
DINGTALK_ALLOWED_USERS=user-id-1

# 多个允许用户（逗号分隔）
# DINGTALK_ALLOWED_USERS=user-id-1,user-id-2

# 可选：群聊门控（与 Slack/Telegram/Discord/WhatsApp 保持一致）
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
        # 在群聊中要求 @提及 后机器人才回复（与 Slack/Telegram/Discord 保持一致）。
        # 单聊忽略此设置——机器人始终在 1:1 对话中回复。
        require_mention: true

        # 平台级白名单。设置后，只有这些钉钉用户 ID 可与机器人交互
        # （语义与 DINGTALK_ALLOWED_USERS 相同，但作用域在此处而非 .env）。
        allowed_users:
          - user-id-1
          - user-id-2
```

- `group_sessions_per_user: true` 在共享群聊中保持每个参与者的上下文隔离
- `require_mention: true` 防止机器人响应每条群消息——仅在有人 @提及 时才回答
- `dingtalk.extra` 下的 `allowed_users` 是 `DINGTALK_ALLOWED_USERS` 的替代方式；若两者同时设置，则合并生效

### 启动 Gateway

配置完成后，启动钉钉 gateway：

```bash
hermes gateway
```

机器人应在几秒内连接到钉钉的 Stream Mode。发送一条消息——单聊或已添加机器人的群聊均可——进行测试。

:::tip
你可以在后台运行 `hermes gateway`，或将其配置为 systemd 服务以持续运行。详见部署文档。
:::

## 功能特性

### AI 卡片

Hermes 可以使用钉钉 AI 卡片代替纯 markdown 消息进行回复。卡片提供更丰富、更结构化的展示，并支持在 agent 生成响应时进行流式更新。

要启用 AI 卡片，在 `config.yaml` 中配置卡片模板 ID：

```yaml
platforms:
  dingtalk:
    enabled: true
    extra:
      card_template_id: "your-card-template-id"
```

你可以在钉钉开发者控制台的应用 AI 卡片设置中找到卡片模板 ID。启用 AI 卡片后，所有回复均以带流式文本更新的卡片形式发送。

### Emoji 反应

Hermes 会自动在你的消息上添加 emoji 反应以显示处理状态：

- 🤔Thinking — 机器人开始处理你的消息时添加
- 🥳Done — 响应完成时添加（替换 Thinking 反应）

这些反应在单聊和群聊中均有效。

### 显示设置

你可以独立于其他平台自定义钉钉的显示行为：

```yaml
display:
  platforms:
    dingtalk:
      show_reasoning: false   # 在回复中显示模型推理/思考过程
      streaming: true         # 启用流式响应（与 AI 卡片配合使用）
      tool_progress: all      # 显示工具执行进度（all/new/off）
      interim_assistant_messages: true  # 显示中间注释消息
```

若要禁用工具进度和中间消息以获得更简洁的体验：

```yaml
display:
  platforms:
    dingtalk:
      tool_progress: off
      interim_assistant_messages: false
```

## 故障排查

### 机器人不响应消息

**原因**：机器人能力未启用，或 `DINGTALK_ALLOWED_USERS` 中不包含你的用户 ID。

**解决方法**：确认应用设置中已启用机器人能力且已选择 Stream Mode。检查你的用户 ID 是否在 `DINGTALK_ALLOWED_USERS` 中。重启 gateway。

### "dingtalk-stream not installed" 错误

**原因**：Python 包 `dingtalk-stream` 未安装。

**解决方法**：安装它：

```bash
pip install dingtalk-stream httpx
```

### "DINGTALK_CLIENT_ID and DINGTALK_CLIENT_SECRET required"

**原因**：凭证未在环境变量或 `.env` 文件中设置。

**解决方法**：确认 `DINGTALK_CLIENT_ID` 和 `DINGTALK_CLIENT_SECRET` 已在 `~/.hermes/.env` 中正确设置。Client ID 是你的 AppKey，Client Secret 是钉钉开发者控制台中的 AppSecret。

### Stream 断开 / 重连循环

**原因**：网络不稳定、钉钉平台维护或凭证问题。

**解决方法**：适配器会以指数退避（2s → 5s → 10s → 30s → 60s）自动重连。检查凭证是否有效，以及应用是否未被停用。确认你的网络允许出站 WebSocket 连接。

### 机器人离线

**原因**：Hermes gateway 未运行，或连接失败。

**解决方法**：检查 `hermes gateway` 是否正在运行。查看终端输出中的错误信息。常见问题：凭证错误、应用被停用、`dingtalk-stream` 或 `httpx` 未安装。

### "No session_webhook available"

**原因**：机器人尝试回复但没有 session webhook URL。通常发生在 webhook 过期或机器人在收到消息和发送回复之间重启的情况下。

**解决方法**：向机器人发送一条新消息——每条传入消息都会提供一个新的 session webhook 用于回复。这是钉钉的正常限制；机器人只能回复最近收到的消息。

## 安全

:::warning
务必设置 `DINGTALK_ALLOWED_USERS` 以限制可与机器人交互的用户。若未设置，gateway 默认拒绝所有用户作为安全措施。只添加你信任的人的用户 ID——已授权用户对 agent 的全部能力拥有完整访问权限，包括工具使用和系统访问。
:::

有关保护 Hermes Agent 部署的更多信息，请参阅[安全指南](../security.md)。

## 注意事项

- **Stream Mode**：无需公网 URL、域名或 webhook 服务器。连接由你的机器通过 WebSocket 发起，可在 NAT 和防火墙后正常工作。
- **AI 卡片**：可选择使用富文本 AI 卡片代替纯 markdown 回复。通过 `card_template_id` 配置。
- **Emoji 反应**：自动添加 🤔Thinking/🥳Done 反应以显示处理状态。
- **Markdown 响应**：回复以钉钉 markdown 格式呈现，支持富文本展示。
- **媒体支持**：传入消息中的图片和文件会自动解析，可由视觉工具处理。
- **消息去重**：适配器在 5 分钟窗口内对消息进行去重，防止同一消息被处理两次。
- **自动重连**：若 stream 连接断开，适配器会以指数退避自动重连。
- **消息长度限制**：每条消息的响应上限为 20,000 个字符，超出部分将被截断。