# QQ Bot

通过**官方 QQ Bot API（v2）**将 Hermes 接入 QQ——支持私聊（C2C）、群组 @-提及、频道及直接消息，并具备语音转写功能。

## 概述

QQ Bot 适配器使用[官方 QQ Bot API](https://bot.q.qq.com/wiki/develop/api-v2/) 实现以下功能：

- 通过持久 **WebSocket** 连接至 QQ Gateway（网关）接收消息
- 通过 **REST API** 发送文本和 Markdown 回复
- 下载并处理图片、语音消息及文件附件
- 使用腾讯内置 ASR 或可配置的 STT（语音转文字）提供商转写语音消息

## 前提条件

1. **QQ Bot 应用** — 在 [q.qq.com](https://q.qq.com) 注册：
   - 创建新应用并记录您的 **App ID** 和 **App Secret**
   - 启用所需 intent（意图）：C2C 消息、群组 @-消息、频道消息
   - 在沙盒模式下配置机器人以进行测试，或发布至生产环境

2. **依赖项** — 适配器需要 `aiohttp` 和 `httpx`：
   ```bash
   pip install aiohttp httpx
   ```

## 配置

### 交互式设置

```bash
hermes gateway setup
```

从平台列表中选择 **QQ Bot** 并按提示操作。

### 手动配置

在 `~/.hermes/.env` 中设置所需环境变量：

```bash
QQ_APP_ID=your-app-id
QQ_CLIENT_SECRET=your-app-secret
```

## 环境变量

| 变量 | 描述 | 默认值 |
|---|---|---|
| `QQ_APP_ID` | QQ Bot App ID（必填） | — |
| `QQ_CLIENT_SECRET` | QQ Bot App Secret（必填） | — |
| `QQBOT_HOME_CHANNEL` | 用于 cron/通知投递的 OpenID | — |
| `QQBOT_HOME_CHANNEL_NAME` | 主频道显示名称 | `Home` |
| `QQ_ALLOWED_USERS` | 允许私聊访问的用户 OpenID 列表（逗号分隔） | 开放（所有用户） |
| `QQ_GROUP_ALLOWED_USERS` | 允许群组访问的群组 OpenID 列表（逗号分隔） | — |
| `QQ_ALLOW_ALL_USERS` | 设为 `true` 以允许所有私聊 | `false` |
| `QQ_PORTAL_HOST` | 覆盖 QQ portal 主机（沙盒路由设为 `sandbox.q.qq.com`） | `q.qq.com` |
| `QQ_STT_API_KEY` | 语音转文字提供商的 API 密钥 | — |
| `QQ_STT_BASE_URL` | （不直接读取——请在 `config.yaml` 中设置 `platforms.qqbot.extra.stt.baseUrl`） | n/a |
| `QQ_STT_MODEL` | STT 模型名称 | `glm-asr` |

## 高级配置

如需精细控制，可在 `~/.hermes/config.yaml` 中添加平台设置：

```yaml
platforms:
  qqbot:
    enabled: true
    extra:
      app_id: "your-app-id"
      client_secret: "your-secret"
      markdown_support: true       # enable QQ markdown (msg_type 2). Config-only; no env-var equivalent.
      dm_policy: "open"          # open | allowlist | disabled
      allow_from:
        - "user_openid_1"
      group_policy: "open"       # open | allowlist | disabled（默认：disabled）
      group_allow_from:
        - "group_openid_1"
      stt:
        provider: "zai"          # zai (GLM-ASR), openai (Whisper), etc.
        baseUrl: "https://open.bigmodel.cn/api/coding/paas/v4"
        apiKey: "your-stt-key"
        model: "glm-asr"
```

## 语音消息（STT）

语音转写分两个阶段进行：

1. **QQ 内置 ASR**（免费，始终优先尝试）——QQ 在语音消息附件中提供 `asr_refer_text`，使用腾讯自有语音识别
2. **已配置的 STT 提供商**（备用）——若 QQ 的 ASR 未返回文本，适配器将调用兼容 OpenAI 的 STT API：

   - **智谱/GLM（zai）**：默认提供商，使用 `glm-asr` 模型
   - **OpenAI Whisper**：设置 `QQ_STT_BASE_URL` 和 `QQ_STT_MODEL`
   - 任何兼容 OpenAI 的 STT 端点

## 故障排查

### 机器人立即断开连接（快速断连）

通常原因如下：
- **App ID / Secret 无效** — 在 q.qq.com 仔细核对您的凭据
- **缺少权限** — 确保机器人已启用所需 intent
- **仅限沙盒的机器人** — 若机器人处于沙盒模式，只能接收来自 QQ 沙盒测试频道的消息

### 语音消息未被转写

1. 检查附件数据中是否存在 QQ 内置的 `asr_refer_text`
2. 若使用自定义 STT 提供商，验证 `QQ_STT_API_KEY` 是否正确设置
3. 查看 gateway 日志中的 STT 错误信息

### 消息未送达

- 在 q.qq.com 验证机器人的 **intent** 是否已启用
- 若私聊访问受限，检查 `QQ_ALLOWED_USERS`
- 对于群组消息，确保机器人被 **@提及**（群组策略可能需要加入白名单）
- 检查 `QQBOT_HOME_CHANNEL` 以确认 cron/通知投递配置

### 连接错误

- 确保已安装 `aiohttp` 和 `httpx`：`pip install aiohttp httpx`
- 检查与 `api.sgroup.qq.com` 及 WebSocket gateway 的网络连通性
- 查看 gateway 日志以获取详细错误信息和重连行为