# QQ Bot

通过**官方 QQ 机器人 API (v2)** 将 Hermes 连接到 QQ —— 支持私信（C2C）、群 @-mentions、频道和带语音转录的直接消息。

## 概览

QQ Bot 适配器使用[官方 QQ 机器人 API](https://bot.q.qq.com/wiki/develop/api-v2/)来：

- 通过到 QQ Gateway 的持久 **WebSocket** 连接接收消息
- 通过 **REST API** 发送文本和 markdown 回复
- 下载和处理图片、语音消息和文件附件
- 使用腾讯内置 ASR 或可配置的 STT 提供商转录语音消息

## 前提条件

1. **QQ 机器人应用** — 在 [q.qq.com](https://q.qq.com) 注册：
   - 创建一个新应用并记下你的 **App ID** 和 **App Secret**
   - 启用所需的 intents：C2C 消息、群 @-消息、频道消息
   - 在沙盒模式下配置你的机器人进行测试，或发布用于生产

2. **依赖** — 适配器需要 `aiohttp` 和 `httpx`：
   ```bash
   pip install aiohttp httpx
   ```

## 配置

### 交互式设置

```bash
hermes gateway setup
```

从平台列表中选择 **QQ Bot** 并按照提示操作。

### 手动配置

在 `~/.hermes/.env` 中设置所需的环境变量：

```bash
QQ_APP_ID=your-app-id
QQ_CLIENT_SECRET=your-app-secret
```

## 环境变量

| 变量 | 描述 | 默认 |
|---|---|---|
| `QQ_APP_ID` | QQ 机器人 App ID（必需） | — |
| `QQ_CLIENT_SECRET` | QQ 机器人 App Secret（必需） | — |
| `QQBOT_HOME_CHANNEL` | 用于 cron/通知递送的 OpenID | — |
| `QQBOT_HOME_CHANNEL_NAME` | 主页频道的显示名称 | `Home` |
| `QQ_ALLOWED_USERS` | 用于 DM 访问的逗号分隔的用户 OpenID | open（所有用户） |
| `QQ_GROUP_ALLOWED_USERS` | 用于群组访问的逗号分隔的群 OpenID | — |
| `QQ_ALLOW_ALL_USERS` | 设置为 `true` 以允许所有 DM | `false` |
| `QQ_PORTAL_HOST` | 覆盖 QQ 门户主机（设置为 `sandbox.q.qq.com` 用于沙盒路由） | `q.qq.com` |
| `QQ_STT_API_KEY` | 语音转文本提供商的 API 密钥 | — |
| `QQ_STT_BASE_URL` | STT 提供商的基础 URL | `https://open.bigmodel.cn/api/coding/paas/v4` |
| `QQ_STT_MODEL` | STT 模型名称 | `glm-asr` |

## 高级配置

要进行细粒度控制，请将平台设置添加到 `~/.hermes/config.yaml`：

```yaml
platforms:
  qq:
    enabled: true
    extra:
      app_id: "your-app-id"
      client_secret: "your-secret"
      markdown_support: true       # 启用 QQ markdown（msg_type 2）。仅配置；没有环境变量等效项。
      dm_policy: "open"          # open | allowlist | disabled
      allow_from:
        - "user_openid_1"
      group_policy: "open"       # open | allowlist | disabled
      group_allow_from:
        - "group_openid_1"
      stt:
        provider: "zai"          # zai (GLM-ASR)、openai (Whisper) 等。
        baseUrl: "https://open.bigmodel.cn/api/coding/paas/v4"
        apiKey: "your-stt-key"
        model: "glm-asr"
```

## 语音消息（STT）

语音转录分两个阶段工作：

1. **QQ 内置 ASR**（免费，始终首先尝试）— QQ 在语音消息附件中提供 `asr_refer_text`，使用腾讯自己的语音识别
2. **配置的 STT 提供商**（回退）— 如果 QQ 的 ASR 未返回文本，适配器调用 OpenAI 兼容的 STT API：

   - **智谱/GLM (zai)**：默认提供商，使用 `glm-asr` 模型
   - **OpenAI Whisper**：设置 `QQ_STT_BASE_URL` 和 `QQ_STT_MODEL`
   - 任何 OpenAI 兼容的 STT 端点

## 故障排除

### 机器人立即断开连接（快速断开）

这通常意味着：
- **无效的 App ID / Secret** — 在 q.qq.com 检查你的凭证
- **缺少权限** — 确保机器人已启用所需的 intents
- **仅沙盒机器人** — 如果机器人在沙盒模式，它只能从 QQ 的沙盒测试频道接收消息

### 语音消息未转录

1. 检查 QQ 内置的 `asr_refer_text` 是否存在于附件数据中
2. 如果使用自定义 STT 提供商，验证 `QQ_STT_API_KEY` 设置正确
3. 检查 gateway 日志中的 STT 错误消息

### 消息未送达

- 验证机器人的 **intents** 在 q.qq.com 已启用
- 如果 DM 访问受限，检查 `QQ_ALLOWED_USERS`
- 对于群消息，确保机器人被 **@mentioned**（群组策略可能需要白名单）
- 检查 `QQBOT_HOME_CHANNEL` 用于 cron/通知递送

### 连接错误

- 确保已安装 `aiohttp` 和 `httpx`：`pip install aiohttp httpx`
- 检查到 `api.sgroup.qq.com` 和 WebSocket Gateway 的网络连接
- 查看 gateway 日志以获取详细的错误消息和重连行为
