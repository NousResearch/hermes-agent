---
sidebar_position: 15
---

# 企业微信回调（自建应用）

将 Hermes 连接到企业微信（WeCom）作为使用回调/Webhook 模型的自建企业应用。

:::info 企业微信机器人 vs 企业微信回调
Hermes 支持两种企业微信集成模式：
- **[企业微信机器人](wecom.md)** — 机器人风格，通过 WebSocket 连接。设置更简单，可在群聊中使用。
- **企业微信回调**（本页）— 自建应用，接收加密的 XML 回调。在用户的企业微信侧边栏中显示为一类应用。支持多企业路由。
:::

## 工作原理

1. 您在企业微信管理后台注册一个自建应用
2. 企业微信将加密的 XML 推送到您的 HTTP 回调端点
3. Hermes 解密消息并将其加入代理队列
4. 立即确认（静默 — 不向用户显示任何内容）
5. 代理处理请求（通常需要 3–30 分钟）
6. 回复通过企业微信 `message/send` API 主动推送

## 前置条件

- 具有管理员访问权限的企业微信企业账户
- `aiohttp` 和 `httpx` Python 包（默认安装已包含）
- 用于回调 URL 的公开可访问服务器（或像 ngrok 这样的隧道）

## 设置

### 1. 在企业微信中创建自建应用

1. 进入[企业微信管理后台](https://work.weixin.qq.com/) → **应用管理** → **创建应用**
2. 记下您的**企业 ID**（显示在管理后台顶部）
3. 在应用设置中，创建一个**应用密钥**
4. 从应用的概述页面记下**应用 AgentId**
5. 在**接收消息**下，配置回调 URL：
   - URL：`http://YOUR_PUBLIC_IP:8645/wecom/callback`
   - Token：生成一个随机 Token（企业微信提供）
   - EncodingAESKey：生成一个密钥（企业微信提供）

### 2. 配置环境变量

添加到您的 `.env` 文件：

```bash
WECOM_CALLBACK_CORP_ID=your-corp-id
WECOM_CALLBACK_CORP_SECRET=your-corp-secret
WECOM_CALLBACK_AGENT_ID=1000002
WECOM_CALLBACK_TOKEN=your-callback-token
WECOM_CALLBACK_ENCODING_AES_KEY=your-43-char-aes-key

# 可选
WECOM_CALLBACK_HOST=0.0.0.0
WECOM_CALLBACK_PORT=8645
WECOM_CALLBACK_ALLOWED_USERS=user1,user2
```

### 3. 启动网关

```bash
hermes gateway
```

（仅在 `hermes gateway install` 注册了 systemd/launchd 服务后才使用 `hermes gateway start`。）

回调适配器在配置的端口上启动 HTTP 服务器。企业微信将通过 GET 请求验证回调 URL，然后开始通过 POST 发送消息。

## 配置参考

在 `config.yaml` 中的 `platforms.wecom_callback.extra` 下设置，或使用环境变量：

| 设置 | 默认值 | 描述 |
|---------|---------|-------------|
| `corp_id` | — | 企业微信企业 Corp ID（必需） |
| `corp_secret` | — | 自建应用的企业密钥（必需） |
| `agent_id` | — | 自建应用的 Agent ID（必需） |
| `token` | — | 回调验证 Token（必需） |
| `encoding_aes_key` | — | 回调加密的 43 字符 AES 密钥（必需） |
| `host` | `0.0.0.0` | HTTP 回调服务器的绑定地址 |
| `port` | `8645` | HTTP 回调服务器的端口 |
| `path` | `/wecom/callback` | 回调端点的 URL 路径 |

## 多应用路由

对于运行多个自建应用的企业（例如跨不同部门或子公司），在 `config.yaml` 中配置 `apps` 列表：

```yaml
platforms:
  wecom_callback:
    enabled: true
    extra:
      host: "0.0.0.0"
      port: 8645
      apps:
        - name: "dept-a"
          corp_id: "ww_corp_a"
          corp_secret: "secret-a"
          agent_id: "1000002"
          token: "token-a"
          encoding_aes_key: "key-a-43-chars..."
        - name: "dept-b"
          corp_id: "ww_corp_b"
          corp_secret: "secret-b"
          agent_id: "1000003"
          token: "token-b"
          encoding_aes_key: "key-b-43-chars..."
```

用户按 `corp_id:user_id` 范围划分以防止跨企业冲突。当用户发送消息时，适配器记录他们属于哪个应用（企业）并通过正确的应用访问令牌路由回复。

## 访问控制

限制哪些用户可以与应用交互：

```bash
# 白名单特定用户
WECOM_CALLBACK_ALLOWED_USERS=zhangsan,lisi,wangwu

# 或允许所有用户
WECOM_CALLBACK_ALLOW_ALL_USERS=true
```

## 端点

适配器暴露：

| 方法 | 路径 | 用途 |
|--------|------|----------|
| GET | `/wecom/callback` | URL 验证握手（企业微信在设置期间发送） |
| POST | `/wecom/callback` | 加密消息回调（企业微信在此发送用户消息） |
| GET | `/health` | 健康检查 — 返回 `{"status": "ok"}` |

## 加密

所有回调负载都使用 EncodingAESKey 通过 AES-CBC 加密。适配器处理：

- **入站**：解密 XML 负载，验证 SHA1 签名
- **出站**：回复通过主动 API 发送（不加密回调响应）

加密实现与腾讯官方 WXBizMsgCrypt SDK 兼容。

## 限制

- **不支持流式传输** — 回复在代理完成后作为完整消息到达
- **不支持打字指示器** — 回调模型不支持打字状态
- **仅支持文本** — 目前支持文本消息输入；图片/文件/语音输入尚未实现。代理通过企业微信平台提示了解出站媒体能力（图片、文档、视频、语音）。
- **响应延迟** — 代理会话需要 3–30 分钟；用户看到处理完成时的回复
