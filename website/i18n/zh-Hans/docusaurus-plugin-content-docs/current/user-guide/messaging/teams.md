---
sidebar_position: 5
title: "Microsoft Teams"
description: "将 Hermes Agent 配置为 Microsoft Teams 机器人"
---

# Microsoft Teams 设置

将 Hermes Agent 连接到 Microsoft Teams 作为机器人。与 Slack 的 Socket Mode 不同，Teams 通过调用**公共 HTTPS Webhook** 传递消息，因此您的实例需要一个公开可访问的端点——可以是开发隧道（本地开发）或真实域名（生产环境）。

## 机器人响应方式

| 场景 | 行为 |
|---------|----------|
| **私信（DM）** | 机器人回复每条消息。无需 @提及。 |
| **群聊** | 只有 @提及机器人时才会响应。 |
| **频道** | 只有 @提及机器人时才会响应。 |

Teams 将 @提及作为带有 `<at>BotName</at>` 标签的常规消息传递，Hermes 会在处理前自动剥离这些标签。

---

## 步骤 1：安装 Teams CLI

`@microsoft/teams.cli` 可自动化机器人注册——无需 Azure 门户。

```bash
npm install -g @microsoft/teams.cli@preview
teams login
```

验证登录状态并查找您自己的 AAD 对象 ID（用于 `TEAMS_ALLOWED_USERS`）：

```bash
teams status --verbose
```

---

## 步骤 2：暴露 Webhook 端口

Teams 无法向 `localhost` 传递消息。对于本地开发，使用任何隧道工具获取公共 HTTPS URL。默认端口是 `3978`——如需更改可使用 `TEAMS_PORT`。

```bash
# devtunnel（Microsoft）
devtunnel create hermes-bot --allow-anonymous
devtunnel port create hermes-bot -p 3978 --protocol https  # 如果更改了 TEAMS_PORT，请替换 3978
devtunnel host hermes-bot

# ngrok
ngrok http 3978  # 如果更改了 TEAMS_PORT，请替换 3978

# cloudflared
cloudflared tunnel --url http://localhost:3978  # 如果更改了 TEAMS_PORT，请替换 3978
```

从输出中复制 `https://` URL——您将在下一步中使用它。开发期间保持隧道运行。

对于生产环境，请将机器人的端点指向您服务器的公共域名（参见[生产部署](#生产部署)）。

---

## 步骤 3：创建机器人

```bash
teams app create \
  --name "Hermes" \
  --endpoint "https://<your-tunnel-url>/api/messages"
```

CLI 输出您的 `CLIENT_ID`、`CLIENT_SECRET` 和 `TENANT_ID`，以及第 6 步的安装链接。保存客户端密钥——它不会再显示。

---

## 步骤 4：配置环境变量

添加到 `~/.hermes/.env`：

```bash
# 必需
TEAMS_CLIENT_ID=<your-client-id>
TEAMS_CLIENT_SECRET=<your-client-secret>
TEAMS_TENANT_ID=<your-tenant-id>

# 限制特定用户访问（推荐）
# 使用 `teams status --verbose` 中的 AAD 对象 ID
TEAMS_ALLOWED_USERS=<your-aad-object-id>
```

---

## 步骤 5：启动网关

```bash
HERMES_UID=$(id -u) HERMES_GID=$(id -g) docker compose up -d gateway
```

这将启动网关。默认 Webhook 端口是 `3978`（可用 `TEAMS_PORT` 覆盖）。检查是否正在运行：

```bash
curl http://localhost:3978/health   # 应该返回: ok
docker logs -f hermes
```

查找：
```
[teams] Webhook server listening on 0.0.0.0:3978/api/messages
```

---

## 步骤 6：在 Teams 中安装应用

```bash
teams app get <teamsAppId> --install-link
```

在浏览器中打开打印的链接——它直接在 Teams 客户端中打开。安装后，向您的机器人发送一条私信——它已准备就绪。

---

## 配置参考

### 环境变量

| 变量 | 描述 |
|----------|-------------|
| `TEAMS_CLIENT_ID` | Azure AD 应用（客户端）ID |
| `TEAMS_CLIENT_SECRET` | Azure AD 客户端密钥 |
| `TEAMS_TENANT_ID` | Azure AD 租户 ID |
| `TEAMS_ALLOWED_USERS` | 允许使用机器人的逗号分隔的 AAD 对象 ID |
| `TEAMS_ALLOW_ALL_USERS` | 设置 `true` 跳过白名单，允许所有人 |
| `TEAMS_HOME_CHANNEL` | 定时/主动消息传递的对话 ID |
| `TEAMS_HOME_CHANNEL_NAME` | 主页频道的显示名称 |
| `TEAMS_PORT` | Webhook 端口（默认：`3978`） |

### config.yaml

或者，通过 `~/.hermes/config.yaml` 进行配置：

```yaml
platforms:
  teams:
    enabled: true
    extra:
      client_id: "your-client-id"
      client_secret: "your-secret"
      tenant_id: "your-tenant-id"
      port: 3978
```

---

## 功能

### 交互式审批卡片

当代理需要运行潜在危险命令时，它会发送一张包含四个按钮的自适应卡片，而不是让您输入 `/approve`：

- **允许一次** — 批准此特定命令
- **允许本次会话** — 批准此模式在本次会话剩余时间内有效
- **始终允许** — 永久批准此模式
- **拒绝** — 拒绝该命令

点击按钮会内联解决审批并将卡片替换为决策结果。

---

## 生产部署

对于永久服务器，跳过 devtunnel 并使用服务器的公共 HTTPS 端点注册您的机器人：

```bash
teams app create \
  --name "Hermes" \
  --endpoint "https://your-domain.com/api/messages"
```

如果您已经创建了机器人，只需更新端点：

```bash
teams app update --id <teamsAppId> --endpoint "https://your-domain.com/api/messages"
```

确保配置的端口（`TEAMS_PORT`，默认为 `3978`）可以从互联网访问，并且您的 TLS 证书有效——Teams 会拒绝自签名证书。

---

## 故障排除

| 问题 | 解决方案 |
|---------|----------|
| `health` 端点正常但机器人不响应 | 检查隧道是否仍在运行，以及机器人的消息传递端点是否与隧道 URL 匹配 |
| 日志中出现 `KeyError: 'teams'` | 重启容器——当前版本已修复此问题 |
| 机器人返回认证错误 | 验证 `TEAMS_CLIENT_ID`、`TEAMS_CLIENT_SECRET` 和 `TEAMS_TENANT_ID` 都设置正确 |
| `No inference provider configured` | 检查 `~/.hermes/.env` 中是否设置了 `ANTHROPIC_API_KEY`（或其他提供商密钥） |
| 机器人收到消息但不处理 | 您的 AAD 对象 ID 可能不在 `TEAMS_ALLOWED_USERS` 中。运行 `teams status --verbose` 查找 |
| 隧道 URL 在重启后更改 | 如果使用命名隧道（`devtunnel create hermes-bot`），devtunnel URL 是持久的。ngrok 和 cloudflared 除非您有付费计划，否则每次运行都会生成新 URL——URL 更改时使用 `teams app update` 更新机器人端点 |
| Teams 显示"此机器人未响应" | Webhook 返回了错误。检查 `docker logs hermes` 获取追踪信息 |
| 日志中出现 `[teams] Failed to connect` | SDK 认证失败。仔细检查您的凭证以及租户 ID 是否与 `teams login` 使用的账户匹配 |

---

## 安全性

:::warning
**始终设置 `TEAMS_ALLOWED_USERS`** 包含授权用户的 AAD 对象 ID。没有它，任何能找到或安装您的机器人的人都可以与它交互。

像对待密码一样对待 `TEAMS_CLIENT_SECRET`——定期通过 Azure 门户或 Teams CLI 轮换它。
:::

- 将凭证存储在 `~/.hermes/.env` 中，权限设置为 `600`（`chmod 600 ~/.hermes/.env`）
- 机器人只接受来自 `TEAMS_ALLOWED_USERS` 中用户的消息；未授权的消息会被静默丢弃
- 您的公共端点（`/api/messages`）通过 Teams Bot Framework 进行认证——没有有效 JWT 的请求会被拒绝
