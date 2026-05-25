---
sidebar_position: 23
title: "Microsoft Graph Webhook 监听器"
description: "在 Hermes 中接收 Microsoft Graph 变更通知（会议、日历、聊天等）"
---

# Microsoft Graph Webhook 监听器

`msgraph_webhook` gateway 平台是一个入站事件监听器。它是 Hermes 接收来自 Microsoft Graph 的**变更通知**的方式——"一个 Teams 会议已结束"、"此聊天中收到了一条新消息"、"此日历事件已更新"。与 `teams` 平台（用户向其发送消息的聊天机器人）不同——此平台是 M365 告知 Hermes 某事已发生，而非来自用户的消息。

目前主要的消费者是 Teams 会议摘要流水线：Graph 在会议产生转录文本时发出通知，流水线获取该内容，Hermes 将摘要发回 Teams。其他 Graph 资源（`/chats/.../messages`、`/users/.../events`）使用同一监听器——流水线消费者通过各自的 PR 接入。

## 前提条件

- Microsoft Graph 应用凭据——[注册 Microsoft Graph 应用程序](/guides/microsoft-graph-app-registration)
- 一个 Microsoft Graph 可访问的**公开 HTTPS URL**（Graph 不会调用私有端点）。测试时可使用 dev tunnel；生产环境需要具有有效证书的真实域名。
- 一个强共享密钥，用作 `clientState` 的值。使用 `openssl rand -hex 32` 生成，并以 `MSGRAPH_WEBHOOK_CLIENT_STATE` 写入 `~/.hermes/.env`。

## 快速开始

最小化 `~/.hermes/config.yaml`：

```yaml
platforms:
  msgraph_webhook:
    enabled: true
    extra:
      port: 8646
      client_state: "replace-with-a-strong-secret"
      accepted_resources:
        - "communications/onlineMeetings"
```

或通过 `~/.hermes/.env` 中的环境变量（启动时自动合并）：

```bash
MSGRAPH_WEBHOOK_ENABLED=true
MSGRAPH_WEBHOOK_PORT=8646
MSGRAPH_WEBHOOK_CLIENT_STATE=<generate-with-openssl-rand-hex-32>
MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES=communications/onlineMeetings
```

启动 gateway：`hermes gateway run`。监听器暴露以下端点：

- `POST /msgraph/webhook` — 来自 Graph 的变更通知
- `GET /msgraph/webhook?validationToken=...` — Graph 订阅验证握手
- `GET /health` — 就绪探针，包含已接受/重复计数器

将监听器公开暴露（反向代理、dev tunnel、ingress）。Graph 订阅的通知 URL 为你的公开 HTTPS 源地址加上 `/msgraph/webhook`：

```
https://ops.example.com/msgraph/webhook
```

## 配置

所有设置位于 `platforms.msgraph_webhook.extra` 下：

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `host` | `0.0.0.0` | HTTP 监听器的绑定地址。 |
| `port` | `8646` | 绑定端口。 |
| `webhook_path` | `/msgraph/webhook` | Graph POST 请求的 URL 路径。 |
| `health_path` | `/health` | 就绪端点。 |
| `client_state` | — | Graph 在每条通知中回传的共享密钥。使用 `hmac.compare_digest` 进行比较——使用 `openssl rand -hex 32` 生成。 |
| `accepted_resources` | `[]`（接受全部） | Graph 资源路径/模式的白名单。末尾 `*` 作为前缀匹配。可容忍开头的 `/`。示例：`["communications/onlineMeetings", "chats/*/messages"]`。 |
| `max_seen_receipts` | `5000` | 通知 ID 的去重缓存大小。达到上限时淘汰最旧的条目。 |
| `allowed_source_cidrs` | `[]`（允许全部） | 可选的源 IP 白名单。见下文。 |

每个设置也有对应的环境变量（`MSGRAPH_WEBHOOK_*`），在 gateway 启动时合并到配置中——参见[环境变量参考](/reference/environment-variables#microsoft-graph-teams-meetings)。

## 安全加固

### clientState 是主要的认证检查

每条 Graph 通知都包含你在订阅时注册的 `clientState` 字符串。监听器使用时序安全比较拒绝任何 `clientState` 不匹配的通知。这是 Microsoft 的官方机制——请将该值视为强共享密钥。

如果未设置 `client_state`，监听器将接受所有格式正确的 POST 请求。**生产环境中请勿在未设置的情况下运行。**

### 源 IP 白名单（生产部署）

在生产环境中，将监听器限制为 Microsoft 公布的 Graph webhook 源 IP 范围。Microsoft 在 [Office 365 IP 地址和 URL Web 服务](https://learn.microsoft.com/en-us/microsoft-365/enterprise/urls-and-ip-address-ranges)中记录了出口范围。配置方式如下：

```yaml
platforms:
  msgraph_webhook:
    enabled: true
    extra:
      client_state: "..."
      allowed_source_cidrs:
        - "52.96.0.0/14"
        - "52.104.0.0/14"
        # ...添加当前 Microsoft 365 "Common" + "Teams" 类别的出口范围
```

或通过环境变量：

```bash
MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS="52.96.0.0/14,52.104.0.0/14"
```

空白名单 = 接受来自任何地址的请求（默认；保留 dev tunnel 工作流）。无效的 CIDR 字符串会记录警告并被忽略。**请每季度审查 Microsoft IP 列表**——它会变更。

### HTTPS 终止

监听器使用纯 HTTP。在你的反向代理（Caddy、Nginx、Cloudflare Tunnel、AWS ALB）处终止 TLS，并通过本地网络代理到监听器。Graph 拒绝向非 HTTPS 端点投递，因此来自 Graph 的未加密流量不存在可达路径。

### 响应规范

成功时，监听器返回 `202 Accepted` 且响应体为空——内部计数器不会出现在响应中。运维人员可通过 `/health` 观察计数。

状态码说明：

| 结果 | 状态码 |
|------|--------|
| 通知已接受或已去重 | 202 |
| 验证握手（带 `validationToken` 的 GET） | 200（原样回传 token） |
| 批次中所有条目的 clientState 均失败 | 403 |
| JSON 格式错误 / 缺少 `value` 数组 / 未知资源 | 400 |
| 源 IP 不在白名单中 | 403 |
| 不带 `validationToken` 的裸 GET | 400 |

## 故障排查

| 问题 | 检查项 |
|------|--------|
| Graph 订阅验证失败 | 公开 URL 可访问，`/msgraph/webhook` 路径匹配，带 `validationToken` 的 GET 在 10 秒内以 `text/plain` 原样回传 token。 |
| 通知 POST 成功但无内容被摄取 | `client_state` 与订阅时注册的值一致。如值已漂移，重新运行 `openssl rand -hex 32` 并创建新订阅。检查 `accepted_resources` 是否包含 Graph 发送的资源路径。 |
| 每条通知均返回 403 | `clientState` 不匹配（伪造，或订阅时使用了不同的值）。使用 `hermes teams-pipeline subscribe --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE" ...` 重新创建订阅（随流水线运行时 PR 一同发布）。 |
| 监听器已启动，但 `curl http://localhost:8646/health` 挂起 | 端口绑定冲突。检查 `ss -tlnp \| grep 8646`，如有需要更改 `port:`。 |
| 来自 Microsoft 的真实 Graph 请求返回 403 | 源 IP 白名单范围过窄。临时移除 `allowed_source_cidrs`，确认流量正常后，将列表扩展至包含当前 Microsoft 出口范围。 |

## 相关文档

- [注册 Microsoft Graph 应用程序](/guides/microsoft-graph-app-registration) — Azure 应用注册前提条件
- [环境变量 → Microsoft Graph](/reference/environment-variables#microsoft-graph-teams-meetings) — 完整环境变量列表
- [Microsoft Teams 机器人设置](/user-guide/messaging/teams) — 允许用户在 Teams 中与 Hermes 聊天的另一平台