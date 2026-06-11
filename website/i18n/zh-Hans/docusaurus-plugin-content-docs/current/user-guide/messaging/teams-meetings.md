---
sidebar_position: 6
title: "Teams 会议"
description: "使用 Microsoft Graph webhook 配置 Microsoft Teams 会议摘要流水线"
---

# Microsoft Teams 会议

当你希望 Hermes 接收 Microsoft Graph 会议事件、优先获取转录文本、在无可用转录时回退到录音加 STT（语音转文字），并将结构化摘要输出到下游 sink 时，请使用 Teams 会议流水线。

本页重点介绍配置与启用：
- Graph 凭据
- webhook 监听器配置
- Teams 投递模式
- 流水线配置结构

关于上线后的日常运维、上线检查及运维工作表，请参阅专项指南：[运维 Teams 会议流水线](/guides/operate-teams-meeting-pipeline)。

## 功能说明

该流水线：
1. 接收 Microsoft Graph webhook 事件
2. 解析会议并优先使用转录文件
3. 在无可用转录时回退到录音下载加 STT
4. 在本地存储持久化任务状态和 sink 记录
5. 可将摘要写入 Notion、Linear 和 Microsoft Teams

运维操作通过 CLI 完成（`teams-pipeline` 子命令由 `teams_pipeline` 插件注册——通过 `hermes plugins enable teams_pipeline` 启用，或在 `config.yaml` 中设置 `plugins.enabled: [teams_pipeline]`）：

```bash
hermes teams-pipeline validate
hermes teams-pipeline list
hermes teams-pipeline maintain-subscriptions
```

## 前提条件

启用会议流水线前，请确保已具备：

- 可正常运行的 Hermes 安装
- 若需要 Teams 出站投递，需完成现有的 [Microsoft Teams bot 配置](/user-guide/messaging/teams)
- 具备订阅所需会议资源权限的 Microsoft Graph 应用凭据
- Microsoft Graph 可调用的公网 HTTPS URL，用于 webhook 投递
- 若需要录音加 STT 回退，需安装 `ffmpeg`

## 第一步：添加 Microsoft Graph 凭据

将 Graph 应用凭据添加到 `~/.hermes/.env`：

```bash
MSGRAPH_TENANT_ID=<tenant-id>
MSGRAPH_CLIENT_ID=<client-id>
MSGRAPH_CLIENT_SECRET=<client-secret>
```

这些凭据用于：
- Graph 客户端基础层
- 订阅维护命令
- 会议解析和文件获取
- 未提供专用 Teams 访问令牌时，通过 Graph 进行 Teams 出站投递

## 第二步：启用 Graph Webhook 监听器

webhook 监听器是一个名为 `msgraph_webhook` 的 gateway 平台。至少需要启用它并设置一个 client state 值：

```bash
MSGRAPH_WEBHOOK_ENABLED=true
MSGRAPH_WEBHOOK_PORT=8646
MSGRAPH_WEBHOOK_CLIENT_STATE=<random-shared-secret>
MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES=communications/onlineMeetings
```

监听器暴露以下端点：
- `/msgraph/webhook` 用于接收 Graph 通知
- `/health` 用于简单健康检查

你需要将公网 HTTPS 端点路由到该监听器。例如，若你的公网域名为 `https://ops.example.com`，Graph 通知 URL 通常为：

```text
https://ops.example.com/msgraph/webhook
```

## 第三步：配置 Teams 投递与流水线行为

会议流水线从现有的 `teams` 平台条目读取运行时配置。流水线专属参数位于 `teams.extra.meeting_pipeline` 下。Teams 出站投递仍使用常规 Teams 平台配置。

`~/.hermes/config.yaml` 示例：

```yaml
platforms:
  msgraph_webhook:
    enabled: true
    extra:
      port: 8646
      client_state: "replace-me"
      accepted_resources:
        - "communications/onlineMeetings"

  teams:
    enabled: true
    extra:
      client_id: "your-teams-client-id"
      client_secret: "your-teams-client-secret"
      tenant_id: "your-teams-tenant-id"

      # outbound summary delivery
      delivery_mode: "graph" # or incoming_webhook
      team_id: "team-id"
      channel_id: "channel-id"
      # incoming_webhook_url: "https://..."

      meeting_pipeline:
        transcript_min_chars: 80
        transcript_required: false
        transcription_fallback: true
        ffmpeg_extract_audio: true
        notion:
          enabled: false
        linear:
          enabled: false
```

## Teams 投递模式

流水线在现有 Teams 插件内支持两种 Teams 摘要投递模式。

### `incoming_webhook`

当你希望通过简单的 webhook 将消息发送到 Teams，而无需通过 Graph 创建频道消息时，使用此模式。

所需配置：

```yaml
platforms:
  teams:
    enabled: true
    extra:
      delivery_mode: "incoming_webhook"
      incoming_webhook_url: "https://..."
```

### `graph`

当你希望 Hermes 通过 Microsoft Graph 将摘要发送到 Teams 聊天或频道时，使用此模式。

支持的目标：
- `chat_id`
- `team_id` + `channel_id`
- 现有 Teams 平台的 `team_id` + `home_channel` 回退

示例：

```yaml
platforms:
  teams:
    enabled: true
    extra:
      delivery_mode: "graph"
      team_id: "team-id"
      channel_id: "channel-id"
```

## 第四步：启动 Gateway

更新配置后正常启动 Hermes：

```bash
hermes gateway run
```

若你在 Docker 中运行 Hermes，按现有部署方式启动 gateway 即可。

检查监听器：

```bash
curl http://localhost:8646/health
```

## 第五步：创建 Graph 订阅

使用插件 CLI 创建和查看订阅。

示例：

```bash
hermes teams-pipeline subscribe \
  --resource communications/onlineMeetings/getAllTranscripts \
  --notification-url https://ops.example.com/msgraph/webhook \
  --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE"

hermes teams-pipeline subscribe \
  --resource communications/onlineMeetings/getAllRecordings \
  --notification-url https://ops.example.com/msgraph/webhook \
  --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE"
```

:::warning Graph 订阅在 72 小时后过期

Microsoft Graph 将 webhook 订阅上限设为 72 小时，且不会自动续期。你**必须**在上线前调度 `hermes teams-pipeline maintain-subscriptions`，否则通知将在手动创建订阅三天后静默停止。请参阅运维手册中的[自动化订阅续期](/guides/operate-teams-meeting-pipeline#automating-subscription-renewal-required-for-production)——提供三种方案（Hermes cron、systemd timer、普通 crontab）。

:::

关于订阅维护和上线后的运维流程，请继续阅读指南：[运维 Teams 会议流水线](/guides/operate-teams-meeting-pipeline)。

## 验证

运行内置验证快照：

```bash
hermes teams-pipeline validate
```

常用辅助检查：

```bash
hermes teams-pipeline token-health
hermes teams-pipeline subscriptions
```

## 故障排查

| 问题 | 检查项 |
|---------|---------------|
| Graph webhook 验证失败 | 确认公网 URL 正确且可访问，并确认 Graph 调用的路径为 `/msgraph/webhook` |
| `hermes teams-pipeline list` 中未出现任务 | 确认 `msgraph_webhook` 已启用，且订阅指向正确的通知 URL |
| 转录优先从未成功 | 检查转录资源的 Graph 权限，以及该会议是否存在转录文件 |
| 录音回退失败 | 确认已安装 `ffmpeg`，且 Graph 应用可访问录音文件 |
| Teams 摘要投递失败 | 重新检查 `delivery_mode`、目标 ID 及 Teams 认证配置 |

## 相关文档

- [Microsoft Teams bot 配置](/user-guide/messaging/teams)
- [运维 Teams 会议流水线](/guides/operate-teams-meeting-pipeline)