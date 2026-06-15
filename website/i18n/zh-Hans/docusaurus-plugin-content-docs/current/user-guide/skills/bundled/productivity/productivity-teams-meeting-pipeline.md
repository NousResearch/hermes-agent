---
title: "Teams Meeting Pipeline"
sidebar_label: "Teams Meeting Pipeline"
description: "通过 Hermes CLI 操作 Teams 会议摘要流水线 — 总结会议、检查流水线状态、重放任务、管理 Microsoft Graph 订阅"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Teams Meeting Pipeline

通过 Hermes CLI 操作 Teams 会议摘要流水线 — 总结会议、检查流水线状态、重放任务、管理 Microsoft Graph 订阅。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/productivity/teams-meeting-pipeline` |
| 版本 | `1.1.0` |
| 作者 | Hermes Agent + Teknium |
| 许可证 | MIT |
| 标签 | `Teams`, `Microsoft Graph`, `Meetings`, `Productivity`, `Operations` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Teams Meeting Pipeline

当用户询问 Microsoft Teams 会议摘要、转录文本、录制内容、行动项、Graph 订阅，或任何与 Teams 会议流水线相关的运维问题时，使用此 skill。支持任意语言 — 以下触发示例并非完整列表。

所有面向运维人员的操作均通过终端工具执行 `hermes teams-pipeline` 子命令完成。此流水线没有新的模型工具 — CLI 是唯一操作界面。

## 使用场景

用户希望：
- 总结 Teams 会议 / 提取行动项 / 获取会议记录
- 检查流水线状态、查看已存储的会议任务，或查看近期会议
- 重放 / 重新运行失败或需要重新生成摘要的已存储任务
- 在更改环境变量或配置后验证 Microsoft Graph 设置
- 排查"会议摘要未送达"或"新会议未被采集"等问题
- 管理 Graph webhook 订阅（创建、续期、删除、查看）
- 设置自动订阅续期（参见下方注意事项）

多语言触发示例（非完整列表）：
- 英语："summarize the Teams meeting"、"pipeline status"、"replay job X"
- 土耳其语："Teams meeting özetle"、"action item çıkar"、"toplantı notu"、"pipeline durumu"、"replay job"

## 前置条件

使用流水线前，请确认以下变量已在 `~/.hermes/.env` 中设置：

```bash
MSGRAPH_TENANT_ID=...
MSGRAPH_CLIENT_ID=...
MSGRAPH_CLIENT_SECRET=...
```

如有缺失，请将用户引导至 `/docs/guides/microsoft-graph-app-registration` 的 Azure 应用注册指南 — 流水线正常运行需要一个已获得管理员授权的 Azure AD 应用注册，并配置相应的 Graph 应用权限。

## 命令参考

### 状态与检查（从这里开始）

```bash
hermes teams-pipeline validate              # 配置快照 — 每次变更后首先运行
hermes teams-pipeline token-health          # Graph token 状态
hermes teams-pipeline token-health --force-refresh   # 强制重新获取 token
hermes teams-pipeline list                  # 近期会议任务
hermes teams-pipeline list --status failed  # 仅显示失败任务
hermes teams-pipeline show <job-id>         # 查看某个任务的完整详情
hermes teams-pipeline subscriptions         # 当前 Graph webhook 订阅
```

### 重新运行 / 调试

```bash
hermes teams-pipeline run <job-id>          # 重放已存储任务（重新生成摘要并重新投递）
hermes teams-pipeline fetch --meeting-id <id>   # 试运行：解析会议及转录文本，不持久化
hermes teams-pipeline fetch --join-web-url "<url>"   # 通过加入链接进行试运行
```

### 订阅管理

```bash
hermes teams-pipeline subscribe \
  --resource communications/onlineMeetings/getAllTranscripts \
  --notification-url https://<your-public-host>/msgraph/webhook \
  --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE"

hermes teams-pipeline renew-subscription <sub-id> --expiration <iso-8601>
hermes teams-pipeline delete-subscription <sub-id>
hermes teams-pipeline maintain-subscriptions            # 续期即将到期的订阅
hermes teams-pipeline maintain-subscriptions --dry-run  # 显示将被续期的内容
```

## 常见问题决策树

- 用户问"为什么今天的会议没有收到摘要？" → 先执行 `list --status failed`，再对相关行执行 `show <job-id>`。如果任务根本不存在，检查 `subscriptions` — webhook 可能已过期（参见下方注意事项）。
- 用户问"设置是否正常？" → 依次执行 `validate`、`token-health`、`subscriptions`。三项均通过后，发起一次测试会议，并检查 `list` 是否出现新行。
- 用户问"重新运行会议 X 的摘要" → 执行 `list` 找到任务 ID，执行 `run <job-id>` 进行重放。若再次失败，执行 `show <job-id>` 查看错误，并用 `fetch --meeting-id` 对制品解析进行试运行。
- 用户问"将会议 X 加入流水线" → 通常无需手动操作 — 流水线由订阅驱动，而非按单次会议触发。如果用户希望对某个历史会议生成摘要，使用 `fetch` 拉取转录文本，并在任务创建后执行 `run`。

## 关键注意事项：Graph 订阅 72 小时后过期

Microsoft Graph 将 webhook 订阅上限设为 72 小时，且**不会自动续期**。如果未调度 `maintain-subscriptions`，手动创建订阅 3 天后会议通知将静默停止。

当用户反馈"昨天流水线还正常，今天没有任何内容进来"时：
1. 执行 `hermes teams-pipeline subscriptions` — 如果结果为空，或所有条目的 `expirationDateTime` 均已过期，即为原因所在。
2. 按上方示例使用 `subscribe` 重新创建订阅。
3. **立即设置自动续期**，可通过 `hermes cron add`、systemd timer 或普通 crontab 实现。运维手册 `/docs/guides/operate-teams-meeting-pipeline#automating-subscription-renewal-required-for-production` 提供了三种方案的完整说明。12 小时间隔是安全的（相对 72 小时上限有 6 倍余量）。

## 其他注意事项

- **转录文本尚未就绪。** Teams 在会议结束后需要一段时间才能生成转录制品。对刚结束的会议执行 `fetch --meeting-id` 可能返回空结果。等待 2-5 分钟后重试，或让 Graph webhook 自然驱动采集。
- **投递模式不匹配。** 如果摘要已生成（`list` 显示成功）但 Teams 中未收到任何内容，检查 `platforms.teams.extra.delivery_mode` 及对应的目标配置（`incoming_webhook_url` 或 `chat_id` 或 `team_id`+`channel_id`）。写入器从 config.yaml 或 `TEAMS_*` 环境变量中读取这些配置。
- **Graph 应用权限。** token 获取正常（`token-health` 通过），但 Graph API 调用返回 401/403，原因是权限已添加但未重新授予管理员同意。请用户重新进入 Azure 门户中的应用注册页面，再次点击"授予管理员同意"。

## 相关文档

当用户需要比本 skill 更深入的内容时，请将其引导至以下资源：
- Azure 应用注册操作指南：`/docs/guides/microsoft-graph-app-registration`
- 完整流水线设置：`/docs/user-guide/messaging/teams-meetings`
- 运维手册（续期自动化、故障排查、上线检查清单）：`/docs/guides/operate-teams-meeting-pipeline`
- Webhook 监听器设置：`/docs/user-guide/messaging/msgraph-webhook`