---
title: "操作 Teams 会议流水线"
description: "Microsoft Teams 会议流水线的运行手册、上线检查清单及操作员工作表"
---

# 操作 Teams 会议流水线

本指南适用于已通过 [Teams Meetings](/user-guide/messaging/teams-meetings) 启用该功能之后的操作阶段。

本页内容：
- 操作员 CLI 流程
- 日常订阅维护
- 故障排查
- 上线检查
- 上线工作表

## 核心操作员命令

### 验证配置快照

```bash
hermes teams-pipeline validate
```

每次配置变更后首先执行此命令。

### 检查 token 健康状态

```bash
hermes teams-pipeline token-health
hermes teams-pipeline token-health --force-refresh
```

当怀疑 auth（认证）状态过期时，使用 `--force-refresh`。

### 检查订阅

```bash
hermes teams-pipeline subscriptions
```

### 续期即将到期的订阅

```bash
hermes teams-pipeline maintain-subscriptions
hermes teams-pipeline maintain-subscriptions --dry-run
```

### 自动化订阅续期（生产环境必须配置）

**Microsoft Graph 订阅最多 72 小时后过期。** 若无任何续期操作，会议通知将在 3 天后静默停止，流水线看起来像是"故障"。这是所有基于 Graph 的集成中最常见的运维故障模式。

你**必须**按计划运行 `maintain-subscriptions`。从以下三种方式中选择一种：

#### 方式一：Hermes cron（若已运行 Hermes gateway，推荐此方式）

Hermes 内置 cron 调度器。`--no-agent` 模式以脚本作为任务执行（而非使用 LLM），`--script` 必须指向 `~/.hermes/scripts/` 下的文件。首先创建脚本：

```bash
mkdir -p ~/.hermes/scripts
cat > ~/.hermes/scripts/maintain-teams-subscriptions.sh <<'EOF'
#!/usr/bin/env bash
exec hermes teams-pipeline maintain-subscriptions
EOF
chmod +x ~/.hermes/scripts/maintain-teams-subscriptions.sh
```

然后注册一个每 12 小时运行一次的纯脚本 cron 任务（相对于 72 小时过期窗口有 6 倍余量）：

```bash
hermes cron create "0 */12 * * *" \
  --name "teams-pipeline-maintain-subscriptions" \
  --no-agent \
  --script maintain-teams-subscriptions.sh \
  --deliver local
```

验证注册情况并查看下次运行时间：

```bash
hermes cron list
hermes cron status        # 调度器状态
```

#### 方式二：systemd timer（推荐用于 Linux 生产部署）

创建 `/etc/systemd/system/hermes-teams-pipeline-maintain.service`：

```ini
[Unit]
Description=Hermes Teams pipeline subscription maintenance
After=network-online.target

[Service]
Type=oneshot
User=hermes
EnvironmentFile=/etc/hermes/env
ExecStart=/usr/local/bin/hermes teams-pipeline maintain-subscriptions
```

以及 `/etc/systemd/system/hermes-teams-pipeline-maintain.timer`：

```ini
[Unit]
Description=Run Hermes Teams pipeline subscription maintenance every 12 hours

[Timer]
OnBootSec=5min
OnUnitActiveSec=12h
Persistent=true

[Install]
WantedBy=timers.target
```

启用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hermes-teams-pipeline-maintain.timer
systemctl list-timers hermes-teams-pipeline-maintain.timer
```

#### 方式三：普通 crontab

```cron
0 */12 * * * /usr/local/bin/hermes teams-pipeline maintain-subscriptions >> /var/log/hermes/teams-pipeline-maintain.log 2>&1
```

确保 cron 环境中包含 `MSGRAPH_*` 凭据。最简单的方法：在 crontab 调用的包装脚本顶部 source `~/.hermes/.env`。

#### 验证续期是否正常工作

设置好计划任务后，在首次计划运行后检查续期活动：

```bash
hermes teams-pipeline subscriptions   # 应显示 expirationDateTime 已推进
hermes teams-pipeline maintain-subscriptions --dry-run   # 大多数时候应显示"0 expiring soon"
```

如果你发现 Graph webhook 在恰好约 72 小时后神秘地"停止工作"，这是首先要检查的地方：续期任务是否实际运行了？

### 查看最近的任务

```bash
hermes teams-pipeline list
hermes teams-pipeline list --status failed
hermes teams-pipeline show <job-id>
```

### 重放已存储的任务

```bash
hermes teams-pipeline run <job-id>
```

### 干运行会议产物拉取

```bash
hermes teams-pipeline fetch --meeting-id <meeting-id>
hermes teams-pipeline fetch --join-web-url "<join-url>"
```

## 日常运行手册

### 首次设置后

按顺序执行：

```bash
hermes teams-pipeline validate
hermes teams-pipeline token-health --force-refresh
hermes teams-pipeline subscriptions
```

然后触发或等待一个真实的会议事件，并确认：

```bash
hermes teams-pipeline list
hermes teams-pipeline show <job-id>
```

### 每日或定期检查

- 运行 `hermes teams-pipeline maintain-subscriptions --dry-run`
- 检查 `hermes teams-pipeline list --status failed`
- 确认 Teams 投递目标仍为正确的聊天或频道

### 变更 webhook URL 或投递目标前

- 更新公共通知 URL 或 Teams 目标配置
- 运行 `hermes teams-pipeline validate`
- 续期或重新创建受影响的订阅
- 确认新事件落入预期的接收端

## 故障排查

### 未创建任何任务

检查：
- `msgraph_webhook` 是否已启用
- 公共通知 URL 是否指向 `/msgraph/webhook`
- 订阅中的 client state 是否与 `MSGRAPH_WEBHOOK_CLIENT_STATE` 匹配
- 订阅是否在远端仍然存在且未过期

### 任务停留在重试状态或在摘要生成前失败

检查：
- 转录权限及可用性
- 录制权限及产物可用性
- 若启用了录制回退，检查 `ffmpeg` 是否可用
- Graph token 健康状态

### 摘要已生成但未投递到 Teams

检查：
- `platforms.teams.enabled: true`
- `delivery_mode`
- webhook 模式下的 `incoming_webhook_url`
- Graph 模式下的 `chat_id` 或 `team_id` 加 `channel_id`
- 若使用 Graph 发帖，检查 Teams auth 配置

### 重复或意外的重放

检查：
- 是否手动通过 `hermes teams-pipeline run` 重放了任务
- 该会议的 sink 记录是否已存在
- 是否在本地配置中有意启用了重发路径

## 上线检查清单

- [ ] Graph 凭据已存在且正确
- [ ] `msgraph_webhook` 已启用且可从公网访问
- [ ] `MSGRAPH_WEBHOOK_CLIENT_STATE` 已设置且与订阅匹配
- [ ] 转录订阅已创建
- [ ] 若需要 STT 回退，录制订阅已创建
- [ ] 若启用录制回退，`ffmpeg` 已安装
- [ ] Teams 出站投递目标已配置并验证
- [ ] Notion 和 Linear 接收端仅在实际需要时配置
- [ ] `hermes teams-pipeline validate` 返回 OK 快照
- [ ] `hermes teams-pipeline token-health --force-refresh` 执行成功
- [ ] **`maintain-subscriptions` 已配置计划任务**（Hermes cron、systemd timer 或 crontab——参见[自动化订阅续期](#automating-subscription-renewal-required-for-production)）。若未配置，Graph 订阅将在 72 小时内静默过期。
- [ ] 一个真实的端到端会议事件已生成存储任务
- [ ] 至少一条摘要已到达预期的投递接收端

## 投递模式决策指南

| 模式 | 适用场景 | 权衡 |
|------|----------|----------|
| `incoming_webhook` | 仅需简单地向 Teams 发帖 | 配置最简单，控制较少 |
| `graph` | 需要通过 Graph 向频道或聊天发帖 | 控制更多，auth 和目标配置更复杂 |

## 操作员工作表

上线前填写：

| 项目 | 值 |
|------|-------|
| 公共通知 URL | |
| Graph 租户 ID | |
| Graph 客户端 ID | |
| Webhook client state | |
| 转录资源订阅 | |
| 录制资源订阅 | |
| Teams 投递模式 | |
| Teams 聊天 ID 或团队/频道 | |
| Notion 数据库 ID | |
| Linear 团队 ID | |
| Store 路径覆盖（如有） | |
| 每日检查负责人 | |

## 变更审查工作表

变更部署前使用：

| 问题 | 答案 |
|----------|--------|
| 是否正在变更公共 webhook URL？ | |
| 是否正在轮换 Graph 凭据？ | |
| 是否正在变更 Teams 投递模式？ | |
| 是否正在迁移到新的 Teams 聊天或频道？ | |
| 订阅是否需要重新创建或续期？ | |
| 是否需要重新进行端到端验证？ | |

## 相关文档

- [Teams Meetings 设置](/user-guide/messaging/teams-meetings)
- [Microsoft Teams bot 设置](/user-guide/messaging/teams)