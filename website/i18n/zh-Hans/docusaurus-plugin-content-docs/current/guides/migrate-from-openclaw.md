---
sidebar_position: 10
title: "从 OpenClaw 迁移"
description: "将 OpenClaw / Clawdbot 配置迁移到 Hermes Agent 的完整指南——包括迁移内容、配置键映射及迁移后的检查事项。"
---

# 从 OpenClaw 迁移

`hermes claw migrate` 将你的 OpenClaw（或旧版 Clawdbot/Moldbot）配置导入 Hermes。本指南详细说明迁移内容、配置键映射以及迁移后的验证步骤。

## 快速开始

```bash
# 预览后迁移（始终先显示预览，再要求确认）
hermes claw migrate

# 仅预览，不做任何更改
hermes claw migrate --dry-run

# 完整迁移，包含 API 密钥，跳过确认
hermes claw migrate --preset full --migrate-secrets --yes
```

迁移操作在执行任何更改前，始终会显示完整的导入预览。请检查列表后确认继续。

默认从 `~/.openclaw/` 读取。旧版 `~/.clawdbot/` 或 `~/.moltbot/` 目录会被自动检测，旧版配置文件名（`clawdbot.json`、`moltbot.json`）同理。

## 选项

| 选项 | 说明 |
|--------|-------------|
| `--dry-run` | 仅预览——显示将迁移的内容后停止。 |
| `--preset <name>` | `full`（所有兼容设置）或 `user-data`（排除基础设施配置）。两种预设默认均不导入密钥——需显式传入 `--migrate-secrets`。 |
| `--overwrite` | 冲突时覆盖已有 Hermes 文件（默认：计划存在冲突时拒绝执行）。 |
| `--migrate-secrets` | 包含 API 密钥。即使使用 `--preset full` 也需要显式指定——没有任何预设会静默导入密钥。 |
| `--no-backup` | 跳过迁移前对 `~/.hermes/` 的 zip 快照备份（默认在执行前写入单个还原点归档，位于 `~/.hermes/backups/pre-migration-*.zip`；可通过 `hermes import` 还原）。 |
| `--source <path>` | 自定义 OpenClaw 目录。 |
| `--workspace-target <path>` | `AGENTS.md` 的放置位置。 |
| `--skill-conflict <mode>` | `skip`（默认）、`overwrite` 或 `rename`。 |
| `--yes` | 跳过预览后的确认提示。 |

## 迁移内容

### Persona（角色设定）、记忆与指令

| 内容 | OpenClaw 来源 | Hermes 目标 | 备注 |
|------|----------------|-------------------|-------|
| Persona | `workspace/SOUL.md` | `~/.hermes/SOUL.md` | 直接复制 |
| 工作区指令 | `workspace/AGENTS.md` | `--workspace-target` 中的 `AGENTS.md` | 需要 `--workspace-target` 标志 |
| 长期记忆 | `workspace/MEMORY.md` | `~/.hermes/memories/MEMORY.md` | 解析为条目，与现有内容合并并去重，使用 `§` 分隔符 |
| 用户档案 | `workspace/USER.md` | `~/.hermes/memories/USER.md` | 与记忆相同的条目合并逻辑 |
| 每日记忆文件 | `workspace/memory/*.md` | `~/.hermes/memories/MEMORY.md` | 所有每日文件合并至主记忆 |

工作区文件还会在 `workspace.default/` 和 `workspace-main/` 作为备用路径进行检测（OpenClaw 在近期版本中将 `workspace/` 重命名为 `workspace-main/`，多 Agent 配置下使用 `workspace-{agentId}`）。

### Skills（技能，4 个来源）

| 来源 | OpenClaw 位置 | Hermes 目标 |
|--------|------------------|-------------------|
| 工作区 skills | `workspace/skills/` | `~/.hermes/skills/openclaw-imports/` |
| 托管/共享 skills | `~/.openclaw/skills/` | `~/.hermes/skills/openclaw-imports/` |
| 个人跨项目 skills | `~/.agents/skills/` | `~/.hermes/skills/openclaw-imports/` |
| 项目级共享 skills | `workspace/.agents/skills/` | `~/.hermes/skills/openclaw-imports/` |

Skill 冲突由 `--skill-conflict` 处理：`skip` 保留现有 Hermes skill，`overwrite` 替换，`rename` 创建带 `-imported` 后缀的副本。

### 模型与 Provider 配置

| 内容 | OpenClaw 配置路径 | Hermes 目标 | 备注 |
|------|---------------------|-------------------|-------|
| 默认模型 | `agents.defaults.model` | `config.yaml` → `model` | 可为字符串或 `{primary, fallbacks}` 对象 |
| 自定义 providers | `models.providers.*` | `config.yaml` → `custom_providers` | 映射 `baseUrl`、`apiType`/`api`——同时处理短格式（"openai"、"anthropic"）和带连字符格式（"openai-completions"、"anthropic-messages"、"google-generative-ai"） |
| Provider API 密钥 | `models.providers.*.apiKey` | `~/.hermes/.env` | 需要 `--migrate-secrets`。参见下方 [API 密钥解析](#api-key-resolution) |

### Agent 行为

| 内容 | OpenClaw 配置路径 | Hermes 配置路径 | 映射规则 |
|------|---------------------|-------------------|---------|
| 最大轮次 | `agents.defaults.timeoutSeconds` | `agent.max_turns` | `timeoutSeconds / 10`，上限 200 |
| 详细模式 | `agents.defaults.verboseDefault` | `agent.verbose` | "off" / "on" / "full" |
| 推理强度 | `agents.defaults.thinkingDefault` | `agent.reasoning_effort` | "always"/"high"/"xhigh" → "high"，"auto"/"medium"/"adaptive" → "medium"，"off"/"low"/"none"/"minimal" → "low" |
| 压缩 | `agents.defaults.compaction.mode` | `compression.enabled` | "off" → false，其他 → true |
| 压缩模型 | `agents.defaults.compaction.model` | `compression.summary_model` | 直接字符串复制 |
| 人工延迟 | `agents.defaults.humanDelay.mode` | `human_delay.mode` | "natural" / "custom" / "off" |
| 人工延迟时间 | `agents.defaults.humanDelay.minMs` / `.maxMs` | `human_delay.min_ms` / `.max_ms` | 直接复制 |
| 时区 | `agents.defaults.userTimezone` | `timezone` | 直接字符串复制 |
| 执行超时 | `tools.exec.timeoutSec` | `terminal.timeout` | 直接复制（字段名为 `timeoutSec`，非 `timeout`） |
| Docker 沙箱 | `agents.defaults.sandbox.backend` | `terminal.backend` | "docker" → "docker" |
| Docker 镜像 | `agents.defaults.sandbox.docker.image` | `terminal.docker_image` | 直接复制 |

### 会话重置策略

| OpenClaw 配置路径 | Hermes 配置路径 | 备注 |
|---------------------|-------------------|-------|
| `session.reset.mode` | `session_reset.mode` | "daily"、"idle" 或两者 |
| `session.reset.atHour` | `session_reset.at_hour` | 每日重置的小时（0–23） |
| `session.reset.idleMinutes` | `session_reset.idle_minutes` | 不活跃分钟数 |

注意：OpenClaw 还有 `session.resetTriggers`（简单字符串数组，如 `["daily", "idle"]`）。若结构化的 `session.reset` 不存在，迁移将回退到从 `resetTriggers` 推断。

### MCP 服务器

| OpenClaw 字段 | Hermes 字段 | 备注 |
|----------------|-------------|-------|
| `mcp.servers.*.command` | `mcp_servers.*.command` | stdio 传输 |
| `mcp.servers.*.args` | `mcp_servers.*.args` | |
| `mcp.servers.*.env` | `mcp_servers.*.env` | |
| `mcp.servers.*.cwd` | `mcp_servers.*.cwd` | |
| `mcp.servers.*.url` | `mcp_servers.*.url` | HTTP/SSE 传输 |
| `mcp.servers.*.tools.include` | `mcp_servers.*.tools.include` | 工具过滤 |
| `mcp.servers.*.tools.exclude` | `mcp_servers.*.tools.exclude` | |

### TTS（文字转语音）

TTS 设置从 OpenClaw 配置的**两个**位置读取，优先级如下：

1. `messages.tts.providers.{provider}.*`（规范位置）
2. 顶层 `talk.providers.{provider}.*`（备用）
3. 旧版扁平键 `messages.tts.{provider}.*`（最旧格式）

| 内容 | Hermes 目标 |
|------|-------------------|
| Provider 名称 | `config.yaml` → `tts.provider` |
| ElevenLabs voice ID | `config.yaml` → `tts.elevenlabs.voice_id` |
| ElevenLabs model ID | `config.yaml` → `tts.elevenlabs.model_id` |
| OpenAI 模型 | `config.yaml` → `tts.openai.model` |
| OpenAI 语音 | `config.yaml` → `tts.openai.voice` |
| Edge TTS 语音 | `config.yaml` → `tts.edge.voice`（OpenClaw 将 "edge" 重命名为 "microsoft"——两者均可识别） |
| TTS 资源文件 | `~/.hermes/tts/`（文件复制） |

### 消息平台

| 平台 | OpenClaw 配置路径 | Hermes `.env` 变量 | 备注 |
|----------|---------------------|----------------------|-------|
| Telegram | `channels.telegram.botToken` 或 `.accounts.default.botToken` | `TELEGRAM_BOT_TOKEN` | Token 可为字符串或 [SecretRef](#secretref-handling)，支持扁平和 accounts 两种布局 |
| Telegram | `credentials/telegram-default-allowFrom.json` | `TELEGRAM_ALLOWED_USERS` | 从 `allowFrom[]` 数组逗号拼接 |
| Discord | `channels.discord.token` 或 `.accounts.default.token` | `DISCORD_BOT_TOKEN` | |
| Discord | `channels.discord.allowFrom` 或 `.accounts.default.allowFrom` | `DISCORD_ALLOWED_USERS` | |
| Slack | `channels.slack.botToken` 或 `.accounts.default.botToken` | `SLACK_BOT_TOKEN` | |
| Slack | `channels.slack.appToken` 或 `.accounts.default.appToken` | `SLACK_APP_TOKEN` | |
| Slack | `channels.slack.allowFrom` 或 `.accounts.default.allowFrom` | `SLACK_ALLOWED_USERS` | |
| WhatsApp | `channels.whatsapp.allowFrom` 或 `.accounts.default.allowFrom` | `WHATSAPP_ALLOWED_USERS` | 通过 Baileys 二维码配对认证——迁移后需重新配对 |
| Signal | `channels.signal.account` 或 `.accounts.default.account` | `SIGNAL_ACCOUNT` | |
| Signal | `channels.signal.httpUrl` 或 `.accounts.default.httpUrl` | `SIGNAL_HTTP_URL` | |
| Signal | `channels.signal.allowFrom` 或 `.accounts.default.allowFrom` | `SIGNAL_ALLOWED_USERS` | |
| Matrix | `channels.matrix.accessToken` 或 `.accounts.default.accessToken` | `MATRIX_ACCESS_TOKEN` | 使用 `accessToken`（非 `botToken`） |
| Mattermost | `channels.mattermost.botToken` 或 `.accounts.default.botToken` | `MATTERMOST_BOT_TOKEN` | |

### 其他配置

| 内容 | OpenClaw 路径 | Hermes 路径 | 备注 |
|------|-------------|-------------|-------|
| 审批模式 | `approvals.exec.mode` | `config.yaml` → `approvals.mode` | "auto"→"off"，"always"→"manual"，"smart"→"smart" |
| 命令白名单 | `exec-approvals.json` | `config.yaml` → `command_allowlist` | 模式合并并去重 |
| 浏览器 CDP URL | `browser.cdpUrl` | `config.yaml` → `browser.cdp_url` | |
| 浏览器无头模式 | `browser.headless` | `config.yaml` → `browser.headless` | |
| Brave 搜索密钥 | `tools.web.search.brave.apiKey` | `.env` → `BRAVE_API_KEY` | 需要 `--migrate-secrets` |
| Gateway 认证 token | `gateway.auth.token` | `.env` → `HERMES_GATEWAY_TOKEN` | 需要 `--migrate-secrets` |
| 工作目录 | `agents.defaults.workspace` | `.env` → `MESSAGING_CWD` | |

### 已归档（无对应 Hermes 等效项）

以下内容保存至 `~/.hermes/migration/openclaw/<timestamp>/archive/` 供人工审查：

| 内容 | 归档文件 | 在 Hermes 中的重建方式 |
|------|-------------|--------------------------|
| `IDENTITY.md` | `archive/workspace/IDENTITY.md` | 合并至 `SOUL.md` |
| `TOOLS.md` | `archive/workspace/TOOLS.md` | Hermes 内置工具说明 |
| `HEARTBEAT.md` | `archive/workspace/HEARTBEAT.md` | 使用 cron 作业执行周期性任务 |
| `BOOTSTRAP.md` | `archive/workspace/BOOTSTRAP.md` | 使用上下文文件或 skills |
| Cron 作业 | `archive/cron-config.json` | 通过 `hermes cron create` 重建 |
| 插件 | `archive/plugins-config.json` | 参见 [插件指南](/user-guide/features/hooks) |
| Hooks/webhooks | `archive/hooks-config.json` | 使用 `hermes webhook` 或 gateway hooks |
| 记忆后端 | `archive/memory-backend-config.json` | 通过 `hermes honcho` 配置 |
| Skills 注册表 | `archive/skills-registry-config.json` | 使用 `hermes skills config` |
| UI/身份 | `archive/ui-identity-config.json` | 使用 `/skin` 命令 |
| 日志 | `archive/logging-diagnostics-config.json` | 在 `config.yaml` 日志部分设置 |
| 多 Agent 列表 | `archive/agents-list.json` | 使用 Hermes profiles |
| 频道绑定 | `archive/bindings.json` | 按平台手动配置 |
| 复杂频道配置 | `archive/channels-deep-config.json` | 手动配置各平台 |

## API 密钥解析

启用 `--migrate-secrets` 时，API 密钥按以下优先级从**四个来源**收集：

1. **配置值** — `openclaw.json` 中的 `models.providers.*.apiKey` 及 TTS provider 密钥
2. **环境文件** — `~/.openclaw/.env`（如 `OPENROUTER_API_KEY`、`ANTHROPIC_API_KEY` 等）
3. **配置 env 子对象** — `openclaw.json` → `"env"` 或 `"env"."vars"`（部分配置将密钥存于此处而非单独的 `.env` 文件）
4. **认证档案** — `~/.openclaw/agents/main/agent/auth-profiles.json`（每个 Agent 的凭据）

配置值优先级最高，后续来源依次填补剩余空缺。

### 支持的密钥目标

`OPENROUTER_API_KEY`、`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`DEEPSEEK_API_KEY`、`GEMINI_API_KEY`、`ZAI_API_KEY`、`MINIMAX_API_KEY`、`ELEVENLABS_API_KEY`、`TELEGRAM_BOT_TOKEN`、`VOICE_TOOLS_OPENAI_KEY`

不在此白名单中的密钥一律不会被复制。

## SecretRef 处理

OpenClaw 配置中 token 和 API 密钥的值支持三种格式：

```json
// 纯字符串
"channels": { "telegram": { "botToken": "123456:ABC-DEF..." } }

// 环境变量模板
"channels": { "telegram": { "botToken": "${TELEGRAM_BOT_TOKEN}" } }

// SecretRef 对象
"channels": { "telegram": { "botToken": { "source": "env", "id": "TELEGRAM_BOT_TOKEN" } } }
```

迁移会解析所有三种格式。对于环境变量模板和 `source: "env"` 的 SecretRef 对象，会从 `~/.openclaw/.env` 和 `openclaw.json` 的 env 子对象中查找值。`source: "file"` 或 `source: "exec"` 的 SecretRef 对象无法自动解析——迁移会对此发出警告，相关值需通过 `hermes config set` 手动添加至 Hermes。

## 迁移后

1. **检查迁移报告** — 完成后打印，包含已迁移、已跳过和冲突项的计数。

2. **审查归档文件** — `~/.hermes/migration/openclaw/<timestamp>/archive/` 中的所有内容需要人工处理。

3. **开启新会话** — 导入的 skills 和记忆条目在新会话中生效，当前会话不受影响。

4. **验证 API 密钥** — 运行 `hermes status` 检查 provider 认证状态。

5. **测试消息平台** — 若迁移了平台 token，重启 gateway：`systemctl --user restart hermes-gateway`

6. **检查会话策略** — 验证 `hermes config get session_reset` 是否符合预期。

7. **重新配对 WhatsApp** — WhatsApp 使用二维码配对（Baileys），不支持 token 迁移。运行 `hermes whatsapp` 进行配对。

8. **清理归档** — 确认一切正常后，运行 `hermes claw cleanup` 将残留的 OpenClaw 目录重命名为 `.pre-migration/`（防止状态混淆）。

## 故障排查

### "OpenClaw directory not found"

迁移依次检查 `~/.openclaw/`、`~/.clawdbot/`、`~/.moltbot/`。若你的安装路径不同，请使用 `--source /path/to/your/openclaw`。

### "No provider API keys found"

根据 OpenClaw 版本不同，密钥可能存储在多个位置：`openclaw.json` 中 `models.providers.*.apiKey` 内联、`~/.openclaw/.env`、`openclaw.json` 的 `"env"` 子对象，或 `agents/main/agent/auth-profiles.json`。迁移会检查所有四个位置。若密钥使用 `source: "file"` 或 `source: "exec"` 的 SecretRef，则无法自动解析——请通过 `hermes config set` 手动添加。

### 迁移后 skills 未出现

导入的 skills 位于 `~/.hermes/skills/openclaw-imports/`。开启新会话后生效，或运行 `/skills` 验证是否已加载。

### TTS 语音未迁移

OpenClaw 在两处存储 TTS 设置：`messages.tts.providers.*` 和顶层 `talk` 配置。迁移会检查两处。若你的 voice ID 是通过 OpenClaw UI 设置的（存储路径不同），可能需要手动设置：`hermes config set tts.elevenlabs.voice_id YOUR_VOICE_ID`。