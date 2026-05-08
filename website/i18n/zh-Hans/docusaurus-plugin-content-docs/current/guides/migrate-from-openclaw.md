---
sidebar_position: 10
title: "从 OpenClaw 迁移"
description: "从 OpenClaw / Clawdbot 完整迁移指南 — 迁移内容、配置映射及迁移后检查事项"
---

# 从 OpenClaw 迁移

`hermes claw migrate` 将你的 OpenClaw（或旧版 Clawdbot/Moldbot）配置导入 Hermes。本指南涵盖迁移内容、配置键映射及迁移后验证事项。

## 快速开始

```bash
# 预览然后迁移（始终先显示预览，再询问确认）
hermes claw migrate

# 仅预览，不做任何更改
hermes claw migrate --dry-run

# 完整迁移包括 API key，跳过确认
hermes claw migrate --preset full --migrate-secrets --yes
```

迁移前始终显示完整的预览列表。审核后确认继续。

默认从 `~/.openclaw/` 读取。旧版 `~/.clawdbot/` 或 `~/.moltbot/` 目录会被自动检测。同样的逻辑适用于旧版配置文件名（`clawdbot.json`、`moltbot.json`）。

## 选项

| 选项 | 说明 |
|--------|-------------|
| `--dry-run` | 仅预览 — 显示迁移内容后停止。 |
| `--preset <name>` | `full`（所有兼容设置）或 `user-data`（不含基础设施配置）。默认预设不导入密钥 — 需要显式传入 `--migrate-secrets`。 |
| `--overwrite` | 覆盖冲突的现有 Hermes 文件（默认：计划有冲突时拒绝应用）。 |
| `--migrate-secrets` | 包含 API key。即使使用 `--preset full` 也需要此选项 — 没有预设会静默导入密钥。 |
| `--no-backup` | 跳过迁移前的 `~/.hermes/` zip 快照（默认会在应用前写入单个恢复点存档，路径为 `~/.hermes/backups/pre-migration-*.zip`；可用 `hermes import` 恢复）。 |
| `--source <path>` | 自定义 OpenClaw 目录。 |
| `--workspace-target <path>` | 放置 `AGENTS.md` 的位置。 |
| `--skill-conflict <mode>` | `skip`（默认）、`overwrite` 或 `rename`。 |
| `--yes` | 跳过预览后的确认提示。 |

## 迁移内容

### 人物设定、记忆和指令

| 项目 | OpenClaw 源 | Hermes 目标 | 说明 |
|------|----------------|-------------------|-------|
| Persona | `workspace/SOUL.md` | `~/.hermes/SOUL.md` | 直接复制 |
| 工作区指令 | `workspace/AGENTS.md` | `--workspace-target` 下的 `AGENTS.md` | 需要 `--workspace-target` 参数 |
| 长期记忆 | `workspace/MEMORY.md` | `~/.hermes/memories/MEMORY.md` | 解析为条目，与现有内容合并去重。使用 `§` 分隔符。 |
| 用户档案 | `workspace/USER.md` | `~/.hermes/memories/USER.md` | 与记忆相同的条目合并逻辑。 |
| 每日记忆文件 | `workspace/memory/*.md` | `~/.hermes/memories/MEMORY.md` | 所有每日文件合并到主记忆。 |

工作区文件也会检查 `workspace.default/` 和 `workspace-main/` 作为备用路径（OpenClaw 在最新版本中将 `workspace/` 重命名为 `workspace-main/`，并使用 `workspace-{agentId}` 用于多智能体设置）。

### Skills（4 个来源）

| 来源 | OpenClaw 路径 | Hermes 目标 |
|--------|------------------|-------------------|
| 工作区 skills | `workspace/skills/` | `~/.hermes/skills/openclaw-imports/` |
| 托管/共享 skills | `~/.openclaw/skills/` | `~/.hermes/skills/openclaw-imports/` |
| 个人跨项目 | `~/.agents/skills/` | `~/.hermes/skills/openclaw-imports/` |
| 项目级共享 | `workspace/.agents/skills/` | `~/.hermes/skills/openclaw-imports/` |

Skill 冲突由 `--skill-conflict` 处理：`skip` 保留现有 Hermes skill，`overwrite` 替换它，`rename` 创建 `-imported` 副本。

### 模型和提供商配置

| 项目 | OpenClaw 配置路径 | Hermes 目标 | 说明 |
|------|---------------------|-------------------|-------|
| 默认模型 | `agents.defaults.model` | `config.yaml` → `model` | 可以是字符串或 `{primary, fallbacks}` 对象 |
| 自定义提供商 | `models.providers.*` | `config.yaml` → `custom_providers` | 映射 `baseUrl`、`apiType`/`api` — 同时处理简写（"openai"、"anthropic"）和连字符形式（"openai-completions"、"anthropic-messages"、"google-generative-ai"） |
| 提供商 API key | `models.providers.*.apiKey` | `~/.hermes/.env` | 需要 `--migrate-secrets`。见下方 [API key 解析](#api-key-resolution)。 |

### 智能体行为

| 项目 | OpenClaw 配置路径 | Hermes 配置路径 | 映射规则 |
|------|---------------------|-------------------|---------|
| 最大轮次 | `agents.defaults.timeoutSeconds` | `agent.max_turns` | `timeoutSeconds / 10`，上限 200 |
| 详细模式 | `agents.defaults.verboseDefault` | `agent.verbose` | "off" / "on" / "full" |
| 推理强度 | `agents.defaults.thinkingDefault` | `agent.reasoning_effort` | "always"/"high"/"xhigh" → "high"，"auto"/"medium"/"adaptive" → "medium"，"off"/"low"/"none"/"minimal" → "low" |
| 压缩 | `agents.defaults.compaction.mode` | `compression.enabled` | "off" → false，其他 → true |
| 压缩模型 | `agents.defaults.compaction.model` | `compression.summary_model` | 直接字符串复制 |
| 人工延迟 | `agents.defaults.humanDelay.mode` | `human_delay.mode` | "natural" / "custom" / "off" |
| 人工延迟时间 | `agents.defaults.humanDelay.minMs` / `.maxMs` | `human_delay.min_ms` / `.max_ms` | 直接复制 |
| 时区 | `agents.defaults.userTimezone` | `timezone` | 直接字符串复制 |
| 执行超时 | `tools.exec.timeoutSec` | `terminal.timeout` | 直接复制（字段是 `timeoutSec`，不是 `timeout`） |
| Docker 沙箱 | `agents.defaults.sandbox.backend` | `terminal.backend` | "docker" → "docker" |
| Docker 镜像 | `agents.defaults.sandbox.docker.image` | `terminal.docker_image` | 直接复制 |

### 会话重置策略

| OpenClaw 配置路径 | Hermes 配置路径 | 说明 |
|---------------------|-------------------|-------|
| `session.reset.mode` | `session_reset.mode` | "daily"、"idle" 或两者 |
| `session.reset.atHour` | `session_reset.at_hour` | 每日重置的小时（0–23） |
| `session.reset.idleMinutes` | `session_reset.idle_minutes` | 空闲分钟数 |

注意：OpenClaw 也有 `session.resetTriggers`（简单字符串数组如 `["daily", "idle"]`）。如果没有结构化的 `session.reset`，迁移会回退到从 `resetTriggers` 推断。

### MCP 服务器

| OpenClaw 字段 | Hermes 字段 | 说明 |
|----------------|-------------|-------|
| `mcp.servers.*.command` | `mcp_servers.*.command` | Stdio 传输 |
| `mcp.servers.*.args` | `mcp_servers.*.args` | |
| `mcp.servers.*.env` | `mcp_servers.*.env` | |
| `mcp.servers.*.cwd` | `mcp_servers.*.cwd` | |
| `mcp.servers.*.url` | `mcp_servers.*.url` | HTTP/SSE 传输 |
| `mcp.servers.*.tools.include` | `mcp_servers.*.tools.include` | 工具过滤 |
| `mcp.servers.*.tools.exclude` | `mcp_servers.*.tools.exclude` | |

### TTS（文字转语音）

TTS 设置从**两个** OpenClaw 配置位置读取，优先级如下：

1. `messages.tts.providers.{provider}.*`（规范位置）
2. 顶层 `talk.providers.{provider}.*`（备用）
3. 旧版平展键 `messages.tts.{provider}.*`（最旧格式）

| 项目 | Hermes 目标 |
|------|-------------------|
| 提供商名称 | `config.yaml` → `tts.provider` |
| ElevenLabs 语音 ID | `config.yaml` → `tts.elevenlabs.voice_id` |
| ElevenLabs 模型 ID | `config.yaml` → `tts.elevenlabs.model_id` |
| OpenAI 模型 | `config.yaml` → `tts.openai.model` |
| OpenAI 语音 | `config.yaml` → `tts.openai.voice` |
| Edge TTS 语音 | `config.yaml` → `tts.edge.voice`（OpenClaw 将 "edge" 重命名为 "microsoft" — 两者都识别） |
| TTS 资源 | `~/.hermes/tts/`（文件复制） |

### 消息平台

| 平台 | OpenClaw 配置路径 | Hermes `.env` 变量 | 说明 |
|----------|---------------------|----------------------|-------|
| Telegram | `channels.telegram.botToken` 或 `.accounts.default.botToken` | `TELEGRAM_BOT_TOKEN` | Token 可以是字符串或 [SecretRef](#secretref-handling)。支持平展和 accounts 布局。 |
| Telegram | `credentials/telegram-default-allowFrom.json` | `TELEGRAM_ALLOWED_USERS` | 从 `allowFrom[]` 数组逗号连接 |
| Discord | `channels.discord.token` 或 `.accounts.default.token` | `DISCORD_BOT_TOKEN` | |
| Discord | `channels.discord.allowFrom` 或 `.accounts.default.allowFrom` | `DISCORD_ALLOWED_USERS` | |
| Slack | `channels.slack.botToken` 或 `.accounts.default.botToken` | `SLACK_BOT_TOKEN` | |
| Slack | `channels.slack.appToken` 或 `.accounts.default.appToken` | `SLACK_APP_TOKEN` | |
| Slack | `channels.slack.allowFrom` 或 `.accounts.default.allowFrom` | `SLACK_ALLOWED_USERS` | |
| WhatsApp | `channels.whatsapp.allowFrom` 或 `.accounts.default.allowFrom` | `WHATSAPP_ALLOWED_USERS` | 认证通过 Baileys QR 配对 — 迁移后需要重新配对 |
| Signal | `channels.signal.account` 或 `.accounts.default.account` | `SIGNAL_ACCOUNT` | |
| Signal | `channels.signal.httpUrl` 或 `.accounts.default.httpUrl` | `SIGNAL_HTTP_URL` | |
| Signal | `channels.signal.allowFrom` 或 `.accounts.default.allowFrom` | `SIGNAL_ALLOWED_USERS` | |
| Matrix | `channels.matrix.accessToken` 或 `.accounts.default.accessToken` | `MATRIX_ACCESS_TOKEN` | 使用 `accessToken`（不是 `botToken`） |
| Mattermost | `channels.mattermost.botToken` 或 `.accounts.default.botToken` | `MATTERMOST_BOT_TOKEN` | |

### 其他配置

| 项目 | OpenClaw 路径 | Hermes 路径 | 说明 |
|------|-------------|-------------|-------|
| 审批模式 | `approvals.exec.mode` | `config.yaml` → `approvals.mode` | "auto"→"off"，"always"→"manual"，"smart"→"smart" |
| 命令白名单 | `exec-approvals.json` | `config.yaml` → `command_allowlist` | 模式合并去重 |
| 浏览器 CDP URL | `browser.cdpUrl` | `config.yaml` → `browser.cdp_url` | |
| 浏览器无头模式 | `browser.headless` | `config.yaml` → `browser.headless` | |
| Brave 搜索 key | `tools.web.search.brave.apiKey` | `.env` → `BRAVE_API_KEY` | 需要 `--migrate-secrets` |
| Gateway 认证 token | `gateway.auth.token` | `.env` → `HERMES_GATEWAY_TOKEN` | 需要 `--migrate-secrets` |
| 工作目录 | `agents.defaults.workspace` | `.env` → `MESSAGING_CWD` | |

### 已归档（无直接 Hermes 等效项）

这些保存到 `~/.hermes/migration/openclaw/<timestamp>/archive/` 供手动审查：

| 项目 | 归档文件 | Hermes 中如何重建 |
|------|-------------|--------------------------|
| `IDENTITY.md` | `archive/workspace/IDENTITY.md` | 合并到 `SOUL.md` |
| `TOOLS.md` | `archive/workspace/TOOLS.md` | Hermes 有内置工具指令 |
| `HEARTBEAT.md` | `archive/workspace/HEARTBEAT.md` | 使用 cron 作业执行定期任务 |
| `BOOTSTRAP.md` | `archive/workspace/BOOTSTRAP.md` | 使用上下文文件或 skills |
| Cron 作业 | `archive/cron-config.json` | 使用 `hermes cron create` 重建 |
| 插件 | `archive/plugins-config.json` | 见 [插件指南](/docs/user-guide/features/hooks) |
| Hooks/Webhooks | `archive/hooks-config.json` | 使用 `hermes webhook` 或 gateway hooks |
| 记忆后端 | `archive/memory-backend-config.json` | 通过 `hermes honcho` 配置 |
| Skills 注册表 | `archive/skills-registry-config.json` | 使用 `hermes skills config` |
| UI/身份 | `archive/ui-identity-config.json` | 使用 `/skin` 命令 |
| 日志记录 | `archive/logging-diagnostics-config.json` | 在 `config.yaml` 日志部分设置 |
| 多智能体列表 | `archive/agents-list.json` | 使用 Hermes profiles |
| 频道绑定 | `archive/bindings.json` | 各平台手动设置 |
| 复杂频道 | `archive/channels-deep-config.json` | 手动平台配置 |

## API key 解析

启用 `--migrate-secrets` 时，API key 从**四个来源**按优先级收集：

1. **配置值** — `openclaw.json` 中 `models.providers.*.apiKey` 和 TTS 提供商密钥
2. **环境文件** — `~/.openclaw/.env`（`OPENROUTER_API_KEY`、`ANTHROPIC_API_KEY` 等）
3. **配置 env 子对象** — `openclaw.json` → `"env"` 或 `"env"."vars"`（某些设置将密钥存储在此处而非单独的 `.env` 文件）
4. **认证配置** — `~/.openclaw/agents/main/agent/auth-profiles.json`（每个智能体的凭证）

配置值优先。每个后续来源填充剩余的空缺。

### 支持的 key 目标

`OPENROUTER_API_KEY`、`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`DEEPSEEK_API_KEY`、`GEMINI_API_KEY`、`ZAI_API_KEY`、`MINIMAX_API_KEY`、`ELEVENLABS_API_KEY`、`TELEGRAM_BOT_TOKEN`、`VOICE_TOOLS_OPENAI_KEY`

此白名单之外的 key 不会被复制。

## SecretRef 处理

OpenClaw 中 token 和 API key 的配置值有三种格式：

```json
// 纯字符串
"channels": { "telegram": { "botToken": "123456:ABC-DEF..." } }

// 环境模板
"channels": { "telegram": { "botToken": "${TELEGRAM_BOT_TOKEN}" } }

// SecretRef 对象
"channels": { "telegram": { "botToken": { "source": "env", "id": "TELEGRAM_BOT_TOKEN" } } }
```

迁移解析所有三种格式。对于环境模板和 `source: "env"` 的 SecretRef 对象，它在 `~/.openclaw/.env` 和 `openclaw.json` env 子对象中查找值。无法自动解析 `source: "file"` 或 `source: "exec"` 的 SecretRef 对象 — 迁移会发出警告，这些值需要通过 `hermes config set` 手动添加到 Hermes。

## 迁移后

1. **检查迁移报告** — 完成后打印迁移、跳过和冲突项目的计数。

2. **审查归档文件** — `~/.hermes/migration/openclaw/<timestamp>/archive/` 中的任何内容需要手动处理。

3. **开启新会话** — 导入的 skills 和记忆条目在新会话中生效，不在当前会话中。

4. **验证 API key** — 运行 `hermes status` 检查提供商认证。

5. **测试消息** — 如果迁移了平台 token，重启 gateway：`systemctl --user restart hermes-gateway`

6. **检查会话策略** — 验证 `hermes config get session_reset` 符合预期。

7. **重新配对 WhatsApp** — WhatsApp 使用 QR 码配对（Baileys），不是 token 迁移。运行 `hermes whatsapp` 配对。

8. **归档清理** — 确认一切正常后，运行 `hermes claw cleanup` 将遗留的 OpenClaw 目录重命名为 `.pre-migration/`（防止状态混淆）。

## 故障排除

### "OpenClaw 目录未找到"

迁移检查 `~/.openclaw/`，然后是 `~/.clawdbot/`，然后是 `~/.moltbot/`。如果你的安装在其他位置，使用 `--source /path/to/your/openclaw`。

### "未找到提供商 API key"

密钥可能存储在多个位置，取决于你的 OpenClaw 版本：`openclaw.json` 内联在 `models.providers.*.apiKey` 下、`~/.openclaw/.env`、`openclaw.json` `"env"` 子对象，或 `agents/main/agent/auth-profiles.json`。迁移检查所有四处。如果密钥使用 `source: "file"` 或 `source: "exec"` SecretRef，无法自动解析 — 通过 `hermes config set` 添加。

### 迁移后 Skills 不显示

导入的 skills 位于 `~/.hermes/skills/openclaw-imports/`。开启新会话才能生效，或运行 `/skills` 验证已加载。

### TTS 语音未迁移

OpenClaw 在两个位置存储 TTS 设置：`messages.tts.providers.*` 和顶层 `talk` 配置。迁移检查两者。如果你的语音 ID 通过 OpenClaw UI 设置（存储在不同路径），可能需要手动设置：`hermes config set tts.elevenlabs.voice_id YOUR_VOICE_ID`。
