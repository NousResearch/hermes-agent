---
sidebar_position: 2
title: "Environment Variables"
description: "Complete reference of all environment variables used by Hermes Agent"
---

# Environment Variables Reference

All variables go in `~/.hermes/.env`. You can also set them with `hermes config set VAR value`.

## LLM Providers

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key (recommended for flexibility) |
| `OPENROUTER_BASE_URL` | Override the OpenRouter-compatible base URL |
| `AI_GATEWAY_API_KEY` | Vercel AI Gateway API key ([ai-gateway.vercel.sh](https://ai-gateway.vercel.sh)) |
| `AI_GATEWAY_BASE_URL` | Override AI Gateway base URL (default: `https://ai-gateway.vercel.sh/v1`) |
| `OPENAI_API_KEY` | API key for custom OpenAI-compatible endpoints (used with `OPENAI_BASE_URL`) |
| `OPENAI_BASE_URL` | Base URL for custom endpoint (VLLM, SGLang, etc.) |
| `COPILOT_GITHUB_TOKEN` | GitHub token for Copilot API — first priority (OAuth `gho_*` or fine-grained PAT `github_pat_*`; classic PATs `ghp_*` are **not supported**) |
| `GH_TOKEN` | GitHub token — second priority for Copilot (also used by `gh` CLI) |
| `GITHUB_TOKEN` | GitHub token — third priority for Copilot |
| `HERMES_COPILOT_ACP_COMMAND` | Override Copilot ACP CLI binary path (default: `copilot`) |
| `COPILOT_CLI_PATH` | Alias for `HERMES_COPILOT_ACP_COMMAND` |
| `HERMES_COPILOT_ACP_ARGS` | Override Copilot ACP arguments (default: `--acp --stdio`) |
| `COPILOT_ACP_BASE_URL` | Override Copilot ACP base URL |
| `GLM_API_KEY` | z.ai / ZhipuAI GLM API key ([z.ai](https://z.ai)) |
| `ZAI_API_KEY` | Alias for `GLM_API_KEY` |
| `Z_AI_API_KEY` | Alias for `GLM_API_KEY` |
| `GLM_BASE_URL` | Override z.ai base URL (default: `https://api.z.ai/api/paas/v4`) |
| `KIMI_API_KEY` | Kimi / Moonshot AI API key ([moonshot.ai](https://platform.moonshot.ai)) |
| `KIMI_BASE_URL` | Override Kimi base URL (default: `https://api.moonshot.ai/v1`) |
| `MINIMAX_API_KEY` | MiniMax API key — global endpoint ([minimax.io](https://www.minimax.io)) |
| `MINIMAX_BASE_URL` | Override MiniMax base URL (default: `https://api.minimax.io/v1`) |
| `MINIMAX_CN_API_KEY` | MiniMax API key — China endpoint ([minimaxi.com](https://www.minimaxi.com)) |
| `MINIMAX_CN_BASE_URL` | Override MiniMax China base URL (default: `https://api.minimaxi.com/v1`) |
| `KILOCODE_API_KEY` | Kilo Code API key ([kilo.ai](https://kilo.ai)) |
| `KILOCODE_BASE_URL` | Override Kilo Code base URL (default: `https://api.kilo.ai/api/gateway`) |
| `HF_TOKEN` | Hugging Face token for Inference Providers ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) |
| `HF_BASE_URL` | Override Hugging Face base URL (default: `https://router.huggingface.co/v1`) |
| `GOOGLE_API_KEY` | Google AI Studio API key ([aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)) |
| `GEMINI_API_KEY` | Alias for `GOOGLE_API_KEY` |
| `GEMINI_BASE_URL` | Override Google AI Studio base URL |
| `ANTHROPIC_API_KEY` | Anthropic Console API key ([console.anthropic.com](https://console.anthropic.com/)) |
| `ANTHROPIC_TOKEN` | Manual or legacy Anthropic OAuth/setup-token override |
| `DASHSCOPE_API_KEY` | Alibaba Cloud DashScope API key for Qwen models ([modelstudio.console.alibabacloud.com](https://modelstudio.console.alibabacloud.com/)) |
| `DASHSCOPE_BASE_URL` | Custom DashScope base URL (default: `https://coding-intl.dashscope.aliyuncs.com/v1`) |
| `DEEPSEEK_API_KEY` | DeepSeek API key for direct DeepSeek access ([platform.deepseek.com](https://platform.deepseek.com/api_keys)) |
| `DEEPSEEK_BASE_URL` | Custom DeepSeek API base URL |
| `OPENCODE_ZEN_API_KEY` | OpenCode Zen API key — pay-as-you-go access to curated models ([opencode.ai](https://opencode.ai/auth)) |
| `OPENCODE_ZEN_BASE_URL` | Override OpenCode Zen base URL |
| `OPENCODE_GO_API_KEY` | OpenCode Go API key — $10/month subscription for open models ([opencode.ai](https://opencode.ai/auth)) |
| `OPENCODE_GO_BASE_URL` | Override OpenCode Go base URL |
| `CLAUDE_CODE_OAUTH_TOKEN` | Explicit Claude Code token override if you export one manually |
| `HERMES_MODEL` | Preferred model name (checked before `LLM_MODEL`, used by gateway) |
| `LLM_MODEL` | Default model name (fallback when not set in config.yaml) |
| `VOICE_TOOLS_OPENAI_KEY` | Preferred OpenAI key for OpenAI speech-to-text and text-to-speech providers |
| `HERMES_LOCAL_STT_COMMAND` | Optional local speech-to-text command template. Supports `{input_path}`, `{output_dir}`, `{language}`, and `{model}` placeholders |
| `HERMES_LOCAL_STT_LANGUAGE` | Default language passed to `HERMES_LOCAL_STT_COMMAND` or auto-detected local `whisper` CLI fallback (default: `en`) |
| `HERMES_HOME` | Override Hermes config directory (default: `~/.hermes`). Also scopes the gateway PID file and systemd service name, so multiple installations can run concurrently |

## Provider Auth (OAuth)

For native Anthropic auth, Hermes prefers Claude Code's own credential files when they exist because those credentials can refresh automatically. Environment variables such as `ANTHROPIC_TOKEN` remain useful as manual overrides, but they are no longer the preferred path for Claude Pro/Max login.

| Variable | Description |
|----------|-------------|
| `HERMES_INFERENCE_PROVIDER` | Override provider selection: `auto`, `openrouter`, `nous`, `openai-codex`, `copilot`, `copilot-acp`, `anthropic`, `huggingface`, `zai`, `kimi-coding`, `minimax`, `minimax-cn`, `kilocode`, `alibaba`, `deepseek`, `opencode-zen`, `opencode-go`, `ai-gateway` (default: `auto`) |
| `HERMES_PORTAL_BASE_URL` | Override Nous Portal URL (for development/testing) |
| `NOUS_INFERENCE_BASE_URL` | Override Nous inference API URL |
| `HERMES_NOUS_MIN_KEY_TTL_SECONDS` | Min agent key TTL before re-mint (default: 1800 = 30min) |
| `HERMES_NOUS_TIMEOUT_SECONDS` | HTTP timeout for Nous credential / token flows |
| `HERMES_DUMP_REQUESTS` | Dump API request payloads to log files (`true`/`false`) |
| `HERMES_PREFILL_MESSAGES_FILE` | Path to a JSON file of ephemeral prefill messages injected at API-call time |
| `HERMES_TIMEZONE` | IANA timezone override (for example `America/New_York`) |

## Tool APIs

| Variable | Description |
|----------|-------------|
| `PARALLEL_API_KEY` | AI-native web search ([parallel.ai](https://parallel.ai/)) |
| `FIRECRAWL_API_KEY` | Web scraping and cloud browser ([firecrawl.dev](https://firecrawl.dev/)) |
| `FIRECRAWL_API_URL` | Custom Firecrawl API endpoint for self-hosted instances (optional) |
| `TAVILY_API_KEY` | Tavily API key for AI-native web search, extract, and crawl ([app.tavily.com](https://app.tavily.com/home)) |
| `EXA_API_KEY` | Exa API key for AI-native web search and contents ([exa.ai](https://exa.ai/)) |
| `BROWSERBASE_API_KEY` | Browser automation ([browserbase.com](https://browserbase.com/)) |
| `BROWSERBASE_PROJECT_ID` | Browserbase project ID |
| `BROWSER_USE_API_KEY` | Browser Use cloud browser API key ([browser-use.com](https://browser-use.com/)) |
| `FIRECRAWL_BROWSER_TTL` | Firecrawl browser session TTL in seconds (default: 300) |
| `BROWSER_CDP_URL` | Chrome DevTools Protocol URL for local browser (set via `/browser connect`, e.g. `ws://localhost:9222`) |
| `CAMOFOX_URL` | Camofox local anti-detection browser URL (default: `http://localhost:9377`) |
| `BROWSER_INACTIVITY_TIMEOUT` | Browser session inactivity timeout in seconds |
| `FAL_KEY` | Image generation ([fal.ai](https://fal.ai/)) |
| `GROQ_API_KEY` | Groq Whisper STT API key ([groq.com](https://groq.com/)) |
| `ELEVENLABS_API_KEY` | ElevenLabs premium TTS voices ([elevenlabs.io](https://elevenlabs.io/)) |
| `STT_GROQ_MODEL` | Override the Groq STT model (default: `whisper-large-v3-turbo`) |
| `GROQ_BASE_URL` | Override the Groq OpenAI-compatible STT endpoint |
| `STT_OPENAI_MODEL` | Override the OpenAI STT model (default: `whisper-1`) |
| `STT_OPENAI_BASE_URL` | Override the OpenAI-compatible STT endpoint |
| `GITHUB_TOKEN` | GitHub token for Skills Hub (higher API rate limits, skill publish) |
| `HONCHO_API_KEY` | Cross-session user modeling ([honcho.dev](https://honcho.dev/)) |
| `HONCHO_BASE_URL` | Base URL for self-hosted Honcho instances (default: Honcho cloud). No API key required for local instances |
| `SUPERMEMORY_API_KEY` | Semantic long-term memory with profile recall and session ingest ([supermemory.ai](https://supermemory.ai)) |
| `TINKER_API_KEY` | RL training ([tinker-console.thinkingmachines.ai](https://tinker-console.thinkingmachines.ai/)) |
| `WANDB_API_KEY` | RL training metrics ([wandb.ai](https://wandb.ai/)) |
| `DAYTONA_API_KEY` | Daytona cloud sandboxes ([daytona.io](https://daytona.io/)) |

## Terminal Backend

| Variable | Description |
|----------|-------------|
| `TERMINAL_ENV` | Backend: `local`, `docker`, `ssh`, `singularity`, `modal`, `daytona` |
| `TERMINAL_DOCKER_IMAGE` | Docker image (default: `nikolaik/python-nodejs:python3.11-nodejs20`) |
| `TERMINAL_DOCKER_FORWARD_ENV` | JSON array of env var names to explicitly forward into Docker terminal sessions. Note: skill-declared `required_environment_variables` are forwarded automatically — you only need this for vars not declared by any skill. |
| `TERMINAL_DOCKER_VOLUMES` | Additional Docker volume mounts (comma-separated `host:container` pairs) |
| `TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE` | Advanced opt-in: mount the launch cwd into Docker `/workspace` (`true`/`false`, default: `false`) |
| `TERMINAL_SINGULARITY_IMAGE` | Singularity image or `.sif` path |
| `TERMINAL_MODAL_IMAGE` | Modal container image |
| `TERMINAL_DAYTONA_IMAGE` | Daytona sandbox image |
| `TERMINAL_TIMEOUT` | Command timeout in seconds |
| `TERMINAL_LIFETIME_SECONDS` | Max lifetime for terminal sessions in seconds |
| `TERMINAL_CWD` | Working directory for all terminal sessions |
| `SUDO_PASSWORD` | Enable sudo without interactive prompt |

For cloud sandbox backends, persistence is filesystem-oriented. `TERMINAL_LIFETIME_SECONDS` controls when Hermes cleans up an idle terminal session, and later resumes may recreate the sandbox rather than keep the same live processes running.

## SSH Backend

| Variable | Description |
|----------|-------------|
| `TERMINAL_SSH_HOST` | Remote server hostname |
| `TERMINAL_SSH_USER` | SSH username |
| `TERMINAL_SSH_PORT` | SSH port (default: 22) |
| `TERMINAL_SSH_KEY` | Path to private key |
| `TERMINAL_SSH_PERSISTENT` | Override persistent shell for SSH (default: follows `TERMINAL_PERSISTENT_SHELL`) |

## Container Resources (Docker, Singularity, Modal, Daytona)

| Variable | Description |
|----------|-------------|
| `TERMINAL_CONTAINER_CPU` | CPU cores (default: 1) |
| `TERMINAL_CONTAINER_MEMORY` | Memory in MB (default: 5120) |
| `TERMINAL_CONTAINER_DISK` | Disk in MB (default: 51200) |
| `TERMINAL_CONTAINER_PERSISTENT` | Persist container filesystem across sessions (default: `true`) |
| `TERMINAL_SANDBOX_DIR` | Host directory for workspaces and overlays (default: `~/.hermes/sandboxes/`) |

## Persistent Shell

| Variable | Description |
|----------|-------------|
| `TERMINAL_PERSISTENT_SHELL` | Enable persistent shell for non-local backends (default: `true`). Also settable via `terminal.persistent_shell` in config.yaml |
| `TERMINAL_LOCAL_PERSISTENT` | Enable persistent shell for local backend (default: `false`) |
| `TERMINAL_SSH_PERSISTENT` | Override persistent shell for SSH backend (default: follows `TERMINAL_PERSISTENT_SHELL`) |

## Messaging

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot token (from @BotFather) |
| `TELEGRAM_ALLOWED_USERS` | Comma-separated user IDs allowed to use the bot |
| `TELEGRAM_HOME_CHANNEL` | Default Telegram chat/channel for cron delivery |
| `TELEGRAM_HOME_CHANNEL_NAME` | Display name for the Telegram home channel |
| `TELEGRAM_WEBHOOK_URL` | Public HTTPS URL for webhook mode (enables webhook instead of polling) |
| `TELEGRAM_WEBHOOK_PORT` | Local listen port for webhook server (default: `8443`) |
| `TELEGRAM_WEBHOOK_SECRET` | Secret token for verifying updates come from Telegram |
| `DISCORD_BOT_TOKEN` | Discord bot token |
| `DISCORD_ALLOWED_USERS` | Comma-separated Discord user IDs allowed to use the bot |
| `DISCORD_HOME_CHANNEL` | Default Discord channel for cron delivery |
| `DISCORD_HOME_CHANNEL_NAME` | Display name for the Discord home channel |
| `DISCORD_REQUIRE_MENTION` | Require an @mention before responding in server channels |
| `DISCORD_FREE_RESPONSE_CHANNELS` | Comma-separated channel IDs where mention is not required |
| `DISCORD_AUTO_THREAD` | Auto-thread long replies when supported |
| `SLACK_BOT_TOKEN` | Slack bot token (`xoxb-...`) |
| `SLACK_APP_TOKEN` | Slack app-level token (`xapp-...`, required for Socket Mode) |
| `SLACK_ALLOWED_USERS` | Comma-separated Slack user IDs |
| `SLACK_HOME_CHANNEL` | Default Slack channel for cron delivery |
| `SLACK_HOME_CHANNEL_NAME` | Display name for the Slack home channel |
| `WHATSAPP_ENABLED` | Enable the WhatsApp bridge (`true`/`false`) |
| `WHATSAPP_MODE` | `bot` (separate number) or `self-chat` (message yourself) |
| `WHATSAPP_ALLOWED_USERS` | Comma-separated phone numbers (with country code, no `+`), or `*` to allow all senders |
| `WHATSAPP_ALLOW_ALL_USERS` | Allow all WhatsApp senders without an allowlist (`true`/`false`) |
| `WHATSAPP_DEBUG` | Log raw message events in the bridge for troubleshooting (`true`/`false`) |
| `SIGNAL_HTTP_URL` | signal-cli daemon HTTP endpoint (for example `http://127.0.0.1:8080`) |
| `SIGNAL_ACCOUNT` | Bot phone number in E.164 format |
| `SIGNAL_ALLOWED_USERS` | Comma-separated E.164 phone numbers or UUIDs |
| `SIGNAL_GROUP_ALLOWED_USERS` | Comma-separated group IDs, or `*` for all groups |
| `SIGNAL_HOME_CHANNEL_NAME` | Display name for the Signal home channel |
| `SIGNAL_IGNORE_STORIES` | Ignore Signal stories/status updates |
| `SIGNAL_ALLOW_ALL_USERS` | Allow all Signal users without an allowlist |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID (shared with telephony skill) |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token (shared with telephony skill) |
| `TWILIO_PHONE_NUMBER` | Twilio phone number in E.164 format (shared with telephony skill) |
| `SMS_WEBHOOK_PORT` | Webhook listener port for inbound SMS (default: `8080`) |
| `SMS_ALLOWED_USERS` | Comma-separated E.164 phone numbers allowed to chat |
| `SMS_ALLOW_ALL_USERS` | Allow all SMS senders without an allowlist |
| `SMS_HOME_CHANNEL` | Phone number for cron job / notification delivery |
| `SMS_HOME_CHANNEL_NAME` | Display name for the SMS home channel |
| `EMAIL_ADDRESS` | Email address for the Email gateway adapter |
| `EMAIL_PASSWORD` | Password or app password for the email account |
| `EMAIL_IMAP_HOST` | IMAP hostname for the email adapter |
| `EMAIL_IMAP_PORT` | IMAP port |
| `EMAIL_SMTP_HOST` | SMTP hostname for the email adapter |
| `EMAIL_SMTP_PORT` | SMTP port |
| `EMAIL_ALLOWED_USERS` | Comma-separated email addresses allowed to message the bot |
| `EMAIL_HOME_ADDRESS` | Default recipient for proactive email delivery |
| `EMAIL_HOME_ADDRESS_NAME` | Display name for the email home target |
| `EMAIL_POLL_INTERVAL` | Email polling interval in seconds |
| `EMAIL_ALLOW_ALL_USERS` | Allow all inbound email senders |
| `DINGTALK_CLIENT_ID` | DingTalk bot AppKey from developer portal ([open.dingtalk.com](https://open.dingtalk.com)) |
| `DINGTALK_CLIENT_SECRET` | DingTalk bot AppSecret from developer portal |
| `DINGTALK_ALLOWED_USERS` | Comma-separated DingTalk user IDs allowed to message the bot |
| `FEISHU_APP_ID` | Feishu/Lark bot App ID from [open.feishu.cn](https://open.feishu.cn/) |
| `FEISHU_APP_SECRET` | Feishu/Lark bot App Secret |
| `FEISHU_DOMAIN` | `feishu` (China) or `lark` (international). Default: `feishu` |
| `FEISHU_CONNECTION_MODE` | `websocket` (recommended) or `webhook`. Default: `websocket` |
| `FEISHU_ENCRYPT_KEY` | Optional encryption key for webhook mode |
| `FEISHU_VERIFICATION_TOKEN` | Optional verification token for webhook mode |
| `FEISHU_ALLOWED_USERS` | Comma-separated Feishu user IDs allowed to message the bot |
| `FEISHU_HOME_CHANNEL` | Feishu chat ID for cron delivery and notifications |
| `WECOM_BOT_ID` | WeCom AI Bot ID from admin console |
| `WECOM_SECRET` | WeCom AI Bot secret |
| `WECOM_WEBSOCKET_URL` | Custom WebSocket URL (default: `wss://openws.work.weixin.qq.com`) |
| `WECOM_ALLOWED_USERS` | Comma-separated WeCom user IDs allowed to message the bot |
| `WECOM_HOME_CHANNEL` | WeCom chat ID for cron delivery and notifications |
| `MATTERMOST_URL` | Mattermost server URL (e.g. `https://mm.example.com`) |
| `MATTERMOST_TOKEN` | Bot token or personal access token for Mattermost |
| `MATTERMOST_ALLOWED_USERS` | Comma-separated Mattermost user IDs allowed to message the bot |
| `MATTERMOST_HOME_CHANNEL` | Channel ID for proactive message delivery (cron, notifications) |
| `MATTERMOST_REQUIRE_MENTION` | Require `@mention` in channels (default: `true`). Set to `false` to respond to all messages. |
| `MATTERMOST_FREE_RESPONSE_CHANNELS` | Comma-separated channel IDs where bot responds without `@mention` |
| `MATTERMOST_REPLY_MODE` | Reply style: `thread` (threaded replies) or `off` (flat messages, default) |
| `MATRIX_HOMESERVER` | Matrix homeserver URL (e.g. `https://matrix.org`) |
| `MATRIX_ACCESS_TOKEN` | Matrix access token for bot authentication |
| `MATRIX_USER_ID` | Matrix user ID (e.g. `@hermes:matrix.org`) — required for password login, optional with access token |
| `MATRIX_PASSWORD` | Matrix password (alternative to access token) |
| `MATRIX_ALLOWED_USERS` | Comma-separated Matrix user IDs allowed to message the bot (e.g. `@alice:matrix.org`) |
| `MATRIX_HOME_ROOM` | Room ID for proactive message delivery (e.g. `!abc123:matrix.org`) |
| `MATRIX_ENCRYPTION` | Enable end-to-end encryption (`true`/`false`, default: `false`) |
| `MATRIX_REQUIRE_MENTION` | Require `@mention` in rooms (default: `true`). Set to `false` to respond to all messages. |
| `MATRIX_FREE_RESPONSE_ROOMS` | Comma-separated room IDs where bot responds without `@mention` |
| `MATRIX_AUTO_THREAD` | Auto-create threads for room messages (default: `true`) |
| `HASS_TOKEN` | Home Assistant Long-Lived Access Token (enables HA platform + tools) |
| `HASS_URL` | Home Assistant URL (default: `http://homeassistant.local:8123`) |
| `WEBHOOK_ENABLED` | Enable the webhook platform adapter (`true`/`false`) |
| `WEBHOOK_PORT` | HTTP server port for receiving webhooks (default: `8644`) |
| `WEBHOOK_SECRET` | Global HMAC secret for webhook signature validation (used as fallback when routes don't specify their own) |
| `API_SERVER_ENABLED` | Enable the OpenAI-compatible API server (`true`/`false`). Runs alongside other platforms. |
| `API_SERVER_KEY` | Bearer token for API server authentication. Strongly recommended; required for any network-accessible deployment. |
| `API_SERVER_CORS_ORIGINS` | Comma-separated browser origins allowed to call the API server directly (for example `http://localhost:3000,http://127.0.0.1:3000`). Default: disabled. |
| `API_SERVER_PORT` | Port for the API server (default: `8642`) |
| `API_SERVER_HOST` | Host/bind address for the API server (default: `127.0.0.1`). Use `0.0.0.0` for network access only with `API_SERVER_KEY` and a narrow `API_SERVER_CORS_ORIGINS` allowlist. |
| `MESSAGING_CWD` | Working directory for terminal commands in messaging mode (default: `~`) |
| `GATEWAY_ALLOWED_USERS` | Comma-separated user IDs allowed across all platforms |
| `GATEWAY_ALLOW_ALL_USERS` | Allow all users without allowlists (`true`/`false`, default: `false`) |

## Agent Behavior

| Variable | Description |
|----------|-------------|
| `HERMES_MAX_ITERATIONS` | Max tool-calling iterations per conversation (default: 90) |
| `HERMES_TOOL_PROGRESS` | Deprecated compatibility variable for tool progress display. Prefer `display.tool_progress` in `config.yaml`. |
| `HERMES_TOOL_PROGRESS_MODE` | Deprecated compatibility variable for tool progress mode. Prefer `display.tool_progress` in `config.yaml`. |
| `HERMES_HUMAN_DELAY_MODE` | Response pacing: `off`/`natural`/`custom` |
| `HERMES_HUMAN_DELAY_MIN_MS` | Custom delay range minimum (ms) |
| `HERMES_HUMAN_DELAY_MAX_MS` | Custom delay range maximum (ms) |
| `HERMES_QUIET` | Suppress non-essential output (`true`/`false`) |
| `HERMES_API_TIMEOUT` | LLM API call timeout in seconds (default: `1800`) |
| `HERMES_EXEC_ASK` | Enable execution approval prompts in gateway mode (`true`/`false`) |
| `HERMES_ENABLE_PROJECT_PLUGINS` | Enable auto-discovery of repo-local plugins from `./.hermes/plugins/` (`true`/`false`, default: `false`) |
| `HERMES_BACKGROUND_NOTIFICATIONS` | Background process notification mode in gateway: `all` (default), `result`, `error`, `off` |
| `HERMES_EPHEMERAL_SYSTEM_PROMPT` | Ephemeral system prompt injected at API-call time (never persisted to sessions) |

## Session Settings

| Variable | Description |
|----------|-------------|
| `SESSION_IDLE_MINUTES` | Reset sessions after N minutes of inactivity (default: 1440) |
| `SESSION_RESET_HOUR` | Daily reset hour in 24h format (default: 4 = 4am) |

## Context Compression (config.yaml only)

Context compression is configured exclusively through the `compression` section in `config.yaml` — there are no environment variables for it.

```yaml
compression:
  enabled: true
  threshold: 0.50
  summary_model: ""                            # empty = use main configured model
  summary_provider: auto
  summary_base_url: null  # Custom OpenAI-compatible endpoint for summaries
```

## Auxiliary Task Overrides

| Variable | Description |
|----------|-------------|
| `AUXILIARY_VISION_PROVIDER` | Override provider for vision tasks |
| `AUXILIARY_VISION_MODEL` | Override model for vision tasks |
| `AUXILIARY_VISION_BASE_URL` | Direct OpenAI-compatible endpoint for vision tasks |
| `AUXILIARY_VISION_API_KEY` | API key paired with `AUXILIARY_VISION_BASE_URL` |
| `AUXILIARY_WEB_EXTRACT_PROVIDER` | Override provider for web extraction/summarization |
| `AUXILIARY_WEB_EXTRACT_MODEL` | Override model for web extraction/summarization |
| `AUXILIARY_WEB_EXTRACT_BASE_URL` | Direct OpenAI-compatible endpoint for web extraction/summarization |
| `AUXILIARY_WEB_EXTRACT_API_KEY` | API key paired with `AUXILIARY_WEB_EXTRACT_BASE_URL` |

For task-specific direct endpoints, Hermes uses the task's configured API key or `OPENAI_API_KEY`. It does not reuse `OPENROUTER_API_KEY` for those custom endpoints.

## Fallback Model (config.yaml only)

The primary model fallback is configured exclusively through `config.yaml` — there are no environment variables for it. Add a `fallback_model` section with `provider` and `model` keys to enable automatic failover when your main model encounters errors.

```yaml
fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

See [Fallback Providers](/docs/user-guide/features/fallback-providers) for full details.

## Provider Routing (config.yaml only)

These go in `~/.hermes/config.yaml` under the `provider_routing` section:

| Key | Description |
|-----|-------------|
| `sort` | Sort providers: `"price"` (default), `"throughput"`, or `"latency"` |
| `only` | List of provider slugs to allow (e.g., `["anthropic", "google"]`) |
| `ignore` | List of provider slugs to skip |
| `order` | List of provider slugs to try in order |
| `require_parameters` | Only use providers supporting all request params (`true`/`false`) |
| `data_collection` | `"allow"` (default) or `"deny"` to exclude data-storing providers |

:::tip
Use `hermes config set` to set environment variables — it automatically saves them to the right file (`.env` for secrets, `config.yaml` for everything else).
:::
