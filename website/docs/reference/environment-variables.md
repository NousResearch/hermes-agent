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
| `HERMES_OPENROUTER_CACHE` | Enable OpenRouter response caching (`1`/`true`/`yes`/`on`). Overrides `openrouter.response_cache` in config.yaml. See [Response Caching](https://openrouter.ai/docs/guides/features/response-caching). |
| `HERMES_OPENROUTER_CACHE_TTL` | Cache TTL in seconds (1-86400). Overrides `openrouter.response_cache_ttl` in config.yaml. |
| `NOUS_BASE_URL` | Override Nous Portal base URL (rarely needed; development/testing only) |
| `NOUS_INFERENCE_BASE_URL` | Override Nous inference endpoint directly |
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
| `KIMI_CN_API_KEY` | Kimi / Moonshot China API key ([moonshot.cn](https://platform.moonshot.cn)) |
| `ARCEEAI_API_KEY` | Arcee AI API key ([chat.arcee.ai](https://chat.arcee.ai/)) |
| `ARCEE_BASE_URL` | Override Arcee base URL (default: `https://api.arcee.ai/api/v1`) |
| `GMI_API_KEY` | GMI Cloud API key ([gmicloud.ai](https://www.gmicloud.ai/)) |
| `GMI_BASE_URL` | Override GMI Cloud base URL (default: `https://api.gmi-serving.com/v1`) |
| `MINIMAX_API_KEY` | MiniMax API key — global endpoint ([minimax.io](https://www.minimax.io)). **Not used by `minimax-oauth`** (OAuth path uses browser login instead). |
| `MINIMAX_BASE_URL` | Override MiniMax base URL (default: `https://api.minimax.io/anthropic` — Hermes uses MiniMax's Anthropic Messages-compatible endpoint). **Not used by `minimax-oauth`**. |
| `MINIMAX_CN_API_KEY` | MiniMax API key — China endpoint ([minimaxi.com](https://www.minimaxi.com)). **Not used by `minimax-oauth`** (OAuth path uses browser login instead). |
| `MINIMAX_CN_BASE_URL` | Override MiniMax China base URL (default: `https://api.minimaxi.com/anthropic`). **Not used by `minimax-oauth`**. |
| `KILOCODE_API_KEY` | Kilo Code API key ([kilo.ai](https://kilo.ai)) |
| `KILOCODE_BASE_URL` | Override Kilo Code base URL (default: `https://api.kilo.ai/api/gateway`) |
| `XIAOMI_API_KEY` | Xiaomi MiMo API key ([platform.xiaomimimo.com](https://platform.xiaomimimo.com)) |
| `XIAOMI_BASE_URL` | Override Xiaomi MiMo base URL (default: `https://api.xiaomimimo.com/v1`) |
| `TOKENHUB_API_KEY` | Tencent TokenHub API key ([tokenhub.tencentmaas.com](https://tokenhub.tencentmaas.com)) |
| `TOKENHUB_BASE_URL` | Override Tencent TokenHub base URL (default: `https://tokenhub.tencentmaas.com/v1`) |
| `AZURE_FOUNDRY_API_KEY` | Azure AI Foundry / Azure OpenAI API key ([ai.azure.com](https://ai.azure.com/)) |
| `AZURE_FOUNDRY_BASE_URL` | Azure AI Foundry endpoint URL (e.g. `https://<resource>.openai.azure.com/openai/v1` for OpenAI-style, or `https://<resource>.services.ai.azure.com/anthropic` for Anthropic-style) |
| `AZURE_ANTHROPIC_KEY` | Azure Anthropic API key for `provider: anthropic` + `base_url` pointing at an Azure Foundry Claude deployment (alternative to `ANTHROPIC_API_KEY` when both Anthropic and Azure Anthropic are configured) |
| `HF_TOKEN` | Hugging Face token for Inference Providers ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) |
| `HF_BASE_URL` | Override Hugging Face base URL (default: `https://router.huggingface.co/v1`) |
| `GOOGLE_API_KEY` | Google AI Studio API key ([aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)) |
| `GEMINI_API_KEY` | Alias for `GOOGLE_API_KEY` |
| `GEMINI_BASE_URL` | Override Google AI Studio base URL |
| `HERMES_GEMINI_CLIENT_ID` | OAuth client ID for `google-gemini-cli` PKCE login (optional; defaults to Google's public gemini-cli client) |
| `HERMES_GEMINI_CLIENT_SECRET` | OAuth client secret for `google-gemini-cli` (optional) |
| `HERMES_GEMINI_PROJECT_ID` | GCP project ID for paid Gemini tiers (free tier auto-provisions) |
| `ANTHROPIC_API_KEY` | Anthropic Console API key ([console.anthropic.com](https://console.anthropic.com/)) |
| `ANTHROPIC_TOKEN` | Manual or legacy Anthropic OAuth/setup-token override |
| `DASHSCOPE_API_KEY` | Alibaba Cloud DashScope API key for Qwen models ([modelstudio.console.alibabacloud.com](https://modelstudio.console.alibabacloud.com/)) |
| `DASHSCOPE_BASE_URL` | Custom DashScope base URL (default: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`; use `https://dashscope.aliyuncs.com/compatible-mode/v1` for mainland-China region) |
| `DEEPSEEK_API_KEY` | DeepSeek API key for direct DeepSeek access ([platform.deepseek.com](https://platform.deepseek.com/api_keys)) |
| `DEEPSEEK_BASE_URL` | Custom DeepSeek API base URL |
| `NVIDIA_API_KEY` | NVIDIA NIM API key — Nemotron and open models ([build.nvidia.com](https://build.nvidia.com)) |
| `NVIDIA_BASE_URL` | Override NVIDIA base URL (default: `https://integrate.api.nvidia.com/v1`; set to `http://localhost:8000/v1` for a local NIM endpoint) |
| `OLLAMA_API_KEY` | Ollama Cloud API key — managed Ollama catalog without local GPU ([ollama.com/settings/keys](https://ollama.com/settings/keys)) |
| `OLLAMA_BASE_URL` | Override Ollama Cloud base URL (default: `https://ollama.com/v1`) |
| `XAI_API_KEY` | xAI (Grok) API key for chat + TTS ([console.x.ai](https://console.x.ai/)) |
| `XAI_BASE_URL` | Override xAI base URL (default: `https://api.x.ai/v1`) |
| `MISTRAL_API_KEY` | Mistral API key for Voxtral TTS and Voxtral STT ([console.mistral.ai](https://console.mistral.ai)) |
| `AWS_REGION` | AWS region for Bedrock inference (e.g. `us-east-1`, `eu-central-1`). Read by boto3. |
| `AWS_PROFILE` | AWS named profile for Bedrock authentication (reads `~/.aws/credentials`). Leave unset to use default boto3 credential chain. |
| `BEDROCK_BASE_URL` | Override Bedrock runtime base URL (default: `https://bedrock-runtime.us-east-1.amazonaws.com`; usually leave unset and use `AWS_REGION` instead) |
| `HERMES_QWEN_BASE_URL` | Qwen Portal base URL override (default: `https://portal.qwen.ai/v1`) |
| `OPENCODE_ZEN_API_KEY` | OpenCode Zen API key — pay-as-you-go access to curated models ([opencode.ai](https://opencode.ai/auth)) |
| `OPENCODE_ZEN_BASE_URL` | Override OpenCode Zen base URL |
| `OPENCODE_GO_API_KEY` | OpenCode Go API key — $10/month subscription for open models ([opencode.ai](https://opencode.ai/auth)) |
| `OPENCODE_GO_BASE_URL` | Override OpenCode Go base URL |
| `CLAUDE_CODE_OAUTH_TOKEN` | Explicit Claude Code token override if you export one manually |
| `HERMES_MODEL` | Override model name at process level (used by cron scheduler; prefer `config.yaml` for normal use) |
| `VOICE_TOOLS_OPENAI_KEY` | Preferred OpenAI key for OpenAI speech-to-text and text-to-speech providers |
| `HERMES_LOCAL_STT_COMMAND` | Optional local speech-to-text command template. Supports `{input_path}`, `{output_dir}`, `{language}`, and `{model}` placeholders |
| `HERMES_LOCAL_STT_LANGUAGE` | Default language passed to `HERMES_LOCAL_STT_COMMAND` or auto-detected local `whisper` CLI fallback (default: `en`) |
| `HERMES_HOME` | Override Hermes config directory (default: `~/.hermes`). Also scopes the gateway PID file and systemd service name, so multiple installations can run concurrently |
| `HERMES_KANBAN_HOME` | Override the shared Hermes root that anchors the kanban board (db + workspaces + worker logs). Falls back to `get_default_hermes_root()` (the parent of any active profile). Useful for tests and unusual deployments |
| `HERMES_KANBAN_DB` | Pin the kanban database file path directly (highest precedence; beats `HERMES_KANBAN_HOME`). The dispatcher injects this into worker subprocess env so profile workers converge on the dispatcher's board |
| `HERMES_KANBAN_WORKSPACES_ROOT` | Pin the kanban workspaces root directly (highest precedence for workspaces; beats `HERMES_KANBAN_HOME`). The dispatcher injects this into worker subprocess env |

## Provider Auth (OAuth)

For native Anthropic auth, Hermes prefers Claude Code's own credential files when they exist because those credentials can refresh automatically. **OAuth against Anthropic requires a Claude Max plan with purchased extra usage credits** — Hermes routes as Claude Code, which only draws from the Max plan's extra/overage credits, not the base Max allowance, and does not work on Claude Pro. Without Max + extra credits, use an API key instead. Environment variables such as `ANTHROPIC_TOKEN` remain useful as manual overrides, but they are no longer the preferred path for Claude Max login.

| Variable | Description |
|----------|-------------|
| `HERMES_INFERENCE_PROVIDER` | Override provider selection: `auto`, `custom`, `openrouter`, `nous`, `openai-codex`, `copilot`, `copilot-acp`, `anthropic`, `huggingface`, `gemini`, `zai`, `kimi-coding`, `kimi-coding-cn`, `minimax`, `minimax-cn`, `minimax-oauth` (browser OAuth login — no API key required; see [MiniMax OAuth guide](../guides/minimax-oauth.md)), `kilocode`, `xiaomi`, `arcee`, `gmi`, `alibaba`, `alibaba-coding-plan` (alias `alibaba_coding`), `deepseek`, `nvidia`, `ollama-cloud`, `xai` (alias `grok`), `google-gemini-cli`, `qwen-oauth`, `bedrock`, `opencode-zen`, `opencode-go`, `ai-gateway`, `tencent-tokenhub` (default: `auto`) |
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
| `TAVILY_BASE_URL` | Override the Tavily API endpoint. Useful for corporate proxies and self-hosted Tavily-compatible search backends. Same pattern as `GROQ_BASE_URL`. |
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
| `HINDSIGHT_TIMEOUT` | Timeout in seconds for Hindsight memory-provider API calls (default: `60`). Bump this if your Hindsight instance is slow to respond during `/sync` or `on_session_switch` and you're seeing timeouts in `errors.log`. |
| `SUPERMEMORY_API_KEY` | Semantic long-term memory with profile recall and session ingest ([supermemory.ai](https://supermemory.ai)) |
| `TINKER_API_KEY` | RL training ([tinker-console.thinkingmachines.ai](https://tinker-console.thinkingmachines.ai/)) |
| `WANDB_API_KEY` | RL training metrics ([wandb.ai](https://wandb.ai/)) |
| `DAYTONA_API_KEY` | Daytona cloud sandboxes ([daytona.io](https://daytona.io/)) |
| `VERCEL_TOKEN` | Vercel Sandbox access token ([vercel.com](https://vercel.com/)) |
| `VERCEL_PROJECT_ID` | Vercel project ID (required with `VERCEL_TOKEN`) |
| `VERCEL_TEAM_ID` | Vercel team ID (required with `VERCEL_TOKEN`) |
| `VERCEL_OIDC_TOKEN` | Vercel short-lived OIDC token (development-only alternative) |

### Langfuse Observability

Environment variables for the bundled [`observability/langfuse`](/docs/user-guide/features/built-in-plugins#observabilitylangfuse) plugin. Set these with `hermes tools → Langfuse Observability` or manually in `~/.hermes/.env`. The plugin must also be enabled (`hermes plugins enable observability/langfuse`) before any of these take effect.

| Variable | Description |
|----------|-------------|
| `HERMES_LANGFUSE_PUBLIC_KEY` | Langfuse project public key (`pk-lf-...`). Required. |
| `HERMES_LANGFUSE_SECRET_KEY` | Langfuse project secret key (`sk-lf-...`). Required. |
| `HERMES_LANGFUSE_BASE_URL` | Langfuse server URL (default: `https://cloud.langfuse.com`). Set for self-hosted. |
| `HERMES_LANGFUSE_ENV` | Environment tag on traces (`production`, `staging`, …) |
| `HERMES_LANGFUSE_RELEASE` | Release/version tag on traces |
| `HERMES_LANGFUSE_SAMPLE_RATE` | SDK sampling rate 0.0–1.0 (default: `1.0`) |
| `HERMES_LANGFUSE_MAX_CHARS` | Per-field truncation for serialized payloads (default: `12000`) |
| `HERMES_LANGFUSE_DEBUG` | `true` enables verbose plugin logging to `agent.log` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | Standard Langfuse SDK names. Accepted as fallbacks when the `HERMES_LANGFUSE_*` equivalents are unset. |

### Nous Tool Gateway

These variables configure the [Tool Gateway](/docs/user-guide/features/tool-gateway) for paid Nous subscribers or self-hosted gateway deployments. Most users don't need to set these — the gateway is configured automatically via `hermes model` or `hermes tools`.

| Variable | Description |
|----------|-------------|
| `TOOL_GATEWAY_DOMAIN` | Base domain for Tool Gateway routing (default: `nousresearch.com`) |
| `TOOL_GATEWAY_SCHEME` | HTTP or HTTPS scheme for gateway URLs (default: `https`) |
| `TOOL_GATEWAY_USER_TOKEN` | Auth token for the Tool Gateway (normally auto-populated from Nous auth) |
| `FIRECRAWL_GATEWAY_URL` | Override URL for the Firecrawl gateway endpoint specifically |

## Terminal Backend

| Variable | Description |
|----------|-------------|
| `TERMINAL_ENV` | Backend: `local`, `docker`, `ssh`, `singularity`, `modal`, `daytona`, `vercel_sandbox` |
| `HERMES_DOCKER_BINARY` | Override the container binary Hermes shells out to (e.g. `podman`, `/usr/local/bin/docker`). When unset, Hermes auto-discovers `docker` or `podman` on `PATH`. Needed when both are installed and you want the non-default, or when the binary lives outside `PATH`. |
| `TERMINAL_DOCKER_IMAGE` | Docker image (default: `nikolaik/python-nodejs:python3.11-nodejs20`) |
| `TERMINAL_DOCKER_FORWARD_ENV` | JSON array of env var names to explicitly forward into Docker terminal sessions. Note: skill-declared `required_environment_variables` are forwarded automatically — you only need this for vars not declared by any skill. |
| `TERMINAL_DOCKER_VOLUMES` | Additional Docker volume mounts (comma-separated `host:container` pairs) |
| `TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE` | Advanced opt-in: mount the launch cwd into Docker `/workspace` (`true`/`false`, default: `false`) |
| `TERMINAL_SINGULARITY_IMAGE` | Singularity image or `.sif` path |
| `TERMINAL_MODAL_IMAGE` | Modal container image |
| `TERMINAL_DAYTONA_IMAGE` | Daytona sandbox image |
| `TERMINAL_VERCEL_RUNTIME` | Vercel Sandbox runtime (`node24`, `node22`, `python3.13`) |
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
| `TELEGRAM_ALLOWED_USERS` | Comma-separated user IDs allowed to use the bot (applies to DMs, groups, and forums) |
| `TELEGRAM_GROUP_ALLOWED_USERS` | Comma-separated sender user IDs authorized in groups/forums only (does NOT grant DM access). Chat-ID-shaped values (starting with `-`) are still honored as chat IDs for backward compat with pre-#17686 configs, with a deprecation warning. |
| `TELEGRAM_GROUP_ALLOWED_CHATS` | Comma-separated group/forum chat IDs; any member is authorized |
| `TELEGRAM_HOME_CHANNEL` | Default Telegram chat/channel for cron delivery |
| `TELEGRAM_HOME_CHANNEL_NAME` | Display name for the Telegram home channel |
| `TELEGRAM_WEBHOOK_URL` | Public HTTPS URL for webhook mode (enables webhook instead of polling) |
| `TELEGRAM_WEBHOOK_PORT` | Local listen port for webhook server (default: `8443`) |
| `TELEGRAM_WEBHOOK_SECRET` | Secret token Telegram echoes back in each update for verification. **Required whenever `TELEGRAM_WEBHOOK_URL` is set** — the gateway refuses to start without it (GHSA-3vpc-7q5r-276h). Generate with `openssl rand -hex 32`. |
| `TELEGRAM_REACTIONS` | Enable emoji reactions on messages during processing (default: `false`) |
| `TELEGRAM_REPLY_TO_MODE` | Reply-reference behavior: `off`, `first` (default), or `all`. Matches the Discord pattern. |
| `TELEGRAM_IGNORED_THREADS` | Comma-separated Telegram forum topic/thread IDs where the bot never responds |
| `TELEGRAM_PROXY` | Proxy URL for Telegram connections — overrides `HTTPS_PROXY`. Supports `http://`, `https://`, `socks5://` |
| `DISCORD_BOT_TOKEN` | Discord bot token |
| `DISCORD_ALLOWED_USERS` | Comma-separated Discord user IDs allowed to use the bot |
| `DISCORD_ALLOWED_ROLES` | Comma-separated Discord role IDs allowed to use the bot (OR with `DISCORD_ALLOWED_USERS`). Auto-enables the Members intent. Useful when moderation teams churn — role grants propagate automatically. |
| `DISCORD_ALLOWED_CHANNELS` | Comma-separated Discord channel IDs. When set, the bot only responds in these channels (plus DMs if allowed). Overrides `config.yaml` `discord.allowed_channels`. |
| `DISCORD_PROXY` | Proxy URL for Discord connections — overrides `HTTPS_PROXY`. Supports `http://`, `https://`, `socks5://` |
| `DISCORD_HOME_CHANNEL` | Default Discord channel for cron delivery |
| `DISCORD_HOME_CHANNEL_NAME` | Display name for the Discord home channel |
| `DISCORD_COMMAND_SYNC_POLICY` | Discord slash-command startup sync policy: `safe` (diff and reconcile), `bulk` (legacy `tree.sync()`), or `off` |
| `DISCORD_REQUIRE_MENTION` | Require an @mention before responding in server channels |
| `DISCORD_FREE_RESPONSE_CHANNELS` | Comma-separated channel IDs where mention is not required |
| `DISCORD_AUTO_THREAD` | Auto-thread long replies when supported |
| `DISCORD_REACTIONS` | Enable emoji reactions on messages during processing (default: `true`) |
| `DISCORD_IGNORED_CHANNELS` | Comma-separated channel IDs where the bot never responds |
| `DISCORD_NO_THREAD_CHANNELS` | Comma-separated channel IDs where bot responds without auto-threading |
| `DISCORD_REPLY_TO_MODE` | Reply-reference behavior: `off`, `first` (default), or `all` |
| `DISCORD_ALLOW_MENTION_EVERYONE` | Allow the bot to ping `@everyone`/`@here` (default: `false`). See [Mention Control](../user-guide/messaging/discord.md#mention-control). |
| `DISCORD_ALLOW_MENTION_ROLES` | Allow the bot to ping `@role` mentions (default: `false`). |
| `DISCORD_ALLOW_MENTION_USERS` | Allow the bot to ping individual `@user` mentions (default: `true`). |
| `DISCORD_ALLOW_MENTION_REPLIED_USER` | Ping the author when replying to their message (default: `true`). |
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
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token (shared with telephony skill; also used for webhook signature validation) |
| `TWILIO_PHONE_NUMBER` | Twilio phone number in E.164 format (shared with telephony skill) |
| `SMS_WEBHOOK_URL` | Public URL for Twilio signature validation — must match the webhook URL in Twilio Console (required) |
| `SMS_WEBHOOK_PORT` | Webhook listener port for inbound SMS (default: `8080`) |
| `SMS_WEBHOOK_HOST` | Webhook bind address (default: `0.0.0.0`) |
| `SMS_INSECURE_NO_SIGNATURE` | Set to `true` to disable Twilio signature validation (local dev only — not for production) |
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
| `FEISHU_ALLOW_BOTS` | `none` (default) / `mentions` / `all` — accept inbound messages from other bots. See [bot-to-bot messaging](../user-guide/messaging/feishu.md#bot-to-bot-messaging) |
| `FEISHU_REQUIRE_MENTION` | `true` (default) / `false` — whether group messages must @mention the bot. Override per-chat via `group_rules.<chat_id>.require_mention`. |
| `FEISHU_HOME_CHANNEL` | Feishu chat ID for cron delivery and notifications |
| `WECOM_BOT_ID` | WeCom AI Bot ID from admin console |
| `WECOM_SECRET` | WeCom AI Bot secret |
| `WECOM_WEBSOCKET_URL` | Custom WebSocket URL (default: `wss://openws.work.weixin.qq.com`) |
| `WECOM_ALLOWED_USERS` | Comma-separated WeCom user IDs allowed to message the bot |
| `WECOM_HOME_CHANNEL` | WeCom chat ID for cron delivery and notifications |
| `WECOM_CALLBACK_CORP_ID` | WeCom enterprise Corp ID for callback self-built app |
| `WECOM_CALLBACK_CORP_SECRET` | Corp secret for the self-built app |
| `WECOM_CALLBACK_AGENT_ID` | Agent ID of the self-built app |
| `WECOM_CALLBACK_TOKEN` | Callback verification token |
| `WECOM_CALLBACK_ENCODING_AES_KEY` | AES key for callback encryption |
| `WECOM_CALLBACK_HOST` | Callback server bind address (default: `0.0.0.0`) |
| `WECOM_CALLBACK_PORT` | Callback server port (default: `8645`) |
| `WECOM_CALLBACK_ALLOWED_USERS` | Comma-separated user IDs for allowlist |
| `WECOM_CALLBACK_ALLOW_ALL_USERS` | Set `true` to allow all users without an allowlist |
| `WEIXIN_ACCOUNT_ID` | Weixin account ID obtained via QR login through iLink Bot API |
| `WEIXIN_TOKEN` | Weixin authentication token obtained via QR login through iLink Bot API |
| `WEIXIN_BASE_URL` | Override Weixin iLink Bot API base URL (default: `https://ilinkai.weixin.qq.com`) |
| `WEIXIN_CDN_BASE_URL` | Override Weixin CDN base URL for media (default: `https://novac2c.cdn.weixin.qq.com/c2c`) |
| `WEIXIN_DM_POLICY` | Direct message policy: `open`, `allowlist`, `pairing`, `disabled` (default: `open`) |
| `WEIXIN_GROUP_POLICY` | Group message policy: `open`, `allowlist`, `disabled` (default: `disabled`) |
| `WEIXIN_ALLOWED_USERS` | Comma-separated Weixin user IDs allowed to DM the bot |
| `WEIXIN_GROUP_ALLOWED_USERS` | Comma-separated Weixin **group chat IDs** (not member user IDs) allowed to interact with the bot. The variable name is legacy — it expects group IDs. Only takes effect when iLink actually delivers group events; QR-login iLink bot identities (`...@im.bot`) typically don't receive ordinary WeChat group messages. |
| `WEIXIN_HOME_CHANNEL` | Weixin chat ID for cron delivery and notifications |
| `WEIXIN_HOME_CHANNEL_NAME` | Display name for the Weixin home channel |
| `WEIXIN_ALLOW_ALL_USERS` | Allow all Weixin users without an allowlist (`true`/`false`) |
| `BLUEBUBBLES_SERVER_URL` | BlueBubbles server URL (e.g. `http://192.168.1.10:1234`) |
| `BLUEBUBBLES_PASSWORD` | BlueBubbles server password |
| `BLUEBUBBLES_WEBHOOK_HOST` | Webhook listener bind address (default: `127.0.0.1`) |
| `BLUEBUBBLES_WEBHOOK_PORT` | Webhook listener port (default: `8645`) |
| `BLUEBUBBLES_HOME_CHANNEL` | Phone/email for cron/notification delivery |
| `BLUEBUBBLES_ALLOWED_USERS` | Comma-separated authorized users |
| `BLUEBUBBLES_ALLOW_ALL_USERS` | Allow all users (`true`/`false`) |
| `QQ_APP_ID` | QQ Bot App ID from [q.qq.com](https://q.qq.com) |
| `QQ_CLIENT_SECRET` | QQ Bot App Secret from [q.qq.com](https://q.qq.com) |
| `QQ_STT_API_KEY` | API key for external STT fallback provider (optional, used when QQ built-in ASR returns no text) |
| `QQ_STT_BASE_URL` | Base URL for external STT provider (optional) |
| `QQ_STT_MODEL` | Model name for external STT provider (optional) |
| `QQ_ALLOWED_USERS` | Comma-separated QQ user openIDs allowed to message the bot |
| `QQ_GROUP_ALLOWED_USERS` | Comma-separated QQ group IDs for group @-message access |
| `QQ_ALLOW_ALL_USERS` | Allow all users (`true`/`false`, overrides `QQ_ALLOWED_USERS`) |
| `QQBOT_HOME_CHANNEL` | QQ user/group openID for cron delivery and notifications |
| `QQBOT_HOME_CHANNEL_NAME` | Display name for the QQ home channel |
| `QQ_PORTAL_HOST` | Override the QQ portal host (set to `sandbox.q.qq.com` to route through the sandbox gateway; default: `q.qq.com`). |
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
| `MATRIX_DEVICE_ID` | Stable Matrix device ID for E2EE persistence across restarts (e.g. `HERMES_BOT`). Without this, E2EE keys rotate every startup and historic-room decrypt breaks. |
| `MATRIX_REACTIONS` | Enable processing-lifecycle emoji reactions on inbound messages (default: `true`). Set to `false` to disable. |
| `MATRIX_REQUIRE_MENTION` | Require `@mention` in rooms (default: `true`). Set to `false` to respond to all messages. |
| `MATRIX_FREE_RESPONSE_ROOMS` | Comma-separated room IDs where bot responds without `@mention` |
| `MATRIX_AUTO_THREAD` | Auto-create threads for room messages (default: `true`) |
| `MATRIX_DM_MENTION_THREADS` | Create a thread when bot is `@mentioned` in a DM (default: `false`) |
| `MATRIX_RECOVERY_KEY` | Recovery key for cross-signing verification after device key rotation. Recommended for E2EE setups with cross-signing enabled. |
| `HASS_TOKEN` | Home Assistant Long-Lived Access Token (enables HA platform + tools) |
| `HASS_URL` | Home Assistant URL (default: `http://homeassistant.local:8123`) |
| `WEBHOOK_ENABLED` | Enable the webhook platform adapter (`true`/`false`) |
| `WEBHOOK_PORT` | HTTP server port for receiving webhooks (default: `8644`) |
| `WEBHOOK_SECRET` | Global HMAC secret for webhook signature validation (used as fallback when routes don't specify their own) |
| `API_SERVER_ENABLED` | Enable the OpenAI-compatible API server (`true`/`false`). Runs alongside other platforms. |
| `API_SERVER_KEY` | Bearer token for API server authentication. Enforced for non-loopback binding. |
| `API_SERVER_CORS_ORIGINS` | Comma-separated browser origins allowed to call the API server directly (for example `http://localhost:3000,http://127.0.0.1:3000`). Default: disabled. |
| `API_SERVER_PORT` | Port for the API server (default: `8642`) |
| `API_SERVER_HOST` | Host/bind address for the API server (default: `127.0.0.1`). Use `0.0.0.0` for network access — requires `API_SERVER_KEY` and a narrow `API_SERVER_CORS_ORIGINS` allowlist. |
| `API_SERVER_MODEL_NAME` | Model name advertised on `/v1/models`. Defaults to the profile name (or `hermes-agent` for the default profile). Useful for multi-user setups where frontends like Open WebUI need distinct model names per connection. |
| `GATEWAY_PROXY_URL` | URL of a remote Hermes API server to forward messages to ([proxy mode](/docs/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos)). When set, the gateway handles platform I/O only — all agent work is delegated to the remote server. Also configurable via `gateway.proxy_url` in `config.yaml`. |
| `GATEWAY_PROXY_KEY` | Bearer token for authenticating with the remote API server in proxy mode. Must match `API_SERVER_KEY` on the remote host. |
| `MESSAGING_CWD` | Working directory for terminal commands in messaging mode (default: `~`) |
| `GATEWAY_ALLOWED_USERS` | Comma-separated user IDs allowed across all platforms |
| `GATEWAY_ALLOW_ALL_USERS` | Allow all users without allowlists (`true`/`false`, default: `false`) |

### Advanced Messaging Tuning

Advanced per-platform knobs for throttling the outbound message batcher. Most users never need to touch these; defaults are set to respect each platform's rate limits without feeling sluggish.

| Variable | Description |
|----------|-------------|
| `HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS` | Grace window before flushing a queued Telegram text chunk (default: `0.6`). |
| `HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS` | Delay between split chunks when a single Telegram message exceeds the length limit (default: `2.0`). |
| `HERMES_TELEGRAM_MEDIA_BATCH_DELAY_SECONDS` | Grace window before flushing queued Telegram media (default: `0.6`). |
| `HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS` | Delay before sending a follow-up after the agent finishes, to avoid racing the last stream chunk. |
| `HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT` / `_READ_TIMEOUT` / `_WRITE_TIMEOUT` / `_POOL_TIMEOUT` | Override the underlying `python-telegram-bot` HTTP timeouts (seconds). |
| `HERMES_TELEGRAM_HTTP_POOL_SIZE` | Max concurrent HTTP connections to the Telegram API. |
| `HERMES_TELEGRAM_DISABLE_FALLBACK_IPS` | Disable the hard-coded Cloudflare fallback IPs used when DNS fails (`true`/`false`). |
| `HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS` | Grace window before flushing a queued Discord text chunk (default: `0.6`). |
| `HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS` | Delay between split chunks when a Discord message exceeds the length limit (default: `2.0`). |
| `HERMES_MATRIX_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | Matrix equivalents of the Telegram batch knobs. |
| `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` / `_MAX_CHARS` / `_MAX_MESSAGES` | Feishu batcher tuning — delay, split delay, max chars per message, max messages per batch. |
| `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | Feishu media flush delay. |
| `HERMES_FEISHU_DEDUP_CACHE_SIZE` | Size of the Feishu webhook dedup cache (default: `1024`). |
| `HERMES_WECOM_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | WeCom batcher tuning. |
| `HERMES_VISION_DOWNLOAD_TIMEOUT` | Timeout in seconds for downloading an image before handing it to vision models (default: `30`). |
| `HERMES_RESTART_DRAIN_TIMEOUT` | Gateway: seconds to wait for active runs to drain on `/restart` before forcing the restart (default: `900`). |
| `HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT` | Per-platform connect timeout during gateway startup (seconds). |
| `HERMES_GATEWAY_BUSY_INPUT_MODE` | Default gateway busy-input behavior: `queue`, `steer`, or `interrupt`. Can be overridden per chat with `/busy`. |
| `HERMES_GATEWAY_BUSY_ACK_ENABLED` | Whether the gateway sends an acknowledgment message (⚡/⏳/⏩) when a user sends input while the agent is busy (default: `true`). Set to `false` to suppress these messages entirely — the input is still queued/steered/interrupts as normal, only the chat reply is silenced. Bridged from `display.busy_ack_enabled` in `config.yaml`. |
| `HERMES_CRON_TIMEOUT` | Inactivity timeout for cron job agent runs in seconds (default: `600`). The agent can run indefinitely while actively calling tools or receiving stream tokens — this only triggers when idle. Set to `0` for unlimited. |
| `HERMES_CRON_SCRIPT_TIMEOUT` | Timeout for pre-run scripts attached to cron jobs in seconds (default: `120`). Override for scripts that need longer execution (e.g., randomized delays for anti-bot timing). Also configurable via `cron.script_timeout_seconds` in `config.yaml`. |
| `HERMES_CRON_MAX_PARALLEL` | Max cron jobs run in parallel per tick (default: `4`). |

## Agent Behavior

| Variable | Description |
|----------|-------------|
| `HERMES_MAX_ITERATIONS` | Max tool-calling iterations per conversation (default: 90) |
| `HERMES_INFERENCE_MODEL` | Override model name at process level (takes priority over `config.yaml` for the session). Also settable via `-m`/`--model` flag. |
| `HERMES_YOLO_MODE` | Set to `1` to bypass dangerous-command approval prompts. Equivalent to `--yolo`. |
| `HERMES_ACCEPT_HOOKS` | Auto-approve any unseen shell hooks declared in `config.yaml` without a TTY prompt. Equivalent to `--accept-hooks` or `hooks_auto_accept: true`. |
| `HERMES_IGNORE_USER_CONFIG` | Skip `~/.hermes/config.yaml` and use built-in defaults (credentials in `.env` still load). Equivalent to `--ignore-user-config`. |
| `HERMES_IGNORE_RULES` | Skip auto-injection of `AGENTS.md`, `SOUL.md`, `.cursorrules`, memory, and preloaded skills. Equivalent to `--ignore-rules`. |
| `HERMES_MD_NAMES` | Comma-separated list of rules-file names to auto-inject (default: `AGENTS.md,CLAUDE.md,.cursorrules,SOUL.md`). |
| `HERMES_TOOL_PROGRESS` | Deprecated compatibility variable for tool progress display. Prefer `display.tool_progress` in `config.yaml`. |
| `HERMES_TOOL_PROGRESS_MODE` | Deprecated compatibility variable for tool progress mode. Prefer `display.tool_progress` in `config.yaml`. |
| `HERMES_HUMAN_DELAY_MODE` | Response pacing: `off`/`natural`/`custom` |
| `HERMES_HUMAN_DELAY_MIN_MS` | Custom delay range minimum (ms) |
| `HERMES_HUMAN_DELAY_MAX_MS` | Custom delay range maximum (ms) |
| `HERMES_QUIET` | Suppress non-essential output (`true`/`false`) |
| `HERMES_API_TIMEOUT` | LLM API call timeout in seconds (default: `1800`) |
| `HERMES_API_CALL_STALE_TIMEOUT` | Non-streaming stale-call timeout in seconds (default: `300`). Auto-disabled for local providers when left unset. Also configurable via `providers.<id>.stale_timeout_seconds` or `providers.<id>.models.<model>.stale_timeout_seconds` in `config.yaml`. |
| `HERMES_STREAM_READ_TIMEOUT` | Streaming socket read timeout in seconds (default: `120`). Auto-increased to `HERMES_API_TIMEOUT` for local providers. Increase if local LLMs time out during long code generation. |
| `HERMES_STREAM_STALE_TIMEOUT` | Stale stream detection timeout in seconds (default: `180`). Auto-disabled for local providers. Triggers connection kill if no chunks arrive within this window. |
| `HERMES_STREAM_RETRIES` | Number of mid-stream reconnect attempts on transient network errors (default: `3`). |
| `HERMES_AGENT_TIMEOUT` | Gateway inactivity timeout for a running agent in seconds (default: `900`). Resets on every tool call and streamed token. Set to `0` to disable. |
| `HERMES_AGENT_TIMEOUT_WARNING` | Gateway: send a warning message after this many seconds of inactivity (default: 75% of `HERMES_AGENT_TIMEOUT`). |
| `HERMES_AGENT_NOTIFY_INTERVAL` | Gateway: interval in seconds between progress notifications on long-running agent turns. |
| `HERMES_CHECKPOINT_TIMEOUT` | Timeout for filesystem checkpoint creation in seconds (default: `30`). |
| `HERMES_EXEC_ASK` | Enable execution approval prompts in gateway mode (`true`/`false`) |
| `HERMES_ENABLE_PROJECT_PLUGINS` | Enable auto-discovery of repo-local plugins from `./.hermes/plugins/` (`true`/`false`, default: `false`) |
| `HERMES_BACKGROUND_NOTIFICATIONS` | Background process notification mode in gateway: `all` (default), `result`, `error`, `off` |
| `HERMES_EPHEMERAL_SYSTEM_PROMPT` | Ephemeral system prompt injected at API-call time (never persisted to sessions) |
| `HERMES_PREFILL_MESSAGES_FILE` | Path to a JSON file of ephemeral prefill messages injected at API-call time. |
| `HERMES_ALLOW_PRIVATE_URLS` | `true`/`false` — allow tools to fetch localhost/private-network URLs. Off by default in gateway mode. |
| `HERMES_REDACT_SECRETS` | `true`/`false` — control secret redaction in logs and shareable outputs (default: `true`). |
| `HERMES_WRITE_SAFE_ROOT` | Optional directory prefix that restricts `write_file`/`patch` writes; paths outside require approval. |
| `HERMES_DISABLE_FILE_STATE_GUARD` | Set to `1` to turn off the "file changed since you read it" guard on `patch`/`write_file`. |
| `HERMES_CORE_TOOLS` | Comma-separated override for the canonical core tool list (advanced; rarely needed). |
| `HERMES_BUNDLED_SKILLS` | Comma-separated override for the list of bundled skills loaded at startup. |
| `HERMES_OPTIONAL_SKILLS` | Comma-separated list of optional-skill names to auto-install on first run. |
| `HERMES_DEBUG_INTERRUPT` | Set to `1` to log detailed interrupt/cancel tracing to `agent.log`. |
| `HERMES_DUMP_REQUESTS` | Dump API request payloads to log files (`true`/`false`) |
| `HERMES_DUMP_REQUEST_STDOUT` | Dump API request payloads to stdout instead of log files. |
| `HERMES_OAUTH_TRACE` | Set to `1` to log OAuth token exchange and refresh attempts. Includes redacted timing info. |
| `HERMES_OAUTH_FILE` | Override the path used for OAuth credential storage (default: `~/.hermes/auth.json`). |
| `HERMES_AGENT_HELP_GUIDANCE` | Append additional guidance text to the system prompt for custom deployments. |
| `HERMES_AGENT_LOGO` | Override the ASCII banner logo at CLI startup. |
| `DELEGATION_MAX_CONCURRENT_CHILDREN` | Max parallel subagents per `delegate_task` batch (default: `3`, floor of 1, no ceiling). Also configurable via `delegation.max_concurrent_children` in `config.yaml` — the config value takes priority. |

## Interface

| Variable | Description |
|----------|-------------|
| `HERMES_TUI` | Launch the [TUI](../user-guide/tui.md) instead of the classic CLI when set to `1`. Equivalent to passing `--tui`. |
| `HERMES_TUI_DIR` | Path to a prebuilt `ui-tui/` directory (must contain `dist/entry.js` and populated `node_modules`). Used by distros and Nix to skip the first-launch `npm install`. |
| `HERMES_TUI_RESUME` | Resume a specific TUI session by ID on launch. When set, `hermes --tui` skips forging a fresh session and picks up the named session instead — useful for re-attaching after a disconnect or terminal crash. |
| `HERMES_TUI_THEME` | Force the TUI color theme: `light`, `dark`, or a raw 6-character background hex (e.g. `ffffff` or `1a1a2e`). When unset, Hermes auto-detects using `COLORFGBG` and terminal background queries; this variable overrides detection on terminals (Ghostty, Warp, iTerm2, etc.) that don't set `COLORFGBG`. |
| `HERMES_INFERENCE_MODEL` | Force the model for `hermes -z` / `hermes chat` without mutating `config.yaml`. Pairs with `HERMES_INFERENCE_PROVIDER`. Useful for scripted callers (sweeper, CI, batch runners) that need to override the default model per run. |

## Session Settings

| Variable | Description |
|----------|-------------|
| `SESSION_IDLE_MINUTES` | Reset sessions after N minutes of inactivity (default: 1440) |
| `SESSION_RESET_HOUR` | Daily reset hour in 24h format (default: 4 = 4am) |

## Context Compression (config.yaml only)

Context compression is configured exclusively through `config.yaml` — there are no environment variables for it. Threshold settings live in the `compression:` block, while the summarization model/provider lives under `auxiliary.compression:`.

```yaml
compression:
  enabled: true
  threshold: 0.50
  target_ratio: 0.20         # fraction of threshold to preserve as recent tail
  protect_last_n: 20         # minimum recent messages to keep uncompressed
```

:::info Legacy migration
Older configs with `compression.summary_model`, `compression.summary_provider`, and `compression.summary_base_url` are automatically migrated to `auxiliary.compression.*` on first load.
:::

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
