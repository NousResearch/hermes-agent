---
sidebar_position: 2
title: "Environment Variables"
description: "Complete reference of all environment variables used by Hermes Agent"
---

# Environment Variables Reference

Hermes reads environment variables from the process environment and, for user-managed secrets, from `~/.hermes/.env`. Keep API keys, bot tokens, OAuth secrets, and other credentials in `.env`; prefer `config.yaml` for non-secret behaviour settings when a config key exists. Some variables below are process-only overrides or internal bridge variables and should not be committed to `.env` just because they are documented here.

## LLM Providers

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key (recommended for flexibility) |
| `OPENROUTER_BASE_URL` | Override the OpenRouter-compatible base URL |
| `HERMES_OPENROUTER_CACHE` | Enable OpenRouter response caching (`1`/`true`/`yes`/`on`). Overrides `openrouter.response_cache` in config.yaml. See [Response Caching](https://openrouter.ai/docs/guides/features/response-caching). |
| `HERMES_OPENROUTER_CACHE_TTL` | Cache TTL in seconds (1-86400). Overrides `openrouter.response_cache_ttl` in config.yaml. |
| `NOUS_BASE_URL` | Override Nous Portal base URL (rarely needed; development/testing only) |
| `NOUS_INFERENCE_BASE_URL` | Override Nous inference endpoint directly |
| `OPENAI_API_KEY` | API key for custom OpenAI-compatible endpoints (used with `OPENAI_BASE_URL`) |
| `OPENAI_BASE_URL` | Base URL for custom endpoint (VLLM, SGLang, etc.) |
| `LM_API_KEY` | API key for LM Studio (`lmstudio` provider). Often a placeholder for local servers |
| `LM_BASE_URL` | LM Studio base URL (default: `http://localhost:1234/v1`) |
| `COPILOT_GITHUB_TOKEN` | GitHub token for Copilot API — first priority (OAuth `gho_*` or fine-grained PAT `github_pat_*`; classic PATs `ghp_*` are **not supported**) |
| `GH_TOKEN` | GitHub token — second priority for Copilot (also used by `gh` CLI) |
| `GITHUB_TOKEN` | GitHub token — third priority for Copilot |
| `HERMES_COPILOT_ACP_COMMAND` | Override Copilot ACP CLI binary path (default: `copilot`) |
| `COPILOT_CLI_PATH` | Alias for `HERMES_COPILOT_ACP_COMMAND` |
| `HERMES_COPILOT_ACP_ARGS` | Override Copilot ACP arguments (default: `--acp --stdio`) |
| `COPILOT_ACP_BASE_URL` | Override Copilot ACP base URL |
| `COPILOT_API_BASE_URL` | Override the Copilot API base URL (`copilot` provider) |
| `GLM_API_KEY` | z.ai / ZhipuAI GLM API key ([z.ai](https://z.ai)) |
| `ZAI_API_KEY` | Alias for `GLM_API_KEY` |
| `Z_AI_API_KEY` | Alias for `GLM_API_KEY` |
| `GLM_BASE_URL` | Override z.ai base URL (default: `https://api.z.ai/api/paas/v4`) |
| `KIMI_API_KEY` | Kimi / Moonshot AI API key ([moonshot.ai](https://platform.moonshot.ai)) |
| `KIMI_CODING_API_KEY` | Alias key for the `kimi-coding` provider (accepted alongside `KIMI_API_KEY`) |
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
| `AZURE_FOUNDRY_API_KEY` | Microsoft Foundry / Azure OpenAI API key ([ai.azure.com](https://ai.azure.com/)). Not needed when `model.auth_mode: entra_id` |
| `AZURE_FOUNDRY_BASE_URL` | Microsoft Foundry endpoint URL (e.g. `https://<resource>.openai.azure.com/openai/v1` for OpenAI-style, or `https://<resource>.services.ai.azure.com/anthropic` for Anthropic-style) |
| `AZURE_ANTHROPIC_KEY` | Azure Anthropic API key for `provider: anthropic` + `base_url` pointing at a Microsoft Foundry Claude deployment (alternative to `ANTHROPIC_API_KEY` when both Anthropic and Azure Anthropic are configured) |
| `AZURE_TENANT_ID` | Entra ID tenant ID (service-principal flows; honored by `azure-identity` when `model.auth_mode: entra_id`) |
| `AZURE_CLIENT_ID` | Entra ID client ID (service principal, workload identity, or user-assigned managed identity) |
| `AZURE_CLIENT_SECRET` | Service principal secret used by `EnvironmentCredential` |
| `AZURE_CLIENT_CERTIFICATE_PATH` | Service principal certificate (alternative to `AZURE_CLIENT_SECRET`) |
| `AZURE_FEDERATED_TOKEN_FILE` | Federated token file path for AKS Workload Identity / OIDC flows |
| `AZURE_AUTHORITY_HOST` | Sovereign-cloud authority override (e.g. `https://login.microsoftonline.us` for Azure Government). See [Azure Foundry guide](/guides/azure-foundry#sovereign-clouds-government-china) |
| `IDENTITY_ENDPOINT` / `MSI_ENDPOINT` | Managed Identity endpoint for App Service, Functions, and Container Apps; VMs usually use IMDS instead and do not set these |
| `HF_TOKEN` | Hugging Face token for Inference Providers ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) |
| `HF_BASE_URL` | Override Hugging Face base URL (default: `https://router.huggingface.co/v1`) |
| `GOOGLE_API_KEY` | Google AI Studio API key ([aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)) |
| `GEMINI_API_KEY` | Alias for `GOOGLE_API_KEY` |
| `GEMINI_BASE_URL` | Override Google AI Studio base URL |
| `ANTHROPIC_API_KEY` | Anthropic Console API key ([console.anthropic.com](https://console.anthropic.com/)) |
| `ANTHROPIC_BASE_URL` | Override the Anthropic API base URL |
| `ANTHROPIC_TOKEN` | Manual or legacy Anthropic OAuth/setup-token override |
| `DASHSCOPE_API_KEY` | Qwen Cloud (Alibaba DashScope) API key for Qwen models ([modelstudio.console.alibabacloud.com](https://modelstudio.console.alibabacloud.com/)) |
| `DASHSCOPE_BASE_URL` | Custom DashScope base URL (default: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`; use `https://dashscope.aliyuncs.com/compatible-mode/v1` for mainland-China region) |
| `ALIBABA_CODING_PLAN_API_KEY` | Qwen Coding Plan API key (`alibaba-coding-plan` provider) |
| `ALIBABA_CODING_PLAN_BASE_URL` | Override the Qwen Coding Plan base URL |
| `DEEPSEEK_API_KEY` | DeepSeek API key for direct DeepSeek access ([platform.deepseek.com](https://platform.deepseek.com/api_keys)) |
| `DEEPSEEK_BASE_URL` | Custom DeepSeek API base URL |
| `NOVITA_API_KEY` | NovitaAI API key — AI-native cloud for Model API, Agent Sandbox, and GPU Cloud ([novita.ai/settings/key-management](https://novita.ai/settings/key-management)) |
| `NOVITA_BASE_URL` | Override NovitaAI base URL (default: `https://api.novita.ai/openai/v1`) |
| `NVIDIA_API_KEY` | NVIDIA NIM API key — Nemotron and open models ([build.nvidia.com](https://build.nvidia.com)) |
| `NVIDIA_BASE_URL` | Override NVIDIA base URL (default: `https://integrate.api.nvidia.com/v1`; set to `http://localhost:8000/v1` for a local NIM endpoint) |
| `STEPFUN_API_KEY` | StepFun API key — Step-series models ([platform.stepfun.com](https://platform.stepfun.com)) |
| `STEPFUN_BASE_URL` | Override StepFun base URL (default: `https://api.stepfun.com/v1`) |
| `OLLAMA_API_KEY` | Ollama Cloud API key — managed Ollama catalog without local GPU ([ollama.com/settings/keys](https://ollama.com/settings/keys)) |
| `OLLAMA_BASE_URL` | Override Ollama Cloud base URL (default: `https://ollama.com/v1`) |
| `XAI_API_KEY` | xAI (Grok) API key for chat + TTS + web search ([console.x.ai](https://console.x.ai/)) |
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
| `HERMES_GIT_BASH_PATH` | **Windows only.** Override `bash.exe` discovery for the terminal tool. Points at any bash — full Git-for-Windows install, WSL bash via symlink, MSYS2, Cygwin. The installer sets this automatically to the PortableGit it provisioned. See the [Windows (Native) Guide](../user-guide/windows-native.md#how-hermes-runs-shell-commands-on-windows) |
| `HERMES_DISABLE_WINDOWS_UTF8` | **Windows only.** Set to `1` to disable the UTF-8 stdio shim (`configure_windows_stdio()`) and fall back to the console's locale code page. Useful for bisecting encoding bugs; rarely the right setting in normal operation |
| `HERMES_KANBAN_HOME` | Override the shared Hermes root that anchors the kanban board (db + workspaces + worker logs). Falls back to `get_default_hermes_root()` (the parent of any active profile). Useful for tests and unusual deployments |
| `HERMES_KANBAN_BOARD` | Pin the active kanban board for this process. Takes precedence over `~/.hermes/kanban/current`; the dispatcher injects this into worker subprocess env so workers physically cannot see tasks on other boards. Defaults to `default`. Slug validation: lowercase alphanumerics + hyphens + underscores, 1-64 chars |
| `HERMES_KANBAN_DB` | Pin the kanban database file path directly (highest precedence; beats `HERMES_KANBAN_BOARD` and `HERMES_KANBAN_HOME`). The dispatcher injects this into worker subprocess env so profile workers converge on the dispatcher's board |
| `HERMES_KANBAN_WORKSPACES_ROOT` | Pin the kanban workspaces root directly (highest precedence for workspaces; beats `HERMES_KANBAN_HOME`). The dispatcher injects this into worker subprocess env |
| `HERMES_KANBAN_DISPATCH_IN_GATEWAY` | Runtime override for `kanban.dispatch_in_gateway`. Set to `0`, `false`, `no`, or `off` to keep the gateway from starting the embedded Kanban dispatcher; any other non-empty value enables it. Useful when a separate dispatcher process owns the board. |

## Provider Auth (OAuth)

For native Anthropic auth, Hermes prefers Claude Code's own credential files when they exist because those credentials can refresh automatically. **OAuth against Anthropic requires a Claude Max plan with purchased extra usage credits** — Hermes routes as Claude Code, which only draws from the Max plan's extra/overage credits, not the base Max allowance, and does not work on Claude Pro. Without Max + extra credits, use an API key instead. Environment variables such as `ANTHROPIC_TOKEN` remain useful as manual overrides, but they are no longer the preferred path for Claude Max login.

| Variable | Description |
|----------|-------------|
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
| `SEARXNG_URL` | SearXNG instance URL for free self-hosted web search — no API key required ([searxng.github.io](https://searxng.github.io/searxng/)) |
| `TAVILY_BASE_URL` | Override the Tavily API endpoint. Useful for corporate proxies and self-hosted Tavily-compatible search backends. Same pattern as `GROQ_BASE_URL`. |
| `EXA_API_KEY` | Exa API key for AI-native web search and contents ([exa.ai](https://exa.ai/)) |
| `BRAVE_SEARCH_API_KEY` | Brave Search API subscription token for web search (free tier available) ([brave.com/search/api](https://brave.com/search/api/)) |
| `BROWSERBASE_API_KEY` | Browser automation ([browserbase.com](https://browserbase.com/)) |
| `BROWSERBASE_PROJECT_ID` | Browserbase project ID |
| `BROWSER_USE_API_KEY` | Browser Use cloud browser API key ([browser-use.com](https://browser-use.com/)) |
| `FIRECRAWL_BROWSER_TTL` | Firecrawl browser session TTL in seconds (default: 300) |
| `BROWSER_CDP_URL` | Chrome DevTools Protocol URL for local browser (set via `/browser connect`, e.g. `ws://localhost:9222`) |
| `CAMOFOX_URL` | Camofox local anti-detection browser URL (default: `http://localhost:9377`) |
| `CAMOFOX_USER_ID` | Optional externally managed Camofox user ID for shared visible sessions |
| `CAMOFOX_SESSION_KEY` | Optional Camofox session key used when creating tabs for `CAMOFOX_USER_ID` |
| `CAMOFOX_ADOPT_EXISTING_TAB` | Set to `true` to reuse an existing Camofox tab before creating a new one |
| `BROWSER_INACTIVITY_TIMEOUT` | Browser session inactivity timeout in seconds |
| `AGENT_BROWSER_ARGS` | Extra Chromium launch flags (comma- or newline-separated). Hermes auto-injects `--no-sandbox,--disable-dev-shm-usage` when running as root or on AppArmor-restricted unprivileged user namespaces (Ubuntu 23.10+, DGX Spark, many container images); set this manually only to override or add other flags. |
| `AGENT_BROWSER_ENGINE` | Browser engine for local mode: `auto` (default — Chromium-family via CDP), or a specific engine override. |
| `FAL_KEY` | Image generation ([fal.ai](https://fal.ai/)) |
| `KREA_API_KEY` | Krea API key for Krea 2 image generation ([krea.ai](https://krea.ai/)) |
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
| `DAYTONA_API_KEY` | Daytona cloud sandboxes ([daytona.io](https://daytona.io/)) |

### Skill API Keys

Secrets consumed by specific bundled / optional skills. Each is only needed if you use the corresponding skill.

| Variable | Used by skill | Description |
|----------|---------------|-------------|
| `NOTION_API_KEY` | `notion` | Notion integration token. |
| `LINEAR_API_KEY` | `linear` | Linear personal API key. |
| `AIRTABLE_API_KEY` | `airtable` | Airtable personal access token. |
| `TENOR_API_KEY` | `gif-search` | Tenor API key for GIF search. |

### Langfuse Observability

Environment variables for the bundled [`observability/langfuse`](/user-guide/features/built-in-plugins#observabilitylangfuse) plugin. Set these in `~/.hermes/.env`. The plugin must also be enabled (`hermes plugins enable observability/langfuse`, or check the box in `hermes plugins`) before any of these take effect.

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

These variables configure the [Tool Gateway](/user-guide/features/tool-gateway) for paid Nous subscribers or self-hosted gateway deployments. Most users don't need to set these — the gateway is configured automatically via `hermes model` or `hermes tools`.

| Variable | Description |
|----------|-------------|
| `TOOL_GATEWAY_DOMAIN` | Base domain for Tool Gateway routing (default: `nousresearch.com`) |
| `TOOL_GATEWAY_SCHEME` | HTTP or HTTPS scheme for gateway URLs (default: `https`) |
| `TOOL_GATEWAY_USER_TOKEN` | Auth token for the Tool Gateway (normally auto-populated from Nous auth) |
| `FIRECRAWL_GATEWAY_URL` | Override URL for the Firecrawl gateway endpoint specifically |

## Terminal Backend

| Variable | Description |
|----------|-------------|
| `TERMINAL_ENV` | Backend: `local`, `docker`, `ssh`, `singularity`, `modal`, `daytona` |
| `HERMES_DOCKER_BINARY` | Override the container binary Hermes shells out to (e.g. `podman`, `/usr/local/bin/docker`). When unset, Hermes auto-discovers `docker` or `podman` on `PATH`. Needed when both are installed and you want the non-default, or when the binary lives outside `PATH`. |
| `TERMINAL_DOCKER_IMAGE` | Docker image (default: `nikolaik/python-nodejs:python3.11-nodejs20`) |
| `TERMINAL_DOCKER_FORWARD_ENV` | JSON array of env var names to explicitly forward into Docker terminal sessions. Note: skill-declared `required_environment_variables` are forwarded automatically — you only need this for vars not declared by any skill. |
| `TERMINAL_DOCKER_VOLUMES` | Additional Docker volume mounts (comma-separated `host:container` pairs) |
| `TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE` | Advanced opt-in: mount the launch cwd into Docker `/workspace` (`true`/`false`, default: `false`) |
| `TERMINAL_SINGULARITY_IMAGE` | Singularity image or `.sif` path |
| `TERMINAL_MODAL_IMAGE` | Modal container image |
| `TERMINAL_DAYTONA_IMAGE` | Daytona sandbox image |
| `TERMINAL_TIMEOUT` | Command timeout in seconds |
| `TERMINAL_LIFETIME_SECONDS` | Max lifetime for terminal sessions in seconds |
| `TERMINAL_CWD` | Deprecated direct override for gateway/cron terminal sessions. Prefer `terminal.cwd` in `config.yaml`; CLI still uses the launch directory. |
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
| `TELEGRAM_ALLOW_ALL_USERS` | Allow any Telegram user to trigger the bot (dev only). |
| `TELEGRAM_GROUP_ALLOWED_USERS` | Comma-separated sender user IDs authorized in groups/forums only (does NOT grant DM access). Chat-ID-shaped values (starting with `-`) are still honored as chat IDs for backward compat with pre-#17686 configs, with a deprecation warning. |
| `TELEGRAM_GROUP_ALLOWED_CHATS` | Comma-separated group/forum chat IDs; any member is authorized |
| `TELEGRAM_HOME_CHANNEL` | Default Telegram chat/channel for cron delivery |
| `TELEGRAM_HOME_CHANNEL_NAME` | Display name for the Telegram home channel |
| `TELEGRAM_CRON_THREAD_ID` | Forum topic ID to receive cron deliveries; overrides `TELEGRAM_HOME_CHANNEL_THREAD_ID` for cron only. Use in topic mode so replies to cron messages open a new session instead of hitting the system lobby (#24409). |
| `TELEGRAM_WEBHOOK_URL` | Public HTTPS URL for webhook mode (enables webhook instead of polling) |
| `TELEGRAM_WEBHOOK_PORT` | Local listen port for webhook server (default: `8443`) |
| `TELEGRAM_WEBHOOK_SECRET` | Secret token Telegram echoes back in each update for verification. **Required whenever `TELEGRAM_WEBHOOK_URL` is set** — the gateway refuses to start without it (GHSA-3vpc-7q5r-276h). Generate with `openssl rand -hex 32`. |
| `TELEGRAM_REACTIONS` | Enable emoji reactions on messages during processing (default: `false`) |
| `TELEGRAM_REQUIRE_MENTION` | Require an explicit trigger before responding in Telegram groups. Equivalent to `telegram.require_mention` in `config.yaml`. |
| `TELEGRAM_MENTION_PATTERNS` | JSON array, newline-separated list, or comma-separated list of regex wake-word patterns accepted when Telegram group mention gating is enabled. Equivalent to `telegram.mention_patterns`. |
| `TELEGRAM_EXCLUSIVE_BOT_MENTIONS` | When enabled, explicit `@...bot` mentions in Telegram groups route only to the mentioned bot usernames before reply or wake-word fallbacks run. Default: `true`. Equivalent to `telegram.exclusive_bot_mentions`. |
| `TELEGRAM_REPLY_TO_MODE` | Reply-reference behavior: `off`, `first` (default), or `all`. Matches the Discord pattern. |
| `TELEGRAM_IGNORED_THREADS` | Comma-separated Telegram forum topic/thread IDs where the bot never responds |
| `TELEGRAM_PROXY` | Proxy URL for Telegram connections — overrides `HTTPS_PROXY`. Supports `http://`, `https://`, `socks5://` |
| `DISCORD_BOT_TOKEN` | Discord bot token |
| `DISCORD_ALLOWED_USERS` | Comma-separated Discord user IDs allowed to use the bot |
| `DISCORD_ALLOW_ALL_USERS` | Allow any Discord user to trigger the bot (dev only). |
| `DISCORD_ALLOWED_ROLES` | Comma-separated Discord role IDs allowed to use the bot (OR with `DISCORD_ALLOWED_USERS`). Auto-enables the Members intent. Useful when moderation teams churn — role grants propagate automatically. |
| `DISCORD_ALLOWED_CHANNELS` | Comma-separated Discord channel IDs. When set, the bot only responds in these channels (plus DMs if allowed). Overrides `config.yaml` `discord.allowed_channels`. |
| `DISCORD_PROXY` | Proxy URL for Discord connections — overrides `HTTPS_PROXY`. Supports `http://`, `https://`, `socks5://` |
| `DISCORD_HOME_CHANNEL` | Default Discord channel for cron delivery |
| `DISCORD_HOME_CHANNEL_NAME` | Display name for the Discord home channel |
| `DISCORD_COMMAND_SYNC_POLICY` | Discord slash-command startup sync policy: `safe` (diff and reconcile), `bulk` (legacy `tree.sync()`), or `off` |
| `DISCORD_REQUIRE_MENTION` | Require an @mention before responding in server channels |
| `DISCORD_FREE_RESPONSE_CHANNELS` | Comma-separated channel IDs where mention is not required |
| `DISCORD_AUTO_THREAD` | Auto-thread long replies when supported |
| `DISCORD_ALLOW_ANY_ATTACHMENT` | When `true`, accept attachments of any file type (not just the built-in PDF/text/zip/office allowlist). Unknown types are cached and surfaced to the agent as a local path so it can inspect them via `terminal` / `read_file` / `ffprobe`. Default `false`. |
| `DISCORD_MAX_ATTACHMENT_BYTES` | Maximum bytes per attachment the gateway will cache. Default `33554432` (32 MiB). Set to `0` for no cap (attachments are held in memory while being written). |
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
| `SLACK_ALLOW_ALL_USERS` | Allow any Slack user to trigger the bot (dev only). |
| `SLACK_HOME_CHANNEL` | Default Slack channel for cron delivery |
| `SLACK_HOME_CHANNEL_NAME` | Display name for the Slack home channel |
| `GOOGLE_CHAT_PROJECT_ID` | GCP project hosting the Pub/Sub topic (falls back to `GOOGLE_CLOUD_PROJECT`) |
| `GOOGLE_CHAT_SUBSCRIPTION_NAME` | Full Pub/Sub subscription path, `projects/{proj}/subscriptions/{sub}` (legacy alias: `GOOGLE_CHAT_SUBSCRIPTION`) |
| `GOOGLE_CHAT_SERVICE_ACCOUNT_JSON` | Path to Service Account JSON, or the JSON inline (falls back to `GOOGLE_APPLICATION_CREDENTIALS`) |
| `GOOGLE_CHAT_ALLOWED_USERS` | Comma-separated user emails allowed to chat with the bot |
| `GOOGLE_CHAT_ALLOW_ALL_USERS` | Allow any Google Chat user to trigger the bot (dev only) |
| `GOOGLE_CHAT_HOME_CHANNEL` | Default space (e.g. `spaces/AAAA...`) for cron delivery |
| `GOOGLE_CHAT_HOME_CHANNEL_NAME` | Display name for the Google Chat home space |
| `GOOGLE_CHAT_MAX_MESSAGES` | Pub/Sub FlowControl max in-flight messages (default: `1`) |
| `GOOGLE_CHAT_MAX_BYTES` | Pub/Sub FlowControl max in-flight bytes (default: `16777216`, 16 MiB) |
| `GOOGLE_CHAT_BOOTSTRAP_SPACES` | Comma-separated extra space IDs to probe at startup when resolving the bot's own `users/{id}` |
| `GOOGLE_CHAT_DEBUG_RAW` | Set to any value to log redacted Pub/Sub envelopes at DEBUG level (debugging only) |
| `WHATSAPP_ENABLED` | Enable the WhatsApp bridge (`true`/`false`) |
| `WHATSAPP_MODE` | `bot` (separate number) or `self-chat` (message yourself) |
| `WHATSAPP_ALLOWED_USERS` | Comma-separated phone numbers (with country code, no `+`), or `*` to allow all senders |
| `WHATSAPP_ALLOW_ALL_USERS` | Allow all WhatsApp senders without an allowlist (`true`/`false`) |
| `WHATSAPP_HOME_CHANNEL` | Default chat ID for cron / notification delivery. |
| `WHATSAPP_HOME_CHANNEL_NAME` | Display name for the WhatsApp home channel. |
| `WHATSAPP_DEBUG` | Log raw message events in the bridge for troubleshooting (`true`/`false`) |
| `WHATSAPP_CLOUD_PHONE_NUMBER_ID` | Meta Phone Number ID from the WhatsApp Business Cloud API (15–17 digits; **not** the phone number itself) |
| `WHATSAPP_CLOUD_ACCESS_TOKEN` | Meta access token (starts with `EAA`); temporary tokens expire after 24h, System User tokens are permanent |
| `WHATSAPP_CLOUD_APP_SECRET` | 32-char hex app secret used to verify inbound webhook signatures |
| `WHATSAPP_CLOUD_VERIFY_TOKEN` | Shared secret for Meta's webhook verification handshake (auto-generated by the setup wizard) |
| `WHATSAPP_CLOUD_ALLOWED_USERS` | Comma-separated `wa_id`s (phone numbers with country code, no `+`) allowed to message the bot |
| `WHATSAPP_CLOUD_ALLOW_ALL_USERS` | Allow all WhatsApp Cloud senders without an allowlist (`true`/`false`) |
| `WHATSAPP_CLOUD_APP_ID` | Optional Meta App ID (for future analytics integration) |
| `WHATSAPP_CLOUD_WABA_ID` | Optional WhatsApp Business Account ID (for future analytics integration) |
| `WHATSAPP_CLOUD_WEBHOOK_HOST` | Interface the inbound webhook server binds to (default `0.0.0.0`) |
| `WHATSAPP_CLOUD_WEBHOOK_PORT` | Port the inbound webhook server binds to (default `8090`) |
| `WHATSAPP_CLOUD_WEBHOOK_PATH` | URL path Meta posts inbound messages to (default `/whatsapp/webhook`) |
| `WHATSAPP_CLOUD_API_VERSION` | Meta Graph API version to call (default `v20.0`) |
| `WHATSAPP_CLOUD_HOME_CHANNEL` | `wa_id` to use as the bot's home channel (for cron jobs etc.) |
| `WHATSAPP_CLOUD_DM_POLICY` | DM gating for the Cloud adapter (`open`/`allowlist`/`disabled`); falls back to `WHATSAPP_DM_POLICY` when unset |
| `WHATSAPP_CLOUD_ALLOW_FROM` | Comma-separated senders allowed when `dm_policy: allowlist` (bare `wa_id`s; Baileys-style JIDs are normalized) |
| `WHATSAPP_CLOUD_GROUP_POLICY` | Group gating for the Cloud adapter (`open`/`allowlist`/`disabled`); falls back to `WHATSAPP_GROUP_POLICY` when unset |
| `WHATSAPP_CLOUD_GROUP_ALLOW_FROM` | Comma-separated group chat IDs allowed when `group_policy: allowlist` |
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
| `DINGTALK_WEBHOOK_URL` | Static robot webhook URL for cross-platform / cron delivery. |
| `DINGTALK_HOME_CHANNEL` | Default conversation ID for cron / notification delivery. |
| `DINGTALK_HOME_CHANNEL_NAME` | Display name for the DingTalk home channel. |
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
| `FEISHU_HOME_CHANNEL_NAME` | Display name for the Feishu home channel. |
| `FEISHU_ALLOW_ALL_USERS` | Allow any Feishu user to trigger the bot (dev only). |
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
| `QQ_SANDBOX` | Enable QQ sandbox mode for development testing (`true`/`false`) |
| `MATTERMOST_URL` | Mattermost server URL (e.g. `https://mm.example.com`) |
| `MATTERMOST_TOKEN` | Bot token or personal access token for Mattermost |
| `MATTERMOST_ALLOWED_USERS` | Comma-separated Mattermost user IDs allowed to message the bot |
| `MATTERMOST_ALLOW_ALL_USERS` | Allow any Mattermost user to trigger the bot (dev only). |
| `MATTERMOST_ALLOWED_CHANNELS` | If set, the bot only responds in these channels (whitelist). |
| `MATTERMOST_HOME_CHANNEL` | Channel ID for proactive message delivery (cron, notifications) |
| `MATTERMOST_REQUIRE_MENTION` | Require `@mention` in channels (default: `true`). Set to `false` to respond to all messages. |
| `MATTERMOST_FREE_RESPONSE_CHANNELS` | Comma-separated channel IDs where bot responds without `@mention` |
| `MATTERMOST_REPLY_MODE` | Reply style: `thread` (threaded replies) or `off` (flat messages, default) |
| `MATRIX_HOMESERVER` | Matrix homeserver URL (e.g. `https://matrix.org`) |
| `MATRIX_ACCESS_TOKEN` | Matrix access token for bot authentication |
| `MATRIX_USER_ID` | Matrix user ID (e.g. `@hermes:matrix.org`) — required for password login, optional with access token |
| `MATRIX_PASSWORD` | Matrix password (alternative to access token) |
| `MATRIX_ALLOWED_USERS` | Comma-separated Matrix user IDs allowed to message the bot (e.g. `@alice:matrix.org`) |
| `MATRIX_ALLOW_ALL_USERS` | Allow any Matrix user to trigger the bot (dev only). |
| `MATRIX_HOME_CHANNEL` | Default room ID for cron / notification delivery. |
| `MATRIX_HOME_CHANNEL_NAME` | Display name for the Matrix home room. |
| `MATRIX_ALLOWED_ROOMS` | Comma-separated Matrix room IDs allowed to trigger bot responses |
| `MATRIX_HOME_ROOM` | Room ID for proactive message delivery (e.g. `!abc123:matrix.org`) |
| `MATRIX_ENCRYPTION` | Enable end-to-end encryption (`true`/`false`, default: `false`) |
| `MATRIX_E2EE_MODE` | Matrix E2EE behavior: `off`, `optional`, or `required`. Overrides `MATRIX_ENCRYPTION` when set. |
| `MATRIX_DEVICE_ID` | Stable Matrix device ID for E2EE persistence across restarts (e.g. `HERMES_BOT`). Without this, E2EE keys rotate every startup and historic-room decrypt breaks. |
| `MATRIX_REACTIONS` | Enable processing-lifecycle emoji reactions on inbound messages (default: `true`). Set to `false` to disable. |
| `MATRIX_REQUIRE_MENTION` | Require `@mention` in rooms (default: `true`). Set to `false` to respond to all messages. |
| `MATRIX_FREE_RESPONSE_ROOMS` | Comma-separated room IDs where bot responds without `@mention` |
| `MATRIX_IGNORE_USER_PATTERNS` | Comma-separated regular expressions for Matrix bridge/appservice ghost user IDs to ignore |
| `MATRIX_PROCESS_NOTICES` | Process inbound Matrix `m.notice` events (default: `false`) |
| `MATRIX_SESSION_SCOPE` | Matrix session scope for project rooms: `auto`, `room`, or `thread` (default: `auto`) |
| `MATRIX_TOOLS_ALLOW_CROSS_ROOM` | Allow Matrix tools to target explicit rooms other than the current room (default: `false`) |
| `MATRIX_TOOLS_ALLOW_CROSS_ROOM_DESTRUCTIVE` | Allow cross-room Matrix redaction/invite-like tools; requires `MATRIX_TOOLS_ALLOW_CROSS_ROOM=true` (default: `false`) |
| `MATRIX_TOOLS_ALLOW_REDACTION` | Allow Matrix message redaction tool execution (default: `false`) |
| `MATRIX_TOOLS_ALLOW_INVITES` | Allow Matrix invite tool execution (default: `false`) |
| `MATRIX_TOOLS_ALLOW_ROOM_CREATE` | Allow Matrix room creation tool execution (default: `false`) |
| `MATRIX_ALLOW_ROOM_MENTIONS` | Allow outbound `@room` mentions to notify all room members (default: `false`) |
| `MATRIX_AUTO_THREAD` | Auto-create threads for room messages (default: `true`) |
| `MATRIX_DM_AUTO_THREAD` | Auto-create threads for DM messages in Matrix (default: `false`) |
| `MATRIX_DM_MENTION_THREADS` | Create a thread when bot is `@mentioned` in a DM (default: `false`) |
| `MATRIX_APPROVAL_REQUIRE_SENDER` | Require approval/model-picker reactions to come from the original requester when known (default: `true`) |
| `MATRIX_APPROVAL_TIMEOUT_SECONDS` | Timeout for Matrix reaction approval/model-picker prompts (default: `300`) |
| `MATRIX_ALLOW_PUBLIC_ROOMS` | Allow Matrix room-creation tools to create public rooms (default: `false`) |
| `MATRIX_MAX_MEDIA_BYTES` | Maximum Matrix media upload/download size in bytes (default: `104857600`) |
| `MATRIX_RECOVERY_KEY` | Recovery key for cross-signing verification after device key rotation. Recommended for E2EE setups with cross-signing enabled. |
| `MATRIX_RECOVERY_KEY_OUTPUT_FILE` | Optional one-time path for a generated Matrix recovery key. Created with mode `0600` and never overwritten. |
| `HASS_TOKEN` | Home Assistant Long-Lived Access Token (enables HA platform + tools) |
| `HASS_URL` | Home Assistant URL (default: `http://homeassistant.local:8123`) |
| `WEBHOOK_ENABLED` | Enable the webhook platform adapter (`true`/`false`) |
| `WEBHOOK_PORT` | HTTP server port for receiving webhooks (default: `8644`) |
| `WEBHOOK_SECRET` | Global HMAC secret for webhook signature validation (used as fallback when routes don't specify their own) |
| `API_SERVER_ENABLED` | Enable the OpenAI-compatible API server (`true`/`false`). Runs alongside other platforms. |
| `API_SERVER_KEY` | Bearer token for API server authentication. Required whenever the API server is enabled. |
| `API_SERVER_CORS_ORIGINS` | Comma-separated browser origins allowed to call the API server directly (for example `http://localhost:3000,http://127.0.0.1:3000`). Default: disabled. |
| `API_SERVER_PORT` | Port for the API server (default: `8642`) |
| `API_SERVER_HOST` | Host/bind address for the API server (default: `127.0.0.1`). `API_SERVER_KEY` is still required on loopback; use a narrow `API_SERVER_CORS_ORIGINS` allowlist for browser access. |
| `API_SERVER_MODEL_NAME` | Model name advertised on `/v1/models`. Defaults to the profile name (or `hermes-agent` for the default profile). Useful for multi-user setups where frontends like Open WebUI need distinct model names per connection. |
| `GATEWAY_PROXY_URL` | URL of a remote Hermes API server to forward messages to ([proxy mode](/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos)). When set, the gateway handles platform I/O only — all agent work is delegated to the remote server. Also configurable via `gateway.proxy_url` in `config.yaml`. |
| `GATEWAY_PROXY_KEY` | Bearer token for authenticating with the remote API server in proxy mode. Must match `API_SERVER_KEY` on the remote host. |
| `MESSAGING_CWD` | Deprecated compatibility fallback for gateway working directory. Prefer `terminal.cwd` in `config.yaml`. |
| `GATEWAY_ALLOWED_USERS` | Comma-separated user IDs allowed across all platforms |
| `GATEWAY_ALLOW_ALL_USERS` | Allow all users without allowlists (`true`/`false`, default: `false`) |

### Web Dashboard & Hermes Desktop

Auth for the [web dashboard](/user-guide/features/web-dashboard) and for connecting [Hermes Desktop to a remote backend](/user-guide/features/web-dashboard#connecting-hermes-desktop-to-a-remote-backend). Per the secrets-only convention, credentials belong in `~/.hermes/.env`; the OAuth `client_id` is better set under `dashboard.oauth` in `config.yaml` (env wins when set).

Three dashboard-auth providers ship in the box. For a remote Hermes Desktop connection or any internet-facing dashboard, the recommended provider is **OAuth (Nous Portal)** — set `HERMES_DASHBOARD_OAUTH_CLIENT_ID` (provision it with `hermes dashboard register`). The bundled **username/password** provider (`HERMES_DASHBOARD_BASIC_AUTH_*`) is the quickest option for a backend on a trusted LAN or behind a VPN, but is not suitable for direct public-internet exposure. To authenticate against your own identity provider, use the **self-hosted OIDC** provider (`HERMES_DASHBOARD_OIDC_*`). Either way, a non-loopback bind (`hermes dashboard --host 0.0.0.0`) engages the auth gate. See [Web Dashboard → Authentication](/user-guide/features/web-dashboard#authentication-gated-mode) for the full picture.

| Variable | Description |
|----------|-------------|
| `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` | Username for the bundled username/password dashboard-auth provider (`plugins/dashboard_auth/basic`). Activates the provider when set together with a password. Overrides `dashboard.basic_auth.username`. |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` | Plaintext password for the basic provider (hashed in-memory at load). Wins over a config `password_hash` so you can rotate via env. Overrides `dashboard.basic_auth.password`. |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` | scrypt password hash for the basic provider (preferred — no plaintext at rest). Compute with `python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"`. Overrides `dashboard.basic_auth.password_hash`. |
| `HERMES_DASHBOARD_BASIC_AUTH_SECRET` | HMAC key (32+ bytes, base64/hex/raw) signing the basic provider's stateless session tokens. Set explicitly so sessions survive restarts / span multiple workers; blank → random per-process (you'll be logged out on every restart). Overrides `dashboard.basic_auth.secret`. |
| `HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS` | Access-token lifetime for the basic provider (default 12h). Overrides `dashboard.basic_auth.session_ttl_seconds`. |
| `HERMES_DASHBOARD_OAUTH_CLIENT_ID` | OAuth client id (`agent:{instance_id}`) for the gated/public dashboard, activating the Nous (`plugins/dashboard_auth/nous`) provider. Overrides `dashboard.oauth.client_id`. Provision it with `hermes dashboard register`. |
| `HERMES_DASHBOARD_PUBLIC_URL` | Complete public URL the dashboard is reached at, for OAuth callback construction behind reverse proxies. Overrides `dashboard.public_url`. |
| `HERMES_DASHBOARD_OIDC_ISSUER` | OIDC issuer URL for the bundled self-hosted OIDC provider (`plugins/dashboard_auth/self_hosted`). Required to activate it. Overrides `dashboard.oauth.self_hosted.issuer`. |
| `HERMES_DASHBOARD_OIDC_CLIENT_ID` | Public OIDC client id (authorization-code + PKCE) for the self-hosted OIDC provider. Required to activate it. Overrides `dashboard.oauth.self_hosted.client_id`. |
| `HERMES_DASHBOARD_OIDC_SCOPES` | Requested OIDC scopes for the self-hosted OIDC provider (default `openid profile email`). Overrides `dashboard.oauth.self_hosted.scopes`. |
| `HERMES_DESKTOP_REMOTE_URL` | (Desktop side) Base URL of the remote backend, e.g. `http://host:9119`. When set, overrides the in-app Gateway URL; you still sign in from the Gateway settings panel (OAuth redirect or username/password, whichever the backend advertises). |
| `HERMES_DESKTOP_HERMES` | Desktop backend command override. Used by packagers/Nix or troubleshooting to point Electron at a specific `hermes` executable after backend probing. |
| `HERMES_DESKTOP_HERMES_ROOT` | Desktop source-checkout override used by `hermes desktop --hermes-root`; checked before the packaged first-launch install or an existing `hermes` on `PATH`. |
| `HERMES_DESKTOP_IGNORE_EXISTING` | Set to `1` to make Desktop ignore an existing `hermes` on `PATH` during backend resolution. Equivalent to `hermes desktop --ignore-existing`. |
| `HERMES_DESKTOP_CWD` | Initial project directory for Desktop chat sessions. Set by `hermes desktop --cwd`. |

### Microsoft Graph (Teams Meetings)

App-only credentials for the Microsoft Graph REST client used by the upcoming Teams meeting summary pipeline. See [Register a Microsoft Graph application](/guides/microsoft-graph-app-registration) for the Azure portal walkthrough and the exact API permissions required.

| Variable | Description |
|----------|-------------|
| `MSGRAPH_TENANT_ID` | Azure AD tenant ID (directory GUID) for the Graph app registration. |
| `MSGRAPH_CLIENT_ID` | Application (client) ID of the Azure app registration. |
| `MSGRAPH_CLIENT_SECRET` | Client secret value for the app registration. Store in `~/.hermes/.env` with `chmod 600`; rotate periodically via the Azure portal. |
| `MSGRAPH_SCOPE` | OAuth2 scope for the client-credentials token request (default: `https://graph.microsoft.com/.default`). |
| `MSGRAPH_AUTHORITY_URL` | Microsoft identity platform authority (default: `https://login.microsoftonline.com`). Override only for national/sovereign clouds (e.g. `https://login.microsoftonline.us` for GCC High). |

### Microsoft Graph Webhook Listener

Inbound change-notification listener for Graph events (Teams meetings, calendar, chat, etc.). See [Microsoft Graph Webhook Listener](/user-guide/messaging/msgraph-webhook) for setup and security hardening.

| Variable | Description |
|----------|-------------|
| `MSGRAPH_WEBHOOK_ENABLED` | Enable the `msgraph_webhook` gateway platform (`true`/`1`/`yes`). |
| `MSGRAPH_WEBHOOK_PORT` | Port the listener binds to (default: `8646`). |
| `MSGRAPH_WEBHOOK_CLIENT_STATE` | Shared secret Graph echoes in every notification; compared with `hmac.compare_digest`. Generate with `openssl rand -hex 32`. |
| `MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES` | Comma-separated allowlist of Graph resource paths/patterns (e.g. `communications/onlineMeetings,chats/*/messages`). Trailing `*` is prefix-matching. Empty = accept all. |
| `MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS` | Comma-separated CIDR ranges allowed to POST to the listener (e.g. `52.96.0.0/14,52.104.0.0/14`). Empty = allow all (default). Restrict to Microsoft Graph's published egress ranges in production. |

### Teams Meeting Summary Delivery

Only used when the [`teams_pipeline` plugin](/user-guide/messaging/msgraph-webhook) is enabled. Settings are also configurable under `platforms.teams.extra` in `config.yaml` — env vars take priority when both are set. See [Microsoft Teams → Meeting Summary Delivery](/user-guide/messaging/teams#meeting-summary-delivery-teams-meeting-pipeline).

| Variable | Description |
|----------|-------------|
| `TEAMS_DELIVERY_MODE` | `graph` or `incoming_webhook`. |
| `TEAMS_INCOMING_WEBHOOK_URL` | Teams-generated webhook URL; required when `TEAMS_DELIVERY_MODE=incoming_webhook`. |
| `TEAMS_GRAPH_ACCESS_TOKEN` | Pre-acquired delegated access token for Graph delivery. Rarely needed — the writer falls back to the `MSGRAPH_*` app credentials when unset. |
| `TEAMS_TEAM_ID` | Target Team ID for channel delivery (`graph` mode). |
| `TEAMS_CHANNEL_ID` | Target channel ID (paired with `TEAMS_TEAM_ID`). |
| `TEAMS_CHAT_ID` | Target 1:1 or group chat ID (alternative to team+channel for `graph` mode). |

### LINE Messaging API

Used by the bundled LINE platform plugin (`plugins/platforms/line/`). See [Messaging Gateway → LINE](/user-guide/messaging/line) for full setup.

| Variable | Description |
|----------|-------------|
| `LINE_CHANNEL_ACCESS_TOKEN` | Long-lived channel access token from the LINE Developers Console (Messaging API tab). Required. |
| `LINE_CHANNEL_SECRET` | Channel secret (Basic settings tab); used for HMAC-SHA256 webhook signature verification. Required. |
| `LINE_HOST` | Webhook bind host (default: `0.0.0.0`). |
| `LINE_PORT` | Webhook bind port (default: `8646`). |
| `LINE_PUBLIC_URL` | Public HTTPS base URL (e.g. `https://my-tunnel.example.com`). Required for image / audio / video sends — LINE only accepts HTTPS-reachable URLs. |
| `LINE_ALLOWED_USERS` | Comma-separated user IDs allowed to DM the bot (`U`-prefixed). |
| `LINE_ALLOWED_GROUPS` | Comma-separated group IDs the bot will respond in (`C`-prefixed). |
| `LINE_ALLOWED_ROOMS` | Comma-separated room IDs the bot will respond in (`R`-prefixed). |
| `LINE_ALLOW_ALL_USERS` | Dev-only escape hatch — accepts any source. Default: `false`. |
| `LINE_HOME_CHANNEL` | Default delivery target for cron jobs with `deliver: line`. |
| `LINE_SLOW_RESPONSE_THRESHOLD` | Seconds before the slow-LLM Template Buttons postback fires (default: `45`). Set `0` to disable and always Push-fallback. |
| `LINE_PENDING_TEXT` | Bubble text shown alongside the postback button. |
| `LINE_BUTTON_LABEL` | Postback button label (default: `Get answer`). |
| `LINE_DELIVERED_TEXT` | Reply when an already-delivered postback is tapped again (default: `Already replied ✅`). |
| `LINE_INTERRUPTED_TEXT` | Reply when a `/stop`-orphaned postback button is tapped (default: `Run was interrupted before completion.`). |

### ntfy (push notifications)

[ntfy](https://ntfy.sh/) is a lightweight HTTP-based push notification service. Subscribe to a topic from the [ntfy mobile app](https://ntfy.sh/docs/subscribe/phone/), publish to that topic to talk to the agent.

| Variable | Description |
|----------|-------------|
| `NTFY_TOPIC` | Topic to subscribe to (incoming messages). Required. |
| `NTFY_SERVER_URL` | Server URL (default: `https://ntfy.sh`). Point at a self-hosted ntfy for privacy. |
| `NTFY_TOKEN` | Optional auth token. Bearer token (e.g. `tk_xyz`) or `user:pass` for Basic auth. |
| `NTFY_PUBLISH_TOPIC` | Topic for outgoing replies (defaults to `NTFY_TOPIC`). |
| `NTFY_MARKDOWN` | Set `true` to send replies with `X-Markdown: true` header. Default: `false`. |
| `NTFY_ALLOWED_USERS` | Allowlist (treated as user IDs; on ntfy these are topic names). Typically set to the same value as `NTFY_TOPIC`. |
| `NTFY_ALLOW_ALL_USERS` | Dev-only escape hatch — only safe on access-controlled private topics. Default: `false`. |
| `NTFY_HOME_CHANNEL` | Default delivery target for cron jobs with `deliver: ntfy`. |
| `NTFY_HOME_CHANNEL_NAME` | Human label for the home channel (defaults to the topic name). |

See [the ntfy messaging guide](/user-guide/messaging/ntfy) — particularly the **identity model** section — before deploying with untrusted topics.

### IRC

Connect Hermes to an IRC server. No external dependencies. See [the IRC messaging guide](/user-guide/messaging/irc).

| Variable | Description |
|----------|-------------|
| `IRC_SERVER` | IRC server hostname (e.g. `irc.libera.chat`). Required. |
| `IRC_CHANNEL` | Channel(s) to join (e.g. `#hermes`); comma-separate for multiple. Required. |
| `IRC_NICKNAME` | Bot nickname (default: `hermes-bot`). Required. |
| `IRC_PORT` | Server port (default: `6697` with TLS, `6667` without). |
| `IRC_USE_TLS` | Use TLS (`true`/`false`; default `true` on port 6697). |
| `IRC_SERVER_PASSWORD` | Server password for the `PASS` command (optional). |
| `IRC_NICKSERV_PASSWORD` | NickServ password for automatic IDENTIFY on connect (optional). |
| `IRC_ALLOWED_USERS` | Comma-separated nicks allowed to talk to the bot. |
| `IRC_ALLOW_ALL_USERS` | Allow anyone in the channel to talk to the bot (dev only). |
| `IRC_HOME_CHANNEL` | Channel for cron / notification delivery (defaults to `IRC_CHANNEL`). |

### SimpleX

Connect Hermes to a [SimpleX Chat](https://simplex.chat/) network via a local `simplex-chat` daemon. See [the SimpleX messaging guide](/user-guide/messaging/simplex).

| Variable | Description |
|----------|-------------|
| `SIMPLEX_WS_URL` | WebSocket URL of the simplex-chat daemon (e.g. `ws://127.0.0.1:5225`). |
| `SIMPLEX_ALLOWED_USERS` | Comma-separated SimpleX contact IDs allowed to talk to the bot. |
| `SIMPLEX_ALLOW_ALL_USERS` | Allow any contact to talk to the bot (dev only — disables allowlist). |
| `SIMPLEX_AUTO_ACCEPT` | Auto-accept incoming contact requests (default: `true`). |
| `SIMPLEX_GROUP_ALLOWED` | Comma-separated SimpleX group IDs the bot should participate in, or `*` to allow any group. Omit to ignore group messages entirely (safer default — a bot in a group otherwise processes every member's traffic). |
| `SIMPLEX_HOME_CHANNEL` | Default contact/group ID for cron / notification delivery. |
| `SIMPLEX_HOME_CHANNEL_NAME` | Human label for the home channel (defaults to the ID). |

### Photon

Connect Hermes to [Photon](https://photon.codes/) / Spectrum (iMessage and other Spectrum platforms) via the Node sidecar. See [the Photon messaging guide](/user-guide/messaging/photon).

| Variable | Description |
|----------|-------------|
| `PHOTON_PROJECT_ID` | Spectrum project id (the project's `spectrumProjectId`; set by `hermes photon setup`). |
| `PHOTON_PROJECT_SECRET` | Project secret paired with the Spectrum project id (set by `hermes photon setup`). |
| `PHOTON_ALLOWED_USERS` | Comma-separated E.164 phone numbers allowed to talk to the bot. |
| `PHOTON_ALLOW_ALL_USERS` | Allow any sender to trigger the bot (dev only — disables allowlist). |
| `PHOTON_REQUIRE_MENTION` | Ignore group-chat messages unless they match a mention wake word (`true`/`false`, default `false`). |
| `PHOTON_MENTION_PATTERNS` | Mention wake-word regexes for group chats (JSON list or comma/newline-separated; defaults to Hermes wake words). |
| `PHOTON_HOME_CHANNEL` | Default Photon target for cron / notification delivery: Spectrum space id, DM GUID, or bare E.164 phone number. |
| `PHOTON_HOME_CHANNEL_NAME` | Human label for the home channel. |
| `PHOTON_MARKDOWN` | Send agent replies as markdown — iMessage renders it natively, other Spectrum platforms degrade to plain text (`true`/`false`, default `true`). |
| `PHOTON_REACTIONS` | Tapback 👀/👍/👎 on messages as processing status and route tapbacks on bot messages to the agent (`true`/`false`, default `false`). |
| `PHOTON_TELEMETRY` | Enable Spectrum SDK telemetry in the sidecar (`true`/`false`, default `false`; toggle with `hermes photon telemetry on|off`). |
| `PHOTON_SIDECAR_PORT` | Loopback port for the Node sidecar control + inbound channel (default `8789`). |
| `PHOTON_SIDECAR_AUTOSTART` | Spawn the Node sidecar on connect (`true`/`false`, default `true`). |
| `PHOTON_NODE_BIN` | Path to the node binary (default: `shutil.which('node')`). |
| `PHOTON_DASHBOARD_HOST` | Photon Dashboard API host (default `https://app.photon.codes`). |
| `PHOTON_SPECTRUM_HOST` | Photon Spectrum API host (default `https://spectrum.photon.codes`). |

### Microsoft Teams (adapter)

The Microsoft Teams platform adapter (Bot Framework / Azure AD), distinct from the [Microsoft Graph (Teams Meetings)](#microsoft-graph-teams-meetings) integration above. See [the Teams messaging guide](/user-guide/messaging/teams).

| Variable | Description |
|----------|-------------|
| `TEAMS_CLIENT_ID` | Azure AD application (Bot Framework) client ID. |
| `TEAMS_CLIENT_SECRET` | Azure AD application client secret. |
| `TEAMS_TENANT_ID` | Azure AD tenant ID hosting the bot application. |
| `TEAMS_PORT` | Webhook listen port (Bot Framework default: `3978`). |
| `TEAMS_ALLOWED_USERS` | Comma-separated Teams user IDs / UPNs allowed to talk to the bot. |
| `TEAMS_ALLOW_ALL_USERS` | Allow any Teams user to trigger the bot (dev only). |
| `TEAMS_HOME_CHANNEL` | Default chat/channel ID for cron / notification delivery. |
| `TEAMS_HOME_CHANNEL_NAME` | Display name for the Teams home channel. |

### Raft

| Variable | Description |
|----------|-------------|
| `RAFT_PROFILE` | Raft agent profile slug — auto-enables the adapter when set. |

### Advanced Messaging Tuning

Advanced per-platform knobs for throttling the outbound message batcher. Most users never need to touch these; defaults are set to respect each platform's rate limits without feeling sluggish.

| Variable | Description |
|----------|-------------|
| `HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS` | Grace window before flushing a queued Telegram text chunk (default: `0.6`). |
| `HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS` | Delay between split chunks when a single Telegram message exceeds the length limit (default: `2.0`). |
| `HERMES_SIMPLEX_TEXT_BATCH_DELAY` | Quiet-period seconds (default: `0.8`) used to concatenate rapid-fire inbound text messages into a single MessageEvent — same pattern as Telegram's text batching. |
| `HERMES_TELEGRAM_MEDIA_BATCH_DELAY_SECONDS` | Grace window before flushing queued Telegram media (default: `0.6`). |
| `HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS` | Delay before sending a follow-up after the agent finishes, to avoid racing the last stream chunk. |
| `HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT` / `_READ_TIMEOUT` / `_WRITE_TIMEOUT` / `_POOL_TIMEOUT` | Override the underlying `python-telegram-bot` HTTP timeouts (seconds). |
| `HERMES_TELEGRAM_INIT_TIMEOUT` | Per-attempt cap (seconds) on the Telegram `initialize()` connect chain during gateway startup, so an unreachable fallback-IP chain can't block startup indefinitely (default: `30`). |
| `HERMES_TELEGRAM_HTTP_POOL_SIZE` | Max concurrent HTTP connections to the Telegram API. |
| `HERMES_TELEGRAM_DISABLE_FALLBACK_IPS` | Disable the hard-coded Cloudflare fallback IPs used when DNS fails (`true`/`false`). |
| `HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS` | Grace window before flushing a queued Discord text chunk (default: `0.6`). |
| `HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS` | Delay between split chunks when a Discord message exceeds the length limit (default: `2.0`). |
| `HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS` | Internal bridge for `discord.liveness_interval_seconds` (config.yaml). Interval for the Discord REST liveness probe that detects zombie clients behind dead proxies/NATs (default: `60`; set to `0` to disable). Prefer setting `discord.liveness_interval_seconds` in `config.yaml`. |
| `HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD` | Internal bridge for `discord.liveness_failure_threshold` (config.yaml). Consecutive probe failures before forcing a Discord reconnect (default: `3`). Prefer setting `discord.liveness_failure_threshold` in `config.yaml`. |
| `HERMES_MATRIX_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | Matrix equivalents of the Telegram batch knobs. |
| `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` / `_MAX_CHARS` / `_MAX_MESSAGES` | Feishu batcher tuning — delay, split delay, max chars per message, max messages per batch. |
| `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | Feishu media flush delay. |
| `HERMES_FEISHU_DEDUP_CACHE_SIZE` | Size of the Feishu webhook dedup cache (default: `1024`). |
| `HERMES_WECOM_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | WeCom batcher tuning. |
| `HERMES_VISION_DOWNLOAD_TIMEOUT` | Timeout in seconds for downloading an image before handing it to vision models (default: `30`). |
| `HERMES_VISION_MAX_CONCURRENCY` | Max concurrent image **encode/resize** bursts across the whole process (override for `auxiliary.vision.max_concurrency`; default: host CPU core count, no ceiling). Bounds only the CPU-bound encode step so a video-frame fan-out can't saturate every core and starve the event loop — the LLM calls stay fully concurrent. Values `< 1` are ignored. |
| `HERMES_RESTART_DRAIN_TIMEOUT` | Gateway: seconds to wait for active runs to drain on `/restart` before forcing the restart (default: `900`). |
| `HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT` | Per-platform connect timeout during gateway startup and reconnect (seconds; `0`/negative waits indefinitely). Applies to the connect attempt *and* the Discord adapter's ready-wait, so accounts with many slash commands to sync don't get killed mid-startup. Bridged from `gateway.platform_connect_timeout` in `config.yaml` (default `30`); this env var is the manual override and wins if set explicitly. |
| `HERMES_GATEWAY_BUSY_INPUT_MODE` | Default gateway busy-input behavior: `queue`, `steer`, or `interrupt`. Can be overridden per chat with `/busy`. |
| `HERMES_GATEWAY_BUSY_ACK_ENABLED` | Whether the gateway sends an acknowledgment message (⚡/⏳/⏩) when a user sends input while the agent is busy (default: `true`). Set to `false` to suppress these messages entirely — the input is still queued/steered/interrupts as normal, only the chat reply is silenced. Bridged from `display.busy_ack_enabled` in `config.yaml`. |
| `HERMES_GATEWAY_NO_SUPERVISE` | Inside the s6-overlay Docker image, opt out of auto-supervision when running `hermes gateway run` and use pre-s6 foreground semantics (no auto-restart, gateway is the container's main process). Truthy values: `1`, `true`, `yes`. Equivalent to the `--no-supervise` CLI flag. No-op outside the s6 image. |
| `HERMES_GATEWAY_BOOTSTRAP_STATE` | Inside the s6-overlay Docker image, declare the gateway's **initial** supervised state on a fresh volume. On a blank volume there is no persisted `gateway_state.json`, so the boot reconciler registers the `gateway-default` slot but leaves it **down** (it only auto-starts when the last recorded state was `running`). Set this to `running` and the first-boot setup hook seeds `gateway_state.json` *before* the reconciler runs, so the gateway comes up on the very first boot. Only the literal value `running` is honoured. First-boot-only: an existing `gateway_state.json` is never overwritten, so a deliberately-stopped gateway stays stopped across restarts. No-op outside the s6 image. |
| `GATEWAY_RELAY_URL` | Experimental relay connector WebSocket base URL. When set, the gateway registers the generic `relay` adapter and dials the connector outbound. Mirrors `gateway.relay_url` in `config.yaml`. |
| `GATEWAY_RELAY_ID` | Relay gateway identifier assigned by `hermes gateway enroll` or managed self-provisioning. Mirrors `gateway.relay_id`. |
| `GATEWAY_RELAY_SECRET` | Per-gateway relay secret used to authenticate the WebSocket. If this is already configured, managed self-provisioning is skipped. Mirrors `gateway.relay_secret`. |
| `GATEWAY_RELAY_DELIVERY_KEY` | Connector-issued delivery key retained for relay/passthrough authentication compatibility. Current relay inbound messages arrive on the outbound WebSocket rather than a gateway-side HTTP receiver. |
| `GATEWAY_RELAY_ENROLL_TOKEN` | Enrollment token consumed by `hermes gateway enroll` when `--token` is not passed explicitly. |
| `GATEWAY_RELAY_PLATFORM` | Optional platform name advertised in the relay capability descriptor. |
| `GATEWAY_RELAY_BOT_ID` | Optional bot identifier advertised in the relay capability descriptor. |
| `GATEWAY_RELAY_ENDPOINT` | Optional gateway endpoint advertised for connector modes that need a callback/passthrough URL; not required for the default WS-only inbound relay path. Mirrors `gateway.relay_endpoint`. |
| `GATEWAY_RELAY_ROUTE_KEYS` | Comma-separated relay route keys advertised to the connector. Mirrors `gateway.relay_route_keys`. |
| `HERMES_FILE_MUTATION_VERIFIER` | Enable the per-turn file-mutation verifier footer (default: `true`). When enabled, Hermes appends an advisory listing any `write_file` / `patch` calls that failed during the turn and were not superseded by a successful write. Set to `0`, `false`, `no`, or `off` to suppress. Mirrors `display.file_mutation_verifier` in `config.yaml`; the env var wins when set. |
| `HERMES_CRON_TIMEOUT` | Inactivity timeout for cron job agent runs in seconds (default: `600`). The agent can run indefinitely while actively calling tools or receiving stream tokens — this only triggers when idle. Set to `0` for unlimited. |
| `HERMES_CRON_SCRIPT_TIMEOUT` | Timeout for pre-run scripts attached to cron jobs in seconds (default: `3600`). Bounds the script only — skill/agent jobs use the separate `HERMES_CRON_TIMEOUT` inactivity budget. Also configurable via `cron.script_timeout_seconds` in `config.yaml`. |
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
| `HERMES_SAFE_MODE` | Troubleshooting mode: disable ALL customizations — skips plugin discovery, MCP server loading, and shell-hook registration. Set automatically by `--safe-mode` (which also sets the two flags above). |
| `HERMES_MD_NAMES` | Comma-separated list of rules-file names to auto-inject (default: `AGENTS.md,CLAUDE.md,.cursorrules,SOUL.md`). |
| `HERMES_TOOL_PROGRESS` | Deprecated compatibility variable for tool progress display. Prefer `display.tool_progress` in `config.yaml`. |
| `HERMES_TOOL_PROGRESS_MODE` | Deprecated compatibility variable for tool progress mode. Prefer `display.tool_progress` in `config.yaml`. |
| `HERMES_HUMAN_DELAY_MODE` | Response pacing: `off`/`natural`/`custom` |
| `HERMES_HUMAN_DELAY_MIN_MS` | Custom delay range minimum (ms) |
| `HERMES_HUMAN_DELAY_MAX_MS` | Custom delay range maximum (ms) |
| `HERMES_QUIET` | Suppress non-essential output (`true`/`false`) |
| `CODEX_HOME` | When [Codex app-server runtime](../user-guide/features/codex-app-server-runtime) is enabled, override the directory Codex CLI reads its config + auth from (default: `~/.codex`). Hermes' migration writes the managed block to `<CODEX_HOME>/config.toml`. |
| `HERMES_KANBAN_TASK` | Set by the kanban dispatcher when spawning a worker (task UUID). Workers and the spawned `hermes-tools` MCP subprocess inherit it so kanban tools gate correctly. Don't set manually. |
| `HERMES_API_TIMEOUT` | LLM API call timeout in seconds (default: `1800`) |
| `HERMES_API_CALL_STALE_TIMEOUT` | Non-streaming stale-call timeout in seconds (default: `90`). Auto-disabled for local providers when left unset, and may scale upward for very large contexts. Also configurable via `providers.<id>.stale_timeout_seconds` or `providers.<id>.models.<model>.stale_timeout_seconds` in `config.yaml`. |
| `HERMES_STREAM_READ_TIMEOUT` | Streaming socket read timeout in seconds (default: `120`). Auto-increased to `HERMES_API_TIMEOUT` for local providers. Increase if local LLMs time out during long code generation. |
| `HERMES_STREAM_STALE_TIMEOUT` | Stale stream detection timeout in seconds (default: `180`). Auto-disabled for local providers. Triggers connection kill if no chunks arrive within this window. |
| `HERMES_STREAM_RETRIES` | Number of mid-stream reconnect attempts on transient network errors (default: `3`). |
| `HERMES_STREAM_STALE_GIVEUP` | Cross-turn circuit breaker: after this many consecutive stale-stream kills with no completed stream, abort each turn immediately with an actionable error instead of re-waiting out the stale timeout (default: `5`, `0` disables). Resets on a completed stream, `/model` switch, or fallback activation. |
| `HERMES_AGENT_TIMEOUT` | Gateway inactivity timeout for a running agent in seconds (default: `1800`, 30 minutes). Resets on every tool call and streamed token. Set to `0` to disable. |
| `HERMES_AGENT_TIMEOUT_WARNING` | Gateway: send a warning message after this many seconds of inactivity (default: 75% of `HERMES_AGENT_TIMEOUT`). |
| `HERMES_AGENT_NOTIFY_INTERVAL` | Gateway: interval in seconds between progress notifications on long-running agent turns. |
| `HERMES_CHECKPOINT_TIMEOUT` | Timeout for filesystem checkpoint creation in seconds (default: `30`). |
| `HERMES_EXEC_ASK` | Enable execution approval prompts in gateway mode (`true`/`false`) |
| `HERMES_ENABLE_PROJECT_PLUGINS` | Enable auto-discovery of repo-local plugins from `./.hermes/plugins/` for both the agent loader and the dashboard web server. Accepts the standard truthy set: `1` / `true` / `yes` / `on` (case-insensitive). Everything else — including `0`, `false`, `no`, `off`, and the empty string — is treated as **disabled** (default). Note: as of GHSA-5qr3-c538-wm9j (#29156) the dashboard web server refuses to auto-import a project plugin's Python `api` file even when this var is enabled — project plugins may extend the UI via static JS/CSS but their backend routes are only loaded when moved under `~/.hermes/plugins/`. |
| `HERMES_PLUGINS_DEBUG` | `1`/`true` to surface verbose plugin-discovery logs on stderr — directories scanned, manifests parsed, skip reasons, and full tracebacks on parse or `register()` failure. Aimed at plugin authors. |
| `HERMES_BACKGROUND_NOTIFICATIONS` | Background process notification mode in gateway: `all` (default), `result`, `error`, `off` |
| `HERMES_EPHEMERAL_SYSTEM_PROMPT` | Ephemeral system prompt injected at API-call time (never persisted to sessions) |
| `HERMES_PREFILL_MESSAGES_FILE` | Path to a JSON file of ephemeral prefill messages injected at API-call time. |
| `HERMES_ALLOW_PRIVATE_URLS` | `true`/`false` — allow tools to fetch localhost/private-network URLs. Off by default in gateway mode. |
| `HERMES_REDACT_SECRETS` | `true`/`false` — control secret redaction in tool output, logs, and chat responses (default: `true`). |
| `HERMES_WRITE_SAFE_ROOT` | Optional directory prefix that restricts `write_file`/`patch` writes; paths outside require approval. Supports multiple directories separated by `os.pathsep` (`:` on Unix, `;` on Windows). |
| `HERMES_DISABLE_LAZY_INSTALLS` | Internal bridge var set automatically in the official Docker image to prevent runtime dependency installs into the immutable `/opt/hermes` tree. The user-facing equivalent is `security.allow_lazy_installs: false` in `config.yaml`; do not set this in `.env`. |
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
| `HERMES_INFERENCE_MODEL` | Force the model for `hermes -z` / `hermes chat` without mutating `config.yaml`. Pairs with the `--provider` flag. Useful for scripted callers (sweeper, CI, batch runners) that need to override the default model per run. |

## Session Settings

| Variable | Description |
|----------|-------------|
| `SESSION_IDLE_MINUTES` | Reset sessions after N minutes of inactivity (default: 1440) |
| `SESSION_RESET_HOUR` | Daily reset hour in 24h format (default: 4 = 4am) |
| `HERMES_SESSION_ID` | **Exported automatically into every tool subprocess** Hermes spawns (`terminal`, `execute_code`, persistent shell, Docker/Singularity backends, delegated subagent runs). Set by the agent to the current session ID; user scripts called from tools can read it to correlate their output, telemetry, or side effects with the originating Hermes session. **You should not set this manually** — overriding it from a parent shell only takes effect outside an agent run, and is overwritten the moment the agent starts a session. |

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

## Fallback Providers (config.yaml only)

The primary model fallback chain is configured exclusively through `config.yaml` — there are no environment variables for it. Add a top-level `fallback_providers` list with `provider` and `model` keys to enable automatic failover when your main model encounters errors. Auxiliary tasks whose provider is `auto` also consult this chain before Hermes' built-in auxiliary discovery chain.

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

The older top-level `fallback_model` single-provider shape is still read for backward compatibility, but new configuration should use `fallback_providers`. For task-specific auxiliary policy, use `auxiliary.<task>.fallback_chain` in `config.yaml`; there is no environment variable equivalent.

See [Fallback Providers](/user-guide/features/fallback-providers) for full details.

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
