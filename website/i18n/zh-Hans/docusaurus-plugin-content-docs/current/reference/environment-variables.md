---
sidebar_position: 2
title: "环境变量"
description: "Hermes Agent 使用的所有环境变量的完整参考"
---

# 环境变量参考

所有变量放在 `~/.hermes/.env` 中。你也可以用 `hermes config set VAR value` 来设置它们。

## LLM Provider

| 变量 | 描述 |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API 密钥（推荐，灵活度高） |
| `OPENROUTER_BASE_URL` | 覆盖 OpenRouter 兼容的 base URL |
| `HERMES_OPENROUTER_CACHE` | 启用 OpenRouter 响应缓存（`1`/`true`/`yes`/`on`）。覆盖 config.yaml 中的 `openrouter.response_cache`。请参见 [响应缓存](https://openrouter.ai/docs/guides/features/response-caching)。 |
| `HERMES_OPENROUTER_CACHE_TTL` | 缓存 TTL 秒数（1-86400）。覆盖 config.yaml 中的 `openrouter.response_cache_ttl`。 |
| `NOUS_BASE_URL` | 覆盖 Nous Portal base URL（很少需要；仅开发/测试） |
| `NOUS_INFERENCE_BASE_URL` | 直接覆盖 Nous 推理端点 |
| `AI_GATEWAY_API_KEY` | Vercel AI Gateway API 密钥（[ai-gateway.vercel.sh](https://ai-gateway.vercel.sh)） |
| `AI_GATEWAY_BASE_URL` | 覆盖 AI Gateway base URL（默认：`https://ai-gateway.vercel.sh/v1`） |
| `OPENAI_API_KEY` | 自定义 OpenAI 兼容端点的 API 密钥（与 `OPENAI_BASE_URL` 一起使用） |
| `OPENAI_BASE_URL` | 自定义端点的 base URL（VLLM、SGLang 等） |
| `COPILOT_GITHUB_TOKEN` | Copilot API 的 GitHub 令牌 — 第一优先级（OAuth `gho_*` 或细粒度 PAT `github_pat_*`；不支持经典 PAT `ghp_*`） |
| `GH_TOKEN` | GitHub 令牌 — Copilot 的第二优先级（也被 `gh` CLI 使用） |
| `GITHUB_TOKEN` | GitHub 令牌 — Copilot 的第三优先级 |
| `HERMES_COPILOT_ACP_COMMAND` | 覆盖 Copilot ACP CLI 二进制路径（默认：`copilot`） |
| `COPILOT_CLI_PATH` | `HERMES_COPILOT_ACP_COMMAND` 的别名 |
| `HERMES_COPILOT_ACP_ARGS` | 覆盖 Copilot ACP 参数（默认：`--acp --stdio`） |
| `COPILOT_ACP_BASE_URL` | 覆盖 Copilot ACP base URL |
| `GLM_API_KEY` | z.ai / ZhipuAI GLM API 密钥（[z.ai](https://z.ai)） |
| `ZAI_API_KEY` | `GLM_API_KEY` 的别名 |
| `Z_AI_API_KEY` | `GLM_API_KEY` 的别名 |
| `GLM_BASE_URL` | 覆盖 z.ai base URL（默认：`https://api.z.ai/api/paas/v4`） |
| `KIMI_API_KEY` | Kimi / Moonshot AI API 密钥（[moonshot.ai](https://platform.moonshot.ai)） |
| `KIMI_BASE_URL` | 覆盖 Kimi base URL（默认：`https://api.moonshot.ai/v1`） |
| `KIMI_CN_API_KEY` | Kimi / Moonshot 中国 API 密钥（[moonshot.cn](https://platform.moonshot.cn)） |
| `ARCEEAI_API_KEY` | Arcee AI API 密钥（[chat.arcee.ai](https://chat.arcee.ai/)） |
| `ARCEE_BASE_URL` | 覆盖 Arcee base URL（默认：`https://api.arcee.ai/api/v1`） |
| `GMI_API_KEY` | GMI Cloud API 密钥（[gmicloud.ai](https://www.gmicloud.ai/)） |
| `GMI_BASE_URL` | 覆盖 GMI Cloud base URL（默认：`https://api.gmi-serving.com/v1`） |
| `MINIMAX_API_KEY` | MiniMax API 密钥 — 全球端点（[minimax.io](https://www.minimax.io)）。**不被 `minimax-oauth` 使用**（OAuth 路径使用浏览器登录）。 |
| `MINIMAX_BASE_URL` | 覆盖 MiniMax base URL（默认：`https://api.minimax.io/anthropic` — Hermes 使用 MiniMax 的 Anthropic Messages 兼容端点）。**不被 `minimax-oauth` 使用**。 |
| `MINIMAX_CN_API_KEY` | MiniMax API 密钥 — 中国端点（[minimaxi.com](https://www.minimaxi.com)）。**不被 `minimax-oauth` 使用**。 |
| `MINIMAX_CN_BASE_URL` | 覆盖 MiniMax 中国 base URL（默认：`https://api.minimaxi.com/anthropic`）。**不被 `minimax-oauth` 使用**。 |
| `KILOCODE_API_KEY` | Kilo Code API 密钥（[kilo.ai](https://kilo.ai)） |
| `KILOCODE_BASE_URL` | 覆盖 Kilo Code base URL（默认：`https://api.kilo.ai/api/gateway`） |
| `XIAOMI_API_KEY` | Xiaomi MiMo API 密钥（[platform.xiaomimimo.com](https://platform.xiaomimimo.com)） |
| `XIAOMI_BASE_URL` | 覆盖 Xiaomi MiMo base URL（默认：`https://api.xiaomimimo.com/v1`） |
| `TOKENHUB_API_KEY` | Tencent TokenHub API 密钥（[tokenhub.tencentmaas.com](https://tokenhub.tencentmaas.com)） |
| `TOKENHUB_BASE_URL` | 覆盖 Tencent TokenHub base URL（默认：`https://tokenhub.tencentmaas.com/v1`） |
| `AZURE_FOUNDRY_API_KEY` | Azure AI Foundry / Azure OpenAI API 密钥（[ai.azure.com](https://ai.azure.com/)） |
| `AZURE_FOUNDRY_BASE_URL` | Azure AI Foundry 端点 URL（如 `https://<resource>.openai.azure.com/openai/v1` 用于 OpenAI 风格，或 `https://<resource>.services.ai.azure.com/anthropic` 用于 Anthropic 风格） |
| `AZURE_ANTHROPIC_KEY` | 当 `provider: anthropic` + `base_url` 指向 Azure Foundry Claude 部署时的 Azure Anthropic API 密钥（当同时配置了 Anthropic 和 Azure Anthropic 时的 `ANTHROPIC_API_KEY` 的替代方案） |
| `HF_TOKEN` | Hugging Face Inference Providers 令牌（[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)） |
| `HF_BASE_URL` | 覆盖 Hugging Face base URL（默认：`https://router.huggingface.co/v1`） |
| `GOOGLE_API_KEY` | Google AI Studio API 密钥（[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)） |
| `GEMINI_API_KEY` | `GOOGLE_API_KEY` 的别名 |
| `GEMINI_BASE_URL` | 覆盖 Google AI Studio base URL |
| `HERMES_GEMINI_CLIENT_ID` | `google-gemini-cli` PKCE 登录的 OAuth 客户端 ID（可选；默认为 Google 公共 gemini-cli 客户端） |
| `HERMES_GEMINI_CLIENT_SECRET` | `google-gemini-cli` 的 OAuth 客户端密钥（可选） |
| `HERMES_GEMINI_PROJECT_ID` | 付费 Gemini 层的 GCP 项目 ID（免费层自动配置） |
| `ANTHROPIC_API_KEY` | Anthropic Console API 密钥（[console.anthropic.com](https://console.anthropic.com/)） |
| `ANTHROPIC_TOKEN` | 手动或旧版 Anthropic OAuth/setup-token 覆盖 |
| `DASHSCOPE_API_KEY` | 用于 Qwen 模型的 Alibaba Cloud DashScope API 密钥（[modelstudio.console.alibabacloud.com](https://modelstudio.console.alibabacloud.com/)） |
| `DASHSCOPE_BASE_URL` | 自定义 DashScope base URL（默认：`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`；中国大陆地区使用 `https://dashscope.aliyuncs.com/compatible-mode/v1`） |
| `DEEPSEEK_API_KEY` | 直接访问 DeepSeek 的 DeepSeek API 密钥（[platform.deepseek.com](https://platform.deepseek.com/api_keys)） |
| `DEEPSEEK_BASE_URL` | 自定义 DeepSeek API base URL |
| `NVIDIA_API_KEY` | NVIDIA NIM API 密钥 — Nemotron 和开放模型（[build.nvidia.com](https://build.nvidia.com)） |
| `NVIDIA_BASE_URL` | 覆盖 NVIDIA base URL（默认：`https://integrate.api.nvidia.com/v1`；本地 NIM 端点设置为 `http://localhost:8000/v1`） |
| `GMI_API_KEY` | GMI Cloud API 密钥 — 开放和推理模型（[inference.gmi.ai](https://inference.gmi.ai)） |
| `GMI_BASE_URL` | 覆盖 GMI Cloud base URL（默认：`https://api.gmi.ai/v1`） |
| `STEPFUN_API_KEY` | StepFun API 密钥 — Step 系列模型（[platform.stepfun.com](https://platform.stepfun.com)） |
| `STEPFUN_BASE_URL` | 覆盖 StepFun base URL（默认：`https://api.stepfun.com/v1`） |
| `OLLAMA_API_KEY` | Ollama Cloud API 密钥 — 托管 Ollama 目录，无需本地 GPU（[ollama.com/settings/keys](https://ollama.com/settings/keys)） |
| `OLLAMA_BASE_URL` | 覆盖 Ollama Cloud base URL（默认：`https://ollama.com/v1`） |
| `XAI_API_KEY` | xAI（Grok）用于聊天 + TTS 的 API 密钥（[console.x.ai](https://console.x.ai/)） |
| `XAI_BASE_URL` | 覆盖 xAI base URL（默认：`https://api.x.ai/v1`） |
| `MISTRAL_API_KEY` | 用于 Voxtral TTS 和 Voxtral STT 的 Mistral API 密钥（[console.mistral.ai](https://console.mistral.ai)） |
| `AWS_REGION` | 用于 Bedrock 推理的 AWS 区域（如 `us-east-1`、`eu-central-1`）。由 boto3 读取。 |
| `AWS_PROFILE` | 用于 Bedrock 认证的 AWS 命名配置文件（读取 `~/.aws/credentials`）。留空使用默认 boto3 凭证链。 |
| `BEDROCK_BASE_URL` | 覆盖 Bedrock 运行时 base URL（默认：`https://bedrock-runtime.us-east-1.amazonaws.com`；通常留空而使用 `AWS_REGION`） |
| `HERMES_QWEN_BASE_URL` | Qwen Portal base URL 覆盖（默认：`https://portal.qwen.ai/v1`） |
| `OPENCODE_ZEN_API_KEY` | OpenCode Zen API 密钥 — 精选模型的按量付费访问（[opencode.ai](https://opencode.ai/auth)） |
| `OPENCODE_ZEN_BASE_URL` | 覆盖 OpenCode Zen base URL |
| `OPENCODE_GO_API_KEY` | OpenCode Go API 密钥 — 每月 $10 订阅，开放模型（[opencode.ai](https://opencode.ai/auth)） |
| `OPENCODE_GO_BASE_URL` | 覆盖 OpenCode Go base URL |
| `CLAUDE_CODE_OAUTH_TOKEN` | 如果你手动导出一个，则显式 Claude Code 令牌覆盖 |
| `HERMES_MODEL` | 在进程级别覆盖模型名称（被 cron 调度器使用；正常用法优先使用 `config.yaml`） |
| `VOICE_TOOLS_OPENAI_KEY` | OpenAI 语音转文本和文本转语音 provider 的首选 OpenAI 密钥 |
| `HERMES_LOCAL_STT_COMMAND` | 可选本地语音转文本命令模板。支持 `{input_path}`、`{output_dir}`、`{language}` 和 `{model}` 占位符 |
| `HERMES_LOCAL_STT_LANGUAGE` | 传递给 `HERMES_LOCAL_STT_COMMAND` 或自动检测本地 `whisper` CLI 后备的默认语言（默认：`en`） |
| `HERMES_HOME` | 覆盖 Hermes 配置目录（默认：`~/.hermes`）。也限定网关 PID 文件和 systemd 服务名称，因此多个安装可以并发运行 |
| `HERMES_KANBAN_HOME` | 覆盖共享 Hermes 根目录，作为 kanban 看板的锚点（db + 工作区 + 工作线程日志）。回退到 `get_default_hermes_root()`（任何活动 profile 的父目录）。对测试和异常部署有用 |
| `HERMES_KANBAN_BOARD` | 为此进程固定活动 kanban 看板。优先于 `~/.hermes/kanban/current`；调度器将其注入工作线程子进程环境，因此工作线程物理上无法看到其他看板上的任务。默认为 `default`。Slug 验证：小写字母数字 + 连字符 + 下划线，1-64 个字符 |
| `HERMES_KANBAN_DB` | 直接固定 kanban 数据库文件路径（最高优先级；高于 `HERMES_KANBAN_BOARD` 和 `HERMES_KANBAN_HOME`）。调度器将其注入工作线程子进程环境，因此 profile 工作线程汇聚到调度器的看板 |
| `HERMES_KANBAN_WORKSPACES_ROOT` | 直接固定 kanban 工作区根目录（工作区最高优先级；高于 `HERMES_KANBAN_HOME`）。调度器将其注入工作线程子进程环境 |

## Provider 认证（OAuth）

对于原生 Anthropic 认证，当 Claude Code 自己的凭证文件存在时，Hermes 优先使用它们，因为这些凭证可以自动刷新。**针对 Anthropic 的 OAuth 需要带有购买额外使用积分的 Claude Max 计划** — Hermes 路由为 Claude Code，它只消耗 Max 计划的额外/超额积分，不是基础 Max 配额，且不适用于 Claude Pro。没有 Max + 额外积分，请改用 API 密钥。环境变量如 `ANTHROPIC_TOKEN` 作为手动覆盖仍然有用，但不再是首选路径。

| 变量 | 描述 |
|----------|-------------|
| `HERMES_INFERENCE_PROVIDER` | 覆盖 provider 选择：`auto`、`custom`、`openrouter`、`nous`、`openai-codex`、`copilot`、`copilot-acp`、`anthropic`、`huggingface`、`gemini`、`zai`、`kimi-coding`、`kimi-coding-cn`、`minimax`、`minimax-cn`、`minimax-oauth`（浏览器 OAuth 登录 — 不需要 API 密钥；请参见 [MiniMax OAuth 指南](../guides/minimax-oauth.md)）、`kilocode`、`xiaomi`、`arcee`、`gmi`、`stepfun`、`alibaba`、`alibaba-coding-plan`（别名 `alibaba_coding`）、`deepseek`、`nvidia`、`ollama-cloud`、`xai`（别名 `grok`）、`google-gemini-cli`、`qwen-oauth`、`bedrock`、`opencode-zen`、`opencode-go`、`ai-gateway`、`tencent-tokenhub`（默认：`auto`） |
| `HERMES_PORTAL_BASE_URL` | 覆盖 Nous Portal URL（用于开发/测试） |
| `NOUS_INFERENCE_BASE_URL` | 覆盖 Nous 推理 API URL |
| `HERMES_NOUS_MIN_KEY_TTL_SECONDS` | 重新颁发前的最小代理密钥 TTL（默认：1800 = 30 分钟） |
| `HERMES_NOUS_TIMEOUT_SECONDS` | Nous 凭证/令牌流程的 HTTP 超时 |
| `HERMES_DUMP_REQUESTS` | 将 API 请求有效载荷转储到日志文件（`true`/`false`） |
| `HERMES_PREFILL_MESSAGES_FILE` | 在 API 调用时注入的临时预填充消息的 JSON 文件路径 |
| `HERMES_TIMEZONE` | IANA 时区覆盖（如 `America/New_York`） |

## 工具 API

| 变量 | 描述 |
|----------|-------------|
| `PARALLEL_API_KEY` | AI 原生网络搜索（[parallel.ai](https://parallel.ai/)） |
| `FIRECRAWL_API_KEY` | 网络抓取和云浏览器（[firecrawl.dev](https://firecrawl.dev/)） |
| `FIRECRAWL_API_URL` | 自托管实例的自定义 Firecrawl API 端点（可选） |
| `TAVILY_API_KEY` | AI 原生网络搜索、提取和抓取的 Tavily API 密钥（[app.tavily.com](https://app.tavily.com/home)） |
| `SEARXNG_URL` | SearXNG 实例 URL，用于免费自托管网络搜索 — 不需要 API 密钥（[searxng.github.io](https://searxng.github.io/searxng/)） |
| `TAVILY_BASE_URL` | 覆盖 Tavily API 端点。对企业代理和自托管 Tavily 兼容搜索后端有用。与 `GROQ_BASE_URL` 相同的模式。 |
| `EXA_API_KEY` | AI 原生网络搜索和内容的 Exa API 密钥（[exa.ai](https://exa.ai/)） |
| `BROWSERBASE_API_KEY` | 浏览器自动化（[browserbase.com](https://browserbase.com/)） |
| `BROWSERBASE_PROJECT_ID` | Browserbase 项目 ID |
| `BROWSER_USE_API_KEY` | Browser Use 云浏览器 API 密钥（[browser-use.com](https://browser-use.com/)） |
| `FIRECRAWL_BROWSER_TTL` | Firecrawl 浏览器会话 TTL 秒数（默认：300） |
| `BROWSER_CDP_URL` | 本地浏览器的 Chrome DevTools Protocol URL（通过 `/browser connect` 设置，如 `ws://localhost:9222`） |
| `CAMOFOX_URL` | Camofox 本地反检测浏览器 URL（默认：`http://localhost:9377`） |
| `BROWSER_INACTIVITY_TIMEOUT` | 浏览器会话不活动超时秒数 |
| `FAL_KEY` | 图像生成（[fal.ai](https://fal.ai/)） |
| `GROQ_API_KEY` | Groq Whisper STT API 密钥（[groq.com](https://groq.com/)） |
| `ELEVENLABS_API_KEY` | ElevenLabs 高级 TTS 语音（[elevenlabs.io](https://elevenlabs.io/)） |
| `STT_GROQ_MODEL` | 覆盖 Groq STT 模型（默认：`whisper-large-v3-turbo`） |
| `GROQ_BASE_URL` | 覆盖 Groq OpenAI 兼容 STT 端点 |
| `STT_OPENAI_MODEL` | 覆盖 OpenAI STT 模型（默认：`whisper-1`） |
| `STT_OPENAI_BASE_URL` | 覆盖 OpenAI 兼容 STT 端点 |
| `GITHUB_TOKEN` | 用于 Skills Hub 的 GitHub 令牌（更高的 API 速率限制，技能发布） |
| `HONCHO_API_KEY` | 跨会话用户建模（[honcho.dev](https://honcho.dev/)） |
| `HONCHO_BASE_URL` | 自托管 Honcho 实例的 base URL（默认：Honcho 云）。本地实例不需要 API 密钥 |
| `HINDSIGHT_TIMEOUT` | Hindsight 记忆 provider API 调用的超时秒数（默认：`60`）。如果在 `/sync` 或 `on_session_switch` 期间 Hindsight 实例响应缓慢而在 `errors.log` 中看到超时，请增大此值。 |
| `SUPERMEMORY_API_KEY` | 带 profile 召回和会话摄取的语义长期记忆（[supermemory.ai](https://supermemory.ai)） |
| `TINKER_API_KEY` | RL 训练（[tinker-console.thinkingmachines.ai](https://tinker-console.thinkingmachines.ai/)） |
| `WANDB_API_KEY` | RL 训练指标（[wandb.ai](https://wandb.ai/)） |
| `DAYTONA_API_KEY` | Daytona 云沙箱（[daytona.io](https://daytona.io/)） |
| `VERCEL_TOKEN` | Vercel Sandbox 访问令牌（[vercel.com](https://vercel.com/)） |
| `VERCEL_PROJECT_ID` | Vercel 项目 ID（需要 `VERCEL_TOKEN`） |
| `VERCEL_TEAM_ID` | Vercel 团队 ID（需要 `VERCEL_TOKEN`） |
| `VERCEL_OIDC_TOKEN` | Vercel 短期 OIDC 令牌（仅开发替代方案） |

### Langfuse 可观测性

捆绑的 [`observability/langfuse`](/docs/user-guide/features/built-in-plugins#observabilitylangfuse) 插件的环境变量。用 `hermes tools → Langfuse Observability` 或在 `~/.hermes/.env` 中手动设置。插件也必须启用（`hermes plugins enable observability/langfuse`）才能生效。

| 变量 | 描述 |
|----------|-------------|
| `HERMES_LANGFUSE_PUBLIC_KEY` | Langfuse 项目公钥（`pk-lf-...`）。必需。 |
| `HERMES_LANGFUSE_SECRET_KEY` | Langfuse 项目密钥（`sk-lf-...`）。必需。 |
| `HERMES_LANGFUSE_BASE_URL` | Langfuse 服务器 URL（默认：`https://cloud.langfuse.com`）。自托管时设置。 |
| `HERMES_LANGFUSE_ENV` | 追踪的环境标签（`production`、`staging`、…） |
| `HERMES_LANGFUSE_RELEASE` | 追踪的发布/版本标签 |
| `HERMES_LANGFUSE_SAMPLE_RATE` | SDK 采样率 0.0–1.0（默认：`1.0`） |
| `HERMES_LANGFUSE_MAX_CHARS` | 序列化有效载荷的每字段截断（默认：`12000`） |
| `HERMES_LANGFUSE_DEBUG` | `true` 启用详细插件日志到 `agent.log` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | 标准 Langfuse SDK 名称。当 `HERMES_LANGFUSE_*` 等效项未设置时作为后备接受。 |

### Nous 工具网关

这些变量配置付费 Nous 订阅者或自托管网关部署的[工具网关](/docs/user-guide/features/tool-gateway)。大多数用户不需要设置这些 — 网关通过 `hermes model` 或 `hermes tools` 自动配置。

| 变量 | 描述 |
|----------|-------------|
| `TOOL_GATEWAY_DOMAIN` | 工具网关路由的基础域（默认：`nousresearch.com`） |
| `TOOL_GATEWAY_SCHEME` | 网关 URL 的 HTTP 或 HTTPS 方案（默认：`https`） |
| `TOOL_GATEWAY_USER_TOKEN` | 工具网关的认证令牌（通常从 Nous auth 自动填充） |
| `FIRECRAWL_GATEWAY_URL` | 专门覆盖 Firecrawl 网关端点的 URL |

## 终端后端

| 变量 | 描述 |
|----------|-------------|
| `TERMINAL_ENV` | 后端：`local`、`docker`、`ssh`、`singularity`、`modal`、`daytona`、`vercel_sandbox` |
| `HERMES_DOCKER_BINARY` | 覆盖 Hermes 调用的容器二进制文件（如 `podman`、`/usr/local/bin/docker`）。未设置时，Hermes 在 `PATH` 上自动发现 `docker` 或 `podman`。当两者都安装而你又想使用非默认的，或者二进制在 `PATH` 之外时需要。 |
| `TERMINAL_DOCKER_IMAGE` | Docker 镜像（默认：`nikolaik/python-nodejs:python3.11-nodejs20`） |
| `TERMINAL_DOCKER_FORWARD_ENV` | 显式转发到 Docker 终端会话的环境变量名 JSON 数组。注意：技能声明的 `required_environment_variables` 会自动转发 — 只有未通过任何技能声明的变量才需要这个。 |
| `TERMINAL_DOCKER_VOLUMES` | 额外的 Docker 卷挂载（逗号分隔的 `host:container` 对） |
| `TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE` | 高级选择性加入：将启动 cwd 挂载到 Docker `/workspace`（`true`/`false`，默认：`false`） |
| `TERMINAL_SINGULARITY_IMAGE` | Singularity 镜像或 `.sif` 路径 |
| `TERMINAL_MODAL_IMAGE` | Modal 容器镜像 |
| `TERMINAL_DAYTONA_IMAGE` | Daytona 沙箱镜像 |
| `TERMINAL_VERCEL_RUNTIME` | Vercel Sandbox 运行时（`node24`、`node22`、`python3.13`） |
| `TERMINAL_TIMEOUT` | 命令超时秒数 |
| `TERMINAL_LIFETIME_SECONDS` | 终端会话的最大生命周期秒数 |
| `TERMINAL_CWD` | 终端会话的工作目录（仅网关/cron；CLI 使用启动目录） |
| `SUDO_PASSWORD` | 启用 sudo 而不需要交互式提示 |

对于云沙箱后端，持久性是文件系统导向的。`TERMINAL_LIFETIME_SECONDS` 控制 Hermes 清理空闲终端会话的时间，稍后恢复可能会重新创建沙箱而不是保持相同的运行进程。

## SSH 后端

| 变量 | 描述 |
|----------|-------------|
| `TERMINAL_SSH_HOST` | 远程服务器主机名 |
| `TERMINAL_SSH_USER` | SSH 用户名 |
| `TERMINAL_SSH_PORT` | SSH 端口（默认：22） |
| `TERMINAL_SSH_KEY` | 私钥路径 |
| `TERMINAL_SSH_PERSISTENT` | 覆盖 SSH 的持久 shell（默认：遵循 `TERMINAL_PERSISTENT_SHELL`） |

## 容器资源（Docker、Singularity、Modal、Daytona）

| 变量 | 描述 |
|----------|-------------|
| `TERMINAL_CONTAINER_CPU` | CPU 核心数（默认：1） |
| `TERMINAL_CONTAINER_MEMORY` | 内存 MB（默认：5120） |
| `TERMINAL_CONTAINER_DISK` | 磁盘 MB（默认：51200） |
| `TERMINAL_CONTAINER_PERSISTENT` | 跨会话持久化容器文件系统（默认：`true`） |
| `TERMINAL_SANDBOX_DIR` | 工作区和覆盖层的主机目录（默认：`~/.hermes/sandboxes/`） |

## 持久 Shell

| 变量 | 描述 |
|----------|-------------|
| `TERMINAL_PERSISTENT_SHELL` | 为非本地后端启用持久 shell（默认：`true`）。也可通过 config.yaml 中的 `terminal.persistent_shell` 设置 |
| `TERMINAL_LOCAL_PERSISTENT` | 为本地后端启用持久 shell（默认：`false`） |
| `TERMINAL_SSH_PERSISTENT` | 覆盖 SSH 后端的持久 shell（默认：遵循 `TERMINAL_PERSISTENT_SHELL`） |

## 消息

| 变量 | 描述 |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot 令牌（来自 @BotFather） |
| `TELEGRAM_ALLOWED_USERS` | 允许使用 bot 的逗号分隔用户 ID（适用于 DM、群组和论坛） |
| `TELEGRAM_GROUP_ALLOWED_USERS` | 仅在群组/论坛中授权的逗号分隔发送者用户 ID（**不**授予 DM 访问权限）。以 `-` 开头的聊天 ID 形状值仍作为聊天 ID 向前兼容 pre-#17686 配置，带弃用警告。 |
| `TELEGRAM_GROUP_ALLOWED_CHATS` | 允许的逗号分隔群组/论坛聊天 ID；任何成员都被授权 |
| `TELEGRAM_HOME_CHANNEL` | cron 交付的默认 Telegram 聊天/频道 |
| `TELEGRAM_HOME_CHANNEL_NAME` | Telegram 主页频道的显示名称 |
| `TELEGRAM_WEBHOOK_URL` | Webhook 模式的公开 HTTPS URL（启用 webhook 而不是轮询） |
| `TELEGRAM_WEBHOOK_PORT` | Webhook 服务器的本地监听端口（默认：`8443`） |
| `TELEGRAM_WEBHOOK_SECRET` | Telegram 在每个更新中回显的机密令牌以进行验证。**只要设置了 `TELEGRAM_WEBHOOK_URL` 就必需** — 没有它网关拒绝启动（GHSA-3vpc-7q5r-276h）。用 `openssl rand -hex 32` 生成。 |
| `TELEGRAM_REACTIONS` | 在处理期间对消息启用 emoji 回应（默认：`false`） |
| `TELEGRAM_REPLY_TO_MODE` | 回复引用行为：`off`、`first`（默认）或 `all`。匹配 Discord 模式。 |
| `TELEGRAM_IGNORED_THREADS` | Bot 永不响应的逗号分隔 Telegram 论坛主题/线程 ID |
| `TELEGRAM_PROXY` | Telegram 连接的代理 URL — 覆盖 `HTTPS_PROXY`。支持 `http://`、`https://`、`socks5://` |
| `DISCORD_BOT_TOKEN` | Discord bot 令牌 |
| `DISCORD_ALLOWED_USERS` | 允许使用 bot 的逗号分隔 Discord 用户 ID |
| `DISCORD_ALLOWED_ROLES` | 允许使用 bot 的逗号分隔 Discord 角色 ID（与 `DISCORD_ALLOWED_USERS` OR）。自动启用 Members intent。当审核团队人员变动时有用 — 角色授权会自动传播。 |
| `DISCORD_ALLOWED_CHANNELS` | 逗号分隔的 Discord 频道 ID。设置后，bot 只在这些频道（加上允许的 DM）中响应。覆盖 `config.yaml` 的 `discord.allowed_channels`。 |
| `DISCORD_PROXY` | Discord 连接的代理 URL — 覆盖 `HTTPS_PROXY`。支持 `http://`、`https://`、`socks5://` |
| `DISCORD_HOME_CHANNEL` | cron 交付的默认 Discord 频道 |
| `DISCORD_HOME_CHANNEL_NAME` | Discord 主页频道的显示名称 |
| `DISCORD_COMMAND_SYNC_POLICY` | Discord 斜杠命令启动同步策略：`safe`（差异和调和）、`bulk`（旧版 `tree.sync()`）或 `off` |
| `DISCORD_REQUIRE_MENTION` | 在服务器频道中响应前需要 @mention |
| `DISCORD_FREE_RESPONSE_CHANNELS` | 不需要 mention 的逗号分隔频道 ID |
| `DISCORD_AUTO_THREAD` | 支持时为长回复自动创建线程 |
| `DISCORD_REACTIONS` | 在处理期间对消息启用 emoji 回应（默认：`true`） |
| `DISCORD_IGNORED_CHANNELS` | Bot 永不响应的逗号分隔频道 ID |
| `DISCORD_NO_THREAD_CHANNELS` | Bot 不自动创建线程的逗号分隔频道 ID |
| `DISCORD_REPLY_TO_MODE` | 回复引用行为：`off`、`first`（默认）或 `all` |
| `DISCORD_ALLOW_MENTION_EVERYONE` | 允许 bot @ping `@everyone`/`@here`（默认：`false`）。请参见 [Mention 控制](../user-guide/messaging/discord.md#mention-control)。 |
| `DISCORD_ALLOW_MENTION_ROLES` | 允许 bot @ping `@role` mentions（默认：`false`）。 |
| `DISCORD_ALLOW_MENTION_USERS` | 允许 bot @ping 单独的 `@user` mentions（默认：`true`）。 |
| `DISCORD_ALLOW_MENTION_REPLIED_USER` | 回复其消息时 ping 作者（默认：`true`）。 |
| `SLACK_BOT_TOKEN` | Slack bot 令牌（`xoxb-...`） |
| `SLACK_APP_TOKEN` | Slack app 级别令牌（`xapp-...`，Socket Mode 必需） |
| `SLACK_ALLOWED_USERS` | 逗号分隔的 Slack 用户 ID |
| `SLACK_HOME_CHANNEL` | cron 交付的默认 Slack 频道 |
| `SLACK_HOME_CHANNEL_NAME` | Slack 主页频道的显示名称 |
| `GOOGLE_CHAT_PROJECT_ID` | 托管 Pub/Sub 主题的 GCP 项目（回退到 `GOOGLE_CLOUD_PROJECT`） |
| `GOOGLE_CHAT_SUBSCRIPTION_NAME` | 完整 Pub/Sub 订阅路径，`projects/{proj}/subscriptions/{sub}`（旧别名：`GOOGLE_CHAT_SUBSCRIPTION`） |
| `GOOGLE_CHAT_SERVICE_ACCOUNT_JSON` | 服务帐号 JSON 的路径，或内联 JSON（回退到 `GOOGLE_APPLICATION_CREDENTIALS`） |
| `GOOGLE_CHAT_ALLOWED_USERS` | 允许与 bot 聊天的逗号分隔用户邮箱 |
| `GOOGLE_CHAT_ALLOW_ALL_USERS` | 允许任何 Google Chat 用户触发 bot（仅开发） |
| `GOOGLE_CHAT_HOME_CHANNEL` | cron 交付的默认空间（如 `spaces/AAAA...`） |
| `GOOGLE_CHAT_HOME_CHANNEL_NAME` | Google Chat 主页空间的显示名称 |
| `GOOGLE_CHAT_MAX_MESSAGES` | Pub/Sub FlowControl 最大飞行中消息数（默认：`1`） |
| `GOOGLE_CHAT_MAX_BYTES` | Pub/Sub FlowControl 最大飞行中字节数（默认：`16777216`，16 MiB） |
| `GOOGLE_CHAT_BOOTSTRAP_SPACES` | 逗号分隔的额外空间 ID，在解析 bot 自己的 `users/{id}` 时启动探测 |
| `GOOGLE_CHAT_DEBUG_RAW` | 设置任何值以在 DEBUG 级别记录编辑的 Pub/Sub 信封（仅调试） |
| `WHATSAPP_ENABLED` | 启用 WhatsApp 桥接（`true`/`false`） |
| `WHATSAPP_MODE` | `bot`（独立号码）或 `self-chat`（给自己发消息） |
| `WHATSAPP_ALLOWED_USERS` | 逗号分隔的电话号码（带国家代码，无 `+`），或 `*` 允许所有发送者 |
| `WHATSAPP_ALLOW_ALL_USERS` | 允许所有 WhatsApp 发送者无需允许列表（`true`/`false`） |
| `WHATSAPP_DEBUG` | 在桥接中记录原始消息事件以进行故障排除（`true`/`false`） |
| `SIGNAL_HTTP_URL` | signal-cli 守护进程 HTTP 端点（如 `http://127.0.0.1:8080`） |
| `SIGNAL_ACCOUNT` | E.164 格式的 bot 电话号码 |
| `SIGNAL_ALLOWED_USERS` | 逗号分隔的 E.164 电话号码或 UUID |
| `SIGNAL_GROUP_ALLOWED_USERS` | 逗号分隔的群组 ID，或 `*` 代表所有群组 |
| `SIGNAL_HOME_CHANNEL_NAME` | Signal 主页频道的显示名称 |
| `SIGNAL_IGNORE_STORIES` | 忽略 Signal 故事/状态更新 |
| `SIGNAL_ALLOW_ALL_USERS` | 允许所有 Signal 用户无需允许列表 |
| `TWILIO_ACCOUNT_SID` | Twilio 帐号 SID（与语音技能共享） |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token（与语音技能共享；也用于 webhook 签名验证） |
| `TWILIO_PHONE_NUMBER` | E.164 格式的 Twilio 电话号码（与语音技能共享） |
| `SMS_WEBHOOK_URL` | Twilio 签名验证的公开 URL — 必须与 Twilio Console 中的 webhook URL 匹配（必需） |
| `SMS_WEBHOOK_PORT` | 入站 SMS 的 webhook 监听端口（默认：`8080`） |
| `SMS_WEBHOOK_HOST` | Webhook 绑定地址（默认：`0.0.0.0`） |
| `SMS_INSECURE_NO_SIGNATURE` | 设置为 `true` 以禁用 Twilio 签名验证（仅本地开发 — 不要用于生产） |
| `SMS_ALLOWED_USERS` | 逗号分隔的 E.164 电话号码允许聊天 |
| `SMS_ALLOW_ALL_USERS` | 允许所有 SMS 发送者无需允许列表 |
| `SMS_HOME_CHANNEL` | cron 作业/通知交付的电话号码 |
| `SMS_HOME_CHANNEL_NAME` | SMS 主页频道的显示名称 |
| `EMAIL_ADDRESS` | Email 网关适配器的邮箱地址 |
| `EMAIL_PASSWORD` | 邮箱密码或应用密码 |
| `EMAIL_IMAP_HOST` | 邮件适配器的 IMAP 主机名 |
| `EMAIL_IMAP_PORT` | IMAP 端口 |
| `EMAIL_SMTP_HOST` | 邮件适配器的 SMTP 主机名 |
| `EMAIL_SMTP_PORT` | SMTP 端口 |
| `EMAIL_ALLOWED_USERS` | 允许向 bot 发消息的逗号分隔邮箱地址 |
| `EMAIL_HOME_ADDRESS` | 主动邮件交付的默认收件人 |
| `EMAIL_HOME_ADDRESS_NAME` | 邮件主页目标的显示名称 |
| `EMAIL_POLL_INTERVAL` | 邮件轮询间隔秒数 |
| `EMAIL_ALLOW_ALL_USERS` | 允许所有入站邮件发送者 |
| `DINGTALK_CLIENT_ID` | 来自开发者门户的 DingTalk bot AppKey（[open.dingtalk.com](https://open.dingtalk.com)） |
| `DINGTALK_CLIENT_SECRET` | 来自开发者门户的 DingTalk bot AppSecret |
| `DINGTALK_ALLOWED_USERS` | 逗号分隔的允许向 bot 发消息的 DingTalk 用户 ID |
| `FEISHU_APP_ID` | 来自 [open.feishu.cn](https://open.feishu.cn/) 的 Feishu/Lark bot App ID |
| `FEISHU_APP_SECRET` | Feishu/Lark bot App Secret |
| `FEISHU_DOMAIN` | `feishu`（中国）或 `lark`（国际）。默认：`feishu` |
| `FEISHU_CONNECTION_MODE` | `websocket`（推荐）或 `webhook`。默认：`websocket` |
| `FEISHU_ENCRYPT_KEY` | Webhook 模式的可选加密密钥 |
| `FEISHU_VERIFICATION_TOKEN` | Webhook 模式的可选验证令牌 |
| `FEISHU_ALLOWED_USERS` | 逗号分隔的允许向 bot 发消息的 Feishu 用户 ID |
| `FEISHU_ALLOW_BOTS` | `none`（默认）/ `mentions` / `all` — 接受来自其他 bot 的入站消息。请参见 [机器人间消息](../user-guide/messaging/feishu.md#bot-to-bot-messaging) |
| `FEISHU_REQUIRE_MENTION` | `true`（默认）/ `false` — 群组消息是否必须 @mention bot。通过 `group_rules.<chat_id>.require_mention` 按聊天覆盖。 |
| `FEISHU_HOME_CHANNEL` | cron 交付和通知的 Feishu 聊天 ID |
| `WECOM_BOT_ID` | 来自管理控制台的 WeCom AI Bot ID |
| `WECOM_SECRET` | WeCom AI Bot secret |
| `WECOM_WEBSOCKET_URL` | 自定义 WebSocket URL（默认：`wss://openws.work.weixin.qq.com`） |
| `WECOM_ALLOWED_USERS` | 逗号分隔的允许向 bot 发消息的 WeCom 用户 ID |
| `WECOM_HOME_CHANNEL` | cron 交付和通知的 WeCom 聊天 ID |
| `WECOM_CALLBACK_CORP_ID` | 回调自建应用的 WeCom 企业 Corp ID |
| `WECOM_CALLBACK_CORP_SECRET` | 自建应用的企业 secret |
| `WECOM_CALLBACK_AGENT_ID` | 自建应用的 Agent ID |
| `WECOM_CALLBACK_TOKEN` | 回调验证令牌 |
| `WECOM_CALLBACK_ENCODING_AES_KEY` | 回调加密的 AES 密钥 |
| `WECOM_CALLBACK_HOST` | 回调服务器绑定地址（默认：`0.0.0.0`） |
| `WECOM_CALLBACK_PORT` | 回调服务器端口（默认：`8645`） |
| `WECOM_CALLBACK_ALLOWED_USERS` | 逗号分隔的用户 ID 允许列表 |
| `WECOM_CALLBACK_ALLOW_ALL_USERS` | 设置 `true` 以允许所有用户无需允许列表 |
| `WEIXIN_ACCOUNT_ID` | 通过 iLink Bot API QR 登录获取的微信账号 ID |
| `WEIXIN_TOKEN` | 通过 iLink Bot API QR 登录获取的微信认证令牌 |
| `WEIXIN_BASE_URL` | 覆盖微信 iLink Bot API base URL（默认：`https://ilinkai.weixin.qq.com`） |
| `WEIXIN_CDN_BASE_URL` | 覆盖微信 CDN base URL 用于媒体（默认：`https://novac2c.cdn.weixin.qq.com/c2c`） |
| `WEIXIN_DM_POLICY` | 直接消息策略：`open`、`allowlist`、`pairing`、`disabled`（默认：`open`） |
| `WEIXIN_GROUP_POLICY` | 群消息策略：`open`、`allowlist`、`disabled`（默认：`disabled`） |
| `WEIXIN_ALLOWED_USERS` | 逗号分隔的允许 DM bot 的微信用户 ID |
| `WEIXIN_GROUP_ALLOWED_USERS` | 逗号分隔的允许与 bot 互动的微信**群聊 ID**（不是成员用户 ID）。变量名是遗留的 — 它期望群 ID。只有当 iLink 实际传递群事件时才生效；QR 登录 iLink bot 身份（`...@im.bot`）通常不会收到普通微信群消息。 |
| `WEIXIN_HOME_CHANNEL` | cron 交付和通知的微信聊天 ID |
| `WEIXIN_HOME_CHANNEL_NAME` | 微信主页频道的显示名称 |
| `WEIXIN_ALLOW_ALL_USERS` | 允许所有微信用户无需允许列表（`true`/`false`） |
| `BLUEBUBBLES_SERVER_URL` | BlueBubbles 服务器 URL（如 `http://192.168.1.10:1234`） |
| `BLUEBUBBLES_PASSWORD` | BlueBubbles 服务器密码 |
| `BLUEBUBBLES_WEBHOOK_HOST` | Webhook 监听器绑定地址（默认：`127.0.0.1`） |
| `BLUEBUBBLES_WEBHOOK_PORT` | Webhook 监听器端口（默认：`8645`） |
| `BLUEBUBBLES_HOME_CHANNEL` | 用于 cron/通知交付的电话/邮箱 |
| `BLUEBUBBLES_ALLOWED_USERS` | 逗号分隔的授权用户 |
| `BLUEBUBBLES_ALLOW_ALL_USERS` | 允许所有用户（`true`/`false`） |
| `QQ_APP_ID` | 来自 [q.qq.com](https://q.qq.com) 的 QQ Bot App ID |
| `QQ_CLIENT_SECRET` | 来自 [q.qq.com](https://q.qq.com) 的 QQ Bot App Secret |
| `QQ_STT_API_KEY` | 外部 STT 后备 provider 的 API 密钥（可选，QQ 内置 ASR 返回无文本时使用） |
| `QQ_STT_BASE_URL` | 外部 STT provider 的 base URL（可选） |
| `QQ_STT_MODEL` | 外部 STT provider 的模型名称（可选） |
| `QQ_ALLOWED_USERS` | 逗号分隔的允许向 bot 发消息的 QQ 用户 openID |
| `QQ_GROUP_ALLOWED_USERS` | 允许群 @-消息的逗号分隔 QQ 群 ID |
| `QQ_ALLOW_ALL_USERS` | 允许所有用户（`true`/`false`，覆盖 `QQ_ALLOWED_USERS`） |
| `QQBOT_HOME_CHANNEL` | 用于 cron 交付和通知的 QQ 用户/群 openID |
| `QQBOT_HOME_CHANNEL_NAME` | QQ 主页频道的显示名称 |
| `QQ_PORTAL_HOST` | 覆盖 QQ portal 主机（设置为 `sandbox.q.qq.com` 以通过沙箱网关路由；默认：`q.qq.com`）。 |
| `MATTERMOST_URL` | Mattermost 服务器 URL（如 `https://mm.example.com`） |
| `MATTERMOST_TOKEN` | Mattermost 的 bot 令牌或个人访问令牌 |
| `MATTERMOST_ALLOWED_USERS` | 逗号分隔的允许向 bot 发消息的 Mattermost 用户 ID |
| `MATTERMOST_HOME_CHANNEL` | 主动消息交付（cron、通知）的频道 ID |
| `MATTERMOST_REQUIRE_MENTION` | 在频道中需要 `@mention`（默认：`true`）。设置为 `false` 以响应所有消息。 |
| `MATTERMOST_FREE_RESPONSE_CHANNELS` | Bot 不需要 `@mention` 即可响应的逗号分隔频道 ID |
| `MATTERMOST_REPLY_MODE` | 回复样式：`thread`（线程回复）或 `off`（扁平消息，默认） |
| `MATRIX_HOMESERVER` | Matrix homeserver URL（如 `https://matrix.org`） |
| `MATRIX_ACCESS_TOKEN` | 用于 bot 认证的 Matrix 访问令牌 |
| `MATRIX_USER_ID` | Matrix 用户 ID（如 `@hermes:matrix.org`） — 密码登录必需，访问令牌可选 |
| `MATRIX_PASSWORD` | Matrix 密码（访问令牌的替代方案） |
| `MATRIX_ALLOWED_USERS` | 逗号分隔的允许向 bot 发消息的 Matrix 用户 ID（如 `@alice:matrix.org`） |
| `MATRIX_HOME_ROOM` | 主动消息交付（cron、通知）的房间 ID（如 `!abc123:matrix.org`） |
| `MATRIX_ENCRYPTION` | 启用端到端加密（`true`/`false`，默认：`false`） |
| `MATRIX_DEVICE_ID` | E2EE 跨重启持久化的稳定 Matrix 设备 ID（如 `HERMES_BOT`）。没有这个，E2EE 密钥每次启动都会轮换，历史房间解密会中断。 |
| `MATRIX_REACTIONS` | 在处理生命周期的 emoji 回应入站消息（默认：`true`）。设置为 `false` 以禁用。 |
| `MATRIX_REQUIRE_MENTION` | 在房间中需要 `@mention`（默认：`true`）。设置为 `false` 以响应所有消息。 |
| `MATRIX_FREE_RESPONSE_ROOMS` | Bot 不需要 `@mention` 即可响应的逗号分隔房间 ID |
| `MATRIX_AUTO_THREAD` | 为房间消息自动创建线程（默认：`true`） |
| `MATRIX_DM_MENTION_THREADS` | 当 bot 在 DM 中被 `@mentioned` 时创建线程（默认：`false`） |
| `MATRIX_RECOVERY_KEY` | 设备密钥轮换后交叉签名验证的恢复密钥。对于启用了交叉签名的 E2EE 设置推荐。 |
| `HASS_TOKEN` | Home Assistant 长期访问令牌（启用 HA 平台 + 工具） |
| `HASS_URL` | Home Assistant URL（默认：`http://homeassistant.local:8123`） |
| `WEBHOOK_ENABLED` | 启用 webhook 平台适配器（`true`/`false`） |
| `WEBHOOK_PORT` | 接收 webhooks 的 HTTP 服务器端口（默认：`8644`） |
| `WEBHOOK_SECRET` | webhook 签名验证的全局 HMAC 密钥（当路由未指定自己的密钥时使用） |
| `API_SERVER_ENABLED` | 启用 OpenAI 兼容 API 服务器（`true`/`false`）。与其他平台一起运行。 |
| `API_SERVER_KEY` | API 服务器认证的 Bearer 令牌。对于非回环绑定强制执行。 |
| `API_SERVER_CORS_ORIGINS` | 允许直接调用 API 服务器的逗号分隔浏览器来源（如 `http://localhost:3000,http://127.0.0.1:3000`）。默认：禁用。 |
| `API_SERVER_PORT` | API 服务器的端口（默认：`8642`） |
| `API_SERVER_HOST` | API 服务器的主机/绑定地址（默认：`127.0.0.1`）。使用 `0.0.0.0` 进行网络访问 — 需要 `API_SERVER_KEY` 和窄的 `API_SERVER_CORS_ORIGINS` 允许列表。 |
| `API_SERVER_MODEL_NAME` | 在 `/v1/models` 上通告的模型名称。默认为 profile 名称（或默认 profile 的 `hermes-agent`）。对于需要每个连接不同模型名称的多用户设置（如 Open WebUI）有用。 |
| `GATEWAY_PROXY_URL` | 转发消息到的远程 Hermes API 服务器的 URL（[代理模式](/docs/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos)）。设置后，网关只处理平台 I/O — 所有代理工作委托给远程服务器。也可通过 `config.yaml` 中的 `gateway.proxy_url` 配置。 |
| `GATEWAY_PROXY_KEY` | 在代理模式中与远程 API 服务器认证的 Bearer 令牌。必须与远程主机上的 `API_SERVER_KEY` 匹配。 |
| `MESSAGING_CWD` | 消息模式中终端命令的工作目录（默认：`~`） |
| `GATEWAY_ALLOWED_USERS` | 所有平台上允许的逗号分隔用户 ID |
| `GATEWAY_ALLOW_ALL_USERS` | 允许所有用户无需允许列表（`true`/`false`，默认：`false`） |

### 高级消息调优

每个平台的出站消息批处理节流高级旋钮。大多数用户永远不需要接触这些；默认值设置为尊重每个平台的速率限制而不会感觉迟缓。

| 变量 | 描述 |
|----------|-------------|
| `HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS` | 刷新排队 Telegram 文本块的宽限期（默认：`0.6`）。 |
| `HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS` | 单个 Telegram 消息超过长度限制时分块之间的延迟（默认：`2.0`）。 |
| `HERMES_TELEGRAM_MEDIA_BATCH_DELAY_SECONDS` | 刷新排队 Telegram 媒体的宽限期（默认：`0.6`）。 |
| `HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS` | 代理完成后发送跟进消息之前的延迟，以避免与最后一个流块竞争。 |
| `HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT` / `_READ_TIMEOUT` / `_WRITE_TIMEOUT` / `_POOL_TIMEOUT` | 覆盖底层 `python-telegram-bot` HTTP 超时（秒）。 |
| `HERMES_TELEGRAM_HTTP_POOL_SIZE` | 到 Telegram API 的最大并发 HTTP 连接数。 |
| `HERMES_TELEGRAM_DISABLE_FALLBACK_IPS` | DNS 失败时禁用硬编码的 Cloudflare 后备 IP（`true`/`false`）。 |
| `HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS` | 刷新排队 Discord 文本块的宽限期（默认：`0.6`）。 |
| `HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS` | Discord 消息超过长度限制时分块之间的延迟（默认：`2.0`）。 |
| `HERMES_MATRIX_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | Matrix 的 Telegram 批处理旋钮等效项。 |
| `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` / `_MAX_CHARS` / `_MAX_MESSAGES` | Feishu 批处理调优 — 延迟、分块延迟、每条消息最大字符数、每批最大消息数。 |
| `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | Feishu 媒体刷新延迟。 |
| `HERMES_FEISHU_DEDUP_CACHE_SIZE` | Feishu webhook 去重缓存大小（默认：`1024`）。 |
| `HERMES_WECOM_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | WeCom 批处理调优。 |
| `HERMES_VISION_DOWNLOAD_TIMEOUT` | 在将图像交给视觉模型之前下载图像的超时秒数（默认：`30`）。 |
| `HERMES_RESTART_DRAIN_TIMEOUT` | 网关：在 `/restart` 上强制重启前等待活动运行耗尽的秒数（默认：`900`）。 |
| `HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT` | 网关启动期间每个平台的连接超时（秒）。 |
| `HERMES_GATEWAY_BUSY_INPUT_MODE` | 默认网关繁忙输入行为：`queue`、`steer` 或 `interrupt`。可以用 `/busy` 按聊天覆盖。 |
| `HERMES_GATEWAY_BUSY_ACK_ENABLED` | 当用户在代理忙碌时发送输入时，网关是否发送确认消息（⚡/⏳/⏩）（默认：`true`）。设置为 `false` 以完全抑制这些消息 — 输入仍然正常排队/转向/中断，只有聊天回复被静音。从 `config.yaml` 中的 `display.busy_ack_enabled` 桥接。 |
| `HERMES_CRON_TIMEOUT` | cron 作业代理运行的不活动超时秒数（默认：`600`）。当代理主动调用工具或接收流令牌时会无限运行 — 这只在空闲时触发。设置为 `0` 则无限制。 |
| `HERMES_CRON_SCRIPT_TIMEOUT` | 附加到 cron 作业的预运行脚本的超时秒数（默认：`120`）。覆盖需要更长执行时间的脚本（如反机器人定时的随机延迟）。也可通过 `config.yaml` 中的 `cron.script_timeout_seconds` 配置。 |
| `HERMES_CRON_MAX_PARALLEL` | 每刻度并行运行的最大 cron 作业数（默认：`4`）。 |

## 代理行为

| 变量 | 描述 |
|----------|-------------|
| `HERMES_MAX_ITERATIONS` | 每轮对话的最大工具调用迭代次数（默认：90） |
| `HERMES_INFERENCE_MODEL` | 在进程级别覆盖模型名称（对会话优先于 `config.yaml`）。也可通过 `-m`/`--model` 标志设置。 |
| `HERMES_YOLO_MODE` | 设置为 `1` 以绕过危险命令确认提示。等同于 `--yolo`。 |
| `HERMES_ACCEPT_HOOKS` | 自动批准 `config.yaml` 中声明的任何未见过的 shell 钩子，无需 TTY 提示。等同于 `--accept-hooks` 或 `hooks_auto_accept: true`。 |
| `HERMES_IGNORE_USER_CONFIG` | 跳过 `~/.hermes/config.yaml` 并使用内置默认值（`.env` 中的凭证仍会加载）。等同于 `--ignore-user-config`。 |
| `HERMES_IGNORE_RULES` | 跳过 `AGENTS.md`、`SOUL.md`、`.cursorrules`、记忆和预加载技能的自动注入。等同于 `--ignore-rules`。 |
| `HERMES_MD_NAMES` | 逗号分隔的自动注入的规则文件名列表（默认：`AGENTS.md,CLAUDE.md,.cursorrules,SOUL.md`）。 |
| `HERMES_TOOL_PROGRESS` | 工具进度显示的已弃用兼容性变量。优先使用 `config.yaml` 中的 `display.tool_progress`。 |
| `HERMES_TOOL_PROGRESS_MODE` | 工具进度模式的已弃用兼容性变量。优先使用 `config.yaml` 中的 `display.tool_progress`。 |
| `HERMES_HUMAN_DELAY_MODE` | 响应节奏：`off`/`natural`/`custom` |
| `HERMES_HUMAN_DELAY_MIN_MS` | 自定义延迟范围最小值（ms） |
| `HERMES_HUMAN_DELAY_MAX_MS` | 自定义延迟范围最大值（ms） |
| `HERMES_QUIET` | 抑制非必要输出（`true`/`false`） |
| `HERMES_API_TIMEOUT` | LLM API 调用超时秒数（默认：`1800`） |
| `HERMES_API_CALL_STALE_TIMEOUT` | 非流式过时调用超时秒数（默认：`300`）。未设置时自动为本地 provider 禁用。也可通过 `config.yaml` 中的 `providers.<id>.stale_timeout_seconds` 或 `providers.<id>.models.<model>.stale_timeout_seconds` 配置。 |
| `HERMES_STREAM_READ_TIMEOUT` | 流式套接字读取超时秒数（默认：`120`）。自动增加到本地 provider 的 `HERMES_API_TIMEOUT`。如果本地 LLM 在长代码生成期间超时，增加此值。 |
| `HERMES_STREAM_STALE_TIMEOUT` | 过时流检测超时秒数（默认：`180`）。自动为本地 provider 禁用。如果在此窗口内没有块到达，则触发连接终止。 |
| `HERMES_STREAM_RETRIES` | 瞬态网络错误时流中重新连接尝试次数（默认：`3`）。 |
| `HERMES_AGENT_TIMEOUT` | 运行中代理的网关不活动超时秒数（默认：`900`）。每次工具调用和流式令牌重置。设置为 `0` 则禁用。 |
| `HERMES_AGENT_TIMEOUT_WARNING` | 网关：在这么多秒不活动后发送警告消息（默认：75% 的 `HERMES_AGENT_TIMEOUT`）。 |
| `HERMES_AGENT_NOTIFY_INTERVAL` | 网关：长运行代理回合之间进度通知的间隔秒数。 |
| `HERMES_CHECKPOINT_TIMEOUT` | 文件系统检查点创建的超时秒数（默认：`30`）。 |
| `HERMES_EXEC_ASK` | 在网关模式下启用执行批准提示（`true`/`false`） |
| `HERMES_ENABLE_PROJECT_PLUGINS` | 启用从 `./.hermes/plugins/` 的 repo 本地插件自动发现（`true`/`false`，默认：`false`） |
| `HERMES_BACKGROUND_NOTIFICATIONS` | 网关中的后台进程通知模式：`all`（默认）、`result`、`error`、`off` |
| `HERMES_EPHEMERAL_SYSTEM_PROMPT` | 在 API 调用时注入的临时系统提示（从不持久化到会话） |
| `HERMES_PREFILL_MESSAGES_FILE` | 在 API 调用时注入的临时预填充消息的 JSON 文件路径。 |
| `HERMES_ALLOW_PRIVATE_URLS` | `true`/`false` — 允许工具获取 localhost/私有网络 URL。网关模式下默认关闭。 |
| `HERMES_REDACT_SECRETS` | `true`/`false` — 控制工具输出、日志和聊天响应中的秘密编辑（默认：`true`）。 |
| `HERMES_WRITE_SAFE_ROOT` | 可选目录前缀，限制 `write_file`/`patch` 写入；路径外需要批准。 |
| `HERMES_DISABLE_FILE_STATE_GUARD` | 设置为 `1` 以关闭 `patch`/`write_file` 上的"文件自读取后已更改"保护。 |
| `HERMES_CORE_TOOLS` | 逗号分隔的规范核心工具列表覆盖（高级；很少需要）。 |
| `HERMES_BUNDLED_SKILLS` | 启动时加载的捆绑技能列表的逗号分隔覆盖。 |
| `HERMES_OPTIONAL_SKILLS` | 首次运行时自动安装的可选技能名称逗号分隔列表。 |
| `HERMES_DEBUG_INTERRUPT` | 设置为 `1` 以将详细的 interrupt/cancel 追踪记录到 `agent.log`。 |
| `HERMES_DUMP_REQUESTS` | 将 API 请求有效载荷转储到日志文件（`true`/`false`） |
| `HERMES_DUMP_REQUEST_STDOUT` | 将 API 请求有效载荷转储到 stdout 而不是日志文件。 |
| `HERMES_OAUTH_TRACE` | 设置为 `1` 以记录 OAuth 令牌交换和刷新尝试。包括编辑的时间信息。 |
| `HERMES_OAUTH_FILE` | OAuth 凭证存储的覆盖路径（默认：`~/.hermes/auth.json`）。 |
| `HERMES_AGENT_HELP_GUIDANCE` | 为自定义部署将额外的指导文本追加到系统提示。 |
| `HERMES_AGENT_LOGO` | CLI 启动时的 ASCII 横幅 logo 覆盖。 |
| `DELEGATION_MAX_CONCURRENT_CHILDREN` | 每个 `delegate_task` 批次的最大并行子代理数（默认：`3`，下限 1，无上限）。也可通过 `config.yaml` 中的 `delegation.max_concurrent_children` 配置 — 配置值优先。 |

## 界面

| 变量 | 描述 |
|----------|-------------|
| `HERMES_TUI` | 设置为 `1` 时启动 [TUI](../user-guide/tui.md) 而不是经典 CLI。等同于传递 `--tui`。 |
| `HERMES_TUI_DIR` | 预构建 `ui-tui/` 目录的路径（必须包含 `dist/entry.js` 和填充的 `node_modules`）。由发行版和 Nix 使用以跳过首次启动 `npm install`。 |
| `HERMES_TUI_RESUME` | 启动时恢复特定 TUI 会话 ID。设置后，`hermes --tui` 跳过生成新会话而是拾取命名会话 — 对断开连接或终端崩溃后重新附加有用。 |
| `HERMES_TUI_THEME` | 强制 TUI 颜色主题：`light`、`dark` 或原始 6 字符背景十六进制（如 `ffffff` 或 `1a1a2e`）。未设置时，Hermes 使用 `COLORFGBG` 和终端背景查询自动检测；对于不设置 `COLORFGBG` 的终端（Ghostty、Warp、iTerm2 等）此变量覆盖检测。 |
| `HERMES_INFERENCE_MODEL` | 强制 `hermes -z` / `hermes chat` 的模型而不修改 `config.yaml`。与 `HERMES_INFERENCE_PROVIDER` 配对。对需要每次运行覆盖默认模型的脚本化调用者（sweeper、CI、批处理运行器）有用。 |

## 会话设置

| 变量 | 描述 |
|----------|-------------|
| `SESSION_IDLE_MINUTES` | N 分钟不活动后重置会话（默认：1440） |
| `SESSION_RESET_HOUR` | 每日重置小时，24 小时制（默认：4 = 凌晨 4 点） |

## 上下文压缩（仅 config.yaml）

上下文压缩仅通过 `config.yaml` 配置 — 没有对应的环境变量。阈值设置在 `compression:` 块中，而摘要模型/provider 位于 `auxiliary.compression:` 下。

```yaml
compression:
  enabled: true
  threshold: 0.50
  target_ratio: 0.20         # 保留为最近尾部的阈值分数
  protect_last_n: 20         # 保持未压缩的最小最近消息数
```

:::info 旧版迁移
带有 `compression.summary_model`、`compression.summary_provider` 和 `compression.summary_base_url` 的旧配置在首次加载时自动迁移到 `auxiliary.compression.*`。
:::

## 辅助任务覆盖

| 变量 | 描述 |
|----------|-------------|
| `AUXILIARY_VISION_PROVIDER` | 视觉任务的 provider 覆盖 |
| `AUXILIARY_VISION_MODEL` | 视觉任务的模型覆盖 |
| `AUXILIARY_VISION_BASE_URL` | 视觉任务的直接 OpenAI 兼容端点 |
| `AUXILIARY_VISION_API_KEY` | 与 `AUXILIARY_VISION_BASE_URL` 配对的 API 密钥 |
| `AUXILIARY_WEB_EXTRACT_PROVIDER` | Web 提取/摘要的 provider 覆盖 |
| `AUXILIARY_WEB_EXTRACT_MODEL` | Web 提取/摘要的模型覆盖 |
| `AUXILIARY_WEB_EXTRACT_BASE_URL` | Web 提取/摘要的直接 OpenAI 兼容端点 |
| `AUXILIARY_WEB_EXTRACT_API_KEY` | 与 `AUXILIARY_WEB_EXTRACT_BASE_URL` 配对的 API 密钥 |

对于特定任务的直接端点，Hermes 使用任务的配置的 API 密钥或 `OPENAI_API_KEY`。它不会为这些自定义端点重用 `OPENROUTER_API_KEY`。

## 备用 Provider（仅 config.yaml）

主模型备用链仅通过 `config.yaml` 配置 — 没有对应的环境变量。添加顶级 `fallback_providers` 列表，包含 `provider` 和 `model` 键以在主模型遇到错误时启用自动故障转移。

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

较旧的顶级 `fallback_model` 单 provider 形状仍被读取以保持向后兼容，但新配置应使用 `fallback_providers`。

请参见 [备用 Provider](/docs/user-guide/features/fallback-providers) 获取完整详情。

## Provider 路由（仅 config.yaml）

这些放在 `~/.hermes/config.yaml` 的 `provider_routing` 部分下：

| 键 | 描述 |
|-----|-------------|
| `sort` | 排序 provider：`"price"`（默认）、`"throughput"` 或 `"latency"` |
| `only` | 允许的 provider slug 列表（如 `["anthropic", "google"]`） |
| `ignore` | 跳过的 provider slug 列表 |
| `order` | 按顺序尝试的 provider slug 列表 |
| `require_parameters` | 仅使用支持所有请求参数的 provider（`true`/`false`） |
| `data_collection` | `"allow"`（默认）或 `"deny"` 以排除可能存储/训练数据的 provider |

:::tip
使用 `hermes config set` 设置环境变量 — 它自动将它们保存到正确的文件（`.env` 用于秘密，其他所有内容到 `config.yaml`）。
:::
