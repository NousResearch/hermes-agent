---
title: "AI 提供商"
sidebar_label: "AI 提供商"
sidebar_position: 1
---

# AI 提供商

本页面介绍如何为 Hermes Agent 配置推理提供商——从 OpenRouter 和 Anthropic 等云 API，到 Ollama 和 vLLM 等自托管端点，再到高级路由和回退配置。使用 Hermes 至少需要配置一个提供商。

## 推理提供商

你至少需要一种连接到 LLM 的方式。使用 `hermes model` 交互式切换提供商和模型，或直接配置：

| 提供商 | 配置方式 |
|----------|-------|
| **Nous Portal** | `hermes model`（OAuth，基于订阅） |
| **OpenAI Codex** | `hermes model`（ChatGPT OAuth，使用 Codex 模型） |
| **GitHub Copilot** | `hermes model`（OAuth 设备代码流程，`COPILOT_GITHUB_TOKEN`、`GH_TOKEN` 或 `gh auth token`） |
| **GitHub Copilot ACP** | `hermes model`（启动本地 `copilot --acp --stdio`） |
| **Anthropic** | `hermes model`（通过 OAuth 的 Claude Max + 额外使用额度；也支持 Anthropic API 密钥或手动设置 token——见下方说明） |
| **OpenRouter** | `~/.hermes/.env` 中的 `OPENROUTER_API_KEY` |
| **AI Gateway** | `~/.hermes/.env` 中的 `AI_GATEWAY_API_KEY`（provider: `ai-gateway`） |
| **z.ai / GLM** | `~/.hermes/.env` 中的 `GLM_API_KEY`（provider: `zai`） |
| **Kimi / Moonshot** | `~/.hermes/.env` 中的 `KIMI_API_KEY`（provider: `kimi-coding`） |
| **Kimi / Moonshot（中国）** | `~/.hermes/.env` 中的 `KIMI_CN_API_KEY`（provider: `kimi-coding-cn`；别名：`kimi-cn`、`moonshot-cn`） |
| **Arcee AI** | `~/.hermes/.env` 中的 `ARCEEAI_API_KEY`（provider: `arcee`；别名：`arcee-ai`、`arceeai`） |
| **GMI Cloud** | `~/.hermes/.env` 中的 `GMI_API_KEY`（provider: `gmi`；别名：`gmi-cloud`、`gmicloud`） |
| **MiniMax** | `~/.hermes/.env` 中的 `MINIMAX_API_KEY`（provider: `minimax`） |
| **MiniMax 中国** | `~/.hermes/.env` 中的 `MINIMAX_CN_API_KEY`（provider: `minimax-cn`） |
| **阿里云** | `~/.hermes/.env` 中的 `DASHSCOPE_API_KEY`（provider: `alibaba`） |
| **阿里编程计划** | `DASHSCOPE_API_KEY`（provider: `alibaba-coding-plan`，别名：`alibaba_coding`）—— 独立计费 SKU，不同端点 |
| **Kilo Code** | `~/.hermes/.env` 中的 `KILOCODE_API_KEY`（provider: `kilocode`） |
| **小米 MiMo** | `~/.hermes/.env` 中的 `XIAOMI_API_KEY`（provider: `xiaomi`，别名：`mimo`、`xiaomi-mimo`） |
| **腾讯 TokenHub** | `~/.hermes/.env` 中的 `TOKENHUB_API_KEY`（provider: `tencent-tokenhub`，别名：`tencent`、`tokenhub`、`tencentmaas`） |
| **OpenCode Zen** | `~/.hermes/.env` 中的 `OPENCODE_ZEN_API_KEY`（provider: `opencode-zen`） |
| **OpenCode Go** | `~/.hermes/.env` 中的 `OPENCODE_GO_API_KEY`（provider: `opencode-go`） |
| **DeepSeek** | `~/.hermes/.env` 中的 `DEEPSEEK_API_KEY`（provider: `deepseek`） |
| **Hugging Face** | `~/.hermes/.env` 中的 `HF_TOKEN`（provider: `huggingface`，别名：`hf`） |
| **Google / Gemini** | `~/.hermes/.env` 中的 `GOOGLE_API_KEY`（或 `GEMINI_API_KEY`）（provider: `gemini`） |
| **Google Gemini（OAuth）** | `hermes model` → 选择"Google Gemini (OAuth)"（provider: `google-gemini-cli`，支持免费配额，浏览器 PKCE 登录） |
| **LM Studio** | `hermes model` → 选择"LM Studio"（provider: `lmstudio`，可选 `LM_API_KEY`） |
| **自定义端点** | `hermes model` → 选择"Custom endpoint"（保存在 `config.yaml`） |

关于官方 API 密钥路径，请参阅 [Google Gemini 指南](/docs/guides/google-gemini)。

:::tip 模型密钥别名
在 `model:` 配置部分，你可以使用 `default:` 或 `model:` 作为模型 ID 的键名。`model: { default: my-model }` 和 `model: { model: my-model }` 的效果完全相同。
:::


### Google Gemini via OAuth（`google-gemini-cli`）

`google-gemini-cli` 提供商使用 Google 的 Cloud Code Assist 后端 —— 与 Google 自家的 `gemini-cli` 工具使用的 API 相同。同时支持**免费配额**（个人账户每日额度慷慨）和**付费层级**（通过 GCP 项目的 Standard/Enterprise）。

**快速开始：**

```bash
hermes model
# → 选择"Google Gemini (OAuth)"
# → 查看策略警告，确认
# → 浏览器打开 accounts.google.com，登录
# → 完成 — Hermes 在首次请求时自动配置你的免费配额
```

Hermes 默认提供 Google 的**公开** `gemini-cli` 桌面 OAuth 客户端 —— 与 Google 在开源 `gemini-cli` 中包含的凭据相同。桌面 OAuth 客户端不是机密客户端（PKCE 提供安全性）。你无需安装 `gemini-cli` 或注册自己的 GCP OAuth 客户端。

**认证原理：**
- 针对 `accounts.google.com` 的 PKCE 授权码流程
- 浏览器回调到 `http://127.0.0.1:8085/oauth2callback`（如果繁忙则回退到临时端口）
- Token 存储在 `~/.hermes/auth/google_oauth.json`（chmod 0600，原子写入，跨进程 `fcntl` 锁）
- 过期前 60 秒自动刷新
- 无头环境（SSH、`HERMES_HEADLESS=1`）→ 粘贴模式回退
- 飞行中刷新去重 —— 两个并发请求不会双重刷新
- `invalid_grant`（已撤销刷新）→ 凭据文件被清除，提示用户重新登录

**推理原理：**
- 流量发送到 `https://cloudcode-pa.googleapis.com/v1internal:generateContent`
  （或流式的 `:streamGenerateContent?alt=sse`），**不是**付费的 `v1beta/openai` 端点
- 请求体包装 `{project, model, user_prompt_id, request}`
- OpenAI 格式的 `messages[]`、`tools[]`、`tool_choice` 被翻译为 Gemini 原生的
  `contents[]`、`tools[].functionDeclarations`、`toolConfig` 格式
- 响应被翻译回 OpenAI 格式，因此 Hermes 的其余部分保持不变

**层级和项目 ID：**

| 你的情况 | 怎么做 |
|---|---|
| 个人 Google 账户，想要免费配额 | 无需操作 —— 登录后开始聊天 |
| Workspace / Standard / Enterprise 账户 | 设置 `HERMES_GEMINI_PROJECT_ID` 或 `GOOGLE_CLOUD_PROJECT` 为你的 GCP 项目 ID |
| VPC-SC 保护的 org | Hermes 检测到 `SECURITY_POLICY_VIOLATED` 并自动强制使用 `standard-tier` |

免费配额在首次使用时自动配置 Google 管理的项目。无需 GCP 设置。

**配额监控：**

```
/gquota
```

显示每个模型的剩余 Code Assist 配额，带进度条：

```
Gemini Code Assist 配额  (项目: 123-abc)

  gemini-2.5-pro                      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░   85%
  gemini-2.5-flash [input]            ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░   92%
```

:::warning 策略风险
Google 认为在第三方软件中使用 Gemini CLI OAuth 客户端违反政策。部分用户报告了账户限制。要获得最低风险体验，请通过 `gemini` 提供商使用你自己的 API 密钥。Hermes 在 OAuth 开始前显示明确警告并需要确认。
:::

**自定义 OAuth 客户端（可选）：**

如果你想注册自己的 Google OAuth 客户端 —— 例如，将配额和同意范围限定为你自己的 GCP 项目 —— 设置：

```bash
HERMES_GEMINI_CLIENT_ID=your-client.apps.googleusercontent.com
HERMES_GEMINI_CLIENT_SECRET=...   # 桌面客户端可选
```

在 [console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials) 注册一个**桌面应用** OAuth 客户端，并启用 Generative Language API。

:::info Codex 说明
OpenAI Codex 提供商通过设备代码进行身份验证（打开 URL，输入代码）。Hermes 将生成的凭据存储在自己的 auth store 中（`~/.hermes/auth.json`），并在存在时可以从 `~/.codex/auth.json` 导入现有的 Codex CLI 凭据。无需安装 Codex CLI。
:::

:::warning
即使使用 Nous Portal、Codex 或自定义端点，某些工具（视觉、Web 摘要、MoA）使用单独的"辅助"模型。默认情况下（`auxiliary.*.provider: "auto"`），Hermes 将这些任务路由到你的**主聊天模型** —— 你在 `hermes model` 中选择的同一模型。你可以单独覆盖每个任务，将其路由到更便宜/更快的模型（例如 OpenRouter 上的 Gemini Flash）—— 见 [辅助模型](/docs/user-guide/configuration#auxiliary-models)。
:::

:::tip Nous Tool Gateway
付费 Nous Portal 订阅者还可以访问 **[Tool Gateway](/docs/user-guide/features/tool-gateway)** —— 通过你的订阅路由 Web 搜索、图像生成、TTS 和浏览器自动化。无需额外 API 密钥。在 `hermes model` 设置期间自动提供，或稍后用 `hermes tools` 启用。
:::

### 模型管理的两个命令

Hermes 有**两个**模型命令服务于不同目的：

| 命令 | 运行位置 | 功能 |
|---------|-------------|--------------|
| **`hermes model`** | 你的终端（任何会话外） | 完整设置向导 —— 添加提供商、运行 OAuth、输入 API 密钥、配置端点 |
| **`/model`** | Hermes 聊天会话内 | 在**已配置**的提供商和模型之间快速切换 |

如果你试图切换到尚未设置的提供商（例如，你只配置了 OpenRouter，想使用 Anthropic），你需要 `hermes model`，而不是 `/model`。首先退出会话（`Ctrl+C` 或 `/quit`），运行 `hermes model`，完成提供商设置，然后开始新会话。

### Anthropic（原生）

直接通过 Anthropic API 使用 Claude 模型 —— 无需 OpenRouter 代理。支持三种认证方式：

:::caution 需要 Claude Max"额外使用"额度
当你通过 `hermes model` → Anthropic OAuth 进行身份验证（或通过 `hermes auth add anthropic --type oauth`）时，Hermes 作为 Claude Code 通过你的 Anthropic 账户路由。**只有当你有 Claude Max 计划并购买了额外使用额度时才能使用。** Max 计划附带的默认使用额度（Claude Code 默认包含的使用量）不被 Hermes 消耗 —— 只有你额外购买的用量/超量额度才会被消耗。Claude Pro 订阅者不能使用此路径。

如果你没有 Max + 额外额度，请使用 `ANTHROPIC_API_KEY` —— 请求按该密钥所属组织的按 token 付费计费（标准 API 定价，独立于任何 Claude 订阅）。
:::

```bash
# 使用 API 密钥（按 token 付费）
export ANTHROPIC_API_KEY=***
hermes chat --provider anthropic --model claude-sonnet-4-6

# 首选：通过 `hermes model` 进行身份验证
# 当可用时，Hermes 直接使用 Claude Code 的凭据存储
hermes model

# 手动使用 setup-token 覆盖（回退 / 传统）
export ANTHROPIC_TOKEN=***  # setup-token 或手动 OAuth token
hermes chat --provider anthropic

# 自动检测 Claude Code 凭据（如果你已使用 Claude Code）
hermes chat --provider anthropic  # 自动读取 Claude Code 凭据文件
```

当你通过 `hermes model` 选择 Anthropic OAuth 时，Hermes 优先使用 Claude Code 自己的凭据存储，而不是将 token 复制到 `~/.hermes/.env`。这样可以保持可刷新的 Claude 凭据的可刷新性。

或永久设置：
```yaml
model:
  provider: "anthropic"
  default: "claude-sonnet-4-6"
```

:::tip 别名
`--provider claude` 和 `--provider claude-code` 也可以作为 `--provider anthropic` 的简写。
:::

### GitHub Copilot

Hermes 支持 GitHub Copilot 作为一级提供商，有两种模式：

**`copilot` — 直接 Copilot API**（推荐）。使用你的 GitHub Copilot 订阅通过 Copilot API 访问 GPT-5.x、Claude、Gemini 和其他模型。

```bash
hermes chat --provider copilot --model gpt-5.4
```

**认证选项**（按此顺序检查）：

1. `COPILOT_GITHUB_TOKEN` 环境变量
2. `GH_TOKEN` 环境变量
3. `GITHUB_TOKEN` 环境变量
4. `gh auth token` CLI 回退

如果未找到 token，`hermes model` 提供**OAuth 设备代码登录** —— 与 Copilot CLI 和 opencode 使用的流程相同。

:::warning Token 类型
Copilot API **不支持**经典个人访问 Token（`ghp_*`）。支持的 token 类型：

| 类型 | 前缀 | 获取方式 |
|------|--------|------------|
| OAuth token | `gho_` | `hermes model` → GitHub Copilot → 用 GitHub 登录 |
| 细粒度 PAT | `github_pat_` | GitHub 设置 → 开发者设置 → 细粒度 token（需要 **Copilot Requests** 权限） |
| GitHub App token | `ghu_` | 通过 GitHub App 安装 |
| **classic PAT** | `ghp_` | ❌ 不支持 |

如果你的 `gh auth token` 返回 `ghp_*` token，请用 `hermes model` 通过 OAuth 代替进行身份验证。
:::

:::info Copilot 在 Hermes 中的认证行为
Hermes 直接将支持的 GitHub token（`gho_*`、`github_pat_*` 或 `ghu_*`）发送到 `api.githubcopilot.com`，并包含 Copilot 特定的 header（`Editor-Version`、`Copilot-Integration-Id`、`Openai-Intent`、`x-initiator`）。

在 HTTP 401 时，Hermes 现在在回退前执行一次性凭据恢复：

1. 通过正常优先级链重新解析 token（`COPILOT_GITHUB_TOKEN` → `GH_TOKEN` → `GITHUB_TOKEN` → `gh auth token`）
2. 用刷新的 header 重建共享的 OpenAI 客户端
3. 重试请求一次

部分旧版社区代理使用 `api.github.com/copilot_internal/v2/token` 交换流程。该端点对于某些账户类型可能不可用（返回 404）。因此，Hermes 将直接 token 认证作为主要路径，依靠运行时凭据刷新 + 重试来保证健壮性。
:::

**API 路由**：GPT-5+ 模型（`gpt-5-mini` 除外）自动使用 Responses API。所有其他模型（GPT-4o、Claude、Gemini 等）使用 Chat Completions。模型从实时 Copilot 目录自动检测。

**`copilot-acp` — Copilot ACP agent 后端**。将本地 Copilot CLI 作为子进程启动：

```bash
hermes chat --provider copilot-acp --model copilot-acp
# 需要 Copilot CLI 在 PATH 中且存在 `copilot login` 会话
```

**永久配置：**
```yaml
model:
  provider: "copilot"
  default: "gpt-5.4"
```

| 环境变量 | 描述 |
|---------------------|-------------|
| `COPILOT_GITHUB_TOKEN` | Copilot API 的 GitHub token（最高优先级） |
| `HERMES_COPILOT_ACP_COMMAND` | 覆盖 Copilot CLI 二进制路径（默认：`copilot`） |
| `HERMES_COPILOT_ACP_ARGS` | 覆盖 ACP 参数（默认：`--acp --stdio`） |

### 一级 API 密钥提供商

这些提供商有内置支持，具有专用提供商 ID。设置 API 密钥并使用 `--provider` 选择：

```bash
# z.ai / ZhipuAI GLM
hermes chat --provider zai --model glm-5
# 需要：`~/.hermes/.env` 中的 GLM_API_KEY

# Kimi / Moonshot AI（国际：api.moonshot.ai）
hermes chat --provider kimi-coding --model kimi-for-coding
# 需要：`~/.hermes/.env` 中的 KIMI_API_KEY

# Kimi / Moonshot AI（中国：api.moonshot.cn）
hermes chat --provider kimi-coding-cn --model kimi-k2.5
# 需要：`~/.hermes/.env` 中的 KIMI_CN_API_KEY

# MiniMax（全球端点）
hermes chat --provider minimax --model MiniMax-M2.7
# 需要：`~/.hermes/.env` 中的 MINIMAX_API_KEY

# MiniMax（中国端点）
hermes chat --provider minimax-cn --model MiniMax-M2.7
# 需要：`~/.hermes/.env` 中的 MINIMAX_CN_API_KEY

# 阿里云 / DashScope（Qwen 模型）
hermes chat --provider alibaba --model qwen3.5-plus
# 需要：`~/.hermes/.env` 中的 DASHSCOPE_API_KEY

# 小米 MiMo
hermes chat --provider xiaomi --model mimo-v2-pro
# 需要：`~/.hermes/.env` 中的 XIAOMI_API_KEY

# 腾讯 TokenHub（Hy3 Preview）
hermes chat --provider tencent-tokenhub --model hy3-preview
# 需要：`~/.hermes/.env` 中的 TOKENHUB_API_KEY

# Arcee AI（Trinity 模型）
hermes chat --provider arcee --model trinity-large-thinking
# 需要：`~/.hermes/.env` 中的 ARCEEAI_API_KEY

# GMI Cloud
# 使用 GMI 的 /v1/models 端点返回的确切模型 ID。
hermes chat --provider gmi --model zai-org/GLM-5.1-FP8
# 需要：`~/.hermes/.env` 中的 GMI_API_KEY
```

或在 `config.yaml` 中永久设置提供商：
```yaml
model:
  provider: "gmi"
  default: "zai-org/GLM-5.1-FP8"
```

可以使用 `GLM_BASE_URL`、`KIMI_BASE_URL`、`MINIMAX_BASE_URL`、`MINIMAX_CN_BASE_URL`、`DASHSCOPE_BASE_URL`、`XIAOMI_BASE_URL`、`GMI_BASE_URL` 或 `TOKENHUB_BASE_URL` 环境变量覆盖基础 URL。

:::note Z.AI 端点自动检测
使用 Z.AI / GLM 提供商时，Hermes 自动探测多个端点（全球、中国、编程变体）以找到接受你 API 密钥的端点。你无需手动设置 `GLM_BASE_URL` —— 工作的端点被自动检测和缓存。
:::

### xAI（Grok）—— Responses API + 提示缓存

xAI 通过 Responses API（`codex_responses` 传输）连接，以在 Grok 4 模型上获得自动推理支持 —— 无需 `reasoning_effort` 参数，服务器默认推理。在 `~/.hermes/.env` 中设置 `XAI_API_KEY`，在 `hermes model` 中选择 xAI，或在 `/model` 中输入 `grok` 作为快捷方式 `grok-4-1-fast-reasoning`。

当使用 xAI 作为提供商时（任何包含 `x.ai` 的基础 URL），Hermes 通过在每个 API 请求中发送 `x-grok-conv-id` header 自动启用提示缓存。这将请求路由到会话内的同一服务器，允许 xAI 的基础设施重用缓存的系统提示和对话历史。

无需配置 —— 当检测到 xAI 端点且有会话 ID 可用时，缓存自动激活。这减少了多轮对话的延迟和成本。

xAI 还附带专用 TTS 端点（`/v1/tts`）。在 `hermes tools` → Voice & TTS 中选择 **xAI TTS**，或参阅 [Voice & TTS](../user-guide/features/tts.md#text-to-speech) 页面的配置。

### Ollama Cloud —— 托管 Ollama 模型，OAuth + API 密钥

[Ollama Cloud](https://ollama.com/cloud) 托管与本地 Ollama 相同的开放权重模型目录，但无需 GPU。在 `hermes model` 中选择 **Ollama Cloud**，从 [ollama.com/settings/keys](https://ollama.com/settings/keys) 粘贴你的 API 密钥，Hermes 自动发现可用模型。

```bash
hermes model
# → 选择"Ollama Cloud"
# → 粘贴你的 OLLAMA_API_KEY
# → 从发现的模型中选择（gpt-oss:120b、glm-4.6:cloud、qwen3-coder:480b-cloud 等）
```

或直接在 `config.yaml` 中：
```yaml
model:
  provider: "ollama-cloud"
  default: "gpt-oss:120b"
```

模型目录从 `ollama.com/v1/models` 动态获取，缓存一小时。`model:tag` 符号（例如 `qwen3-coder:480b-cloud`）通过规范化保持 —— 不要使用破折号。

:::tip Ollama Cloud vs 本地 Ollama
两者使用相同的 OpenAI 兼容 API。云是一级提供商（`--provider ollama-cloud`、`OLLAMA_API_KEY`）；本地 Ollama 通过自定义端点流程访问（基础 URL `http://localhost:11434/v1`，无需密钥）。使用云处理你无法在本地运行的大型模型；使用本地进行隐私保护或离线工作。
:::

### AWS Bedrock

通过 AWS Bedrock 使用 Anthropic Claude、Amazon Nova、DeepSeek v3.2、Meta Llama 4 等模型。使用 AWS SDK（`boto3`）凭据链 —— 无需 API 密钥，只需标准 AWS 认证。

```bash
# 最简单 —— ~/.aws/credentials 中的命名配置文件
hermes chat --provider bedrock --model us.anthropic.claude-sonnet-4-6

# 或使用显式环境变量
AWS_PROFILE=myprofile AWS_REGION=us-east-1 hermes chat --provider bedrock --model us.anthropic.claude-sonnet-4-6
```

或在 `config.yaml` 中永久设置：
```yaml
model:
  provider: "bedrock"
  default: "us.anthropic.claude-sonnet-4-6"
bedrock:
  region: "us-east-1"          # 或设置 AWS_REGION
  # profile: "myprofile"       # 或设置 AWS_PROFILE
  # discovery: true            # 从 IAM 自动发现区域
  # guardrail:                 # 可选 Bedrock Guardrails
  #   id: "your-guardrail-id"
  #   version: "DRAFT"
```

认证使用标准 boto3 链：显式 `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`、`~/.aws/credentials` 中的 `AWS_PROFILE`、EC2/ECS/Lambda 上的 IAM 角色、IMDS 或 SSO。如果你已经使用 AWS CLI 认证，则无需环境变量。

Bedrock 在底层使用 **Converse API** —— 请求被翻译为 Bedrock 的模型无关格式，因此相同配置适用于 Claude、Nova、DeepSeek 和 Llama 模型。只有在调用非默认区域端点时才需要设置 `BEDROCK_BASE_URL`。

有关 IAM 设置、区域选择和跨区域推理的详细说明，请参阅 [AWS Bedrock 指南](/docs/guides/aws-bedrock)。

### Qwen Portal（OAuth）

阿里 Qwen Portal，带基于浏览器的 OAuth 登录。在 `hermes model` 中选择 **Qwen OAuth (Portal)**，通过浏览器登录，Hermes 持久化 refresh token。

```bash
hermes model
# → 选择"Qwen OAuth (Portal)"
# → 浏览器打开；用你的阿里账户登录
# → 确认 —— 凭据保存到 ~/.hermes/auth.json

hermes chat   # 使用 portal.qwen.ai/v1 端点
```

或配置 `config.yaml`：
```yaml
model:
  provider: "qwen-oauth"
  default: "qwen3-coder-plus"
```

仅在门户端点迁移时才需要设置 `HERMES_QWEN_BASE_URL`（默认：`https://portal.qwen.ai/v1`）。

:::tip Qwen OAuth vs DashScope（阿里）
`qwen-oauth` 使用面向消费者的 Qwen Portal，带 OAuth 登录 —— 适合个人用户。`alibaba` 提供商使用 DashScope 企业 API，带 `DASHSCOPE_API_KEY` —— 适合编程式/生产工作负载。两者都路由到 Qwen 系列模型，但位于不同端点。
:::

### 阿里编程计划

如果你订阅了阿里的**编程计划**（与标准 DashScope API 访问分开的定价 SKU），Hermes 将其作为独立的一级提供商公开：`alibaba-coding-plan`。端点：`https://coding-intl.dashscope.aliyuncs.com/v1`。与常规 `alibaba` 提供商一样 OpenAI 兼容，但基础 URL 和计费范围不同。

```yaml
model:
  provider: alibaba_coding     # alibaba-coding-plan 的别名
  model: qwen3-coder-plus
```

或从 CLI：

```bash
hermes chat --provider alibaba_coding --model qwen3-coder-plus
```

`alibaba_coding` 使用你的 `alibaba` 条目已经在使用的相同 `DASHSCOPE_API_KEY` —— 无需单独密钥，只是不同的路由目标。在这个提供商注册之前，在 `config.yaml` 中设置 `provider: alibaba_coding` 的用户会静默回退到 OpenRouter 路由。

### MiniMax（OAuth）

MiniMax-M2.7 通过浏览器 OAuth 登录 —— 无需 API 密钥。在 `hermes model` 中选择 **MiniMax (OAuth)**，通过浏览器登录，Hermes 持久化 access + refresh token。底层使用 Anthropic Messages 兼容端点（`/anthropic`）。

```bash
hermes model
# → 选择"MiniMax (OAuth)"
# → 浏览器打开；用你的 MiniMax 账户登录（全球或中国区域）
# → 确认 —— 凭据保存到 ~/.hermes/auth.json

hermes chat   # 使用 api.minimax.io/anthropic 端点
```

或配置 `config.yaml`：
```yaml
model:
  provider: "minimax-oauth"
  default: "MiniMax-M2.7"
```

支持的模型：`MiniMax-M2.7`（主模型）和 `MiniMax-M2.7-highspeed`（作为默认辅助模型连接）。OAuth 路径忽略 `MINIMAX_API_KEY` / `MINIMAX_BASE_URL`。

:::tip MiniMax OAuth vs API 密钥
`minimax-oauth` 使用 MiniMax 的面向消费者 Portal，带 OAuth 登录 —— 无需计费设置。`minimax` 和 `minimax-cn` 提供商使用 `MINIMAX_API_KEY` / `MINIMAX_CN_API_KEY` —— 用于编程式访问。详见 [MiniMax OAuth 指南](/docs/guides/minimax-oauth)。
:::

### NVIDIA NIM

通过 [build.nvidia.com](https://build.nvidia.com)（免费 API 密钥）或本地 NIM 端点使用 Nemotron 和其他开源模型。

```bash
# 云端（build.nvidia.com）
hermes chat --provider nvidia --model nvidia/nemotron-3-super-120b-a12b
# 需要：`~/.hermes/.env` 中的 NVIDIA_API_KEY

# 本地 NIM 端点 —— 覆盖基础 URL
NVIDIA_BASE_URL=http://localhost:8000/v1 hermes chat --provider nvidia --model nvidia/nemotron-3-super-120b-a12b
```

或在 `config.yaml` 中永久设置：
```yaml
model:
  provider: "nvidia"
  default: "nvidia/nemotron-3-super-120b-a12b"
```

:::tip 本地 NIM
对于本地部署（DGX Spark、本地 GPU），设置 `NVIDIA_BASE_URL=http://localhost:8000/v1`。NIM 暴露与 build.nvidia.com 相同的 OpenAI 兼容 chat completions API，因此云端和本地之间的切换只需更改一行环境变量。
:::

### GMI Cloud

通过 [GMI Cloud](https://inference.gmi.ai) 使用开放和推理模型 —— OpenAI 兼容 API，API 密钥认证。

```bash
# GMI Cloud
hermes chat --provider gmi --model deepseek-ai/DeepSeek-R1
# 需要：`~/.hermes/.env` 中的 GMI_API_KEY
```

或在 `config.yaml` 中永久设置：
```yaml
model:
  provider: "gmi"
  default: "deepseek-ai/DeepSeek-R1"
```

可以使用 `GMI_BASE_URL` 覆盖基础 URL（默认：`https://api.gmi.ai/v1`）。

### StepFun

通过 [StepFun](https://platform.stepfun.com) 使用 Step 系列模型 —— OpenAI 兼容 API，API 密钥认证。

```bash
# StepFun
hermes chat --provider stepfun --model step-3-mini
# 需要：`~/.hermes/.env` 中的 STEPFUN_API_KEY
```

或在 `config.yaml` 中永久设置：
```yaml
model:
  provider: "stepfun"
  default: "step-3-mini"
```

可以使用 `STEPFUN_BASE_URL` 覆盖基础 URL（默认：`https://api.stepfun.com/v1`）。

### Hugging Face 推理提供商

[Hugging Face 推理提供商](https://huggingface.co/docs/inference-providers) 通过统一 OpenAI 兼容端点（`router.huggingface.co/v1`）路由到 20+ 开放模型。请求自动路由到最快可用的后端（Groq、Together、SambaNova 等），并有自动故障转移。

```bash
# 使用任何可用模型
hermes chat --provider huggingface --model Qwen/Qwen3-235B-A22B-Thinking-2507
# 需要：`~/.hermes/.env` 中的 HF_TOKEN

# 短别名
hermes chat --provider hf --model deepseek-ai/DeepSeek-V3.2
```

或在 `config.yaml` 中永久设置：
```yaml
model:
  provider: "huggingface"
  default: "Qwen/Qwen3-235B-A22B-Thinking-2507"
```

在 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 获取你的 token —— 确保启用"Make calls to Inference Providers"权限。包含免费配额（$0.10/月信用额度， provider 费率无加价）。

你可以在模型名称后附加路由后缀：`:fastest`（默认）、`:cheapest` 或 `:provider_name` 以强制使用特定后端。

可以使用 `HF_BASE_URL` 覆盖基础 URL。

## 自定义和自托管 LLM 提供商

Hermes Agent 与**任何 OpenAI 兼容 API 端点**配合使用。如果服务器实现了 `/v1/chat/completions`，你可以将 Hermes 指向它。这意味着你可以使用本地模型、GPU 推理服务器、多提供商路由器或任何第三方 API。

### 通用设置

配置自定义端点的三种方式：

**交互式设置（推荐）：**
```bash
hermes model
# 选择"Custom endpoint (self-hosted / VLLM / etc.)"
# 输入：API 基础 URL、API 密钥、模型名称
```

**手动配置（`config.yaml`）：**
```yaml
# 在 ~/.hermes/config.yaml 中
model:
  default: your-model-name
  provider: custom
  base_url: http://localhost:8000/v1
  api_key: your-key-or-leave-empty-for-local
```

:::warning 旧版环境变量
`.env` 中的 `OPENAI_BASE_URL` 和 `LLM_MODEL` 已被**移除**。Hermes 的任何部分都不再读取它们 —— `config.yaml` 是模型和端点配置的单一事实来源。如果你 `.env` 中有陈旧条目，它们会在下次 `hermes setup` 或配置迁移时自动清除。请直接使用 `hermes model` 或编辑 `config.yaml`。
:::

两种方法都持久化到 `config.yaml`，它是模型、提供商和基础 URL 的事实来源。

### 使用 `/model` 切换模型

:::warning hermes model vs /model
**`hermes model`**（在聊天会话外的终端中运行）是**完整提供商设置向导**。用它添加新提供商、运行 OAuth 流程、输入 API 密钥和配置自定义端点。

**`/model`**（在活跃的 Hermes 聊天会话中输入）只能**在你已设置的提供商和模型之间切换**。它不能添加新提供商、运行 OAuth 或提示输入 API 密钥。如果你只配置了一个提供商（例如 OpenRouter），`/model` 将只显示该提供商的模型。

**要添加新提供商：** 退出你的会话（`Ctrl+C` 或 `/quit`），运行 `hermes model`，设置新提供商，然后开始新会话。
:::

配置了至少一个自定义端点后，你可以在会话中切换模型：

```
/model custom:qwen-2.5          # 切换到自定义端点上的模型
/model custom                    # 从端点自动检测模型
/model openrouter:claude-sonnet-4 # 切换回云提供商
```

如果你配置了**命名自定义提供商**（见下方），使用三段式语法：

```
/model custom:local:qwen-2.5    # 使用"local"自定义提供商和 qwen-2.5 模型
/model custom:work:llama3       # 使用"work"自定义提供商和 llama3 模型
```

切换提供商时，Hermes 将基础 URL 和提供商持久化到配置，以便更改在重启后保留。当从自定义端点切换到内置提供商时，陈旧的基础 URL 会自动清除。

:::tip
`/model custom`（裸，不带模型名）查询你端点的 `/models` API，如果恰好加载了一个模型则自动选择。适用于运行单个模型的本地服务器。
:::

以下所有内容遵循相同模式 —— 只需更改 URL、密钥和模型名称。

---

### Ollama —— 本地模型，零配置

[Ollama](https://ollama.com/) 用一条命令在本地运行开放权重模型。最适合：快速本地实验、隐私敏感工作、离线使用。支持通过 OpenAI 兼容 API 进行工具调用。

```bash
# 安装并运行模型
ollama pull qwen2.5-coder:32b
ollama serve   # 在端口 11434 启动
```

然后配置 Hermes：

```bash
hermes model
# 选择"Custom endpoint (self-hosted / VLLM / etc.)"
# 输入 URL: http://localhost:11434/v1
# 跳过 API 密钥（Ollama 不需要）
# 输入模型名称（例如 qwen2.5-coder:32b）
```

或直接在 `config.yaml` 中配置：

```yaml
model:
  default: qwen2.5-coder:32b
  provider: custom
  base_url: http://localhost:11434/v1
  context_length: 32768   # 见下方警告
```

:::caution Ollama 默认上下文长度非常低
Ollama **不**使用模型的全部上下文窗口。默认值取决于你的 VRAM：

| 可用 VRAM | 默认上下文 |
|----------------|----------------|
| 少于 24 GB | **4,096 tokens** |
| 24–48 GB | 32,768 tokens |
| 48+ GB | 256,000 tokens |

对于使用工具的 agent，**你需要至少 16k–32k 上下文**。在 4k 时，系统提示 + 工具模式本身就可以填满窗口，没有对话空间。

**如何增加**（任选其一）：

```bash
# 选项 1：通过环境变量设置服务器范围（推荐）
OLLAMA_CONTEXT_LENGTH=32768 ollama serve

# 选项 2：对于 systemd 管理的 Ollama
sudo systemctl edit ollama.service
# 添加：Environment="OLLAMA_CONTEXT_LENGTH=32768"
# 然后：sudo systemctl daemon-reload && sudo systemctl restart ollama

# 选项 3：烘焙到自定义模型（每个模型持久化）
echo -e "FROM qwen2.5-coder:32b\nPARAMETER num_ctx 32768" > Modelfile
ollama create qwen2.5-coder-32k -f Modelfile
```

**你无法通过 OpenAI 兼容 API**（`/v1/chat/completions`）设置上下文长度。必须通过 Modelfile 在服务器端或通过 Modelfile 配置。这是将 Ollama 与 Hermes 等工具集成时的 #1 困惑来源。
:::

**验证你的上下文设置正确：**

```bash
ollama ps
# 查看 CONTEXT 列 —— 应该显示你配置的值
```

:::tip
用 `ollama list` 列出可用模型。用 `ollama pull <model>` 从 [Ollama 库](https://ollama.com/library) 拉取任何模型。Ollama 自动处理 GPU 卸载 —— 大多数设置无需配置。
:::

---

### vLLM —— 高性能 GPU 推理

[vLLM](https://docs.vllm.ai/) 是生产 LLM 服务的标准。最适合：GPU 硬件上的最大吞吐量、服务大型模型、连续批处理。

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --port 8000 \
  --max-model-len 65536 \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

然后配置 Hermes：

```bash
hermes model
# 选择"Custom endpoint (self-hosted / VLLM / etc.)"
# 输入 URL: http://localhost:8000/v1
# 跳过 API 密钥（如果你用 --api-key 配置了 vLLM 则输入）
# 输入模型名称: meta-llama/Llama-3.1-70B-Instruct
```

**上下文长度：** vLLM 默认读取模型的 `max_position_embeddings`。如果超过你的 GPU 内存，它会报错并要求你降低 `--max-model-len`。你也可以使用 `--max-model-len auto` 自动找到适合的最大值。设置 `--gpu-memory-utilization 0.95`（默认 0.9）可以在 VRAM 中挤入更多上下文。

**工具调用需要显式标志：**

| 标志 | 用途 |
|------|---------|
| `--enable-auto-tool-choice` | `tool_choice: "auto"`（Hermes 默认值）所需 |
| `--tool-call-parser <name>` | 模型工具调用格式的解析器 |

支持的解析器：`hermes`（Qwen 2.5、Hermes 2/3）、`llama3_json`（Llama 3.x）、`mistral`、`deepseek_v3`、`deepseek_v31`、`xlam`、`pythonic`。没有这些标志，工具调用将不工作 —— 模型将把工具调用输出为文本。

:::tip
vLLM 支持人类可读的尺寸：`--max-model-len 64k`（小写 k = 1000，大写 K = 1024）。
:::

---

### SGLang —— 带 RadixAttention 的快速服务

[SGLang](https://github.com/sgl-project/sglang) 是 vLLM 的替代方案，具有 KV 缓存重用的 RadixAttention。最适合：多轮对话（前缀缓存）、约束解码、结构化输出。

```bash
pip install "sglang[all]"
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --port 30000 \
  --context-length 65536 \
  --tp 2 \
  --tool-call-parser qwen
```

然后配置 Hermes：

```bash
hermes model
# 选择"Custom endpoint (self-hosted / VLLM / etc.)"
# 输入 URL: http://localhost:30000/v1
# 输入模型名称: meta-llama/Llama-3.1-70B-Instruct
```

**上下文长度：** SGLang 默认从模型配置读取。使用 `--context-length` 覆盖。如果需要超过模型声明的最大值，设置 `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`。

**工具调用：** 使用 `--tool-call-parser` 以及适合你模型系列对应的解析器：`qwen`（Qwen 2.5）、`llama3`、`llama4`、`deepseekv3`、`mistral`、`glm`。没有此标志，工具调用会作为纯文本返回。

:::caution SGLang 默认 128 最大输出 tokens
如果响应似乎被截断，请在请求中添加 `max_tokens` 或在服务器上设置 `--default-max-tokens`。SGLang 的默认值仅为 128 tokens/响应（如果请求中未指定）。
:::

---

### llama.cpp / llama-server —— CPU 和 Metal 推理

[llama.cpp](https://github.com/ggml-org/llama.cpp) 在 CPU、Apple Silicon（Metal）和消费级 GPU 上运行量化模型。最适合：在没有数据中心 GPU 的情况下运行模型、Mac 用户、边缘部署。

```bash
# 构建并启动 llama-server
cmake -B build && cmake --build build --config Release
./build/bin/llama-server \
  --jinja -fa \
  -c 32768 \
  -ngl 99 \
  -m models/qwen2.5-coder-32b-instruct-Q4_K_M.gguf \
  --port 8080 --host 0.0.0.0
```

**上下文长度（`-c`）：** 最近的构建默认为 `0`，从 GGUF 元数据中读取模型的训练上下文。对于 128k+ 训练上下文的模型，这可能会 OOM（尝试分配完整 KV 缓存）。显式设置 `-c` 为你需要的大小（agent 使用 32k–64k 是好范围）。如果使用并行槽（`-np`），总上下文在槽之间分配 —— 使用 `-c 32768 -np 4`，每个槽仅获得 8k。

然后配置 Hermes 指向它：

```bash
hermes model
# 选择"Custom endpoint (self-hosted / VLLM / etc.)"
# 输入 URL: http://localhost:8080/v1
# 跳过 API 密钥（本地服务器不需要）
# 输入模型名称 —— 或留空以在只加载一个模型时自动检测
```

这将端点保存到 `config.yaml`，以便在会话之间持久化。

:::caution `--jinja` 对于工具调用是必需的
没有 `--jinja`，llama-server 完全忽略 `tools` 参数。模型会尝试通过在响应文本中写入 JSON 来调用工具，但 Hermes 不会将其识别为工具调用 —— 你会看到原始 JSON（如 `{"name": "web_search", ...}`）作为消息打印，而不是实际搜索。

原生工具调用支持（最佳性能）：Llama 3.x、Qwen 2.5（包括 Coder）、Hermes 2/3、Mistral、DeepSeek、Functionary。所有其他模型使用通用处理程序，可以工作但可能效率较低。详见 [llama.cpp function calling 文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)。

你可以检查 `http://localhost:8080/props` 验证工具支持是否激活 —— `chat_template` 字段应该存在。
:::

:::tip
从 [Hugging Face](https://huggingface.co/models?library=gguf) 下载 GGUF 模型。Q4_K_M 量化提供质量和内存使用的最佳平衡。
:::

---

### LM Studio —— 带本地模型的桌面应用

[LM Studio](https://lmstudio.ai/) 是一个用于运行带 GUI 本地模型的桌面应用。最适合：喜欢可视化界面的用户、快速模型测试、macOS/Windows/Linux 开发者。

从 LM Studio 应用启动服务器（Developer 选项卡 → Start Server），或使用 CLI：

```bash
lms server start                        # 在端口 1234 启动
lms load qwen2.5-coder --context-length 32768
```

然后配置 Hermes：

```bash
hermes model
# 选择"LM Studio"
# 按 Enter 使用 http://localhost:1234/v1
# 从发现的模型中选择
# 如果启用了 LM Studio 服务器认证，在提示时输入 LM_API_KEY
```

Hermes 将自动使用 64K 上下文长度加载 LM Studio 模型。

在 LM Studio 中更改上下文长度：

1. 点击模型选择器旁边的齿轮图标
2. 将"Context Length"设置为至少 64000 以获得流畅体验
3. 重新加载模型以使更改生效
4. 如果你的机器无法容纳 64000，考虑使用更小上下文长度的更大模型。

或者使用 CLI：`lms load model-name --context-length 64000`

你可以用 CLI 估计模型是否合适：`lms load model-name --context-length 64000 --estimate-only`

要设置每个模型的持久默认值：My Models 选项卡 → 模型上的齿轮图标 → 设置上下文大小。
:::

**工具调用：** 自 LM Studio 0.3.6 起支持。具有原生工具调用训练的模型（Qwen 2.5、Llama 3.x、Mistral、Hermes）会被自动检测并显示工具徽章。其他模型使用可能不太可靠的通用回退。

---

### WSL2 网络（Windows 用户）

由于 Hermes Agent 需要 Unix 环境，Windows 用户在 WSL2 内运行它。如果你的模型服务器（Ollama、LM Studio 等）运行在 **Windows 主机**上，你需要桥接网络差距 —— WSL2 使用虚拟网络适配器，有自己的子网，所以 WSL2 内的 `localhost` 指的是 Linux VM，**不是** Windows 主机。

:::tip 两者都在 WSL2 中？没问题的。
如果你的模型服务器也在 WSL2 内运行（vLLM、SGLang 和 llama-server 的常见情况），`localhost` 按预期工作 —— 它们共享同一网络命名空间。跳过本节。
:::

#### 选项 1：镜像网络模式（推荐）

在 **Windows 11 22H2+** 上可用，镜像模式使 `localhost` 在 Windows 和 WSL2 之间双向工作 —— 最简单的修复。

1. 创建或编辑 `%USERPROFILE%\.wslconfig`（例如 `C:\Users\YourName\.wslconfig`）：
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```

2. 从 PowerShell 重启 WSL：
   ```powershell
   wsl --shutdown
   ```

3. 重新打开 WSL2 终端。`localhost` 现在可以访问 Windows 服务：
   ```bash
   curl http://localhost:11434/v1/models   # Windows 上的 Ollama —— 工作正常
   ```

:::note Hyper-V 防火墙
在某些 Windows 11 版本上，Hyper-V 防火墙默认阻止镜像连接。如果启用镜像模式后 `localhost` 仍然不工作，请在**管理员 PowerShell** 中运行：
```powershell
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
```
:::

#### 选项 2：使用 Windows 主机 IP（Windows 10 / 旧版本）

如果你无法使用镜像模式，从 WSL2 内部找到 Windows 主机 IP 并用它代替 `localhost`：

```bash
# 获取 Windows 主机 IP（WSL2 虚拟网络的默认网关）
ip route show | grep -i default | awk '{ print $3 }'
# 示例输出：172.29.192.1
```

在你的 Hermes 配置中使用该 IP：

```yaml
model:
  default: qwen2.5-coder:32b
  provider: custom
  base_url: http://172.29.192.1:11434/v1   # Windows 主机 IP，不是 localhost
```

:::tip 动态助手
主机 IP 可能在 WSL2 重启时更改。你可以在 shell 中动态获取：
```bash
export WSL_HOST=$(ip route show | grep -i default | awk '{ print $3 }')
echo "Windows 主机在: $WSL_HOST"
curl http://$WSL_HOST:11434/v1/models   # 测试 Ollama
```

或者使用你机器的 mDNS 名称（WSL2 中需要 `libnss-mdns`）：
```bash
sudo apt install libnss-mdns
curl http://$(hostname).local:11434/v1/models
```
:::

#### 服务器绑定地址（NAT 模式必需）

如果你使用**选项 2**（带主机 IP 的 NAT 模式），Windows 上的模型服务器必须接受来自 `127.0.0.1` 外部的连接。默认情况下，大多数服务器只在 localhost 上监听 —— WSL2 连接在 NAT 模式下来自不同的虚拟子网，会被拒绝。在镜像模式下，`localhost` 直接映射，所以默认 `127.0.0.1` 绑定工作正常。

| 服务器 | 默认绑定 | 如何修复 |
|--------|-------------|------------|
| **Ollama** | `127.0.0.1` | 在启动 Ollama 前设置 `OLLAMA_HOST=0.0.0.0` 环境变量（Windows 系统设置 → 环境变量，或编辑 Ollama 服务） |
| **LM Studio** | `127.0.0.1` | 在 Developer 选项卡 → Server 设置中启用 **"Serve on Network"** |
| **llama-server** | `127.0.0.1` | 在启动命令中添加 `--host 0.0.0.0` |
| **vLLM** | `0.0.0.0` | 默认已绑定到所有接口 |
| **SGLang** | `127.0.0.1` | 在启动命令中添加 `--host 0.0.0.0` |

**Windows 上的 Ollama（详细）：** Ollama 作为 Windows 服务运行。设置 `OLLAMA_HOST`：
1. 打开**系统属性** → **环境变量**
2. 添加新的**系统变量**：`OLLAMA_HOST` = `0.0.0.0`
3. 重启 Ollama 服务（或重启）

#### Windows 防火墙

Windows 防火墙将 WSL2 视为单独的网络（在 NAT 和镜像模式下都是如此）。如果执行上述步骤后连接仍然失败，为你的模型服务器端口添加防火墙规则：

```powershell
# 在管理员 PowerShell 中运行 —— 将 PORT 替换为你的服务器端口
New-NetFirewallRule -DisplayName "Allow WSL2 to Model Server" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11434
```

常用端口：Ollama `11434`、vLLM `8000`、SGLang `30000`、llama-server `8080`、LM Studio `1234`。

#### 快速验证

从 WSL2 内部，测试是否可以到达你的模型服务器：

```bash
# 将 URL 替换为你的服务器地址和端口
curl http://localhost:11434/v1/models          # 镜像模式
curl http://172.29.192.1:11434/v1/models       # NAT 模式（使用你的实际主机 IP）
```

如果你得到列出模型的 JSON 响应，你就可以开始了。将相同的 URL 用作 Hermes 配置中的 `base_url`。

---

### 本地模型故障排除

这些问题影响**所有**与 Hermes 一起使用的本地推理服务器。

#### WSL2 到 Windows 托管模型服务器的"连接被拒绝"

如果你在 WSL2 内运行 Hermes 而模型服务器在 Windows 主机上，`http://localhost:<port>` 在 WSL2 的默认 NAT 网络模式下不工作。见上方 [WSL2 网络](#wsl2-networking-windows-users) 的修复。

#### 工具调用显示为文本而不是执行

模型输出类似 `{"name": "web_search", "arguments": {...}}` 的内容作为消息，而不是实际调用工具。

**原因：** 你的服务器未启用工具调用，或模型不支持服务器的 tool calling 实现。

| 服务器 | 修复 |
|--------|---------|
| **llama.cpp** | 在启动命令中添加 `--jinja` |
| **vLLM** | 添加 `--enable-auto-tool-choice --tool-call-parser hermes` |
| **SGLang** | 添加 `--tool-call-parser qwen`（或适当的解析器） |
| **Ollama** | 默认启用工具调用 —— 确保你的模型支持（用 `ollama show model-name` 检查） |
| **LM Studio** | 更新到 0.3.6+ 并使用具有原生工具支持的模型 |

#### 模型似乎忘记上下文或给出不连贯的响应

**原因：** 上下文窗口太小。当对话超过上下文限制时，大多数服务器会静默丢弃旧消息。Hermes 的系统提示 + 工具模式本身可以使用 4k–8k tokens。

**诊断：**

```bash
# 检查 Hermes 认为的上下文是什么
# 查看启动行："Context limit: X tokens"

# 检查你服务器的实际上下文
# Ollama: ollama ps（CONTEXT 列）
# llama.cpp: curl http://localhost:8080/props | jq '.default_generation_settings.n_ctx'
# vLLM: 检查启动参数中的 --max-model-len
```

**修复：** 为 agent 使用设置上下文至少 **32,768 tokens**。见上方每个服务器部分的具体标志。

#### 启动时显示"Context limit: 2048 tokens"

Hermes 从服务器的 `/v1/models` 端点自动检测上下文长度。如果服务器报告的值很低（或根本不报告），Hermes 使用模型声明的限制，可能不正确。

**修复：** 在 `config.yaml` 中显式设置：

```yaml
model:
  default: your-model
  provider: custom
  base_url: http://localhost:11434/v1
  context_length: 32768
```

#### 响应在句子中间被截断

**可能原因：**
1. **服务器上的输出上限（`max_tokens`）太低** —— SGLang 默认每响应 128 tokens。在服务器上设置 `--default-max-tokens` 或在 config.yaml 中用 `model.max_tokens` 配置 Hermes。注意：`max_tokens` 仅控制响应长度 —— 与对话历史可以多长无关（那是 `context_length`）。
2. **上下文耗尽** —— 模型填满了其上下文窗口。增加 `model.context_length` 或在 Hermes 中启用[上下文压缩](/docs/user-guide/configuration#context-compression)。

---

### LiteLLM Proxy —— 多提供商网关

[LiteLLM](https://docs.litellm.ai/) 是一个 OpenAI 兼容代理，在单一 API 后统一 100+ LLM 提供商。最适合：在配置不变的情况下在提供商之间切换、负载均衡、回退链、预算控制。

```bash
# 安装并启动
pip install "litellm[proxy]"
litellm --model anthropic/claude-sonnet-4 --port 4000

# 或使用配置文件处理多个模型：
litellm --config litellm_config.yaml --port 4000
```

然后用 `hermes model` → Custom endpoint → `http://localhost:4000/v1` 配置 Hermes。

带回退的 `litellm_config.yaml` 示例：
```yaml
model_list:
  - model_name: "best"
    litellm_params:
      model: anthropic/claude-sonnet-4
      api_key: sk-ant-...
  - model_name: "best"
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-...
router_settings:
  routing_strategy: "latency-based-routing"
```

---

### ClawRouter —— 成本优化路由

BlockRunAI 的 [ClawRouter](https://github.com/BlockRunAI/ClawRouter) 是一个本地路由代理，根据查询复杂度自动选择模型。它在 14 个维度上对请求进行分类，并路由到可以处理任务的最便宜模型。通过 USDC 加密货币支付（无需 API 密钥）。

```bash
# 安装并启动
npx @blockrun/clawrouter    # 在端口 8402 启动
```

然后用 `hermes model` → Custom endpoint → `http://localhost:8402/v1` → 模型名称 `blockrun/auto` 配置 Hermes。

路由配置文件：
| 配置文件 | 策略 | 节省 |
|---------|----------|-------|
| `blockrun/auto` | 平衡质量/成本 | 74-100% |
| `blockrun/eco` | 最便宜可能 | 95-100% |
| `blockrun/premium` | 最佳质量模型 | 0% |
| `blockrun/free` | 仅免费模型 | 100% |
| `blockrun/agentic` | 为工具使用优化 | 视情况 |

:::note
ClawRouter 需要在 Base 或 Solana 上有 USDC 资助的钱包进行支付。所有请求通过 BlockRun 的后端 API 路由。运行 `npx @blockrun/clawrouter doctor` 检查钱包状态。
:::

---

### 其他兼容提供商

任何具有 OpenAI 兼容 API 的服务都可以使用。一些流行选项：

| 提供商 | 基础 URL | 说明 |
|----------|----------|-------|
| [Together AI](https://together.ai) | `https://api.together.xyz/v1` | 云托管开放模型 |
| [Groq](https://groq.com) | `https://api.groq.com/openai/v1` | 超快推理 |
| [DeepSeek](https://deepseek.com) | `https://api.deepseek.com/v1` | DeepSeek 模型 |
| [Fireworks AI](https://fireworks.ai) | `https://api.fireworks.ai/inference/v1` | 快速开放模型托管 |
| [GMI Cloud](https://www.gmicloud.ai/) | `https://api.gmi-serving.com/v1` | 托管 OpenAI 兼容推理 |
| [Cerebras](https://cerebras.ai) | `https://api.cerebras.ai/v1` | 晶圆级芯片推理 |
| [Mistral AI](https://mistral.ai) | `https://api.mistral.ai/v1` | Mistral 模型 |
| [OpenAI](https://openai.com) | `https://api.openai.com/v1` | 直接 OpenAI 访问 |
| [Azure OpenAI](https://azure.microsoft.com) | `https://YOUR.openai.azure.com/` | 企业 OpenAI |
| [LocalAI](https://localai.io) | `http://localhost:8080/v1` | 自托管、多模型 |
| [Jan](https://jan.ai) | `http://localhost:1337/v1` | 带本地模型的桌面应用 |

用 `hermes model` → Custom endpoint 或在 `config.yaml` 中配置其中任何一个：

```yaml
model:
  default: meta-llama/Llama-3.1-70B-Instruct-Turbo
  provider: custom
  base_url: https://api.together.xyz/v1
  api_key: your-together-key
```

---

### 上下文长度检测

:::note 两个容易混淆的设置
**`context_length`** 是**总上下文窗口** —— 输入和输出 tokens 的总预算（例如 Claude Opus 4.6 为 200,000）。Hermes 用它来决定何时压缩历史并验证 API 请求。

**`model.max_tokens`** 是**输出上限** —— 模型可以在*单个响应*中生成的最大 tokens 数。它与对话历史可以多长无关。行业标准名称 `max_tokens` 是常见混淆来源；Anthropic 的原生 API 已将其重命名为 `max_output_tokens` 以明确。

当自动检测窗口大小时出错时设置 `context_length`。
只有当你需要限制单个响应的长度时才设置 `model.max_tokens`。
:::

Hermes 使用多源解析链来检测模型和提供商的正确上下文窗口：

1. **配置覆盖** —— `config.yaml` 中的 `model.context_length`（最高优先级）
2. **自定义提供商每个模型** —— `custom_providers[].models.<id>.context_length`
3. **持久缓存** —— 之前发现的值（跨重启保留）
4. **端点 `/models`** —— 查询你服务器的 API（本地/自定义端点）
5. **Anthropic `/v1/models`** —— 查询 Anthropic API 的 `max_input_tokens`（仅 API 密钥用户）
6. **OpenRouter API** —— 来自 OpenRouter 的实时模型元数据
7. **Nous Portal** —— 将 Nous 模型 ID 后缀匹配到 OpenRouter 元数据
8. **[models.dev](https://models.dev)** —— 社区维护的注册表，包含 100+ 提供商上 3800+ 模型特定上下文长度
9. **回退默认值** —— 广泛的模型系列模式（128K 默认）

对于大多数设置，这开箱即用。该系统具有提供商感知能力 —— 相同模型可以根据服务提供商有不同的上下文限制（例如，`claude-opus-4.6` 在 Anthropic 直连为 1M，但在 GitHub Copilot 上为 128K）。

要显式设置上下文长度，将 `context_length` 添加到你的模型配置：

```yaml
model:
  default: "qwen3.5:9b"
  base_url: "http://localhost:8080/v1"
  context_length: 131072  # tokens
```

对于自定义端点，你也可以按模型设置上下文长度：

```yaml
custom_providers:
  - name: "My Local LLM"
    base_url: "http://localhost:11434/v1"
    models:
      qwen3.5:27b:
        context_length: 32768
      deepseek-r1:70b:
        context_length: 65536
```

`hermes model` 在配置自定义端点时会提示输入上下文长度。留空以进行自动检测。

:::tip 何时手动设置
- 你正在使用自定义 `num_ctx` 低于模型最大值的 Ollama
- 你想将上下文限制在模型最大值以下（例如，在 128k 模型上使用 8k 以节省 VRAM）
- 你在不支持 `/v1/models` 的代理后面运行
:::

---

### 命名自定义提供商

如果你使用多个自定义端点（例如，本地开发服务器和远程 GPU 服务器），你可以在 `config.yaml` 中将它们定义为命名自定义提供商：

```yaml
custom_providers:
  - name: local
    base_url: http://localhost:8080/v1
    # api_key 省略 —— Hermes 对无密钥本地服务器使用"no-key-required"
  - name: work
    base_url: https://gpu-server.internal.corp/v1
    key_env: CORP_API_KEY
    api_mode: chat_completions   # 可选，从 URL 自动检测
  - name: anthropic-proxy
    base_url: https://proxy.example.com/anthropic
    key_env: ANTHROPIC_PROXY_KEY
    api_mode: anthropic_messages  # 用于 Anthropic 兼容代理
```

用三段式语法在会话中切换：

```
/model custom:local:qwen-2.5       # 使用"local"端点和 qwen-2.5 模型
/model custom:work:llama3-70b      # 使用"work"端点和 llama3-70b 模型
/model custom:anthropic-proxy:claude-sonnet-4  # 使用代理
```

你也可以从交互式 `hermes model` 菜单中选择命名自定义提供商。

---

###  Cookbook：Together AI、Groq、Perplexity

[其他兼容提供商](#other-compatible-providers) 中列出的云提供商都使用 OpenAI REST 方言，因此在 `custom_providers:` 下的连接方式相同。以下是三个实用示例。每个放入 `~/.hermes/config.yaml`，匹配的 API 密钥放入 `~/.hermes/.env`。

#### Together AI

以明显低于第一方 API 的价格托管开放权重模型（Llama、MiniMax、Gemma、DeepSeek、Qwen）。多模型集群的良好默认值。

```yaml
# ~/.hermes/config.yaml
custom_providers:
  - name: together
    base_url: https://api.together.xyz/v1
    key_env: TOGETHER_API_KEY
    # api_mode: chat_completions  # 默认 —— 无需设置

model:
  default: MiniMaxAI/MiniMax-M2.7   # 或 together.ai/models 上的任何模型
  provider: custom:together
```

```bash
# ~/.hermes/.env
TOGETHER_API_KEY=your-together-key
```

在会话中切换模型：

```
/model custom:together:meta-llama/Llama-3.3-70B-Instruct-Turbo
/model custom:together:google/gemma-4-31b-it
/model custom:together:deepseek-ai/DeepSeek-V3
```

Together 的 `/v1/models` 端点可用，因此 `hermes model` 可以自动发现可用模型。

#### Groq

超快推理（Llama-3.3-70B 约 500 tok/s）。目录较小，但对延迟敏感的交互使用效果出色。

```yaml
# ~/.hermes/config.yaml
custom_providers:
  - name: groq
    base_url: https://api.groq.com/openai/v1
    key_env: GROQ_API_KEY

model:
  default: llama-3.3-70b-versatile
  provider: custom:groq
```

```bash
# ~/.hermes/.env
GROQ_API_KEY=your-groq-key
```

#### Perplexity

当你想要模型自动进行实时网络搜索和引文时很有用。对可用模型有严格限制 —— 查看 [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api) 了解当前列表。

```yaml
# ~/.hermes/config.yaml
custom_providers:
  - name: perplexity
    base_url: https://api.perplexity.ai
    key_env: PERPLEXITY_API_KEY

model:
  default: sonar
  provider: custom:perplexity
```

```bash
# ~/.hermes/.env
PERPLEXITY_API_KEY=your-perplexity-key
```

#### 在一个配置中组合多个提供商

这三个示例可以组合 —— 一起使用，用 `/model custom:<name>:<model>` 逐轮切换：

```yaml
custom_providers:
  - name: together
    base_url: https://api.together.xyz/v1
    key_env: TOGETHER_API_KEY
  - name: groq
    base_url: https://api.groq.com/openai/v1
    key_env: GROQ_API_KEY
  - name: perplexity
    base_url: https://api.perplexity.ai
    key_env: PERPLEXITY_API_KEY

model:
  default: MiniMaxAI/MiniMax-M2.7
  provider: custom:together      # 启动到 Together；之后自由切换
```

:::tip 故障排除
- 在 CLI 验证器修复 #15083 后，`hermes doctor` 应该不会为这些名称打印任何 `Unknown provider` 警告。
- 如果提供商的 `/v1/models` 端点不可达（Perplexity 是常见情况），`hermes model` 会持久化模型并显示警告而不是硬拒绝 —— 见 #15136。
- 要完全跳过 `custom_providers:` 并使用裸 `provider: custom` 配合 `CUSTOM_BASE_URL` 环境变量，见 #15103。
:::

---

### 选择正确的设置

| 使用场景 | 推荐 |
|----------|-------------|
| **只是想让它工作** | OpenRouter（默认）或 Nous Portal |
| **本地模型，易于设置** | Ollama |
| **生产 GPU 服务** | vLLM 或 SGLang |
| **Mac / 无 GPU** | Ollama 或 llama.cpp |
| **多提供商路由** | LiteLLM Proxy 或 OpenRouter |
| **成本优化** | ClawRouter 或带有 `sort: "price"` 的 OpenRouter |
| **最大隐私** | Ollama、vLLM 或 llama.cpp（完全本地） |
| **企业 / Azure** | 带有自定义端点的 Azure OpenAI |
| **中文 AI 模型** | z.ai (GLM)、Kimi/Moonshot（`kimi-coding` 或 `kimi-coding-cn`）、MiniMax、小米 MiMo 或腾讯 TokenHub（一级提供商） |

:::tip
你可以随时使用 `hermes model` 在提供商之间切换 —— 无需重新启动。你的对话历史、记忆和技能会自动延续，无论你使用哪个提供商。
:::

## 可选 API 密钥

| 功能 | 提供商 | 环境变量 |
|---------|----------|--------------|
| 网页抓取 | [Firecrawl](https://firecrawl.dev/) | `FIRECRAWL_API_KEY`、`FIRECRAWL_API_URL` |
| 浏览器自动化 | [Browserbase](https://browserbase.com/) | `BROWSERBASE_API_KEY`、`BROWSERBASE_PROJECT_ID` |
| 图像生成 | [FAL](https://fal.ai/) | `FAL_KEY` |
| 高级 TTS 语音 | [ElevenLabs](https://elevenlabs.io/) | `ELEVENLABS_API_KEY` |
| OpenAI TTS + 语音转录 | [OpenAI](https://platform.openai.com/api-keys) | `VOICE_TOOLS_OPENAI_KEY` |
| Mistral TTS + 语音转录 | [Mistral](https://console.mistral.ai/) | `MISTRAL_API_KEY` |
| RL 训练 | [Tinker](https://tinker-console.thinkingmachines.ai/) + [WandB](https://wandb.ai/) | `TINKER_API_KEY`、`WANDB_API_KEY` |
| 跨会话用户建模 | [Honcho](https://honcho.dev/) | `HONCHO_API_KEY` |
| 语义长期记忆 | [Supermemory](https://supermemory.ai) | `SUPERMEMORY_API_KEY` |

### 自托管 Firecrawl

默认情况下，Hermes 使用 [Firecrawl 云 API](https://firecrawl.dev/) 进行网页搜索和抓取。如果你更喜欢在本地运行 Firecrawl，可以将 Hermes 指向自托管的实例。有关完整的设置说明，请参阅 Firecrawl 的 [SELF_HOST.md](https://github.com/firecrawl/firecrawl/blob/main/SELF_HOST.md)。

**你将获得：** 无需 API 密钥、无速率限制、无按页费用、完全数据主权。

**你将失去：** 云版本使用 Firecrawl 专有的"Fire-engine"进行高级反机器人绕过（Cloudflare、CAPTCHAs、IP 轮换）。自托管使用基础 fetch + Playwright，因此某些受保护的站点可能会失败。搜索使用 DuckDuckGo 而不是 Google。

**设置：**

1. 克隆并启动 Firecrawl Docker 栈（5 个容器：API、Playwright、Redis、RabbitMQ、PostgreSQL —— 需要约 4-8 GB RAM）：
   ```bash
   git clone https://github.com/firecrawl/firecrawl
   cd firecrawl
   # 在 .env 中，设置：USE_DB_AUTHENTICATION=false, HOST=0.0.0.0, PORT=3002
   docker compose up -d
   ```

2. 将 Hermes 指向你的实例（无需 API 密钥）：
   ```bash
   hermes config set FIRECRAWL_API_URL http://localhost:3002
   ```

你也可以同时设置 `FIRECRAWL_API_KEY` 和 `FIRECRAWL_API_URL`（如果你的自托管实例启用了身份验证）。

## OpenRouter 提供商路由

使用 OpenRouter 时，你可以控制请求在提供商之间的路由方式。在 `~/.hermes/config.yaml` 中添加 `provider_routing` 部分：

```yaml
provider_routing:
  sort: "throughput"          # "price"（默认）、"throughput" 或 "latency"
  # only: ["anthropic"]      # 仅使用这些提供商
  # ignore: ["deepinfra"]    # 跳过这些提供商
  # order: ["anthropic", "google"]  # 按此顺序尝试提供商
  # require_parameters: true  # 仅使用支持所有请求参数的提供商
  # data_collection: "deny"   # 排除可能存储/训练数据的提供商
```

**快捷方式：** 在任何模型名称后附加 `:nitro` 以进行吞吐量排序（例如 `anthropic/claude-sonnet-4:nitro`），或附加 `:floor` 以进行价格排序。

## 备用模型

配置备用 provider:model，当你的主模型失败（速率限制、服务器错误、身份验证失败）时，Hermes 会自动切换到它：

```yaml
fallback_model:
  provider: openrouter                    # 必需
  model: anthropic/claude-sonnet-4        # 必需
  # base_url: http://localhost:8000/v1    # 可选，用于自定义端点
  # key_env: MY_CUSTOM_KEY               # 可选，自定义端点 API 密钥的环境变量名
```

激活后，备用方案会在会话中途切换模型和提供商，而不会丢失你的对话。它在**每个会话中最多触发一次**。

支持的提供商：`openrouter`、`nous`、`openai-codex`、`copilot`、`copilot-acp`、`anthropic`、`gemini`、`google-gemini-cli`、`qwen-oauth`、`huggingface`、`zai`、`kimi-coding`、`kimi-coding-cn`、`minimax`、`minimax-cn`、`minimax-oauth`、`deepseek`、`nvidia`、`xai`、`ollama-cloud`、`bedrock`、`ai-gateway`、`opencode-zen`、`opencode-go`、`kilocode`、`xiaomi`、`arcee`、`gmi`、`stepfun`、`alibaba`、`tencent-tokenhub`、`custom`。

:::tip
备用方案仅通过 `config.yaml` 配置 —— 没有用于它的环境变量。有关触发条件、支持的提供商以及它与辅助任务和委托的交互方式的完整详细信息，请参阅[备用提供商](/docs/user-guide/features/fallback-providers)。
:::

---

## 另请参阅

- [配置](/docs/user-guide/configuration) —— 常规配置（目录结构、配置优先级、终端后端、记忆、压缩等）
- [环境变量](/docs/reference/environment-variables) —— 所有环境变量的完整参考
