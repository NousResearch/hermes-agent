---
title: "AI 提供商"
sidebar_label: "AI 提供商"
sidebar_position: 1
---

# AI 提供商

本页介绍如何为 Hermes Agent 配置推理提供商——从 OpenRouter、Anthropic 等云端 API，到 Ollama、vLLM 等自托管端点，再到高级路由与故障转移配置。使用 Hermes 至少需要配置一个提供商。

## 推理提供商

你需要至少一种方式连接到 LLM。使用 `hermes model` 交互式切换提供商和模型，或直接配置：

| 提供商 | 配置方式 |
|----------|-------|
| **Nous Portal** | `hermes model`（OAuth，订阅制） |
| **OpenAI Codex** | `hermes model`（ChatGPT OAuth，使用 Codex 模型） |
| **GitHub Copilot** | `hermes model`（OAuth 设备码流程，`COPILOT_GITHUB_TOKEN`、`GH_TOKEN` 或 `gh auth token`） |
| **GitHub Copilot ACP** | `hermes model`（在本地生成 `copilot --acp --stdio` 子进程） |
| **Anthropic** | `hermes model`（Claude Max + 额外用量积分，通过 OAuth；也支持 Anthropic API key 或手动 setup-token——见下方说明） |
| **OpenRouter** | `~/.hermes/.env` 中的 `OPENROUTER_API_KEY` |
| **NovitaAI** | `~/.hermes/.env` 中的 `NOVITA_API_KEY`（provider: `novita`，200+ 模型，Model API、Agent Sandbox、GPU Cloud） |
| **AI Gateway** | `~/.hermes/.env` 中的 `AI_GATEWAY_API_KEY`（provider: `ai-gateway`） |
| **z.ai / GLM** | `~/.hermes/.env` 中的 `GLM_API_KEY`（provider: `zai`） |
| **Kimi / Moonshot** | `~/.hermes/.env` 中的 `KIMI_API_KEY`（provider: `kimi-coding`） |
| **Kimi / Moonshot（中国）** | `~/.hermes/.env` 中的 `KIMI_CN_API_KEY`（provider: `kimi-coding-cn`；别名：`kimi-cn`、`moonshot-cn`） |
| **Arcee AI** | `~/.hermes/.env` 中的 `ARCEEAI_API_KEY`（provider: `arcee`；别名：`arcee-ai`、`arceeai`） |
| **GMI Cloud** | `~/.hermes/.env` 中的 `GMI_API_KEY`（provider: `gmi`；别名：`gmi-cloud`、`gmicloud`） |
| **MiniMax** | `~/.hermes/.env` 中的 `MINIMAX_API_KEY`（provider: `minimax`） |
| **MiniMax 中国** | `~/.hermes/.env` 中的 `MINIMAX_CN_API_KEY`（provider: `minimax-cn`） |
| **xAI（Grok）— Responses API** | `~/.hermes/.env` 中的 `XAI_API_KEY`（provider: `xai`） |
| **xAI Grok OAuth（SuperGrok）** | `hermes model` → "xAI Grok OAuth (SuperGrok / Premium+)"——浏览器登录，无需 API key。参见[指南](../guides/xai-grok-oauth.md) |
| **Qwen Cloud（阿里 DashScope）** | `~/.hermes/.env` 中的 `DASHSCOPE_API_KEY`（provider: `alibaba`） |
| **阿里云（Coding Plan）** | `DASHSCOPE_API_KEY`（provider: `alibaba-coding-plan`，别名：`alibaba_coding`）——独立计费 SKU，不同端点 |
| **Kilo Code** | `~/.hermes/.env` 中的 `KILOCODE_API_KEY`（provider: `kilocode`） |
| **小米 MiMo** | `~/.hermes/.env` 中的 `XIAOMI_API_KEY`（provider: `xiaomi`，别名：`mimo`、`xiaomi-mimo`） |
| **腾讯 TokenHub** | `~/.hermes/.env` 中的 `TOKENHUB_API_KEY`（provider: `tencent-tokenhub`，别名：`tencent`、`tokenhub`、`tencentmaas`） |
| **OpenCode Zen** | `~/.hermes/.env` 中的 `OPENCODE_ZEN_API_KEY`（provider: `opencode-zen`） |
| **OpenCode Go** | `~/.hermes/.env` 中的 `OPENCODE_GO_API_KEY`（provider: `opencode-go`） |
| **DeepSeek** | `~/.hermes/.env` 中的 `DEEPSEEK_API_KEY`（provider: `deepseek`） |
| **Hugging Face** | `~/.hermes/.env` 中的 `HF_TOKEN`（provider: `huggingface`，别名：`hf`） |
| **Google / Gemini** | `~/.hermes/.env` 中的 `GOOGLE_API_KEY`（或 `GEMINI_API_KEY`）（provider: `gemini`） |
| **Google Gemini（OAuth）** | `hermes model` → "Google Gemini (OAuth)"（provider: `google-gemini-cli`，支持免费层，浏览器 PKCE 登录） |
| **LM Studio** | `hermes model` → "LM Studio"（provider: `lmstudio`，可选 `LM_API_KEY`） |
| **自定义端点** | `hermes model` → 选择"Custom endpoint"（保存在 `config.yaml`） |

官方 API key 路径请参见专属的 [Google Gemini 指南](/guides/google-gemini)。

:::tip 模型 key 别名
在 `model:` 配置节中，可以使用 `default:` 或 `model:` 作为模型 ID 的键名。`model: { default: my-model }` 和 `model: { model: my-model }` 效果完全相同。
:::


### Nous Portal

[Nous Portal](https://portal.nousresearch.com) 是 Nous Research 的统一订阅网关，也是**运行 Hermes Agent 的推荐方式**。一次 OAuth 登录即可访问 300+ 前沿智能体模型（Claude、GPT、Gemini、DeepSeek、Qwen、Kimi、GLM、MiniMax、Grok 等），以及 [Tool Gateway](/user-guide/features/tool-gateway)（网页搜索、图像生成、TTS、浏览器自动化）和 [Nous Chat](https://chat.nousresearch.com)——费用从你的 Nous 订阅中扣除，无需单独管理各提供商账户。

```bash
hermes setup --portal     # 全新安装——一条命令完成 OAuth + 提供商 + 网关配置
hermes model              # 已有安装——从列表中选择"Nous Portal"
hermes portal status      # 随时查看登录状态和路由信息
```

还没有订阅？前往 [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription) 购买。

**完整详情：** 参见专属的 [Nous Portal 集成页面](/integrations/nous-portal)（订阅内容、模型目录、故障排查）以及分步指南[使用 Nous Portal 运行 Hermes Agent](/guides/run-hermes-with-nous-portal)。


:::info Codex 说明
OpenAI Codex 提供商通过设备码（device code）认证——打开一个 URL 并输入验证码。Hermes 将生成的凭据存储在 `~/.hermes/auth.json` 的自有认证存储中，并在存在 `~/.codex/auth.json` 时可导入现有的 Codex CLI 凭据。无需安装 Codex CLI。

如果 token 刷新因终端错误（HTTP 4xx、`invalid_grant`、授权被撤销等）失败，Hermes 会将该刷新 token 标记为失效并停止重试，避免出现大量重复的认证失败。下一次请求会显示类型化的重新认证提示。运行 `hermes auth add codex-oauth`（或 `hermes model` → OpenAI Codex）开始新的设备码登录；成功交换后隔离状态自动解除。
:::

:::warning
即使使用 Nous Portal、Codex 或自定义端点，某些工具（视觉、网页摘要、MoA）仍会使用单独的"辅助"模型。默认情况下（`auxiliary.*.provider: "auto"`），Hermes 将这些任务路由到你的**主聊天模型**——即你在 `hermes model` 中选择的同一模型。你可以单独覆盖每个任务，将其路由到更便宜/更快的模型（例如 OpenRouter 上的 Gemini Flash）——参见[辅助模型](/user-guide/configuration#auxiliary-models)。
:::

:::tip Nous Tool Gateway
付费 Nous Portal 订阅者还可访问 **[Tool Gateway](/user-guide/features/tool-gateway)**——网页搜索、图像生成、TTS 和浏览器自动化，均通过你的订阅路由。无需额外 API key。全新安装时，`hermes setup --portal` 一条命令即可完成登录、设置 Nous 为提供商并开启网关。现有用户可通过 `hermes model` 或 `hermes tools` 按工具启用。随时使用 `hermes portal status` 查看路由状态。
:::

### 模型管理的两个命令

Hermes 有**两个**模型命令，用途不同：

| 命令 | 运行位置 | 功能 |
|---------|-------------|--------------|
| **`hermes model`** | 终端（任何会话之外） | 完整配置向导——添加提供商、运行 OAuth、输入 API key、配置端点 |
| **`/model`** | Hermes 聊天会话内部 | 在**已配置的**提供商和模型之间快速切换 |

如果你想切换到尚未配置的提供商（例如你只配置了 OpenRouter，想使用 Anthropic），需要使用 `hermes model`，而不是 `/model`。先退出会话（`Ctrl+C` 或 `/quit`），运行 `hermes model`，完成提供商配置，然后开启新会话。


### Anthropic（原生）

通过 Anthropic API 直接使用 Claude 模型——无需 OpenRouter 代理。支持三种认证方式：

:::caution 需要 Claude Max"额外用量"积分
通过 `hermes model` → Anthropic OAuth（或 `hermes auth add anthropic --type oauth`）认证时，Hermes 以 Claude Code 身份路由到你的 Anthropic 账户。**仅当你订阅了 Claude Max 计划且购买了额外用量积分时才有效。** Claude Max 基础计划的配额（Claude Code 默认包含的用量）不会被 Hermes 消耗——只有你额外购买的超额积分才会被使用。Claude Pro 订阅者无法使用此路径。

如果你没有 Max + 额外积分，请改用 `ANTHROPIC_API_KEY`——请求将按 token 计费，从该 key 所属组织扣费（标准 API 定价，与任何 Claude 订阅无关）。
:::

```bash
# 使用 API key（按 token 计费）
export ANTHROPIC_API_KEY=***
hermes chat --provider anthropic --model claude-sonnet-4-6

# 推荐：通过 `hermes model` 认证
# 如果已使用 Claude Code，Hermes 会直接使用其凭据存储
hermes model

# 使用 setup-token 手动覆盖（备用/旧版）
export ANTHROPIC_TOKEN=***  # setup-token 或手动 OAuth token
hermes chat --provider anthropic

# 自动检测 Claude Code 凭据（如果你已使用 Claude Code）
hermes chat --provider anthropic  # 自动读取 Claude Code 凭据文件
```

通过 `hermes model` 选择 Anthropic OAuth 时，Hermes 优先使用 Claude Code 自身的凭据存储，而不是将 token 复制到 `~/.hermes/.env`。这样可以保持 Claude 凭据的可刷新性。

或永久设置：
```yaml
model:
  provider: "anthropic"
  default: "claude-sonnet-4-6"
```

:::tip 别名
`--provider claude` 和 `--provider claude-code` 也可作为 `--provider anthropic` 的简写。
:::

### GitHub Copilot

Hermes 以一等提供商身份支持 GitHub Copilot，提供两种模式：

**`copilot` — 直连 Copilot API**（推荐）。使用你的 GitHub Copilot 订阅，通过 Copilot API 访问 GPT-5.x、Claude、Gemini 等模型。

```bash
hermes chat --provider copilot --model gpt-5.4
```

**认证选项**（按以下顺序检查）：

1. `COPILOT_GITHUB_TOKEN` 环境变量
2. `GH_TOKEN` 环境变量
3. `GITHUB_TOKEN` 环境变量
4. `gh auth token` CLI 回退

如果未找到 token，`hermes model` 会提供 **OAuth 设备码登录**——与 Copilot CLI 和 opencode 使用的流程相同。

:::warning Token 类型
Copilot API **不**支持经典个人访问 token（`ghp_*`）。支持的 token 类型：

| 类型 | 前缀 | 获取方式 |
|------|--------|------------|
| OAuth token | `gho_` | `hermes model` → GitHub Copilot → 使用 GitHub 登录 |
| 细粒度 PAT | `github_pat_` | GitHub 设置 → 开发者设置 → 细粒度 token（需要 **Copilot Requests** 权限） |
| GitHub App token | `ghu_` | 通过 GitHub App 安装获取 |

如果你的 `gh auth token` 返回 `ghp_*` token，请使用 `hermes model` 通过 OAuth 认证。
:::

:::info Hermes 中的 Copilot 认证行为
Hermes 将支持的 GitHub token（`gho_*`、`github_pat_*` 或 `ghu_*`）直接发送到 `api.githubcopilot.com`，并附带 Copilot 专用请求头（`Editor-Version`、`Copilot-Integration-Id`、`Openai-Intent`、`x-initiator`）。

收到 HTTP 401 时，Hermes 在回退前会执行一次性凭据恢复：

1. 通过正常优先级链重新解析 token（`COPILOT_GITHUB_TOKEN` → `GH_TOKEN` → `GITHUB_TOKEN` → `gh auth token`）
2. 使用刷新后的请求头重建共享 OpenAI 客户端
3. 重试请求一次

部分旧版社区代理使用 `api.github.com/copilot_internal/v2/token` 交换流程。该端点对某些账户类型可能不可用（返回 404）。因此 Hermes 以直接 token 认证为主路径，依靠运行时凭据刷新 + 重试保证健壮性。
:::

**API 路由**：GPT-5+ 模型（`gpt-5-mini` 除外）自动使用 Responses API。其他所有模型（GPT-4o、Claude、Gemini 等）使用 Chat Completions。模型从 Copilot 实时目录自动检测。

**`copilot-acp` — Copilot ACP 智能体后端**。将本地 Copilot CLI 作为子进程启动：

```bash
hermes chat --provider copilot-acp --model copilot-acp
# 需要 PATH 中存在 GitHub Copilot CLI 且已完成 `copilot login`
```

**永久配置：**
```yaml
model:
  provider: "copilot"
  default: "gpt-5.4"
```

| 环境变量 | 说明 |
|---------------------|-------------|
| `COPILOT_GITHUB_TOKEN` | Copilot API 的 GitHub token（最高优先级） |
| `HERMES_COPILOT_ACP_COMMAND` | 覆盖 Copilot CLI 二进制路径（默认：`copilot`） |
| `HERMES_COPILOT_ACP_ARGS` | 覆盖 ACP 参数（默认：`--acp --stdio`） |

### 一等 API Key 提供商

这些提供商内置支持，具有专属提供商 ID。设置 API key 后使用 `--provider` 选择：

```bash
# NovitaAI Model API
hermes chat --provider novita --model moonshotai/kimi-k2.5
# 需要：~/.hermes/.env 中的 NOVITA_API_KEY

# z.ai / ZhipuAI GLM
hermes chat --provider zai --model glm-5
# 需要：~/.hermes/.env 中的 GLM_API_KEY

# Kimi / Moonshot AI（国际版：api.moonshot.ai）
hermes chat --provider kimi-coding --model kimi-for-coding
# 需要：~/.hermes/.env 中的 KIMI_API_KEY

# Kimi / Moonshot AI（中国版：api.moonshot.cn）
hermes chat --provider kimi-coding-cn --model kimi-k2.5
# 需要：~/.hermes/.env 中的 KIMI_CN_API_KEY

# MiniMax（全球端点）
hermes chat --provider minimax --model MiniMax-M2.7
# 需要：~/.hermes/.env 中的 MINIMAX_API_KEY

# MiniMax（中国端点）
hermes chat --provider minimax-cn --model MiniMax-M2.7
# 需要：~/.hermes/.env 中的 MINIMAX_CN_API_KEY

# Qwen Cloud / DashScope（Qwen 模型）
hermes chat --provider alibaba --model qwen3.5-plus
# 需要：~/.hermes/.env 中的 DASHSCOPE_API_KEY

# 小米 MiMo
hermes chat --provider xiaomi --model mimo-v2-pro
# 需要：~/.hermes/.env 中的 XIAOMI_API_KEY

# 腾讯 TokenHub（Hy3 Preview）
hermes chat --provider tencent-tokenhub --model hy3-preview
# 需要：~/.hermes/.env 中的 TOKENHUB_API_KEY

# Arcee AI（Trinity 模型）
hermes chat --provider arcee --model trinity-large-thinking
# 需要：~/.hermes/.env 中的 ARCEEAI_API_KEY

# GMI Cloud
# 使用 GMI /v1/models 端点返回的精确模型 ID。
hermes chat --provider gmi --model zai-org/GLM-5.1-FP8
# 需要：~/.hermes/.env 中的 GMI_API_KEY
```

或在 `config.yaml` 中永久设置提供商：
```yaml
model:
  provider: "gmi"
  default: "zai-org/GLM-5.1-FP8"
```

基础 URL 可通过 `NOVITA_BASE_URL`、`GLM_BASE_URL`、`KIMI_BASE_URL`、`MINIMAX_BASE_URL`、`MINIMAX_CN_BASE_URL`、`DASHSCOPE_BASE_URL`、`XIAOMI_BASE_URL`、`GMI_BASE_URL` 或 `TOKENHUB_BASE_URL` 环境变量覆盖。

:::note Z.AI 端点自动检测
使用 Z.AI / GLM 提供商时，Hermes 会自动探测多个端点（全球版、中国版、编程版）以找到接受你 API key 的端点。无需手动设置 `GLM_BASE_URL`——可用端点会被自动检测并缓存。
:::

### xAI（Grok）— Responses API + Prompt 缓存

xAI 通过 Responses API（`codex_responses` 传输）接入，自动支持 Grok 4 模型的推理——无需 `reasoning_effort` 参数，服务端默认进行推理。在 `~/.hermes/.env` 中设置 `XAI_API_KEY` 并在 `hermes model` 中选择 xAI，或直接用 `grok` 作为快捷方式输入 `/model grok-4-1-fast-reasoning`。

SuperGrok 和 X Premium+ 订阅者可以用浏览器 OAuth 登录，无需 API key——在 `hermes model` 中选择 **xAI Grok OAuth (SuperGrok / Premium+)**，或运行 `hermes auth add xai-oauth`。同一 OAuth bearer token 会被 xAI 直连工具（TTS、图像生成、视频生成、转录）自动复用。完整流程参见 [xAI Grok OAuth 指南](../guides/xai-grok-oauth.md)——如果 Hermes 运行在远程主机上，还需参见 [SSH / 远程主机上的 OAuth](../guides/oauth-over-ssh.md) 了解所需的 `ssh -L` 隧道配置。

使用 xAI 作为提供商时（任何包含 `x.ai` 的基础 URL），Hermes 会在每次 API 请求中自动发送 `x-grok-conv-id` 请求头以启用 prompt（提示词）缓存。这会将同一会话的请求路由到同一服务器，使 xAI 基础设施能够复用已缓存的系统 prompt 和对话历史。

无需任何配置——检测到 xAI 端点且存在会话 ID 时，缓存自动激活。这可降低多轮对话的延迟和成本。

xAI 还提供专属 TTS 端点（`/v1/tts`）。在 `hermes tools` → 语音与 TTS 中选择 **xAI TTS**，或参见[语音与 TTS](../user-guide/features/tts.md#text-to-speech) 页面了解配置。

### NovitaAI

[NovitaAI](https://novita.ai) 是面向开发者和智能体的 AI 原生云平台。三条产品线：200+ 模型的 Model API、用于构建和运行 AI 智能体的 Agent Sandbox，以及可扩展计算的 GPU Cloud，均可从同一平台访问。

```bash
# 使用任意可用模型
hermes chat --provider novita --model moonshotai/kimi-k2.5
# 需要：~/.hermes/.env 中的 NOVITA_API_KEY

# 短别名
hermes chat --provider novita-ai --model deepseek/deepseek-v3-0324
```

或在 `config.yaml` 中永久设置：
```yaml
model:
  provider: "novita"
  default: "moonshotai/kimi-k2.5"
  base_url: "https://api.novita.ai/openai/v1"
```

在 [novita.ai/settings/key-management](https://novita.ai/settings/key-management) 获取 API key。基础 URL 可通过 `NOVITA_BASE_URL` 覆盖。

### Ollama Cloud — 托管 Ollama 模型，OAuth + API Key

[Ollama Cloud](https://ollama.com/cloud) 托管与本地 Ollama 相同的开源模型目录，无需 GPU。在 `hermes model` 中选择 **Ollama Cloud**，粘贴来自 [ollama.com/settings/keys](https://ollama.com/settings/keys) 的 API key，Hermes 会自动发现可用模型。

```bash
hermes model
# → 选择"Ollama Cloud"
# → 粘贴你的 OLLAMA_API_KEY
# → 从已发现的模型中选择（gpt-oss:120b、glm-4.6:cloud、qwen3-coder:480b-cloud 等）
```

或直接编辑 `config.yaml`：
```yaml
model:
  provider: "ollama-cloud"
  default: "gpt-oss:120b"
```

模型目录从 `ollama.com/v1/models` 动态获取，缓存一小时。`model:tag` 格式（如 `qwen3-coder:480b-cloud`）在规范化过程中保留——不要使用连字符。

:::tip Ollama Cloud 与本地 Ollama
两者使用相同的 OpenAI 兼容 API。Cloud 是一等提供商（`--provider ollama-cloud`，`OLLAMA_API_KEY`）；本地 Ollama 通过自定义端点流程访问（基础 URL `http://localhost:11434/v1`，无需 key）。对于无法在本地运行的大模型使用 Cloud；对于隐私保护或离线工作使用本地。
:::

### AWS Bedrock

通过 AWS Bedrock 使用 Anthropic Claude、Amazon Nova、DeepSeek v3.2、Meta Llama 4 等模型。使用 AWS SDK（`boto3`）凭据链——无需 API key，使用标准 AWS 认证即可。

```bash
# 最简方式——~/.aws/credentials 中的命名 profile
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
  # guardrail:                 # 可选的 Bedrock Guardrails
  #   guardrail_identifier: "your-guardrail-id"
  #   guardrail_version: "DRAFT"
```

认证使用标准 boto3 链：显式 `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`、`~/.aws/credentials` 中的 `AWS_PROFILE`、EC2/ECS/Lambda 上的 IAM 角色、IMDS 或 SSO。如果已通过 AWS CLI 认证，无需设置任何环境变量。

Bedrock 底层使用 **Converse API**——请求被转换为 Bedrock 的模型无关格式，因此同一配置适用于 Claude、Nova、DeepSeek 和 Llama 模型。仅在调用非默认区域端点时才需设置 `BEDROCK_BASE_URL`。

参见 [AWS Bedrock 指南](/guides/aws-bedrock)，了解 IAM 配置、区域选择和跨区域推理的详细步骤。

### Qwen Portal（OAuth）

阿里巴巴 Qwen Portal，支持基于浏览器的 OAuth 登录。在 `hermes model` 中选择 **Qwen OAuth (Portal)**，通过浏览器登录，Hermes 会持久化刷新 token。

```bash
hermes model
# → 选择"Qwen OAuth (Portal)"
# → 浏览器打开；使用阿里巴巴账户登录
# → 确认——凭据保存到 ~/.hermes/auth.json

hermes chat   # 使用 portal.qwen.ai/v1 端点
```

或配置 `config.yaml`：
```yaml
model:
  provider: "qwen-oauth"
  default: "qwen3-coder-plus"
```

仅在 portal 端点迁移时才需设置 `HERMES_QWEN_BASE_URL`（默认：`https://portal.qwen.ai/v1`）。

:::tip Qwen OAuth 与 Qwen Cloud（阿里 DashScope）
`qwen-oauth` 使用面向消费者的 Qwen Portal，通过 OAuth 登录——适合个人用户。`alibaba` 提供商使用 Qwen Cloud（阿里 DashScope），需要 `DASHSCOPE_API_KEY`——适合程序化/生产工作负载。两者都路由到 Qwen 系列模型，但端点不同。
:::

### 阿里云（Coding Plan）

如果你订阅了阿里巴巴的 **Coding Plan**（独立于标准 DashScope API 访问的计费 SKU），Hermes 将其作为独立的一等提供商暴露：`alibaba-coding-plan`。端点：`https://coding-intl.dashscope.aliyuncs.com/v1`。与常规 `alibaba` 提供商一样兼容 OpenAI，但基础 URL 和计费面不同。

```yaml
model:
  provider: alibaba_coding     # alibaba-coding-plan 的别名
  model: qwen3-coder-plus
```

或通过 CLI：

```bash
hermes chat --provider alibaba_coding --model qwen3-coder-plus
```

`alibaba_coding` 使用与 `alibaba` 条目相同的 `DASHSCOPE_API_KEY`——无需单独的 key，只是路由目标不同。在此提供商注册之前，在 `config.yaml` 中设置 `provider: alibaba_coding` 的用户会静默回退到 OpenRouter 路由。

### MiniMax（OAuth）

通过浏览器 OAuth 登录使用 MiniMax-M2.7——无需 API key。在 `hermes model` 中选择 **MiniMax (OAuth)**，通过浏览器登录，Hermes 会持久化访问 token 和刷新 token。底层使用 Anthropic Messages 兼容端点（`/anthropic`）。

```bash
hermes model
# → 选择"MiniMax (OAuth)"
# → 浏览器打开；使用 MiniMax 账户登录（全球或中国区）
# → 确认——凭据保存到 ~/.hermes/auth.json

hermes chat   # 使用 api.minimax.io/anthropic 端点
```

或配置 `config.yaml`：
```yaml
model:
  provider: "minimax-oauth"
  default: "MiniMax-M2.7"
```

支持的模型：`MiniMax-M2.7`（主模型）和 `MiniMax-M2.7-highspeed`（默认辅助模型）。OAuth 路径忽略 `MINIMAX_API_KEY` / `MINIMAX_BASE_URL`。

:::tip MiniMax OAuth 与 API key
`minimax-oauth` 使用 MiniMax 面向消费者的 portal，通过 OAuth 登录——无需设置计费。`minimax` 和 `minimax-cn` 提供商使用 `MINIMAX_API_KEY` / `MINIMAX_CN_API_KEY`——用于程序化访问。完整流程参见 [MiniMax OAuth 指南](/guides/minimax-oauth)。
:::

### NVIDIA NIM

通过 [build.nvidia.com](https://build.nvidia.com)（免费 API key）或本地 NIM 端点使用 Nemotron 及其他开源模型。

```bash
# 云端（build.nvidia.com）
hermes chat --provider nvidia --model nvidia/nemotron-3-super-120b-a12b
# 需要：~/.hermes/.env 中的 NVIDIA_API_KEY

# 本地 NIM 端点——覆盖基础 URL
NVIDIA_BASE_URL=http://localhost:8000/v1 hermes chat --provider nvidia --model nvidia/nemotron-3-super-120b-a12b
```

或在 `config.yaml` 中永久设置：
```yaml
model:
  provider: "nvidia"
  default: "nvidia/nemotron-3-super-120b-a12b"
```

:::tip 本地 NIM
对于本地部署（DGX Spark、本地 GPU），设置 `NVIDIA_BASE_URL=http://localhost:8000/v1`。NIM 暴露与 build.nvidia.com 相同的 OpenAI 兼容 chat completions API，因此在云端和本地之间切换只需修改一行环境变量。
:::

Hermes 会在每次向 `build.nvidia.com` 发送请求时自动附加 NIM 计费来源请求头——无需任何配置。这会在 NVIDIA 计费仪表板中将消耗路由到正确的来源。

### GMI Cloud

通过 [GMI Cloud](https://www.gmicloud.ai/) 使用开源和推理模型——OpenAI 兼容 API，API key 认证。

```bash
# GMI Cloud
hermes chat --provider gmi --model deepseek-ai/DeepSeek-R1
# 需要：~/.hermes/.env 中的 GMI_API_KEY
```

或在 `config.yaml` 中永久设置：
```yaml
model:
  provider: "gmi"
  default: "deepseek-ai/DeepSeek-R1"
```

基础 URL 可通过 `GMI_BASE_URL` 覆盖（默认：`https://api.gmi-serving.com/v1`）。

### StepFun

通过 [StepFun](https://platform.stepfun.com) 使用 Step 系列模型——OpenAI 兼容 API，API key 认证。

```bash
# StepFun
hermes chat --provider stepfun --model step-3-mini
# 需要：~/.hermes/.env 中的 STEPFUN_API_KEY
```

或在 `config.yaml` 中永久设置：
```yaml
model:
  provider: "stepfun"
  default: "step-3-mini"
```

基础 URL 可通过 `STEPFUN_BASE_URL` 覆盖（默认：`https://api.stepfun.com/v1`）。

### Hugging Face 推理提供商

[Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers) 通过统一的 OpenAI 兼容端点（`router.huggingface.co/v1`）路由到 20+ 开源模型。请求自动路由到最快的可用后端（Groq、Together、SambaNova 等），并支持自动故障转移。

```bash
# 使用任意可用模型
hermes chat --provider huggingface --model Qwen/Qwen3-235B-A22B-Thinking-2507
# 需要：~/.hermes/.env 中的 HF_TOKEN

# 短别名
hermes chat --provider hf --model deepseek-ai/DeepSeek-V3.2
```

或在 `config.yaml` 中永久设置：
```yaml
model:
  provider: "huggingface"
  default: "Qwen/Qwen3-235B-A22B-Thinking-2507"
```

在 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 获取 token——确保启用"Make calls to Inference Providers"权限。包含免费层（每月 $0.10 积分，不加价）。

可在模型名称后附加路由后缀：`:fastest`（默认）、`:cheapest`，或 `:provider_name` 强制指定后端。

基础 URL 可通过 `HF_BASE_URL` 覆盖。

### 通过 OAuth 使用 Google Gemini（`google-gemini-cli`）

`google-gemini-cli` 提供商使用 Google 的 Cloud Code Assist 后端——与 Google 自己的 `gemini-cli` 工具使用的 API 相同。支持**免费层**（个人账户每日配额充足）和**付费层**（通过 GCP 项目的 Standard/Enterprise）。

**快速开始：**

```bash
hermes model
# → 选择"Google Gemini (OAuth)"
# → 查看政策警告，确认
# → 浏览器打开 accounts.google.com，登录
# → 完成——Hermes 在首次请求时自动开通免费层
```

Hermes 默认使用 Google 的**公开** `gemini-cli` 桌面 OAuth 客户端——与 Google 在其开源 `gemini-cli` 中包含的凭据相同。桌面 OAuth 客户端不是机密客户端（PKCE 提供安全保障）。你无需安装 `gemini-cli` 或注册自己的 GCP OAuth 客户端。

**认证工作原理：**
- 针对 `accounts.google.com` 的 PKCE 授权码流程
- 浏览器回调地址 `http://127.0.0.1:8085/oauth2callback`（端口占用时自动回退到临时端口）
- Token 存储在 `~/.hermes/auth/google_oauth.json`（chmod 0600，原子写入，跨进程 `fcntl` 锁）
- 到期前 60 秒自动刷新
- 无头环境（SSH、`HERMES_HEADLESS=1`）→ 粘贴模式回退
- 并发刷新去重——两个并发请求不会触发双重刷新
- `invalid_grant`（刷新 token 被撤销）→ 凭据文件被清除，提示用户重新登录

**推理工作原理：**
- 流量发送到 `https://cloudcode-pa.googleapis.com/v1internal:generateContent`
  （流式传输为 `:streamGenerateContent?alt=sse`），而非付费的 `v1beta/openai` 端点
- 请求体封装为 `{project, model, user_prompt_id, request}`
- OpenAI 格式的 `messages[]`、`tools[]`、`tool_choice` 被转换为 Gemini 原生的
  `contents[]`、`tools[].functionDeclarations`、`toolConfig` 格式
- 响应转换回 OpenAI 格式，Hermes 其余部分无感知

**层级与项目 ID：**

| 你的情况 | 操作 |
|---|---|
| 个人 Google 账户，使用免费层 | 无需操作——登录即可开始聊天 |
| Workspace / Standard / Enterprise 账户 | 将 `HERMES_GEMINI_PROJECT_ID` 或 `GOOGLE_CLOUD_PROJECT` 设置为你的 GCP 项目 ID |
| VPC-SC 保护的组织 | Hermes 检测到 `SECURITY_POLICY_VIOLATED` 后自动强制使用 `standard-tier` |

免费层在首次使用时自动开通 Google 托管项目。无需 GCP 配置。

**配额监控：**

```
/gquota
```

以进度条显示每个模型的剩余 Code Assist 配额：

```
Gemini Code Assist quota  (project: 123-abc)

  gemini-2.5-pro                      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░   85%
  gemini-2.5-flash [input]            ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░   92%
```

:::warning 政策风险
Google 认为将 Gemini CLI OAuth 客户端用于第三方软件违反政策。部分用户反映账户受到限制。为降低风险，建议改用 `gemini` 提供商并通过 API key 访问。Hermes 会在 OAuth 开始前显示警告并要求明确确认。
:::

**自定义 OAuth 客户端（可选）：**

如果你希望注册自己的 Google OAuth 客户端——例如将配额和授权范围限定在自己的 GCP 项目内——请设置：

```bash
HERMES_GEMINI_CLIENT_ID=your-client.apps.googleusercontent.com
HERMES_GEMINI_CLIENT_SECRET=...   # 桌面客户端可选
```

在 [console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials) 注册一个**桌面应用** OAuth 客户端，并启用 Generative Language API。

## 自定义与自托管 LLM 提供商

Hermes Agent 可与**任何 OpenAI 兼容 API 端点**配合使用。只要服务器实现了 `/v1/chat/completions`，就可以将 Hermes 指向它。这意味着你可以使用本地模型、GPU 推理服务器、多提供商路由器或任何第三方 API。

### 通用配置

配置自定义端点的三种方式：

**交互式配置（推荐）：**
```bash
hermes model
# 选择"Custom endpoint (self-hosted / VLLM / etc.)"
# 输入：API 基础 URL、API key、模型名称
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
`.env` 中的 `OPENAI_BASE_URL` 和 `LLM_MODEL` 已**移除**。Hermes 的任何部分都不再读取这两个变量——`config.yaml` 是模型和端点配置的唯一来源。如果你的 `.env` 中有过时条目，下次运行 `hermes setup` 或配置迁移时会自动清除。请使用 `hermes model` 或直接编辑 `config.yaml`。
:::

两种方式都会持久化到 `config.yaml`，该文件是模型、提供商和基础 URL 的唯一来源。

### 使用 `/model` 切换模型

:::warning hermes model 与 /model
**`hermes model`**（在终端中运行，任何聊天会话之外）是**完整的提供商配置向导**。用于添加新提供商、运行 OAuth 流程、输入 API key 和配置自定义端点。

**`/model`**（在活跃的 Hermes 聊天会话中输入）只能在**已配置的**提供商和模型之间**切换**。它无法添加新提供商、运行 OAuth 或提示输入 API key。如果你只配置了一个提供商（如 OpenRouter），`/model` 只会显示该提供商的模型。

**添加新提供商：** 退出会话（`Ctrl+C` 或 `/quit`），运行 `hermes model`，配置新提供商，然后开启新会话。
:::

配置好至少一个自定义端点后，可以在会话中途切换模型：

```
/model custom:qwen-2.5          # 切换到自定义端点上的某个模型
/model custom                    # 从端点自动检测模型
/model openrouter:claude-sonnet-4 # 切换回云端提供商
```

如果你配置了**命名自定义提供商**（见下文），使用三段式语法：

```
/model custom:local:qwen-2.5    # 使用"local"自定义提供商和 qwen-2.5 模型
/model custom:work:llama3       # 使用"work"自定义提供商和 llama3
```

切换提供商时，Hermes 会将基础 URL 和提供商持久化到配置中，使更改在重启后保留。从自定义端点切换到内置提供商时，过时的基础 URL 会自动清除。

:::tip
`/model custom`（不带模型名称）会查询端点的 `/models` API，如果只加载了一个模型则自动选择。适用于运行单个模型的本地服务器。
:::

以下所有内容遵循相同模式——只需更改 URL、key 和模型名称。

---

### Ollama — 本地模型，零配置

[Ollama](https://ollama.com/) 用一条命令在本地运行开源模型。最适合：快速本地实验、隐私敏感工作、离线使用。通过 OpenAI 兼容 API 支持工具调用。

```bash
# 安装并运行模型
ollama pull qwen2.5-coder:32b
ollama serve   # 在端口 11434 启动
```

然后配置 Hermes：

```bash
hermes model
# 选择"Custom endpoint (self-hosted / VLLM / etc.)"
# 输入 URL：http://localhost:11434/v1
# 跳过 API key（Ollama 不需要）
# 输入模型名称（如 qwen2.5-coder:32b）
```

或直接配置 `config.yaml`：

```yaml
model:
  default: qwen2.5-coder:32b
  provider: custom
  base_url: http://localhost:11434/v1
  context_length: 32768   # 见下方警告
```

:::caution Ollama 默认上下文长度非常短
Ollama **默认不使用**模型的完整上下文窗口。根据你的显存，默认值为：

| 可用显存 | 默认上下文 |
|----------------|----------------|
| 小于 24 GB | **4,096 tokens** |
| 24–48 GB | 32,768 tokens |
| 48+ GB | 256,000 tokens |

对于带工具的智能体使用，**至少需要 16k–32k 上下文**。在 4k 时，系统 prompt 加工具 schema 就可能填满窗口，没有空间留给对话。

**如何增加**（选择其一）：

```bash
# 方式 1：通过环境变量设置服务器全局值（推荐）
OLLAMA_CONTEXT_LENGTH=32768 ollama serve

# 方式 2：对于 systemd 管理的 Ollama
sudo systemctl edit ollama.service
# 添加：Environment="OLLAMA_CONTEXT_LENGTH=32768"
# 然后：sudo systemctl daemon-reload && sudo systemctl restart ollama

# 方式 3：烘焙到自定义模型中（每个模型持久生效）
echo -e "FROM qwen2.5-coder:32b\nPARAMETER num_ctx 32768" > Modelfile
ollama create qwen2.5-coder-32k -f Modelfile
```

**无法通过 OpenAI 兼容 API**（`/v1/chat/completions`）设置上下文长度。必须在服务端或通过 Modelfile 配置。这是将 Ollama 与 Hermes 等工具集成时最常见的困惑来源。
:::

**验证上下文设置是否正确：**

```bash
ollama ps
# 查看 CONTEXT 列——应显示你配置的值
```

:::tip
使用 `ollama list` 列出可用模型。使用 `ollama pull <model>` 从 [Ollama 库](https://ollama.com/library) 拉取任意模型。Ollama 自动处理 GPU 卸载——大多数配置无需手动设置。
:::

---

### vLLM — 高性能 GPU 推理

[vLLM](https://docs.vllm.ai/) 是生产 LLM 服务的标准方案。最适合：GPU 硬件上的最大吞吐量、大模型服务、连续批处理。

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
# 输入 URL：http://localhost:8000/v1
# 跳过 API key（或输入你配置 vLLM 时设置的 --api-key）
# 输入模型名称：meta-llama/Llama-3.1-70B-Instruct
```

**上下文长度：** vLLM 默认读取模型的 `max_position_embeddings`。如果超出显存，会报错并要求降低 `--max-model-len`。也可使用 `--max-model-len auto` 自动找到能放入显存的最大值。设置 `--gpu-memory-utilization 0.95`（默认 0.9）可将更多上下文放入显存。

**工具调用需要显式标志：**

| 标志 | 用途 |
|------|---------|
| `--enable-auto-tool-choice` | `tool_choice: "auto"` 所必需（Hermes 的默认值） |
| `--tool-call-parser <name>` | 模型工具调用格式的解析器 |

支持的解析器：`hermes`（Qwen 2.5、Hermes 2/3）、`llama3_json`（Llama 3.x）、`mistral`、`deepseek_v3`、`deepseek_v31`、`xlam`、`pythonic`。没有这些标志，工具调用将无法工作——模型会将工具调用以文本形式输出。

:::tip
vLLM 支持人类可读的大小：`--max-model-len 64k`（小写 k = 1000，大写 K = 1024）。
:::

---

### SGLang — 带 RadixAttention 的快速服务

[SGLang](https://github.com/sgl-project/sglang) 是 vLLM 的替代方案，具有用于 KV 缓存复用的 RadixAttention。最适合：多轮对话（前缀缓存）、约束解码、结构化输出。

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
# 输入 URL：http://localhost:30000/v1
# 输入模型名称：meta-llama/Llama-3.1-70B-Instruct
```

**上下文长度：** SGLang 默认从模型配置读取。使用 `--context-length` 覆盖。如果需要超过模型声明的最大值，设置 `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`。

**工具调用：** 使用 `--tool-call-parser` 并选择适合你模型系列的解析器：`qwen`（Qwen 2.5）、`llama3`、`llama4`、`deepseekv3`、`mistral`、`glm`。没有此标志，工具调用将以纯文本返回。

:::caution SGLang 默认最大输出 128 tokens
如果响应看起来被截断，在请求中添加 `max_tokens` 或在服务器上设置 `--default-max-tokens`。SGLang 的默认值是每次响应仅 128 tokens（如果请求中未指定）。
:::

---

### llama.cpp / llama-server — CPU 与 Metal 推理

[llama.cpp](https://github.com/ggml-org/llama.cpp) 在 CPU、Apple Silicon（Metal）和消费级 GPU 上运行量化模型。最适合：无数据中心 GPU 的模型运行、Mac 用户、边缘部署。

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

**上下文长度（`-c`）：** 近期版本默认为 `0`，从 GGUF 元数据读取模型的训练上下文。对于训练上下文超过 128k 的模型，这可能因尝试分配完整 KV 缓存而导致 OOM。请显式设置 `-c` 为你需要的值（32k–64k 是智能体使用的合理范围）。如果使用并行槽（`-np`），总上下文在槽之间分配——`-c 32768 -np 4` 时每个槽只有 8k。

然后配置 Hermes 指向它：

```bash
hermes model
# 选择"Custom endpoint (self-hosted / VLLM / etc.)"
# 输入 URL：http://localhost:8080/v1
# 跳过 API key（本地服务器不需要）
# 输入模型名称——或留空以在只加载一个模型时自动检测
```

这会将端点保存到 `config.yaml`，在会话间持久保留。

:::caution `--jinja` 是工具调用的必要条件
没有 `--jinja`，llama-server 会完全忽略 `tools` 参数。模型会尝试在响应文本中写入 JSON 来调用工具，但 Hermes 不会将其识别为工具调用——你会看到原始 JSON（如 `{"name": "web_search", ...}`）作为消息打印出来，而不是实际执行搜索。

原生工具调用支持（最佳性能）：Llama 3.x、Qwen 2.5（包括 Coder）、Hermes 2/3、Mistral、DeepSeek、Functionary。其他所有模型使用通用处理器，可以工作但效率可能较低。完整列表参见 [llama.cpp 函数调用文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)。

可通过检查 `http://localhost:8080/props` 验证工具支持是否已激活——`chat_template` 字段应存在。
:::

:::tip
从 [Hugging Face](https://huggingface.co/models?library=gguf) 下载 GGUF 模型。Q4_K_M 量化在质量与内存使用之间提供最佳平衡。
:::

---

### LM Studio — 带本地模型的桌面应用

[LM Studio](https://lmstudio.ai/) 是一款带 GUI 的本地模型运行桌面应用。最适合：偏好可视化界面的用户、快速模型测试、macOS/Windows/Linux 开发者。

从 LM Studio 应用启动服务器（开发者标签页 → 启动服务器），或使用 CLI：

```bash
lms server start                        # 在端口 1234 启动
lms load qwen2.5-coder --context-length 32768
```

然后配置 Hermes：

```bash
hermes model
# 选择"LM Studio"
# 按 Enter 使用 http://localhost:1234/v1
# 从已发现的模型中选择
# 如果启用了 LM Studio 服务器认证，在提示时输入 LM_API_KEY
```

Hermes 会自动以 64K 上下文长度加载 LM Studio 模型。

在 LM Studio 中更改上下文长度：

1. 点击模型选择器旁的齿轮图标
2. 将"Context Length"设置为至少 64000 以获得流畅体验
3. 重新加载模型使更改生效
4. 如果你的机器无法容纳 64000，考虑使用上下文长度更大的小模型。

或使用 CLI：`lms load model-name --context-length 64000`

可使用 CLI 估算模型是否能放入内存：`lms load model-name --context-length 64000 --estimate-only`

设置每个模型的持久默认值：我的模型标签页 → 模型上的齿轮图标 → 设置上下文大小。
:::

**工具调用：** 自 LM Studio 0.3.6 起支持。具有原生工具调用训练的模型（Qwen 2.5、Llama 3.x、Mistral、Hermes）会被自动检测并显示工具徽章。其他模型使用通用回退，可靠性可能较低。

---

### WSL2 网络（Windows 用户）

由于 Hermes Agent 需要 Unix 环境，Windows 用户在 WSL2 内运行它。如果你的模型服务器（Ollama、LM Studio 等）运行在 **Windows 主机**上，需要桥接网络——WSL2 使用具有独立子网的虚拟网络适配器，因此 WSL2 内的 `localhost` 指向 Linux 虚拟机，**而非** Windows 主机。

:::tip 都在 WSL2 内？没问题。
如果你的模型服务器也在 WSL2 内运行（vLLM、SGLang 和 llama-server 的常见情况），`localhost` 可以正常工作——它们共享同一网络命名空间。跳过本节。
:::

#### 方式 1：镜像网络模式（推荐）

适用于 **Windows 11 22H2+**，镜像模式使 `localhost` 在 Windows 和 WSL2 之间双向工作——最简单的解决方案。

1. 创建或编辑 `%USERPROFILE%\.wslconfig`（如 `C:\Users\YourName\.wslconfig`）：
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
   curl http://localhost:11434/v1/models   # Windows 上的 Ollama——正常工作
   ```

:::note Hyper-V 防火墙
在某些 Windows 11 版本上，Hyper-V 防火墙默认阻止镜像连接。如果启用镜像模式后 `localhost` 仍无法工作，在**管理员 PowerShell** 中运行：
```powershell
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
```
:::

#### 方式 2：使用 Windows 主机 IP（Windows 10 / 旧版本）

如果无法使用镜像模式，从 WSL2 内部找到 Windows 主机 IP 并使用它代替 `localhost`：

```bash
# 获取 Windows 主机 IP（WSL2 虚拟网络的默认网关）
ip route show | grep -i default | awk '{ print $3 }'
# 示例输出：172.29.192.1
```

在 Hermes 配置中使用该 IP：

```yaml
model:
  default: qwen2.5-coder:32b
  provider: custom
  base_url: http://172.29.192.1:11434/v1   # Windows 主机 IP，非 localhost
```

:::tip 动态获取
WSL2 重启后主机 IP 可能变化。可在 shell 中动态获取：
```bash
export WSL_HOST=$(ip route show | grep -i default | awk '{ print $3 }')
echo "Windows host at: $WSL_HOST"
curl http://$WSL_HOST:11434/v1/models   # 测试 Ollama
```

或使用机器的 mDNS 名称（需要 WSL2 中的 `libnss-mdns`）：
```bash
sudo apt install libnss-mdns
curl http://$(hostname).local:11434/v1/models
```
:::

#### 服务器绑定地址（NAT 模式必需）

如果使用**方式 2**（NAT 模式加主机 IP），Windows 上的模型服务器必须接受来自 `127.0.0.1` 以外的连接。默认情况下，大多数服务器只监听 localhost——NAT 模式下 WSL2 的连接来自不同的虚拟子网，会被拒绝。在镜像模式下，`localhost` 直接映射，因此默认的 `127.0.0.1` 绑定可以正常工作。

| 服务器 | 默认绑定 | 修复方式 |
|--------|-------------|------------|
| **Ollama** | `127.0.0.1` | 启动 Ollama 前设置 `OLLAMA_HOST=0.0.0.0` 环境变量（Windows 系统设置 → 环境变量，或编辑 Ollama 服务） |
| **LM Studio** | `127.0.0.1` | 在开发者标签页 → 服务器设置中启用**"Serve on Network"** |
| **llama-server** | `127.0.0.1` | 在启动命令中添加 `--host 0.0.0.0` |
| **vLLM** | `0.0.0.0` | 默认已绑定所有接口 |
| **SGLang** | `127.0.0.1` | 在启动命令中添加 `--host 0.0.0.0` |

**Windows 上的 Ollama（详细步骤）：** Ollama 作为 Windows 服务运行。设置 `OLLAMA_HOST`：
1. 打开**系统属性** → **环境变量**
2. 添加新的**系统变量**：`OLLAMA_HOST` = `0.0.0.0`
3. 重启 Ollama 服务（或重启电脑）

#### Windows 防火墙

Windows 防火墙将 WSL2 视为独立网络（在 NAT 和镜像模式下均如此）。如果按上述步骤操作后连接仍然失败，为模型服务器端口添加防火墙规则：

```powershell
# 在管理员 PowerShell 中运行——将 PORT 替换为你服务器的端口
New-NetFirewallRule -DisplayName "Allow WSL2 to Model Server" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11434
```

常用端口：Ollama `11434`、vLLM `8000`、SGLang `30000`、llama-server `8080`、LM Studio `1234`。

#### 快速验证

从 WSL2 内部测试是否能访问模型服务器：

```bash
# 将 URL 替换为你服务器的地址和端口
curl http://localhost:11434/v1/models          # 镜像模式
curl http://172.29.192.1:11434/v1/models       # NAT 模式（使用你的实际主机 IP）
```

如果收到列出模型的 JSON 响应，说明配置正确。在 Hermes 配置中使用相同的 URL 作为 `base_url`。

---

### 本地模型故障排查

以下问题影响与 Hermes 配合使用的**所有**本地推理服务器。

#### 从 WSL2 连接 Windows 托管模型服务器时"连接被拒绝"

如果你在 WSL2 内运行 Hermes 而模型服务器在 Windows 主机上，在 WSL2 默认 NAT 网络模式下 `http://localhost:<port>` 无法工作。参见上方的 [WSL2 网络](#wsl2-networking-windows-users) 了解解决方案。

#### 工具调用以文本形式出现而非执行

模型输出类似 `{"name": "web_search", "arguments": {...}}` 的消息，而不是实际调用工具。

**原因：** 你的服务器未启用工具调用，或模型不支持通过服务器的工具调用实现。

| 服务器 | 修复方式 |
|--------|-----|
| **llama.cpp** | 在启动命令中添加 `--jinja` |
| **vLLM** | 添加 `--enable-auto-tool-choice --tool-call-parser hermes` |
| **SGLang** | 添加 `--tool-call-parser qwen`（或适当的解析器） |
| **Ollama** | 工具调用默认启用——确保你的模型支持（使用 `ollama show model-name` 检查） |
| **LM Studio** | 更新到 0.3.6+ 并使用具有原生工具支持的模型 |

#### 模型似乎忘记上下文或给出不连贯的响应

**原因：** 上下文窗口太小。当对话超过上下文限制时，大多数服务器会静默丢弃较早的消息。Hermes 的系统 prompt 加工具 schema 单独就可能占用 4k–8k tokens。

**诊断：**

```bash
# 检查 Hermes 认为的上下文大小
# 查看启动行："Context limit: X tokens"

# 检查服务器的实际上下文
# Ollama：ollama ps（CONTEXT 列）
# llama.cpp：curl http://localhost:8080/props | jq '.default_generation_settings.n_ctx'
# vLLM：检查启动参数中的 --max-model-len
```

**修复：** 将上下文设置为至少 **32,768 tokens** 用于智能体使用。参见上方各服务器章节了解具体标志。

#### 启动时显示"Context limit: 2048 tokens"

Hermes 从服务器的 `/v1/models` 端点自动检测上下文长度。如果服务器报告的值较低（或根本不报告），Hermes 使用模型声明的限制，该值可能不正确。

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
1. **服务器上的输出上限（`max_tokens`）过低** — SGLang 默认每次响应 128 tokens。在服务器上设置 `--default-max-tokens`，或在 config.yaml 中配置 `model.max_tokens`。注意：`max_tokens` 只控制响应长度——与对话历史可以有多长无关（那是 `context_length`）。
2. **上下文耗尽** — 模型填满了上下文窗口。增加 `model.context_length` 或在 Hermes 中启用[上下文压缩](/user-guide/configuration#context-compression)。

---

### LiteLLM Proxy — 多提供商网关

[LiteLLM](https://docs.litellm.ai/) 是一个 OpenAI 兼容代理，将 100+ LLM 提供商统一在单一 API 后面。最适合：无需更改配置即可切换提供商、负载均衡、故障转移链、预算控制。

```bash
# 安装并启动
pip install "litellm[proxy]"
litellm --model anthropic/claude-sonnet-4 --port 4000

# 或使用配置文件支持多个模型：
litellm --config litellm_config.yaml --port 4000
```

然后通过 `hermes model` → 自定义端点 → `http://localhost:4000/v1` 配置 Hermes。

带故障转移的 `litellm_config.yaml` 示例：
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

### ClawRouter — 成本优化路由

[ClawRouter](https://github.com/BlockRunAI/ClawRouter) 由 BlockRunAI 开发，是一个本地路由代理，根据查询复杂度自动选择模型。它从 14 个维度对请求进行分类，并路由到能处理该任务的最便宜模型。支付方式为 USDC 加密货币（无需 API key）。

```bash
# 安装并启动
npx @blockrun/clawrouter    # 在端口 8402 启动
```

然后通过 `hermes model` → 自定义端点 → `http://localhost:8402/v1` → 模型名称 `blockrun/auto` 配置 Hermes。

路由配置文件：
| 配置文件 | 策略 | 节省 |
|---------|----------|---------|
| `blockrun/auto` | 质量/成本均衡 | 74-100% |
| `blockrun/eco` | 尽可能便宜 | 95-100% |
| `blockrun/premium` | 最佳质量模型 | 0% |
| `blockrun/free` | 仅免费模型 | 100% |
| `blockrun/agentic` | 针对工具使用优化 | 不定 |

:::note
ClawRouter 需要在 Base 或 Solana 上有 USDC 充值的钱包用于支付。所有请求通过 BlockRun 的后端 API 路由。运行 `npx @blockrun/clawrouter doctor` 检查钱包状态。
:::

---

### 其他兼容提供商

任何具有 OpenAI 兼容 API 的服务均可使用。一些常用选项：

| 提供商 | 基础 URL | 说明 |
|----------|----------|-------|
| [Together AI](https://together.ai) | `https://api.together.xyz/v1` | 云托管开源模型 |
| [Groq](https://groq.com) | `https://api.groq.com/openai/v1` | 超快推理 |
| [DeepSeek](https://deepseek.com) | `https://api.deepseek.com/v1` | DeepSeek 模型 |
| [Fireworks AI](https://fireworks.ai) | `https://api.fireworks.ai/inference/v1` | 快速开源模型托管 |
| [GMI Cloud](https://www.gmicloud.ai/) | `https://api.gmi-serving.com/v1` | 托管 OpenAI 兼容推理 |
| [Cerebras](https://cerebras.ai) | `https://api.cerebras.ai/v1` | 晶圆级芯片推理 |
| [Mistral AI](https://mistral.ai) | `https://api.mistral.ai/v1` | Mistral 模型 |
| [OpenAI](https://openai.com) | `https://api.openai.com/v1` | 直连 OpenAI |
| [Azure OpenAI](https://azure.microsoft.com) | `https://YOUR.openai.azure.com/` | 企业级 OpenAI |
| [LocalAI](https://localai.io) | `http://localhost:8080/v1` | 自托管，多模型 |
| [Jan](https://jan.ai) | `http://localhost:1337/v1` | 带本地模型的桌面应用 |

通过 `hermes model` → 自定义端点，或在 `config.yaml` 中配置任意上述服务：

```yaml
model:
  default: meta-llama/Llama-3.1-70B-Instruct-Turbo
  provider: custom
  base_url: https://api.together.xyz/v1
  api_key: your-together-key
```

---

### 上下文长度检测

:::note 两个设置，容易混淆
**`context_length`** 是**总上下文窗口**——输入和输出 token 的合计预算（例如 Claude Opus 4.6 为 200,000）。Hermes 用它来决定何时压缩历史记录以及验证 API 请求。

**`model.max_tokens`** 是**输出上限**——模型在*单次响应*中最多可生成的 token 数。与对话历史可以有多长无关。行业标准名称 `max_tokens` 是常见的混淆来源；Anthropic 的原生 API 已将其重命名为 `max_output_tokens` 以更清晰。

当自动检测获取的窗口大小不正确时，设置 `context_length`。
仅当需要限制单次响应长度时，才设置 `model.max_tokens`。
:::

Hermes 使用多源解析链来检测模型和提供商的正确上下文窗口：

1. **配置覆盖** — config.yaml 中的 `model.context_length`（最高优先级）
2. **自定义提供商按模型** — `custom_providers[].models.<id>.context_length`
3. **持久缓存** — 之前发现的值（重启后保留）
4. **端点 `/models`** — 查询服务器 API（本地/自定义端点）
5. **Anthropic `/v1/models`** — 查询 Anthropic API 获取 `max_input_tokens`（仅 API key 用户）
6. **OpenRouter API** — 来自 OpenRouter 的实时模型元数据
7. **Nous Portal** — 将 Nous 模型 ID 后缀匹配到 OpenRouter 元数据
8. **[models.dev](https://models.dev)** — 社区维护的注册表，包含 100+ 提供商 3800+ 模型的提供商特定上下文长度
9. **回退默认值** — 广泛的模型系列模式（默认 128K）

大多数配置开箱即用。该系统具有提供商感知能力——同一模型在不同服务商处可能有不同的上下文限制（例如 `claude-opus-4.6` 在 Anthropic 直连时为 1M，在 GitHub Copilot 上为 128K）。

要显式设置上下文长度，在模型配置中添加 `context_length`：

```yaml
model:
  default: "qwen3.5:9b"
  base_url: "http://localhost:8080/v1"
  context_length: 131072  # tokens
```

对于自定义端点，也可以按模型设置上下文长度：

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

`hermes model` 在配置自定义端点时会提示输入上下文长度。留空则自动检测。

:::tip 何时手动设置
- 你使用的 Ollama 自定义 `num_ctx` 低于模型最大值
- 你想将上下文限制在模型最大值以下（例如在 128k 模型上使用 8k 以节省显存）
- 你在不暴露 `/v1/models` 的代理后面运行
:::

---

### 命名自定义提供商

如果你使用多个自定义端点（例如本地开发服务器和远程 GPU 服务器），可以在 `config.yaml` 中将它们定义为命名自定义提供商：

```yaml
custom_providers:
  - name: local
    base_url: http://localhost:8080/v1
    # api_key 省略——Hermes 对无 key 的本地服务器使用"no-key-required"
  - name: work
    base_url: https://gpu-server.internal.corp/v1
    key_env: CORP_API_KEY
    api_mode: chat_completions   # 由 `hermes model` → 自定义端点向导显式设置；自动检测仍作为回退
  - name: anthropic-proxy
    base_url: https://proxy.example.com/anthropic
    key_env: ANTHROPIC_PROXY_KEY
    api_mode: anthropic_messages  # 用于 Anthropic 兼容代理
```

某些 OpenAI 兼容端点需要特定于提供商的请求体字段。在对应的自定义提供商中添加 `extra_body` 映射，Hermes 会将其合并到该端点的每个 chat-completions 请求中：

```yaml
custom_providers:
  - name: gemma-local
    base_url: http://localhost:8080/v1
    model: google/gemma-4-31b-it
    extra_body:
      enable_thinking: true
      reasoning_effort: high
```

使用你服务器文档中的格式。例如，vLLM Gemma 部署和某些 NVIDIA NIM 端点期望 `enable_thinking` 在 `chat_template_kwargs` 下，而不是作为顶级 `extra_body` 字段：

```yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: true
```

`hermes model` → 自定义端点向导现在会显式提示 `api_mode` 并将你的答案持久化到 `config.yaml`。当字段留空时，基于 URL 的自动检测（例如 `/anthropic` 路径 → `anthropic_messages`）仍作为回退。

使用三段式语法在会话中途切换：

```
/model custom:local:qwen-2.5       # 使用"local"端点和 qwen-2.5
/model custom:work:llama3-70b      # 使用"work"端点和 llama3-70b
/model custom:anthropic-proxy:claude-sonnet-4  # 使用代理
```

也可以从交互式 `hermes model` 菜单中选择命名自定义提供商。

---

### 实战配置：Together AI、Groq、Perplexity

[其他兼容提供商](#other-compatible-providers) 中列出的云提供商都使用 OpenAI 的 REST 方言，因此在 `custom_providers:` 下的接入方式相同。以下是三个可直接使用的配置示例。每个示例放入 `~/.hermes/config.yaml`，对应的 API key 放入 `~/.hermes/.env`。

#### Together AI

托管开源模型（Llama、MiniMax、Gemma、DeepSeek、Qwen），价格显著低于一方 API。适合多模型场景的默认选择。

```yaml
# ~/.hermes/config.yaml
custom_providers:
  - name: together
    base_url: https://api.together.xyz/v1
    key_env: TOGETHER_API_KEY
    # api_mode: chat_completions  # 默认——无需设置

model:
  default: MiniMaxAI/MiniMax-M2.7   # 或 together.ai/models 中的任意模型
  provider: custom:together
```

```bash
# ~/.hermes/.env
TOGETHER_API_KEY=your-together-key
```

会话中途切换模型：

```
/model custom:together:meta-llama/Llama-3.3-70B-Instruct-Turbo
/model custom:together:google/gemma-4-31b-it
/model custom:together:deepseek-ai/DeepSeek-V3
```

Together 的 `/v1/models` 端点可用，因此 `hermes model` 可以自动发现可用模型。

#### Groq

超快推理（Llama-3.3-70B 约 500 tok/s）。模型目录较小，但对延迟敏感的交互式使用效果出色。

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

当你需要自动进行实时网页搜索和引用的模型时很有用。对可用模型有严格限制——查看 [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api) 获取当前列表。

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

#### 在单个配置中使用多个提供商

三个示例可以组合使用——同时使用所有提供商，并通过 `/model custom:<name>:<model>` 按轮次切换：

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
  provider: custom:together      # 启动时使用 Together；之后可自由切换
```

:::tip 故障排查
- `hermes doctor` 对于上述任何名称都不应打印 `Unknown provider` 警告（在 #15083 的 CLI 验证器修复之后）。
- 如果某个提供商的 `/v1/models` 端点不可达（Perplexity 是常见情况），`hermes model` 会在警告后持久化模型而不是硬性拒绝——参见 #15136。
- 要完全跳过 `custom_providers:` 并使用带 `CUSTOM_BASE_URL` 环境变量的裸 `provider: custom`，参见 #15103。
:::

---

### 选择合适的配置

| 使用场景 | 推荐方案 |
|----------|-------------|
| **只想让它工作** | OpenRouter（默认）或 Nous Portal |
| **本地模型，简单配置** | Ollama |
| **生产 GPU 服务** | vLLM 或 SGLang |
| **Mac / 无 GPU** | Ollama 或 llama.cpp |
| **多提供商路由** | LiteLLM Proxy 或 OpenRouter |
| **成本优化** | ClawRouter 或带 `sort: "price"` 的 OpenRouter |
| **最大隐私保护** | Ollama、vLLM 或 llama.cpp（完全本地） |
| **企业 / Azure** | Azure OpenAI 加自定义端点 |
| **中国 AI 模型** | z.ai（GLM）、Kimi/Moonshot（`kimi-coding` 或 `kimi-coding-cn`）、MiniMax、小米 MiMo 或腾讯 TokenHub（一等提供商） |

:::tip
可以随时使用 `hermes model` 切换提供商——无需重启。无论使用哪个提供商，你的对话历史、记忆和技能都会保留。
:::

## 可选 API Key

| 功能 | 提供商 | 环境变量 |
|---------|----------|--------------|
| 网页抓取 | [Firecrawl](https://firecrawl.dev/) | `FIRECRAWL_API_KEY`、`FIRECRAWL_API_URL` |
| 浏览器自动化 | [Browserbase](https://browserbase.com/) | `BROWSERBASE_API_KEY`、`BROWSERBASE_PROJECT_ID` |
| 图像生成 | [FAL](https://fal.ai/) | `FAL_KEY` |
| 高级 TTS 语音 | [ElevenLabs](https://elevenlabs.io/) | `ELEVENLABS_API_KEY` |
| OpenAI TTS + 语音转录 | [OpenAI](https://platform.openai.com/api-keys) | `VOICE_TOOLS_OPENAI_KEY` |
| Mistral TTS + 语音转录 | [Mistral](https://console.mistral.ai/) | `MISTRAL_API_KEY` |
| 跨会话用户建模 | [Honcho](https://honcho.dev/) | `HONCHO_API_KEY` |
| 语义长期记忆 | [Supermemory](https://supermemory.ai) | `SUPERMEMORY_API_KEY` |

### 自托管 Firecrawl

默认情况下，Hermes 使用 [Firecrawl 云 API](https://firecrawl.dev/) 进行网页搜索和抓取。如果你希望在本地运行 Firecrawl，可以将 Hermes 指向自托管实例。完整配置说明参见 Firecrawl 的 [SELF_HOST.md](https://github.com/firecrawl/firecrawl/blob/main/SELF_HOST.md)。

**优势：** 无需 API key，无速率限制，无按页计费，完全数据主权。

**劣势：** 云版本使用 Firecrawl 专有的"Fire-engine"进行高级反爬虫绕过（Cloudflare、CAPTCHA、IP 轮换）。自托管版本使用基础 fetch + Playwright，某些受保护的网站可能失败。搜索使用 DuckDuckGo 而非 Google。

**配置步骤：**

1. 克隆并启动 Firecrawl Docker 栈（5 个容器：API、Playwright、Redis、RabbitMQ、PostgreSQL——需要约 4-8 GB RAM）：
   ```bash
   git clone https://github.com/firecrawl/firecrawl
   cd firecrawl
   # 在 .env 中设置：USE_DB_AUTHENTICATION=false, HOST=0.0.0.0, PORT=3002
   docker compose up -d
   ```

2. 将 Hermes 指向你的实例（无需 API key）：
   ```bash
   hermes config set FIRECRAWL_API_URL http://localhost:3002
   ```

如果你的自托管实例启用了认证，也可以同时设置 `FIRECRAWL_API_KEY` 和 `FIRECRAWL_API_URL`。

## OpenRouter 提供商路由

使用 OpenRouter 时，可以控制请求如何在提供商之间路由。在 `~/.hermes/config.yaml` 中添加 `provider_routing` 节：

```yaml
provider_routing:
  sort: "throughput"          # "price"（默认）、"throughput" 或 "latency"
  # only: ["anthropic"]      # 仅使用这些提供商
  # ignore: ["deepinfra"]    # 跳过这些提供商
  # order: ["anthropic", "google"]  # 按此顺序尝试提供商
  # require_parameters: true  # 仅使用支持所有请求参数的提供商
  # data_collection: "deny"   # 排除可能存储/训练数据的提供商
```

**快捷方式：** 在任意模型名称后附加 `:nitro` 进行吞吐量排序（如 `anthropic/claude-sonnet-4:nitro`），或附加 `:floor` 进行价格排序。

## OpenRouter Pareto Code 路由器

OpenRouter 提供一个实验性编程模型路由器 `openrouter/pareto-code`，自动将请求路由到满足编程质量标准的最便宜模型（按 [Artificial Analysis](https://artificialanalysis.ai/) 排名）。选择此模型并在 `~/.hermes/config.yaml` 中调整 `min_coding_score` 参数：

```yaml
model:
  provider: openrouter
  model: openrouter/pareto-code

openrouter:
  min_coding_score: 0.65   # 0.0–1.0；越高 = 越强（越贵）的编程模型。默认 0.65。
```

说明：

- `min_coding_score` **仅**在 `model.model` 为 `openrouter/pareto-code` 时发送。对其他任何模型该值无效。
- 设置为空字符串（或删除该行）让 OpenRouter 选择最强的可用编程模型——这是省略 plugins 块时的文档行为。
- 在给定日期内，按分数选择是确定性的，但随着 Pareto 前沿移动（新模型、基准更新），实际选择的模型可能变化。
- 参见 OpenRouter 的 [Pareto Router 文档](https://openrouter.ai/docs/guides/routing/routers/pareto-router) 了解完整路由器行为。
- 要将 Pareto Code 路由器用于特定**辅助任务**（压缩、视觉等）而非主智能体，在该任务下设置 `extra_body.plugins`——参见[辅助模型 → OpenRouter 路由与辅助任务的 Pareto Code](/user-guide/configuration#openrouter-routing--pareto-code-for-auxiliary-tasks)。

## 故障转移提供商

配置一个备用提供商链，当主模型失败时（速率限制、服务器错误、认证失败）Hermes 按顺序尝试。规范格式是顶级 `fallback_providers:` 列表：

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
  - provider: anthropic
    model: claude-sonnet-4
    # base_url: http://localhost:8000/v1    # 可选，用于自定义端点
    # api_mode: chat_completions           # 可选覆盖
```

为向后兼容，旧版单对 `fallback_model:` 字典仍被接受：

```yaml
fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

激活时，故障转移在不丢失对话的情况下中途切换模型和提供商。链按条目逐一尝试；每个会话激活一次。

支持的提供商：`openrouter`、`nous`、`openai-codex`、`copilot`、`copilot-acp`、`anthropic`、`gemini`、`google-gemini-cli`、`qwen-oauth`、`huggingface`、`zai`、`kimi-coding`、`kimi-coding-cn`、`minimax`、`minimax-cn`、`minimax-oauth`、`deepseek`、`nvidia`、`xai`、`xai-oauth`、`ollama-cloud`、`bedrock`、`ai-gateway`、`azure-foundry`、`opencode-zen`、`opencode-go`、`kilocode`、`xiaomi`、`arcee`、`gmi`、`stepfun`、`lmstudio`、`alibaba`、`alibaba-coding-plan`、`tencent-tokenhub`、`custom`。

:::tip
故障转移仅通过 `config.yaml` 配置——或通过 `hermes fallback` 交互式配置。有关触发时机、链推进方式以及与辅助任务和委托的交互，参见[故障转移提供商](/user-guide/features/fallback-providers)。
:::

---

## 另请参阅

- [配置](/user-guide/configuration) — 通用配置（目录结构、配置优先级、终端后端、记忆、压缩等）
- [环境变量](/reference/environment-variables) — 所有环境变量的完整参考