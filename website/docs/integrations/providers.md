---
title: "AI Providers"
sidebar_label: "AI Providers"
sidebar_position: 1
---

# AI Providers

This page covers setting up inference providers for Hermes Agent — from cloud APIs like OpenRouter and Anthropic, to self-hosted endpoints like Ollama and vLLM, to advanced routing and fallback configurations. You need at least one provider configured to use Hermes.

## Inference Providers

You need at least one way to connect to an LLM. Use `hermes model` to switch providers and models interactively, or configure directly:

| Provider | Setup |
|----------|-------|
| **Nous Portal** | `hermes model` (OAuth, subscription-based) |
| **OpenAI Codex** | `hermes model` (ChatGPT OAuth, uses Codex models) |
| **GitHub Copilot** | `hermes model` (OAuth device code flow, `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, or `gh auth token`) |
| **GitHub Copilot ACP** | `hermes model` (spawns local `copilot --acp --stdio`) |
| **Anthropic** | `hermes model` (Claude Pro/Max via Claude Code auth, Anthropic API key, or manual setup-token) |
| **OpenRouter** | `OPENROUTER_API_KEY` in `~/.hermes/.env` |
| **AI Gateway** | `AI_GATEWAY_API_KEY` in `~/.hermes/.env` (provider: `ai-gateway`) |
| **z.ai / GLM** | `GLM_API_KEY` in `~/.hermes/.env` (provider: `zai`) |
| **Kimi / Moonshot** | `KIMI_API_KEY` in `~/.hermes/.env` (provider: `kimi-coding`) |
| **MiniMax** | `MINIMAX_API_KEY` in `~/.hermes/.env` (provider: `minimax`) |
| **MiniMax China** | `MINIMAX_CN_API_KEY` in `~/.hermes/.env` (provider: `minimax-cn`) |
| **Alibaba Cloud** | `DASHSCOPE_API_KEY` in `~/.hermes/.env` (provider: `alibaba`, aliases: `dashscope`, `qwen`) |
| **Kilo Code** | `KILOCODE_API_KEY` in `~/.hermes/.env` (provider: `kilocode`) |
| **OpenCode Zen** | `OPENCODE_ZEN_API_KEY` in `~/.hermes/.env` (provider: `opencode-zen`) |
| **OpenCode Go** | `OPENCODE_GO_API_KEY` in `~/.hermes/.env` (provider: `opencode-go`) |
| **DeepSeek** | `DEEPSEEK_API_KEY` in `~/.hermes/.env` (provider: `deepseek`) |
| **Hugging Face** | `HF_TOKEN` in `~/.hermes/.env` (provider: `huggingface`, aliases: `hf`) |
| **Custom Endpoint** | `hermes model` (saved in `config.yaml`) or `OPENAI_BASE_URL` + `OPENAI_API_KEY` in `~/.hermes/.env` |

:::tip Model key alias
In the `model:` config section, you can use either `default:` or `model:` as the key name for your model ID. Both `model: { default: my-model }` and `model: { model: my-model }` work identically.
:::

:::info Codex Note
The OpenAI Codex provider authenticates via device code (open a URL, enter a code). Hermes stores the resulting credentials in its own auth store under `~/.hermes/auth.json` and can import existing Codex CLI credentials from `~/.codex/auth.json` when present. No Codex CLI installation is required.
:::

:::warning
Even when using Nous Portal, Codex, or a custom endpoint, some tools (vision, web summarization, MoA) use a separate "auxiliary" model — by default Gemini Flash via OpenRouter. An `OPENROUTER_API_KEY` enables these tools automatically. You can also configure which model and provider these tools use — see [Auxiliary Models](/docs/user-guide/configuration#auxiliary-models).
:::

### Anthropic (Native)

Use Claude models directly through the Anthropic API — no OpenRouter proxy needed. Supports three auth methods:

```bash
# With an API key (pay-per-token)
export ANTHROPIC_API_KEY=***
hermes chat --provider anthropic --model claude-sonnet-4-6

# Preferred: authenticate through `hermes model`
# Hermes will use Claude Code's credential store directly when available
hermes model

# Manual override with a setup-token (fallback / legacy)
export ANTHROPIC_TOKEN=***  # setup-token or manual OAuth token
hermes chat --provider anthropic

# Auto-detect Claude Code credentials (if you already use Claude Code)
hermes chat --provider anthropic  # reads Claude Code credential files automatically
```

When you choose Anthropic OAuth through `hermes model`, Hermes prefers Claude Code's own credential store over copying the token into `~/.hermes/.env`. That keeps refreshable Claude credentials refreshable.

Or set it permanently:
```yaml
model:
  provider: "anthropic"
  default: "claude-sonnet-4-6"
```

:::tip Aliases
`--provider claude` and `--provider claude-code` also work as shorthand for `--provider anthropic`.
:::

### GitHub Copilot

Hermes supports GitHub Copilot as a first-class provider with two modes:

**`copilot` — Direct Copilot API** (recommended). Uses your GitHub Copilot subscription to access GPT-5.x, Claude, Gemini, and other models through the Copilot API.

```bash
hermes chat --provider copilot --model gpt-5.4
```

**Authentication options** (checked in this order):

1. `COPILOT_GITHUB_TOKEN` environment variable
2. `GH_TOKEN` environment variable
3. `GITHUB_TOKEN` environment variable
4. `gh auth token` CLI fallback

If no token is found, `hermes model` offers an **OAuth device code login** — the same flow used by the Copilot CLI and opencode.

:::warning Token types
The Copilot API does **not** support classic Personal Access Tokens (`ghp_*`). Supported token types:

| Type | Prefix | How to get |
|------|--------|------------|
| OAuth token | `gho_` | `hermes model` → GitHub Copilot → Login with GitHub |
| Fine-grained PAT | `github_pat_` | GitHub Settings → Developer settings → Fine-grained tokens (needs **Copilot Requests** permission) |
| GitHub App token | `ghu_` | Via GitHub App installation |

If your `gh auth token` returns a `ghp_*` token, use `hermes model` to authenticate via OAuth instead.
:::

**API routing**: GPT-5+ models (except `gpt-5-mini`) automatically use the Responses API. All other models (GPT-4o, Claude, Gemini, etc.) use Chat Completions. Models are auto-detected from the live Copilot catalog.

**`copilot-acp` — Copilot ACP agent backend**. Spawns the local Copilot CLI as a subprocess:

```bash
hermes chat --provider copilot-acp --model copilot-acp
# Requires the GitHub Copilot CLI in PATH and an existing `copilot login` session
```

**Permanent config:**
```yaml
model:
  provider: "copilot"
  default: "gpt-5.4"
```

| Environment variable | Description |
|---------------------|-------------|
| `COPILOT_GITHUB_TOKEN` | GitHub token for Copilot API (first priority) |
| `HERMES_COPILOT_ACP_COMMAND` | Override the Copilot CLI binary path (default: `copilot`) |
| `HERMES_COPILOT_ACP_ARGS` | Override ACP args (default: `--acp --stdio`) |

### First-Class Chinese AI Providers

These providers have built-in support with dedicated provider IDs. Set the API key and use `--provider` to select:

```bash
# z.ai / ZhipuAI GLM
hermes chat --provider zai --model glm-4-plus
# Requires: GLM_API_KEY in ~/.hermes/.env

# Kimi / Moonshot AI
hermes chat --provider kimi-coding --model moonshot-v1-auto
# Requires: KIMI_API_KEY in ~/.hermes/.env

# MiniMax (global endpoint)
hermes chat --provider minimax --model MiniMax-M2.7
# Requires: MINIMAX_API_KEY in ~/.hermes/.env

# MiniMax (China endpoint)
hermes chat --provider minimax-cn --model MiniMax-M2.7
# Requires: MINIMAX_CN_API_KEY in ~/.hermes/.env

# Alibaba Cloud / DashScope (Qwen models)
hermes chat --provider alibaba --model qwen3.5-plus
# Requires: DASHSCOPE_API_KEY in ~/.hermes/.env
```

Or set the provider permanently in `config.yaml`:
```yaml
model:
  provider: "zai"       # or: kimi-coding, minimax, minimax-cn, alibaba
  default: "glm-4-plus"
```

Base URLs can be overridden with `GLM_BASE_URL`, `KIMI_BASE_URL`, `MINIMAX_BASE_URL`, `MINIMAX_CN_BASE_URL`, or `DASHSCOPE_BASE_URL` environment variables.

### Hugging Face Inference Providers

[Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers) routes to 20+ open models through a unified OpenAI-compatible endpoint (`router.huggingface.co/v1`). Requests are automatically routed to the fastest available backend (Groq, Together, SambaNova, etc.) with automatic failover.

```bash
# Use any available model
hermes chat --provider huggingface --model Qwen/Qwen3-235B-A22B-Thinking-2507
# Requires: HF_TOKEN in ~/.hermes/.env

# Short alias
hermes chat --provider hf --model deepseek-ai/DeepSeek-V3.2
```

Or set it permanently in `config.yaml`:
```yaml
model:
  provider: "huggingface"
  default: "Qwen/Qwen3-235B-A22B-Thinking-2507"
```

Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — make sure to enable the "Make calls to Inference Providers" permission. Free tier included ($0.10/month credit, no markup on provider rates).

You can append routing suffixes to model names: `:fastest` (default), `:cheapest`, or `:provider_name` to force a specific backend.

The base URL can be overridden with `HF_BASE_URL`.

## Custom & Self-Hosted LLM Providers

Hermes Agent works with **any OpenAI-compatible API endpoint**. If a server implements `/v1/chat/completions`, you can point Hermes at it. This means you can use local models, GPU inference servers, multi-provider routers, or any third-party API.

### General Setup

Three ways to configure a custom endpoint:

**Interactive setup (recommended):**
```bash
hermes model
# Select "Custom endpoint (self-hosted / VLLM / etc.)"
# Enter: API base URL, API key, Model name
```

**Manual config (`config.yaml`):**
```yaml
# In ~/.hermes/config.yaml
model:
  default: your-model-name
  provider: custom
  base_url: http://localhost:8000/v1
  api_key: your-key-or-leave-empty-for-local
```

**Environment variables (`.env` file):**
```bash
# Add to ~/.hermes/.env
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=your-key     # Any non-empty string for local servers
LLM_MODEL=your-model-name
```

All three approaches end up in the same runtime path. `hermes model` persists provider, model, and base URL to `config.yaml` so later sessions keep using that endpoint even if env vars are not set.

### Switching Models with `/model`

Once a custom endpoint is configured, you can switch models mid-session:

```
/model custom:qwen-2.5          # Switch to a model on your custom endpoint
/model custom                    # Auto-detect the model from the endpoint
/model openrouter:claude-sonnet-4 # Switch back to a cloud provider
```

If you have **named custom providers** configured (see below), use the triple syntax:

```
/model custom:local:qwen-2.5    # Use the "local" custom provider with model qwen-2.5
/model custom:work:llama3       # Use the "work" custom provider with llama3
```

When switching providers, Hermes persists the base URL and provider to config so the change survives restarts. When switching away from a custom endpoint to a built-in provider, the stale base URL is automatically cleared.

:::tip
`/model custom` (bare, no model name) queries your endpoint's `/models` API and auto-selects the model if exactly one is loaded. Useful for local servers running a single model.
:::

Everything below follows this same pattern — just change the URL, key, and model name.

---

### Ollama — Local Models, Zero Config

[Ollama](https://ollama.com/) runs open-weight models locally with one command. Best for: quick local experimentation, privacy-sensitive work, offline use.

```bash
# Install and run a model
ollama pull llama3.1:70b
ollama serve   # Starts on port 11434

# Configure Hermes
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama           # Any non-empty string
LLM_MODEL=llama3.1:70b
```

Ollama's OpenAI-compatible endpoint supports chat completions, streaming, and tool calling (for supported models). No GPU required for smaller models — Ollama handles CPU inference automatically.

:::tip
List available models with `ollama list`. Pull any model from the [Ollama library](https://ollama.com/library) with `ollama pull <model>`.
:::

---

### vLLM — High-Performance GPU Inference

[vLLM](https://docs.vllm.ai/) is the standard for production LLM serving. Best for: maximum throughput on GPU hardware, serving large models, continuous batching.

```bash
# Start vLLM server
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --port 8000 \
  --tensor-parallel-size 2    # Multi-GPU

# Configure Hermes
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=dummy
LLM_MODEL=meta-llama/Llama-3.1-70B-Instruct
```

vLLM supports tool calling, structured output, and multi-modal models. Use `--enable-auto-tool-choice` and `--tool-call-parser hermes` for Hermes-format tool calling with NousResearch models.

---

### SGLang — Fast Serving with RadixAttention

[SGLang](https://github.com/sgl-project/sglang) is an alternative to vLLM with RadixAttention for KV cache reuse. Best for: multi-turn conversations (prefix caching), constrained decoding, structured output.

```bash
# Start SGLang server
pip install "sglang[all]"
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --port 8000 \
  --tp 2

# Configure Hermes
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=dummy
LLM_MODEL=meta-llama/Llama-3.1-70B-Instruct
```

---

### llama.cpp / llama-server — CPU & Metal Inference

[llama.cpp](https://github.com/ggml-org/llama.cpp) runs quantized models on CPU, Apple Silicon (Metal), and consumer GPUs. Best for: running models without a datacenter GPU, Mac users, edge deployment.

```bash
# Build and start llama-server
cmake -B build && cmake --build build --config Release
./build/bin/llama-server \
  -m models/llama-3.1-8b-instruct-Q4_K_M.gguf \
  --port 8080 --host 0.0.0.0

# Configure Hermes
OPENAI_BASE_URL=http://localhost:8080/v1
OPENAI_API_KEY=dummy
LLM_MODEL=llama-3.1-8b-instruct
```

:::tip
Download GGUF models from [Hugging Face](https://huggingface.co/models?library=gguf). Q4_K_M quantization offers the best balance of quality vs. memory usage.
:::

---

### LiteLLM Proxy — Multi-Provider Gateway

[LiteLLM](https://docs.litellm.ai/) is an OpenAI-compatible proxy that unifies 100+ LLM providers behind a single API. Best for: switching between providers without config changes, load balancing, fallback chains, budget controls.

```bash
# Install and start
pip install "litellm[proxy]"
litellm --model anthropic/claude-sonnet-4 --port 4000

# Or with a config file for multiple models:
litellm --config litellm_config.yaml --port 4000

# Configure Hermes
OPENAI_BASE_URL=http://localhost:4000/v1
OPENAI_API_KEY=sk-your-litellm-key
LLM_MODEL=anthropic/claude-sonnet-4
```

Example `litellm_config.yaml` with fallback:
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

### ClawRouter — Cost-Optimized Routing

[ClawRouter](https://github.com/BlockRunAI/ClawRouter) by BlockRunAI is a local routing proxy that auto-selects models based on query complexity. It classifies requests across 14 dimensions and routes to the cheapest model that can handle the task. Payment is via USDC cryptocurrency (no API keys).

```bash
# Install and start
npx @blockrun/clawrouter    # Starts on port 8402

# Configure Hermes
OPENAI_BASE_URL=http://localhost:8402/v1
OPENAI_API_KEY=dummy
LLM_MODEL=blockrun/auto     # or: blockrun/eco, blockrun/premium, blockrun/agentic
```

Routing profiles:
| Profile | Strategy | Savings |
|---------|----------|---------|
| `blockrun/auto` | Balanced quality/cost | 74-100% |
| `blockrun/eco` | Cheapest possible | 95-100% |
| `blockrun/premium` | Best quality models | 0% |
| `blockrun/free` | Free models only | 100% |
| `blockrun/agentic` | Optimized for tool use | varies |

:::note
ClawRouter requires a USDC-funded wallet on Base or Solana for payment. All requests route through BlockRun's backend API. Run `npx @blockrun/clawrouter doctor` to check wallet status.
:::

---

### Other Compatible Providers

Any service with an OpenAI-compatible API works. Some popular options:

| Provider | Base URL | Notes |
|----------|----------|-------|
| [Together AI](https://together.ai) | `https://api.together.xyz/v1` | Cloud-hosted open models |
| [Groq](https://groq.com) | `https://api.groq.com/openai/v1` | Ultra-fast inference |
| [DeepSeek](https://deepseek.com) | `https://api.deepseek.com/v1` | DeepSeek models |
| [Fireworks AI](https://fireworks.ai) | `https://api.fireworks.ai/inference/v1` | Fast open model hosting |
| [Cerebras](https://cerebras.ai) | `https://api.cerebras.ai/v1` | Wafer-scale chip inference |
| [Mistral AI](https://mistral.ai) | `https://api.mistral.ai/v1` | Mistral models |
| [OpenAI](https://openai.com) | `https://api.openai.com/v1` | Direct OpenAI access |
| [Azure OpenAI](https://azure.microsoft.com) | `https://YOUR.openai.azure.com/` | Enterprise OpenAI |
| [LocalAI](https://localai.io) | `http://localhost:8080/v1` | Self-hosted, multi-model |
| [Jan](https://jan.ai) | `http://localhost:1337/v1` | Desktop app with local models |

```bash
# Example: Together AI
OPENAI_BASE_URL=https://api.together.xyz/v1
OPENAI_API_KEY=your-together-key
LLM_MODEL=meta-llama/Llama-3.1-70B-Instruct-Turbo
```

---

### Context Length Detection

Hermes uses a multi-source resolution chain to detect the correct context window for your model and provider:

1. **Config override** — `model.context_length` in config.yaml (highest priority)
2. **Custom provider per-model** — `custom_providers[].models.<id>.context_length`
3. **Persistent cache** — previously discovered values (survives restarts)
4. **Endpoint `/models`** — queries your server's API (local/custom endpoints)
5. **Anthropic `/v1/models`** — queries Anthropic's API for `max_input_tokens` (API-key users only)
6. **OpenRouter API** — live model metadata from OpenRouter
7. **Nous Portal** — suffix-matches Nous model IDs against OpenRouter metadata
8. **[models.dev](https://models.dev)** — community-maintained registry with provider-specific context lengths for 3800+ models across 100+ providers
9. **Fallback defaults** — broad model family patterns (128K default)

For most setups this works out of the box. The system is provider-aware — the same model can have different context limits depending on who serves it (e.g., `claude-opus-4.6` is 1M on Anthropic direct but 128K on GitHub Copilot).

To set the context length explicitly, add `context_length` to your model config:

```yaml
model:
  default: "qwen3.5:9b"
  base_url: "http://localhost:8080/v1"
  context_length: 131072  # tokens
```

For custom endpoints, you can also set context length per model:

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

`hermes model` will prompt for context length when configuring a custom endpoint. Leave it blank for auto-detection.

:::tip When to set this manually
- You're using Ollama with a custom `num_ctx` that's lower than the model's maximum
- You want to limit context below the model's maximum (e.g., 8k on a 128k model to save VRAM)
- You're running behind a proxy that doesn't expose `/v1/models`
:::

---

### Named Custom Providers

If you work with multiple custom endpoints (e.g., a local dev server and a remote GPU server), you can define them as named custom providers in `config.yaml`:

```yaml
custom_providers:
  - name: local
    base_url: http://localhost:8080/v1
    # api_key omitted — Hermes uses "no-key-required" for keyless local servers
  - name: work
    base_url: https://gpu-server.internal.corp/v1
    api_key: corp-api-key
    api_mode: chat_completions   # optional, auto-detected from URL
  - name: anthropic-proxy
    base_url: https://proxy.example.com/anthropic
    api_key: proxy-key
    api_mode: anthropic_messages  # for Anthropic-compatible proxies
```

Switch between them mid-session with the triple syntax:

```
/model custom:local:qwen-2.5       # Use the "local" endpoint with qwen-2.5
/model custom:work:llama3-70b      # Use the "work" endpoint with llama3-70b
/model custom:anthropic-proxy:claude-sonnet-4  # Use the proxy
```

You can also select named custom providers from the interactive `hermes model` menu.

---

### Choosing the Right Setup

| Use Case | Recommended |
|----------|-------------|
| **Just want it to work** | OpenRouter (default) or Nous Portal |
| **Local models, easy setup** | Ollama |
| **Production GPU serving** | vLLM or SGLang |
| **Mac / no GPU** | Ollama or llama.cpp |
| **Multi-provider routing** | LiteLLM Proxy or OpenRouter |
| **Cost optimization** | ClawRouter or OpenRouter with `sort: "price"` |
| **Maximum privacy** | Ollama, vLLM, or llama.cpp (fully local) |
| **Enterprise / Azure** | Azure OpenAI with custom endpoint |
| **Chinese AI models** | z.ai (GLM), Kimi/Moonshot, or MiniMax (first-class providers) |

:::tip
You can switch between providers at any time with `hermes model` — no restart required. Your conversation history, memory, and skills carry over regardless of which provider you use.
:::

## Optional API Keys

| Feature | Provider | Env Variable |
|---------|----------|--------------|
| Web scraping | [Firecrawl](https://firecrawl.dev/) | `FIRECRAWL_API_KEY`, `FIRECRAWL_API_URL` |
| Browser automation | [Browserbase](https://browserbase.com/) | `BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID` |
| Image generation | [FAL](https://fal.ai/) | `FAL_KEY` |
| Premium TTS voices | [ElevenLabs](https://elevenlabs.io/) | `ELEVENLABS_API_KEY` |
| OpenAI TTS + voice transcription | [OpenAI](https://platform.openai.com/api-keys) | `VOICE_TOOLS_OPENAI_KEY` |
| RL Training | [Tinker](https://tinker-console.thinkingmachines.ai/) + [WandB](https://wandb.ai/) | `TINKER_API_KEY`, `WANDB_API_KEY` |
| Cross-session user modeling | [Honcho](https://honcho.dev/) | `HONCHO_API_KEY` |

### Self-Hosting Firecrawl

By default, Hermes uses the [Firecrawl cloud API](https://firecrawl.dev/) for web search and scraping. If you prefer to run Firecrawl locally, you can point Hermes at a self-hosted instance instead. See Firecrawl's [SELF_HOST.md](https://github.com/firecrawl/firecrawl/blob/main/SELF_HOST.md) for complete setup instructions.

**What you get:** No API key required, no rate limits, no per-page costs, full data sovereignty.

**What you lose:** The cloud version uses Firecrawl's proprietary "Fire-engine" for advanced anti-bot bypassing (Cloudflare, CAPTCHAs, IP rotation). Self-hosted uses basic fetch + Playwright, so some protected sites may fail. Search uses DuckDuckGo instead of Google.

**Setup:**

1. Clone and start the Firecrawl Docker stack (5 containers: API, Playwright, Redis, RabbitMQ, PostgreSQL — requires ~4-8 GB RAM):
   ```bash
   git clone https://github.com/firecrawl/firecrawl
   cd firecrawl
   # In .env, set: USE_DB_AUTHENTICATION=false, HOST=0.0.0.0, PORT=3002
   docker compose up -d
   ```

2. Point Hermes at your instance (no API key needed):
   ```bash
   hermes config set FIRECRAWL_API_URL http://localhost:3002
   ```

You can also set both `FIRECRAWL_API_KEY` and `FIRECRAWL_API_URL` if your self-hosted instance has authentication enabled.

## OpenRouter Provider Routing

When using OpenRouter, you can control how requests are routed across providers. Add a `provider_routing` section to `~/.hermes/config.yaml`:

```yaml
provider_routing:
  sort: "throughput"          # "price" (default), "throughput", or "latency"
  # only: ["anthropic"]      # Only use these providers
  # ignore: ["deepinfra"]    # Skip these providers
  # order: ["anthropic", "google"]  # Try providers in this order
  # require_parameters: true  # Only use providers that support all request params
  # data_collection: "deny"   # Exclude providers that may store/train on data
```

**Shortcuts:** Append `:nitro` to any model name for throughput sorting (e.g., `anthropic/claude-sonnet-4:nitro`), or `:floor` for price sorting.

## Fallback Model

Configure a backup provider:model that Hermes switches to automatically when your primary model fails (rate limits, server errors, auth failures):

```yaml
fallback_model:
  provider: openrouter                    # required
  model: anthropic/claude-sonnet-4        # required
  # base_url: http://localhost:8000/v1    # optional, for custom endpoints
  # api_key_env: MY_CUSTOM_KEY           # optional, env var name for custom endpoint API key
```

When activated, the fallback swaps the model and provider mid-session without losing your conversation. It fires **at most once** per session.

Supported providers: `openrouter`, `nous`, `openai-codex`, `copilot`, `anthropic`, `huggingface`, `zai`, `kimi-coding`, `minimax`, `minimax-cn`, `custom`.

:::tip
Fallback is configured exclusively through `config.yaml` — there are no environment variables for it. For full details on when it triggers, supported providers, and how it interacts with auxiliary tasks and delegation, see [Fallback Providers](/docs/user-guide/features/fallback-providers).
:::

## Smart Model Routing

Optional cheap-vs-strong routing lets Hermes keep your main model for complex work while sending very short/simple turns to a cheaper model.

```yaml
smart_model_routing:
  enabled: true
  max_simple_chars: 160
  max_simple_words: 28
  cheap_model:
    provider: openrouter
    model: google/gemini-2.5-flash
    # base_url: http://localhost:8000/v1  # optional custom endpoint
    # api_key_env: MY_CUSTOM_KEY          # optional env var name for that endpoint's API key
```

How it works:
- If a turn is short, single-line, and does not look code/tool/debug heavy, Hermes may route it to `cheap_model`
- If the turn looks complex, Hermes stays on your primary model/provider
- If the cheap route cannot be resolved cleanly, Hermes falls back to the primary model automatically

This is intentionally conservative. It is meant for quick, low-stakes turns like:
- short factual questions
- quick rewrites
- lightweight summaries

It will avoid routing prompts that look like:
- coding/debugging work
- tool-heavy requests
- long or multi-line analysis asks

Use this when you want lower latency or cost without fully changing your default model.

---

## See Also

- [Configuration](/docs/user-guide/configuration) — General configuration (directory structure, config precedence, terminal backends, memory, compression, and more)
- [Environment Variables](/docs/reference/environment-variables) — Complete reference of all environment variables
