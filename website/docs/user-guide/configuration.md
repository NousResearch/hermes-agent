---
sidebar_position: 2
title: "Configuration"
description: "Configure Hermes Agent — config.yaml, providers, models, API keys, and more"
---

# Configuration

All settings are stored in the `~/.hermes/` directory for easy access.

## Directory Structure

```text
~/.hermes/
├── config.yaml     # Settings (model, terminal, TTS, compression, etc.)
├── .env            # API keys and secrets
├── auth.json       # OAuth provider credentials (Nous Portal, etc.)
├── SOUL.md         # Optional: global persona (agent embodies this personality)
├── memories/       # Persistent memory (MEMORY.md, USER.md)
├── skills/         # Agent-created skills (managed via skill_manage tool)
├── cron/           # Scheduled jobs
├── sessions/       # Gateway sessions
└── logs/           # Logs (errors.log, gateway.log — secrets auto-redacted)
```

## Managing Configuration

```bash
hermes config              # View current configuration
hermes config edit         # Open config.yaml in your editor
hermes config set KEY VAL  # Set a specific value
hermes config check        # Check for missing options (after updates)
hermes config migrate      # Interactively add missing options

# Examples:
hermes config set model anthropic/claude-opus-4
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...  # Saves to .env
```

:::tip
The `hermes config set` command automatically routes values to the right file — API keys are saved to `.env`, everything else to `config.yaml`.
:::

## Configuration Precedence

Settings are resolved in this order (highest priority first):

1. **CLI arguments** — e.g., `hermes chat --model anthropic/claude-sonnet-4` (per-invocation override)
2. **`~/.hermes/config.yaml`** — the primary config file for all non-secret settings
3. **`~/.hermes/.env`** — fallback for env vars; **required** for secrets (API keys, tokens, passwords)
4. **Built-in defaults** — hardcoded safe defaults when nothing else is set

:::info Rule of Thumb
Secrets (API keys, bot tokens, passwords) go in `.env`. Everything else (model, terminal backend, compression settings, memory limits, toolsets) goes in `config.yaml`. When both are set, `config.yaml` wins for non-secret settings.
:::

## Inference Providers

You need at least one way to connect to an LLM. Use `hermes model` to switch providers and models interactively, or configure directly:

| Provider | Setup |
|----------|-------|
| **Nous Portal** | `hermes model` (OAuth, subscription-based) |
| **OpenAI Codex** | `hermes model` (ChatGPT OAuth, uses Codex models) |
| **Anthropic** | `hermes model` (Claude Pro/Max via Claude Code auth, Anthropic API key, or manual setup-token) |
| **OpenRouter** | `OPENROUTER_API_KEY` in `~/.hermes/.env` |
| **z.ai / GLM** | `GLM_API_KEY` in `~/.hermes/.env` (provider: `zai`) |
| **Kimi / Moonshot** | `KIMI_API_KEY` in `~/.hermes/.env` (provider: `kimi-coding`) |
| **MiniMax** | `MINIMAX_API_KEY` in `~/.hermes/.env` (provider: `minimax`) |
| **MiniMax China** | `MINIMAX_CN_API_KEY` in `~/.hermes/.env` (provider: `minimax-cn`) |
| **Custom Endpoint** | `OPENAI_BASE_URL` + `OPENAI_API_KEY` in `~/.hermes/.env` |

:::info Codex Note
The OpenAI Codex provider authenticates via device code (open a URL, enter a code). Hermes stores the resulting credentials in its own auth store under `~/.hermes/auth.json` and can import existing Codex CLI credentials from `~/.codex/auth.json` when present. No Codex CLI installation is required.
:::

:::warning
Even when using Nous Portal, Codex, or a custom endpoint, some tools (vision, web summarization, MoA) use a separate "auxiliary" model — by default Gemini Flash via OpenRouter. An `OPENROUTER_API_KEY` enables these tools automatically. You can also configure which model and provider these tools use — see [Auxiliary Models](#auxiliary-models) below.
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
hermes chat --provider minimax --model MiniMax-Text-01
# Requires: MINIMAX_API_KEY in ~/.hermes/.env

# MiniMax (China endpoint)
hermes chat --provider minimax-cn --model MiniMax-Text-01
# Requires: MINIMAX_CN_API_KEY in ~/.hermes/.env
```

Or set the provider permanently in `config.yaml`:
```yaml
model:
  provider: "zai"       # or: kimi-coding, minimax, minimax-cn
  default: "glm-4-plus"
```

Base URLs can be overridden with `GLM_BASE_URL`, `KIMI_BASE_URL`, `MINIMAX_BASE_URL`, or `MINIMAX_CN_BASE_URL` environment variables.

## Custom & Self-Hosted LLM Providers

Hermes Agent works with **any OpenAI-compatible API endpoint**. If a server implements `/v1/chat/completions`, you can point Hermes at it. This means you can use local models, GPU inference servers, multi-provider routers, or any third-party API.

### General Setup

Two ways to configure a custom endpoint:

**Interactive (recommended):**
```bash
hermes model
# Select "Custom endpoint (self-hosted / VLLM / etc.)"
# Enter: API base URL, API key, Model name
```

**Manual (`.env` file):**
```bash
# Add to ~/.hermes/.env
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=your-key-or-dummy
LLM_MODEL=your-model-name
```

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
pip install sglang[all]
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
pip install litellm[proxy]
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
| Web scraping | [Firecrawl](https://firecrawl.dev/) | `FIRECRAWL_API_KEY` |
| Browser automation | [Browserbase](https://browserbase.com/) | `BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID` |
| Image generation | [FAL](https://fal.ai/) | `FAL_KEY` |
| Premium TTS voices | [ElevenLabs](https://elevenlabs.io/) | `ELEVENLABS_API_KEY` |
| OpenAI TTS + voice transcription | [OpenAI](https://platform.openai.com/api-keys) | `VOICE_TOOLS_OPENAI_KEY` |
| RL Training | [Tinker](https://tinker-console.thinkingmachines.ai/) + [WandB](https://wandb.ai/) | `TINKER_API_KEY`, `WANDB_API_KEY` |
| Cross-session user modeling | [Honcho](https://honcho.dev/) | `HONCHO_API_KEY` |

### Self-Hosting Firecrawl

By default, Hermes uses the [Firecrawl cloud API](https://firecrawl.dev/) for web search and scraping. If you prefer to run Firecrawl locally, you can point Hermes at a self-hosted instance instead.

**What you get:** No API key required, no rate limits, no per-page costs, full data sovereignty.

**What you lose:** The cloud version uses Firecrawl's proprietary "Fire-engine" for advanced anti-bot bypassing (Cloudflare, CAPTCHAs, IP rotation). Self-hosted uses basic fetch + Playwright, so some protected sites may fail. Search uses DuckDuckGo instead of Google.

**Setup:**

1. Clone and start the Firecrawl Docker stack (5 containers: API, Playwright, Redis, RabbitMQ, PostgreSQL — requires ~4-8 GB RAM):
   ```bash
   git clone https://github.com/mendableai/firecrawl
   cd firecrawl
   # In .env, set: USE_DB_AUTHENTICATION=false
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

## Terminal Backend Configuration

Configure which environment the agent uses for terminal commands:

```yaml
terminal:
  backend: local    # or: docker, ssh, singularity, modal, daytona
  cwd: "."          # Working directory ("." = current dir)
  timeout: 180      # Command timeout in seconds

  # Docker-specific settings
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_volumes:                    # Share host directories with the container
    - "/home/user/projects:/workspace/projects"
    - "/home/user/data:/data:ro"     # :ro for read-only

  # Container resource limits (docker, singularity, modal, daytona)
  container_cpu: 1                   # CPU cores
  container_memory: 5120             # MB (default 5GB)
  container_disk: 51200              # MB (default 50GB)
  container_persistent: true         # Persist filesystem across sessions
```

### Common Terminal Backend Issues

If terminal commands fail immediately or the terminal tool is reported as disabled, check the following:

- **Local backend**
  - No special requirements. This is the safest default when you are just getting started.

- **Docker backend**
  - Ensure Docker Desktop (or the Docker daemon) is installed and running.
  - Hermes needs to be able to find the `docker` CLI. It checks your `$PATH` first and also probes common Docker Desktop install locations on macOS. Run:
    ```bash
    docker version
    ```
    If this fails, fix your Docker installation or switch back to the local backend:
    ```bash
    hermes config set terminal.backend local
    ```

- **SSH backend**
  - Both `TERMINAL_SSH_HOST` and `TERMINAL_SSH_USER` must be set, for example:
    ```bash
    export TERMINAL_ENV=ssh
    export TERMINAL_SSH_HOST=my-server.example.com
    export TERMINAL_SSH_USER=ubuntu
    ```
  - If either value is missing, Hermes will log a clear error and refuse to use the SSH backend.

- **Modal backend**
  - You need either a `MODAL_TOKEN_ID` environment variable or a `~/.modal.toml` config file.
  - If neither is present, the backend check fails and Hermes will report that the Modal backend is not available.

When in doubt, set `terminal.backend` back to `local` and verify that commands run there first.

### Docker Volume Mounts

When using the Docker backend, `docker_volumes` lets you share host directories with the container. Each entry uses standard Docker `-v` syntax: `host_path:container_path[:options]`.

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/projects:/workspace/projects"   # Read-write (default)
    - "/home/user/datasets:/data:ro"              # Read-only
    - "/home/user/outputs:/outputs"               # Agent writes, you read
```

This is useful for:
- **Providing files** to the agent (datasets, configs, reference code)
- **Receiving files** from the agent (generated code, reports, exports)
- **Shared workspaces** where both you and the agent access the same files

Can also be set via environment variable: `TERMINAL_DOCKER_VOLUMES='["/host:/container"]'` (JSON array).

See [Code Execution](features/code-execution.md) and the [Terminal section of the README](features/tools.md) for details on each backend.

## Memory Configuration

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
```

## Git Worktree Isolation

Enable isolated git worktrees for running multiple agents in parallel on the same repo:

```yaml
worktree: true    # Always create a worktree (same as hermes -w)
# worktree: false # Default — only when -w flag is passed
```

When enabled, each CLI session creates a fresh worktree under `.worktrees/` with its own branch. Agents can edit files, commit, push, and create PRs without interfering with each other. Clean worktrees are removed on exit; dirty ones are kept for manual recovery.

You can also list gitignored files to copy into worktrees via `.worktreeinclude` in your repo root:

```
# .worktreeinclude
.env
.venv/
node_modules/
```

## Context Compression

```yaml
compression:
  enabled: true
  threshold: 0.50              # Compress at 50% of context limit by default
  summary_model: "google/gemini-3-flash-preview"   # Model for summarization
  # summary_provider: "auto"   # "auto", "openrouter", "nous", "main"
```

The `summary_model` must support a context length at least as large as your main model's, since it receives the full middle section of the conversation for compression.

## Iteration Budget Pressure

When the agent is working on a complex task with many tool calls, it can burn through its iteration budget (default: 90 turns) without realizing it's running low. Budget pressure automatically warns the model as it approaches the limit:

| Threshold | Level | What the model sees |
|-----------|-------|---------------------|
| **70%** | Caution | `[BUDGET: 63/90. 27 iterations left. Start consolidating.]` |
| **90%** | Warning | `[BUDGET WARNING: 81/90. Only 9 left. Respond NOW.]` |

Warnings are injected into the last tool result's JSON (as a `_budget_warning` field) rather than as separate messages — this preserves prompt caching and doesn't disrupt the conversation structure.

```yaml
agent:
  max_turns: 90                # Max iterations per conversation turn (default: 90)
```

Budget pressure is enabled by default. The agent sees warnings naturally as part of tool results, encouraging it to consolidate its work and deliver a response before running out of iterations.

## Auxiliary Models

Hermes uses lightweight "auxiliary" models for side tasks like image analysis, web page summarization, and browser screenshot analysis. By default, these use **Gemini Flash** via OpenRouter or Nous Portal — you don't need to configure anything.

To use a different model, add an `auxiliary` section to `~/.hermes/config.yaml`:

```yaml
auxiliary:
  # Image analysis (vision_analyze tool + browser screenshots)
  vision:
    provider: "auto"           # "auto", "openrouter", "nous", "main"
    model: ""                  # e.g. "openai/gpt-4o", "google/gemini-2.5-flash"

  # Web page summarization + browser page text extraction
  web_extract:
    provider: "auto"
    model: ""                  # e.g. "google/gemini-2.5-flash"
```

### Changing the Vision Model

To use GPT-4o instead of Gemini Flash for image analysis:

```yaml
auxiliary:
  vision:
    model: "openai/gpt-4o"
```

Or via environment variable (in `~/.hermes/.env`):

```bash
AUXILIARY_VISION_MODEL=openai/gpt-4o
```

### Provider Options

| Provider | Description | Requirements |
|----------|-------------|-------------|
| `"auto"` | Best available (default). Vision tries OpenRouter → Nous → Codex. | — |
| `"openrouter"` | Force OpenRouter — routes to any model (Gemini, GPT-4o, Claude, etc.) | `OPENROUTER_API_KEY` |
| `"nous"` | Force Nous Portal | `hermes login` |
| `"codex"` | Force Codex OAuth (ChatGPT account). Supports vision (gpt-5.3-codex). | `hermes model` → Codex |
| `"main"` | Use your custom endpoint (`OPENAI_BASE_URL` + `OPENAI_API_KEY`). Works with OpenAI, local models, or any OpenAI-compatible API. | `OPENAI_BASE_URL` + `OPENAI_API_KEY` |

### Common Setups

**Using OpenAI API key for vision:**
```yaml
# In ~/.hermes/.env:
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=sk-...

auxiliary:
  vision:
    provider: "main"
    model: "gpt-4o"       # or "gpt-4o-mini" for cheaper
```

**Using OpenRouter for vision** (route to any model):
```yaml
auxiliary:
  vision:
    provider: "openrouter"
    model: "openai/gpt-4o"      # or "google/gemini-2.5-flash", etc.
```

**Using Codex OAuth** (ChatGPT Pro/Plus account — no API key needed):
```yaml
auxiliary:
  vision:
    provider: "codex"     # uses your ChatGPT OAuth token
    # model defaults to gpt-5.3-codex (supports vision)
```

**Using a local/self-hosted model:**
```yaml
auxiliary:
  vision:
    provider: "main"      # uses your OPENAI_BASE_URL endpoint
    model: "my-local-model"
```

:::tip
If you use Codex OAuth as your main model provider, vision works automatically — no extra configuration needed. Codex is included in the auto-detection chain for vision.
:::

:::warning
**Vision requires a multimodal model.** If you set `provider: "main"`, make sure your endpoint supports multimodal/vision — otherwise image analysis will fail.
:::

### Environment Variables

You can also configure auxiliary models via environment variables instead of `config.yaml`:

| Setting | Environment Variable |
|---------|---------------------|
| Vision provider | `AUXILIARY_VISION_PROVIDER` |
| Vision model | `AUXILIARY_VISION_MODEL` |
| Web extract provider | `AUXILIARY_WEB_EXTRACT_PROVIDER` |
| Web extract model | `AUXILIARY_WEB_EXTRACT_MODEL` |
| Compression provider | `CONTEXT_COMPRESSION_PROVIDER` |
| Compression model | `CONTEXT_COMPRESSION_MODEL` |

:::tip
Run `hermes config` to see your current auxiliary model settings. Overrides only show up when they differ from the defaults.
:::

## Reasoning Effort

Control how much "thinking" the model does before responding:

```yaml
agent:
  reasoning_effort: ""   # empty = medium (default). Options: xhigh (max), high, medium, low, minimal, none
```

When unset (default), reasoning effort defaults to "medium" — a balanced level that works well for most tasks. Setting a value overrides it — higher reasoning effort gives better results on complex tasks at the cost of more tokens and latency.

You can also change the reasoning effort at runtime with the `/reasoning` command:

```
/reasoning           # Show current effort level and display state
/reasoning high      # Set reasoning effort to high
/reasoning none      # Disable reasoning
/reasoning show      # Show model thinking above each response
/reasoning hide      # Hide model thinking
```

## TTS Configuration

```yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai"
  edge:
    voice: "en-US-AriaNeural"   # 322 voices, 74 languages
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
```

This controls both the `text_to_speech` tool and spoken replies in voice mode (`/voice tts` in the CLI or messaging gateway).

## Display Settings

```yaml
display:
  tool_progress: all      # off | new | all | verbose
  skin: default           # Built-in or custom CLI skin (see user-guide/features/skins)
  personality: "kawaii"  # Legacy cosmetic field still surfaced in some summaries
  compact: false          # Compact output mode (less whitespace)
  resume_display: full    # full (show previous messages on resume) | minimal (one-liner only)
  bell_on_complete: false # Play terminal bell when agent finishes (great for long tasks)
  show_reasoning: false   # Show model reasoning/thinking above each response (toggle with /reasoning show|hide)
```

| Mode | What you see |
|------|-------------|
| `off` | Silent — just the final response |
| `new` | Tool indicator only when the tool changes |
| `all` | Every tool call with a short preview (default) |
| `verbose` | Full args, results, and debug logs |

## Speech-to-Text (STT)

```yaml
stt:
  provider: "local"            # "local" | "groq" | "openai"
  local:
    model: "base"              # tiny, base, small, medium, large-v3
  openai:
    model: "whisper-1"         # whisper-1 | gpt-4o-mini-transcribe | gpt-4o-transcribe
  # model: "whisper-1"         # Legacy fallback key still respected
```

Provider behavior:

- `local` uses `faster-whisper` running on your machine. Install it separately with `pip install faster-whisper`.
- `groq` uses Groq's Whisper-compatible endpoint and reads `GROQ_API_KEY`.
- `openai` uses the OpenAI speech API and reads `VOICE_TOOLS_OPENAI_KEY`.

If the requested provider is unavailable, Hermes falls back automatically in this order: `local` → `groq` → `openai`.

Groq and OpenAI model overrides are environment-driven:

```bash
STT_GROQ_MODEL=whisper-large-v3-turbo
STT_OPENAI_MODEL=whisper-1
GROQ_BASE_URL=https://api.groq.com/openai/v1
STT_OPENAI_BASE_URL=https://api.openai.com/v1
```

## Voice Mode (CLI)

```yaml
voice:
  record_key: "ctrl+b"         # Push-to-talk key inside the CLI
  max_recording_seconds: 120    # Hard stop for long recordings
  auto_tts: false               # Enable spoken replies automatically when /voice on
  silence_threshold: 200        # RMS threshold for speech detection
  silence_duration: 3.0         # Seconds of silence before auto-stop
```

Use `/voice on` in the CLI to enable microphone mode, `record_key` to start/stop recording, and `/voice tts` to toggle spoken replies. See [Voice Mode](/docs/user-guide/features/voice-mode) for end-to-end setup and platform-specific behavior.

## Quick Commands

Define custom commands that run shell commands without invoking the LLM — zero token usage, instant execution. Especially useful from messaging platforms (Telegram, Discord, etc.) for quick server checks or utility scripts.

```yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  disk:
    type: exec
    command: df -h /
  update:
    type: exec
    command: cd ~/.hermes/hermes-agent && git pull && pip install -e .
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

Usage: type `/status`, `/disk`, `/update`, or `/gpu` in the CLI or any messaging platform. The command runs locally on the host and returns the output directly — no LLM call, no tokens consumed.

- **30-second timeout** — long-running commands are killed with an error message
- **Priority** — quick commands are checked before skill commands, so you can override skill names
- **Autocomplete** — quick commands are resolved at dispatch time and are not shown in the built-in slash-command autocomplete tables
- **Type** — only `exec` is supported (runs a shell command); other types show an error
- **Works everywhere** — CLI, Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant

## Human Delay

Simulate human-like response pacing in messaging platforms:

```yaml
human_delay:
  mode: "off"                  # off | natural | custom
  min_ms: 500                  # Minimum delay (custom mode)
  max_ms: 2000                 # Maximum delay (custom mode)
```

## Code Execution

Configure the sandboxed Python code execution tool:

```yaml
code_execution:
  timeout: 300                 # Max execution time in seconds
  max_tool_calls: 50           # Max tool calls within code execution
```

## Browser

Configure browser automation behavior:

```yaml
browser:
  inactivity_timeout: 120        # Seconds before auto-closing idle sessions
  record_sessions: false         # Auto-record browser sessions as WebM videos to ~/.hermes/browser_recordings/
```

## Checkpoints

Automatic filesystem snapshots before destructive file operations. See the [Checkpoints feature page](/docs/user-guide/features/checkpoints) for details.

```yaml
checkpoints:
  enabled: false                 # Enable automatic checkpoints (also: hermes --checkpoints)
  max_snapshots: 50              # Max checkpoints to keep per directory
```


## Delegation

Configure subagent behavior for the delegate tool:

```yaml
delegation:
  max_iterations: 50           # Max iterations per subagent
  default_toolsets:             # Toolsets available to subagents
    - terminal
    - file
    - web
  # model: "google/gemini-3-flash-preview"  # Override model (empty = inherit parent)
  # provider: "openrouter"                  # Override provider (empty = inherit parent)
```

**Subagent provider:model override:** By default, subagents inherit the parent agent's provider and model. Set `delegation.provider` and `delegation.model` to route subagents to a different provider:model pair — e.g., use a cheap/fast model for narrowly-scoped subtasks while your primary agent runs an expensive reasoning model.

The delegation provider uses the same credential resolution as CLI/gateway startup. All configured providers are supported: `openrouter`, `nous`, `zai`, `kimi-coding`, `minimax`, `minimax-cn`. When a provider is set, the system automatically resolves the correct base URL, API key, and API mode — no manual credential wiring needed.

**Precedence:** `delegation.provider` in config → parent provider (inherited). `delegation.model` in config → parent model (inherited). Setting just `model` without `provider` changes only the model name while keeping the parent's credentials (useful for switching models within the same provider like OpenRouter).

## Clarify

Configure the clarification prompt behavior:

```yaml
clarify:
  timeout: 120                 # Seconds to wait for user clarification response
```

## Context Files (SOUL.md, AGENTS.md)

Hermes uses two different context scopes:

| File | Purpose | Scope |
|------|---------|-------|
| `AGENTS.md` | Project-specific instructions, coding conventions | Working directory / project tree |
| `SOUL.md` | Default persona for this Hermes instance | `~/.hermes/SOUL.md` or `$HERMES_HOME/SOUL.md` |
| `.cursorrules` | Cursor IDE rules (also detected) | Working directory |
| `.cursor/rules/*.mdc` | Cursor rule files (also detected) | Working directory |

- **AGENTS.md** is hierarchical: if subdirectories also have AGENTS.md, all are combined.
- **SOUL.md** is now global to the Hermes instance and is loaded only from `HERMES_HOME`.
- Hermes automatically seeds a default `SOUL.md` if one does not already exist.
- An empty `SOUL.md` contributes nothing to the system prompt.
- All loaded context files are capped at 20,000 characters with smart truncation.

See also:
- [Personality & SOUL.md](/docs/user-guide/features/personality)
- [Context Files](/docs/user-guide/features/context-files)

## Working Directory

| Context | Default |
|---------|---------|
| **CLI (`hermes`)** | Current directory where you run the command |
| **Messaging gateway** | Home directory `~` (override with `MESSAGING_CWD`) |
| **Docker / Singularity / Modal / SSH** | User's home directory inside the container or remote machine |

Override the working directory:
```bash
# In ~/.hermes/.env or ~/.hermes/config.yaml:
MESSAGING_CWD=/home/myuser/projects    # Gateway sessions
TERMINAL_CWD=/workspace                # All terminal sessions
```
