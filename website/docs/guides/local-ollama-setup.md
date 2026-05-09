---
sidebar_position: 9
title: "Run Hermes Locally with Ollama — Zero API Cost"
description: "Step-by-step guide to running Hermes Agent entirely on your own machine with Ollama and open-weight models like Gemma 4, no cloud API keys or paid subscriptions needed"
---

# Run Hermes Locally with Ollama — Zero API Cost

## The Problem

Cloud LLM APIs charge per token. A heavy coding session can cost $5–20. For personal projects, learning, or privacy-sensitive work, that adds up — and you're sending every conversation to a third party.

## What This Guide Solves

You'll set up Hermes Agent running entirely on your own hardware, using [Ollama](https://ollama.com) as the model backend. No API keys, no subscriptions, no data leaving your machine. Once configured, Hermes works exactly like it does with OpenRouter or Anthropic — terminal commands, file editing, web browsing, delegation — but the model runs locally.

By the end, you'll have:

- Ollama serving one or more open-weight models
- Hermes connected to Ollama as a custom endpoint
- A working local agent that can edit files, run commands, and browse the web
- Optional: a Telegram/Discord bot powered entirely by your own hardware

## What You Need

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB (for 3B models) | 32+ GB (for 27B+ models) |
| **Storage** | 5 GB free | 30+ GB (for multiple models) |
| **CPU** | 4 cores | 8+ cores (AMD EPYC, Ryzen, Intel Xeon) |
| **GPU** | Not required | NVIDIA GPU with 8+ GB VRAM speeds things up significantly |

:::tip CPU-only works, but expect slower responses
Ollama runs on CPU-only servers. A 9B model on a modern 8-core CPU gives ~10 tokens/sec. A 31B model on CPU is slower (~2–5 tokens/sec) — each response takes 30–120 seconds, but it works. A GPU dramatically improves this. For CPU-only setups, widen the API timeout via the env var (it's not a `config.yaml` key):

```bash
# ~/.hermes/.env
HERMES_API_TIMEOUT=1800   # 30 minutes — generous for slow local models
```
:::

## Step 1: Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify it's running:

```bash
ollama --version
curl http://localhost:11434/api/tags   # Should return {"models":[]}
```

## Step 2: Pull a Model

Choose based on your hardware:

| Model | Size on Disk | RAM Needed | Tool Calling | Best For |
|-------|-------------|------------|:------------:|----------|
| `gemma4:31b` | ~20 GB | 24+ GB | Yes | Best quality — strong tool use and reasoning |
| `gemma2:27b` | ~16 GB | 20+ GB | No | Conversational tasks, no tool use |
| `gemma2:9b` | ~5 GB | 8+ GB | No | Fast chat, Q&A — cannot call tools |
| `llama3.2:3b` | ~2 GB | 4+ GB | No | Lightweight quick answers only |

:::warning Tool calling matters
Hermes is an **agentic** assistant — it edits files, runs commands, and browses the web through tool calls. Models without tool-call support can only chat; they can't take actions. For the full Hermes experience, use a model that supports tools (like `gemma4:31b`).
:::

Pull your chosen model:

```bash
ollama pull gemma4:31b
```

:::info Multiple models
You can pull several models and switch between them inside Hermes with `/model`. Ollama loads the active model into memory on demand and unloads idle ones automatically.
:::

Verify the model works:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:31b",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 50
  }'
```

You should see a JSON response with the model's reply.

## Step 3: Configure Hermes

Run the Hermes setup wizard:

```bash
hermes setup
```

When prompted for a provider, select **Custom Endpoint** and enter:

- **Base URL:** `http://localhost:11434/v1`
- **API Key:** Leave empty or type `no-key` (Ollama doesn't need one)
- **Model:** `gemma4:31b` (or whichever model you pulled)

Alternatively, edit `~/.hermes/config.yaml` directly:

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"
```

## Step 4: Start Using Hermes

```bash
hermes
```

That's it. You're now running a fully local agent. Try it out:

```
You: List all Python files in this directory and count the lines of code in each

You: Read the README.md and summarize what this project does

You: Create a Python script that fetches the weather for Ho Chi Minh City
```

Hermes will use the terminal tool, file operations, and your local model — no cloud calls.

## Step 5: Pick the Right Model for Your Task

Not every task needs the biggest model. Here's a practical guide:

| Task | Recommended Model | Why |
|------|-------------------|-----|
| File edits, code, terminal commands | `gemma4:31b` | Only model with reliable tool calling |
| Quick Q&A (no tool use needed) | `gemma2:9b` | Fast responses for conversational tasks |
| Lightweight chat | `llama3.2:3b` | Fastest, but very limited capabilities |

:::note
For full agentic work (editing files, running commands, browsing), `gemma4:31b` is currently the best local option with tool-call support. Check [Ollama's model library](https://ollama.com/library) for newer models — tool-calling support is expanding rapidly.
:::

Switch models on the fly inside a session:

```
/model gemma2:9b
```

## Step 6: Optimize for Speed

### Increase Ollama's Context Window

By default, Ollama uses a 2048-token context. For agentic work (tool calls, long conversations), you need more:

```bash
# Create a Modelfile that extends context
cat > /tmp/Modelfile << 'EOF'
FROM gemma4:31b
PARAMETER num_ctx 16384
EOF

ollama create gemma4-16k -f /tmp/Modelfile
```

Then update your Hermes config to use `gemma4-16k` as the model name.

### Keep the Model Loaded

By default, Ollama unloads models after 5 minutes of inactivity. For a persistent gateway bot, keep it loaded:

```bash
# Set keep-alive to 24 hours
curl http://localhost:11434/api/generate \
  -d '{"model": "gemma4:31b", "keep_alive": "24h"}'
```

Or set it globally in Ollama's environment:

```bash
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_KEEP_ALIVE=24h"
```

### Use GPU Offloading (If Available)

If you have an NVIDIA GPU, Ollama automatically offloads layers to it. Check with:

```bash
ollama ps   # Shows which model is loaded and how many GPU layers
```

For a 31B model on a 12 GB GPU, you'll get partial offload (~40 layers on GPU, rest on CPU), which still gives a significant speedup.

## Step 7: Run as a Gateway Bot (Optional)

Once Hermes works locally in the CLI, you can expose it as a Telegram or Discord bot — still running entirely on your hardware.

### Telegram

1. Create a bot via [@BotFather](https://t.me/BotFather) and get the token
2. Add to your `~/.hermes/config.yaml`:

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"

platforms:
  telegram:
    enabled: true
    token: "YOUR_TELEGRAM_BOT_TOKEN"
```

3. Start the gateway:

```bash
hermes gateway
```

Now message your bot on Telegram — it responds using your local model.

### Discord

1. Create a Discord application at [discord.com/developers](https://discord.com/developers/applications)
2. Add to config:

```yaml
platforms:
  discord:
    enabled: true
    token: "YOUR_DISCORD_BOT_TOKEN"
```

3. Start: `hermes gateway`

## Step 8: Set Up Fallbacks (Optional)

Local models can struggle with complex tasks. Set up a cloud fallback that only activates when the local model fails:

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"

fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

This way, 90% of your usage is free (local), and only the hard tasks hit the paid API.

## Troubleshooting

### "Connection refused" on startup

Ollama isn't running. Start it:

```bash
sudo systemctl start ollama
# or
ollama serve
```

### Slow responses

- **Check model size vs RAM:** If your model needs more RAM than available, it swaps to disk. Use a smaller model or add RAM.
- **Check `ollama ps`:** If no GPU layers are offloaded, responses are CPU-bound. This is normal for CPU-only servers.
- **Reduce context:** Large conversations slow down inference. Use `/compress` regularly, or set a lower compression threshold in config.

### Model doesn't follow tool calls

Smaller models (3B, 7B) sometimes ignore tool-call instructions and produce plain text instead of structured function calls. Solutions:

- **Use a bigger model** — `gemma4:31b` or `gemma2:27b` handle tool calls much better than 3B/7B models.
- **Hermes has auto-repair** — it detects malformed tool calls and attempts to fix them automatically.
- **Set up a fallback** — if the local model fails 3 times, Hermes falls back to a cloud provider.

### Context window errors

The default Ollama context (2048 tokens) is too small for agentic work. See [Step 6](#step-6-optimize-for-speed) to increase it.

## Cost Comparison

Here's what running locally saves compared to cloud APIs, based on a typical coding session (~100K tokens input, ~20K tokens output):

| Provider | Cost per Session | Monthly (daily use) |
|----------|-----------------|---------------------|
| Anthropic Claude Sonnet | ~$0.80 | ~$24 |
| OpenRouter (GPT-4o) | ~$0.60 | ~$18 |
| **Ollama (local)** | **$0.00** | **$0.00** |

Your only cost is electricity — roughly $0.01–0.05 per session depending on hardware.

## What Works Well Locally

- **File editing and code generation** — models 9B+ handle this well
- **Terminal commands** — Hermes wraps the command, runs it, reads output regardless of model
- **Web browsing** — the browser tool does the fetching; the model just interprets results
- **Cron jobs and scheduled tasks** — work identically to cloud setups
- **Multi-platform gateway** — Telegram, Discord, Slack all work with local models

## What's Better with Cloud Models

- **Very complex multi-step reasoning** — 70B+ or cloud models like Claude Opus are noticeably better
- **Long context windows** — cloud models offer 100K–1M tokens; local models are typically 8K–32K
- **Speed on large responses** — cloud inference is faster than CPU-only local for long generations

The sweet spot: use local for everyday tasks, set up a cloud fallback for the hard stuff.
