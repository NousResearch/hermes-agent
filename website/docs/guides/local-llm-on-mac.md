---
sidebar_position: 2
title: "Run Local LLMs on Mac"
description: "Set up a local OpenAI-compatible LLM server on macOS with llama.cpp or MLX, including model selection, memory optimization, and real benchmarks on Apple Silicon"
---

# Run Local LLMs on Mac

:::tip Desktop users: there's a one-click path
On the Hermes desktop app, **Settings → Providers → Local Models** installs
and manages a local llama.cpp server for you — model downloads, memory
fitting, and context sizing included. See [Local Models](../user-guide/local-models.md).
This guide is for manual setup: MLX, custom builds, or servers you want to
run yourself.
:::

This guide walks you through running a local LLM server on macOS with an OpenAI-compatible API. You get full privacy, zero API costs, and surprisingly good performance on Apple Silicon.

We cover three backends:

| Backend | Install | Best at | Format |
|---------|---------|---------|--------|
| **llama.cpp** | `brew install llama.cpp` | Fastest time-to-first-token, quantized KV cache for low memory | GGUF |
| **omlx** | [omlx.ai](https://omlx.ai) | Fastest token generation, native Metal optimization | MLX (safetensors) |
| **vllm-metal** | [Installation guide](https://docs.vllm.ai/projects/vllm-metal/en/latest/installation/) | vLLM's serving API on Apple Silicon | MLX (safetensors) |

All three expose an OpenAI-compatible `/v1/chat/completions` endpoint. Hermes works with any of them — just point it at `http://localhost:8080` or `http://localhost:8000`.

:::info Apple Silicon only
This guide targets Macs with Apple Silicon (M1 and later). Intel Macs will work with llama.cpp but without GPU acceleration — expect significantly slower performance.
:::

---

## Choosing a model

For getting started, we recommend **Qwen3.5-9B** — it's a strong reasoning model that fits comfortably in 8GB+ of unified memory with quantization.

| Variant | Size on disk | RAM needed (128K context) | Backend |
|---------|-------------|---------------------------|---------|
| Qwen3.5-9B-Q4_K_M (GGUF) | 5.3 GB | ~10 GB with quantized KV cache | llama.cpp |
| Qwen3.5-9B-mlx-lm-mxfp4 (MLX) | ~5 GB | ~12 GB | omlx |

**Memory rule of thumb:** model size + KV cache. A 9B Q4 model is ~5 GB. The KV cache at 128K context with Q4 quantization adds ~4-5 GB. With default (f16) KV cache, that balloons to ~16 GB. The quantized KV cache flags in llama.cpp are the key trick for memory-constrained systems.

For larger models (27B, 35B), you'll need 32 GB+ of unified memory. The 9B is the sweet spot for 8-16 GB machines.

---

## Option A: llama.cpp

llama.cpp is the most portable local LLM runtime. On macOS it uses Metal for GPU acceleration out of the box.

### Install

```bash
brew install llama.cpp
```

This gives you the `llama-server` command globally.

### Download the model

You need a GGUF-format model. The easiest source is Hugging Face via the `huggingface-cli`:

```bash
brew install huggingface-cli
```

Then download:

```bash
huggingface-cli download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir ~/models
```

:::tip Gated models
Some models on Hugging Face require authentication. Run `huggingface-cli login` first if you get a 401 or 404 error.
:::

### Start the server

```bash
llama-server -m ~/models/Qwen3.5-9B-Q4_K_M.gguf \
  -ngl 99 \
  -c 131072 \
  -np 1 \
  -fa on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --host 0.0.0.0
```

Here's what each flag does:

| Flag | Purpose |
|------|---------|
| `-ngl 99` | Offload all layers to GPU (Metal). Use a high number to ensure nothing stays on CPU. |
| `-c 131072` | Context window size (128K tokens). Reduce this if you're low on memory. |
| `-np 1` | Number of parallel slots. Keep at 1 for single-user use — more slots split your memory budget. |
| `-fa on` | Flash attention. Reduces memory usage and speeds up long-context inference. |
| `--cache-type-k q4_0` | Quantize the key cache to 4-bit. **This is the big memory saver.** |
| `--cache-type-v q4_0` | Quantize the value cache to 4-bit. Together with the above, this cuts KV cache memory by ~75% vs f16. |
| `--host 0.0.0.0` | Listen on all interfaces. Use `127.0.0.1` if you don't need network access. |

The server is ready when you see:

```
main: server is listening on http://0.0.0.0:8080
srv  update_slots: all slots are idle
```

### Memory optimization for constrained systems

The `--cache-type-k q4_0 --cache-type-v q4_0` flags are the most important optimization for systems with limited memory. Here's the impact at 128K context:

| KV cache type | KV cache memory (128K ctx, 9B model) |
|---------------|--------------------------------------|
| f16 (default) | ~16 GB |
| q8_0 | ~8 GB |
| **q4_0** | **~4 GB** |

On an 8 GB Mac, use `q4_0` KV cache and choose a smaller model that can still fit Hermes' 64K minimum context. On 16 GB, you can comfortably do 128K context. On 32 GB+, you can run larger models or multiple parallel slots.

If you're still running out of memory, reduce context only while staying at or above Hermes' 64K minimum; otherwise switch to a smaller model or smaller quantization (Q3_K_M instead of Q4_K_M).

### Test it

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }' | jq .choices[0].message.content
```

### Get the model name

If you forget the model name, query the models endpoint:

```bash
curl -s http://localhost:8080/v1/models | jq '.data[].id'
```

---

## Option B: MLX via omlx

[omlx](https://omlx.ai) is a macOS-native app that manages and serves MLX models. MLX is Apple's own machine learning framework, optimized specifically for Apple Silicon's unified memory architecture.

### Install

Download and install from [omlx.ai](https://omlx.ai). It provides a GUI for model management and a built-in server.

### Download the model

Use the omlx app to browse and download models. Search for `Qwen3.5-9B-mlx-lm-mxfp4` and download it. Models are stored locally (typically in `~/.omlx/models/`).

### Start the server

omlx serves models on `http://127.0.0.1:8000` by default. Start serving from the app UI, or use the CLI if available.

### Test it

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-mlx-lm-mxfp4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }' | jq .choices[0].message.content
```

### List available models

omlx can serve multiple models simultaneously:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'
```

---

## Option C: vllm-metal

[vllm-metal](https://github.com/vllm-project/vllm-metal) runs vLLM on Apple Silicon and requires macOS 15 or later. Install it with Homebrew:

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
```

Start an OpenAI-compatible server with an MLX-format Qwen3.5-9B model:

```bash
vllm serve mlx-community/Qwen3.5-9B-4bit \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 65536 --max-num-seqs 1 \
  --gpu-memory-utilization 0.3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

The tool and reasoning parsers above are for Qwen3.5. The memory budget is 30% of unified memory; if startup reports insufficient capacity, increase `--gpu-memory-utilization` or use a smaller model while keeping at least Hermes' 64K context minimum.

In `hermes model`, choose **Custom endpoint**, enter `http://127.0.0.1:8000/v1`, leave the API key blank, and select `mlx-community/Qwen3.5-9B-4bit`. Set the context length to `65536` to match the server. See the [vllm-metal installation guide](https://docs.vllm.ai/projects/vllm-metal/en/latest/installation/) for other installation options.

---

## Benchmarks: llama.cpp, oMLX, and vllm-metal

Tested on October 31, 2026, on an **Apple M5 Pro with 64 GB unified memory**, running macOS 27.0.1 and **Qwen3.5-9B**. vllm-metal and oMLX loaded identical `mlx-community/Qwen3.5-9B-4bit` weights; llama.cpp used `unsloth/Qwen3.5-9B-GGUF` in Q4_K_M format. The MLX and GGUF quantizations differ.

Five shared prompts, three runs each, with one request at a time and a 65,536-token context limit. Engines ran sequentially, with thinking and prompt-cache reuse disabled. Two warmup requests per engine excluded model loading from the measurements. All 45 measured requests had matching prompt token counts and produced exactly 512 output tokens. llama.cpp used f16 K/V cache for this comparison, rather than the quantized cache in Option A.

### Results

| Metric | llama.cpp b10731 | oMLX 0.6.4 | vllm-metal 0.32.0 |
|--------|-----------------|------------|------------------|
| **TTFT (avg)** | 149.6 ms | 339.7 ms | 116.3 ms |
| **TTFT (p50)** | 146.5 ms | 338.7 ms | 116.5 ms |
| **Generation (avg)** | 42.4 tok/s | 54.0 tok/s | 52.9 tok/s |
| **Generation (p50)** | 42.5 tok/s | 54.1 tok/s | 52.8 tok/s |
| **Total time (512 tokens, avg)** | 12.21 s | 9.80 s | 9.77 s |

See the [benchmark details](https://github.com/NousResearch/hermes-agent/pull/147355) for model revisions and the measurement method.

### What this means

**vllm-metal** had the lowest time to first token in this workload. **oMLX** had a small generation-rate advantage, while its total response time was close to vllm-metal's. These short, single-request measurements do not establish performance for long-context agent turns, cached conversations, or concurrent workloads.

### Which one should you pick?

| Use case | Recommendation |
|----------|---------------|
| Lowest time to first token in this benchmark | vllm-metal |
| 512-token responses in this benchmark | oMLX or vllm-metal |
| Memory-constrained (8-16 GB) | llama.cpp (quantized KV cache is unmatched) |
| Serving multiple models simultaneously | omlx (built-in multi-model support) |
| Maximum compatibility (Linux too) | llama.cpp |

---

## Connect to Hermes

Once your local server is running:

```bash
hermes model
```

Select **Custom endpoint** and follow the prompts. It will ask for the base URL and model name — use the values from whichever backend you set up above.

---

## Timeouts

Hermes automatically detects local endpoints (localhost, LAN IPs) and relaxes its streaming timeouts. No configuration needed for most setups.

If you still hit timeout errors (e.g. very large contexts on slow hardware), you can override the streaming read timeout:

```bash
# In your .env — raise from the 120s default to 30 minutes
HERMES_STREAM_READ_TIMEOUT=1800
```

| Timeout | Default | Local auto-adjustment | Env var override |
|---------|---------|----------------------|------------------|
| Stream read (socket-level) | 120s | Raised to 1800s | `HERMES_STREAM_READ_TIMEOUT` |
| Stale stream detection | 180s | Disabled entirely | `HERMES_STREAM_STALE_TIMEOUT` |
| API call (non-streaming) | 1800s | No change needed | `HERMES_API_TIMEOUT` |

The stream read timeout is the one most likely to cause issues — it's the socket-level deadline for receiving the next chunk of data. During prefill on large contexts, local models may produce no output for minutes while processing the prompt. The auto-detection handles this transparently.

:::tip A silent first turn is usually prefill, not a hang
Hermes sends its system prompt and tool schemas on every call, so on slower hardware the first turn can involve minutes of silence while the model processes that prompt before generating anything. That's prefill at work, not a stalled session. See [Slow first response (prefill)](./local-ollama-setup.md#slow-first-response-prefill) in the Ollama guide for mitigations like keeping the model loaded and trimming the fixed prompt with `hermes prompt-size`.
:::
