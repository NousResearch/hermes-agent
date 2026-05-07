---
sidebar_position: 10
title: "Run Hermes Locally with llama.cpp"
description: "Use llama.cpp's llama-server as a fully local OpenAI-compatible backend for Hermes Agent — zero cloud API cost"
---

# Run Hermes Locally with llama.cpp

llama.cpp ships an HTTP server (`llama-server`) that speaks the OpenAI
chat-completions wire format. Hermes has a built-in `llama-cpp` provider
that points at it — once the server is running, `hermes chat --provider
llama-cpp` works with no further config.

This guide is the llama.cpp counterpart to [Run Hermes Locally with
Ollama](./local-ollama-setup.md). Pick whichever backend you prefer;
llama.cpp gives you finer control over GGUF files, quantization, and
threading, while Ollama wraps that in a friendlier model manager.

## What you need

- a `.gguf` model file (any quant) — grab one from any source you trust
- `llama-server` on your `$PATH`

## Step 1: Install llama.cpp

```bash
# macOS
brew install llama.cpp

# Linux (prebuilt)
# https://github.com/ggml-org/llama.cpp/releases  →  pick a llama-bXXXX-bin-* tarball

# Or from pip
pip install 'llama-cpp-python[server]'
```

Verify:

```bash
llama-server --help | head
```

## Step 2: Boot the server

The repo ships a launcher that lines up with Hermes' built-in defaults:

```bash
./scripts/start-llama-server.sh ~/models/your-model.gguf
```

That brings the server up at `http://127.0.0.1:8088/v1` — exactly where
Hermes' `llama-cpp` provider looks by default.

Override with environment variables:

```bash
PORT=9000 CTX=32768 N_GPU_LAYERS=-1 ./scripts/start-llama-server.sh ~/models/foo.gguf
```

If you change the port or host, also tell Hermes:

```bash
export LLAMA_CPP_BASE_URL=http://127.0.0.1:9000/v1
```

## Step 3: Point Hermes at it

```bash
hermes chat --provider llama-cpp
```

Or set it permanently in `~/.hermes/config.yaml`:

```yaml
model:
  provider: llama-cpp
  default: <model-id-as-reported-by-/v1/models>
```

The model id Hermes uses comes straight from `${LLAMA_CPP_BASE_URL}/models`,
which `llama-server` populates from whatever GGUF you loaded. You can also
just pass `--model anything` — `llama-server` ignores the field when only
one model is loaded.

## Optional: API key

`llama-server` runs without auth by default. If you start it with
`--api-key <token>`, mirror that in Hermes:

```bash
export LLAMA_CPP_API_KEY=<token>
```

## Reference

- Provider id: `llama-cpp`
- Aliases: `llamacpp`, `llama.cpp`, `llama_cpp`, `llama-server`
- Default base URL: `http://127.0.0.1:8088/v1`
- Env vars: `LLAMA_CPP_API_KEY`, `LLAMA_CPP_BASE_URL`
- Profile source: [`plugins/model-providers/llama-cpp/`](https://github.com/NousResearch/hermes-agent/tree/main/plugins/model-providers/llama-cpp)
- Launcher: [`scripts/start-llama-server.sh`](https://github.com/NousResearch/hermes-agent/blob/main/scripts/start-llama-server.sh)
