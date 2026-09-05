---
sidebar_position: 4
title: Local Models
description: Run models entirely on your own machine — no account, no API key, nothing leaves your computer.
---

# Local Models

Hermes can run open models entirely on your own machine. It downloads and
manages the inference engine (llama.cpp), picks the right build of each
model for your hardware, and handles memory so you never configure context
sizes, GPU layers, or quantization. You pick a model; Hermes does the rest.

Nothing leaves your computer: no account, no API key, and no network access
after a model is downloaded.

## Getting started

1. Open **Settings → Providers → Local Models** (or choose **Run models
   locally** during onboarding).
2. Click **Install runtime**. Hermes downloads the official llama.cpp
   build for your hardware (a few hundred MB), verifies it, and keeps it
   updated.
3. Pick a model from the catalog and click **Download**.
4. Click **Use**. New chats now run on the local model.

That's the whole flow. The server starts and stops with Hermes, restarts
survive app restarts, and switching back to a cloud provider is one click
in the model picker.

## How Hermes picks what to download

Every model in the catalog is priced against **your machine** before you
download anything. Each row shows:

- **Memory fit** — green (*Fits your GPU*: runs entirely in GPU memory),
  amber (*Uses system RAM*: works, but slower), or red (*Too big for this
  machine*).
- **Context** — the window the model starts with and the maximum it can
  grow to.
- The download size of the build selected for your hardware.

Models ship in several quality grades (quantizations). Hermes picks the
highest-quality build that runs fully on your GPU; machines with less
memory get a more compact build of the same model with the same
guarantees. Below 4-bit the quality loss is too severe, so Hermes never
offers builds smaller than that — a machine that can't run the 4-bit
build spilled to system RAM simply can't run that model.

Models that don't fit stay visible with the reason, so you always know
what a hardware upgrade would unlock.

## How memory management works

Local models live or die by memory placement, so Hermes manages it
end-to-end and exposes no knobs:

- **Models start at a context window that fully fits your GPU** and grow
  toward their native maximum as your conversation needs more room. You
  may see "Context window grown" in the status feed during long sessions
  — that's the window expanding, not an error.
- **Every recommended model gets at least a 64K context window.** When a
  model is larger than your GPU's memory, Hermes deliberately places the
  overflow in system RAM in the order that hurts least (expert weights
  first, never the attention cache), trading some speed to protect the
  context guarantee.
- **Conversation compression only kicks in at the model's maximum
  window** — growth always comes first.
- Idle models are unloaded after 15 minutes to free GPU memory; they
  reload automatically on the next message.

## The status bar

Right-click the status bar and enable **System resources** to see live GPU
utilization, GPU memory, and RAM while local models run. The context meter
always reflects the window the model is actually running with.

## Finding more models

The catalog is a curated starting point, not a boundary. The **Find more
models** section on the same page searches all of Hugging Face:

- Results show download counts and a per-file fit check sized to your
  machine, so you know before downloading whether a build runs fully on
  your GPU.
- Anything you download behaves exactly like a catalog model — Hermes
  reads the model file itself to pick its context window and memory
  placement. The only difference: community models don't carry our
  "validated" testing badge.
- Already have a `.gguf` file on disk? **Add model file** links it into
  your library without copying it (the original stays where it is), and
  it's usable immediately.

## Using your own llama-server

If a llama-server is already running on your machine, Hermes detects it
and uses it instead of starting its own. Point a custom endpoint at any
OpenAI-compatible server for full manual control — the managed runtime is
a default, not a requirement. For manual setups (Ollama, MLX, custom
builds, headless CLI machines), see
[Run Hermes Locally with Ollama](/guides/local-ollama-setup) and
[Run Local LLMs on Mac](/guides/local-llm-on-mac).

## Configuration

The managed runtime is controlled by the `local_runtime` section of
`config.yaml`. The desktop UI writes these values for you; they're
documented for CLI and headless use:

```yaml
local_runtime:
  enabled: false     # true = start the managed server with Hermes.
                     # The desktop "Use" button sets this automatically.
  backend: auto      # auto | cuda | metal | vulkan | hip | cpu
  tag: b10362        # pinned llama.cpp release; Hermes updates it with
                     # each release after re-validation
  server_extra_args: []  # extra llama-server flags, e.g. ["--main-gpu", "1"]
                     # on multi-GPU machines. Takes effect on the next
                     # server start (Stop, then Start in Local Models).
                     # --host/--port/--api-key/--models-dir are refused.
```

Models and runtime builds live under the Hermes home directory
(`models/` and `runtimes/llamacpp/`). Selecting a local model as your
main model uses the standard `model.provider: llamacpp` +
`model.default` settings — the same shape as every other provider.

## Requirements and limits

- **Windows and Linux:** NVIDIA GPU (CUDA) or CPU. **macOS:** Apple
  Silicon (Metal). Vulkan builds serve AMD GPUs.
- A GPU with 8 GB+ of memory runs the small catalog models comfortably;
  16 GB+ runs the 27–35B models at high quality.
- Model downloads are byte-size checked against the catalog during the
  transfer; an incomplete download is deleted and reported, never
  half-used. (Only the runtime engine zips are SHA-256 verified.)
- Deleting a model removes every file it staged, including vision
  adapters and speculative-decoding companions.
