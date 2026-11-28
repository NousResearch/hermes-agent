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

## Desktop availability and downloads

Local Models is on by default in the desktop app on macOS, Windows, and Linux
(x64 and arm64), on every release channel. On other platforms, launch the app
with `--local` to show it. A runtime can already be bundled; its absence
triggers the managed-tool install path, not an arbitrary latest llama.cpp
download.

Use **Pause** and **Resume** on engine installs, engine updates, catalog models,
Hugging Face downloads, and quickstart. Paused jobs remain visible when you
leave and reopen Local Models. Pause stops the transfer at a chunk boundary;
unpacking, verification, and server activation are separate phases.

PM downloads every engine component, including CUDA runtime DLLs, from the
URLs and SHA-256 pins in `pm/lock.json`. Model weights, split GGUF parts,
vision projectors, and draft models use the same downloader. Byte progress
covers the whole download plan, including completed files and resumed ranges.
Quickstart labels engine and model progress as separate stages.

Partial downloads use PM's writable `cache/partials` area, outside a signed
app package. Range-capable hosts resume missing bytes. Hosts without Range
support restart the current file. Completed files are reused.

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

From a terminal, `hermes model` → **Local models** runs the same setup: it
lists the catalog priced for your machine (★ marks the recommendation),
installs the pinned engine, downloads the model with live progress, starts
the server, and makes the model your default. Models already on disk are
offered too. Ctrl+C pauses a download; choosing the model again resumes it.

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

Hermes recommends the highest-quality model that runs entirely in GPU or
unified memory at a predicted 20 tokens per second or more. A machine's
maker can set a different default, and Hermes recommends that model
whenever it fits. When no model reaches the floor, Hermes recommends none,
and you can still choose any model that fits.

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
- **Memory fit includes the launch configuration**, not just the model file:
  context state, runtime buffers, the vision projector, and MTP buffers all
  count. For multi-token prediction (MTP), Hermes uses smaller batches when
  larger batches would spill at the same context window. MTP stays enabled.
  The same calculation runs when a grown window is restored after restart.
- **Conversation compression follows a growth check.** If a larger window
  cannot fit, generation is too slow, or the native maximum is reached,
  Hermes compresses instead of claiming a window the server did not receive.
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
a default, not a requirement. You can enter the server root (for example
`http://127.0.0.1:8080`) or the full `/v1` URL: the endpoint test tries
both and saves the variant that actually served `/models`, so chat
requests go to the same prefix the model list came from. For manual setups (Ollama, MLX, custom
builds, headless CLI machines), see
[Run Hermes Locally with Ollama](../guides/local-ollama-setup.md) and
[Run Local LLMs on Mac](../guides/local-llm-on-mac.md).

## Running Hermes with local or small models

The managed runtime above sizes everything for you. These tips cover the other
cases: a server you run yourself (llama.cpp, Ollama, LM Studio, vLLM, SGLang, any
custom OpenAI-compatible endpoint) or a small model that needs a leaner setup.
They come from reports by people running Hermes on their own hardware. Every
key below is a regular `~/.hermes/config.yaml` setting.

### Set the context length explicitly

Hermes reads the window from the server when it can, but many local servers
report their training maximum or a small default, not the window they actually
allocated. Set the value yourself so that compression and the context meter use
the right number:

```yaml
model:
  provider: custom
  base_url: http://127.0.0.1:8080/v1
  default: my-model
  context_length: 65536     # the window the server really serves
```

- The number must match the server: `-c` for llama.cpp, `--max-model-len` for
  vLLM, and `num_ctx` / `OLLAMA_CONTEXT_LENGTH` for Ollama. With Ollama you can
  also set `model.ollama_num_ctx` so that Hermes requests that window on each
  call.
- Hermes needs at least **64,000 tokens**, and a smaller window is refused at
  startup. The one exception is `provider: lmstudio` with an explicit positive
  `model.context_length`, which can run below that. llama.cpp parallel slots
  split the total (`-c 65536 -np 4` gives each slot 16K), so use `-np 1` or
  raise `-c`.
- Check the `Context limit: … tokens` line at startup to confirm the value
  Hermes is using.

### Turn reasoning off for non-reasoning instruct models

When the effort is unset, a custom endpoint gets `medium`. For a plain instruct
model, or a hybrid model you want in its non-thinking mode, turn it off:

```yaml
agent:
  reasoning_effort: none
```

When reasoning is off, Hermes sends `reasoning_effort: "none"` on custom
endpoints, plus `think: false` on Ollama URLs. To turn it off for only some
models, use `agent.reasoning_overrides` (see
[Reasoning Effort](./configuration.md#reasoning-effort)). In a session,
`/reasoning none --global` does the same thing and saves it to config.

### Trim enabled toolsets to cut prompt tokens

Every enabled tool's schema is sent on every call. On a slow local model, this
fixed prompt is most of the prefill time before the first token. Measure it,
then turn off what you don't use:

```bash
hermes prompt-size            # byte breakdown of system prompt + tool schemas
hermes tools                  # per-platform tool checklist
```

```yaml
agent:
  disabled_toolsets:          # removed everywhere (CLI and every gateway platform)
    - browser
    - image_gen
```

For a single run, `hermes chat --toolsets file,terminal` enables only the
toolsets you list. See
[Global Toolset Disable](./configuration.md#global-toolset-disable).

### Fall back to a local endpoint

A local server works well as a safety net behind a cloud primary, or behind a
second local server. `fallback_providers` is tried in order when the primary
fails with rate-limit, overload or connection errors:

```yaml
fallback_providers:
  - provider: custom
    model: my-local-model
    base_url: http://127.0.0.1:8080/v1
  - provider: llamacpp      # the managed runtime; skipped when it isn't running
    model: my-managed-model
```

`hermes fallback add` writes the same list interactively. See
[Fallback Providers](./features/fallback-providers.md).

### Tune compression for small windows

Compression starts when context usage passes a trigger that Hermes derives from
`compression.threshold`:

- Windows under 512K use a ratio of at least `0.75`. A 128K model compacts at
  about 98K.
- The trigger never drops below 64,000 tokens. Near that minimum it is capped
  at 85% of the window instead, so a 65,536-token model compacts at about 55.7K
  whether `threshold` is 0.50, 0.75 or 0.80. Only a `threshold` above 0.85 moves
  it later.

To compact **earlier** on a small window, set an absolute cap. Hermes uses the
lower of the cap and the derived trigger:

```yaml
compression:
  threshold_tokens: 40000   # compress at 40K or the derived trigger, whichever is lower
```

The `Context limit: … (compress at …)` startup line shows the trigger in effect.
The summary itself is written by the `auxiliary.compression` model. If your
main model is slow, point that model at a faster endpoint (see
[Auxiliary Models](./configuration.md#auxiliary-models)).

### llama.cpp: cache and checkpoint RAM

`llama-server` keeps reusable prompt caches and per-slot context checkpoints in
**system RAM**, separate from the model weights and the KV cache. Agent sessions
have long, growing prompts, so these can take several GB on top of the model.
On a RAM-limited machine, cap them with server flags (not Hermes settings):

```bash
# --cache-ram: MiB for the prompt cache (0 disables, -1 = no limit)
# --ctx-checkpoints: max context checkpoints kept per slot
llama-server -m model.gguf -c 65536 -np 1 --cache-ram 2048 --ctx-checkpoints 4
```

Flag names and defaults change between llama.cpp builds, so check
`llama-server --help` for yours. If a busy server streams slowly enough to trip
Hermes's stall detector, raise `agent.local_stream_stale_timeout` (default
900 seconds).

## Configuration

The managed runtime is controlled by the `local_runtime` section of
`config.yaml`. The desktop UI writes these values for you; they're
documented for CLI and headless use:

```yaml
local_runtime:
  enabled: false     # true = start the managed server with Hermes.
                     # The desktop "Use" button sets this automatically.
  backend: auto      # auto | cuda | metal | vulkan | hip | cpu
  detect_ports: [8081]  # extra ports to probe for a llama-server you run
                        # yourself (the default probe is :8080 only)
```

Running `llama-server` yourself on a fixed port works with the same
`model.provider: llamacpp` selection — either list the port in
`local_runtime.detect_ports`, or define the endpoint explicitly under
`providers:` (an explicit entry wins over server detection):

```yaml
providers:
  llamacpp:
    base_url: http://127.0.0.1:8081/v1
    model: my-model
```

The `/model` → **Local** picker row and `provider: llamacpp` resolve to that
server; with no server reachable the error names the local runtime ("the local
model server isn't running") instead of an unknown-provider or missing-API-key
message.

Engine versions come only from PM's lockfile, not a `local_runtime.tag`
override. Boot uses an installed PM engine without downloading. When a new
pin is available, install it with the desktop update button.

Models live in the machine-shared `models/` directory. Engine binaries live
in PM's store; `runtimes/llamacpp/` holds mutable presets and server state.
Selecting a local model as your main model uses the standard
`model.provider: llamacpp` + `model.default` settings.

## Requirements and limits

- **Windows:** CUDA on supported NVIDIA targets, Vulkan on x64, or CPU.
  **Linux:** CUDA on x64 and arm64 NVIDIA (needs a CUDA 13 driver), Vulkan, or CPU.
  **macOS:** Metal or CPU. HIP/ROCm is an explicit choice on supported x64
  targets. Unsupported backend/target pairs fail before any download.
- `backend: auto` prefers CUDA for NVIDIA and Vulkan for AMD and Intel GPUs,
  then falls back to the builds listed above for your platform (CPU when no
  GPU is found). Behind a Vulkan or HIP engine, a discrete card is sized from
  its own memory (system RAM is spill space) and each launch subtracts what
  other programs hold on it; an integrated GPU shares system RAM and is sized
  from that. Quickstart installs the engine before choosing the model, so the
  pick reflects the card.
- A GPU with 8 GB+ of memory runs the small catalog models comfortably;
  16 GB+ runs the 27–35B models at high quality.
- Model completeness is checked against the server's response, not catalog
  size estimates. Interrupted transfers retain partials for resume; incomplete
  files are not published. Engine archives are SHA-256 verified before use.
- Deleting a model removes every file it staged, including vision
  adapters and speculative-decoding companions.
