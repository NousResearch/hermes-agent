---
title: "Comfyui"
sidebar_label: "Comfyui"
description: "Generate images, video, and audio with ComfyUI — install, launch, manage nodes/models, run workflows with parameter injection"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Comfyui

Generate images, video, and audio with ComfyUI — install, launch, manage nodes/models, run workflows with parameter injection. Uses the official comfy-cli for lifecycle and direct REST API for execution.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/creative/comfyui` |
| Version | `4.1.0` |
| Author | ['kshitijk4poor', 'alt-glitch'] |
| License | MIT |
| Platforms | macos, linux, windows |
| Tags | `comfyui`, `image-generation`, `stable-diffusion`, `flux`, `creative`, `generative-ai`, `video-generation` |
| Related skills | [`stable-diffusion-image-generation`](/docs/user-guide/skills/optional/mlops/mlops-stable-diffusion), `image_gen` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# ComfyUI

Generate images, video, and audio through ComfyUI using the official `comfy-cli` for
setup/management and direct REST API calls for workflow execution.

**Reference files in this skill:**

- `references/official-cli.md` — comfy-cli command reference (install, launch, nodes, models)
- `references/rest-api.md` — ComfyUI REST API endpoints (local + cloud)
- `references/workflow-format.md` — workflow JSON format, common node types, parameter mapping

**Scripts in this skill:**

- `scripts/hardware_check.py` — detect GPU/VRAM/Apple Silicon, decide local vs Comfy Cloud
- `scripts/comfyui_setup.sh` — full setup automation (hardware check + install + launch + verify)
- `scripts/extract_schema.py` — reads workflow JSON, outputs which parameters are controllable
- `scripts/run_workflow.py` — injects user args, submits workflow, monitors progress, downloads outputs
- `scripts/check_deps.py` — checks if required custom nodes and models are installed

## When to Use

- User asks to generate images with Stable Diffusion, SDXL, Flux, or other diffusion models
- User wants to run a specific ComfyUI workflow
- User wants to chain generative steps (txt2img → upscale → face restore)
- User needs ControlNet, inpainting, img2img, or other advanced pipelines
- User asks to manage ComfyUI queue, check models, or install custom nodes
- User wants video/audio generation via AnimateDiff, Hunyuan, AudioCraft, etc.

## Architecture: Two Layers

<!-- ascii-guard-ignore -->
```
┌─────────────────────────────────────────────────────┐
│ Layer 1: comfy-cli (official)                       │
│   Setup, lifecycle, nodes, models                   │
│   comfy install / launch / stop / node / model      │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│ Layer 2: REST API + skill scripts                   │
│   Workflow execution, param injection, monitoring   │
│   POST /api/prompt, GET /api/view, WebSocket        │
│   scripts/run_workflow.py, extract_schema.py        │
└─────────────────────────────────────────────────────┘
```
<!-- ascii-guard-ignore-end -->

**Why two layers?** The official CLI handles installation and server management excellently
but has minimal workflow execution support (just raw file submission, no param injection,
no structured output). The REST API fills that gap — the scripts in this skill handle the
param injection, execution monitoring, and output download that the CLI doesn't do.

## Quick Start

### Detect Environment

```bash
# What's available?
command -v comfy >/dev/null 2>&1 && echo "comfy-cli: installed"
curl -s http://127.0.0.1:8188/system_stats 2>/dev/null && echo "server: running"

# Can this machine actually run ComfyUI locally? (GPU/VRAM/Apple Silicon check)
python3 scripts/hardware_check.py
```

If nothing is installed, go to **Setup & Onboarding** below — but always run the
hardware check first, before picking an install path.
If the server is already running, skip to **Core Workflow**.

## Core Workflow

### Step 1: Get a Workflow

Users provide workflow JSON files. These come from:
- ComfyUI web editor → "Save (API Format)" button
- Community downloads (civitai, Reddit, Discord)
- The `scripts/` directory of this skill (example workflows)

**The workflow must be in API format** (node IDs as keys with `class_type`).
If user has editor format (has `nodes[]` and `links[]` at top level), they
need to re-export using "Save (API Format)" in the ComfyUI web editor.

### Step 2: Understand What's Controllable

```bash
python3 scripts/extract_schema.py workflow_api.json
```

Output (JSON):
```json
{
  "parameters": {
    "prompt": {"node_id": "6", "field": "text", "type": "string", "value": "a cat"},
    "negative_prompt": {"node_id": "7", "field": "text", "type": "string", "value": "bad quality"},
    "seed": {"node_id": "3", "field": "seed", "type": "int", "value": 42},
    "steps": {"node_id": "3", "field": "steps", "type": "int", "value": 20},
    "width": {"node_id": "5", "field": "width", "type": "int", "value": 512},
    "height": {"node_id": "5", "field": "height", "type": "int", "value": 512}
  }
}
```

### Step 3: Run with Parameters

**Local:**
```bash
python3 scripts/run_workflow.py \
  --workflow workflow_api.json \
  --args '{"prompt": "a beautiful sunset over mountains", "seed": 123, "steps": 30}' \
  --output-dir ./outputs
```

**Cloud:**
```bash
python3 scripts/run_workflow.py \
  --workflow workflow_api.json \
  --args '{"prompt": "a beautiful sunset", "seed": 123}' \
  --host https://cloud.comfy.org \
  --api-key "$COMFY_CLOUD_API_KEY" \
  --output-dir ./outputs
```

### Step 4: Present Results

The script outputs JSON with file paths:
```json
{
  "status": "success",
  "outputs": [
    {"file": "./outputs/ComfyUI_00001_.png", "node_id": "9", "type": "image"}
  ]
}
```

Show images to the user via `vision_analyze` or return the file path directly.

## Decision Tree

| User says | Tool | Command |
|-----------|------|---------|
| "install ComfyUI" | comfy-cli | `comfy install` |
| "start ComfyUI" | comfy-cli | `comfy launch --background` |
| "stop ComfyUI" | comfy-cli | `comfy stop` |
| "install X node" | comfy-cli | `comfy node install <name>` |
| "download X model" | comfy-cli | `comfy model download --url <url>` |
| "list installed models" | comfy-cli | `comfy model list` |
| "list installed nodes" | comfy-cli | `comfy node show installed` |
| "generate an image" | script | `run_workflow.py --args '{"prompt": "..."}'` |
| "use this image" (img2img) | REST | upload image, then run_workflow.py |
| "what can I change in this workflow?" | script | `extract_schema.py workflow.json` |
| "check if workflow deps are met" | script | `check_deps.py workflow.json` |
| "what's in the queue?" | REST | `curl http://HOST:8188/queue` |
| "cancel that" | REST | `curl -X POST http://HOST:8188/interrupt` |
| "free GPU memory" | REST | `curl -X POST http://HOST:8188/free` |

## Setup & Onboarding

When a user asks to set up ComfyUI, the FIRST thing to do is ask them whether
they want **Comfy Cloud** (hosted, zero install, API key) or **Local** (install
ComfyUI on their machine). Do NOT start running install commands or hardware
checks until they've answered.

**Official docs:** https://docs.comfy.org/installation
**CLI docs:** https://docs.comfy.org/comfy-cli/getting-started
**Cloud docs:** https://docs.comfy.org/get_started/cloud

### Step 0: Ask Local vs Cloud (ALWAYS FIRST)

Present the tradeoff clearly and wait for the user to choose. Suggested script:

> "Do you want to run ComfyUI locally on your machine, or use Comfy Cloud?
>
> - **Comfy Cloud** — hosted on RTX 6000 Pro GPUs, all models pre-installed, zero setup. Requires an API key (paid subscription). Best if you don't have a capable GPU or want to skip installation.
> - **Local** — free, but your machine MUST meet the hardware requirements:
>   - NVIDIA GPU with **≥6 GB VRAM** (≥8 GB recommended for SDXL, ≥12 GB for Flux/video), OR
>   - AMD GPU with ROCm support (Linux), OR
>   - Apple Silicon Mac (M1 or newer) with **≥16 GB unified memory** (≥32 GB recommended).
>   - Intel Macs and machines with no GPU will NOT work — use Cloud instead.
>
> Which would you like?"

Route based on their answer:

- **User picks Cloud** → skip to **Path A** (no hardware check needed).
- **User picks Local** → go to **Step 1: Hardware Check** to verify their machine actually meets the requirements, then pick an install path from Paths B-E based on the verdict.
- **User is unsure / asks for a recommendation** → run the hardware check anyway and let the verdict decide.

### Step 1: Verify Hardware (ONLY if user chose local)

```bash
python3 scripts/hardware_check.py --json
```

It detects OS, GPU (NVIDIA CUDA / AMD ROCm / Apple Silicon / Intel Arc), VRAM,
and unified/system RAM, then returns a verdict plus a suggested `comfy-cli` flag:

| Verdict    | Meaning                                                   | Action                                          |
|------------|-----------------------------------------------------------|-------------------------------------------------|
| `ok`       | ≥8 GB VRAM (discrete) OR ≥32 GB unified (Apple Silicon)   | Local install — use `comfy_cli_flag` from report |
| `marginal` | SD1.5 works; SDXL tight; Flux/video unlikely              | Local OK for light workflows, else **Path A (Cloud)** |
| `cloud`    | No usable GPU, &lt;6 GB VRAM, &lt;16 GB Apple unified, Intel Mac | **User chose local but their machine doesn't meet requirements** — surface the `notes` and ask if they want to switch to Cloud |

Hardware thresholds the skill enforces:

- **Discrete GPU minimum:** 6 GB VRAM. Below that, most modern models won't load.
- **Apple Silicon:** M1 or newer (ARM64). Intel Macs have no MPS backend — Cloud only.
- **Apple Silicon memory:** 16 GB unified minimum. 8 GB M1/M2 will swap/OOM on SDXL/Flux.
- **No accelerator at all:** CPU-only is listed as a comfy-cli option but a single SDXL
  image takes 10+ minutes — treat it as unusable and route to Cloud.

If verdict is `cloud` but the user explicitly wanted local, DO NOT proceed
silently. Show the `notes` array verbatim, explain which requirement they
don't meet, and ask whether they want to (a) switch to Cloud or (b) force
a local install anyway (marginal/cloud-verdict local installs will OOM or
be unusably slow on modern models).

The report's `comfy_cli_flag` field gives you the exact flag for Step 2 below:
`--nvidia`, `--amd`, or `--m-series`. For Intel Arc, use Path E (manual install).

Surface the `notes` array verbatim to the user so they understand why a
particular path was recommended.

### Choosing an Installation Path

Use the hardware check result first. The table below is a fallback for when the user
has already told you their hardware or you need to narrow down between multiple
viable paths:

| Situation | Recommended Path |
|-----------|-----------------|
| `verdict: cloud` from hardware check | **Path A: Comfy Cloud** |
| No GPU / just want to try it | **Path A: Comfy Cloud** (zero setup) |
| Windows + NVIDIA GPU + non-technical | **Path B: ComfyUI Desktop** (one-click installer) |
| Windows + NVIDIA GPU + technical | **Path C: Portable** or **Path D: comfy-cli** |
| Linux + any GPU | **Path D: comfy-cli** (easiest) or Path E manual |
| macOS + Apple Silicon | **Path B: ComfyUI Desktop** or **Path D: comfy-cli** |
| Headless / server / CI | **Path D: comfy-cli** |

For the fully automated path (hardware check → install → launch), just run:

```bash
bash scripts/comfyui_setup.sh
```

It runs `hardware_check.py` internally, refuses to install locally when the verdict
is `cloud`, picks the right `comfy-cli` flag otherwise, then installs and launches.

---

### Path A: Comfy Cloud (No Local Install)

For users without a capable GPU or who want zero setup.
Powered by RTX 6000 Pro GPUs, all models pre-installed.

**Docs:** https://docs.comfy.org/get_started/cloud

1. Go to https://comfy.org/cloud and sign up
2. Get an API key at https://platform.comfy.org/login
   - Click `+ New` in API Keys section → Generate
   - Save immediately (only visible once)
3. Set the key:
   ```bash
   export COMFY_CLOUD_API_KEY="comfyui-xxxxxxxxxxxx"
   ```
4. Run workflows via the script or web UI:
   ```bash
   python3 scripts/run_workflow.py \
     --workflow workflow_api.json \
     --args '{"prompt": "a cat"}' \
     --host https://cloud.comfy.org \
     --api-key "$COMFY_CLOUD_API_KEY" \
     --output-dir ./outputs
   ```

**Pricing:** https://www.comfy.org/cloud/pricing
Subscription required. Concurrent limits: Free/Standard: 1 job, Creator: 3, Pro: 5.

---

### Path B: ComfyUI Desktop (Windows/macOS)

One-click installer for non-technical users. Currently Beta.

**Docs:** https://docs.comfy.org/installation/desktop

- **Windows (NVIDIA):** https://download.comfy.org/windows/nsis/x64
- **macOS (Apple Silicon):** Available from https://comfy.org (download page)

Steps:
1. Download and run installer
2. Select GPU type (NVIDIA recommended, or CPU mode)
3. Choose install location (SSD recommended, ~15GB needed)
4. Optionally migrate from existing ComfyUI Portable install
5. Desktop launches automatically — web UI opens in browser

Desktop manages its own Python environment. For CLI access to the bundled env:
```bash
cd <install_dir>/ComfyUI
.venv/Scripts/activate   # Windows
# or use the built-in terminal in the Desktop UI
```

**Limitations:** Desktop uses stable releases (may lag behind latest).
Linux not supported for Desktop — use comfy-cli or manual install.

---

### Path C: ComfyUI Portable (Windows Only)

Standalone package with embedded Python. Extract and run. No install.

**Docs:** https://docs.comfy.org/installation/comfyui_portable_windows

1. Download from https://github.com/comfyanonymous/ComfyUI/releases
   - Standard: Python 3.13 + CUDA 13.0 (modern NVIDIA GPUs)
   - Alt: PyTorch CUDA 12.6 + Python 3.12 (NVIDIA 10 series and older)
   - AMD (experimental)
2. Extract with 7-Zip
3. Run `run_nvidia_gpu.bat` (or `run_cpu.bat`)
4. Wait for "To see the GUI go to: http://127.0.0.1:8188"

Update: run `update/update_comfyui.bat` (latest commit) or
`update/update_comfyui_stable.bat` (latest stable release).

---

### Path D: comfy-cli (All Platforms — Recommended for Agents)

The official CLI is the best path for headless/automated setups.

**Docs:** https://docs.comfy.org/comfy-cli/getting-started
**Repo:** https://github.com/Comfy-Org/comfy-cli

#### Prerequisites
- Python 3.10+ (3.13 recommended)
- pip (or conda/uv)
- GPU drivers installed (CUDA for NVIDIA, ROCm for AMD)

#### Install comfy-cli

```bash
pip install comfy-cli
# or
uvx --from comfy-cli comfy --help
```

Disable analytics (avoids interactive prompt):
```bash
comfy --skip-prompt tracking disable
```

#### Install ComfyUI

```bash
# Interactive (prompts for GPU type)
comfy install

# Non-interactive variants:
comfy --skip-prompt install --nvidia              # NVIDIA (CUDA)
comfy --skip-prompt install --amd                 # AMD (ROCm, Linux)
comfy --skip-prompt install --m-series            # Apple Silicon (MPS)
comfy --skip-prompt install --cpu                 # CPU only (slow)

# With faster dependency resolution:
comfy --skip-prompt install --nvidia --fast-deps
```

Default location: `~/comfy/ComfyUI` (Linux), `~/Documents/comfy/ComfyUI` (macOS/Win).
Override with: `comfy --workspace /custom/path install`

#### Launch Server

```bash
comfy launch --background              # background daemon on :8188
comfy launch                           # foreground (see logs)
comfy launch -- --listen 0.0.0.0       # accessible on LAN
comfy launch -- --port 8190            # custom port
comfy launch -- --lowvram              # low VRAM mode (6GB cards)
```

Verify server is running:
```bash
curl -s http://127.0.0.1:8188/system_stats | python3 -m json.tool
```

Stop background server:
```bash
comfy stop
```

---

### Path E: Manual Install (Advanced / All Hardware)

For full control or unsupported hardware (Ascend NPU, Cambricon MLU, Intel Arc).

**Docs:** https://docs.comfy.org/installation/manual_install
**GitHub:** https://github.com/comfyanonymous/ComfyUI

```bash
# 1. Create environment
conda create -n comfyenv python=3.13
conda activate comfyenv

# 2. Clone
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 3. Install PyTorch (pick your hardware)
# NVIDIA:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
# AMD (ROCm 6.4):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
# Apple Silicon:
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
# Intel Arc:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/xpu
# CPU only:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# 4. Install ComfyUI deps
pip install -r requirements.txt

# 5. Run
python main.py
# With options: python main.py --listen 0.0.0.0 --port 8188
```

---

### Post-Install: Download Models

ComfyUI needs at least one checkpoint model to generate images.

**Using comfy-cli:**
```bash
# SDXL (general purpose, ~6.5GB)
comfy model download \
  --url "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
  --relative-path models/checkpoints

# SD 1.5 (lighter, ~4GB, good for low VRAM)
comfy model download \
  --url "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors" \
  --relative-path models/checkpoints

# From CivitAI (may need API token):
comfy model download \
  --url "https://civitai.com/api/download/models/128713" \
  --relative-path models/checkpoints \
  --set-civitai-api-token "YOUR_TOKEN"

# LoRA adapters:
comfy model download --url "<URL>" --relative-path models/loras
```

**Manual download:** Place `.safetensors` / `.ckpt` files directly into the
`ComfyUI/models/checkpoints/` directory (or `loras/`, `vae/`, etc.).

List installed models:
```bash
comfy model list
```

---

### Post-Install: Install Custom Nodes

Custom nodes extend ComfyUI's capabilities (upscaling, video, ControlNet, etc.).

```bash
comfy node install comfyui-impact-pack           # popular utility pack
comfy node install comfyui-animatediff-evolved    # video generation
comfy node install comfyui-controlnet-aux         # ControlNet preprocessors
comfy node install comfyui-essentials             # common helpers
comfy node update all                            # update all nodes
```

Check what's installed:
```bash
comfy node show installed
```

Install deps for a specific workflow:
```bash
comfy node install-deps --workflow=workflow_api.json
```

---

### Post-Install: Verify Setup

```bash
# Check server is responsive
curl -s http://127.0.0.1:8188/system_stats | python3 -m json.tool

# Check a workflow's dependencies
python3 scripts/check_deps.py workflow_api.json --host 127.0.0.1 --port 8188

# Test a generation
python3 scripts/run_workflow.py \
  --workflow workflow_api.json \
  --args '{"prompt": "test image, high quality"}' \
  --output-dir ./test-outputs
```

## Image Upload (img2img / Inpainting)

Upload files directly via REST:

```bash
# Upload input image
curl -X POST "http://127.0.0.1:8188/upload/image" \
  -F "image=@photo.png" -F "type=input" -F "overwrite=true"
# Returns: {"name": "photo.png", "subfolder": "", "type": "input"}

# Upload mask for inpainting
curl -X POST "http://127.0.0.1:8188/upload/mask" \
  -F "image=@mask.png" -F "type=input" \
  -F 'original_ref={"filename":"photo.png","subfolder":"","type":"input"}'
```

Then reference the uploaded filename in workflow args:
```bash
python3 scripts/run_workflow.py --workflow inpaint.json \
  --args '{"image": "photo.png", "mask": "mask.png", "prompt": "fill with flowers"}'
```

## Cloud Execution

Base URL: `https://cloud.comfy.org`
Auth: `X-API-Key` header

```bash
# Submit workflow
python3 scripts/run_workflow.py \
  --workflow workflow_api.json \
  --args '{"prompt": "cyberpunk city"}' \
  --host https://cloud.comfy.org \
  --api-key "$COMFY_CLOUD_API_KEY" \
  --output-dir ./outputs \
  --timeout 300

# Upload image for cloud workflows
curl -X POST "https://cloud.comfy.org/api/upload/image" \
  -H "X-API-Key: $COMFY_CLOUD_API_KEY" \
  -F "image=@input.png" -F "type=input" -F "overwrite=true"
```

Concurrent job limits:
| Tier | Concurrent Jobs |
|------|----------------|
| Free/Standard | 1 |
| Creator | 3 |
| Pro | 5 |

Extra submissions queue automatically.

## Queue & System Management

```bash
# Check queue
curl -s http://127.0.0.1:8188/queue | python3 -m json.tool

# Clear pending queue
curl -X POST http://127.0.0.1:8188/queue -d '{"clear": true}'

# Cancel running job
curl -X POST http://127.0.0.1:8188/interrupt

# Free GPU memory (unload all models)
curl -X POST http://127.0.0.1:8188/free -H "Content-Type: application/json" \
  -d '{"unload_models": true, "free_memory": true}'

# System stats (VRAM, RAM, GPU info)
curl -s http://127.0.0.1:8188/system_stats | python3 -m json.tool
```

## Pitfalls

1. **API format required** — `comfy run` and the scripts only accept API-format workflow JSON.
   If the user has editor format (from "Save" not "Save (API Format)"), they need to
   re-export. Check: API format has `class_type` in each node object, editor format has
   top-level `nodes` and `links` arrays.

2. **Server must be running** — All execution requires a live server. `comfy launch --background`
   starts one. Check with `curl http://127.0.0.1:8188/system_stats`.

3. **Model names are exact** — Case-sensitive, includes file extension. Use
   `comfy model list` to discover what's installed.

4. **Missing custom nodes** — "class_type not found" means a required node isn't installed.
   Run `check_deps.py` to find what's missing, then `comfy node install <name>`.

5. **Working directory** — `comfy-cli` auto-detects the ComfyUI workspace. If commands
   fail with "no workspace found", use `comfy --workspace /path/to/ComfyUI <command>`
   or `comfy set-default /path/to/ComfyUI`.

6. **Cloud vs local output download** — Cloud `/api/view` returns a 302 redirect to a
   signed URL. Always follow redirects (`curl -L`). The `run_workflow.py` script handles
   this automatically.

7. **Timeout for video/audio** — Long generations (video, high step counts) can take
   minutes. Pass `--timeout 600` to `run_workflow.py`. Default is 120 seconds.

8. **tracking prompt** — First run of `comfy` may prompt for analytics tracking consent.
   Use `comfy --skip-prompt tracking disable` to skip it non-interactively.

9. **comfy-cli invocation via uvx** — If comfy-cli is not installed globally, invoke with
   `uvx --from comfy-cli comfy <command>`. All examples in this skill use bare `comfy`
   but prepend `uvx --from comfy-cli` if needed.

## Verification Checklist

- [ ] `hardware_check.py` verdict is `ok` OR the user explicitly chose Comfy Cloud
- [ ] `comfy` available on PATH (or `uvx --from comfy-cli comfy --help` works)
- [ ] `curl http://127.0.0.1:8188/system_stats` returns JSON
- [ ] `comfy model list` shows at least one checkpoint
- [ ] Workflow JSON is in API format (has `class_type` keys)
- [ ] `check_deps.py` reports no missing nodes/models
- [ ] Test run completes and outputs are saved
