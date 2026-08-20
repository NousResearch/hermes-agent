---
sidebar_position: 2.3
title: "System Requirements"
description: "Hardware, OS, and dependency requirements for running Hermes Agent — for both cloud API and local model paths."
---

# System Requirements

What you need to run Hermes Agent depends on **how** you run it. This page covers both paths — cloud API and local model — so you can size your hardware before installing.

---

## Hard Blockers

These conditions prevent Hermes from working regardless of how good your hardware is:

| Condition | Detail |
|-----------|--------|
| **Intel Macs** | Hermes does **not** support macOS on Intel (x86) processors. Apple Silicon (M1 and later) only. |
| **Minimum 64K context** | Hermes requires a model with at least **64,000 tokens** of context. Models with smaller windows cannot maintain enough working memory for multi-step tool-calling and are rejected at startup. Most hosted models (Claude, GPT, Gemini, Qwen, DeepSeek) meet this easily. For local models, set the context size to at least 64K. |
| **Python ≥3.11, <3.14** | The installer (via `uv`) manages Python automatically — you don't need it pre-installed. But your system must be capable of running Python 3.11, 3.12, or 3.13. |

---

## Path A: Cloud API (recommended for most users)

The cloud API path is the primary, recommended setup. Hermes runs locally as a thin client — the model runs on a remote provider. Hardware requirements are minimal.

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **RAM** | 2 GB | 4+ GB |
| **Disk (free)** | 4 GB | 8+ GB |
| **CPU** | 2 cores (x86_64 or ARM64) | 4+ cores |
| **GPU** | Not required | Not required |
| **Network** | Broadband internet (any speed) | Low-latency connection for responsive chat |

### What takes up space

| Component | Approximate size |
|-----------|-----------------|
| Repo clone (git history + working tree) | ~300 MB |
| Python virtual environment | ~500 MB |
| uv + Python 3.11 runtime | ~500 MB |
| Node.js v22 | ~80 MB |
| ripgrep, ffmpeg | ~50 MB |
| **Total** | **~1.5 GB** |

You'll also accumulate skills, sessions, and cached data over time — budget at least **4 GB** free to start.

### Supported operating systems

| OS / Architecture | Tier |
|------------------|------|
| Linux (x86_64, aarch64) | Tier 1 |
| Windows 10 / 11 (x86_64, aarch64) | Tier 1 |
| macOS Apple Silicon (M1+) | Tier 1 |
| Docker container | Tier 1 |
| Android / Termux (aarch64) | Tier 2 |
| Nix (macOS, Linux, NixOS) | Tier 2 (best-effort) |

See [Platform Support](./platform-support.md) for the full support matrix.

### Software your system already needs

The installer handles almost everything — but these must be present on your machine:

- **Git** (required across all platforms)
- **curl** + **xz-utils** (Linux only — needed to download Node.js)
- **build-essential** or **g++** (Linux only — needed to compile native modules for the desktop app)

:::tip
You do **not** need Python, Node.js, ripgrep, or ffmpeg pre-installed. The installer detects what's missing and downloads it automatically.
:::

---

## Path B: Local Model (Ollama / llama.cpp)

Running models locally is demanding. Hardware requirements are dominated by the model you choose.

| Resource | Minimum (3B models) | Recommended (9B models) | Heavy (27B+ models) |
|----------|---------------------|------------------------|---------------------|
| **RAM** | 8 GB | 16 GB | 32+ GB |
| **Disk (free)** | 10 GB | 20 GB | 50+ GB |
| **CPU** | 4 cores | 8+ cores | 16+ cores |
| **GPU** | Not required | 8+ GB VRAM (NVIDIA) | 24+ GB VRAM |
| **Network** | Required for initial model download | Broadband | Broadband |

### Local model sizing

| Model | Disk | RAM | Tool Calling |
|-------|------|-----|:------------:|
| 3B models (e.g. Llama 3.2) | ~2 GB | 4+ GB | No |
| 9B models (e.g. Qwen3.5-9B, Gemma 2) | ~5 GB | 8+ GB | Varies |
| 27B models (e.g. Gemma 2:27b) | ~16 GB | 20+ GB | Varies |
| 31B+ models (e.g. Gemma 4:31b) | ~20 GB | 24+ GB | Yes |

:::warning Tool calling matters
Hermes is an **agentic** assistant — it edits files, runs commands, and browses the web through tool calls. Not all local models support tool calling, and even those that do may produce unreliable results. For the full Hermes experience on local hardware, use a model that is tested for tool use (see the detailed local guides below).
:::

### Detailed local LLM guides

For step-by-step setup instructions, see:

- **[Local Ollama Setup](/guides/local-ollama-setup.md)** — CPU-only and GPU-accelerated models via Ollama
- **[Local LLMs on Mac](/guides/local-llm-on-mac.md)** — llama.cpp and MLX on Apple Silicon

---

## Path C: Docker Container

Running Hermes in Docker requires no additional hardware beyond the cloud API path — the container bundles everything.

| Resource | Minimum |
|----------|---------|
| **RAM** | 2 GB (container overhead + Hermes) |
| **Disk** | 2 GB for the image + 4 GB for the data mount |
| **CPU** | x86_64 or aarch64 host |
| **Docker** | Docker Engine 24+ or compatible (Podman, Orbstack) |

The full [Docker guide](/user-guide/docker.md) covers setup, volume mounts, and production deployment.

---

## Desktop App (Electron)

When running Hermes Desktop (the Electron shell), add these requirements:

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **RAM** | +1 GB (above agent requirements) | +2 GB |
| **Disk** | +500 MB | +1 GB |
| **OS** | macOS (Apple Silicon) or Windows 10/11 | — |

The desktop app bundles the CLI agent alongside the Electron shell — hardware must meet both the desktop and the Path A (cloud API) requirements unless you're pairing it with a local model.

---

## Sizing Guidelines by Use Case

| Use Case | RAM | Disk | CPU | GPU |
|----------|-----|------|-----|-----|
| CLI chat (cloud API) | 2 GB | 4 GB | 2 cores | — |
| Gateway / always-on bot (cloud API) | 2 GB | 8 GB | 2 cores | — |
| CLI + TUI + cron jobs (cloud API) | 4 GB | 8 GB | 4 cores | — |
| Local 3B–9B model | 16 GB | 20 GB | 4 cores | Optional (8 GB VRAM) |
| Local 27B+ model | 32 GB | 50 GB | 8 cores | Recommended (24+ GB VRAM) |
| Full stack (local model + gateway + dashboard) | 32 GB | 50 GB | 8 cores | Recommended |

---

## What About ARM / Raspberry Pi?

Hermes runs on ARM64 (aarch64) platforms. Linux on aarch64 is Tier 1. This includes Raspberry Pi 4 / 5 running a 64-bit OS, Oracle Ampere A1 instances, and Apple Silicon Macs (in Linux VMs or Docker).

For a Raspberry Pi:
- **Cloud API path** works well on a Pi 4 (4 GB) or Pi 5 (8 GB).
- **Local model path** is not practical — even a 3B Q4 model struggles with inference speed on the Pi's ARM CPU.

---

## Related Pages

- [Installation](./installation.md) — step-by-step install instructions
- [Platform Support](./platform-support.md) — full OS and distribution support matrix
- [Quickstart](./quickstart.md) — first conversation in under 5 minutes
- [Local Ollama Setup](/guides/local-ollama-setup.md) — run models locally with Ollama
- [Local LLMs on Mac](/guides/local-llm-on-mac.md) — Apple Silicon optimizations
- [Docker](/user-guide/docker.md) — containerized deployment
