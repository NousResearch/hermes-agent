---
title: "Obliteratus — Abliterate refusals from local open-weight LLMs"
sidebar_label: "Obliteratus"
description: "Abliterate refusals from local open-weight LLMs"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Obliteratus

Abliterate refusals from local open-weight LLMs.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/mlops/obliteratus` |
| Path | `optional-skills/mlops/obliteratus` |
| Version | `2.1.0` |
| Author | mr-r0b0t (@am423), Hermes Agent |
| License | MIT |
| Platforms | linux, macos |
| Tags | `Abliteration`, `Uncensoring`, `Refusal-Removal`, `LLM`, `Weight-Projection`, `SVD`, `HuggingFace`, `Model-Surgery` |
| Related skills | [`serving-llms-vllm`](/docs/user-guide/skills/optional/mlops/mlops-inference-serving-llms-vllm), [`llama-cpp`](/docs/user-guide/skills/optional/mlops/mlops-inference-llama-cpp), [`huggingface-tokenizers`](/docs/user-guide/skills/optional/mlops/mlops-huggingface-tokenizers) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# OBLITERATUS Skill

OBLITERATUS (package `0.1.3`, AGPL-3.0) projects refusal directions out of a local HuggingFace checkpoint without fine-tuning. It does not retrain, does not serve the result, and does not guarantee 0% refusal on models under ~1B. Never `import obliteratus` inside Hermes (MIT); invoke the CLI through the `terminal` tool only.

Upstream: https://github.com/elder-plinius/OBLITERATUS

## When to Use

- User wants to abliterate, uncensor, or remove refusals/guardrails from an open-weight LLM
- The model is already on disk as a HuggingFace directory, or a hub id that will resolve locally
- Residual refusals remain after a first pass (`self-improve`, `--residue-file`)
- Qwen3.8 hybrid Gated DeltaNet local surgery

Do not use for hosted APIs, for importing the library into Hermes, or as a substitute for serving — load `serving-llms-vllm` or `llama-cpp` after the weights are saved.

## Prerequisites

- Python >= 3.10 and a CUDA GPU for anything above ~1B (CPU is tiny-models only)
- A local snapshot: `config.json` plus weights. Confirm with `read_file` on `config.json`
- Optional `HF_TOKEN` only for gated hub ids; local directories do not need it

Install (confirm with the user first — PyTorch-scale deps) through `terminal`:

```bash
git clone https://github.com/elder-plinius/OBLITERATUS.git
cd OBLITERATUS
pip install -e ".[quantization]"
obliteratus --version
```

- Qwen3.8 / hybrid Gated DeltaNet: also `pip install -e ".[qwen-hybrid]"` (`transformers>=5.8`, `flash-linear-attention>=0.5.2`, `causal-conv1d>=1.7.0`)
- Gradio UI: `pip install -e ".[spaces]"`
- After a CUDA wheel override, do not launch via `uv run` — it can resync the CPU wheel locked for portable CI

## How to Run

The `model` argument is a HuggingFace id **or a local directory**. Always pass `--output-dir`.

```bash
obliteratus obliterate /abs/path/to/ckpt \
  --method advanced \
  --dtype bfloat16 \
  --output-dir ./liberated \
  --gpus 0
```

VRAM-tight (requires `.[quantization]`):

```bash
obliteratus obliterate /abs/path/to/ckpt \
  --method advanced \
  --quantization 4bit \
  --dtype float16 \
  --output-dir ./liberated
```

Qwen3.8: one CUDA device, no generic multi-GPU shard. Load `references/local-models.md` with `skill_view`.

## Quick Reference

| Command | Purpose |
|---|---|
| `obliteratus obliterate <id-or-path>` | Weight-projection refusal removal (`abliterate` is a hidden alias) |
| `obliteratus info <id-or-path>` | Architecture print |
| `obliteratus models --tier {tiny,small,medium,large,frontier}` | Curated presets |
| `obliteratus recommend <id-or-path>` | Telemetry-driven method/params (`--insights` for global) |
| `obliteratus gpu-calc <id-or-path> --gpu-mem 80` | Minimum GPU count estimate |
| `obliteratus self-improve <path> --audit <json> --output-dir <dir>` | Residue / hard-negative follow-up |
| `obliteratus capability-check --abliterated <p> --stock <p>` | MMLU vs stock (`--quick`) |
| `obliteratus tourney <id-or-path>` | Method tournament |
| `obliteratus runs launch -- <args>` | Detached durable run |
| `obliteratus ui --port 7860` | Local Gradio |
| `obliteratus interactive` | Guided wizard |

`--method`: `basic`, `advanced` (default), `aggressive`, `spectral_cascade`, `informed`, `surgical`, `optimized`, `som`, `inverted`, `nuclear`.

`--direction-method`: `diff_means`, `svd`, `leace`, `som`. `--quantization`: `4bit`, `8bit`.

## Procedure

1. Confirm the checkpoint with `read_file` on `<ckpt>/config.json`. Prefer the on-disk tree; do not re-download a sibling official copy.
2. Check GPU idle state through `terminal`. Pin an idle device with `--gpus N`. Do not steal a live serve.
3. Size the job: `obliteratus gpu-calc <ckpt> --gpu-mem <GB>`. Native BF16 surgery needs ~2 bytes/param plus activations; `--quantization 4bit` is load-time only. FP8/NVFP4 checkpoints dequantize automatically — peak VRAM is the **BF16** size; saved output is BF16.
4. Optional: `obliteratus recommend <ckpt>` then `obliteratus info <ckpt>`.
5. Run `obliteratus obliterate` with `--method advanced` unless `references/methods-guide.md` says otherwise (precision MoE → `surgical`, quality auto-tune → `optimized`).
6. Read the printed metrics. Target refusal rate ≤ 5% (ideally ~0–3%). Fail-closed defaults: `--max-perplexity-increase 3.0` (multiple of stock), `--min-coherence-retention 0.5`, `--max-degenerate-fraction 0.2`, `--verify-sample-size 30`.
7. If refusals remain: `obliteratus self-improve <ckpt> --audit <json> --output-dir <next>`, or rerun with `--residue-file`, more `--n-directions`, `--refinement-passes 3`, `--direction-method svd`, or `--method aggressive`.
8. Optional capability gate: `obliteratus capability-check --abliterated <out> --stock <ckpt> --quick`.
9. The output is a standard HuggingFace directory. Serve with the related serving skills — not with `import obliteratus`.

Load `references/local-models.md` when the checkpoint is a local path, Qwen3.8, FP8/NVFP4, or a residue loop. Load `references/methods-guide.md` for method choice. Load `references/analysis-modules.md` only when the user asks to map refusal geometry first.

## Pitfalls

1. **AGPL boundary.** Never `from obliteratus import ...` in Hermes or other MIT/Apache code. CLI / subprocess only.
2. **`--quantization` values are `4bit` and `8bit`.** Not `bitsandbytes-4bit`. Extra `.[quantization]` must be installed or the flag fails.
3. **Qwen3.8 hybrid runtime.** Needs `.[qwen-hybrid]`, FLA + causal-conv1d CUDA kernels, the full text model on **one** CUDA device with 15% headroom. Generic PyTorch fallback and `device_map="auto"` sharding are rejected. A failed pristine load is never modified.
4. **`--large-model` is for 120B+**, not 70B. Conservative defaults: fewer directions, 1 pass.
5. **Sub-1B models respond poorly.** Expect leftover refusal. 3B+ is the realistic near-zero-refusal regime.
6. **`informed` is slower and experimental.** Default is `advanced`.
7. **`aggressive` can damage coherence** on small models. Use only when `advanced` leaves >10% refusal on 3B+.
8. **Quantized-load ≠ quantized-save.** Abliterate, then re-quantize the BF16 output if you need a 4-bit serving artifact.
9. **Multi-GPU is memory, not speed.** `--gpus 0,1` shards via accelerate pipeline parallel; Qwen3.8 forbids that path.
10. **Spectral "incomplete" is not a fail.** Trust measured refusal rate, coherence, and degeneracy.
11. **`--contribute` is opt-in telemetry.** Off by default on CLI (Spaces turns it on).
12. **Do not use `uv run` after replacing the CPU-locked CI torch wheel with CUDA.**

## Verification

```bash
obliteratus --version
obliteratus info /abs/path/to/ckpt
```

A liberated tree is valid when it contains `config.json` plus safetensors, `obliteratus info <output-dir>` prints the architecture, and the CLI summary shows refusal rate ≤ 5% with degeneracy below `0.2`.
