---
title: "Obliteratus — OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）"
sidebar_label: "Obliteratus"
description: "OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）"
---
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
# Obliteratus
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
OBLITERATUS: abliterate LLM refusals (diff-in-means).
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## 技能元数据
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
| | |
|---|---|
| | |
| 路径 | `skills/mlops/inference/obliteratus` |
| Version | `2.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Dependencies | `obliteratus`, `torch`, `transformers`, `bitsandbytes`, `accelerate`, `safetensors` |
| 标签 | `消除拒绝、去审查、拒绝移除、LLM、权重投影、SVD、机制可解释性、HuggingFace、模型手术` |
| Related skills | `vllm`, `gguf`, [`huggingface-tokenizers`](/docs/user-guide/skills/optional/mlops/mlops-huggingface-tokenizers) |
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## 参考：完整 SKILL.md
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. 这是代理在技能激活时看到的指令。
:::
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
# Obliteratus
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## What's inside
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
9 CLI methods, 28 analysis modules, 116 model presets across 5 compute tiers, tournament evaluation, and telemetry-driven recommendations.
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
Remove refusal behaviors (guardrails) from open-weight LLMs without retraining or fine-tuning. Uses mechanistic interpretability techniques — including diff-in-means, SVD, whitened SVD, LEACE concept erasure, SAE decomposition, Bayesian kernel projection, and more — to identify and surgically excise refusal directions from model weights while preserving reasoning capabilities.
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
**License warning:** OBLITERATUS is AGPL-3.0. NEVER import it as a Python library. Always invoke via CLI (`obliteratus` command) or subprocess. This keeps Hermes Agent's MIT license clean.
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## Video Guide
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
Walkthrough of OBLITERATUS used by a Hermes agent to abliterate Gemma:
https://www.youtube.com/watch?v=8fG9BrNTeHs ("OBLITERATUS: An AI Agent Removed Gemma 4's Safety Guardrails")
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
Useful when the user wants a visual overview of the end-to-end workflow before running it themselves.
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## When to Use This Skill
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
Trigger when the user:
- Wants to "uncensor" or "abliterate" an LLM
- Asks about removing refusal/guardrails from a model
- Wants to create an uncensored version of Llama, Qwen, Mistral, etc.
- Mentions "refusal removal", "abliteration", "weight projection"
- Wants to analyze how a model's refusal mechanism works
- References OBLITERATUS, abliterator, or refusal directions
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## Step 1: Installation
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
Check if already installed:
```bash
obliteratus --version 2>/dev/null && echo "INSTALLED" || echo "NOT INSTALLED"
```
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
If not installed, clone and install from GitHub:
```bash
git clone https://github.com/elder-plinius/OBLITERATUS.git
cd OBLITERATUS
pip install -e .
# Obliteratus
# Obliteratus
```
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
**IMPORTANT:** Confirm with user before installing. This pulls in ~5-10GB of dependencies (PyTorch, Transformers, bitsandbytes, etc.).
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## Step 2: Check Hardware
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
Before anything, check what GPU is available:
```bash
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU: {gpu}')
    print(f'VRAM: {vram:.1f} GB')
    if vram < 4: print('TIER: tiny (models under 1B)')
    elif vram < 8: print('TIER: small (models 1-4B)')
    elif vram < 16: print('TIER: medium (models 4-9B with 4bit quant)')
    elif vram < 32: print('TIER: large (models 8-32B with 4bit quant)')
    else: print('TIER: frontier (models 32B+)')
else:
    print('NO GPU - only tiny models (under 1B) on CPU')
"
```
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
### VRAM Requirements (with 4-bit quantization)
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
| VRAM     | Max Model Size  | Example Models                              |
|:---------|:----------------|:--------------------------------------------|
| CPU only | ~1B params      | GPT-2, TinyLlama, SmolLM                    |
| 4-8 GB   | ~4B params      | Qwen2.5-1.5B, Phi-3.5 mini, Llama 3.2 3B   |
| 8-16 GB  | ~9B params      | Llama 3.1 8B, Mistral 7B, Gemma 2 9B       |
| 24 GB    | ~32B params     | Qwen3-32B, Llama 3.1 70B (tight), Command-R |
| 48 GB+   | ~72B+ params    | Qwen2.5-72B, DeepSeek-R1                    |
| Multi-GPU| 200B+ params    | Llama 3.1 405B, DeepSeek-V3 (685B MoE)      |
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## Step 3: Browse Available Models & Get Recommendations
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
```bash
# Obliteratus
obliteratus models --tier medium
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
# Obliteratus
obliteratus info <model_name>
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
# Obliteratus
obliteratus recommend <model_name>
obliteratus recommend <model_name> --insights  # global cross-architecture rankings
```
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## Step 4: Choose a Method
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
### Method Selection Guide
**Default / recommended for most cases: `advanced`.** It uses multi-direction SVD with norm-preserving projection and is well-tested.
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
| Situation                         | Recommended Method | Why                                      |
|:----------------------------------|:-------------------|:-----------------------------------------|
| Default / most models             | `advanced`         | Multi-direction SVD, norm-preserving, reliable |
| Quick test / prototyping          | `basic`            | Fast, simple, good enough to evaluate    |
| Dense model (Llama, Mistral)      | `advanced`         | Multi-direction, norm-preserving         |
| MoE model (DeepSeek, Mixtral)     | `nuclear`          | Expert-granular, handles MoE complexity  |
| Reasoning model (R1 distills)     | `surgical`         | CoT-aware, preserves chain-of-thought    |
| Stubborn refusals persist         | `aggressive`       | Whitened SVD + head surgery + jailbreak   |
| Want reversible changes           | Use steering vectors (see Analysis section) |
| Maximum quality, time no object   | `optimized`        | Bayesian search for best parameters      |
| Experimental auto-detection       | `informed`         | Auto-detects alignment type — experimental, may not always outperform advanced |
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
### 9 CLI Methods
- **basic** — Single refusal direction via diff-in-means. Fast (~5-10 min for 8B).
- **advanced** (DEFAULT, RECOMMENDED) — Multiple SVD directions, norm-preserving projection, 2 refinement passes. Medium speed (~10-20 min).
- **aggressive** — Whitened SVD + jailbreak-contrastive + attention head surgery. Higher risk of coherence damage.
- **spectral_cascade** — DCT frequency-domain decomposition. Research/novel approach.
- **informed** — Runs analysis DURING abliteration to auto-configure. Experimental — slower and less predictable than advanced.
- **surgical** — SAE features + neuron masking + head surgery + per-expert. Very slow (~1-2 hrs). Best for reasoning models.
- **optimized** — Bayesian hyperparameter search (Optuna TPE). Longest runtime but finds optimal parameters.
- **inverted** — Flips the refusal direction. Model becomes actively willing.
- **nuclear** — Maximum force combo for stubborn MoE models. Expert-granular.
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
### Direction Extraction Methods (--direction-method flag)
- **diff_means** (default) — Simple difference-in-means between refused/complied activations. Robust.
- **svd** — Multi-direction SVD extraction. Better for complex alignment.
- **leace** — LEACE (Linear Erasure via Closed-form Estimation). Optimal linear erasure.
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
### 4 Python-API-Only Methods
(NOT available via CLI — require Python import, which violates AGPL boundary. Mention to user only if they explicitly want to use OBLITERATUS as a library in their own AGPL project.)
- failspy, gabliteration, heretic, rdo
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## Step 5: Run Abliteration
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
### Standard usage
```bash
# Obliteratus
obliteratus obliterate <model_name> --method advanced --output-dir ./abliterated-models
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
# Obliteratus
obliteratus obliterate <model_name> --method advanced --quantization 4bit --output-dir ./abliterated-models
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
# Obliteratus
obliteratus obliterate <model_name> --method advanced --quantization 4bit --large-model --output-dir ./abliterated-models
```
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
### Fine-tuning parameters
```bash
obliteratus obliterate <model_name> \
  --method advanced \
  --direction-method diff_means \
  --n-directions 4 \
  --refinement-passes 2 \
  --regularization 0.1 \
  --quantization 4bit \
  --output-dir ./abliterated-models \
  --contribute  # opt-in telemetry for community research
```
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
### Key flags
| Flag | Description | Default |
|:-----|:------------|:--------|
| `--method` | Abliteration method | advanced |
| `--direction-method` | Direction extraction | diff_means |
| `--n-directions` | Number of refusal directions (1-32) | method-dependent |
| `--refinement-passes` | Iterative passes (1-5) | 2 |
| `--regularization` | Regularization strength (0.0-1.0) | 0.1 |
| `--quantization` | Load in 4bit or 8bit | none (full precision) |
| `--large-model` | Conservative defaults for 120B+ | false |
| `--output-dir` | Where to save the abliterated model | ./obliterated_model |
| `--contribute` | Share anonymized results for research | false |
| `--verify-sample-size` | Number of test prompts for refusal check | 20 |
| `--dtype` | Model dtype (float16, bfloat16) | auto |
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
### Other execution modes
```bash
# Obliteratus
obliteratus interactive
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
# Obliteratus
obliteratus ui --port 7860
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
# Obliteratus
obliteratus run config.yaml --preset quick
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
# Obliteratus
obliteratus tourney <model_name>
```
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## Step 6: Verify Results
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
After abliteration, check the output metrics:
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
| Metric | Good Value | Warning |
|:-------|:-----------|:--------|
| Refusal rate | &lt; 5% (ideally ~0%) | > 10% means refusals persist |
| Perplexity change | &lt; 10% increase | > 15% means coherence damage |
| KL divergence | &lt; 0.1 | > 0.5 means significant distribution shift |
| Coherence | High / passes qualitative check | Degraded responses, repetition |
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
### If refusals persist (> 10%)
1. Try `aggressive` method
2. Increase `--n-directions` (e.g., 8 or 16)
3. Add `--refinement-passes 3`
4. Try `--direction-method svd` instead of diff_means
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
### If coherence is damaged (perplexity > 15% increase)
1. Reduce `--n-directions` (try 2)
2. Increase `--regularization` (try 0.3)
3. Reduce `--refinement-passes` to 1
4. Try `basic` method (gentler)
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## Step 7: Use the Abliterated Model
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
The output is a standard HuggingFace model directory.
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
```bash
# Obliteratus
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./abliterated-models/<model>')
tokenizer = AutoTokenizer.from_pretrained('./abliterated-models/<model>')
inputs = tokenizer('How do I pick a lock?', return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
# Obliteratus
huggingface-cli upload <username>/<model-name>-abliterated ./abliterated-models/<model>
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
# Obliteratus
vllm serve ./abliterated-models/<model>
```
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## CLI Command Reference
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
| Command | Description |
|:--------|:------------|
| `obliteratus obliterate` | Main abliteration command |
| `obliteratus info <model>` | Print model architecture details |
| `obliteratus models --tier <tier>` | Browse curated models by compute tier |
| `obliteratus recommend <model>` | Telemetry-driven method/param suggestion |
| `obliteratus interactive` | Guided setup wizard |
| `obliteratus tourney <model>` | Tournament: all methods head-to-head |
| `obliteratus run <config.yaml>` | Execute ablation study from YAML |
| `obliteratus strategies` | List all registered ablation strategies |
| `obliteratus report <results.json>` | Regenerate visual reports |
| `obliteratus ui` | Launch Gradio web interface |
| `obliteratus aggregate` | Summarize community telemetry data |
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## Analysis Modules
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
OBLITERATUS includes 28 analysis modules for mechanistic interpretability.
See `skill_view(name="obliteratus", file_path="references/analysis-modules.md")` for the full reference.
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
### Quick analysis commands
```bash
# Obliteratus
obliteratus run analysis-config.yaml --preset quick
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
# Obliteratus
# Obliteratus
# Obliteratus
# Obliteratus
# Obliteratus
# Obliteratus
```
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
### Steering Vectors (Reversible Alternative)
Instead of permanent weight modification, use inference-time steering:
```python
# Obliteratus
from obliteratus.analysis.steering_vectors import SteeringVectorFactory, SteeringHookManager
```
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## Ablation Strategies
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
Beyond direction-based abliteration, OBLITERATUS includes structural ablation strategies:
- **Embedding Ablation** — Target embedding layer components
- **FFN Ablation** — Feed-forward network block removal
- **Head Pruning** — Attention head pruning
- **Layer Removal** — Full layer removal
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
List all available: `obliteratus strategies`
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## Evaluation
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
OBLITERATUS includes built-in evaluation tools:
- Refusal rate benchmarking
- Perplexity comparison (before/after)
- LM Eval Harness integration for academic benchmarks
- Head-to-head competitor comparison
- Baseline performance tracking
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## Platform Support
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
- **CUDA** — Full support (NVIDIA GPUs)
- **Apple Silicon (MLX)** — Supported via MLX backend
- **CPU** — Supported for tiny models (&lt; 1B params)
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## YAML Config Templates
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
Load templates for reproducible runs via `skill_view`:
- `templates/abliteration-config.yaml` — Standard single-model config
- `templates/analysis-study.yaml` — Pre-abliteration analysis study
- `templates/batch-abliteration.yaml` — Multi-model batch processing
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## Telemetry
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
OBLITERATUS can optionally contribute anonymized run data to a global research dataset.
Enable with `--contribute` flag. No personal data is collected — only model name, method, metrics.
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## Common Pitfalls
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
1. **Don't use `informed` as default** — it's experimental and slower. Use `advanced` for reliable results.
2. **Models under ~1B respond poorly to abliteration** — their refusal behaviors are shallow and fragmented, making clean direction extraction difficult. Expect partial results (20-40% remaining refusal). Models 3B+ have cleaner refusal directions and respond much better (often 0% refusal with `advanced`).
3. **`aggressive` can make things worse** — on small models it can damage coherence and actually increase refusal rate. Only use it if `advanced` leaves > 10% refusals on a 3B+ model.
4. **Always check perplexity** — if it spikes > 15%, the model is damaged. Reduce aggressiveness.
5. **MoE models need special handling** — use `nuclear` method for Mixtral, DeepSeek-MoE, etc.
6. **Quantized models can't be re-quantized** — abliterate the full-precision model, then quantize the output.
7. **VRAM estimation is approximate** — 4-bit quant helps but peak usage can spike during extraction.
8. **Reasoning models are sensitive** — use `surgical` for R1 distills to preserve chain-of-thought.
9. **Check `obliteratus recommend`** — telemetry data may have better parameters than defaults.
10. **AGPL license** — never `import obliteratus` in MIT/Apache projects. CLI invocation only.
11. **Large models (70B+)** — always use `--large-model` flag for conservative defaults.
12. **Spectral certification RED is common** — the spectral check often flags "incomplete" even when practical refusal rate is 0%. Check actual refusal rate rather than relying on spectral certification alone.
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
## Complementary Skills
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。
- **vllm** — Serve abliterated models with high throughput
- **gguf** — Convert abliterated models to GGUF for llama.cpp
- **huggingface-tokenizers** — Work with model tokenizers
OBLITERATUS：消除 LLM 拒绝行为（diff-in-means）。