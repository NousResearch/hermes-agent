---
title: "vLLM 推理服务 — vLLM：高吞吐 LLM 推理服务、OpenAI API、量化"
sidebar_label: "vLLM 推理服务"
description: "vLLM：高吞吐 LLM 推理服务、OpenAI API、量化"
---
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
# vLLM 推理服务
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
vLLM: high-throughput LLM serving, OpenAI API, quantization.
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
## 技能元数据
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
| | |
|---|---|
| | |
| 路径 | `skills/mlops/inference/vllm` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `vllm`, `torch`, `transformers` |
| 标签 | `vLLM、推理服务、PagedAttention、连续批处理、高吞吐、生产环境、OpenAI API、量化、张量并行` |
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
## 参考：完整 SKILL.md
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. 这是代理在技能激活时看到的指令。
:::
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
# vLLM 推理服务
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
## When to use
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Use when deploying production LLM APIs, optimizing inference latency/throughput, or serving models with limited GPU memory. Supports OpenAI-compatible endpoints, quantization (GPTQ/AWQ/FP8), and tensor parallelism.
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
## Quick start
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
vLLM achieves 24x higher throughput than standard transformers through PagedAttention (block-based KV cache) and continuous batching (mixing prefill/decode requests).
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Installation**:
```bash
pip install vllm
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Basic offline inference**:
```python
from vllm import LLM, SamplingParams
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
llm = LLM(model="meta-llama/Llama-3-8B-Instruct")
sampling = SamplingParams(temperature=0.7, max_tokens=256)
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
outputs = llm.generate(["Explain quantum computing"], sampling)
print(outputs[0].outputs[0].text)
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**OpenAI-compatible server**:
```bash
vllm serve meta-llama/Llama-3-8B-Instruct
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
# vLLM 推理服务
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')
print(client.chat.completions.create(
    model='meta-llama/Llama-3-8B-Instruct',
    messages=[{'role': 'user', 'content': 'Hello!'}]
).choices[0].message.content)
"
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
## Common workflows
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
### Workflow 1: Production API deployment
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Copy this checklist and track progress:
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
```
Deployment Progress:
- [ ] Step 1: Configure server settings
- [ ] Step 2: Test with limited traffic
- [ ] Step 3: Enable monitoring
- [ ] Step 4: Deploy to production
- [ ] Step 5: Verify performance metrics
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Step 1: Configure server settings**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Choose configuration based on your model size:
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
```bash
# vLLM 推理服务
vllm serve meta-llama/Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --port 8000
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
# vLLM 推理服务
vllm serve meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --quantization awq \
  --port 8000
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
# vLLM 推理服务
vllm serve meta-llama/Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --enable-metrics \
  --metrics-port 9090 \
  --port 8000 \
  --host 0.0.0.0
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Step 2: Test with limited traffic**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Run load test before production:
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
```bash
# vLLM 推理服务
pip install locust
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
# vLLM 推理服务
# vLLM 推理服务
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Verify TTFT (time to first token) &lt; 500ms and throughput > 100 req/sec.
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Step 3: Enable monitoring**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
vLLM exposes Prometheus metrics on port 9090:
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
```bash
curl http://localhost:9090/metrics | grep vllm
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Key metrics to monitor:
- `vllm:time_to_first_token_seconds` - Latency
- `vllm:num_requests_running` - Active requests
- `vllm:gpu_cache_usage_perc` - KV cache utilization
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Step 4: Deploy to production**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Use Docker for consistent deployment:
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
```bash
# vLLM 推理服务
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Step 5: Verify performance metrics**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Check that deployment meets targets:
- TTFT &lt; 500ms (for short prompts)
- Throughput > target req/sec
- GPU utilization > 80%
- No OOM errors in logs
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
### Workflow 2: Offline batch inference
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
For processing large datasets without server overhead.
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Copy this checklist:
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
```
Batch Processing:
- [ ] Step 1: Prepare input data
- [ ] Step 2: Configure LLM engine
- [ ] Step 3: Run batch inference
- [ ] Step 4: Process results
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Step 1: Prepare input data**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
```python
# vLLM 推理服务
prompts = []
with open("prompts.txt") as f:
    prompts = [line.strip() for line in f]
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
print(f"Loaded {len(prompts)} prompts")
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Step 2: Configure LLM engine**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
```python
from vllm import LLM, SamplingParams
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    tensor_parallel_size=2,  # Use 2 GPUs
    gpu_memory_utilization=0.9,
    max_model_len=4096
)
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
sampling = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
    stop=["</s>", "\n\n"]
)
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Step 3: Run batch inference**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
vLLM automatically batches requests for efficiency:
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
```python
# vLLM 推理服务
outputs = llm.generate(prompts, sampling)
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
# vLLM 推理服务
# vLLM 推理服务
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Step 4: Process results**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
```python
# vLLM 推理服务
results = []
for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    results.append({
        "prompt": prompt,
        "generated": generated,
        "tokens": len(output.outputs[0].token_ids)
    })
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
# vLLM 推理服务
import json
with open("results.jsonl", "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
print(f"Processed {len(results)} prompts")
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
### Workflow 3: Quantized model serving
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Fit large models in limited GPU memory.
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
```
Quantization Setup:
- [ ] Step 1: Choose quantization method
- [ ] Step 2: Find or create quantized model
- [ ] Step 3: Launch with quantization flag
- [ ] Step 4: Verify accuracy
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Step 1: Choose quantization method**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
- **AWQ**: Best for 70B models, minimal accuracy loss
- **GPTQ**: Wide model support, good compression
- **FP8**: Fastest on H100 GPUs
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Step 2: Find or create quantized model**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Use pre-quantized models from HuggingFace:
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
```bash
# vLLM 推理服务
# vLLM 推理服务
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Step 3: Launch with quantization flag**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
```bash
# vLLM 推理服务
vllm serve TheBloke/Llama-2-70B-AWQ \
  --quantization awq \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
# vLLM 推理服务
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Step 4: Verify accuracy**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Test outputs match expected quality:
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
```python
# vLLM 推理服务
# vLLM 推理服务
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
## When to use vs alternatives
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Use vLLM when:**
- Deploying production LLM APIs (100+ req/sec)
- Serving OpenAI-compatible endpoints
- Limited GPU memory but need large models
- Multi-user applications (chatbots, assistants)
- Need low latency with high throughput
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Use alternatives instead:**
- **llama.cpp**: CPU/edge inference, single-user
- **HuggingFace transformers**: Research, prototyping, one-off generation
- **TensorRT-LLM**: NVIDIA-only, need absolute maximum performance
- **Text-Generation-Inference**: Already in HuggingFace ecosystem
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
## Common issues
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Issue: Out of memory during model loading**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Reduce memory usage:
```bash
vllm serve MODEL \
  --gpu-memory-utilization 0.7 \
  --max-model-len 4096
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Or use quantization:
```bash
vllm serve MODEL --quantization awq
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Issue: Slow first token (TTFT > 1 second)**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Enable prefix caching for repeated prompts:
```bash
vllm serve MODEL --enable-prefix-caching
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
For long prompts, enable chunked prefill:
```bash
vllm serve MODEL --enable-chunked-prefill
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Issue: Model not found error**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Use `--trust-remote-code` for custom models:
```bash
vllm serve MODEL --trust-remote-code
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Issue: Low throughput (&lt;50 req/sec)**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Increase concurrent sequences:
```bash
vllm serve MODEL --max-num-seqs 512
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Check GPU utilization with `nvidia-smi` - should be >80%.
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Issue: Inference slower than expected**
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Verify tensor parallelism uses power of 2 GPUs:
```bash
vllm serve MODEL --tensor-parallel-size 4  # Not 3
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Enable speculative decoding for faster generation:
```bash
vllm serve MODEL --speculative-model DRAFT_MODEL
```
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
## Advanced topics
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Server deployment patterns**: See [references/server-deployment.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/vllm/references/server-deployment.md) for Docker, Kubernetes, and load balancing configurations.
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Performance optimization**: See [references/optimization.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/vllm/references/optimization.md) for PagedAttention tuning, continuous batching details, and benchmark results.
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Quantization guide**: See [references/quantization.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/vllm/references/quantization.md) for AWQ/GPTQ/FP8 setup, model preparation, and accuracy comparisons.
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
**Troubleshooting**: See [references/troubleshooting.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/vllm/references/troubleshooting.md) for detailed error messages, debugging steps, and performance diagnostics.
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
## Hardware requirements
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
- **Small models (7B-13B)**: 1x A10 (24GB) or A100 (40GB)
- **Medium models (30B-40B)**: 2x A100 (40GB) with tensor parallelism
- **Large models (70B+)**: 4x A100 (40GB) or 2x A100 (80GB), use AWQ/GPTQ
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
Supported platforms: NVIDIA (primary), AMD ROCm, Intel GPUs, TPUs
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
## Resources
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。
- Official docs: https://docs.vllm.ai
- GitHub: https://github.com/vllm-project/vllm
- Paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
- Community: https://discuss.vllm.ai
vLLM：高吞吐 LLM 推理服务、OpenAI API、量化。