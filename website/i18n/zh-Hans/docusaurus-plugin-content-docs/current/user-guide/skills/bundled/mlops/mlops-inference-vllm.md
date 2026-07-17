---
title: "Serving Llms Vllm — vLLM：高吞吐量 LLM 服务、OpenAI API、量化"
sidebar_label: "Serving Llms Vllm"
description: "vLLM：高吞吐量 LLM 服务、OpenAI API、量化"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Serving Llms Vllm

vLLM：高吞吐量 LLM 服务、OpenAI API、量化。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/mlops/inference/vllm` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖 | `vllm`, `torch`, `transformers` |
| 平台 | linux, macos |
| 标签 | `vLLM`, `Inference Serving`, `PagedAttention`, `Continuous Batching`, `High Throughput`, `Production`, `OpenAI API`, `Quantization`, `Tensor Parallelism` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# vLLM - 高性能 LLM 服务

## 适用场景

在部署生产级 LLM API、优化推理延迟/吞吐量，或在 GPU 显存有限的情况下服务模型时使用。支持 OpenAI 兼容端点、量化（GPTQ/AWQ/FP8）以及张量并行。

## 快速开始

vLLM 通过 PagedAttention（基于块的 KV 缓存）和 continuous batching（混合 prefill/decode 请求）实现比标准 transformers 高 24 倍的吞吐量。

**安装**：
```bash
pip install vllm
```

**基础离线推理**：
```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B-Instruct")
sampling = SamplingParams(temperature=0.7, max_tokens=256)

outputs = llm.generate(["Explain quantum computing"], sampling)
print(outputs[0].outputs[0].text)
```

**OpenAI 兼容服务器**：
```bash
vllm serve meta-llama/Llama-3-8B-Instruct

# Query with OpenAI SDK
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')
print(client.chat.completions.create(
    model='meta-llama/Llama-3-8B-Instruct',
    messages=[{'role': 'user', 'content': 'Hello!'}]
).choices[0].message.content)
"
```

## 常见工作流

### 工作流 1：生产 API 部署

复制此清单并跟踪进度：

```
Deployment Progress:
- [ ] Step 1: Configure server settings
- [ ] Step 2: Test with limited traffic
- [ ] Step 3: Enable monitoring
- [ ] Step 4: Deploy to production
- [ ] Step 5: Verify performance metrics
```

**步骤 1：配置服务器设置**

根据模型大小选择配置：

```bash
# For 7B-13B models on single GPU
vllm serve meta-llama/Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --port 8000

# For 30B-70B models with tensor parallelism
vllm serve meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --quantization awq \
  --port 8000

# For production with caching and metrics
vllm serve meta-llama/Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --enable-metrics \
  --metrics-port 9090 \
  --port 8000 \
  --host 0.0.0.0
```

**步骤 2：使用有限流量测试**

在生产前运行负载测试：

```bash
# Install load testing tool
pip install locust

# Create test_load.py with sample requests
# Run: locust -f test_load.py --host http://localhost:8000
```

验证 TTFT（首 token 时间）&lt; 500ms，吞吐量 > 100 req/sec。

**步骤 3：启用监控**

vLLM 在端口 9090 上暴露 Prometheus 指标：

```bash
curl http://localhost:9090/metrics | grep vllm
```

需监控的关键指标：
- `vllm:time_to_first_token_seconds` - 延迟
- `vllm:num_requests_running` - 活跃请求数
- `vllm:gpu_cache_usage_perc` - KV 缓存利用率

**步骤 4：部署到生产环境**

使用 Docker 实现一致性部署：

```bash
# Run vLLM in Docker
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching
```

**步骤 5：验证性能指标**

检查部署是否达到目标：
- TTFT &lt; 500ms（短 prompt 情况下）
- 吞吐量 > 目标 req/sec
- GPU 利用率 > 80%
- 日志中无 OOM 错误

### 工作流 2：离线批量推理

用于处理大型数据集，无需服务器开销。

复制此清单：

```
Batch Processing:
- [ ] Step 1: Prepare input data
- [ ] Step 2: Configure LLM engine
- [ ] Step 3: Run batch inference
- [ ] Step 4: Process results
```

**步骤 1：准备输入数据**

```python
# Load prompts from file
prompts = []
with open("prompts.txt") as f:
    prompts = [line.strip() for line in f]

print(f"Loaded {len(prompts)} prompts")
```

**步骤 2：配置 LLM 引擎**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    tensor_parallel_size=2,  # Use 2 GPUs
    gpu_memory_utilization=0.9,
    max_model_len=4096
)

sampling = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
    stop=["</s>", "\n\n"]
)
```

**步骤 3：运行批量推理**

vLLM 自动对请求进行批处理以提升效率：

```python
# Process all prompts in one call
outputs = llm.generate(prompts, sampling)

# vLLM handles batching internally
# No need to manually chunk prompts
```

**步骤 4：处理结果**

```python
# Extract generated text
results = []
for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    results.append({
        "prompt": prompt,
        "generated": generated,
        "tokens": len(output.outputs[0].token_ids)
    })

# Save to file
import json
with open("results.jsonl", "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

print(f"Processed {len(results)} prompts")
```

### 工作流 3：量化模型服务

在有限 GPU 显存中运行大型模型。

```
Quantization Setup:
- [ ] Step 1: Choose quantization method
- [ ] Step 2: Find or create quantized model
- [ ] Step 3: Launch with quantization flag
- [ ] Step 4: Verify accuracy
```

**步骤 1：选择量化方法**

- **AWQ**：最适合 70B 模型，精度损失极小
- **GPTQ**：模型支持范围广，压缩效果好
- **FP8**：在 H100 GPU 上速度最快

**步骤 2：查找或创建量化模型**

使用 HuggingFace 上的预量化模型：

```bash
# Search for AWQ models
# Example: TheBloke/Llama-2-70B-AWQ
```

**步骤 3：使用量化标志启动**

```bash
# Using pre-quantized model
vllm serve TheBloke/Llama-2-70B-AWQ \
  --quantization awq \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95

# Results: 70B model in ~40GB VRAM
```

**步骤 4：验证精度**

测试输出是否符合预期质量：

```python
# Compare quantized vs non-quantized responses
# Verify task-specific performance unchanged
```

## 与替代方案的对比

**使用 vLLM 的场景：**
- 部署生产级 LLM API（100+ req/sec）
- 提供 OpenAI 兼容端点
- GPU 显存有限但需要运行大型模型
- 多用户应用（聊天机器人、助手）
- 需要低延迟与高吞吐量并存

**改用替代方案的场景：**
- **llama.cpp**：CPU/边缘推理，单用户场景
- **HuggingFace transformers**：研究、原型开发、一次性生成
- **TensorRT-LLM**：仅限 NVIDIA，追求绝对最高性能
- **Text-Generation-Inference**：已在 HuggingFace 生态系统中

## 常见问题

**问题：模型加载时内存不足**

减少内存使用：
```bash
vllm serve MODEL \
  --gpu-memory-utilization 0.7 \
  --max-model-len 4096
```

或使用量化：
```bash
vllm serve MODEL --quantization awq
```

**问题：首 token 速度慢（TTFT > 1 秒）**

对重复 prompt 启用前缀缓存：
```bash
vllm serve MODEL --enable-prefix-caching
```

对长 prompt，启用分块 prefill：
```bash
vllm serve MODEL --enable-chunked-prefill
```

**问题：模型未找到错误**

对自定义模型使用 `--trust-remote-code`：
```bash
vllm serve MODEL --trust-remote-code
```

**问题：吞吐量低（&lt;50 req/sec）**

增加并发序列数：
```bash
vllm serve MODEL --max-num-seqs 512
```

使用 `nvidia-smi` 检查 GPU 利用率——应高于 80%。

**问题：推理速度低于预期**

验证张量并行使用的 GPU 数量为 2 的幂次：
```bash
vllm serve MODEL --tensor-parallel-size 4  # Not 3
```

启用推测解码以加速生成：
```bash
vllm serve MODEL --speculative-model DRAFT_MODEL
```

## 高级主题

**服务器部署模式**：参见 [references/server-deployment.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/vllm/references/server-deployment.md)，了解 Docker、Kubernetes 和负载均衡配置。

**性能优化**：参见 [references/optimization.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/vllm/references/optimization.md)，了解 PagedAttention 调优、continuous batching 详情及基准测试结果。

**量化指南**：参见 [references/quantization.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/vllm/references/quantization.md)，了解 AWQ/GPTQ/FP8 配置、模型准备及精度对比。

**故障排查**：参见 [references/troubleshooting.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/vllm/references/troubleshooting.md)，了解详细错误信息、调试步骤及性能诊断。

## 硬件要求

- **小型模型（7B-13B）**：1x A10（24GB）或 A100（40GB）
- **中型模型（30B-40B）**：2x A100（40GB），使用张量并行
- **大型模型（70B+）**：4x A100（40GB）或 2x A100（80GB），使用 AWQ/GPTQ

支持平台：NVIDIA（主要）、AMD ROCm、Intel GPU、TPU

## 资源

- 官方文档：https://docs.vllm.ai
- GitHub：https://github.com/vllm-project/vllm
- 论文："Efficient Memory Management for Large Language Model Serving with PagedAttention"（SOSP 2023）
- 社区：https://discuss.vllm.ai