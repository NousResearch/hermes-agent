---
title: "Slime Rl 训练ing — 使用 slime（一个 Megatron+SGLang 框架）进行 LLM 后训练 RL 的指南"
sidebar_label: "Slime Rl 训练ing"
description: "使用 slime（一个 Megatron+SGLang 框架）进行 LLM 后训练 RL 的指南"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Slime Rl 训练ing

使用 slime（一个 Megatron+SGLang 框架）进行 LLM 后训练 RL 的指南。适用于训练 GLM 模型、实现自定义数据生成工作流，或需要深度集成 Megatron-LM 以实现 RL 扩展的场景。

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/mlops/slime` |
| Path | `optional-skills/mlops/slime` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `sglang-router>=0.2.3`, `ray`, `torch>=2.0.0`, `transformers>=4.40.0` |
| Tags | `Reinforcement Learning`, `Megatron-LM`, `SGLang`, `GRPO`, `Post-训练ing`, `GLM` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# slime: LLM Post-训练ing Framework for RL Scaling

slime 是清华 THUDM 团队的 LLM 后训练框架，支撑 GLM-4.5、GLM-4.6 和 GLM-4.7。它将用于训练的 Megatron-LM 与用于高吞吐量 rollout 生成的 SGLang 相连接。

## When to Use slime

**Choose slime when you need:**
- Megatron-LM native training with SGLang inference
- Custom data generation workflows with flexible data buffers
- 训练ing GLM, Qwen3, DeepSeek V3, or Llama 3 models
- Research-grade framework with production backing (Z.ai)

**Consider alternatives when:**
- You need enterprise-grade stability features → use **miles**
- You want flexible backend swapping → use **verl**
- You need PyTorch-native abstractions → use **torchforge**

## Key Features

- **训练ing**: Megatron-LM with full parallelism support (TP, PP, DP, SP)
- **Rollout**: SGLang-based high-throughput generation with router
- **Data Buffer**: Flexible prompt management and sample storage
- **Models**: GLM-4.x, Qwen3, DeepSeek V3/R1, Llama 3

## 架构概述

<!-- ascii-guard-ignore -->
```
┌─────────────────────────────────────────────────────────┐
│                    Data Buffer                          │
│ - Prompt initialization and management                  │
│ - Custom data generation and filtering                  │
│ - Rollout sample storage                                │
└─────────────┬───────────────────────────┬───────────────┘
              │                           │
┌─────────────▼───────────┐ ┌─────────────▼───────────────┐
│ 训练ing (Megatron-LM)  │ │ Rollout (SGLang + Router)   │
│ - Actor model training  │ │ - Response generation       │
│ - Critic (optional)     │ │ - Reward/verifier output    │
│ - Weight sync to rollout│ │ - Multi-turn support        │
└─────────────────────────┘ └─────────────────────────────┘
```
<!-- ascii-guard-ignore-end -->

## 安装

```bash
# 推荐: Docker
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -it slimerl/slime:latest /bin/bash

# Inside container
cd /root/slime && pip install -e . --no-deps
```

### 来自 Source

```bash
git clone https://github.com/THUDM/slime.git
cd slime
pip install -r requirements.txt
pip install -e .
```

## Quick Start: GRPO 训练ing

```bash
# Source model configuration
source scripts/models/qwen3-4B.sh

# 启动 training
python train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --rollout-num-gpus 4 \
    --advantage-estimator grpo \
    --use-kl-loss --kl-loss-coef 0.001 \
    --rollout-batch-size 32 \
    --n-samples-per-prompt 8 \
    --global-batch-size 256 \
    --num-rollout 3000 \
    --prompt-data /path/to/data.jsonl \
    ${MODEL_ARGS[@]} ${CKPT_ARGS[@]}
```

---

## Workflow 1: Standard GRPO 训练ing

Use this workflow for training reasoning models with group-relative advantages.

### Prerequisites Checklist
- [ ] Docker environment or Megatron-LM + SGLang installed
- [ ] Model checkpoint (HuggingFace or Megatron format)
- [ ] 训练ing data in JSONL format

### Step 1: Prepare Data

```python
# data.jsonl format
{"prompt": "What is 2 + 2?", "label": "4"}
{"prompt": "Solve: 3x = 12", "label": "x = 4"}
```

Or with chat format:
```python
{
    "prompt": [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 15 + 27?"}
    ],
    "label": "42"
}
```

### Step 2: 配置 Model

Choose a pre-configured model script:

```bash
# List available models
ls scripts/models/
# glm4-9B.sh, qwen3-4B.sh, qwen3-30B-A3B.sh, deepseek-v3.sh, llama3-8B.sh, ...

# Source your model
source scripts/models/qwen3-4B.sh
```

### Step 3: 启动 训练ing

```bash
python train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus 8 \
    --advantage-estimator grpo \
    --use-kl-loss \
    --kl-loss-coef 0.001 \
    --prompt-data /path/to/train.jsonl \
    --input-key prompt \
    --label-key label \
    --apply-chat-template \
    --rollout-batch-size 32 \
    --n-samples-per-prompt 8 \
    --global-batch-size 256 \
    --num-rollout 3000 \
    --save-interval 100 \
    --eval-interval 50 \
    ${MODEL_ARGS[@]}
```

### Step 4: Monitor 训练ing
- [ ] Check TensorBoard: `tensorboard --logdir outputs/`
- [ ] Verify reward curves are increasing
- [ ] Monitor GPU utilization across nodes

---

## Workflow 2: Asynchronous 训练ing

Use async mode for higher throughput by overlapping rollout and training.

### When to Use Async
- Large models with long generation times
- High GPU idle time in synchronous mode
- Sufficient memory for buffering

### 启动 Async 训练ing

```bash
python train_async.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus 8 \
    --advantage-estimator grpo \
    --async-buffer-size 4 \
    --prompt-data /path/to/train.jsonl \
    ${MODEL_ARGS[@]}
```

### Async-Specific Parameters

```bash
--async-buffer-size 4        # Number of rollouts to buffer
--update-weights-interval 2  # Sync weights every N rollouts
```

---

## Workflow 3: Multi-Turn Agentic 训练ing

Use this workflow for training agents with tool use or multi-step reasoning.

### Prerequisites
- [ ] Custom generate function for multi-turn logic
- [ ] Tool/environment interface

### Step 1: Define Custom Generate Function

```python
# custom_generate.py
async def custom_generate(args, samples, evaluation=False):
    """Multi-turn generation with tool calling."""
    for sample in samples:
        conversation = sample.prompt

        for turn in range(args.max_turns):
            # Generate response
            response = await generate_single(conversation)

            # Check for tool call
            tool_call = extract_tool_call(response)
            if tool_call:
                tool_result = execute_tool(tool_call)
                conversation.append({"role": "assistant", "content": response})
                conversation.append({"role": "tool", "content": tool_result})
            else:
                break

        sample.response = response
        sample.reward = compute_reward(sample)

    return samples
```

### Step 2: 启动 with Custom Function

```bash
python train.py \
    --custom-generate-function-path custom_generate.py \
    --max-turns 5 \
    --prompt-data /path/to/agent_data.jsonl \
    ${MODEL_ARGS[@]}
```

See `examples/search-r1/` for a complete multi-turn search example.

---

## Configuration Reference

### Three Argument Categories

slime uses three types of arguments:

**1. Megatron Arguments** (passed directly):
```bash
--tensor-model-parallel-size 2
--pipeline-model-parallel-size 1
--num-layers 32
--hidden-size 4096
```

**2. SGLang Arguments** (prefixed with `--sglang-`):
```bash
--sglang-mem-fraction-static 0.8
--sglang-context-length 8192
--sglang-log-level INFO
```

**3. slime Arguments**:
```bash
# Resource allocation
--actor-num-nodes 1
--actor-num-gpus-per-node 8
--rollout-num-gpus 8
--colocate  # Share GPUs between training/inference

# Data
--prompt-data /path/to/data.jsonl
--input-key prompt
--label-key label

# 训练ing loop
--num-rollout 3000
--rollout-batch-size 32
--n-samples-per-prompt 8
--global-batch-size 256

# Algorithm
--advantage-estimator grpo  # or: gspo, ppo, reinforce_plus_plus
--use-kl-loss
--kl-loss-coef 0.001
```

### Key Constraints

```
rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout
```

Example: 32 × 8 = 256 × 1

---

## Data Buffer System

slime's data buffer enables flexible data management:

### Basic Data Source

```python
class RolloutDataSource:
    def get_samples(self, num_samples):
        """Fetch prompts from dataset."""
        return self.dataset.sample(num_samples)

    def add_samples(self, samples):
        """Called after generation (no-op by default)."""
        pass
```

### Buffered Data Source (Off-Policy)

```python
class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self):
        self.buffer = []

    def add_samples(self, samples):
        """Store generated samples for reuse."""
        self.buffer.extend(samples)

    def buffer_filter(self, args, buffer, num_samples):
        """Custom selection logic (prioritized, stratified, etc.)."""
        return select_best(buffer, num_samples)
```

---

## Common Issues and Solutions

### Issue: SGLang Engine Crash

**Symptoms**: Inference engine dies mid-training

**Solutions**:
```bash
# Enable fault tolerance
--use-fault-tolerance

# Increase memory allocation
--sglang-mem-fraction-static 0.85

# Reduce batch size
--rollout-batch-size 16
```

### Issue: Weight Sync Timeout

**Symptoms**: 训练ing hangs after rollout

**Solutions**:
```bash
# Increase sync interval
--update-weights-interval 5

# Use colocated mode (no network transfer)
--colocate
```

### Issue: OOM During 训练ing

**Symptoms**: CUDA OOM in backward pass

**Solutions**:
```bash
# Enable gradient checkpointing
--recompute-activations

# Reduce micro-batch size
--micro-batch-size 1

# Enable sequence parallelism
--sequence-parallel
```

### Issue: Slow Data Loading

**Symptoms**: GPU idle during data fetch

**Solutions**:
```bash
# Increase data workers
--num-data-workers 4

# Use streaming dataset
--streaming-data
```

---

## 支持 Models

| Model Family | Configurations |
|--------------|----------------|
| GLM | GLM-4.5, GLM-4.6, GLM-4.7, GLM-Z1-9B |
| Qwen | Qwen3 (4B, 8B, 30B-A3B), Qwen3-MoE, Qwen2.5 |
| DeepSeek | V3, V3.1, R1 |
| Llama | Llama 3 (8B, 70B) |
| Others | Kimi K2, Moonlight-16B |

Each model has pre-configured scripts in `scripts/models/`.

---

## Advanced Topics

### Co-location Mode

Share GPUs between training and inference to reduce memory:

```bash
python train.py \
    --colocate \
    --actor-num-gpus-per-node 8 \
    --sglang-mem-fraction-static 0.4 \
    ${MODEL_ARGS[@]}
```

### Custom Reward Model

```python
# custom_rm.py
class CustomRewardModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def compute_reward(self, prompts, responses):
        inputs = self.tokenize(prompts, responses)
        scores = self.model(inputs)
        return scores.tolist()
```

```bash
--custom-rm-path custom_rm.py
```

### Evaluation Multi-Task

```bash
--eval-prompt-data aime /path/to/aime.jsonl \
--eval-prompt-data gsm8k /path/to/gsm8k.jsonl \
--n-samples-per-eval-prompt 16
```

---

## Resources

- **Documentation**: https://thudm.github.io/slime/
- **GitHub**: https://github.com/THUDM/slime
- **Blog**: https://lmsys.org/blog/2025-07-09-slime/
- **Examples**: See `examples/` directory for 14+ worked examples
