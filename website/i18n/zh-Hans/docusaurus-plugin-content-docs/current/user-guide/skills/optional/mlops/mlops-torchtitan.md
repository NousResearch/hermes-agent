---
title: "Distributed Llm Pretraining Torchtitan"
sidebar_label: "Distributed Llm Pretraining Torchtitan"
description: "使用 torchtitan 提供 PyTorch 原生分布式 LLM 预训练，支持 4D 并行（FSDP2、TP、PP、CP）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Distributed Llm Pretraining Torchtitan

使用 torchtitan 提供 PyTorch 原生分布式 LLM 预训练，支持 4D 并行（FSDP2、TP、PP、CP）。适用于在 8 到 512+ GPU 规模下预训练 Llama 3.1、DeepSeek V3 或自定义模型，支持 Float8、torch.compile 及分布式检查点。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/torchtitan` 安装 |
| 路径 | `optional-skills/mlops/torchtitan` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖 | `torch>=2.6.0`, `torchtitan>=0.2.0`, `torchao>=0.5.0` |
| 平台 | linux, macos |
| 标签 | `Model Architecture`, `Distributed Training`, `TorchTitan`, `FSDP2`, `Tensor Parallel`, `Pipeline Parallel`, `Context Parallel`, `Float8`, `Llama`, `Pretraining` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# TorchTitan - PyTorch 原生分布式 LLM 预训练

## 快速开始

TorchTitan 是 PyTorch 官方的大规模 LLM 预训练平台，支持可组合的 4D 并行（FSDP2、TP、PP、CP），在 H100 GPU 上相比基线可实现 65%+ 的加速。

**安装**：
```bash
# 从 PyPI 安装（稳定版）
pip install torchtitan

# 从源码安装（最新特性，需要 PyTorch nightly）
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install -r requirements.txt
```

**下载 tokenizer**：
```bash
# 从 https://huggingface.co/settings/tokens 获取 HF token
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
```

**在 8 个 GPU 上启动训练**：
```bash
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh
```

## 常用工作流

### 工作流 1：在单节点上预训练 Llama 3.1 8B

复制此检查清单：

```
单节点预训练：
- [ ] 步骤 1：下载 tokenizer
- [ ] 步骤 2：配置训练
- [ ] 步骤 3：启动训练
- [ ] 步骤 4：监控与检查点
```

**步骤 1：下载 tokenizer**

```bash
python scripts/download_hf_assets.py \
  --repo_id meta-llama/Llama-3.1-8B \
  --assets tokenizer \
  --hf_token=YOUR_HF_TOKEN
```

**步骤 2：配置训练**

编辑或创建 TOML 配置文件：

```toml
# llama3_8b_custom.toml
[job]
dump_folder = "./outputs"
description = "Llama 3.1 8B training"

[model]
name = "llama3"
flavor = "8B"
hf_assets_path = "./assets/hf/Llama-3.1-8B"

[optimizer]
name = "AdamW"
lr = 3e-4

[lr_scheduler]
warmup_steps = 200

[training]
local_batch_size = 2
seq_len = 8192
max_norm = 1.0
steps = 1000
dataset = "c4"

[parallelism]
data_parallel_shard_degree = -1  # Use all GPUs for FSDP

[activation_checkpoint]
mode = "selective"
selective_ac_option = "op"

[checkpoint]
enable = true
folder = "checkpoint"
interval = 500
```

**步骤 3：启动训练**

```bash
# 单节点 8 个 GPU
CONFIG_FILE="./llama3_8b_custom.toml" ./run_train.sh

# 或显式使用 torchrun
torchrun --nproc_per_node=8 \
  -m torchtitan.train \
  --job.config_file ./llama3_8b_custom.toml
```

**步骤 4：监控与检查点**

TensorBoard 日志保存至 `./outputs/tb/`：
```bash
tensorboard --logdir ./outputs/tb
```

### 工作流 2：使用 SLURM 进行多节点训练

```
多节点训练：
- [ ] 步骤 1：为规模配置并行度
- [ ] 步骤 2：设置 SLURM 脚本
- [ ] 步骤 3：提交作业
- [ ] 步骤 4：从检查点恢复
```

**步骤 1：为规模配置并行度**

在 256 个 GPU（32 个节点）上训练 70B 模型：
```toml
[parallelism]
data_parallel_shard_degree = 32  # FSDP across 32 ranks
tensor_parallel_degree = 8        # TP within node
pipeline_parallel_degree = 1      # No PP for 70B
context_parallel_degree = 1       # Increase for long sequences
```

**步骤 2：设置 SLURM 脚本**

```bash
#!/bin/bash
#SBATCH --job-name=llama70b
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

srun torchrun \
  --nnodes=32 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m torchtitan.train \
  --job.config_file ./llama3_70b.toml
```

**步骤 3：提交作业**

```bash
sbatch multinode_trainer.slurm
```

**步骤 4：从检查点恢复**

若配置的文件夹中存在检查点，训练将自动恢复。

### 工作流 3：为 H100 启用 Float8 训练

Float8 在 H100 GPU 上可提供 30-50% 的加速。

```
Float8 训练：
- [ ] 步骤 1：安装 torchao
- [ ] 步骤 2：配置 Float8
- [ ] 步骤 3：启动并开启 compile
```

**步骤 1：安装 torchao**

```bash
USE_CPP=0 pip install git+https://github.com/pytorch/ao.git
```

**步骤 2：配置 Float8**

在 TOML 配置中添加：
```toml
[model]
converters = ["quantize.linear.float8"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true
precompute_float8_dynamic_scale_for_fsdp = true
filter_fqns = ["output"]  # Exclude output layer

[compile]
enable = true
components = ["model", "loss"]
```

**步骤 3：启动并开启 compile**

```bash
CONFIG_FILE="./llama3_8b.toml" ./run_train.sh \
  --model.converters="quantize.linear.float8" \
  --quantize.linear.float8.enable_fsdp_float8_all_gather \
  --compile.enable
```

### 工作流 4：405B 模型的 4D 并行

```
4D 并行（FSDP + TP + PP + CP）：
- [ ] 步骤 1：创建种子检查点
- [ ] 步骤 2：配置 4D 并行
- [ ] 步骤 3：在 512 个 GPU 上启动
```

**步骤 1：创建种子检查点**

跨 PP 阶段一致初始化所必需：
```bash
NGPU=1 CONFIG_FILE=./llama3_405b.toml ./run_train.sh \
  --checkpoint.enable \
  --checkpoint.create_seed_checkpoint \
  --parallelism.data_parallel_shard_degree 1 \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.pipeline_parallel_degree 1
```

**步骤 2：配置 4D 并行**

```toml
[parallelism]
data_parallel_shard_degree = 8   # FSDP
tensor_parallel_degree = 8       # TP within node
pipeline_parallel_degree = 8     # PP across nodes
context_parallel_degree = 1      # CP for long sequences

[training]
local_batch_size = 32
seq_len = 8192
```

**步骤 3：在 512 个 GPU 上启动**

```bash
# 64 节点 x 8 GPU = 512 GPU
srun torchrun --nnodes=64 --nproc_per_node=8 \
  -m torchtitan.train \
  --job.config_file ./llama3_405b.toml
```

## 何时使用 vs 替代方案

**使用 TorchTitan 的场景：**
- 从头预训练 LLM（8B 到 405B+）
- 需要无第三方依赖的 PyTorch 原生方案
- 需要可组合的 4D 并行（FSDP2、TP、PP、CP）
- 在支持 Float8 的 H100 上训练
- 需要与 torchtune/HuggingFace 互操作的检查点

**使用替代方案的场景：**
- **Megatron-LM**：仅限 NVIDIA 部署时追求最高性能
- **DeepSpeed**：更广泛的 ZeRO 优化生态，支持推理
- **Axolotl/TRL**：微调而非预训练
- **LitGPT**：教学用途，小规模训练

## 常见问题

**问题：大模型内存不足**

启用激活检查点并减小批次大小：
```toml
[activation_checkpoint]
mode = "full"  # Instead of "selective"

[training]
local_batch_size = 1
```

或使用梯度累积：
```toml
[training]
local_batch_size = 1
global_batch_size = 32  # Accumulates gradients
```

**问题：TP 异步集合通信导致内存占用过高**

设置环境变量：
```bash
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
```

**问题：Float8 训练未见加速**

Float8 仅对大型 GEMM 有效。过滤小层：
```toml
[quantize.linear.float8]
filter_fqns = ["attention.wk", "attention.wv", "output", "auto_filter_small_kn"]
```

**问题：更改并行度后检查点加载失败**

使用 DCP 的重分片功能：
```bash
# 将分片检查点转换为单文件
python -m torch.distributed.checkpoint.format_utils \
  dcp_to_torch checkpoint/step-1000 checkpoint.pt
```

**问题：Pipeline 并行初始化失败**

请先创建种子检查点（参见工作流 4，步骤 1）。

## 支持的模型

| 模型 | 规模 | 状态 |
|-------|-------|--------|
| Llama 3.1 | 8B, 70B, 405B | 生产可用 |
| Llama 4 | 多种 | 实验性 |
| DeepSeek V3 | 16B, 236B, 671B (MoE) | 实验性 |
| GPT-OSS | 20B, 120B (MoE) | 实验性 |
| Qwen 3 | 多种 | 实验性 |
| Flux | 扩散模型 | 实验性 |

## 性能基准（H100）

| 模型 | GPU 数 | 并行策略 | TPS/GPU | 技术 |
|-------|------|-------------|---------|------------|
| Llama 8B | 8 | FSDP | 5,762 | 基线 |
| Llama 8B | 8 | FSDP+compile+FP8 | 8,532 | +48% |
| Llama 70B | 256 | FSDP+TP+AsyncTP | 876 | 2D 并行 |
| Llama 405B | 512 | FSDP+TP+PP | 128 | 3D 并行 |

## 进阶主题

**FSDP2 配置**：参见 [references/fsdp.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/torchtitan/references/fsdp.md)，了解 FSDP2 与 FSDP1 的详细对比及 ZeRO 等价关系。

**Float8 训练**：参见 [references/float8.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/torchtitan/references/float8.md)，了解 tensorwise 与 rowwise 缩放方案。

**检查点**：参见 [references/checkpoint.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/torchtitan/references/checkpoint.md)，了解 HuggingFace 转换与异步检查点。

**添加自定义模型**：参见 [references/custom-models.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/torchtitan/references/custom-models.md)，了解 TrainSpec 协议。

## 资源

- GitHub：https://github.com/pytorch/torchtitan
- 论文：https://arxiv.org/abs/2410.06511
- ICLR 2025：https://iclr.cc/virtual/2025/poster/29620
- PyTorch 论坛：https://discuss.pytorch.org/c/distributed/torchtitan/44