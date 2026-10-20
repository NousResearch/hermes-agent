---
title: "Simpo 训练 — 用于 LLM 对齐的简单偏好优化"
sidebar_label: "Simpo 训练"
description: "用于 LLM 对齐的简单偏好优化"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Simpo 训练

用于 LLM 对齐的简单偏好优化（Simple Preference Optimization）。无需参考模型的 DPO 替代方案，性能更优（在 AlpacaEval 2.0 上提升 +6.4 分）。无需参考模型，比 DPO 更高效。当需要比 DPO/PPO 更简单、更快速的训练时，可用于偏好对齐。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/simpo` 安装 |
| 路径 | `optional-skills/mlops/simpo` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `torch`, `transformers`, `datasets`, `trl`, `accelerate` |
| 平台 | linux, macos, windows |
| 标签 | `Post-Training`, `SimPO`, `Preference Optimization`, `Alignment`, `DPO Alternative`, `Reference-Free`, `LLM Alignment`, `Efficient Training` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# SimPO - 简单偏好优化

## 快速开始

SimPO 是一种无需参考模型的偏好优化方法，性能优于 DPO。

**安装**：
```bash
# Create environment
conda create -n simpo python=3.10 && conda activate simpo

# Install PyTorch 2.2.2
# Visit: https://pytorch.org/get-started/locally/

# Install alignment-handbook
git clone https://github.com/huggingface/alignment-handbook.git
cd alignment-handbook
python -m pip install .

# Install Flash Attention 2
python -m pip install flash-attn --no-build-isolation
```

**训练**（Mistral 7B）：
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_simpo.py \
  training_configs/mistral-7b-base-simpo.yaml
```

## 常见工作流

### 工作流 1：从基础模型训练（Mistral 7B）

**配置文件**（`mistral-7b-base-simpo.yaml`）：
```yaml
# Model
model_name_or_path: mistralai/Mistral-7B-v0.1
torch_dtype: bfloat16

# Dataset
dataset_mixer:
  HuggingFaceH4/ultrafeedback_binarized: 1.0
dataset_splits:
  - train_prefs
  - test_prefs

# SimPO hyperparameters
beta: 2.0                  # Reward scaling (2.0-10.0)
gamma_beta_ratio: 0.5       # Target margin (0-1)
loss_type: sigmoid          # sigmoid or hinge
sft_weight: 0.0             # Optional SFT regularization

# Training
learning_rate: 5e-7         # Critical: 3e-7 to 1e-6
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 8

# Output
output_dir: ./outputs/mistral-7b-simpo
```

**启动训练**：
```bash
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_simpo.py training_configs/mistral-7b-base-simpo.yaml
```

### 工作流 2：微调指令模型（Llama 3 8B）

**配置文件**（`llama3-8b-instruct-simpo.yaml`）：
```yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct

dataset_mixer:
  argilla/ultrafeedback-binarized-preferences-cleaned: 1.0

beta: 2.5
gamma_beta_ratio: 0.5
learning_rate: 5e-7
sft_weight: 0.1             # Add SFT loss to preserve capabilities

num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
output_dir: ./outputs/llama3-8b-simpo
```

**启动**：
```bash
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_simpo.py training_configs/llama3-8b-instruct-simpo.yaml
```

### 工作流 3：推理密集型任务（较低学习率）

**适用于数学/代码任务**：
```yaml
model_name_or_path: deepseek-ai/deepseek-math-7b-base

dataset_mixer:
  argilla/distilabel-math-preference-dpo: 1.0

beta: 5.0                   # Higher for stronger signal
gamma_beta_ratio: 0.7       # Larger margin
learning_rate: 3e-7         # Lower LR for reasoning
sft_weight: 0.0

num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

## 何时使用及替代方案

**适合使用 SimPO 的场景**：
- 希望比 DPO 训练更简单（无需参考模型）
- 拥有偏好数据（chosen/rejected 对）
- 需要比 DPO 更好的性能
- 计算资源有限
- 单节点训练即可满足需求

**算法选择**：
- **SimPO**：最简单、性能最优、无需参考模型
- **DPO**：需要参考模型基线，更为保守
- **PPO**：最大控制度，需要奖励模型，配置复杂
- **GRPO**：内存高效的 RL，无需 critic

**改用其他方案的场景**：
- **OpenRLHF**：多节点分布式训练，PPO/GRPO
- **TRL**：需要在单一框架中使用多种方法
- **DPO**：需要建立已有基线对比

## 常见问题

**问题：损失发散**

降低学习率：
```yaml
learning_rate: 3e-7  # Reduce from 5e-7
```

降低 beta：
```yaml
beta: 1.0  # Reduce from 2.0
```

**问题：模型遗忘原有能力**

添加 SFT 正则化：
```yaml
sft_weight: 0.1  # Add SFT loss component
```

**问题：偏好分离效果差**

提高 beta 和 margin：
```yaml
beta: 5.0            # Increase from 2.0
gamma_beta_ratio: 0.8  # Increase from 0.5
```

**问题：训练时显存不足（OOM）**

减小批次大小：
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # Maintain effective batch
```

启用梯度检查点：
```yaml
gradient_checkpointing: true
```

## 进阶主题

**损失函数**：参见 [references/loss-functions.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/simpo/references/loss-functions.md)，了解 sigmoid 与 hinge 损失、数学公式及各自适用场景。

**超参数调优**：参见 [references/hyperparameters.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/simpo/references/hyperparameters.md)，了解 beta、gamma、学习率选择指南及针对不同模型规模的建议。

**数据集准备**：参见 [references/datasets.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/simpo/references/datasets.md)，了解偏好数据格式、质量过滤及自定义数据集创建方法。

## 硬件要求

- **GPU**：推荐 NVIDIA A100/H100
- **显存**：
  - 7B 模型：1× A100 40GB（DeepSpeed ZeRO-3）
  - 8B 模型：2× A100 40GB
  - 70B 模型：8× A100 80GB
- **单节点**：DeepSpeed ZeRO-3 即可满足
- **混合精度**：推荐 BF16

**内存优化**：
- DeepSpeed ZeRO-3（默认配置）
- 梯度检查点
- Flash Attention 2

## 资源

- 论文：https://arxiv.org/abs/2405.14734（NeurIPS 2024）
- GitHub：https://github.com/princeton-nlp/SimPO
- 模型：https://huggingface.co/princeton-nlp
- Alignment Handbook：https://github.com/huggingface/alignment-handbook