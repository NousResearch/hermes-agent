---
title: "使用 TRL 进行微调 — TRL：面向 LLM RLHF 的 SFT、DPO、PPO、GRPO 及奖励建模"
sidebar_label: "使用 TRL 进行微调"
description: "TRL：面向 LLM RLHF 的 SFT、DPO、PPO、GRPO 及奖励建模"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 使用 TRL 进行微调

TRL：面向 LLM RLHF 的 SFT、DPO、PPO、GRPO 及奖励建模。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/trl-fine-tuning` 安装 |
| 路径 | `optional-skills/mlops/training/trl-fine-tuning` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `trl`, `transformers`, `datasets`, `peft`, `accelerate`, `torch` |
| 平台 | linux, macos, windows |
| 标签 | `Post-Training`, `TRL`, `Reinforcement Learning`, `Fine-Tuning`, `SFT`, `DPO`, `PPO`, `GRPO`, `RLHF`, `Preference Alignment`, `HuggingFace` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# TRL - Transformer Reinforcement Learning

## 快速开始

TRL 提供用于将语言模型与人类偏好对齐的后训练（post-training）方法。

**安装**：
```bash
pip install trl transformers datasets peft accelerate
```

**监督微调（SFT）**（指令微调）：
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,  # Prompt-completion pairs
)
trainer.train()
```

**DPO**（偏好对齐）：
```python
from trl import DPOTrainer, DPOConfig

config = DPOConfig(output_dir="model-dpo", beta=0.1)
trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=preference_dataset,  # chosen/rejected pairs
    processing_class=tokenizer
)
trainer.train()
```

## 常见工作流

### 工作流 1：完整 RLHF 流水线（SFT → 奖励模型 → PPO）

从基础模型到人类对齐模型的完整流水线。

复制此检查清单：

```
RLHF Training:
- [ ] Step 1: Supervised fine-tuning (SFT)
- [ ] Step 2: Train reward model
- [ ] Step 3: PPO reinforcement learning
- [ ] Step 4: Evaluate aligned model
```

**第 1 步：监督微调**

在指令跟随数据上训练基础模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Load instruction dataset
dataset = load_dataset("trl-lib/Capybara", split="train")

# Configure training
training_args = SFTConfig(
    output_dir="Qwen2.5-0.5B-SFT",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch"
)

# Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)
trainer.train()
trainer.save_model()
```

**第 2 步：训练奖励模型**

训练模型以预测人类偏好：

```python
from transformers import AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig

# Load SFT model as base
model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen2.5-0.5B-SFT",
    num_labels=1  # Single reward score
)
tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-0.5B-SFT")

# Load preference data (chosen/rejected pairs)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# Configure training
training_args = RewardConfig(
    output_dir="Qwen2.5-0.5B-Reward",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=1e-5
)

# Train reward model
trainer = RewardTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset
)
trainer.train()
trainer.save_model()
```

**第 3 步：PPO 强化学习**

使用奖励模型优化策略：

```bash
python -m trl.scripts.ppo \
    --model_name_or_path Qwen2.5-0.5B-SFT \
    --reward_model_path Qwen2.5-0.5B-Reward \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --output_dir Qwen2.5-0.5B-PPO \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 64 \
    --total_episodes 10000
```

**第 4 步：评估**

```python
from transformers import pipeline

# Load aligned model
generator = pipeline("text-generation", model="Qwen2.5-0.5B-PPO")

# Test
prompt = "Explain quantum computing to a 10-year-old"
output = generator(prompt, max_length=200)[0]["generated_text"]
print(output)
```

### 工作流 2：使用 DPO 进行简单偏好对齐

无需奖励模型即可对齐模型偏好。

复制此检查清单：

```
DPO Training:
- [ ] Step 1: Prepare preference dataset
- [ ] Step 2: Configure DPO
- [ ] Step 3: Train with DPOTrainer
- [ ] Step 4: Evaluate alignment
```

**第 1 步：准备偏好数据集**

数据集格式：
```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "I don't know."
}
```

加载数据集：
```python
from datasets import load_dataset

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
# Or load your own
# dataset = load_dataset("json", data_files="preferences.json")
```

**第 2 步：配置 DPO**

```python
from trl import DPOConfig

config = DPOConfig(
    output_dir="Qwen2.5-0.5B-DPO",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=5e-7,
    beta=0.1,  # KL penalty strength
    max_prompt_length=512,
    max_length=1024,
    logging_steps=10
)
```

**第 3 步：使用 DPOTrainer 训练**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer
)

trainer.train()
trainer.save_model()
```

**CLI 替代方式**：
```bash
trl dpo \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name argilla/Capybara-Preferences \
    --output_dir Qwen2.5-0.5B-DPO \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-7 \
    --beta 0.1
```

### 工作流 3：使用 GRPO 进行内存高效的在线 RL

以最小内存占用进行强化学习训练。

关于深入的 GRPO 指导——奖励函数设计、关键训练洞察（损失行为、模式崩溃、调参）以及高级多阶段模式——请参阅 **[references/grpo-training.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/references/grpo-training.md)**。生产就绪的训练脚本位于 **[templates/basic_grpo_training.py](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/templates/basic_grpo_training.py)**。

复制此检查清单：

```
GRPO Training:
- [ ] Step 1: Define reward function
- [ ] Step 2: Configure GRPO
- [ ] Step 3: Train with GRPOTrainer
```

**第 1 步：定义奖励函数**

```python
def reward_function(completions, **kwargs):
    """
    Compute rewards for completions.

    Args:
        completions: List of generated texts

    Returns:
        List of reward scores (floats)
    """
    rewards = []
    for completion in completions:
        # Example: reward based on length and unique words
        score = len(completion.split())  # Favor longer responses
        score += len(set(completion.lower().split()))  # Reward unique words
        rewards.append(score)
    return rewards
```

或使用奖励模型：
```python
from transformers import pipeline

reward_model = pipeline("text-classification", model="reward-model-path")

def reward_from_model(completions, prompts, **kwargs):
    # Combine prompt + completion
    full_texts = [p + c for p, c in zip(prompts, completions)]
    # Get reward scores
    results = reward_model(full_texts)
    return [r["score"] for r in results]
```

**第 2 步：配置 GRPO**

```python
from trl import GRPOConfig

config = GRPOConfig(
    output_dir="Qwen2-GRPO",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    num_generations=4,  # Generate 4 completions per prompt
    max_new_tokens=128
)
```

**第 3 步：使用 GRPOTrainer 训练**

```python
from datasets import load_dataset
from trl import GRPOTrainer

# Load prompt-only dataset
dataset = load_dataset("trl-lib/tldr", split="train")

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_function,  # Your reward function
    args=config,
    train_dataset=dataset
)

trainer.train()
```

**CLI**：
```bash
trl grpo \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/tldr \
    --output_dir Qwen2-GRPO \
    --num_generations 4
```

## 何时使用 TRL 及替代方案

**适合使用 TRL 的场景：**
- 需要将模型与人类偏好对齐
- 拥有偏好数据（chosen/rejected 对）
- 希望使用强化学习（PPO、GRPO）
- 需要训练奖励模型
- 执行完整 RLHF 流水线

**方法选择**：
- **SFT**：拥有 prompt-completion 对，需要基础指令跟随
- **DPO**：拥有偏好数据，需要简单对齐（无需奖励模型）
- **PPO**：拥有奖励模型，需要对 RL 进行最大程度的控制
- **GRPO**：内存受限，需要在线 RL
- **奖励模型**：构建 RLHF 流水线，需要对生成内容评分

**改用替代方案的场景：**
- **HuggingFace Trainer**：无需 RL 的基础微调
- **Axolotl**：基于 YAML 的训练配置
- **LitGPT**：教学用途、极简微调
- **Unsloth**：快速 LoRA 训练

## 常见问题

**问题：DPO 训练时显存溢出（OOM）**

减小批次大小和序列长度：
```python
config = DPOConfig(
    per_device_train_batch_size=1,  # Reduce from 4
    max_length=512,  # Reduce from 1024
    gradient_accumulation_steps=8  # Maintain effective batch
)
```

或启用梯度检查点：
```python
model.gradient_checkpointing_enable()
```

**问题：对齐质量差**

调整 beta 参数：
```python
# Higher beta = more conservative (stays closer to reference)
config = DPOConfig(beta=0.5)  # Default 0.1

# Lower beta = more aggressive alignment
config = DPOConfig(beta=0.01)
```

**问题：奖励模型无法学习**

检查损失类型和学习率：
```python
config = RewardConfig(
    learning_rate=1e-5,  # Try different LR
    num_train_epochs=3  # Train longer
)
```

确保偏好数据集有明确的优劣区分：
```python
# Verify dataset
print(dataset[0])
# Should have clear chosen > rejected
```

**问题：PPO 训练不稳定**

调整 KL 系数：
```python
config = PPOConfig(
    kl_coef=0.1,  # Increase from 0.05
    cliprange=0.1  # Reduce from 0.2
)
```

## 高级主题

**SFT 训练指南**：参阅 [references/sft-training.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/references/sft-training.md)，了解数据集格式、chat template、packing 策略及多 GPU 训练。

**DPO 变体**：参阅 [references/dpo-variants.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/references/dpo-variants.md)，了解 IPO、cDPO、RPO 及其他 DPO 损失函数与推荐超参数。

**奖励建模**：参阅 [references/reward-modeling.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/references/reward-modeling.md)，了解结果奖励与过程奖励、Bradley-Terry 损失及奖励模型评估。

**在线 RL 方法**：参阅 [references/online-rl.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/references/online-rl.md)，了解 PPO、GRPO、RLOO 及 OnlineDPO 的详细配置。

**GRPO 深度解析**：参阅 [references/grpo-training.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/references/grpo-training.md)，获取专家级 GRPO 模式——奖励函数设计理念、训练洞察（为何损失上升、模式崩溃检测）、超参数调优、多阶段训练及故障排查。生产就绪模板位于 [templates/basic_grpo_training.py](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/templates/basic_grpo_training.py)。

## 硬件要求

- **GPU**：NVIDIA（需要 CUDA）
- **显存（VRAM）**：取决于模型和方法
  - SFT 7B：16GB（使用 LoRA）
  - DPO 7B：24GB（存储参考模型）
  - PPO 7B：40GB（策略模型 + 奖励模型）
  - GRPO 7B：24GB（内存效率更高）
- **多 GPU**：通过 `accelerate` 支持
- **混合精度**：推荐 BF16（A100/H100）

**内存优化**：
- 所有方法均可使用 LoRA/QLoRA
- 启用梯度检查点
- 使用更小的批次大小配合梯度累积

## 资源

- 文档：https://huggingface.co/docs/trl/
- GitHub：https://github.com/huggingface/trl
- 论文：
  - "Training language models to follow instructions with human feedback"（InstructGPT，2022）
  - "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"（DPO，2023）
  - "Group Relative Policy Optimization"（GRPO，2024）
- 示例：https://github.com/huggingface/trl/tree/main/examples/scripts