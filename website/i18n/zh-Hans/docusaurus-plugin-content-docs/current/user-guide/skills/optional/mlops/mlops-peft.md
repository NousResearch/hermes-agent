---
title: "Peft Fine Tuning — 使用 LoRA、QLoRA 及 25+ 种方法对 LLM 进行参数高效微调"
sidebar_label: "Peft Fine Tuning"
description: "使用 LoRA、QLoRA 及 25+ 种方法对 LLM 进行参数高效微调"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Peft Fine Tuning

使用 LoRA、QLoRA 及 25+ 种方法对 LLM 进行参数高效微调（Parameter-efficient fine-tuning）。适用场景：在显存有限的情况下微调大型模型（7B–70B）、需要以极低精度损失训练不足 1% 的参数，或用于多适配器（multi-adapter）服务。HuggingFace 官方库，与 transformers 生态深度集成。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/peft` 安装 |
| 路径 | `optional-skills/mlops/peft` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖 | `peft>=0.13.0`, `transformers>=4.45.0`, `torch>=2.0.0`, `bitsandbytes>=0.43.0` |
| 平台 | linux, macos, windows |
| 标签 | `Fine-Tuning`, `PEFT`, `LoRA`, `QLoRA`, `Parameter-Efficient`, `Adapters`, `Low-Rank`, `Memory Optimization`, `Multi-Adapter` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# PEFT（参数高效微调）

通过 LoRA、QLoRA 及 25+ 种适配器方法，仅训练不足 1% 的参数来微调 LLM。

## 何时使用 PEFT

**在以下情况使用 PEFT/LoRA：**
- 在消费级 GPU（RTX 4090、A100）上微调 7B–70B 模型
- 需要训练不足 1% 的参数（6MB 适配器 vs 14GB 完整模型）
- 希望通过多个任务专属适配器快速迭代
- 从单一基础模型部署多个微调变体

**在以下情况使用 QLoRA（PEFT + 量化）：**
- 在单张 24GB GPU 上微调 70B 模型
- 显存是主要瓶颈
- 可接受相比完整微调约 5% 的质量损失

**在以下情况改用完整微调：**
- 训练小型模型（参数量 < 1B）
- 需要最高质量且有充足算力预算
- 显著的领域偏移需要更新全部权重

## 快速开始

### 安装

```bash
# 基础安装
pip install peft

# 含量化支持（推荐）
pip install peft bitsandbytes

# 完整工具栈
pip install peft transformers accelerate bitsandbytes datasets
```

### LoRA 微调（标准方式）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# 加载基础模型
model_name = "meta-llama/Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # 秩（Rank），范围 8-64，越高容量越大
    lora_alpha=32,                 # 缩放因子（通常为 2*r）
    lora_dropout=0.05,             # 正则化 dropout
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 注意力层
    bias="none"                    # 不训练偏置项
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出：trainable params: 13,631,488 || all params: 8,043,307,008 || trainable%: 0.17%

# 准备数据集
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

def tokenize(example):
    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    return tokenizer(text, truncation=True, max_length=512, padding="max_length")

tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

# 训练
training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=lambda data: {"input_ids": torch.stack([f["input_ids"] for f in data]),
                                 "attention_mask": torch.stack([f["attention_mask"] for f in data]),
                                 "labels": torch.stack([f["input_ids"] for f in data])}
)

trainer.train()

# 仅保存适配器（6MB vs 16GB）
model.save_pretrained("./lora-llama-adapter")
```

### QLoRA 微调（显存高效方式）

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4（最适合 LLM）
    bnb_4bit_compute_dtype="bfloat16",   # 以 bf16 计算
    bnb_4bit_use_double_quant=True       # 嵌套量化
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    quantization_config=bnb_config,
    device_map="auto"
)

# 为训练做准备（启用梯度检查点）
model = prepare_model_for_kbit_training(model)

# QLoRA 的 LoRA 配置
lora_config = LoraConfig(
    r=64,                              # 70B 模型使用更高秩
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
# 70B 模型现在可在单张 24GB GPU 上运行！
```

## LoRA 参数选择

### 秩（r）——容量与效率的权衡

| 秩 | 可训练参数量 | 显存 | 质量 | 适用场景 |
|------|-----------------|--------|---------|----------|
| 4 | ~3M | 极低 | 较低 | 简单任务、原型验证 |
| **8** | ~7M | 低 | 良好 | **推荐起始点** |
| **16** | ~14M | 中等 | 更好 | **通用微调** |
| 32 | ~27M | 较高 | 高 | 复杂任务 |
| 64 | ~54M | 高 | 最高 | 领域适配、70B 模型 |

### Alpha（lora_alpha）——缩放因子

```python
# 经验法则：alpha = 2 * rank
LoraConfig(r=16, lora_alpha=32)  # 标准
LoraConfig(r=16, lora_alpha=16)  # 保守（学习率效果较低）
LoraConfig(r=16, lora_alpha=64)  # 激进（学习率效果较高）
```

### 按架构选择目标模块

```python
# Llama / Mistral / Qwen
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# GPT-2 / GPT-Neo
target_modules = ["c_attn", "c_proj", "c_fc"]

# Falcon
target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

# BLOOM
target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

# 自动检测所有线性层
target_modules = "all-linear"  # PEFT 0.6.0+
```

## 加载与合并适配器

### 加载已训练的适配器

```python
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM

# 方式一：使用 PeftModel 加载
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = PeftModel.from_pretrained(base_model, "./lora-llama-adapter")

# 方式二：直接加载（推荐）
model = AutoPeftModelForCausalLM.from_pretrained(
    "./lora-llama-adapter",
    device_map="auto"
)
```

### 将适配器合并到基础模型

```python
# 合并以用于部署（无适配器开销）
merged_model = model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("./llama-merged")
tokenizer.save_pretrained("./llama-merged")

# 推送到 Hub
merged_model.push_to_hub("username/llama-finetuned")
```

### 多适配器服务

```python
from peft import PeftModel

# 加载基础模型及第一个适配器
model = AutoPeftModelForCausalLM.from_pretrained("./adapter-task1")

# 加载额外适配器
model.load_adapter("./adapter-task2", adapter_name="task2")
model.load_adapter("./adapter-task3", adapter_name="task3")

# 运行时切换适配器
model.set_adapter("task1")  # 使用 task1 适配器
output1 = model.generate(**inputs)

model.set_adapter("task2")  # 切换到 task2
output2 = model.generate(**inputs)

# 禁用适配器（使用基础模型）
with model.disable_adapter():
    base_output = model.generate(**inputs)
```

## PEFT 方法对比

| 方法 | 可训练参数占比 | 显存 | 速度 | 最适场景 |
|--------|------------|--------|-------|----------|
| **LoRA** | 0.1–1% | 低 | 快 | 通用微调 |
| **QLoRA** | 0.1–1% | 极低 | 中等 | 显存受限场景 |
| AdaLoRA | 0.1–1% | 低 | 中等 | 自动秩选择 |
| IA3 | 0.01% | 极小 | 最快 | 少样本适配 |
| Prefix Tuning | 0.1% | 低 | 中等 | 生成控制 |
| Prompt Tuning | 0.001% | 极小 | 快 | 简单任务适配 |
| P-Tuning v2 | 0.1% | 低 | 中等 | NLU 任务 |

### IA3（最少参数）

```python
from peft import IA3Config

ia3_config = IA3Config(
    target_modules=["q_proj", "v_proj", "k_proj", "down_proj"],
    feedforward_modules=["down_proj"]
)
model = get_peft_model(model, ia3_config)
# 仅训练 0.01% 的参数！
```

### Prefix Tuning

```python
from peft import PrefixTuningConfig

prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,      # 前置 token 数量
    prefix_projection=True       # 使用 MLP 投影
)
model = get_peft_model(model, prefix_config)
```

## 集成模式

### 与 TRL（SFTTrainer）集成

```python
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir="./output", max_seq_length=512),
    train_dataset=dataset,
    peft_config=lora_config,  # 直接传入 LoRA 配置
)
trainer.train()
```

### 与 Axolotl（YAML 配置）集成

```yaml
# axolotl config.yaml
adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
lora_target_linear: true  # 针对所有线性层
```

### 与 vLLM（推理）集成

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

# 加载支持 LoRA 的基础模型
llm = LLM(model="meta-llama/Llama-3.1-8B", enable_lora=True)

# 使用适配器进行推理
outputs = llm.generate(
    prompts,
    lora_request=LoRARequest("adapter1", 1, "./lora-adapter")
)
```

## 性能基准

### 显存占用（Llama 3.1 8B）

| 方法 | GPU 显存 | 可训练参数量 |
|--------|-----------|------------------|
| 完整微调 | 60+ GB | 8B（100%） |
| LoRA r=16 | 18 GB | 14M（0.17%） |
| QLoRA r=16 | 6 GB | 14M（0.17%） |
| IA3 | 16 GB | 800K（0.01%） |

### 训练速度（A100 80GB）

| 方法 | Tokens/秒 | 相对完整微调 |
|--------|-----------|------------|
| 完整微调 | 2,500 | 1x |
| LoRA | 3,200 | 1.3x |
| QLoRA | 2,100 | 0.84x |

### 质量（MMLU 基准）

| 模型 | 完整微调 | LoRA | QLoRA |
|-------|---------|------|-------|
| Llama 2-7B | 45.3 | 44.8 | 44.1 |
| Llama 2-13B | 54.8 | 54.2 | 53.5 |

## 常见问题

### 训练时 CUDA 显存不足（OOM）

```python
# 方案一：启用梯度检查点
model.gradient_checkpointing_enable()

# 方案二：减小批大小 + 增大梯度累积步数
TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16
)

# 方案三：使用 QLoRA
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
```

### 适配器未生效

```python
# 验证适配器是否激活
print(model.active_adapters)  # 应显示适配器名称

# 检查可训练参数
model.print_trainable_parameters()

# 确保模型处于训练模式
model.train()
```

### 质量下降

```python
# 提高秩
LoraConfig(r=32, lora_alpha=64)

# 针对更多模块
target_modules = "all-linear"

# 使用更多训练数据和更多轮次
TrainingArguments(num_train_epochs=5)

# 降低学习率
TrainingArguments(learning_rate=1e-4)
```

## 最佳实践

1. **从 r=8–16 开始**，质量不足时再提高
2. **以 alpha = 2 * rank 为起始点**
3. **同时针对注意力层和 MLP 层**以获得最佳质量/效率比
4. **启用梯度检查点**以节省显存
5. **频繁保存适配器**（文件小，便于回滚）
6. **合并前在留出数据上评估**
7. **70B+ 模型在消费级硬件上使用 QLoRA**

## 参考资料

- **[高级用法](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/peft/references/advanced-usage.md)** — DoRA、LoftQ、秩稳定化、自定义模块
- **[故障排查](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/peft/references/troubleshooting.md)** — 常见错误、调试、优化

## 资源

- **GitHub**：https://github.com/huggingface/peft
- **文档**：https://huggingface.co/docs/peft
- **LoRA 论文**：arXiv:2106.09685
- **QLoRA 论文**：arXiv:2305.14314
- **模型**：https://huggingface.co/models?library=peft