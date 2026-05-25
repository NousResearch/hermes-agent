---
title: "Evaluating Llms Harness — lm-eval-harness: benchmark LLMs (MMLU, GSM8K, etc"
sidebar_label: "Evaluating Llms Harness"
description: "lm-eval-harness：对 LLM 进行基准测试（MMLU、GSM8K 等）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Evaluating Llms Harness

lm-eval-harness：对 LLM 进行基准测试（MMLU、GSM8K 等）。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/mlops/evaluation/lm-evaluation-harness` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `lm-eval`, `transformers`, `vllm` |
| 平台 | linux, macos |
| 标签 | `Evaluation`, `LM Evaluation Harness`, `Benchmarking`, `MMLU`, `HumanEval`, `GSM8K`, `EleutherAI`, `Model Quality`, `Academic Benchmarks`, `Industry Standard` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# lm-evaluation-harness - LLM 基准测试

## 内容概览

在 60+ 个学术基准（MMLU、HumanEval、GSM8K、TruthfulQA、HellaSwag）上评估 LLM。适用于基准测试模型质量、比较模型、报告学术结果或跟踪训练进度。行业标准工具，被 EleutherAI、HuggingFace 及各大实验室广泛使用。支持 HuggingFace、vLLM 及 API。

## 快速开始

lm-evaluation-harness 使用标准化 prompt（提示词）和指标，在 60+ 个学术基准上评估 LLM。

**安装**：
```bash
pip install lm-eval
```

**评估任意 HuggingFace 模型**：
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu,gsm8k,hellaswag \
  --device cuda:0 \
  --batch_size 8
```

**查看可用任务**：
```bash
lm_eval --tasks list
```

## 常用工作流

### 工作流 1：标准基准评估

在核心基准（MMLU、GSM8K、HumanEval）上评估模型。

复制此检查清单：

```
基准评估：
- [ ] 步骤 1：选择基准套件
- [ ] 步骤 2：配置模型
- [ ] 步骤 3：运行评估
- [ ] 步骤 4：分析结果
```

**步骤 1：选择基准套件**

**核心推理基准**：
- **MMLU**（Massive Multitask Language Understanding）- 57 个科目，多项选择
- **GSM8K** - 小学数学应用题
- **HellaSwag** - 常识推理
- **TruthfulQA** - 真实性与事实性
- **ARC**（AI2 Reasoning Challenge）- 科学题目

**代码基准**：
- **HumanEval** - Python 代码生成（164 道题）
- **MBPP**（Mostly Basic Python Problems）- Python 编程

**标准套件**（推荐用于模型发布）：
```bash
--tasks mmlu,gsm8k,hellaswag,truthfulqa,arc_challenge
```

**步骤 2：配置模型**

**HuggingFace 模型**：
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,dtype=bfloat16 \
  --tasks mmlu \
  --device cuda:0 \
  --batch_size auto  # Auto-detect optimal batch size
```

**量化模型（4-bit/8-bit）**：
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,load_in_4bit=True \
  --tasks mmlu \
  --device cuda:0
```

**自定义 checkpoint**：
```bash
lm_eval --model hf \
  --model_args pretrained=/path/to/my-model,tokenizer=/path/to/tokenizer \
  --tasks mmlu \
  --device cuda:0
```

**步骤 3：运行评估**

```bash
# Full MMLU evaluation (57 subjects)
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu \
  --num_fewshot 5 \  # 5-shot evaluation (standard)
  --batch_size 8 \
  --output_path results/ \
  --log_samples  # Save individual predictions

# Multiple benchmarks at once
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu,gsm8k,hellaswag,truthfulqa,arc_challenge \
  --num_fewshot 5 \
  --batch_size 8 \
  --output_path results/llama2-7b-eval.json
```

**步骤 4：分析结果**

结果保存至 `results/llama2-7b-eval.json`：

```json
{
  "results": {
    "mmlu": {
      "acc": 0.459,
      "acc_stderr": 0.004
    },
    "gsm8k": {
      "exact_match": 0.142,
      "exact_match_stderr": 0.006
    },
    "hellaswag": {
      "acc_norm": 0.765,
      "acc_norm_stderr": 0.004
    }
  },
  "config": {
    "model": "hf",
    "model_args": "pretrained=meta-llama/Llama-2-7b-hf",
    "num_fewshot": 5
  }
}
```

### 工作流 2：跟踪训练进度

在训练过程中评估 checkpoint。

```
训练进度跟踪：
- [ ] 步骤 1：设置定期评估
- [ ] 步骤 2：选择快速基准
- [ ] 步骤 3：自动化评估
- [ ] 步骤 4：绘制学习曲线
```

**步骤 1：设置定期评估**

每 N 个训练步骤评估一次：

```bash
#!/bin/bash
# eval_checkpoint.sh

CHECKPOINT_DIR=$1
STEP=$2

lm_eval --model hf \
  --model_args pretrained=$CHECKPOINT_DIR/checkpoint-$STEP \
  --tasks gsm8k,hellaswag \
  --num_fewshot 0 \  # 0-shot for speed
  --batch_size 16 \
  --output_path results/step-$STEP.json
```

**步骤 2：选择快速基准**

适合频繁评估的快速基准：
- **HellaSwag**：单 GPU 约 10 分钟
- **GSM8K**：约 5 分钟
- **PIQA**：约 2 分钟

不适合频繁评估（耗时过长）：
- **MMLU**：约 2 小时（57 个科目）
- **HumanEval**：需要执行代码

**步骤 3：自动化评估**

集成到训练脚本中：

```python
# In training loop
if step % eval_interval == 0:
    model.save_pretrained(f"checkpoints/step-{step}")

    # Run evaluation
    os.system(f"./eval_checkpoint.sh checkpoints step-{step}")
```

或使用 PyTorch Lightning callback：

```python
from pytorch_lightning import Callback

class EvalHarnessCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        step = trainer.global_step
        checkpoint_path = f"checkpoints/step-{step}"

        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path)

        # Run lm-eval
        os.system(f"lm_eval --model hf --model_args pretrained={checkpoint_path} ...")
```

**步骤 4：绘制学习曲线**

```python
import json
import matplotlib.pyplot as plt

# Load all results
steps = []
mmlu_scores = []

for file in sorted(glob.glob("results/step-*.json")):
    with open(file) as f:
        data = json.load(f)
        step = int(file.split("-")[1].split(".")[0])
        steps.append(step)
        mmlu_scores.append(data["results"]["mmlu"]["acc"])

# Plot
plt.plot(steps, mmlu_scores)
plt.xlabel("Training Step")
plt.ylabel("MMLU Accuracy")
plt.title("Training Progress")
plt.savefig("training_curve.png")
```

### 工作流 3：比较多个模型

用于模型比较的基准套件。

```
模型比较：
- [ ] 步骤 1：定义模型列表
- [ ] 步骤 2：运行评估
- [ ] 步骤 3：生成对比表格
```

**步骤 1：定义模型列表**

```bash
# models.txt
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-13b-hf
mistralai/Mistral-7B-v0.1
microsoft/phi-2
```

**步骤 2：运行评估**

```bash
#!/bin/bash
# eval_all_models.sh

TASKS="mmlu,gsm8k,hellaswag,truthfulqa"

while read model; do
    echo "Evaluating $model"

    # Extract model name for output file
    model_name=$(echo $model | sed 's/\//-/g')

    lm_eval --model hf \
      --model_args pretrained=$model,dtype=bfloat16 \
      --tasks $TASKS \
      --num_fewshot 5 \
      --batch_size auto \
      --output_path results/$model_name.json

done < models.txt
```

**步骤 3：生成对比表格**

```python
import json
import pandas as pd

models = [
    "meta-llama-Llama-2-7b-hf",
    "meta-llama-Llama-2-13b-hf",
    "mistralai-Mistral-7B-v0.1",
    "microsoft-phi-2"
]

tasks = ["mmlu", "gsm8k", "hellaswag", "truthfulqa"]

results = []
for model in models:
    with open(f"results/{model}.json") as f:
        data = json.load(f)
        row = {"Model": model.replace("-", "/")}
        for task in tasks:
            # Get primary metric for each task
            metrics = data["results"][task]
            if "acc" in metrics:
                row[task.upper()] = f"{metrics['acc']:.3f}"
            elif "exact_match" in metrics:
                row[task.upper()] = f"{metrics['exact_match']:.3f}"
        results.append(row)

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

输出：
```
| Model                  | MMLU  | GSM8K | HELLASWAG | TRUTHFULQA |
|------------------------|-------|-------|-----------|------------|
| meta-llama/Llama-2-7b  | 0.459 | 0.142 | 0.765     | 0.391      |
| meta-llama/Llama-2-13b | 0.549 | 0.287 | 0.801     | 0.430      |
| mistralai/Mistral-7B   | 0.626 | 0.395 | 0.812     | 0.428      |
| microsoft/phi-2        | 0.560 | 0.613 | 0.682     | 0.447      |
```

### 工作流 4：使用 vLLM 评估（更快的推理）

使用 vLLM 后端可获得 5-10 倍的评估速度提升。

```
vLLM 评估：
- [ ] 步骤 1：安装 vLLM
- [ ] 步骤 2：配置 vLLM 后端
- [ ] 步骤 3：运行评估
```

**步骤 1：安装 vLLM**

```bash
pip install vllm
```

**步骤 2：配置 vLLM 后端**

```bash
lm_eval --model vllm \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8 \
  --tasks mmlu \
  --batch_size auto
```

**步骤 3：运行评估**

vLLM 比标准 HuggingFace 快 5-10 倍：

```bash
# Standard HF: ~2 hours for MMLU on 7B model
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu \
  --batch_size 8

# vLLM: ~15-20 minutes for MMLU on 7B model
lm_eval --model vllm \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,tensor_parallel_size=2 \
  --tasks mmlu \
  --batch_size auto
```

## 何时使用及替代方案

**在以下情况使用 lm-evaluation-harness：**
- 为学术论文进行模型基准测试
- 在标准任务上比较模型质量
- 跟踪训练进度
- 报告标准化指标（所有人使用相同 prompt）
- 需要可复现的评估结果

**改用以下替代方案：**
- **HELM**（Stanford）：更广泛的评估（公平性、效率、校准）
- **AlpacaEval**：使用 LLM 作为评判的指令跟随评估
- **MT-Bench**：多轮对话评估
- **自定义脚本**：特定领域评估

## 常见问题

**问题：评估速度过慢**

使用 vLLM 后端：
```bash
lm_eval --model vllm \
  --model_args pretrained=model-name,tensor_parallel_size=2
```

或减少 few-shot 示例数：
```bash
--num_fewshot 0  # Instead of 5
```

或评估 MMLU 子集：
```bash
--tasks mmlu_stem  # Only STEM subjects
```

**问题：显存不足**

减小 batch size：
```bash
--batch_size 1  # Or --batch_size auto
```

使用量化：
```bash
--model_args pretrained=model-name,load_in_8bit=True
```

启用 CPU offloading：
```bash
--model_args pretrained=model-name,device_map=auto,offload_folder=offload
```

**问题：结果与已报告数值不一致**

检查 few-shot 数量：
```bash
--num_fewshot 5  # Most papers use 5-shot
```

检查确切任务名称：
```bash
--tasks mmlu  # Not mmlu_direct or mmlu_fewshot
```

验证模型与 tokenizer 匹配：
```bash
--model_args pretrained=model-name,tokenizer=same-model-name
```

**问题：HumanEval 未执行代码**

安装执行依赖：
```bash
pip install human-eval
```

启用代码执行：
```bash
lm_eval --model hf \
  --model_args pretrained=model-name \
  --tasks humaneval \
  --allow_code_execution  # Required for HumanEval
```

## 进阶主题

**基准描述**：参见 [references/benchmark-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/lm-evaluation-harness/references/benchmark-guide.md)，了解所有 60+ 个任务的详细说明、测量内容及结果解读。

**自定义任务**：参见 [references/custom-tasks.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/lm-evaluation-harness/references/custom-tasks.md)，了解如何创建特定领域的评估任务。

**API 评估**：参见 [references/api-evaluation.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/lm-evaluation-harness/references/api-evaluation.md)，了解如何评估 OpenAI、Anthropic 及其他 API 模型。

**多 GPU 策略**：参见 [references/distributed-eval.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/lm-evaluation-harness/references/distributed-eval.md)，了解数据并行与张量并行评估方案。

## 硬件要求

- **GPU**：NVIDIA（CUDA 11.8+），支持 CPU 运行（速度极慢）
- **显存**：
  - 7B 模型：16GB（bf16）或 8GB（8-bit）
  - 13B 模型：28GB（bf16）或 14GB（8-bit）
  - 70B 模型：需要多 GPU 或量化
- **耗时**（7B 模型，单张 A100）：
  - HellaSwag：10 分钟
  - GSM8K：5 分钟
  - MMLU（完整）：2 小时
  - HumanEval：20 分钟

## 资源

- GitHub：https://github.com/EleutherAI/lm-evaluation-harness
- 文档：https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs
- 任务库：60+ 个任务，包括 MMLU、GSM8K、HumanEval、TruthfulQA、HellaSwag、ARC、WinoGrande 等
- 排行榜：https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard（使用本工具）