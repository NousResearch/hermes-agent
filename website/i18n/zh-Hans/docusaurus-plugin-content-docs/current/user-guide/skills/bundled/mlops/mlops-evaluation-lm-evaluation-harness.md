---
title: "LLM 评估基准 — lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）"
sidebar_label: "LLM 评估基准"
description: "lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）"
---
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
# LLM 评估基准
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
lm-eval-harness: benchmark LLMs (MMLU, GSM8K, etc.).
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
## 技能元数据
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
| | |
|---|---|
| | |
| 路径 | `skills/mlops/evaluation/lm-evaluation-harness` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `lm-eval`, `transformers`, `vllm` |
| 标签 | `评估、LM Evaluation Harness、基准测试、MMLU、HumanEval、GSM8K、EleutherAI、模型质量、学术基准、行业标准` |
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
## 参考：完整 SKILL.md
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. 这是代理在技能激活时看到的指令。
:::
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
# LLM 评估基准
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
## What's inside
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Evaluates LLMs across 60+ academic benchmarks (MMLU, HumanEval, GSM8K, TruthfulQA, HellaSwag). Use when benchmarking model quality, comparing models, reporting academic results, or tracking training progress. Industry standard used by EleutherAI, HuggingFace, and major labs. Supports HuggingFace, vLLM, APIs.
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
## Quick start
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
lm-evaluation-harness evaluates LLMs across 60+ academic benchmarks using standardized prompts and metrics.
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Installation**:
```bash
pip install lm-eval
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Evaluate any HuggingFace model**:
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu,gsm8k,hellaswag \
  --device cuda:0 \
  --batch_size 8
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**View available tasks**:
```bash
lm_eval --tasks list
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
## Common workflows
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
### Workflow 1: Standard benchmark evaluation
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Evaluate model on core benchmarks (MMLU, GSM8K, HumanEval).
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Copy this checklist:
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
```
Benchmark Evaluation:
- [ ] Step 1: Choose benchmark suite
- [ ] Step 2: Configure model
- [ ] Step 3: Run evaluation
- [ ] Step 4: Analyze results
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Step 1: Choose benchmark suite**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Core reasoning benchmarks**:
- **MMLU** (Massive Multitask Language Understanding) - 57 subjects, multiple choice
- **GSM8K** - Grade school math word problems
- **HellaSwag** - Common sense reasoning
- **TruthfulQA** - Truthfulness and factuality
- **ARC** (AI2 Reasoning Challenge) - Science questions
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Code benchmarks**:
- **HumanEval** - Python code generation (164 problems)
- **MBPP** (Mostly Basic Python Problems) - Python coding
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Standard suite** (recommended for model releases):
```bash
--tasks mmlu,gsm8k,hellaswag,truthfulqa,arc_challenge
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Step 2: Configure model**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**HuggingFace model**:
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,dtype=bfloat16 \
  --tasks mmlu \
  --device cuda:0 \
  --batch_size auto  # Auto-detect optimal batch size
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Quantized model (4-bit/8-bit)**:
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,load_in_4bit=True \
  --tasks mmlu \
  --device cuda:0
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Custom checkpoint**:
```bash
lm_eval --model hf \
  --model_args pretrained=/path/to/my-model,tokenizer=/path/to/tokenizer \
  --tasks mmlu \
  --device cuda:0
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Step 3: Run evaluation**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
```bash
# LLM 评估基准
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu \
  --num_fewshot 5 \  # 5-shot evaluation (standard)
  --batch_size 8 \
  --output_path results/ \
  --log_samples  # Save individual predictions
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
# LLM 评估基准
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu,gsm8k,hellaswag,truthfulqa,arc_challenge \
  --num_fewshot 5 \
  --batch_size 8 \
  --output_path results/llama2-7b-eval.json
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Step 4: Analyze results**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Results saved to `results/llama2-7b-eval.json`:
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
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
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
### Workflow 2: Track training progress
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Evaluate checkpoints during training.
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
```
Training Progress Tracking:
- [ ] Step 1: Set up periodic evaluation
- [ ] Step 2: Choose quick benchmarks
- [ ] Step 3: Automate evaluation
- [ ] Step 4: Plot learning curves
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Step 1: Set up periodic evaluation**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Evaluate every N training steps:
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
```bash
#!/bin/bash
# LLM 评估基准
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
CHECKPOINT_DIR=$1
STEP=$2
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
lm_eval --model hf \
  --model_args pretrained=$CHECKPOINT_DIR/checkpoint-$STEP \
  --tasks gsm8k,hellaswag \
  --num_fewshot 0 \  # 0-shot for speed
  --batch_size 16 \
  --output_path results/step-$STEP.json
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Step 2: Choose quick benchmarks**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Fast benchmarks for frequent evaluation:
- **HellaSwag**: ~10 minutes on 1 GPU
- **GSM8K**: ~5 minutes
- **PIQA**: ~2 minutes
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Avoid for frequent eval (too slow):
- **MMLU**: ~2 hours (57 subjects)
- **HumanEval**: Requires code execution
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Step 3: Automate evaluation**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Integrate with training script:
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
```python
# LLM 评估基准
if step % eval_interval == 0:
    model.save_pretrained(f"checkpoints/step-{step}")
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
    # Run evaluation
    os.system(f"./eval_checkpoint.sh checkpoints step-{step}")
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Or use PyTorch Lightning callbacks:
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
```python
from pytorch_lightning import Callback
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
class EvalHarnessCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        step = trainer.global_step
        checkpoint_path = f"checkpoints/step-{step}"
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path)
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
        # Run lm-eval
        os.system(f"lm_eval --model hf --model_args pretrained={checkpoint_path} ...")
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Step 4: Plot learning curves**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
```python
import json
import matplotlib.pyplot as plt
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
# LLM 评估基准
steps = []
mmlu_scores = []
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
for file in sorted(glob.glob("results/step-*.json")):
    with open(file) as f:
        data = json.load(f)
        step = int(file.split("-")[1].split(".")[0])
        steps.append(step)
        mmlu_scores.append(data["results"]["mmlu"]["acc"])
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
# LLM 评估基准
plt.plot(steps, mmlu_scores)
plt.xlabel("Training Step")
plt.ylabel("MMLU Accuracy")
plt.title("Training Progress")
plt.savefig("training_curve.png")
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
### Workflow 3: Compare multiple models
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Benchmark suite for model comparison.
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
```
Model Comparison:
- [ ] Step 1: Define model list
- [ ] Step 2: Run evaluations
- [ ] Step 3: Generate comparison table
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Step 1: Define model list**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
```bash
# LLM 评估基准
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-13b-hf
mistralai/Mistral-7B-v0.1
microsoft/phi-2
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Step 2: Run evaluations**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
```bash
#!/bin/bash
# LLM 评估基准
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
TASKS="mmlu,gsm8k,hellaswag,truthfulqa"
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
while read model; do
    echo "Evaluating $model"
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
    # Extract model name for output file
    model_name=$(echo $model | sed 's/\//-/g')
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
    lm_eval --model hf \
      --model_args pretrained=$model,dtype=bfloat16 \
      --tasks $TASKS \
      --num_fewshot 5 \
      --batch_size auto \
      --output_path results/$model_name.json
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
done < models.txt
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Step 3: Generate comparison table**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
```python
import json
import pandas as pd
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
models = [
    "meta-llama-Llama-2-7b-hf",
    "meta-llama-Llama-2-13b-hf",
    "mistralai-Mistral-7B-v0.1",
    "microsoft-phi-2"
]
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
tasks = ["mmlu", "gsm8k", "hellaswag", "truthfulqa"]
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
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
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Output:
```
| Model                  | MMLU  | GSM8K | HELLASWAG | TRUTHFULQA |
|------------------------|-------|-------|-----------|------------|
| meta-llama/Llama-2-7b  | 0.459 | 0.142 | 0.765     | 0.391      |
| meta-llama/Llama-2-13b | 0.549 | 0.287 | 0.801     | 0.430      |
| mistralai/Mistral-7B   | 0.626 | 0.395 | 0.812     | 0.428      |
| microsoft/phi-2        | 0.560 | 0.613 | 0.682     | 0.447      |
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
### Workflow 4: Evaluate with vLLM (faster inference)
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Use vLLM backend for 5-10x faster evaluation.
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
```
vLLM Evaluation:
- [ ] Step 1: Install vLLM
- [ ] Step 2: Configure vLLM backend
- [ ] Step 3: Run evaluation
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Step 1: Install vLLM**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
```bash
pip install vllm
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Step 2: Configure vLLM backend**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
```bash
lm_eval --model vllm \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8 \
  --tasks mmlu \
  --batch_size auto
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Step 3: Run evaluation**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
vLLM is 5-10× faster than standard HuggingFace:
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
```bash
# LLM 评估基准
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu \
  --batch_size 8
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
# LLM 评估基准
lm_eval --model vllm \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,tensor_parallel_size=2 \
  --tasks mmlu \
  --batch_size auto
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
## When to use vs alternatives
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Use lm-evaluation-harness when:**
- Benchmarking models for academic papers
- Comparing model quality across standard tasks
- Tracking training progress
- Reporting standardized metrics (everyone uses same prompts)
- Need reproducible evaluation
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Use alternatives instead:**
- **HELM** (Stanford): Broader evaluation (fairness, efficiency, calibration)
- **AlpacaEval**: Instruction-following evaluation with LLM judges
- **MT-Bench**: Conversational multi-turn evaluation
- **Custom scripts**: Domain-specific evaluation
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
## Common issues
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Issue: Evaluation too slow**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Use vLLM backend:
```bash
lm_eval --model vllm \
  --model_args pretrained=model-name,tensor_parallel_size=2
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Or reduce fewshot examples:
```bash
--num_fewshot 0  # Instead of 5
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Or evaluate subset of MMLU:
```bash
--tasks mmlu_stem  # Only STEM subjects
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Issue: Out of memory**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Reduce batch size:
```bash
--batch_size 1  # Or --batch_size auto
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Use quantization:
```bash
--model_args pretrained=model-name,load_in_8bit=True
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Enable CPU offloading:
```bash
--model_args pretrained=model-name,device_map=auto,offload_folder=offload
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Issue: Different results than reported**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Check fewshot count:
```bash
--num_fewshot 5  # Most papers use 5-shot
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Check exact task name:
```bash
--tasks mmlu  # Not mmlu_direct or mmlu_fewshot
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Verify model and tokenizer match:
```bash
--model_args pretrained=model-name,tokenizer=same-model-name
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Issue: HumanEval not executing code**
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Install execution dependencies:
```bash
pip install human-eval
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
Enable code execution:
```bash
lm_eval --model hf \
  --model_args pretrained=model-name \
  --tasks humaneval \
  --allow_code_execution  # Required for HumanEval
```
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
## Advanced topics
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Benchmark descriptions**: See [references/benchmark-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/lm-evaluation-harness/references/benchmark-guide.md) for detailed description of all 60+ tasks, what they measure, and interpretation.
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Custom tasks**: See [references/custom-tasks.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/lm-evaluation-harness/references/custom-tasks.md) for creating domain-specific evaluation tasks.
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**API evaluation**: See [references/api-evaluation.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/lm-evaluation-harness/references/api-evaluation.md) for evaluating OpenAI, Anthropic, and other API models.
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
**Multi-GPU strategies**: See [references/distributed-eval.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/lm-evaluation-harness/references/distributed-eval.md) for data parallel and tensor parallel evaluation.
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
## Hardware requirements
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
- **GPU**: NVIDIA (CUDA 11.8+), works on CPU (very slow)
- **VRAM**:
  - 7B model: 16GB (bf16) or 8GB (8-bit)
  - 13B model: 28GB (bf16) or 14GB (8-bit)
  - 70B model: Requires multi-GPU or quantization
- **Time** (7B model, single A100):
  - HellaSwag: 10 minutes
  - GSM8K: 5 minutes
  - MMLU (full): 2 hours
  - HumanEval: 20 minutes
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
## Resources
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。
- GitHub: https://github.com/EleutherAI/lm-evaluation-harness
- Docs: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs
- Task library: 60+ tasks including MMLU, GSM8K, HumanEval, TruthfulQA, HellaSwag, ARC, WinoGrande, etc.
- Leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard (uses this harness)
lm-eval-harness：基准测试 LLM（MMLU、GSM8K 等）。