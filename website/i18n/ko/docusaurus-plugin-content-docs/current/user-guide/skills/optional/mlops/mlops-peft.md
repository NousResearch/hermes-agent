---
title: "Peft — 제한된 GPU 메모리에서 LoRA로 대규모 LLM 미세 조정"
sidebar_label: "Peft"
description: "제한된 GPU 메모리에서 LoRA로 대규모 LLM 미세 조정"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Peft

제한된 GPU 메모리에서 LoRA로 대규모 LLM을 미세 조정합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/peft`로 설치 |
| 경로 | `optional-skills/mlops/peft` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `peft>=0.13.0`, `transformers>=4.45.0`, `torch>=2.0.0`, `bitsandbytes>=0.43.0` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Fine-Tuning`, `PEFT`, `LoRA`, `QLoRA`, `Parameter-Efficient`, `Adapters`, `Low-Rank`, `Memory Optimization`, `Multi-Adapter` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# PEFT (Parameter-Efficient Fine-Tuning)

LoRA, QLoRA 및 25가지 이상의 어댑터 방법을 사용해 매개변수의 &lt;1%를 학습하여 LLM을 미세 조정합니다.

## PEFT를 사용할 때

**다음과 같은 경우 PEFT/LoRA를 사용하세요:**
- 소비자용 GPU(RTX 4090, A100)에서 7B~70B 모델을 미세 조정할 때
- 매개변수의 &lt;1%를 학습해야 할 때(전체 모델 14GB에 비해 어댑터는 6MB)
- 작업별 어댑터를 여러 개 사용해 빠르게 반복하고 싶을 때
- 하나의 기본 모델에서 여러 미세 조정 변형을 배포할 때

**다음과 같은 경우 QLoRA(PEFT + 양자화)를 사용하세요:**
- 단일 24GB GPU에서 70B 모델을 미세 조정할 때
- 메모리가 주요 제약 조건일 때
- 전체 미세 조정 대비 약 5%의 품질 저하를 수용할 수 있을 때

**다음과 같은 경우 대신 전체 미세 조정을 사용하세요:**
- 1B 매개변수 미만의 소형 모델을 학습할 때
- 최고 품질이 필요하고 연산 예산이 충분할 때
- 상당한 도메인 변화로 모든 가중치를 업데이트해야 할 때

## 빠른 시작

### 설치

```bash
# Basic installation
pip install peft

# With quantization support (recommended)
pip install peft bitsandbytes

# Full stack
pip install peft transformers accelerate bitsandbytes datasets
```

### LoRA 미세 조정(표준)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# Load base model
model_name = "meta-llama/Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # Rank (8-64, higher = more capacity)
    lora_alpha=32,                 # Scaling factor (typically 2*r)
    lora_dropout=0.05,             # Dropout for regularization
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
    bias="none"                    # Don't train biases
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 13,631,488 || all params: 8,043,307,008 || trainable%: 0.17%

# Prepare dataset
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

def tokenize(example):
    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    return tokenizer(text, truncation=True, max_length=512, padding="max_length")

tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

# Training
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

# Save adapter only (6MB vs 16GB)
model.save_pretrained("./lora-llama-adapter")
```

### QLoRA 미세 조정(메모리 효율적)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 (best for LLMs)
    bnb_4bit_compute_dtype="bfloat16",   # Compute in bf16
    bnb_4bit_use_double_quant=True       # Nested quantization
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for training (enables gradient checkpointing)
model = prepare_model_for_kbit_training(model)

# LoRA config for QLoRA
lora_config = LoraConfig(
    r=64,                              # Higher rank for 70B
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
# 70B model now fits on single 24GB GPU!
```

## LoRA 매개변수 선택

### Rank (r) - 용량과 효율의 균형

| 순위 | 학습 가능한 매개변수 | 메모리 | 품질 | 사용 사례 |
|------|-----------------|--------|---------|----------|
| 4 | ~3M | 최소 | 낮음 | 단순한 작업, 프로토타이핑 |
| **8** | ~7M | 낮음 | 좋음 | **권장 시작점** |
| **16** | ~14M | 중간 | 더 좋음 | **일반적인 미세 조정** |
| 32 | ~27M | 높음 | 높음 | 복잡한 작업 |
| 64 | ~54M | 높음 | 최고 | 도메인 적응, 70B 모델 |

### Alpha (lora_alpha) - 스케일링 계수

```python
# Rule of thumb: alpha = 2 * rank
LoraConfig(r=16, lora_alpha=32)  # Standard
LoraConfig(r=16, lora_alpha=16)  # Conservative (lower learning rate effect)
LoraConfig(r=16, lora_alpha=64)  # Aggressive (higher learning rate effect)
```

### 아키텍처별 대상 모듈

```python
# Llama / Mistral / Qwen
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# GPT-2 / GPT-Neo
target_modules = ["c_attn", "c_proj", "c_fc"]

# Falcon
target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

# BLOOM
target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

# Auto-detect all linear layers
target_modules = "all-linear"  # PEFT 0.6.0+
```

## 어댑터 로드 및 병합

### 학습된 어댑터 로드

```python
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM

# Option 1: Load with PeftModel
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = PeftModel.from_pretrained(base_model, "./lora-llama-adapter")

# Option 2: Load directly (recommended)
model = AutoPeftModelForCausalLM.from_pretrained(
    "./lora-llama-adapter",
    device_map="auto"
)
```

### 어댑터를 기본 모델에 병합

```python
# Merge for deployment (no adapter overhead)
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./llama-merged")
tokenizer.save_pretrained("./llama-merged")

# Push to Hub
merged_model.push_to_hub("username/llama-finetuned")
```

### 멀티 어댑터 서빙

```python
from peft import PeftModel

# Load base with first adapter
model = AutoPeftModelForCausalLM.from_pretrained("./adapter-task1")

# Load additional adapters
model.load_adapter("./adapter-task2", adapter_name="task2")
model.load_adapter("./adapter-task3", adapter_name="task3")

# Switch between adapters at runtime
model.set_adapter("task1")  # Use task1 adapter
output1 = model.generate(**inputs)

model.set_adapter("task2")  # Switch to task2
output2 = model.generate(**inputs)

# Disable adapters (use base model)
with model.disable_adapter():
    base_output = model.generate(**inputs)
```

## PEFT 방법 비교

| 방법 | 학습 가능 비율 | 메모리 | 속도 | 적합한 용도 |
|--------|------------|--------|-------|----------|
| **LoRA** | 0.1-1% | 낮음 | 빠름 | 일반적인 미세 조정 |
| **QLoRA** | 0.1-1% | 매우 낮음 | 중간 | 메모리가 제한된 환경 |
| AdaLoRA | 0.1-1% | 낮음 | 중간 | 자동 순위 선택 |
| IA3 | 0.01% | 최소 | 가장 빠름 | 퓨샷 적응 |
| Prefix Tuning | 0.1% | 낮음 | 중간 | 생성 제어 |
| Prompt Tuning | 0.001% | 최소 | 빠름 | 단순한 작업 적응 |
| P-Tuning v2 | 0.1% | 낮음 | 중간 | NLU 작업 |

### IA3(최소 매개변수)

```python
from peft import IA3Config

ia3_config = IA3Config(
    target_modules=["q_proj", "v_proj", "k_proj", "down_proj"],
    feedforward_modules=["down_proj"]
)
model = get_peft_model(model, ia3_config)
# Trains only 0.01% of parameters!
```

### Prefix Tuning

```python
from peft import PrefixTuningConfig

prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,      # Prepended tokens
    prefix_projection=True       # Use MLP projection
)
model = get_peft_model(model, prefix_config)
```

## 통합 패턴

### TRL과 함께(SFTTrainer)

```python
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir="./output", max_seq_length=512),
    train_dataset=dataset,
    peft_config=lora_config,  # Pass LoRA config directly
)
trainer.train()
```

### Axolotl과 함께(YAML 구성)

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
lora_target_linear: true  # Target all linear layers
```

### vLLM과 함께(추론)

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

# Load base model with LoRA support
llm = LLM(model="meta-llama/Llama-3.1-8B", enable_lora=True)

# Serve with adapter
outputs = llm.generate(
    prompts,
    lora_request=LoRARequest("adapter1", 1, "./lora-adapter")
)
```

## 성능 벤치마크

### 메모리 사용량(Llama 3.1 8B)

| 방법 | GPU 메모리 | 학습 가능한 매개변수 |
|--------|-----------|------------------|
| 전체 미세 조정 | 60+ GB | 8B (100%) |
| LoRA r=16 | 18 GB | 14M (0.17%) |
| QLoRA r=16 | 6 GB | 14M (0.17%) |
| IA3 | 16 GB | 800K (0.01%) |

### 학습 속도(A100 80GB)

| 방법 | 토큰/초 | 전체 FT 대비 |
|--------|-----------|------------|
| 전체 FT | 2,500 | 1x |
| LoRA | 3,200 | 1.3x |
| QLoRA | 2,100 | 0.84x |

### 품질(MMLU 벤치마크)

| 모델 | 전체 FT | LoRA | QLoRA |
|-------|---------|------|-------|
| Llama 2-7B | 45.3 | 44.8 | 44.1 |
| Llama 2-13B | 54.8 | 54.2 | 53.5 |

## 일반적인 문제

### 학습 중 CUDA OOM

```python
# Solution 1: Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 2: Reduce batch size + increase accumulation
TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16
)

# Solution 3: Use QLoRA
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
```

### 어댑터가 적용되지 않음

```python
# Verify adapter is active
print(model.active_adapters)  # Should show adapter name

# Check trainable parameters
model.print_trainable_parameters()

# Ensure model in training mode
model.train()
```

### 품질 저하

```python
# Increase rank
LoraConfig(r=32, lora_alpha=64)

# Target more modules
target_modules = "all-linear"

# Use more training data and epochs
TrainingArguments(num_train_epochs=5)

# Lower learning rate
TrainingArguments(learning_rate=1e-4)
```

## 모범 사례

1. **r=8~16으로 시작**하고 품질이 부족하면 값을 높이세요.
2. 시작점으로 **alpha = 2 * rank**를 사용하세요.
3. 최상의 품질/효율을 위해 **어텐션 + MLP 레이어를 대상**으로 지정하세요.
4. 메모리를 절약하려면 **그래디언트 체크포인팅을 활성화**하세요.
5. **어댑터를 자주 저장**하세요(파일이 작아 쉽게 롤백할 수 있습니다).
6. 병합하기 전에 **보류 데이터로 평가**하세요.
7. 소비자용 하드웨어에서 70B 이상 모델에는 **QLoRA를 사용**하세요.

## 참고 자료

- **[고급 사용법](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/peft/references/advanced-usage.md)** - DoRA, LoftQ, 순위 안정화, 사용자 지정 모듈
- **[문제 해결](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/peft/references/troubleshooting.md)** - 일반적인 오류, 디버깅, 최적화

## 리소스

- **GitHub**: https://github.com/huggingface/peft
- **문서**: https://huggingface.co/docs/peft
- **LoRA 논문**: arXiv:2106.09685
- **QLoRA 논문**: arXiv:2305.14314
- **모델**: https://huggingface.co/models?library=peft
