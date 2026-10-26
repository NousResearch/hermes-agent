---
title: "Saelens — 모델 특성을 해석하기 위한 희소 오토인코더 학습"
sidebar_label: "Saelens"
description: "모델 특성을 해석하기 위한 희소 오토인코더 학습"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 수정하세요. */}

# Saelens

모델 특성을 해석하기 위한 희소 오토인코더를 학습합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/saelens`로 설치 |
| 경로 | `optional-skills/mlops/saelens` |
| 버전 | `1.0.1` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `sae-lens>=6.0.0`, `transformer-lens>=2.0.0`, `torch>=2.0.0` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Sparse Autoencoders`, `SAE`, `Mechanistic Interpretability`, `Feature Discovery`, `Superposition` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# SAELens: 기계론적 해석 가능성을 위한 희소 오토인코더

SAELens는 희소 오토인코더(SAE)를 학습하고 분석하기 위한 기본 라이브러리입니다. 희소 오토인코더는 다의미적 신경망 활성화를 희소하고 해석 가능한 특성으로 분해하는 기법이며, Anthropic의 획기적인 단일 의미성 연구를 기반으로 합니다.

**GitHub**: [jbloomAus/SAELens](https://github.com/jbloomAus/SAELens) (별 1,100개 이상)

## 문제: 다의미성과 중첩

신경망의 개별 뉴런은 **다의미적**입니다. 즉, 의미적으로 서로 다른 여러 맥락에서 활성화됩니다. 모델이 뉴런 수보다 더 많은 특성을 표현하기 위해 **중첩**을 사용하기 때문에 이런 현상이 발생하며, 이로 인해 해석 가능성이 어려워집니다.

**SAE는 이를 해결합니다.** 밀집 활성화를 희소하고 단일 의미적인 특성으로 분해합니다. 일반적으로 어떤 입력에 대해서도 소수의 특성만 활성화되며, 각 특성은 해석 가능한 개념에 대응합니다.

## SAELens를 사용할 때

**다음이 필요할 때 SAELens를 사용하세요:**
- 모델 활성화에서 해석 가능한 특성 발견
- 모델이 학습한 개념 파악
- 중첩과 특성 기하학 연구
- 특성 기반 조정 또는 절제 수행
- 안전과 관련된 특성(기만, 편향, 유해 콘텐츠) 분석

**다음과 같은 경우에는 대안을 고려하세요:**
- 기본 활성화 분석이 필요함 → **TransformerLens**를 직접 사용
- 인과적 개입 실험을 원함 → **pyvene** 또는 **TransformerLens** 사용
- 프로덕션 조정이 필요함 → 직접 활성화 엔지니어링 고려

## 설치

```bash
pip install sae-lens
```

요구 사항: Python 3.10+, transformer-lens>=2.0.0

## 핵심 개념

### SAE가 학습하는 것

SAE는 희소 병목을 통해 모델 활성화를 재구성하도록 학습합니다:

```
Input Activation → Encoder → Sparse Features → Decoder → Reconstructed Activation
    (d_model)       ↓        (d_sae >> d_model)    ↓         (d_model)
                 sparsity                      reconstruction
                 penalty                          loss
```

**손실 함수**: `MSE(original, reconstructed) + L1_coefficient × L1(features)`

### 핵심 검증(Anthropic 연구)

"Towards Monosemanticity"에서 사람 평가자들은 **SAE 특성의 70%가 진정으로 해석 가능하다**고 판단했습니다. 발견된 특성에는 다음이 포함됩니다:
- DNA 서열, 법률 언어, HTTP 요청
- 히브리어 텍스트, 영양 관련 문장, 코드 구문
- 감정, 명명된 엔터티, 문법 구조

## 워크플로 1: 사전 학습된 SAE 로드 및 분석

### 단계별 안내

```python
from transformer_lens import HookedTransformer
from sae_lens import SAE

# 1. Load model and pre-trained SAE
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
# In sae-lens v6, SAE.from_pretrained() returns JUST the SAE (not a tuple).
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)
# If you also need the cfg dict and feature sparsity, use:
# sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(...)

# 2. Get model activations
tokens = model.to_tokens("The capital of France is Paris")
_, cache = model.run_with_cache(tokens)
activations = cache["resid_pre", 8]  # [batch, pos, d_model]

# 3. Encode to SAE features
sae_features = sae.encode(activations)  # [batch, pos, d_sae]
print(f"Active features: {(sae_features > 0).sum()}")

# 4. Find top features for each position
for pos in range(tokens.shape[1]):
    top_features = sae_features[0, pos].topk(5)
    token = model.to_str_tokens(tokens[0, pos:pos+1])[0]
    print(f"Token '{token}': features {top_features.indices.tolist()}")

# 5. Reconstruct activations
reconstructed = sae.decode(sae_features)
reconstruction_error = (activations - reconstructed).norm()
```

### 사용 가능한 사전 학습 SAE

| Release | Model | Layers |
|---------|-------|--------|
| `gpt2-small-res-jb` | GPT-2 Small | 여러 잔차 스트림 |
| `gemma-2b-res` | Gemma 2B | 잔차 스트림 |
| Various on HuggingFace | `saelens` 태그 검색 | 다양함 |

### 체크리스트
- [ ] TransformerLens로 모델 로드
- [ ] 대상 레이어에 맞는 SAE 로드
- [ ] 활성화를 희소 특성으로 인코딩
- [ ] 토큰별 최상위 활성화 특성 식별
- [ ] 재구성 품질 검증

## 워크플로 2: 사용자 정의 SAE 학습

### 단계별 안내

```python
from sae_lens import (
    LanguageModelSAETrainingRunner,
    LanguageModelSAERunnerConfig,
    StandardTrainingSAEConfig,
    LoggingConfig,
)

# 1. Configure training (v6 uses a NESTED config: SAE-specific options live in a
#    `sae=` sub-config, and logging options live in a `logger=` sub-config).
#    Note: `architecture`, `d_sae`, `l1_coefficient` etc. are now on the SAE sub-config,
#    and legacy flat options like `hook_layer`, `activation_fn`, `log_to_wandb` were removed.
cfg = LanguageModelSAERunnerConfig(
    # SAE architecture + sparsity (nested)
    sae=StandardTrainingSAEConfig(
        d_in=768,          # Model dimension
        d_sae=768 * 8,     # Expansion factor of 8
        l1_coefficient=8e-5,  # Sparsity penalty
        apply_b_dec_to_input=True,
        normalize_activations="expected_average_only_in",
    ),

    # Data-generating function (model + hook point)
    model_name="gpt2-small",
    hook_name="blocks.8.hook_resid_pre",  # layer is inferred from hook_name (no hook_layer)

    # Training
    lr=4e-4,
    l1_warm_up_steps=1000,
    train_batch_size_tokens=4096,
    training_tokens=100_000_000,

    # Data
    dataset_path="monology/pile-uncopyrighted",
    context_size=128,

    # Logging (nested)
    logger=LoggingConfig(
        log_to_wandb=True,
        wandb_project="sae-training",
    ),

    # Checkpointing
    checkpoint_path="checkpoints",
    n_checkpoints=5,
)

# 2. Train
trainer = LanguageModelSAETrainingRunner(cfg)  # SAETrainingRunner still works as an alias
sae = trainer.run()

# 3. Evaluate
print(f"L0 (avg active features): {trainer.metrics['l0']}")
print(f"CE Loss Recovered: {trainer.metrics['ce_loss_score']}")
```

> **v6 마이그레이션 참고:** 다른 SAE 유형을 사용하려면 `sae=` 하위 설정을 교체하세요 —
> `GatedTrainingSAEConfig`, `TopKTrainingSAEConfig`(`k`를 직접 설정), 또는
> `JumpReLUTrainingSAEConfig`(`l0_coefficient` 사용). 레거시 평면 옵션
> (`architecture`, `expansion_factor`, `hook_layer`, `activation_fn`/`activation_fn_kwargs`,
> `use_ghost_grads`, ghost grads, b_dec/decoder init options)은 v6에서 제거되었습니다.

### 주요 하이퍼파라미터

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `d_sae` | d_model의 4-16배 | 더 많은 특성, 더 높은 용량 |
| `l1_coefficient` | 5e-5 to 1e-4 | 값이 높을수록 더 희소하지만 정확도는 낮아짐 |
| `lr` | 1e-4 to 1e-3 | 표준 옵티마이저 학습률 |
| `l1_warm_up_steps` | 500-2000 | 초기 특성 소실 방지 |

### 평가 지표

| Metric | Target | Meaning |
|--------|--------|---------|
| **L0** | 50-200 | 토큰당 평균 활성 특성 수 |
| **CE Loss Score** | 80-95% | 원본 대비 복원된 교차 엔트로피 |
| **Dead Features** | &lt;5% | 한 번도 활성화되지 않는 특성 |
| **Explained Variance** | >90% | 재구성 품질 |

### 체크리스트
- [ ] 대상 레이어와 훅 지점 선택
- [ ] 확장 계수 설정(d_sae = 4-16× d_model)
- [ ] 원하는 희소성에 맞게 L1 계수 조정
- [ ] 특성 소실 방지를 위한 L1 워밍업 활성화
- [ ] 학습 중 지표 모니터링(W&B)
- [ ] L0 및 CE 손실 복원 검증
- [ ] 소실 특성 비율 확인

## 워크플로 3: 특성 분석 및 조정

### 개별 특성 분석

```python
from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch

model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae = SAE.from_pretrained(  # v6 returns just the SAE
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)

# Find what activates a specific feature
feature_idx = 1234
test_texts = [
    "The scientist conducted an experiment",
    "I love chocolate cake",
    "The code compiles successfully",
    "Paris is beautiful in spring",
]

for text in test_texts:
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens)
    features = sae.encode(cache["resid_pre", 8])
    activation = features[0, :, feature_idx].max().item()
    print(f"{activation:.3f}: {text}")
```

### 특성 조정

```python
def steer_with_feature(model, sae, prompt, feature_idx, strength=5.0):
    """Add SAE feature direction to residual stream."""
    tokens = model.to_tokens(prompt)

    # Get feature direction from decoder
    feature_direction = sae.W_dec[feature_idx]  # [d_model]

    def steering_hook(activation, hook):
        # Add scaled feature direction at all positions
        activation += strength * feature_direction
        return activation

    # Generate with steering
    output = model.generate(
        tokens,
        max_new_tokens=50,
        fwd_hooks=[("blocks.8.hook_resid_pre", steering_hook)]
    )
    return model.to_string(output[0])
```

### 특성 귀속

```python
# Which features most affect a specific output?
tokens = model.to_tokens("The capital of France is")
_, cache = model.run_with_cache(tokens)

# Get features at final position
features = sae.encode(cache["resid_pre", 8])[0, -1]  # [d_sae]

# Get logit attribution per feature
# Feature contribution = feature_activation × decoder_weight × unembedding
W_dec = sae.W_dec  # [d_sae, d_model]
W_U = model.W_U    # [d_model, vocab]

# Contribution to "Paris" logit
paris_token = model.to_single_token(" Paris")
feature_contributions = features * (W_dec @ W_U[:, paris_token])

top_features = feature_contributions.topk(10)
print("Top features for 'Paris' prediction:")
for idx, val in zip(top_features.indices, top_features.values):
    print(f"  Feature {idx.item()}: {val.item():.3f}")
```

## 일반적인 문제 및 해결 방법

> 아래의 모든 예시는 v6 중첩 설정을 사용합니다. SAE별 옵션은 `sae=` 하위 설정에 넣고
> (`StandardTrainingSAEConfig` / `TopKTrainingSAEConfig` / 기타), 학습 관련 설정은 최상위 `LanguageModelSAERunnerConfig`에 둡니다.

### 문제: 높은 소실 특성 비율
```python
from sae_lens import LanguageModelSAERunnerConfig, StandardTrainingSAEConfig

# WRONG: no warm-up, features die early
cfg = LanguageModelSAERunnerConfig(
    sae=StandardTrainingSAEConfig(d_in=768, d_sae=768*8, l1_coefficient=1e-4),
    l1_warm_up_steps=0,  # Bad!
)

# RIGHT: warm up the L1 penalty (v6 removed ghost grads; warm-up is the lever now)
cfg = LanguageModelSAERunnerConfig(
    sae=StandardTrainingSAEConfig(d_in=768, d_sae=768*8, l1_coefficient=8e-5),
    l1_warm_up_steps=1000,  # Gradually increase
)
```

### 문제: 낮은 CE 복원으로 인한 부실한 재구성
```python
# Reduce sparsity penalty and/or add capacity (both on the SAE sub-config)
cfg = LanguageModelSAERunnerConfig(
    sae=StandardTrainingSAEConfig(
        d_in=768,
        d_sae=768 * 16,       # More capacity
        l1_coefficient=5e-5,  # Lower = better reconstruction
    ),
)
```

### 문제: 특성을 해석할 수 없음
```python
from sae_lens import LanguageModelSAERunnerConfig, StandardTrainingSAEConfig, TopKTrainingSAEConfig

# Increase sparsity (higher L1)
cfg = LanguageModelSAERunnerConfig(
    sae=StandardTrainingSAEConfig(d_in=768, d_sae=768*8, l1_coefficient=1e-4),
)
# Or use a TopK SAE (k is set directly in v6, not via activation_fn_kwargs)
cfg = LanguageModelSAERunnerConfig(
    sae=TopKTrainingSAEConfig(d_in=768, d_sae=768*8, k=50),  # Exactly 50 active features
)
```

### 문제: 학습 중 메모리 오류
```python
cfg = LanguageModelSAERunnerConfig(
    sae=StandardTrainingSAEConfig(d_in=768, d_sae=768*8, l1_coefficient=8e-5),
    train_batch_size_tokens=2048,  # Reduce batch size
    store_batch_size_prompts=4,    # Fewer prompts in buffer
    n_batches_in_buffer=8,         # Smaller activation buffer
)
```

## Neuronpedia와의 통합

[neuronpedia.org](https://neuronpedia.org)에서 사전 학습된 SAE 특성을 찾아보세요:

```python
# Features are indexed by SAE ID
# Example: gpt2-small layer 8 feature 1234
# → neuronpedia.org/gpt2-small/8-res-jb/1234
```

## 주요 클래스 참고

| Class | Purpose |
|-------|---------|
| `SAE` | 희소 오토인코더 모델 |
| `LanguageModelSAERunnerConfig` | 최상위 학습 설정(`sae=` 및 `logger=`를 포함) |
| `StandardTrainingSAEConfig` / `TopKTrainingSAEConfig` / `GatedTrainingSAEConfig` / `JumpReLUTrainingSAEConfig` | SAE 유형별 하위 설정(v6) |
| `LoggingConfig` | 로깅/W&B 하위 설정(v6) |
| `LanguageModelSAETrainingRunner` | 학습 루프 관리자(별칭: `SAETrainingRunner`) |
| `ActivationsStore` | 활성화 수집 및 배치 처리 |
| `HookedSAETransformer` | TransformerLens + SAE 통합 |

## 참고 문서

자세한 API 문서, 튜토리얼, 고급 사용법은 `references/` 폴더를 참고하세요:

| File | Contents |
|------|----------|
| [references/README.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/saelens/references/README.md) | 개요 및 빠른 시작 안내 |
| [references/api.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/saelens/references/api.md) | SAE, TrainingSAE, 설정에 대한 전체 API 참고 |
| [references/tutorials.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/saelens/references/tutorials.md) | 학습, 분석, 조정을 위한 단계별 튜토리얼 |

## 외부 리소스

### 튜토리얼
- [기본 로드 및 분석](https://github.com/jbloomAus/SAELens/blob/main/tutorials/basic_loading_and_analysing.ipynb)
- [희소 오토인코더 학습](https://github.com/jbloomAus/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb)
- [ARENA SAE 커리큘럼](https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab)

### 논문
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) - Anthropic (2023)
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) - Anthropic (2024)
- [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600) - Cunningham et al. (ICLR 2024)

### 공식 문서
- [SAELens 문서](https://jbloomaus.github.io/SAELens/)
- [Neuronpedia](https://neuronpedia.org) - 특성 브라우저

## SAE 아키텍처

| Architecture | Description | Use Case |
|--------------|-------------|----------|
| **Standard** | ReLU + L1 패널티 | 범용 |
| **Gated** | 학습된 게이팅 메커니즘 | 더 나은 희소성 제어 |
| **TopK** | 정확히 K개의 활성 특성 | 일관된 희소성 |

```python
from sae_lens import LanguageModelSAERunnerConfig, TopKTrainingSAEConfig

# TopK SAE (exactly 50 features active) — `k` is set on the SAE sub-config in v6
cfg = LanguageModelSAERunnerConfig(
    sae=TopKTrainingSAEConfig(d_in=768, d_sae=768*8, k=50),
)
```
