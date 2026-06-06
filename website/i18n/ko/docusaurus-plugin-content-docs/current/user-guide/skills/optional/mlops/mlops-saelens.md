---
title: "Sparse Autoencoder Training"
sidebar_label: "Sparse Autoencoder Training"
description: "SAELens를 사용하여 신경망 활성화를 해석 가능한 특징으로 분해하는 희소 오토인코더(SAE)를 학습하고 분석하기 위한 지침을 제공합니다"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 희소 오토인코더 학습 (Sparse Autoencoder Training)

SAELens를 사용하여 신경망 활성화를 해석 가능한 특징으로 분해하는 희소 오토인코더(SAE)를 학습하고 분석하기 위한 지침을 제공합니다. 언어 모델에서 해석 가능한 특징을 발견하거나, 중첩(superposition)을 분석하거나, 단일 의미 표현(monosemantic representations)을 연구할 때 사용합니다.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/saelens`로 설치 |
| Path | `optional-skills/mlops/saelens` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `sae-lens>=6.0.0`, `transformer-lens>=2.0.0`, `torch>=2.0.0` |
| Platforms | linux, macos, windows |
| Tags | `Sparse Autoencoders`, `SAE`, `Mechanistic Interpretability`, `Feature Discovery`, `Superposition` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되어 있을 때 에이전트가 지침으로 보는 내용입니다.
:::

# SAELens: 기계적 해석 가능성(Mechanistic Interpretability)을 위한 희소 오토인코더

SAELens는 다의적(polysemantic) 신경망 활성화를 희소하고 해석 가능한 특징으로 분해하는 기술인 희소 오토인코더(SAE)를 학습하고 분석하기 위한 기본 라이브러리입니다. Anthropic의 단일 의미성(monosemanticity)에 대한 획기적인 연구를 기반으로 합니다.

**GitHub**: [jbloomAus/SAELens](https://github.com/jbloomAus/SAELens) (1,100+ stars)

## 문제: 다의성(Polysemanticity) & 중첩(Superposition)

신경망의 개별 뉴런은 **다의적**입니다. 즉, 여러 의미론적으로 구별되는 컨텍스트에서 활성화됩니다. 이는 모델이 뉴런 수보다 더 많은 특징을 표현하기 위해 **중첩(superposition)**을 사용하기 때문에 발생하며, 이는 해석 가능성을 어렵게 만듭니다.

**SAE는 이를 해결합니다** - 밀집된 활성화를 희소하고 단일 의미적인 특징으로 분해함으로써 해결합니다. 일반적으로 주어진 입력에 대해 소수의 특징만 활성화되며, 각 특징은 해석 가능한 개념에 대응합니다.

## SAELens를 사용해야 할 때

**다음과 같은 경우에 SAELens를 사용하세요:**
- 모델 활성화에서 해석 가능한 특징을 발견할 때
- 모델이 학습한 개념을 이해할 때
- 중첩 및 특징 기하학(feature geometry)을 연구할 때
- 특징 기반 스티어링(steering) 또는 제거(ablation)를 수행할 때
- 안전 관련 특징(기만, 편향, 유해 콘텐츠)을 분석할 때

**다음과 같은 경우 대안을 고려하세요:**
- 기본적인 활성화 분석이 필요할 때 → **TransformerLens**를 직접 사용하세요
- 인과적 개입 실험을 원할 때 → **pyvene** 또는 **TransformerLens**를 사용하세요
- 프로덕션 환경의 스티어링이 필요할 때 → 직접적인 활성화 엔지니어링을 고려하세요

## 설치

```bash
pip install sae-lens
```

요구 사항: Python 3.10+, transformer-lens>=2.0.0

## 핵심 개념

### SAE가 학습하는 것

SAE는 희소 병목(sparse bottleneck)을 통해 모델 활성화를 재구성하도록 학습됩니다:

```
입력 활성화(Input Activation) → 인코더(Encoder) → 희소 특징(Sparse Features) → 디코더(Decoder) → 재구성된 활성화(Reconstructed Activation)
    (d_model)                 ↓                (d_sae >> d_model)            ↓                  (d_model)
                           희소성 패널티                                    재구성 손실
                           (sparsity penalty)                               (reconstruction loss)
```

**손실 함수(Loss Function)**: `MSE(original, reconstructed) + L1_coefficient × L1(features)`

### 주요 검증 (Anthropic Research)

"Towards Monosemanticity" 논문에서 인간 평가자들은 **SAE 특징의 70%가 진정으로 해석 가능**하다는 것을 발견했습니다. 발견된 특징은 다음과 같습니다:
- DNA 시퀀스, 법률 용어, HTTP 요청
- 히브리어 텍스트, 영양 성분 표시, 코드 구문
- 감정, 개체명, 문법 구조

## 워크플로 1: 사전 학습된 SAE 로드 및 분석

### 단계별

```python
from transformer_lens import HookedTransformer
from sae_lens import SAE

# 1. 모델과 사전 학습된 SAE 로드
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)

# 2. 모델 활성화 가져오기
tokens = model.to_tokens("The capital of France is Paris")
_, cache = model.run_with_cache(tokens)
activations = cache["resid_pre", 8]  # [batch, pos, d_model]

# 3. SAE 특징으로 인코딩
sae_features = sae.encode(activations)  # [batch, pos, d_sae]
print(f"활성화된 특징 수: {(sae_features > 0).sum()}")

# 4. 각 위치에 대한 상위 특징 찾기
for pos in range(tokens.shape[1]):
    top_features = sae_features[0, pos].topk(5)
    token = model.to_str_tokens(tokens[0, pos:pos+1])[0]
    print(f"Token '{token}': features {top_features.indices.tolist()}")

# 5. 활성화 재구성
reconstructed = sae.decode(sae_features)
reconstruction_error = (activations - reconstructed).norm()
```

### 사용 가능한 사전 학습된 SAE

| 릴리스(Release) | 모델 | 레이어 |
|---------|-------|--------|
| `gpt2-small-res-jb` | GPT-2 Small | 다중 residual 스트림 |
| `gemma-2b-res` | Gemma 2B | Residual 스트림 |
| HuggingFace의 다양한 항목 | 태그 검색 `saelens` | 다양함 |

### 체크리스트
- [ ] TransformerLens로 모델 로드
- [ ] 대상 레이어에 맞는 일치하는 SAE 로드
- [ ] 활성화를 희소 특징으로 인코딩
- [ ] 토큰당 상위 활성화 특징 식별
- [ ] 재구성 품질 검증

## 워크플로 2: 커스텀 SAE 학습

### 단계별

```python
from sae_lens import SAE, LanguageModelSAERunnerConfig, SAETrainingRunner

# 1. 학습 구성
cfg = LanguageModelSAERunnerConfig(
    # 모델
    model_name="gpt2-small",
    hook_name="blocks.8.hook_resid_pre",
    hook_layer=8,
    d_in=768,  # 모델 차원

    # SAE 아키텍처
    architecture="standard",  # 또는 "gated", "topk"
    d_sae=768 * 8,  # 확장 계수(Expansion factor) 8
    activation_fn="relu",

    # 학습
    lr=4e-4,
    l1_coefficient=8e-5,  # 희소성 페널티
    l1_warm_up_steps=1000,
    train_batch_size_tokens=4096,
    training_tokens=100_000_000,

    # 데이터
    dataset_path="monology/pile-uncopyrighted",
    context_size=128,

    # 로깅
    log_to_wandb=True,
    wandb_project="sae-training",

    # 체크포인트
    checkpoint_path="checkpoints",
    n_checkpoints=5,
)

# 2. 학습
trainer = SAETrainingRunner(cfg)
sae = trainer.run()

# 3. 평가
print(f"L0 (평균 활성 특징 수): {trainer.metrics['l0']}")
print(f"CE Loss 복구율: {trainer.metrics['ce_loss_score']}")
```

### 주요 하이퍼파라미터

| 매개변수 | 일반적인 값 | 효과 |
|-----------|---------------|--------|
| `d_sae` | 4-16× d_model | 더 많은 특징, 더 높은 용량 |
| `l1_coefficient` | 5e-5 ~ 1e-4 | 높을수록 = 더 희소함, 정확도 감소 |
| `lr` | 1e-4 ~ 1e-3 | 표준 옵티마이저 학습률 |
| `l1_warm_up_steps` | 500-2000 | 특징이 조기에 죽는 현상(dead features) 방지 |

### 평가 지표

| 지표 | 목표 | 의미 |
|--------|--------|---------|
| **L0** | 50-200 | 토큰당 평균 활성 특징 수 |
| **CE Loss Score** | 80-95% | 원본 대비 교차 엔트로피 복구율 |
| **Dead Features** | &lt;5% | 한 번도 활성화되지 않는 특징 |
| **Explained Variance** | >90% | 재구성 품질 |

### 체크리스트
- [ ] 대상 레이어 및 훅(hook) 지점 선택
- [ ] 확장 계수 설정 (d_sae = 4-16× d_model)
- [ ] 원하는 희소성을 위해 L1 계수 조정
- [ ] Dead feature를 방지하기 위해 L1 웜업 활성화
- [ ] 학습 중 지표 모니터링 (W&B)
- [ ] L0 및 CE 손실 복구율 검증
- [ ] Dead feature 비율 확인

## 워크플로 3: 특징 분석 및 스티어링

### 개별 특징 분석

```python
from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch

model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae, _, _ = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)

# 특정 특징을 활성화하는 원인 찾기
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

### 특징 스티어링(Feature Steering)

```python
def steer_with_feature(model, sae, prompt, feature_idx, strength=5.0):
    """SAE 특징 방향을 residual 스트림에 추가합니다."""
    tokens = model.to_tokens(prompt)

    # 디코더에서 특징 방향 가져오기
    feature_direction = sae.W_dec[feature_idx]  # [d_model]

    def steering_hook(activation, hook):
        # 모든 위치에 스케일된 특징 방향 추가
        activation += strength * feature_direction
        return activation

    # 스티어링과 함께 생성
    output = model.generate(
        tokens,
        max_new_tokens=50,
        fwd_hooks=[("blocks.8.hook_resid_pre", steering_hook)]
    )
    return model.to_string(output[0])
```

### 특징 기여도 분석(Feature Attribution)

```python
# 어떤 특징이 특정 출력에 가장 큰 영향을 미치는지?
tokens = model.to_tokens("The capital of France is")
_, cache = model.run_with_cache(tokens)

# 마지막 위치에서 특징 가져오기
features = sae.encode(cache["resid_pre", 8])[0, -1]  # [d_sae]

# 특징별 로짓 기여도 가져오기
# 특징 기여도 = feature_activation × decoder_weight × unembedding
W_dec = sae.W_dec  # [d_sae, d_model]
W_U = model.W_U    # [d_model, vocab]

# "Paris" 로짓에 대한 기여도
paris_token = model.to_single_token(" Paris")
feature_contributions = features * (W_dec @ W_U[:, paris_token])

top_features = feature_contributions.topk(10)
print("'Paris' 예측에 대한 상위 특징:")
for idx, val in zip(top_features.indices, top_features.values):
    print(f"  Feature {idx.item()}: {val.item():.3f}")
```

## 일반적인 문제 및 해결책

### 문제: 높은 Dead feature 비율
```python
# 잘못된 예: 웜업이 없어서 특징이 일찍 죽음
cfg = LanguageModelSAERunnerConfig(
    l1_coefficient=1e-4,
    l1_warm_up_steps=0,  # Bad!
)

# 올바른 예: L1 페널티 웜업
cfg = LanguageModelSAERunnerConfig(
    l1_coefficient=8e-5,
    l1_warm_up_steps=1000,  # 점진적으로 증가
    use_ghost_grads=True,   # dead features 되살리기
)
```

### 문제: 열악한 재구성 (낮은 CE 복구율)
```python
# 희소성 페널티 줄이기
cfg = LanguageModelSAERunnerConfig(
    l1_coefficient=5e-5,  # 낮을수록 = 더 나은 재구성
    d_sae=768 * 16,       # 더 많은 용량
)
```

### 문제: 특징이 해석 불가능함
```python
# 희소성 증가 (더 높은 L1)
cfg = LanguageModelSAERunnerConfig(
    l1_coefficient=1e-4,  # 높을수록 = 더 희소함, 더 해석 가능함
)
# 또는 TopK 아키텍처 사용
cfg = LanguageModelSAERunnerConfig(
    architecture="topk",
    activation_fn_kwargs={"k": 50},  # 정확히 50개의 특징만 활성화
)
```

### 문제: 학습 중 메모리 오류
```python
cfg = LanguageModelSAERunnerConfig(
    train_batch_size_tokens=2048,  # 배치 크기 줄이기
    store_batch_size_prompts=4,    # 버퍼에 있는 프롬프트 수 줄이기
    n_batches_in_buffer=8,         # 활성화 버퍼 크기 줄이기
)
```

## Neuronpedia 통합

[neuronpedia.org](https://neuronpedia.org)에서 사전 학습된 SAE 특징을 찾아볼 수 있습니다:

```python
# 특징은 SAE ID로 인덱싱됩니다.
# 예시: gpt2-small 레이어 8 특징 1234
# → neuronpedia.org/gpt2-small/8-res-jb/1234
```

## 핵심 클래스 참조

| 클래스 | 목적 |
|-------|---------|
| `SAE` | 희소 오토인코더 모델 |
| `LanguageModelSAERunnerConfig` | 학습 구성 |
| `SAETrainingRunner` | 학습 루프 관리자 |
| `ActivationsStore` | 활성화 수집 및 배치 처리 |
| `HookedSAETransformer` | TransformerLens + SAE 통합 |

## 참고 문서

자세한 API 문서, 튜토리얼 및 고급 사용법은 `references/` 폴더를 참조하세요:

| 파일 | 내용 |
|------|----------|
| [references/README.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/saelens/references/README.md) | 개요 및 빠른 시작 가이드 |
| [references/api.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/saelens/references/api.md) | SAE, TrainingSAE, 구성에 대한 전체 API 참조 |
| [references/tutorials.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/saelens/references/tutorials.md) | 학습, 분석, 스티어링에 대한 단계별 튜토리얼 |

## 외부 리소스

### 튜토리얼
- [기본 로딩 & 분석](https://github.com/jbloomAus/SAELens/blob/main/tutorials/basic_loading_and_analysing.ipynb)
- [희소 오토인코더 학습](https://github.com/jbloomAus/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb)
- [ARENA SAE 커리큘럼](https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab)

### 논문
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) - Anthropic (2023)
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) - Anthropic (2024)
- [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600) - Cunningham et al. (ICLR 2024)

### 공식 문서
- [SAELens Docs](https://jbloomaus.github.io/SAELens/)
- [Neuronpedia](https://neuronpedia.org) - 특징 브라우저

## SAE 아키텍처

| 아키텍처 | 설명 | 사용 사례 |
|--------------|-------------|----------|
| **Standard** | ReLU + L1 페널티 | 범용 |
| **Gated** | 학습된 게이팅 메커니즘 | 더 나은 희소성 제어 |
| **TopK** | 정확히 K개의 특징만 활성화 | 일관된 희소성 |

```python
# TopK SAE (정확히 50개의 특징만 활성화)
cfg = LanguageModelSAERunnerConfig(
    architecture="topk",
    activation_fn="topk",
    activation_fn_kwargs={"k": 50},
)
```
