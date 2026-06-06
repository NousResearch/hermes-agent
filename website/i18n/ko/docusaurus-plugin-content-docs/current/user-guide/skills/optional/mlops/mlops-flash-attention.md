---
title: "Optimizing Attention Flash"
sidebar_label: "Optimizing Attention Flash"
description: "Flash Attention을 사용하여 트랜스포머 어텐션을 최적화하여 2~4배의 속도 향상과 10~20배의 메모리 절감을 제공합니다."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Optimizing Attention Flash

Flash Attention을 사용하여 트랜스포머 어텐션을 최적화하여 2~4배의 속도 향상과 10~20배의 메모리 절감을 달성합니다. 긴 시퀀스(>512 토큰)를 사용하여 트랜스포머를 학습하거나 실행할 때, 어텐션으로 인한 GPU 메모리 문제가 발생할 때, 또는 더 빠른 추론이 필요할 때 사용하세요. PyTorch 기본 SDPA, flash-attn 라이브러리, H100 FP8 및 슬라이딩 윈도우 어텐션을 지원합니다.

## Skill metadata

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/flash-attention`로 설치 |
| Path | `optional-skills/mlops/flash-attention` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `flash-attn`, `torch`, `transformers` |
| Platforms | linux, macos |
| Tags | `Optimization`, `Flash Attention`, `Attention Optimization`, `Memory Efficiency`, `Speed Optimization`, `Long Context`, `PyTorch`, `SDPA`, `H100`, `FP8`, `Transformers` |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Flash Attention - 빠르고 메모리 효율적인 어텐션

## 빠른 시작

Flash Attention은 IO 인식 타일링(tiling) 및 재계산을 통해 트랜스포머 어텐션에서 2~4배의 속도 향상과 10~20배의 메모리 감소를 제공합니다.

**PyTorch 기본 제공 기능 (가장 쉬움, PyTorch 2.2 이상)**:
```python
import torch
import torch.nn.functional as F

q = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)  # [배치, 헤드, 시퀀스, 차원]
k = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)

# 사용 가능한 경우 Flash Attention을 자동으로 사용합니다.
out = F.scaled_dot_product_attention(q, k, v)
```

**flash-attn 라이브러리 (더 많은 기능 제공)**:
```bash
pip install flash-attn --no-build-isolation
```

```python
from flash_attn import flash_attn_func

# q, k, v: [배치, 시퀀스 길이, 헤드 수, 헤드 차원]
out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
```

## 일반적인 워크플로우

### 워크플로우 1: 기존 PyTorch 모델에서 활성화하기

다음 체크리스트를 복사하세요:

```
Flash Attention 통합:
- [ ] 1단계: PyTorch 버전 확인 (≥2.2)
- [ ] 2단계: Flash Attention 백엔드 활성화
- [ ] 3단계: 프로파일링으로 속도 향상 확인
- [ ] 4단계: 기준선(baseline)과 정확도가 일치하는지 테스트
```

**1단계: PyTorch 버전 확인**

```bash
python -c "import torch; print(torch.__version__)"
# 2.2.0 이상이어야 합니다.
```

2.2 미만인 경우 업그레이드하세요:
```bash
pip install --upgrade torch
```

**2단계: Flash Attention 백엔드 활성화**

표준 어텐션을 다음으로 교체하세요:
```python
# 이전 (표준 어텐션)
attn_weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d_k), dim=-1)
out = attn_weights @ v

# 이후 (Flash Attention)
import torch.nn.functional as F
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

Flash Attention 백엔드를 강제로 사용하려면:
```python
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    out = F.scaled_dot_product_attention(q, k, v)
```

**3단계: 프로파일링으로 속도 향상 확인**

```python
import torch.utils.benchmark as benchmark

def test_attention(use_flash):
    q, k, v = [torch.randn(2, 8, 2048, 64, device='cuda', dtype=torch.float16) for _ in range(3)]

    if use_flash:
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            return F.scaled_dot_product_attention(q, k, v)
    else:
        attn = (q @ k.transpose(-2, -1) / 8.0).softmax(dim=-1)
        return attn @ v

# 벤치마크
t_flash = benchmark.Timer(stmt='test_attention(True)', globals=globals())
t_standard = benchmark.Timer(stmt='test_attention(False)', globals=globals())

print(f"Flash: {t_flash.timeit(100).mean:.3f}s")
print(f"표준: {t_standard.timeit(100).mean:.3f}s")
```

기대 결과: >512 토큰 시퀀스에서 2~4배 속도 향상.

**4단계: 기준선과 정확도가 일치하는지 테스트**

```python
# 출력 비교
q, k, v = [torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16) for _ in range(3)]

# Flash Attention
out_flash = F.scaled_dot_product_attention(q, k, v)

# 표준 어텐션
attn_weights = torch.softmax(q @ k.transpose(-2, -1) / 8.0, dim=-1)
out_standard = attn_weights @ v

# 차이 확인
diff = (out_flash - out_standard).abs().max()
print(f"최대 차이: {diff:.6f}")
# float16의 경우 &lt;1e-3 이내여야 합니다.
```

### 워크플로우 2: 고급 기능을 위해 flash-attn 라이브러리 사용하기

다중 쿼리 어텐션(Multi-query attention), 슬라이딩 윈도우, 또는 H100 FP8과 같은 기능이 필요할 때 사용합니다.

다음 체크리스트를 복사하세요:

```
flash-attn 라이브러리 설정:
- [ ] 1단계: flash-attn 라이브러리 설치
- [ ] 2단계: 어텐션 코드 수정
- [ ] 3단계: 고급 기능 활성화
- [ ] 4단계: 성능 벤치마킹
```

**1단계: flash-attn 라이브러리 설치**

```bash
# NVIDIA GPU (CUDA 12.0 이상)
pip install flash-attn --no-build-isolation

# 설치 확인
python -c "from flash_attn import flash_attn_func; print('Success')"
```

**2단계: 어텐션 코드 수정**

```python
from flash_attn import flash_attn_func

# 입력: [배치 크기, 시퀀스 길이, 헤드 수, 헤드 차원]
# [배치, 헤드, 시퀀스, 차원] 형태인 경우 transpose를 수행해야 합니다.
q = q.transpose(1, 2)  # [배치, 시퀀스, 헤드, 차원]
k = k.transpose(1, 2)
v = v.transpose(1, 2)

out = flash_attn_func(
    q, k, v,
    dropout_p=0.1,
    causal=True,  # 자기회귀(autoregressive) 모델용
    window_size=(-1, -1),  # 슬라이딩 윈도우 없음
    softmax_scale=None  # 자동 스케일
)

out = out.transpose(1, 2)  # 다시 [배치, 헤드, 시퀀스, 차원]으로 변경
```

**3단계: 고급 기능 활성화**

다중 쿼리 어텐션 (헤드 간 K/V 공유):
```python
from flash_attn import flash_attn_func

# q: [배치, 시퀀스, q_헤드_수, 차원]
# k, v: [배치, 시퀀스, kv_헤드_수, 차원]  # 더 적은 수의 KV 헤드
out = flash_attn_func(q, k, v)  # 자동으로 MQA를 처리합니다.
```

슬라이딩 윈도우 어텐션 (로컬 어텐션):
```python
# 이전/이후 256 토큰의 윈도우에만 어텐션 적용
out = flash_attn_func(
    q, k, v,
    window_size=(256, 256),  # (왼쪽, 오른쪽) 윈도우
    causal=True
)
```

**4단계: 성능 벤치마킹**

```python
import torch
from flash_attn import flash_attn_func
import time

q, k, v = [torch.randn(4, 4096, 32, 64, device='cuda', dtype=torch.float16) for _ in range(3)]

# 웜업(Warmup)
for _ in range(10):
    _ = flash_attn_func(q, k, v)

# 벤치마크
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    out = flash_attn_func(q, k, v)
    torch.cuda.synchronize()
end = time.time()

print(f"반복당 소요 시간: {(end-start)/100*1000:.2f}ms")
print(f"할당된 메모리: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
```

### 워크플로우 3: H100 FP8 최적화 (FlashAttention-3)

H100 GPU에서 최고의 성능을 내기 위한 최적화입니다.

```
FP8 설정:
- [ ] 1단계: H100 GPU 사용 가능 여부 확인
- [ ] 2단계: FP8을 지원하는 flash-attn 설치
- [ ] 3단계: 입력을 FP8로 변환
- [ ] 4단계: FP8 어텐션으로 실행
```

**1단계: H100 GPU 확인**

```bash
nvidia-smi --query-gpu=name --format=csv
# "H100" 또는 "H800"이 표시되어야 합니다.
```

**2단계: FP8을 지원하는 flash-attn 설치**

```bash
pip install flash-attn --no-build-isolation
# H100용 FP8 지원이 포함되어 있습니다.
```

**3단계: 입력을 FP8로 변환**

```python
import torch

q = torch.randn(2, 4096, 32, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 4096, 32, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 4096, 32, 64, device='cuda', dtype=torch.float16)

# float8_e4m3 (FP8)로 변환
q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = k.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)
```

**4단계: FP8 어텐션으로 실행**

```python
from flash_attn import flash_attn_func

# FlashAttention-3은 H100에서 자동으로 FP8 커널을 사용합니다.
out = flash_attn_func(q_fp8, k_fp8, v_fp8)
# 결과: ~1.2 PFLOPS, FP16보다 1.5~2배 빠름
```

## 사용 시기 및 대안

**Flash Attention을 사용하는 경우:**
- >512 토큰 시퀀스로 트랜스포머를 학습할 때
- 긴 컨텍스트(>2K 토큰)로 추론을 실행할 때
- GPU 메모리가 부족할 때 (표준 어텐션에서 OOM 발생)
- 정확도 손실 없이 2~4배의 속도 향상이 필요할 때
- PyTorch 2.2 이상을 사용 중이거나 flash-attn을 설치할 수 있을 때

**대신 사용할 수 있는 대안:**
- **표준 어텐션**: &lt;256 토큰의 시퀀스 (오버헤드 대비 이점이 적음)
- **xFormers**: (속도 외에) 더 다양한 어텐션 변형이 필요할 때
- **메모리 효율적인 어텐션**: CPU 추론 (Flash Attention은 GPU가 필요함)

## 일반적인 문제

**문제: ImportError: cannot import flash_attn**

no-build-isolation 플래그와 함께 설치하세요:
```bash
pip install flash-attn --no-build-isolation
```

또는 CUDA 툴킷을 먼저 설치하세요:
```bash
conda install cuda -c nvidia
pip install flash-attn --no-build-isolation
```

**문제: 예상보다 느림 (속도 향상 없음)**

Flash Attention의 이점은 시퀀스 길이가 길어질수록 커집니다:
- &lt;512 토큰: 최소한의 속도 향상 (10-20%)
- 512-2K 토큰: 2-3배 속도 향상
- >2K 토큰: 3-4배 속도 향상

시퀀스 길이가 충분히 긴지 확인하세요.

**문제: RuntimeError: CUDA error**

GPU가 Flash Attention을 지원하는지 확인하세요:
```python
import torch
print(torch.cuda.get_device_capability())
# Turing 아키텍처 이상의 경우 ≥(7, 5)이어야 합니다.
```

Flash Attention의 요구 사항:
- Ampere (A100, A10): ✅ 완벽 지원
- Turing (T4): ✅ 지원됨
- Volta (V100): ❌ 지원되지 않음

**문제: 정확도 저하**

dtype이 float32가 아닌 float16 또는 bfloat16인지 확인하세요:
```python
q = q.to(torch.float16)  # 또는 torch.bfloat16
```

Flash Attention은 속도를 위해 float16/bfloat16을 사용합니다. Float32는 지원하지 않습니다.

## 고급 주제

**HuggingFace Transformers와의 통합**: BERT, GPT, Llama 모델에서 Flash Attention을 활성화하는 방법은 [references/transformers-integration.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/flash-attention/references/transformers-integration.md)를 참조하세요.

**성능 벤치마크**: GPU 및 시퀀스 길이별 자세한 속도 및 메모리 비교는 [references/benchmarks.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/flash-attention/references/benchmarks.md)를 참조하세요.

## 하드웨어 요구 사항

- **GPU**: NVIDIA Ampere 이상 (A100, A10, A30) 또는 AMD MI200 이상
- **VRAM**: 표준 어텐션과 동일 (Flash Attention은 메모리 사용량을 늘리지 않음)
- **CUDA**: 12.0 이상 (최소 11.8)
- **PyTorch**: 기본 지원을 위해 2.2 이상 필요

**지원되지 않음**: V100 (Volta), CPU 추론

## 리소스

- 논문: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022)
- 논문: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (ICLR 2024)
- 블로그: https://tridao.me/blog/2024/flash3/
- GitHub: https://github.com/Dao-AILab/flash-attention
- PyTorch 문서: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
