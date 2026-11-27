---
title: "Flash Attention — 긴 시퀀스 트랜스포머 학습 및 추론 가속"
sidebar_label: "Flash Attention"
description: "긴 시퀀스 트랜스포머 학습 및 추론 가속"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 소스 SKILL.md를 편집하세요. */}

# Flash Attention

긴 시퀀스 트랜스포머 학습 및 추론을 가속합니다.

## Skill 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/flash-attention`으로 설치 |
| 경로 | `optional-skills/mlops/flash-attention` |
| 버전 | `1.0.1` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `flash-attn`, `torch`, `transformers` |
| 플랫폼 | linux, macos |
| 태그 | `Optimization`, `Flash Attention`, `Attention Optimization`, `Memory Efficiency`, `Speed Optimization`, `Long Context`, `PyTorch`, `SDPA`, `H100`, `FP8`, `Transformers` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# Flash Attention - 빠르고 메모리 효율적인 Attention

## 빠른 시작

Flash Attention은 IO 인식 타일링과 재계산을 통해 트랜스포머 attention에서 2~4배의 속도 향상과 10~20배의 메모리 절감을 제공합니다.

**PyTorch 네이티브(가장 쉬운 방법, PyTorch 2.2+)**:
```python
import torch
import torch.nn.functional as F

q = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)  # [batch, heads, seq, dim]
k = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)

# Automatically uses Flash Attention if available
out = F.scaled_dot_product_attention(q, k, v)
```

**flash-attn 라이브러리(더 많은 기능)**:
```bash
pip install flash-attn --no-build-isolation
```

```python
from flash_attn import flash_attn_func

# q, k, v: [batch, seqlen, nheads, headdim]
out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
```

## 일반적인 워크플로

### 워크플로 1: 기존 PyTorch 모델에서 활성화

다음 체크리스트를 복사하세요:

```
Flash Attention Integration:
- [ ] Step 1: Check PyTorch version (≥2.2)
- [ ] Step 2: Enable Flash Attention backend
- [ ] Step 3: Verify speedup with profiling
- [ ] Step 4: Test accuracy matches baseline
```

**1단계: PyTorch 버전 확인**

```bash
python -c "import torch; print(torch.__version__)"
# Should be ≥2.2.0
```

2.2 미만이면 업그레이드하세요:
```bash
pip install --upgrade torch
```

**2단계: Flash Attention 백엔드 활성화**

표준 attention을 다음과 같이 교체하세요:
```python
# Before (standard attention)
attn_weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d_k), dim=-1)
out = attn_weights @ v

# After (Flash Attention)
import torch.nn.functional as F
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

Flash Attention 백엔드를 강제합니다(`torch.backends.cuda.sdp_kernel`은 더 이상 사용되지 않으므로 `SDPBackend`와 함께 `torch.nn.attention.sdpa_kernel`을 사용하세요):
```python
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v)
```

**3단계: 프로파일링으로 속도 향상 확인**

```python
import torch.utils.benchmark as benchmark

def test_attention(use_flash):
    q, k, v = [torch.randn(2, 8, 2048, 64, device='cuda', dtype=torch.float16) for _ in range(3)]

    if use_flash:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v)
    else:
        attn = (q @ k.transpose(-2, -1) / 8.0).softmax(dim=-1)
        return attn @ v

# Benchmark
t_flash = benchmark.Timer(stmt='test_attention(True)', globals=globals())
t_standard = benchmark.Timer(stmt='test_attention(False)', globals=globals())

print(f"Flash: {t_flash.timeit(100).mean:.3f}s")
print(f"Standard: {t_standard.timeit(100).mean:.3f}s")
```

예상 결과: 512 토큰을 초과하는 시퀀스에서 2~4배 속도 향상.

**4단계: 정확도가 기준선과 일치하는지 테스트**

```python
# Compare outputs
q, k, v = [torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16) for _ in range(3)]

# Flash Attention
out_flash = F.scaled_dot_product_attention(q, k, v)

# Standard attention
attn_weights = torch.softmax(q @ k.transpose(-2, -1) / 8.0, dim=-1)
out_standard = attn_weights @ v

# Check difference
diff = (out_flash - out_standard).abs().max()
print(f"Max difference: {diff:.6f}")
# Should be <1e-3 for float16
```

### 워크플로 2: 고급 기능을 위해 flash-attn 라이브러리 사용

멀티 쿼리 attention, 슬라이딩 윈도우 또는 H100 FP8이 필요한 경우에 사용합니다.

다음 체크리스트를 복사하세요:

```
flash-attn Library Setup:
- [ ] Step 1: Install flash-attn library
- [ ] Step 2: Modify attention code
- [ ] Step 3: Enable advanced features
- [ ] Step 4: Benchmark performance
```

**1단계: flash-attn 라이브러리 설치**

```bash
# NVIDIA GPUs (CUDA 12.0+)
pip install flash-attn --no-build-isolation

# Verify installation
python -c "from flash_attn import flash_attn_func; print('Success')"
```

**2단계: attention 코드 수정**

```python
from flash_attn import flash_attn_func

# Input: [batch_size, seq_len, num_heads, head_dim]
# Transpose from [batch, heads, seq, dim] if needed
q = q.transpose(1, 2)  # [batch, seq, heads, dim]
k = k.transpose(1, 2)
v = v.transpose(1, 2)

out = flash_attn_func(
    q, k, v,
    dropout_p=0.1,
    causal=True,  # For autoregressive models
    window_size=(-1, -1),  # No sliding window
    softmax_scale=None  # Auto-scale
)

out = out.transpose(1, 2)  # Back to [batch, heads, seq, dim]
```

**3단계: 고급 기능 활성화**

멀티 쿼리 attention(헤드 간 K/V 공유):
```python
from flash_attn import flash_attn_func

# q: [batch, seq, num_q_heads, dim]
# k, v: [batch, seq, num_kv_heads, dim]  # Fewer KV heads
out = flash_attn_func(q, k, v)  # Automatically handles MQA
```

슬라이딩 윈도우 attention(로컬 attention):
```python
# Only attend to window of 256 tokens before/after
out = flash_attn_func(
    q, k, v,
    window_size=(256, 256),  # (left, right) window
    causal=True
)
```

**4단계: 성능 벤치마크**

```python
import torch
from flash_attn import flash_attn_func
import time

q, k, v = [torch.randn(4, 4096, 32, 64, device='cuda', dtype=torch.float16) for _ in range(3)]

# Warmup
for _ in range(10):
    _ = flash_attn_func(q, k, v)

# Benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    out = flash_attn_func(q, k, v)
    torch.cuda.synchronize()
end = time.time()

print(f"Time per iteration: {(end-start)/100*1000:.2f}ms")
print(f"Memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
```

### 워크플로 3: H100 FP8 최적화(FlashAttention-3)

Hopper GPU(H100)에서 최대 성능을 얻을 때 사용합니다.

> **중요:** pip 패키지 `flash-attn`(2.8.x)은 **FlashAttention-2만** 포함하며 FA3 또는 FP8 H100 커널은 포함하지 않습니다. 또한 `flash_attn_func`는 FP8을 자동으로 사용하지 **않습니다**. FlashAttention-3은 저장소의 `hopper/` 디렉터리에서 소스 코드로 컴파일하는 별도의 **베타** 빌드이며, `flash_attn_interface` 모듈을 통해 제공됩니다. FA3은 FP16/BF16 순전파+역전파와 **FP8 순전파만** 지원합니다.

```
FP8 Setup:
- [ ] Step 1: Verify Hopper (H100) GPU available
- [ ] Step 2: Build & install FlashAttention-3 from source (hopper/)
- [ ] Step 3: Use the FA3 interface (FP8 forward)
```

**1단계: H100 GPU 확인**

```bash
nvidia-smi --query-gpu=name --format=csv
# Should show "H100" or "H800"
```

**2단계: 소스 코드에서 FlashAttention-3 빌드 및 설치**

FA3은 `pip install flash-attn`에 포함되지 **않습니다**. `hopper/` 하위 디렉터리에서 빌드하세요:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
# (compilation is heavy and requires a CUDA toolchain + Hopper GPU)
```

**3단계: FA3 인터페이스 사용(FP8 순전파)**

FA3은 자체 모듈 `flash_attn_interface`를 노출합니다(FA2의 `flash_attn`과는 다름).
FP8은 **순전파 전용** 경로이며 `float8_e4m3fn` 입력을 요구합니다:

```python
import torch
from flash_attn_interface import flash_attn_func  # FA3 (hopper build), not `flash_attn`

# q, k, v: [batch, seqlen, nheads, headdim]
q = torch.randn(2, 4096, 32, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 4096, 32, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 4096, 32, 64, device='cuda', dtype=torch.float16)

# FP8 forward (inference / forward-only): cast to float8_e4m3fn
q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = k.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)

out = flash_attn_func(q_fp8, k_fp8, v_fp8, causal=True)
# FP16/BF16 forward+backward is also supported by the FA3 interface.
```

## 대안과 비교해 언제 사용할지

**다음과 같은 경우 Flash Attention 사용:**
- 512 토큰을 초과하는 시퀀스로 트랜스포머 학습
- 긴 컨텍스트(2K 토큰 초과)로 추론 실행
- GPU 메모리가 제한됨(표준 attention에서 OOM 발생)
- 정확도 손실 없이 2~4배 속도 향상이 필요함
- PyTorch 2.2+를 사용하거나 flash-attn을 설치할 수 있음

**다음과 같은 경우 대안 사용:**
- **표준 attention**: 256 토큰 미만의 시퀀스(오버헤드를 감수할 가치가 없음)
- **xFormers**: 더 많은 attention 변형이 필요함(속도만 필요한 경우가 아님)
- **메모리 효율적인 attention**: CPU 추론(Flash Attention에는 GPU가 필요함)

## 일반적인 문제

**문제: ImportError: cannot import flash_attn**

빌드 격리 없이 설치하세요:
```bash
pip install flash-attn --no-build-isolation
```

또는 먼저 CUDA 툴킷을 설치하세요:
```bash
conda install cuda -c nvidia
pip install flash-attn --no-build-isolation
```

**문제: 예상보다 느림(속도 향상 없음)**

Flash Attention의 이점은 시퀀스 길이가 길수록 커집니다:
- 512 토큰 미만: 최소한의 속도 향상(10~20%)
- 512~2K 토큰: 2~3배 속도 향상
- 2K 토큰 초과: 3~4배 속도 향상

시퀀스 길이가 충분한지 확인하세요.

**문제: RuntimeError: CUDA error**

GPU가 Flash Attention을 지원하는지 확인하세요:
```python
import torch
print(torch.cuda.get_device_capability())
# Should be ≥(7, 5) for Turing+
```

Flash Attention에는 다음이 필요합니다:
- Ampere(A100, A10): ✅ 완전 지원
- Turing(T4): ✅ 지원
- Volta(V100): ❌ 지원되지 않음

**문제: 정확도 저하**

dtype가 float16 또는 bfloat16인지 확인하세요(float32는 사용하지 않음):
```python
q = q.to(torch.float16)  # Or torch.bfloat16
```

Flash Attention은 속도를 위해 float16/bfloat16을 사용합니다. float32는 지원되지 않습니다.

## 고급 주제

**HuggingFace Transformers와의 통합**: BERT, GPT, Llama 모델에서 Flash Attention을 활성화하는 방법은 [references/transformers-integration.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/flash-attention/references/transformers-integration.md)를 참조하세요.

**성능 벤치마크**: GPU와 시퀀스 길이별 자세한 속도 및 메모리 비교는 [references/benchmarks.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/flash-attention/references/benchmarks.md)를 참조하세요.

## 하드웨어 요구 사항

- **GPU**: NVIDIA Ampere 이상(A100, A10, A30) 또는 AMD MI200 이상
- **VRAM**: 표준 attention과 동일(Flash Attention은 메모리를 늘리지 않음)
- **CUDA**: 12.0 이상(최소 11.8)
- **PyTorch**: 네이티브 지원에는 2.2 이상 필요

**지원되지 않음**: V100(Volta), CPU 추론

## 리소스

- 논문: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022)
- 논문: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (ICLR 2024)
- 블로그: https://tridao.me/blog/2024/flash3/
- GitHub: https://github.com/Dao-AILab/flash-attention
- PyTorch 문서: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
