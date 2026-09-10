---
title: "Tensorrt Llm — NVIDIA GPU에서의 고처리량 LLM 추론"
sidebar_label: "Tensorrt Llm"
description: "NVIDIA GPU에서의 고처리량 LLM 추론"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Tensorrt Llm

NVIDIA GPU에서 고성능으로 LLM 추론을 최적화하기 위한 NVIDIA의 오픈 소스 라이브러리입니다.

## Skill 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/mlops/tensorrt-llm`으로 설치 |
| 경로 | `optional-skills/mlops/tensorrt-llm` |
| 버전 | `1.0.1` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `tensorrt-llm`, `torch` |
| 플랫폼 | linux, macos |
| 태그 | `Inference Serving`, `TensorRT-LLM`, `NVIDIA`, `Inference Optimization`, `High Throughput`, `Low Latency`, `Production`, `FP8`, `INT4`, `In-Flight Batching`, `Multi-GPU` |

## 참조: 전체 SKILL.md

:::info
다음은 이 skill이 트리거될 때 Hermes가 로드하는 완전한 skill 정의입니다. skill이 활성화되었을 때 에이전트가 보는 지침입니다.
:::

# TensorRT-LLM

NVIDIA GPU에서 고성능 LLM 추론을 최적화하기 위한 NVIDIA의 오픈 소스 라이브러리입니다.

## TensorRT-LLM을 사용할 때

**다음과 같은 경우 TensorRT-LLM을 사용합니다.**
- NVIDIA GPU(A100, H100, GB200)에 배포할 때
- 최대 처리량이 필요할 때(Llama 3에서 초당 24,000개 이상의 토큰)
- 실시간 애플리케이션에 낮은 지연 시간이 필요할 때
- 양자화된 모델(FP8, INT4, FP4)을 사용할 때
- 여러 GPU 또는 노드로 확장할 때

**다음과 같은 경우에는 대신 vLLM을 사용합니다.**
- 더 간단한 설정과 Python 우선 API가 필요할 때
- TensorRT 컴파일 없이 PagedAttention을 사용하려 할 때
- AMD GPU 또는 NVIDIA가 아닌 하드웨어에서 작업할 때

**다음과 같은 경우에는 대신 llama.cpp를 사용합니다.**
- CPU 또는 Apple Silicon에 배포할 때
- NVIDIA GPU 없이 엣지 배포가 필요할 때
- 더 간단한 GGUF 양자화 형식을 원할 때

## 빠른 시작

### 설치

```bash
# Docker (recommended) — images are on NGC (nvcr.io), not Docker Hub.
# Replace x.y.z with the desired version (e.g. 1.2.1). Browse tags on NGC:
# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags
docker pull nvcr.io/nvidia/tensorrt-llm/release:x.y.z

# pip install (current stable GA)
pip install tensorrt_llm

# Requires CUDA 13.2.1, TensorRT 10.x, Python 3.10-3.12
```

### 기본 추론

```python
from tensorrt_llm import LLM, SamplingParams

# Initialize model
llm = LLM(model="meta-llama/Meta-Llama-3-8B")

# Configure sampling
sampling_params = SamplingParams(
    max_tokens=100,
    temperature=0.7,
    top_p=0.9
)

# Generate
prompts = ["Explain quantum computing"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.text)
```

### trtllm-serve로 서빙하기

```bash
# Start server (automatic model download and compilation)
trtllm-serve meta-llama/Meta-Llama-3-8B \
    --tp_size 4 \              # Tensor parallelism (4 GPUs)
    --max_batch_size 256 \
    --max_num_tokens 4096

# Client request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

## 주요 기능

### 성능 최적화
- **In-flight batching**: 생성 중 동적 배칭
- **Paged KV cache**: 효율적인 메모리 관리
- **Flash Attention**: 최적화된 어텐션 커널
- **양자화**: 2~4배 빠른 추론을 위한 FP8, INT4, FP4
- **CUDA 그래프**: 커널 실행 오버헤드 감소

### 병렬 처리
- **Tensor parallelism (TP)**: 여러 GPU에 모델 분할
- **Pipeline parallelism (PP)**: 레이어 단위 분산
- **Expert parallelism**: Mixture-of-Experts 모델용
- **Multi-node**: 단일 머신을 넘어 확장

### 고급 기능
- **Speculative decoding**: 초안 모델을 사용한 더 빠른 생성
- **LoRA serving**: 효율적인 다중 어댑터 배포
- **Disaggregated serving**: 프리필과 생성을 분리

## 일반적인 패턴

### 양자화된 모델(FP8)

```python
from tensorrt_llm import LLM

# Load FP8 quantized model (2× faster, 50% memory)
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B",
    dtype="fp8",
    max_num_tokens=8192
)

# Inference same as before
outputs = llm.generate(["Summarize this article..."])
```

### 다중 GPU 배포

```python
# Tensor parallelism across 8 GPUs
llm = LLM(
    model="meta-llama/Meta-Llama-3-405B",
    tensor_parallel_size=8,
    dtype="fp8"
)
```

### 배치 추론

```python
# Process 100 prompts efficiently
prompts = [f"Question {i}: ..." for i in range(100)]

outputs = llm.generate(
    prompts,
    sampling_params=SamplingParams(max_tokens=200)
)

# Automatic in-flight batching for maximum throughput
```

## 성능 벤치마크

**Meta Llama 3-8B** (H100 GPU):
- 처리량: 초당 24,000 토큰
- 지연 시간: 토큰당 약 10ms
- PyTorch 대비: **100배 빠름**

**Llama 3-70B** (8× A100 80GB):
- FP8 양자화: FP16보다 2배 빠름
- 메모리: FP8 사용 시 50% 감소

## 지원 모델

- **LLaMA 제품군**: Llama 2, Llama 3, CodeLlama
- **GPT 제품군**: GPT-2, GPT-J, GPT-NeoX
- **Qwen**: Qwen, Qwen2, QwQ
- **DeepSeek**: DeepSeek-V2, DeepSeek-V3
- **Mixtral**: Mixtral-8x7B, Mixtral-8x22B
- **Vision**: LLaVA, Phi-3-vision
- HuggingFace의 **100개 이상의 모델**

## 참고 자료

- **[최적화 가이드](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/tensorrt-llm/references/optimization.md)** - 양자화, 배칭, KV 캐시 튜닝
- **[다중 GPU 설정](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/tensorrt-llm/references/multi-gpu.md)** - 텐서/파이프라인 병렬 처리, 다중 노드
- **[서빙 가이드](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/tensorrt-llm/references/serving.md)** - 프로덕션 배포, 모니터링, 오토스케일링

## 리소스

- **문서**: https://nvidia.github.io/TensorRT-LLM/
- **GitHub**: https://github.com/NVIDIA/TensorRT-LLM
- **모델**: https://huggingface.co/models?library=tensorrt_llm
