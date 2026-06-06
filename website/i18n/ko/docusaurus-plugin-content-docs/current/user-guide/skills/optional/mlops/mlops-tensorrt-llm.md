---
title: "Tensorrt Llm — 최대 처리량과 최저 지연 시간을 위한 NVIDIA TensorRT LLM 추론 최적화"
sidebar_label: "Tensorrt Llm"
description: "최대 처리량과 최저 지연 시간을 위해 NVIDIA TensorRT를 사용하여 LLM 추론을 최적화합니다."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Tensorrt Llm

최대 처리량과 최저 지연 시간을 달성하기 위해 NVIDIA TensorRT로 LLM 추론을 최적화합니다. NVIDIA GPU(A100/H100)의 프로덕션 배포에서 PyTorch보다 10~100배 빠른 추론 속도가 필요하거나 양자화(FP8/INT4), 인플라이트(in-flight) 배칭, 멀티 GPU 확장 등을 모델 서빙에 활용해야 할 때 사용합니다.

## Skill metadata

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/tensorrt-llm`로 설치 |
| Path | `optional-skills/mlops/tensorrt-llm` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `tensorrt-llm`, `torch` |
| Platforms | linux, macos |
| Tags | `Inference Serving`, `TensorRT-LLM`, `NVIDIA`, `Inference Optimization`, `High Throughput`, `Low Latency`, `Production`, `FP8`, `INT4`, `In-Flight Batching`, `Multi-GPU` |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# TensorRT-LLM

NVIDIA GPU에서 최첨단 성능을 갖춘 LLM 추론을 최적화하기 위한 NVIDIA의 오픈 소스 라이브러리입니다.

## TensorRT-LLM을 사용해야 하는 경우

**사용 시기:**
- NVIDIA GPU (A100, H100, GB200)에 배포할 때
- 최대 처리량이 필요할 때 (Llama 3 기준 24,000+ 토큰/초)
- 실시간 애플리케이션을 위한 짧은 지연 시간이 필요할 때
- 양자화 모델 (FP8, INT4, FP4)을 활용할 때
- 여러 GPU나 노드로 확장해야 할 때

**대신 vLLM을 사용하는 경우:**
- 더 간단한 설정과 Python 우선(Python-first) API가 필요할 때
- TensorRT 컴파일 없이 PagedAttention을 사용하고 싶을 때
- AMD GPU 또는 NVIDIA가 아닌 다른 하드웨어에서 작업할 때

**대신 llama.cpp를 사용하는 경우:**
- CPU나 Apple Silicon에 배포할 때
- NVIDIA GPU 없이 엣지(edge) 환경 배포가 필요할 때
- 더 단순한 GGUF 양자화 형식을 원할 때

## 빠른 시작

### 설치

```bash
# Docker (권장)
docker pull nvidia/tensorrt_llm:latest

# pip 설치
pip install tensorrt_llm==1.2.0rc3

# 요구 사항: CUDA 13.0.0, TensorRT 10.13.2, Python 3.10-3.12
```

### 기본 추론

```python
from tensorrt_llm import LLM, SamplingParams

# 모델 초기화
llm = LLM(model="meta-llama/Meta-Llama-3-8B")

# 샘플링 설정
sampling_params = SamplingParams(
    max_tokens=100,
    temperature=0.7,
    top_p=0.9
)

# 생성
prompts = ["양자 컴퓨팅에 대해 설명해줘"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.text)
```

### trtllm-serve를 이용한 서빙

```bash
# 서버 시작 (자동으로 모델 다운로드 및 컴파일)
trtllm-serve meta-llama/Meta-Llama-3-8B \
    --tp_size 4 \              # 텐서 병렬 처리 (4 GPU)
    --max_batch_size 256 \
    --max_num_tokens 4096

# 클라이언트 요청
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B",
    "messages": [{"role": "user", "content": "안녕하세요!"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

## 주요 기능

### 성능 최적화
- **인플라이트(In-flight) 배칭**: 생성 중 동적 배칭 처리
- **페이징 KV 캐시 (Paged KV cache)**: 효율적인 메모리 관리
- **Flash Attention**: 최적화된 어텐션 커널
- **양자화**: FP8, INT4, FP4를 통한 2~4배 빠른 추론
- **CUDA 그래프**: 커널 실행 오버헤드 감소

### 병렬 처리 (Parallelism)
- **텐서 병렬 처리 (TP)**: 모델을 여러 GPU로 분할
- **파이프라인 병렬 처리 (PP)**: 계층별(Layer-wise) 분산
- **전문가 병렬 처리 (Expert parallelism)**: MoE(Mixture-of-Experts) 모델용
- **다중 노드 (Multi-node)**: 단일 기기를 넘어서는 확장성

### 고급 기능
- **추측 디코딩 (Speculative decoding)**: 초안(draft) 모델을 이용한 빠른 생성
- **LoRA 서빙**: 여러 어댑터를 효율적으로 배포
- **분리 서빙 (Disaggregated serving)**: 프리필(prefill)과 생성 단계 분리

## 일반적인 패턴

### 양자화 모델 (FP8)

```python
from tensorrt_llm import LLM

# FP8 양자화 모델 불러오기 (2배 빠름, 50% 메모리 절감)
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B",
    dtype="fp8",
    max_num_tokens=8192
)

# 추론은 이전과 동일합니다
outputs = llm.generate(["이 기사를 요약해줘..."])
```

### 다중 GPU 배포

```python
# 8개 GPU에서의 텐서 병렬 처리
llm = LLM(
    model="meta-llama/Meta-Llama-3-405B",
    tensor_parallel_size=8,
    dtype="fp8"
)
```

### 배치 추론

```python
# 100개의 프롬프트를 효율적으로 처리
prompts = [f"질문 {i}: ..." for i in range(100)]

outputs = llm.generate(
    prompts,
    sampling_params=SamplingParams(max_tokens=200)
)

# 최대 처리량을 위한 자동 인플라이트(in-flight) 배칭
```

## 성능 벤치마크

**Meta Llama 3-8B** (H100 GPU):
- 처리량: 24,000 토큰/초
- 지연 시간: 토큰당 ~10ms
- vs PyTorch: **100배 빠름**

**Llama 3-70B** (8× A100 80GB):
- FP8 양자화: FP16보다 2배 빠름
- 메모리: FP8로 50% 절감

## 지원 모델

- **LLaMA 제품군**: Llama 2, Llama 3, CodeLlama
- **GPT 제품군**: GPT-2, GPT-J, GPT-NeoX
- **Qwen**: Qwen, Qwen2, QwQ
- **DeepSeek**: DeepSeek-V2, DeepSeek-V3
- **Mixtral**: Mixtral-8x7B, Mixtral-8x22B
- **Vision**: LLaVA, Phi-3-vision
- HuggingFace의 **100개 이상의 모델**

## 참고 자료 (References)

- **[최적화 가이드](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/tensorrt-llm/references/optimization.md)** - 양자화, 배칭, KV 캐시 튜닝
- **[다중 GPU 설정](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/tensorrt-llm/references/multi-gpu.md)** - 텐서/파이프라인 병렬 처리, 다중 노드
- **[서빙 가이드](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/tensorrt-llm/references/serving.md)** - 프로덕션 배포, 모니터링, 오토스케일링

## 리소스

- **공식 문서**: https://nvidia.github.io/TensorRT-LLM/
- **GitHub**: https://github.com/NVIDIA/TensorRT-LLM
- **모델**: https://huggingface.co/models?library=tensorrt_llm
