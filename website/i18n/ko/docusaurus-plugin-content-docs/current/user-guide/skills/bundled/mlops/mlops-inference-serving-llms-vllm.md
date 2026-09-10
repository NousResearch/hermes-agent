---
title: "LLM 제공 vLLM — vLLM: 고처리량 LLM 제공, OpenAI API, 양자화"
sidebar_label: "LLM 제공 vLLM"
description: "vLLM: 고처리량 LLM 제공, OpenAI API, 양자화"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# LLM 제공 vLLM

vLLM: 고처리량 LLM 제공, OpenAI API, 양자화.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 번들 포함(기본 설치) |
| 경로 | `skills/mlops/inference/serving-llms-vllm` |
| 버전 | `1.0.1` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `vllm`, `torch`, `transformers` |
| 플랫폼 | linux, macos |
| 태그 | `vLLM`, `Inference Serving`, `PagedAttention`, `Continuous Batching`, `High Throughput`, `Production`, `OpenAI API`, `Quantization`, `Tensor Parallelism` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# vLLM - 고성능 LLM 제공

## 사용 시점

운영 환경의 LLM API를 배포하거나, 추론 지연 시간/처리량을 최적화하거나, GPU 메모리가 제한된 환경에서 모델을 제공할 때 사용합니다. OpenAI 호환 엔드포인트, 양자화(GPTQ/AWQ/FP8), 텐서 병렬 처리를 지원합니다.

## 빠른 시작

vLLM은 PagedAttention(블록 기반 KV 캐시)과 연속 배칭(prefill/decode 요청 혼합)을 통해 표준 transformers보다 24배 높은 처리량을 달성합니다.

**설치**:
```bash
pip install vllm
```

**기본 오프라인 추론**:
```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
sampling = SamplingParams(temperature=0.7, max_tokens=256)

outputs = llm.generate(["Explain quantum computing"], sampling)
print(outputs[0].outputs[0].text)
```

**OpenAI 호환 서버**:
```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct

# Query with OpenAI SDK
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')
print(client.chat.completions.create(
    model='meta-llama/Meta-Llama-3-8B-Instruct',
    messages=[{'role': 'user', 'content': 'Hello!'}]
).choices[0].message.content)
"
```

## 일반적인 작업 흐름

### 작업 흐름 1: 운영 환경 API 배포

이 체크리스트를 복사하고 진행 상황을 추적하세요.

```
Deployment Progress:
- [ ] Step 1: Configure server settings
- [ ] Step 2: Test with limited traffic
- [ ] Step 3: Enable monitoring
- [ ] Step 4: Deploy to production
- [ ] Step 5: Verify performance metrics
```

**1단계: 서버 설정 구성**

모델 크기에 따라 설정을 선택하세요.

```bash
# For 7B-13B models on single GPU
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --port 8000

# For 30B-70B models with tensor parallelism
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --quantization awq \
  --port 8000

# For production with caching (Prometheus metrics are exposed
# automatically at /metrics on the API port)
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --port 8000 \
  --host 0.0.0.0
```

**2단계: 제한된 트래픽으로 테스트**

운영 환경에 배포하기 전에 부하 테스트를 실행하세요.

```bash
# Install load testing tool
pip install locust

# Create test_load.py with sample requests
# Run: locust -f test_load.py --host http://localhost:8000
```

TTFT(첫 토큰까지의 시간)가 &lt; 500ms이고 처리량이 > 100 req/sec인지 확인하세요.

**3단계: 모니터링 활성화**

vLLM은 API 포트(기본값 8000)의 `/metrics`에서 Prometheus 메트릭을 제공합니다.

```bash
curl http://localhost:8000/metrics | grep vllm
```

모니터링할 주요 메트릭:
- `vllm:time_to_first_token_seconds` - 지연 시간
- `vllm:num_requests_running` - 활성 요청
- `vllm:gpu_cache_usage_perc` - KV 캐시 사용률

**4단계: 운영 환경에 배포**

일관된 배포를 위해 Docker를 사용하세요.

```bash
# Run vLLM in Docker
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching
```

**5단계: 성능 메트릭 확인**

배포가 목표를 충족하는지 확인하세요.
- 짧은 프롬프트의 TTFT &lt; 500ms
- 처리량 > 목표 req/sec
- GPU 사용률 > 80%
- 로그에 OOM 오류 없음

### 작업 흐름 2: 오프라인 배치 추론

서버 오버헤드 없이 대규모 데이터셋을 처리할 때 사용합니다.

이 체크리스트를 복사하세요.

```
Batch Processing:
- [ ] Step 1: Prepare input data
- [ ] Step 2: Configure LLM engine
- [ ] Step 3: Run batch inference
- [ ] Step 4: Process results
```

**1단계: 입력 데이터 준비**

```python
# Load prompts from file
prompts = []
with open("prompts.txt") as f:
    prompts = [line.strip() for line in f]

print(f"Loaded {len(prompts)} prompts")
```

**2단계: LLM 엔진 구성**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=2,  # Use 2 GPUs
    gpu_memory_utilization=0.9,
    max_model_len=4096
)

sampling = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
    stop=["</s>", "\n\n"]
)
```

**3단계: 배치 추론 실행**

vLLM은 효율성을 위해 요청을 자동으로 배치 처리합니다.

```python
# Process all prompts in one call
outputs = llm.generate(prompts, sampling)

# vLLM handles batching internally
# No need to manually chunk prompts
```

**4단계: 결과 처리**

```python
# Extract generated text
results = []
for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    results.append({
        "prompt": prompt,
        "generated": generated,
        "tokens": len(output.outputs[0].token_ids)
    })

# Save to file
import json
with open("results.jsonl", "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

print(f"Processed {len(results)} prompts")
```

### 작업 흐름 3: 양자화된 모델 제공

제한된 GPU 메모리에 대규모 모델을 맞출 때 사용합니다.

```
Quantization Setup:
- [ ] Step 1: Choose quantization method
- [ ] Step 2: Find or create quantized model
- [ ] Step 3: Launch with quantization flag
- [ ] Step 4: Verify accuracy
```

**1단계: 양자화 방법 선택**

- **AWQ**: 정확도 손실이 최소이며 70B 모델에 가장 적합
- **GPTQ**: 폭넓은 모델 지원, 뛰어난 압축
- **FP8**: H100 GPU에서 가장 빠름

**2단계: 양자화된 모델 찾기 또는 생성**

HuggingFace에서 사전 양자화된 모델을 사용하세요.

```bash
# Search for AWQ models
# Example: TheBloke/Llama-2-70B-AWQ
```

**3단계: 양자화 플래그로 실행**

```bash
# Using pre-quantized model
vllm serve TheBloke/Llama-2-70B-AWQ \
  --quantization awq \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95

# Results: 70B model in ~40GB VRAM
```

**4단계: 정확도 확인**

출력 품질이 예상과 일치하는지 테스트하세요.

```python
# Compare quantized vs non-quantized responses
# Verify task-specific performance unchanged
```

## 대안과 비교한 사용 시점

**다음과 같은 경우 vLLM을 사용하세요:**
- 운영 환경의 LLM API 배포(100+ req/sec)
- OpenAI 호환 엔드포인트 제공
- GPU 메모리는 제한되어 있지만 대규모 모델이 필요한 경우
- 다중 사용자 애플리케이션(챗봇, 어시스턴트)
- 높은 처리량과 함께 낮은 지연 시간이 필요한 경우

**다음과 같은 경우에는 대안을 사용하세요:**
- **llama.cpp**: CPU/엣지 추론, 단일 사용자
- **HuggingFace transformers**: 연구, 프로토타이핑, 일회성 생성
- **TensorRT-LLM**: NVIDIA 전용이며 절대적인 최고 성능이 필요한 경우
- **Text-Generation-Inference**: 이미 HuggingFace 생태계에 속한 경우

## 일반적인 문제

**문제: 모델 로딩 중 메모리 부족**

메모리 사용량을 줄이세요.
```bash
vllm serve MODEL \
  --gpu-memory-utilization 0.7 \
  --max-model-len 4096
```

또는 양자화를 사용하세요.
```bash
vllm serve MODEL --quantization awq
```

**문제: 첫 토큰이 느림(TTFT > 1초)**

반복되는 프롬프트에 프리픽스 캐싱을 활성화하세요.
```bash
vllm serve MODEL --enable-prefix-caching
```

긴 프롬프트에는 청크 분할 프리필을 활성화하세요.
```bash
vllm serve MODEL --enable-chunked-prefill
```

**문제: 모델을 찾을 수 없음 오류**

사용자 지정 모델에는 `--trust-remote-code`를 사용하세요.
```bash
vllm serve MODEL --trust-remote-code
```

**문제: 처리량이 낮음(&lt;50 req/sec)**

동시 시퀀스 수를 늘리세요.
```bash
vllm serve MODEL --max-num-seqs 512
```

`nvidia-smi`로 GPU 사용률을 확인하세요. 80%보다 높아야 합니다.

**문제: 추론이 예상보다 느림**

텐서 병렬 처리가 2의 거듭제곱인 GPU 수를 사용하는지 확인하세요.
```bash
vllm serve MODEL --tensor-parallel-size 4  # Not 3
```

더 빠른 생성을 위해 추측 디코딩을 활성화하세요(JSON으로 설정 전달).
`--speculative-model`은 제거되었으며 `--speculative-config`로 대체되었습니다.
```bash
vllm serve MODEL \
  --speculative-config '{"model": "DRAFT_MODEL", "num_speculative_tokens": 5, "method": "draft_model"}'
```

## 고급 주제

**서버 배포 패턴**: Docker, Kubernetes, 로드 밸런싱 설정은 [references/server-deployment.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/serving-llms-vllm/references/server-deployment.md)를 참조하세요.

**성능 최적화**: PagedAttention 튜닝, 연속 배칭 세부 정보, 벤치마크 결과는 [references/optimization.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/serving-llms-vllm/references/optimization.md)를 참조하세요.

**양자화 가이드**: AWQ/GPTQ/FP8 설정, 모델 준비, 정확도 비교는 [references/quantization.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/serving-llms-vllm/references/quantization.md)를 참조하세요.

**문제 해결**: 자세한 오류 메시지, 디버깅 단계, 성능 진단은 [references/troubleshooting.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/serving-llms-vllm/references/troubleshooting.md)를 참조하세요.

## 하드웨어 요구 사항

- **소형 모델(7B-13B)**: 1x A10(24GB) 또는 A100(40GB)
- **중형 모델(30B-40B)**: 텐서 병렬 처리를 사용하는 2x A100(40GB)
- **대형 모델(70B+)**: 4x A100(40GB) 또는 2x A100(80GB), AWQ/GPTQ 사용

지원 플랫폼: NVIDIA(주요), AMD ROCm, Intel GPU, TPU

## 리소스

- 공식 문서: https://docs.vllm.ai
- GitHub: https://github.com/vllm-project/vllm
- 논문: "PagedAttention를 활용한 대규모 언어 모델 제공을 위한 효율적인 메모리 관리"(SOSP 2023)
- 커뮤니티: https://discuss.vllm.ai
