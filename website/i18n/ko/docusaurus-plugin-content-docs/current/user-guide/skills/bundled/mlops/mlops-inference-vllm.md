---
title: "Serving Llms Vllm — vLLM: 고처리량(high-throughput) LLM 서빙, OpenAI API, 양자화"
sidebar_label: "Serving Llms Vllm"
description: "vLLM: 고처리량(high-throughput) LLM 서빙, OpenAI API, 양자화"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Serving Llms Vllm

vLLM: 고처리량(high-throughput) LLM 서빙, OpenAI API, 양자화.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 내장 (기본으로 설치됨) |
| 경로 | `skills/mlops/inference/vllm` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `vllm`, `torch`, `transformers` |
| 플랫폼 | linux, macos |
| 태그 | `vLLM`, `Inference Serving`, `PagedAttention`, `Continuous Batching`, `High Throughput`, `Production`, `OpenAI API`, `Quantization`, `Tensor Parallelism` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# vLLM - 고성능 LLM 서빙

## 사용 시기

프로덕션 LLM API를 배포하거나, 추론 대기 시간(latency)/처리량(throughput)을 최적화하거나, 제한된 GPU 메모리로 모델을 서빙해야 할 때 사용합니다. OpenAI 호환 엔드포인트, 양자화(GPTQ/AWQ/FP8) 및 텐서 병렬 처리를 지원합니다.

## 빠른 시작

vLLM은 PagedAttention (블록 기반 KV 캐시) 및 연속 배칭(Continuous batching - prefill/decode 요청 혼합)을 통해 표준 transformers보다 24배 더 높은 처리량을 달성합니다.

**설치**:
```bash
pip install vllm
```

**기본 오프라인 추론**:
```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B-Instruct")
sampling = SamplingParams(temperature=0.7, max_tokens=256)

outputs = llm.generate(["Explain quantum computing"], sampling)
print(outputs[0].outputs[0].text)
```

**OpenAI 호환 서버**:
```bash
vllm serve meta-llama/Llama-3-8B-Instruct

# OpenAI SDK로 쿼리
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')
print(client.chat.completions.create(
    model='meta-llama/Llama-3-8B-Instruct',
    messages=[{'role': 'user', 'content': 'Hello!'}]
).choices[0].message.content)
"
```

## 일반적인 워크플로우

### 워크플로우 1: 프로덕션 API 배포

다음 체크리스트를 복사하여 진행 상황을 추적하세요:

```
배포 진행 상황:
- [ ] 1단계: 서버 설정 구성
- [ ] 2단계: 제한된 트래픽으로 테스트
- [ ] 3단계: 모니터링 활성화
- [ ] 4단계: 프로덕션에 배포
- [ ] 5단계: 성능 지표 확인
```

**1단계: 서버 설정 구성**

모델 크기에 따라 구성을 선택하세요:

```bash
# 단일 GPU에서의 7B-13B 모델
vllm serve meta-llama/Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --port 8000

# 텐서 병렬 처리가 있는 30B-70B 모델
vllm serve meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --quantization awq \
  --port 8000

# 캐싱 및 지표가 있는 프로덕션용
vllm serve meta-llama/Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --enable-metrics \
  --metrics-port 9090 \
  --port 8000 \
  --host 0.0.0.0
```

**2단계: 제한된 트래픽으로 테스트**

프로덕션 전에 부하 테스트를 실행합니다:

```bash
# 부하 테스트 도구 설치
pip install locust

# 샘플 요청이 있는 test_load.py 생성
# 실행: locust -f test_load.py --host http://localhost:8000
```

TTFT (첫 번째 토큰까지 걸리는 시간, time to first token)가 500ms 미만이고 처리량이 초당 100 요청 이상인지 확인합니다.

**3단계: 모니터링 활성화**

vLLM은 포트 9090에서 Prometheus 지표를 노출합니다:

```bash
curl http://localhost:9090/metrics | grep vllm
```

모니터링해야 할 주요 지표:
- `vllm:time_to_first_token_seconds` - 대기 시간(Latency)
- `vllm:num_requests_running` - 활성 요청 수
- `vllm:gpu_cache_usage_perc` - KV 캐시 활용도

**4단계: 프로덕션에 배포**

일관된 배포를 위해 Docker를 사용합니다:

```bash
# Docker에서 vLLM 실행
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching
```

**5단계: 성능 지표 확인**

배포가 목표를 달성하는지 확인합니다:
- TTFT < 500ms (짧은 프롬프트의 경우)
- 처리량 > 목표 요청 수/초 (req/sec)
- GPU 활용도 > 80%
- 로그에 OOM 오류가 없음

### 워크플로우 2: 오프라인 배치 추론

서버 오버헤드 없이 대규모 데이터셋을 처리하기 위함입니다.

다음 체크리스트를 복사하세요:

```
배치 처리:
- [ ] 1단계: 입력 데이터 준비
- [ ] 2단계: LLM 엔진 구성
- [ ] 3단계: 배치 추론 실행
- [ ] 4단계: 결과 처리
```

**1단계: 입력 데이터 준비**

```python
# 파일에서 프롬프트 로드
prompts = []
with open("prompts.txt") as f:
    prompts = [line.strip() for line in f]

print(f"Loaded {len(prompts)} prompts")
```

**2단계: LLM 엔진 구성**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    tensor_parallel_size=2,  # 2개의 GPU 사용
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

vLLM은 효율성을 위해 자동으로 요청을 배치(batch) 처리합니다:

```python
# 한 번의 호출로 모든 프롬프트 처리
outputs = llm.generate(prompts, sampling)

# vLLM이 내부적으로 배치를 처리합니다
# 프롬프트를 수동으로 청크(chunk)로 나눌 필요가 없습니다
```

**4단계: 결과 처리**

```python
# 생성된 텍스트 추출
results = []
for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    results.append({
        "prompt": prompt,
        "generated": generated,
        "tokens": len(output.outputs[0].token_ids)
    })

# 파일에 저장
import json
with open("results.jsonl", "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

print(f"Processed {len(results)} prompts")
```

### 워크플로우 3: 양자화된 모델 서빙

제한된 GPU 메모리에 대형 모델을 맞추기 위함입니다.

```
양자화 설정:
- [ ] 1단계: 양자화 방법 선택
- [ ] 2단계: 양자화된 모델 찾기 또는 생성
- [ ] 3단계: 양자화 플래그와 함께 실행
- [ ] 4단계: 정확도 확인
```

**1단계: 양자화 방법 선택**

- **AWQ**: 70B 모델에 가장 적합, 최소한의 정확도 손실
- **GPTQ**: 폭넓은 모델 지원, 좋은 압축률
- **FP8**: H100 GPU에서 가장 빠름

**2단계: 양자화된 모델 찾기 또는 생성**

HuggingFace에서 사전 양자화된 모델을 사용합니다:

```bash
# AWQ 모델 검색
# 예: TheBloke/Llama-2-70B-AWQ
```

**3단계: 양자화 플래그와 함께 실행**

```bash
# 사전 양자화된 모델 사용
vllm serve TheBloke/Llama-2-70B-AWQ \
  --quantization awq \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95

# 결과: ~40GB VRAM에 70B 모델을 올림
```

**4단계: 정확도 확인**

출력된 결과물이 기대 품질과 일치하는지 테스트합니다:

```python
# 양자화된 것과 비양자화된 응답 비교
# 작업별 성능이 변하지 않았는지 확인
```

## 언제 사용하고 대안은 무엇인가

**다음과 같은 경우 vLLM을 사용하세요:**
- 프로덕션 LLM API를 배포할 때 (초당 100+ 요청)
- OpenAI 호환 엔드포인트를 서빙할 때
- GPU 메모리는 제한되어 있지만 큰 모델이 필요할 때
- 다중 사용자 애플리케이션 (챗봇, 어시스턴트)
- 높은 처리량과 함께 낮은 대기 시간이 필요할 때

**대신 다음 대안을 사용하는 것이 좋은 경우:**
- **llama.cpp**: CPU/엣지(edge) 추론, 단일 사용자
- **HuggingFace transformers**: 연구, 프로토타이핑, 일회성 생성
- **TensorRT-LLM**: NVIDIA 전용, 절대적인 최대 성능이 필요할 때
- **Text-Generation-Inference**: 이미 HuggingFace 생태계 내에 있을 때

## 일반적인 문제

**문제: 모델 로딩 중 메모리 부족 (OOM)**

메모리 사용량을 줄입니다:
```bash
vllm serve MODEL \
  --gpu-memory-utilization 0.7 \
  --max-model-len 4096
```

또는 양자화를 사용합니다:
```bash
vllm serve MODEL --quantization awq
```

**문제: 첫 번째 토큰이 느림 (TTFT > 1초)**

반복되는 프롬프트에 대해 프리픽스 캐싱(prefix caching)을 활성화합니다:
```bash
vllm serve MODEL --enable-prefix-caching
```

긴 프롬프트의 경우, 청크 단위 prefill(chunked prefill)을 활성화합니다:
```bash
vllm serve MODEL --enable-chunked-prefill
```

**문제: 모델을 찾을 수 없음 오류**

커스텀 모델의 경우 `--trust-remote-code`를 사용합니다:
```bash
vllm serve MODEL --trust-remote-code
```

**문제: 처리량이 낮음 (&lt;50 req/sec)**

동시 시퀀스 수를 늘립니다:
```bash
vllm serve MODEL --max-num-seqs 512
```

`nvidia-smi`로 GPU 활용도를 확인하세요 - >80% 이상이어야 합니다.

**문제: 추론이 예상보다 느림**

텐서 병렬 처리가 2의 거듭제곱 수의 GPU를 사용하는지 확인합니다:
```bash
vllm serve MODEL --tensor-parallel-size 4  # 3이 아님
```

더 빠른 생성을 위해 투기적 디코딩(speculative decoding)을 활성화합니다:
```bash
vllm serve MODEL --speculative-model DRAFT_MODEL
```

## 고급 주제

**서버 배포 패턴**: Docker, Kubernetes 및 로드 밸런싱 구성에 대해서는 [references/server-deployment.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/vllm/references/server-deployment.md)를 참조하세요.

**성능 최적화**: PagedAttention 튜닝, 연속 배칭 세부 정보 및 벤치마크 결과에 대해서는 [references/optimization.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/vllm/references/optimization.md)를 참조하세요.

**양자화 가이드**: AWQ/GPTQ/FP8 설정, 모델 준비 및 정확도 비교에 대해서는 [references/quantization.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/vllm/references/quantization.md)를 참조하세요.

**문제 해결**: 자세한 오류 메시지, 디버깅 단계 및 성능 진단에 대해서는 [references/troubleshooting.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/vllm/references/troubleshooting.md)를 참조하세요.

## 하드웨어 요구 사항

- **작은 모델 (7B-13B)**: 1x A10 (24GB) 또는 A100 (40GB)
- **중간 모델 (30B-40B)**: 텐서 병렬 처리가 있는 2x A100 (40GB)
- **큰 모델 (70B+)**: 4x A100 (40GB) 또는 2x A100 (80GB), AWQ/GPTQ 사용 권장

지원 플랫폼: NVIDIA (기본), AMD ROCm, Intel GPU, TPU

## 리소스

- 공식 문서: https://docs.vllm.ai
- GitHub: https://github.com/vllm-project/vllm
- 논문: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
- 커뮤니티: https://discuss.vllm.ai
