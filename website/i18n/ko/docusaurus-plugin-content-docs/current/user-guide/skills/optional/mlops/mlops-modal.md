---
title: "Modal — ML 작업과 모델 API를 위한 서버리스 GPU 클라우드"
sidebar_label: "Modal"
description: "ML 작업과 모델 API를 위한 서버리스 GPU 클라우드"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Modal

ML 작업과 모델 API를 위한 서버리스 GPU 클라우드입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/modal`로 설치 |
| 경로 | `optional-skills/mlops/modal` |
| 버전 | `1.0.1` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `modal>=1.0` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Infrastructure`, `Serverless`, `GPU`, `Cloud`, `Deployment`, `Modal` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 실행될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# Modal 서버리스 GPU

Modal의 서버리스 GPU 클라우드 플랫폼에서 ML 워크로드를 실행하는 방법을 안내합니다.

## Modal을 사용할 때

**다음과 같은 경우 Modal을 사용합니다.**
- 인프라를 직접 관리하지 않고 GPU 집약적인 ML 워크로드를 실행해야 하는 경우
- 자동 확장되는 API로 ML 모델을 배포하는 경우
- 배치 처리 작업(학습, 추론, 데이터 처리)을 실행하는 경우
- 유휴 비용 없이 초 단위 GPU 사용량에 따라 지불해야 하는 경우
- ML 애플리케이션을 빠르게 프로토타이핑하는 경우
- 예약 작업(cron과 유사한 워크로드)을 실행하는 경우

**주요 기능:**
- **서버리스 GPU**: 필요할 때 T4, L4, A10G, L40S, A100, H100, H200, B200 제공
- **Python 네이티브**: YAML 없이 Python 코드로 인프라 정의
- **자동 확장**: 즉시 0개에서 100개 이상의 GPU까지 확장
- **1초 미만의 콜드 스타트**: 빠른 컨테이너 시작을 위한 Rust 기반 인프라
- **컨테이너 캐싱**: 빠른 반복 작업을 위해 이미지 레이어 캐시
- **웹 엔드포인트**: 다운타임 없이 함수를 REST API로 배포

**대신 다음 대안을 사용합니다.**
- **RunPod**: 영속 상태가 필요한 장시간 실행 pod
- **Lambda Labs**: 예약 GPU 인스턴스
- **SkyPilot**: 멀티 클라우드 오케스트레이션 및 비용 최적화
- **Kubernetes**: 복잡한 멀티 서비스 아키텍처

## 빠른 시작

### 설치

```bash
pip install modal
modal setup  # Opens browser for authentication
```

### GPU Hello World

```python
import modal

app = modal.App("hello-gpu")

@app.function(gpu="T4")
def gpu_info():
    import subprocess
    return subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout

@app.local_entrypoint()
def main():
    print(gpu_info.remote())
```

실행: `modal run hello_gpu.py`

### 기본 추론 엔드포인트

```python
import modal

app = modal.App("text-generation")
image = modal.Image.debian_slim().pip_install("transformers", "torch", "accelerate")

@app.cls(gpu="A10G", image=image)
class TextGenerator:
    @modal.enter()
    def load_model(self):
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model="gpt2", device=0)

    @modal.method()
    def generate(self, prompt: str) -> str:
        return self.pipe(prompt, max_length=100)[0]["generated_text"]

@app.local_entrypoint()
def main():
    print(TextGenerator().generate.remote("Hello, world"))
```

## 핵심 개념

### 주요 구성 요소

| 구성 요소 | 용도 |
|-----------|---------|
| `App` | 함수와 리소스를 담는 컨테이너 |
| `Function` | 컴퓨팅 사양을 갖춘 서버리스 함수 |
| `Cls` | 수명 주기 훅을 사용하는 클래스 기반 함수 |
| `Image` | 컨테이너 이미지 정의 |
| `Volume` | 모델/데이터용 영속 스토리지 |
| `Secret` | 보안 자격 증명 저장소 |

### 실행 모드

| 명령 | 설명 |
|---------|-------------|
| `modal run script.py` | 실행 후 종료 |
| `modal serve script.py` | 라이브 리로드를 사용한 개발 |
| `modal deploy script.py` | 영속적인 클라우드 배포 |

## GPU 구성

### 사용 가능한 GPU

| GPU | VRAM | 적합한 용도 |
|-----|------|----------|
| `T4` | 16GB | 예산 중심 추론, 소형 모델 |
| `L4` | 24GB | 추론, Ada Lovelace 아키텍처 |
| `A10G` | 24GB | 학습/추론, T4보다 3.3배 빠름 |
| `L40S` | 48GB | 추론에 권장 (최고의 비용 대비 성능) |
| `A100-40GB` | 40GB | 대형 모델 학습 |
| `A100-80GB` | 80GB | 초대형 모델 |
| `H100` | 80GB | 가장 빠름, FP8 + Transformer Engine |
| `H200` | 141GB | H100에서 자동 업그레이드, 4.8TB/s 대역폭 |
| `B200` | 최신 | Blackwell 아키텍처 |

### GPU 사양 패턴

```python
# Single GPU
@app.function(gpu="A100")

# Specific memory variant
@app.function(gpu="A100-80GB")

# Multiple GPUs (up to 8)
@app.function(gpu="H100:4")

# GPU with fallbacks
@app.function(gpu=["H100", "A100", "L40S"])

# Any available GPU
@app.function(gpu="any")
```

## 컨테이너 이미지

```python
# Basic image with pip
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.1.0", "transformers==4.36.0", "accelerate"
)

# From CUDA base
image = modal.Image.from_registry(
    "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
    add_python="3.11"
).pip_install("torch", "transformers")

# With system packages
image = modal.Image.debian_slim().apt_install("git", "ffmpeg").pip_install("whisper")
```

## 영속 스토리지

```python
volume = modal.Volume.from_name("model-cache", create_if_missing=True)

@app.function(gpu="A10G", volumes={"/models": volume})
def load_model():
    import os
    model_path = "/models/llama-7b"
    if not os.path.exists(model_path):
        model = download_model()
        model.save_pretrained(model_path)
        volume.commit()  # Persist changes
    return load_from_path(model_path)
```

## 웹 엔드포인트

### FastAPI 엔드포인트 데코레이터

```python
@app.function()
@modal.fastapi_endpoint(method="POST")
def predict(text: str) -> dict:
    return {"result": model.predict(text)}
```

### 전체 ASGI 앱

```python
from fastapi import FastAPI
web_app = FastAPI()

@web_app.post("/predict")
async def predict(text: str):
    return {"result": await model.predict.remote.aio(text)}

@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
```

### 웹 엔드포인트 유형

| 데코레이터 | 사용 사례 |
|-----------|----------|
| `@modal.fastapi_endpoint()` | 단순 함수 → API |
| `@modal.asgi_app()` | 전체 FastAPI/Starlette 앱 |
| `@modal.wsgi_app()` | Django/Flask 앱 |
| `@modal.web_server(port)` | 임의의 HTTP 서버 |

## 동적 배치 처리

```python
@app.function()
@modal.batched(max_batch_size=32, wait_ms=100)
async def batch_predict(inputs: list[str]) -> list[dict]:
    # Inputs automatically batched
    return model.batch_predict(inputs)
```
## 시크릿 관리

```bash
# Create secret
modal secret create huggingface HF_TOKEN=hf_xxx
```

```python
@app.function(secrets=[modal.Secret.from_name("huggingface")])
def download_model():
    import os
    token = os.environ["HF_TOKEN"]
```

## 예약

```python
@app.function(schedule=modal.Cron("0 0 * * *"))  # Daily midnight
def daily_job():
    pass

@app.function(schedule=modal.Period(hours=1))
def hourly_job():
    pass
```

## 성능 최적화

### 콜드 스타트 완화

```python
# Modal 1.0 autoscaler params: scaledown_window (was container_idle_timeout).
# Input concurrency moved to the @modal.concurrent decorator.
@app.function(scaledown_window=300)  # Keep warm 5 min
@modal.concurrent(max_inputs=10)     # Handle concurrent requests per container
def inference():
    pass
```

### 모델 로딩 모범 사례

```python
@app.cls(gpu="A100")
class Model:
    @modal.enter()  # Run once at container start
    def load(self):
        self.model = load_model()  # Load during warm-up

    @modal.method()
    def predict(self, x):
        return self.model(x)
```

## 병렬 처리

```python
@app.function()
def process_item(item):
    return expensive_computation(item)

@app.function()
def run_parallel():
    items = list(range(1000))
    # Fan out to parallel containers
    results = list(process_item.map(items))
    return results
```

## 일반 구성

```python
@app.function(
    gpu="A100",
    memory=32768,              # 32GB RAM
    cpu=4,                     # 4 CPU cores
    timeout=3600,              # 1 hour max
    scaledown_window=120,      # Keep warm 2 min (was container_idle_timeout)
    retries=3,                 # Retry on failure
    max_containers=10,         # Max concurrent containers (was concurrency_limit)
    min_containers=1,          # Keep N containers warm (was keep_warm)
)
def my_function():
    pass
```

> **Modal 1.0 자동 확장기 이름 변경** ([마이그레이션 가이드](https://modal.com/docs/guide/modal-1-0-migration) 참고):
> - `container_idle_timeout` → `scaledown_window`
> - `concurrency_limit` → `max_containers`
> - `keep_warm` → `min_containers`
> - `allow_concurrent_inputs=N` → `@modal.concurrent(max_inputs=N)` 데코레이터

## 디버깅

```python
# Test locally
if __name__ == "__main__":
    result = my_function.local()

# View logs
# modal app logs my-app
```

## 일반적인 문제

| 문제 | 해결 방법 |
|-------|----------|
| 콜드 스타트 지연 | `scaledown_window`을 늘리고 `@modal.enter()` 사용 |
| GPU OOM | 더 큰 GPU(`A100-80GB`) 사용, 그래디언트 체크포인팅 활성화 |
| 이미지 빌드 실패 | 의존성 버전을 고정하고 CUDA 호환성 확인 |
| 타임아웃 오류 | `timeout`을 늘리고 체크포인팅 추가 |

## 참고 자료

- **[고급 사용법](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/modal/references/advanced-usage.md)** - 멀티 GPU, 분산 학습, 비용 최적화
- **[문제 해결](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/modal/references/troubleshooting.md)** - 일반적인 문제와 해결 방법

## 리소스

- **문서**: https://modal.com/docs
- **예제**: https://github.com/modal-labs/modal-examples
- **가격**: https://modal.com/pricing
- **Discord**: https://discord.gg/modal
