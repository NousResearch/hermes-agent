---
title: "Modal Serverless Gpu — ML 워크로드를 실행하기 위한 서버리스 GPU 클라우드 플랫폼"
sidebar_label: "Modal Serverless Gpu"
description: "ML 워크로드를 실행하기 위한 서버리스 GPU 클라우드 플랫폼"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Modal Serverless Gpu

ML 워크로드를 실행하기 위한 서버리스 GPU 클라우드 플랫폼입니다. 인프라 관리 없이 온디맨드 GPU 접근이 필요하거나, ML 모델을 API로 배포하거나, 자동 확장 기능이 있는 배치(일괄 처리) 작업을 실행할 때 사용하세요.

## Skill metadata

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/modal`로 설치 |
| Path | `optional-skills/mlops/modal` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `modal>=0.64.0` |
| Platforms | linux, macos, windows |
| Tags | `Infrastructure`, `Serverless`, `GPU`, `Cloud`, `Deployment`, `Modal` |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Modal 서버리스 GPU

Modal의 서버리스 GPU 클라우드 플랫폼에서 ML 워크로드를 실행하기 위한 포괄적인 가이드입니다.

## Modal을 사용해야 하는 경우

**사용 시기:**
- 인프라를 관리하지 않고 GPU 집약적인 ML 워크로드를 실행할 때
- 자동 확장되는 API 형태로 ML 모델을 배포할 때
- 배치 처리 작업(학습, 추론, 데이터 처리)을 실행할 때
- 유휴 비용 없이 초당 과금되는 GPU가 필요할 때
- ML 애플리케이션을 빠르게 프로토타이핑할 때
- 예약된 작업(cron과 유사한 워크로드)을 실행할 때

**주요 기능:**
- **서버리스 GPU**: T4, L4, A10G, L40S, A100, H100, H200, B200 온디맨드 지원
- **Python 네이티브**: YAML 없이 Python 코드로 인프라 정의
- **자동 확장(Auto-scaling)**: 0개로 축소하거나 즉시 100개 이상의 GPU로 확장
- **1초 미만의 콜드 스타트**: 빠른 컨테이너 실행을 위한 Rust 기반 인프라
- **컨테이너 캐싱**: 빠른 반복 작업을 위해 캐시되는 이미지 레이어
- **웹 엔드포인트**: 무중단 업데이트를 통해 기능을 REST API로 배포

**대신 사용할 수 있는 대안:**
- **RunPod**: 상태가 유지되는 장기 실행 파드용
- **Lambda Labs**: 예약된 GPU 인스턴스용
- **SkyPilot**: 멀티 클라우드 오케스트레이션 및 비용 최적화용
- **Kubernetes**: 복잡한 다중 서비스 아키텍처용

## 빠른 시작

### 설치

```bash
pip install modal
modal setup  # 인증을 위해 브라우저를 엽니다.
```

### GPU를 사용한 Hello World

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

| 구성 요소 | 목적 |
|-----------|---------|
| `App` | 함수와 리소스를 담는 컨테이너 |
| `Function` | 컴퓨팅 사양이 포함된 서버리스 함수 |
| `Cls` | 라이프사이클 훅이 포함된 클래스 기반 함수 |
| `Image` | 컨테이너 이미지 정의 |
| `Volume` | 모델/데이터를 위한 영구 스토리지 |
| `Secret` | 안전한 자격 증명(credential) 저장소 |

### 실행 모드

| 명령어 | 설명 |
|---------|-------------|
| `modal run script.py` | 실행 후 종료 |
| `modal serve script.py` | 라이브 리로드가 적용된 개발 모드 |
| `modal deploy script.py` | 영구적인 클라우드 배포 |

## GPU 구성

### 사용 가능한 GPU

| GPU | VRAM | 최적 용도 |
|-----|------|----------|
| `T4` | 16GB | 예산 친화적인 추론, 소형 모델 |
| `L4` | 24GB | 추론, Ada Lovelace 아키텍처 |
| `A10G` | 24GB | 학습/추론, T4 대비 3.3배 빠름 |
| `L40S` | 48GB | 추론 권장 (비용 대비 성능 최고) |
| `A100-40GB` | 40GB | 대형 모델 학습 |
| `A100-80GB` | 80GB | 초대형 모델 학습 |
| `H100` | 80GB | 가장 빠름, FP8 + 트랜스포머 엔진 |
| `H200` | 141GB | H100에서 자동 업그레이드, 4.8TB/s 대역폭 |
| `B200` | 최신 | Blackwell 아키텍처 |

### GPU 지정 패턴

```python
# 단일 GPU
@app.function(gpu="A100")

# 특정 메모리 변형 모델
@app.function(gpu="A100-80GB")

# 다중 GPU (최대 8개)
@app.function(gpu="H100:4")

# 폴백(fallback)을 포함한 GPU
@app.function(gpu=["H100", "A100", "L40S"])

# 사용 가능한 아무 GPU
@app.function(gpu="any")
```

## 컨테이너 이미지

```python
# pip를 포함한 기본 이미지
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.1.0", "transformers==4.36.0", "accelerate"
)

# CUDA 베이스를 활용한 이미지
image = modal.Image.from_registry(
    "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
    add_python="3.11"
).pip_install("torch", "transformers")

# 시스템 패키지가 포함된 이미지
image = modal.Image.debian_slim().apt_install("git", "ffmpeg").pip_install("whisper")
```

## 영구 스토리지 (Persistent storage)

```python
volume = modal.Volume.from_name("model-cache", create_if_missing=True)

@app.function(gpu="A10G", volumes={"/models": volume})
def load_model():
    import os
    model_path = "/models/llama-7b"
    if not os.path.exists(model_path):
        model = download_model()
        model.save_pretrained(model_path)
        volume.commit()  # 변경 사항 영구 저장
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

## 동적 일괄 처리 (Dynamic batching)

```python
@app.function()
@modal.batched(max_batch_size=32, wait_ms=100)
async def batch_predict(inputs: list[str]) -> list[dict]:
    # 입력이 자동으로 일괄 처리됨
    return model.batch_predict(inputs)
```

## Secret 관리

```bash
# Secret 생성
modal secret create huggingface HF_TOKEN=hf_xxx
```

```python
@app.function(secrets=[modal.Secret.from_name("huggingface")])
def download_model():
    import os
    token = os.environ["HF_TOKEN"]
```

## 스케줄링

```python
@app.function(schedule=modal.Cron("0 0 * * *"))  # 매일 자정
def daily_job():
    pass

@app.function(schedule=modal.Period(hours=1))
def hourly_job():
    pass
```

## 성능 최적화

### 콜드 스타트 완화

```python
@app.function(
    container_idle_timeout=300,  # 5분 동안 웜(warm) 상태 유지
    allow_concurrent_inputs=10,  # 동시 요청 처리 허용
)
def inference():
    pass
```

### 모델 로딩 모범 사례

```python
@app.cls(gpu="A100")
class Model:
    @modal.enter()  # 컨테이너 시작 시 한 번 실행
    def load(self):
        self.model = load_model()  # 웜업 중 로드

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
    # 병렬 컨테이너로 분산(Fan out)
    results = list(process_item.map(items))
    return results
```

## 일반적인 설정

```python
@app.function(
    gpu="A100",
    memory=32768,              # 32GB RAM
    cpu=4,                     # 4 CPU 코어
    timeout=3600,              # 최대 1시간
    container_idle_timeout=120,# 2분 동안 웜 상태 유지
    retries=3,                 # 실패 시 재시도
    concurrency_limit=10,      # 최대 동시 컨테이너 수
)
def my_function():
    pass
```

## 디버깅

```python
# 로컬에서 테스트
if __name__ == "__main__":
    result = my_function.local()

# 로그 보기
# modal app logs my-app
```

## 일반적인 문제

| 문제 | 해결 방법 |
|-------|----------|
| 콜드 스타트 지연 | `container_idle_timeout`을 늘리고, `@modal.enter()`를 사용 |
| GPU OOM | 더 큰 GPU(`A100-80GB`)를 사용하거나, 그래디언트 체크포인팅 활성화 |
| 이미지 빌드 실패 | 의존성 버전을 고정(pin)하고, CUDA 호환성 확인 |
| 시간 초과(Timeout) 에러 | `timeout` 값을 늘리거나 체크포인팅 추가 |

## 참고 자료 (References)

- **[고급 사용법 (Advanced Usage)](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/modal/references/advanced-usage.md)** - 멀티 GPU, 분산 학습, 비용 최적화
- **[문제 해결 (Troubleshooting)](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/modal/references/troubleshooting.md)** - 일반적인 문제 및 해결책

## 리소스

- **공식 문서**: https://modal.com/docs
- **예제**: https://github.com/modal-labs/modal-examples
- **가격**: https://modal.com/pricing
- **Discord**: https://discord.gg/modal
