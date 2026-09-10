---
sidebar_position: 2
title: "Mac에서 로컬 LLM 실행하기"
description: "모델 선택, 메모리 최적화, Apple Silicon의 실제 벤치마크를 포함해 llama.cpp 또는 MLX로 macOS에서 OpenAI 호환 로컬 LLM 서버를 설정합니다"
---

# Mac에서 로컬 LLM 실행하기

이 가이드에서는 OpenAI 호환 API를 사용해 macOS에서 로컬 LLM 서버를 실행하는 방법을 안내합니다. 완전한 개인정보 보호, API 비용 제로, 그리고 Apple Silicon에서 기대 이상으로 뛰어난 성능을 얻을 수 있습니다.

두 가지 백엔드를 다룹니다.

| 백엔드 | 설치 | 가장 뛰어난 점 | 형식 |
|---------|---------|---------|--------|
| **llama.cpp** | `brew install llama.cpp` | 가장 빠른 첫 토큰까지의 시간, 메모리가 부족할 때 양자화된 KV 캐시 | GGUF |
| **omlx** | [omlx.ai](https://omlx.ai) | 가장 빠른 토큰 생성, 네이티브 Metal 최적화 | MLX (safetensors) |

두 백엔드 모두 OpenAI 호환 `/v1/chat/completions` 엔드포인트를 제공합니다. Hermes는 둘 중 어느 쪽과도 작동하므로 `http://localhost:8080` 또는 `http://localhost:8000`을 가리키기만 하면 됩니다.

:::info Apple Silicon 전용
이 가이드는 Apple Silicon(M1 이상)이 탑재된 Mac을 대상으로 합니다. Intel Mac에서는 llama.cpp가 작동하지만 GPU 가속을 사용할 수 없으므로 성능이 크게 느려질 수 있습니다.
:::

---

## 모델 선택

처음 시작한다면 **Qwen3.5-9B**를 권장합니다. 강력한 추론 모델이며 양자화를 사용하면 통합 메모리 8GB 이상에서 여유 있게 실행할 수 있습니다.

| 변형 | 디스크 사용량 | 필요한 RAM(128K 컨텍스트) | 백엔드 |
|---------|-------------|---------------------------|---------|
| Qwen3.5-9B-Q4_K_M (GGUF) | 5.3 GB | 양자화된 KV 캐시 사용 시 약 10 GB | llama.cpp |
| Qwen3.5-9B-mlx-lm-mxfp4 (MLX) | 약 5 GB | 약 12 GB | omlx |

**메모리 경험칙:** 모델 크기 + KV 캐시입니다. 9B Q4 모델은 약 5GB입니다. 128K 컨텍스트에서 Q4 양자화를 사용하면 KV 캐시에 약 4~5GB가 추가됩니다. 기본값(f16) KV 캐시를 사용하면 약 16GB까지 늘어납니다. llama.cpp의 양자화된 KV 캐시 플래그가 메모리가 제한된 시스템에서 핵심적인 비결입니다.

더 큰 모델(27B, 35B)을 사용하려면 통합 메모리 32GB 이상이 필요합니다. 9B는 8~16GB 시스템에 가장 적합한 크기입니다.

---

## 옵션 A: llama.cpp

llama.cpp는 가장 이식성이 높은 로컬 LLM 런타임입니다. macOS에서는 기본적으로 Metal을 사용해 GPU를 가속합니다.

### 설치

```bash
brew install llama.cpp
```

이 명령으로 `llama-server` 명령을 전역에서 사용할 수 있습니다.

### 모델 다운로드

GGUF 형식의 모델이 필요합니다. 가장 간단한 출처는 `huggingface-cli`를 통한 Hugging Face입니다.

```bash
brew install huggingface-cli
```

그런 다음 다운로드합니다.

```bash
huggingface-cli download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir ~/models
```

:::tip 접근이 제한된 모델
Hugging Face의 일부 모델은 인증이 필요합니다. 401 또는 404 오류가 발생하면 먼저 `huggingface-cli login`을 실행하세요.
:::

### 서버 시작

```bash
llama-server -m ~/models/Qwen3.5-9B-Q4_K_M.gguf \
  -ngl 99 \
  -c 131072 \
  -np 1 \
  -fa on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --host 0.0.0.0
```

각 플래그의 역할은 다음과 같습니다.

| 플래그 | 용도 |
|------|---------|
| `-ngl 99` | 모든 레이어를 GPU(Metal)로 오프로드합니다. CPU에 남는 것이 없도록 높은 값을 사용하세요. |
| `-c 131072` | 컨텍스트 창 크기(128K 토큰)입니다. 메모리가 부족하면 줄이세요. |
| `-np 1` | 병렬 슬롯 수입니다. 단일 사용자라면 1로 유지하세요. 슬롯을 늘리면 메모리 예산이 분할됩니다. |
| `-fa on` | 플래시 어텐션입니다. 메모리 사용량을 줄이고 긴 컨텍스트 추론을 빠르게 합니다. |
| `--cache-type-k q4_0` | 키 캐시를 4비트로 양자화합니다. **메모리를 크게 절약하는 핵심 설정입니다.** |
| `--cache-type-v q4_0` | 값 캐시를 4비트로 양자화합니다. 위 설정과 함께 사용하면 f16 대비 KV 캐시 메모리를 약 75% 줄입니다. |
| `--host 0.0.0.0` | 모든 인터페이스에서 수신합니다. 네트워크 접근이 필요 없다면 `127.0.0.1`을 사용하세요. |

다음 메시지가 표시되면 서버가 준비된 것입니다.

```
main: server is listening on http://0.0.0.0:8080
srv  update_slots: all slots are idle
```

### 제한된 시스템의 메모리 최적화

`--cache-type-k q4_0 --cache-type-v q4_0` 플래그는 메모리가 제한된 시스템에서 가장 중요한 최적화입니다. 128K 컨텍스트에서의 영향은 다음과 같습니다.

| KV 캐시 유형 | KV 캐시 메모리(128K 컨텍스트, 9B 모델) |
|---------------|--------------------------------------|
| f16 (기본값) | 약 16 GB |
| q8_0 | 약 8 GB |
| **q4_0** | **약 4 GB** |

8GB Mac에서는 `q4_0` KV 캐시를 사용하고 Hermes의 64K 최소 컨텍스트에 맞으면서도 실행 가능한 더 작은 모델을 선택하세요. 16GB에서는 128K 컨텍스트를 여유 있게 사용할 수 있습니다. 32GB 이상이면 더 큰 모델이나 여러 병렬 슬롯을 실행할 수 있습니다.

그래도 메모리가 부족하다면 Hermes의 64K 최소값 이상을 유지하는 범위에서만 컨텍스트를 줄이세요. 그렇지 않으면 더 작은 모델이나 더 낮은 양자화(Q4_K_M 대신 Q3_K_M)로 전환하세요.

### 테스트

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }' | jq .choices[0].message.content
```

### 모델 이름 확인

모델 이름이 기억나지 않으면 모델 엔드포인트를 조회하세요.

```bash
curl -s http://localhost:8080/v1/models | jq '.data[].id'
```

---

## 옵션 B: omlx를 통한 MLX

[omlx](https://omlx.ai)는 MLX 모델을 관리하고 제공하는 macOS 네이티브 앱입니다. MLX는 Apple의 자체 머신러닝 프레임워크로, Apple Silicon의 통합 메모리 아키텍처에 맞게 특별히 최적화되어 있습니다.

### 설치

[omlx.ai](https://omlx.ai)에서 다운로드해 설치하세요. 모델 관리용 GUI와 내장 서버를 제공합니다.

### 모델 다운로드

omlx 앱에서 모델을 찾아 다운로드하세요. `Qwen3.5-9B-mlx-lm-mxfp4`를 검색해 다운로드합니다. 모델은 로컬에 저장됩니다(일반적으로 `~/.omlx/models/`).

### 서버 시작

omlx는 기본적으로 `http://127.0.0.1:8000`에서 모델을 제공합니다. 앱 UI에서 제공을 시작하거나, 사용 가능한 경우 CLI를 사용하세요.

### 테스트

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-mlx-lm-mxfp4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }' | jq .choices[0].message.content
```

### 사용 가능한 모델 목록

omlx는 여러 모델을 동시에 제공할 수 있습니다.

```bash
curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'
```

---

## 벤치마크: llama.cpp와 MLX 비교

두 백엔드는 동일한 시스템(Apple M5 Max, 통합 메모리 128GB)에서 동일한 모델(Qwen3.5-9B)을 비슷한 양자화 수준(GGUF는 Q4_K_M, MLX는 mxfp4)으로 실행해 테스트했습니다. 서로 다른 프롬프트 5개를 각각 3회 실행했으며, 리소스 경합을 피하기 위해 백엔드를 순차적으로 테스트했습니다.

### 결과

| 지표 | llama.cpp (Q4_K_M) | MLX (mxfp4) | 승자 |
|--------|-------------------|-------------|--------|
| **TTFT (평균)** | **67 ms** | 289 ms | llama.cpp(4.3배 빠름) |
| **TTFT (p50)** | **66 ms** | 286 ms | llama.cpp(4.3배 빠름) |
| **생성 (평균)** | 70 tok/s | **96 tok/s** | MLX(37% 빠름) |
| **생성 (p50)** | 70 tok/s | **96 tok/s** | MLX(37% 빠름) |
| **총 시간(512 토큰)** | 7.3s | **5.5s** | MLX(25% 빠름) |

### 의미

- **llama.cpp**는 프롬프트 처리에 뛰어납니다. 플래시 어텐션과 양자화된 KV 캐시 파이프라인 덕분에 약 66ms 만에 첫 토큰을 얻을 수 있습니다. 체감 반응성이 중요한 대화형 애플리케이션(챗봇, 자동 완성)을 구축한다면 의미 있는 장점입니다.

- **MLX**는 생성이 시작된 뒤 토큰을 약 37% 더 빠르게 생성합니다. 배치 작업, 긴 형식의 생성, 또는 초기 지연 시간보다 전체 완료 시간이 중요한 작업에서는 MLX가 더 빨리 끝납니다.

- 두 백엔드 모두 **매우 일관적**입니다. 실행 간 편차가 무시할 수 있을 정도로 작았으므로 이 수치를 신뢰해도 됩니다.

### 어느 쪽을 선택해야 할까요?

| 사용 사례 | 권장 사항 |
|----------|---------------|
| 대화형 채팅, 지연 시간이 짧은 도구 | llama.cpp |
| 긴 형식의 생성, 대량 처리 | MLX (omlx) |
| 메모리가 제한된 환경(8~16GB) | llama.cpp(양자화된 KV 캐시는 따라올 것이 없음) |
| 여러 모델을 동시에 제공 | omlx(내장 멀티 모델 지원) |
| 최대 호환성(Linux 포함) | llama.cpp |

---

## Hermes에 연결

로컬 서버가 실행되면 다음을 입력하세요.

```bash
hermes model
```

**Custom endpoint**를 선택하고 안내에 따르세요. 기본 URL과 모델 이름을 묻는데, 위에서 설정한 백엔드의 값을 사용하면 됩니다.

---

## 타임아웃

Hermes는 로컬 엔드포인트(localhost, LAN IP)를 자동으로 감지하고 스트리밍 타임아웃을 완화합니다. 대부분의 설정에서는 별도의 구성이 필요하지 않습니다.

그래도 타임아웃 오류가 발생한다면(예: 느린 하드웨어에서 매우 큰 컨텍스트를 사용하는 경우) 스트리밍 읽기 타임아웃을 재정의할 수 있습니다.

```bash
# In your .env — raise from the 120s default to 30 minutes
HERMES_STREAM_READ_TIMEOUT=1800
```

| 타임아웃 | 기본값 | 로컬 자동 조정 | 환경 변수 재정의 |
|---------|---------|----------------------|------------------|
| 스트림 읽기(소켓 수준) | 120s | 1800s로 증가 | `HERMES_STREAM_READ_TIMEOUT` |
| 오래된 스트림 감지 | 180s | 완전히 비활성화 | `HERMES_STREAM_STALE_TIMEOUT` |
| API 호출(비스트리밍) | 1800s | 변경 필요 없음 | `HERMES_API_TIMEOUT` |

문제를 일으킬 가능성이 가장 높은 것은 스트림 읽기 타임아웃입니다. 다음 데이터 청크를 받을 때까지의 소켓 수준 마감 시간입니다. 큰 컨텍스트에서 프리필을 수행하는 동안 로컬 모델은 프롬프트를 처리하느라 몇 분 동안 아무 출력도 내놓지 않을 수 있습니다. 자동 감지가 이를 투명하게 처리합니다.

:::tip 첫 번째 턴에 아무 소리도 없는 것은 대개 프리필이며 멈춘 것이 아닙니다
Hermes는 모든 호출마다 시스템 프롬프트와 도구 스키마를 전송하므로, 느린 하드웨어에서는 모델이 해당 프롬프트를 처리한 뒤 생성하기 시작할 때까지 첫 번째 턴에 몇 분 동안 아무 출력이 없을 수 있습니다. 이는 세션이 멈춘 것이 아니라 프리필이 진행 중인 것입니다. 완화 방법(모델을 로드한 상태로 유지하고 `hermes prompt-size`로 고정 프롬프트 줄이기)은 Ollama 가이드의 [느린 첫 응답(프리필)](./local-ollama-setup.md#slow-first-response-prefill)을 참고하세요.
:::
