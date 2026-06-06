---
sidebar_position: 2
title: "Mac에서 로컬 LLM 실행"
description: "llama.cpp 또는 MLX를 사용하여 macOS에 OpenAI 호환 로컬 LLM 서버를 설정합니다. 모델 선택, 메모리 최적화 및 Apple Silicon에서의 실제 벤치마크 결과를 포함합니다."
---

# Mac에서 로컬 LLM 실행

이 가이드는 macOS에서 OpenAI 호환 API를 지원하는 로컬 LLM 서버를 실행하는 방법을 안내합니다. 완전한 개인정보 보호와 API 비용 무료의 이점을 얻을 수 있으며, Apple Silicon에서 놀랍도록 뛰어난 성능을 경험할 수 있습니다.

두 가지 백엔드를 다룹니다:

| 백엔드 | 설치 방법 | 장점 | 포맷 |
|---------|---------|---------|--------|
| **llama.cpp** | `brew install llama.cpp` | 첫 번째 토큰 생성 시간이 가장 빠르며, 낮은 메모리를 위한 KV 캐시 양자화(quantized KV cache) 지원 | GGUF |
| **omlx** | [omlx.ai](https://omlx.ai) | 토큰 생성 속도가 가장 빠르며, 네이티브 Metal 최적화 | MLX (safetensors) |

두 가지 모두 OpenAI 호환 `/v1/chat/completions` 엔드포인트를 노출합니다. Hermes는 두 백엔드 모두와 함께 작동합니다 — `http://localhost:8080` 또는 `http://localhost:8000`을 가리키기만 하면 됩니다.

:::info Apple Silicon 전용
이 가이드는 Apple Silicon(M1 이상)이 장착된 Mac을 대상으로 합니다. 인텔 Mac에서도 llama.cpp가 작동하지만 GPU 가속은 되지 않으므로 성능이 눈에 띄게 느릴 수 있습니다.
:::

---

## 모델 선택

시작하려면 **Qwen3.5-9B**를 권장합니다. 강력한 추론 능력을 갖추고 있으며 양자화를 통해 8GB 이상의 통합 메모리에 쉽게 들어맞습니다.

| 모델 버전 | 디스크 용량 | 필요 RAM (128K 컨텍스트) | 백엔드 |
|---------|-------------|---------------------------|---------|
| Qwen3.5-9B-Q4_K_M (GGUF) | 5.3 GB | KV 캐시 양자화 시 ~10 GB | llama.cpp |
| Qwen3.5-9B-mlx-lm-mxfp4 (MLX) | ~5 GB | ~12 GB | omlx |

**메모리 경험 법칙:** 모델 크기 + KV 캐시. 9B Q4 모델은 약 5GB입니다. 128K 컨텍스트에 Q4 양자화를 적용한 KV 캐시는 약 4-5GB가 추가됩니다. 기본(f16) KV 캐시를 사용할 경우, 요구량은 약 16GB까지 늘어납니다. llama.cpp의 양자화된 KV 캐시 플래그는 메모리가 제한된 시스템에서 핵심 비법입니다.

더 큰 모델(27B, 35B)의 경우, 32GB 이상의 통합 메모리가 필요합니다. 9B는 8~16GB 머신에 최적화된 스위트 스팟입니다.

---

## 옵션 A: llama.cpp

llama.cpp는 휴대성이 가장 뛰어난 로컬 LLM 런타임입니다. macOS에서는 기본적으로 Metal을 사용하여 GPU를 가속화합니다.

### 설치

```bash
brew install llama.cpp
```

설치하면 `llama-server` 명령어를 전역적으로 사용할 수 있습니다.

### 모델 다운로드

GGUF 포맷의 모델이 필요합니다. `huggingface-cli`를 통해 Hugging Face에서 다운로드하는 것이 가장 쉽습니다:

```bash
brew install huggingface-cli
```

그런 다음 다운로드합니다:

```bash
huggingface-cli download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir ~/models
```

:::tip Gated models(비공개 모델)
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

각 플래그의 역할은 다음과 같습니다:

| 플래그 | 목적 |
|------|---------|
| `-ngl 99` | 모든 레이어를 GPU(Metal)로 오프로드합니다. CPU에 남는 것이 없도록 높은 숫자를 사용하세요. |
| `-c 131072` | 컨텍스트 창 크기(128K 토큰). 메모리가 부족한 경우 이 값을 줄이세요. |
| `-np 1` | 병렬 슬롯의 수입니다. 단일 사용자의 경우 1로 유지하세요 — 더 많은 슬롯은 메모리 예산을 분할합니다. |
| `-fa on` | 플래시 어텐션(Flash attention). 메모리 사용량을 줄이고 긴 컨텍스트의 추론 속도를 높입니다. |
| `--cache-type-k q4_0` | Key 캐시를 4비트로 양자화합니다. **이것이 가장 큰 메모리 절약 요인입니다.** |
| `--cache-type-v q4_0` | Value 캐시를 4비트로 양자화합니다. 위 옵션과 함께 사용하면 f16에 비해 KV 캐시 메모리가 약 75% 감소합니다. |
| `--host 0.0.0.0` | 모든 인터페이스에서 수신(listen) 대기합니다. 네트워크 액세스가 필요 없다면 `127.0.0.1`을 사용하세요. |

서버가 준비되면 다음 메시지가 표시됩니다:

```
main: server is listening on http://0.0.0.0:8080
srv  update_slots: all slots are idle
```

### 제한된 시스템을 위한 메모리 최적화

`--cache-type-k q4_0 --cache-type-v q4_0` 플래그는 메모리가 제한된 시스템에서 가장 중요한 최적화입니다. 128K 컨텍스트에서의 영향은 다음과 같습니다:

| KV 캐시 타입 | KV 캐시 메모리 (128K ctx, 9B 모델) |
|---------------|--------------------------------------|
| f16 (기본값) | ~16 GB |
| q8_0 | ~8 GB |
| **q4_0** | **~4 GB** |

8GB Mac에서는 `q4_0` KV 캐시를 사용하고, Hermes의 최소 컨텍스트인 64K를 충족하는 더 작은 모델을 선택하세요. 16GB에서는 128K 컨텍스트를 쾌적하게 실행할 수 있습니다. 32GB 이상에서는 더 큰 모델이나 다중 병렬 슬롯을 실행할 수 있습니다.

여전히 메모리가 부족한 경우 Hermes의 64K 최소 요구 사항 이상을 유지하는 선에서 컨텍스트를 줄이거나, 더 작은 모델 또는 더 작은 양자화 레벨(Q4_K_M 대신 Q3_K_M)로 전환하세요.

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

모델 이름을 잊은 경우 모델 엔드포인트를 쿼리하세요:

```bash
curl -s http://localhost:8080/v1/models | jq '.data[].id'
```

---

## 옵션 B: omlx를 통한 MLX

[omlx](https://omlx.ai)는 MLX 모델을 관리하고 제공하는 macOS 전용 앱입니다. MLX는 Apple Silicon의 통합 메모리 아키텍처에 특별히 최적화된 Apple 자체 머신 러닝 프레임워크입니다.

### 설치

[omlx.ai](https://omlx.ai)에서 다운로드하여 설치합니다. 모델 관리를 위한 GUI와 내장 서버를 제공합니다.

### 모델 다운로드

omlx 앱을 사용하여 모델을 찾아보고 다운로드합니다. `Qwen3.5-9B-mlx-lm-mxfp4`를 검색하여 다운로드하세요. 모델은 로컬(일반적으로 `~/.omlx/models/`)에 저장됩니다.

### 서버 시작

omlx는 기본적으로 `http://127.0.0.1:8000`에서 모델을 서빙합니다. 앱 UI에서 시작하거나 가능한 경우 CLI를 사용하세요.

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

### 사용 가능한 모델 나열

omlx는 여러 모델을 동시에 서빙할 수 있습니다:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'
```

---

## 벤치마크: llama.cpp vs MLX

동일한 컴퓨터(Apple M5 Max, 128GB 통합 메모리)에서 동일한 모델(Qwen3.5-9B)을 비슷한 양자화 수준(GGUF의 경우 Q4_K_M, MLX의 경우 mxfp4)으로 두 백엔드를 테스트했습니다. 다양한 유형의 프롬프트 5개로 각 3회 실행했으며 리소스 충돌을 피하기 위해 백엔드를 순차적으로 테스트했습니다.

### 결과

| 지표 | llama.cpp (Q4_K_M) | MLX (mxfp4) | 승자 |
|--------|-------------------|-------------|--------|
| **TTFT (평균)** | **67 ms** | 289 ms | llama.cpp (4.3배 빠름) |
| **TTFT (p50)** | **66 ms** | 286 ms | llama.cpp (4.3배 빠름) |
| **생성 속도 (평균)** | 70 tok/s | **96 tok/s** | MLX (37% 빠름) |
| **생성 속도 (p50)** | 70 tok/s | **96 tok/s** | MLX (37% 빠름) |
| **총 소요 시간 (512 토큰)** | 7.3s | **5.5s** | MLX (25% 빠름) |

### 의미하는 바

- **llama.cpp**는 프롬프트 처리에 뛰어납니다 — 플래시 어텐션 + 양자화된 KV 캐시 파이프라인 덕분에 첫 번째 토큰을 약 66ms 만에 얻을 수 있습니다. 챗봇이나 자동 완성 등 사용자가 체감하는 응답성이 중요한 대화형 애플리케이션을 구축하는 경우 이는 의미 있는 이점입니다.

- **MLX**는 본격적인 생성이 시작되면 토큰 생성 속도가 약 37% 더 빠릅니다. 일괄 작업, 긴 문장 생성 등 초기 지연 시간(latency)보다 총 완료 시간이 더 중요한 작업의 경우 MLX가 더 빨리 작업을 끝냅니다.

- 두 백엔드 모두 **매우 일관적**입니다 — 여러 번 실행해도 분산이 거의 없었습니다. 이 결과 수치를 신뢰할 수 있습니다.

### 어떤 것을 선택해야 할까요?

| 사용 사례 | 권장 사항 |
|----------|---------------|
| 대화형 채팅, 지연 시간이 짧은 도구 | llama.cpp |
| 긴 문장 생성, 일괄 처리 | MLX (omlx) |
| 메모리가 제한된 경우 (8-16 GB) | llama.cpp (양자화된 KV 캐시가 타의 추종을 불허함) |
| 여러 모델을 동시에 서빙하는 경우 | omlx (내장된 다중 모델 지원) |
| 최대 호환성 (Linux 포함) | llama.cpp |

---

## Hermes에 연결하기

로컬 서버가 실행되면:

```bash
hermes model
```

**Custom endpoint(사용자 지정 엔드포인트)**를 선택하고 안내에 따르세요. 기본 URL과 모델 이름을 묻는 메시지가 표시됩니다 — 위에서 설정한 백엔드의 값을 사용하세요.

---

## 타임아웃 (Timeouts)

Hermes는 로컬 엔드포인트(localhost, LAN IP)를 자동으로 감지하고 스트리밍 타임아웃을 완화합니다. 대부분의 설정에서는 추가 구성이 필요하지 않습니다.

여전히 타임아웃 오류가 발생하는 경우(예: 느린 하드웨어에서 매우 큰 컨텍스트 사용 시), 스트리밍 읽기 타임아웃을 오버라이드(재정의)할 수 있습니다:

```bash
# .env 파일에서 — 120초 기본값을 30분으로 늘리기
HERMES_STREAM_READ_TIMEOUT=1800
```

| 타임아웃 | 기본값 | 로컬 자동 조정 | 환경 변수 오버라이드 |
|---------|---------|----------------------|------------------|
| 스트림 읽기 (소켓 레벨) | 120s | 1800s로 증가 | `HERMES_STREAM_READ_TIMEOUT` |
| 오래된(stale) 스트림 감지 | 180s | 완전히 비활성화됨 | `HERMES_STREAM_STALE_TIMEOUT` |
| API 호출 (비 스트리밍) | 1800s | 변경 필요 없음 | `HERMES_API_TIMEOUT` |

스트림 읽기 타임아웃은 문제를 일으킬 가능성이 가장 높은 요소로, 다음 데이터 청크(chunk)를 수신하기 위한 소켓 수준의 제한 시간입니다. 대규모 컨텍스트를 사전 준비(prefill)하는 동안, 로컬 모델은 프롬프트를 처리하느라 몇 분 동안 아무런 출력도 내보내지 않을 수 있습니다. 자동 감지 기능이 이를 투명하게 처리합니다.
