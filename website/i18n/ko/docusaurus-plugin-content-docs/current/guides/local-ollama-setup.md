---
sidebar_position: 9
title: "Ollama로 로컬에서 Hermes 실행하기 — API 비용 0원"
description: "클라우드 API 키나 유료 구독 없이 Ollama와 Gemma 4 같은 오픈 웨이트(open-weight) 모델을 사용하여 Hermes 에이전트를 내 컴퓨터에서 온전히 실행하기 위한 단계별 가이드"
---

# Ollama로 로컬에서 Hermes 실행하기 — API 비용 0원

## 문제점

클라우드 LLM API는 토큰당 비용을 청구합니다. 코딩 작업을 집중해서 하면 세션당 5~20달러의 비용이 들 수 있습니다. 개인 프로젝트, 학습 또는 개인정보에 민감한 작업을 할 때 이 비용은 누적되며, 무엇보다 모든 대화 내용을 제3자에게 전송하게 됩니다.

## 이 가이드의 해결책

이 가이드에서는 모델 백엔드로 [Ollama](https://ollama.com)를 사용하여 여러분의 하드웨어에서 전적으로 실행되는 Hermes Agent를 설정합니다. API 키, 구독이 필요 없으며, 기기 외부로 데이터가 유출되지도 않습니다. 한 번 구성하고 나면, 로컬 모델이 실행된다는 점만 다를 뿐 OpenRouter나 Anthropic을 사용할 때와 동일하게 Hermes가 작동합니다 (터미널 명령어, 파일 편집, 웹 브라우징, 위임 등 모두 지원).

이 과정을 마치면 다음을 갖게 됩니다:

- 하나 이상의 오픈 웨이트 모델을 서비스하는 Ollama
- 사용자 지정 엔드포인트로서 Ollama에 연결된 Hermes
- 파일을 편집하고, 명령어를 실행하며, 웹을 탐색할 수 있는 동작하는 로컬 에이전트
- 선택 사항: 전적으로 개인의 하드웨어로 구동되는 Telegram/Discord 봇

## 필요 사항 (What You Need)

| 구성 요소 | 최소 사양 | 권장 사양 |
|-----------|---------|-------------|
| **RAM** | 8 GB (3B 모델용) | 32+ GB (27B 이상 모델용) |
| **저장 공간** | 여유 5 GB | 30+ GB (다중 모델 사용 시) |
| **CPU** | 4 코어 | 8+ 코어 (AMD EPYC, Ryzen, Intel Xeon) |
| **GPU** | 필수 아님 | 8+ GB VRAM을 탑재한 NVIDIA GPU (속도가 크게 향상됨) |

:::tip CPU 전용도 작동하지만 응답 속도가 느릴 수 있습니다
Ollama는 CPU 전용 서버에서도 실행됩니다. 최신 8코어 CPU에서 9B 모델을 실행하면 초당 약 10토큰이 처리됩니다. CPU에서 31B 모델을 실행하면 더 느리며(초당 약 2~5토큰), 각 응답에 30~120초가 걸리지만 작동은 합니다. GPU를 사용하면 이 속도가 획기적으로 향상됩니다. CPU 전용 설정의 경우, 환경 변수를 통해 API 시간 초과(timeout) 여유를 넉넉하게 설정하세요 (`config.yaml` 키가 아닙니다):

```bash
# ~/.hermes/.env
HERMES_API_TIMEOUT=1800   # 30분 — 느린 로컬 모델을 위한 넉넉한 시간
```
:::

## 1단계: Ollama 설치

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

실행 중인지 확인합니다:

```bash
ollama --version
curl http://localhost:11434/api/tags   # {"models":[]} 반환 확인
```

## 2단계: 모델 다운로드 (Pull a Model)

하드웨어 사양에 맞춰 선택하세요:

| 모델 | 디스크 공간 | 필요 RAM | 도구 호출(Tool Calling) | 적합한 용도 |
|-------|-------------|------------|:------------:|----------|
| `gemma4:31b` | 약 20 GB | 24+ GB | 지원함 (Yes) | 최고 품질 — 강력한 도구 사용 및 추론 능력 |
| `gemma2:27b` | 약 16 GB | 20+ GB | 미지원 (No) | 도구 사용이 필요 없는 대화형 태스크 |
| `gemma2:9b` | 약 5 GB | 8+ GB | 미지원 (No) | 빠른 채팅, Q&A — 도구 호출 불가능 |
| `llama3.2:3b` | 약 2 GB | 4+ GB | 미지원 (No) | 매우 가볍고 빠른 답변 전용 |

:::warning 도구 호출(Tool calling)의 중요성
Hermes는 도구 호출을 통해 파일을 편집하고, 명령어를 실행하고, 웹을 탐색하는 **에이전트 역할(agentic)** 을 수행하는 어시스턴트입니다. 도구 호출을 지원하지 않는 모델은 단지 채팅만 할 수 있으며 동작(action)을 취할 수 없습니다. Hermes의 전체 경험을 누리려면 도구를 지원하는 모델(예: `gemma4:31b`)을 사용해야 합니다.
:::

선택한 모델을 다운로드합니다:

```bash
ollama pull gemma4:31b
```

:::info 여러 모델(Multiple models) 사용
여러 개의 모델을 다운로드한 다음 Hermes 세션 내에서 `/model` 명령어로 모델을 전환할 수 있습니다. Ollama는 활성화된 모델을 필요할 때 메모리에 로드하고 유휴 상태인 모델은 자동으로 언로드(unload)합니다.
:::

모델이 작동하는지 확인합니다:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:31b",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 50
  }'
```

모델의 답변이 포함된 JSON 응답이 표시되어야 합니다.

## 3단계: Hermes 구성

Hermes 설정 마법사를 실행합니다:

```bash
hermes setup
```

제공자(provider)를 입력하라는 메시지가 표시되면 **Custom Endpoint**를 선택하고 다음 값을 입력합니다:

- **Base URL:** `http://localhost:11434/v1`
- **API Key:** 비워 두거나 `no-key` 입력 (Ollama는 키가 필요하지 않음)
- **Model:** `gemma4:31b` (또는 다운로드한 모델명)

또는 `~/.hermes/config.yaml`을 직접 편집할 수도 있습니다:

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"
```

## 4단계: Hermes 사용 시작하기

```bash
hermes
```

이것으로 끝입니다. 이제 완전히 로컬로 작동하는 에이전트를 실행 중입니다. 시도해 보세요:

```
You: 이 디렉토리에 있는 모든 Python 파일을 나열하고 각각의 코드 라인 수를 세어줘

You: README.md를 읽고 이 프로젝트가 하는 일을 요약해줘

You: 호치민시의 날씨를 가져오는 Python 스크립트를 작성해줘
```

Hermes는 터미널 도구, 파일 작업 및 로컬 모델을 사용하여 처리할 것이며 클라우드 호출은 발생하지 않습니다.

## 5단계: 작업에 맞는 올바른 모델 선택

모든 작업에 가장 큰 모델이 필요한 것은 아닙니다. 다음은 실용적인 가이드입니다:

| 작업 | 권장 모델 | 이유 |
|------|-------------------|-----|
| 파일 편집, 코드, 터미널 명령어 | `gemma4:31b` | 신뢰할 수 있는 도구 호출을 지원하는 유일한 모델 |
| 빠른 Q&A (도구 사용 필요 없음) | `gemma2:9b` | 대화형 작업에 대한 빠른 응답 속도 |
| 가벼운 채팅 | `llama3.2:3b` | 가장 빠르지만, 능력이 매우 제한적임 |

:::note
파일 편집, 명령어 실행, 브라우징 등 본격적인 에이전트 작업(agentic work)의 경우 도구 호출을 지원하는 `gemma4:31b`가 현재 로컬 옵션 중 최고입니다. 도구 호출 지원이 빠르게 확장되고 있으므로 [Ollama의 모델 라이브러리](https://ollama.com/library)에서 최신 모델들을 계속 확인해 보세요.
:::

세션 내에서 모델을 즉시 전환할 수 있습니다:

```
/model gemma2:9b
```

## 6단계: 속도 최적화

### Ollama의 컨텍스트 창(Context Window) 늘리기

기본적으로 Ollama는 2048개의 토큰 컨텍스트를 사용합니다. Hermes가 도구를 사용하여 에이전트 작업을 수행하려면 최소 64,000개의 토큰이 필요합니다:

```bash
# 컨텍스트를 확장하는 Modelfile 생성
cat > /tmp/Modelfile << 'EOF'
FROM gemma4:31b
PARAMETER num_ctx 64000
EOF

ollama create gemma4-64k -f /tmp/Modelfile
```

그런 다음 Hermes 설정(`config.yaml`)을 업데이트하여 모델명으로 `gemma4-64k`를 사용합니다.

### 모델 로드 상태 유지하기

기본적으로 Ollama는 5분 동안 활동이 없으면 모델을 언로드합니다. 영구적인 게이트웨이 봇으로 실행하려면 모델을 계속 로드된 상태로 유지하세요:

```bash
# 활성 유지 시간을 24시간으로 설정
curl http://localhost:11434/api/generate \
  -d '{"model": "gemma4:31b", "keep_alive": "24h"}'
```

또는 Ollama의 전역 환경 변수에 설정할 수 있습니다:

```bash
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_KEEP_ALIVE=24h"
```

### GPU 오프로딩 사용 (사용 가능한 경우)

NVIDIA GPU가 있는 경우 Ollama는 자동으로 모델의 레이어를 GPU로 오프로드합니다. 다음 명령어로 확인할 수 있습니다:

```bash
ollama ps   # 로드된 모델과 GPU에 오프로드된 레이어 수를 보여줌
```

12 GB GPU에서 31B 모델을 실행하는 경우 부분적인 오프로드(GPU에 약 40개 레이어, 나머지는 CPU에 할당)가 발생하지만, 그래도 여전히 상당한 속도 향상을 얻을 수 있습니다.

## 7단계: 게이트웨이 봇으로 실행 (선택 사항)

Hermes가 CLI에서 로컬로 작동하게 되면 Telegram이나 Discord 봇으로 노출시킬 수 있습니다 — 여전히 전적으로 사용자의 하드웨어에서 실행됩니다.

### Telegram

1. [@BotFather](https://t.me/BotFather)를 통해 봇을 생성하고 토큰을 발급받습니다.
2. `~/.hermes/config.yaml`에 추가합니다:

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"

platforms:
  telegram:
    enabled: true
    token: "YOUR_TELEGRAM_BOT_TOKEN"
```

3. 게이트웨이를 시작합니다:

```bash
hermes gateway
```

이제 Telegram에서 봇에게 메시지를 보내면 로컬 모델을 사용하여 응답합니다.

### Discord

1. [discord.com/developers](https://discord.com/developers/applications)에서 Discord 애플리케이션을 생성합니다.
2. 설정 파일에 추가합니다:

```yaml
platforms:
  discord:
    enabled: true
    token: "YOUR_DISCORD_BOT_TOKEN"
```

3. 시작: `hermes gateway`

## 8단계: 대체 수단 (Fallback) 설정 (선택 사항)

로컬 모델은 복잡한 작업을 처리하는 데 어려움을 겪을 수 있습니다. 로컬 모델이 실패할 때만 활성화되는 클라우드 대체(fallback) 수단을 설정하세요:

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"

fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

이렇게 하면 사용량의 90%는 무료(로컬)로 처리되고, 어려운 작업에만 유료 API를 호출하게 됩니다.

## 문제 해결 (Troubleshooting)

### 시작 시 "Connection refused"

Ollama가 실행 중이 아닙니다. 시작하세요:

```bash
sudo systemctl start ollama
# 또는
ollama serve
```

### 느린 응답

- **모델 크기 대비 RAM 확인:** 모델이 사용 가능한 메모리보다 많은 RAM을 필요로 하면 디스크 스왑(swap)이 발생합니다. 더 작은 모델을 사용하거나 RAM을 늘리세요.
- **`ollama ps` 확인:** GPU 레이어가 오프로드되지 않았다면 응답은 전적으로 CPU에 바인딩됩니다. CPU 전용 서버에서는 정상적인 동작입니다.
- **컨텍스트 줄이기:** 대화 내용이 길어지면 추론 속도가 느려집니다. `/compress` 명령어를 정기적으로 사용하거나 설정 파일에서 더 낮은 압축 임계값을 설정하세요.

### 모델이 도구 호출(Tool calls)을 따르지 않음

크기가 작은 모델(3B, 7B)은 도구 호출 지침을 무시하고 구조화된 함수 호출 대신 일반 텍스트를 생성하는 경우가 종종 있습니다. 해결 방법:

- **더 큰 모델 사용** — `gemma4:31b` 또는 `gemma2:27b`는 3B/7B 모델보다 도구 호출을 훨씬 잘 처리합니다.
- **Hermes의 자동 복구** — 형식이 잘못된 도구 호출을 감지하고 자동으로 수정을 시도합니다.
- **대체 수단(fallback) 설정** — 로컬 모델이 3번 실패하면 Hermes는 구성해둔 클라우드 제공자로 넘어갑니다.

### 컨텍스트 창 오류

기본 Ollama 컨텍스트 크기(2048 토큰)는 에이전트 작업을 수행하기에 너무 작습니다. [6단계](#6단계-속도-최적화)를 참고하여 컨텍스트 크기를 늘리세요.

## 비용 비교

일반적인 코딩 세션(입력 약 100K 토큰, 출력 약 20K 토큰)을 기준으로 로컬 실행 시 클라우드 API와 비교하여 절약되는 비용은 다음과 같습니다:

| 제공자 (Provider) | 세션당 비용 | 월 예상 (매일 사용 시) |
|----------|-----------------|---------------------|
| Anthropic Claude Sonnet | 약 $0.80 | 약 $24 |
| OpenRouter (GPT-4o) | 약 $0.60 | 약 $18 |
| **Ollama (로컬)** | **$0.00** | **$0.00** |

여러분이 지불해야 할 유일한 비용은 전기세 뿐입니다 — 하드웨어에 따라 다르지만 세션당 약 0.01~0.05달러 수준입니다.

## 로컬에서 잘 작동하는 작업 (What Works Well Locally)

- **파일 편집 및 코드 생성** — 9B 이상의 모델이 이를 잘 처리합니다.
- **터미널 명령어** — 어떤 모델을 쓰더라도 Hermes가 명령어를 래핑하고 실행하며 출력을 읽습니다.
- **웹 브라우징** — 브라우저 도구가 가져오는 역할을 하고, 모델은 결과를 해석하기만 하면 됩니다.
- **크론 작업 및 예약된 태스크** — 클라우드 설정과 동일하게 작동합니다.
- **다중 플랫폼 게이트웨이** — Telegram, Discord, Slack 모두 로컬 모델과 함께 잘 작동합니다.

## 클라우드 모델을 사용할 때 더 좋은 작업 (What's Better with Cloud Models)

- **매우 복잡한 다단계 추론** — 70B 이상의 모델이나 Claude Opus와 같은 클라우드 모델이 눈에 띄게 우수합니다.
- **긴 컨텍스트 창(Long context windows)** — 클라우드 모델은 100K~1M 토큰을 제공하지만, 로컬 런타임은 따로 설정하지 않는 한 Hermes의 최소 권장 사양인 64K 미만을 기본값으로 하는 경우가 많습니다.
- **대량의 응답 생성 속도** — 답변 길이가 길어지는 경우 CPU 전용의 로컬 환경보다 클라우드 추론 속도가 훨씬 빠릅니다.

최적의 사용법: 일상적인 작업에는 로컬 모델을 사용하고, 어려운 작업을 위해 클라우드를 대체 수단(fallback)으로 설정하세요.
