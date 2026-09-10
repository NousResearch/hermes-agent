---
sidebar_position: 9
title: "Ollama로 Hermes 로컬 실행 — API 비용 0원"
description: "Ollama와 Gemma 4 같은 오픈 웨이트 모델을 사용해 클라우드 API 키나 유료 구독 없이 자신의 컴퓨터에서 Hermes Agent를 완전히 실행하는 단계별 안내"
---

# Ollama로 Hermes 로컬 실행 — API 비용 0원

## 문제

클라우드 LLM API는 토큰당 요금을 부과합니다. 무거운 코딩 세션에는 5~20달러가 들 수 있습니다. 개인 프로젝트, 학습 또는 개인정보 보호가 중요한 작업에서는 이 비용이 쌓이고, 모든 대화를 제3자에게 보내게 됩니다.

## 이 안내서에서 해결하는 것

[Ollama](https://ollama.com)를 모델 백엔드로 사용해 자신의 하드웨어에서 완전히 실행되는 Hermes Agent를 설정합니다. API 키도, 구독도 필요 없고 데이터가 컴퓨터 밖으로 나가지 않습니다. 설정이 끝나면 Hermes는 OpenRouter나 Anthropic을 사용할 때와 정확히 같은 방식으로 작동합니다. 터미널 명령, 파일 편집, 웹 브라우징, 위임을 사용할 수 있지만 모델은 로컬에서 실행됩니다.

이 안내를 마치면 다음을 갖추게 됩니다.

- 하나 이상의 오픈 웨이트 모델을 제공하는 Ollama
- 사용자 지정 엔드포인트로 Ollama에 연결된 Hermes
- 파일을 편집하고, 명령을 실행하고, 웹을 탐색할 수 있는 작동하는 로컬 에이전트
- 선택 사항: 자신의 하드웨어만으로 구동되는 Telegram/Discord 봇

## 필요한 것

| 구성 요소 | 최소 사양 | 권장 사양 |
|-----------|---------|-------------|
| **RAM** | 8 GB (3B 모델용) | 32+ GB (27B+ 모델용) |
| **저장 공간** | 5 GB 여유 공간 | 30+ GB (여러 모델용) |
| **CPU** | 4코어 | 8+코어 (AMD EPYC, Ryzen, Intel Xeon) |
| **GPU** | 필요 없음 | 8+ GB VRAM의 NVIDIA GPU를 사용하면 속도가 크게 향상됨 |

:::tip CPU만 사용해도 되지만 응답이 느릴 수 있습니다
Ollama는 CPU만 사용하는 서버에서도 실행됩니다. 최신 8코어 CPU에서 9B 모델은 초당 약 10토큰을 생성합니다. CPU에서 31B 모델은 더 느립니다(초당 약 2~5토큰). 각 응답에 30~120초가 걸리지만 작동합니다. GPU를 사용하면 크게 향상됩니다. CPU만 사용하는 설정에서는 환경 변수로 API 제한 시간을 늘리세요(`config.yaml` 키가 아닙니다).

```bash
# ~/.hermes/.env
HERMES_API_TIMEOUT=1800   # 30 minutes — generous for slow local models
```
:::

## 1단계: Ollama 설치

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

실행 중인지 확인합니다.

```bash
ollama --version
curl http://localhost:11434/api/tags   # Should return {"models":[]}
```

## 2단계: 모델 가져오기

하드웨어에 맞춰 선택하세요.

| 모델 | 디스크 크기 | 필요한 RAM | 도구 호출 | 적합한 용도 |
|-------|-------------|------------|:------------:|----------|
| `gemma4:31b` | ~20 GB | 24+ GB | 예 | 최고 품질 — 강력한 도구 사용 및 추론 |
| `gemma2:27b` | ~16 GB | 20+ GB | 아니요 | 대화 작업, 도구 사용 없음 |
| `gemma2:9b` | ~5 GB | 8+ GB | 아니요 | 빠른 채팅, Q&A — 도구 호출 불가 |
| `llama3.2:3b` | ~2 GB | 4+ GB | 아니요 | 가벼운 빠른 답변만 |

:::warning 도구 호출이 중요합니다
Hermes는 **에이전트형** 어시스턴트입니다. 도구 호출을 통해 파일을 편집하고, 명령을 실행하고, 웹을 탐색합니다. 도구 호출을 지원하지 않는 모델은 채팅만 할 수 있고 작업을 수행할 수 없습니다. Hermes의 모든 기능을 사용하려면 도구를 지원하는 모델(예: `gemma4:31b`)을 사용하세요.
:::

선택한 모델을 가져옵니다.

```bash
ollama pull gemma4:31b
```

:::info 여러 모델
여러 모델을 가져온 뒤 Hermes에서 `/model`로 전환할 수 있습니다. Ollama는 필요할 때 활성 모델을 메모리에 로드하고, 유휴 상태가 되면 자동으로 언로드합니다.
:::

모델이 작동하는지 확인합니다.

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

Hermes 설정 마법사를 실행합니다.

```bash
hermes setup
```

공급자를 묻는 메시지가 표시되면 **사용자 지정 엔드포인트**를 선택하고 다음을 입력합니다.

- **기본 URL:** `http://localhost:11434/v1`
- **API 키:** 비워 두거나 `no-key` 입력(Ollama에는 필요하지 않음)
- **모델:** `gemma4:31b`(또는 가져온 모델)

또는 `~/.hermes/config.yaml`을 직접 편집합니다.

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"
```

## 4단계: Hermes 사용 시작

```bash
hermes
```

이제 완전히 로컬에서 실행되는 에이전트를 갖췄습니다. 다음과 같이 시도해 보세요.

```
You: List all Python files in this directory and count the lines of code in each

You: Read the README.md and summarize what this project does

You: Create a Python script that fetches the weather for Ho Chi Minh City
```

Hermes는 터미널 도구, 파일 작업 및 로컬 모델을 사용하므로 클라우드 호출이 없습니다.

## 5단계: 작업에 맞는 모델 선택

모든 작업에 가장 큰 모델이 필요한 것은 아닙니다. 다음은 실용적인 안내입니다.

| 작업 | 권장 모델 | 이유 |
|-----|-------------------|-----|
| 파일 편집, 코드, 터미널 명령 | `gemma4:31b` | 안정적인 도구 호출을 지원하는 유일한 모델 |
| 빠른 Q&A(도구 사용 불필요) | `gemma2:9b` | 대화 작업에 빠른 응답 |
| 가벼운 채팅 | `llama3.2:3b` | 가장 빠르지만 기능이 매우 제한적 |

:::note
전체 에이전트 작업(편집, 명령 실행, 브라우징)에는 `gemma4:31b`가 현재 도구 호출을 지원하는 최고의 로컬 옵션입니다. 최신 모델은 [Ollama 모델 라이브러리](https://ollama.com/library)에서 확인하세요. 도구 호출 지원은 빠르게 확대되고 있습니다.
:::

세션 중에 모델을 즉시 전환할 수 있습니다.

```
/model gemma2:9b
```

## 6단계: 속도 최적화

### Ollama의 컨텍스트 창 늘리기

기본적으로 Ollama는 2048토큰 컨텍스트를 사용합니다. 도구를 사용하는 에이전트 작업에는 Hermes가 최소 64,000토큰을 요구합니다.

```bash
# Create a Modelfile that extends context
cat > /tmp/Modelfile << 'EOF'
FROM gemma4:31b
PARAMETER num_ctx 64000
EOF

ollama create gemma4-64k -f /tmp/Modelfile
```

그런 다음 Hermes 구성에서 모델 이름으로 `gemma4-64k`를 사용하도록 업데이트합니다.

### 모델을 로드된 상태로 유지

기본적으로 Ollama는 5분 동안 사용하지 않으면 모델을 언로드합니다. 지속적인 게이트웨이 봇에서는 모델을 로드된 상태로 유지하세요.

```bash
# Set keep-alive to 24 hours
curl http://localhost:11434/api/generate \
  -d '{"model": "gemma4:31b", "keep_alive": "24h"}'
```

또는 Ollama 환경에서 전역으로 설정합니다.

```bash
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_KEEP_ALIVE=24h"
```

### GPU 오프로딩 사용(가능한 경우)

NVIDIA GPU가 있으면 Ollama가 자동으로 레이어를 GPU로 오프로딩합니다. 다음으로 확인하세요.

```bash
ollama ps   # Shows which model is loaded and how many GPU layers
```

12 GB GPU에서 31B 모델을 사용하면 부분 오프로딩(약 40개 레이어는 GPU, 나머지는 CPU)이 이루어지며, 그래도 속도가 크게 향상됩니다.

## 7단계: 게이트웨이 봇으로 실행(선택 사항)

CLI에서 Hermes가 로컬로 작동하면 Telegram 또는 Discord 봇으로 노출할 수 있습니다. 이 경우에도 여전히 하드웨어에서 완전히 실행됩니다.

### Telegram

1. [@BotFather](https://t.me/BotFather)를 통해 봇을 만들고 토큰을 받습니다
2. `~/.hermes/config.yaml`에 추가합니다.

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

3. 게이트웨이를 시작합니다.

```bash
hermes gateway
```

이제 Telegram에서 봇에 메시지를 보내면 로컬 모델을 사용해 응답합니다.

### Discord

1. [discord.com/developers](https://discord.com/developers/applications)에서 Discord 애플리케이션을 만듭니다
2. 구성에 추가합니다.

```yaml
platforms:
  discord:
    enabled: true
    token: "YOUR_DISCORD_BOT_TOKEN"
```

3. 시작합니다: `hermes gateway`

## 8단계: 폴백 설정(선택 사항)

로컬 모델은 복잡한 작업에 어려움을 겪을 수 있습니다. 로컬 모델이 실패할 때만 활성화되는 클라우드 폴백을 설정하세요.

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"

fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

이렇게 하면 사용량의 90%는 무료(로컬)로 처리되고, 어려운 작업만 유료 API에 전달됩니다.

## 문제 해결

### 시작 시 "Connection refused"

Ollama가 실행되고 있지 않습니다. 시작하세요.

```bash
sudo systemctl start ollama
# or
ollama serve
```

### 응답이 느림

- **모델 크기와 RAM 확인:** 모델에 필요한 RAM이 사용 가능한 양보다 많으면 디스크로 스왑됩니다. 더 작은 모델을 사용하거나 RAM을 추가하세요.
- **`ollama ps` 확인:** GPU 레이어가 오프로딩되지 않았다면 응답이 CPU에 의존하는 상태입니다. CPU만 사용하는 서버에서는 정상입니다.
- **컨텍스트 줄이기:** 대화가 길수록 추론이 느려집니다. `/compress`를 정기적으로 사용하거나 구성에서 압축 임계값을 낮게 설정하세요.

### 첫 응답이 느림(prefill)

Hermes는 모든 API 호출에서 대화 내용에 앞서 고정 페이로드(모든 활성화된 도구의 시스템 프롬프트와 도구 스키마)를 보냅니다. CPU만 사용하거나 VRAM이 적은 설정에서는 해당 프롬프트를 처리하는 *prefill* 단계가 첫 번째 턴을 지배합니다. 모델이 프롬프트를 처리하는 동안 몇 분간 아무 출력도 없이 멈춘 듯 보일 수 있으며, 그 뒤 정상적인 속도로 생성합니다. 이는 멈춤이 아니라 예상된 동작입니다. [Mac 로컬 LLM 안내서](./local-llm-on-mac.md#timeouts)에도 같은 현상이 설명되어 있습니다. 큰 컨텍스트에서 prefill을 수행하는 동안 로컬 모델은 프롬프트 처리로 몇 분간 출력을 내지 않을 수 있으며, Hermes는 로컬 엔드포인트의 스트림 읽기 제한 시간을 120초에서 1800초로 자동 상향합니다(`HERMES_STREAM_READ_TIMEOUT`).

도움이 되는 방법:

- **모델을 로드된 상태로 유지** — Ollama는 5분이 지나면 유휴 모델을 언로드하므로 다음 prefill 전에 전체 재로드가 추가됩니다. `OLLAMA_KEEP_ALIVE=24h`를 설정하세요([6단계](#keep-the-model-loaded) 참조).
- **API 제한 시간 늘리기** — `~/.hermes/.env`에 `HERMES_API_TIMEOUT=1800`을 설정하세요([필요한 것](#what-you-need) 참조).
- **고정 프롬프트 측정 및 축소** — `hermes prompt-size`를 실행해 시스템 프롬프트와 도구 스키마의 바이트별 구성을 확인한 다음, `hermes tools`로 사용하지 않는 도구 세트를 비활성화하고 `hermes skills`로 필요하지 않은 스킬을 제거하세요.
- **GPU 오프로딩 사용** — 부분 오프로딩만으로도 속도가 크게 향상됩니다([6단계](#use-gpu-offloading-if-available) 참조).

### 모델이 도구 호출을 따르지 않음

도구 호출을 지원하지 않는 모델은 구조화된 함수 호출 대신 일반 텍스트를 생성합니다. 해결 방법:

- **도구 호출을 지원하는 모델 사용** — 위에 나열된 모델 중 안정적인 도구 호출을 지원하는 것은 `gemma4:31b`뿐입니다.
- **Hermes에는 자동 복구 기능이 있음** — 잘못된 도구 호출을 감지하고 자동으로 수정을 시도합니다.
- **폴백 설정** — 로컬 모델이 3번 실패하면 Hermes가 클라우드 공급자로 폴백합니다.

모델이 실제로 도구를 실행하는 대신 답변에 `{"name": "web_search", ...}`와 같은 원시 JSON을 출력한다면, 이는 보통 모델이 아니라 *서버*의 문제입니다. 도구 호출이 활성화되지 않았거나 도구 호출 형식이 파싱되지 않은 것입니다. [실행되지 않고 텍스트로 표시되는 도구 호출](/integrations/providers#tool-calls-appear-as-text-instead-of-executing)의 서버별 수정 표를 참조하세요(llama.cpp에는 `--jinja`, vLLM에는 `--enable-auto-tool-choice --tool-call-parser hermes` 등이 필요합니다).

### 컨텍스트 창 오류

기본 Ollama 컨텍스트(2048토큰)는 에이전트 작업에 너무 작습니다. 늘리는 방법은 [6단계](#step-6-optimize-for-speed)를 참조하세요.

## 비용 비교

일반적인 코딩 세션(입력 약 10만 토큰, 출력 약 2만 토큰)을 기준으로 로컬 실행 시 클라우드 API 대비 절약되는 비용은 다음과 같습니다.

| 공급자 | 세션당 비용 | 월간(매일 사용) |
|----------|-----------------|---------------------|
| Anthropic Claude Sonnet | ~$0.80 | ~$24 |
| OpenRouter (GPT-4o) | ~$0.60 | ~$18 |
| **Ollama (local)** | **$0.00** | **$0.00** |

유일한 비용은 전기 요금이며, 하드웨어에 따라 세션당 대략 0.01~0.05달러입니다.

## 로컬에서 잘 작동하는 것

- **파일 편집 및 코드 생성** — 9B 이상 모델은 이를 잘 처리합니다
- **터미널 명령** — Hermes가 명령을 감싸 실행하고 모델과 관계없이 출력을 읽습니다
- **웹 브라우징** — 브라우저 도구가 가져오기를 수행하고, 모델은 결과를 해석하기만 합니다
- **Cron 작업 및 예약된 작업** — 클라우드 설정과 동일하게 작동합니다
- **멀티 플랫폼 게이트웨이** — Telegram, Discord, Slack이 모두 로컬 모델과 작동합니다

## 클라우드 모델이 더 나은 점

- **매우 복잡한 다단계 추론** — 70B+ 또는 Claude Opus 같은 클라우드 모델이 눈에 띄게 더 뛰어납니다
- **긴 컨텍스트 창** — 클라우드 모델은 10만~100만 토큰을 제공하며, 로컬 런타임은 구성하지 않으면 Hermes의 최소 64K보다 작은 값을 기본으로 사용하는 경우가 많습니다
- **긴 응답의 속도** — 긴 생성에서는 클라우드 추론이 CPU만 사용하는 로컬 추론보다 빠릅니다

가장 좋은 조합은 일상적인 작업에는 로컬을 사용하고, 어려운 작업에는 클라우드 폴백을 설정하는 것입니다.
