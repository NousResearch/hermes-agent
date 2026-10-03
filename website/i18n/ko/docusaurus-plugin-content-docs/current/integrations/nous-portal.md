---
sidebar_position: 1
title: "Nous Portal"
description: "하나의 구독으로 300개 이상의 최첨단 모델과 Tool Gateway를 이용하는 Hermes Agent 권장 실행 방법"
---

# Nous Portal

[Nous Portal](https://portal.nousresearch.com)은 Nous Research의 통합 구독 게이트웨이이며, **Hermes Agent를 실행하는 데 권장되는 방법**입니다. 하나의 OAuth 로그인으로 직접 연결해야 했던 모든 모델 연구소, 검색 API, 이미지 생성기, 브라우저 제공업체의 개별 계정, API 키, 결제 관계를 일일이 관리할 필요가 없어집니다.

설정할 시간이 한 가지밖에 없다면 이것부터 설정하세요. 가장 빠른 방법은 다음과 같습니다.

```bash
hermes setup --portal
```

이 단일 명령은 Portal OAuth를 실행하고, Nous 모델을 선택하게 하며, `config.yaml`에서 Nous를 추론 제공업체로 설정하고 Tool Gateway를 켭니다. 이제 바로 `hermes chat`을 사용할 수 있습니다.

아직 구독이 없나요? [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription)에서 가입한 다음 돌아와 위 명령을 실행하세요.

## 구독에 포함된 항목

### 하나의 청구서로 이용하는 300개 이상의 최첨단 모델

Portal은 생태계 전반에서 선별한 에이전트 모델 카탈로그를 프록시합니다. 연구소마다 하나씩 크레딧 잔액을 관리하는 대신 Nous 구독으로 청구됩니다.

| 제품군 | 모델 |
|--------|--------|
| **Anthropic Claude** | Opus 4.7, Opus 4.6, Sonnet 4.6, Haiku 4.5 |
| **OpenAI** | GPT-5.5, GPT-5.5 Pro, GPT-5.4 Mini, GPT-5.4 Nano, GPT-5.3 Codex |
| **Google Gemini** | Gemini 3 Pro Preview, Gemini 3 Flash Preview, Gemini 3.1 Pro Preview, Gemini 3.1 Flash Lite Preview |
| **DeepSeek** | DeepSeek V4 Pro |
| **Qwen** | Qwen3.7-Max, Qwen3.6-35B-A3B |
| **Kimi / Moonshot** | Kimi K2.6 |
| **GLM / Zhipu** | GLM-5.1 |
| **MiniMax** | MiniMax M2.7 |
| **xAI** | Grok 4.3 |
| **NVIDIA** | Nemotron-3 Super 120B-A12B |
| **Tencent** | Hunyuan 3 Preview |
| **Xiaomi** | MiMo V2.5 Pro |
| **StepFun** | Step 3.5 Flash |
| **Hermes** | Hermes-4-70B, Hermes-4-405B (채팅, [아래 참고](#a-note-on-hermes-4)) |
| **+ 그 외 모든 항목** | 280개 이상의 추가 모델 — 전체 에이전트 최첨단 모델 |

내부적으로 Portal은 각 모델을 가장 적합한 백엔드로 라우팅합니다. 일부 모델은 OpenRouter를 거치고, 다른 모델은 독점 또는 보조 제공업체를 거치며, 특정 모델의 라우팅은 시간이 지나면서 변경될 수 있습니다. 어느 경우든 모든 비용은 Nous 구독으로 청구됩니다. `/model`을 사용하면 세션 중간에도 코드에는 Claude Sonnet 4.6을, 긴 컨텍스트에는 Gemini 3 Pro를 전환해 사용할 수 있습니다. 새 자격 증명, 충전, 잔액 부족으로 인한 예상치 못한 오류가 필요하지 않습니다.

:::note
라우팅은 모델별로 이루어지며 항상 OpenRouter를 거치는 것은 아니므로, OpenRouter 전용 요청 확장 기능(예: `provider` 라우팅 선호도, `session_id` 고정 라우팅, 최상위 `cache_control`)은 Portal API 계약의 일부가 아니며 모델을 제공하는 백엔드에 따라 무시될 수 있습니다.
:::

### Nous Tool Gateway

같은 구독으로 [Tool Gateway](/user-guide/features/tool-gateway)가 활성화되며, Hermes Agent의 도구 호출을 Nous가 관리하는 인프라를 통해 라우팅합니다. 다섯 개의 백엔드와 하나의 로그인으로 이용할 수 있습니다.

| 도구 | 파트너 | 기능 |
|------|---------|--------------|
| **웹 검색 및 추출** | Firecrawl | 에이전트급 검색과 전체 페이지 추출입니다. Firecrawl API 키나 속도 제한을 직접 관리할 필요가 없습니다. |
| **이미지 생성** | FAL | 하나의 엔드포인트에서 9개 모델을 제공합니다: FLUX 2 Klein 9B, FLUX 2 Pro, Z-Image Turbo, Nano Banana Pro (Gemini 3 Pro Image), GPT Image 1.5, GPT Image 2, Ideogram V3, Recraft V4 Pro, Qwen Image. |
| **텍스트 음성 변환** | OpenAI TTS | 별도의 OpenAI 키 없이 고품질 TTS를 제공합니다. 메시징 플랫폼 전반에서 [음성 모드](/user-guide/features/voice-mode)를 활성화합니다. |
| **클라우드 브라우저 자동화** | Browser Use | `browser_navigate`, `browser_click`, `browser_type`, `browser_vision`을 위한 헤드리스 Chromium 세션입니다. Browserbase 계정이 필요하지 않습니다. |
| **클라우드 터미널 샌드박스** | Modal | 코드 실행을 위한 서버리스 터미널 샌드박스입니다(선택적 추가 기능). |

게이트웨이가 없다면 이 기능들을 각각 연결하기 위해 Firecrawl 계정, FAL 계정, Browser Use 계정, OpenAI 키, Modal 계정이 필요합니다. 가입 5번, 대시보드 5개, 충전 절차 5개를 각각 관리해야 합니다. 게이트웨이를 사용하면 모두 하나의 구독을 통해 라우팅됩니다.

특정 게이트웨이 도구만 활성화할 수도 있습니다(예: 이미지 생성은 제외하고 웹 검색만 활성화). 자세한 내용은 아래의 [게이트웨이와 자체 백엔드 혼합](#mixing-the-gateway-with-your-own-backends)을 참고하세요.

### dotfile에 자격 증명을 저장하지 않음

모든 것이 OAuth로 인증된 하나의 Portal 세션을 통해 라우팅되므로, 오래 유지되는 API 키가 십여 개 들어 있는 `.env` 파일이 쌓이지 않습니다. 디스크에 저장되는 유일한 자격 증명은 `~/.hermes/auth.json`의 refresh token이며, Hermes는 요청마다 이 토큰으로 단기 JWT를 발급합니다. 자세한 내용은 아래의 [토큰 처리](#token-handling)를 참고하세요.

### 크로스 플랫폼 동등성

[Native Windows](/user-guide/windows-native)에서는 도구별 API 키 설정이 가장 번거로운 부분입니다. Firecrawl 계정, FAL 계정, Browser Use 계정, Windows에서 사용할 OpenAI 키를 각각 설치하는 일이 유용한 에이전트를 갖추는 데 가장 큰 마찰을 일으킵니다. Portal 구독은 이를 간소화합니다. 하나의 OAuth로 모델과 모든 게이트웨이 도구를 사용할 수 있으므로, Windows 사용자도 네 개의 백엔드를 직접 설정하지 않고 macOS/Linux와 같은 경험을 얻습니다.

## Hermes 4에 대한 참고

Nous Research 자체의 **Hermes 4** 제품군(Hermes-4-70B, Hermes-4-405B)은 Portal에서 크게 할인된 요금으로 이용할 수 있습니다. 이 모델들은 수학, 과학, 지시 따르기, 스키마 준수, 역할극, 장문 작성에 강한 **최첨단 하이브리드 추론 채팅 모델**입니다.

하지만 **Hermes Agent 내부에서 사용하는 것은 권장하지 않습니다**. Hermes 4는 에이전트가 의존하는 빠른 도구 호출 루프가 아니라 채팅과 추론에 맞춰 조정되었습니다. 연구 워크플로에 사용하거나 다른 도구에서 [구독 프록시](/user-guide/features/subscription-proxy)를 통해 사용하세요. 에이전트 작업에는 대신 카탈로그에서 최첨단 에이전트 모델을 선택하세요.

```bash
/model anthropic/claude-sonnet-4.6     # best general-purpose agentic model
/model openai/gpt-5.5-pro              # strong reasoning + tool calling
/model google/gemini-3-pro-preview     # huge context window
/model deepseek/deepseek-v4-pro        # cost-effective coder
```

Portal의 자체 [모델 정보 페이지](https://portal.nousresearch.com/info)에도 같은 경고가 있으므로, 이는 Hermes 측의 의견이 아니라 Nous Research의 공식 안내입니다.

## 설정

### 새로 설치하기 — 한 번의 명령

```bash
hermes setup --portal
```

이 명령 하나로 전체 설정을 실행합니다.

1. OAuth 로그인을 위해 브라우저에서 portal.nousresearch.com을 엽니다.
2. refresh token을 `~/.hermes/auth.json`에 저장합니다.
3. 선별된 목록에서 Nous 모델을 선택하게 합니다(또는 건너뛰어 현재 모델을 유지합니다).
4. 모델을 선택하면 `~/.hermes/config.yaml`에서 Nous를 추론 제공업체로 설정합니다.
5. Tool Gateway(웹, 이미지, TTS, 브라우저 라우팅)를 켭니다.
6. 터미널로 돌아와 `hermes chat`을 실행할 수 있게 합니다.

아직 구독이 없다면 먼저 [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription)에서 가입하세요.

### 기존 설치 — 다른 제공업체와 함께 Portal 추가

이미 OpenRouter, Anthropic 또는 다른 제공업체로 Hermes를 설정했고 Portal도 함께 추가하려는 경우 다음을 실행하세요.

```bash
hermes model
# pick "Nous Portal" from the provider list
# browser opens, sign in, done
```

기존 제공업체 설정은 그대로 유지됩니다. 세션 중간에는 `/model`로, 세션 사이에는 `hermes model`로 전환할 수 있습니다. Portal은 유일한 제공업체가 아니라 사용 가능한 제공업체 중 하나가 됩니다.

### 헤드리스 / SSH / 원격 설정

OAuth에는 브라우저가 필요하지만 loopback 콜백은 Hermes가 실행 중인 시스템에서 동작합니다. 원격 호스트의 경우 [SSH를 통한 OAuth / 원격 호스트](/guides/oauth-over-ssh)를 참고하세요. 동일한 방식이 다른 OAuth 기반 제공업체와 Portal에도 적용됩니다(`ssh -L` 포트 포워딩).

### 프로필 설정

[Hermes 프로필](/user-guide/profiles)을 사용하는 경우 Portal refresh token은 공유 토큰 저장소를 통해 모든 프로필에서 자동으로 공유됩니다. 어떤 프로필에서든 한 번 로그인하면 나머지 프로필이 자동으로 이를 사용하므로 프로필마다 OAuth 흐름을 반복할 필요가 없습니다.

## Portal을 일상적으로 사용하기

### 연결된 항목 확인

```bash
hermes portal            # log in to Nous Portal + set it up (one-shot onboarding)
hermes portal info       # login status, subscription info, model + gateway routing
hermes portal status     # alias for `portal info`
hermes portal tools      # detailed Tool Gateway catalog with per-tool routing
hermes portal open       # open the subscription management page in your browser
```

하위 명령 없이 `hermes portal`을 실행하면 사람이 읽기 쉬운 `hermes auth add nous --type oauth`의 별칭입니다. 로그인하고 Nous 모델을 선택하게 하며, Nous를 추론 제공업체로 설정하고 Tool Gateway 사용 여부를 묻습니다(`hermes setup --portal`과 동일하며 최초 빠른 설정과 같은 Nous 흐름입니다).

`hermes portal info`는 다음과 같은 상위 수준의 개요를 제공합니다.

```
  Nous Portal
  ───────────
  Auth:    ✓ logged in
  Portal:  https://portal.nousresearch.com
  Model:   ✓ using Nous as inference provider

  Tool Gateway
  ────────────
  Web search & extract  via Nous Portal
  Image generation      via Nous Portal
  Text-to-speech        via Nous Portal
  Browser automation    via Nous Portal
  Cloud terminal        not configured
```

### 모델 전환

세션 내부에서 다음을 실행합니다.

```bash
/model anthropic/claude-sonnet-4.6
/model openai/gpt-5.5-pro
/model google/gemini-3-pro-preview
```

또는 선택기를 엽니다.

```bash
/model
# arrow keys, enter to select
```

세션 외부에서(새 제공업체를 추가할 때 유용한 전체 설정 마법사):

```bash
hermes model
```

### 게이트웨이와 자체 백엔드 혼합

이미 Browserbase 계정이 있고 이를 계속 사용하면서 웹 검색과 이미지 생성은 Nous를 통해 라우팅하려는 경우에도 지원됩니다. `hermes tools`를 사용해 도구별 백엔드를 선택하세요.

```bash
hermes tools
# → Web search       → "Nous Subscription"
# → Image generation → "Nous Subscription"
# → Browser          → "Browserbase"  (your existing key)
# → TTS              → "Nous Subscription"
```

Tool Gateway는 전체를 한꺼번에 켜거나 끄는 방식이 아니라 도구별로 선택할 수 있습니다. 관리형 백엔드는 Nous Portal 로그인 여부와 관계없이 `hermes tools`에 표시됩니다. 인증하기 전에 "Nous Subscription"을 선택하면 Hermes가 Portal 로그인을 인라인으로 실행합니다(추론 제공업체를 변경하거나 다른 도구를 건드리지 않습니다). 도구별 전체 설정 매트릭스는 [Tool Gateway 문서](/user-guide/features/tool-gateway)를 참고하세요.

### 구독 관리

언제든지 요금제를 관리하고 사용량을 확인하거나 업그레이드/취소할 수 있습니다.

- **웹:** [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription)
- **CLI 바로가기:** `hermes portal open`(기본 브라우저에서 같은 페이지를 엽니다)

## 설정 참고

`hermes setup --portal`을 실행한 후 `~/.hermes/config.yaml`은 다음과 같은 형태가 됩니다.

```yaml
model:
  provider: nous
  default: anthropic/claude-sonnet-4.6     # or whatever model you picked
  base_url: https://inference-api.nousresearch.com/v1
```

Tool Gateway 설정은 각 도구에 해당하는 섹션 아래에 있습니다.

```yaml
web:
  backend: firecrawl
  use_gateway: true   # web search/extract routes through Tool Gateway

image_gen:
  use_gateway: true

tts:
  provider: openai
  use_gateway: true

browser:
  cloud_provider: browser-use
  use_gateway: true
```

OAuth refresh token은 `~/.hermes/auth.json`에 별도로 저장됩니다(`config.yaml`에는 저장되지 않습니다. 자격 증명과 설정을 분리하는 것은 의도된 설계입니다).

## 토큰 처리

Hermes는 장기 API 키를 재사용하는 대신 저장된 Portal refresh token으로 각 추론 호출마다 단기 JWT를 발급합니다. 토큰 수명 주기는 완전히 자동으로 관리됩니다. 갱신, 발급, 일시적인 401 발생 시 재시도가 자동으로 수행되며 사용자가 토큰을 볼 일은 없습니다.

Portal이 refresh token을 무효화하면(비밀번호 변경, 수동 폐기, 세션 만료) 무효화된 refresh token은 **로컬에서 격리**됩니다. 따라서 Hermes가 이를 반복 사용하지 않아 동일한 401 오류가 계속 발생하지 않습니다. 다음 호출에서는 "재인증 필요"라는 명확한 메시지가 표시됩니다. `hermes auth add nous`를 실행해 다시 로그인하세요. 다음 로그인에 성공하면 격리가 해제됩니다.

## 문제 해결

### `hermes portal info`에 "not logged in"이 표시됨

OAuth 흐름을 완료하지 않았거나 refresh token이 삭제되었습니다. 다음을 실행하세요.

```bash
hermes portal
```

또는 `hermes model`을 사용해 Nous Portal을 다시 선택하세요.

### 세션 중간에 "re-authentication required" 메시지가 표시됨

Portal refresh token이 무효화되었습니다(비밀번호 변경, 수동 폐기 또는 세션 만료). `hermes auth add nous`를 실행하면 다음 요청에서 새 자격 증명을 사용합니다. 이전 토큰의 격리는 다시 로그인에 성공하면 자동으로 해제됩니다.

### Portal에서 제공하지 않는 특정 제공업체 모델을 사용하고 싶음

Portal은 각 모델을 적합한 백엔드로 라우팅합니다. 일부는 OpenRouter를 거치고, 일부는 독점 또는 보조 제공업체를 거치므로 OpenRouter가 지원하는 대부분의 모델을 일반적으로 이용할 수 있습니다. 특정 모델이 `/model`에 나타나지 않으면 OpenRouter 형식의 slug를 직접 입력해 보세요.

```bash
/model anthropic/claude-opus-4.6
```

모델이 실제로 누락된 경우 [이슈를 열어 주세요](https://github.com/NousResearch/hermes-agent/issues). Hermes에는 Portal의 카탈로그가 표시되며, 누락은 대개 업데이트할 수 있는 라우팅 설정을 의미합니다.

### Portal 계정에 청구 내역이 표시되지 않음

먼저 `hermes portal info`를 확인하세요. 다른 제공업체를 사용 중이라고 표시된다면(`Model: currently openrouter`가 `using Nous as inference provider` 대신 표시되는 경우) 로컬 설정이 변경된 것입니다. `hermes model`을 실행하고 Nous Portal을 선택하면 다음 요청부터 구독을 통해 라우팅됩니다.

## 함께 보기

- **[Tool Gateway](/user-guide/features/tool-gateway)** — 모든 게이트웨이 도구, 도구별 설정, 요금에 대한 전체 세부 정보
- **[구독 프록시](/user-guide/features/subscription-proxy)** — Hermes가 아닌 도구(다른 에이전트, 스크립트, 타사 클라이언트)에서 Portal 구독 사용
- **[음성 모드](/user-guide/features/voice-mode)** — Portal의 OpenAI TTS를 사용하는 음성 대화
- **[AI 제공업체](/integrations/providers)** — 대안을 비교하려는 경우 전체 제공업체 카탈로그
- **[SSH를 통한 OAuth](/guides/oauth-over-ssh)** — 원격 호스트 또는 브라우저만 사용할 수 있는 환경에서 로그인
- **[프로필](/user-guide/profiles)** — 하나의 Portal 로그인을 공유하는 여러 Hermes 설정
