---
sidebar_position: 1
title: "Nous Portal"
description: "하나의 구독, 300개 이상의 최첨단 모델, Tool Gateway 및 Nous Chat — Hermes Agent를 실행하는 권장 방법"
---

# Nous Portal

[Nous Portal](https://portal.nousresearch.com)은 Nous Research의 통합 구독 게이트웨이이며 **Hermes Agent를 실행하는 데 권장되는 방법**입니다. 하나의 OAuth 로그인으로 개별 계정, API 키, 그리고 모든 모델 연구소, 검색 API, 이미지 생성기 및 브라우저 프로바이더에 걸쳐 수동으로 설정해야 했던 번거로운 청구 관계를 해결합니다.

시간이 없어서 단 하나만 설정해야 한다면, 이것을 설정하세요. 가장 빠른 방법은 다음과 같습니다:

```bash
hermes setup --portal
```

이 단일 명령어는 Portal OAuth를 실행하고, Nous 모델을 선택하게 하며, `config.yaml`에서 인퍼런스 프로바이더를 Nous로 설정하고, Tool Gateway를 켭니다. 이 과정을 마치면 즉시 `hermes chat`을 사용할 준비가 됩니다.

아직 구독하지 않으셨나요? [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription)에서 가입한 후, 다시 돌아와 위의 명령어를 실행하세요.

## 구독에 포함된 내용

### 300개 이상의 최첨단 모델을 단일 청구로

Portal은 에코시스템 전반에 걸쳐 선별된 에이전트형 모델 카탈로그를 프록시합니다 — 각 연구소별로 크레딧 잔액을 관리하는 대신 귀하의 Nous 구독에 통합하여 청구됩니다.

| 제품군 (Family) | 모델 |
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
| **Hermes** | Hermes-4-70B, Hermes-4-405B (채팅 용도, [아래 참고](#hermes-4에-대한-참고사항) 확인) |
| **+ 기타 모든 것** | 280개 이상의 추가 모델 — 에이전트형 최첨단 모델 전체 |

내부적으로는 OpenRouter를 통해 라우팅이 이루어지므로, 모델의 가용성과 장애 조치(failover) 동작은 OpenRouter 키를 통해 얻을 수 있는 것과 동일합니다 — 단지 귀하의 Nous 구독에 청구될 뿐입니다. 코드 작업 시에는 Claude Sonnet 4.6을, 긴 컨텍스트 작업 시에는 Gemini 3 Pro를 세션 중간에 `/model` 명령어로 쉽게 전환하세요. 새로운 자격 증명이나 금액 충전, 잔액 부족 오류로 당황할 필요가 없습니다.

### Nous Tool Gateway

동일한 구독으로 Hermes Agent의 도구 호출을 Nous 관리 인프라를 통해 라우팅하는 [Tool Gateway](/user-guide/features/tool-gateway)의 잠금을 해제할 수 있습니다. 5개의 백엔드를 한 번의 로그인으로 이용할 수 있습니다:

| 도구 | 파트너 | 기능 |
|------|---------|--------------|
| **웹 검색 및 추출** | Firecrawl | 에이전트 수준의 검색 및 전체 페이지 추출. Firecrawl API 키나 속도 제한 모니터링이 필요 없습니다. |
| **이미지 생성** | FAL | 단일 엔드포인트 아래 9개의 모델: FLUX 2 Klein 9B, FLUX 2 Pro, Z-Image Turbo, Nano Banana Pro (Gemini 3 Pro Image), GPT Image 1.5, GPT Image 2, Ideogram V3, Recraft V4 Pro, Qwen Image. |
| **텍스트 음성 변환(TTS)** | OpenAI TTS | 별도의 OpenAI 키 없이 사용하는 고품질 TTS. 메시징 플랫폼 전반에서 [음성 모드(voice mode)](/user-guide/features/voice-mode)를 활성화합니다. |
| **클라우드 브라우저 자동화** | Browser Use | `browser_navigate`, `browser_click`, `browser_type`, `browser_vision`을 위한 헤드리스 Chromium 세션. Browserbase 계정이 필요하지 않습니다. |
| **클라우드 터미널 샌드박스** | Modal | 코드 실행을 위한 서버리스 터미널 샌드박스 (선택적 애드온). |

게이트웨이 없이 이들을 각각 연결하려면 Firecrawl 계정, FAL 계정, Browser Use 계정, OpenAI 키, Modal 계정이 필요합니다. 즉, 5번의 가입, 5개의 개별 대시보드, 5번의 충전 절차가 필요합니다. 하지만 게이트웨이를 사용하면 이 모든 것이 하나의 구독을 통해 라우팅됩니다.

특정 게이트웨이 도구(예: 웹 검색은 활성화하지만 이미지 생성은 비활성화)만 활성화할 수도 있습니다 — 아래의 [게이트웨이와 자체 백엔드 혼합 사용](#게이트웨이와-자체-백엔드-혼합-사용)을 참조하세요.

### Nous Chat

귀하의 Portal 계정은 동일한 모델 카탈로그를 제공하는 Nous Research의 웹 채팅 인터페이스인 [chat.nousresearch.com](https://chat.nousresearch.com)도 지원합니다. 터미널을 사용할 수 없거나 에이전트가 아닌 일반적인 대화 작업에 유용합니다.

### 구성 파일(dotfiles)에 자격 증명 없음

모든 것이 하나의 OAuth 인증 Portal 세션을 통해 라우팅되므로, 12개가 넘는 수명이 긴 API 키로 가득 찬 `.env` 파일을 만들 필요가 없습니다. `~/.hermes/auth.json`에 있는 새로 고침 토큰(refresh token)이 디스크에 있는 유일한 자격 증명이며, Hermes는 요청마다 이 토큰을 사용해 수명이 짧은 JWT를 생성합니다 — 아래의 [토큰 처리](#토큰-처리)를 참조하세요.

### 크로스 플랫폼 동등성

[네이티브 Windows](/user-guide/windows-native)에서는 도구별 API 키 설정이 가장 까다로운 부분입니다. Windows에서 유용한 에이전트를 구성하려면 Firecrawl 계정, FAL 계정, Browser Use 계정, OpenAI 키를 설치해야 하는 과정이 가장 마찰이 심합니다. Portal 구독은 이를 원활하게 해줍니다. 하나의 OAuth로 모델과 모든 게이트웨이 도구를 처리하므로, Windows 사용자도 수동으로 4개의 백엔드를 구성할 필요 없이 macOS/Linux와 동일한 경험을 할 수 있습니다.

## Hermes 4에 대한 참고사항

Nous Research 자체의 **Hermes 4** 제품군(Hermes-4-70B, Hermes-4-405B)은 Portal을 통해 대폭 할인된 가격으로 제공됩니다. 이들은 수학, 과학, 지시 수행, 스키마 준수, 롤플레잉 및 긴 형식의 글쓰기에 강력한 **최첨단 하이브리드 추론 채팅 모델**입니다.

그러나 **Hermes Agent 내에서 사용하는 것은 권장되지 않습니다**. Hermes 4는 채팅과 추론에 맞춰 튜닝되었으며, 에이전트가 의존하는 빠른 도구 호출 루프에는 적합하지 않습니다. [Nous Chat](https://chat.nousresearch.com)이나 연구 워크플로우에, 또는 다른 도구의 [구독 프록시](/user-guide/features/subscription-proxy)를 통해 사용하세요. 그러나 에이전트 작업을 위해서는 카탈로그에서 다음의 최첨단 에이전트형 모델 중 하나를 선택하세요:

```bash
/model anthropic/claude-sonnet-4.6     # 최고의 다목적 에이전트 모델
/model openai/gpt-5.5-pro              # 강력한 추론 + 도구 호출 기능
/model google/gemini-3-pro-preview     # 거대한 컨텍스트 창
/model deepseek/deepseek-v4-pro        # 비용 효율적인 코더
```

Portal의 자체 [모델 정보 페이지](https://portal.nousresearch.com/info)에도 동일한 경고가 있으므로, 이는 Hermes 측의 개인적인 의견이 아닙니다 — Nous Research의 공식 지침입니다.

## 설정

### 신규 설치 — 단일 명령어

```bash
hermes setup --portal
```

이 명령어는 한 번에 전체 설정을 실행합니다:

1. OAuth 로그인을 위해 브라우저에서 portal.nousresearch.com을 엽니다.
2. 새로 고침 토큰을 `~/.hermes/auth.json`에 저장합니다.
3. 선별된 목록에서 Nous 모델을 선택하게 합니다 (또는 현재 모델을 유지하기 위해 건너뛸 수 있습니다).
4. 모델을 선택하면 `~/.hermes/config.yaml`에서 인퍼런스 프로바이더를 Nous로 설정합니다.
5. Tool Gateway (웹, 이미지, TTS, 브라우저 라우팅)를 켭니다.
6. `hermes chat`을 할 수 있는 준비된 상태로 터미널로 돌아갑니다.

아직 구독하지 않았다면 [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription)에서 먼저 가입하세요.

### 기존 설치 — 다른 프로바이더와 함께 Portal 추가

이미 OpenRouter, Anthropic 또는 다른 프로바이더로 Hermes를 구성한 상태에서 이들과 함께 Portal을 추가하려면:

```bash
hermes model
# 프로바이더 목록에서 "Nous Portal"을 선택합니다.
# 브라우저가 열리면 로그인하여 완료합니다.
```

기존의 프로바이더 설정은 그대로 유지됩니다. 세션 중간에는 `/model`로, 세션 간에는 `hermes model`로 프로바이더를 전환할 수 있습니다 — Portal은 유일한 프로바이더가 아닌, 사용 가능한 프로바이더 중 하나가 됩니다.

### 헤드리스 / SSH / 원격 설정

OAuth는 브라우저가 필요하지만 루프백 콜백은 Hermes가 실행 중인 머신에서 작동합니다. 원격 호스트의 경우 [SSH/원격 호스트를 통한 OAuth](/guides/oauth-over-ssh)를 참조하세요 — Portal에도 다른 OAuth 기반 프로바이더와 동일한 패턴이 적용됩니다 (Cloud Shell / Codespaces와 같은 브라우저 전용 환경을 위한 `ssh -L` 포트 포워딩, `--manual-paste`).

### 프로필 설정

[Hermes 프로필](/user-guide/profiles)을 사용하는 경우 Portal 새로 고침 토큰은 공유 토큰 저장소를 통해 모든 프로필에 자동으로 공유됩니다. 프로필 중 하나에서 한 번만 로그인하면 나머지 프로필은 이를 자동으로 인식하므로, 프로필마다 OAuth 흐름을 반복할 필요가 없습니다.

## 일상적인 Portal 사용

### 연결 상태 점검

```bash
hermes portal            # Nous Portal 로그인 + 설정 (1회성 온보딩)
hermes portal info       # 로그인 상태, 구독 정보, 모델 + 게이트웨이 라우팅 확인
hermes portal tools      # 도구별 라우팅이 포함된 상세 Tool Gateway 카탈로그
hermes portal open       # 브라우저에서 구독 관리 페이지 열기
```

`hermes portal` (하위 명령 없음)은 `hermes auth add nous --type oauth`의 읽기 쉬운 별칭입니다. 로그인을 수행하고, Nous 모델을 선택하게 하며, Nous를 인퍼런스 프로바이더로 설정하고, Tool Gateway 옵트인(선택)을 제공합니다 (`hermes setup --portal`과 동일하며 첫 빠른 설정과 동일한 Nous 흐름).

`hermes portal info`는 다음과 같은 상위 수준의 개요를 제공합니다:

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

세션 내부에서:

```bash
/model anthropic/claude-sonnet-4.6
/model openai/gpt-5.5-pro
/model google/gemini-3-pro-preview
```

또는 모델 선택기를 엽니다:

```bash
/model
# 방향키를 사용하여 이동하고, 엔터키로 선택합니다.
```

세션 외부에서 (새로운 프로바이더를 추가할 때 유용한 전체 설정 마법사):

```bash
hermes model
```

### 게이트웨이와 자체 백엔드 혼합 사용

예를 들어 이미 Browserbase 계정이 있고 이를 계속 사용하면서 웹 검색과 이미지 생성은 Nous를 통해 라우팅하고 싶다면 이 역시 지원됩니다. `hermes tools`를 사용하여 도구별 백엔드를 선택하세요:

```bash
hermes tools
# → Web search       → "Nous Subscription"
# → Image generation → "Nous Subscription"
# → Browser          → "Browserbase"  (기존 사용하던 키)
# → TTS              → "Nous Subscription"
```

Tool Gateway는 전부 아니면 전무의 방식이 아니라 도구별로 선택(opt-in)할 수 있습니다. Nous Portal 로그인 여부와 관계없이 관리되는 백엔드는 `hermes tools`에 표시됩니다. 인증하기 전에 "Nous Subscription"을 선택하면 Hermes는 인라인으로 Portal 로그인을 실행합니다 (이는 기존 도구나 인퍼런스 프로바이더 설정을 변경하지 않습니다). 전체 도구별 구성 매트릭스는 [Tool Gateway 문서](/user-guide/features/tool-gateway)를 참조하세요.

### 구독 관리

플랜 관리, 사용량 확인, 업그레이드/취소를 언제든지 할 수 있습니다:

- **웹:** [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription)
- **CLI 바로가기:** `hermes portal open` (기본 브라우저에서 동일한 페이지를 엽니다)

## 구성 참조

`hermes setup --portal`을 실행한 후, `~/.hermes/config.yaml`은 다음과 같이 표시됩니다:

```yaml
model:
  provider: nous
  default: anthropic/claude-sonnet-4.6     # 또는 선택한 다른 모델
  base_url: https://inference-api.nousresearch.com/v1
```

Tool Gateway 설정은 각각의 해당 도구 섹션 아래에 위치합니다:

```yaml
web:
  backend: nous       # 웹 검색/추출이 Tool Gateway를 통해 라우팅됩니다.

image_gen:
  provider: nous

tts:
  provider: nous

browser:
  backend: nous
```

OAuth 새로 고침 토큰은 별도의 `~/.hermes/auth.json`에 저장됩니다 (의도적으로 자격 증명과 구성을 분리하기 위해 `config.yaml`에는 저장되지 않습니다).

## 토큰 처리

Hermes는 수명이 긴 API 키를 재사용하는 대신, 인퍼런스를 호출할 때마다 저장된 Portal 새로 고침 토큰에서 수명이 짧은 JWT를 생성합니다. 토큰 수명 주기(새로 고침, 생성, 일시적인 401 오류 시 재시도)는 완전히 자동으로 이루어지며, 사용자에게는 노출되지 않습니다.

Portal이 새로 고침 토큰을 무효화하는 경우 (비밀번호 변경, 수동 취소, 세션 만료), 무효화된 새로 고침 토큰은 **로컬에 격리**되므로 Hermes는 재사용을 중지하고 동일한 401 오류가 연속적으로 표시되지 않도록 합니다. 다음 번 호출 시에는 "재인증 필요"라는 명확한 메시지가 나타납니다. `hermes auth add nous`를 실행하여 다시 로그인하세요; 성공적으로 로그인하면 격리가 해제됩니다.

## 문제 해결

### `hermes portal info`에 "not logged in"으로 표시됨

OAuth 흐름을 완료하지 않았거나 새로 고침 토큰이 지워졌습니다. 다음을 실행하세요:

```bash
hermes portal
```

또는 `hermes model`을 사용하고 Nous Portal을 다시 선택하세요.

### 세션 중 "re-authentication required" 메시지를 받음

비밀번호 변경, 수동 취소 또는 세션 만료로 인해 Portal 새로 고침 토큰이 무효화되었습니다. `hermes auth add nous`를 실행하면 다음 요청에서 새 자격 증명을 사용합니다. 성공적으로 재로그인하면 이전 토큰의 격리 상태가 자동으로 해제됩니다.

### Portal에 없는 특정 프로바이더 모델을 사용하고 싶음

Portal은 OpenRouter를 통해 프록시하므로, OpenRouter가 지원하는 모든 모델은 일반적으로 사용할 수 있습니다. 특정 모델이 `/model`에 나타나지 않는다면 OpenRouter 형식의 슬러그(slug)를 직접 시도해 보세요:

```bash
/model anthropic/claude-opus-4.6
```

정말로 누락된 모델이 있다면 [이슈를 등록](https://github.com/NousResearch/hermes-agent/issues)해 주세요 — 저희는 Portal의 카탈로그를 Hermes에 표면화하며, 누락된 부분은 대개 라우팅 구성을 업데이트하여 해결할 수 있습니다.

### 청구 내역이 내 Portal 계정에 나타나지 않음

먼저 `hermes portal info`를 확인하세요 — 다른 프로바이더를 사용 중이라고 표시된다면 (`using Nous as inference provider` 대신 `Model: currently openrouter` 등), 로컬 구성이 변경된 것입니다. `hermes model`을 실행하여 Nous Portal을 선택하면, 다음 요청부터는 귀하의 구독을 통해 라우팅됩니다.

## 함께 보기

- **[Tool Gateway](/user-guide/features/tool-gateway)** — 모든 게이트웨이 도구, 도구별 구성 및 가격에 대한 전체 세부 정보
- **[Subscription proxy(구독 프록시)](/user-guide/features/subscription-proxy)** — Hermes가 아닌 다른 도구 (다른 에이전트, 스크립트, 타사 클라이언트)에서 Portal 구독 사용
- **[Voice mode(음성 모드)](/user-guide/features/voice-mode)** — Portal의 OpenAI TTS를 사용하는 음성 대화
- **[AI Providers](/integrations/providers)** — 대안을 비교하고 싶은 경우 전체 프로바이더 카탈로그
- **[OAuth over SSH](/guides/oauth-over-ssh)** — 원격 호스트 또는 브라우저 전용 환경에서의 로그인
- **[Profiles](/user-guide/profiles)** — 하나의 Portal 로그인을 공유하는 여러 Hermes 구성
