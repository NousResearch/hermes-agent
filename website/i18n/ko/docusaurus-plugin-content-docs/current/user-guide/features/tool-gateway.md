---
title: "Nous 도구 게이트웨이"
description: "하나의 구독으로 모든 도구를 이용하세요. 웹 검색, 이미지 생성, TTS, 클라우드 브라우저를 추가 API 키 없이 모두 Nous Portal을 통해 라우팅합니다."
sidebar_label: "도구 게이트웨이"
sidebar_position: 2
---

# Nous 도구 게이트웨이

**구독 하나. 모든 도구가 기본 제공됩니다.**

도구 게이트웨이는 모든 유료 [Nous Portal](https://portal.nousresearch.com) 구독에 포함됩니다. Hermes의 도구 호출(웹 검색, 이미지 생성, 텍스트 음성 변환, 클라우드 브라우저 자동화)을 Nous가 이미 운영 중인 인프라를 통해 라우팅하므로, 에이전트를 유용하게 만들기 위해 Firecrawl, FAL, OpenAI, Browser Use 또는 다른 서비스에 따로 가입할 필요가 없습니다.

<div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap', margin: '1.5rem 0'}}>
  <a href="https://portal.nousresearch.com/manage-subscription" style={{background: 'var(--ifm-color-primary)', color: 'white', padding: '0.75rem 1.5rem', borderRadius: '6px', textDecoration: 'none', fontWeight: 'bold'}}>구독 시작 또는 관리 →</a>
</div>

## 포함된 기능

| | 도구 | 제공되는 기능 |
|---|---|---|
| 🔍 | **웹 검색 및 추출** | Firecrawl을 통한 에이전트급 웹 검색과 전체 페이지 추출. 게이트웨이가 확장을 처리하므로 속도 제한을 걱정할 필요가 없습니다. |
| 🎨 | **이미지 생성** | 하나의 엔드포인트에서 9개 모델 사용: **FLUX 2 Klein 9B**, **FLUX 2 Pro**, **Z-Image Turbo**, **Nano Banana Pro** (Gemini 3 Pro Image), **GPT Image 1.5**, **GPT Image 2**, **Ideogram V3**, **Recraft V4 Pro**, **Qwen Image**. 플래그로 생성마다 선택하거나 Hermes가 기본으로 FLUX 2 Klein을 사용하게 할 수 있습니다. |
| 🔊 | **텍스트 음성 변환** | `text_to_speech` 도구에 연결된 OpenAI TTS 음성. Telegram에 음성 메모를 보내고, 파이프라인용 오디오를 생성하고, 무엇이든 내레이션할 수 있습니다. |
| 🌐 | **클라우드 브라우저 자동화** | Browser Use를 통한 헤드리스 Chromium 세션. `browser_navigate`, `browser_click`, `browser_type`, `browser_vision` 등 에이전트를 구동하는 모든 기본 기능을 Browserbase 계정 없이 사용할 수 있습니다. |

네 가지 모두 Nous 구독에 사용량 기준으로 청구됩니다. 원하는 조합을 사용하세요. 웹과 이미지에는 게이트웨이를 사용하면서 TTS에는 자체 ElevenLabs 키를 유지하거나, 모든 기능을 Nous를 통해 라우팅할 수 있습니다.

## 이 기능이 있는 이유

실제로 *무언가를 할 수 있는* 에이전트를 만들려면 각기 별도의 가입, 속도 제한, 청구, 특성을 가진 5개 이상의 API 구독을 이어 붙여야 합니다. 게이트웨이는 이를 하나의 계정으로 통합합니다.

- **청구서 하나.** Nous에 결제하면 나머지는 저희가 처리합니다.
- **가입 한 번.** 관리해야 할 Firecrawl, FAL, Browser Use 또는 OpenAI 오디오 계정이 없습니다.
- **키 하나.** Nous Portal OAuth 하나로 모든 도구를 사용할 수 있습니다.
- **동일한 품질.** 직접 키를 사용하는 경로와 동일한 백엔드이며, 저희가 앞단에서 연결해 둔 것뿐입니다.

언제든 자체 키를 도구별로 원하는 때에 가져올 수 있습니다. 게이트웨이는 종속이 아니라 지름길입니다.

## 시작하기

진입 방법은 세 가지입니다. 현재 상황에 맞는 방법을 선택하세요.

```bash
hermes setup --portal     # Fresh install: Nous OAuth + set Nous as provider + turn on the Tool Gateway in one go
```

```bash
hermes model              # Switch your inference provider to Nous Portal — Hermes then offers to turn on the gateway for all tools
```

```bash
hermes tools              # Enable the gateway per-tool — pick "Nous Subscription" for any tool you want
```

`hermes setup --portal`과 `hermes model`은 한 번에 처리하는 경로입니다. 한 번 로그인하고, 원하면 모든 도구를 게이트웨이로 전환합니다. `hermes tools`는 맞춤 선택 경로로, 원하는 도구만 한 번에 하나씩 켭니다.

**먼저 로그인할 필요는 없습니다.** `hermes tools`에서는 Nous Portal에 로그인한 적이 없어도 Nous가 관리하는 백엔드(웹 검색, 이미지, 비디오, TTS, 브라우저)가 항상 표시됩니다. 하나를 선택하면 아직 인증되지 않은 경우 Hermes가 그 자리에서 Portal 로그인을 실행하므로, 미리 `hermes model`을 실행할 필요가 없습니다. Nous OAuth가 이미 활성화되어 있다면 백엔드를 선택하는 즉시 추가 질문 없이 활성화됩니다. 이 경로는 로그인하고 선택한 도구 하나를 켜기만 하며, 추론 제공자를 전환하지 않고 다른 모든 도구에 게이트웨이를 활성화할지 묻지도 않습니다.

언제든 현재 활성 상태를 확인하세요.

```bash
hermes portal info        # Portal auth + Tool Gateway routing summary
hermes portal tools       # Gateway catalog with current routing per tool
hermes status             # Full system status (Tool Gateway is one section)
```

`hermes portal info`는 다음과 같은 섹션을 표시합니다.

```
◆ Nous Tool Gateway
  Nous Portal     ✓ managed tools available
  Web tools       ✓ active via Nous subscription
  Image gen       ✓ active via Nous subscription
  TTS             ✓ active via Nous subscription
  Browser         ○ active via Browser Use key
```

"active via Nous subscription"으로 표시된 도구는 게이트웨이를 통해 처리됩니다. 그 외 도구는 자체 키를 사용합니다.

## 이용 자격

도구 게이트웨이는 **유료 구독** 기능입니다. Nous 무료 계정은 추론에 Portal을 사용할 수 있지만 관리형 도구는 포함하지 않습니다. 게이트웨이를 사용하려면 [요금제를 업그레이드하세요](https://portal.nousresearch.com/manage-subscription).

일부 계정에는 **무료 도구 풀**도 제공됩니다. 이는 유료 구독 없이 게이트웨이 도구 호출을 처리하는 소규모 관리형 도구 사용량입니다. 무료 풀이 제공되는 경우 게이트웨이가 이를 표시하고 최초 사용 시 설정 안내를 보여 주므로, 바로 동의하고 관리형 도구를 사용할 수 있습니다.

## 조합해서 사용하기

게이트웨이는 도구별로 적용됩니다. 원하는 기능에만 켜세요.

- **모든 도구를 Nous를 통해 사용** — 가장 간단합니다. 구독 하나면 끝입니다.
- **웹과 이미지에는 게이트웨이, TTS는 자체 사용** — ElevenLabs 음성은 유지하고 나머지는 Nous에 맡깁니다.
- **키가 없는 기능에만 게이트웨이 사용** — "Browserbase에는 이미 돈을 내고 있지만 Firecrawl 계정은 만들고 싶지 않다"는 경우에도 문제없이 사용할 수 있습니다.

다음 명령으로 언제든 도구를 전환할 수 있습니다.

```bash
hermes tools          # Interactive picker for each tool category
```

도구를 선택하고 제공자로 **Nous Subscription**(또는 원하는 직접 제공자)을 선택하세요. 설정 파일을 편집할 필요가 없습니다. 아직 Nous Portal에 로그인하지 않았다면 **Nous Subscription**을 선택할 때 Portal 로그인이 바로 시작되므로, 먼저 `hermes model`을 통해 인증할 필요가 없습니다.

## 개별 이미지 모델 사용

이미지 생성은 속도를 위해 기본적으로 FLUX 2 Klein 9B를 사용합니다. `image_generate` 도구에 모델 ID를 전달하여 호출별로 재정의하세요.

| 모델 | ID | 적합한 용도 |
|---|---|---|
| FLUX 2 Klein 9B | `fal-ai/flux-2/klein/9b` | 빠른 생성, 좋은 기본값 |
| FLUX 2 Pro | `fal-ai/flux-2-pro` | 더 높은 충실도의 FLUX |
| Z-Image Turbo | `fal-ai/z-image/turbo` | 스타일화된 이미지, 빠른 생성 |
| Nano Banana Pro | `fal-ai/nano-banana-pro` | Google Gemini 3 Pro Image |
| GPT Image 1.5 | `fal-ai/gpt-image-1.5` | OpenAI 이미지 생성, 텍스트+이미지 |
| GPT Image 2 | `fal-ai/gpt-image-2` | OpenAI 최신 모델 |
| Ideogram V3 | `fal-ai/ideogram/v3` | 뛰어난 프롬프트 준수 및 타이포그래피 |
| Recraft V4 Pro | `fal-ai/recraft/v4/pro/text-to-image` | 벡터 스타일, 그래픽 디자인 |
| Qwen Image | `fal-ai/qwen-image` | Alibaba 멀티모달 |

목록은 계속 변경됩니다. `hermes tools` → Image Generation에서 현재 목록을 확인하세요.

---

## 설정 참고

대부분의 사용자는 이를 직접 수정할 필요가 없습니다. `hermes model`과 `hermes tools`가 모든 작업 흐름을 대화형으로 처리합니다. 이 섹션은 `config.yaml`을 직접 작성하거나 설정을 스크립팅할 때 사용합니다.

### 도구별 `use_gateway` 플래그

각 도구의 설정 블록은 `use_gateway` 불리언을 받습니다.

```yaml
web:
  backend: firecrawl
  use_gateway: true

image_gen:
  use_gateway: true

tts:
  provider: openai
  use_gateway: true

browser:
  cloud_provider: browser-use
  use_gateway: true
```

우선순위: `use_gateway: true`는 `.env`에 직접 키가 있더라도 Nous를 통해 라우팅합니다. `use_gateway: false`(또는 미설정)는 사용 가능한 직접 키를 사용하고, 직접 키가 없을 때만 게이트웨이로 대체합니다.

### 게이트웨이 비활성화

```yaml
web:
  use_gateway: false   # Hermes now uses FIRECRAWL_API_KEY from .env
```

게이트웨이가 아닌 제공자를 선택하면 `hermes tools`가 자동으로 플래그를 지우므로, 보통은 이 작업도 자동으로 처리됩니다.

### 자체 호스팅 게이트웨이(고급)

자체 Nous 호환 게이트웨이를 운영하시나요? `~/.hermes/.env`에서 엔드포인트를 재정의하세요.

```bash
TOOL_GATEWAY_DOMAIN=your-domain.example.com
TOOL_GATEWAY_SCHEME=https
TOOL_GATEWAY_USER_TOKEN=your-token        # normally auto-populated from Portal login
FIRECRAWL_GATEWAY_URL=https://...         # override one endpoint specifically
```

이 설정은 맞춤 인프라 환경(엔터프라이즈 배포, 개발 환경)을 위한 것입니다. 일반 구독자는 설정할 필요가 없습니다.

## FAQ

### Telegram / Discord / 다른 메시징 게이트웨이에서도 작동하나요?

예. 도구 게이트웨이는 CLI가 아니라 도구 실행 계층에서 작동합니다. 도구를 호출할 수 있는 모든 인터페이스(CLI, Telegram, Discord, Slack, IRC, Teams, API 서버 등)는 투명하게 이 기능의 혜택을 받습니다.

### 구독이 만료되면 어떻게 되나요?

게이트웨이를 통해 라우팅되는 도구는 구독을 갱신하거나 `hermes tools`를 통해 직접 API 키로 바꿀 때까지 작동하지 않습니다. Hermes가 Portal을 안내하는 명확한 오류를 표시합니다.

### 도구별 사용량이나 비용을 볼 수 있나요?

예. [Nous Portal 대시보드](https://portal.nousresearch.com)에서 도구별로 사용량을 나누어 보여 주므로 어떤 도구가 청구액을 늘리는지 확인할 수 있습니다.

### Modal(서버리스 터미널)도 포함되나요?

Modal은 기본 도구 게이트웨이 묶음의 일부가 아니라 Nous 구독을 통한 **선택적 추가 기능**입니다. 셸 실행을 위한 원격 샌드박스가 필요할 때 `hermes setup terminal` 또는 `config.yaml`에서 직접 설정하세요.

### 게이트웨이를 활성화할 때 기존 API 키를 삭제해야 하나요?

아니요. 키를 `.env`에 그대로 두세요. `use_gateway: true`이면 Hermes가 직접 키를 건너뛰고 게이트웨이를 사용합니다. 플래그를 `false`로 되돌리면 키가 다시 소스로 사용됩니다. 게이트웨이는 종속이 아닙니다.
