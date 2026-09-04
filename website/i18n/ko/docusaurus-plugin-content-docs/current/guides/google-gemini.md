---
sidebar_position: 16
title: "Google Gemini"
description: "Google Gemini와 Hermes Agent 사용 — 네이티브 AI Studio API, API 키 설정, 도구 호출, 스트리밍 및 할당량 안내"
---

# Google Gemini

Hermes Agent는 **Google AI Studio / Gemini API**를 네이티브 공급자로 지원합니다 — OpenAI 호환 엔드포인트가 아닙니다. 이를 통해 Hermes는 도구 호출, 스트리밍, 멀티모달 입력 및 Gemini 전용 응답 메타데이터를 유지하면서 내부 OpenAI 형태의 메시지와 도구 루프를 Gemini의 네이티브 `generateContent` API로 변환할 수 있습니다.

## 사전 요구 사항

- **Google AI Studio API 키** — [aistudio.google.com/apikey](https://aistudio.google.com/apikey)에서 생성
- **결제가 활성화된 Google Cloud 프로젝트** — 에이전트 사용에 권장됩니다. Hermes가 사용자 턴마다 여러 모델 호출을 수행할 수 있으므로 Gemini 무료 등급은 장시간 에이전트 세션에 너무 작습니다.
- **Hermes 설치** — 네이티브 Gemini 공급자에는 추가 Python 패키지가 필요하지 않습니다.

:::tip API 키 경로
`GOOGLE_API_KEY` 또는 `GEMINI_API_KEY`를 설정하세요. Hermes는 `gemini` 공급자에 대해 두 이름을 모두 확인합니다.
:::

## 빠른 시작

```bash
# Add your Gemini API key
echo "GOOGLE_API_KEY=..." >> ~/.hermes/.env

# Select Gemini as your provider
hermes model
# → Choose "More providers..." → "Google AI Studio"
# → Hermes checks your key tier and shows Gemini models
# → Select a model

# Start chatting
hermes chat
```

직접 설정을 편집하려면 네이티브 Gemini API 기본 URL을 사용하세요.

```yaml
model:
  default: gemini-3-flash-preview
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## 설정

`hermes model`을 실행하면 `~/.hermes/config.yaml`에 다음이 포함됩니다.

```yaml
model:
  default: gemini-3-flash-preview
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

그리고 `~/.hermes/.env`에는 다음이 포함됩니다.

```bash
GOOGLE_API_KEY=...
```

### 네이티브 Gemini API

권장 엔드포인트는 다음과 같습니다.

```text
https://generativelanguage.googleapis.com/v1beta
```

Hermes는 이 엔드포인트를 감지하고 네이티브 Gemini 어댑터를 생성합니다. 내부적으로 Hermes는 여전히 에이전트 루프에서 OpenAI 형태의 메시지를 유지한 다음 각 요청을 Gemini의 네이티브 스키마로 변환합니다.

- `messages[]` → Gemini `contents[]`
- 시스템 프롬프트 → Gemini `systemInstruction`
- 도구 스키마 → Gemini `functionDeclarations`
- 도구 결과 → Gemini `functionResponse` parts
- 스트리밍 응답 → Hermes 루프를 위한 OpenAI 형태의 스트림 청크

:::note Gemini 3 사고 서명
Gemini 3 도구 사용의 경우 Hermes는 함수 호출 파트에 연결된 `thoughtSignature` 값을 보존하고 다음 도구 턴에서 다시 전달합니다. 이를 통해 다단계 에이전트 워크플로의 검증에 중요한 경로를 처리합니다.

Gemini 3는 다른 응답 파트에도 사고 서명을 연결할 수 있습니다. Hermes의 네이티브 어댑터는 현재 에이전트 도구 루프에 최적화되어 있으므로 모든 비도구 호출 서명을 파트 수준의 완전한 충실도로 아직 다시 전달하지는 않습니다.
:::

### 네이티브 엔드포인트 우선 사용

Google은 OpenAI 호환 엔드포인트도 제공합니다.

```text
https://generativelanguage.googleapis.com/v1beta/openai/
```

Hermes 에이전트 세션에서는 위의 네이티브 Gemini 엔드포인트를 우선 사용하세요. Hermes에는 다중 턴 도구 사용, 도구 호출 결과, 스트리밍, 멀티모달 입력 및 Gemini 응답 메타데이터를 Gemini의 `generateContent` API에 직접 매핑할 수 있는 네이티브 Gemini 어댑터가 포함되어 있습니다. OpenAI 호환 엔드포인트는 OpenAI API 호환성이 특별히 필요할 때 여전히 유용합니다.

이전에 `GEMINI_BASE_URL`을 `/openai` URL로 설정했다면 제거하거나 변경하세요.

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

## 사용 가능한 모델

`hermes model` 선택기에는 Hermes의 공급자 레지스트리에서 관리하는 Gemini 모델이 표시됩니다. 일반적인 선택지는 다음과 같습니다.

| 모델 | ID | 참고 |
|-------|----|-------|
| Gemini 3.1 Pro Preview | `gemini-3.1-pro-preview` | 사용 가능한 경우 가장 뛰어난 프리뷰 모델 |
| Gemini 3 Pro Preview | `gemini-3-pro-preview` | 강력한 추론 및 코딩 모델 |
| Gemini 3 Flash Preview | `gemini-3-flash-preview` | 속도와 성능의 균형을 이루는 권장 기본값 |
| Gemini 3.1 Flash Lite Preview | `gemini-3.1-flash-lite-preview` | 사용 가능한 경우 가장 빠르고 비용이 낮은 옵션 |

모델 사용 가능 여부는 시간이 지나면서 변경됩니다. 모델이 사라졌거나 키에서 활성화되지 않았다면 `hermes model`을 다시 실행하고 현재 목록에서 하나를 선택하세요.

:::info 모델 ID
`provider: gemini`일 때는 `gemini-3-flash-preview` 같은 Gemini 네이티브 모델 ID를 사용하고, `google/gemini-3-flash-preview` 같은 OpenRouter 스타일 ID는 사용하지 마세요.
:::

### 최신 별칭

Google은 Pro 및 Flash Gemini 제품군을 위한 변경 가능한 별칭을 게시합니다. Hermes 설정을 변경하지 않고 Google이 모델을 자동으로 업데이트하도록 하려는 경우 `gemini-pro-latest`와 `gemini-flash-latest`가 유용합니다.

| 별칭 | 현재 추적 대상 | 참고 |
|-------|----------------|-------|
| `gemini-pro-latest` | 최신 Gemini Pro 모델 | Google의 현재 Pro 기본값을 사용하려는 경우 적합 |
| `gemini-flash-latest` | 최신 Gemini Flash 모델 | Google의 현재 Flash 기본값을 사용하려는 경우 적합 |

```yaml
model:
  default: gemini-pro-latest
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

엄격한 재현성이 필요하다면 `gemini-3.1-pro-preview` 또는 `gemini-3-flash-preview` 같은 명시적 모델 ID를 우선 사용하세요.

### Gemini API를 통한 Gemma

Google은 Gemini API를 통해 Gemma 모델도 제공합니다. Hermes는 이를 Google 모델로 인식하지만, 새 사용자가 장시간 에이전트 세션에 평가 등급 모델을 실수로 선택하지 않도록 처리량이 매우 낮은 Gemma 항목은 기본 모델 선택기에서 숨깁니다.

유용한 평가용 ID는 다음과 같습니다.

| 모델 | ID | 참고 |
|-------|----|-------|
| Gemma 4 31B IT | `gemma-4-31b-it` | 호환성 및 품질 평가에 유용한 더 큰 Gemma 모델 |
| Gemma 4 26B A4B IT | `gemma-4-26b-a4b-it` | 사용 가능한 경우 더 작은 활성 매개변수 변형 |

이 모델은 Gemini API 키에서 평가용 옵션으로 사용하는 것이 가장 적합합니다. Google의 Gemma API 요금은 무료 등급만 제공되며 사용량 제한이 프로덕션 Gemini 모델보다 낮으므로, Hermes 에이전트를 지속적으로 사용하려면 일반적으로 유료 Gemini 모델, 자체 호스팅 배포 또는 적절한 할당량을 제공하는 다른 공급자로 전환해야 합니다.

선택기에서 숨겨진 Gemma 모델을 사용하려면 직접 설정하세요.

```yaml
model:
  default: gemma-4-31b-it
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## 세션 중 모델 전환

대화 중 `/model` 명령을 사용하세요.

```text
/model gemini-3-flash-preview
/model gemini-flash-latest
/model gemini-3-pro-preview
/model gemini-pro-latest
/model gemma-4-31b-it
/model gemini-3.1-flash-lite-preview
```

아직 Gemini를 설정하지 않았다면 세션을 종료하고 먼저 `hermes model`을 실행하세요. `/model`은 이미 설정된 공급자와 모델 사이를 전환하며, 새 API 키를 수집하지는 않습니다.

## 진단

```bash
hermes doctor
```

doctor는 다음을 확인합니다.

- `GOOGLE_API_KEY` 또는 `GEMINI_API_KEY`를 사용할 수 있는지
- 설정된 공급자 인증 정보를 확인할 수 있는지

## Gateway(메시징 플랫폼)

Gemini는 모든 Hermes Gateway 플랫폼(Telegram, Discord, Slack, WhatsApp, LINE, Feishu 등)에서 작동합니다. Gemini를 공급자로 설정한 다음 평소처럼 Gateway를 시작하세요.

```bash
hermes gateway setup
hermes gateway start
```

Gateway는 `config.yaml`을 읽고 동일한 Gemini 공급자 설정을 사용합니다.

## 문제 해결

### "Gemini native client requires an API key"

Hermes가 사용할 수 있는 API 키를 찾지 못했습니다. 다음 중 하나를 `~/.hermes/.env`에 추가하세요.

```bash
GOOGLE_API_KEY=...
# or
GEMINI_API_KEY=...
```

그런 다음 `hermes model`을 다시 실행하세요.

### "This Google API key is on the free tier"

Hermes는 설정 중 Gemini API 키를 확인합니다. 도구 사용, 재시도, 압축 및 보조 작업에 여러 모델 호출이 필요할 수 있으므로 무료 등급 할당량은 에이전트를 몇 차례 실행한 후 소진될 수 있습니다.

키가 연결된 Google Cloud 프로젝트에서 결제를 활성화하고, 필요한 경우 키를 다시 생성한 다음 다음을 실행하세요.

```bash
hermes model
```

### "404 model not found"

선택한 모델을 계정, 리전 또는 키에서 사용할 수 없습니다. `hermes model`을 다시 실행하고 다른 Gemini 모델을 선택하세요.

### `hermes model`에 Gemma 모델이 표시되지 않음

Hermes는 기본적으로 처리량이 낮은 Gemma 모델을 선택기에서 숨길 수 있습니다. 평가를 위해 하나를 사용하려면 `~/.hermes/config.yaml`에 모델 ID를 직접 설정하세요.

### Gemma에서 "429 quota exceeded"

Gemini API를 통해 제공되는 Gemma 모델은 평가에 유용하지만 Gemini API 무료 등급 한도가 낮습니다. 호환성 테스트에 사용한 후 지속적인 에이전트 세션에는 유료 Gemini 모델이나 다른 공급자로 전환하세요.

### OpenAI 호환 엔드포인트가 설정됨

`~/.hermes/.env`에서 다음을 확인하세요.

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

네이티브 엔드포인트로 변경하거나 재정의를 제거하세요.

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

### 도구 호출이 스키마 오류와 함께 실패함

Hermes를 업그레이드하고 `hermes model`을 다시 실행하세요. 네이티브 Gemini 어댑터는 Gemini의 더 엄격한 함수 선언 형식에 맞게 도구 스키마를 정리합니다. 이전 빌드나 사용자 지정 엔드포인트에서는 그렇지 않을 수 있습니다.

## 관련 문서

- [AI 공급자](/integrations/providers)
- [설정](/user-guide/configuration)
- [대체 공급자](/user-guide/features/fallback-providers)
- [AWS Bedrock](/guides/aws-bedrock) — AWS 자격 증명을 사용하는 네이티브 클라우드 공급자 통합
