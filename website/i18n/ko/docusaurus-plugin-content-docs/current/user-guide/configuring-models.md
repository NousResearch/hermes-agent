---
sidebar_position: 3
---

# 모델 구성

Hermes는 두 종류의 모델 슬롯을 사용합니다:

- **주 모델** — 에이전트가 사고하는 데 사용하는 모델입니다. 모든 사용자 메시지, 모든 도구 호출 루프, 모든 스트리밍 응답이 이 모델을 거칩니다.
- **보조 모델** — 에이전트가 오프로딩하는 소규모 부가 작업에 사용하는 모델입니다. 컨텍스트 압축, 비전(이미지 분석), 웹 페이지 요약, 승인 점수 산정, MCP 도구 라우팅, 세션 제목 생성, 스킬 검색에 사용되며, 각각 별도의 슬롯을 가지고 독립적으로 재정의할 수 있습니다.

이 페이지에서는 대시보드에서 두 모델을 모두 구성하는 방법을 다룹니다. 설정 파일이나 CLI를 선호한다면 아래쪽의 [대체 방법](#alternative-methods)으로 이동하세요.

:::tip 가장 빠른 방법: Nous Portal
[Nous Portal](/user-guide/features/tool-gateway)은 하나의 구독으로 300개 이상의 모델을 제공합니다. 새로 설치한 경우 `hermes setup --portal`을 실행해 로그인하고 한 번에 Nous를 제공자로 설정하세요. `hermes portal info`로 현재 연결된 항목을 확인할 수 있습니다.

- Portal 구독자는 **토큰 사용량에 따라 과금되는 제공자에서 10% 할인**도 받습니다.
:::

:::note `model:` 스키마 — 빈 문자열과 매핑
새로 설치하면 함께 제공되는 기본 설정에서 `model: ""`(아직 "구성되지 않음"을 의미하는 빈 문자열 센티널)을 사용합니다. 처음 `hermes setup` 또는 `hermes model`을 실행하면 해당 키가 `provider`, `default`, `base_url`, `api_mode` 하위 키를 가진 매핑으로 즉시 업그레이드됩니다. 이 페이지와 [`profiles.md`](./profiles.md) / [`configuration.md`](./configuration.md) 전반에 표시되는 형태입니다. `config.yaml`에서 빈 문자열을 발견하면 `hermes model`을 실행하거나 대시보드에서 **변경**을 클릭하세요. 그러면 Hermes가 딕셔너리 형태를 기록합니다.
:::

## 모델 페이지

대시보드를 열고 사이드바에서 **모델**을 클릭하세요. 두 섹션이 표시됩니다:

1. **모델 설정** — 모델을 슬롯에 할당하는 상단 패널입니다.
2. **사용량 분석** — 선택한 기간에 세션을 실행한 모든 모델을 순위별 카드로 보여주며, 토큰 수, 비용, 기능 배지를 표시합니다.

![모델 페이지 개요](/img/docs/dashboard-models/overview.png)

상단 카드는 **모델 설정** 패널입니다. 주 행에는 새 세션에 사용할 에이전트 모델이 항상 표시됩니다. **변경**을 클릭하면 선택기가 열립니다.

## 주 모델 설정

주 모델 행에서 **변경**을 클릭하세요:

![모델 선택기 대화 상자](/img/docs/dashboard-models/picker-dialog.png)

선택기에는 두 열이 있습니다:

- **왼쪽** — 인증된 제공자입니다. 설정을 완료한 제공자(API 키 설정, OAuth 인증, 사용자 지정 엔드포인트 정의)만 여기에 표시됩니다. 제공자가 보이지 않으면 **키**로 이동해 인증 정보를 추가하세요.
- **오른쪽** — 선택한 제공자에 대한 엄선된 모델 목록입니다. 이는 Hermes가 해당 제공자에 권장하는 에이전트 모델이며, 원시 `/models` 결과가 아닙니다(OpenRouter의 경우 TTS, 이미지 생성기, 재순위 모델을 포함해 400개 이상의 모델이 포함됩니다).

필터 상자에 입력하면 제공자 이름, 슬러그 또는 모델 ID로 범위를 좁힐 수 있습니다.

모델을 선택하고 **전환**을 누르면 Hermes가 `~/.hermes/config.yaml`의 `model` 섹션에 기록합니다. **이는 새 세션에만 적용됩니다** — 이미 열어 둔 채팅 탭은 시작할 때 사용한 모델로 계속 실행됩니다. 현재 채팅에서 모델을 즉시 바꾸려면 채팅 안에서 `/model` 슬래시 명령을 사용하세요.

### 세션 중 전환과 컨텍스트 경고

**활성 세션 안에서 모델을 전환하면**(Herm TUI 모델 선택기, `hermes` CLI 또는 Telegram/Discord의 `/model`) Hermes는 **다음 메시지**가 새 모델의 컨텍스트 창에 대해 **사전 컨텍스트 압축**을 실행할지 추정합니다. 세션이 해당 모델의 압축 임계값( [컨텍스트 압축](./configuration.md#context-compression) 참조)에 이미 근접했거나 이를 초과한 경우, 전환 응답에 경고가 포함됩니다. 이는 비용이 높은 모델 알림에 사용되는 동일한 `warning_message` 경로입니다. 전환은 즉시 적용되며, 압축은 모델이 응답하기 전에 **전환 후 첫 사용자 메시지에서** 실행됩니다.

:::warning 세션 중 전환은 프롬프트 캐시를 초기화합니다
프롬프트 캐시는 요청을 처리하는 모델을 기준으로 지정되므로, 대화 중 모델을 변경하면(명시적인 `/model` 전환, [자동 폴백](./features/fallback-providers.md), 또는 [자격 증명 풀](./features/credential-pools.md)이 다른 계정으로 순환하는 경우) 다음 메시지는 캐시된(약 75~90% 할인된) 요금 대신 전체 대화를 입력 토큰의 정가로 다시 읽습니다. 긴 세션에서는 이 일회성 재읽기 비용이 두 모델 간 토큰당 가격 차이보다 훨씬 클 수 있습니다. 필요할 때 전환하되, 대화 초반이나 새 세션을 시작한 직후에 하는 편이 좋습니다.
:::

## 보조 모델 설정

**보조 항목 표시**를 클릭하면 11개의 작업 슬롯이 나타납니다:

![보조 패널 확장](/img/docs/dashboard-models/auxiliary-expanded.png)

모든 보조 작업은 기본적으로 `auto`입니다. 즉 Hermes가 해당 작업에도 주 모델을 사용하려고 시도합니다. 해당 경로를 사용할 수 없거나 용량 관련 오류가 발생하면 `auto`는 작업별 `auxiliary.<task>.fallback_chain`, 그다음 주 `fallback_providers` / `fallback_model` 체인, 마지막으로 Hermes에 내장된 보조 모델 검색 체인을 차례로 따릅니다. 부가 작업에 더 저렴하거나 빠른 모델을 사용하고 싶다면 특정 작업을 재정의하세요.

### 일반적인 재정의 패턴

| 작업 | 재정의할 때 |
|---|---|
| **제목 생성** | 제목 생성 지연 시간이나 비용이 주 모델과의 일치보다 중요할 때입니다. 검증된 flash 모델을 고정하거나 `auxiliary.title_generation.prefer_fast_model: true`를 설정해 Hermes가 제공자의 빠른 티어를 선택하도록 하세요. |
| **비전** | 주 모델에 비전 지원이 없을 때입니다. `google/gemini-2.5-flash` 또는 `gpt-4o-mini`를 지정하세요. |
| **압축** | 컨텍스트를 요약하려고 Opus/M2.7에서 추론 토큰을 소모하고 있을 때입니다. 빠른 채팅 모델이면 1/50 비용으로 처리할 수 있습니다. |
| **승인** | `approval_mode: smart`를 사용할 때입니다 — 빠르고 저렴한 모델(haiku, flash, gpt-5-mini)이 위험도가 낮은 명령을 자동 승인할지 결정합니다. 여기에 비싼 모델을 쓰는 것은 낭비입니다. |
| **웹 추출** | `web_extract`를 많이 사용할 때입니다. 압축과 같은 논리로, 요약에는 추론이 필요하지 않습니다. |
| **스킬 허브** | `hermes skills search`가 사용합니다. 보통 `auto`로 충분합니다. |
| **MCP** | MCP 도구 라우팅에 사용합니다. 보통 `auto`로 충분합니다. |
| **트리아지 지정자** | Kanban 트리아지 지정자(`hermes kanban specify`)를 라우팅합니다. 이 지정자는 대략적인 한 줄 설명을 구체적인 사양으로 확장합니다. 저렴하면서 성능이 좋은 모델이 적합합니다. |
| **Kanban 분해기** | Kanban 작업 분해를 라우팅합니다 — 트리아지 작업을 전문 프로필을 위한 하위 작업 그래프로 나눕니다. |
| **프로필 설명자** | 프로필 설명 생성(`hermes profile describe --auto` / 대시보드 자동 생성 버튼)을 라우팅합니다. 짧고 저렴한 호출입니다. |
| **큐레이터** | 큐레이터의 스킬 사용량 검토 단계를 라우팅합니다. 추론 모델에서 몇 분이 걸릴 수 있으므로 더 저렴한 보조 모델이 유용한 경우가 많습니다. |

### 작업별 재정의

아무 보조 행에서나 **변경**을 클릭하세요. 동일한 선택기가 열리고 동일한 방식으로 동작합니다 — 제공자와 모델을 선택하고 **전환**을 누르세요. 행이 `auto (주 모델 사용)` 대신 `provider · model`을 표시하도록 업데이트됩니다.

### 모두 auto로 재설정

과도하게 조정했으며 처음부터 다시 시작하고 싶다면 보조 섹션 상단의 **모두 auto로 재설정**을 클릭하세요. 모든 슬롯이 주 모델을 사용하도록 돌아갑니다.

## "다음 용도로 사용" 바로 가기

페이지의 모든 모델 카드에는 **다음 용도로 사용** 드롭다운이 있습니다. 빠른 방법은 분석에서 보이는 모델을 선택하고 **다음 용도로 사용**을 클릭한 다음, 한 번의 클릭으로 주 슬롯이나 특정 보조 작업에 할당하는 것입니다:

![다음 용도로 사용 드롭다운](/img/docs/dashboard-models/use-as-dropdown.png)

드롭다운에는 다음 항목이 있습니다:

- **주 모델** — 주 행에서 **변경**을 클릭하는 것과 같습니다.
- **모든 보조 작업** — 이 모델을 11개 보조 슬롯 모두에 한 번에 할당합니다. 모든 부가 작업에 저렴한 flash 모델을 사용하고 싶을 때 유용합니다.
- **개별 작업 옵션** — 비전, 웹 추출, 압축 등입니다. 각 작업에 현재 할당된 모델에는 `current`가 표시됩니다.

현재 무언가에 할당된 카드에는 `main` 또는 `aux · <task>` 배지가 표시됩니다 — 과거에 사용한 모델이 어디에 연결되어 있는지 한눈에 확인할 수 있습니다.

## `config.yaml`에 기록되는 내용

대시보드를 통해 저장하면 Hermes가 `~/.hermes/config.yaml`에 기록합니다:

**주 모델:**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4.7
  base_url: ''        # cleared on provider switch
  api_mode: chat_completions
```

**보조 재정의(예 — gemini-flash의 비전):**
```yaml
auxiliary:
  vision:
    provider: openrouter
    model: google/gemini-2.5-flash
    base_url: ''
    api_key: ''
    timeout: 120
    extra_body: {}
    download_timeout: 30
```

**auto 상태의 보조 모델(기본값):**
```yaml
auxiliary:
  compression:
    provider: auto
    model: ''
    base_url: ''
    # ... other fields unchanged
```

`model: ''`과 함께 `provider: auto`를 사용하면 Hermes가 해당 작업에 주 모델을 사용하되, 주 경로가 보조 호출을 처리할 수 없는 경우에도 폴백 정책을 적용합니다.

선택적인 작업별 폴백 체인은 동일한 보조 작업 아래에 둡니다:

```yaml
auxiliary:
  title_generation:
    provider: auto
    model: ''
    fallback_chain:
      - provider: openrouter
        model: inclusionai/ring-2.6-1t:free
```

`fallback_chain`이 없으면 `auto`는 내장 보조 모델 검색 체인보다 먼저 최상위 `fallback_providers` 체인을 사용합니다.

## 제공자별 요청 옵션

제공자 항목(`providers:` 딕셔너리의 `providers.<name>` 또는 레거시 `custom_providers` 목록의 항목)은 Hermes가 엔드포인트와 통신하는 방식을 정하는 두 가지 옵션을 받습니다:

**`extra_headers`** — 해당 제공자의 기본 URL로 라우팅되는 모든 LLM 요청에 추가할 HTTP 헤더의 매핑입니다. URL/프로필 기본값과 사용자 헤더 재정의 이후 마지막에 적용되므로 자격 증명 교체와 클라이언트 재생성 후에도 유지됩니다. Cloudflare Access 서비스 토큰, 프록시 인증 또는 사용자 지정 bearer 스킴에 유용합니다:

```yaml
providers:
  my-gateway:
    api: https://llm.internal.example.com/v1
    api_key: sk-...
    extra_headers:
      CF-Access-Client-Id: "xxxx.access"
      CF-Access-Client-Secret: "yyyy"
```

헤더 값에는 일반적으로 자격 증명이 포함되므로 Hermes는 이를 절대 로그에 기록하지 않습니다. `extra_headers`는 OpenAI 호환 경로에 적용되며 `anthropic_messages` 및 `bedrock_converse` API 모드에서는 사용되지 않습니다.

**`discover_models`** — 엔드포인트의 `/models` 목록 조회를 건너뛰고 항목에 구성한 `models`만 사용하려면 `false`(기본값 `true`)로 설정하세요. 모델 목록 조회가 느리거나 불안정하거나 불필요하게 많은 게이트웨이에 유용합니다:

```yaml
providers:
  my-gateway:
    api: https://llm.internal.example.com/v1
    discover_models: false
    models:
      - my-finetune-v2
      - my-finetune-v1
```

검색을 끄면 모델 선택기(`hermes model`, `/model`)가 실시간 탐색 결과 대신 구성된 목록을 표시합니다.

Anthropic 호환 게이트웨이가 요청을 받은 뒤에만 이름 없는 모델 별칭을 해석하는 경우, 모델별 `prompt_caching` 기능을 사용해 해당 별칭을 네이티브 프롬프트 캐시 마커에 등록하세요:

```yaml
providers:
  anthropic-proxy:
    api: https://gateway.example.com/anthropic
    transport: anthropic_messages
    models:
      fable:
        context_length: 1000000
        prompt_caching: true
```

Hermes는 별칭을 다시 쓰지 않고 이 선언을 정확한 제공자 경로 및 런타임 모델 ID에 맞춥니다. 모델의 캐시 마커를 명시적으로 비활성화하려면 `prompt_caching: false`로 설정하세요. 생략하면 Hermes는 일반적인 제공자 및 모델 기능 감지를 유지합니다.

:::note 레거시 형식
이전 설정에서는 최상위 `custom_providers:` 목록(`api` 대신 `base_url` 사용)을 사용했습니다. 이 형식은 여전히 작동하며 `hermes update`(설정 v12)에서 `providers:` 딕셔너리로 자동 마이그레이션됩니다.
:::

## 언제 적용되나요?

- **CLI**(`hermes chat`): 다음 `hermes chat` 호출부터 적용됩니다.
- **게이트웨이**(Telegram, Discord, Slack 등): 다음 *새* 세션부터 적용됩니다. 기존 세션은 모델을 유지합니다. 모든 세션이 변경 사항을 적용하도록 하려면 게이트웨이를 재시작하세요(`hermes gateway restart`).
- **대시보드 채팅 탭**(`/chat`): 다음 새 PTY부터 적용됩니다. 현재 열려 있는 채팅은 모델을 유지하므로, 즉시 바꾸려면 채팅 안에서 `/model`을 사용하세요.

변경 사항은 실행 중인 세션의 프롬프트 캐시를 절대 무효화하지 않습니다. 이는 의도된 동작입니다. 세션 안에서 주 모델을 바꾸려면 캐시를 재설정해야 하고(시스템 프롬프트에 모델별 내용이 포함됨), 이 작업은 채팅 내부의 명시적인 `/model` 슬래시 명령에 한해서만 수행합니다.

## 문제 해결

### 선택기에 "인증된 제공자 없음"이 표시됨

Hermes는 작동하는 자격 증명이 있는 경우에만 제공자를 나열합니다. 사이드바의 **키**를 확인하세요 — API 키, 성공한 OAuth 인증 또는 사용자 지정 엔드포인트 URL 중 하나가 표시되어야 합니다. 원하는 제공자가 없다면 `hermes setup`을 실행해 연결하거나 **키**로 이동해 환경 변수를 추가하세요.

### 실행 중인 채팅에서 주 모델이 변경되지 않음

정상입니다. 대시보드는 `config.yaml`에 기록하며 새 세션이 이를 읽습니다. 현재 열려 있는 채팅은 실행 중인 에이전트 프로세스이므로 생성될 때 사용한 모델을 계속 사용합니다. 채팅 안에서 `/model <name>`을 사용해 해당 세션만 즉시 바꾸세요.

### 보조 재정의가 "적용되지 않음"

다음 세 가지를 확인하세요:

1. **새 세션을 시작했나요?** 기존 채팅은 설정을 다시 읽지 않습니다.
2. **`provider`가 `auto`가 아닌 값으로 설정되어 있나요?** 필드에 `auto`가 표시되면 해당 작업은 여전히 주 모델을 사용 중입니다. **변경**을 클릭하고 실제 제공자를 선택하세요.
3. **제공자가 인증되어 있나요?** 작업에 `minimax`를 할당했지만 MiniMax API 키가 없다면 해당 작업은 openrouter 기본값으로 폴백하며 `agent.log`에 경고를 기록합니다.

### 모델을 선택했는데 Hermes가 제공자를 바꿈

OpenRouter(또는 다른 애그리게이터)에서는 이름 없는 모델 이름이 먼저 애그리게이터 내부에서 해석됩니다. 따라서 OpenRouter에서 `claude-sonnet-4`는 `anthropic/claude-sonnet-4.6`이 되어 OpenRouter 인증을 그대로 사용합니다. 하지만 네이티브 Anthropic 인증에서 `claude-sonnet-4`를 입력하면 `claude-sonnet-4-6`으로 유지됩니다. 예상치 못한 제공자 전환이 보이면 현재 제공자가 예상한 값인지 확인하세요 — 선택기는 대화 상자 상단에 현재 주 모델을 항상 표시합니다.

## 대체 방법

### CLI 슬래시 명령

어떤 `hermes chat` 세션에서든 다음을 실행할 수 있습니다:

```
/model gpt-5.4 --provider openrouter             # session-only
/model gpt-5.4 --provider openrouter --global    # also persists to config.yaml
/model claude-opus-4.6 --once                    # next turn only, then auto-restores
```

`--global`은 대시보드의 **변경** 버튼과 동일한 작업을 수행하며, 실행 중인 세션도 즉시 전환합니다.

`--once`는 한 번의 턴 동안만 전환한 뒤 이전 모델을 복원합니다 — 성공, 오류 또는 중단 여부와 관계없이 동일합니다. 아무것도 저장되지 않습니다. 턴 중 게이트웨이가 재시작되어도 원래 모델로 돌아옵니다. 어려운 질문 하나를 비싼 모델로 올려 처리하거나("이번 한 번만 Opus에게 물어보기"), 일회성 질의에 저렴한 모델로 낮출 때 유용합니다.

:::note 프롬프트 캐시 비용
한 번의 턴 동안 전환하면 제공자의 프롬프트 캐시 접두사가 두 번(전환해 나갈 때와 돌아올 때) 끊깁니다. 캐시된 접두사를 사용하는 제공자(Anthropic, OpenAI)의 긴 세션에서는 다음 턴에 입력 비용을 전부 다시 지불하게 됩니다 — `--once`는 짧은 세션이나 저렴한 모델에서 비싼 모델로 올리는 경우에 유리하지만, 긴 비싼 세션 중 빠른 부가 질문 하나를 처리하는 경우에는 절약되는 비용보다 더 많이 들 수 있습니다.
:::

### 사용자 지정 별칭

자주 사용하는 모델에 짧은 이름을 직접 정의한 다음, CLI나 모든 메시징 플랫폼에서 `/model <alias>`를 사용하세요. 동일한 형식이 두 가지 있으므로 작업 방식에 맞는 것을 선택하면 됩니다.

**표준 형식(최상위 `model_aliases:`)** — 제공자와 base_url을 모두 제어합니다:

```yaml
# ~/.hermes/config.yaml
model_aliases:
  fav:
    model: claude-sonnet-4.6
    provider: anthropic
  grok:
    model: grok-4
    provider: x-ai
```

**짧은 문자열 형식(`model.aliases.<name>: provider/model`)** — `hermes config set`은 스칼라 값만 기록할 수 있어 셸에서 편리하지만, 사용자 지정 `base_url`은 전달할 수 없습니다:

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

두 경로 모두 동일한 로더(`hermes_cli/model_switch.py`)로 전달됩니다. `model_aliases:`에 선언된 항목은 같은 이름의 `model.aliases:` 항목보다 우선합니다.

그 다음 채팅에서 `/model fav` 또는 `/model grok`을 사용하세요. 사용자 별칭은 내장된 짧은 이름(`sonnet`, `kimi`, `opus` 등)을 가립니다. 전체 참고 문서는 [사용자 지정 모델 별칭](/reference/slash-commands#custom-model-aliases)을 참조하세요.

### `hermes model` 하위 명령

```bash
hermes model            # Interactive provider + model picker (the canonical way to switch defaults)
```

`hermes model`은 제공자를 선택하고, 인증한 다음(OAuth 흐름은 브라우저를 열고 API 키 제공자는 키를 요청함), 해당 제공자의 엄선된 카탈로그에서 특정 모델을 선택하는 과정을 안내합니다. 선택한 내용은 `~/.hermes/config.yaml`의 `model.provider` 및 `model.default`에 기록됩니다.

선택기를 실행하지 않고 제공자/모델을 나열하려면 아래의 대시보드나 REST 엔드포인트를 사용하세요. CLI가 현재 실제로 사용할 값을 확인하려면 `hermes config get model --json`과 `hermes status`를 사용하세요.

### 직접 설정 편집

`~/.hermes/config.yaml`을 편집하고 해당 설정을 읽는 프로그램을 재시작하세요. 전체 스키마는 [설정 참고 문서](./configuration.md)를 참조하세요.

### REST API

대시보드는 세 개의 엔드포인트를 사용합니다. 스크립팅에 유용합니다:

```bash
# List authenticated providers + curated model lists
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/options

# Read current main + auxiliary assignments
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/auxiliary

# Set the main model
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"main","provider":"openrouter","model":"anthropic/claude-opus-4.7"}' \
  http://localhost:PORT/api/model/set

# Override a single auxiliary task
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"vision","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# Assign one model to every auxiliary task
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# Reset all auxiliary tasks to auto
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"__reset__","provider":"","model":""}' \
  http://localhost:PORT/api/model/set
```

세션 토큰은 시작 시 대시보드 HTML에 주입되며 서버를 재시작할 때마다 순환됩니다. 실행 중인 대시보드를 대상으로 스크립트를 실행한다면 브라우저 개발자 도구에서(`window.__HERMES_SESSION_TOKEN__`) 토큰을 가져오세요.
