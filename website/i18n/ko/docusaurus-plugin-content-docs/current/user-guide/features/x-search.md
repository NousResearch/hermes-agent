---
title: X (Twitter) 검색
description: xAI의 기본 제공 x_search Responses 도구를 사용해 에이전트 안에서 X(Twitter)의 게시물과 스레드를 검색합니다 — SuperGrok OAuth 로그인 또는 XAI_API_KEY로 사용할 수 있습니다.
sidebar_label: X (Twitter) 검색
sidebar_position: 7
---

# X (Twitter) 검색

`x_search` 도구를 사용하면 에이전트가 X(Twitter)의 게시물, 프로필, 스레드를 직접 검색할 수 있습니다. 이 도구는 `https://api.x.ai/v1/responses`의 Responses API에 있는 xAI의 기본 제공 `x_search` 도구를 기반으로 하며 — Grok 자체가 서버 측에서 검색을 실행하고 원본 게시물의 인용이 포함된 종합 결과를 반환합니다.

현재 논의, 반응 또는 **X에서** 제기된 주장을 구체적으로 원할 때는 `web_search` 대신 **이 도구를 사용하세요**. 일반 웹 페이지에는 계속 `web_search` / `web_extract`를 사용하세요.

## `x_search`와 `xurl` 비교

Hermes는 서로 다른 두 가지 X 표면을 제공할 수 있습니다.

| 표면 | 용도 | 사용하지 않는 용도 |
|---------|------------|-------------------|
| `x_search` | 읽기 전용 공개 X 탐색: 현재 논의, 반응, 주장, 프로필, 스레드, 인용이 포함된 종합 답변. | 게시, 답글, 좋아요, DM, 미디어 업로드, 삭제 또는 인증된 X 계정의 상태 변경이 일어났음을 입증하는 작업. |
| `xurl` skill | 정확하거나 인증이 필요한 X API 작업: `post`, `reply`, `read`, `like`, `dm`, 타임라인, 멘션, 미디어 업로드, 계정별 읽기, 원시 v2 엔드포인트. | `x_search`를 사용할 수 있고 인증된 계정 컨텍스트가 필요하지 않은 경우의 광범위한 Grok 종합 공개 X 조사. |

혼합 워크플로에서는 `x_search`로 후보 공개 게시물을 찾은 다음, 대상 게시물/사용자/작업이 명확해지면 `xurl read` 또는 다른 정확한 `xurl` 명령으로 전환하세요. 상태를 변경하는 X 작업은 `xurl` 출력 또는 X API 응답으로 확인해야 합니다. `x_search` 답변은 쓰기가 발생했다는 증거가 아닙니다.

:::tip
어쨌든 Portal에서 xAI 모델을 결제하고 있다면 Live Search 호출은 채팅에 설정된 동일한 xAI 키를 기준으로 청구됩니다. [Nous Portal](/integrations/nous-portal)을 참고하세요.
:::

## 인증

**xAI 자격 증명 경로 중 하나라도** 사용할 수 있으면 `x_search`가 등록됩니다.

| 자격 증명 | 출처 | 설정 |
|------------|--------|-------|
| **SuperGrok / X Premium+ OAuth** (권장) | `accounts.x.ai`에서 브라우저 로그인, 자동 갱신 | `hermes auth add xai-oauth` — [xAI Grok OAuth (SuperGrok / X Premium+)](../../guides/xai-grok-oauth.md) 참고 |
| **`XAI_API_KEY`** | 유료 xAI API 키 | `~/.hermes/.env`에 설정 |

두 경로 모두 동일한 엔드포인트에 동일한 페이로드로 요청하며 — 유일한 차이는 bearer 토큰입니다. **둘 다 설정된 경우에는 SuperGrok OAuth가 우선**하므로 유료 API 비용이 아니라 구독 할당량으로 x_search가 실행됩니다.

도구의 `check_fn`은 모델의 도구 목록이 다시 생성될 때마다 xAI 자격 증명 확인자를 실행합니다. `True` 반환은 bearer를 가져올 수 있고 비어 있지 않으며, 만료된 경우 성공적으로 갱신되었다는 뜻입니다. 갱신에 실패한 폐기된 토큰은 스키마에서 도구를 숨깁니다. 모델은 해당 도구를 볼 수 없게 됩니다.

## 도구 활성화

xAI 자격 증명(OAuth 토큰 또는 `XAI_API_KEY`)이 있으면 자동으로 활성화됩니다. 원하지 않는 경우 `hermes tools` → Search → x_search에서 명시적으로 비활성화하세요.

```bash
hermes tools
# → 🐦 X (Twitter) Search   (press space to toggle on)
```

선택 화면에서는 두 가지 자격 증명 옵션을 제공합니다.

1. **xAI Grok OAuth (SuperGrok / Premium+)** — 아직 로그인하지 않았다면 `accounts.x.ai`의 브라우저를 엽니다.
2. **xAI API key** — `XAI_API_KEY`를 입력하라는 메시지를 표시합니다.

어느 옵션을 선택해도 게이팅 조건을 충족합니다. 이미 보유한 자격 증명을 선택하면 됩니다. 두 옵션 모두 도구가 동일하게 작동합니다. 둘 다 설정된 경우 호출 시 OAuth가 우선됩니다.

## 구성

```yaml
# ~/.hermes/config.yaml
x_search:
  # xAI model used for the Responses call.
  # grok-4.5 is the recommended default; any Grok model
  # with x_search tool access works.
  model: grok-4.5

  # Optional reasoning effort: low, medium, high, or xhigh. When omitted,
  # the selected model's default applies. xhigh is supported only by
  # models that document it, such as grok-4.20-multi-agent.
  # reasoning_effort: low

  # Request timeout in seconds. x_search can take 60–120s for
  # complex queries — the default is generous. Minimum: 30.
  timeout_seconds: 180

  # Number of automatic retries on 5xx / ReadTimeout / ConnectionError.
  # Each retry backs off (1.5x attempt seconds, capped at 5s).
  retries: 2
```

`reasoning_effort`는 `reasoning: {effort: ...}`로 xAI Responses API에 전송됩니다. 구성 가능한 추론을 지원하지 않는 모델에서는 설정하지 않은 상태로 두세요. 잘못된 값은 API 요청이 실행되기 전에 실패합니다.

## 도구 매개변수

에이전트는 다음 인수로 `x_search`를 호출합니다.

| 매개변수 | 유형 | 설명 |
|-----------|------|-------------|
| `query` | string (필수) | X에서 조회할 내용. |
| `allowed_x_handles` | string array | 포함할 핸들의 선택 목록(최대 10개). 선행 `@`는 제거됩니다. |
| `excluded_x_handles` | string array | 제외할 핸들의 선택 목록(최대 10개). `allowed_x_handles`와 함께 사용할 수 없습니다. |
| `from_date` | string | 선택 사항인 `YYYY-MM-DD` 시작 날짜. |
| `to_date` | string | 선택 사항인 `YYYY-MM-DD` 종료 날짜. |
| `enable_image_understanding` | boolean | 일치하는 게시물에 첨부된 이미지를 xAI가 분석하도록 요청합니다. |
| `enable_video_understanding` | boolean | 일치하는 게시물에 첨부된 동영상을 xAI가 분석하도록 요청합니다. |

도구는 다음 JSON을 반환합니다.

- `answer` — Grok의 종합 텍스트 응답
- `citations` — Responses API 최상위 필드에서 반환된 인용
- `inline_citations` — 메시지 본문에서 추출한 `url_citation` 주석(`url`, `title`, `start_index`, `end_index` 포함)
- `degraded` — 범위 좁히기 필터(`allowed_x_handles`, `excluded_x_handles`, `from_date`, `to_date`)가 하나라도 설정되었고 두 인용 채널이 모두 비어 있을 때 `true`. 이 경우 `answer`는 X 색인이 아니라 모델 자체 지식으로 종합된 것이므로 출처가 없는 것으로 취급해야 합니다. 그 외에는 `false`입니다(필터를 설정하지 않은 경우도 포함 — 출처 없는 광범위한 답변은 필터 누락이 아니라 단순한 답변입니다).
- `degraded_reason` — 활성화된 필터를 나타내는 짧은 문자열 또는 `degraded`가 `false`일 때 `null`
- `credential_source` — OAuth가 확인된 경우 `"xai-oauth"`, API 키가 확인된 경우 `"xai"`
- `model`, `query`, `provider`, `tool`, `success`

### 날짜 검증

HTTP 호출 전에 `from_date` / `to_date`가 클라이언트 측에서 검증됩니다.

- 제공된 경우 두 값 모두 `YYYY-MM-DD`로 해석되어야 합니다.
- 두 값이 모두 설정된 경우 `from_date`는 `to_date` 이전이거나 같아야 합니다.
- `from_date`는 UTC 기준 오늘보다 늦을 수 없습니다 — 아직 시작되지 않은 기간에는 게시물이 존재할 수 없으므로 호출 결과에 인용이 0개일 것이 보장됩니다.
- 미래의 `to_date`는 허용됩니다(호출자가 게시물이 도착할 때 포착하기 위해 "어제부터 내일까지"를 합법적으로 요청할 수 있습니다).

검증 실패는 구조화된 `{"error": "..."}` 도구 결과로 표시되며, xAI로 보내는 HTTP 호출로 이어지지 않습니다.

## 예시

에이전트와 대화하기:

> X에서 새로운 Grok 이미지 기능에 대해 사람들이 뭐라고 말하는지 알려줘. @xai의 응답에 집중해.

에이전트는 다음을 수행합니다.

1. `query="reactions to new Grok image features"`, `allowed_x_handles=["xai"]`로 `x_search`를 호출합니다.
2. 특정 게시물로 연결되는 인용 목록과 종합 답변을 받습니다.
3. 답변과 참고 자료를 함께 반환합니다.

다음 사용자 요청이 "가장 좋은 게시물에 답글 달아" 또는 "그 게시물에 좋아요 눌러"라면 에이전트는 `xurl` skill로 전환하고 정확한 대상 게시물을 확인한 후 X API 작업을 사용해야 합니다. `x_search`는 탐색 도구로 남습니다.

## 문제 해결

### "사용 가능한 xAI 자격 증명이 없습니다"

두 인증 경로가 모두 실패하면 도구에서 이 메시지를 표시합니다. `~/.hermes/.env`에 `XAI_API_KEY`를 설정하거나 `hermes auth add xai-oauth`를 실행하고 브라우저 로그인을 완료하세요. 그런 다음 에이전트가 도구 레지스트리를 다시 읽도록 세션을 재시작하세요.

### "이 모델에서는 `x_search`가 활성화되지 않았습니다"

구성된 `x_search.model`이 서버 측 `x_search` 도구에 액세스할 수 없습니다. `grok-4.5`(기본값) 또는 이를 지원하는 다른 Grok 모델로 전환하세요. 현재 지원 모델 목록은 [xAI 문서](https://docs.x.ai/)에서 확인하세요.

### 도구가 스키마에 표시되지 않음

가능한 원인은 두 가지입니다.

1. **도구 세트가 활성화되지 않았습니다.** `hermes tools`를 실행하고 `🐦 X (Twitter) Search`가 선택되어 있는지 확인하세요.
2. **xAI 자격 증명이 없습니다.** `check_fn`이 False를 반환하므로 스키마에 도구가 표시되지 않습니다. `hermes auth status`로 xai-oauth 로그인 상태를 확인하고, API 키 경로를 사용하는 경우 `XAI_API_KEY`가 설정되어 있는지 확인하세요.

### `degraded: true` — 인용이 없는 답변

`allowed_x_handles`, `excluded_x_handles` 또는 날짜 범위를 사용했는데 응답이 `degraded: true`로 반환되면 xAI의 X 색인에서 일치하는 게시물을 반환하지 않았지만 Grok이 자체 학습 데이터로 종합 답변을 생성한 것입니다. 이 답변에는 출처가 없으므로 실제 X 결과로 취급하지 마세요.

확인해 볼 만한 원인은 다음과 같습니다.

- **핸들 오타.** `@`를 제거하고 철자를 다시 확인한 다음 계정이 존재하는지 확인하세요.
- **날짜 범위가 너무 좁거나** 오늘 게시물을 지나도록 설정되었습니다. 범위를 넓혀 다시 시도하세요.
- **xAI 색인 누락.** 정기적으로 게시하는 활성 계정도 `x_search`에 간헐적으로 표시되지 않을 수 있습니다. 몇 분 후 다시 시도하거나, 특정 핸들의 타임라인을 정확히 읽어야 할 때는 `xurl` skill을 사용해 직접 X API를 읽으세요.

## 함께 보기

- [xAI Grok OAuth (SuperGrok / Premium+)](../../guides/xai-grok-oauth.md) — OAuth 설정 안내
- [xurl 스킬](../skills/bundled/social-media/social-media-xurl.md) — 인증된 계정 작업을 위한 공식 X API CLI
- [웹 검색 및 추출](web-search.md) — 일반(X가 아닌) 웹 검색용
- [도구 참고 자료](../../reference/tools-reference.md) — 전체 도구 카탈로그
