---
title: X (Twitter) 검색 (X (Twitter) Search)
description: SuperGrok OAuth 로그인 또는 XAI_API_KEY와 함께 작동하는 xAI의 내장 x_search Responses 도구를 사용하여 에이전트 내에서 X (Twitter) 게시물 및 스레드를 검색합니다.
sidebar_label: X (Twitter) 검색 (X (Twitter) Search)
sidebar_position: 7
---

# X (Twitter) 검색 (X (Twitter) Search)

`x_search` 도구를 사용하면 에이전트가 X (Twitter) 게시물, 프로필 및 스레드를 직접 검색할 수 있습니다. 이는 `https://api.x.ai/v1/responses`에 위치한 Responses API에서 xAI의 내장 `x_search` 도구의 지원을 받습니다 — Grok 자체가 서버 측에서 검색을 실행하고 원래 게시물에 대한 인용과 함께 종합된 결과를 반환합니다.

**X 내의** 현재 토론, 반응 또는 주장이 구체적으로 필요할 때 **`web_search` 대신 이 도구를 사용하세요**. 일반 웹 페이지의 경우 `web_search` / `web_extract`를 계속 사용하십시오.

:::tip
어쨌든 xAI 모델을 사용하기 위해 Portal 요금을 지불하고 있다면 라이브 검색 호출 비용은 채팅에 구성된 것과 동일한 xAI 키에 청구됩니다. [Nous Portal](/integrations/nous-portal)을 참고하세요.
:::

## 인증 (Authentication)

`x_search`는 xAI 자격 증명 경로 중 **하나**가 사용 가능할 때 등록됩니다:

| 자격 증명 (Credential) | 소스 (Source) | 설정 (Setup) |
|------------|--------|-------|
| **SuperGrok / X Premium+ OAuth** (선호됨) | 브라우저 로그인은 `accounts.x.ai`에서 수행되며 자동으로 새로 고쳐집니다 | `hermes auth add xai-oauth` — [xAI Grok OAuth (SuperGrok / X Premium+)](../../guides/xai-grok-oauth.md)를 참고하세요 |
| **`XAI_API_KEY`** | 유료 xAI API 키 | `~/.hermes/.env`에 설정 |

둘 다 동일한 페이로드를 사용하여 동일한 엔드포인트에 도달합니다 — 유일한 차이점은 베어러(bearer) 토큰입니다. **둘 다 구성된 경우 SuperGrok OAuth가 우선순위를 갖습니다**. 따라서 x_search는 유료 API 지출 대신 구독 쿼터(quota)에 대해 실행됩니다.

도구의 `check_fn`은 모델의 도구 목록이 재빌드될 때마다 xAI 자격 증명 확인자(resolver)를 실행합니다. `True`가 반환되면 베어러를 가져올 수 있고, 비어 있지 않으며 (만료된 경우) 새로 고침에 성공했음을 의미합니다. 새로 고침에 실패한 취소된 토큰은 스키마에서 도구를 숨깁니다. 그러면 모델은 그것을 볼 수 없습니다.

## 도구 활성화 (Enabling the tool)

xAI 자격 증명(OAuth 토큰 또는 `XAI_API_KEY`)이 있는 경우 자동으로 활성화됩니다. 원하지 않는 경우 `hermes tools` → Search → x_search를 통해 명시적으로 비활성화하세요.

```bash
hermes tools
# → 🐦 X (Twitter) Search   (스페이스바를 눌러 켜기/끄기 전환)
```

선택기(picker)는 두 가지 자격 증명 선택을 제공합니다:

1. **xAI Grok OAuth (SuperGrok / Premium+)** — 아직 로그인하지 않은 경우 브라우저를 `accounts.x.ai`로 엽니다.
2. **xAI API 키** — `XAI_API_KEY`에 대한 프롬프트를 표시합니다.

어느 쪽을 선택하든 게이팅(gating)을 충족합니다. 이미 가지고 있는 자격 증명을 선택할 수 있으며 도구는 두 경우 모두 동일하게 작동합니다. 둘 다 구성된 경우 호출 시 OAuth가 선호됩니다.

## 구성 (Configuration)

```yaml
# ~/.hermes/config.yaml
x_search:
  # Responses 호출에 사용되는 xAI 모델.
  # grok-4.20-reasoning이 권장되는 기본값이며 x_search 도구 권한이 있는
  # 모든 Grok 모델이 작동합니다.
  model: grok-4.20-reasoning

  # 요청 시간 초과 시간(초). 복잡한 쿼리의 경우 x_search가 
  # 60~120초가 걸릴 수 있으므로 넉넉한 기본값이 제공됩니다. 최소값: 30.
  timeout_seconds: 180

  # 5xx / ReadTimeout / ConnectionError 시 자동 재시도 횟수.
  # 각 재시도는 지연(백오프)됩니다 (1.5x 시도 시간, 최대 5초).
  retries: 2
```

## 도구 매개변수 (Tool parameters)

에이전트는 다음 인수를 사용하여 `x_search`를 호출합니다:

| 매개변수 (Parameter) | 유형 (Type) | 설명 (Description) |
|-----------|------|-------------|
| `query` | 문자열 (필수) | X에서 검색할 내용입니다. |
| `allowed_x_handles` | 문자열 배열 | **독점적으로** 포함할 핸들의 선택적 목록 (최대 10개). 선행 `@`는 제거됩니다. |
| `excluded_x_handles` | 문자열 배열 | 제외할 핸들의 선택적 목록 (최대 10개). `allowed_x_handles`와 상호 배타적입니다. |
| `from_date` | 문자열 | 선택적 `YYYY-MM-DD` 시작 날짜. |
| `to_date` | 문자열 | 선택적 `YYYY-MM-DD` 종료 날짜. |
| `enable_image_understanding` | 부울 | 일치하는 게시물에 첨부된 이미지를 분석하도록 xAI에 요청합니다. |
| `enable_video_understanding` | 부울 | 일치하는 게시물에 첨부된 비디오를 분석하도록 xAI에 요청합니다. |

도구는 다음을 포함하는 JSON을 반환합니다:

- `answer` — Grok의 합성된 텍스트 응답
- `citations` — Responses API 최상위 필드에서 반환된 인용
- `inline_citations` — 메시지 본문에서 추출된 `url_citation` 어노테이션 (각각에 `url`, `title`, `start_index`, `end_index` 포함)
- `degraded` — 축소 필터(`allowed_x_handles`, `excluded_x_handles`, `from_date`, `to_date`)가 설정되어 **있고** 두 인용 채널이 모두 비어 있을 때 `true`입니다. 이 경우 `answer`는 X 인덱스 대신 모델 자체의 지식에서 합성된 것이므로, 출처가 불분명한 것으로 취급하십시오. 그렇지 않으면 `false`입니다 ("필터가 설정되지 않은" 경우 포함 — 광범위하고 출처가 없는 답변은 단지 답변일 뿐 필터 누락이 아닙니다)
- `degraded_reason` — 활성화된 필터의 이름을 나타내는 짧은 문자열. `degraded`가 `false`일 때는 `null`입니다.
- `credential_source` — OAuth가 처리된 경우 `"xai-oauth"`, API 키가 처리된 경우 `"xai"`
- `model`, `query`, `provider`, `tool`, `success`

### 날짜 유효성 검사 (Date validation)

HTTP 호출 전에 클라이언트 측에서 `from_date` / `to_date`의 유효성이 검사됩니다:

- 두 날짜 모두 (제공된 경우) `YYYY-MM-DD` 형식으로 파싱되어야 합니다.
- 둘 다 설정된 경우 `from_date`는 `to_date`와 같거나 그 이전이어야 합니다.
- `from_date`는 오늘 UTC 이후일 수 없습니다 — 아직 시작되지 않은 시간 범위에는 게시물이 있을 수 없으므로, 이러한 호출은 항상 0개의 인용을 반환할 것이 보장되기 때문입니다.
- 미래의 `to_date`는 허용됩니다 (호출자가 도착하는 대로 게시물을 포착하기 위해 "어제부터 내일까지"와 같이 정당하게 요청할 수 있습니다).

유효성 검사 실패는 HTTP 호출이 아니라 구조화된 `{"error": "..."}` 도구 결과로 나타납니다.

## 예시 (Example)

에이전트와의 대화:

> 새로운 Grok 이미지 기능에 대해 사람들이 X에서 뭐라고 하나요? @xai의 답변에 집중해주세요.

에이전트의 작업:

1. `query="reactions to new Grok image features"`, `allowed_x_handles=["xai"]`를 사용하여 `x_search` 호출
2. 합성된 답변과 특정 게시물에 연결되는 인용 목록을 받습니다.
3. 답변 및 참조와 함께 답장합니다.

## 문제 해결 (Troubleshooting)

### "No xAI credentials available"

두 인증 경로가 모두 실패할 때 도구에서 이 메시지를 표시합니다. `~/.hermes/.env`에 `XAI_API_KEY`를 설정하거나 `hermes auth add xai-oauth`를 실행하고 브라우저 로그인을 완료하세요. 그런 다음 에이전트가 도구 레지스트리를 다시 읽을 수 있도록 세션을 다시 시작하십시오.

### "`x_search` is not enabled for this model"

구성된 `x_search.model`은 서버 측 `x_search` 도구에 액세스할 수 없습니다. `grok-4.20-reasoning` (기본값) 또는 이를 지원하는 다른 Grok 모델로 전환하십시오. 현재 목록은 [xAI 설명서](https://docs.x.ai/)를 참조하세요.

### 스키마에 도구가 나타나지 않음

두 가지 가능한 원인이 있습니다:

1. **도구 세트가 활성화되지 않음.** `hermes tools`를 실행하고 `🐦 X (Twitter) Search`가 선택되어 있는지 확인합니다.
2. **xAI 자격 증명 없음.** check_fn이 False를 반환하므로 스키마가 숨겨진 상태를 유지합니다. `hermes auth status`를 실행하여 xai-oauth 로그인 상태를 확인하고, (API 키 경로를 사용하는 경우) `XAI_API_KEY`가 설정되어 있는지 확인합니다.

### `degraded: true` — 인용 없는 답변

`allowed_x_handles`, `excluded_x_handles` 또는 날짜 범위를 사용했는데 응답이 `degraded: true`와 함께 다시 나타나는 경우, xAI의 X 인덱스는 일치하는 게시물을 반환하지 않았지만 Grok은 자체 학습 데이터에서 합성된 답변을 생성했습니다. 답변에는 출처가 없습니다 — 실제 X 결과로 취급하지 마세요.

확인해 볼 가치가 있는 원인:

- **핸들의 오타.** `@`를 제거하고 철자를 두 번 확인한 후 계정이 존재하는지 확인합니다.
- **날짜 범위가 너무 좁거나** 오늘의 게시물을 벗어났습니다; 범위를 넓히고 다시 시도하세요.
- **xAI 인덱스 차이.** 활발한 계정 중 일부는 정기적으로 게시하더라도 `x_search`에 간헐적으로 노출되지 않습니다. 몇 분 후에 다시 시도하거나, 정확한 핸들의 타임라인이 필요한 경우 직접 X API 읽기를 위해 `xurl` 기술을 사용하세요.

## 참고 항목

- [xAI Grok OAuth (SuperGrok / Premium+)](../../guides/xai-grok-oauth.md) — OAuth 설정 가이드
- [웹 검색 & 추출 (Web Search & Extract)](web-search.md) — 일반적인 (X가 아닌) 웹 검색용
- [도구 참조 (Tools Reference)](../../reference/tools-reference.md) — 전체 도구 카탈로그
