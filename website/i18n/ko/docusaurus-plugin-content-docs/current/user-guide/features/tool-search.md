---
title: Tool Search
sidebar_position: 95
---

# 도구 검색 (Tool Search)

세션에 많은 MCP 서버나 핵심이 아닌 플러그인 도구가 연결되어 있을 때, 해당 JSON 스키마는 매 턴마다 컨텍스트 윈도우의 상당한 부분을 소비할 수 있습니다. 심지어 그중 몇 개만 사용자가 실제로 요청한 것과 관련이 있을 때도 마찬가지입니다.

**Tool Search(도구 검색)** 는 이 문제를 해결하기 위한 Hermes의 선택적(opt-in) 점진적 공개(progressive-disclosure) 계층입니다. 활성화되면 모델에 표시되는 도구 배열에서 MCP 및 플러그인 도구가 세 개의 브리지(bridge) 도구로 대체되며, 모델은 온디맨드로 각 특정 도구의 스키마를 로드합니다.

:::info 내장 Hermes 도구는 결코 지연되지 않습니다
Hermes의 핵심 기능을 구성하는 도구들(`terminal`, `read_file`, `write_file`, `patch`, `search_files`, `todo`, `memory`, `browser_*`, `web_search`, `web_extract`, `clarify`, `execute_code`, `delegate_task`, `session_search`, `send_message`, 그리고 나머지 `_HERMES_CORE_TOOLS`)은 *항상* 직접 로드됩니다. 오직 MCP 도구와 핵심이 아닌 플러그인 도구만 지연(deferral) 대상이 됩니다.
:::

## 작동 방식

Tool Search가 턴에 대해 활성화되면, 모델은 지연된 도구들을 대신하여 세 개의 새로운 도구를 보게 됩니다.

```
tool_search(query, limit?)     — 지연된 도구 카탈로그 검색
tool_describe(name)            — 하나의 도구에 대한 전체 스키마 로드
tool_call(name, arguments)     — 지연된 도구 호출
```

일반적인 상호 작용은 다음과 같습니다.

```
Model: tool_search("create a github issue")
  → { matches: [{ name: "mcp_github_create_issue", ... }, ...] }
Model: tool_describe("mcp_github_create_issue")
  → { parameters: { type: "object", properties: { ... } } }
Model: tool_call("mcp_github_create_issue", { title: "...", body: "..." })
  → { ok: true, issue_number: 42 }
```

모델이 `tool_call`을 호출하면 Hermes는 **브리지를 언래핑(unwrap)** 하고 모델이 도구를 직접 호출한 것과 똑같이 기본 도구를 디스패치합니다. 도구 호출 전 훅(pre-tool-call hooks), 가드레일, 승인 프롬프트 및 도구 호출 후 훅은 `tool_call`이 아닌 실제 도구 이름을 기준으로 실행됩니다. CLI 및 게이트웨이의 활동 피드(activity feed)도 언래핑되므로 브리지가 아닌 기본 도구가 표시됩니다.

## 언제 활성화되나요?

기본적으로 Tool Search는 `auto` 모드로 실행됩니다. 지연 가능한 도구 스키마가 활성 모델의 컨텍스트 윈도우의 10% 이상을 소비할 때만 활성화됩니다. 그 미만에서는 도구 배열 어셈블리가 단순한 패스스루(pass-through)가 되며 오버헤드가 발생하지 않습니다.

이 결정은 도구 배열이 빌드될 때마다 다시 평가되므로 다음이 적용됩니다.

- 몇 개의 MCP 도구와 긴 컨텍스트 모델만 있는 세션은 결코 Tool Search를 활성화하지 않습니다.
- 많은 MCP 서버가 연결된 세션(일반적으로 15개 이상의 도구)은 활성화를 시작합니다.
- 세션 중간에 MCP 서버를 제거하면 다음 어셈블리에서 올바르게 직접 노출로 돌아갑니다.

## 설정 (Configuration)

```yaml
tools:
  tool_search:
    enabled: auto       # auto (기본값), on, 또는 off
    threshold_pct: 10   # 컨텍스트의 백분율 — auto 모드에서만 사용됨
    search_default_limit: 5
    max_search_limit: 20
```

| 키 | 기본값 | 의미 |
| --- | --- | --- |
| `enabled` | `auto` | `auto`는 임계값 위에서 활성화됩니다. `on`은 지연 가능한 도구가 하나라도 있으면 항상 활성화됩니다. `off`는 완전히 비활성화합니다. |
| `threshold_pct` | `10` | `auto` 모드가 시작되는 컨텍스트 길이의 백분율입니다. 범위 0–100. |
| `search_default_limit` | `5` | 모델이 `limit` 없이 `tool_search`를 호출할 때 반환되는 히트 수입니다. |
| `max_search_limit` | `20` | 모델이 `limit`를 통해 요청할 수 있는 하드 상한선입니다. 범위 1–50. |

레거시 부울(boolean) 형태로도 전환할 수 있습니다.

```yaml
tools:
  tool_search: true   # {enabled: auto}와 동일
```

## 언제 사용하지 말아야 하나요

Tool Search는 지연된 스키마에 대한 절약을 위해 턴당 고정 토큰 비용(세 개의 브리지 도구 스키마, 약 300 토큰)과 적어도 한 번의 추가 왕복(검색 → 설명 → 호출)을 거래합니다. 도구가 많고 턴당 도구를 적게 사용할 때는 확실한 이득이 됩니다. 전체 도구 수가 적을 때는 오버헤드가 됩니다.

`auto` 기본값이 이를 처리합니다. 무조건 `enabled: on`으로 설정하면 작은 툴셋에서 턴당 약간의 비용이 발생할 것으로 예상해야 합니다.

## 피할 수 없는 절충점 (Trade-offs that don't go away)

이러한 점들은 프롬프트 캐시 무결성(prompt-cache integrity) 불변성에서 비롯됩니다. 이 구현에 국한된 것이 아니라 모든 점진적 공개(progressive-disclosure) 설계에 내재되어 있습니다.

- **콜드 도구에 대한 한 번의 추가 왕복(One extra round trip on cold tools).** 모델이 지연된 도구를 처음 필요로 할 때, 스키마를 찾아 로드하는 데 한두 번의 추가 모델 호출을 소비합니다. 정적 측면에서의 토큰 절약은 실제이지만 일부는 런타임에 지불됩니다.
- **지연된 스키마에 대한 캐시 이점 없음(No cache benefit on deferred schemas).** 로드된 `tool_describe` 결과는 대화 기록에 들어가므로 후속 턴에서는 캐시되지만 시스템 프롬프트 캐시 접두사의 이점을 얻지는 못합니다.
- **모델 품질 의존성(Model-quality dependence).** Tool Search는 모델이 원하는 도구에 대해 합리적인 검색 쿼리를 작성할 수 있다고 가정합니다. 작은 모델은 이를 잘 수행하지 못합니다. 발표된 Anthropic 수치(Tool search 유무에 따른 Opus 4의 49% → 74%)는 상향적인 측면을 보여주지만 동시에 여전히 ~26 포인트의 정확도가 검색 실패라는 것을 보여줍니다.
- **툴셋 편집 시 캐시 무효화(Toolset edits invalidate cache).** 세션 중간에 도구를 추가하거나 제거하면 브리지 도구의 설명(지연된 도구의 수가 포함됨)과 카탈로그가 변경되므로 프롬프트 캐시가 무효화됩니다. 이것은 모든 툴셋 편집과 동일한 절충점입니다.

## 구현 세부 사항 (Implementation details)

- **검색(Retrieval):** 토큰화된 도구 이름 + 설명 + 매개변수 이름에 대한 BM25. BM25가 긍정적인 점수의 히트를 반환하지 않을 때 도구 이름의 리터럴 하위 문자열 일치로 폴백합니다. 이는 제로 IDF의 퇴화된 사례(예: 모든 도구 이름에 "github"가 포함된 카탈로그에서 `"github"` 검색)를 방지합니다.
- **카탈로그는 턴 간에 상태 비저장입니다(Catalog is stateless across turns).** 매 어셈블리마다 현재 도구 정의 목록에서 다시 빌드됩니다 — 세션 키 `Map`이 없습니다. 이것은 저장된 카탈로그가 라이브 도구 레지스트리와 동기화되지 않는(drift out of sync) 클래스의 버그를 방지합니다.
- **카탈로그의 범위는 세션의 툴셋으로 지정됩니다.** `tool_search`, `tool_describe`, `tool_call`은 세션에 실제로 부여된 도구만 보고 호출합니다. 도구 집합의 하위 집합으로 제한된 하위 에이전트, 칸반 워커 또는 게이트웨이 세션은 브리지를 사용하여 해당 하위 집합 외부의 도구를 발견하거나 호출할 수 없습니다 — 지연된 카탈로그는 전체 프로세스 레지스트리가 아니라 세션 자체의 활성화/비활성화된 툴셋 중 지연 가능한 부분입니다.
- **JS 샌드박스 없음.** Hermes는 더 간단한 "구조화된 도구(structured tools)" 모드(검색 / 설명 / 일반 함수로 호출)를 사용합니다. 일부 다른 구현이 제공하는 JS 샌드박스 "코드 모드"는 표면적이 큽니다. 우리는 이를 건너뜁니다.

## 참고 항목

- `tools/tool_search.py` — 구현
- `tests/tools/test_tool_search.py` — 회귀(regression) 제품군
- 설계를 구체화한 연구에 대한 원본 구현 PR의 `openclaw-tool-search-report` PDF
