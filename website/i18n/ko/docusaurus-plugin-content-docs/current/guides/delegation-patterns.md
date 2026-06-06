---
sidebar_position: 13
title: "위임 및 병렬 작업 (Delegation & Parallel Work)"
description: "서브에이전트 위임을 언제, 어떻게 사용해야 하는가 — 병렬 리서치, 코드 리뷰 및 다중 파일 작업을 위한 패턴"
---

# 위임 및 병렬 작업 (Delegation & Parallel Work)

Hermes는 하위 작업(subtasks)을 병렬로 처리할 수 있도록 격리된 자식 에이전트(서브에이전트)를 생성할 수 있습니다. 각 서브에이전트는 자체 대화, 터미널 세션, 도구 모음을 갖습니다. 최종 요약본만 반환되며 — 중간 과정의 도구 호출 내역은 컨텍스트 창(context window)에 들어가지 않습니다.

전체 기능 레퍼런스는 [Subagent Delegation](/user-guide/features/delegation)을 참조하세요.

---

## 언제 위임해야 하는가 (When to Delegate)

**위임하기 좋은 사례:**
- 깊은 추론이 필요한 하위 작업 (디버깅, 코드 리뷰, 리서치 종합)
- 중간 데이터로 인해 컨텍스트가 넘쳐날 우려가 있는 작업
- 병렬적이고 독립적인 워크스트림 (리서치 A와 B를 동시에 진행)
- 에이전트가 선입견 없이 깨끗한 상태(fresh-context)에서 접근하기를 원할 때

**다른 방법을 사용해야 하는 경우:**
- 단일 도구 호출 → 도구를 직접 사용하세요
- 각 단계 사이에 로직이 들어가는 기계적인 다단계 작업 → `execute_code` 사용
- 사용자의 상호작용이 필요한 작업 → 서브에이전트는 `clarify`를 사용할 수 없습니다
- 빠른 파일 편집 → 직접 편집하세요
- 현재 턴을 넘어서 지속되어야 하는 장기 실행 작업 → `cronjob` 또는 `terminal(background=True, notify_on_complete=True)`을 사용하세요. `delegate_task`는 **동기식(synchronous)**입니다: 부모 턴이 중단되면 활성화된 자식들도 취소되고 그들의 작업은 폐기됩니다.

---

## 패턴: 병렬 리서치

세 가지 주제를 동시에 리서치하고 구조화된 요약본을 받아보세요:

```
다음 세 가지 주제를 병렬로 조사해줘:
1. 브라우저 외부에서의 WebAssembly의 현재 상태
2. 2025년 RISC-V 서버 칩 채택 현황
3. 실용적인 양자 컴퓨팅 응용 사례

최근 동향과 주요 플레이어에 초점을 맞춰줘.
```

내부적으로 Hermes는 다음과 같이 사용합니다:

```python
delegate_task(tasks=[
    {
        "goal": "2025년 브라우저 외부 WebAssembly 리서치",
        "context": "초점: 런타임(Wasmtime, Wasmer), 클라우드/엣지 사용 사례, WASI 진행 상황",
        "toolsets": ["web"]
    },
    {
        "goal": "RISC-V 서버 칩 채택 현황 리서치",
        "context": "초점: 출시되는 서버 칩, 도입 중인 클라우드 제공업체, 소프트웨어 생태계",
        "toolsets": ["web"]
    },
    {
        "goal": "실용적인 양자 컴퓨팅 응용 사례 리서치",
        "context": "초점: 오류 수정의 돌파구, 실제 사용 사례, 주요 기업",
        "toolsets": ["web"]
    }
])
```

세 가지 작업이 동시에 실행됩니다. 각 서브에이전트는 독립적으로 웹을 검색하고 요약을 반환합니다. 그런 다음 부모 에이전트가 이를 일관된 브리핑으로 종합합니다.

---

## 패턴: 코드 리뷰

선입견 없이 코드에 접근할 수 있도록 깨끗한 컨텍스트의 서브에이전트에게 보안 리뷰를 위임하세요:

```
src/auth/ 에 있는 인증 모듈의 보안 문제를 검토해줘.
SQL 인젝션, JWT 검증 문제, 비밀번호 처리 및 세션 관리를 확인해.
발견된 문제는 수정하고 테스트를 실행해줘.
```

핵심은 `context` 필드입니다 — 서브에이전트에게 필요한 모든 정보가 포함되어야 합니다:

```python
delegate_task(
    goal="src/auth/ 의 보안 문제를 검토하고 발견된 내용 수정하기",
    context="""프로젝트 경로: /home/user/webapp. Python 3.11, Flask, PyJWT, bcrypt 사용.
    인증 파일: src/auth/login.py, src/auth/jwt.py, src/auth/middleware.py
    테스트 명령어: pytest tests/auth/ -v
    초점: SQL 인젝션, JWT 검증, 비밀번호 해싱, 세션 관리.
    발견된 문제를 수정하고 테스트가 통과하는지 확인하세요.""",
    toolsets=["terminal", "file"]
)
```

:::warning 컨텍스트 문제 (The Context Problem)
서브에이전트는 귀하의 기존 대화에 대해 **전혀 모릅니다**. 완전히 새로운 상태에서 시작합니다. "우리가 이야기하던 버그를 수정해"라고 위임하면, 서브에이전트는 어떤 버그인지 알 수 없습니다. 항상 파일 경로, 오류 메시지, 프로젝트 구조 및 제약 조건을 명시적으로 전달하세요.
:::

---

## 패턴: 대안 비교 (Compare Alternatives)

동일한 문제에 대한 여러 접근 방식을 병렬로 평가한 후 가장 좋은 것을 선택하세요:

```
우리의 Django 앱에 전체 텍스트 검색(full-text search)을 추가해야 해. 다음 세 가지 접근 방식을 병렬로 평가해줘:
1. PostgreSQL tsvector (내장)
2. django-elasticsearch-dsl을 통한 Elasticsearch
3. meilisearch-python을 통한 Meilisearch

각각에 대해: 설정 복잡도, 쿼리 기능, 리소스 요구 사항, 유지 관리 오버헤드를 평가해. 
비교하고 하나를 추천해줘.
```

각 서브에이전트가 독립적으로 한 가지 옵션을 리서치합니다. 격리되어 있기 때문에 교차 오염이 없으며 — 각 평가는 그 자체의 장단점에 기반합니다. 부모 에이전트는 세 가지 요약을 모두 받아 비교를 수행합니다.

---

## 패턴: 다중 파일 리팩토링

대규모 리팩토링 작업을 병렬 서브에이전트들에게 나누어, 각각 코드베이스의 다른 부분을 처리하게 하세요:

```python
delegate_task(tasks=[
    {
        "goal": "모든 API 엔드포인트 핸들러가 새 응답 형식을 사용하도록 리팩토링",
        "context": """프로젝트 경로: /home/user/api-server.
        파일: src/handlers/users.py, src/handlers/auth.py, src/handlers/billing.py
        이전 형식: return {"data": result, "status": "ok"}
        새 형식: return APIResponse(data=result, status=200).to_dict()
        가져오기: from src.responses import APIResponse
        이후 테스트 실행: pytest tests/handlers/ -v""",
        "toolsets": ["terminal", "file"]
    },
    {
        "goal": "모든 클라이언트 SDK 메서드가 새 응답 형식을 처리하도록 업데이트",
        "context": """프로젝트 경로: /home/user/api-server.
        파일: sdk/python/client.py, sdk/python/models.py
        이전 파싱: result = response.json()["data"]
        새 파싱: result = response.json()["data"] (키는 동일하나 상태 코드 확인 추가)
        sdk/python/tests/test_client.py 도 업데이트할 것""",
        "toolsets": ["terminal", "file"]
    },
    {
        "goal": "새 응답 형식을 반영하도록 API 문서 업데이트",
        "context": """프로젝트 경로: /home/user/api-server.
        문서 위치: docs/api/. 형식: 코드 예제가 포함된 Markdown.
        모든 응답 예제를 이전 형식에서 새 형식으로 업데이트.
        docs/api/overview.md 에 스키마를 설명하는 '응답 형식(Response Format)' 섹션 추가.""",
        "toolsets": ["terminal", "file"]
    }
])
```

:::tip
각 서브에이전트는 자체 터미널 세션을 얻습니다. 서로 다른 파일을 편집하는 한 서로 간섭하지 않고 동일한 프로젝트 디렉토리에서 작업할 수 있습니다. 두 서브에이전트가 같은 파일을 건드릴 가능성이 있다면, 병렬 작업이 완료된 후 해당 파일을 직접 처리하세요.
:::

---

## 패턴: 수집 후 분석 (Gather Then Analyze)

기계적인 데이터 수집에는 `execute_code`를 사용하고, 추론이 많이 필요한 분석은 위임하세요:

```python
# 1단계: 기계적 수집 (여기서는 추론이 필요 없으므로 execute_code가 더 낫습니다)
execute_code("""
from hermes_tools import web_search, web_extract

results = []
for query in ["2026년 1분기 AI 투자금", "2026년 AI 스타트업 인수", "2026년 AI IPO"]:
    r = web_search(query, limit=5)
    for item in r["data"]["web"]:
        results.append({"title": item["title"], "url": item["url"], "desc": item["description"]})

# 상위 5개의 가장 관련성 높은 페이지에서 전체 콘텐츠 추출
urls = [r["url"] for r in results[:5]]
content = web_extract(urls)

# 분석 단계를 위해 저장
import json
with open("/tmp/ai-funding-data.json", "w") as f:
    json.dump({"search_results": results, "extracted": content["results"]}, f)
print(f"{len(results)}개의 결과를 수집하고, {len(content['results'])}개의 페이지를 추출했습니다")
""")

# 2단계: 추론이 많은 분석 (여기서는 위임이 더 낫습니다)
delegate_task(
    goal="AI 자금 조달 데이터를 분석하고 시장 보고서 작성",
    context="""/tmp/ai-funding-data.json 에 있는 원시 데이터에는 2026년 1분기 AI 펀딩, 인수 및 IPO에 대한
    검색 결과와 추출된 웹 페이지가 포함되어 있습니다.
    구조화된 시장 보고서를 작성하세요: 주요 거래, 동향, 주목할 만한 기업 및 전망. 
    1억 달러 이상의 거래에 초점을 맞추세요.""",
    toolsets=["terminal", "file"]
)
```

이는 흔히 가장 효율적인 패턴입니다: `execute_code`가 10번 이상의 순차적 도구 호출을 저렴하게 처리하고, 서브에이전트가 깨끗한 컨텍스트에서 비용이 많이 드는 단일 추론 작업을 수행합니다.

---

## 도구 모음 선택 (Toolset Selection)

서브에이전트에게 필요한 것을 기준으로 도구 모음을 선택하세요:

| 작업 유형 | 도구 모음 | 이유 |
|-----------|----------|-----|
| 웹 리서치 | `["web"]` | web_search 및 web_extract만 필요 |
| 코드 작업 | `["terminal", "file"]` | 쉘 액세스 + 파일 작업 필요 |
| 풀 스택 | `["terminal", "file", "web"]` | 메시징을 제외한 모든 도구 필요 |
| 읽기 전용 분석 | `["file"]` | 파일만 읽을 수 있으며 쉘은 불필요 |

도구 모음을 제한하면 서브에이전트가 집중력을 유지하고 우발적인 부작용(예: 리서치 서브에이전트가 쉘 명령어를 실행하는 것)을 방지할 수 있습니다.

---

## 제약 조건 (Constraints)

- **기본 3개의 병렬 작업**: 한 번에 실행되는 서브에이전트의 기본 동시성(concurrency)은 3입니다 (config.yaml의 `delegation.max_concurrent_children`을 통해 설정 가능, 상한선 없음, 하한선은 1).
- **중첩 위임은 옵트인(opt-in) 방식**: 최하위 서브에이전트(기본값)는 `delegate_task`, `clarify`, `memory`, `send_message`, `execute_code`를 호출할 수 없습니다. 오케스트레이터 서브에이전트(`role="orchestrator"`)는 추가 위임을 위해 `delegate_task`를 유지하지만, 이는 `delegation.max_spawn_depth`가 기본값인 1 이상으로 높아졌을 때만 가능합니다 (하한 1, 상한 없음). 나머지 4개의 도구는 여전히 차단됩니다. `delegation.orchestrator_enabled: false`를 통해 전역적으로 비활성화할 수 있습니다.

### 동시성 및 깊이 튜닝

| 설정 | 기본값 | 범위 | 효과 |
|--------|---------|-------|--------|
| `max_concurrent_children` | 3 | >=1 | `delegate_task` 호출당 병렬 배치 크기 |
| `max_spawn_depth` | 1 | >=1 | 서브에이전트가 자식을 스폰(생성)할 수 있는 위임 깊이 |

예시: 중첩 서브에이전트를 포함하여 30개의 병렬 작업자 실행:

```yaml
delegation:
  max_concurrent_children: 30
  max_spawn_depth: 2
```

- **개별 터미널** — 각 서브에이전트는 별도의 작업 디렉토리와 상태를 가진 자체 터미널 세션을 얻습니다.
- **대화 기록 없음** — 서브에이전트는 부모 에이전트가 `delegate_task` 호출 시 전달하는 `goal`과 `context`만 봅니다.
- **기본 50회 반복** — 간단한 작업의 경우 비용을 절약하기 위해 `max_iterations`를 더 낮게 설정하세요.
- **지속성 없음 (Not durable)** — `delegate_task`는 동기식이며 부모 턴 내부에서 실행됩니다. 부모가 중단되면(새로운 사용자 메시지, `/stop`, `/new`) 활성화된 모든 자식은 취소(`status="interrupted"`)되고 그들의 작업은 폐기됩니다. 현재 턴을 넘어 유지되어야 하는 작업의 경우 `cronjob`이나 `terminal(background=True, notify_on_complete=True)`을 사용하세요.

---

## 팁 (Tips)

**목표를 구체적으로 설정하세요.** "버그 수정해"는 너무 모호합니다. "api/handlers.py 47번째 줄에서 process_request()가 parse_body()로부터 None을 받는 곳의 TypeError를 수정해"가 서브에이전트가 작업하기에 충분한 정보입니다.

**파일 경로를 포함하세요.** 서브에이전트는 프로젝트 구조를 모릅니다. 관련 파일에 대한 절대 경로, 프로젝트 루트 및 테스트 명령어를 항상 포함하세요.

**컨텍스트 격리를 위해 위임을 사용하세요.** 때로는 신선한 시각이 필요할 때가 있습니다. 위임을 하면 문제를 명확하게 설명하게 되며, 서브에이전트는 진행 중이던 대화에 쌓인 가정이 없는 상태에서 문제에 접근합니다.

**결과를 확인하세요.** 서브에이전트의 요약은 그야말로 요약일 뿐입니다. 서브에이전트가 "버그를 수정했고 테스트를 통과했습니다"라고 말하더라도, 직접 테스트를 실행하거나 diff(변경 사항)를 읽어 확인하세요.

---

*모든 매개변수, ACP 통합 및 고급 구성을 포함한 전체 위임 레퍼런스는 [Subagent Delegation](/user-guide/features/delegation)을 참조하세요.*
