---
sidebar_position: 13
title: "위임 및 병렬 작업"
description: "병렬 조사, 코드 리뷰, 다중 파일 작업에서 서브에이전트 위임을 언제 어떻게 사용할지 설명합니다"
---

# 위임 및 병렬 작업

Hermes는 격리된 자식 에이전트를 생성하여 작업을 병렬로 수행할 수 있습니다. 각 서브에이전트는 자체 대화, 터미널 세션, 도구 세트를 가집니다. 최종 요약만 돌아오며, 중간 도구 호출은 컨텍스트 창에 들어오지 않습니다.

전체 기능 참고 자료는 [서브에이전트 위임](/user-guide/features/delegation)을 참조하세요.

---

## 위임할 때

**위임에 적합한 후보:**
- 추론이 많이 필요한 하위 작업(디버깅, 코드 리뷰, 조사 결과 종합)
- 중간 데이터가 컨텍스트를 가득 채울 작업
- 서로 독립적인 병렬 작업 흐름(조사 A와 B를 동시에 수행)
- 편견 없이 접근하도록 새 컨텍스트가 필요한 작업

**다른 것을 사용하세요:**
- 단일 도구 호출 → 도구를 직접 사용합니다
- 중간에 로직이 필요한 기계적인 다단계 작업 → `execute_code`
- 사용자 상호작용이 필요한 작업 → 서브에이전트는 `clarify`를 사용할 수 없습니다
- 빠른 파일 편집 → 직접 수행합니다
- 세션 종료 또는 프로세스 재시작 후에도 살아남아야 하는 지속적인 장기 작업 → `cronjob` 또는 `terminal(background=True, notify_on_complete=True)`. 최상위 위임은 비동기이지만 여전히 프로세스 로컬입니다.

---

## 패턴: 병렬 조사

세 가지 주제를 동시에 조사하고 구조화된 요약을 받습니다:

```
Research these three topics in parallel:
1. Current state of WebAssembly outside the browser
2. RISC-V server chip adoption in 2025
3. Practical quantum computing applications

Focus on recent developments and key players.
```

내부적으로 Hermes는 다음을 사용합니다:

```python
delegate_task(tasks=[
    {
        "goal": "Research WebAssembly outside the browser in 2025",
        "context": "Focus on: runtimes (Wasmtime, Wasmer), cloud/edge use cases, WASI progress"
    },
    {
        "goal": "Research RISC-V server chip adoption",
        "context": "Focus on: server chips shipping, cloud providers adopting, software ecosystem"
    },
    {
        "goal": "Research practical quantum computing applications",
        "context": "Focus on: error correction breakthroughs, real-world use cases, key companies"
    }
])
```

세 작업은 모두 동시에 실행됩니다. 각 서브에이전트는 독립적으로 웹을 검색하고 요약을 반환합니다. 그러면 부모 에이전트가 이를 하나의 일관된 브리핑으로 종합합니다.

---

## 패턴: 코드 리뷰

선입견 없이 접근하는 새 컨텍스트의 서브에이전트에 보안 검토를 위임합니다:

```
Review the authentication module at src/auth/ for security issues.
Check for SQL injection, JWT validation problems, password handling,
and session management. Fix anything you find and run the tests.
```

핵심은 `context` 필드입니다 — 서브에이전트가 필요로 하는 모든 것을 포함해야 합니다:

```python
delegate_task(
    goal="Review src/auth/ for security issues and fix any found",
    context="""Project at /home/user/webapp. Python 3.11, Flask, PyJWT, bcrypt.
    Auth files: src/auth/login.py, src/auth/jwt.py, src/auth/middleware.py
    Test command: pytest tests/auth/ -v
    Focus on: SQL injection, JWT validation, password hashing, session management.
    Fix issues found and verify tests pass."""
)
```

:::warning 컨텍스트 문제
서브에이전트는 여러분의 대화에 대해 **전혀 아무것도 모릅니다.** 완전히 새로 시작하며, 부모가 전달한 `goal`과 `context`만 받습니다. "우리가 논의하던 버그를 수정해"라고 위임하면 서브에이전트는 어떤 버그인지 알 수 없습니다. 항상 파일 경로, 오류 메시지, 프로젝트 구조, 제약 조건을 함께 전달하세요.
:::

---

## 패턴: 대안 비교

같은 문제에 대한 여러 접근 방식을 병렬로 평가한 다음 최선의 방법을 선택합니다:

```
I need to add full-text search to our Django app. Evaluate three approaches
in parallel:
1. PostgreSQL tsvector (built-in)
2. Elasticsearch via django-elasticsearch-dsl
3. Meilisearch via meilisearch-python

For each: setup complexity, query capabilities, resource requirements,
and maintenance overhead. Compare them and recommend one.
```

각 서브에이전트는 하나의 선택지를 독립적으로 조사합니다. 격리되어 있으므로 서로 오염되지 않으며, 각 평가는 고유한 장점에 따라 독립적으로 이루어집니다. 부모 에이전트는 세 가지 요약을 받아 비교하고 추천합니다.

---

## 패턴: 다중 파일 리팩터링

대규모 리팩터링 작업을 병렬 서브에이전트로 나누고, 각 서브에이전트가 코드베이스의 서로 다른 부분을 담당하게 합니다:

```python
delegate_task(tasks=[
    {
        "goal": "Refactor all API endpoint handlers to use the new response format",
        "context": """Project at /home/user/api-server.
        Files: src/handlers/users.py, src/handlers/auth.py, src/handlers/billing.py
        Old format: return {"data": result, "status": "ok"}
        New format: return APIResponse(data=result, status=200).to_dict()
        Import: from src.responses import APIResponse
        Run tests after: pytest tests/handlers/ -v"""
    },
    {
        "goal": "Update all client SDK methods to handle the new response format",
        "context": """Project at /home/user/api-server.
        Files: sdk/python/client.py, sdk/python/models.py
        Old parsing: result = response.json()["data"]
        New parsing: result = response.json()["data"] (same key, but add status code checking)
        Also update sdk/python/tests/test_client.py"""
    },
    {
        "goal": "Update API documentation to reflect the new response format",
        "context": """Project at /home/user/api-server.
        Docs at: docs/api/. Format: Markdown with code examples.
        Update all response examples from old format to new format.
        Add a 'Response Format' section to docs/api/overview.md explaining the schema."""
    }
])
```

:::tip
각 서브에이전트는 별도의 터미널 세션을 가집니다. 서로 다른 파일을 편집하는 한 같은 프로젝트 디렉터리에서 서로 방해하지 않고 작업할 수 있습니다. 두 서브에이전트가 같은 파일을 건드릴 가능성이 있다면 해당 파일은 직접 처리하세요.
:::

---

## 패턴: 수집 후 분석

기계적인 데이터 수집에는 `execute_code`를 사용한 다음, 추론이 많이 필요한 분석은 위임합니다:

```python
# Step 1: Mechanical gathering (execute_code is better here — no reasoning needed)
execute_code("""
from hermes_tools import web_search, web_extract

results = []
for query in ["AI funding Q1 2026", "AI startup acquisitions 2026", "AI IPOs 2026"]:
    r = web_search(query, limit=5)
    for item in r["data"]["web"]:
        results.append({"title": item["title"], "url": item["url"], "desc": item["description"]})

# Extract full content from top 5 most relevant
urls = [r["url"] for r in results[:5]]
content = web_extract(urls)

# Save for the analysis step
import json
with open("/tmp/ai-funding-data.json", "w") as f:
    json.dump({"search_results": results, "extracted": content["results"]}, f)
print(f"Collected {len(results)} results, extracted {len(content['results'])} pages")
""")

# Step 2: Reasoning-heavy analysis (delegation is better here)
delegate_task(
    goal="Analyze AI funding data and write a market report",
    context="""Raw data at /tmp/ai-funding-data.json contains search results and
    extracted web pages about AI funding, acquisitions, and IPOs in Q1 2026.
    Write a structured market report: key deals, trends, notable players,
    and outlook. Focus on deals over $100M."""
)
```

이는 흔히 가장 효율적인 방식입니다. `execute_code`가 10개 이상의 순차적인 도구 호출을 저렴하게 처리하고, 이후 서브에이전트가 깔끔한 컨텍스트에서 단 한 번의 비용 높은 추론 작업을 수행합니다.

---

## 상속되는 도구 액세스

서브에이전트는 부모의 활성화된 도구 세트를 상속합니다. `delegate_task`는 모델을 향한 `toolsets` 매개변수를 받지 않으므로, 위임된 작업은 스스로 기능을 추가할 수 없습니다. 자식에게 웹, 터미널, 파일 또는 기타 액세스가 필요하다면 대화를 시작하기 전에 부모의 도구를 구성하세요. Hermes는 `clarify`, `memory`, `send_message`처럼 자식에게 차단된 도구를 계속 제거하지만, 자식은 프로그래밍 방식의 도구 호출을 위해 `execute_code`를 유지합니다.

---

## 제약 조건

- **기본 병렬 작업 3개**: 배치의 기본값은 동시에 실행되는 서브에이전트 3개입니다(`delegation.max_concurrent_children`로 구성 가능하며, 하드 상한은 없고 최솟값은 1)
- **중첩 위임은 선택 사항**: 리프 서브에이전트(기본값)는 `delegate_task`, `clarify`, `memory`, `execute_code`를 호출할 수 없습니다. 오케스트레이터 서브에이전트(`role="orchestrator"`)는 추가 위임을 위해 `delegate_task`를 유지하지만, `delegation.max_spawn_depth`가 기본값 1보다 높을 때만 가능합니다(최솟값 1, 상한 없음). 나머지 세 가지는 계속 차단됩니다. `delegation.orchestrator_enabled: false`로 전역 비활성화할 수 있습니다.

### 동시성 및 깊이 조정

| 설정 | 기본값 | 범위 | 효과 |
|--------|---------|-------|--------|
| `max_concurrent_children` | 3 | >=1 | 배치별 동시 실행 수 |
| `max_spawn_depth` | 1 | >=1 | 위임할 수 있는 단계 수 |

서브에이전트가 중첩 위임을 수행하는 30개의 병렬 작업자를 실행하는 예:

```yaml
delegation:
  max_concurrent_children: 30
  max_spawn_depth: 2
```

- **별도 터미널** — 각 서브에이전트는 별도의 터미널 세션과 작업 디렉터리, 상태를 가집니다
- **대화 기록 없음** — 서브에이전트에는 부모가 전달하는 `goal`과 `context`만 보이며 대화 기록은 보이지 않습니다
- **기본 50회 반복** — 단순한 작업은 비용을 절약하도록 `max_iterations`를 낮게 설정합니다
- **영속적이지 않음** — 최상위 위임은 백그라운드에서 실행되고 나중에 결과를 게시하지만, 소유 세션과 Hermes 프로세스에 연결되어 있습니다. 세션 종료, `/stop`, `/new`, 또는 프로세스 재시작으로 진행 중인 작업이 취소되거나 고립될 수 있습니다. 이러한 경계를 넘어 살아남아야 하는 작업에는 `cronjob` 또는 `terminal(background=True, notify_on_complete=True)`를 사용하세요.

---

## 팁

**목표를 구체적으로 작성하세요.** "버그를 수정해"는 너무 모호합니다. "`process_request()`가 `parse_body()`에서 `None`을 받아 `api/handlers.py` 47번째 줄에서 발생하는 TypeError를 수정해"라고 하면 서브에이전트가 충분한 정보를 얻습니다.

**파일 경로를 포함하세요.** 서브에이전트는 프로젝트 구조를 모릅니다. 관련 파일의 절대 경로, 프로젝트 루트, 테스트 명령을 항상 포함하세요.

**컨텍스트 격리를 위해 위임을 사용하세요.** 때로는 새로운 관점이 필요합니다. 위임하면 문제를 명확히 설명해야 하므로, 서브에이전트는 대화 중 형성된 가정 없이 접근합니다.

**결과를 확인하세요.** 서브에이전트의 요약은 어디까지나 요약일 뿐입니다. 서브에이전트가 "수정했고 테스트가 통과했다"고 말하더라도 직접 테스트를 실행하거나 diff를 읽어 확인하세요.

---

*전체 위임 참고 자료 — 모든 매개변수, ACP 통합, 고급 설정은 [위임](/user-guide/features/delegation)을 참조하세요.*
