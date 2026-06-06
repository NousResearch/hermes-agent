---
sidebar_position: 8
title: "코드 실행"
description: "RPC 도구 액세스를 사용한 프로그래밍 방식의 Python 실행 — 다단계 워크플로우를 단일 턴으로 압축합니다."
---

# 코드 실행 (프로그래밍 방식의 도구 호출)

`execute_code` 도구를 사용하면 에이전트가 Hermes 도구를 프로그래밍 방식으로 호출하는 Python 스크립트를 작성하여 다단계 워크플로우를 단일 LLM 턴으로 압축할 수 있습니다. 스크립트는 에이전트 호스트의 하위 프로세스에서 실행되며, Unix 도메인 소켓 RPC를 통해 Hermes와 통신합니다.

## 작동 방식

1. 에이전트가 `from hermes_tools import ...`를 사용하여 Python 스크립트를 작성합니다.
2. Hermes는 RPC 함수가 포함된 `hermes_tools.py` 스텁(stub) 모듈을 생성합니다.
3. Hermes는 Unix 도메인 소켓을 열고 RPC 리스너 스레드를 시작합니다.
4. 스크립트는 하위 프로세스에서 실행됩니다 — 도구 호출은 소켓을 통해 다시 Hermes로 전달됩니다.
5. 스크립트의 `print()` 출력만 LLM으로 반환되며, 중간 도구 결과는 컨텍스트 창(context window)에 들어가지 않습니다.

```python
# 에이전트는 다음과 같은 스크립트를 작성할 수 있습니다:
from hermes_tools import web_search, web_extract

results = web_search("Python 3.13 features", limit=5)
for r in results["data"]["web"]:
    content = web_extract([r["url"]])
    # ... 필터링 및 처리 ...
print(summary)
```

**스크립트 내부에서 사용 가능한 도구:** `web_search`, `web_extract`, `read_file`, `write_file`, `search_files`, `patch`, `terminal` (포그라운드 전용).

## 에이전트가 이 기능을 사용하는 경우

에이전트는 다음과 같은 경우 `execute_code`를 사용합니다:

- 호출 사이에 처리 로직이 있는 **3개 이상의 도구 호출**
- 대량의 데이터 필터링 또는 조건 분기
- 결과에 대한 반복 루프(Loops over results)

핵심 이점: 중간 도구 결과는 컨텍스트 창에 들어가지 않습니다. 최종 `print()` 출력만 반환되므로 토큰 사용량이 극적으로 감소합니다.

## 실용적인 예

### 데이터 처리 파이프라인

```python
from hermes_tools import search_files, read_file
import json

# 모든 설정 파일을 찾고 데이터베이스 설정을 추출
matches = search_files("database", path=".", file_glob="*.yaml", limit=20)
configs = []
for match in matches.get("matches", []):
    content = read_file(match["path"])
    configs.append({"file": match["path"], "preview": content["content"][:200]})

print(json.dumps(configs, indent=2))
```

### 다단계 웹 리서치

```python
from hermes_tools import web_search, web_extract
import json

# 검색, 추출, 요약을 단 한 번에 처리
results = web_search("Rust async runtime comparison 2025", limit=5)
summaries = []
for r in results["data"]["web"]:
    page = web_extract([r["url"]])
    for p in page.get("results", []):
        if p.get("content"):
            summaries.append({
                "title": r["title"],
                "url": r["url"],
                "excerpt": p["content"][:500]
            })

print(json.dumps(summaries, indent=2))
```

### 대량 파일 리팩토링

```python
from hermes_tools import search_files, read_file, patch

# 사용되지 않는 API를 사용하는 모든 Python 파일을 찾아 수정
matches = search_files("old_api_call", path="src/", file_glob="*.py")
fixed = 0
for match in matches.get("matches", []):
    result = patch(
        path=match["path"],
        old_string="old_api_call(",
        new_string="new_api_call(",
        replace_all=True
    )
    if "error" not in str(result):
        fixed += 1

print(f"Fixed {fixed} files out of {len(matches.get('matches', []))} matches")
```

### 빌드 및 테스트 파이프라인

```python
from hermes_tools import terminal, read_file
import json

# 테스트 실행, 결과 구문 분석 및 보고
result = terminal("cd /project && python -m pytest --tb=short -q 2>&1", timeout=120)
output = result.get("output", "")

# 테스트 출력 파싱
passed = output.count(" passed")
failed = output.count(" failed")
errors = output.count(" error")

report = {
    "passed": passed,
    "failed": failed,
    "errors": errors,
    "exit_code": result.get("exit_code", -1),
    "summary": output[-500:] if len(output) > 500 else output
}

print(json.dumps(report, indent=2))
```

## 실행 모드

`execute_code`에는 `~/.hermes/config.yaml`의 `code_execution.mode`로 제어되는 두 가지 실행 모드가 있습니다:

| 모드 | 작업 디렉토리 | Python 인터프리터 |
|------|-------------------|--------------------|
| **`project`** (기본값) | 세션의 작업 디렉토리 (`terminal()`과 동일) | 활성화된 `VIRTUAL_ENV` / `CONDA_PREFIX` python, 실패 시 Hermes 자체 python 사용 |
| `strict` | 사용자의 프로젝트와 격리된 임시 스테이징 디렉토리 | `sys.executable` (Hermes 자체 python) |

**`project` 모드를 그대로 두어야 하는 경우:** `terminal()`에서와 동일하게 `import pandas`, `from my_project import foo`, 또는 `open(".env")`와 같은 상대 경로가 작동하기를 원할 때입니다. 이것이 대부분의 경우에 필요합니다.

**`strict`로 전환해야 하는 경우:** 최대의 재현성이 필요한 경우입니다 — 사용자가 활성화한 venv에 관계없이 매 세션마다 동일한 인터프리터를 원하거나, 상대 경로를 통해 프로젝트 파일을 실수로 읽을 위험을 막기 위해 스크립트를 프로젝트 트리에서 격리하려는 경우입니다.

```yaml
# ~/.hermes/config.yaml
code_execution:
  mode: project   # 또는 "strict"
```

`project` 모드에서의 대체 동작: `VIRTUAL_ENV` / `CONDA_PREFIX`가 설정되지 않았거나, 깨졌거나, 3.8 이전의 Python을 가리키는 경우 리졸버는 `sys.executable`로 깨끗하게 대체합니다 — 에이전트가 작동하는 인터프리터 없이 남겨지는 일은 없습니다.

보안에 중요한 불변 규칙(invariants)은 두 모드에서 동일합니다:

- 환경 정화 (API 키, 토큰, 자격 증명 제거)
- 도구 허용 목록 (스크립트는 `execute_code`, `delegate_task` 또는 MCP 도구를 재귀적으로 호출할 수 없음)
- 리소스 제한 (시간 초과, stdout 상한, 도구 호출 상한)

모드 전환은 스크립트가 실행되는 위치와 실행할 인터프리터만 변경할 뿐, 볼 수 있는 자격 증명이나 호출할 수 있는 도구를 변경하지는 않습니다.

## 리소스 제한

| 리소스 | 제한 | 참고 |
|----------|-------|-------|
| **Timeout (시간 초과)** | 5분 (300초) | 스크립트는 SIGTERM으로 강제 종료되며, 5초의 유예 기간 후 SIGKILL로 종료됩니다. |
| **Stdout** | 50 KB | 출력이 잘리며 `[output truncated at 50KB]` 알림이 표시됩니다. |
| **Stderr** | 10 KB | 디버깅을 위해 0이 아닌 종료 코드가 발생할 때 출력에 포함됩니다. |
| **Tool calls (도구 호출)** | 실행당 50회 | 한도에 도달하면 오류가 반환됩니다. |

모든 제한은 `config.yaml`을 통해 구성 가능합니다:

```yaml
# ~/.hermes/config.yaml 파일에서
code_execution:
  mode: project      # project (기본값) | strict
  timeout: 300       # 스크립트당 최대 초 단위 시간 (기본값: 300)
  max_tool_calls: 50 # 실행당 최대 도구 호출 횟수 (기본값: 50)
```

## 스크립트 내부 도구 호출의 작동 방식

스크립트에서 `web_search("query")`와 같은 함수를 호출하면:

1. 호출은 JSON으로 직렬화되어 Unix 도메인 소켓을 통해 부모 프로세스로 전송됩니다.
2. 부모는 표준 `handle_function_call` 핸들러를 통해 디스패치합니다.
3. 결과는 소켓을 통해 다시 전송됩니다.
4. 함수는 파싱된 결과를 반환합니다.

즉, 스크립트 내부의 도구 호출은 일반적인 도구 호출과 동일하게 동작합니다 — 동일한 속도 제한, 동일한 오류 처리, 동일한 기능이 적용됩니다. 유일한 제한은 `terminal()`이 포그라운드 전용이라는 점입니다(`background` 또는 `pty` 매개변수 없음).

## 오류 처리

스크립트가 실패하면 에이전트는 구조화된 오류 정보를 수신합니다:

- **0이 아닌 종료 코드 (Non-zero exit code)**: stderr가 출력에 포함되어 에이전트가 전체 트레이스백을 볼 수 있습니다.
- **시간 초과 (Timeout)**: 스크립트가 강제 종료되고 에이전트는 `"Script timed out after 300s and was killed."`를 봅니다.
- **중단 (Interruption)**: 실행 중에 사용자가 새 메시지를 보내면 스크립트가 종료되고 에이전트는 `[execution interrupted — user sent a new message]`를 봅니다.
- **도구 호출 제한 (Tool call limit)**: 50번 호출 제한에 도달하면 후속 도구 호출에서 오류 메시지가 반환됩니다.

응답에는 항상 `status` (success/error/timeout/interrupted), `output`, `tool_calls_made`, 그리고 `duration_seconds`가 포함됩니다.

## 보안

:::danger 보안 모델
하위 프로세스는 **최소한의 환경**으로 실행됩니다. 기본적으로 API 키, 토큰 및 자격 증명은 제거됩니다. 스크립트는 독점적으로 RPC 채널을 통해서만 도구에 액세스할 수 있습니다 — 명시적으로 허용되지 않는 한 환경 변수에서 비밀(secrets)을 읽을 수 없습니다.
:::

이름에 `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `PASSWD` 또는 `AUTH`가 포함된 환경 변수는 제외됩니다. 안전한 시스템 변수(`PATH`, `HOME`, `LANG`, `SHELL`, `PYTHONPATH`, `VIRTUAL_ENV` 등)만 전달됩니다.

### 스킬 환경 변수 패스스루 (Skill Environment Variable Passthrough)

스킬이 프론트매터(frontmatter)에서 `required_environment_variables`를 선언하면 스킬이 로드된 후 해당 변수들은 `execute_code`와 `terminal` 하위 프로세스에 **자동으로 전달(passthrough)**됩니다. 이를 통해 임의의 코드에 대한 보안 태세를 약화시키지 않으면서 스킬이 선언된 API 키를 사용할 수 있습니다.

스킬과 관련되지 않은 사용 사례의 경우 `config.yaml`에 명시적으로 변수를 허용할 수 있습니다:

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

전체 세부 사항은 [보안 가이드](/user-guide/security#environment-variable-passthrough)를 참조하세요.

### 하위 프로세스의 `HERMES_*` 변수

하위 프로세스는 정확한 이름을 기준으로 소수의 고정된 운영용 `HERMES_*` 변수만 수신합니다:

- `HERMES_HOME`
- `HERMES_PROFILE`
- `HERMES_CONFIG`
- `HERMES_ENV`

(여기에 `HERMES_RPC_DIR` / `HERMES_RPC_SOCKET` / `TZ` / `HOME`이 추가되는데, Hermes가 RPC 채널이 작동하도록 명시적으로 주입합니다).

:::note 동작 변경
이전 버전에서는 `HERMES_`로 시작하는 **모든** 변수가 하위 프로세스로 전달되었습니다. 보안 강화를 위해 이 광범위한 접두사가 제거되었습니다: 비밀번호 부분 문자열과 일치하지 않는 `HERMES_*`로 이름 지정된 구성(예: `HERMES_BASE_URL`, `HERMES_KANBAN_DB` 또는 `HERMES_*_WEBHOOK` 엔드포인트)이 임의의 샌드박스화된 코드에 노출될 수 있었기 때문입니다.

`execute_code` 스크립트 — 또는 스크립트가 가져올 때(import) 사용하는 리포지토리/플러그인 모듈 — 가 위의 4가지 운영용 이름 외의 `HERMES_*` 변수에 의존했다면 이제 하위 프로세스에서 해당 변수가 **설정되지 않음(unset)**으로 나타날 것입니다. 이는 의도적인 동작이며 버그가 아닙니다.
:::

**해결 방법 — 변수를 명시적으로 다시 선택하십시오.** 두 경로 모두 변수를 `execute_code` *및* `terminal` 하위 프로세스로 통과시키며, 어느 경로도 비밀 제거 보장을 약화시키지 않습니다(Hermes 관리형 제공자 자격 증명은 이 방법으로 다시 허용될 수 없습니다):

1. **머신당 `config.yaml` 사용** — 정확한 변수 이름을 패스스루 허용 목록에 추가합니다:

   ```yaml
   terminal:
     env_passthrough:
       - HERMES_KANBAN_DB
       - HERMES_BASE_URL
   ```

2. **스킬당 스킬의 프론트매터 사용** — 해당 스킬이 로드될 때마다 자동으로 등록되도록 선언합니다:

   ```yaml
   required_environment_variables:
     - HERMES_KANBAN_DB
   ```

**진단.** 하위 프로세스가 하나 이상의 허용되지 않은 `HERMES_*` 변수를 삭제하면, Hermes는 해당 변수 이름을 명시하고 `env_passthrough` 회피책을 가리키는 한 줄짜리 `debug` 로그를 방출합니다. 디버그 로깅으로 실행하고(`hermes logs --level DEBUG`, 또는 `~/.hermes/logs/agent.log` 확인), 스크립트가 `HERMES_*` 변수가 누락된 것처럼 작동할 경우 `execute_code: dropped N non-allowlisted HERMES_* var(s)` 메시지를 찾으십시오.

Hermes는 항상 실행 후 정리되는 임시 스테이징 디렉토리에 스크립트 및 자동 생성된 `hermes_tools.py` RPC 스텁을 씁니다. `strict` 모드에서는 스크립트가 그곳에서 직접 *실행*됩니다. `project` 모드에서는 세션의 작업 디렉토리에서 실행됩니다(스테이징 디렉토리는 `PYTHONPATH`에 남아 있으므로 import는 여전히 해결됨). 하위 프로세스는 시간 초과 또는 중단 시 깔끔하게 종료될 수 있도록 자체 프로세스 그룹에서 실행됩니다.

## execute_code vs terminal

| 사용 사례 | execute_code | terminal |
|----------|-------------|----------|
| 호출 사이에 도구가 있는 다단계 워크플로우 | ✅ | ❌ |
| 단순한 쉘 명령어 | ❌ | ✅ |
| 대규모 도구 출력 필터링/처리 | ✅ | ❌ |
| 빌드 또는 테스트 제품군 실행 | ❌ | ✅ |
| 검색 결과 반복 처리(Looping) | ✅ | ❌ |
| 대화형/백그라운드 프로세스 | ❌ | ✅ |
| 환경 변수에 API 키가 필요함 | ⚠️ [패스스루](/user-guide/security#environment-variable-passthrough)를 통해서만 | ✅ (대부분 통과됨) |

**경험 법칙:** 프로그램적으로 로직 사이에 Hermes 도구를 호출해야 하는 경우 `execute_code`를 사용하십시오. 쉘 명령, 빌드 및 프로세스를 실행하려면 `terminal`을 사용하십시오.

## 플랫폼 지원

코드 실행은 Unix 도메인 소켓이 필요하며 **Linux 및 macOS에서만 사용 가능**합니다. Windows에서는 자동으로 비활성화되며 — 에이전트는 일반적인 순차적 도구 호출로 폴백합니다.
