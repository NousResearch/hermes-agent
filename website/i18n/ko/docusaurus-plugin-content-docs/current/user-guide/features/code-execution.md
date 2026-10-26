---
sidebar_position: 8
title: "코드 실행"
description: "RPC 도구 액세스를 지원하는 프로그래밍 방식의 Python 실행 — 여러 단계의 워크플로를 한 번의 턴으로 통합"
---

# 코드 실행(프로그래밍 방식의 도구 호출)

`execute_code` 도구를 사용하면 에이전트가 Hermes 도구를 프로그래밍 방식으로 호출하는 Python 스크립트를 작성할 수 있어, 여러 단계의 워크플로를 하나의 LLM 턴으로 통합할 수 있습니다. 스크립트는 에이전트 호스트의 자식 프로세스에서 실행되며, Unix 도메인 소켓 RPC를 통해 Hermes와 통신합니다.

## 작동 방식

1. 에이전트가 `from hermes_tools import ...`를 사용해 Python 스크립트를 작성합니다.
2. Hermes가 RPC 함수를 포함한 `hermes_tools.py` 스텁 모듈을 생성합니다.
3. Hermes가 Unix 도메인 소켓을 열고 RPC 리스너 스레드를 시작합니다.
4. 스크립트가 자식 프로세스에서 실행됩니다. 도구 호출은 소켓을 통해 Hermes로 돌아갑니다.
5. 스크립트의 `print()` 출력만 LLM에 반환되고, 중간 도구 결과는 컨텍스트 창에 들어가지 않습니다.

```python
# The agent can write scripts like:
from hermes_tools import web_search, web_extract

results = web_search("Python 3.13 features", limit=5)
for r in results["data"]["web"]:
    content = web_extract([r["url"]])
    # ... filter and process ...
print(summary)
```

**스크립트 내부에서 사용할 수 있는 도구:** `web_search`, `web_extract`, `read_file`, `write_file`, `search_files`, `patch`, `terminal`(포그라운드 전용)

## 에이전트가 이를 사용하는 경우

에이전트는 다음과 같은 경우 `execute_code`를 사용합니다.

- **처리 로직이 호출 사이에 있는 도구 호출이 3개 이상인 경우**
- 대량 데이터 필터링 또는 조건부 분기
- 결과를 순회하는 루프

핵심 이점은 중간 도구 결과가 컨텍스트 창에 들어가지 않는다는 점입니다. 최종 `print()` 출력만 돌아오므로 토큰 사용량이 크게 줄어듭니다.

## 실제 예시

### 데이터 처리 파이프라인

```python
from hermes_tools import search_files, read_file
import json

# Find all config files and extract database settings
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

# Search, extract, and summarize in one turn
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

### 대량 파일 리팩터링

```python
from hermes_tools import search_files, read_file, patch

# Find all Python files using deprecated API and fix them
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

# Run tests, parse results, and report
result = terminal("cd /project && python -m pytest --tb=short -q 2>&1", timeout=120)
output = result.get("output", "")

# Parse test output
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

`execute_code`에는 `~/.hermes/config.yaml`의 `code_execution.mode`로 제어되는 두 가지 실행 모드가 있습니다.

| 모드 | 작업 디렉터리 | Python 인터프리터 |
|------|---------------|------------------|
| **`project`**(기본값) | 세션의 작업 디렉터리(`terminal()`과 동일) | 활성화된 `VIRTUAL_ENV` / `CONDA_PREFIX` Python, 없으면 Hermes 자체 Python |
| `strict` | 사용자의 프로젝트와 격리된 임시 스테이징 디렉터리 | `sys.executable`(Hermes 자체 Python) |

**`project`로 유지할 경우:** `import pandas`, `from my_project import foo` 또는 `open(".env")` 같은 상대 경로가 `terminal()`과 동일하게 작동해야 할 때 사용합니다. 대부분의 경우 이것이 원하는 방식입니다.

**`strict`로 전환할 경우:** 재현성을 극대화해야 할 때 사용합니다. 사용자가 어떤 가상 환경을 활성화했는지와 관계없이 매 세션 동일한 인터프리터를 사용하고, 스크립트를 프로젝트 트리와 격리해 상대 경로를 통한 프로젝트 파일의 우발적 접근 위험을 없애려는 경우입니다.

```yaml
# ~/.hermes/config.yaml
code_execution:
  mode: project   # or "strict"
```

`project` 모드의 대체 동작: `VIRTUAL_ENV` / `CONDA_PREFIX`가 설정되지 않았거나 손상되었거나 3.8보다 오래된 Python을 가리키면, 확인자는 `sys.executable`로 정상적으로 대체됩니다. 작동하는 인터프리터 없이 에이전트가 남는 일은 없습니다.

두 모드에서 보안상 중요한 불변 조건은 동일합니다.

- 환경 정리(API 키, 토큰, 자격 증명 제거)
- 도구 허용 목록(스크립트는 `execute_code`를 재귀적으로 호출하거나 `delegate_task` 또는 MCP 도구를 호출할 수 없음)
- 리소스 제한(시간 초과, stdout 상한, 도구 호출 상한)

모드를 전환하면 스크립트가 실행되는 위치와 사용하는 인터프리터만 바뀌며, 스크립트가 볼 수 있는 자격 증명이나 호출할 수 있는 도구는 바뀌지 않습니다.

## 리소스 제한

| 리소스 | 제한 | 참고 |
|----------|-------|-------|
| **시간 초과** | 5분(300초) | 스크립트가 SIGTERM으로 종료되고, 5초의 유예 후 SIGKILL로 종료됩니다. |
| **Stdout** | 50KB | `[output truncated at 50KB]` 알림과 함께 출력이 잘립니다. |
| **Stderr** | 10KB | 0이 아닌 종료 시 디버깅을 위해 출력에 포함됩니다. |
| **도구 호출** | 실행당 50회 | 한도에 도달하면 오류가 반환됩니다. |

모든 제한은 `config.yaml`을 통해 구성할 수 있습니다.

```yaml
# In ~/.hermes/config.yaml
code_execution:
  mode: project      # project (default) | strict
  timeout: 300       # Max seconds per script (default: 300)
  max_tool_calls: 50 # Max tool calls per execution (default: 50)
```

## 스크립트 내부에서 도구 호출이 작동하는 방식

스크립트가 `web_search("query")` 같은 함수를 호출하면 다음과 같이 처리됩니다.

1. 호출이 JSON으로 직렬화되어 Unix 도메인 소켓을 통해 부모 프로세스로 전송됩니다.
2. 부모 프로세스가 표준 `handle_function_call` 핸들러를 통해 디스패치합니다.
3. 결과가 소켓을 통해 다시 전송됩니다.
4. 함수가 파싱된 결과를 반환합니다.

따라서 스크립트 내부의 도구 호출은 일반 도구 호출과 동일하게 작동합니다. 동일한 속도 제한, 동일한 오류 처리, 동일한 기능을 사용합니다. 유일한 제한은 `terminal()`이 포그라운드 전용이라는 점입니다(`background` 또는 `pty` 매개변수는 사용할 수 없음).

## 오류 처리

스크립트가 실패하면 에이전트는 구조화된 오류 정보를 받습니다.

- **0이 아닌 종료 코드:** 에이전트가 전체 트레이스백을 볼 수 있도록 stderr가 출력에 포함됩니다.
- **시간 초과:** 스크립트가 종료되고 에이전트에는 `"Script timed out after 300s and was killed."`가 표시됩니다.
- **중단:** 실행 중 사용자가 새 메시지를 보내면 스크립트가 종료되고 에이전트에는 `[execution interrupted — user sent a new message]`가 표시됩니다.
- **도구 호출 한도:** 50회 호출 한도에 도달하면 이후 도구 호출에서 오류 메시지가 반환됩니다.

응답에는 항상 `status`(success/error/timeout/interrupted), `output`, `tool_calls_made`, `duration_seconds`가 포함됩니다.

## 보안

:::danger 보안 모델
자식 프로세스는 **최소 환경**에서 실행됩니다. API 키, 토큰, 자격 증명은 기본적으로 제거됩니다. 스크립트는 RPC 채널을 통해서만 도구에 접근하며, 명시적으로 허용하지 않는 한 환경 변수에서 비밀을 읽을 수 없습니다.
:::

이름에 `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `PASSWD` 또는 `AUTH`가 포함된 환경 변수는 제외됩니다. 안전한 시스템 변수(`PATH`, `HOME`, `LANG`, `SHELL`, `PYTHONPATH`, `VIRTUAL_ENV` 등)만 전달됩니다.

### 스킬 환경 변수 전달

스킬의 프런트매터에 `required_environment_variables`를 선언하면 해당 변수가 스킬을 로드한 후 `execute_code`와 `terminal` 자식 프로세스 모두에 **자동으로 전달**됩니다. 이를 통해 임의의 코드에 대한 보안 태세를 약화하지 않고도 스킬이 선언한 API 키를 사용할 수 있습니다.

스킬이 아닌 사용 사례에서는 `config.yaml`에서 변수를 명시적으로 허용 목록에 추가할 수 있습니다.

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

자세한 내용은 [보안 가이드](/user-guide/security#environment-variable-passthrough)를 참조하세요.

### 자식 프로세스의 `HERMES_*` 변수

자식 프로세스는 정확한 이름으로 지정된 소수의 운영용 `HERMES_*` 변수만 받습니다.

- `HERMES_HOME`
- `HERMES_PROFILE`
- `HERMES_CONFIG`
- `HERMES_ENV`

(그리고 RPC 채널이 작동하도록 Hermes가 명시적으로 주입하는 `HERMES_RPC_DIR` / `HERMES_RPC_SOCKET` / `TZ` / `HOME`도 받습니다.)

:::note 동작 변경
이전 버전은 이름이 `HERMES_`로 시작하는 **모든** 변수를 자식 프로세스로 전달했습니다. 보안 강화를 위해 이 광범위한 접두사 전달이 제거되었습니다. 이 방식은 비밀 문자열을 포함하지 않는 `HERMES_*` 이름의 설정(예: `HERMES_BASE_URL`, `HERMES_KANBAN_DB` 또는 `HERMES_*_WEBHOOK` 엔드포인트)을 임의의 샌드박스 코드로 유출할 수 있었습니다.

`execute_code` 스크립트 또는 가져오기 시점에 해당 변수를 사용하는 저장소/플러그인 모듈이 위의 네 가지 운영용 이름 이외의 `HERMES_*` 변수에 의존했다면, 이제 자식 프로세스에서는 해당 변수가 **설정되지 않은 상태**로 확인됩니다. 이는 의도적인 제거이며 버그가 아닙니다.
:::

**해결 방법 — 변수를 명시적으로 다시 허용합니다.** 두 경로 모두 해당 변수를 `execute_code` 및 `terminal` 자식 프로세스로 전달하며, 어느 경로도 비밀 제거 보장을 약화하지 않습니다(Hermes가 관리하는 제공자 자격 증명은 이 방법으로 다시 허용할 수 없습니다).

1. **컴퓨터별, `config.yaml`에서** — 정확한 변수 이름을 전달 허용 목록에 추가합니다.

   ```yaml
   terminal:
     env_passthrough:
       - HERMES_KANBAN_DB
       - HERMES_BASE_URL
   ```

2. **스킬별, 스킬의 프런트매터에서** — 해당 스킬이 로드될 때마다 자동으로 등록되도록 선언합니다.

   ```yaml
   required_environment_variables:
     - HERMES_KANBAN_DB
   ```

**진단 방법.** 자식 프로세스가 허용 목록에 없는 `HERMES_*` 변수 하나 이상을 제거하면 Hermes는 해당 변수의 이름과 `env_passthrough` 우회 경로를 가리키는 한 줄짜리 `debug` 로그를 기록합니다. 디버그 로깅(`hermes logs --level DEBUG`)으로 실행하거나 `~/.hermes/logs/agent.log`를 확인하고, 스크립트가 `HERMES_*` 변수 누락처럼 동작할 때 `execute_code: dropped N non-allowlisted HERMES_* var(s)`를 찾으세요.

Hermes는 항상 스크립트와 자동 생성된 `hermes_tools.py` RPC 스텁을 실행 후 정리되는 임시 스테이징 디렉터리에 기록합니다. `strict` 모드에서는 스크립트도 그곳에서 실행되고, `project` 모드에서는 세션의 작업 디렉터리에서 실행됩니다(가져오기가 계속 해결되도록 스테이징 디렉터리는 `PYTHONPATH`에 남습니다). 자식 프로세스는 자체 프로세스 그룹에서 실행되므로 시간 초과나 중단 시 깔끔하게 종료할 수 있습니다.

## execute_code와 terminal 비교

| 사용 사례 | execute_code | terminal |
|----------|-------------|----------|
| 호출 사이에 도구 호출이 있는 다단계 워크플로 | ✅ | ❌ |
| 간단한 셸 명령 | ❌ | ✅ |
| 대량 도구 출력 필터링/처리 | ✅ | ❌ |
| 빌드 또는 테스트 모음 실행 | ❌ | ✅ |
| 검색 결과 순회 | ✅ | ❌ |
| 대화형/백그라운드 프로세스 | ❌ | ✅ |
| 환경 변수에 API 키 필요 | ⚠️ [passthrough](/user-guide/security#environment-variable-passthrough)를 통해서만 가능 | ✅(대부분 전달됨) |

**경험칙:** 로직을 삽입해 Hermes 도구를 프로그래밍 방식으로 호출해야 할 때는 `execute_code`를 사용하세요. 셸 명령, 빌드 및 프로세스를 실행할 때는 `terminal`을 사용하세요.

## 플랫폼 지원

코드 실행은 **Linux, macOS 및 Windows**에서 사용할 수 있습니다. Linux와 macOS에서는 RPC 채널이 Unix 도메인 소켓을 사용하고, `AF_UNIX`가 불안정한 Windows에서는 Hermes가 샌드박스 RPC 전송을 위해 자동으로 루프백 TCP 소켓으로 대체합니다. 원격 터미널 백엔드(Docker/SSH/Modal 등)는 대신 파일 기반 RPC 전송을 사용하며, 백엔드 내부에 Python 3도 필요합니다.
