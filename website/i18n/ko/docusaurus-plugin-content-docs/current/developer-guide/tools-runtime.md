---
sidebar_position: 9
title: "도구 런타임 (Tools Runtime)"
description: "도구 레지스트리, 도구 모음, 디스패치 및 터미널 환경의 런타임 동작"
---

# 도구 런타임 (Tools Runtime)

Hermes 도구들은 중앙 레지스트리/디스패치 시스템을 통해 실행되고 도구 모음(toolsets)으로 그룹화되는 자가 등록(self-registering) 함수들입니다.

주요 파일:

- `tools/registry.py`
- `model_tools.py`
- `toolsets.py`
- `tools/terminal_tool.py`
- `tools/environments/*`

## 도구 등록 모델 (Tool registration model)

각 도구 모듈은 임포트 시점에 `registry.register(...)`를 호출합니다.

`model_tools.py`는 도구 모듈을 가져오고(import) 발견하며 모델에서 사용하는 스키마 목록을 구성할 책임이 있습니다.

### `registry.register()` 작동 방식

`tools/`에 있는 모든 도구 파일은 모듈 수준에서 `registry.register()`를 호출하여 자신을 선언합니다. 함수 서명은 다음과 같습니다:

```python
registry.register(
    name="terminal",               # 고유한 도구 이름 (API 스키마에서 사용됨)
    toolset="terminal",            # 이 도구가 속한 도구 모음
    schema={...},                  # OpenAI 함수 호출 스키마 (설명, 매개변수)
    handler=handle_terminal,       # 도구가 호출될 때 실행되는 함수
    check_fn=check_terminal,       # 선택 사항: 가용성에 대해 True/False를 반환함
    requires_env=["SOME_VAR"],     # 선택 사항: 필요한 환경 변수 (UI 표시용)
    is_async=False,                # 핸들러가 비동기(async) 코루틴인지 여부
    description="Run commands",    # 사람이 읽을 수 있는 설명
    emoji="💻",                    # 스피너/진행 상황 표시를 위한 이모지
)
```

각 호출은 도구 이름을 키로 사용하여 싱글톤 `ToolRegistry._tools` 딕셔너리에 저장되는 `ToolEntry`를 생성합니다. 여러 도구 모음에서 이름 충돌이 발생하면 경고가 기록되고 나중에 등록된 것이 우선합니다.

### 검색: `discover_builtin_tools()`

`model_tools.py`를 가져올 때 `tools/registry.py`에서 `discover_builtin_tools()`를 호출합니다. 이 함수는 AST 파싱을 사용하여 모든 `tools/*.py` 파일을 스캔하여 최상위 수준의 `registry.register()` 호출을 포함하는 모듈을 찾은 다음 이들을 가져옵니다(import):

```python
# tools/registry.py (단순화됨)
def discover_builtin_tools(tools_dir=None):
    tools_path = Path(tools_dir) if tools_dir else Path(__file__).parent
    for path in sorted(tools_path.glob("*.py")):
        if path.name in {"__init__.py", "registry.py", "mcp_tool.py"}:
            continue
        if _module_registers_tools(path):  # 최상위 registry.register()에 대한 AST 확인
            importlib.import_module(f"tools.{path.stem}")
```

이러한 자동 검색(auto-discovery)을 통해 새 도구 파일이 자동으로 인식되므로 유지 관리해야 할 수동 목록이 없습니다. AST 확인은 최상위 `registry.register()` 호출만 일치시키고(함수 내부의 호출은 제외), 따라서 `tools/` 내부의 도우미(helper) 모듈은 가져오지 않습니다.

가져오기 할 때마다 각 모듈의 `registry.register()` 호출이 트리거됩니다. 선택적 도구에서의 오류(예: 이미지 생성용 `fal_client` 누락)는 잡혀서(caught) 기록되며 — 다른 도구가 로드되는 것을 막지 않습니다.

핵심 도구를 검색한 후 MCP 도구와 플러그인 도구도 검색됩니다:

1. **MCP 도구** — `tools.mcp_tool.discover_mcp_tools()`가 MCP 서버 설정을 읽고 외부 서버로부터 도구를 등록합니다.
2. **플러그인 도구** — `hermes_cli.plugins.discover_plugins()`가 추가 도구를 등록할 수 있는 사용자/프로젝트/pip 플러그인을 로드합니다.

## 도구 가용성 검사 (`check_fn`)

각 도구는 선택적으로 `check_fn`을 제공할 수 있습니다 — 도구를 사용할 수 있을 때 `True`를 반환하고 사용할 수 없을 때 `False`를 반환하는 콜러블(callable) 함수입니다. 일반적인 검사 내용은 다음과 같습니다:

- **API 키 존재 여부** — 예: 웹 검색의 경우 `lambda: bool(os.environ.get("SERP_API_KEY"))`
- **실행 중인 서비스** — 예: Honcho 서버가 구성되었는지 확인
- **설치된 바이너리** — 예: 브라우저 도구에서 `playwright`를 사용할 수 있는지 확인

`registry.get_definitions()`가 모델을 위한 스키마 목록을 빌드할 때, 각 도구의 `check_fn()`을 실행합니다:

```python
# registry.py 에서 단순화됨
if entry.check_fn:
    try:
        available = bool(entry.check_fn())
    except Exception:
        available = False   # 예외 발생 시 = 사용할 수 없음
    if not available:
        continue            # 이 도구는 완전히 건너뜀
```

주요 동작:
- 검사 결과는 **호출당 캐시됩니다** — 여러 도구가 동일한 `check_fn`을 공유하는 경우 한 번만 실행됩니다.
- `check_fn()`의 예외는 "사용 불가"로 취급됩니다 (fail-safe).
- `is_toolset_available()` 메서드는 도구 모음의 `check_fn`이 통과하는지 확인하여 UI 표시 및 도구 모음 확인(resolution)에 사용됩니다.

## 도구 모음 확인 (Toolset resolution)

도구 모음(toolsets)은 도구들의 이름 있는 묶음(bundle)입니다. Hermes는 다음을 통해 이를 확인합니다:

- 명시적으로 활성화/비활성화된 도구 모음 목록
- 플랫폼 프리셋 (예: `hermes-cli`, `hermes-telegram` 등)
- 동적 MCP 도구 모음
- `hermes-acp`와 같이 엄선된 특수 목적용 세트

### `get_tool_definitions()`가 도구를 필터링하는 방법

주요 진입점은 `model_tools.get_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode)`입니다:

1. **`enabled_toolsets`가 제공된 경우** — 해당 도구 모음에 속한 도구만 포함됩니다. 각 도구 모음 이름은 복합 도구 모음을 개별 도구 이름으로 확장하는 `resolve_toolset()`을 통해 해석됩니다.

2. **`disabled_toolsets`가 제공된 경우** — 모든 도구 모음에서 시작하여 비활성화된 항목을 뺍니다.

3. **둘 다 아닌 경우** — 알려진 모든 도구 모음을 포함합니다.

4. **레지스트리 필터링** — 결정된 도구 이름 집합이 `registry.get_definitions()`로 전달되며, 여기서 `check_fn` 필터링을 적용하고 OpenAI 형식의 스키마를 반환합니다.

5. **동적 스키마 패치** — 필터링 이후에, `execute_code`와 `browser_navigate` 스키마는 실제로 필터링을 통과한 도구만 참조하도록 동적으로 조정됩니다 (모델이 사용할 수 없는 도구를 환각(hallucination)하는 것을 방지).

### 레거시 도구 모음 이름

이전 버전과의 호환성을 위해 `_tools` 접미사가 있는 이전 도구 모음 이름(예: `web_tools`, `terminal_tools`)은 `_LEGACY_TOOLSET_MAP`을 통해 최신 도구 이름에 매핑됩니다.

## 디스패치 (Dispatch)

런타임 시에 도구들은 중앙 레지스트리를 통해 디스패치(dispatch)되며, 메모리/할 일(todo)/세션 검색 처리와 같은 일부 에이전트 레벨 도구에 대해서는 에이전트 루프의 예외 처리가 있습니다.

### 디스패치 흐름: 모델의 tool_call → 핸들러 실행

모델이 `tool_call`을 반환할 때의 흐름은 다음과 같습니다:

```
tool_call을 포함한 모델 응답
    ↓
run_agent.py 에이전트 루프
    ↓
model_tools.handle_function_call(name, args, task_id, user_task)
    ↓
[에이전트 루프 도구인가?] → 에이전트 루프에서 직접 처리됨 (todo, memory, session_search, delegate_task)
    ↓
[플러그인 사전-훅(pre-hook)] → invoke_hook("pre_tool_call", ...)
    ↓
registry.dispatch(name, args, **kwargs)
    ↓
이름으로 ToolEntry 찾기
    ↓
[비동기(Async) 핸들러?] → _run_async()를 통해 연결됨
[동기(Sync) 핸들러?]    → 직접 호출됨
    ↓
결과 문자열 반환 (또는 JSON 오류 반환)
    ↓
[플러그인 사후-훅(post-hook)] → invoke_hook("post_tool_call", ...)
```

### 오류 래핑 (Error wrapping)

모든 도구 실행은 두 단계의 오류 처리에 래핑되어 있습니다:

1. **`registry.dispatch()`** — 핸들러에서 발생하는 모든 예외를 잡아서 JSON 형태인 `{"error": "Tool execution failed: ExceptionType: message"}`로 반환합니다.

2. **`handle_function_call()`** — 전체 디스패치를 2차 try/except로 래핑하여 `{"error": "Error executing tool_name: message"}`를 반환합니다.

이를 통해 모델은 항상 잘 구성된 JSON 문자열을 수신하며, 처리되지 않은 예외를 받지 않게 됩니다.

### 에이전트 루프 도구 (Agent-loop tools)

4개의 도구는 에이전트 수준 상태(TodoStore, MemoryStore 등)가 필요하므로 레지스트리 디스패치 전에 인터셉트(intercept)됩니다:

- `todo` — 계획(planning) 및 태스크 추적
- `memory` — 영구 메모리 쓰기
- `session_search` — 세션 간 정보 회상
- `delegate_task` — 하위 에이전트 세션 생성

이러한 도구들의 스키마도 레지스트리에 여전히 등록되어 있지만( `get_tool_definitions` 사용을 위해), 디스패치가 어떻게든 이 도구들에 직접 도달할 경우 핸들러가 스텁(stub) 오류를 반환합니다.

### 비동기 연결 (Async bridging)

도구 핸들러가 비동기일 경우, `_run_async()`는 이를 동기 디스패치 경로와 연결합니다:

- **CLI 경로 (실행 중인 루프 없음)** — 캐시된 비동기 클라이언트를 유지하기 위해 영구 이벤트 루프를 사용합니다.
- **게이트웨이 경로 (실행 중인 루프 있음)** — `asyncio.run()`을 사용하여 일회성 스레드를 시작합니다.
- **워커 스레드 (병렬 도구)** — 스레드 로컬 스토리지에 저장된 스레드별 영구 루프를 사용합니다.

## DANGEROUS_PATTERNS 승인 흐름

터미널 도구는 `tools/approval.py`에 정의된 위험한 명령어 승인 시스템을 통합합니다:

1. **패턴 감지** — `DANGEROUS_PATTERNS`는 다음과 같은 파괴적인 작업을 커버하는 `(regex, description)` 튜플의 목록입니다:
   - 재귀적 삭제 (`rm -rf`)
   - 파일 시스템 포맷 (`mkfs`, `dd`)
   - SQL 파괴적 연산 (`WHERE`가 없는 `DROP TABLE`, `DELETE FROM`)
   - 시스템 설정 덮어쓰기 (`> /etc/`)
   - 서비스 조작 (`systemctl stop`)
   - 원격 코드 실행 (`curl | sh`)
   - 포크 폭탄(Fork bombs), 프로세스 강제 종료 등

2. **감지** — 모든 터미널 명령어를 실행하기 전에 `detect_dangerous_command(command)`가 모든 패턴에 대해 검사합니다.

3. **승인 프롬프트** — 일치하는 패턴이 발견된 경우:
   - **CLI 모드** — 인터랙티브 프롬프트가 사용자에게 승인, 거부 또는 영구 허용을 요청합니다.
   - **게이트웨이 모드** — 비동기 승인 콜백이 메시징 플랫폼으로 요청을 보냅니다.
   - **스마트 승인** — 선택적으로, 보조 LLM이 패턴과 일치하는 저위험 명령어(예: `rm -rf node_modules/`는 안전하지만 "재귀적 삭제" 패턴과 일치함)를 자동으로 승인할 수 있습니다.

4. **세션 상태** — 승인은 세션별로 추적됩니다. 한 세션에서 "재귀적 삭제"를 승인하면 후속 `rm -rf` 명령어는 다시 프롬프트를 띄우지 않습니다.

5. **영구 허용 목록** — "영구 허용(allow permanently)" 옵션은 `config.yaml`의 `command_allowlist`에 패턴을 기록하여 세션을 거쳐도 유지되게 합니다.

## 터미널/런타임 환경 (Terminal/runtime environments)

터미널 시스템은 여러 백엔드를 지원합니다:

- local
- docker
- ssh
- singularity
- modal
- daytona

또한 다음 기능들도 지원합니다:

- 태스크별 작업 디렉토리(cwd) 재정의
- 백그라운드 프로세스 관리
- PTY 모드
- 위험한 명령어에 대한 승인 콜백

## 동시성 (Concurrency)

도구 호출은 도구의 구성 및 상호 작용 요구 사항에 따라 순차적으로 또는 동시에(concurrently) 실행될 수 있습니다.

## 관련 문서 (Related docs)

- [도구 모음 참조 (Toolsets Reference)](../reference/toolsets-reference.md)
- [내장 도구 참조 (Built-in Tools Reference)](../reference/tools-reference.md)
- [에이전트 루프 내부](./agent-loop.md)
- [ACP 내부](./acp-internals.md)
