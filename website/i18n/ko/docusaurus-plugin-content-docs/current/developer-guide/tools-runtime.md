---
sidebar_position: 9
title: "도구 런타임"
description: "도구 레지스트리, 도구 세트, 디스패치 및 터미널 환경의 런타임 동작"
---

# 도구 런타임

Hermes 도구는 도구 세트로 그룹화되고 중앙 레지스트리/디스패치 시스템을 통해 실행되는 자체 등록 함수입니다.

주요 파일:

- `tools/registry.py`
- `model_tools.py`
- `toolsets.py`
- `tools/terminal_tool.py`
- `tools/environments/*`

## 도구 등록 모델

각 도구 모듈은 가져올 때 `registry.register(...)`를 호출합니다.

`model_tools.py`는 도구 모듈을 가져오고 검색하며 모델이 사용하는 스키마 목록을 빌드합니다.

### `registry.register()` 작동 방식

`tools/`의 모든 도구 파일은 모듈 수준에서 `registry.register()`를 호출하여 자신을 선언합니다. 함수 시그니처는 다음과 같습니다.

```python
registry.register(
    name="terminal",               # Unique tool name (used in API schemas)
    toolset="terminal",            # Toolset this tool belongs to
    schema={...},                  # Model-facing schema (description, parameters)
    handler=handle_terminal,       # The function that executes when the tool is called
    check_fn=check_terminal,       # Optional: returns True/False for availability
    requires_env=["SOME_VAR"],     # Optional: env vars needed (for UI display)
    is_async=False,                # Whether the handler is an async coroutine
    description="Run commands",    # Optional ToolEntry registry metadata
    emoji="💻",                    # Emoji for spinner/progress display
)
```

각 호출은 도구 이름을 키로 하는 싱글턴 `ToolRegistry._tools` 딕셔너리에 저장된 `ToolEntry`를 생성합니다. **서로 다른** 도구 세트에 속한 기존 도구를 가리는 등록은 `override=True`를 전달하지 않는 한 거부됩니다(오류 로그가 기록됨). 기본 제공 도구를 플러그인이 재정의하려면 `config.yaml`에서 운영자가 `plugins.entries.<plugin_id>.allow_tool_override: true`를 선택해야 합니다.

`schema["description"]`은 모델에 표시되는 설명의 기준입니다. 별도의 `description=` 인수는 `ToolEntry.description`을 채우며, 생략되면 레지스트리 메타데이터가 스키마 설명으로 대체됩니다. `get_definitions()`는 `entry.schema`에서 OpenAI 함수 정의를 빌드하고 `entry.description`을 설명이 없는 스키마에 복사하지 않습니다. 따라서 `description=`만으로는 모델에 도구를 설명할 수 없으며, 두 값이 다르면 모델에는 스키마 값이 표시됩니다. 소비자가 의도적으로 다른 메타데이터를 필요로 하는 경우가 아니라면 설명은 스키마에 한 번만 정의하는 것이 좋습니다.

### 검색: `discover_builtin_tools()`

`model_tools.py`를 가져올 때 `tools/registry.py`의 `discover_builtin_tools()`를 호출합니다. 이 함수는 AST 파싱을 사용하여 모든 `tools/*.py` 파일을 검사하고 최상위 수준의 `registry.register()` 호출을 포함하는 모듈을 찾은 다음 가져옵니다.

```python
# tools/registry.py (simplified)
def discover_builtin_tools(tools_dir=None):
    tools_path = Path(tools_dir) if tools_dir else Path(__file__).parent
    for path in sorted(tools_path.glob("*.py")):
        if path.name in {"__init__.py", "registry.py", "mcp_tool.py"}:
            continue
        if _module_registers_tools(path):  # AST check for top-level registry.register()
            importlib.import_module(f"tools.{path.stem}")
```

이 자동 검색 덕분에 새 도구 파일은 자동으로 선택되며 수동으로 관리할 목록이 필요하지 않습니다. AST 검사는 최상위 수준의 `registry.register()` 호출만 일치시키므로 `tools/`의 도우미 모듈은 가져오지 않습니다.

각 가져오기는 해당 모듈의 `registry.register()` 호출을 실행합니다. 선택적 도구(예: 이미지 생성에 필요한 `fal_client`가 없는 경우)의 오류는 포착되어 로그에 기록되며, 다른 도구의 로드를 막지 않습니다.

핵심 도구 검색 후 MCP 도구와 플러그인 도구도 검색됩니다.

1. **MCP 도구** — `tools.mcp_tool.discover_mcp_tools()`가 MCP 서버 설정을 읽고 외부 서버의 도구를 등록합니다.
2. **플러그인 도구** — `hermes_cli.plugins.discover_plugins()`가 사용자/프로젝트/pip 플러그인을 로드하며 추가 도구를 등록할 수 있습니다.

## 도구 사용 가능 여부 확인(`check_fn`)

각 도구는 선택적으로 `check_fn`을 제공할 수 있습니다. 이는 도구를 사용할 수 있을 때 `True`, 그렇지 않을 때 `False`를 반환하는 호출 가능 객체입니다. 일반적인 확인 항목은 다음과 같습니다.

- **API 키 존재 여부** — 예: 웹 검색을 위한 `lambda: bool(os.environ.get("SERP_API_KEY"))`
- **서비스 실행 여부** — 예: Honcho 서버가 구성되었는지 확인
- **바이너리 설치 여부** — 예: 브라우저 도구에 `playwright`를 사용할 수 있는지 확인

`registry.get_definitions()`가 모델용 스키마 목록을 빌드할 때 각 도구의 `check_fn()`을 실행합니다.

```python
# Simplified from registry.py
if entry.check_fn:
    try:
        available = bool(entry.check_fn())
    except Exception:
        available = False   # Exceptions = unavailable
    if not available:
        continue            # Skip this tool entirely
```

주요 동작:
- 확인 결과는 **호출별로 캐시**됩니다. 여러 도구가 동일한 `check_fn`을 공유하면 한 번만 실행됩니다.
- `check_fn()`의 예외는 "사용할 수 없음"으로 처리됩니다(안전 우선).
- `is_toolset_available()` 메서드는 도구 세트의 `check_fn`이 통과하는지 확인하며, UI 표시와 도구 세트 확인에 사용됩니다.

## 도구 세트 확인

도구 세트는 도구의 이름이 지정된 묶음입니다. Hermes는 다음을 통해 도구 세트를 확인합니다.

- 명시적으로 활성화/비활성화된 도구 세트 목록
- 플랫폼 프리셋(`hermes-cli`, `hermes-telegram` 등)
- 동적 MCP 도구 세트
- `hermes-acp`와 같은 선별된 특수 목적 세트

### `get_tool_definitions()`가 도구를 필터링하는 방식

주요 진입점은 `model_tools.get_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode)`입니다.

1. **`enabled_toolsets`가 제공된 경우** — 해당 도구 세트의 도구만 포함됩니다. 각 도구 세트 이름은 `resolve_toolset()`을 통해 확인되며, 복합 도구 세트를 개별 도구 이름으로 확장합니다.

2. **`disabled_toolsets`가 제공된 경우** — 모든 도구 세트에서 시작한 다음 비활성화된 세트를 제외합니다.

3. **둘 다 없는 경우** — 알려진 모든 도구 세트를 포함합니다.

4. **레지스트리 필터링** — 확인된 도구 이름 집합이 `registry.get_definitions()`에 전달되고, 여기서 `check_fn` 필터링을 적용하여 OpenAI 형식의 스키마를 반환합니다.

5. **동적 스키마 패치** — 필터링 후 `execute_code` 및 `browser_navigate` 스키마가 동적으로 조정되어 실제로 필터링을 통과한 도구만 참조합니다(모델이 사용할 수 없는 도구를 환각하는 것을 방지).

### 레거시 도구 세트 이름

이전 도구 세트 이름의 `_tools` 접미사(예: `web_tools`, `terminal_tools`)는 하위 호환성을 위해 `_LEGACY_TOOLSET_MAP`을 통해 최신 도구 이름으로 매핑됩니다.

## 디스패치

런타임에서 도구는 중앙 레지스트리를 통해 디스패치되며, 메모리/할 일/세션 검색 처리와 같은 일부 에이전트 수준 도구에는 에이전트 루프 예외가 적용됩니다.

### 디스패치 흐름: 모델 `tool_call` → 핸들러 실행

모델이 `tool_call`을 반환하면 흐름은 다음과 같습니다.

```
Model response with tool_call
    ↓
run_agent.py agent loop
    ↓
model_tools.handle_function_call(name, args, task_id, user_task)
    ↓
[Agent-loop tools?] → handled directly by agent loop (todo, memory, session_search, delegate_task)
    ↓
[Plugin pre-hook] → invoke_hook("pre_tool_call", ...)
    ↓
registry.dispatch(name, args, **kwargs)
    ↓
Look up ToolEntry by name
    ↓
[Async handler?] → bridge via _run_async()
[Sync handler?]  → call directly
    ↓
Return result string (or JSON error)
    ↓
[Plugin post-hook] → invoke_hook("post_tool_call", ...)
```

### 오류 래핑

모든 도구 실행은 두 수준의 오류 처리로 래핑됩니다.

1. **`registry.dispatch()`** — 핸들러의 모든 예외를 포착하고 `{"error": "Tool execution failed: ExceptionType: message"}`를 JSON으로 반환합니다.

2. **`handle_function_call()`** — 전체 디스패치를 보조 `try/except`로 래핑하고 `{"error": "Error executing tool_name: message"}`를 반환합니다.

이를 통해 모델은 처리되지 않은 예외가 아닌 올바른 형식의 JSON 문자열을 항상 받습니다.

### 에이전트 루프 도구

네 도구는 에이전트 수준의 상태(`TodoStore`, `MemoryStore` 등)가 필요하므로 레지스트리 디스패치 전에 가로챕니다.

- `todo` — 계획/작업 추적
- `memory` — 영구 메모리 쓰기
- `session_search` — 세션 간 검색
- `delegate_task` — 하위 에이전트 세션 생성

이 도구들의 스키마는 여전히 레지스트리에 등록되지만(`get_tool_definitions`용), 디스패치가 어떤 이유로든 해당 도구에 도달하면 핸들러는 스텁 오류를 반환합니다.

### 비동기 브리징

도구 핸들러가 비동기인 경우 `_run_async()`가 동기 디스패치 경로와 연결합니다.

- **CLI 경로(실행 중인 루프 없음)** — 캐시된 비동기 클라이언트를 유지하기 위해 영구 이벤트 루프를 사용
- **게이트웨이 경로(실행 중인 루프 있음)** — `asyncio.run()`을 사용하는 일회성 스레드를 생성
- **작업자 스레드(병렬 도구)** — 스레드 로컬 저장소에 저장된 스레드별 영구 루프를 사용

## `DANGEROUS_PATTERNS` 승인 흐름

터미널 도구는 `tools/approval.py`에 정의된 위험 명령 승인 시스템과 통합됩니다.

1. **패턴 감지** — `DANGEROUS_PATTERNS`는 파괴적인 작업을 다루는 `(regex, description)` 튜플 목록입니다.
   - 재귀 삭제(`rm -rf`)
   - 파일 시스템 포맷(`mkfs`, `dd`)
   - SQL 파괴적 작업(`DROP TABLE`, `WHERE` 없는 `DELETE FROM`)
   - 시스템 설정 덮어쓰기(`> /etc/`)
   - 서비스 조작(`systemctl stop`)
   - 원격 코드 실행(`curl | sh`)
   - 포크 폭탄, 프로세스 종료 등

2. **감지** — 터미널 명령을 실행하기 전에 `detect_dangerous_command(command)`가 모든 패턴과 대조합니다.

3. **승인 요청** — 일치 항목이 발견되면:
   - **CLI 모드** — 대화형 프롬프트에서 사용자에게 승인, 거부 또는 영구 허용을 요청
   - **게이트웨이 모드** — 비동기 승인 콜백이 메시징 플랫폼에 요청을 보냄
   - **스마트 승인** — 선택적으로 보조 LLM이 패턴과 일치하지만 위험도가 낮은 명령(예: `rm -rf node_modules/`는 안전하지만 "재귀 삭제"와 일치)을 자동 승인

4. **세션 상태** — 승인은 세션별로 추적됩니다. 세션에서 "재귀 삭제"를 승인하면 이후의 `rm -rf` 명령에는 다시 묻지 않습니다.

5. **영구 허용 목록** — "영구 허용" 옵션은 패턴을 `config.yaml`의 `command_allowlist`에 기록하여 세션 간에 유지합니다.

## 터미널/런타임 환경

터미널 시스템은 여러 백엔드를 지원합니다.

- local
- docker
- ssh
- singularity
- modal
- daytona
- vercel_sandbox

또한 다음을 지원합니다.

- 작업별 cwd 재정의
- 백그라운드 프로세스 관리
- PTY 모드
- 위험 명령의 승인 콜백

## 동시성

도구 호출은 도구 조합과 상호작용 요구 사항에 따라 순차적으로 또는 동시에 실행될 수 있습니다.

## 관련 문서

- [도구 세트 참조](../reference/toolsets-reference.md)
- [기본 제공 도구 참조](../reference/tools-reference.md)
- [에이전트 루프 내부](./agent-loop.md)
- [ACP 내부](./acp-internals.md)
