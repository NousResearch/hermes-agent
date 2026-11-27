---
sidebar_position: 11
sidebar_label: "플러그인"
title: "플러그인"
description: "플러그인 시스템으로 사용자 지정 도구, 훅, 통합 기능을 추가하여 Hermes 확장하기"
---

# 플러그인

Hermes에는 코어 코드를 수정하지 않고 사용자 지정 도구, 훅, 통합 기능을 추가할 수 있는 플러그인 시스템이 있습니다.

자신이나 팀, 또는 특정 프로젝트를 위한 사용자 지정 도구를 만들고 싶다면 일반적으로 이 방법이 가장 적합합니다. 개발자 가이드의 [도구 추가](/developer-guide/adding-tools) 페이지는 `tools/` 및 `toolsets.py`에 있는 Hermes 내장 코어 도구를 위한 문서입니다.

**→ [Hermes 플러그인 빌드](/developer-guide/plugins)** — 완전히 작동하는 예제와 함께 단계별로 안내합니다.

## 빠른 개요

`plugin.yaml`과 Python 코드를 포함한 디렉터리를 `~/.hermes/plugins/`에 넣습니다.

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml      # manifest
├── __init__.py      # register() — wires schemas to handlers
├── schemas.py       # tool schemas (what the LLM sees)
└── tools.py         # tool handlers (what runs when called)
```

Hermes를 시작하면 도구가 내장 도구와 함께 나타납니다. 모델은 즉시 도구를 호출할 수 있습니다.

### 최소 작동 예제

다음은 `hello_world` 도구를 추가하고 훅을 통해 모든 도구 호출을 기록하는 완전한 플러그인입니다.

**`~/.hermes/plugins/hello-world/plugin.yaml`**

```yaml
name: hello-world
version: "1.0"
description: A minimal example plugin
```

**`~/.hermes/plugins/hello-world/__init__.py`**

```python
"""Minimal Hermes plugin — registers a tool and a hook."""

import json


def register(ctx):
    # --- Tool: hello_world ---
    schema = {
        "name": "hello_world",
        "description": "Returns a friendly greeting for the given name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name to greet",
                }
            },
            "required": ["name"],
        },
    }

    def handle_hello(params, **kwargs):
        del kwargs
        name = params.get("name", "World")
        return json.dumps({"success": True, "greeting": f"Hello, {name}!"})

    ctx.register_tool(
        name="hello_world",
        toolset="hello_world",
        schema=schema,
        handler=handle_hello,
    )

    # --- Hook: log every tool call ---
    def on_tool_call(tool_name, params, result):
        print(f"[hello-world] tool called: {tool_name}")

    ctx.register_hook("post_tool_call", on_tool_call)
```

두 파일을 모두 `~/.hermes/plugins/hello-world/`에 넣고 Hermes를 다시 시작하면 모델이 즉시 `hello_world`를 호출할 수 있습니다. 훅은 모든 도구 호출 후에 로그 한 줄을 출력합니다.

모델에 표시되는 도구 설명은 `schema["description"]`에 작성합니다. 선택 사항인 `ctx.register_tool(description=...)` 값은 별도의 `ToolEntry` 레지스트리 메타데이터입니다. 이 값을 생략하면 스키마 설명이 기본값이 되지만, Hermes는 `description`이 없는 스키마에 설명을 다시 복사하지 않습니다. 텍스트는 스키마에 한 번만 정의하는 것을 권장합니다. 두 값을 모두 제공한다면 서로 동기화된 상태로 유지하세요. 모델에는 스키마 값이 표시됩니다.

`./.hermes/plugins/` 아래의 프로젝트 로컬 플러그인은 기본적으로 비활성화되어 있습니다. 신뢰하는 저장소에서만 Hermes를 시작하기 전에 `HERMES_ENABLE_PROJECT_PLUGINS=true`를 설정하여 활성화하세요.

## 플러그인으로 할 수 있는 일

아래의 모든 `ctx.*` API는 플러그인의 `register(ctx)` 함수 안에서 사용할 수 있습니다.

| 기능 | 방법 |
|-----------|-----|
| 도구 추가 | `ctx.register_tool(name=..., toolset=..., schema=..., handler=...)` |
| 훅 추가 | `ctx.register_hook("post_tool_call", callback)` |
| 슬래시 명령 추가 | `ctx.register_command(name, handler, description)` — CLI 및 게이트웨이 세션에 `/name` 추가 |
| 명령에서 도구 디스패치 | `ctx.dispatch_tool(name, args)` — 부모 에이전트 컨텍스트가 자동으로 연결된 등록 도구 호출 |
| CLI 명령 추가 | `ctx.register_cli_command(name, help, setup_fn, handler_fn)` — `hermes <plugin> <subcommand>` 추가 |
| 메시지 주입 | `ctx.inject_message(content, role="user", session_key=...)` - [메시지 주입](#injecting-messages) 참고 |
| 데이터 파일 제공 | `Path(__file__).parent / "data" / "file.yaml"` |
| 스킬 번들링 | `ctx.register_skill(name, path)` — `plugin:skill`로 네임스페이스가 지정되고 `skill_view("plugin:skill")`을 통해 로드 |
| 환경 변수에 따른 게이트 | 플러그인.yaml의 `requires_env: [API_KEY]` — `hermes plugins install` 중 입력 요청 |
| pip으로 배포 | `[project.entry-points."hermes_agent.plugins"]` |
| 게이트웨이 플랫폼 등록 (Discord, Telegram, IRC, …) | `ctx.register_platform(name, label, adapter_factory, check_fn, ...)` — [플랫폼 어댑터 추가](/developer-guide/adding-platform-adapters) 참고 |
| 이미지 생성 백엔드 등록 | `ctx.register_image_gen_provider(provider)` — [이미지 생성 제공자 플러그인](/developer-guide/image-gen-provider-plugin) 참고 |
| 동영상 생성 백엔드 등록 | `ctx.register_video_gen_provider(provider)` — [동영상 생성 제공자 플러그인](/developer-guide/video-gen-provider-plugin) 참고 |
| 컨텍스트 압축 엔진 등록 | `ctx.register_context_engine(engine)` — [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin) 참고 |
| 사용자 승인 프롬프트 라우팅 | `ctx.register_approval_transport(name, present_fn)` — [승인 전송 방식](#approval-transports) 참고 |
| 메모리 백엔드 등록 | `plugins/memory/<name>/__init__.py`에서 `MemoryProvider`를 상속 — [메모리 제공자 플러그인](/developer-guide/memory-provider-plugin) 참고 (별도의 검색 시스템 사용) |
| 호스트가 소유한 LLM 호출 실행 | `ctx.llm.complete(...)` / `ctx.llm.complete_structured(...)` — 선택적 JSON 스키마 검증과 함께 사용자의 활성 모델 및 인증을 빌려 일회성 완성을 실행합니다. [플러그인 LLM 액세스](/developer-guide/plugin-llm-access) 참고 |
| MCP 도구 호출 (기능 게이트 적용) | `ctx.call_mcp(server, tool, arguments, timeout=30)` — [플러그인에서 MCP 서버 호출](#calling-mcp-servers-from-plugins) 참고 |
| 추론 백엔드(LLM 제공자) 등록 | `plugins/model-providers/<name>/__init__.py`에서 `register_provider(ProviderProfile(...))` — [모델 제공자 플러그인](/developer-guide/model-provider-plugin) 참고 (별도의 검색 시스템 사용) |

## 플러그인 검색

| 출처 | 경로 | 사용 사례 |
|--------|------|----------|
| 번들 | `<repo>/plugins/` | Hermes와 함께 제공 — [내장 플러그인](/user-guide/features/built-in-plugins) 참고 |
| 사용자 | `~/.hermes/plugins/` | 개인 플러그인 |
| 프로젝트 | `.hermes/plugins/` | 프로젝트별 플러그인 (`HERMES_ENABLE_PROJECT_PLUGINS=true` 필요) |
| pip | `hermes_agent.plugins` entry_points | 배포된 패키지 |
| Nix | `services.hermes-agent.extraPlugins` / `extraPythonPackages` | NixOS 선언형 설치 — [Nix 설정](/getting-started/nix-setup#plugins) 참고 |

이름이 충돌하면 나중에 검색된 출처가 이전 출처를 덮어쓰므로, 번들 플러그인과 이름이 같은 사용자 플러그인이 이를 대체합니다.

### 플러그인 하위 카테고리

각 출처 내에서 Hermes는 플러그인을 특수 검색 시스템으로 라우팅하는 하위 카테고리 디렉터리도 인식합니다.

| 하위 디렉터리 | 포함 내용 | 검색 시스템 |
|---|---|---|
| `plugins/` (루트) | 일반 플러그인 — 도구, 훅, 슬래시 명령, CLI 명령, 번들 스킬 | `PluginManager` (종류: `standalone` 또는 `backend`) |
| `plugins/platforms/<name>/` | 게이트웨이 채널 어댑터 (`ctx.register_platform()`) | `PluginManager` (종류: `platform`, 한 단계 더 깊음) |
| `plugins/image_gen/<name>/` | 이미지 생성 백엔드 (`ctx.register_image_gen_provider()`) | `PluginManager` (종류: `backend`, 한 단계 더 깊음) |
| `plugins/memory/<name>/` | 메모리 제공자 (`MemoryProvider` 상속) | `plugins/memory/__init__.py`의 **자체 로더** (종류: `exclusive` — 한 번에 하나만 활성) |
| `plugins/context_engine/<name>/` | 컨텍스트 압축 엔진 (`ctx.register_context_engine()`) | `plugins/context_engine/__init__.py`의 **자체 로더** (한 번에 하나만 활성) |
| `plugins/model-providers/<name>/` | LLM 제공자 프로필 (`register_provider(ProviderProfile(...))`) | `providers/__init__.py`의 **자체 로더** (첫 `get_provider_profile()` 호출 시 지연 검색) |

`~/.hermes/plugins/model-providers/<name>/` 및 `~/.hermes/plugins/memory/<name>/`의 사용자 플러그인은 이름이 같은 번들 플러그인을 덮어씁니다. `register_provider()` / `register_memory_provider()`에서 마지막으로 쓴 값이 적용됩니다. 디렉터리를 넣으면 저장소를 편집하지 않고도 내장 플러그인을 대체할 수 있습니다.

## 플러그인은 선택적으로 활성화됩니다 (몇 가지 예외 있음)

**일반 플러그인과 사용자가 설치한 백엔드는 기본적으로 비활성화됩니다.** 검색을 통해 플러그인을 찾을 수 있으므로 `hermes plugins`와 `/plugins`에 표시되지만, `~/.hermes/config.yaml`의 `plugins.enabled`에 플러그인 이름을 추가하기 전에는 훅이나 도구가 있는 어떤 항목도 로드되지 않습니다. 이를 통해 명시적으로 동의하지 않은 서드파티 코드가 실행되는 것을 막습니다.

```yaml
plugins:
  enabled:
    - my-tool-plugin
    - disk-cleanup
  disabled:       # optional deny-list — always wins if a name appears in both
    - noisy-plugin
```

상태를 전환하는 방법은 세 가지입니다.

```bash
hermes plugins                    # interactive toggle (space to check/uncheck)
hermes plugins enable <name>      # add to allow-list
hermes plugins disable <name>     # remove from allow-list + add to disabled
```

`hermes plugins install owner/repo`를 실행하면 `Enable 'name' now? [y/N]`이라는 질문이 표시되며 기본값은 아니요입니다. 스크립트 설치에서 질문을 건너뛰려면 `--enable` 또는 `--no-enable`을 사용하세요.

재현 가능한 설치를 위해 완전하고 변경 불가능한 커밋을 고정하세요 (태그, 브랜치, 축약 SHA는 허용되지 않음):

```bash
hermes plugins install owner/repo --ref 0123456789abcdef0123456789abcdef01234567
```

Hermes는 커밋을 분리된 HEAD 상태로 체크아웃하고 `HEAD`가 요청한 SHA와 정확히 일치하는지 확인한 다음, 현재 프로필에 정식 출처, 설치된 리비전, 고정 상태를 기록합니다. `hermes plugins update`는 고정된 플러그인을 이동하지 않습니다. `hermes plugins install <source> --force --ref <new-commit>`으로 새 정확한 커밋을 명시적으로 선택하세요. 프로필 로컬 설치 메타데이터에는 설정 값, 환경 값, 시크릿 또는 기능 권한이 포함되지 않습니다.

### 허용 목록이 게이트하지 않는 항목

여러 플러그인 카테고리는 `plugins.enabled`를 우회합니다. 이는 Hermes의 내장 표면에 속하며 기본적으로 게이트하면 기본 기능이 중단되기 때문입니다.

| 플러그인 종류 | 대신 활성화하는 방법 |
|---|---|
| **번들 플랫폼 플러그인** (`plugins/platforms/` 아래의 IRC, Teams 등) | 제공되는 모든 게이트웨이 채널을 사용할 수 있도록 자동 로드됩니다. 실제 채널은 `config.yaml`의 `gateway.platforms.<name>.enabled`로 켭니다. |
| **번들 백엔드** (`plugins/image_gen/` 등의 이미지 생성 제공자) | 기본 백엔드가 "그냥 작동"하도록 자동 로드됩니다. 선택은 `config.yaml`의 `<category>.provider`에서 합니다 (예: `image_gen.provider: openai`). |
| **메모리 제공자** (`plugins/memory/`) | 모두 검색되며 `config.yaml`의 `memory.provider`로 선택한 정확히 하나가 활성화됩니다. |
| **컨텍스트 엔진** (`plugins/context_engine/`) | 모두 검색되며 `config.yaml`의 `context.engine`으로 선택한 하나가 활성화됩니다. |
| **모델 제공자** (`plugins/model-providers/`) | `plugins/model-providers/` 아래의 모든 번들 제공자는 첫 `get_provider_profile()` 호출 시 검색되고 등록됩니다. 사용자는 `--provider` 또는 `config.yaml`로 한 번에 하나를 선택합니다. |
| **pip로 설치한 `backend` 플러그인** | `plugins.enabled`로 선택적으로 활성화합니다 (일반 플러그인과 동일). |
| **사용자 설치 플랫폼** (`~/.hermes/plugins/platforms/` 아래) | `plugins.enabled`로 선택적으로 활성화합니다 — 서드파티 게이트웨이 어댑터는 명시적인 동의가 필요합니다. |

요약하면, **"항상 작동하는" 번들 인프라는 자동으로 로드되고 서드파티 일반 플러그인은 선택적으로 활성화됩니다.** `plugins.enabled` 허용 목록은 사용자가 `~/.hermes/plugins/`에 넣은 임의 코드에 대한 게이트입니다.

### 승인 전송 방식

승인 전송 방식은 기존 Hermes 도구 승인 요청을 **사람이 보고 답하는 위치**를 바꿉니다. 명령에 승인이 필요한지 결정하지 않으며 인증 정책 API도 아닙니다.

```python
def present(request):
    # Deliver request.command and request.description to your UI, wait for
    # its authenticated human response, then return a request-bound decision.
    choice = send_to_my_ui_and_wait(request)  # once/session/always/deny
    return request.respond(choice)


def register(ctx):
    ctx.register_approval_transport("my-ui", present)
```

`present`는 동기 또는 비동기일 수 있습니다. Hermes는 이를 제한된 워커에서 실행하고 플러그인이 처리하지 않더라도 표준 `approvals.timeout`을 적용합니다. 요청은 변경할 수 없으며, 삭제된 표시 텍스트, 호스트 표시 클래스(`cli` 또는 `gateway`), 호스트 타임아웃, 허용된 선택지, 불투명한 요청 ID/다이제스트를 포함합니다.
`request.respond(choice)`의 결과를 반환하세요. 연결되지 않은 딕셔너리와 오래되었거나 변경된 요청 ID/다이제스트는 거부됩니다. 플러그인은 호스트가 제공하지 않은 범위를 반환할 수 없습니다 (예를 들어 한 번만 가능한 요청에서 `always`를 반환할 수 없음).

등록만으로는 아무 일도 일어나지 않습니다. 플러그인을 활성화하는 것과 해당 전송 방식을 명시적으로 선택하는 것은 별도의 동의 단계입니다.

```yaml
plugins:
  enabled: [my-approval-plugin]

security:
  approval:
    transport: my-ui
    transport_fallback: deny     # default
```

전송 예외, 타임아웃, 사용할 수 없는 등록, 잘못된 선택, 오래된 응답은 기본적으로 거부됩니다. 선택한 전송이 실패했을 때 일반 CLI/TUI/게이트웨이/ACP 표면에 프롬프트를 의도적으로 표시하려면 `transport_fallback: builtin`으로 설정하세요. 이 정확한 선택이 없으면 Hermes는 다른 표면에 프롬프트를 생성하지 않습니다.

Hermes는 여전히 강제 차단, sudo-stdin 보호, 사용자 거부 규칙, 요청 연결, 허용 범위, 지속성, 훅 및 최종 인증을 관리합니다. 강제 차단 명령은 모든 전송 콜백보다 먼저 차단됩니다. 이 인터페이스에는 의도적으로 **플러그인 승인 정책, 자동 허용 콜백 또는 필수 `pre_tool_call` 정책**이 없습니다. 향후 승인 정책 기능은 플러그인 기능 동의 모델을 사용할 수 있지만, 전송 방식 선택이 그 권한을 부여하지는 않습니다.

### 기존 사용자 마이그레이션

선택적 플러그인이 있는 Hermes 버전(설정 스키마 v21 이상)으로 업그레이드하면 `~/.hermes/plugins/` 아래에 이미 설치되어 있고 `plugins.disabled`에 아직 들어 있지 않은 사용자 플러그인은 **자동으로** `plugins.enabled`에 추가됩니다. 기존 설정은 계속 작동합니다. 번들 독립형 플러그인은 자동으로 추가되지 않으므로 기존 사용자도 명시적으로 활성화해야 합니다. (번들 플랫폼/백엔드 플러그인은 애초에 게이트되지 않았으므로 추가할 필요가 없었습니다.)

## 사용 가능한 훅

플러그인은 현재 `hermes_cli.plugins.VALID_HOOKS`에서 허용하는 26개의 수명 주기 이벤트를 등록할 수 있습니다. 정확한 실행 시점, 반환 처리, 페이로드 필드, 개인정보 보호 참고 사항은 **[이벤트 훅 카탈로그](/user-guide/features/hooks#shipped-plugin-hook-catalog)**를 기준으로 합니다.

| 설명 범주 | 제공되는 훅 |
|---|---|
| **지시/제어** | `pre_tool_call`, `pre_llm_call`, `pre_verify`, `pre_gateway_dispatch` |
| 변환 | `transform_tool_result`, `transform_terminal_output`, `transform_llm_output`, `pre_transcription` |
| 관찰자 | `post_tool_call`, `post_llm_call`, `pre_api_request`, `post_api_request`, `api_request_error`, `on_stream_start`, `on_stream_delta`, `on_stream_end`, `on_interim_message`, `on_session_start`, `on_session_end`, `on_session_finalize`, `on_session_reset`, `on_skill_lifecycle`, `subagent_start`, `subagent_stop`, `pre_approval_request`, `post_approval_response`, `pre_command`, `kanban_task_claimed`, `kanban_task_completed`, `kanban_task_blocked` |

이 범주는 현재 동작을 설명하는 것이며 향후 명명 규칙을 정의하지 않습니다. 플러그인 미들웨어는 별도의 레지스트리/표면으로 유지됩니다.
## 플러그인 유형

Hermes에는 네 가지 종류의 플러그인이 있습니다.

| 유형 | 하는 일 | 선택 | 위치 |
|------|-------------|-----------|----------|
| **일반 플러그인** | 도구, 훅, 슬래시 명령, CLI 명령 추가 | 다중 선택 (활성화/비활성화) | `~/.hermes/plugins/` |
| **메모리 제공자** | 내장 메모리 대체 또는 보강 | 단일 선택 (하나 활성) | `plugins/memory/` |
| **컨텍스트 엔진** | 내장 컨텍스트 압축기 대체 | 단일 선택 (하나 활성) | `plugins/context_engine/` |
| **모델 제공자** | 추론 백엔드 선언 (OpenRouter, Anthropic, …) | 다중 등록, `--provider` / `config.yaml`로 선택 | `plugins/model-providers/` |

메모리 제공자와 컨텍스트 엔진은 **제공자 플러그인**이며 각 유형에서 한 번에 하나만 활성화할 수 있습니다. 모델 제공자도 플러그인이지만 여러 개가 동시에 로드되며, 사용자가 `--provider` 또는 `config.yaml`로 한 번에 하나를 선택합니다. 일반 플러그인은 원하는 조합으로 활성화할 수 있습니다.

## 플러그형 인터페이스 — 각각 어디로 가야 하는가

위 표는 네 가지 플러그인 카테고리를 보여주지만, "일반 플러그인" 안에서 `PluginContext`는 여러 가지 서로 다른 확장 지점을 제공합니다. 또한 Hermes는 Python 플러그인 시스템 외부의 확장(설정 기반 백엔드, 셸 훅 명령, 외부 서버 등)도 허용합니다. 무엇을 만들고 싶은지에 맞는 문서를 찾으려면 다음 표를 사용하세요.

| 추가하려는 것 | 방법 | 작성 가이드 |
|---|---|---|
| LLM이 호출할 수 있는 **도구** | Python 플러그인 — `ctx.register_tool()` | [Hermes 플러그인 빌드](/developer-guide/plugins) · [도구 추가](/developer-guide/adding-tools) |
| **수명 주기 훅** (LLM 전/후, 세션 시작/종료, 도구 필터) | Python 플러그인 — `ctx.register_hook()` | [훅 참고](/user-guide/features/hooks) · [Hermes 플러그인 빌드](/developer-guide/plugins) |
| CLI/게이트웨이용 **슬래시 명령** | Python 플러그인 — `ctx.register_command()` | [Hermes 플러그인 빌드](/developer-guide/plugins) · [CLI 확장](/developer-guide/extending-the-cli) |
| `hermes <thing>`용 **하위 명령** | Python 플러그인 — `ctx.register_cli_command()` | [CLI 확장](/developer-guide/extending-the-cli) |
| 플러그인이 제공하는 번들 **스킬** | Python 플러그인 — `ctx.register_skill()` | [스킬 만들기](/developer-guide/creating-skills) |
| **추론 백엔드** (LLM 제공자: OpenAI 호환, Codex, Anthropic-Messages, Bedrock) | `plugins/model-providers/<name>/`에서 제공자 플러그인 — `register_provider(ProviderProfile(...))` | **[모델 제공자 플러그인](/developer-guide/model-provider-plugin)** · [제공자 추가](/developer-guide/adding-providers) |
| **게이트웨이 채널** (Discord / Telegram / IRC / Teams 등) | `plugins/platforms/<name>/`에서 플랫폼 플러그인 — `ctx.register_platform()` | [플랫폼 어댑터 추가](/developer-guide/adding-platform-adapters) |
| **메모리 백엔드** (Honcho, Mem0, Supermemory 등) | 메모리 플러그인 — `plugins/memory/<name>/`에서 `MemoryProvider` 상속 | [메모리 제공자 플러그인](/developer-guide/memory-provider-plugin) |
| **컨텍스트 압축 전략** | 컨텍스트 엔진 플러그인 — `ctx.register_context_engine()` | [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin) |
| **이미지 생성 백엔드** (DALL·E, SDXL 등) | 백엔드 플러그인 — `ctx.register_image_gen_provider()` | [이미지 생성 제공자 플러그인](/developer-guide/image-gen-provider-plugin) |
| **동영상 생성 백엔드** (Veo, Kling, Pixverse, Grok-Imagine, Runway 등) | 백엔드 플러그인 — `ctx.register_video_gen_provider()` | [동영상 생성 제공자 플러그인](/developer-guide/video-gen-provider-plugin) |
| **TTS 백엔드** (모든 CLI — Piper, VoxCPM, Kokoro, xtts, 음성 복제 스크립트 등) | 설정 기반 (권장) — `config.yaml`의 `tts.providers.<name>` 아래에 `type: command`로 선언. 또는 셸 템플릿 이상의 기능이 필요한 Python SDK/스트리밍 엔진은 Python 백엔드 플러그인 — `ctx.register_tts_provider()` | [TTS 설정](/user-guide/features/tts#custom-command-providers) · [Python 플러그인 가이드](/user-guide/features/tts#python-plugin-providers) |
| **STT 백엔드** (모든 CLI — whisper.cpp, 사용자 지정 whisper 바이너리, 로컬 ASR CLI) | 설정 기반 (권장) — `config.yaml`의 `stt.providers.<name>` 아래에 `type: command`로 선언하거나 레거시 단일 명령 우회 수단인 `HERMES_LOCAL_STT_COMMAND`를 설정. 또는 Python SDK 엔진(OpenRouter, SenseAudio, Gemini-STT 등)은 Python 백엔드 플러그인 — `ctx.register_transcription_provider()` | [STT 설정](/user-guide/features/tts#stt-custom-command-providers) · [Python 플러그인 가이드](/user-guide/features/tts#python-plugin-providers-stt) |
| MCP를 통한 **외부 도구** (filesystem, GitHub, Linear, Notion, 모든 MCP 서버) | 설정 기반 — `config.yaml`에 `mcp_servers.<name>`을 `command:` / `url:`과 함께 선언. Hermes가 서버의 도구를 자동으로 검색하여 내장 도구와 함께 등록 | [MCP](/user-guide/features/mcp) |
| **추가 스킬 출처** (사용자 지정 GitHub 저장소, 비공개 스킬 인덱스) | CLI — `hermes skills tap add <repo>` | [스킬 허브](/user-guide/features/skills#skills-hub) · [사용자 지정 탭 게시](/user-guide/features/skills#publishing-a-custom-skill-tap) |
| **게이트웨이 이벤트 훅** (`gateway:startup`, `session:start`, `agent:end`, `command:*`에서 실행) | `~/.hermes/hooks/<name>/`에 `HOOK.yaml` + `handler.py` 배치 | [이벤트 훅](/user-guide/features/hooks#gateway-event-hooks) |
| **셸 훅** (이벤트에서 셸 명령 실행 — 알림, 감사 로그, 데스크톱 알림) | 설정 기반 — `config.yaml`의 `hooks:` 아래에 선언 | [셸 훅](/user-guide/features/hooks#shell-hooks) |

:::note
모든 것이 Python 플러그인은 아닙니다. 일부 확장 표면은 **설정 기반 셸 명령**(TTS, STT, 셸 훅)을 의도적으로 사용하므로, 이미 가지고 있는 CLI를 Python을 작성하지 않고도 플러그인으로 만들 수 있습니다. 다른 확장 표면은 에이전트가 연결하여 도구를 자동 등록하는 **외부 서버**(MCP)입니다. 또 일부는 자체 매니페스트 형식을 가진 **드롭인 디렉터리**(게이트웨이 훅)입니다. 통합 방식에 맞는 표면을 선택하세요. 위 표의 작성 가이드에서 각각 플레이스홀더, 검색, 예제를 설명합니다.
:::

## NixOS 선언형 플러그인

NixOS에서는 모듈 옵션으로 플러그인을 선언적으로 설치할 수 있으므로 `hermes plugins install`이 필요하지 않습니다. 자세한 내용은 **[Nix 설정 가이드](/getting-started/nix-setup#plugins)**를 참고하세요.

```nix
services.hermes-agent = {
  # Directory plugin (source tree with plugin.yaml)
  extraPlugins = [ (pkgs.fetchFromGitHub { ... }) ];
  # Entry-point plugin (pip package)
  extraPythonPackages = [ (pkgs.python312Packages.buildPythonPackage { ... }) ];
  # Enable in config
  settings.plugins.enabled = [ "my-plugin" ];
};
```

선언형 플러그인은 `nix-managed-` 접두사가 붙은 심볼릭 링크로 연결됩니다. 수동으로 설치한 플러그인과 함께 사용할 수 있으며 Nix 설정에서 제거하면 자동으로 정리됩니다.

## 플러그인 관리

```bash
hermes plugins                               # unified interactive UI
hermes plugins list                          # table: enabled / disabled / not enabled
hermes plugins search <term>                 # search the community plugin index
hermes plugins install <name>                # install by index name (resolved to repo @ pinned ref)
hermes plugins install user/repo             # install from Git, then prompt Enable? [y/N]
hermes plugins install user/repo --enable    # install AND enable (no prompt)
hermes plugins install user/repo --no-enable # install but leave disabled (no prompt)
hermes plugins update my-plugin              # pull latest (local edits are autostashed and re-applied)
hermes plugins remove my-plugin              # uninstall
hermes plugins enable my-plugin              # add to allow-list
hermes plugins disable my-plugin             # remove from allow-list + add to disabled
hermes plugins capabilities [my-plugin]      # declared vs granted capabilities
```

### 플러그인 기능과 동의

플러그인은 `plugin.yaml`에 원하는 권한 있는 호스트 표면을 선언할 수 있습니다.

```yaml
name: my-plugin
capabilities:
  - tools.override        # replace built-in tools
  - llm.model_override    # pick the model for host-owned LLM calls
```

플러그인이 기능을 선언하면 `hermes plugins install`(및 `hermes plugins enable`)이 한 줄짜리 위험 설명과 함께 목록을 표시하고 한 번 질문합니다. 동의하면 `plugins.entries.<id>.granted_capabilities`에 동의 해시 및 타임스탬프와 함께 권한이 기록됩니다. 거부해도 플러그인은 활성 상태로 남지만 해당 기능은 꺼져 있습니다. 잘 작성된 플러그인은 `ctx.has_capability()`로 확인하고 정상적으로 기능을 축소합니다.

**업데이트 시 재동의:** 플러그인 업데이트에서 아직 부여하지 않은 기능을 선언하면 `hermes plugins update`가 추가된 기능을 표시하고 다시 질문합니다. 동의하기 전까지 새 기능은 꺼져 있습니다. 플러그인 업데이트가 조용히 접근 권한을 넓힐 수는 없습니다.

**비대화형 세션은 닫힌 상태로 실패:** TTY 없이 설치하거나 업데이트하면 설치는 완료되지만 선언된 기능은 부여되지 않습니다. 나중에 기능을 부여하려면 대화형으로 `hermes plugins enable <id>`를 실행하세요.

언제든지 상태를 확인할 수 있습니다.

```bash
hermes plugins capabilities             # all plugins with declared/granted capabilities
hermes plugins capabilities my-plugin   # one plugin, declared vs granted
```

기능 ID는 이전 기능별 설정 게이트와 1:1로 대응합니다. 이 게이트는 계속 작동하지만 동의 흐름을 우선하도록 **사용 중단 예정**입니다.

| 기능 | 레거시 키 (`plugins.entries.<id>.…`) |
|---|---|
| `tools.override` | `allow_tool_override` |
| `llm.provider_override` | `llm.allow_provider_override` |
| `llm.model_override` | `llm.allow_model_override` |
| `llm.agent_id_override` | `llm.allow_agent_id_override` |
| `llm.profile_override` | `llm.allow_profile_override` |
| `llm.task_override` | `llm.allow_task_override` |
| `gateway.platform_actions` | `allow_platform_actions` |

기능이 부여되었거나 레거시 키가 설정되어 있으면 게이트가 열립니다. 기존 설정은 변경 없이 계속 작동합니다.

:::warning 샌드박스가 아님
기능은 **동의 및 감사 계층**이지 격리 기능이 아닙니다. 플러그인은 일반적인 인프로세스 Python 코드로 실행되므로 악성 플러그인은 여기의 모든 게이트를 무시할 수 있습니다. 기능을 부여한다는 것은 플러그인 작성자를 신뢰한다는 뜻이지, 코드를 감사했다는 뜻이 아니며 Hermes가 플러그인 코드를 검토했다는 뜻도 아닙니다. 신뢰할 수 있는 출처의 플러그인만 설치하세요.
:::

### 플랫폼 작업

`ctx.platform_actions`는 실행 중인 게이트웨이 어댑터 레지스트리를 통해 연결된 채팅 플랫폼에서 작동하기 위한, 권한 기능으로 게이트되는 최소한의 동사 집합을 플러그인에 제공합니다. 이는 어댑터를 몽키 패치하는 대신 승인된 방식입니다. **기본값은 꺼져 있습니다.** 호출할 때마다 `gateway.platform_actions` 기능(레거시 키 `plugins.entries.<id>.allow_platform_actions`)을 다시 확인하며, 부여되지 않은 호출은 동작하지 않고 구조화된 오류를 반환합니다.

v1 동사 (둘 다 `async`이고, 일반 딕셔너리를 반환하며, 훅 디스패치로 예외를 전파하지 않음):

```python
result = await ctx.platform_actions.add_reaction(
    platform="telegram", chat_id="-100123", message_id="456", emoji="👍",
)
result = await ctx.platform_actions.set_thread_title(
    platform="discord", chat_id="123", thread_id="456", title="New title",
)
if not result["ok"]:
    print(result["error"], result.get("detail"))
```

성공 결과는 `{"ok": True, "action": <verb>}`입니다. 실패 결과는 안정적인 오류 코드와 함께 `{"ok": False, "error": <code>, "detail": <str>}`입니다:
`capability_not_granted`, `invalid_argument`, `gateway_unavailable`, `unknown_platform`, `adapter_not_registered`, `adapter_disconnected`, `unsupported_platform_action`, `action_failed`. 작업을 실행하기 전에 대상 어댑터가 존재하고 연결되어 있는지 확인합니다. 연결이 끊겼거나 없는 어댑터는 예외가 아닌 구조화된 오류로 처리됩니다.

v1에서 지원하는 플랫폼은 Telegram과 Discord입니다. Telegram의 `add_reaction`은 봇의 반응을 *설정*합니다 (Bot API는 봇의 이전 반응을 누적하지 않고 대체함). 허용되거나 거부된 모든 작업은 플러그인 ID, 동사, 플랫폼, 결과와 함께 로그에 기록됩니다.

:::warning 보안 참고 사항
플랫폼 작업은 **봇으로서 메시지를 보낼 수 있는 권한**입니다. 권한을 부여받은 플러그인은 훅을 발생시킨 채팅뿐 아니라 게이트웨이 봇이 접근할 수 있는 모든 채팅에서 반응을 달고 스레드 이름을 바꿀 수 있습니다. 신뢰하는 플러그인에만 `gateway.platform_actions`를 부여하고, 수행하는 작업을 정확히 문서화한 플러그인을 우선하세요.
원시 플랫폼 SDK 페이로드/핸들 접근은 이 표면에 의도적으로 포함되지 않습니다. #64176 2차 설계 수정에 따르면 자체 기능(`gateway.raw_events`)과 "안정성 보장 없음" 라벨 및 별도 설계가 필요하며, 아직 제공되지 않았습니다.
:::

### 커뮤니티 플러그인 검색

`hermes plugins search <term>`은 **커뮤니티 플러그인 인덱스**를 검색합니다. 커뮤니티 플러그인의 정적이며 기계 판독 가능한 JSON 카탈로그입니다. 이름, 설명, 태그를 대상으로 퍼지 검색이 수행됩니다.

```bash
hermes plugins search telegram               # fuzzy search
hermes plugins search                        # browse the whole index
hermes plugins search --capability platform  # filter by declared capability
hermes plugins search media --json           # machine-readable output
hermes plugins search --refresh              # bypass the 24h local cache
```

플러그인을 찾았다면 이름만 사용하여 설치합니다. 이름은 인덱스를 통해 `owner/repo` 및 인덱스에서 고정한 커밋으로 해석됩니다.

```bash
hermes plugins install hermes-media-studio
```

이름이 둘 이상의 항목과 일치하면 후보가 표시되고 아무것도 설치되지 않습니다. 명시적인 `owner/repo` 또는 Git URL 식별자는 인덱스를 건드리지 않으며 이전과 똑같이 작동합니다. 명시적인 `--ref <sha>`는 언제나 인덱스 고정을 덮어씁니다.

**인덱스를 가져오는 방법.** 인덱스는 정식 URL에 있습니다 (`https://raw.githubusercontent.com/NousResearch/hermes-plugin-index/main/index.json`, `hermes config set plugins.index_url <url>`로 재정의 가능). 가져온 데이터는 24시간 동안 `~/.hermes/cache/plugin_index.json`에 캐시됩니다. 원격에 연결할 수 없으면 오래된 캐시를 사용하고, 캐시가 전혀 없으면 Hermes에 번들된 시드 사본을 사용하므로 검색이 완전히 오프라인에서도 작동합니다.

**인덱스 항목 형식.** 각 항목은 JSON 객체입니다.

```json
{
  "name": "hermes-media-studio",
  "description": "Generative media workspace plugin.",
  "author": "NousResearch",
  "tags": ["media", "image-gen"],
  "repo": "NousResearch/hermes-media-studio",
  "ref": "<40-char commit SHA>",
  "subdir": null,
  "homepage": "https://github.com/NousResearch/hermes-media-studio",
  "capabilities": ["tools", "dashboard"],
  "api_version": 1,
  "added_at": "2026-08-12"
}
```

`repo`는 `owner/name` 형식의 GitHub 식별자이고, `ref`는 변경할 수 없는 커밋 SHA이며, 선택적인 `subdir`은 모노레포를 지원합니다. 번들된 시드 파일(저장소의 `hermes_cli/data/plugin_index.json`)이 형식의 참고 자료입니다.

**플러그인 제출.** 인덱스는 일반 JSON 파일로 관리됩니다. [hermes-plugin-index](https://github.com/NousResearch/hermes-plugin-index) 저장소에 풀 리퀘스트를 제출하고 항목(이름, 설명, 작성자, 태그, `owner/repo`, 고정 커밋 SHA)을 추가하세요. 검토 대상은 항목의 *메타데이터*뿐입니다.

:::warning 인덱스에 포함되었다고 감사된 것은 아님
커뮤니티 인덱스에 포함되었다는 것은 항목의 메타데이터가 검토되었다는 뜻일 뿐이며, **코드 감사가 아닙니다.** 설치는 여전히 일반 동의/검토 흐름을 따릅니다 (플러그인 설치는 기본적으로 비활성화되고, 활성화는 명시적인 단계이며, 도구 대체 권한에는 별도 부여가 필요함). 활성화하기 전에 플러그인의 소스 코드를 검토하세요.
:::

### 플러그인 팩

**플러그인 팩**은 여러 플러그인을 고정하는 선언적이고 공유 가능한 YAML 파일(`hermes-pack.yaml`)입니다. 모드팩을 공유하는 것과 비슷합니다. 설치하면 일반적인 고정 설치로 분산되며 런타임에 새로운 것이 추가되지는 않습니다.

```yaml
name: voice-assistant-pack
description: STT + streaming TTS + approval relay
author: hyper
version: 1.0.0
plugins:
  - name: hermes-media-studio            # bare community-index name…
    ref: e8d59971d2b7901405b39dac7b03bdd616272d0d
  - repo: owner/approval-relay           # …or explicit owner/repo (or git URL)
    ref: 8f3c2d1a9b4e5f6071829304a5b6c7d8e9f00112
    subdir: plugins/relay                # optional monorepo path
config:                                  # optional, non-secret seeds only
  hermes-media-studio:
    default_model: flux-3
skills: []                               # declared list only (not auto-installed yet)
```

```bash
hermes plugins pack show ./hermes-pack.yaml     # dry-run review
hermes plugins pack install ./hermes-pack.yaml  # review → confirm → install
hermes plugins pack export > hermes-pack.yaml   # snapshot the current install
hermes plugins pack export --enabled-only       # only plugins.enabled
```

**공급망 방침.** 모든 항목의 `ref`는 정확히 40자인 커밋 SHA여야 합니다. 태그와 브랜치 이름은 항목 이름을 포함한 오류와 함께 거부되며, 이는 커뮤니티 인덱스와 동일한 규칙입니다. 팩 설치는 `hermes plugins install --ref <sha>`와 완전히 동일한 고정 설치 경로를 사용하고 `plugins/.install-metadata.json`에 동일한 출처를 기록하므로 같은 팩을 두 번 설치해도 동일하게 해석됩니다. 팩은 [매니페스트 v2 필드](/developer-guide/plugins)(`manifest_version`, `api_version`, `requires_plugins`)를 기반으로 합니다. 각 플러그인의 자체 매니페스트는 일반 설치 경로를 통해 계속 검증됩니다.

**동의는 일괄 부여되지 않습니다.** `pack install`은 필수 검토 화면(모든 플러그인, 출처, 고정 ref, 선언된 기능)을 표시한 후 팩 내용에 대해 한 번 확인을 요청합니다. 그 뒤 각 플러그인의 선언된 기능은 단일 `hermes plugins install`과 동일하게 플러그인별 표준 기능 동의 프롬프트를 거칩니다. `--yes`는 없으며 비대화형 세션에서는 팩을 설치할 수 없습니다.

**시크릿은 팩에 포함되지 않습니다.** `config:` 시드는 비밀이 아닌 `plugins.entries.<id>` 키로 제한됩니다. 비밀처럼 보이는 키 이름(`*token*`, `*key*`, `*password*`, …), 기능 부여, 사용 중단된 `allow_*` 신뢰 게이트는 설치 시 거부되고 내보낼 때 제거됩니다. 시크릿이 필요한 플러그인은 자체 `requires_env`에 이를 선언하며 평소처럼 설치 중 입력을 요청합니다. `plugins.entries.<id>`에 있는 기존 사용자 값은 언제나 팩 시드보다 우선합니다.

**부분 실패.** 각 플러그인은 독립적으로 설치됩니다. 실패는 플러그인별로 보고되고 나머지는 계속 진행되며, 하나라도 실패하면 명령이 0이 아닌 상태로 종료됩니다.

**내보내기 주의 사항.** `pack export`에는 알려진 Git 출처가 있는 플러그인(`hermes plugins install`로 설치된 플러그인)만 포함됩니다. 로컬 전용 플러그인은 출력되는 YAML에 경고 주석으로 나열되며 설치 가능한 항목으로 포함되지 않습니다.

`skills:` 목록은 설치 시 파싱되고 표시되지만 아직 자동 설치되지는 않습니다. 지금은 수동으로 설치하세요 (`hermes skills`). 스킬 허브 ID를 팩 설치에 연결하는 작업은 문서화된 후속 확장 지점입니다.

### 대화형 UI

인자 없이 `hermes plugins`를 실행하면 통합 대화형 화면이 열립니다.

```
Plugins
  ↑↓ navigate  SPACE toggle  ENTER configure/confirm  ESC done

  General Plugins
 → [✓] my-tool-plugin — Custom search tool
   [ ] webhook-notifier — Event hooks
   [ ] disk-cleanup — Auto-cleanup of ephemeral files [bundled]

  Provider Plugins
     Memory Provider          ▸ honcho
     Context Engine           ▸ compressor
```

- **일반 플러그인 섹션** — 체크박스이며 SPACE로 전환합니다. 선택됨 = `plugins.enabled`에 있음, 선택 해제됨 = `plugins.disabled`에 있음 (명시적으로 끔).
- **제공자 플러그인 섹션** — 현재 선택 항목을 보여줍니다. ENTER를 눌러 활성 제공자 하나를 선택하는 라디오 선택기로 들어갑니다.
- 번들 플러그인은 `[bundled]` 태그와 함께 같은 목록에 표시됩니다.

제공자 플러그인 선택은 `config.yaml`에 저장됩니다.

```yaml
memory:
  provider: "honcho"      # empty string = built-in only

context:
  engine: "compressor"    # default built-in compressor
```

### 활성화됨 vs. 비활성화됨 vs. 둘 다 아님

플러그인은 세 가지 상태 중 하나입니다.

| 상태 | 의미 | `plugins.enabled`에 있음? | `plugins.disabled`에 있음? |
|---|---|---|---|
| `enabled` | 다음 세션에서 로드됨 | 예 | 아니요 |
| `disabled` | 명시적으로 꺼짐 — `enabled`에도 있어도 로드되지 않음 | (무관) | 예 |
| `not enabled` | 검색되었지만 아직 선택적으로 활성화되지 않음 | 아니요 | 아니요 |

새로 설치했거나 번들된 플러그인의 기본값은 `not enabled`입니다. `hermes plugins list`는 세 가지 상태를 모두 구분하여 표시하므로 명시적으로 꺼진 항목과 단지 활성화를 기다리는 항목을 구별할 수 있습니다.

실행 중인 세션에서 `/plugins`를 실행하면 현재 로드된 플러그인을 확인할 수 있습니다.

## 메시지 주입

플러그인은 `ctx.inject_message()`를 사용하여 CLI 대화 또는 알고 있는 게이트웨이 세션에 메시지를 주입할 수 있습니다.

```python
# Active CLI conversation
ctx.inject_message("New data arrived from the webhook", role="user")

# Existing gateway conversation
ctx.inject_message(
    "New data arrived from the webhook",
    role="user",
    session_key="agent:main:telegram:dm:123456789",
)
```

**시그니처:** `ctx.inject_message(content: str, role: str = "user", *, session_key: str | None = None) -> bool`

CLI 모드:

- 에이전트가 **유휴 상태**(사용자 입력을 기다리는 중)라면 메시지가 다음 입력으로 대기열에 들어가고 새 턴이 시작됩니다.
- 에이전트가 **턴 중간**(실행 중)이라면 메시지가 현재 작업을 중단합니다. 사용자가 새 메시지를 입력하고 Enter를 누른 것과 같습니다.
- `"user"`가 아닌 역할의 경우 콘텐츠 앞에 `[role]`이 붙습니다 (예: `[system] ...`).
- 메시지가 성공적으로 대기열에 들어가면 `True`를 반환합니다.

게이트웨이 모드:

- `session_key`가 필요하며 기존 게이트웨이 세션을 식별해야 합니다. 이는 CLI 세션 ID가 아니라 안정적인 라우팅 키입니다.
- Hermes는 해당 세션에 저장된 플랫폼, 채팅, 스레드, 프로필 및 대화 기록을 재사용합니다. 플러그인은 이 API로 새 채팅 경로를 제공할 수 없습니다.
- Hermes는 디스패치 전에 저장된 경로를 게이트웨이의 현재 인증 규칙과 다시 대조합니다.
- 어댑터 시점 또는 업스트림 인증 결정에만 의존한 경로는 Hermes가 현재 코어 허용 목록, 페어링 또는 명시적 모두 허용 설정으로 다시 검증할 수 없다면 거부됩니다.
- 주입된 텍스트는 항상 대화 입력입니다. 슬래시 명령을 실행하거나 도구를 승인하거나 대기 중인 확인 및 명확화 프롬프트를 해결할 수 없습니다.
- 디스패치가 대기 중인 동안 경로와 대화가 고정됩니다. 처리가 시작되기 전에 주제 복구로 경로가 변경되거나 세션이 교체되면 Hermes는 요청을 버립니다.
- 요청은 플랫폼 어댑터의 일반 메시지 경로로 들어갑니다. 활성 세션은 경쟁하는 턴을 시작하는 대신 기존 바쁜 세션 대기열을 사용합니다.
- 실시간 게이트웨이가 비동기 디스패치를 수락하면 `True`를 반환합니다. 이는 에이전트 턴이나 플랫폼 전달이 완료되었다는 뜻은 아닙니다.
- `session_key`가 누락되었거나 권한이 없거나 실시간 게이트웨이가 요청을 수락할 수 없으면 `False`를 반환합니다. 비동기 수락 후 발견된 알 수 없거나 라우팅할 수 없는 세션 키는 게이트웨이 로그에 기록됩니다.

이를 통해 원격 제어 뷰어, 메시징 브리지 또는 웹훅 수신기 같은 플러그인이 외부 소스의 메시지를 대화로 전달할 수 있습니다.

게이트웨이 주입은 에이전트 응답을 외부 메시징 플랫폼으로 보낼 수 있습니다. 모든 플러그인에서 기본적으로 비활성화되어 있습니다. `config.yaml`에서 플러그인별로 부여하세요.

```yaml
plugins:
  entries:
    my-plugin:
      allow_gateway_injection: true
```

:::warning
게이트웨이 주입은 신뢰하는 플러그인에만 부여하세요. Hermes는 이 호스트 API 권한을 확인하고 기존 세션 경로로 제한하지만 Python 플러그인은 인프로세스로 실행되며 이 설정은 샌드박스가 아닙니다.
:::

:::note
이 플러그인 API는 외부 프로세스를 위한 공개 HTTP 엔드포인트나 CLI 명령을 제공하지 않습니다. 플러그인은 대상 게이트웨이 `session_key`를 이미 알고 있어야 합니다. 예를 들어 신뢰할 수 있는 자체 설정이나 이전에 보존한 세션 상태에서 알아야 합니다.
:::

## 플러그인에서 MCP 서버 호출

`ctx.call_mcp()`를 사용하면 플러그인이 사용자가 설정한 MCP 서버 중 하나의 도구를 동기적으로 호출할 수 있습니다. 모든 훅이나 도구 핸들러에서 사용할 수 있으며 Hermes의 기존 네이티브 MCP 클라이언트를 통해 라우팅됩니다 (모델이 호출하는 MCP 도구와 동일한 연결, 신뢰 등급 게이트, 회로 차단기 및 재연결 로직을 사용하며 병렬 클라이언트를 만들지 않음).

```python
result = ctx.call_mcp(
    "knowledge_rag",            # server name from mcp.servers
    "query_knowledge",          # tool on that server
    {"query": "deploy runbook"},
    timeout=30,                 # seconds; clamped to 1–600
)
if result["ok"]:
    print(result["result"])
else:
    print("MCP error:", result["error"])
```

**시그니처:** `ctx.call_mcp(server: str, tool: str, arguments: dict | None = None, timeout: float = 30) -> dict`

안정적인 봉투를 반환합니다. `{"ok": True, "result": ...}`(서버가 제공하는 경우 `structuredContent`도 포함) 또는 `{"ok": False, "error": "..."}`입니다. 약 64KB를 초과하는 결과는 잘리고 `"truncated": True`로 표시됩니다.

### 보안: 기본 꺼짐, 서버별 허용 목록

플러그인은 기본적으로 MCP에 접근할 수 없습니다. 운영자가 `config.yaml`에서 각 서버를 명시적으로 허용해야 합니다.

```yaml
plugins:
  entries:
    my-plugin:
      mcp_allowlist: ["knowledge_rag", "github"]
```

- 목록에 없는 서버를 호출하면 설정할 정확한 구성 키를 명시한 `PermissionError`가 발생합니다.
- 권한은 서버별 및 플러그인별입니다. 설정된 모든 서버에 대한 암묵적 권한이 아니며 `"*"` 와일드카드도 적용되지 않습니다.
- 모든 호출에는 제한 시간(기본 30초)이 적용되므로 멈춘 MCP 서버가 호출한 훅이나 도구 파이프라인을 멈출 수 없습니다.
- MCP 서버는 신뢰할 수 없는 콘텐츠를 반환합니다. `result`를 지시가 아닌 데이터로 취급하세요. 검증 없이 권한 있는 결정(승인, 명령 실행)에 전달하지 마세요.

:::warning
`mcp_allowlist`를 부여하면 플러그인이 해당 MCP 서버에 대해 모델과 동일한 접근 권한을 갖게 됩니다. 여기에는 서버가 노출하는 쓰기 가능한 도구도 포함됩니다 (서버의 `trust` 등급 게이트 적용). 플러그인에 정말 필요한 서버만 허용하세요.
:::

핸들러 계약, 스키마 형식, 훅 동작, 오류 처리, 일반적인 실수는 **[전체 가이드](/developer-guide/plugins)**를 참고하세요.
