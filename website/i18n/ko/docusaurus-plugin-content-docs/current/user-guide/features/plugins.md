---
sidebar_position: 11
sidebar_label: "플러그인 (Plugins)"
title: "플러그인 (Plugins)"
description: "플러그인 시스템을 통해 사용자 지정 도구, 훅 및 통합으로 Hermes 확장하기"
---

# 플러그인 (Plugins)

Hermes에는 핵심 코드를 수정하지 않고 사용자 지정 도구, 훅 및 통합을 추가할 수 있는 플러그인 시스템이 있습니다.

본인, 팀 또는 한 프로젝트를 위한 사용자 지정 도구를 만들고 싶다면, 일반적으로 이 방법이 올바른 경로입니다. 개발자 가이드의 [도구 추가(Adding Tools)](/developer-guide/adding-tools) 페이지는 `tools/` 및 `toolsets.py`에 있는 기본 제공 Hermes 핵심 도구를 위한 것입니다.

**→ [Hermes 플러그인 구축하기](/guides/build-a-hermes-plugin)** — 완전한 작동 예시가 포함된 단계별 가이드.

## 빠른 개요

`plugin.yaml`과 Python 코드가 있는 디렉토리를 `~/.hermes/plugins/`에 넣습니다.

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml      # 매니페스트
├── __init__.py      # register() — 스키마를 핸들러에 연결
├── schemas.py       # 도구 스키마 (LLM이 보는 내용)
└── tools.py         # 도구 핸들러 (호출 시 실행되는 내용)
```

Hermes를 시작하세요 — 도구가 기본 제공 도구와 나란히 나타납니다. 모델은 이를 즉시 호출할 수 있습니다.

### 최소 작동 예시

다음은 `hello_world` 도구를 추가하고 훅(hook)을 통해 모든 도구 호출을 기록하는 완전한 플러그인입니다.

**`~/.hermes/plugins/hello-world/plugin.yaml`**

```yaml
name: hello-world
version: "1.0"
description: 최소한의 예제 플러그인
```

**`~/.hermes/plugins/hello-world/__init__.py`**

```python
"""최소한의 Hermes 플러그인 — 도구와 훅을 등록합니다."""

import json


def register(ctx):
    # --- 도구: hello_world ---
    schema = {
        "name": "hello_world",
        "description": "주어진 이름에 대한 친근한 인사말을 반환합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "인사할 이름",
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
        description="주어진 이름에 대한 친근한 인사말을 반환합니다.",
    )

    # --- 훅: 모든 도구 호출 기록 ---
    def on_tool_call(tool_name, params, result):
        print(f"[hello-world] 도구 호출됨: {tool_name}")

    ctx.register_hook("post_tool_call", on_tool_call)
```

두 파일을 모두 `~/.hermes/plugins/hello-world/`에 넣고 Hermes를 다시 시작하면 모델이 즉시 `hello_world`를 호출할 수 있습니다. 훅은 모든 도구 호출 후에 로그 줄을 인쇄합니다.

`./.hermes/plugins/` 아래의 프로젝트 로컬 플러그인은 기본적으로 비활성화되어 있습니다. 신뢰할 수 있는 리포지토리에 대해서만 활성화하려면 Hermes를 시작하기 전에 `HERMES_ENABLE_PROJECT_PLUGINS=true`를 설정하세요.

## 플러그인이 할 수 있는 일

아래의 모든 `ctx.*` API는 플러그인의 `register(ctx)` 함수 내에서 사용할 수 있습니다.

| 기능 | 방법 |
|-----------|-----|
| 도구 추가 | `ctx.register_tool(name=..., toolset=..., schema=..., handler=...)` |
| 훅 추가 | `ctx.register_hook("post_tool_call", callback)` |
| 슬래시 명령어 추가 | `ctx.register_command(name, handler, description)` — CLI 및 게이트웨이 세션에서 `/name` 추가 |
| 명령어에서 도구 디스패치 | `ctx.dispatch_tool(name, args)` — 상위 에이전트 컨텍스트가 자동 연결된 등록된 도구 호출 |
| CLI 명령어 추가 | `ctx.register_cli_command(name, help, setup_fn, handler_fn)` — `hermes <plugin> <subcommand>` 추가 |
| 메시지 주입 | `ctx.inject_message(content, role="user")` — [메시지 주입](#injecting-messages) 참조 |
| 데이터 파일 포함 | `Path(__file__).parent / "data" / "file.yaml"` |
| 번들 스킬 | `ctx.register_skill(name, path)` — `plugin:skill` 네임스페이스 지정, `skill_view("plugin:skill")`을 통해 로드됨 |
| 환경 변수 게이트웨이 | plugin.yaml의 `requires_env: [API_KEY]` — `hermes plugins install` 중 프롬프트 표시 |
| pip를 통한 배포 | `[project.entry-points."hermes_agent.plugins"]` |
| 게이트웨이 플랫폼 등록 (Discord, Telegram, IRC 등) | `ctx.register_platform(name, label, adapter_factory, check_fn, ...)` — [플랫폼 어댑터 추가](/developer-guide/adding-platform-adapters) 참조 |
| 이미지 생성 백엔드 등록 | `ctx.register_image_gen_provider(provider)` — [이미지 생성 공급자 플러그인](/developer-guide/image-gen-provider-plugin) 참조 |
| 비디오 생성 백엔드 등록 | `ctx.register_video_gen_provider(provider)` — [비디오 생성 공급자 플러그인](/developer-guide/video-gen-provider-plugin) 참조 |
| 컨텍스트 압축 엔진 등록 | `ctx.register_context_engine(engine)` — [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin) 참조 |
| 메모리 백엔드 등록 | `plugins/memory/<name>/__init__.py`에서 `MemoryProvider` 서브클래싱 — [메모리 공급자 플러그인](/developer-guide/memory-provider-plugin) 참조 (별도의 검색 시스템 사용) |
| 호스트 소유 LLM 호출 실행 | `ctx.llm.complete(...)` / `ctx.llm.complete_structured(...)` — 선택적 JSON 스키마 유효성 검사를 통해 원샷(one-shot) 완성을 위해 사용자의 활성 모델 및 인증 차용. [플러그인 LLM 액세스](/developer-guide/plugin-llm-access) 참조 |
| 추론 백엔드 등록 (LLM 공급자) | `plugins/model-providers/<name>/__init__.py`의 `register_provider(ProviderProfile(...))` — [모델 공급자 플러그인](/developer-guide/model-provider-plugin) 참조 (별도의 검색 시스템 사용) |

## 플러그인 검색 (Plugin discovery)

| 출처 | 경로 | 사용 사례 |
|--------|------|----------|
| 번들 | `<repo>/plugins/` | Hermes와 함께 제공됨 — [기본 제공 플러그인](/user-guide/features/built-in-plugins) 참조 |
| 사용자 | `~/.hermes/plugins/` | 개인 플러그인 |
| 프로젝트 | `.hermes/plugins/` | 프로젝트별 플러그인 (`HERMES_ENABLE_PROJECT_PLUGINS=true` 필요) |
| pip | `hermes_agent.plugins` entry_points | 배포된 패키지 |
| Nix | `services.hermes-agent.extraPlugins` / `extraPythonPackages` | NixOS 선언적 설치 — [Nix 설정](/getting-started/nix-setup#plugins) 참조 |

나중 출처는 이름 충돌 시 이전 출처를 덮어쓰므로 번들 플러그인과 동일한 이름의 사용자 플러그인이 이를 대체합니다.

### 플러그인 하위 카테고리

각 출처 내에서 Hermes는 플러그인을 특수 검색 시스템으로 라우팅하는 하위 카테고리 디렉토리도 인식합니다.

| 하위 디렉토리 | 내용 | 검색 시스템 |
|---|---|---|
| `plugins/` (root) | 일반 플러그인 — 도구, 훅, 슬래시 명령어, CLI 명령어, 번들 스킬 | `PluginManager` (종류: `standalone` 또는 `backend`) |
| `plugins/platforms/<name>/` | 게이트웨이 채널 어댑터 (`ctx.register_platform()`) | `PluginManager` (종류: `platform`, 한 단계 더 깊음) |
| `plugins/image_gen/<name>/` | 이미지 생성 백엔드 (`ctx.register_image_gen_provider()`) | `PluginManager` (종류: `backend`, 한 단계 더 깊음) |
| `plugins/memory/<name>/` | 메모리 공급자 (`MemoryProvider` 서브클래스) | `plugins/memory/__init__.py`의 **자체 로더** (종류: `exclusive` — 한 번에 하나만 활성화됨) |
| `plugins/context_engine/<name>/` | 컨텍스트 압축 엔진 (`ctx.register_context_engine()`) | `plugins/context_engine/__init__.py`의 **자체 로더** (한 번에 하나만 활성화됨) |
| `plugins/model-providers/<name>/` | LLM 공급자 프로필 (`register_provider(ProviderProfile(...))`) | `providers/__init__.py`의 **자체 로더** (첫 번째 `get_provider_profile()` 호출 시 지연 스캔됨) |

`~/.hermes/plugins/model-providers/<name>/` 및 `~/.hermes/plugins/memory/<name>/`의 사용자 플러그인은 동일한 이름의 번들 플러그인을 덮어씁니다 — `register_provider()` / `register_memory_provider()`에서 마지막 작성자가 승리합니다. 리포지토리 편집 없이 디렉토리를 놓기만 하면 내장 기능이 대체됩니다.

## 플러그인은 옵트인 방식입니다 (몇 가지 예외 제외)

**일반 플러그인 및 사용자 설치 백엔드는 기본적으로 비활성화되어 있습니다** — 검색을 통해 찾을 수는 있지만 (`hermes plugins` 및 `/plugins`에 표시되도록), `~/.hermes/config.yaml`의 `plugins.enabled`에 플러그인 이름을 추가할 때까지 훅이나 도구를 포함한 어떤 것도 로드되지 않습니다. 이는 타사 코드가 사용자의 명시적인 동의 없이 실행되는 것을 방지합니다.

```yaml
plugins:
  enabled:
    - my-tool-plugin
    - disk-cleanup
  disabled:       # 선택적 거부 목록 — 두 곳 모두에 이름이 나타나면 항상 승리합니다.
    - noisy-plugin
```

상태를 전환하는 세 가지 방법:

```bash
hermes plugins                    # 대화형 토글 (스페이스로 선택/해제)
hermes plugins enable <name>      # 허용 목록에 추가
hermes plugins disable <name>     # 허용 목록에서 제거 + 비활성화 목록에 추가
```

`hermes plugins install owner/repo`를 실행한 후 `Enable 'name' now? [y/N]`(지금 '이름'을 활성화하시겠습니까?)이라는 프롬프트가 표시되며 기본값은 no입니다. `--enable` 또는 `--no-enable`을 사용하여 스크립트 기반 설치의 프롬프트를 건너뛸 수 있습니다.

### 허용 목록이 제한하지 않는 것

몇 가지 플러그인 카테고리는 `plugins.enabled`를 무시합니다. 이는 Hermes 내장 기능의 일부이며 기본적으로 차단되면 기본 기능이 손상될 수 있습니다.

| 플러그인 종류 | 활성화 방법 |
|---|---|
| **번들 플랫폼 플러그인** (`plugins/platforms/` 아래의 IRC, Teams 등) | 함께 제공되는 모든 게이트웨이 채널을 사용할 수 있도록 자동 로드됩니다. 실제 채널은 `config.yaml`의 `gateway.platforms.<name>.enabled`를 통해 켜집니다. |
| **번들 백엔드** (`plugins/image_gen/` 아래의 이미지 생성 공급자 등) | 기본 백엔드가 "그냥 작동"하도록 자동 로드됩니다. 선택은 `config.yaml`의 `<category>.provider`를 통해 이루어집니다 (예: `image_gen.provider: openai`). |
| **메모리 공급자** (`plugins/memory/`) | 모두 검색됩니다; 정확히 하나만 활성화되며, `config.yaml`의 `memory.provider`에 의해 선택됩니다. |
| **컨텍스트 엔진** (`plugins/context_engine/`) | 모두 검색됩니다; 하나만 활성화되며, `config.yaml`의 `context.engine`에 의해 선택됩니다. |
| **모델 공급자** (`plugins/model-providers/`) | `plugins/model-providers/` 아래의 모든 번들 공급자는 첫 번째 `get_provider_profile()` 호출 시 검색되고 등록됩니다. 사용자는 `--provider` 또는 `config.yaml`을 통해 한 번에 하나씩 선택합니다. |
| **Pip를 통해 설치된 `backend` 플러그인** | `plugins.enabled`를 통해 옵트인 (일반 플러그인과 동일). |
| **사용자 설치 플랫폼** (`~/.hermes/plugins/platforms/` 아래) | `plugins.enabled`를 통해 옵트인 — 타사 게이트웨이 어댑터는 명시적인 동의가 필요합니다. |

요약하자면: **번들형 "항상 작동하는" 인프라는 자동으로 로드되며, 타사 일반 플러그인은 옵트인 방식입니다.** `plugins.enabled` 허용 목록은 사용자가 `~/.hermes/plugins/`에 넣는 임의의 코드에 대한 차단문(gate)입니다.

### 기존 사용자를 위한 마이그레이션

옵트인 플러그인이 있는 Hermes 버전(구성 스키마 v21 이상)으로 업그레이드하면, `~/.hermes/plugins/` 아래에 이미 설치되어 있고 `plugins.disabled`에 없는 모든 사용자 플러그인은 **자동으로 기존 권한이 인정되어(grandfathered)** `plugins.enabled`에 포함됩니다. 기존 설정은 계속 작동합니다. 번들형 독립 실행형 플러그인은 이전 권한이 인정되지 않으므로 기존 사용자도 명시적으로 옵트인해야 합니다. (번들 플랫폼/백엔드 플러그인은 애초에 차단된 적이 없으므로 권한 인정이 필요하지 않았습니다.)

## 사용 가능한 훅 (Available hooks)

플러그인은 이러한 수명 주기 이벤트에 대한 콜백을 등록할 수 있습니다. 자세한 내용, 콜백 서명 및 예시는 **[이벤트 훅 페이지(Event Hooks page)](/user-guide/features/hooks#plugin-hooks)**를 참조하세요.

| 훅 | 실행 시기 |
|------|-----------|
| [`pre_tool_call`](/user-guide/features/hooks#pre_tool_call) | 도구가 실행되기 전 |
| [`post_tool_call`](/user-guide/features/hooks#post_tool_call) | 도구가 반환된 후 |
| [`pre_llm_call`](/user-guide/features/hooks#pre_llm_call) | 턴당 한 번, LLM 루프 시작 전 — `{"context": "..."}`을 반환하여 [사용자 메시지에 컨텍스트 주입](/user-guide/features/hooks#pre_llm_call) 가능 |
| [`post_llm_call`](/user-guide/features/hooks#post_llm_call) | 턴당 한 번, LLM 루프 후 (성공한 턴만) |
| [`on_session_start`](/user-guide/features/hooks#on_session_start) | 새 세션 생성됨 (첫 번째 턴만) |
| [`on_session_end`](/user-guide/features/hooks#on_session_end) | 모든 `run_conversation` 호출 종료 시 + CLI 종료 핸들러 |
| [`on_session_finalize`](/user-guide/features/hooks#on_session_finalize) | CLI/게이트웨이가 활성 세션을 해제함 (`/new`, GC, CLI 종료) |
| [`on_session_reset`](/user-guide/features/hooks#on_session_reset) | 게이트웨이가 새 세션 키로 교체함 (`/new`, `/reset`, `/clear`, 유휴 교체) |
| [`subagent_stop`](/user-guide/features/hooks#subagent_stop) | `delegate_task`가 완료된 후 하위(child)당 한 번 |
| [`pre_gateway_dispatch`](/user-guide/features/hooks#pre_gateway_dispatch) | 인증 및 디스패치 전, 게이트웨이가 사용자 메시지를 수신함. 흐름에 영향을 주기 위해 `{"action": "skip" \| "rewrite" \| "allow", ...}` 반환 가능. |

## 플러그인 유형

Hermes에는 4가지 종류의 플러그인이 있습니다.

| 유형 | 역할 | 선택 방식 | 위치 |
|------|-------------|-----------|----------|
| **일반 플러그인** | 도구, 훅, 슬래시 명령어, CLI 명령어 추가 | 다중 선택 (활성화/비활성화) | `~/.hermes/plugins/` |
| **메모리 공급자** | 내장 메모리 교체 또는 보강 | 단일 선택 (하나만 활성) | `plugins/memory/` |
| **컨텍스트 엔진** | 내장 컨텍스트 압축기 교체 | 단일 선택 (하나만 활성) | `plugins/context_engine/` |
| **모델 공급자** | 추론 백엔드 선언 (OpenRouter, Anthropic 등) | 다중 등록, `--provider` / `config.yaml`에 의해 선택됨 | `plugins/model-providers/` |

메모리 공급자와 컨텍스트 엔진은 **공급자 플러그인(provider plugins)**입니다 — 한 번에 각 유형 중 하나만 활성화될 수 있습니다. 모델 공급자도 플러그인이지만 많은 모델이 동시에 로드됩니다. 사용자는 `--provider` 또는 `config.yaml`을 통해 한 번에 하나씩 선택합니다. 일반 플러그인은 어떤 조합으로든 활성화할 수 있습니다.

## 플러그 가능 인터페이스 — 각 기능별 가이드

위의 표에는 플러그인의 4가지 범주가 나와 있지만, "일반 플러그인" 내에서도 `PluginContext`는 여러 개의 고유한 확장 지점을 노출합니다. 또한 Hermes는 Python 플러그인 시스템(구성 기반 백엔드, 셸 후킹된 명령어, 외부 서버 등) 외부에서도 확장을 허용합니다. 구축하려는 기능에 맞는 문서를 찾으려면 이 표를 사용하세요.

| 추가하려는 기능… | 방법 | 작성 가이드 |
|---|---|---|
| LLM이 호출할 수 있는 **도구** | Python 플러그인 — `ctx.register_tool()` | [Hermes 플러그인 구축하기](/guides/build-a-hermes-plugin) · [도구 추가](/developer-guide/adding-tools) |
| **수명 주기 훅** (LLM 이전/이후, 세션 시작/종료, 도구 필터) | Python 플러그인 — `ctx.register_hook()` | [훅 참조](/user-guide/features/hooks) · [Hermes 플러그인 구축하기](/guides/build-a-hermes-plugin) |
| CLI / 게이트웨이용 **슬래시 명령어** | Python 플러그인 — `ctx.register_command()` | [Hermes 플러그인 구축하기](/guides/build-a-hermes-plugin) · [CLI 확장](/developer-guide/extending-the-cli) |
| `hermes <thing>`을 위한 **하위 명령어(subcommand)** | Python 플러그인 — `ctx.register_cli_command()` | [CLI 확장](/developer-guide/extending-the-cli) |
| 플러그인이 제공하는 **번들 스킬** | Python 플러그인 — `ctx.register_skill()` | [스킬 생성](/developer-guide/creating-skills) |
| **추론 백엔드** (LLM 공급자: OpenAI 호환, Codex, Anthropic-Messages, Bedrock) | 공급자 플러그인 — `plugins/model-providers/<name>/`의 `register_provider(ProviderProfile(...))` | **[모델 공급자 플러그인](/developer-guide/model-provider-plugin)** · [공급자 추가](/developer-guide/adding-providers) |
| **게이트웨이 채널** (Discord / Telegram / IRC / Teams 등) | 플랫폼 플러그인 — `plugins/platforms/<name>/`의 `ctx.register_platform()` | [플랫폼 어댑터 추가](/developer-guide/adding-platform-adapters) |
| **메모리 백엔드** (Honcho, Mem0, Supermemory 등) | 메모리 플러그인 — `plugins/memory/<name>/`에서 `MemoryProvider` 서브클래싱 | [메모리 공급자 플러그인](/developer-guide/memory-provider-plugin) |
| **컨텍스트 압축 전략** | 컨텍스트 엔진 플러그인 — `ctx.register_context_engine()` | [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin) |
| **이미지 생성 백엔드** (DALL·E, SDXL 등) | 백엔드 플러그인 — `ctx.register_image_gen_provider()` | [이미지 생성 공급자 플러그인](/developer-guide/image-gen-provider-plugin) |
| **비디오 생성 백엔드** (Veo, Kling, Pixverse, Grok-Imagine, Runway 등) | 백엔드 플러그인 — `ctx.register_video_gen_provider()` | [비디오 생성 공급자 플러그인](/developer-guide/video-gen-provider-plugin) |
| **TTS 백엔드** (모든 CLI — Piper, VoxCPM, Kokoro, xtts, 음성 복제 스크립트 등) | 구성 기반(권장) — `config.yaml`에 `type: command`와 함께 `tts.providers.<name>` 아래에 선언. 또는 Python 백엔드 플러그인 — 셸 템플릿 이상의 기능이 필요한 Python-SDK / 스트리밍 엔진의 경우 `ctx.register_tts_provider()`. | [TTS 설정](/user-guide/features/tts#custom-command-providers) · [Python 플러그인 가이드](/user-guide/features/tts#python-plugin-providers) |
| **STT 백엔드** (모든 CLI — whisper.cpp, 사용자 지정 whisper 바이너리, 로컬 ASR CLI) | 구성 기반(권장) — `config.yaml`에 `type: command`와 함께 `stt.providers.<name>` 아래에 선언하거나, 레거시 단일 명령 탈출구를 위해 `HERMES_LOCAL_STT_COMMAND` 설정. 또는 Python 백엔드 플러그인 — Python-SDK 엔진(OpenRouter, SenseAudio, Gemini-STT 등)의 경우 `ctx.register_transcription_provider()`. | [STT 설정](/user-guide/features/tts#stt-custom-command-providers) · [Python 플러그인 가이드](/user-guide/features/tts#python-plugin-providers-stt) |
| **MCP를 통한 외부 도구** (파일 시스템, GitHub, Linear, Notion, 모든 MCP 서버) | 구성 기반 — `config.yaml`에 `command:` / `url:`과 함께 `mcp_servers.<name>` 선언. Hermes는 서버의 도구를 자동 검색하여 기본 기능과 함께 등록합니다. | [MCP](/user-guide/features/mcp) |
| **추가 스킬 소스** (사용자 지정 GitHub 저장소, 프라이빗 스킬 인덱스) | CLI — `hermes skills tap add <repo>` | [스킬 허브](/user-guide/features/skills#skills-hub) · [사용자 지정 탭 게시하기](/user-guide/features/skills#publishing-a-custom-skill-tap) |
| **게이트웨이 이벤트 훅** (`gateway:startup`, `session:start`, `agent:end`, `command:*` 등에서 발생) | `HOOK.yaml` + `handler.py`를 `~/.hermes/hooks/<name>/`에 드롭 | [이벤트 훅](/user-guide/features/hooks#gateway-event-hooks) |
| **셸 훅** (이벤트 발생 시 셸 명령어 실행 — 알림, 감사 로그, 바탕화면 알림) | 구성 기반 — `config.yaml`의 `hooks:` 아래에 선언 | [셸 훅](/user-guide/features/hooks#shell-hooks) |

:::note
모든 것이 Python 플러그인은 아닙니다. 일부 확장 표면은 Python을 작성하지 않고도 기존 CLI를 플러그인으로 전환할 수 있도록 의도적으로 **구성 기반 셸 명령어**(TTS, STT, 셸 훅)를 사용합니다. 다른 확장은 에이전트가 연결하여 도구를 자동으로 등록하는 **외부 서버**(MCP)입니다. 그리고 일부는 자체 매니페스트 형식을 가진 **드롭인 디렉토리**(게이트웨이 훅)입니다. 사용 사례에 맞는 통합 스타일을 위해 적절한 표면을 선택하세요. 위 표의 작성 가이드에서는 각각의 플레이스홀더, 검색 및 예시를 다룹니다.
:::

## NixOS 선언적 플러그인

NixOS에서는 `hermes plugins install`이 필요 없이 모듈 옵션을 통해 선언적으로 플러그인을 설치할 수 있습니다. 자세한 내용은 **[Nix 설정 가이드](/getting-started/nix-setup#plugins)**를 참조하세요.

```nix
services.hermes-agent = {
  # 디렉토리 플러그인 (plugin.yaml이 있는 소스 트리)
  extraPlugins = [ (pkgs.fetchFromGitHub { ... }) ];
  # 진입점 플러그인 (pip 패키지)
  extraPythonPackages = [ (pkgs.python312Packages.buildPythonPackage { ... }) ];
  # 구성에서 활성화
  settings.plugins.enabled = [ "my-plugin" ];
};
```

선언적 플러그인은 `nix-managed-` 접두사와 함께 심볼릭 링크됩니다. 이들은 수동으로 설치된 플러그인과 공존하며 Nix 설정에서 제거될 때 자동으로 정리됩니다.

## 플러그인 관리

```bash
hermes plugins                               # 통합 대화형 UI
hermes plugins list                          # 표 형식: 활성화됨 / 비활성화됨 / 활성화되지 않음
hermes plugins install user/repo             # Git에서 설치 후 Enable? [y/N] 프롬프트 표시
hermes plugins install user/repo --enable    # 설치 및 활성화 (프롬프트 없음)
hermes plugins install user/repo --no-enable # 설치하지만 비활성 상태 유지 (프롬프트 없음)
hermes plugins update my-plugin              # 최신 버전 가져오기
hermes plugins remove my-plugin              # 제거
hermes plugins enable my-plugin              # 허용 목록에 추가
hermes plugins disable my-plugin             # 허용 목록에서 제거 + 비활성화 목록에 추가
```

### 대화형 UI

인수 없이 `hermes plugins`를 실행하면 복합 대화형 화면이 열립니다.

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

- **일반 플러그인(General Plugins) 섹션** — 확인란, 스페이스바를 눌러 토글. 선택됨 = `plugins.enabled`에 포함, 선택 해제됨 = `plugins.disabled`에 포함 (명시적 비활성화).
- **공급자 플러그인(Provider Plugins) 섹션** — 현재 선택을 보여줍니다. Enter를 누르면 하나의 활성 공급자를 선택하는 라디오 선택기(radio picker)로 들어갑니다.
- 번들 플러그인은 `[bundled]` 태그와 함께 동일한 목록에 표시됩니다.

공급자 플러그인 선택 항목은 `config.yaml`에 저장됩니다.

```yaml
memory:
  provider: "honcho"      # 빈 문자열 = 내장 기능만

context:
  engine: "compressor"    # 기본 내장 압축기
```

### 활성화됨(Enabled) vs 비활성화됨(Disabled) vs 둘 다 아님

플러그인은 세 가지 상태 중 하나입니다.

| 상태 | 의미 | `plugins.enabled`에 있나요? | `plugins.disabled`에 있나요? |
|---|---|---|---|
| `enabled` | 다음 세션에 로드됨 | 예 | 아니요 |
| `disabled` | 명시적으로 꺼짐 — `enabled`에 있더라도 로드되지 않음 | (무관함) | 예 |
| `not enabled` | 발견되었지만 한 번도 옵트인 되지 않음 | 아니요 | 아니요 |

새로 설치되거나 번들로 제공되는 플러그인의 기본값은 `not enabled`(활성화되지 않음)입니다. `hermes plugins list`는 명시적으로 꺼져 있는 것과 단순히 활성화 대기 중인 것을 구분할 수 있도록 3가지 상태를 모두 보여줍니다.

실행 중인 세션에서 `/plugins`는 현재 로드된 플러그인을 보여줍니다.

## 메시지 주입 (Injecting Messages)

플러그인은 `ctx.inject_message()`를 사용하여 활성 대화에 메시지를 주입할 수 있습니다.

```python
ctx.inject_message("웹훅에서 새 데이터가 도착했습니다", role="user")
```

**서명:** `ctx.inject_message(content: str, role: str = "user") -> bool`

작동 방식:

- 에이전트가 **유휴 상태(idle)**인 경우(사용자 입력을 기다리는 경우) 메시지가 다음 입력으로 큐에 대기하고 새 턴을 시작합니다.
- 에이전트가 **턴 중간(mid-turn)**인 경우(적극적으로 실행 중) 사용자가 새 메시지를 입력하고 Enter 키를 누르는 것과 동일하게 현재 작업을 중단합니다.
- `"user"`가 아닌 역할의 경우 콘텐츠 앞에 `[role]`(예: `[system] ...`)이 추가됩니다.
- 메시지가 큐에 성공적으로 추가되면 `True`를 반환하고, CLI 참조를 사용할 수 없는 경우(예: 게이트웨이 모드) `False`를 반환합니다.

이 기능을 통해 원격 제어 뷰어, 메시징 브릿지 또는 웹훅 수신기와 같은 플러그인이 외부 소스의 메시지를 대화에 공급할 수 있습니다.

:::note
`inject_message`는 CLI 모드에서만 사용할 수 있습니다. 게이트웨이 모드에서는 CLI 참조가 없으며 메서드는 `False`를 반환합니다.
:::

핸들러 계약, 스키마 형식, 훅 동작, 오류 처리 및 일반적인 실수에 대해서는 **[전체 가이드](/guides/build-a-hermes-plugin)**를 참조하세요.
