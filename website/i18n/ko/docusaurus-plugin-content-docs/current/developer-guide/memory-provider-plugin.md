---
sidebar_position: 8
title: "Memory Provider Plugins"
description: "Hermes 에이전트용 메모리 프로바이더 플러그인을 빌드하는 방법"
---

# 메모리 프로바이더 플러그인 빌드하기

메모리 프로바이더 플러그인은 내장된 MEMORY.md 및 USER.md를 넘어 Hermes 에이전트에게 지속적이고 세션 간 공유되는 지식을 제공합니다. 이 가이드는 이를 빌드하는 방법을 다룹니다.

:::tip
메모리 프로바이더는 두 가지 **프로바이더 플러그인** 유형 중 하나입니다. 다른 하나는 내장된 컨텍스트 압축기를 대체하는 [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin)입니다. 둘 다 동일한 패턴(단일 선택, 구성 기반, `hermes plugins`를 통한 관리)을 따릅니다.
:::

## 디렉토리 구조

각 메모리 프로바이더는 `plugins/memory/<name>/`에 위치합니다:

```
plugins/memory/my-provider/
├── __init__.py      # MemoryProvider 구현 + register() 진입점
├── plugin.yaml      # 메타데이터 (이름, 설명, 훅)
└── README.md        # 설정 지침, 설정 참조, 도구
```

## MemoryProvider ABC

여러분의 플러그인은 `agent/memory_provider.py`에 있는 `MemoryProvider` 추상 기본 클래스(ABC)를 구현합니다:

```python
from agent.memory_provider import MemoryProvider

class MyMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "my-provider"

    def is_available(self) -> bool:
        """이 프로바이더가 활성화될 수 있는지 확인합니다. 네트워크 호출은 하지 않습니다."""
        return bool(os.environ.get("MY_API_KEY"))

    def initialize(self, session_id: str, **kwargs) -> None:
        """에이전트 시작 시 한 번 호출됩니다.

        kwargs에는 항상 다음이 포함됩니다:
          hermes_home (str): 활성 HERMES_HOME 경로. 저장용으로 사용합니다.
        """
        self._api_key = os.environ.get("MY_API_KEY", "")
        self._session_id = session_id

    # ... 나머지 메서드 구현
```

## 필수 메서드

### 핵심 라이프사이클

| 메서드 | 호출 시점 | 구현 필수? |
|--------|-----------|-----------------|
| `name` (property) | 항상 | **예** |
| `is_available()` | 에이전트 초기화, 활성화 전 | **예** — 네트워크 호출 없음 |
| `initialize(session_id, **kwargs)` | 에이전트 시작 시 | **예** |
| `get_tool_schemas()` | 초기화 후, 도구 주입 시 | **예** |
| `handle_tool_call(tool_name, args, **kwargs)` | 에이전트가 여러분의 도구를 사용할 때 | **예** (도구가 있는 경우) |

### 구성(Config)

| 메서드 | 목적 | 구현 필수? |
|--------|---------|-----------------|
| `get_config_schema()` | `hermes memory setup`에 대한 구성 필드 선언 | **예** |
| `save_config(values, hermes_home)` | 비밀이 아닌 구성을 네이티브 위치에 저장 | **예** (환경 변수 전용인 경우 제외) |

### 선택적 훅(Optional Hooks)

| 메서드 | 호출 시점 | 사용 사례 |
|--------|-----------|----------|
| `system_prompt_block()` | 시스템 프롬프트 조립 시 | 정적 프로바이더 정보 |
| `prefetch(query, *, session_id="")` | 각 API 호출 전 | 리콜된 컨텍스트 반환 |
| `queue_prefetch(query)` | 각 턴 후 | 다음 턴을 위한 예열(Pre-warm) |
| `sync_turn(user, assistant, *, session_id="")` | 각 완료된 턴 후 | 대화 저장(Persist) |
| `on_session_end(messages)` | 대화 종료 시 | 최종 추출/플러시 |
| `on_pre_compress(messages)` | 컨텍스트 압축 전 | 폐기 전 인사이트 저장 |
| `on_memory_write(action, target, content)` | 내장 메모리 쓰기 시 | 백엔드에 미러링 |
| `shutdown()` | 프로세스 종료 시 | 연결 정리 |

## 구성 스키마 (Config Schema)

`get_config_schema()`는 `hermes memory setup`에서 사용하는 필드 디스크립터 목록을 반환합니다:

```python
def get_config_schema(self):
    return [
        {
            "key": "api_key",
            "description": "My Provider API key",
            "secret": True,           # → .env에 작성됨
            "required": True,
            "env_var": "MY_API_KEY",   # 명시적 환경 변수 이름
            "url": "https://my-provider.com/keys",  # 키를 얻는 곳
        },
        {
            "key": "region",
            "description": "Server region",
            "default": "us-east",
            "choices": ["us-east", "eu-west", "ap-south"],
        },
        {
            "key": "project",
            "description": "Project identifier",
            "default": "hermes",
        },
    ]
```

`secret: True`와 `env_var`가 있는 필드는 `.env`로 들어갑니다. 비밀이 아닌 필드는 `save_config()`로 전달됩니다.

:::tip 최소 스키마 vs 전체 스키마
`get_config_schema()`의 모든 필드는 `hermes memory setup` 중 입력을 요청받습니다. 옵션이 많은 프로바이더는 스키마를 최소한으로 유지해야 합니다 — 사용자가 **반드시** 구성해야 하는 필드(API 키, 필수 자격 증명)만 포함하세요. 설정 마법사를 빠르게 유지하면서도 고급 구성을 지원하기 위해 설정 파일 참조(예: `$HERMES_HOME/myprovider.json`)에 선택적 설정을 문서화하세요. 예시로 Supermemory 프로바이더를 참고하세요. API 키만 묻고 다른 모든 옵션은 `supermemory.json`에 있습니다.
:::

## 구성 저장 (Save Config)

```python
def save_config(self, values: dict, hermes_home: str) -> None:
    """비밀이 아닌 구성을 네이티브 위치에 작성합니다."""
    import json
    from pathlib import Path
    config_path = Path(hermes_home) / "my-provider.json"
    config_path.write_text(json.dumps(values, indent=2))
```

환경 변수 전용 프로바이더의 경우, 아무것도 하지 않는 기본값(no-op)으로 둡니다.

## 플러그인 진입점 (Plugin Entry Point)

```python
def register(ctx) -> None:
    """메모리 플러그인 검색 시스템에 의해 호출됩니다."""
    ctx.register_memory_provider(MyMemoryProvider())
```

## plugin.yaml

```yaml
name: my-provider
version: 1.0.0
description: "이 프로바이더가 수행하는 작업에 대한 짧은 설명입니다."
hooks:
  - on_session_end    # 구현하는 훅 목록
```

## 스레딩 계약 (Threading Contract)

**`sync_turn()`은 반드시 논블로킹(non-blocking)이어야 합니다.** 백엔드에 지연(API 호출, LLM 처리)이 있는 경우 백그라운드 데몬 스레드에서 작업을 실행하세요:

```python
def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
    def _sync():
        try:
            self._api.ingest(user_content, assistant_content, session_id=session_id, messages=messages)
        except Exception as e:
            logger.warning("Sync failed: %s", e)

    if self._sync_thread and self._sync_thread.is_alive():
        self._sync_thread.join(timeout=5.0)
    self._sync_thread = threading.Thread(target=_sync, daemon=True)
    self._sync_thread.start()
```

`messages`는 완료된 턴 시점의 선택적인 OpenAI 스타일 대화 컨텍스트입니다. 존재하는 경우 사용자/어시스턴트 메시지, 어시스턴트 도구 호출 및 도구 결과 메시지가 포함됩니다. 원시 턴 컨텍스트가 필요하지 않은 프로바이더는 `messages` 매개변수를 생략할 수 있습니다; Hermes는 이전 서명(legacy signature)으로 계속 호출할 것입니다.

클라우드 프로바이더는 `messages`의 어떤 부분이 기기 외부에 전송되는지 문서화해야 합니다. 도구 호출 및 도구 결과에는 파일 경로, 명령어 출력 또는 기타 작업 공간 데이터가 포함될 수 있습니다.

## 프로필 격리 (Profile Isolation)

모든 저장소 경로는 하드코딩된 `~/.hermes`가 아닌 `initialize()`의 `hermes_home` 키워드 인자(kwarg)를 **반드시** 사용해야 합니다:

```python
# 올바름 — 프로필 범위 지정
from hermes_constants import get_hermes_home
data_dir = get_hermes_home() / "my-provider"

# 틀림 — 모든 프로필에서 공유됨
data_dir = Path("~/.hermes/my-provider").expanduser()
```

## 테스트

종단 간(End-to-End) 패턴에 대해서는 `tests/agent/test_memory_provider.py` 및 인접한 메모리 테스트(`tests/agent/test_memory_session_switch.py`, `tests/agent/test_memory_user_id.py`, `tests/run_agent/test_memory_provider_init.py`)를 참조하세요.

```python
from agent.memory_manager import MemoryManager

mgr = MemoryManager()
mgr.add_provider(my_provider)
mgr.initialize_all(session_id="test-1", platform="cli")

# 도구 라우팅 테스트
result = mgr.handle_tool_call("my_tool", {"action": "add", "content": "test"})

# 라이프사이클 테스트
mgr.sync_all("user msg", "assistant msg")
mgr.on_session_end([])
mgr.shutdown_all()
```

## CLI 명령어 추가하기

메모리 프로바이더 플러그인은 고유한 CLI 하위 명령 트리(예: `hermes my-provider status`, `hermes my-provider config`)를 등록할 수 있습니다. 이는 명명 규칙 기반의 검색 시스템을 사용합니다 — 핵심(core) 파일을 수정할 필요가 없습니다.

### 작동 방식

1. 플러그인 디렉토리에 `cli.py` 파일을 추가합니다.
2. argparse 트리를 빌드하는 `register_cli(subparser)` 함수를 정의합니다.
3. 메모리 플러그인 시스템은 시작 시 `discover_plugin_cli_commands()`를 통해 이를 검색합니다.
4. 명령어가 `hermes <provider-name> <subcommand>` 아래에 나타납니다.

**활성 프로바이더 제한:** CLI 명령어는 여러분의 프로바이더가 설정의 활성 `memory.provider`일 때만 나타납니다. 사용자가 여러분의 프로바이더를 구성하지 않았다면, `hermes --help`에 명령어가 표시되지 않습니다.

### 예시

```python
# plugins/memory/my-provider/cli.py

def my_command(args):
    """argparse에 의해 디스패치되는 핸들러."""
    sub = getattr(args, "my_command", None)
    if sub == "status":
        print("Provider is active and connected.")
    elif sub == "config":
        print("Showing config...")
    else:
        print("Usage: hermes my-provider <status|config>")

def register_cli(subparser) -> None:
    """hermes my-provider argparse 트리를 빌드합니다.

    argparse 설정 시 discover_plugin_cli_commands()에 의해 호출됩니다.
    """
    subs = subparser.add_subparsers(dest="my_command")
    subs.add_parser("status", help="Show provider status")
    subs.add_parser("config", help="Show provider config")
    subparser.set_defaults(func=my_command)
```

### 참조 구현

13개의 하위 명령, 교차 프로필 관리(`--target-profile`) 및 구성 읽기/쓰기가 있는 전체 예제는 `plugins/memory/honcho/cli.py`를 참조하세요.

### CLI 포함 디렉토리 구조

```
plugins/memory/my-provider/
├── __init__.py      # MemoryProvider 구현 + register()
├── plugin.yaml      # 메타데이터
├── cli.py           # register_cli(subparser) — CLI 명령어
└── README.md        # 설정 지침
```

## 단일 프로바이더 규칙

한 번에 **단 하나**의 외부 메모리 프로바이더만 활성화될 수 있습니다. 사용자가 두 번째 프로바이더를 등록하려고 하면, MemoryManager가 경고와 함께 이를 거부합니다. 이는 도구 스키마 비대화 및 충돌하는 백엔드를 방지합니다.
