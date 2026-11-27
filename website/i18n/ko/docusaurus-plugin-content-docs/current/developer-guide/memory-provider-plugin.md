---
sidebar_position: 8
title: "메모리 제공자 플러그인"
description: "Hermes Agent용 메모리 제공자 플러그인을 만드는 방법"
---

# 메모리 제공자 플러그인 만들기

메모리 제공자 플러그인은 기본 제공되는 MEMORY.md와 USER.md를 넘어, 세션 간에 유지되는 지식을 Hermes Agent에 제공합니다. 이 가이드에서는 메모리 제공자를 만드는 방법을 설명합니다.

:::tip
메모리 제공자는 두 가지 **제공자 플러그인** 유형 중 하나입니다. 다른 하나는 기본 제공 컨텍스트 압축기를 대체하는 [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin)입니다. 둘 다 단일 선택 방식이며 설정으로 구동되고 `hermes plugins`로 관리된다는 동일한 패턴을 따릅니다.
:::

## 설치 레이아웃

Hermes는 다음 네 가지 소스에서 메모리 제공자를 검색하며, 우선순위는 다음과 같습니다.

| 소스 | 위치 | 비고 |
|---|---|---|
| 번들 | `plugins/memory/<name>/` | Hermes와 함께 제공됩니다. 새 제공자는 받지 않습니다 — [CONTRIBUTING](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)을 참조하세요. |
| 사용자 | `$HERMES_HOME/plugins/<name>/` | 프로필별로 사용자가 추가합니다. |
| 프로젝트 | `./.hermes/plugins/<name>/` | `HERMES_ENABLE_PROJECT_PLUGINS=1`을 통해 선택적으로 활성화합니다. |
| 패키지 | `hermes_agent.memory_providers` 엔트리 포인트 | `pip install`만 하면 되며, 복사할 필요가 없습니다. |

이름이 충돌하면 앞선 소스가 우선하므로, 작업 트리에 추가한 디렉터리가 배포된 제공자를 가릴 수 없습니다.

:::note
이는 일반 플러그인 시스템의 나중 우선 순서와 반대입니다. 메모리 제공자는 *이름*(`memory.provider`)으로 활성화되므로, 가려지면 단순히 도구를 재정의하는 것이 아니라 에이전트의 메모리를 조용히 다른 곳으로 전환하게 됩니다.
:::

검색은 *열거*만 수행하며 제공자를 가져오지는 않습니다. `memory.provider`가 해당 제공자의 이름을 지정하기 전까지는 아무것도 실행되지 않습니다.

### 디렉터리 제공자

디렉터리 제공자는 Hermes에 번들로 포함될 때 `plugins/memory/<name>/`에, 사용자가 설치할 때 `$HERMES_HOME/plugins/<name>/`에, 프로젝트 로컬 제공자일 때 `./.hermes/plugins/<name>/`에 둡니다.

```
plugins/memory/my-provider/
├── __init__.py      # MemoryProvider implementation + register() entry point
├── plugin.yaml      # Metadata (name, description, hooks)
└── README.md        # Setup instructions, config reference, tools
```

### 패키지 제공자

pip로 설치되는 제공자는 `hermes_agent.memory_providers` 그룹에 엔트리 포인트를 게시합니다. 엔트리 포인트 이름은 사용자가 `memory.provider`에서 선택하는 제공자 이름이며, 값은 제공자의 `register(ctx)` 함수를 가리킵니다.

```toml title="pyproject.toml"
[project.entry-points."hermes_agent.memory_providers"]
my-provider = "my_provider:register"
```

엔트리 포인트는 **패키지** 또는 패키지 내부의 `register(ctx)`를 가리키도록 하고, 구현, 스킬 및 기타 리소스는 일반적인 Python 패키지 레이아웃에 보관하세요. `$HERMES_HOME/plugins/` 아래에 복사할 필요가 없습니다.

패키지 엔트리 포인트는 디렉터리 설치와 동일한 모든 것을 제공합니다. 여기에는 Hermes가 가져오기 대신 디스크에서 읽는 두 파일인 `config_schema.py`(대시보드 설정 패널)와 `cli.py`(사용자의 `hermes <provider>` 하위 명령)도 포함됩니다. 둘 다 패키지의 `__init__.py` 옆에서 검색되므로, 둘 중 하나를 제공한다면 단일 모듈이 아니라 패키지를 가리키세요.

## MemoryProvider ABC

플러그인은 `agent/memory_provider.py`의 `MemoryProvider` 추상 기본 클래스를 구현합니다.

```python
from agent.memory_provider import MemoryProvider

class MyMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "my-provider"

    def is_available(self) -> bool:
        """Check if this provider can activate. NO network calls."""
        return bool(os.environ.get("MY_API_KEY"))

    def initialize(self, session_id: str, **kwargs) -> None:
        """Called once at agent startup.

        kwargs always includes:
          hermes_home (str): Active HERMES_HOME path. Use for storage.
        """
        self._api_key = os.environ.get("MY_API_KEY", "")
        self._session_id = session_id

    # ... implement remaining methods
```

## 필수 메서드

### 핵심 수명 주기

| 메서드 | 호출 시점 | 구현해야 하나요? |
|--------|-----------|-----------------|
| `name` (property) | 항상 | **예** |
| `is_available()` | 에이전트 초기화, 활성화 전 | **예** — 네트워크 호출 금지 |
| `initialize(session_id, **kwargs)` | 에이전트 시작 | **예** |
| `get_tool_schemas()` | 초기화 후, 도구 주입 시 | **예** |
| `handle_tool_call(tool_name, args, **kwargs)` | 에이전트가 도구를 사용할 때 | **예** (도구가 있는 경우) |

### 설정

| 메서드 | 목적 | 구현해야 하나요? |
|--------|------|-----------------|
| `get_config_schema()` | `hermes memory setup`용 설정 필드 선언 | **예** |
| `save_config(values, hermes_home)` | 비밀이 아닌 설정을 기본 위치에 기록 | **예** (환경 변수만 사용하는 경우 제외) |

### 선택적 훅

| 메서드 | 호출 시점 | 사용 사례 |
|--------|-----------|----------|
| `system_prompt_block()` | 시스템 프롬프트 조합 | 정적 제공자 정보 |
| `prefetch(query, *, session_id="")` | 각 API 호출 전 | 회상한 컨텍스트 반환 |
| `queue_prefetch(query, *, session_id="")` | 각 턴 후 | 다음 턴을 위한 사전 준비 |
| `sync_turn(user, assistant, *, session_id="", messages=None)` | 각 턴 완료 후 | 대화 저장 |
| `on_session_end(messages)` | 대화 종료 시 | 최종 추출/플러시 |
| `on_pre_compress(messages)` | 컨텍스트 압축 전 | 버리기 전에 인사이트 저장 |
| `on_memory_write(action, target, content)` | 기본 제공 메모리 쓰기 시 | 백엔드에 미러링 |
| `shutdown()` | 프로세스 종료 | 연결 정리 |

## 설정 스키마

`get_config_schema()`는 `hermes memory setup`에서 사용하는 필드 설명자 목록을 반환합니다.

```python
def get_config_schema(self):
    return [
        {
            "key": "api_key",
            "description": "My Provider API key",
            "secret": True,           # → written to .env
            "required": True,
            "env_var": "MY_API_KEY",   # explicit env var name
            "url": "https://my-provider.com/keys",  # where to get it
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

`secret: True`와 `env_var`가 있는 필드는 `.env`로 이동합니다. 비밀이 아닌 필드는 `save_config()`에 전달됩니다.

:::tip 최소 스키마와 전체 스키마
`get_config_schema()`의 모든 필드는 `hermes memory setup` 중에 사용자에게 묻습니다. 옵션이 많은 제공자는 스키마를 최소화해야 합니다 — 사용자가 **반드시** 설정해야 하는 필드(API 키, 필수 자격 증명)만 포함하세요. 선택적 설정은 설정 파일 참조(예: `$HERMES_HOME/myprovider.json`)에 문서화하여 설정 중에 모두 묻지 않도록 하세요. 이렇게 하면 고급 설정을 지원하면서도 설정 마법사를 빠르게 유지할 수 있습니다. 예시는 Supermemory 제공자를 참조하세요 — 이 제공자는 API 키만 묻고, 다른 모든 옵션은 `supermemory.json`에 둡니다.
:::

## 설정 저장

```python
def save_config(self, values: dict, hermes_home: str) -> None:
    """Write non-secret config to your native location."""
    import json
    from pathlib import Path
    config_path = Path(hermes_home) / "my-provider.json"
    config_path.write_text(json.dumps(values, indent=2))
```

환경 변수만 사용하는 제공자는 기본 no-op을 그대로 두면 됩니다.

## 플러그인 엔트리 포인트

```python
def register(ctx) -> None:
    """Called by the memory plugin discovery system."""
    ctx.register_memory_provider(MyMemoryProvider())
```

제공자는 같은 콜백에서 읽기 전용 스킬을 노출할 수도 있습니다. 스킬은 엔트리 포인트 이름으로 한정되며, 해당 메모리 제공자가 활성화된 경우에만 로드됩니다.

```python
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"

def register(ctx) -> None:
    ctx.register_memory_provider(MyMemoryProvider())
    ctx.register_skill(
        "maintenance",
        SKILLS_DIR / "maintenance" / "SKILL.md",
        "Maintain the provider's memory store",
    )
```

`my-provider` 엔트리 포인트가 활성화되면 `skill_view()`를 통해 `my-provider:maintenance`로 스킬을 사용할 수 있습니다.

## plugin.yaml

```yaml
name: my-provider
version: 1.0.0
description: "Short description of what this provider does."
hooks:
  - on_session_end    # list hooks you implement
```

## 스레딩 계약

**`sync_turn()`은 블로킹되지 않아야 합니다.** 백엔드에 지연 시간(API 호출, LLM 처리)이 있다면 데몬 스레드에서 작업을 실행하세요.

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

`messages`는 완료된 턴 시점의 선택적 OpenAI 형식 대화 컨텍스트입니다. 제공되면 사용자/어시스턴트 메시지, 어시스턴트 도구 호출, 도구 결과 메시지가 포함됩니다. 원시 턴 컨텍스트가 필요하지 않은 제공자는 `messages` 매개변수를 생략할 수 있으며, Hermes는 계속 기존 시그니처로 호출합니다.

클라우드 제공자는 `messages`의 어떤 부분이 디바이스 밖으로 전송되는지 문서화해야 합니다. 도구 호출과 도구 결과에는 파일 경로, 명령 출력 또는 기타 작업 공간 데이터가 포함될 수 있습니다.

## 프로필 격리

모든 저장 경로는 하드코딩된 `~/.hermes`가 아니라 `initialize()`의 `hermes_home` kwarg를 사용해야 합니다.

```python
# CORRECT — profile-scoped
from hermes_constants import get_hermes_home
data_dir = get_hermes_home() / "my-provider"

# WRONG — shared across all profiles
data_dir = Path("~/.hermes/my-provider").expanduser()
```

## 테스트

엔드투엔드 패턴은 `tests/agent/test_memory_provider.py`와 인접한 메모리 테스트(`tests/agent/test_memory_session_switch.py`, `tests/agent/test_memory_user_id.py`, `tests/run_agent/test_memory_provider_init.py`)를 참조하세요.

```python
from agent.memory_manager import MemoryManager

mgr = MemoryManager()
mgr.add_provider(my_provider)
mgr.initialize_all(session_id="test-1", platform="cli")

# Test tool routing
result = mgr.handle_tool_call("my_tool", {"action": "add", "content": "test"})

# Test lifecycle
mgr.sync_all("user msg", "assistant msg")
mgr.on_session_end([])
mgr.shutdown_all()
```

## CLI 명령 추가

메모리 제공자 플러그인은 자체 CLI 하위 명령 트리(예: `hermes my-provider status`, `hermes my-provider config`)를 등록할 수 있습니다. 이는 관례 기반 검색 시스템을 사용하므로 코어 파일을 변경할 필요가 없습니다.

### 작동 방식

1. 플러그인 디렉터리에 `cli.py` 파일 추가
2. argparse 트리를 구성하는 `register_cli(subparser)` 함수 정의
3. 메모리 플러그인 시스템이 시작 시 `discover_plugin_cli_commands()`를 통해 해당 파일 검색
4. 명령이 `hermes <provider-name> <subcommand>` 아래에 표시

**활성 제공자 게이팅:** CLI 명령은 설정에서 제공자가 활성 `memory.provider`일 때만 표시됩니다. 사용자가 제공자를 설정하지 않았다면 해당 명령이 `hermes --help`에 나타나지 않습니다.

### 예시

```python
# plugins/memory/my-provider/cli.py

def my_command(args):
    """Handler dispatched by argparse."""
    sub = getattr(args, "my_command", None)
    if sub == "status":
        print("Provider is active and connected.")
    elif sub == "config":
        print("Showing config...")
    else:
        print("Usage: hermes my-provider <status|config>")

def register_cli(subparser) -> None:
    """Build the hermes my-provider argparse tree.

    Called by discover_plugin_cli_commands() at argparse setup time.
    """
    subs = subparser.add_subparsers(dest="my_command")
    subs.add_parser("status", help="Show provider status")
    subs.add_parser("config", help="Show provider config")
    subparser.set_defaults(func=my_command)
```

### 참조 구현

13개 하위 명령, 프로필 간 관리(`--target-profile`), 설정 읽기/쓰기를 포함한 전체 예시는 `plugins/memory/honcho/cli.py`를 참조하세요.

### CLI가 포함된 디렉터리 구조

```
plugins/memory/my-provider/
├── __init__.py      # MemoryProvider implementation + register()
├── plugin.yaml      # Metadata
├── cli.py           # register_cli(subparser) — CLI commands
└── README.md        # Setup instructions
```

## 단일 제공자 규칙

외부 메모리 제공자는 한 번에 **하나만** 활성화할 수 있습니다. 사용자가 두 번째 제공자를 등록하려고 하면 MemoryManager가 경고와 함께 거부합니다. 이를 통해 도구 스키마가 비대해지거나 백엔드가 충돌하는 것을 방지합니다.
