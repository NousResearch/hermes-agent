---
sidebar_position: 9
---

# 플랫폼 어댑터 추가

이 가이드는 Hermes 게이트웨이에 새로운 메시징 플랫폼을 추가하는 방법을 설명합니다. 플랫폼 어댑터는 Hermes를 외부 메시징 서비스(Telegram, Discord, WeCom 등)에 연결하여 사용자가 해당 서비스를 통해 에이전트와 상호작용할 수 있게 합니다.

:::tip
플랫폼을 추가하는 방법은 두 가지입니다:
- **플러그인** (커뮤니티/서드파티에 권장): 플러그인 디렉터리를 `~/.hermes/plugins/`에 넣기만 하면 됩니다 — 코어 코드를 변경할 필요가 없습니다. 아래의 [플러그인 경로](#plugin-path-recommended)를 참고하세요.
- **내장 방식**: 코드, 설정, 문서 전반에 걸쳐 20개 이상의 파일을 수정합니다. 아래의 [내장 방식 단계별 체크리스트](#step-by-step-checklist-built-in-path)를 사용하세요.
:::

## 아키텍처 개요

```
User ↔ Messaging Platform ↔ Platform Adapter ↔ Gateway Runner ↔ AIAgent
```

모든 어댑터는 `gateway/platforms/base.py`의 `BasePlatformAdapter`를 확장하고 다음을 구현합니다:

- **`connect()`** — 연결 수립(WebSocket, long-poll, HTTP 서버 등) *(추상 메서드)*
- **`disconnect()`** — 정상 종료 *(추상 메서드)*
- **`send()`** — 채팅으로 텍스트 메시지 전송 *(추상 메서드)*
- **`send_typing()`** — 입력 중 표시(선택적 재정의)
- **`get_chat_info()`** — 채팅 메타데이터 반환(선택적 재정의)

인바운드 메시지는 어댑터가 수신한 뒤 `self.handle_message(event)`를 통해 전달하며, 기본 클래스가 이를 게이트웨이 실행기로 라우팅합니다.

## 플러그인 경로(권장)

플러그인 시스템을 사용하면 Hermes 코어 코드를 수정하지 않고도 플랫폼 어댑터를 추가할 수 있습니다. 플러그인은 다음 두 파일로 구성된 디렉터리입니다:

```
~/.hermes/plugins/my-platform/
  plugin.yaml      # Plugin metadata
  adapter.py       # Adapter class + register() entry point
```

### plugin.yaml

플러그인 메타데이터입니다. `requires_env` 및 `optional_env` 블록은 `hermes config` UI 항목을 자동으로 채웁니다(아래 [환경 변수 Hermes config에 표시](#surfacing-env-vars-in-hermes-config) 참고).

```yaml
name: my-platform
label: My Platform
kind: platform
version: 1.0.0
description: My custom messaging platform adapter
author: Your Name
requires_env:
  - MY_PLATFORM_TOKEN          # bare string works
  - name: MY_PLATFORM_CHANNEL  # or rich dict for better UX
    description: "Channel to join"
    prompt: "Channel"
    password: false
optional_env:
  - name: MY_PLATFORM_HOME_CHANNEL
    description: "Default channel for cron delivery"
    password: false
```

#### 아웃바운드 클라이언트 도구: `provides_tools`

`kind: platform` 플러그인은 **지연 로드**됩니다. 게이트웨이, cron 또는 `send_message` 경로가 먼저 플랫폼 레지스트리에 플랫폼을 요청할 때만 어댑터 모듈(및 해당 SDK import)이 로드됩니다. 플러그인이 모든 세션에서 에이전트가 호출할 수 있어야 하는 아웃바운드 *클라이언트 도구*도 제공한다면(번들된 `a2a` 플러그인의 `a2a_call` / `a2a_discover` 등), 전용 `tools.py`에 `register_tools(ctx)` 함수를 넣고 매니페스트에 선언하세요:

```yaml
provides_tools:
  - my_platform_call
  - my_platform_list
```

`provides_tools`를 선언하면 Hermes는 플러그인 탐색 중 `tools.py`만 import하고 모든 프로세스(CLI와 TUI 포함)에 클라이언트 도구를 등록합니다. 반면 어댑터는 계속 지연 로드됩니다. 패키지의 `__init__.py`는 import를 가볍게 유지하고, eager import 비용을 낮게 유지할 수 있도록 `register()` 내부에서 어댑터를 가져오세요. 이 필드가 없으면 아무것도 바뀌지 않습니다. 플러그인 전체가 지연 로드 상태로 유지됩니다.

사용자는 다른 도구와 마찬가지로 플랫폼별로 도구 세트를 활성화합니다. 예를 들어 `hermes tools enable my_platform --platform cli`를 실행하거나 `config.yaml`의 `platform_toolsets` 아래에 도구 세트 키를 나열하면 됩니다. 플러그인 플랫폼 이름도 유효한 `--platform` 대상이므로, 해당 플랫폼의 인바운드 세션에 자체 아웃바운드 도구를 부여할 수 있습니다.

### adapter.py

```python
import os
from gateway.platforms.base import (
    BasePlatformAdapter, SendResult, MessageEvent, MessageType,
)
from gateway.config import Platform, PlatformConfig


class MyPlatformAdapter(BasePlatformAdapter):
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("my_platform"))
        extra = config.extra or {}
        self.token = os.getenv("MY_PLATFORM_TOKEN") or extra.get("token", "")

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        # Connect to the platform API, start listeners
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        # Send message via platform API
        return SendResult(success=True, message_id="...")

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}


def check_requirements() -> bool:
    return bool(os.getenv("MY_PLATFORM_TOKEN"))


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(os.getenv("MY_PLATFORM_TOKEN") or extra.get("token"))


def _env_enablement() -> dict | None:
    token = os.getenv("MY_PLATFORM_TOKEN", "").strip()
    channel = os.getenv("MY_PLATFORM_CHANNEL", "").strip()
    if not (token and channel):
        return None
    seed = {"token": token, "channel": channel}
    home = os.getenv("MY_PLATFORM_HOME_CHANNEL")
    if home:
        seed["home_channel"] = {"chat_id": home, "name": "Home"}
    return seed


def register(ctx):
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="my_platform",
        label="My Platform",
        adapter_factory=lambda cfg: MyPlatformAdapter(cfg),
        # PASSIVE probe — "are deps/config present right now?".  Called from
        # status displays and config loading, so it must NEVER pip-install.
        check_fn=check_requirements,
        # ACTIVE installer (optional) — only for platforms with a
        # lazy-installable SDK.  create_adapter() calls it when check_fn
        # returns False, right before the gateway connects the platform.
        # Typically wraps tools.lazy_deps.ensure_and_bind(...).  Omit it
        # and a False check_fn is a hard block.
        # ensure_deps_fn=ensure_requirements,
        validate_config=validate_config,
        required_env=["MY_PLATFORM_TOKEN"],
        install_hint="pip install my-platform-sdk",
        # Env-driven auto-configuration — seeds PlatformConfig.extra from
        # env vars before adapter construction. See "Env-Driven Auto-
        # Configuration" section below.
        env_enablement_fn=_env_enablement,
        # Cron home-channel delivery support. Lets deliver=my_platform cron
        # jobs route without editing cron/scheduler.py. See "Cron Delivery"
        # section below.
        cron_deliver_env_var="MY_PLATFORM_HOME_CHANNEL",
        # Per-platform user authorization env vars
        allowed_users_env="MY_PLATFORM_ALLOWED_USERS",
        allow_all_env="MY_PLATFORM_ALLOW_ALL_USERS",
        # Message length limit for smart chunking (0 = no limit)
        max_message_length=4000,
        # LLM guidance injected into system prompt
        platform_hint=(
            "You are chatting via My Platform. "
            "It supports markdown formatting."
        ),
        # Display
        emoji="💬",
    )

    # Optional: register platform-specific tools
    ctx.register_tool(
        name="my_platform_search",
        toolset="my_platform",
        schema={...},
        handler=my_search_handler,
    )
```

### 설정

사용자는 `config.yaml`에서 플랫폼을 설정합니다:

```yaml
gateway:
  platforms:
    my_platform:
      enabled: true
      extra:
        token: "..."
        channel: "#general"
```

또는 환경 변수를 사용할 수도 있습니다(어댑터가 `__init__`에서 읽습니다).

### 플러그인 시스템이 자동으로 처리하는 항목

`ctx.register_platform()`을 호출하면 다음 통합 지점이 자동으로 처리됩니다 — 코어 코드 변경이 필요하지 않습니다:

| 통합 지점 | 작동 방식 |
|---|---|
| 게이트웨이 어댑터 생성 | 내장 if/elif 체인보다 먼저 레지스트리 확인 |
| 설정 파싱 | `Platform._missing_()`이 모든 플랫폼 이름을 허용 |
| 연결된 플랫폼 검증 | 레지스트리 `validate_config()` 호출 |
| 사용자 인증 | `allowed_users_env` / `allow_all_env` 확인 |
| 환경 변수만으로 자동 활성화 | `env_enablement_fn`이 `PlatformConfig.extra` + `home_channel` 시드 |
| YAML 설정 브리지 | `apply_yaml_config_fn`이 `config.yaml` 키를 환경 변수 / extra로 변환 |
| Cron 전송 | `cron_deliver_env_var`가 `deliver=<name>` 작동을 지원 |
| `hermes config` UI 항목 | `plugin.yaml`의 `requires_env` / `optional_env`가 자동으로 채움 |
| 전송 엔진(`tools/send_message_tool.py`) | 활성 게이트웨이 어댑터를 통해 라우팅 |
| 웹훅 크로스 플랫폼 전송 | 알려진 플랫폼을 레지스트리에서 확인 |
| `/update` 명령 접근 | `allow_update_command` 플래그 |
| 채널 디렉터리 | 열거 시 플러그인 플랫폼 포함 |
| 시스템 프롬프트 힌트 | `platform_hint`를 LLM 컨텍스트에 주입 |
| 메시지 분할 | 스마트 분할을 위한 `max_message_length` |
| PII 삭제 | `pii_safe` 플래그 |
| `hermes status` | 플러그인 플랫폼을 `(plugin)` 태그와 함께 표시 |
| `hermes gateway setup` | 플러그인 플랫폼이 설정 메뉴에 표시 |
| `hermes tools` / `hermes skills` | 플랫폼별 설정에 플러그인 플랫폼 포함 |
| 토큰 잠금(다중 프로필) | `connect()`에서 `acquire_scoped_lock()` 사용 |
| 고아 설정 경고 | 플러그인이 없을 때 설명적인 로그 출력 |

## 독립 전송 경로 확장

독립형 플랫폼은 `ctx.register_platform()`으로 생성된 동일한 `PlatformEntry`에 전송 동작을 선언하여 호스트가 주도하는 아웃바운드 전송에 참여할 수 있습니다. 이를 통해 직접 `hermes send --to ...` 및 cron `deliver=platform:...`을 사용할 수 있습니다. `send_message`는 의도적으로 에이전트가 호출할 수 있는 모델 도구가 아닙니다. 플러그인은 에이전트가 스스로 아웃바운드 메시지를 시작할 수 있게 하는 동등한 모델 표면을 등록해서는 안 됩니다.

```python
async def _send_request(args, chat_id, platform_name, pconfig):
    # `args` contains the host-driven send request fields.
    message_id = await client.send(
        address=chat_id,
        body=args["message"],
        subject=args.get("subject"),
    )
    return {"success": True, "platform": platform_name,
            "chat_id": chat_id, "message_id": message_id}


def _parse_address(raw):
    normalized = raw.strip().lower()
    if normalized.startswith("@") and "@" in normalized[1:]:
        return normalized, None  # (chat_id, optional thread_id)
    return None                 # continue to channel-directory resolution


def _validate_address(address):
    # True accepts; False rejects; a string rejects with that diagnostic.
    return True if address.endswith("@example.com") else "unsupported domain"


def register(ctx):
    ctx.register_platform(
        name="fmsg",
        label="Fixture Message",
        adapter_factory=lambda cfg: FmsgAdapter(cfg),
        check_fn=check_requirements,
        parse_target_ref_fn=_parse_address,
        validate_target_ref_fn=_validate_address,
        # May be a regular function or async def. Hermes awaits any awaitable
        # result, including callable objects and functools.partial wrappers.
        send_message_handler=_send_request,
        # Prefer this lower-level hook when cron must send from a process
        # without the live gateway.
        standalone_sender_fn=_standalone_send,
    )
```

대상 해석은 세 가지 아웃바운드 표면에서 공유됩니다. 먼저 파서 출력이 정규화되고 채널 디렉터리 ID는 신뢰됩니다. 플러그인 파서는 네이티브 대상 구문을 명시적으로 허용해야 합니다. 해석되지 않은 문자열은 불투명하게 그대로 전달되지 않습니다. 알 수 없는 플랫폼과 검증기 실패는 조용히 전송을 시도하는 대신 진단 메시지를 반환합니다. 플러그인 강제 재로드/프로필 전환 시 소유한 항목이 등록 해제되므로 파서와 핸들러가 다음 프로필로 유출될 수 없습니다.

## 환경 변수 기반 자동 설정

대부분의 사용자는 `config.yaml`을 편집하는 대신 `~/.hermes/.env`에 환경 변수를 추가하여 플랫폼을 설정합니다. `env_enablement_fn` 훅을 사용하면 어댑터가 생성되기 **전에** 플러그인이 해당 환경 변수를 가져올 수 있습니다. 따라서 플랫폼 SDK를 인스턴스화하지 않고도 `hermes gateway status`, `get_connected_platforms()`, cron 전송이 올바른 상태를 확인할 수 있습니다.

```python
def _env_enablement() -> dict | None:
    """Seed PlatformConfig.extra from env vars.

    Called by the platform registry during load_gateway_config().
    Return None when the platform isn't minimally configured — the
    caller then skips auto-enabling. Return a dict to seed extras.

    The special 'home_channel' key is extracted and becomes a proper
    HomeChannel dataclass on the PlatformConfig; every other key is
    merged into PlatformConfig.extra.
    """
    token = os.getenv("MY_PLATFORM_TOKEN", "").strip()
    channel = os.getenv("MY_PLATFORM_CHANNEL", "").strip()
    if not (token and channel):
        return None
    seed = {"token": token, "channel": channel}
    home = os.getenv("MY_PLATFORM_HOME_CHANNEL")
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("MY_PLATFORM_HOME_CHANNEL_NAME", "Home"),
        }
    return seed


def register(ctx):
    ctx.register_platform(
        name="my_platform",
        label="My Platform",
        adapter_factory=lambda cfg: MyPlatformAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        env_enablement_fn=_env_enablement,
        # ... other fields
    )
```
## `config.yaml` 설정을 환경 변수로 연결하기

일부 사용자는 환경 변수 대신 `config.yaml` 키(`my_platform.require_mention`, `my_platform.allowed_channels` 등)를 설정하는 방식을 선호합니다. `apply_yaml_config_fn` 훅을 사용하면 코어 `gateway/config.py`가 플랫폼의 YAML 스키마를 알아야 하도록 강제하지 않고도 플러그인이 이 변환을 직접 처리할 수 있습니다.

```python
import os

def _apply_yaml_config(yaml_cfg: dict, platform_cfg: dict) -> dict | None:
    """Translate config.yaml `my_platform:` keys into env vars / extras.

    yaml_cfg     — the full top-level parsed config.yaml dict
    platform_cfg — the platform's own sub-dict (yaml_cfg.get("my_platform", {}))

    May mutate os.environ directly (use `not os.getenv(...)` guards to
    preserve env > YAML precedence) and/or return a dict to merge into
    PlatformConfig.extra. Return None or {} for no extras.
    """
    if "require_mention" in platform_cfg and not os.getenv("MY_PLATFORM_REQUIRE_MENTION"):
        os.environ["MY_PLATFORM_REQUIRE_MENTION"] = str(platform_cfg["require_mention"]).lower()
    allowed = platform_cfg.get("allowed_channels")
    if allowed is not None and not os.getenv("MY_PLATFORM_ALLOWED_CHANNELS"):
        if isinstance(allowed, list):
            allowed = ",".join(str(v) for v in allowed)
        os.environ["MY_PLATFORM_ALLOWED_CHANNELS"] = str(allowed)
    return None  # nothing extra to merge into PlatformConfig.extra

def register(ctx):
    ctx.register_platform(
        name="my_platform",
        ...,
        apply_yaml_config_fn=_apply_yaml_config,
    )
```

이 훅은 `load_gateway_config()`에서 공통 키(`unauthorized_dm_behavior`, `notice_delivery`, `reply_prefix`, `require_mention` 등)를 처리하는 일반 공유 키 루프가 끝난 후, `_apply_env_overrides()`가 실행되기 전에 호출됩니다. 따라서 플러그인은 플랫폼별 키만 연결하면 됩니다.

훅에서 발생한 예외는 무시되고 디버그 수준으로 로그에 기록됩니다. 즉, 문제가 있는 플러그인 하나 때문에 게이트웨이 설정 로드 전체가 중단되지 않습니다.


## Cron 전송

`deliver=my_platform` Cron 작업이 설정된 홈 채널로 라우팅되도록 하려면, 기본 채팅방/룸/채널 ID를 담은 환경 변수의 이름을 `cron_deliver_env_var`로 지정합니다.

```python
ctx.register_platform(
    name="my_platform",
    ...
    cron_deliver_env_var="MY_PLATFORM_HOME_CHANNEL",
)
```

스케줄러는 `deliver=my_platform` 작업의 홈 대상 위치를 확인할 때 이 환경 변수를 읽으며, `_KNOWN_DELIVERY_PLATFORMS`와 유사한 검사에서도 해당 플랫폼을 유효한 Cron 대상으로 취급합니다. `env_enablement_fn`이 `home_channel` 딕셔너리를 채우는 경우(위 참조)에는 이것이 우선합니다. `cron_deliver_env_var`는 환경 변수 시딩 전에 실행되는 Cron 작업을 위한 대체 수단입니다.

### 프로세스 외부에서 Cron 전송하기

`cron_deliver_env_var`는 플랫폼을 `deliver=` 대상으로 인식하게 합니다. 게이트웨이와 별도의 프로세스에서 Cron 작업이 실행될 때(즉, `hermes cron run`이 `hermes gateway`와 별도로 실행될 때) 실제 전송까지 성공하게 하려면 `standalone_sender_fn`을 등록합니다.

```python
async def _standalone_send(
    pconfig,
    chat_id,
    message,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
):
    """Open an ephemeral connection / acquire a fresh token, send, and close."""
    # ... open connection, send message, return result ...
    return {"success": True, "message_id": "..."}
    # or {"error": "..."}

ctx.register_platform(
    name="my_platform",
    ...
    cron_deliver_env_var="MY_PLATFORM_HOME_CHANNEL",
    standalone_sender_fn=_standalone_send,
)
```

이 훅이 필요한 이유는 다음과 같습니다. 기본 제공 플랫폼(Telegram, Discord, Slack 등)은 `tools/send_message_tool.py`에 직접 REST 헬퍼를 제공하므로 게이트웨이와 같은 프로세스를 유지하지 않아도 Cron 전송이 가능합니다. 반면 플러그인 플랫폼은 전통적으로 `_gateway_runner_ref()`에 의존했는데, 이 함수는 게이트웨이 프로세스 외부에서는 `None`을 반환합니다. 따라서 `standalone_sender_fn`이 없으면 Cron 측 전송이 `No live adapter for platform '<name>'` 오류와 함께 실패합니다.

이 함수는 실행 중인 어댑터와 동일한 `pconfig` 및 `chat_id`를 받으며, 선택적 키워드 인자인 `thread_id`, `media_files`, `force_document`도 받습니다. `{"success": True, "message_id": ...}`를 반환하면 전송 성공으로 처리되고, `{"error": "..."}`를 반환하면 해당 메시지가 Cron의 `delivery_errors`에 표시됩니다. 함수 내부에서 발생한 예외는 디스패처가 받아 `Plugin standalone send failed: <reason>`으로 보고합니다. 참고 구현은 `plugins/platforms/{irc,teams,google_chat}/adapter.py`에 있습니다.

## `hermes config`에서 환경 변수 표시하기

`hermes_cli/config.py`는 import 시점에 `plugins/platforms/*/plugin.yaml`을 스캔하고 `requires_env` 및 (선택 사항인) `optional_env` 블록에서 `OPTIONAL_ENV_VARS`를 자동으로 채웁니다. 설명, 프롬프트, 비밀번호 플래그, URL을 제대로 제공하려면 풍부한 딕셔너리 형식을 사용하세요. 그러면 CLI 설정 UI가 이를 자동으로 활용합니다.

```yaml
# plugins/platforms/my_platform/plugin.yaml
name: my_platform-platform
label: My Platform
kind: platform
version: 1.0.0
description: >
  My Platform gateway adapter for Hermes Agent.
author: Your Name
requires_env:
  - name: MY_PLATFORM_TOKEN
    description: "Bot API token from the My Platform console"
    prompt: "My Platform bot token"
    url: "https://my-platform.example.com/bots"
    password: true
  - name: MY_PLATFORM_CHANNEL
    description: "Channel to join (e.g. #hermes)"
    prompt: "Channel"
    password: false
optional_env:
  - name: MY_PLATFORM_HOME_CHANNEL
    description: "Default channel for cron delivery (defaults to MY_PLATFORM_CHANNEL)"
    prompt: "Home channel (or empty)"
    password: false
  - name: MY_PLATFORM_ALLOWED_USERS
    description: "Comma-separated user IDs allowed to talk to the bot"
    prompt: "Allowed users (comma-separated)"
    password: false
```

**지원되는 딕셔너리 키:** `name`(필수), `description`, `prompt`, `url`, `password`(bool; 생략하면 값이 `*_TOKEN` / `*_SECRET` / `*_KEY` / `*_PASSWORD` / `*_JSON` 접미사인지 자동 감지), `category`(기본값은 `"messaging"`).

일반 문자열 항목(`- MY_PLATFORM_TOKEN`)도 계속 사용할 수 있습니다. 이 경우 플러그인의 `label`에서 자동으로 유도한 일반 설명이 적용됩니다. 동일한 변수가 `OPTIONAL_ENV_VARS`에 하드코딩된 항목으로 이미 존재하면 해당 항목이 우선합니다(하위 호환). `plugin.yaml` 형식은 대체 수단으로 동작합니다.

## 플랫폼별 느린 LLM 응답 UX

일부 플랫폼에는 느린 LLM 응답을 표시하는 방식을 바꾸는 제약이 있습니다.

- **LINE**은 수신 이벤트 후 약 60초가 지나면 만료되는 일회성 *reply token*을 발급합니다. 이 토큰으로 답장하는 것은 무료지만, 과금되는 Push API로 대체하는 것은 무료가 아닙니다. 제한 시간까지 LLM이 끝나지 않았다면 선택지는 "유료 Push 할당량을 소모"하거나 "reply token이 만료되기 전에 더 영리하게 사용하는 것"입니다.
- **WhatsApp**은 24시간이 지나면 세션을 비활성 상태로 표시하며, 그 이후에는 템플릿 메시지만 허용됩니다.
- **SMS**에는 입력 중 표시나 점진적 업데이트라는 개념이 없습니다. 긴 응답은 봇이 오프라인인 것처럼 보일 뿐입니다.

이는 기본 `BasePlatformAdapter`가 미리 예상할 수 없는 실제 제약입니다. 플러그인 표면은 어댑터가 기본 입력 중 루프 위에 플랫폼별 UX를 추가할 수 있는 여지를 의도적으로 남겨 두며, kwarg 목록을 확장하지 않습니다.

### 패턴: `_keep_typing`을 서브클래싱해 진행 중 UX 추가하기

`BasePlatformAdapter._keep_typing`은 입력 중 표시의 heartbeat입니다. LLM이 생성하는 동안 백그라운드 작업으로 실행되고, 응답이 전달되면 취소됩니다. 임계값에서 플랫폼별 동작(예: 45초 후 "아직 생각 중" 말풍선 전송)을 추가하려면 어댑터에서 `_keep_typing`을 재정의하고, `super()._keep_typing()`과 함께 자체 작업을 예약한 다음 `finally`에서 정리합니다.

```python
class LineAdapter(BasePlatformAdapter):
    async def _keep_typing(self, chat_id: str, *args, **kwargs) -> None:
        if self.slow_response_threshold <= 0:
            await super()._keep_typing(chat_id, *args, **kwargs)
            return

        async def _fire_at_threshold() -> None:
            try:
                await asyncio.sleep(self.slow_response_threshold)
            except asyncio.CancelledError:
                raise
            # Platform-specific work here — for LINE, send a Template
            # Buttons "Get answer" bubble using the cached reply token
            # so the user can fetch the cached response later via a
            # fresh (free) reply token from the postback callback.
            await self._send_slow_response_button(chat_id)

        side_task = asyncio.create_task(_fire_at_threshold())
        try:
            await super()._keep_typing(chat_id, *args, **kwargs)
        finally:
            if not side_task.done():
                side_task.cancel()
                try:
                    await side_task
                except (asyncio.CancelledError, Exception):
                    pass
```

핵심 사항:

- **항상 `await super()._keep_typing(...)`을 호출하세요.** 입력 중 heartbeat는 독립적으로 유용하므로 대체하지 말고 그 위에 추가해야 합니다.
- **`finally`에서 보조 작업을 정리하세요.** LLM이 완료되거나 `/stop`으로 실행이 취소되면 게이트웨이가 입력 중 작업을 취소합니다. 보조 작업도 이 취소를 확인해야 합니다. 그렇지 않으면 작업이 남아 응답이 이미 전달된 뒤 실행될 수 있습니다.
- **`interrupt_session_activity`와 함께 사용해** 사용자가 `/stop`을 입력했을 때 고아 UX 상태를 해결하세요. LINE의 경우 postback 캐시 항목을 `PENDING`에서 `ERROR`로 전환하여, 지속적으로 표시되는 "Get answer" 버튼이 반복되는 대신 "Run was interrupted" 메시지를 전달하게 한다는 뜻입니다.

### 패턴: 즉시 전송하는 대신 캐시를 거치도록 `send` 서브클래싱하기

느린 응답 UX가 나중에 가져올 수 있도록 응답을 캐시한다면(LINE의 postback 흐름처럼) `send` 재정의는 다음 세 가지 모드를 인식해야 합니다.

1. **이 채팅에 대기 중인 postback이 활성 상태** → `request_id` 아래에 응답을 캐시하고, 화면에 보이는 내용은 전송하지 않습니다.
2. **시스템 busy-ack**(`⚡ Interrupting`, `⏳ Queued`, `⏩ Steered`) → 캐시를 우회하고 화면에 보이도록 전송하여 사용자가 자신의 입력에 대한 게이트웨이 응답을 확인하게 합니다.
3. **일반 응답** → 평소처럼 reply-token-or-push를 통해 전송합니다.

```python
async def send(self, chat_id: str, content: str, **kw) -> SendResult:
    if _is_system_bypass(content):
        return await self._send_text_chunks(chat_id, content, force_push=False)
    pending_rid = self._pending_buttons.get(chat_id)
    if pending_rid:
        self._cache.set_ready(pending_rid, content)
        return SendResult(success=True, message_id=pending_rid)
    return await self._send_text_chunks(chat_id, content, force_push=False)
```

`_SYSTEM_BYPASS_PREFIXES`는 게이트웨이 자체의 busy acknowledgment 접두사(`⚡`, `⏳`, `⏩`, `💾`)입니다. 캐시된 UX 상태와 관계없이 항상 이 메시지가 화면에 보이도록 통과시켜야 합니다.

### 이 패턴이 적합한 경우

다음 조건을 모두 만족할 때 입력 중 루프 재정의 방식을 사용하세요.

- 플랫폼의 아웃바운드 API에 엄격한 시간 제한이 있고(일회성 reply token, 만료되는 고정 세션 등), 그리고
- 해당 플랫폼에서 화면에 보이는 진행 중 말풍선이 허용되는 UX인 경우.

다음과 같은 경우에는 더 단순한 `slow_response_threshold = 0` 상시 Push 방식을 사용하세요.

- 플랫폼에 무료와 유료의 의미 있는 차이가 없거나,
- 사용자 커뮤니티가 상호작용 가능한 중간 말풍선보다 "로딩… 로딩… 완료"라는 침묵 후 응답을 선호하는 경우.

LINE은 두 방식을 모두 지원합니다. 기본 임계값은 무료 postback 가져오기를 위해 45초이며, `LINE_SLOW_RESPONSE_THRESHOLD=0`으로 설정하면 "항상 Push 대체" 방식으로 돌아갑니다.

### 참고 구현

전체 LINE postback 구현은 `plugins/platforms/line/adapter.py`를 참조하세요. 여기에는 `RequestCache` 상태 머신(`PENDING → READY → DELIVERED`, `/stop`을 위한 `ERROR` 포함), 임계값에서 Template Buttons 말풍선을 표시하는 `_keep_typing` 재정의, 캐시를 통해 라우팅하는 `send` 재정의, 고아 PENDING 항목을 해결하는 `interrupt_session_activity` 재정의가 포함되어 있습니다.

### 참고 구현(플러그인 경로)

완전히 동작하는 예제는 저장소의 `plugins/platforms/irc/`를 참조하세요. 외부 의존성이 전혀 없는 완전한 비동기 IRC 어댑터입니다. `plugins/platforms/teams/`는 Bot Framework / Adaptive Cards를, `plugins/platforms/google_chat/`는 OAuth 기반 REST API를, `plugins/platforms/line/`은 플랫폼별 느린 LLM 응답 UX를 갖춘 webhook 기반 Messaging API를 다룹니다.

## 단계별 체크리스트 (기본 경로)

:::note
이 체크리스트는 Hermes 핵심 코드베이스에 플랫폼을 직접 추가할 때 사용하는 절차입니다. 일반적으로 공식 지원 플랫폼을 추가하는 핵심 기여자가 수행합니다. 커뮤니티/서드파티 플랫폼은 위의 [플러그인 경로](#plugin-path-recommended)를 사용해야 합니다.
:::

### 1. 플랫폼 열거형

`gateway/config.py`의 `Platform` 열거형에 플랫폼을 추가합니다.

```python
class Platform(Enum):
    # ... existing platforms ...
    NEWPLAT = "newplat"
```

### 2. 어댑터 파일

`plugins/platforms/newplat/adapter.py`를 생성합니다.

```python
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter, MessageEvent, MessageType, SendResult,
)

def check_newplat_requirements() -> bool:
    """Return True if dependencies are available."""
    return SOME_SDK_AVAILABLE

class NewPlatAdapter(BasePlatformAdapter):
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.NEWPLAT)
        # Read config from config.extra dict
        extra = config.extra or {}
        self._api_key = extra.get("api_key") or os.getenv("NEWPLAT_API_KEY", "")

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        # Set up connection, start polling/webhook
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._running = False
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        # Send message via platform API
        return SendResult(success=True, message_id="...")

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}
```

인바운드 메시지의 경우 `MessageEvent`를 만들고 `self.handle_message(event)`를 호출합니다.

```python
source = self.build_source(
    chat_id=chat_id,
    chat_name=name,
    chat_type="dm",  # or "group"
    user_id=user_id,
    user_name=user_name,
)
event = MessageEvent(
    text=content,
    message_type=MessageType.TEXT,
    source=source,
    message_id=msg_id,
)
await self.handle_message(event)
```

### 3. 게이트웨이 설정 (`gateway/config.py`)

수정 지점은 세 곳입니다.

1. **`get_connected_platforms()`** — 플랫폼에 필요한 자격 증명 확인을 추가합니다.
2. **`load_gateway_config()`** — 토큰 환경 변수 매핑 항목을 추가합니다: `Platform.NEWPLAT: "NEWPLAT_TOKEN"`
3. **`_apply_env_overrides()`** — 모든 `NEWPLAT_*` 환경 변수를 설정에 매핑합니다.

### 4. 게이트웨이 실행기 (`gateway/run.py`)

수정 지점은 여섯 곳입니다.

1. **`_create_adapter()`** — `elif platform == Platform.NEWPLAT:` 분기를 추가합니다.
2. **`_is_user_authorized()`의 allowed_users 매핑** — `Platform.NEWPLAT: "NEWPLAT_ALLOWED_USERS"`
3. **`_is_user_authorized()`의 allow_all 매핑** — `Platform.NEWPLAT: "NEWPLAT_ALLOW_ALL_USERS"`
4. **초기 환경 변수 확인용 `_any_allowlist` 튜플** — `"NEWPLAT_ALLOWED_USERS"`를 추가합니다.
5. **초기 환경 변수 확인용 `_allow_all` 튜플** — `"NEWPLAT_ALLOW_ALL_USERS"`를 추가합니다.
6. **`_UPDATE_ALLOWED_PLATFORMS` frozenset** — `Platform.NEWPLAT`을 추가합니다.

### 5. 플랫폼 간 전달

1. **`gateway/platforms/webhook.py`** — 전달 유형 튜플에 `"newplat"`을 추가합니다.
2. **`cron/scheduler.py`** — `_KNOWN_DELIVERY_PLATFORMS` frozenset과 `_deliver_result()` 플랫폼 매핑에 추가합니다.

### 6. CLI 통합

1. **`hermes_cli/config.py`** — 모든 `NEWPLAT_*` 변수를 `_EXTRA_ENV_KEYS`에 추가합니다.
2. **`hermes_cli/gateway.py`** — 키, 레이블, 이모지, token_var, setup_instructions, vars를 포함한 항목을 `_PLATFORMS` 목록에 추가합니다.
3. **`hermes_cli/platforms.py`** — 레이블과 default_toolset을 포함한 `PlatformInfo` 항목을 추가합니다(`skills_config` 및 `tools_config` TUI에서 사용됨).
4. **`hermes_cli/setup.py`** — `_setup_newplat()` 함수를 추가하고(`gateway.py`에 위임 가능), 메시징 플랫폼 목록에 튜플을 추가합니다.
5. **`hermes_cli/status.py`** — 플랫폼 감지 항목을 추가합니다: `"NewPlat": ("NEWPLAT_TOKEN", "NEWPLAT_HOME_CHANNEL")`
6. **`hermes_cli/dump.py`** — 플랫폼 감지 딕셔너리에 `"newplat": "NEWPLAT_TOKEN"`을 추가합니다.

### 7. 도구

1. **`tools/send_message_tool.py`** — 플랫폼 매핑에 `"newplat": Platform.NEWPLAT`을 추가합니다.
2. **`tools/cronjob_tools.py`** — 전달 대상 설명 문자열에 `newplat`을 추가합니다.

### 8. 도구 세트

1. **`toolsets.py`** — `_HERMES_CORE_TOOLS`를 포함하는 `"hermes-newplat"` 도구 세트 정의를 추가합니다.
2. **`toolsets.py`** — `"hermes-gateway"`의 includes 목록에 `"hermes-newplat"`을 추가합니다.

### 9. 선택 사항: 플랫폼 힌트

**`agent/prompt_builder.py`** — 플랫폼에 특정 렌더링 제약(마크다운 미지원, 메시지 길이 제한 등)이 있는 경우 `PLATFORM_HINTS` 딕셔너리에 항목을 추가합니다. 이를 통해 시스템 프롬프트에 플랫폼별 지침을 주입할 수 있습니다.

```python
PLATFORM_HINTS = {
    # ...
    "newplat": (
        "You are chatting via NewPlat. It supports markdown formatting "
        "but has a 4000-character message limit."
    ),
}
```

모든 플랫폼에 힌트가 필요한 것은 아닙니다. 에이전트의 동작이 달라져야 하는 경우에만 추가합니다.

### 10. 테스트

다음을 다루는 `tests/gateway/test_newplat.py`를 생성합니다.

- 설정으로부터의 어댑터 생성
- 메시지 이벤트 구성
- 전송 메서드(외부 API를 모의 처리)
- 플랫폼별 기능(암호화, 라우팅 등)

### 11. 문서

| 파일 | 추가할 내용 |
|------|-------------|
| `website/docs/user-guide/messaging/newplat.md` | 전체 플랫폼 설정 페이지 |
| `website/docs/user-guide/messaging/index.md` | 플랫폼 비교 표, 아키텍처 다이어그램, 도구 세트 표, 보안 섹션, 다음 단계 링크 |
| `website/docs/reference/environment-variables.md` | 모든 NEWPLAT_* 환경 변수 |
| `website/docs/reference/toolsets-reference.md` | hermes-newplat 도구 세트 |
| `website/docs/integrations/index.md` | 플랫폼 링크 |
| `website/sidebars.ts` | 문서 페이지의 사이드바 항목 |
| `website/docs/developer-guide/architecture.md` | 어댑터 수 + 목록 |
| `website/docs/developer-guide/gateway-internals.md` | 어댑터 파일 목록 |

## 동등성 감사

새 플랫폼 PR을 완료로 표시하기 전에 기존 플랫폼을 기준으로 동등성 감사를 실행합니다.

```bash
# Find every .py file mentioning the reference platform
search_files "bluebubbles" output_mode="files_only" file_glob="*.py"

# Find every .py file mentioning the new platform
search_files "newplat" output_mode="files_only" file_glob="*.py"

# Any file in the first set but not the second is a potential gap
```

`.md` 및 `.ts` 파일에 대해서도 반복합니다. 각 차이를 조사합니다. 플랫폼 열거형이라서 업데이트가 필요한 것인지, 아니면 플랫폼별 참조라서 건너뛰어도 되는 것인지 확인합니다.

## 일반적인 패턴

### 롱 폴링 어댑터

어댑터가 롱 폴링(예: Telegram 또는 Weixin)을 사용하는 경우 폴링 루프 태스크를 사용합니다.

```python
async def connect(self):
    self._poll_task = asyncio.create_task(self._poll_loop())
    self._mark_connected()

async def _poll_loop(self):
    while self._running:
        messages = await self._fetch_updates()
        for msg in messages:
            await self.handle_message(self._build_event(msg))
```

### 콜백/웹훅 어댑터

플랫폼이 엔드포인트로 메시지를 푸시하는 경우(예: WeCom Callback) HTTP 서버를 실행합니다.

```python
async def connect(self):
    self._app = web.Application()
    self._app.router.add_post("/callback", self._handle_callback)
    # ... start aiohttp server
    self._mark_connected()

async def _handle_callback(self, request):
    event = self._build_event(await request.text())
    await self._message_queue.put(event)
    return web.Response(text="success")  # Acknowledge immediately
```

응답 기한이 매우 짧은 플랫폼(예: WeCom의 5초 제한)의 경우 항상 즉시 확인 응답을 보내고, 이후 API를 통해 에이전트의 답변을 능동적으로 전달합니다. 에이전트 세션은 3~30분 동안 실행되므로 콜백 응답 시간 내에 인라인 답변을 제공하는 것은 현실적이지 않습니다.

### 토큰 잠금

어댑터가 고유한 자격 증명으로 지속적인 연결을 유지하는 경우, 두 프로필이 동일한 자격 증명을 사용하지 못하도록 범위가 지정된 잠금을 추가합니다.

```python
from gateway.status import acquire_scoped_lock, release_scoped_lock

async def connect(self, *, is_reconnect: bool = False):
    acquired, _existing = acquire_scoped_lock("newplat", self._token)
    if not acquired:
        logger.error("Token already in use by another profile")
        return False
    # ... connect

async def disconnect(self):
    release_scoped_lock("newplat", self._token)
```

## 참조 구현

| 어댑터 | 패턴 | 복잡도 | 적합한 참조 대상 |
|---------|---------|------------|-------------------|
| `bluebubbles.py` | REST + 웹훅 | 중간 | 간단한 REST API 통합 |
| `weixin.py` | 롱 폴링 + CDN | 높음 | 미디어 처리, 암호화 |
| `plugins/platforms/wecom/callback_adapter.py` | 콜백/웹훅 | 중간 | HTTP 서버, AES 암호화, 다중 앱 |
| `plugins/platforms/irc/adapter.py` | 롱 폴링 + IRC 프로토콜 | 높음 | 범위가 지정된 토큰 잠금을 사용하는 완전한 기능의 플러그인 어댑터 |
