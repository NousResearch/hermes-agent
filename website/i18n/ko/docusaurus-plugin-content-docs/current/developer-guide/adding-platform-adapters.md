---
sidebar_position: 9
---

# 플랫폼 어댑터 추가하기 (Adding a Platform Adapter)

이 가이드는 Hermes 게이트웨이에 새로운 메시징 플랫폼을 추가하는 방법을 설명합니다. 플랫폼 어댑터는 Hermes를 외부 메시징 서비스(Telegram, Discord, WeCom 등)에 연결하여 사용자가 해당 서비스를 통해 에이전트와 상호 작용할 수 있도록 합니다.

:::tip
플랫폼을 추가하는 두 가지 방법이 있습니다:
- **플러그인** (커뮤니티/서드파티 권장): `~/.hermes/plugins/`에 플러그인 디렉터리를 추가합니다 — 코어 코드 수정이 전혀 필요하지 않습니다. 아래의 [플러그인 경로](#plugin-path-recommended)를 참조하세요.
- **내장(Built-in)**: 코드, 구성, 문서 등 20여 개의 파일을 수정해야 합니다. 아래의 [내장 방식 체크리스트](#step-by-step-checklist-built-in-path)를 사용하세요.
:::

## 아키텍처 개요

```
사용자 ↔ 메시징 플랫폼 ↔ 플랫폼 어댑터 ↔ 게이트웨이 러너 ↔ AIAgent
```

모든 어댑터는 `gateway/platforms/base.py`의 `BasePlatformAdapter`를 상속하며 다음을 구현합니다:

- **`connect()`** — 연결 설정 (WebSocket, 롱 폴링(long-poll), HTTP 서버 등) *(추상 메서드)*
- **`disconnect()`** — 안전한 종료 *(추상 메서드)*
- **`send()`** — 채팅에 텍스트 메시지 전송 *(추상 메서드)*
- **`send_typing()`** — 타이핑 표시기 표시 (선택적 재정의)
- **`get_chat_info()`** — 채팅 메타데이터 반환 (선택적 재정의)

수신된 메시지는 어댑터에 의해 수신되고 기본 클래스(base class)가 게이트웨이 러너로 라우팅하는 `self.handle_message(event)`를 통해 전달됩니다.

## 플러그인 경로 (권장됨)

플러그인 시스템을 사용하면 핵심 Hermes 코드를 수정하지 않고 플랫폼 어댑터를 추가할 수 있습니다. 플러그인은 두 개의 파일을 포함하는 디렉터리입니다:

```
~/.hermes/plugins/my-platform/
  PLUGIN.yaml      # 플러그인 메타데이터
  adapter.py       # 어댑터 클래스 + register() 엔트리 포인트
```

### PLUGIN.yaml

플러그인 메타데이터입니다. `requires_env` 및 `optional_env` 블록은 `hermes config` UI 항목을 자동으로 채웁니다. (아래의 [`hermes config`에 환경 변수 노출하기](#surfacing-env-vars-in-hermes-config) 참고).

```yaml
name: my-platform
label: My Platform
kind: platform
version: 1.0.0
description: My custom messaging platform adapter
author: Your Name
requires_env:
  - MY_PLATFORM_TOKEN          # 일반 문자열로 가능
  - name: MY_PLATFORM_CHANNEL  # 또는 더 나은 UX를 위한 딕셔너리 형태
    description: "Channel to join"
    prompt: "Channel"
    password: false
optional_env:
  - name: MY_PLATFORM_HOME_CHANNEL
    description: "Default channel for cron delivery"
    password: false
```

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

    async def connect(self) -> bool:
        # 플랫폼 API에 연결하고 리스너 시작
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        # 플랫폼 API를 통해 메시지 전송
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
    """플러그인 엔트리 포인트 — Hermes 플러그인 시스템에 의해 호출됨."""
    ctx.register_platform(
        name="my_platform",
        label="My Platform",
        adapter_factory=lambda cfg: MyPlatformAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        required_env=["MY_PLATFORM_TOKEN"],
        install_hint="pip install my-platform-sdk",
        # 환경 변수 기반 자동 구성 — 어댑터 구성 전에 환경 변수에서
        # PlatformConfig.extra를 초기화합니다.
        # 아래의 "환경 변수 기반 자동 구성" 섹션을 참고하세요.
        env_enablement_fn=_env_enablement,
        # cron 홈 채널 전송(delivery) 지원. cron/scheduler.py를 편집하지 않고도
        # deliver=my_platform cron 작업이 올바르게 전달되도록 합니다.
        # 아래의 "Cron 전송" 섹션을 참고하세요.
        cron_deliver_env_var="MY_PLATFORM_HOME_CHANNEL",
        # 플랫폼별 사용자 권한 부여 환경 변수
        allowed_users_env="MY_PLATFORM_ALLOWED_USERS",
        allow_all_env="MY_PLATFORM_ALLOW_ALL_USERS",
        # 스마트 청킹(분할)을 위한 메시지 길이 제한 (0 = 무제한)
        max_message_length=4000,
        # 시스템 프롬프트에 주입되는 LLM 가이드라인
        platform_hint=(
            "You are chatting via My Platform. "
            "It supports markdown formatting."
        ),
        # 디스플레이 아이콘
        emoji="💬",
    )

    # 선택적: 플랫폼 전용 도구 등록
    ctx.register_tool(
        name="my_platform_search",
        toolset="my_platform",
        schema={...},
        handler=my_search_handler,
    )
```

### 설정 (Configuration)

사용자는 `config.yaml`에서 플랫폼을 구성합니다:

```yaml
gateway:
  platforms:
    my_platform:
      enabled: true
      extra:
        token: "..."
        channel: "#general"
```

또는 어댑터가 `__init__`에서 읽는 환경 변수를 통해 구성할 수 있습니다.

### 플러그인 시스템이 자동으로 처리하는 항목

`ctx.register_platform()`을 호출하면 코어 코드를 수정하지 않아도 다음 통합 사항이 자동으로 처리됩니다:

| 통합 항목 | 작동 방식 |
|---|---|
| 게이트웨이 어댑터 생성 | 내장(built-in) if/elif 체인 이전에 레지스트리 확인됨 |
| 구성 분석(Config parsing) | `Platform._missing_()`이 어떤 플랫폼 이름이든 수락 |
| 연결된 플랫폼 유효성 검사 | 레지스트리의 `validate_config()` 호출 |
| 사용자 권한 검사 | `allowed_users_env` / `allow_all_env` 확인됨 |
| 환경 변수 전용 자동 활성화 | `env_enablement_fn`이 `PlatformConfig.extra` + `home_channel` 초기화 |
| YAML 설정 브릿지 | `apply_yaml_config_fn`이 `config.yaml` 키를 환경 변수 / 추가(extra) 항목으로 변환 |
| Cron 전송 | `cron_deliver_env_var`를 통해 `deliver=<name>` 작동 |
| `hermes config` UI 항목 | `plugin.yaml`의 `requires_env` / `optional_env`를 자동으로 채움 |
| send_message 도구 | 활성 게이트웨이 어댑터를 통해 라우팅됨 |
| Webhook 교차 플랫폼 전송 | 레지스트리에서 알려진 플랫폼 확인 |
| `/update` 명령어 접근 | `allow_update_command` 플래그로 결정 |
| 채널 목록 | 열거형 데이터에 플러그인 플랫폼이 포함됨 |
| 시스템 프롬프트 힌트 | `platform_hint`가 LLM 컨텍스트에 주입됨 |
| 메시지 청킹(분할) | 스마트 분할을 위한 `max_message_length` 참조 |
| PII 마스킹 처리 | `pii_safe` 플래그로 결정 |
| `hermes status` | 플러그인 플랫폼을 `(plugin)` 태그와 함께 표시 |
| `hermes gateway setup` | 설정 메뉴에 플러그인 플랫폼 표시 |
| `hermes tools` / `hermes skills` | 플랫폼별 설정에 플러그인 플랫폼 추가 |
| 토큰 락(다중 프로필) | `connect()` 내부에서 `acquire_scoped_lock()` 사용 |
| 고아(orphaned) 구성 경고 | 플러그인이 누락된 경우 명확한 로그 제공 |

## 환경 변수 기반 자동 구성 (Env-Driven Auto-Configuration)

대부분의 사용자는 `config.yaml`을 직접 편집하는 대신 `~/.hermes/.env`에 환경 변수를 넣어 플랫폼을 설정합니다. `env_enablement_fn` 훅(hook)을 사용하면 어댑터가 생성되기 **전에** 플러그인이 이러한 환경 변수를 인식할 수 있으므로, 플랫폼 SDK를 초기화하지 않고도 `hermes gateway status`, `get_connected_platforms()` 및 cron 전송이 올바른 상태를 확인할 수 있습니다.

```python
def _env_enablement() -> dict | None:
    """환경 변수에서 PlatformConfig.extra를 초기화합니다.

    load_gateway_config() 동안 플랫폼 레지스트리에 의해 호출됩니다.
    플랫폼 구성이 최소 기준을 충족하지 않으면 None을 반환하고 자동 활성화를
    건너뜁니다. 구성 요소로 활용할 딕셔너리를 반환합니다.

    특수 키인 'home_channel'이 추출되어 PlatformConfig의 적절한 HomeChannel
    데이터 클래스가 되며, 다른 모든 키는 PlatformConfig.extra에 병합됩니다.
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
        # ... 다른 필드들
    )
```

## YAML→환경 변수 설정 브릿지 (YAML→env Config Bridge)

일부 사용자는 환경 변수보다 `config.yaml`의 키(`my_platform.require_mention`, `my_platform.allowed_channels` 등) 설정을 선호합니다. `apply_yaml_config_fn` 훅을 사용하면, 코어 `gateway/config.py`가 해당 플랫폼의 YAML 스키마를 알도록 강제하는 대신 플러그인이 직접 이 변환 작업을 담당할 수 있습니다.

```python
import os

def _apply_yaml_config(yaml_cfg: dict, platform_cfg: dict) -> dict | None:
    """config.yaml `my_platform:` 키를 환경 변수 또는 extras로 변환합니다.

    yaml_cfg     — 파싱된 전체 최상위 config.yaml 딕셔너리
    platform_cfg — 플랫폼의 하위 딕셔너리 (yaml_cfg.get("my_platform", {}))

    os.environ을 직접 변경(환경 변수 우선순위를 유지하기 위해 `not os.getenv(...)` 
    가드 사용)하거나 PlatformConfig.extra에 병합할 딕셔너리를 반환할 수 있습니다. 
    추가 항목이 없다면 None 또는 {}를 반환하세요.
    """
    if "require_mention" in platform_cfg and not os.getenv("MY_PLATFORM_REQUIRE_MENTION"):
        os.environ["MY_PLATFORM_REQUIRE_MENTION"] = str(platform_cfg["require_mention"]).lower()
    allowed = platform_cfg.get("allowed_channels")
    if allowed is not None and not os.getenv("MY_PLATFORM_ALLOWED_CHANNELS"):
        if isinstance(allowed, list):
            allowed = ",".join(str(v) for v in allowed)
        os.environ["MY_PLATFORM_ALLOWED_CHANNELS"] = str(allowed)
    return None  # PlatformConfig.extra에 병합할 내용 없음

def register(ctx):
    ctx.register_platform(
        name="my_platform",
        ...,
        apply_yaml_config_fn=_apply_yaml_config,
    )
```

이 훅은 `load_gateway_config()` 중에 호출되며, 일반적인 공통 키(`unauthorized_dm_behavior`, `notice_delivery`, `reply_prefix`, `require_mention` 등) 처리 루프 이후와 `_apply_env_overrides()` 이전에 실행되므로 플러그인은 **플랫폼 전용** 키만 연결하면 됩니다.

훅에서 발생한 예외는 무시되고 디버그 수준(debug level)으로 로그에 기록되므로, 오작동하는 플러그인이 게이트웨이 구성 로드 프로세스를 중단시키지 않습니다.

## Cron 전송 (Cron Delivery)

`deliver=my_platform` 형태의 cron 작업이 구성된 홈 채널로 라우팅되도록 하려면, 기본 채팅/방/채널 ID를 저장하는 환경 변수 이름으로 `cron_deliver_env_var`를 설정하세요:

```python
ctx.register_platform(
    name="my_platform",
    ...
    cron_deliver_env_var="MY_PLATFORM_HOME_CHANNEL",
)
```

스케줄러는 `deliver=my_platform` 작업에 대한 홈 대상을 확인할 때 이 환경 변수를 읽고, `_KNOWN_DELIVERY_PLATFORMS` 스타일의 검사에서 플랫폼을 유효한 cron 대상으로 처리합니다. `env_enablement_fn`이 `home_channel` 딕셔너리를 초기화하면 이 값이 우선순위를 가지며 — `cron_deliver_env_var`는 환경 초기화 전에 실행되는 cron 작업을 위한 대체 수단입니다.

### 프로세스 외부 cron 전송 (Out-of-process cron delivery)

`cron_deliver_env_var`는 여러분의 플랫폼을 인식 가능한 `deliver=` 대상으로 만듭니다. cron 작업이 게이트웨이와 분리된 프로세스(즉, `hermes gateway`와 별개의 `hermes cron run`)에서 실행될 때 실제 전송이 성공하도록 하려면 `standalone_sender_fn`을 등록하세요:

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
    """일회성(ephemeral) 연결을 열거나 / 새 토큰을 획득하여 메시지를 보내고 닫습니다."""
    # ... 연결 열기, 메시지 전송, 결과 반환 ...
    return {"success": True, "message_id": "..."}
    # 또는 {"error": "..."}

ctx.register_platform(
    name="my_platform",
    ...
    cron_deliver_env_var="MY_PLATFORM_HOME_CHANNEL",
    standalone_sender_fn=_standalone_send,
)
```

이 훅이 필요한 이유: 기본 플랫폼(Telegram, Discord, Slack 등)은 `tools/send_message_tool.py`에 직접적인 REST 헬퍼를 포함하고 있어 게이트웨이와 동일한 프로세스를 유지하지 않아도 cron에서 전송할 수 있습니다. 반면 기존 플러그인 플랫폼은 `_gateway_runner_ref()`에 의존했기 때문에 게이트웨이 프로세스 외부에서는 이 함수가 `None`을 반환하여, `standalone_sender_fn`이 없다면 cron 전송 시 `No live adapter for platform '<name>'` 오류와 함께 실패했습니다.

이 함수는 활성 어댑터와 동일하게 `pconfig` 및 `chat_id`를 받으며, 선택적으로 `thread_id`, `media_files`, `force_document` 키워드 인수를 추가로 받습니다. `{"success": True, "message_id": ...}`를 반환하면 전송이 성공한 것으로 처리되고, `{"error": "..."}`를 반환하면 cron의 `delivery_errors`에 에러 내용이 노출됩니다. 함수 내부에서 발생한 예외는 디스패처에 의해 포착되어 `Plugin standalone send failed: <reason>`으로 보고됩니다. 참조 구현은 `plugins/platforms/{irc,teams,google_chat}/adapter.py`에 있습니다.

## `hermes config`에 환경 변수 노출하기

`hermes_cli/config.py`는 가져오기(import) 시점에 `plugins/platforms/*/plugin.yaml`을 검사하고 `requires_env` 및 (선택 사항인) `optional_env` 블록에서 `OPTIONAL_ENV_VARS`를 자동으로 채웁니다. `dict` 형식을 사용하여 적절한 설명, 프롬프트, 비밀번호 플래그, URL을 제공하면 CLI 설정 UI에서 이를 그대로 반영합니다.

```yaml
# plugins/platforms/my_platform/plugin.yaml
name: my_platform-platform
label: My Platform
kind: platform
version: 1.0.0
description: >
  Hermes Agent용 My Platform 게이트웨이 어댑터입니다.
author: Your Name
requires_env:
  - name: MY_PLATFORM_TOKEN
    description: "My Platform 콘솔에서 받은 Bot API 토큰"
    prompt: "My Platform bot token"
    url: "https://my-platform.example.com/bots"
    password: true
  - name: MY_PLATFORM_CHANNEL
    description: "참여할 채널 (예: #hermes)"
    prompt: "Channel"
    password: false
optional_env:
  - name: MY_PLATFORM_HOME_CHANNEL
    description: "cron 전송을 위한 기본 채널 (MY_PLATFORM_CHANNEL을 기본값으로 함)"
    prompt: "Home channel (or empty)"
    password: false
  - name: MY_PLATFORM_ALLOWED_USERS
    description: "봇과 대화할 수 있는 사용자 ID 목록 (쉼표로 구분)"
    prompt: "Allowed users (comma-separated)"
    password: false
```

**지원되는 딕셔너리 키:** `name` (필수), `description`, `prompt`, `url`, `password` (부울 값; 생략 시 `*_TOKEN` / `*_SECRET` / `*_KEY` / `*_PASSWORD` / `*_JSON` 접미사로 자동 감지), `category` (기본값은 `"messaging"`).

일반 문자열 항목(`- MY_PLATFORM_TOKEN`)도 작동합니다 — 이 경우 플러그인의 `label`에서 자동으로 생성된 일반적인 설명이 제공됩니다. 동일한 변수에 대해 `OPTIONAL_ENV_VARS`에 이미 하드코딩된 항목이 있는 경우 해당 항목이 우선 적용되며 (하위 호환성 유지), plugin.yaml 양식은 대체재(fallback)로 작동합니다.

## 플랫폼 전용 지연 LLM UX (Platform-Specific Slow-LLM UX)

일부 플랫폼에는 느린 LLM 응답이 표시되는 방식을 변경하는 제약 조건이 있습니다:

- **LINE**: 인바운드 이벤트 발생 후 약 60초 만료되는 일회성 *응답 토큰(reply token)*을 발급합니다. 토큰을 사용한 응답은 무료이지만, 종량제(metered) Push API로 폴백하는 것은 무료가 아닙니다. LLM이 마감 시간 전에 처리를 완료하지 못한 경우 선택지는 "유료 푸시 할당량 소진" 또는 "만료 전 응답 토큰을 영리하게 활용하기"입니다.
- **WhatsApp**: 24시간 후 세션을 비활성화 상태로 표시하며, 그 이후에는 템플릿 메시지만 허용됩니다.
- **SMS**: 타이핑 표시기나 점진적인 업데이트의 개념이 없습니다 — 응답이 늦어지면 봇이 오프라인 상태인 것처럼 보입니다.

이러한 제약은 기본 `BasePlatformAdapter`가 예측할 수 없는 실제 상황입니다. 플러그인 인터페이스는 어댑터가 kwarg 목록을 확장하지 않고도 기본 타이핑 루프 위에 플랫폼 고유 UX 레이어를 입힐 수 있는 여지를 남겨두도록 의도적으로 설계되었습니다.

### 패턴: 진행 중 UX 레이어를 입히기 위해 `_keep_typing` 서브클래싱

`BasePlatformAdapter._keep_typing`은 타이핑을 표시하는 심박(heartbeat)입니다. 이 기능은 LLM이 생성되는 동안 백그라운드 태스크로 실행되며, 응답이 전송되면 취소됩니다. 특정 임계값(예: 45초에 "아직 생각 중" 버블 전송)에 도달할 때 플랫폼 전용 동작을 레이어로 입히려면, 어댑터의 `_keep_typing`을 오버라이드하고, `super()._keep_typing()`과 함께 실행할 사용자 정의 태스크를 예약한 후 `finally` 블록에서 해당 태스크를 정리(teardown)하세요:

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
            # 여기에 플랫폼 전용 작업 삽입 — LINE의 경우 캐시된 응답 토큰을 사용하여
            # Template Buttons "답변 가져오기" 버블을 보냅니다.
            # 이 버튼의 포스트백 콜백을 통해 사용자는 새로운 (무료) 응답 토큰으로
            # 캐시된 응답을 추후 불러올 수 있습니다.
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

- **항상 `await super()._keep_typing(...)`을 호출하세요.** 타이핑 심박 표시는 그 자체로 유용합니다 — 교체하지 말고 덧입히세요.
- **`finally` 블록 안에서 보조 태스크(side task)를 정리하세요.** LLM이 완료되거나 `/stop` 명령어로 인해 실행이 취소되면 게이트웨이가 타이핑 태스크를 취소합니다. 보조 태스크도 이러한 취소 상태를 감지해야 합니다. 그렇지 않으면 응답이 이미 전송된 이후에 뒤늦게 실행될 수 있습니다.
- 사용자가 `/stop` 명령을 내릴 때 잔여 UX 상태를 해결하려면 **`interrupt_session_activity`와 함께 사용하세요.** LINE의 경우 캐시 항목 상태를 `PENDING`에서 `ERROR`로 변경하여, 화면에 남아 있는 "답변 가져오기" 버튼을 누를 때 계속 대기하지 않고 "실행이 중단되었습니다" 메시지가 나오게 해야 합니다.

### 패턴: 캐시를 통한 라우팅을 위해 `send` 서브클래싱

지연 응답 UX가 나중에 검색할 수 있도록 응답을 캐시한다면 (LINE의 포스트백 흐름), 재정의된 `send` 메서드는 다음 세 가지 모드를 인식해야 합니다:

1. **이 채팅에서 대기 중인(pending) 포스트백 버튼이 활성화됨** → 눈에 보이는 메시지를 보내지 말고, request_id로 캐시에 응답을 저장합니다.
2. **시스템 대기-수신 응답 (busy-ack)** (`⚡ 인터럽트 중`, `⏳ 대기열 추가`, `⏩ 제어됨`) → 캐시 상태에 상관없이 즉시 화면에 전송하여 사용자가 입력에 대한 게이트웨이의 반응을 볼 수 있게 합니다.
3. **일반적인 응답** → 정상적으로 응답 토큰이나 Push API를 사용하여 전송합니다.

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

`_SYSTEM_BYPASS_PREFIXES`는 게이트웨이 자체의 대기/수신확인 접두어(`⚡`, `⏳`, `⏩`, `💾`)들입니다. 캐시된 UX 상태에 관계없이, 이 접두어들은 항상 사용자 화면에 즉시 전송되어야 합니다.

### 이 패턴을 사용해야 할 때

다음 경우에 타이핑 루프 재정의(override) 접근 방식을 사용하세요:

- 플랫폼의 아웃바운드 API에 엄격한 시간 창(time-window) 제약이 있는 경우 (단일 사용 응답 토큰, 만료되는 고정 세션 등) 및
- 해당 플랫폼에서 *눈에 띄는 중간 진행 상태 메시지 버블*이 사용자 경험(UX) 측면에서 수용 가능한 경우.

다음 경우에 더 간단한 `slow_response_threshold = 0` 방식을 사용하여 항상 푸시(Push) 대체 방식을 취하세요:

- 플랫폼에 무료/유료 전송의 의미 있는 차이가 없는 경우, 또는
- 사용자들이 진행 중 표시되는 중간 대화창보다 조용히 "로딩 중... 완료" 형태로 처리되는 응답을 선호하는 경우.

LINE은 두 가지 옵션을 모두 지원합니다. 무료 포스트백 호출의 기본 임계값은 45초이며, `LINE_SLOW_RESPONSE_THRESHOLD=0`을 설정하면 "항상 푸시 전송" 모드로 되돌아갑니다.

### 참고 구현 (Reference Implementation)

완벽한 LINE 포스트백 구현의 경우 `plugins/platforms/line/adapter.py`를 참조하세요 — 상태 머신 역할을 하는 `RequestCache` (`PENDING → READY → DELIVERED`, `/stop`에 대한 `ERROR`), 임계값에 도달하면 템플릿 버튼을 표시하는 `_keep_typing` 훅, 응답을 캐시로 라우팅하는 `send` 재정의, 남겨진 `PENDING` 항목을 해결하기 위한 `interrupt_session_activity` 재정의가 포함되어 있습니다.

### 참조 구현 목록 (플러그인 경로)

외부 의존성이 전혀 없는 완전한 비동기 IRC 어댑터가 작동하는 전체 예제는 리포지토리의 `plugins/platforms/irc/`를 참조하세요. `plugins/platforms/teams/`는 Bot Framework / Adaptive Cards, `plugins/platforms/google_chat/`은 OAuth 기반 REST API, `plugins/platforms/line/`은 플랫폼 전용 지연 LLM UX가 있는 웹훅 기반 Messaging API의 예제입니다.

---

## 내장 방식 체크리스트 (Step-by-Step Checklist (Built-in Path))

:::note
이 체크리스트는 플랫폼을 Hermes 코어 코드베이스에 직접 추가하기 위한 것으로, 일반적으로 공식 지원 플랫폼에 대해 핵심 기여자가 수행합니다. 커뮤니티나 서드파티 플랫폼의 경우 위의 [플러그인 경로](#plugin-path-recommended)를 사용하는 것이 좋습니다.
:::

### 1. Platform 열거형 (Enum)

`gateway/config.py`의 `Platform` 열거형에 당신의 플랫폼을 추가하세요:

```python
class Platform(str, Enum):
    # ... 기존 플랫폼들 ...
    NEWPLAT = "newplat"
```

### 2. 어댑터 파일

`gateway/platforms/newplat.py` 생성:

```python
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter, MessageEvent, MessageType, SendResult,
)

def check_newplat_requirements() -> bool:
    """종속성을 사용할 수 있으면 True를 반환합니다."""
    return SOME_SDK_AVAILABLE

class NewPlatAdapter(BasePlatformAdapter):
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.NEWPLAT)
        # config.extra 딕셔너리에서 설정 읽기
        extra = config.extra or {}
        self._api_key = extra.get("api_key") or os.getenv("NEWPLAT_API_KEY", "")

    async def connect(self) -> bool:
        # 연결 설정, 폴링/웹훅 시작
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._running = False
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        # 플랫폼 API를 통해 메시지 전송
        return SendResult(success=True, message_id="...")

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}
```

인바운드 메시지의 경우 `MessageEvent`를 빌드하고 `self.handle_message(event)`를 호출합니다:

```python
source = self.build_source(
    chat_id=chat_id,
    chat_name=name,
    chat_type="dm",  # 또는 "group"
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

### 3. 게이트웨이 구성 (`gateway/config.py`)

세 가지 수정 지점:

1. **`get_connected_platforms()`** — 플랫폼 필수 자격 증명에 대한 검사 추가
2. **`load_gateway_config()`** — 토큰 환경 변수 매핑 추가: `Platform.NEWPLAT: "NEWPLAT_TOKEN"`
3. **`_apply_env_overrides()`** — 모든 `NEWPLAT_*` 환경 변수를 구성에 매핑

### 4. 게이트웨이 러너 (`gateway/run.py`)

다섯 가지 수정 지점:

1. **`_create_adapter()`** — `elif platform == Platform.NEWPLAT:` 분기 추가
2. **`_is_user_authorized()`의 allowed_users 맵** — `Platform.NEWPLAT: "NEWPLAT_ALLOWED_USERS"`
3. **`_is_user_authorized()`의 allow_all 맵** — `Platform.NEWPLAT: "NEWPLAT_ALLOW_ALL_USERS"`
4. **조기 환경 변수 검사 `_any_allowlist` 튜플** — `"NEWPLAT_ALLOWED_USERS"` 추가
5. **조기 환경 변수 검사 `_allow_all` 튜플** — `"NEWPLAT_ALLOW_ALL_USERS"` 추가
6. **`_UPDATE_ALLOWED_PLATFORMS` frozenset** — `Platform.NEWPLAT` 추가

### 5. 교차 플랫폼 전송 (Cross-Platform Delivery)

1. **`gateway/platforms/webhook.py`** — 전송 유형 튜플에 `"newplat"` 추가
2. **`cron/scheduler.py`** — `_KNOWN_DELIVERY_PLATFORMS` frozenset 및 `_deliver_result()` 플랫폼 맵에 추가

### 6. CLI 연동

1. **`hermes_cli/config.py`** — `_EXTRA_ENV_KEYS`에 모든 `NEWPLAT_*` 변수 추가
2. **`hermes_cli/gateway.py`** — `_PLATFORMS` 리스트에 key, label, emoji, token_var, setup_instructions, vars 추가
3. **`hermes_cli/platforms.py`** — (`skills_config` 및 `tools_config` TUI가 사용하는) label과 default_toolset을 포함한 `PlatformInfo` 항목 추가
4. **`hermes_cli/setup.py`** — `_setup_newplat()` 함수 추가 (`gateway.py`에 위임 가능) 및 메시징 플랫폼 리스트에 튜플 추가
5. **`hermes_cli/status.py`** — 플랫폼 탐지 항목 추가: `"NewPlat": ("NEWPLAT_TOKEN", "NEWPLAT_HOME_CHANNEL")`
6. **`hermes_cli/dump.py`** — 플랫폼 탐지 딕셔너리에 `"newplat": "NEWPLAT_TOKEN"` 추가

### 7. 도구 (Tools)

1. **`tools/send_message_tool.py`** — 플랫폼 맵에 `"newplat": Platform.NEWPLAT` 추가
2. **`tools/cronjob_tools.py`** — 전송 대상 설명 문자열에 `newplat` 추가

### 8. 도구 세트 (Toolsets)

1. **`toolsets.py`** — `_HERMES_CORE_TOOLS`와 함께 `"hermes-newplat"` 도구 세트 정의 추가
2. **`toolsets.py`** — `"hermes-gateway"` 포함 목록에 `"hermes-newplat"` 추가

### 9. 선택 사항: 플랫폼 힌트

**`agent/prompt_builder.py`** — 플랫폼에 특정한 렌더링 제한(마크다운 미지원, 메시지 길이 제한 등)이 있는 경우, `_PLATFORM_HINTS` 딕셔너리에 항목을 추가하세요. 이렇게 하면 시스템 프롬프트에 플랫폼 전용 가이드라인이 추가됩니다:

```python
_PLATFORM_HINTS = {
    # ...
    "newplat": (
        "You are chatting via NewPlat. It supports markdown formatting "
        "but has a 4000-character message limit."
    ),
}
```

모든 플랫폼에 힌트가 필요한 것은 아니며, 에이전트의 동작이 달라야 할 경우에만 추가하세요.

### 10. 테스트

다음을 다루는 `tests/gateway/test_newplat.py`를 생성하세요:

- 구성을 통한 어댑터 생성
- 메시지 이벤트 구성
- Send 메서드 (외부 API 모킹)
- 플랫폼 특정 기능 (암호화, 라우팅 등)

### 11. 문서화

| 파일 | 추가할 내용 |
|------|-------------|
| `website/docs/user-guide/messaging/newplat.md` | 전체 플랫폼 설정 페이지 |
| `website/docs/user-guide/messaging/index.md` | 플랫폼 비교 표, 아키텍처 다이어그램, 도구 세트 표, 보안 섹션, 다음 단계 링크 |
| `website/docs/reference/environment-variables.md` | 모든 NEWPLAT_* 환경 변수 |
| `website/docs/reference/toolsets-reference.md` | hermes-newplat 도구 세트 |
| `website/docs/integrations/index.md` | 플랫폼 링크 |
| `website/sidebars.ts` | 문서 페이지를 위한 사이드바 항목 |
| `website/docs/developer-guide/architecture.md` | 어댑터 수 및 목록 |
| `website/docs/developer-guide/gateway-internals.md` | 어댑터 파일 목록 |

## 동등성 감사 (Parity Audit)

새로운 플랫폼 PR을 완료된 것으로 표시하기 전에 기존에 확립된 플랫폼과 비교하여 동등성 감사를 실행하세요:

```bash
# 참조 플랫폼(bluebubbles)이 언급된 모든 .py 파일을 찾습니다
search_files "bluebubbles" output_mode="files_only" file_glob="*.py"

# 새 플랫폼이 언급된 모든 .py 파일을 찾습니다
search_files "newplat" output_mode="files_only" file_glob="*.py"

# 첫 번째 집합에는 있지만 두 번째 집합에는 없는 파일은 잠재적인 누락 지점입니다
```

`.md` 및 `.ts` 파일에 대해서도 이를 반복합니다. 누락된 부분을 각각 조사하세요 — 플랫폼 열거형인지(업데이트 필요), 아니면 플랫폼 전용 참조인지(건너뛰기)?

## 일반적인 패턴

### 롱 폴링(Long-Poll) 어댑터

어댑터가 롱 폴링(Telegram 또는 Weixin 등)을 사용하는 경우 폴링 루프 태스크를 사용하세요:

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

플랫폼이 메시지를 엔드포인트로 푸시하는 경우(WeCom Callback 등) HTTP 서버를 실행하세요:

```python
async def connect(self):
    self._app = web.Application()
    self._app.router.add_post("/callback", self._handle_callback)
    # ... aiohttp 서버 시작
    self._mark_connected()

async def _handle_callback(self, request):
    event = self._build_event(await request.text())
    await self._message_queue.put(event)
    return web.Response(text="success")  # 즉시 확인 응답
```

응답 기한이 엄격한 플랫폼(예: WeCom의 5초 제한)의 경우 항상 즉시 확인(ack)을 응답하고 에이전트의 답변은 나중에 API를 통해 주도적으로 전송해야 합니다. 에이전트 세션은 3~30분 동안 실행되므로 콜백 응답 창 내에서 바로 응답하는 것은 불가능합니다.

### 토큰 락 (Token Locks)

어댑터가 고유 자격 증명으로 지속적인 연결을 유지하는 경우 두 프로필이 동일한 자격 증명을 사용하는 것을 방지하기 위해 스코프 락(scoped lock)을 추가하세요:

```python
from gateway.status import acquire_scoped_lock, release_scoped_lock

async def connect(self):
    if not acquire_scoped_lock("newplat", self._token):
        logger.error("Token already in use by another profile")
        return False
    # ... 연결

async def disconnect(self):
    release_scoped_lock("newplat", self._token)
```

## 참조 구현

| 어댑터 | 패턴 | 복잡성 | 참조 목적 |
|---------|---------|------------|-------------------|
| `bluebubbles.py` | REST + webhook | 중간 | 단순한 REST API 연동 |
| `weixin.py` | Long-poll + CDN | 높음 | 미디어 처리, 암호화 |
| `wecom_callback.py` | Callback/webhook | 중간 | HTTP 서버, AES 암호, 멀티 앱 |
| `telegram.py` | Long-poll + Bot API | 높음 | 그룹, 스레드를 포함한 모든 기능을 갖춘 어댑터 |
