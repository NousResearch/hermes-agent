---
title: Home Assistant
description: Control your smart home with Hermes Agent via Home Assistant integration.
sidebar_label: Home Assistant
sidebar_position: 5
---

# Home Assistant 통합

Hermes Agent는 두 가지 방법으로 [Home Assistant](https://www.home-assistant.io/)와 통합됩니다:

1. **게이트웨이 플랫폼** — 웹소켓(WebSocket)을 통해 실시간 상태 변경을 구독하고 이벤트에 응답합니다.
2. **스마트 홈 도구(tools)** — REST API를 통해 장치를 쿼리하고 제어하기 위한 4개의 LLM 호출 가능 도구입니다.

## 설정

### 1. 장기 실행 액세스 토큰(Long-Lived Access Token) 생성

1. Home Assistant 인스턴스를 엽니다.
2. **프로필(Profile)**로 이동합니다 (사이드바에서 이름 클릭).
3. **장기 실행 액세스 토큰(Long-Lived Access Tokens)**으로 스크롤합니다.
4. **토큰 만들기(Create Token)**를 클릭하고 "Hermes Agent"와 같은 이름을 지정합니다.
5. 토큰을 복사합니다.

### 2. 환경 변수 구성

```bash
# ~/.hermes/.env에 추가

# 필수: 장기 실행 액세스 토큰
HASS_TOKEN=your-long-lived-access-token

# 선택 사항: HA URL (기본값: http://homeassistant.local:8123)
HASS_URL=http://192.168.1.100:8123
```

:::info
`HASS_TOKEN`이 설정되면 `homeassistant` 툴셋이 자동으로 활성화됩니다. 게이트웨이 플랫폼과 기기 제어 도구 모두 이 단일 토큰으로 활성화됩니다.
:::

### 3. 게이트웨이 시작

```bash
hermes gateway
```

Home Assistant는 다른 메시징 플랫폼(Telegram, Discord 등)과 함께 연결된 플랫폼으로 표시됩니다.

## 사용 가능한 도구 (Available Tools)

Hermes Agent는 스마트 홈 제어를 위해 4개의 도구를 등록합니다:

### `ha_list_entities`

도메인이나 영역(area)별로 선택적으로 필터링하여 Home Assistant 엔터티 목록을 나열합니다.

**매개변수:**
- `domain` *(선택)* — 엔터티 도메인별 필터링: `light`, `switch`, `climate`, `sensor`, `binary_sensor`, `cover`, `fan`, `media_player` 등.
- `area` *(선택)* — 영역/방 이름별 필터링(friendly name과 일치하는지 확인): `living room`, `kitchen`, `bedroom` 등.

**예시:**
```
거실의 모든 조명 나열해줘
```

엔터티 ID, 상태 및 친숙한 이름(friendly name)을 반환합니다.

### `ha_get_state`

모든 속성(밝기, 색상, 설정 온도, 센서 판독값 등)을 포함하여 단일 엔터티의 세부 상태를 가져옵니다.

**매개변수:**
- `entity_id` *(필수)* — 쿼리할 엔터티. 예: `light.living_room`, `climate.thermostat`, `sensor.temperature`

**예시:**
```
climate.thermostat의 현재 상태는 뭐야?
```

반환값: 상태, 모든 속성, 마지막 변경/업데이트 타임스탬프.

### `ha_list_services`

장치 제어에 사용할 수 있는 서비스(작업)를 나열합니다. 각 장치 유형에서 수행할 수 있는 작업과 허용하는 매개변수를 보여줍니다.

**매개변수:**
- `domain` *(선택)* — 도메인별 필터링. 예: `light`, `climate`, `switch`

**예시:**
```
climate 장치에는 어떤 서비스가 제공돼?
```

### `ha_call_service`

장치를 제어하기 위해 Home Assistant 서비스를 호출합니다.

**매개변수:**
- `domain` *(필수)* — 서비스 도메인: `light`, `switch`, `climate`, `cover`, `media_player`, `fan`, `scene`, `script`
- `service` *(필수)* — 서비스 이름: `turn_on`, `turn_off`, `toggle`, `set_temperature`, `set_hvac_mode`, `open_cover`, `close_cover`, `set_volume_level`
- `entity_id` *(선택)* — 대상 엔터티. 예: `light.living_room`
- `data` *(선택)* — JSON 객체 형태의 추가 매개변수

**예시:**

```
거실 조명 켜
→ ha_call_service(domain="light", service="turn_on", entity_id="light.living_room")
```

```
온도 조절기를 난방 모드(heat mode)로 22도로 설정해
→ ha_call_service(domain="climate", service="set_temperature",
    entity_id="climate.thermostat", data={"temperature": 22, "hvac_mode": "heat"})
```

```
거실 조명을 밝기 50%의 파란색으로 설정해
→ ha_call_service(domain="light", service="turn_on",
    entity_id="light.living_room", data={"brightness": 128, "color_name": "blue"})
```

## 게이트웨이 플랫폼: 실시간 이벤트

Home Assistant 게이트웨이 어댑터는 웹소켓(WebSocket)을 통해 연결하고 `state_changed` 이벤트를 구독합니다. 장치 상태가 변경되고 필터와 일치하면 이벤트가 에이전트에 메시지로 전달됩니다.

### 이벤트 필터링

:::warning 필수 구성
기본적으로 **이벤트는 전달되지 않습니다**. 이벤트를 수신하려면 `watch_domains`, `watch_entities` 또는 `watch_all` 중 하나 이상을 구성해야 합니다. 필터가 없으면 시작 시 경고가 기록되고 모든 상태 변경이 조용히 삭제됩니다.
:::

`~/.hermes/config.yaml` 파일의 Home Assistant 플랫폼 `extra` 섹션에서 에이전트가 볼 수 있는 이벤트를 구성합니다:

```yaml
platforms:
  homeassistant:
    enabled: true
    extra:
      watch_domains:
        - climate
        - binary_sensor
        - alarm_control_panel
        - light
      watch_entities:
        - sensor.front_door_battery
      ignore_entities:
        - sensor.uptime
        - sensor.cpu_usage
        - sensor.memory_usage
      cooldown_seconds: 30
```

| 설정 | 기본값 | 설명 |
|---------|---------|-------------|
| `watch_domains` | *(없음)* | 지정된 엔터티 도메인만 감시합니다 (예: `climate`, `light`, `binary_sensor`) |
| `watch_entities` | *(없음)* | 지정된 특정 엔터티 ID만 감시합니다 |
| `watch_all` | `false` | **모든** 상태 변경 이벤트를 수신하려면 `true`로 설정하세요 (대부분의 설정에는 권장되지 않음) |
| `ignore_entities` | *(없음)* | 지정된 엔터티를 항상 무시합니다 (도메인/엔터티 필터 전에 적용됨) |
| `cooldown_seconds` | `30` | 동일한 엔터티의 이벤트 간 최소 간격(초) |

:::tip
먼저 `climate`, `binary_sensor`, `alarm_control_panel`과 같이 유용한 자동화를 대부분 커버하는 핵심 도메인들로 시작해 보세요. 필요한 도메인을 점진적으로 추가하세요. `ignore_entities`를 사용하면 CPU 온도나 업타임 카운터와 같이 노이즈가 많은 센서를 억제할 수 있습니다.
:::

### 이벤트 형식

상태 변경 이벤트는 도메인에 따라 사람이 읽기 쉬운 형식의 메시지로 변환됩니다:

| 도메인 | 형식 |
|--------|--------|
| `climate` | "HVAC mode changed from 'off' to 'heat' (current: 21, target: 23)" |
| `sensor` | "changed from 21°C to 22°C" |
| `binary_sensor` | "triggered" / "cleared" |
| `light`, `switch`, `fan` | "turned on" / "turned off" |
| `alarm_control_panel` | "alarm state changed from 'armed_away' to 'triggered'" |
| *(other)* | "changed from 'old' to 'new'" |

### 에이전트 응답

에이전트가 보내는 발신 메시지는 **Home Assistant 지속 알림(persistent notifications)**으로 전달됩니다 (`persistent_notification.create` 사용). 이는 HA 알림 패널에 "Hermes Agent"라는 제목으로 표시됩니다.

### 연결 관리

- 실시간 이벤트를 위한 30초 하트비트가 포함된 **웹소켓(WebSocket)**
- 백오프가 적용된 **자동 재연결**: 5초 → 10초 → 30초 → 60초
- 발신 알림을 위한 **REST API** (웹소켓 충돌을 피하기 위한 별도 세션)
- **권한 부여** — HA 이벤트는 항상 승인됩니다 (`HASS_TOKEN`이 연결을 인증하므로 사용자 허용 목록이 필요하지 않음)

## 보안

Home Assistant 도구는 다음의 보안 제한을 강제합니다:

:::warning 차단된 도메인
HA 호스트에서 임의의 코드 실행을 방지하기 위해 다음 서비스 도메인이 **차단**됩니다:

- `shell_command` — 임의 셸 명령
- `command_line` — 명령을 실행하는 센서/스위치
- `python_script` — 스크립트 기반 Python 실행
- `pyscript` — 광범위한 스크립팅 통합
- `hassio` — 애드온 제어, 호스트 종료/재부팅
- `rest_command` — HA 서버에서의 HTTP 요청 (SSRF 백터)

이러한 도메인의 서비스를 호출하려고 하면 오류가 반환됩니다.
:::

엔터티 ID는 인젝션 공격을 방지하기 위해 `^[a-z_][a-z0-9_]*\.[a-z0-9_]+$` 패턴에 맞는지 검증됩니다.

## 예시 자동화

### 아침 루틴 (Morning Routine)

```
User: Start my morning routine

Agent:
1. ha_call_service(domain="light", service="turn_on",
     entity_id="light.bedroom", data={"brightness": 128})
2. ha_call_service(domain="climate", service="set_temperature",
     entity_id="climate.thermostat", data={"temperature": 22})
3. ha_call_service(domain="media_player", service="turn_on",
     entity_id="media_player.kitchen_speaker")
```

### 보안 확인 (Security Check)

```
User: Is the house secure?

Agent:
1. ha_list_entities(domain="binary_sensor")
     → 문/창문 센서 확인
2. ha_get_state(entity_id="alarm_control_panel.home")
     → 알람 상태 확인
3. ha_list_entities(domain="lock")
     → 잠금 상태 확인
4. Reports: "All doors closed, alarm is armed_away, all locks engaged."
```

### 반응형 자동화 (게이트웨이 이벤트 활용)

게이트웨이 플랫폼으로 연결된 경우 에이전트는 이벤트에 반응할 수 있습니다:

```
[Home Assistant] Front Door: triggered (was cleared)

Agent automatically:
1. ha_get_state(entity_id="binary_sensor.front_door")
2. ha_call_service(domain="light", service="turn_on",
     entity_id="light.hallway")
3. Sends notification: "Front door opened. Hallway lights turned on."
```

## 문제 해결

**환경 변수가 반영되지 않음.**
어댑터는 시작 시 자동 병합되는 `~/.hermes/.env` 또는 `config.yaml`에서 자격 증명을 읽습니다. 해당 파일이 활성 Hermes 프로필 홈 디렉터리 아래에 있는지, URL/토큰 주위에 불필요한 따옴표가 없는지 다시 확인하세요. 편집 후에는 게이트웨이를 재시작해야 합니다 — 환경 변수 변경 사항은 프로세스 시작 시에만 적용됩니다.

**`conversation entity not found` / 에이전트가 응답하지 않음.**
Home Assistant의 대화 API를 사용하려면 설정된 *Assist* 대화 에이전트가 필요합니다. HA에서 **설정(Settings) → 음성 비서(Voice assistants) → 비서 추가(Add assistant)**로 이동하여 결과로 나타나는 엔터티 ID를 확인하세요 (예: `conversation.home_assistant` 또는 `conversation.openai_<name>`). 어댑터의 `conversation_entity` 설정에 해당 엔터티 ID를 설정하세요. 인스턴스에 기본값이 존재하지 않을 수 있습니다.

**REST 인증 실패 (`401 Unauthorized`).**
토큰은 HA 사용자 프로필 페이지(**프로필 → 보안 → 장기 실행 액세스 토큰**)에서 생성된 *장기 실행 액세스 토큰(Long-Lived Access Token)*이어야 합니다. 수명이 짧은 UI 세션 토큰은 작동하지 않습니다. 또한 기본 URL이 스킴과 포트를 모두 포함하는지(예: `http://homeassistant.local:8123`) 그리고 Hermes를 실행하는 호스트에서 도달할 수 있는지 확인하세요. `curl -H "Authorization: Bearer <token>" <url>/api/` 명령을 실행하면 `{"message": "API running."}`이 반환되어야 합니다.
