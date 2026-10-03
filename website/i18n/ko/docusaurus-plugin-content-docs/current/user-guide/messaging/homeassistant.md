---
title: Home Assistant
description: Home Assistant 통합을 통해 Hermes Agent로 스마트 홈을 제어합니다.
sidebar_label: Home Assistant
sidebar_position: 5
---

# Home Assistant 통합

Hermes Agent는 [Home Assistant](https://www.home-assistant.io/)와 두 가지 방식으로 통합됩니다.

1. **Gateway 플랫폼** — WebSocket을 통해 실시간 상태 변경을 구독하고 이벤트에 응답합니다
2. **스마트 홈 도구** — REST API를 통해 장치를 조회하고 제어하는 LLM 호출 가능 도구 4개

## 설정

### 1. 장기 액세스 토큰 생성

1. Home Assistant 인스턴스를 엽니다
2. **프로필**(사이드바에서 이름 클릭)로 이동합니다
3. **장기 액세스 토큰**까지 스크롤합니다
4. **토큰 생성**을 클릭하고 "Hermes Agent"와 같은 이름을 지정합니다
5. 토큰을 복사합니다

### 2. 환경 변수 설정

```bash
# Add to ~/.hermes/.env

# Required: your Long-Lived Access Token
HASS_TOKEN=your-long-lived-access-token

# Optional: HA URL (default: http://homeassistant.local:8123)
HASS_URL=http://192.168.1.100:8123
```

:::info
`HASS_TOKEN`이 설정되면 `homeassistant` 도구 세트가 자동으로 활성화됩니다. Gateway 플랫폼과 장치 제어 도구가 이 하나의 토큰으로 모두 활성화됩니다.
:::

### 3. Gateway 시작

```bash
hermes gateway
```

Home Assistant가 다른 메시징 플랫폼(Telegram, Discord 등)과 함께 연결된 플랫폼으로 표시됩니다.

## 사용 가능한 도구

Hermes Agent는 스마트 홈 제어를 위해 네 가지 도구를 등록합니다.

### `ha_list_entities`

Home Assistant 엔터티를 나열하며, 필요한 경우 도메인이나 영역으로 필터링합니다.

**매개변수:**
- `domain` *(선택 사항)* — 엔터티 도메인으로 필터링합니다: `light`, `switch`, `climate`, `sensor`, `binary_sensor`, `cover`, `fan`, `media_player` 등
- `area` *(선택 사항)* — 영역/방 이름으로 필터링합니다(친숙한 이름과 일치): `living room`, `kitchen`, `bedroom` 등

**예시:**
```
List all lights in the living room
```

엔터티 ID, 상태 및 친숙한 이름을 반환합니다.

### `ha_get_state`

단일 엔터티의 상세 상태를 가져옵니다. 밝기, 색상, 온도 설정값, 센서 측정값 등 모든 속성이 포함됩니다.

**매개변수:**
- `entity_id` *(필수)* — 조회할 엔터티입니다. 예: `light.living_room`, `climate.thermostat`, `sensor.temperature`

**예시:**
```
What's the current state of climate.thermostat?
```

반환값: 상태, 모든 속성, 마지막 변경/업데이트 타임스탬프

### `ha_list_services`

장치 제어에 사용할 수 있는 서비스(동작)를 나열합니다. 각 장치 유형에서 수행할 수 있는 동작과 허용되는 매개변수를 보여줍니다.

**매개변수:**
- `domain` *(선택 사항)* — 도메인으로 필터링합니다. 예: `light`, `climate`, `switch`

**예시:**
```
What services are available for climate devices?
```

### `ha_call_service`

Home Assistant 서비스를 호출하여 장치를 제어합니다.

**매개변수:**
- `domain` *(필수)* — 서비스 도메인: `light`, `switch`, `climate`, `cover`, `media_player`, `fan`, `scene`, `script`
- `service` *(필수)* — 서비스 이름: `turn_on`, `turn_off`, `toggle`, `set_temperature`, `set_hvac_mode`, `open_cover`, `close_cover`, `set_volume_level`
- `entity_id` *(선택 사항)* — 대상 엔터티. 예: `light.living_room`
- `data` *(선택 사항)* — JSON 객체 형식의 추가 매개변수

**예시:**

```
Turn on the living room lights
→ ha_call_service(domain="light", service="turn_on", entity_id="light.living_room")
```

```
Set the thermostat to 22 degrees in heat mode
→ ha_call_service(domain="climate", service="set_temperature",
    entity_id="climate.thermostat", data={"temperature": 22, "hvac_mode": "heat"})
```

```
Set living room lights to blue at 50% brightness
→ ha_call_service(domain="light", service="turn_on",
    entity_id="light.living_room", data={"brightness": 128, "color_name": "blue"})
```

## Gateway 플랫폼: 실시간 이벤트

Home Assistant Gateway 어댑터는 WebSocket으로 연결하고 `state_changed` 이벤트를 구독합니다. 장치 상태가 변경되고 설정한 필터와 일치하면 해당 변경 사항이 메시지로 에이전트에 전달됩니다.

### 이벤트 필터링

:::warning 필수 설정
기본적으로 **어떤 이벤트도 전달되지 않습니다**. 이벤트를 받으려면 `watch_domains`, `watch_entities`, `watch_all` 중 하나 이상을 설정해야 합니다. 필터가 없으면 시작 시 경고가 기록되고 모든 상태 변경이 조용히 삭제됩니다.
:::

`~/.hermes/config.yaml`의 Home Assistant 플랫폼 `extra` 섹션에서 에이전트가 확인할 이벤트를 설정합니다.

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
| `watch_domains` | *(없음)* | 이 엔터티 도메인만 감시합니다(예: `climate`, `light`, `binary_sensor`) |
| `watch_entities` | *(없음)* | 이 특정 엔터티 ID만 감시합니다 |
| `watch_all` | `false` | 모든 상태 변경을 받으려면 `true`로 설정합니다(대부분의 설정에는 권장하지 않음) |
| `ignore_entities` | *(없음)* | 항상 이 엔터티들을 무시합니다(도메인/엔터티 필터보다 먼저 적용) |
| `cooldown_seconds` | `30` | 동일한 엔터티에 대한 이벤트 사이의 최소 시간(초) |

:::tip
집중된 도메인 집합으로 시작하세요 — `climate`, `binary_sensor`, `alarm_control_panel`은 가장 유용한 자동화 대부분을 다룹니다. 필요에 따라 더 추가하세요. CPU 온도나 가동 시간 카운터처럼 이벤트가 잦은 센서를 억제하려면 `ignore_entities`를 사용하세요.
:::

### 이벤트 형식

상태 변경은 도메인에 따라 사람이 읽을 수 있는 메시지로 형식화됩니다.

| 도메인 | 형식 |
|--------|--------|
| `climate` | "HVAC mode changed from 'off' to 'heat' (current: 21, target: 23)" |
| `sensor` | "changed from 21°C to 22°C" |
| `binary_sensor` | "triggered" / "cleared" |
| `light`, `switch`, `fan` | "turned on" / "turned off" |
| `alarm_control_panel` | "alarm state changed from 'armed_away' to 'triggered'" |
| *(기타)* | "changed from 'old' to 'new'" |

### 에이전트 응답

에이전트의 발신 메시지는 **Home Assistant 영구 알림**(`persistent_notification.create`)으로 전달됩니다. 알림 패널에 "Hermes Agent"라는 제목으로 표시됩니다.

### 연결 관리

- 실시간 이벤트를 위한 **WebSocket** 및 30초 하트비트
- 백오프를 적용한 **자동 재연결**: 5초 → 10초 → 30초 → 60초
- 발신 알림을 위한 **REST API**(WebSocket 충돌을 방지하기 위한 별도 세션)
- **인증** — `HASS_TOKEN`이 연결을 인증하므로 HA 이벤트는 항상 인증됩니다(사용자 허용 목록 불필요)

## 보안

Home Assistant 도구는 다음과 같은 보안 제한을 적용합니다.

:::warning 차단된 도메인
HA 호스트에서 임의의 코드가 실행되는 것을 방지하기 위해 다음 서비스 도메인은 **차단됩니다**:

- `shell_command` — 임의의 셸 명령
- `command_line` — 명령을 실행하는 센서/스위치
- `python_script` — Python 스크립트 실행
- `pyscript` — 더 광범위한 스크립팅 통합
- `hassio` — 애드온 제어, 호스트 종료/재부팅
- `rest_command` — HA 서버에서 보내는 HTTP 요청(SSRF 벡터)

이 도메인에서 서비스를 호출하려고 하면 오류가 반환됩니다.
:::

인젝션 공격을 방지하기 위해 엔터티 ID는 `^[a-z_][a-z0-9_]*\.[a-z0-9_]+$` 패턴에 따라 검증됩니다.

## 자동화 예시

### 아침 루틴

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

### 보안 확인

```
User: Is the house secure?

Agent:
1. ha_list_entities(domain="binary_sensor")
     → checks door/window sensors
2. ha_get_state(entity_id="alarm_control_panel.home")
     → checks alarm status
3. ha_list_entities(domain="lock")
     → checks lock states
4. Reports: "All doors closed, alarm is armed_away, all locks engaged."
```

### 반응형 자동화(Gateway 이벤트 사용)

Gateway 플랫폼으로 연결되면 에이전트가 이벤트에 반응할 수 있습니다:

```
[Home Assistant] Front Door: triggered (was cleared)

Agent automatically:
1. ha_get_state(entity_id="binary_sensor.front_door")
2. ha_call_service(domain="light", service="turn_on",
     entity_id="light.hallway")
3. Sends notification: "Front door opened. Hallway lights turned on."
```

## 문제 해결

**환경 변수가 반영되지 않습니다.**
어댑터는 `~/.hermes/.env`(시작 시 자동 병합) 또는
`config.yaml`에서 인증 정보를 읽습니다. 파일이 활성 Hermes 프로필
홈 아래에 있는지, URL/토큰 주위에 불필요한 따옴표가 없는지 확인하세요. 수정 후 Gateway를
다시 시작하세요 — 환경 변수 변경은 프로세스 시작 시에만 적용됩니다.
