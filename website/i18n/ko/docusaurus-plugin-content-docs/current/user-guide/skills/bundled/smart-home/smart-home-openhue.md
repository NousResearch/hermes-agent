---
title: "Openhue — OpenHue CLI로 Philips Hue 조명, 장면, 방 제어"
sidebar_label: "Openhue"
description: "OpenHue CLI로 Philips Hue 조명, 장면, 방 제어"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Openhue

OpenHue CLI로 Philips Hue 조명, 장면, 방을 제어합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들됨(기본 설치) |
| 경로 | `skills/smart-home/openhue` |
| 버전 | `1.0.1` |
| 작성자 | community |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Smart-Home`, `Hue`, `Lights`, `IoT`, `Automation` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# OpenHue CLI

터미널에서 Hue Bridge를 통해 Philips Hue 조명과 장면을 제어합니다.

## 사전 요구 사항

```bash
# Linux (pre-built binary — releases ship tarballs, not bare binaries)
curl -sL "https://github.com/openhue/openhue-cli/releases/latest/download/openhue_Linux_x86_64.tar.gz" \
  | tar -xz -C /tmp openhue \
  && install -m 0755 /tmp/openhue ~/.local/bin/openhue
# (use openhue_Linux_arm64.tar.gz on ARM64)

# macOS
brew install openhue/cli/openhue-cli
```

처음 실행할 때는 Hue Bridge의 버튼을 눌러 페어링해야 합니다. 브리지는 동일한 로컬 네트워크에 있어야 합니다.

## 사용 시점

- "조명 켜기/끄기"
- "거실 조명 어둡게 하기"
- "장면 설정" 또는 "영화 모드"
- 특정 Hue 방, 구역 또는 개별 전구 제어
- 밝기, 색상 또는 색온도 조정

## 주요 명령

### 리소스 나열

```bash
openhue get light       # List all lights
openhue get room        # List all rooms
openhue get scene       # List all scenes
```

### 조명 제어

```bash
# Turn on/off
openhue set light "Bedroom Lamp" --on
openhue set light "Bedroom Lamp" --off

# Brightness (0-100)
openhue set light "Bedroom Lamp" --on --brightness 50

# Color temperature (warm to cool: 153-500 mirek)
openhue set light "Bedroom Lamp" --on --temperature 300

# Color (by name or hex)
openhue set light "Bedroom Lamp" --on --color red
openhue set light "Bedroom Lamp" --on --rgb "#FF5500"
```

### 방 제어

```bash
# Turn off entire room
openhue set room "Bedroom" --off

# Set room brightness
openhue set room "Bedroom" --on --brightness 30
```

### 장면

```bash
openhue set scene "Relax" --room "Bedroom"
openhue set scene "Concentrate" --room "Office"
```

## 빠른 프리셋

```bash
# Bedtime (dim warm)
openhue set room "Bedroom" --on --brightness 20 --temperature 450

# Work mode (bright cool)
openhue set room "Office" --on --brightness 100 --temperature 250

# Movie mode (dim)
openhue set room "Living Room" --on --brightness 10

# Everything off
openhue set room "Bedroom" --off
openhue set room "Office" --off
openhue set room "Living Room" --off
```

## 참고

- 브리지는 Hermes를 실행하는 컴퓨터와 동일한 로컬 네트워크에 있어야 합니다.
- 처음 실행할 때는 인증을 위해 Hue Bridge의 버튼을 직접 눌러야 합니다.
- 색상은 색상 지원 전구에서만 작동합니다(흰색 전용 모델 제외).
- 조명 및 방 이름은 대소문자를 구분합니다. 정확한 이름은 `openhue get light`로 확인하세요.
- 예약 조명(예: 취침 시간에 어둡게, 기상 시간에 밝게)을 위한 cron 작업과 잘 어울립니다.
