---
title: "Openhue — Control Philips Hue lights, scenes, rooms via OpenHue CLI"
sidebar_label: "Openhue"
description: "Control Philips Hue lights, scenes, rooms via OpenHue CLI"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Openhue

OpenHue CLI를 통해 Philips Hue 조명, 장면(scenes), 방(rooms) 제어하기.

## 스킬 메타데이터

| | |
|---|---|
| Source | 번들 (기본 설치) |
| Path | `skills/smart-home/openhue` |
| Version | `1.0.0` |
| Author | community |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Smart-Home`, `Hue`, `Lights`, `IoT`, `Automation` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# OpenHue CLI

터미널에서 Hue 브리지를 통해 Philips Hue 조명과 장면을 제어합니다.

## 사전 요구 사항

```bash
# Linux (미리 빌드된 바이너리)
curl -sL https://github.com/openhue/openhue-cli/releases/latest/download/openhue-linux-amd64 -o ~/.local/bin/openhue && chmod +x ~/.local/bin/openhue

# macOS
brew install openhue/cli/openhue-cli
```

처음 실행할 때는 페어링을 위해 Hue 브리지의 버튼을 눌러야 합니다. 브리지는 동일한 로컬 네트워크 상에 있어야 합니다.

## 사용 시기

- "조명 켜줘/꺼줘"
- "거실 조명 어둡게 해줘"
- "장면 설정해줘" 또는 "영화 모드"
- 특정 Hue 방(rooms), 구역(zones) 또는 개별 전구 제어
- 밝기, 색상 또는 색온도 조절

## 일반적인 명령어

### 리소스 목록 보기

```bash
openhue get light       # 모든 조명 목록
openhue get room        # 모든 방 목록
openhue get scene       # 모든 장면 목록
```

### 조명 제어

```bash
# 켜기/끄기
openhue set light "Bedroom Lamp" --on
openhue set light "Bedroom Lamp" --off

# 밝기 (0-100)
openhue set light "Bedroom Lamp" --on --brightness 50

# 색온도 (따뜻한 색에서 차가운 색: 153-500 mirek)
openhue set light "Bedroom Lamp" --on --temperature 300

# 색상 (이름 또는 16진수)
openhue set light "Bedroom Lamp" --on --color red
openhue set light "Bedroom Lamp" --on --rgb "#FF5500"
```

### 방 제어

```bash
# 방 전체 끄기
openhue set room "Bedroom" --off

# 방 밝기 설정
openhue set room "Bedroom" --on --brightness 30
```

### 장면 (Scenes)

```bash
openhue set scene "Relax" --room "Bedroom"
openhue set scene "Concentrate" --room "Office"
```

### 빠른 프리셋 (Presets)

```bash
# 취침 시간 (어둡고 따뜻하게)
openhue set room "Bedroom" --on --brightness 20 --temperature 450

# 작업 모드 (밝고 차갑게)
openhue set room "Office" --on --brightness 100 --temperature 250

# 영화 모드 (어둡게)
openhue set room "Living Room" --on --brightness 10

# 모두 끄기
openhue set room "Bedroom" --off
openhue set room "Office" --off
openhue set room "Living Room" --off
```

## 참고 사항

- 브리지는 Hermes를 실행하는 머신과 동일한 로컬 네트워크 상에 있어야 합니다.
- 첫 실행 시 권한 부여를 위해 Hue 브리지의 버튼을 물리적으로 눌러야 합니다.
- 색상 조절은 컬러 지원 전구에서만 작동합니다 (백색 전용 모델 제외).
- 조명과 방 이름은 대소문자를 구분합니다 — 정확한 이름은 `openhue get light`를 사용하여 확인하세요.
- 조명 예약 관리를 위한 cron 작업과 아주 잘 작동합니다 (예: 취침 시 어둡게, 기상 시 밝게).
