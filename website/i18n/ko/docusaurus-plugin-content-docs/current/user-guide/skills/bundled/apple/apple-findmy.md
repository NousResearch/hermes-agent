---
title: "Findmy — macOS의 FindMy.app으로 Apple 기기/AirTag 추적"
sidebar_label: "Findmy"
description: "macOS의 FindMy.app으로 Apple 기기/AirTag 추적"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Findmy

macOS의 FindMy.app으로 Apple 기기/AirTag을 추적합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 제공(기본 설치됨) |
| 경로 | `skills/apple/findmy` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | macos |
| 태그 | `FindMy`, `AirTag`, `location`, `tracking`, `macOS`, `Apple` |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# 나의 찾기(Apple)

macOS의 FindMy.app을 통해 Apple 기기와 AirTag을 추적합니다. Apple은 FindMy용 CLI를
제공하지 않으므로, 이 스킬은 AppleScript로 앱을 열고 화면 캡처를 사용해 기기 위치를
읽습니다.

## 사전 요구 사항

- **Find My 앱이 설치되어 있고 iCloud에 로그인된 macOS**
- Find My에 기기/AirTag이 이미 등록되어 있어야 함
- 터미널의 화면 기록 권한(System Settings → Privacy → Screen Recording)
- **선택 사항이지만 권장**: 더 나은 UI 자동화를 위해 `peekaboo` 설치:
  `brew install steipete/tap/peekaboo`

## 사용 시점

- 사용자가 "내 [기기/고양이/열쇠/가방]이 어디 있어?"라고 물을 때
- AirTag 위치 추적
- 기기 위치 확인(iPhone, iPad, Mac, AirPods)
- 반려동물 또는 물품의 이동을 시간에 따라 모니터링(AirTag 순찰 경로)

## 방법 1: AppleScript + 스크린샷(기본)

### FindMy 열기 및 탐색

```bash
# Open Find My app
osascript -e 'tell application "FindMy" to activate'

# Wait for it to load
sleep 3

# Take a screenshot of the Find My window
screencapture -w -o /tmp/findmy.png
```

그런 다음 `vision_analyze`를 사용해 스크린샷을 읽습니다.
```
vision_analyze(image_url="/tmp/findmy.png", question="What devices/items are shown and what are their locations?")
```

### 탭 간 전환

```bash
# Switch to Devices tab
osascript -e '
tell application "System Events"
    tell process "FindMy"
        click button "Devices" of toolbar 1 of window 1
    end tell
end tell'

# Switch to Items tab (AirTags)
osascript -e '
tell application "System Events"
    tell process "FindMy"
        click button "Items" of toolbar 1 of window 1
    end tell
end tell'
```

## 방법 2: Peekaboo UI 자동화(권장)

`peekaboo`가 설치되어 있다면 더 안정적인 UI 상호작용을 위해 사용합니다.

```bash
# Open Find My
osascript -e 'tell application "FindMy" to activate'
sleep 3

# Capture and annotate the UI
peekaboo see --app "FindMy" --annotate --path /tmp/findmy-ui.png

# Click on a specific device/item by element ID
peekaboo click --on B3 --app "FindMy"

# Capture the detail view
peekaboo image --app "FindMy" --path /tmp/findmy-detail.png
```

그런 다음 vision으로 분석합니다.
```
vision_analyze(image_url="/tmp/findmy-detail.png", question="What is the location shown for this device/item? Include address and coordinates if visible.")
```

## 워크플로: 시간에 따른 AirTag 위치 추적

AirTag(예: 고양이의 순찰 경로 추적)를 모니터링할 때:

```bash
# 1. Open FindMy to Items tab
osascript -e 'tell application "FindMy" to activate'
sleep 3

# 2. Click on the AirTag item (stay on page — AirTag only updates when page is open)

# 3. Periodically capture location
while true; do
    screencapture -w -o /tmp/findmy-$(date +%H%M%S).png
    sleep 300  # Every 5 minutes
done
```

각 스크린샷을 vision으로 분석해 좌표를 추출한 다음 경로를 정리합니다.

## 제한 사항

- FindMy에는 **CLI 또는 API가 없으므로** UI 자동화를 사용해야 함
- FindMy 페이지가 활성 상태로 표시되는 동안에만 AirTag 위치가 업데이트됨
- 위치 정확도는 FindMy 네트워크에서 주변 Apple 기기에 따라 달라짐
- 스크린샷을 찍으려면 화면 기록 권한이 필요함
- AppleScript UI 자동화는 macOS 버전 간에 작동하지 않을 수 있음

## 규칙

1. AirTag을 추적할 때 FindMy 앱을 전경에 유지합니다(최소화하면 업데이트가 중지됨).
2. 스크린샷 내용을 읽을 때 `vision_analyze`를 사용합니다 — 픽셀을 직접 파싱하지 마세요.
3. 지속적으로 추적하려면 cronjob을 사용해 주기적으로 위치를 캡처하고 기록합니다.
4. 개인정보를 존중하세요 — 사용자가 소유한 기기/물품만 추적합니다.
