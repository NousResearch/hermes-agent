---
title: "Findmy — FindMy를 통한 Apple 기기/AirTag 추적"
sidebar_label: "Findmy"
description: "FindMy를 통한 Apple 기기/AirTag 추적"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py 스크립트를 통해 해당 스킬의 SKILL.md에서 자동 생성됩니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Findmy

macOS의 FindMy.app을 통해 Apple 기기/AirTag를 추적합니다.

## 스킬 메타데이터 (Skill metadata)

| | |
|---|---|
| 출처 (Source) | 번들 (기본적으로 설치됨) |
| 경로 (Path) | `skills/apple/findmy` |
| 버전 (Version) | `1.0.0` |
| 작성자 (Author) | Hermes Agent |
| 라이선스 (License) | MIT |
| 플랫폼 (Platforms) | macos |
| 태그 (Tags) | `FindMy`, `AirTag`, `location`, `tracking`, `macOS`, `Apple` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것이 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Find My (Apple)

macOS의 FindMy.app을 통해 Apple 기기 및 AirTag를 추적합니다. Apple은 FindMy에 대한 CLI를 제공하지 않으므로, 이 스킬은 AppleScript를 사용하여 앱을 열고 화면 캡처를 사용하여 기기 위치를 읽습니다.

## 전제 조건 (Prerequisites)

- Find My 앱이 있고 iCloud에 로그인된 **macOS**
- Find My에 이미 등록된 기기/AirTag
- 터미널에 대한 화면 기록(Screen Recording) 권한 (시스템 설정 → 개인정보 보호 및 보안 → 화면 기록)
- **선택 사항이지만 권장됨**: 더 나은 UI 자동화를 위해 `peekaboo` 설치:
  `brew install steipete/tap/peekaboo`

## 사용 시기 (When to Use)

- 사용자가 "내 [기기/고양이/열쇠/가방]은(는) 어디에 있어?"라고 물을 때
- AirTag 위치 추적
- 기기 위치 확인 (iPhone, iPad, Mac, AirPods)
- 시간에 따른 반려동물이나 물품의 이동 모니터링 (AirTag 순찰 경로)

## 방법 1: AppleScript + 스크린샷 (기본)

### FindMy 열기 및 탐색

```bash
# Find My 앱 열기
osascript -e 'tell application "FindMy" to activate'

# 로드될 때까지 대기
sleep 3

# Find My 창 스크린샷 캡처
screencapture -w -o /tmp/findmy.png
```

그런 다음 `vision_analyze`를 사용하여 스크린샷을 읽습니다:
```
vision_analyze(image_url="/tmp/findmy.png", question="어떤 기기/물품이 표시되며 위치는 어디인가요?")
```

### 탭 전환

```bash
# 기기(Devices) 탭으로 전환
osascript -e '
tell application "System Events"
    tell process "FindMy"
        click button "Devices" of toolbar 1 of window 1
    end tell
end tell'

# 물품(Items) 탭으로 전환 (AirTags)
osascript -e '
tell application "System Events"
    tell process "FindMy"
        click button "Items" of toolbar 1 of window 1
    end tell
end tell'
```

## 방법 2: Peekaboo UI 자동화 (권장)

`peekaboo`가 설치된 경우 더 안정적인 UI 상호 작용을 위해 이를 사용합니다:

```bash
# Find My 열기
osascript -e 'tell application "FindMy" to activate'
sleep 3

# UI 캡처 및 주석 달기
peekaboo see --app "FindMy" --annotate --path /tmp/findmy-ui.png

# 요소 ID를 사용하여 특정 기기/물품 클릭
peekaboo click --on B3 --app "FindMy"

# 세부 정보 보기 캡처
peekaboo image --app "FindMy" --path /tmp/findmy-detail.png
```

그런 다음 비전으로 분석합니다:
```
vision_analyze(image_url="/tmp/findmy-detail.png", question="이 기기/물품에 표시된 위치는 어디인가요? 주소와 좌표가 보이면 포함하세요.")
```

## 워크플로: 시간에 따른 AirTag 위치 추적

AirTag를 모니터링하는 경우 (예: 고양이의 순찰 경로 추적):

```bash
# 1. FindMy를 열어 물품(Items) 탭으로 이동
osascript -e 'tell application "FindMy" to activate'
sleep 3

# 2. AirTag 항목을 클릭합니다. (페이지 유지 — AirTag는 페이지가 열려 있을 때만 업데이트됨)

# 3. 주기적으로 위치 캡처
while true; do
    screencapture -w -o /tmp/findmy-$(date +%H%M%S).png
    sleep 300  # 5분마다
done
```

비전을 사용하여 각 스크린샷을 분석하여 좌표를 추출한 다음 경로를 컴파일합니다.

## 제한 사항 (Limitations)

- FindMy에는 **CLI나 API가 없습니다** — UI 자동화를 사용해야 합니다.
- AirTag는 FindMy 페이지가 활발하게 표시되는 동안에만 위치를 업데이트합니다.
- 위치 정확도는 FindMy 네트워크 내 주변 Apple 기기에 따라 달라집니다.
- 스크린샷을 찍으려면 화면 기록 권한이 필요합니다.
- AppleScript UI 자동화는 macOS 버전에 따라 작동하지 않을 수 있습니다.

## 규칙 (Rules)

1. AirTag를 추적할 때는 FindMy 앱을 포그라운드에 유지하십시오 (최소화하면 업데이트가 중지됨).
2. 스크린샷 내용을 읽으려면 `vision_analyze`를 사용하십시오 — 픽셀을 구문 분석(parse)하려고 시도하지 마십시오.
3. 지속적인 추적의 경우 cronjob을 사용하여 위치를 주기적으로 캡처하고 기록하십시오.
4. 개인정보를 존중하십시오 — 사용자가 소유한 기기/물품만 추적하십시오.
