---
title: "Apple Reminders — remindctl을 통한 Apple Reminders: 추가, 목록 조회, 완료"
sidebar_label: "Apple Reminders"
description: "remindctl을 통한 Apple Reminders: 추가, 목록 조회, 완료"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Apple Reminders

remindctl을 통해 Apple Reminders를 관리합니다. 추가, 목록 조회, 완료를 지원합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치) |
| 경로 | `skills/apple/apple-reminders` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | macos |
| 태그 | `Reminders`, `tasks`, `todo`, `macOS`, `Apple` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# Apple Reminders

터미널에서 `remindctl`을 사용해 Apple Reminders를 직접 관리합니다. 작업은 iCloud를 통해 모든 Apple 기기에서 동기화됩니다.

## 사전 요구 사항

- **Reminders.app이 설치된 macOS**
- 설치: `brew install steipete/tap/remindctl`
- 메시지가 표시되면 Reminders 권한 부여
- 확인: `remindctl status` / 요청: `remindctl authorize`

## 사용 시점

- 사용자가 "reminder" 또는 "Reminders app"을 언급하는 경우
- iOS와 동기화되는 기한이 있는 개인 할 일을 생성하는 경우
- Apple Reminders 목록을 관리하는 경우
- 사용자가 iPhone/iPad에 작업이 표시되기를 원하는 경우

## 사용하지 않는 경우

- 에이전트 알림 예약 → 대신 cronjob 도구 사용
- 캘린더 일정 → Apple Calendar 또는 Google Calendar 사용
- 프로젝트 작업 관리 → GitHub Issues, Notion 등을 사용
- 사용자가 "remind me"라고 했지만 에이전트 알림을 의미하는 경우 → 먼저 확인

## 빠른 참조

### 미리 알림 보기

```bash
remindctl                    # Today's reminders
remindctl today              # Today
remindctl tomorrow           # Tomorrow
remindctl week               # This week
remindctl overdue            # Past due
remindctl all                # Everything
remindctl 2026-01-04         # Specific date
```

### 목록 관리

```bash
remindctl list               # List all lists
remindctl list Work          # Show specific list
remindctl list Projects --create    # Create list
remindctl list Work --delete        # Delete list
```

### 미리 알림 생성

```bash
remindctl add "Buy milk"
remindctl add --title "Call mom" --list Personal --due tomorrow
remindctl add --title "Meeting prep" --due "2026-02-15 09:00"
```

### 기한과 알람 / 사전 알림 시간

`--due`와 `--alarm`은 서로 다른 필드입니다.

- `--due`는 미리 알림의 기한 날짜/시간을 설정합니다.
- `--alarm`은 EventKit 알람/알림 트리거를 설정합니다. 시간이 지정된 기한 미리 알림은 기본적으로 기한 시간에 알람이 설정될 수 있지만, 사용자가 더 이른 사전 알림을 요청하면 `--alarm`을 명시적으로 전달합니다.

오후 2시가 기한이고 30분 전에 알림을 받으려면 다음과 같이 합니다.

```bash
remindctl add --title "Hairdresser" --due "2026-05-15 14:00" --alarm "2026-05-15 13:30"
```

기존 미리 알림을 편집하려면 다음과 같이 합니다.

```bash
remindctl edit 87354 --due "2026-05-15 14:00" --alarm "2026-05-15 13:30"
```

Reminders UI는 알림이 실행되는 시간이므로 항목을 알람 시간 기준으로 표시하거나 그룹화할 수 있습니다. 기한 시간이 변경되었다고 가정하지 말고 JSON으로 확인합니다.

```bash
remindctl today --json
```

예상 형태:

- `dueDate`: 실제 기한 시간
- `alarmDate`: 알림 / 사전 알림 시간

Apple의 공개 `EKReminder` 문서에는 미리 알림 전용 속성만 나열되어 있습니다. 알람 지원은 `--alarm` 플래그를 통해 remindctl이 노출하는 상속된 `EKCalendarItem` 동작에서 제공됩니다.

### 완료 / 삭제

```bash
remindctl complete 1 2 3          # Complete by ID
remindctl delete 4A83 --force     # Delete by ID
```

### 출력 형식

```bash
remindctl today --json       # JSON for scripting
remindctl today --plain      # TSV format
remindctl today --quiet      # Counts only
```

## 날짜 형식

`--due` 및 날짜 필터에서 허용되는 형식:
- `today`, `tomorrow`, `yesterday`
- `YYYY-MM-DD`
- `YYYY-MM-DD HH:mm`
- ISO 8601 (`2026-01-04T12:34:56Z`)

## 규칙

1. 사용자가 "remind me"라고 하면 Apple Reminders(휴대폰과 동기화)인지 에이전트 cronjob 알림인지 확인합니다.
2. 생성하기 전에 미리 알림 내용과 기한을 항상 확인합니다.
3. 프로그래밍 방식으로 파싱할 때는 `--json`을 사용합니다.
