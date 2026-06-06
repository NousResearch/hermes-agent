---
title: "Apple Reminders — remindctl을 통한 Apple Reminders: 추가, 목록 조회, 완료"
sidebar_label: "Apple Reminders"
description: "remindctl을 통한 Apple Reminders: 추가, 목록 조회, 완료"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Apple Reminders

remindctl을 통한 Apple Reminders: 추가, 목록 조회, 완료.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/apple/apple-reminders` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | macos |
| 태그 | `Reminders`, `tasks`, `todo`, `macOS`, `Apple` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Apple Reminders

`remindctl`을 사용하여 터미널에서 직접 Apple Reminders를 관리합니다. 작업은 iCloud를 통해 모든 Apple 기기 간에 동기화됩니다.

## 전제 조건

- Reminders.app이 있는 **macOS**
- 설치: `brew install steipete/tap/remindctl`
- 메시지가 표시될 때 Reminders 권한 부여
- 확인: `remindctl status` / 요청: `remindctl authorize`

## 사용 시기

- 사용자가 "미리 알림" 또는 "Reminders 앱"을 언급할 때
- iOS에 동기화되는 마감일이 있는 개인 할 일 생성 시
- Apple Reminders 목록을 관리할 때
- 사용자가 iPhone/iPad에 작업이 표시되기를 원할 때

## 사용하지 말아야 할 시기

- 에이전트 알림 예약 → 대신 cronjob 도구를 사용하세요
- 캘린더 이벤트 → Apple Calendar 또는 Google Calendar를 사용하세요
- 프로젝트 작업 관리 → GitHub Issues, Notion 등을 사용하세요
- 사용자가 "나에게 알려줘(remind me)"라고 말하지만 실제로는 에이전트 알림을 의미하는 경우 → 먼저 명확히 확인하세요

## 빠른 참조

### 미리 알림 보기

```bash
remindctl                    # 오늘의 미리 알림
remindctl today              # 오늘
remindctl tomorrow           # 내일
remindctl week               # 이번 주
remindctl overdue            # 기한 지남
remindctl all                # 모든 항목
remindctl 2026-01-04         # 특정 날짜
```

### 목록 관리

```bash
remindctl list               # 모든 목록 보기
remindctl list Work          # 특정 목록 보기
remindctl list Projects --create    # 목록 생성
remindctl list Work --delete        # 목록 삭제
```

### 미리 알림 생성

```bash
remindctl add "우유 사기"
remindctl add --title "엄마에게 전화하기" --list Personal --due tomorrow
remindctl add --title "회의 준비" --due "2026-02-15 09:00"
```

### 기한(Due Time) vs 알람(Alarm) / 사전 알림(Early Nudge)

`--due`와 `--alarm`은 서로 다른 필드입니다:

- `--due`는 미리 알림의 마감 날짜/시간을 설정합니다.
- `--alarm`은 EventKit 알람/알림 트리거를 설정합니다. 기한이 설정된 미리 알림은 마감 시간에 알람이 울리도록 기본 설정될 수 있지만, 사용자가 더 이른 사전 알림을 원할 때는 명시적으로 `--alarm`을 전달하세요.

오후 2시가 마감이고 30분 전에 알림이 울려야 하는 미리 알림의 경우:

```bash
remindctl add --title "미용실" --due "2026-05-15 14:00" --alarm "2026-05-15 13:30"
```

기존 미리 알림 수정:

```bash
remindctl edit 87354 --due "2026-05-15 14:00" --alarm "2026-05-15 13:30"
```

알림이 실행되는 시점이기 때문에 Reminders UI는 항목을 알람 시간별로 표시하거나 그룹화할 수 있습니다. 마감 시간이 변경되었다고 가정하지 말고 대신 JSON을 통해 확인하세요:

```bash
remindctl today --json
```

예상되는 형태:

- `dueDate`: 실제 마감 시간
- `alarmDate`: 알림 / 사전 알림 시간

Apple의 공개 `EKReminder` 문서에는 미리 알림 전용 속성만 나열되어 있습니다. 알람 지원은 remindctl의 `--alarm` 플래그에 의해 노출되는 상속된 `EKCalendarItem` 동작에서 비롯됩니다.

### 완료 / 삭제

```bash
remindctl complete 1 2 3          # ID로 완료
remindctl delete 4A83 --force     # ID로 삭제
```

### 출력 형식

```bash
remindctl today --json       # 스크립팅을 위한 JSON
remindctl today --plain      # TSV 형식
remindctl today --quiet      # 개수만 표시
```

## 날짜 형식

`--due` 및 날짜 필터에서 지원됨:
- `today`, `tomorrow`, `yesterday`
- `YYYY-MM-DD`
- `YYYY-MM-DD HH:mm`
- ISO 8601 (`2026-01-04T12:34:56Z`)

## 규칙

1. 사용자가 "나에게 알려줘(remind me)"라고 할 때 명확히 하세요: Apple Reminders(휴대폰 동기화) vs 에이전트 cronjob 알림
2. 생성하기 전에 항상 미리 알림 내용과 마감일을 확인하세요
3. 프로그래밍 방식의 파싱을 위해 `--json`을 사용하세요
