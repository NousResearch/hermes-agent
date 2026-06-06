---
title: "Canvas — Canvas LMS 연동 — API 토큰 인증을 사용하여 등록된 강의 및 과제 가져오기"
sidebar_label: "Canvas"
description: "Canvas LMS 연동 — API 토큰 인증을 사용하여 등록된 강의 및 과제 가져오기"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Canvas

Canvas LMS 연동 — API 토큰 인증을 사용하여 등록된 강의 및 과제를 가져옵니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택적(Optional) — `hermes skills install official/productivity/canvas` 명령어로 설치 |
| 경로 | `optional-skills/productivity/canvas` |
| 버전 | `1.0.0` |
| 작성자 | community |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Canvas`, `LMS`, `Education`, `Courses`, `Assignments` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 내용은 스킬이 활성화되어 있을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# Canvas LMS — 강의 및 과제 액세스

강의 및 과제 목록을 확인하기 위한 Canvas LMS의 읽기 전용 액세스입니다.

## 스크립트

- `scripts/canvas_api.py` — Canvas API 호출을 위한 Python CLI

## 설정

1. 브라우저에서 Canvas 인스턴스에 로그인합니다.
2. **Account → Settings(계정 → 설정)**으로 이동합니다. (프로필 아이콘을 클릭한 다음 설정을 클릭합니다.)
3. **Approved Integrations(승인된 통합)** 섹션으로 스크롤하여 **+ New Access Token(+ 새 액세스 토큰)**을 클릭합니다.
4. 토큰의 이름(예: "Hermes Agent")을 지정하고 선택 사항으로 만료일을 설정한 다음 **Generate Token(토큰 생성)**을 클릭합니다.
5. 토큰을 복사하여 `~/.hermes/.env` 파일에 추가합니다:

```
CANVAS_API_TOKEN=your_token_here
CANVAS_BASE_URL=https://yourschool.instructure.com
```

기본 URL(Base URL)은 Canvas에 로그인했을 때 브라우저에 표시되는 주소입니다(끝에 슬래시 제외).

## 사용법

```bash
CANVAS="python $HERMES_HOME/skills/productivity/canvas/scripts/canvas_api.py"

# 활성 강의 목록 표시
$CANVAS list_courses --enrollment-state active

# 모든 강의 목록 표시 (모든 상태)
$CANVAS list_courses

# 특정 강의의 과제 목록 표시
$CANVAS list_assignments 12345

# 기한을 기준으로 정렬된 과제 목록 표시
$CANVAS list_assignments 12345 --order-by due_at
```

## 출력 형식

**list_courses** 반환값:
```json
[{"id": 12345, "name": "Intro to CS", "course_code": "CS101", "workflow_state": "available", "start_at": "...", "end_at": "..."}]
```

**list_assignments** 반환값:
```json
[{"id": 67890, "name": "Homework 1", "due_at": "2025-02-15T23:59:00Z", "points_possible": 100, "submission_types": ["online_upload"], "html_url": "...", "description": "...", "course_id": 12345}]
```

참고: 과제 설명은 500자로 잘립니다. `html_url` 필드는 Canvas의 전체 과제 페이지로 연결됩니다.

## API 레퍼런스 (curl)

```bash
# 강의 목록
curl -s -H "Authorization: Bearer $CANVAS_API_TOKEN" \
  "$CANVAS_BASE_URL/api/v1/courses?enrollment_state=active&per_page=10"

# 강의의 과제 목록
curl -s -H "Authorization: Bearer $CANVAS_API_TOKEN" \
  "$CANVAS_BASE_URL/api/v1/courses/COURSE_ID/assignments?per_page=10&order_by=due_at"
```

Canvas는 페이지네이션을 위해 `Link` 헤더를 사용합니다. Python 스크립트가 페이지네이션을 자동으로 처리합니다.

## 규칙

- 이 스킬은 **읽기 전용**입니다 — 데이터를 가져오기만 하며 강의나 과제를 절대 수정하지 않습니다.
- 처음 사용할 때 `$CANVAS list_courses`를 실행하여 인증을 확인합니다. 401 오류가 발생하면 사용자가 설정할 수 있도록 안내합니다.
- Canvas는 10분당 약 700회의 요청으로 속도를 제한합니다; 제한에 도달한 경우 `X-Rate-Limit-Remaining` 헤더를 확인합니다.

## 문제 해결 (Troubleshooting)

| 문제 | 해결 방법 |
|---------|-----|
| 401 Unauthorized | 토큰이 잘못되었거나 만료되었습니다 — Canvas 설정에서 다시 생성하세요. |
| 403 Forbidden | 토큰에 이 강의에 대한 권한이 없습니다. |
| 빈 강의 목록 | `--enrollment-state active`를 시도하거나 플래그를 생략하여 모든 상태를 확인하세요. |
| 잘못된 학교 | `CANVAS_BASE_URL`이 브라우저의 URL과 일치하는지 확인하세요. |
| Timeout 오류 | Canvas 인스턴스와의 네트워크 연결을 확인하세요. |
