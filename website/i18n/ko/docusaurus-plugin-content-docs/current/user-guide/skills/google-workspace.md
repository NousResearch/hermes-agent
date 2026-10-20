---
sidebar_position: 2
sidebar_label: "Google Workspace"
title: "Google Workspace — Gmail, Calendar, Drive, Sheets 및 Docs"
description: "OAuth2로 인증된 Google API를 통해 이메일을 보내고, 캘린더 이벤트를 관리하고, Drive를 검색하고, Sheets를 읽고 쓰고, Docs에 액세스합니다"
---

# Google Workspace Skill

Hermes용 Gmail, Calendar, Drive, Contacts, Sheets 및 Docs 통합입니다. 자동 토큰 갱신과 함께 OAuth2를 사용합니다. 더 폭넓은 기능을 위해 가능한 경우 [Google Workspace CLI (`gws`)](https://github.com/googleworkspace/cli)를 우선 사용하고, 그렇지 않으면 Google의 Python 클라이언트 라이브러리로 대체합니다.

**Skill 경로:** `skills/productivity/google-workspace/`

## 설정

설정은 전적으로 에이전트가 진행합니다 — Hermes에 Google Workspace 설정을 요청하면 각 단계를 안내합니다. 흐름은 다음과 같습니다.

1. **Google Cloud 프로젝트를 생성**하고 필요한 API(Gmail, Calendar, Drive, Sheets, Docs, People)를 활성화합니다.
2. **OAuth 2.0 사용자 인증 정보**(데스크톱 앱 유형)를 생성하고 클라이언트 보안 정보 JSON을 다운로드합니다.
3. **인증** — Hermes가 인증 URL을 생성하면 브라우저에서 승인하고 리디렉션 URL을 다시 붙여넣습니다.
4. **완료** — 그 시점부터 토큰이 자동으로 갱신됩니다.

:::tip 이메일만 사용하는 경우
이메일만 필요하고(Calendar/Drive/Sheets는 필요하지 않은 경우) **himalaya** skill을 대신 사용하세요 — Gmail 앱 비밀번호로 작동하며 2분이면 설정할 수 있습니다. Google Cloud 프로젝트가 필요하지 않습니다.
:::

## Gmail

### 검색

```bash
$GAPI gmail search "is:unread" --max 10
$GAPI gmail search "from:boss@company.com newer_than:1d"
$GAPI gmail search "has:attachment filename:pdf newer_than:7d"
```

각 메시지에 대해 `id`, `from`, `subject`, `date`, `snippet`, `labels`가 포함된 JSON을 반환합니다.

### 읽기

```bash
$GAPI gmail get MESSAGE_ID
```

전체 메시지 본문을 텍스트로 반환합니다(일반 텍스트를 우선하고 HTML을 대체 수단으로 사용).

### 보내기

```bash
# Basic send
$GAPI gmail send --to user@example.com --subject "Hello" --body "Message text"

# HTML email
$GAPI gmail send --to user@example.com --subject "Report" \
  --body "<h1>Q4 Results</h1><p>Details here</p>" --html

# Custom From header (display name + email)
$GAPI gmail send --to user@example.com --subject "Hello" \
  --from '"Research Agent" <user@example.com>' --body "Message text"

# With CC
$GAPI gmail send --to user@example.com --cc "team@example.com" \
  --subject "Update" --body "FYI"
```

### 사용자 지정 From 헤더

`--from` 플래그를 사용하면 발신 이메일의 보낸 사람 표시 이름을 사용자 지정할 수 있습니다. 여러 에이전트가 하나의 Gmail 계정을 공유하지만 수신자에게 서로 다른 이름을 표시하고 싶을 때 유용합니다.

```bash
# Agent 1
$GAPI gmail send --to client@co.com --subject "Research Summary" \
  --from '"Research Agent" <shared@company.com>' --body "..."

# Agent 2  
$GAPI gmail send --to client@co.com --subject "Code Review" \
  --from '"Code Assistant" <shared@company.com>' --body "..."
```

**작동 방식:** `--from` 값은 MIME 메시지의 RFC 5322 `From` 헤더로 설정됩니다. Gmail에서는 추가 구성 없이 인증된 자신의 이메일 주소에 표시 이름을 사용자 지정할 수 있습니다. 수신자에게는 사용자 지정 표시 이름(예: "Research Agent")이 보이고 이메일 주소는 그대로 유지됩니다.

**중요:** 인증된 계정과 *다른 이메일 주소*를 `--from`에 사용하면 Gmail에서 해당 주소를 Gmail 설정 → 계정 → 다음 주소에서 메일 보내기의 [다른 주소로 보내기 별칭](https://support.google.com/mail/answer/22370)으로 구성해야 합니다.

`--from` 플래그는 `send`와 `reply` 모두에서 작동합니다.

```bash
$GAPI gmail reply MESSAGE_ID \
  --from '"Support Bot" <shared@company.com>' --body "We're on it"
```

### 답장

```bash
$GAPI gmail reply MESSAGE_ID --body "Thanks, that works for me."
```

원본 메시지의 스레드 ID를 사용하고 `In-Reply-To` 및 `References` 헤더를 설정하여 답장을 자동으로 같은 스레드에 연결합니다.

### 라벨

```bash
# List all labels
$GAPI gmail labels

# Add/remove labels
$GAPI gmail modify MESSAGE_ID --add-labels LABEL_ID
$GAPI gmail modify MESSAGE_ID --remove-labels UNREAD
```

## Calendar

```bash
# List events (defaults to next 7 days)
$GAPI calendar list
$GAPI calendar list --start 2026-03-01T00:00:00Z --end 2026-03-07T23:59:59Z

# Create event (timezone required)
$GAPI calendar create --summary "Team Standup" \
  --start 2026-03-01T10:00:00-07:00 --end 2026-03-01T10:30:00-07:00

# With location and attendees
$GAPI calendar create --summary "Lunch" \
  --start 2026-03-01T12:00:00Z --end 2026-03-01T13:00:00Z \
  --location "Cafe" --attendees "alice@co.com,bob@co.com"

# Delete event
$GAPI calendar delete EVENT_ID
```

:::warning
Calendar 일정의 시간에는 **반드시** 시간대 오프셋(예: `-07:00`)이 포함되어야 하며, UTC(`Z`)를 사용할 수도 있습니다. `2026-03-01T10:00:00`처럼 시간대가 없는 날짜/시간은 모호하므로 UTC로 처리됩니다.
:::

## Drive

```bash
$GAPI drive search "quarterly report" --max 10
$GAPI drive search "mimeType='application/pdf'" --raw-query --max 5
```

## Sheets

```bash
# Read a range
$GAPI sheets get SHEET_ID "Sheet1!A1:D10"

# Write to a range
$GAPI sheets update SHEET_ID "Sheet1!A1:B2" --values '[["Name","Score"],["Alice","95"]]'

# Append rows
$GAPI sheets append SHEET_ID "Sheet1!A:C" --values '[["new","row","data"]]'
```

## Docs

```bash
$GAPI docs get DOC_ID
```

문서 제목과 전체 텍스트 콘텐츠를 반환합니다.

## Contacts

```bash
$GAPI contacts list --max 20
```

## 출력 형식

모든 명령은 JSON을 반환합니다. 서비스별 주요 필드는 다음과 같습니다.

| 명령 | 필드 |
|---------|--------|
| `gmail search` | `id`, `threadId`, `from`, `to`, `subject`, `date`, `snippet`, `labels` |
| `gmail get` | `id`, `threadId`, `from`, `to`, `subject`, `date`, `labels`, `body` |
| `gmail send/reply` | `status`, `id`, `threadId` |
| `calendar list` | `id`, `summary`, `start`, `end`, `location`, `description`, `htmlLink` |
| `calendar create` | `status`, `id`, `summary`, `htmlLink` |
| `drive search` | `id`, `name`, `mimeType`, `modifiedTime`, `webViewLink` |
| `contacts list` | `name`, `emails`, `phones` |
| `sheets get` | 셀 값의 2차원 배열 |

## 문제 해결

| 문제 | 해결 방법 |
|---------|-----|
| `NOT_AUTHENTICATED` | 설정을 실행하세요 (Hermes에 Google Workspace 설정을 요청). |
| `REFRESH_FAILED` | 토큰이 취소되었습니다 — 인증 단계를 다시 실행하세요. |
| `HttpError 403: Insufficient Permission` | scope가 없습니다 — 필요한 서비스에 맞는 scope로 취소한 뒤 다시 인증하세요. |
| `HttpError 403: Access Not Configured` | Google Cloud Console에서 API가 활성화되지 않았습니다. |
| `ModuleNotFoundError` | `--install-deps`와 함께 설정 스크립트를 실행하세요. |
