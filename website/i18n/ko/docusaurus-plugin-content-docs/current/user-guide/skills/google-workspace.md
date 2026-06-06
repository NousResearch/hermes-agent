---
sidebar_position: 2
sidebar_label: "Google Workspace"
title: "Google Workspace — Gmail, Calendar, Drive, Sheets & Docs"
description: "OAuth2 인증된 Google API를 통해 이메일 전송, 캘린더 이벤트 관리, 드라이브 검색, 스프레드시트 읽기/쓰기 및 문서에 액세스합니다."
---

# Google Workspace 스킬

Hermes를 위한 Gmail, Calendar, Drive, Contacts, Sheets, Docs 연동 기능입니다. 자동 토큰 갱신 기능이 있는 OAuth2를 사용합니다. 더 넓은 범위를 다루기 위해 가능한 경우 [Google Workspace CLI (`gws`)](https://github.com/googleworkspace/cli)를 선호하며, 그렇지 않은 경우 Google의 Python 클라이언트 라이브러리로 폴백(fallback)합니다.

**스킬 경로:** `skills/productivity/google-workspace/`

## 설정

설정은 전적으로 에이전트 주도 방식입니다 — Hermes에게 Google Workspace 설정을 요청하면 각 단계를 안내해 줍니다. 전체 흐름은 다음과 같습니다:

1. **Google Cloud 프로젝트 생성** 및 필요한 API(Gmail, Calendar, Drive, Sheets, Docs, People) 활성화
2. **OAuth 2.0 자격 증명(credentials) 생성** (데스크톱 앱 유형) 및 클라이언트 보안 비밀(client secret) JSON 다운로드
3. **인증(Authorize)** — Hermes가 인증 URL을 생성하면, 브라우저에서 승인한 후 리디렉션된 URL을 다시 붙여넣습니다.
4. **완료** — 이 시점부터 토큰이 자동으로 갱신됩니다.

:::tip 이메일만 사용하는 경우
Calendar/Drive/Sheets 없이 이메일 기능만 필요하다면, **himalaya** 스킬을 사용하세요 — Gmail 앱 비밀번호를 사용하며 설정에 2분밖에 걸리지 않습니다. Google Cloud 프로젝트가 필요하지 않습니다.
:::

## Gmail

### 검색하기

```bash
$GAPI gmail search "is:unread" --max 10
$GAPI gmail search "from:boss@company.com newer_than:1d"
$GAPI gmail search "has:attachment filename:pdf newer_than:7d"
```

각 메시지에 대한 `id`, `from`, `subject`, `date`, `snippet`, `labels`를 JSON 형식으로 반환합니다.

### 읽기

```bash
$GAPI gmail get MESSAGE_ID
```

전체 메시지 본문을 텍스트로 반환합니다 (일반 텍스트를 선호하며 HTML로 폴백합니다).

### 보내기

```bash
# 기본 전송
$GAPI gmail send --to user@example.com --subject "Hello" --body "Message text"

# HTML 이메일
$GAPI gmail send --to user@example.com --subject "Report" \
  --body "<h1>Q4 Results</h1><p>Details here</p>" --html

# 커스텀 발신자(From) 헤더 (표시 이름 + 이메일)
$GAPI gmail send --to user@example.com --subject "Hello" \
  --from '"Research Agent" <user@example.com>' --body "Message text"

# 참조(CC) 포함
$GAPI gmail send --to user@example.com --cc "team@example.com" \
  --subject "Update" --body "FYI"
```

### 커스텀 발신자(From) 헤더

`--from` 플래그를 사용하면 발신 이메일의 표시 이름(display name)을 커스텀할 수 있습니다. 이는 여러 에이전트가 동일한 Gmail 계정을 공유하지만 수신자에게는 다른 이름으로 보이게 하고 싶을 때 유용합니다:

```bash
# 에이전트 1
$GAPI gmail send --to client@co.com --subject "Research Summary" \
  --from '"Research Agent" <shared@company.com>' --body "..."

# 에이전트 2  
$GAPI gmail send --to client@co.com --subject "Code Review" \
  --from '"Code Assistant" <shared@company.com>' --body "..."
```

**작동 방식:** `--from` 값은 MIME 메시지의 RFC 5322 `From` 헤더로 설정됩니다. Gmail은 추가 구성 없이 인증된 이메일 주소의 표시 이름을 커스텀할 수 있도록 허용합니다. 수신자는 커스텀된 표시 이름(예: "Research Agent")을 보게 되며 이메일 주소는 동일하게 유지됩니다.

**중요:** `--from`에 인증된 계정이 아닌 *다른 이메일 주소*를 사용하는 경우, Gmail의 설정 → 계정 및 가져오기 → 다른 주소에서 메일 보내기에서 해당 주소를 [Send As alias(별칭으로 보내기)](https://support.google.com/mail/answer/22370)로 구성해야 합니다.

`--from` 플래그는 `send`와 `reply` 모두에서 작동합니다:

```bash
$GAPI gmail reply MESSAGE_ID \
  --from '"Support Bot" <shared@company.com>' --body "We're on it"
```

### 답장하기

```bash
$GAPI gmail reply MESSAGE_ID --body "Thanks, that works for me."
```

자동으로 답장을 스레드로 묶고(`In-Reply-To` 및 `References` 헤더 설정) 원본 메시지의 스레드 ID를 사용합니다.

### 라벨 (Labels)

```bash
# 모든 라벨 나열
$GAPI gmail labels

# 라벨 추가/제거
$GAPI gmail modify MESSAGE_ID --add-labels LABEL_ID
$GAPI gmail modify MESSAGE_ID --remove-labels UNREAD
```

## Calendar

```bash
# 이벤트 목록 나열 (기본값은 향후 7일)
$GAPI calendar list
$GAPI calendar list --start 2026-03-01T00:00:00Z --end 2026-03-07T23:59:59Z

# 이벤트 생성 (시간대 필수)
$GAPI calendar create --summary "Team Standup" \
  --start 2026-03-01T10:00:00-07:00 --end 2026-03-01T10:30:00-07:00

# 위치 및 참석자 포함
$GAPI calendar create --summary "Lunch" \
  --start 2026-03-01T12:00:00Z --end 2026-03-01T13:00:00Z \
  --location "Cafe" --attendees "alice@co.com,bob@co.com"

# 이벤트 삭제
$GAPI calendar delete EVENT_ID
```

:::warning
Calendar 시간은 **반드시** 시간대 오프셋(예: `-07:00`)을 포함하거나 UTC(`Z`)를 사용해야 합니다. `2026-03-01T10:00:00`과 같은 시간대 없는 데이터는 모호하며 UTC로 간주됩니다.
:::

## Drive

```bash
$GAPI drive search "quarterly report" --max 10
$GAPI drive search "mimeType='application/pdf'" --raw-query --max 5
```

## Sheets

```bash
# 범위(range) 읽기
$GAPI sheets get SHEET_ID "Sheet1!A1:D10"

# 범위에 쓰기
$GAPI sheets update SHEET_ID "Sheet1!A1:B2" --values '[["Name","Score"],["Alice","95"]]'

# 행 추가 (Append)
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

## 출력 형식 (Output Format)

모든 명령어는 JSON을 반환합니다. 서비스별 주요 필드:

| 명령어 | 필드 |
|---------|--------|
| `gmail search` | `id`, `threadId`, `from`, `to`, `subject`, `date`, `snippet`, `labels` |
| `gmail get` | `id`, `threadId`, `from`, `to`, `subject`, `date`, `labels`, `body` |
| `gmail send/reply` | `status`, `id`, `threadId` |
| `calendar list` | `id`, `summary`, `start`, `end`, `location`, `description`, `htmlLink` |
| `calendar create` | `status`, `id`, `summary`, `htmlLink` |
| `drive search` | `id`, `name`, `mimeType`, `modifiedTime`, `webViewLink` |
| `contacts list` | `name`, `emails`, `phones` |
| `sheets get` | 2D 배열 형태의 셀(cell) 값 |

## 문제 해결 (Troubleshooting)

| 문제 | 해결 방법 |
|---------|-----|
| `NOT_AUTHENTICATED` | 설정을 실행하세요 (Hermes에게 Google Workspace를 설정해 달라고 요청) |
| `REFRESH_FAILED` | 토큰이 취소됨 — 인증 단계를 다시 실행하세요 |
| `HttpError 403: Insufficient Permission` | 범위(Scope) 누락 — 권한을 취소하고 올바른 서비스 권한을 주어 다시 인증하세요 |
| `HttpError 403: Access Not Configured` | Google Cloud Console에서 API가 활성화되지 않았습니다 |
| `ModuleNotFoundError` | `--install-deps`와 함께 설정 스크립트를 실행하세요 |
