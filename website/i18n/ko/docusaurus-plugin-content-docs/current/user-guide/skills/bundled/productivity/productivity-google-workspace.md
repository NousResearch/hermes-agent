---
title: "Google Workspace — gws CLI 또는 Python을 통한 Gmail, Calendar, Drive, Docs, Sheets"
sidebar_label: "Google Workspace"
description: "gws CLI 또는 Python을 통한 Gmail, Calendar, Drive, Docs, Sheets"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Google Workspace

gws CLI 또는 Python을 통한 Gmail, Calendar, Drive, Docs, Sheets.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들됨 (기본 설치) |
| 경로 | `skills/productivity/google-workspace` |
| 버전 | `1.2.0` |
| 작성자 | Nous Research |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Google`, `Gmail`, `Calendar`, `Drive`, `Sheets`, `Docs`, `Contacts`, `Email`, `OAuth` |
| 관련 스킬 | [`himalaya`](/docs/user-guide/skills/bundled/email/email-himalaya) |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Google Workspace

Hermes가 관리하는 OAuth와 얇은 CLI 래퍼를 통해 Gmail, Calendar, Drive, Contacts, Sheets, Docs를 사용합니다. `gws`가 설치되어 있으면 더 폭넓은 Google Workspace 지원을 위해 이를 실행 백엔드로 사용하고, 그렇지 않으면 번들된 Python 클라이언트 구현으로 대체합니다.

## 참고 자료

- `references/gmail-search-syntax.md` — Gmail 검색 연산자 (is:unread, from:, newer_than: 등)
- `references/daily-brief.md` — 일일/아침 브리핑 절차: Gmail과 Calendar의 일정 + 충돌 + 회의 준비 + 긴급 메일. 사용자가 아침 브리핑, 회의 준비 또는 "내 캘린더에 무엇이 있고 어떤 이메일에 주의해야 하나요?"라고 요청할 때 로드합니다.

## 스크립트

- `scripts/setup.py` — OAuth2 설정 (한 번 실행하여 인증)
- `scripts/google_api.py` — 호환성 래퍼 CLI. 가능한 경우 `gws`를 우선 사용하면서 Hermes의 기존 JSON 출력 계약을 유지합니다.

## 최초 설정

설정은 완전히 비대화형입니다 — CLI, Telegram, Discord 또는 어떤 플랫폼에서도 작동하도록 단계별로 진행합니다.

먼저 약칭을 정의합니다:

```bash
GSETUP="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/setup.py"
```

### 0단계: 이미 설정되었는지 확인

```bash
$GSETUP --check
```

`AUTHENTICATED`가 출력되면 사용법으로 건너뜁니다 — 설정이 이미 완료되었습니다.

### 1단계: 분류 — 사용자에게 필요한 사항 확인

OAuth 설정을 시작하기 전에 사용자에게 다음 두 가지 질문을 하세요:

**질문 1: "어떤 Google 서비스를 사용해야 하나요? 이메일만 필요한가요, 아니면 Calendar/Drive/Sheets/Docs도 필요한가요?"**

- **이메일만** → 이 스킬은 전혀 필요하지 않습니다. 대신 `himalaya` 스킬을 사용하세요 — Gmail 앱 비밀번호 (설정 → 보안 → 앱 비밀번호)로 작동하며 설정에 2분이 걸립니다. Google Cloud 프로젝트가 필요하지 않습니다.
  `himalaya` 스킬을 로드하고 해당 설정 지침을 따르세요.

- **이메일 + Calendar** → 이 스킬로 계속 진행하되, 인증 시 `--services email,calendar`를 사용하여 실제로 필요한 범위만 동의 화면에 표시되도록 하세요.

- **Calendar/Drive/Sheets/Docs만** → 이 스킬로 계속 진행하고 `calendar,drive,sheets,docs`처럼 더 좁은 `--services` 집합을 사용하세요.

- **전체 Workspace 액세스** → 이 스킬로 계속 진행하고 기본 `all` 서비스 집합을 사용하세요.

**질문 2: "Google 계정에서 Advanced Protection을 사용하나요 (로그인에 하드웨어 보안 키 필요)? 확실하지 않다면 아마 사용하지 않을 가능성이 높습니다 — 직접 등록했어야 하는 기능입니다."**

- **아니요 / 확실하지 않음** → 일반 설정입니다. 아래를 계속 진행하세요.
- **예** → 4단계가 작동하기 전에 Workspace 관리자가 조직의 허용된 앱 목록에 OAuth 클라이언트 ID를 추가해야 합니다. 미리 알려 주세요.

### 2단계: OAuth 자격 증명 생성 (최초 1회, 약 5분)

사용자에게 다음을 안내하세요:

> Google Cloud OAuth 클라이언트가 필요합니다. 최초 1회 설정입니다:
>
> 1. 프로젝트를 생성하거나 선택하세요:
>    https://console.cloud.google.com/projectselector2/home/dashboard
> 2. API 라이브러리에서 필요한 API를 활성화하세요:
>    https://console.cloud.google.com/apis/library
>    활성화할 항목: Gmail API, Google Calendar API, Google Drive API,
>    Google Sheets API, Google Docs API, People API
> 3. 여기에서 OAuth 클라이언트를 생성하세요:
>    https://console.cloud.google.com/apis/credentials
>    자격 증명 → 자격 증명 만들기 → OAuth 2.0 클라이언트 ID
> 4. 애플리케이션 유형: "데스크톱 앱" → 만들기
> 5. 앱이 아직 테스트 중이면 여기에서 사용자의 Google 계정을 테스트 사용자로 추가하세요:
>    https://console.cloud.google.com/auth/audience
>    대상 → 테스트 사용자 → 사용자 추가
> 6. JSON 파일을 다운로드하고 파일 경로를 알려 주세요
>
> 중요한 Hermes CLI 참고 사항: 파일 경로가 `/`로 시작하면 CLI에서 경로만 단독으로 보내지 마세요. 슬래시 명령으로 오인될 수 있습니다. 대신 `The JSON file path is: ~/Downloads/client_secret_....json`처럼 문장으로 보내세요.

경로를 받으면:

```bash
$GSETUP --client-secret /path/to/client_secret.json
```

파일 경로 대신 원시 클라이언트 ID / 클라이언트 보안 비밀번호 값을 붙여넣으면 유효한 Desktop OAuth JSON 파일을 직접 작성하여 명시적인 위치 (예: `~/Downloads/hermes-google-client-secret.json`)에 저장한 다음 해당 파일을 대상으로 `--client-secret`을 실행하세요.

### 3단계: 인증 URL 가져오기

1단계에서 선택한 서비스 집합을 사용하세요. 예:

```bash
$GSETUP --auth-url --services email,calendar --format json
$GSETUP --auth-url --services calendar,drive,sheets,docs --format json
$GSETUP --auth-url --services all --format json
```

이 명령은 `auth_url` 필드가 포함된 JSON을 반환하고 정확한 URL을 `~/.hermes/google_oauth_last_url.txt`에도 저장합니다.

이 단계의 에이전트 규칙:
- `auth_url` 필드를 추출하고 해당 URL을 한 줄로 사용자에게 보내세요.
- 승인 후 브라우저가 `http://localhost:1`에서 실패할 가능성이 높지만 이는 예상된 동작이라고 알려 주세요.
- 브라우저 주소 표시줄에서 리디렉션된 URL 전체를 복사하라고 안내하세요.
- 사용자에게 `Error 403: access_denied`가 표시되면 `https://console.cloud.google.com/auth/audience`로 바로 이동하여 자신을 테스트 사용자로 추가하라고 안내하세요.

### 4단계: 코드 교환

사용자는 `http://localhost:1/?code=4/0A...&scope=...` 같은 URL 또는 코드 문자열만 붙여넣습니다. 둘 다 사용할 수 있습니다. `--auth-url` 단계는 보류 중인 OAuth 세션을 로컬에 저장하므로 헤드리스 시스템에서도 나중에 `--auth-code`로 PKCE 교환을 완료할 수 있습니다:

```bash
$GSETUP --auth-code "THE_URL_OR_CODE_THE_USER_PASTED" --format json
```

코드가 만료되었거나 이미 사용되었거나 이전 브라우저 탭에서 가져온 것이어서 `--auth-code`가 실패하면 이제 `fresh_auth_url`을 반환합니다. 이 경우 즉시 새 URL을 사용자에게 보내고 최신 브라우저 리디렉션만 사용하여 다시 시도하게 하세요.

### 5단계: 확인

```bash
$GSETUP --check
```

`AUTHENTICATED`가 출력되어야 합니다. 설정이 완료되었습니다 — 이제부터 토큰이 자동으로 갱신됩니다.

### 참고

- 토큰은 `~/.hermes/google_token.json`에 저장되며 자동으로 갱신됩니다.
- 보류 중인 OAuth 세션 상태/검증자는 교환이 완료될 때까지 `~/.hermes/google_oauth_pending.json`에 임시로 저장됩니다.
- `gws`가 설치되어 있으면 `google_api.py`는 동일한 `~/.hermes/google_token.json` 자격 증명 파일을 사용하도록 이를 지정합니다. 사용자는 별도의 `gws auth login` 흐름을 실행할 필요가 없습니다.
- 취소하려면: `$GSETUP --revoke`

## 사용법

모든 명령은 API 스크립트를 통해 실행됩니다. `GAPI`를 약칭으로 설정하세요:

```bash
GAPI="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/google_api.py"
```

### Gmail

```bash
# Search (returns JSON array with id, from, subject, date, snippet)
$GAPI gmail search "is:unread" --max 10
$GAPI gmail search "from:boss@company.com newer_than:1d"
$GAPI gmail search "has:attachment filename:pdf newer_than:7d"

# Read full message (returns JSON with body text)
$GAPI gmail get MESSAGE_ID

# Send
$GAPI gmail send --to user@example.com --subject "Hello" --body "Message text"
$GAPI gmail send --to user@example.com --subject "Report" --body "<h1>Q4</h1><p>Details...</p>" --html
$GAPI gmail send --to user@example.com --subject "Hello" --from '"Research Agent" <user@example.com>' --body "Message text"

# Reply (automatically threads and sets In-Reply-To)
$GAPI gmail reply MESSAGE_ID --body "Thanks, that works for me."
$GAPI gmail reply MESSAGE_ID --from '"Support Bot" <user@example.com>' --body "Thanks"

# Labels
$GAPI gmail labels
$GAPI gmail modify MESSAGE_ID --add-labels LABEL_ID
$GAPI gmail modify MESSAGE_ID --remove-labels UNREAD
```

### Calendar

```bash
# List events (defaults to next 7 days)
$GAPI calendar list
$GAPI calendar list --start 2026-03-01T00:00:00Z --end 2026-03-07T23:59:59Z

# Create event (ISO 8601 with timezone required)
$GAPI calendar create --summary "Team Standup" --start 2026-03-01T10:00:00-06:00 --end 2026-03-01T10:30:00-06:00
$GAPI calendar create --summary "Lunch" --start 2026-03-01T12:00:00Z --end 2026-03-01T13:00:00Z --location "Cafe"
$GAPI calendar create --summary "Review" --start 2026-03-01T14:00:00Z --end 2026-03-01T15:00:00Z --attendees "alice@co.com,bob@co.com"

# Delete event
$GAPI calendar delete EVENT_ID
```

### Drive

```bash
# Search existing files
$GAPI drive search "quarterly report" --max 10
$GAPI drive search "mimeType='application/pdf'" --raw-query --max 5

# Get metadata for a single file
$GAPI drive get FILE_ID

# Upload a local file (auto-detects MIME type)
$GAPI drive upload /path/to/report.pdf
$GAPI drive upload /path/to/image.png --name "Logo.png" --parent FOLDER_ID

# Download (binary files download as-is; Google-native files export to a
# sensible default — Docs→pdf, Sheets→csv, Slides→pdf, Drawings→png)
$GAPI drive download FILE_ID
$GAPI drive download DOC_ID --output ~/doc.pdf
$GAPI drive download DOC_ID --export-mime text/plain --output ~/doc.txt

# Create a folder
$GAPI drive create-folder "Reports"
$GAPI drive create-folder "Q4" --parent FOLDER_ID

# Share
$GAPI drive share FILE_ID --email alice@example.com --role reader
$GAPI drive share FILE_ID --email alice@example.com --role writer --notify
$GAPI drive share FILE_ID --type anyone --role reader        # anyone with link
$GAPI drive share FILE_ID --type domain --domain example.com --role reader

# Delete — defaults to trash (reversible). Use --permanent to skip the trash.
$GAPI drive delete FILE_ID
$GAPI drive delete FILE_ID --permanent
```

### Contacts

```bash
$GAPI contacts list --max 20
```

### Sheets

```bash
# Create a new spreadsheet
$GAPI sheets create --title "Q4 Budget"
$GAPI sheets create --title "Inventory" --sheet-name "Stock"

# Read
$GAPI sheets get SHEET_ID "Sheet1!A1:D10"

# Write
$GAPI sheets update SHEET_ID "Sheet1!A1:B2" --values '[["Name","Score"],["Alice","95"]]'

# Append rows
$GAPI sheets append SHEET_ID "Sheet1!A:C" --values '[["new","row","data"]]'
```

### Docs

```bash
# Read
$GAPI docs get DOC_ID

# Create a new Doc (optionally seeded with body text)
$GAPI docs create --title "Meeting Notes"
$GAPI docs create --title "Draft" --body "First paragraph..."

# Append text to the end of an existing Doc
$GAPI docs append DOC_ID --text "Additional content to append"
```

## 출력 형식

모든 명령은 JSON을 반환합니다. `jq`로 파싱하거나 직접 읽으세요. 주요 필드:

- **Gmail 검색**: `[{id, threadId, from, to, subject, date, snippet, labels}]`
- **Gmail 가져오기**: `{id, threadId, from, to, subject, date, labels, body}`
- **Gmail 보내기/답장**: `{status: "sent", id, threadId}`
- **Calendar 목록**: `[{id, summary, start, end, location, description, htmlLink}]`
- **Calendar 생성**: `{status: "created", id, summary, htmlLink}`
- **Drive 검색**: `[{id, name, mimeType, modifiedTime, webViewLink}]`
- **Drive 가져오기**: `{id, name, mimeType, modifiedTime, size, webViewLink, parents, owners}`
- **Drive 업로드**: `{status: "uploaded", id, name, mimeType, webViewLink}`
- **Drive 다운로드**: `{status: "downloaded", id, name, path, mimeType}`
- **Drive 폴더 생성**: `{status: "created", id, name, webViewLink}`
- **Drive 공유**: `{status: "shared", permissionId, fileId, role, type}`
- **Drive 삭제**: `{status: "trashed" | "deleted", fileId, permanent}`
- **Contacts 목록**: `[{name, emails: [...], phones: [...]}]`
- **Sheets 가져오기**: `[[cell, cell, ...], ...]`
- **Sheets 생성**: `{status: "created", spreadsheetId, title, spreadsheetUrl}`
- **Docs 생성**: `{status: "created", documentId, title, url}`
- **Docs 추가**: `{status: "appended", documentId, inserted_at, characters}`

## 규칙

1. **먼저 사용자에게 확인하지 않고 이메일을 보내거나, Calendar 일정을 생성/삭제하거나, Drive 파일을 삭제하거나, 파일을 공유하거나, Docs/Sheets를 수정하지 마세요.** 수행할 작업 (수신자, 파일 ID, 콘텐츠, 공유 역할)을 보여 주고 승인을 요청하세요. `drive delete`에서는 기본 휴지통 (복구 가능)을 `--permanent`보다 우선하세요.
2. **최초 사용 전에 인증을 확인하세요** — `setup.py --check`를 실행합니다. 실패하면 사용자가 설정하도록 안내하세요.
3. 복잡한 쿼리에는 Gmail 검색 구문 참고 자료를 사용하세요 — `skill_view("google-workspace", file_path="references/gmail-search-syntax.md")`로 로드합니다.
4. **Calendar 시간에는 시간대가 포함되어야 합니다** — 항상 오프셋이 있는 ISO 8601 (예: `2026-03-01T10:00:00-06:00`) 또는 UTC (`Z`)를 사용하세요.
5. **속도 제한을 준수하세요** — 빠르게 연속해서 API를 호출하지 마세요. 가능하면 읽기 작업을 일괄 처리하세요.

## 문제 해결

| 문제 | 해결 방법 |
|---------|-----|
| `NOT_AUTHENTICATED` | 위의 설정 2~5단계를 실행하세요 |
| `REFRESH_FAILED` | 토큰이 취소되었거나 만료됨 — 3~5단계를 다시 수행하세요 |
| `HttpError 403: Insufficient Permission` | API 범위가 없음 — `$GSETUP --revoke`를 실행한 후 3~5단계를 다시 수행하세요 |
| `AUTHENTICATED (partial)` 또는 "Token missing scopes" | 새로운 쓰기 기능 (Drive 쓰기/삭제, Docs 생성/편집)에는 재인증이 필요합니다. `$GSETUP --revoke`를 실행한 후 3~5단계를 다시 수행하여 업그레이드된 범위를 부여하세요. |
| `HttpError 403: Access Not Configured` | API가 활성화되지 않음 — 사용자가 Google Cloud Console에서 활성화해야 합니다 |
| `ModuleNotFoundError` | `$GSETUP --install-deps`를 실행하세요 |
| Advanced Protection이 인증을 차단함 | Workspace 관리자가 OAuth 클라이언트 ID를 허용 목록에 추가해야 합니다 |

## 액세스 취소

```bash
$GSETUP --revoke
```
