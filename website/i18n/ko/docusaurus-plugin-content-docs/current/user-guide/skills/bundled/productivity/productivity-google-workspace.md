---
title: "Google Workspace — gws CLI 또는 Python을 통한 Gmail, Calendar, Drive, Docs, Sheets"
sidebar_label: "Google Workspace"
description: "gws CLI 또는 Python을 통한 Gmail, Calendar, Drive, Docs, Sheets"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Google Workspace

gws CLI 또는 Python을 통한 Gmail, Calendar, Drive, Docs, Sheets.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/productivity/google-workspace` |
| Version | `1.1.0` |
| Author | Nous Research |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Google`, `Gmail`, `Calendar`, `Drive`, `Sheets`, `Docs`, `Contacts`, `Email`, `OAuth` |
| Related skills | [`himalaya`](/docs/user-guide/skills/bundled/email/email-himalaya) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Google Workspace

Hermes에서 관리하는 OAuth 및 얇은 CLI 래퍼를 통한 Gmail, Calendar, Drive, Contacts, Sheets 및 Docs. `gws`가 설치된 경우, 스킬은 더 넓은 범위의 Google Workspace 커버리지를 위해 실행 백엔드로 `gws`를 사용합니다. 그렇지 않으면 내장된 Python 클라이언트 구현으로 폴백(fallback)합니다.

## 참고 자료 (References)

- `references/gmail-search-syntax.md` — Gmail 검색 연산자 (is:unread, from:, newer_than: 등)

## 스크립트 (Scripts)

- `scripts/setup.py` — OAuth2 설정 (권한을 부여하기 위해 한 번 실행)
- `scripts/google_api.py` — 호환성 래퍼 CLI. Hermes의 기존 JSON 출력 형식을 유지하면서 가능한 경우 작업을 위해 `gws`를 선호합니다.

## 초기 설정 (First-Time Setup)

설정은 완전히 비대화형입니다 — CLI, Telegram, Discord 또는 어느 플랫폼에서든 작동하도록 단계별로 진행합니다.

먼저 단축 명령어를 정의하세요:

```bash
GSETUP="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/setup.py"
```

### 0단계: 이미 설정되어 있는지 확인

```bash
$GSETUP --check
```

`AUTHENTICATED`가 출력되면, 사용법으로 건너뛰세요 — 설정이 이미 완료되었습니다.

### 1단계: 선별 — 사용자가 무엇을 필요로 하는지 묻기

OAuth 설정을 시작하기 전에 사용자에게 두 가지 질문을 하세요:

**질문 1: "어떤 Google 서비스가 필요하십니까? 이메일만 필요하신가요, 아니면 캘린더/드라이브/스프레드시트/문서도 필요하신가요?"**

- **이메일만 필요** → 이 스킬이 전혀 필요하지 않습니다. 대신 `himalaya` 스킬을 사용하세요 — Gmail 앱 비밀번호(설정 → 보안 → 앱 비밀번호)로 작동하며 설정하는 데 2분밖에 걸리지 않습니다. Google Cloud 프로젝트가 필요하지 않습니다. himalaya 스킬을 로드하고 해당 설정 지침을 따르세요.

- **이메일 + 캘린더** → 이 스킬을 계속 진행하되, 인증 중 `--services email,calendar`를 사용하여 동의 화면에서 실제로 필요한 권한만 요청하도록 하세요.

- **캘린더/드라이브/스프레드시트/문서만 필요** → 이 스킬을 계속 진행하고 `calendar,drive,sheets,docs`와 같이 더 좁은 `--services` 세트를 사용하세요.

- **전체 Workspace 액세스** → 이 스킬을 계속 진행하고 기본 `all` 서비스 세트를 사용하세요.

**질문 2: "Google 계정에서 고급 보호(로그인 시 하드웨어 보안 키 필요)를 사용하고 있습니까? 확실하지 않다면 아마 사용하지 않는 것입니다 — 이것은 명시적으로 등록해야 하는 기능입니다."**

- **아니오 / 확실하지 않음** → 정상 설정. 아래를 계속 진행하세요.
- **예** → 4단계가 작동하려면 Workspace 관리자가 조직의 허용된 앱 목록에 OAuth 클라이언트 ID를 추가해야 합니다. 이 사실을 미리 알려주세요.

### 2단계: OAuth 자격 증명 만들기 (1회성, 약 5분 소요)

사용자에게 다음과 같이 안내하세요:

> Google Cloud OAuth 클라이언트가 필요합니다. 이는 일회성 설정입니다:
>
> 1. 프로젝트를 생성하거나 선택합니다:
>    https://console.cloud.google.com/projectselector2/home/dashboard
> 2. API 라이브러리에서 필요한 API를 활성화합니다:
>    https://console.cloud.google.com/apis/library
>    활성화할 항목: Gmail API, Google Calendar API, Google Drive API, Google Sheets API, Google Docs API, People API
> 3. 여기에서 OAuth 클라이언트를 생성합니다:
>    https://console.cloud.google.com/apis/credentials
>    사용자 인증 정보 → 사용자 인증 정보 만들기 → OAuth 클라이언트 ID
> 4. 애플리케이션 유형: "데스크톱 앱" → 만들기
> 5. 앱이 여전히 테스트 중인 경우, 여기에서 사용자의 Google 계정을 테스트 사용자로 추가하세요:
>    https://console.cloud.google.com/auth/audience
>    잠재고객 → 테스트 사용자 → 사용자 추가
> 6. JSON 파일을 다운로드하고 파일 경로를 저에게 알려주세요.
>
> 중요한 Hermes CLI 참고: 파일 경로가 `/`로 시작하는 경우 CLI에서 단일 메시지로 단순 경로만 보내지 마세요. 슬래시 명령어로 오인될 수 있습니다. 대신 문장 안에 넣어 보내세요. 예:
> `The JSON 파일 경로는: /home/user/Downloads/client_secret_....json 입니다.`

사용자가 경로를 제공하면:

```bash
$GSETUP --client-secret /path/to/client_secret.json
```

사용자가 파일 경로 대신 원시(raw) 클라이언트 ID / 클라이언트 암호 값을 직접 붙여넣는 경우, 사용자를 위해 유효한 데스크톱 OAuth JSON 파일을 작성하고 명시적인 위치(예: `~/Downloads/hermes-google-client-secret.json`)에 저장한 다음 해당 파일에 대해 `--client-secret`을 실행하세요.

### 3단계: 인증 URL 받기

1단계에서 선택한 서비스 세트를 사용합니다. 예:

```bash
$GSETUP --auth-url --services email,calendar --format json
$GSETUP --auth-url --services calendar,drive,sheets,docs --format json
$GSETUP --auth-url --services all --format json
```

이것은 `auth_url` 필드가 있는 JSON을 반환하고 정확한 URL을 `~/.hermes/google_oauth_last_url.txt`에 저장합니다.

이 단계에 대한 에이전트 규칙:
- `auth_url` 필드를 추출하여 정확히 해당 URL을 사용자에게 한 줄로 보냅니다.
- 승인 후 브라우저가 `http://localhost:1`에서 오류가 발생할 가능성이 있으며 이는 정상적인 현상임을 사용자에게 알려주세요.
- 브라우저 주소 표시줄에서 리디렉션된 전체(ENTIRE) URL을 복사하도록 지시하세요.
- 사용자가 `Error 403: access_denied` 오류를 받는 경우, 자신을 테스트 사용자로 추가하기 위해 `https://console.cloud.google.com/auth/audience`로 직접 이동하도록 안내하세요.

### 4단계: 코드 교환 (Exchange the code)

사용자는 `http://localhost:1/?code=4/0A...&scope=...`와 같은 URL을 다시 붙여넣거나 코드 문자열만 붙여넣을 것입니다. 둘 다 작동합니다. `--auth-url` 단계는 임시 보류 중인 OAuth 세션을 로컬에 저장하므로 `--auth-code`가 나중에 헤드리스 시스템에서도 PKCE 교환을 완료할 수 있습니다:

```bash
$GSETUP --auth-code "THE_URL_OR_CODE_THE_USER_PASTED" --format json
```

만약 코드가 만료되었거나, 이미 사용되었거나, 이전 브라우저 탭에서 복사한 것이라서 `--auth-code`가 실패하면 새로운 `fresh_auth_url`을 반환합니다. 이 경우 즉시 새 URL을 사용자에게 보내고 가장 최신의 브라우저 리디렉션으로만 다시 시도하도록 하세요.

### 5단계: 검증 (Verify)

```bash
$GSETUP --check
```

`AUTHENTICATED`가 출력되어야 합니다. 설정이 완료되었습니다 — 이제부터 토큰은 자동으로 갱신됩니다.

### 참고 사항 (Notes)

- 토큰은 `~/.hermes/google_token.json`에 저장되며 자동 갱신됩니다.
- 보류 중인 OAuth 세션 상태/verifier는 교환이 완료될 때까지 `~/.hermes/google_oauth_pending.json`에 임시로 저장됩니다.
- `gws`가 설치된 경우 `google_api.py`는 동일한 `~/.hermes/google_token.json` 자격 증명 파일을 가리킵니다. 사용자가 별도로 `gws auth login` 절차를 실행할 필요가 없습니다.
- 액세스 권한 취소: `$GSETUP --revoke`

## 사용법 (Usage)

모든 명령어는 API 스크립트를 거칩니다. 단축 명령어로 `GAPI`를 설정하세요:

```bash
GAPI="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/google_api.py"
```

### Gmail

```bash
# 검색 (id, from, subject, date, snippet이 포함된 JSON 배열 반환)
$GAPI gmail search "is:unread" --max 10
$GAPI gmail search "from:boss@company.com newer_than:1d"
$GAPI gmail search "has:attachment filename:pdf newer_than:7d"

# 전체 메시지 읽기 (body text가 포함된 JSON 반환)
$GAPI gmail get MESSAGE_ID

# 보내기
$GAPI gmail send --to user@example.com --subject "Hello" --body "Message text"
$GAPI gmail send --to user@example.com --subject "Report" --body "<h1>Q4</h1><p>Details...</p>" --html
$GAPI gmail send --to user@example.com --subject "Hello" --from '"Research Agent" <user@example.com>' --body "Message text"

# 회신 (자동으로 스레드화하고 In-Reply-To를 설정)
$GAPI gmail reply MESSAGE_ID --body "Thanks, that works for me."
$GAPI gmail reply MESSAGE_ID --from '"Support Bot" <user@example.com>' --body "Thanks"

# 라벨
$GAPI gmail labels
$GAPI gmail modify MESSAGE_ID --add-labels LABEL_ID
$GAPI gmail modify MESSAGE_ID --remove-labels UNREAD
```

### Calendar

```bash
# 이벤트 나열 (기본값: 향후 7일)
$GAPI calendar list
$GAPI calendar list --start 2026-03-01T00:00:00Z --end 2026-03-07T23:59:59Z

# 이벤트 생성 (시간대가 포함된 ISO 8601 필수)
$GAPI calendar create --summary "Team Standup" --start 2026-03-01T10:00:00-06:00 --end 2026-03-01T10:30:00-06:00
$GAPI calendar create --summary "Lunch" --start 2026-03-01T12:00:00Z --end 2026-03-01T13:00:00Z --location "Cafe"
$GAPI calendar create --summary "Review" --start 2026-03-01T14:00:00Z --end 2026-03-01T15:00:00Z --attendees "alice@co.com,bob@co.com"

# 이벤트 삭제
$GAPI calendar delete EVENT_ID
```

### Drive

```bash
# 기존 파일 검색
$GAPI drive search "quarterly report" --max 10
$GAPI drive search "mimeType='application/pdf'" --raw-query --max 5

# 단일 파일 메타데이터 가져오기
$GAPI drive get FILE_ID

# 로컬 파일 업로드 (MIME 유형 자동 감지)
$GAPI drive upload /path/to/report.pdf
$GAPI drive upload /path/to/image.png --name "Logo.png" --parent FOLDER_ID

# 다운로드 (이진 파일은 그대로 다운로드되며, Google 기본 파일은 적절한 기본값으로 내보내짐 — Docs→pdf, Sheets→csv, Slides→pdf, Drawings→png)
$GAPI drive download FILE_ID
$GAPI drive download DOC_ID --output ~/doc.pdf
$GAPI drive download DOC_ID --export-mime text/plain --output ~/doc.txt

# 폴더 생성
$GAPI drive create-folder "Reports"
$GAPI drive create-folder "Q4" --parent FOLDER_ID

# 공유
$GAPI drive share FILE_ID --email alice@example.com --role reader
$GAPI drive share FILE_ID --email alice@example.com --role writer --notify
$GAPI drive share FILE_ID --type anyone --role reader        # 링크가 있는 모든 사용자
$GAPI drive share FILE_ID --type domain --domain example.com --role reader

# 삭제 — 기본적으로 휴지통으로 이동(복원 가능). 휴지통을 건너뛰려면 --permanent를 사용하세요.
$GAPI drive delete FILE_ID
$GAPI drive delete FILE_ID --permanent
```

### Contacts

```bash
$GAPI contacts list --max 20
```

### Sheets

```bash
# 새 스프레드시트 생성
$GAPI sheets create --title "Q4 Budget"
$GAPI sheets create --title "Inventory" --sheet-name "Stock"

# 읽기
$GAPI sheets get SHEET_ID "Sheet1!A1:D10"

# 쓰기
$GAPI sheets update SHEET_ID "Sheet1!A1:B2" --values '[["Name","Score"],["Alice","95"]]'

# 행 추가
$GAPI sheets append SHEET_ID "Sheet1!A:C" --values '[["new","row","data"]]'
```

### Docs

```bash
# 읽기
$GAPI docs get DOC_ID

# 새 문서 생성 (선택적으로 본문 텍스트 포함)
$GAPI docs create --title "Meeting Notes"
$GAPI docs create --title "Draft" --body "First paragraph..."

# 기존 문서 끝에 텍스트 추가
$GAPI docs append DOC_ID --text "Additional content to append"
```

## 출력 형식 (Output Format)

모든 명령어는 JSON을 반환합니다. `jq`로 구문 분석하거나 직접 읽으세요. 주요 필드:

- **Gmail search**: `[{id, threadId, from, to, subject, date, snippet, labels}]`
- **Gmail get**: `{id, threadId, from, to, subject, date, labels, body}`
- **Gmail send/reply**: `{status: "sent", id, threadId}`
- **Calendar list**: `[{id, summary, start, end, location, description, htmlLink}]`
- **Calendar create**: `{status: "created", id, summary, htmlLink}`
- **Drive search**: `[{id, name, mimeType, modifiedTime, webViewLink}]`
- **Drive get**: `{id, name, mimeType, modifiedTime, size, webViewLink, parents, owners}`
- **Drive upload**: `{status: "uploaded", id, name, mimeType, webViewLink}`
- **Drive download**: `{status: "downloaded", id, name, path, mimeType}`
- **Drive create-folder**: `{status: "created", id, name, webViewLink}`
- **Drive share**: `{status: "shared", permissionId, fileId, role, type}`
- **Drive delete**: `{status: "trashed" | "deleted", fileId, permanent}`
- **Contacts list**: `[{name, emails: [...], phones: [...]}]`
- **Sheets get**: `[[cell, cell, ...], ...]`
- **Sheets create**: `{status: "created", spreadsheetId, title, spreadsheetUrl}`
- **Docs create**: `{status: "created", documentId, title, url}`
- **Docs append**: `{status: "appended", documentId, inserted_at, characters}`

## 규칙 (Rules)

1. **사용자에게 먼저 확인받지 않고 절대 이메일 전송, 캘린더 이벤트 생성/삭제, Drive 파일 삭제, 파일 공유, Docs/Sheets 수정을 수행하지 마세요.** 실행될 내용(수신자, 파일 ID, 콘텐츠, 공유 역할)을 보여주고 승인을 요청하세요. `drive delete`의 경우 `--permanent`보다 기본 휴지통(복원 가능)을 선호하세요.
2. **첫 사용 전에 인증 상태를 확인하세요** — `setup.py --check`를 실행합니다. 실패하면 사용자가 설정을 진행하도록 안내하세요.
3. **복잡한 쿼리의 경우 Gmail 검색 구문 참조를 사용하세요** — `skill_view("google-workspace", file_path="references/gmail-search-syntax.md")`로 로드합니다.
4. **캘린더 시간에는 시간대가 포함되어야 합니다** — 항상 오프셋(예: `2026-03-01T10:00:00-06:00`) 또는 UTC(`Z`)가 포함된 ISO 8601 형식을 사용하세요.
5. **요청 속도 제한을 준수하세요** — API를 너무 빠르게 연속적으로 호출하지 마세요. 가능하면 읽기 작업을 일괄 처리하세요.

## 문제 해결 (Troubleshooting)

| 문제 | 해결책 |
|---------|-----|
| `NOT_AUTHENTICATED` | 위 설정의 2~5단계를 실행하세요. |
| `REFRESH_FAILED` | 토큰이 취소되었거나 만료됨 — 3~5단계를 다시 실행하세요. |
| `HttpError 403: Insufficient Permission` | API 범위 누락 — `$GSETUP --revoke` 후 3~5단계를 다시 실행하세요. |
| `AUTHENTICATED (partial)` 또는 "Token missing scopes" | 새로운 쓰기 기능(Drive 쓰기/삭제, Docs 생성/편집)에는 재인증이 필요합니다. `$GSETUP --revoke` 후 업그레이드된 권한을 부여하기 위해 3~5단계를 다시 실행하세요. |
| `HttpError 403: Access Not Configured` | API가 활성화되지 않음 — 사용자가 Google Cloud Console에서 활성화해야 합니다. |
| `ModuleNotFoundError` | `$GSETUP --install-deps`를 실행하세요. |
| 고급 보호로 인해 인증이 차단됨 | Workspace 관리자가 OAuth 클라이언트 ID를 허용 목록에 추가해야 합니다. |

## 액세스 권한 취소 (Revoking Access)

```bash
$GSETUP --revoke
```
