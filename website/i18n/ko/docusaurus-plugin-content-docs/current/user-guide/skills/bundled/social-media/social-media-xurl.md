---
title: "Xurl — xurl CLI를 통한 X/Twitter: 게시, 검색, DM, 미디어, v2 API"
sidebar_label: "Xurl"
description: "xurl CLI를 통한 X/Twitter: 게시, 검색, DM, 미디어, v2 API"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Xurl

xurl CLI를 통한 X/Twitter: 게시, 검색, DM, 미디어, v2 API.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/social-media/xurl` |
| 버전 | `1.1.1` |
| 작성자 | xdevplatform + openclaw + Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos |
| 태그 | `twitter`, `x`, `social-media`, `xurl`, `official-api` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# xurl — 공식 CLI를 통한 X (Twitter) API

`xurl`은 X 개발자 플랫폼의 공식 X API용 CLI입니다. 일반적인 작업을 위한 바로가기 명령어와 모든 v2 엔드포인트에 대한 순수 curl 스타일의 액세스를 모두 지원합니다. 모든 명령어는 stdout으로 JSON을 반환합니다.

이 스킬의 사용 목적:
- 게시물 작성, 답글, 인용, 삭제
- 게시물 검색 및 타임라인/멘션 읽기
- 마음에 들어요, 재게시, 북마크
- 팔로우, 언팔로우, 차단, 뮤트
- 다이렉트 메시지 (DM)
- 미디어 업로드 (이미지 및 비디오)
- 모든 X API v2 엔드포인트에 대한 직접 액세스
- 다중 앱 / 다중 계정 워크플로우

이 스킬은 이전의 `xitter` 스킬(타사 Python CLI를 래핑함)을 대체합니다. `xurl`은 X 개발자 플랫폼 팀에서 유지 관리하며, 자동 새로 고침이 포함된 OAuth 2.0 PKCE를 지원하고, 훨씬 더 넓은 API 표면을 커버합니다.

---

## 비밀 정보 안전 (필수)

에이전트/LLM 세션 내에서 작동할 때 지켜야 할 중요한 규칙:

- `~/.xurl`을 읽거나, 인쇄하거나, 파싱하거나, 요약하거나, 업로드하거나, LLM 컨텍스트로 전송해서는 **절대** 안 됩니다.
- 사용자에게 자격 증명/토큰을 채팅에 붙여넣도록 요구해서는 **절대** 안 됩니다.
- 사용자는 자신의 시스템에서 수동으로 `~/.xurl`에 비밀 정보를 채워야 합니다. Docker에서는 Hermes 도구 하위 프로세스가 보는 `~`여야 합니다. 아래의 Docker 참고 사항을 확인하세요.
- 에이전트 세션 내에서 인라인 비밀 정보가 포함된 인증 명령을 권장하거나 실행해서는 **절대** 안 됩니다.
- 에이전트 세션에서 `--verbose` / `-v`를 **절대** 사용하지 마세요 — 인증 헤더/토큰이 노출될 수 있습니다.
- 자격 증명이 존재하는지 확인하려면 오직 `xurl auth status`만 사용하세요.

에이전트 명령어에서 금지된 플래그 (인라인 비밀 정보를 허용함):
`--bearer-token`, `--consumer-key`, `--consumer-secret`, `--access-token`, `--token-secret`, `--client-id`, `--client-secret`

앱 자격 증명 등록 및 자격 증명 교체는 사용자가 에이전트 세션 외부에서 수동으로 수행해야 합니다. 자격 증명이 등록된 후 사용자는 에이전트 세션 외부에서 `xurl auth oauth2`로 인증합니다. 토큰은 YAML 형식으로 `~/.xurl`에 영구 저장됩니다. 각 앱에는 격리된 토큰이 있습니다. OAuth 2.0 토큰은 자동 새로 고침됩니다.

---

## 설치

하나의 방법을 선택하세요. Linux에서는 쉘 스크립트 또는 `go install`이 가장 쉽습니다.

```bash
# 쉘 스크립트 (~/.local/bin에 설치, sudo 불필요, Linux + macOS에서 작동)
curl -fsSL https://raw.githubusercontent.com/xdevplatform/xurl/main/install.sh | bash

# Homebrew (macOS)
brew install --cask xdevplatform/tap/xurl

# npm
npm install -g @xdevplatform/xurl

# Go
go install github.com/xdevplatform/xurl@latest
```

확인:

```bash
xurl --help
xurl auth status
```

`xurl`이 설치되어 있지만 `auth status`에 앱이나 토큰이 표시되지 않는 경우, 사용자는 수동으로 인증을 완료해야 합니다 — 다음 섹션을 참조하세요.

---

## 일회성 사용자 설정 (사용자가 에이전트 외부에서 실행)

이 단계는 비밀 정보를 붙여넣는 작업을 포함하므로 에이전트가 아닌 **사용자가 직접 수행해야 합니다**. 사용자에게 이 블록을 안내하고 직접 실행하지 마세요.

1. https://developer.x.com/en/portal/dashboard 에서 앱을 생성하거나 엽니다.
2. 리디렉션 URI를 `http://localhost:8080/callback`으로 설정합니다.
3. 앱의 Client ID와 Client Secret을 복사합니다.
4. 앱을 로컬에 등록합니다 (사용자가 실행):
   ```bash
   xurl auth apps add my-app --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET
   ```
5. 인증합니다 (토큰을 앱에 바인딩하기 위해 `--app`을 지정):
   ```bash
   xurl auth oauth2 --app my-app
   ```
   (OAuth 2.0 PKCE 플로우를 위한 브라우저가 열립니다.)

   OAuth 후 `/2/users/me` 조회 시 X에서 `UsernameNotFound` 오류나 403을 반환하는 경우, 핸들을 명시적으로 전달하세요 (xurl v1.1.0 이상):
   ```bash
   xurl auth oauth2 --app my-app YOUR_USERNAME
   ```
   이렇게 하면 토큰이 핸들에 바인딩되고 손상된 `/2/users/me` 호출을 건너뜁니다.
6. 모든 명령어가 해당 앱을 사용하도록 기본 앱으로 설정합니다:
   ```bash
   xurl auth default my-app
   ```
7. 확인:
   ```bash
   xurl auth status
   xurl whoami
   ```

이후 에이전트는 추가 설정 없이 아래의 모든 명령어를 사용할 수 있습니다. OAuth 2.0 토큰은 자동으로 새로 고침됩니다.

> **일반적인 함정:** `xurl auth oauth2`에서 `--app my-app`을 생략하면, OAuth 토큰이 client-id나 client-secret이 없는 내장 `default` 앱 프로필에 저장됩니다. OAuth 플로우가 성공한 것처럼 보여도 명령어는 인증 오류로 실패합니다. 이 문제가 발생하면 `xurl auth oauth2 --app my-app` 및 `xurl auth default my-app`을 다시 실행하세요.

> **Docker HOME 함정:** 공식 Hermes Docker 레이아웃에서 `/opt/data`는 `HERMES_HOME`이지만, Hermes 도구 하위 프로세스는 `/opt/data/home`을 `HOME`으로 사용합니다. 즉, Hermes가 실행하는 `xurl` 명령어의 `~/.xurl`은 `/opt/data/.xurl`이 아니라 `/opt/data/home/.xurl`로 해석됩니다. 동일한 HOME으로 사용자 설정을 실행하세요:
> ```bash
> HOME=/opt/data/home xurl auth apps add my-app --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET
> HOME=/opt/data/home xurl auth oauth2 --app my-app YOUR_USERNAME
> HOME=/opt/data/home xurl auth default my-app YOUR_USERNAME
> HOME=/opt/data/home xurl auth status
> ```
> `HOME=/opt/data xurl auth status`는 성공하지만 `HOME=/opt/data/home xurl auth status`에 앱이나 토큰이 표시되지 않는 경우, Hermes 도구 호출이 자격 증명을 볼 수 없습니다.

---

## 빠른 참조

| 작업 | 명령어 |
| --- | --- |
| 게시 | `xurl post "Hello world!"` |
| 답글 | `xurl reply POST_ID "Nice post!"` |
| 인용 | `xurl quote POST_ID "My take"` |
| 게시물 삭제 | `xurl delete POST_ID` |
| 게시물 읽기 | `xurl read POST_ID` |
| 게시물 검색 | `xurl search "QUERY" -n 10` |
| 내 정보 | `xurl whoami` |
| 사용자 조회 | `xurl user @handle` |
| 홈 타임라인 | `xurl timeline -n 20` |
| 멘션 | `xurl mentions -n 10` |
| 마음에 들어요 / 취소 | `xurl like POST_ID` / `xurl unlike POST_ID` |
| 재게시 / 취소 | `xurl repost POST_ID` / `xurl unrepost POST_ID` |
| 북마크 / 제거 | `xurl bookmark POST_ID` / `xurl unbookmark POST_ID` |
| 북마크 / 마음에 들어요 목록 | `xurl bookmarks -n 10` / `xurl likes -n 10` |
| 팔로우 / 언팔로우 | `xurl follow @handle` / `xurl unfollow @handle` |
| 팔로잉 / 팔로워 | `xurl following -n 20` / `xurl followers -n 20` |
| 차단 / 해제 | `xurl block @handle` / `xurl unblock @handle` |
| 뮤트 / 해제 | `xurl mute @handle` / `xurl unmute @handle` |
| DM 보내기 | `xurl dm @handle "message"` |
| DM 목록 | `xurl dms -n 10` |
| 미디어 업로드 | `xurl media upload path/to/file.mp4` |
| 미디어 상태 | `xurl media status MEDIA_ID` |
| 앱 목록 | `xurl auth apps list` |
| 앱 제거 | `xurl auth apps remove NAME` |
| 기본 앱 설정 | `xurl auth default APP_NAME [USERNAME]` |
| 요청별 앱 지정 | `xurl --app NAME /2/users/me` |
| 인증 상태 | `xurl auth status` |

참고:
- `POST_ID`는 전체 URL(예: `https://x.com/user/status/1234567890`)도 허용합니다 — xurl이 ID를 추출합니다.
- 사용자 이름은 선행 `@`가 있든 없든 작동합니다.

---

## 명령어 상세정보

### 게시 (Posting)

```bash
xurl post "Hello world!"
xurl post "Check this out" --media-id MEDIA_ID
xurl post "Thread pics" --media-id 111 --media-id 222

xurl reply 1234567890 "Great point!"
xurl reply https://x.com/user/status/1234567890 "Agreed!"
xurl reply 1234567890 "Look at this" --media-id MEDIA_ID

xurl quote 1234567890 "Adding my thoughts"
xurl delete 1234567890
```

### 읽기 및 검색 (Reading & Search)

```bash
xurl read 1234567890
xurl read https://x.com/user/status/1234567890

xurl search "golang"
xurl search "from:elonmusk" -n 20
xurl search "#buildinpublic lang:en" -n 15
```

X Articles의 경우, `read` 바로가기 대신 순수 API 모드를 사용하세요. `xurl read`는 게시물 ID 또는 게시물 URL을 기대합니다; `/2/tweets/...` 엔드포인트 앞에 `read`를 넣지 마세요. `article` 트윗 필드를 요청하고 JSON 응답에서 `data.article.plain_text`를 추출하세요:

```bash
xurl --app APP_NAME '/2/tweets/2057909493250539891?expansions=author_id,attachments.media_keys,referenced_tweets.id&tweet.fields=created_at,lang,public_metrics,context_annotations,entities,possibly_sensitive,conversation_id,in_reply_to_user_id,referenced_tweets,article'
```

### 사용자, 타임라인, 멘션

```bash
xurl whoami
xurl user elonmusk
xurl user @XDevelopers

xurl timeline -n 25
xurl mentions -n 20
```

### 참여 (Engagement)

```bash
xurl like 1234567890
xurl unlike 1234567890

xurl repost 1234567890
xurl unrepost 1234567890

xurl bookmark 1234567890
xurl unbookmark 1234567890

xurl bookmarks -n 20
xurl likes -n 20
```

### 소셜 그래프 (Social Graph)

```bash
xurl follow @XDevelopers
xurl unfollow @XDevelopers

xurl following -n 50
xurl followers -n 50

# 다른 사용자의 그래프
xurl following --of elonmusk -n 20
xurl followers --of elonmusk -n 20

xurl block @spammer
xurl unblock @spammer
xurl mute @annoying
xurl unmute @annoying
```

### 다이렉트 메시지 (Direct Messages)

```bash
xurl dm @someuser "Hey, saw your post!"
xurl dms -n 25
```

### 미디어 업로드 (Media Upload)

```bash
# 유형 자동 감지
xurl media upload photo.jpg
xurl media upload video.mp4

# 명시적 유형/카테고리
xurl media upload --media-type image/jpeg --category tweet_image photo.jpg

# 비디오는 서버 측 처리가 필요합니다 — 상태 확인 (또는 폴링)
xurl media status MEDIA_ID
xurl media status --wait MEDIA_ID

# 전체 워크플로우
xurl media upload meme.png                  # media id 반환
xurl post "lol" --media-id MEDIA_ID
```

---

## 직접 API 접근 (Raw API Access)

바로가기 명령어는 일반적인 작업을 다룹니다. 그 외의 경우에는 모든 X API v2 엔드포인트에 대해 curl 스타일의 순수 모드를 사용하세요:

```bash
# GET
xurl /2/users/me

# JSON 본문과 함께 POST
xurl -X POST /2/tweets -d '{"text":"Hello world!"}'

# DELETE / PUT / PATCH
xurl -X DELETE /2/tweets/1234567890

# 사용자 지정 헤더
xurl -H "Content-Type: application/json" /2/some/endpoint

# 스트리밍 강제 실행
xurl -s /2/tweets/search/stream

# 전체 URL도 작동합니다
xurl https://api.x.com/2/users/me
```

---

## 전역 플래그 (Global Flags)

| 플래그 | 짧은 플래그 | 설명 |
| --- | --- | --- |
| `--app` | | 특정 등록된 앱 사용 (기본값 재정의) |
| `--auth` | | 인증 유형 강제 지정: `oauth1`, `oauth2`, 또는 `app` |
| `--username` | `-u` | 사용할 OAuth2 계정 (여러 개 존재하는 경우) |
| `--verbose` | `-v` | **에이전트 세션에서 금지됨** — 인증 헤더 유출 |
| `--trace` | `-t` | `X-B3-Flags: 1` 추적 헤더 추가 |

---

## 스트리밍 (Streaming)

스트리밍 엔드포인트는 자동 감지됩니다. 알려진 엔드포인트는 다음과 같습니다:

- `/2/tweets/search/stream`
- `/2/tweets/sample/stream`
- `/2/tweets/sample10/stream`

`-s`를 사용하여 어떤 엔드포인트에서든 스트리밍을 강제할 수 있습니다.

---

## 출력 형식 (Output Format)

모든 명령어는 stdout으로 JSON을 반환합니다. 구조는 X API v2를 따릅니다:

```json
{ "data": { "id": "1234567890", "text": "Hello world!" } }
```

오류도 JSON 형식입니다:

```json
{ "errors": [ { "message": "Not authorized", "code": 403 } ] }
```

---

## 일반적인 워크플로우

### 이미지와 함께 게시
```bash
xurl media upload photo.jpg
xurl post "Check out this photo!" --media-id MEDIA_ID
```

### 대화에 답글 달기
```bash
xurl read https://x.com/user/status/1234567890
xurl reply 1234567890 "Here are my thoughts..."
```

### 검색 및 참여
```bash
xurl search "topic of interest" -n 10
xurl like POST_ID_FROM_RESULTS
xurl reply POST_ID_FROM_RESULTS "Great point!"
```

### 내 활동 확인
```bash
xurl whoami
xurl mentions -n 20
xurl timeline -n 20
```

### 다중 앱 (자격 증명이 수동으로 사전 구성됨)
```bash
xurl auth default prod alice               # prod 앱, alice 사용자
xurl --app staging /2/users/me             # staging에 대한 일회성 실행
```

---

## 오류 처리 (Error Handling)

- 오류 발생 시 0이 아닌 종료 코드를 반환합니다.
- API 오류도 여전히 stdout에 JSON으로 출력되므로 파싱할 수 있습니다.
- 인증 오류 → 사용자에게 에이전트 세션 외부에서 `xurl auth oauth2`를 다시 실행하도록 합니다.
- 발신자의 사용자 ID가 필요한 명령어(마음에 들어요, 재게시, 북마크, 팔로우 등)는 `/2/users/me`를 통해 자동으로 ID를 가져옵니다. 해당 단계의 인증 실패는 인증 오류로 나타납니다.

---

## 에이전트 워크플로우

1. 전제 조건 확인: `xurl --help` 및 `xurl auth status`.
2. **기본 앱에 자격 증명이 있는지 확인합니다.** `auth status` 출력을 파싱합니다. 기본 앱은 `▸`로 표시됩니다. 기본 앱이 `oauth2: (none)`으로 표시되지만 다른 앱에 유효한 oauth2 사용자가 있는 경우, 사용자에게 `xurl auth default <that-app>`을 실행하여 수정하도록 알려줍니다. 이것은 가장 흔한 설정 실수입니다 — 사용자가 사용자 지정 이름으로 앱을 추가했지만 기본 앱으로 설정하지 않아 xurl이 계속 비어있는 `default` 프로필을 시도하는 경우입니다.
3. 인증이 완전히 누락된 경우, 중지하고 사용자를 "일회성 사용자 설정" 섹션으로 안내합니다 — 스스로 앱을 등록하거나 비밀 정보를 전달하려고 시도하지 **마세요**.
4. 연결 가능성을 확인하기 위해 비용이 적게 드는 읽기 작업(`xurl whoami`, `xurl user @handle`, `xurl search ... -n 3`)부터 시작합니다.
5. 쓰기 작업(게시, 답글, 마음에 들어요, 재게시, DM, 팔로우, 차단, 삭제) 전에 대상 게시물/사용자와 사용자의 의도를 확인합니다.
6. JSON 출력을 직접 사용합니다 — 모든 응답은 이미 구조화되어 있습니다.
7. `~/.xurl` 내용을 대화에 다시 붙여넣지 마세요.

---

## 문제 해결 (Troubleshooting)

| 증상 | 원인 | 해결책 |
| --- | --- | --- |
| 성공적인 OAuth 흐름 이후의 인증 오류 | 토큰이 명명된 앱 대신 (client-id/secret이 없는) `default` 앱에 저장됨 | `xurl auth oauth2 --app my-app` 이후 `xurl auth default my-app` |
| OAuth 중 `unauthorized_client` 오류 | X 대시보드에서 앱 유형이 "Native App"으로 설정됨 | 사용자 인증 설정(User Authentication Settings)에서 "Web app, automated app or bot"으로 변경 |
| OAuth 직후 `/2/users/me`에서 `UsernameNotFound` 또는 403 오류 | X가 `/2/users/me`에서 사용자 이름을 안정적으로 반환하지 않음 | `xurl auth oauth2 --app my-app YOUR_USERNAME` (xurl v1.1.0 이상)을 다시 실행하여 핸들을 명시적으로 전달 |
| 모든 요청에 대해 401 오류 | 토큰이 만료되었거나 잘못된 기본 앱 | `xurl auth status`를 확인 — `▸`가 oauth2 토큰이 있는 앱을 가리키는지 확인 |
| `client-forbidden` / `client-not-enrolled` | X 플랫폼 등록 문제 | 대시보드 → Apps → Manage → "Pay-per-use" 패키지로 이동 → Production 환경 |
| `CreditsDepleted` | X API 잔고 $0 | Developer Console → Billing에서 크레딧 구매 (최소 $5) |
| 이미지 업로드 시 `media processing failed` | 기본 카테고리가 `amplify_video`임 | `--category tweet_image --media-type image/png` 추가 |
| X 대시보드에 두 개의 "Client Secret" 값이 있음 | UI 버그 — 첫 번째 값은 실제로는 Client ID임 | "Keys and tokens" 페이지에서 확인; ID는 `MTpjaQ`로 끝남 |

---

## 참고 사항

- **속도 제한 (Rate limits):** X는 엔드포인트별로 속도 제한을 시행합니다. 429는 대기 후 재시도하라는 의미입니다. 쓰기 엔드포인트(게시, 답글, 마음에 들어요, 재게시)는 읽기보다 더 엄격한 제한을 갖습니다.
- **범위 (Scopes):** OAuth 2.0 토큰은 광범위한 범위를 사용합니다. 특정 작업에 대한 403은 일반적으로 토큰에 범위가 누락되었음을 의미합니다 — 사용자가 `xurl auth oauth2`를 다시 실행하도록 합니다.
- **토큰 새로 고침:** OAuth 2.0 토큰은 자동으로 새로 고침됩니다. 아무것도 할 필요가 없습니다.
- **다중 앱:** 각 앱에는 격리된 자격 증명/토큰이 있습니다. `xurl auth default` 또는 `--app`으로 전환하세요.
- **앱당 다중 계정:** `-u / --username`으로 선택하거나 `xurl auth default APP USER`로 기본값을 설정하세요.
- **토큰 저장소:** `~/.xurl`은 YAML입니다. Docker에서는 Hermes 하위 프로세스 HOME(공식 이미지의 경우 `/opt/data/home`)을 사용하여 토큰이 `/opt/data/home/.xurl` 아래에 저장되도록 합니다. 이 파일을 읽거나 LLM 컨텍스트로 전송하지 마세요.
- **비용:** 의미 있는 X API 사용은 일반적으로 유료입니다. 많은 실패는 코드 문제가 아니라 요금제/권한 문제입니다.

---

## 출처 (Attribution)

- 업스트림 CLI: https://github.com/xdevplatform/xurl (X 개발자 플랫폼 팀, Chris Park 등)
- 업스트림 에이전트 스킬: https://github.com/openclaw/openclaw/blob/main/skills/xurl/SKILL.md
- Hermes 버전: Hermes 스킬 규약에 맞게 재형식화됨; 안전 가드레일은 원본 그대로 보존됨.
