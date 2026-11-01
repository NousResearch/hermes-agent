---
title: "Xurl — xurl CLI를 통한 X/Twitter: 원문 게시물 검색, 게시, DM, 미디어"
sidebar_label: "Xurl"
description: "xurl CLI를 통한 X/Twitter: 원문 게시물 검색, 게시, DM, 미디어"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Xurl

xurl CLI를 통한 X/Twitter: 원문 게시물 검색, 게시, DM, 미디어.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 제공(기본 설치됨) |
| 경로 | `skills/social-media/xurl` |
| 버전 | `1.1.3` |
| 작성자 | xdevplatform + openclaw + Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos |
| 태그 | `twitter`, `x`, `social-media`, `xurl`, `official-api` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# xurl — 공식 CLI를 통한 X(Twitter) API

`xurl`은 X API를 위한 X 개발자 플랫폼의 공식 CLI입니다. 일반적인 작업을 위한 단축 명령과 모든 v2 엔드포인트에 접근할 수 있는 원시 curl 스타일 액세스를 모두 지원합니다. 모든 명령은 JSON을 stdout으로 반환합니다.

다음 작업에 이 스킬을 사용하세요:
- 게시물 게시, 답글, 인용, 삭제
- 원문 게시물 검색(상호작용할 수 있는 실제 게시물 JSON과 ID) 및 타임라인/멘션 읽기
- 좋아요, 리포스트, 북마크
- 팔로우, 언팔로우, 차단, 뮤트
- 다이렉트 메시지
- 미디어 업로드(이미지 및 동영상)
- 모든 X API v2 엔드포인트에 대한 원시 액세스
- 여러 앱 / 여러 계정 워크플로

이 스킬은 서드파티 Python CLI를 감싸던 기존 `xitter` 스킬을 대체합니다. `xurl`은 X 개발자 플랫폼 팀이 유지 관리하며 OAuth 2.0 PKCE와 자동 갱신을 지원하고, 훨씬 더 넓은 API 표면을 다룹니다.

---

## 시크릿 안전(필수)

에이전트/LLM 세션 내부에서 작업할 때의 핵심 규칙:

- **`~/.xurl`을 LLM 컨텍스트로 절대 읽거나, 출력하거나, 파싱하거나, 요약하거나, 업로드하거나, 전송하지 마세요.**
- **사용자에게 자격 증명/토큰을 채팅에 붙여넣으라고 절대 요청하지 마세요.**
- 사용자는 자신의 컴퓨터에서 직접 `~/.xurl`에 시크릿을 입력해야 합니다. Docker에서는 아래 Docker 참고 사항에 설명된 대로 Hermes 도구 서브프로세스에서 보이는 `~`이어야 합니다.
- **에이전트 세션에서 인라인 시크릿을 포함한 인증 명령을 절대 권장하거나 실행하지 마세요.**
- **에이전트 세션에서는 절대 `--verbose` / `-v`를 사용하지 마세요.** 인증 헤더/토큰이 노출될 수 있습니다.
- 자격 증명이 존재하는지 확인할 때는 `xurl auth status`만 사용하세요.

에이전트 명령에서 금지된 플래그(인라인 시크릿을 허용함):
`--bearer-token`, `--consumer-key`, `--consumer-secret`, `--access-token`, `--token-secret`, `--client-id`, `--client-secret`

앱 자격 증명 등록과 자격 증명 교체는 에이전트 세션 외부에서 사용자가 직접 수행해야 합니다. 자격 증명이 등록된 후 사용자는 에이전트 세션 외부에서 `xurl auth oauth2`로 인증합니다. 토큰은 YAML 형식으로 `~/.xurl`에 저장됩니다. 각 앱에는 격리된 토큰이 있습니다. OAuth 2.0 토큰은 자동으로 갱신됩니다.

---

## 설치

방법을 하나만 선택하세요. Linux에서는 셸 스크립트나 `go install`이 가장 간단합니다.

```bash
# Shell script (installs to ~/.local/bin, no sudo, works on Linux + macOS)
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

`xurl`이 설치되어 있지만 `auth status`에 앱이나 토큰이 표시되지 않으면 사용자가 다음 섹션을 참고해 직접 인증을 완료해야 합니다.

---

## 최초 사용자 설정(사용자가 외부에서 실행)

이 단계는 사용자가 직접 수행해야 하며, 에이전트가 수행해서는 안 됩니다. 시크릿 붙여넣기가 필요하기 때문입니다. 사용자에게 이 블록을 안내하되, 대신 실행하지 마세요.

1. https://developer.x.com/en/portal/dashboard 에서 앱을 생성하거나 엽니다.
2. 리디렉션 URI를 `http://localhost:8080/callback`으로 설정합니다.
3. 앱의 Client ID와 Client Secret을 복사합니다.
4. 앱을 로컬에 등록합니다(사용자가 실행):
   ```bash
   xurl auth apps add my-app --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET
   ```
5. 인증합니다(토큰을 앱에 연결하려면 `--app`을 지정):
   ```bash
   xurl auth oauth2 --app my-app
   ```
   (OAuth 2.0 PKCE 플로를 위해 브라우저가 열립니다.)

   X가 post-OAuth `/2/users/me` 조회에서 `UsernameNotFound` 오류 또는 403을 반환하면 핸들을 명시적으로 전달하세요(xurl v1.1.0 이상):
   ```bash
   xurl auth oauth2 --app my-app YOUR_USERNAME
   ```
   이렇게 하면 토큰이 해당 핸들에 연결되고 문제가 있는 `/2/users/me` 호출을 건너뜁니다.
6. 모든 명령이 이 앱을 사용하도록 앱을 기본값으로 설정합니다.
   ```bash
   xurl auth default my-app
   ```
7. 확인합니다.
   ```bash
   xurl auth status
   xurl whoami
   ```

이후에는 추가 설정 없이 에이전트가 아래의 모든 명령을 사용할 수 있습니다. OAuth 2.0 토큰은 자동으로 갱신됩니다.

> **흔한 문제:** `xurl auth oauth2`에서 `--app my-app`을 생략하면 OAuth 토큰이 내장된 `default` 앱 프로필에 저장됩니다. 이 프로필에는 client-id나 client-secret이 없습니다. OAuth 플로가 성공한 것처럼 보여도 명령은 인증 오류로 실패합니다. 이 경우 `xurl auth oauth2 --app my-app`과 `xurl auth default my-app`을 다시 실행하세요.

> **Docker HOME 문제:** 공식 Hermes Docker 레이아웃에서 `/opt/data`는 `HERMES_HOME`이지만 Hermes 도구 서브프로세스는 `/opt/data/home`을 `HOME`으로 사용합니다. 따라서 Hermes가 실행하는 `xurl` 명령에서 `~/.xurl`은 `/opt/data/.xurl`이 아니라 `/opt/data/home/.xurl`로 해석됩니다. 동일한 HOME으로 사용자 설정을 실행하세요.
> ```bash
> HOME=/opt/data/home xurl auth apps add my-app --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET
> HOME=/opt/data/home xurl auth oauth2 --app my-app YOUR_USERNAME
> HOME=/opt/data/home xurl auth default my-app YOUR_USERNAME
> HOME=/opt/data/home xurl auth status
> ```
> `HOME=/opt/data xurl auth status`는 성공하지만 `HOME=/opt/data/home xurl auth status`에 앱이나 토큰이 표시되지 않는다면 Hermes 도구 호출에서는 자격 증명을 볼 수 없습니다.

---

## 빠른 참조

| 작업 | 명령 |
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
| 좋아요 / 좋아요 취소 | `xurl like POST_ID` / `xurl unlike POST_ID` |
| 리포스트 / 취소 | `xurl repost POST_ID` / `xurl unrepost POST_ID` |
| 북마크 / 삭제 | `xurl bookmark POST_ID` / `xurl unbookmark POST_ID` |
| 북마크 / 좋아요 목록 | `xurl bookmarks -n 10` / `xurl likes -n 10` |
| 팔로우 / 언팔로우 | `xurl follow @handle` / `xurl unfollow @handle` |
| 팔로잉 / 팔로워 | `xurl following -n 20` / `xurl followers -n 20` |
| 차단 / 차단 해제 | `xurl block @handle` / `xurl unblock @handle` |
| 뮤트 / 뮤트 해제 | `xurl mute @handle` / `xurl unmute @handle` |
| DM 보내기 | `xurl dm @handle "message"` |
| DM 목록 | `xurl dms -n 10` |
| 미디어 업로드 | `xurl media upload path/to/file.mp4` |
| 미디어 상태 | `xurl media status MEDIA_ID` |
| 앱 목록 | `xurl auth apps list` |
| 앱 제거 | `xurl auth apps remove NAME` |
| 기본 앱 설정 | `xurl auth default APP_NAME [USERNAME]` |
| 요청별 앱 | `xurl --app NAME /2/users/me` |
| 인증 상태 | `xurl auth status` |

참고:
- `POST_ID`에는 전체 URL도 사용할 수 있습니다(예: `https://x.com/user/status/1234567890`). xurl이 ID를 추출합니다.
- 사용자 이름은 앞에 `@`이 있어도 되고 없어도 됩니다.

---

## 명령 세부 정보

### 게시

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

### 읽기 및 검색

`xurl search`는 인증된 계정으로 X 색인을 조회하고 원문 게시물 객체(게시물 ID, 작성자, 전체 텍스트)를 반환하므로 결과에 즉시 상호작용(답글, 좋아요, 리포스트, 인용)할 수 있습니다. 주제에 대한 요약 답변이 아니라 실제 게시물이 필요할 때 사용하세요.

```bash
xurl read 1234567890
xurl read https://x.com/user/status/1234567890

xurl search "golang"
xurl search "from:elonmusk" -n 20
xurl search "#buildinpublic lang:en" -n 15
```

X Articles에는 `read` 단축 명령 대신 원시 API 모드를 사용하세요. `xurl read`는 게시물 ID 또는 게시물 URL을 예상하므로 `/2/tweets/...` 엔드포인트 앞에 `read`를 넣지 마세요. `article` 트윗 필드를 요청하고 JSON 응답의 `data.article.plain_text`를 수집하세요.

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

### 상호작용

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

### 소셜 그래프

```bash
xurl follow @XDevelopers
xurl unfollow @XDevelopers

xurl following -n 50
xurl followers -n 50

# Another user's graph
xurl following --of elonmusk -n 20
xurl followers --of elonmusk -n 20

xurl block @spammer
xurl unblock @spammer
xurl mute @annoying
xurl unmute @annoying
```

### 다이렉트 메시지

```bash
xurl dm @someuser "Hey, saw your post!"
xurl dms -n 25
```

### 미디어 업로드

```bash
# Auto-detect type
xurl media upload photo.jpg
xurl media upload video.mp4

# Explicit type/category
xurl media upload --media-type image/jpeg --category tweet_image photo.jpg

# Videos need server-side processing — check status (or poll)
xurl media status MEDIA_ID
xurl media status --wait MEDIA_ID

# Full workflow
xurl media upload meme.png                  # returns media id
xurl post "lol" --media-id MEDIA_ID
```

---

## 원시 API 액세스

단축 명령은 일반적인 작업을 다룹니다. 그 외의 작업에는 모든 X API v2 엔드포인트를 대상으로 원시 curl 스타일 모드를 사용하세요.

```bash
# GET
xurl /2/users/me

# POST with JSON body
xurl -X POST /2/tweets -d '{"text":"Hello world!"}'

# DELETE / PUT / PATCH
xurl -X DELETE /2/tweets/1234567890

# Custom headers
xurl -H "Content-Type: application/json" /2/some/endpoint

# Force streaming
xurl -s /2/tweets/search/stream

# Full URLs also work
xurl https://api.x.com/2/users/me
```

---

## 전역 플래그

| 플래그 | 단축형 | 설명 |
| --- | --- | --- |
| `--app` | | 특정 등록 앱 사용(기본값 재정의) |
| `--auth` | | 인증 유형 강제: `oauth1`, `oauth2`, 또는 `app` |
| `--username` | `-u` | 사용할 OAuth2 계정(여러 계정이 있는 경우) |
| `--verbose` | `-v` | **에이전트 세션에서 금지** — 인증 헤더가 유출됨 |
| `--trace` | `-t` | `X-B3-Flags: 1` 추적 헤더 추가 |

---

## 스트리밍

스트리밍 엔드포인트는 자동으로 감지됩니다. 알려진 엔드포인트에는 다음이 포함됩니다.

- `/2/tweets/search/stream`
- `/2/tweets/sample/stream`
- `/2/tweets/sample10/stream`

어떤 엔드포인트에서든 `-s`로 스트리밍을 강제할 수 있습니다.

---

## 출력 형식

모든 명령은 JSON을 stdout으로 반환합니다. 구조는 X API v2를 반영합니다.

```json
{ "data": { "id": "1234567890", "text": "Hello world!" } }
```

오류도 JSON으로 반환됩니다.

```json
{ "errors": [ { "message": "Not authorized", "code": 403 } ] }
```

---

## 일반적인 워크플로

### 이미지와 함께 게시
```bash
xurl media upload photo.jpg
xurl post "Check out this photo!" --media-id MEDIA_ID
```

### 대화에 답글
```bash
xurl read https://x.com/user/status/1234567890
xurl reply 1234567890 "Here are my thoughts..."
```

### 검색 후 상호작용
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

### 여러 앱(자격 증명은 수동으로 사전 설정)
```bash
xurl auth default prod alice               # prod app, alice user
xurl --app staging /2/users/me             # one-off against staging
```

---

## 오류 처리

- 오류가 발생하면 0이 아닌 종료 코드를 반환합니다.
- API 오류도 JSON으로 stdout에 출력되므로 파싱할 수 있습니다.
- 인증 오류가 발생하면 에이전트 세션 외부에서 사용자가 `xurl auth oauth2`를 다시 실행하도록 안내하세요.
- 호출자의 사용자 ID가 필요한 명령(좋아요, 리포스트, 북마크, 팔로우 등)은 `/2/users/me`를 통해 자동으로 ID를 가져옵니다. 해당 요청에서 인증에 실패하면 인증 오류가 표시됩니다.

---

## 에이전트 워크플로

1. 사전 요구 사항을 확인합니다: `xurl --help` 및 `xurl auth status`.
2. `xurl search`를 사용하기 전에 의도를 확인합니다. 실제 게시물 객체, 인증된 계정 컨텍스트 또는 X 쓰기 작업으로 이어지는 결과가 필요한 경우 사용하세요. 사용자가 참여할 수 있는 게시물을 원하고 주제에 대한 요약만 원하는 것이 아닐 때 적합한 표면입니다.
3. **기본 앱에 자격 증명이 있는지 확인합니다.** `auth status` 출력을 파싱합니다. 기본 앱에는 `▸`가 표시됩니다. 기본 앱에 `oauth2: (none)`이 표시되지만 다른 앱에 유효한 oauth2 사용자가 있다면 사용자가 `xurl auth default <that-app>`을 실행하도록 안내하세요. 가장 흔한 설정 실수는 사용자가 사용자 지정 이름으로 앱을 추가했지만 기본값으로 설정하지 않아 xurl이 빈 `default` 프로필을 계속 사용하려는 것입니다.
4. 인증이 완전히 누락된 경우 중지하고 "최초 사용자 설정" 섹션을 안내하세요. 앱을 등록하거나 직접 시크릿을 전달하려고 하지 마세요.
5. 저비용 읽기 작업(`xurl whoami`, `xurl user @handle`, `xurl search ... -n 3`)으로 시작해 연결 가능성을 확인합니다.
6. 쓰기 작업(게시, 답글, 좋아요, 리포스트, DM, 팔로우, 차단, 삭제)을 수행하기 전에 대상 게시물/사용자와 사용자의 의도를 확인합니다.
7. 상태를 변경하는 X 작업이 실제로 발생했다는 증거는 `xurl` 명령 출력(또는 원시 X API 응답)뿐입니다. 검색 결과, 요약 또는 이전 컨텍스트를 근거로 쓰기 작업이 완료되었다고 절대 보고하지 마세요.
8. JSON 출력을 직접 사용하세요. 모든 응답은 이미 구조화되어 있습니다.
9. `~/.xurl`의 내용을 대화에 붙여넣지 마세요.

---

## 문제 해결

| 증상 | 원인 | 해결 방법 |
| --- | --- | --- |
| OAuth 플로가 성공한 후 인증 오류 | 토큰이 이름을 지정한 앱이 아니라 `default` 앱에 저장됨(client-id/secret 없음) | `xurl auth oauth2 --app my-app`을 실행한 다음 `xurl auth default my-app` 실행 |
| OAuth 중 `unauthorized_client` | X 대시보드에서 앱 유형이 "Native App"으로 설정됨 | User Authentication Settings에서 "Web app, automated app or bot"으로 변경 |
| OAuth 직후 `/2/users/me`에서 `UsernameNotFound` 또는 403 | X가 `/2/users/me`에서 사용자 이름을 안정적으로 반환하지 않음 | `xurl auth oauth2 --app my-app YOUR_USERNAME`을 다시 실행해 핸들을 명시적으로 전달(xurl v1.1.0 이상) |
| 모든 요청에서 401 | 토큰 만료 또는 잘못된 기본 앱 | `xurl auth status`를 확인하고 `▸`가 oauth2 토큰이 있는 앱을 가리키는지 확인 |
| `client-forbidden` / `client-not-enrolled` | X 플랫폼 등록 문제 | Dashboard → Apps → Manage → "Pay-per-use" 패키지 → Production 환경으로 이동 |
| `CreditsDepleted` | X API 잔액이 $0 | Developer Console → Billing에서 크레딧 구매(최소 $5) |
| 이미지 업로드에서 `media processing failed` | 기본 카테고리가 `amplify_video`임 | `--category tweet_image --media-type image/png` 추가 |
| X 대시보드에 "Client Secret" 값이 2개 | UI 버그 — 첫 번째 값은 실제로 Client ID | "Keys and tokens" 페이지에서 확인. ID는 `MTpjaQ`로 끝남 |

---

## 참고 사항

- **속도 제한:** X는 엔드포인트별 속도 제한을 적용합니다. 429가 발생하면 기다렸다가 다시 시도하세요. 쓰기 엔드포인트(게시, 답글, 좋아요, 리포스트)는 읽기보다 제한이 더 엄격합니다.
- **스코프:** OAuth 2.0 토큰은 넓은 스코프를 사용합니다. 특정 작업에서 403이 발생하면 보통 토큰에 해당 스코프가 없다는 뜻이므로 사용자가 `xurl auth oauth2`를 다시 실행하도록 하세요.
- **토큰 갱신:** OAuth 2.0 토큰은 자동으로 갱신됩니다. 별도로 할 일이 없습니다.
- **여러 앱:** 각 앱에는 격리된 자격 증명/토큰이 있습니다. `xurl auth default` 또는 `--app`으로 전환하세요.
- **앱당 여러 계정:** `-u / --username`으로 선택하거나 `xurl auth default APP USER`로 기본값을 설정하세요.
- **토큰 저장:** `~/.xurl`은 YAML입니다. Docker에서는 Hermes 서브프로세스 HOME(공식 이미지에서는 `/opt/data/home`)을 사용해 토큰이 `/opt/data/home/.xurl` 아래에 저장되도록 하세요. 이 파일을 LLM 컨텍스트로 절대 읽거나 전송하지 마세요.
- **비용:** X API 액세스는 의미 있는 사용량에 대해 일반적으로 유료입니다. 많은 실패는 코드 문제가 아니라 요금제/권한 문제입니다.

---

## 출처

- 업스트림 CLI: https://github.com/xdevplatform/xurl (X 개발자 플랫폼 팀, Chris Park 외)
- 업스트림 에이전트 스킬: https://github.com/openclaw/openclaw/blob/main/skills/xurl/SKILL.md
- Hermes 적용: Hermes 스킬 규칙에 맞게 형식을 다시 지정했으며, 안전 가드레일은 원문 그대로 유지했습니다.
