---
title: "Here.Now — 정적 사이트를 {slug}에 게시하기"
sidebar_label: "Here.Now"
description: "정적 사이트를 {slug}에 게시하기"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Here.Now

정적 사이트를 &#123;slug&#125;.here.now에 게시하고 에이전트 간 핸드오프를 위해 클라우드 드라이브(Drives)에 개인 파일을 저장합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택적(Optional) — `hermes skills install official/productivity/here-now` 명령어로 설치 |
| 경로 | `optional-skills/productivity/here-now` |
| 버전 | `1.15.3` |
| 작성자 | here.now |
| 라이선스 | MIT |
| 플랫폼 | macos, linux |
| 태그 | `here.now`, `herenow`, `publish`, `deploy`, `hosting`, `static-site`, `web`, `share`, `URL`, `drive`, `storage` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 내용은 스킬이 활성화되어 있을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# here.now

here.now를 통해 에이전트는 웹사이트를 게시하고 클라우드 드라이브(Drives)에 개인 파일을 저장할 수 있습니다.

here.now를 다음 두 가지 작업에 사용하세요:

- **사이트(Sites)**: 웹사이트와 파일을 `{slug}.here.now`에 게시합니다.
- **드라이브(Drives)**: 클라우드 폴더에 비공개 에이전트 파일을 저장합니다.

## 최신 문서 (Current docs)

**here.now의 기능, 특징 또는 워크플로우에 대한 질문에 답하기 전에 최신 문서를 읽어보세요:**

→ **https://here.now/docs**

다음을 수행할 때 문서를 읽으세요:

- 대화에서 here.now와 관련된 첫 번째 상호작용 시
- 사용자가 방법을 물어볼 때마다
- 사용자가 무엇이 가능하고, 지원되며, 권장되는지 물어볼 때마다
- 사용자에게 기능이 지원되지 않는다고 말하기 전

최신 문서가 필요한 주제 (로컬 스킬 텍스트에만 의존하지 마세요):

- 드라이브 및 드라이브 공유 (Drives and Drive sharing)
- 커스텀 도메인 (custom domains)
- 결제 및 결제 제한 (payments and payment gating)
- 포킹 (forking)
- 프록시 라우팅 및 서비스 변수 (proxy routes and service variables)
- 핸들 및 링크 (handles and links)
- 한도 및 할당량 (limits and quotas)
- SPA 라우팅 (SPA routing)
- 오류 처리 및 문제 해결 (error handling and remediation)
- 기능 가용성 (feature availability)

**만약 문서와 실제 API 동작이 일치하지 않는다면, 실제 API 동작을 신뢰하세요.**

문서 가져오기에 실패하거나 시간이 초과되면, 로컬 스킬과 라이브 API/스크립트 출력으로 계속 진행합니다. 활성 작업을 수행할 때는 라이브 API 동작을 선호하세요.

## 요구 사항 (Requirements)

- 필수 바이너리: `curl`, `file`, `jq`
- 선택적 환경 변수: `$HERENOW_API_KEY`
- 선택적 드라이브 토큰 변수: `$HERENOW_DRIVE_TOKEN`
- 선택적 자격 증명 파일: `~/.herenow/credentials`
- 스킬 헬퍼 경로:
  - 사이트 게시용: `${HERMES_SKILL_DIR}/scripts/publish.sh`
  - 비공개 드라이브 저장소용: `${HERMES_SKILL_DIR}/scripts/drive.sh`

## 사이트 생성

```bash
PUBLISH="${HERMES_SKILL_DIR}/scripts/publish.sh"
bash "$PUBLISH" {file-or-dir} --client hermes
```

라이브 URL을 출력합니다 (예: `https://bright-canvas-a7k2.here.now/`).

내부적으로 이것은 3단계 흐름(생성/업데이트 -> 파일 업로드 -> 완료)입니다. 완료(finalize)에 성공해야 사이트가 활성화됩니다.

API 키가 없으면 24시간 후에 만료되는 **익명 사이트(anonymous site)**를 생성합니다.
저장된 API 키가 있으면 사이트가 영구적으로 유지됩니다.

**파일 구조:** HTML 사이트의 경우, 하위 디렉토리가 아닌 게시할 디렉토리의 루트(root)에 `index.html`을 배치하세요. 해당 디렉토리의 내용이 사이트의 루트가 됩니다. 예를 들어, `my-site/index.html`이 존재하는 경우 `my-site/`를 게시하세요 — `my-site/`를 포함하는 상위 폴더를 게시하지 마세요.

HTML 없이 원본(raw) 파일을 게시할 수도 있습니다. 단일 파일의 경우 풍부한 자동 뷰어(이미지, PDF, 비디오, 오디오)가 제공됩니다. 여러 파일의 경우 폴더 탐색 및 이미지 갤러리가 있는 디렉토리 목록이 자동 생성됩니다.

## 기존 사이트 업데이트

```bash
PUBLISH="${HERMES_SKILL_DIR}/scripts/publish.sh"
bash "$PUBLISH" {file-or-dir} --slug {slug} --client hermes
```

익명 사이트를 업데이트할 때 스크립트는 `.herenow/state.json`에서 `claimToken`을 자동으로 로드합니다. 오버라이드(override)하려면 `--claim-token {token}`을 전달하세요.

인증된 상태로 업데이트하려면 저장된 API 키가 필요합니다.

## 드라이브 사용 (Use a Drive)

사용자가 문서, 컨텍스트, 메모리, 계획, 에셋, 미디어, 연구 자료, 코드 및 웹사이트로 게시되지 않으면서 유지되어야 하는 모든 에이전트 파일을 위해 비공개 클라우드 스토리지를 원할 때 드라이브를 사용하세요.

로그인한 모든 계정에는 `My Drive`라는 이름의 기본 드라이브가 있습니다.

```bash
DRIVE="${HERMES_SKILL_DIR}/scripts/drive.sh"
bash "$DRIVE" default
bash "$DRIVE" ls "My Drive"
bash "$DRIVE" put "My Drive" notes/today.md --from ./notes/today.md
bash "$DRIVE" cat "My Drive" notes/today.md
bash "$DRIVE" share "My Drive" --perms write --prefix notes/ --ttl 7d
```

에이전트 간 핸드오프를 위해서는 범위가 지정된(scoped) 드라이브 토큰을 사용하세요. `herenow_drive` 공유 블록을 받은 경우, `api_base`에 대해 해당 `token`을 `Authorization: Bearer <token>`으로 사용하고, `pathPrefix`가 존재하는 경우 이를 존중하며, 쓰기 작업 시 ETag를 보존하세요. `pathPrefix`가 `null`이면 드라이브에 대한 전체 액세스 권한을 의미합니다. 스킬이 사용 가능한 경우 `drive.sh`를 선호하고, 그렇지 않으면 나열된 API 작업을 직접 호출하세요.

## API 키 저장 (API key storage)

게시 스크립트는 다음 소스에서 API 키를 읽습니다 (첫 번째로 일치하는 항목 적용):

1. `--api-key {key}` 플래그 (CI/스크립팅 전용 — 대화형 사용에서는 피하세요)
2. `$HERENOW_API_KEY` 환경 변수
3. `~/.herenow/credentials` 파일 (에이전트에게 권장됨)

키를 저장하려면 자격 증명 파일에 기록합니다:

```bash
mkdir -p ~/.herenow && echo "{API_KEY}" > ~/.herenow/credentials && chmod 600 ~/.herenow/credentials
```

**중요**: API 키를 받은 후 즉시 저장하세요 — 위의 명령어를 당신(에이전트)이 직접 실행하세요. 사용자에게 수동으로 실행하라고 요청하지 마세요. 대화형 세션에서 CLI 플래그(예: `--api-key`)를 통해 키를 전달하는 것은 피하세요. 자격 증명 파일이 권장되는 저장 방식입니다.

자격 증명이나 로컬 상태 파일(`~/.herenow/credentials`, `.herenow/state.json`)을 소스 제어(버전 관리)에 절대 커밋하지 마세요.

## API 키 얻기

익명 사이트(24시간)에서 영구 사이트로 업그레이드하려면:

1. 사용자에게 이메일 주소를 묻습니다.
2. 일회용 로그인 코드를 요청합니다:

```bash
curl -sS https://here.now/api/auth/agent/request-code \
  -H "content-type: application/json" \
  -d '{"email": "user@example.com"}'
```

3. 사용자에게 이렇게 말합니다: "받은 편지함에서 here.now가 보낸 로그인 코드를 확인하고 여기에 붙여넣어 주세요."
4. 코드를 확인하고 API 키를 얻습니다:

```bash
curl -sS https://here.now/api/auth/agent/verify-code \
  -H "content-type: application/json" \
  -d '{"email":"user@example.com","code":"ABCD-2345"}'
```

5. 반환된 `apiKey`를 당신(에이전트)이 직접 저장합니다 (사용자에게 이 작업을 요청하지 마세요):

```bash
mkdir -p ~/.herenow && echo "{API_KEY}" > ~/.herenow/credentials && chmod 600 ~/.herenow/credentials
```

## 상태 파일 (State file)

모든 사이트 생성/업데이트 후에, 스크립트는 작업 디렉토리의 `.herenow/state.json`에 기록합니다:

```json
{
  "publishes": {
    "bright-canvas-a7k2": {
      "siteUrl": "https://bright-canvas-a7k2.here.now/",
      "claimToken": "abc123",
      "claimUrl": "https://here.now/claim?slug=bright-canvas-a7k2&token=abc123",
      "expiresAt": "2026-02-18T01:00:00.000Z"
    }
  }
}
```

사이트를 생성하거나 업데이트하기 전에 이 파일을 확인하여 이전의 slug를 찾을 수 있습니다.
`.herenow/state.json`은 내부 캐시로만 취급하세요.
이 로컬 파일 경로를 URL로 제시하지 말고, 인증 모드, 만료 또는 클레임(claim) URL의 진실의 원천(source of truth)으로 사용하지 마세요.

## 사용자에게 말해야 할 내용

게시된 사이트의 경우:

- 항상 현재 실행된 스크립트의 `siteUrl`을 공유하세요.
- 스크립트의 표준 에러(stderr)에서 `publish_result.*` 라인을 읽고 따라 인증 모드를 결정하세요.
- `publish_result.auth_mode=authenticated`인 경우: 사이트가 **영구적**이며 그들의 계정에 저장되었다고 사용자에게 알리세요. 클레임 URL은 필요하지 않습니다.
- `publish_result.auth_mode=anonymous`인 경우: 사이트가 **24시간 후에 만료됨**을 사용자에게 알리세요. 그들이 사이트를 영구적으로 유지할 수 있도록 클레임 URL(`publish_result.claim_url`이 비어있지 않고 `https://`로 시작하는 경우)을 공유하세요. 클레임 토큰은 한 번만 반환되며 복구할 수 없음을 경고하세요.
- 사용자에게 클레임 URL이나 인증 상태를 확인하기 위해 `.herenow/state.json`을 직접 검사하라고 절대 말하지 마세요.

드라이브의 경우:

- 드라이브 파일을 공개 URL로 설명하지 마세요.
- 드라이브 콘텐츠는 범위가 지정된 토큰으로 공유되지 않는 한 비공개(private)라고 사용자에게 알리세요.
- 다른 에이전트와 액세스를 공유할 때는 좁은 `pathPrefix`와 짧은 TTL을 가진 범위 지정 토큰을 선호하세요.

## publish.sh 옵션

| 플래그                   | 설명                                  |
| ---------------------- | -------------------------------------------- |
| `--slug {slug}`        | 새 사이트를 생성하는 대신 기존 사이트를 업데이트 |
| `--claim-token {token}`| 익명 업데이트를 위한 클레임 토큰 오버라이드 |
| `--title {text}`       | 뷰어 제목 (HTML이 아닌 사이트용) |
| `--description {text}` | 뷰어 설명 |
| `--ttl {seconds}`      | 만료 시간 설정 (인증된 상태에서만) |
| `--client {name}`      | 출처 표시를 위한 에이전트 이름 (예: `hermes`) |
| `--base-url {url}`     | API 기본 URL (기본값: `https://here.now`) |
| `--allow-nonherenow-base-url` | 기본 `--base-url`이 아닌 곳으로 인증 정보를 보내는 것을 허용 |
| `--api-key {key}`      | API 키 오버라이드 (자격 증명 파일 방식이 권장됨) |
| `--spa`                | SPA 라우팅 활성화 (알 수 없는 경로의 경우 index.html 제공) |
| `--forkable`           | 다른 사람이 이 사이트를 포크(fork)할 수 있도록 허용 |

## publish.sh 이상의 기능

드라이브 작업을 위해 `drive.sh` 또는 드라이브 API를 사용하세요. 삭제, 메타데이터, 비밀번호, 결제, 도메인, 핸들, 링크, 변수, 프록시 라우팅, 포킹, 복제 및 기타 계정 및 사이트 관리에 대한 광범위한 내용은 최신 문서를 참조하세요:

→ **https://here.now/docs**

전체 문서: https://here.now/docs
