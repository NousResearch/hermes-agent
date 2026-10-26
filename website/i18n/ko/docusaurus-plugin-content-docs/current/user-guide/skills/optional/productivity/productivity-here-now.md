---
title: "Here Now — {slug}.here.now에 사이트를 게시하고 Drive에 파일 저장"
sidebar_label: "Here Now"
description: "{slug}.here.now에 사이트를 게시하고 Drive에 파일 저장"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Here Now

&#123;slug&#125;.here.now에 사이트를 게시하고 Drive에 파일을 저장합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/productivity/here-now`로 설치 |
| 경로 | `optional-skills/productivity/here-now` |
| 버전 | `1.15.3` |
| 작성자 | here.now |
| 라이선스 | MIT |
| 플랫폼 | macos, linux |
| 태그 | `here.now`, `herenow`, `publish`, `deploy`, `hosting`, `static-site`, `web`, `share`, `URL`, `drive`, `storage` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 보게 되는 내용입니다.
:::

# here.now

here.now를 사용하면 에이전트가 웹사이트를 게시하고 비공개 파일을 클라우드 Drive에 저장할 수 있습니다.

here.now는 다음 두 가지 작업에 사용합니다.

- **사이트**: `{slug}.here.now`에서 웹사이트와 파일을 게시합니다.
- **Drive**: 비공개 에이전트 파일을 클라우드 폴더에 저장합니다.

## 최신 문서

**here.now의 기능, 특징 또는 워크플로에 관한 질문에 답하기 전에 최신 문서를 읽으세요.**

→ **https://here.now/docs**

문서를 읽어야 하는 경우:

- 대화에서 here.now와 관련된 상호작용이 처음 발생했을 때
- 사용자가 작업 방법을 물을 때마다
- 사용자가 무엇이 가능하고, 지원되며, 권장되는지 물을 때마다
- 기능이 지원되지 않는다고 말하기 전에

다음 주제에는 현재 문서가 필요합니다(로컬 스킬 텍스트만으로 판단하지 마세요).

- Drive와 Drive 공유
- 사용자 지정 도메인
- 결제와 결제 게이팅
- 포크
- 프록시 라우트와 서비스 변수
- 핸들과 링크
- 제한과 할당량
- SPA 라우팅
- 오류 처리와 해결
- 기능 제공 여부

문서와 실제 API 동작이 일치하지 않으면 실제 API 동작을 신뢰하세요.

문서 가져오기가 실패하거나 시간 초과되면 로컬 스킬과 실제 API/스크립트 출력으로 계속 진행하세요. 활성 작업에서는 실제 API 동작을 우선하세요.

## 요구 사항

- 필수 바이너리: `curl`, `file`, `jq`
- 선택적 환경 변수: `$HERENOW_API_KEY`
- 선택적 Drive 토큰 변수: `$HERENOW_DRIVE_TOKEN`
- 선택적 자격 증명 파일: `~/.herenow/credentials`
- 스킬 도우미 경로:
  - 게시용 `${HERMES_SKILL_DIR}/scripts/publish.sh`
  - 비공개 Drive 저장용 `${HERMES_SKILL_DIR}/scripts/drive.sh`

## 사이트 만들기

```bash
PUBLISH="${HERMES_SKILL_DIR}/scripts/publish.sh"
bash "$PUBLISH" {file-or-dir} --client hermes
```

실제 URL을 출력합니다(예: `https://bright-canvas-a7k2.here.now/`).

내부적으로는 생성/업데이트 → 파일 업로드 → 완료라는 3단계 흐름입니다. 완료 단계가 성공할 때까지 사이트는 공개되지 않습니다.

API 키가 없으면 24시간 후 만료되는 **익명 사이트**가 생성됩니다.
저장된 API 키가 있으면 사이트가 영구적으로 유지됩니다.

**파일 구조:** HTML 사이트의 경우 게시하는 디렉터리의 루트에 `index.html`을 두고 하위 디렉터리 안에는 두지 마세요. 디렉터리의 내용이 사이트 루트가 됩니다. 예를 들어 `my-site/index.html`이 있는 `my-site/`를 게시하세요. `my-site/`를 포함하는 상위 폴더를 게시하면 안 됩니다.

HTML 없이 원시 파일만 게시할 수도 있습니다. 단일 파일에는 다양한 형식을 지원하는 자동 뷰어(이미지, PDF, 동영상, 오디오)가 제공됩니다. 여러 파일에는 폴더 탐색과 이미지 갤러리가 포함된 자동 생성 디렉터리 목록이 제공됩니다.

## 기존 사이트 업데이트

```bash
PUBLISH="${HERMES_SKILL_DIR}/scripts/publish.sh"
bash "$PUBLISH" {file-or-dir} --slug {slug} --client hermes
```

스크립트는 익명 사이트를 업데이트할 때 `.herenow/state.json`에서 `claimToken`을 자동으로 로드합니다. 이를 덮어쓰려면 `--claim-token {token}`을 전달하세요.

인증된 업데이트에는 저장된 API 키가 필요합니다.

## Drive 사용

사용자가 웹사이트로 게시하지 않고도 유지해야 하는 문서, 컨텍스트, 메모리, 계획, 에셋, 미디어, 조사 자료, 코드 및 기타 에이전트 파일을 비공개 클라우드 저장소에 보관하려는 경우 Drive를 사용하세요.

로그인한 모든 계정에는 `My Drive`라는 기본 Drive가 있습니다.

```bash
DRIVE="${HERMES_SKILL_DIR}/scripts/drive.sh"
bash "$DRIVE" default
bash "$DRIVE" ls "My Drive"
bash "$DRIVE" put "My Drive" notes/today.md --from ./notes/today.md
bash "$DRIVE" cat "My Drive" notes/today.md
bash "$DRIVE" share "My Drive" --perms write --prefix notes/ --ttl 7d
```

에이전트 간 인계에는 범위가 지정된 Drive 토큰을 사용하세요. `herenow_drive` 공유 블록을 받으면 해당 `token`을 `api_base`에 대한 `Authorization: Bearer <token>`으로 사용하고, 제공된 경우 `pathPrefix`를 준수하며, 쓰기 작업에서는 ETag을 유지하세요. 스킬을 사용할 수 있으면 `drive.sh`를 우선 사용하고, 그렇지 않으면 나열된 API 작업을 직접 호출하세요.

## API 키 저장

게시 스크립트는 다음 소스에서 API 키를 읽습니다(먼저 일치하는 항목 사용).

1. `--api-key {key}` 플래그(CI/스크립팅 전용 — 대화형 사용에서는 피하세요)
2. `$HERENOW_API_KEY` 환경 변수
3. `~/.herenow/credentials` 파일(에이전트에 권장)

키를 저장하려면 자격 증명 파일에 기록하세요.

```bash
mkdir -p ~/.herenow && echo "{API_KEY}" > ~/.herenow/credentials && chmod 600 ~/.herenow/credentials
```

**중요:** API 키를 받은 후 즉시 저장하세요 — 위 명령을 직접 실행하세요. 사용자에게 수동으로 실행하도록 요청하지 마세요. 대화형 세션에서는 CLI 플래그(예: `--api-key`)로 키를 전달하지 말고, 권장되는 자격 증명 파일을 사용하세요.

자격 증명이나 로컬 상태 파일(`~/.herenow/credentials`, `.herenow/state.json`)은 절대 소스 제어에 커밋하지 마세요.

## API 키 받기

익명 사이트를 영구 사이트로 업그레이드하려면 다음 단계를 따르세요.

1. 사용자에게 이메일 주소를 요청합니다.
2. 일회성 로그인 코드 요청:

```bash
curl -sS https://here.now/api/auth/agent/request-code \
  -H "content-type: application/json" \
  -d '{"email": "user@example.com"}'
```

3. 사용자에게 다음과 같이 말합니다. "here.now에서 보낸 로그인 코드가 받은 편지함에 도착했는지 확인하고 여기에 붙여 넣어 주세요."
4. 코드를 확인하고 API 키 받기:

```bash
curl -sS https://here.now/api/auth/agent/verify-code \
  -H "content-type: application/json" \
  -d '{"email":"user@example.com","code":"ABCD-2345"}'
```

5. 반환된 `apiKey`를 직접 저장하세요(사용자에게 저장을 요청하지 마세요).

```bash
mkdir -p ~/.herenow && echo "{API_KEY}" > ~/.herenow/credentials && chmod 600 ~/.herenow/credentials
```

## 상태 파일

사이트를 만들거나 업데이트할 때마다 스크립트는 작업 디렉터리의 `.herenow/state.json`에 다음과 같이 기록합니다.

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

사이트를 만들거나 업데이트하기 전에 이 파일을 확인하여 이전 슬러그를 찾을 수 있습니다.
`.herenow/state.json`은 내부 캐시로만 취급하세요.
인증 모드, 만료 또는 클레임 URL의 기준 정보로 이 파일을 사용하거나 이 로컬 파일 경로를 URL로 제시하지 마세요.

## 사용자에게 알려야 할 내용

게시된 사이트의 경우:

- 항상 현재 스크립트 실행에서 나온 `siteUrl`을 공유하세요.
- 스크립트 stderr의 `publish_result.*` 줄을 읽어 인증 모드를 확인하고 따르세요.
- `publish_result.auth_mode=authenticated`인 경우 사이트가 **영구적**이며 사용자 계정에 저장되었다고 알려 주세요. 클레임 URL은 필요하지 않습니다.
- `publish_result.auth_mode=anonymous`인 경우 사이트가 **24시간 후 만료**된다고 알려 주세요. `publish_result.claim_url`이 비어 있지 않고 `https://`로 시작하면 사용자가 영구적으로 유지할 수 있도록 클레임 URL을 공유하세요. 클레임 토큰은 한 번만 반환되며 복구할 수 없다고 경고하세요.
- 인증 상태나 클레임 URL을 확인하기 위해 `.herenow/state.json`을 살펴보라고 절대 말하지 마세요.

Drive의 경우:

- Drive 파일을 공개 URL로 설명하지 마세요.
- 공유 범위가 지정된 토큰을 사용해 공유하지 않는 한 Drive 콘텐츠는 비공개라고 알려 주세요.
- 다른 에이전트와 액세스를 공유할 때는 범위가 좁고 TTL이 짧은 토큰을 우선 사용하세요.

## publish.sh 옵션

| 플래그 | 설명 |
| ---------------------- | -------------------------------------------- |
| `--slug {slug}` | 새로 만드는 대신 기존 사이트 업데이트 |
| `--claim-token {token}`| 익명 업데이트에 사용할 클레임 토큰 덮어쓰기 |
| `--title {text}` | 뷰어 제목(HTML이 아닌 사이트) |
| `--description {text}` | 뷰어 설명 |
| `--ttl {seconds}` | 만료 시간 설정(인증된 경우에만)               |
| `--client {name}` | 기여 표시에 사용할 에이전트 이름(예: `hermes`)    |
| `--base-url {url}` | API 기본 URL(기본값: `https://here.now`)          |
| `--allow-nonherenow-base-url` | 기본값이 아닌 `--base-url`로 인증 정보 전송 허용 |
| `--api-key {key}` | API 키 재정의(자격 증명 파일 우선)    |
| `--spa` | SPA 라우팅 활성화(알 수 없는 경로에 index.html 제공) |
| `--forkable` | 다른 사용자가 이 사이트를 포크할 수 있도록 허용                           |

## publish.sh 외 기능

Drive 작업에는 `drive.sh` 또는 Drive API를 사용하세요. 삭제, 메타데이터, 비밀번호, 결제, 도메인, 핸들, 링크, 변수, 프록시 라우트, 포크, 복제 등 더 광범위한 계정 및 사이트 관리 방법은 최신 문서를 참조하세요.

→ **https://here.now/docs**

전체 문서: https://here.now/docs
