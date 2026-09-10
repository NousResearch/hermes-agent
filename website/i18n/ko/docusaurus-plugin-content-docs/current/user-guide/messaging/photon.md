---
sidebar_position: 18
---

# Photon iMessage

[Photon][photon]을 통해 Hermes를 **iMessage**에 연결하세요. Photon은 Apple 회선 할당과 악용 방지 계층을 처리하는 관리형 서비스이므로, 직접 Mac 릴레이를 실행할 필요가 없습니다.

무료 요금제는 Photon의 공유 iMessage 회선 풀을 사용합니다. 수신자에 따라 서로 다른 발신 번호가 표시될 수 있지만 각 대화는 안정적으로 유지됩니다. 유료 Business 요금제에서는 모든 사용자에게 동일한 전용 번호가 제공됩니다. 이 플러그인은 두 요금제를 모두 지원하며, 무료 요금제로 시작하는 것을 권장합니다.

:::info 무료로 시작
Photon의 공유 회선 풀은 무료입니다. Hermes에서 첫 iMessage를 보내기 위해 구독할 필요가 없습니다 — 계정에 연결할 수 있는 전화번호만 있으면 됩니다.
:::

## 아키텍처

Photon은 Discord나 Slack과 같은 **지속적 연결** 채널입니다 — **웹훅, 공개 URL, 관리할 signing secret이 없습니다.**

`spectrum-ts` SDK는 양방향 통신을 위해 Photon으로 장기 실행 **gRPC 스트림**을 유지합니다. SDK는 TypeScript 전용이므로 Hermes는 작은 관리형 **Node 사이드카**에서 SDK를 실행하고 루프백을 통해 통신합니다:

- **인바운드** — 사이드카가 SDK의 `app.messages` gRPC 스트림을 소비하고 각 메시지를 루프백 `GET /inbound`(NDJSON)를 통해 Python 어댑터로 전달합니다. 어댑터는 중복을 제거하고 에이전트로 디스패치하며 스트림이 끊기면 자동으로 재연결합니다.
- **아웃바운드** — 응답을 사이드카에 루프백 POST로 보내면 사이드카가 SDK에서 `space.send(...)`를 호출합니다.

Python 플러그인이 사이드카를 자동으로 시작하고, 관리하며, 종료합니다.

## 사전 요구 사항

- Photon 계정 — [app.photon.codes][app]에서 가입하세요
- PATH에 **Node.js 18.17 이상** (`node --version`)
- iMessage를 받을 수 있는 전화번호(계정 연결에 사용)

이것으로 충분합니다 — 설정할 공개 URL이나 터널은 없습니다.

## 최초 설정

통합 gateway 마법사를 실행하고 **Photon iMessage**를 선택하세요:

```bash
hermes gateway setup
```

…또는 Photon 설정을 직접 실행하세요(마법사도 동일한 흐름을 호출합니다):

```bash
# Device-code login + project + user + sidecar deps, all in one
hermes photon setup --phone +15551234567
```

설정 순서는 다음과 같습니다:

1. **디바이스 로그인**(`client_id=photon-cli`) — 승인을 위해 `https://app.photon.codes/`를 열고 bearer token을 저장합니다.
2. 계정에서 `Hermes Agent` 프로젝트를 찾거나 생성합니다.
3. **Spectrum을 활성화**하고 프로젝트의 Spectrum id를 읽은 다음 프로젝트 secret을 교체합니다.
4. **전화번호를 Spectrum 사용자로 등록**합니다 — 해당 번호의 사용자가 이미 있으면 건너뛰므로 다시 실행해도 안전합니다.
5. **할당된 iMessage 회선을 출력**합니다 — 에이전트에게 문자를 보낼 때 사용하는 번호입니다.
6. 플러그인의 사이드카 디렉터리에서 `npm install`을 실행합니다. 읽기 전용/불변 설치 트리(호스팅된 Docker 이미지, Podman, Nix)에서는 사이드카가 자동으로 `~/.hermes/photon/sidecar` 아래의 쓰기 가능한 미러로 대체됩니다. 명시적인 위치를 고정하려면 `PHOTON_SIDECAR_DIR`을 설정하세요.

런타임 자격 증명은 `~/.hermes/.env`에 기록됩니다(`PHOTON_PROJECT_ID` = Spectrum project id, `PHOTON_PROJECT_SECRET`). 다른 모든 채널이 토큰을 저장하는 곳과 같습니다. 관리 메타데이터(device token, dashboard project id)는 `credential_pool.photon` / `credential_pool.photon_project` 아래 `~/.hermes/auth.json`에 저장됩니다.

## 사용자 인증

Photon은 다른 모든 Hermes 채널과 동일한 인증 모델을 사용합니다. 다음 방법 중 하나를 선택하세요:

**DM 페어링(기본값).** 알 수 없는 번호가 Photon 회선으로 메시지를 보내면 Hermes가 페어링 코드를 답장합니다. 다음 명령으로 승인하세요:

```bash
hermes pairing approve photon <CODE>
```

`hermes pairing list`를 사용하면 대기 중인 코드와 승인된 사용자를 볼 수 있습니다.

**특정 번호를 사전 인증**(`~/.hermes/.env`에서):

```bash
PHOTON_ALLOWED_USERS=+15551234567,+15559876543
```

**공개 액세스**(개발 전용, `~/.hermes/.env`에서):

```bash
PHOTON_ALLOW_ALL_USERS=true
```

`PHOTON_ALLOWED_USERS`가 설정되면 알 수 없는 발신자는 페어링 코드 안내 없이 조용히 무시됩니다(allowlist는 의도적으로 액세스를 제한했음을 나타냅니다).

### 그룹 채팅에서 멘션 요구

기본적으로 Hermes는 승인된 모든 DM과 그룹 메시지에 응답합니다. 그룹 채팅을 명시적으로 선택해야 하도록 하려면 멘션 게이팅을 활성화하세요(DM은 계속 항상 작동합니다):

```yaml
gateway:
  platforms:
    photon:
      enabled: true
      require_mention: true
```

`require_mention: true`이면 웨이크 워드 패턴과 일치하지 않는 그룹 채팅 메시지는 무시됩니다. 기본값은 `Hermes` 및 `@Hermes agent` 변형과 일치합니다. 사용자 지정 에이전트 이름은 정규식 패턴을 설정하세요:

```yaml
gateway:
  platforms:
    photon:
      require_mention: true
      mention_patterns:
        - '(?<![\w@])@?amos\b[,:\-]?'
```

두 키 모두 환경 변수(`PHOTON_REQUIRE_MENTION`, `PHOTON_MENTION_PATTERNS`)도 사용할 수 있습니다. 이는 BlueBubbles iMessage 채널이 사용하는 것과 동일한 멘션 게이팅 모델입니다.

## gateway 시작

```bash
hermes gateway start
```

다음과 비슷한 내용이 표시됩니다:

```
[photon] connected — sidecar on 127.0.0.1:8789, streaming inbound over gRPC
```

할당된 번호로 iMessage를 보내면 Hermes가 답장합니다.

## 상태 및 문제 해결

```bash
hermes photon status
```

저장된 자격 증명, 사이드카 상태, 등록된 번호, Hermes가 사용하는 할당된 iMessage 회선을 출력합니다. Photon token과 dashboard project를 사용할 수 있으면 `status`가 새 회선을 프로비저닝하지 않고 dashboard에서 누락된 번호 행을 갱신합니다.

```
Photon iMessage status
──────────────────────
  device token        : ✓ stored
  dashboard project   : 3c90c3cc-0d44-4b50-...
  spectrum project id : sp-...
  project secret      : ✓ stored
  my number           : +15551234567
  assigned number     : +16282679185
  node binary         : /usr/bin/node
  sidecar deps        : ✓ installed
```

일반적인 문제:

- **`sidecar deps : ✗ run hermes photon install-sidecar`** — Node는 설치되어 있지만 `spectrum-ts`가 없습니다. 안내된 명령을 실행하세요.
- **`device token : ✗ missing`** — 로그인하려면 `hermes photon setup`을 실행하세요.
- **`No iMessage line assigned yet`** — Spectrum은 활성화되었지만 회선이 프로비저닝되지 않았습니다. `hermes photon setup`을 다시 실행하거나 [dashboard][app]를 확인하세요.
- **사이드카가 시작되지 않음** — `node --version`이 18.17 이상인지, `hermes photon install-sidecar`가 오류 없이 완료되었는지 확인하세요.

## 현재 제한 사항

- **인바운드 첨부 파일은 메타데이터만 지원합니다.** 인바운드 이벤트에는 파일 이름과 MIME 유형이 포함됩니다. 에이전트는 표시자를 보지만 아직 바이트를 읽을 수는 없습니다. SDK가 `content.read()`를 통해 첨부 파일 바이트를 제공하므로 이는 사이드카 후속 작업입니다.
- **아웃바운드 첨부 파일은 지원됩니다.** Hermes는 사이드카의 `/send-attachment` 엔드포인트를 통해 spectrum-ts의 `attachment()` / `voice()` 콘텐츠 빌더로 이미지, 음성 메모, 동영상, 문서를 보냅니다. 캡션은 미디어 다음에 별도의 iMessage 말풍선으로 도착합니다.
- **네이티브 투표가 지원됩니다.** Hermes는 사이드카의 `/send-poll` 엔드포인트를 통해 spectrum-ts의 `poll()` 빌더로 투표 콘텐츠를 보냅니다.
- **메시지 효과가 지원됩니다.** Hermes는 사이드카의 `/send-effect` 엔드포인트를 통해 spectrum-ts의 iMessage `effect()` 빌더로 네이티브 iMessage 말풍선/화면 효과가 적용된 텍스트를 보냅니다.
- **Photon 무료 할당량:** 서버당 하루 5,000개 메시지, 공유 회선당 하루 50회의 새 대화 시작. 증설 가능 — `help@photon.codes`로 이메일을 보내세요.
- **Cron 및 독립 실행형 전송에는 gateway 실행이 필요합니다.** 프로세스 외부 전송자(cron 작업, `hermes send`, dashboard)는 gateway가 생성한 사이드카를 재사용합니다 — 사이드카가 상태 확인을 통과하면 기록되고 중지되면 삭제되는 `<hermes-home>/runtime/photon-sidecar.json`에서 포트/token을 읽습니다. 독립 실행형 전송에서 gateway가 중지된 것으로 나타나면 먼저 gateway를 시작(또는 재시작)하세요.
- **공유/무료 요금제 회선은 새 대상과의 대화를 시작할 수 없습니다.** Photon 측 정책상 공유 회선은 해당 번호가 먼저 회선으로 문자를 보낸 후에만 그 번호로 메시지를 보낼 수 있습니다. Hermes가 올바르게 설정되어 있어도 완전히 새로운 수신자에게 보내는 cron/독립 실행형 전송은 Photon에서 거부됩니다 — 수신자가 먼저 회선으로 메시지를 보내게 하거나 전용 회선으로 전환하세요.

## 환경 변수

| 변수                  | 기본값            | 참고                                      |
|---------------------------|--------------------|--------------------------------------------|
| `PHOTON_PROJECT_ID`       | `.env`에서 가져옴        | Spectrum project id(SDK의 `projectId`); setup에서 설정 |
| `PHOTON_PROJECT_SECRET`   | `.env`에서 가져옴        | Project secret; setup에서 설정               |
| `PHOTON_SIDECAR_PORT`     | `8789`             | 사이드카 제어 + 인바운드 채널용 루프백 포트 |
| `PHOTON_SIDECAR_AUTOSTART`| `true`             | 어댑터가 사이드카를 생성할지 여부     |
| `PHOTON_NODE_BIN`         | `which node`       | Node 바이너리 경로 재정의              |
| `PHOTON_HOME_CHANNEL`     | (설정되지 않음)            | cron / 알림의 기본 space id  |
| `PHOTON_HOME_CHANNEL_NAME`| (설정되지 않음)            | 홈 채널의 사람이 읽을 수 있는 레이블        |
| `PHOTON_ALLOWED_USERS`    | (설정되지 않음)            | 쉼표로 구분된 E.164 allowlist            |
| `PHOTON_ALLOW_ALL_USERS`  | `false`             | 개발 전용 — 모든 발신자 허용               |
| `PHOTON_REQUIRE_MENTION`  | `false`             | 그룹에서 응답하기 전에 웨이크 워드 요구 |
| `PHOTON_MENTION_PATTERNS` | Hermes wake words  | 그룹 멘션용 JSON 목록 / 쉼표 / 줄바꿈 정규식 패턴 |
| `PHOTON_DASHBOARD_HOST`   | `app.photon.codes` | dashboard / device-login 호스트 재정의 |
| `PHOTON_SPECTRUM_HOST`    | `spectrum.photon.codes` | Spectrum API 호스트 재정의 |

[photon]: https://photon.codes/
[app]: https://app.photon.codes/
