---
sidebar_position: 11
title: "Feishu / Lark"
description: "Hermes Agent를 Feishu 또는 Lark 봇으로 설정"
---

# Feishu / Lark 설정

Hermes Agent는 완전한 기능을 갖춘 봇으로 Feishu와 Lark를 통합합니다. 연결하면 다이렉트 메시지나 그룹 채팅에서 에이전트와 대화하고, 홈 채팅에서 cron 작업 결과를 받고, 일반 게이트웨이 흐름을 통해 텍스트, 이미지, 오디오, 파일 첨부를 보낼 수 있습니다.

통합은 두 가지 연결 모드를 모두 지원합니다.

- `websocket` — 권장 방식입니다. Hermes가 아웃바운드 연결을 열므로 공개 webhook 엔드포인트가 필요하지 않습니다.
- `webhook` — Feishu/Lark가 HTTP를 통해 게이트웨이로 이벤트를 푸시하도록 하려는 경우에 유용합니다.

## Hermes의 동작 방식

| 컨텍스트 | 동작 |
|---------|----------|
| 다이렉트 메시지 | Hermes는 모든 메시지에 응답합니다. |
| 그룹 채팅 | Hermes는 채팅에서 봇이 @멘션된 경우에만 응답합니다. |
| 공유 그룹 채팅 | 기본적으로 공유 채팅 내 세션 기록은 사용자별로 격리됩니다. |

이 공유 채팅 동작은 `config.yaml`로 제어합니다.

```yaml
group_sessions_per_user: true
```

하나의 채팅마다 대화를 하나만 공유하려는 경우에만 `false`로 설정하세요.

## 1단계: Feishu / Lark 앱 만들기

### 권장: 스캔하여 만들기(명령 하나)

```bash
hermes gateway setup
```

**Feishu / Lark**를 선택하고 Feishu 또는 Lark 모바일 앱으로 QR 코드를 스캔하세요. Hermes가 올바른 권한으로 봇 애플리케이션을 자동으로 만들고 인증 정보를 저장합니다.

### 대안: 수동 설정

스캔하여 만들기를 사용할 수 없으면 마법사가 수동 입력으로 전환됩니다.

1. Feishu 또는 Lark 개발자 콘솔을 엽니다.
   - Feishu: [https://open.feishu.cn/](https://open.feishu.cn/)
   - Lark: [https://open.larksuite.com/](https://open.larksuite.com/)
2. 새 앱을 만듭니다.
3. **Credentials & Basic Info**에서 **App ID**와 **App Secret**을 복사합니다.
4. 앱의 **Bot** 기능을 활성화합니다.
5. `hermes gateway setup`을 실행하고 **Feishu / Lark**를 선택한 다음, 메시지가 표시되면 인증 정보를 입력합니다.

:::warning
App Secret을 비공개로 유지하세요. App Secret을 가진 사람은 누구나 앱을 사칭할 수 있습니다.
:::

### 권한 설정

Feishu 개발자 콘솔에서 **Permission Management**로 이동하여 다음 범위를 추가합니다. 권한 페이지에서 일괄 가져오기를 사용할 수 있습니다.

**필수 권한:**

| 범위 | 용도 |
|-------|---------|
| `im:message` | 메시지 수신 및 읽기 |
| `im:message:send_as_bot` | 봇으로 메시지 보내기 |
| `im:resource` | 사용자가 보낸 이미지, 파일, 오디오에 액세스 |
| `im:chat` | 채팅/그룹 메타데이터에 액세스 |
| `im:chat:readonly` | 채팅 목록 및 멤버십 읽기 |

**권장 권한(전체 기능 사용):**

| 범위 | 용도 |
|-------|---------|
| `im:message.reactions:readonly` | 이모지 반응 이벤트 수신 |
| `admin:app.info:readonly` | @멘션 게이트를 위한 봇 ID 자동 감지 |
| `contact:user.id:readonly` | 허용 목록 일치에 사용할 사용자 ID 확인 |

### 이벤트 설정

**Events and Callbacks**에서 다음을 수행합니다.

1. 연결 모드를 **Long Connection (WebSocket)**(권장)으로 설정하거나 webhook URL을 구성합니다.
2. **Event Configuration** 섹션에서 다음을 구독합니다.
   - `im.message.receive_v1` — 메시지 수신에 필요

### 앱 게시

권한과 이벤트를 설정한 후 **Version Management**로 이동하여 앱의 새 버전을 게시합니다. 버전이 게시되고 승인될 때까지 권한이 적용되지 않습니다(엔터프라이즈 앱의 경우 관리자 승인이 필요할 수 있음).

## 2단계: 연결 모드 선택

### 권장: WebSocket 모드

Hermes를 노트북, 워크스테이션 또는 비공개 서버에서 실행하는 경우 WebSocket 모드를 사용하세요. 공개 URL이 필요하지 않습니다. 공식 Lark SDK가 자동 재연결 기능이 있는 지속적인 아웃바운드 WebSocket 연결을 열고 유지합니다.

```bash
FEISHU_CONNECTION_MODE=websocket
```

**요구 사항:** `websockets` Python 패키지가 설치되어 있어야 합니다. SDK가 연결 수명 주기, heartbeat, 자동 재연결을 내부적으로 처리합니다.

**작동 방식:** 어댑터는 백그라운드 executor 스레드에서 Lark SDK의 WebSocket 클라이언트를 실행합니다. 인바운드 이벤트(메시지, 반응, 카드 작업)는 메인 asyncio 루프로 전달됩니다. 연결이 끊기면 SDK가 자동으로 재연결을 시도합니다.

### 선택 사항: Webhook 모드

접근 가능한 HTTP 엔드포인트 뒤에서 이미 Hermes를 실행하고 있을 때만 webhook 모드를 사용하세요.

```bash
FEISHU_CONNECTION_MODE=webhook
```

Webhook 모드에서 Hermes는 HTTP 서버(`aiohttp` 사용)를 시작하고 다음 경로에서 Feishu 엔드포인트를 제공합니다.

```text
/feishu/webhook
```

**요구 사항:** `aiohttp` Python 패키지가 설치되어 있어야 합니다.

Webhook 서버의 바인드 주소와 경로를 사용자 지정할 수 있습니다.

```bash
FEISHU_WEBHOOK_HOST=127.0.0.1   # default: 127.0.0.1
FEISHU_WEBHOOK_PORT=8765         # default: 8765
FEISHU_WEBHOOK_PATH=/feishu/webhook  # default: /feishu/webhook
```

Feishu가 URL 확인 challenge(`type: url_verification`)를 보내면 webhook이 자동으로 응답하므로 Feishu 개발자 콘솔에서 구독 설정을 완료할 수 있습니다. challenge 응답은 설정된 경우 `FEISHU_VERIFICATION_TOKEN`에 의해 보호됩니다. 토큰이 없거나 일치하지 않는 challenge 요청은 거부되므로, 인증되지 않은 원격 사용자가 공격자가 제어하는 challenge 데이터를 반향하여 엔드포인트 제어를 증명할 수 없습니다.

## 3단계: Hermes 설정

### 옵션 A: 대화형 설정

```bash
hermes gateway setup
```

**Feishu / Lark**를 선택하고 메시지에 따라 입력합니다.

### 옵션 B: 수동 설정

다음을 `~/.hermes/.env`에 추가합니다.

```bash
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=secret_xxx
FEISHU_DOMAIN=feishu
FEISHU_CONNECTION_MODE=websocket

# Optional but strongly recommended
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
FEISHU_HOME_CHANNEL=oc_xxx
```

`FEISHU_DOMAIN`은 다음 값을 허용합니다.

- `feishu` — Feishu 중국
- `lark` — Lark 국제

## 4단계: 게이트웨이 시작

```bash
hermes gateway
```

그런 다음 Feishu/Lark에서 봇에게 메시지를 보내 연결이 활성 상태인지 확인합니다.

## 홈 채팅

Feishu/Lark 채팅에서 `/set-home`을 사용하여 cron 작업 결과와 플랫폼 간 알림을 받을 홈 채널로 지정합니다.

미리 설정할 수도 있습니다.

```bash
FEISHU_HOME_CHANNEL=oc_xxx
```

## 보안

### 사용자 허용 목록

프로덕션 환경에서는 Feishu Open ID 허용 목록을 설정합니다.

```bash
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
```

허용 목록을 비워 두면 봇에 접근할 수 있는 누구나 봇을 사용할 수 있습니다. 그룹 채팅에서는 메시지를 처리하기 전에 발신자의 open_id를 기준으로 허용 목록을 확인합니다.

### Webhook 암호화 키

Webhook 모드에서 실행할 때 암호화 키를 설정하여 인바운드 webhook 페이로드의 서명 검증을 활성화합니다.

```bash
FEISHU_ENCRYPT_KEY=your-encrypt-key
```

이 키는 Feishu 앱 설정의 **Event Subscriptions** 섹션에서 찾을 수 있습니다. 설정하면 어댑터는 다음 서명 알고리즘을 사용하여 모든 webhook 요청을 검증합니다.

```
SHA256(timestamp + nonce + encrypt_key + body)
```

계산된 해시는 timing-safe 비교를 사용하여 `x-lark-signature` 헤더와 비교합니다. 유효하지 않거나 누락된 서명이 있는 요청은 HTTP 401로 거부됩니다.

:::tip
WebSocket 모드에서는 SDK 자체가 서명 검증을 처리하므로 `FEISHU_ENCRYPT_KEY`는 선택 사항입니다. Webhook 모드에서는 프로덕션 환경에 강력히 권장됩니다.
:::

### 검증 토큰

Webhook 페이로드 내부의 `token` 필드를 확인하는 추가 인증 계층입니다.

```bash
FEISHU_VERIFICATION_TOKEN=your-verification-token
```

이 토큰도 Feishu 앱의 **Event Subscriptions** 섹션에서 찾을 수 있습니다. 설정하면 모든 인바운드 webhook 페이로드의 `header` 객체에 일치하는 `token`이 있어야 합니다. 일치하지 않는 토큰은 HTTP 401로 거부됩니다.

심층 방어를 위해 `FEISHU_ENCRYPT_KEY`와 `FEISHU_VERIFICATION_TOKEN`을 함께 사용할 수 있습니다.

## 그룹 메시지 정책

`FEISHU_GROUP_POLICY` 환경 변수는 Hermes가 그룹 채팅에서 응답할지와 응답 방식을 제어합니다.

```bash
FEISHU_GROUP_POLICY=allowlist   # default
```

| 값 | 동작 |
|-------|----------|
| `open` | Hermes는 모든 그룹의 모든 사용자로부터 온 @멘션에 응답합니다. |
| `allowlist` | Hermes는 `FEISHU_ALLOWED_USERS`에 등록된 사용자의 @멘션에만 응답합니다. |
| `disabled` | Hermes는 모든 그룹 메시지를 완전히 무시합니다. |

모든 모드에서 메시지를 처리하기 전에 그룹에서 봇을 명시적으로 @멘션(또는 @all)해야 합니다. 다이렉트 메시지는 항상 이 게이트를 우회합니다.

`FEISHU_REQUIRE_MENTION=false`로 설정하면 @멘션 없이도 Hermes가 모든 그룹 트래픽을 읽도록 할 수 있습니다.

```bash
FEISHU_REQUIRE_MENTION=false
```

채팅별 제어를 위해 `group_rules` 항목에 `require_mention`을 설정하세요. 아래 [그룹별 액세스 제어](#per-group-access-control)를 참조하세요.

### 봇 ID

Hermes는 시작할 때 봇의 `open_id`와 표시 이름을 자동으로 감지합니다. 자동 감지로 Feishu API에 접근할 수 없거나 앱에서 테넌트 범위 사용자 ID를 사용하는 경우에만 직접 설정하면 됩니다.

```bash
FEISHU_BOT_OPEN_ID=ou_xxx     # only when auto-detection fails
FEISHU_BOT_USER_ID=xxx        # required if your app uses sender_id_type=user_id
FEISHU_BOT_NAME=MyBot         # only when auto-detection fails
```

## 봇 간 메시징

기본적으로 Hermes는 다른 봇이 보낸 메시지를 무시합니다. Hermes가 A2A 오케스트레이션에 참여하거나 같은 그룹의 다른 봇으로부터 알림을 받도록 하려면 봇 간 메시징을 활성화하세요.

```bash
FEISHU_ALLOW_BOTS=mentions   # default: none
```

| 값 | 동작 |
|-------|----------|
| `none` | 다른 봇의 모든 메시지를 무시합니다(기본값). |
| `mentions` | 상대 봇이 Hermes를 @멘션한 경우에만 수락합니다. |
| `all` | 상대 봇의 모든 메시지를 수락합니다. |

`config.yaml`의 `feishu.allow_bots`로도 설정할 수 있습니다(둘 다 설정된 경우 env가 우선합니다).

상대 봇은 `FEISHU_ALLOWED_USERS`에 추가할 필요가 없습니다. 해당 허용 목록은 사람 발신자에게만 적용됩니다.

상대 봇 이름을 표시하려면 `application:bot.basic_info:read` 범위를 부여합니다. 이 범위가 없어도 상대 봇은 올바르게 라우팅되지만 `open_id`로 표시됩니다.

## 대화형 카드 작업

사용자가 봇이 보낸 버튼을 클릭하거나 대화형 카드와 상호작용하면 어댑터는 이를 합성된 `/card` 명령 이벤트로 라우팅합니다.

- 버튼 클릭은 다음으로 변환됩니다: `/card button {"key": "value", ...}`
- 카드 정의의 작업 `value` 페이로드가 JSON으로 포함됩니다.
- 카드 작업은 중복 처리를 방지하기 위해 15분의 기간으로 중복 제거됩니다.

게이트웨이 기반 업데이트 프롬프트는 일반 텍스트 답변으로 대체하지 않고 네이티브 Feishu `Yes` / `No` 카드를 사용합니다. `hermes update --gateway`에 확인이 필요하면 어댑터는 선택한 답변을 Hermes의 `.update_response` 파일에 기록하고 카드를 해결된 상태로 인라인 교체합니다.

카드 작업 이벤트는 `MessageType.COMMAND`로 디스패치되므로 일반 명령 처리 파이프라인을 거칩니다.

**명령 승인**도 같은 방식으로 작동합니다. 에이전트가 위험한 명령을 실행해야 하면 Allow Once / Session / Always / Deny 버튼이 있는 대화형 카드를 보냅니다. 사용자가 버튼을 클릭하면 카드 작업 콜백이 승인 결정을 에이전트로 전달합니다.

### 필수 Feishu 앱 설정

대화형 카드를 사용하려면 Feishu Developer Console에서 세 가지 설정 단계를 수행해야 합니다. 하나라도 누락되면 사용자가 카드 버튼을 클릭할 때 **200340** 오류가 발생합니다.

1. **카드 작업 이벤트 구독:**
   **Event Subscriptions**에서 구독 이벤트에 `card.action.trigger`를 추가합니다.
2. **Interactive Card 기능 활성화:**
   **App Features > Bot**에서 **Interactive Card** 토글이 활성화되어 있는지 확인합니다. 이를 통해 Feishu는 앱이 카드 작업 콜백을 수신할 수 있음을 알 수 있습니다.
3. **카드 요청 URL 설정(webhook 모드만 해당):**
   **App Features > Bot > Message Card Request URL**에서 URL을 이벤트 webhook과 동일한 엔드포인트로 설정합니다(예: `https://your-server:8765/feishu/webhook`). WebSocket 모드에서는 SDK가 자동으로 처리합니다.

:::warning
세 단계를 모두 수행하지 않으면 Feishu가 대화형 카드를 성공적으로 *보내기는* 하지만(전송에는 `im:message:send` 권한만 필요), 버튼을 클릭할 때마다 200340 오류가 반환됩니다. 카드 자체는 작동하는 것처럼 보이며, 오류는 사용자가 상호작용할 때만 나타납니다.
:::

## 문서 댓글 지능형 답변

채팅 외에도 어댑터는 **Feishu/Lark 문서**에 남겨진 `@` 멘션에도 답변할 수 있습니다. 사용자가 문서에 댓글(로컬 텍스트 선택 또는 문서 전체 댓글)을 작성하고 봇을 @멘션하면 Hermes가 문서와 주변 댓글 스레드를 읽고 스레드 안에 LLM 답변을 인라인으로 게시합니다.

`drive.notice.comment_add_v1` 이벤트를 기반으로 하는 핸들러는 다음을 수행합니다.

- 문서 내용과 댓글 타임라인을 병렬로 가져옵니다(문서 전체 스레드는 메시지 20개, 로컬 선택 스레드는 12개).
- 단일 댓글 세션으로 범위가 지정된 `feishu_doc` + `feishu_drive` 도구 세트로 에이전트를 실행합니다.
- 답변을 4000자로 나누어 스레드 답글로 게시합니다.
- 문서별 세션을 1시간 동안, 최대 50개 메시지로 캐시하므로 같은 문서의 후속 댓글이 컨텍스트를 유지합니다.

### 3단계 액세스 제어

문서 댓글 답변은 **명시적 허가만 허용**되며, 암시적인 모두 허용 모드는 없습니다. 권한은 다음 순서로 확인됩니다(필드별 최초 일치 규칙 적용).

1. **정확한 문서** — 특정 문서 토큰으로 범위가 지정된 규칙
2. **와일드카드** — 문서 패턴과 일치하는 규칙
3. **최상위** — 워크스페이스의 기본 규칙

각 규칙에는 두 가지 정책을 사용할 수 있습니다.

- **`allowlist`** — 사용자/테넌트의 정적 목록
- **`pairing`** — 정적 목록 ∪ 런타임 승인 저장소. 운영자가 실시간으로 액세스 권한을 부여할 수 있는 롤아웃에 유용합니다.

규칙은 `~/.hermes/feishu_comment_rules.json`에 저장되고(pairing 허가는 `~/.hermes/feishu_comment_pairing.json`), mtime 캐시 기반 핫 리로드가 적용됩니다. 편집 내용은 게이트웨이를 재시작하지 않아도 다음 댓글 이벤트부터 적용됩니다.

CLI:

```bash
# Inspect current rules and pairing state
python -m gateway.platforms.feishu_comment_rules status

# Simulate an access check for a specific doc + user
python -m gateway.platforms.feishu_comment_rules check <fileType:fileToken> <user_open_id>

# Manage pairing grants at runtime
python -m gateway.platforms.feishu_comment_rules pairing list
python -m gateway.platforms.feishu_comment_rules pairing add <user_open_id>
python -m gateway.platforms.feishu_comment_rules pairing remove <user_open_id>
```

### 필수 Feishu 앱 설정

이미 부여한 채팅/카드 권한에 더해 drive 댓글 이벤트를 추가합니다.

- **Event Subscriptions**에서 `drive.notice.comment_add_v1`을 구독합니다.
- 핸들러가 문서 내용을 읽을 수 있도록 `docs:doc:readonly` 및 `drive:drive:readonly` 범위를 부여합니다.

## 회의 초대 이벤트

사람 참가자를 초대하는 것과 같은 방식으로 Hermes Feishu/Lark 봇을 화상 회의에 초대할 수 있습니다. 봇이 회의 초대 이벤트를 받으면 Hermes가 자동으로 에이전트 턴을 시작하여 회의에 참가하려고 시도할 수 있습니다.

`vc.bot.meeting_invited_v1` 이벤트를 기반으로 하는 흐름은 다음과 같습니다.

- 사용자가 Feishu/Lark 화상 회의에 봇을 초대합니다.
- Feishu/Lark가 Hermes에 회의 초대 이벤트를 보냅니다.
- Hermes가 초대한 사람, 회의 주제, 회의 번호를 추출합니다.
- 초대한 사람이 일반 게이트웨이 허용 목록 또는 pairing 정책으로 권한을 부여받은 경우 에이전트가 회의 번호를 받고 자동으로 참가를 시도합니다.
- 초대 형식이 잘못되었거나 에이전트가 참가할 수 없으면 Hermes가 이벤트를 무시하거나 초대한 사람에게 간단한 설명을 답합니다.

초대한 사람과 `meeting_no`를 모두 포함하지 않는 잘못된 초대는 무시됩니다.

### 필수 Feishu 앱 설정

이미 부여한 채팅/카드 권한에 더해 화상 회의 초대 이벤트를 추가합니다.

- **Event Subscriptions**에서 `vc.bot.meeting_invited_v1`을 구독합니다.
- Feishu/Lark 개발자 콘솔에서 해당 이벤트에 대해 요청하는 Video Conferencing 권한 범위를 활성화합니다.
- Hermes가 초대한 사람에게 답할 수 있도록 `im:message` 및 `im:message:send_as_bot`을 활성화 상태로 유지합니다.
- 게이트웨이 사용자 허용 목록 또는 pairing 정책이 초대한 사람을 허가하는지 확인합니다. 회의 초대는 일반 게이트웨이 액세스 검사를 우회하지 않습니다.

## 미디어 지원

### 인바운드(수신)

어댑터는 사용자로부터 다음 미디어 유형을 받아 캐시합니다.

| 유형 | 확장자 | 처리 방식 |
|------|-----------|-------------------|
| **이미지** | .jpg, .jpeg, .png, .gif, .webp, .bmp | Feishu API를 통해 다운로드하고 로컬에 캐시 |
| **오디오** | .ogg, .mp3, .wav, .m4a, .aac, .flac, .opus, .webm | 다운로드하고 캐시하며, 작은 텍스트 파일은 자동 추출 |
| **비디오** | .mp4, .mov, .avi, .mkv, .webm, .m4v, .3gp | 다운로드하고 문서로 캐시 |
| **파일** | .pdf, .doc, .docx, .xls, .xlsx, .ppt, .pptx 및 기타 | 다운로드하고 문서로 캐시 |

인라인 이미지와 파일 첨부를 포함한 서식 있는 텍스트(post) 메시지의 미디어도 추출하고 캐시합니다.

작은 텍스트 기반 문서(.txt, .md)는 파일 내용이 메시지 텍스트에 자동으로 삽입되므로 에이전트가 도구 없이 직접 읽을 수 있습니다.

### 아웃바운드(전송)

| 메서드 | 전송 내용 |
|--------|--------------|
| `send` | 텍스트 또는 서식 있는 post 메시지(markdown 내용에 따라 자동 감지) |
| `send_image` / `send_image_file` | 이미지를 Feishu에 업로드한 후 네이티브 이미지 말풍선으로 전송(선택적 캡션 포함) |
| `send_document` | 파일을 Feishu API에 업로드한 후 파일 첨부로 전송 |
| `send_voice` | 오디오 파일을 Feishu 파일 첨부로 업로드 |
| `send_video` | 비디오를 업로드하고 네이티브 미디어 메시지로 전송 |
| `send_animation` | GIF는 파일 첨부로 전환(Feishu에는 네이티브 GIF 말풍선이 없음) |

파일 업로드 라우팅은 확장자에 따라 자동으로 이루어집니다.

- `.ogg`, `.opus` → `opus` 오디오로 업로드
- `.mp4`, `.mov`, `.avi`, `.m4v` → `mp4` 미디어로 업로드
- `.pdf`, `.doc(x)`, `.xls(x)`, `.ppt(x)` → 문서 유형으로 업로드
- 그 외 모든 파일 → 일반 stream 파일로 업로드

## Markdown 렌더링 및 Post 대체

아웃바운드 텍스트에 markdown 서식(제목, 굵게, 목록, 코드 블록, 링크 등)이 포함되면 어댑터는 일반 텍스트가 아니라 `md` 태그가 삽입된 Feishu **post** 메시지로 자동 전송합니다. 이를 통해 Feishu 클라이언트에서 서식 있는 렌더링이 가능합니다.

Feishu API가 post 페이로드를 거부하는 경우(예: 지원되지 않는 markdown 구성으로 인해) 어댑터는 자동으로 markdown을 제거한 일반 텍스트 전송으로 대체합니다. 이 2단계 대체를 통해 메시지가 항상 전달됩니다.

일반 텍스트 메시지(markdown이 감지되지 않음)는 단순한 `text` 메시지 유형으로 전송됩니다.

## 처리 상태 반응

에이전트가 작업하는 동안 봇은 사용자의 메시지에 `Typing` 반응을 표시합니다. 답변이 도착하면 반응이 지워지고, 처리에 실패하면 `CrossMark`로 대체됩니다.

끄려면 `FEISHU_REACTIONS=false`로 설정합니다.

## 버스트 보호 및 일괄 처리

어댑터에는 빠르게 몰리는 메시지로 에이전트가 과부하되지 않도록 디바운싱이 포함되어 있습니다.

### 텍스트 일괄 처리

사용자가 짧은 시간에 여러 텍스트 메시지를 보내면 디스패치 전에 하나의 이벤트로 병합됩니다.

| 설정 | 환경 변수 | 기본값 |
|---------|---------|---------|
| 대기 시간 | `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` | 0.6s |
| 일괄 처리당 최대 메시지 수 | `HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES` | 8 |
| 일괄 처리당 최대 문자 수 | `HERMES_FEISHU_TEXT_BATCH_MAX_CHARS` | 4000 |

### 미디어 일괄 처리

짧은 시간에 전송된 여러 미디어 첨부(예: 여러 이미지를 드래그하여 전송)는 하나의 이벤트로 병합됩니다.

| 설정 | 환경 변수 | 기본값 |
|---------|---------|---------|
| 대기 시간 | `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | 0.8s |

### 채팅별 직렬화

대화의 일관성을 유지하기 위해 같은 채팅 내 메시지는 순차적으로 처리됩니다(한 번에 하나씩). 채팅마다 자체 잠금이 있으므로 서로 다른 채팅의 메시지는 동시에 처리됩니다.

## 속도 제한(Webhook 모드)

Webhook 모드에서 어댑터는 악용을 방지하기 위해 IP별 속도 제한을 적용합니다.

- **기간:** 60초 슬라이딩 윈도우
- **제한:** (app_id, path, IP) 조합별 기간당 120개 요청
- **추적 상한:** 추적되는 고유 키 최대 4096개(무제한 메모리 증가 방지)

제한을 초과한 요청은 HTTP 429(Too Many Requests)를 받습니다.

### Webhook 이상 징후 추적

어댑터는 IP 주소별 연속 오류 응답을 추적합니다. 6시간 윈도우 내 같은 IP에서 연속 오류가 25회 발생하면 경고를 기록합니다. 이는 잘못 구성된 클라이언트나 탐색 시도를 감지하는 데 도움이 됩니다.

추가 webhook 보호 기능:

- **본문 크기 제한:** 최대 1MB
- **본문 읽기 시간 초과:** 30초
- **Content-Type 적용:** `application/json`만 허용

## WebSocket 조정

`websocket` 모드를 사용할 때 재연결 및 ping 동작을 사용자 지정할 수 있습니다.

```yaml
platforms:
  feishu:
    extra:
      ws_reconnect_interval: 120   # Seconds between reconnect attempts (default: 120)
      ws_ping_interval: 30         # Seconds between WebSocket pings (optional; SDK default if unset)
```

| 설정 | 구성 키 | 기본값 | 설명 |
|---------|---------|---------|-------------|
| 재연결 간격 | `ws_reconnect_interval` | 120s | 재연결 시도 사이에 기다리는 시간 |
| Ping 간격 | `ws_ping_interval` | _(SDK default)_ | WebSocket keepalive ping 빈도 |

## 그룹별 액세스 제어

전역 `FEISHU_GROUP_POLICY` 외에도 `config.yaml`의 `group_rules`를 사용하여 그룹 채팅별 세밀한 규칙을 설정할 수 있습니다.

```yaml
platforms:
  feishu:
    extra:
      default_group_policy: "open"     # Default for groups not in group_rules
      admins:                          # Users who can manage bot settings
        - "ou_admin_open_id"
      group_rules:
        "oc_group_chat_id_1":
          policy: "allowlist"          # open | allowlist | blacklist | admin_only | disabled
          allowlist:
            - "ou_user_open_id_1"
            - "ou_user_open_id_2"
        "oc_group_chat_id_2":
          policy: "admin_only"
        "oc_group_chat_id_3":
          policy: "blacklist"
          blacklist:
            - "ou_blocked_user"
        "oc_free_chat":
          policy: "open"
          require_mention: false       # overrides FEISHU_REQUIRE_MENTION for this chat
```

| 정책 | 설명 |
|--------|-------------|
| `open` | 그룹 내 누구나 봇을 사용할 수 있음 |
| `allowlist` | 그룹의 `allowlist`에 있는 사용자만 봇을 사용할 수 있음 |
| `blacklist` | 그룹의 `blacklist`에 있는 사용자를 제외한 모든 사람이 봇을 사용할 수 있음 |
| `admin_only` | 전역 `admins` 목록의 사용자만 이 그룹에서 봇을 사용할 수 있음 |
| `disabled` | 봇이 이 그룹의 모든 메시지를 무시함 |

특정 채팅의 @멘션 요구 사항을 건너뛰려면 `group_rules` 항목에 `require_mention: false`를 설정합니다. 생략하면 해당 채팅은 전역 `FEISHU_REQUIRE_MENTION` 값을 상속합니다.

`group_rules`에 나열되지 않은 그룹은 `default_group_policy`로 대체됩니다(기본값은 `FEISHU_GROUP_POLICY` 값).

## 중복 제거

인바운드 메시지는 24시간 TTL의 메시지 ID를 사용하여 중복 제거됩니다. 중복 제거 상태는 재시작 후에도 `~/.hermes/feishu_seen_message_ids.json`에 저장됩니다.

| 설정 | 환경 변수 | 기본값 |
|---------|---------|---------|
| 캐시 크기 | `HERMES_FEISHU_DEDUP_CACHE_SIZE` | 2048개 항목 |

## 모든 환경 변수

| 변수 | 필수 | 기본값 | 설명 |
|----------|----------|---------|-------------|
| `FEISHU_APP_ID` | ✅ | — | Feishu/Lark App ID |
| `FEISHU_APP_SECRET` | ✅ | — | Feishu/Lark App Secret |
| `FEISHU_DOMAIN` | — | `feishu` | `feishu`(중국) 또는 `lark`(국제) |
| `FEISHU_CONNECTION_MODE` | — | `websocket` | `websocket` 또는 `webhook` |
| `FEISHU_ALLOWED_USERS` | — | _(empty)_ | 사용자 허용 목록에 사용할 쉼표로 구분된 open_id 목록 |
| `FEISHU_ALLOW_BOTS` | — | `none` | 다른 봇의 메시지 수락: `none`, `mentions` 또는 `all` |
| `FEISHU_REQUIRE_MENTION` | — | `true` | 그룹 메시지에서 봇을 @멘션해야 하는지 여부 |
| `FEISHU_HOME_CHANNEL` | — | — | cron/알림 출력을 위한 채팅 ID |
| `FEISHU_ENCRYPT_KEY` | — | _(empty)_ | webhook 서명 검증용 암호화 키 |
| `FEISHU_VERIFICATION_TOKEN` | — | _(empty)_ | webhook 페이로드 인증용 검증 토큰 |
| `FEISHU_GROUP_POLICY` | — | `allowlist` | 그룹 메시지 정책: `open`, `allowlist`, `disabled` |
| `FEISHU_BOT_OPEN_ID` | — | _(empty)_ | 봇의 open_id(@멘션 감지용) |
| `FEISHU_BOT_USER_ID` | — | _(empty)_ | 봇의 user_id(@멘션 감지용) |
| `FEISHU_BOT_NAME` | — | _(empty)_ | 봇의 표시 이름(@멘션 감지용) |
| `FEISHU_WEBHOOK_HOST` | — | `127.0.0.1` | webhook 서버 바인드 주소 |
| `FEISHU_WEBHOOK_PORT` | — | `8765` | webhook 서버 포트 |
| `FEISHU_WEBHOOK_PATH` | — | `/feishu/webhook` | webhook 엔드포인트 경로 |
| `HERMES_FEISHU_DEDUP_CACHE_SIZE` | — | `2048` | 추적할 중복 제거 메시지 ID의 최대 개수 |
| `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` | — | `0.6` | 텍스트 버스트 디바운싱 대기 시간 |
| `HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES` | — | `8` | 텍스트 일괄 처리에서 병합할 최대 메시지 수 |
| `HERMES_FEISHU_TEXT_BATCH_MAX_CHARS` | — | `4000` | 텍스트 일괄 처리에서 병합할 최대 문자 수 |
| `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | — | `0.8` | 미디어 버스트 디바운싱 대기 시간 |

WebSocket 및 그룹별 ACL 설정은 `platforms.feishu.extra` 아래의 `config.yaml`을 통해 구성합니다(위의 [WebSocket 조정](#websocket-tuning) 및 [그룹별 액세스 제어](#per-group-access-control) 참조).

## 문제 해결

| 문제 | 해결 방법 |
|---------|-----|
| `lark-oapi not installed` | SDK 설치: `pip install lark-oapi` |
| `websockets not installed; websocket mode unavailable` | websockets 설치: `pip install websockets` |
| `aiohttp not installed; webhook mode unavailable` | aiohttp 설치: `pip install aiohttp` |
| `FEISHU_APP_ID or FEISHU_APP_SECRET not set` | 두 환경 변수를 설정하거나 `hermes gateway setup`으로 구성 |
| `Another local Hermes gateway is already using this Feishu app_id` | 하나의 Hermes 인스턴스만 같은 app_id를 사용할 수 있습니다. 먼저 다른 게이트웨이를 중지하세요. |
| 그룹에서 봇이 응답하지 않음 | 봇이 @멘션되었는지 확인하고, `FEISHU_GROUP_POLICY`를 확인하며, 정책이 `allowlist`인 경우 발신자가 `FEISHU_ALLOWED_USERS`에 있는지 확인 |
| `Webhook rejected: invalid verification token` | `FEISHU_VERIFICATION_TOKEN`이 Feishu 앱의 Event Subscriptions 설정에 있는 토큰과 일치하는지 확인 |
| `Webhook rejected: invalid signature` | `FEISHU_ENCRYPT_KEY`가 Feishu 앱 설정의 암호화 키와 일치하는지 확인 |
| Post 메시지가 일반 텍스트로 표시됨 | Feishu API가 post 페이로드를 거부했습니다. 이는 정상적인 대체 동작입니다. 자세한 내용은 로그를 확인하세요. |
| 봇이 이미지/파일을 받지 못함 | Feishu 앱에 `im:message` 및 `im:resource` 권한 범위를 부여 |
| 봇 ID가 자동 감지되지 않음 | 대개 Feishu의 봇 정보 엔드포인트에 연결할 때 발생한 일시적인 네트워크 문제입니다. 우회 방법으로 `FEISHU_BOT_OPEN_ID`와 `FEISHU_BOT_NAME`을 직접 설정하세요. |
| `FEISHU_ALLOW_BOTS`를 활성화한 후에도 상대 봇 메시지가 무시됨 | Hermes가 아직 자신의 ID를 확인하지 못한 것입니다. `FEISHU_BOT_OPEN_ID`를 설정하고, 앱에서 `sender_id_type=user_id`를 사용하는 경우 `FEISHU_BOT_USER_ID`도 설정하세요. |
| 상대 봇이 이름 대신 `ou_xxxxxx`로 표시됨 | `application:bot.basic_info:read` 범위를 부여하세요. |
| 승인 버튼을 클릭할 때 오류 200340 | **Interactive Card** 기능을 활성화하고 Feishu Developer Console에서 **Card Request URL**을 구성하세요. 위의 [필수 Feishu 앱 설정](#required-feishu-app-configuration)을 참조하세요. |
| `Webhook rate limit exceeded` | 같은 IP에서 분당 120개를 초과하는 요청이 발생했습니다. 일반적으로 잘못된 설정이나 루프가 원인입니다. |

## 도구 세트

Feishu / Lark는 Telegram 및 다른 게이트웨이 기반 메시징 플랫폼과 동일한 핵심 도구를 포함하는 `hermes-feishu` 플랫폼 프리셋을 사용합니다.
