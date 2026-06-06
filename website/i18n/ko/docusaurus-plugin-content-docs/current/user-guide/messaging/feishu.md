---
sidebar_position: 11
title: "Feishu / Lark"
description: "Hermes Agent를 Feishu 또는 Lark 봇으로 설정하기"
---

# Feishu / Lark 설정

Hermes Agent는 Feishu 및 Lark와 완전한 기능을 갖춘 봇으로 통합됩니다. 연결이 완료되면 다이렉트 메시지나 그룹 채팅에서 에이전트와 대화하고, 홈 채팅에서 cron 작업 결과를 받으며, 일반적인 게이트웨이 흐름을 통해 텍스트, 이미지, 오디오 및 파일 첨부물을 보낼 수 있습니다.

이 통합은 두 가지 연결 모드를 지원합니다:

- `websocket` — 권장됨; Hermes가 아웃바운드 연결을 열며 퍼블릭 웹훅 엔드포인트가 필요하지 않습니다.
- `webhook` — Feishu/Lark가 HTTP를 통해 게이트웨이로 이벤트를 푸시하도록 하려는 경우에 유용합니다.

## Hermes 동작 방식

| 컨텍스트 | 동작 |
|---------|----------|
| 다이렉트 메시지(DM) | Hermes는 모든 메시지에 응답합니다. |
| 그룹 채팅 | Hermes는 채팅에서 봇이 @멘션된 경우에만 응답합니다. |
| 공유 그룹 채팅 | 기본적으로 세션 기록은 공유 채팅 내에서 사용자별로 격리됩니다. |

이 공유 채팅 동작은 `config.yaml`에 의해 제어됩니다:

```yaml
group_sessions_per_user: true
```

각 채팅당 하나의 공유 대화를 명시적으로 원하는 경우에만 `false`로 설정하세요.

## 1단계: Feishu / Lark 앱 만들기

### 권장: 스캔하여 생성 (단일 명령어)

```bash
hermes gateway setup
```

**Feishu / Lark**를 선택하고 Feishu 또는 Lark 모바일 앱으로 QR 코드를 스캔하세요. Hermes가 자동으로 올바른 권한을 가진 봇 애플리케이션을 생성하고 자격 증명을 저장합니다.

### 대안: 수동 설정

스캔하여 생성 기능을 사용할 수 없는 경우, 마법사가 수동 입력 모드로 전환됩니다:

1. Feishu 또는 Lark 개발자 콘솔을 엽니다:
   - Feishu: [https://open.feishu.cn/](https://open.feishu.cn/)
   - Lark: [https://open.larksuite.com/](https://open.larksuite.com/)
2. 새 앱을 생성합니다.
3. **Credentials & Basic Info**에서 **App ID**와 **App Secret**을 복사합니다.
4. 앱의 **Bot** 기능을 활성화합니다.
5. `hermes gateway setup`을 실행하고 **Feishu / Lark**를 선택한 후, 프롬프트에 자격 증명을 입력합니다.

:::warning
App Secret을 비공개로 유지하세요. 이를 가진 사람은 누구나 귀하의 앱을 사칭할 수 있습니다.
:::

### 권한 구성

Feishu 개발자 콘솔에서 **Permission Management**로 이동하여 다음 범위를 추가합니다. 권한 페이지에서 일괄 가져오기(bulk-import)를 할 수 있습니다.

**필수 권한:**

| 범위(Scope) | 목적 |
|-------|---------|
| `im:message` | 메시지 수신 및 읽기 |
| `im:message:send_as_bot` | 봇으로서 메시지 보내기 |
| `im:resource` | 사용자가 보낸 이미지, 파일, 오디오 접근 |
| `im:chat` | 채팅/그룹 메타데이터 접근 |
| `im:chat:readonly` | 채팅 목록 및 멤버십 읽기 |

**권장 권한 (전체 기능을 위해):**

| 범위(Scope) | 목적 |
|-------|---------|
| `im:message.reactions:readonly` | 이모지 반응 이벤트 수신 |
| `admin:app.info:readonly` | @멘션 게이팅을 위한 봇 신원 자동 감지 |
| `contact:user.id:readonly` | 허용 목록(allowlist) 일치를 위한 사용자 ID 확인 |

### 이벤트 구성

**Events and Callbacks** 메뉴에서:

1. 연결 모드를 **Long Connection (WebSocket)**(권장)으로 설정하거나 웹훅 URL을 구성합니다.
2. **Event Configuration** 섹션에서 다음을 구독합니다:
   - `im.message.receive_v1` — 메시지 수신에 필요

### 앱 게시

권한과 이벤트를 구성한 후, **Version Management**로 이동하여 새 버전의 앱을 게시합니다. 권한은 버전이 게시되고 승인되기 전까지 적용되지 않습니다(엔터프라이즈 앱의 경우 관리자 승인이 필요할 수 있음).

## 2단계: 연결 모드 선택

### 권장: WebSocket 모드

노트북, 워크스테이션 또는 프라이빗 서버에서 Hermes를 실행할 때는 WebSocket 모드를 사용하세요. 퍼블릭 URL이 필요하지 않습니다. 공식 Lark SDK가 자동으로 재연결되는 영구적인 아웃바운드 WebSocket 연결을 열고 유지합니다.

```bash
FEISHU_CONNECTION_MODE=websocket
```

**요구 사항:** `websockets` Python 패키지가 설치되어 있어야 합니다. SDK가 연결 수명 주기, 하트비트 및 자동 재연결을 내부적으로 처리합니다.

**작동 방식:** 어댑터는 백그라운드 실행기 스레드(executor thread)에서 Lark SDK의 WebSocket 클라이언트를 실행합니다. 인바운드 이벤트(메시지, 반응, 카드 액션)가 메인 asyncio 루프로 디스패치됩니다. 연결이 끊어지면 SDK가 자동으로 재연결을 시도합니다.

### 선택 사항: Webhook 모드

이미 접근 가능한 HTTP 엔드포인트 뒤에서 Hermes를 실행하고 있는 경우에만 웹훅 모드를 사용하세요.

```bash
FEISHU_CONNECTION_MODE=webhook
```

웹훅 모드에서 Hermes는 HTTP 서버(`aiohttp` 사용)를 시작하고 다음 위치에서 Feishu 엔드포인트를 제공합니다:

```text
/feishu/webhook
```

**요구 사항:** `aiohttp` Python 패키지가 설치되어 있어야 합니다.

웹훅 서버 바인드 주소 및 경로를 사용자 지정할 수 있습니다:

```bash
FEISHU_WEBHOOK_HOST=127.0.0.1   # 기본값: 127.0.0.1
FEISHU_WEBHOOK_PORT=8765         # 기본값: 8765
FEISHU_WEBHOOK_PATH=/feishu/webhook  # 기본값: /feishu/webhook
```

Feishu가 URL 확인(Verification) 챌린지(`type: url_verification`)를 보내면 웹훅이 자동으로 응답하여 Feishu 개발자 콘솔에서 구독 설정을 완료할 수 있게 합니다. 챌린지 응답은 `FEISHU_VERIFICATION_TOKEN`이 설정된 경우 이에 의해 제어됩니다. 인증되지 않은 원격 시스템이 공격자가 제어하는 챌린지 데이터를 에코(echo)하여 엔드포인트 제어권을 증명하지 못하도록 토큰이 누락되거나 일치하지 않는 챌린지 요청은 거부됩니다.

## 3단계: Hermes 구성

### 옵션 A: 대화형 설정

```bash
hermes gateway setup
```

**Feishu / Lark**를 선택하고 프롬프트에 내용을 채웁니다.

### 옵션 B: 수동 구성

`~/.hermes/.env` 파일에 다음을 추가합니다:

```bash
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=secret_xxx
FEISHU_DOMAIN=feishu
FEISHU_CONNECTION_MODE=websocket

# 선택 사항이지만 강력히 권장됨
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
FEISHU_HOME_CHANNEL=oc_xxx
```

`FEISHU_DOMAIN` 허용 값:

- `feishu` (Feishu 중국 버전)
- `lark` (Lark 인터내셔널 버전)

## 4단계: 게이트웨이 시작

```bash
hermes gateway
```

그런 다음 Feishu/Lark에서 봇에게 메시지를 보내 연결이 정상적으로 작동하는지 확인합니다.

## 홈 채팅 (Home Chat)

Feishu/Lark 채팅에서 `/set-home`을 사용하여 해당 채팅을 cron 작업 결과 및 교차 플랫폼 알림을 위한 홈 채널로 지정하세요.

미리 구성할 수도 있습니다:

```bash
FEISHU_HOME_CHANNEL=oc_xxx
```

## 보안

### 사용자 허용 목록 (User Allowlist)

프로덕션 용도의 경우 Feishu Open ID의 허용 목록을 설정하세요:

```bash
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
```

허용 목록을 비워두면 봇에 도달할 수 있는 누구나 봇을 사용할 수 있습니다. 그룹 채팅에서는 메시지가 처리되기 전에 발신자의 open_id와 허용 목록이 대조됩니다.

### 웹훅 암호화 키 (Webhook Encryption Key)

웹훅 모드에서 실행할 때는 인바운드 웹훅 페이로드의 서명 확인을 활성화하기 위해 암호화 키를 설정하세요:

```bash
FEISHU_ENCRYPT_KEY=your-encrypt-key
```

이 키는 Feishu 앱 구성의 **Event Subscriptions** 섹션에서 찾을 수 있습니다. 설정하면 어댑터는 다음 서명 알고리즘을 사용하여 모든 웹훅 요청을 확인합니다:

```
SHA256(timestamp + nonce + encrypt_key + body)
```

계산된 해시는 타이밍 공격에 안전한 비교 방식(timing-safe comparison)을 통해 `x-lark-signature` 헤더와 비교됩니다. 서명이 없거나 유효하지 않은 요청은 HTTP 401 오류와 함께 거부됩니다.

:::tip
WebSocket 모드에서는 SDK 자체가 서명 검증을 처리하므로 `FEISHU_ENCRYPT_KEY`는 선택 사항입니다. 웹훅 모드에서는 프로덕션 환경에서 설정할 것을 강력히 권장합니다.
:::

### 확인 토큰 (Verification Token)

웹훅 페이로드 내부의 `token` 필드를 확인하는 추가 인증 계층입니다:

```bash
FEISHU_VERIFICATION_TOKEN=your-verification-token
```

이 토큰도 Feishu 앱의 **Event Subscriptions** 섹션에서 찾을 수 있습니다. 설정된 경우, 모든 인바운드 웹훅 페이로드의 `header` 객체에는 일치하는 `token`이 포함되어야 합니다. 일치하지 않는 토큰은 HTTP 401과 함께 거부됩니다.

심층 방어(defense in depth)를 위해 `FEISHU_ENCRYPT_KEY`와 `FEISHU_VERIFICATION_TOKEN`을 함께 사용할 수 있습니다.

## 그룹 메시지 정책

`FEISHU_GROUP_POLICY` 환경 변수는 그룹 채팅에서 Hermes의 응답 여부 및 방식을 제어합니다:

```bash
FEISHU_GROUP_POLICY=allowlist   # 기본값
```

| 값 | 동작 |
|-------|----------|
| `open` | Hermes는 모든 그룹의 모든 사용자의 @멘션에 응답합니다. |
| `allowlist` | Hermes는 `FEISHU_ALLOWED_USERS`에 나열된 사용자의 @멘션에만 응답합니다. |
| `disabled` | Hermes는 모든 그룹 메시지를 완전히 무시합니다. |

모든 모드에서, 메시지가 처리되려면 그룹 내에서 봇이 명시적으로 @멘션(또는 @all) 되어야 합니다. 다이렉트 메시지는 항상 이 관문을 우회합니다.

`FEISHU_REQUIRE_MENTION=false`로 설정하면 Hermes가 @멘션을 요구하지 않고 모든 그룹 트래픽을 읽도록 허용합니다:

```bash
FEISHU_REQUIRE_MENTION=false
```

채팅별로 제어하려면 `group_rules` 항목에 `require_mention`을 설정하세요. 아래 [그룹별 접근 제어](#per-group-access-control)를 참고하세요.

### 봇 신원 파악 (Bot Identity)

Hermes는 시작 시 봇의 `open_id`와 표시 이름(display name)을 자동으로 감지합니다. 자동 감지가 Feishu API에 도달할 수 없거나 앱이 테넌트(tenant) 범위의 사용자 ID를 사용하는 경우에만 수동으로 설정하면 됩니다:

```bash
FEISHU_BOT_OPEN_ID=ou_xxx     # 자동 감지 실패 시에만
FEISHU_BOT_USER_ID=xxx        # 앱이 sender_id_type=user_id 를 사용하는 경우 필수
FEISHU_BOT_NAME=MyBot         # 자동 감지 실패 시에만
```

## 봇 대 봇 (Bot-to-Bot) 메시징

기본적으로 Hermes는 다른 봇이 보낸 메시지를 무시합니다. Hermes가 A2A 오케스트레이션에 참여하거나 동일한 그룹의 다른 봇으로부터 알림을 받아야 할 때 봇 대 봇 메시징을 활성화하세요.

```bash
FEISHU_ALLOW_BOTS=mentions   # 기본값: none
```

| 값 | 동작 |
|-------|----------|
| `none` | 다른 봇의 모든 메시지 무시 (기본값). |
| `mentions` | 상대 봇이 Hermes를 @멘션할 때만 허용. |
| `all` | 모든 상대 봇 메시지 허용. |

`config.yaml` 의 `feishu.allow_bots`로도 설정 가능합니다 (둘 다 설정된 경우 env 변수 우선).

상대 봇은 `FEISHU_ALLOWED_USERS`에 추가할 필요가 없습니다. 해당 허용 목록은 인간 발신자에게만 적용됩니다.

상대 봇 이름을 표시하려면 `application:bot.basic_info:read` 범위를 부여하세요. 이 권한이 없어도 상대 봇 라우팅은 올바르게 작동하지만 이름 대신 `open_id` 로 표시됩니다.

## 상호작용형 카드 액션 (Interactive Card Actions)

사용자가 봇이 보낸 상호작용형 카드의 버튼을 클릭하거나 상호작용할 때, 어댑터는 이를 합성(synthetic) `/card` 명령 이벤트로 라우팅합니다:

- 버튼 클릭은 다음과 같이 변환됩니다: `/card button {"key": "value", ...}`
- 카드 정의의 액션 `value` 페이로드가 JSON으로 포함됩니다.
- 카드 액션은 중복 처리를 방지하기 위해 15분 창(window) 내에서 중복이 제거됩니다.

게이트웨이 기반의 업데이트 프롬프트는 일반 텍스트 답장으로 예외처리(fallback)하지 않고 네이티브 Feishu `Yes` / `No` 카드를 사용합니다. `hermes update --gateway` 명령이 확인을 필요로 할 때 어댑터는 선택된 답변을 Hermes의 `.update_response` 파일에 기록하고 해당 카드를 해결된 상태(resolved state)로 인라인 교체합니다.

카드 액션 이벤트는 `MessageType.COMMAND`와 함께 디스패치되므로 일반적인 명령어 처리 파이프라인을 거쳐 흐르게 됩니다.

이것은 **명령어 승인** 방식이기도 합니다. 에이전트가 위험한 명령을 실행해야 할 때 한 번만 허용 / 세션 허용 / 항상 허용 / 거부 버튼이 포함된 대화형 카드를 보냅니다. 사용자가 버튼을 클릭하면 카드 액션 콜백이 승인 결정 결과를 에이전트에 다시 전달합니다.

### Feishu 앱 필수 설정

상호작용형 카드 기능을 사용하려면 Feishu 개발자 콘솔에서 **세 가지** 설정 단계가 필요합니다. 이 중 하나라도 누락되면 사용자가 카드 버튼을 클릭할 때 오류 **200340**이 발생합니다.

1. **카드 액션 이벤트 구독:**
   **Event Subscriptions** 에서 `card.action.trigger`를 구독 이벤트에 추가하세요.

2. **Interactive Card 기능 활성화:**
   **App Features > Bot** 에서 **Interactive Card** 토글이 켜져 있는지 확인하세요. 이는 앱이 카드 액션 콜백을 받을 수 있음을 Feishu에 알리는 역할입니다.

3. **카드 요청 URL 구성 (Webhook 모드만 해당):**
   **App Features > Bot > Message Card Request URL** 에 이벤트 웹훅과 동일한 엔드포인트를 설정하세요 (예: `https://your-server:8765/feishu/webhook`). WebSocket 모드에서는 SDK가 이를 자동으로 처리합니다.

:::warning
세 단계가 모두 갖춰지지 않으면 Feishu는 상호작용형 카드를 성공적으로 *보내지만*(보내는 데는 `im:message:send` 권한만 필요함), 버튼을 클릭할 때 오류 200340을 반환합니다. 카드가 제대로 작동하는 것처럼 보이며 사용자가 상호작용할 때에만 오류가 표면화됩니다.
:::

## 문서 댓글 스마트 답글 (Document Comment Intelligent Reply)

채팅 외에도 어댑터는 **Feishu/Lark 문서**에 남겨진 `@` 멘션에도 응답할 수 있습니다. 사용자가 문서에 댓글(로컬 텍스트 선택 또는 전체 문서 댓글)을 달고 봇을 @멘션하면, Hermes는 문서와 해당 댓글 스레드를 읽은 다음 스레드에 인라인으로 LLM 답변을 게시합니다.

이 기능은 `drive.notice.comment_add_v1` 이벤트에 의해 구동되며, 핸들러의 역할은 다음과 같습니다:

- 문서 내용과 댓글 타임라인을 병렬로 가져옵니다(전체 문서 스레드는 20개 메시지, 로컬 선택 스레드는 12개 메시지).
- 해당 단일 댓글 세션 범위에 맞춘 `feishu_doc` + `feishu_drive` 도구 세트와 함께 에이전트를 실행합니다.
- 답변을 4000자 단위로 나누어 스레드에 답글로 게시합니다.
- 동일한 문서에 달린 후속 댓글들이 문맥을 유지할 수 있도록 문서별 세션을 메시지 50개 제한으로 1시간 동안 캐싱합니다.

### 3계층 접근 제어 (3-Tier Access Control)

문서 댓글의 답변 기능은 **명시적으로 승인된 경우에만(explicit-grant only)** 작동하며, 암시적인 모두 허용(allow-all) 모드는 없습니다. 권한은 다음 순서로 확인되며 필드별로 첫 번째로 일치하는 규칙이 적용됩니다:

1. **정확한 문서(Exact doc)** — 특정 문서 토큰으로 범위가 지정된 규칙
2. **와일드카드(Wildcard)** — 여러 문서의 패턴과 일치하는 규칙
3. **최상위(Top-level)** — 워크스페이스에 적용되는 기본 규칙

규칙당 두 가지 정책을 사용할 수 있습니다:

- **`allowlist`** — 허용된 사용자 / 테넌트의 정적(static) 목록.
- **`pairing`** — 정적 목록 ∪ 런타임에 승인된 저장소. 중재자가 실시간으로 권한을 부여할 수 있어 배포(rollout) 시 유용합니다.

규칙은 수정 시간(mtime) 기준으로 캐싱되는 핫-리로딩(hot-reload) 방식을 통해 `~/.hermes/feishu_comment_rules.json`에 저장됩니다 (페어링 권한 부여 내역은 `~/.hermes/feishu_comment_pairing.json`). 파일 내용을 수정하면 게이트웨이를 다시 시작할 필요 없이 다음 댓글 이벤트에 바로 적용됩니다.

CLI 명령:

```bash
# 현재 규칙 및 페어링 상태 검사
python -m gateway.platforms.feishu_comment_rules status

# 특정 문서 + 사용자에 대한 접근 확인 시뮬레이션
python -m gateway.platforms.feishu_comment_rules check <fileType:fileToken> <user_open_id>

# 런타임 환경에서 페어링 부여 관리
python -m gateway.platforms.feishu_comment_rules pairing list
python -m gateway.platforms.feishu_comment_rules pairing add <user_open_id>
python -m gateway.platforms.feishu_comment_rules pairing remove <user_open_id>
```

### Feishu 앱 필수 구성

앞서 부여받은 채팅/카드 권한 외에 문서 댓글 이벤트를 추가해야 합니다:

- **Event Subscriptions**에서 `drive.notice.comment_add_v1`을 구독합니다.
- 핸들러가 문서 내용을 읽을 수 있도록 `docs:doc:readonly` 및 `drive:drive:readonly` 범위를 부여합니다.

## 화상 회의 초대 이벤트 (Meeting Invitation Events)

사람을 화상 회의에 초대하는 것과 같은 방식으로 Hermes Feishu/Lark 봇을 초대할 수 있습니다. 봇이 화상 회의 초대 이벤트를 수신하면 Hermes가 자동으로 화상 회의 참여를 시도하는 에이전트 턴을 시작할 수 있습니다.

이 기능은 `vc.bot.meeting_invited_v1` 이벤트에 의해 구동되며, 다음 흐름을 따릅니다:

- 사용자가 Feishu/Lark 화상 회의에 봇을 초대합니다.
- Feishu/Lark가 Hermes에 회의 초대 이벤트를 전송합니다.
- Hermes는 초대자, 회의 주제 및 회의 번호를 추출합니다.
- 초대자가 일반 게이트웨이 허용 목록이나 페어링 정책을 통해 승인된 사용자인 경우, 에이전트가 회의 번호를 받아 자동으로 참여를 시도합니다.
- 초대가 형식이 잘못되었거나 에이전트가 참여할 수 없는 경우, Hermes는 이벤트를 삭제하거나 초대자에게 간결한 설명과 함께 회신합니다.

초대자와 `meeting_no`를 모두 포함하지 않는 잘못된 형식의 초대는 무시됩니다.

### Feishu 앱 필수 구성

앞서 부여받은 채팅/카드 권한 외에 화상 회의 초대 이벤트를 추가해야 합니다:

- **Event Subscriptions**에서 `vc.bot.meeting_invited_v1`을 구독합니다.
- 해당 이벤트에 대해 Feishu/Lark 개발자 콘솔에서 요청하는 Video Conferencing 권한 범위를 활성화합니다.
- Hermes가 초대자에게 회신할 수 있도록 `im:message` 및 `im:message:send_as_bot` 권한을 계속 유지합니다.
- 게이트웨이 사용자 허용 목록 또는 페어링 정책이 초대자에게 권한을 부여하도록 확인합니다. 회의 초대도 일반 게이트웨이 액세스 확인을 거칩니다.

## 미디어 지원

### 인바운드 (수신)

어댑터는 사용자로부터 미디어 첨부 파일을 받아 에이전트 처리를 위해 로컬에 캐싱합니다:

| 유형 | 처리 방법 |
|------|-----------------|
| **이미지** | 다운로드되어 로컬에 캐시됨. URL 기반과 base64로 인코딩된 이미지 모두 지원. |
| **파일** | 다운로드되어 캐시됨. 원래 메시지의 파일 이름이 유지됨. |
| **음성** | 사용 가능한 경우 음성 메시지 텍스트 변환 추출. |
| **혼합 메시지** | WeCom 혼합형 메시지(텍스트 + 이미지)를 파싱하여 모든 구성 요소 추출. |

**인용 메시지:** 사용자가 답장하는 대상을 에이전트가 알 수 있도록, 인용된(답장된) 메시지의 미디어 또한 추출됩니다.

### 아웃바운드 (발신)

| 메서드 | 발신 내용 | 크기 제한 |
|--------|--------------|------------|
| `send` | 마크다운 텍스트 메시지 | 4000자 |
| `send_image` / `send_image_file` | 네이티브 이미지 메시지 | 10 MB |
| `send_document` | 파일 첨부물 | 20 MB |
| `send_voice` | 음성 메시지 (네이티브 음성의 경우 AMR 형식만 가능) | 2 MB |
| `send_video` | 동영상 메시지 | 10 MB |

**청크 단위 업로드 (Chunked upload):** 파일은 3단계 프로토콜(init → chunks → finish)을 통해 512KB 단위의 청크로 업로드됩니다. 이 과정은 어댑터가 자동으로 처리합니다.

**자동 다운그레이드:** 미디어가 네이티브 유형의 크기 제한을 초과하지만 절대 파일 제한(20MB)을 넘지 않을 경우 자동으로 일반 첨부 파일 형식으로 전환되어 전송됩니다:

- 이미지 > 10 MB → 파일로 전송
- 비디오 > 10 MB → 파일로 전송
- 음성 > 2 MB → 파일로 전송
- 비 AMR 오디오 → 파일로 전송 (WeCom은 네이티브 음성에 대해 AMR만 지원)

절대 제한인 20MB를 초과하는 파일은 거부되며 채팅창에 안내 메시지가 전송됩니다.

## 마크다운 렌더링 및 Post Fallback

발신 텍스트에 마크다운 포맷(제목, 굵게, 목록, 코드 블록, 링크 등)이 포함되어 있으면, 어댑터는 이를 일반 텍스트 대신 `md` 태그가 포함된 Feishu **post** 메시지로 자동 전송합니다. 이렇게 하면 Feishu 클라이언트에서 리치(rich) 렌더링이 가능합니다.

Feishu API가 (예: 지원되지 않는 마크다운 구문 때문에) post 페이로드를 거부하는 경우, 어댑터는 마크다운을 제거한 채 일반 텍스트로 보내는 방식으로 자동으로 예외처리를 수행합니다. 이 2단계 방식은 메시지가 항상 전달되도록 보장합니다.

일반 텍스트 메시지(마크다운이 감지되지 않음)는 단순 `text` 메시지 유형으로 전송됩니다.

## 처리 상태 반응

에이전트가 작업하는 동안 봇은 사용자의 메시지에 `Typing` 반응을 표시합니다. 답장이 오면 반응이 지워지고, 처리에 실패하면 `CrossMark`로 바뀝니다.

끄려면 `FEISHU_REACTIONS=false`를 설정하세요.

## 버스트 보호(Burst Protection) 및 일괄 처리(Batching)

어댑터에는 에이전트에 부하가 걸리는 것을 방지하기 위해 단기간에 쏟아지는 연속된 메시지에 대한 디바운싱(debouncing) 기능이 포함되어 있습니다.

### 텍스트 일괄 처리

사용자가 짧은 시간에 여러 텍스트 메시지를 보낼 경우 전송 전에 이 메시지들을 단일 이벤트로 병합합니다.

| 설정 | 환경 변수 | 기본값 |
|---------|---------|---------|
| 대기 시간 | `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` | 0.6초 |
| 일괄 처리당 최대 메시지 수 | `HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES` | 8 |
| 일괄 처리당 최대 문자 수 | `HERMES_FEISHU_TEXT_BATCH_MAX_CHARS` | 4000 |

### 미디어 일괄 처리

짧은 시간에 여러 미디어 첨부 파일이 전송된 경우(예: 이미지를 여러 장 드래그) 단일 이벤트로 병합합니다.

| 설정 | 환경 변수 | 기본값 |
|---------|---------|---------|
| 대기 시간 | `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | 0.8초 |

### 채팅별 직렬화 (Per-Chat Serialization)

같은 채팅방 내의 메시지는 대화의 일관성을 유지하기 위해 (한 번에 하나씩) 직렬로 처리됩니다. 각 채팅에는 고유한 잠금(lock) 기능이 있으므로 서로 다른 채팅방의 메시지는 동시에 처리됩니다.

## 속도 제한 (Webhook 모드)

웹훅 모드에서 어댑터는 오남용을 방지하기 위해 IP별로 속도 제한을 적용합니다.

- **기간 (Window):** 60초의 슬라이딩 윈도우
- **제한 (Limit):** (app_id, path, IP) 조합 당 윈도우별 120개 요청
- **추적 제한 (Tracking cap):** 최대 4096개의 고유 키 추적 (무한한 메모리 증가 방지)

제한을 초과하는 요청에는 HTTP 429(Too Many Requests)가 반환됩니다.

### 웹훅 이상 탐지 (Webhook Anomaly Tracking)

어댑터는 각 IP 주소에 대해 연속된 오류 응답을 추적합니다. 동일한 IP에서 6시간 동안 25회의 연속 오류가 발생하면 경고가 로깅됩니다. 이는 잘못 구성된 클라이언트나 탐색 시도를 감지하는 데 도움이 됩니다.

웹훅의 추가 보호 기능:
- **본문 크기 제한:** 최대 1 MB
- **본문 읽기 시간 초과:** 30초
- **Content-Type 강제:** 오직 `application/json`만 허용

## WebSocket 튜닝

`websocket` 모드를 사용할 때는 재연결 및 핑(ping) 동작을 사용자에 맞춰 조정할 수 있습니다:

```yaml
platforms:
  feishu:
    extra:
      ws_reconnect_interval: 120   # 재연결 시도 사이의 시간(초) (기본값: 120)
      ws_ping_interval: 30         # WebSocket 핑 전송 주기(초) (선택 사항; 설정되지 않은 경우 SDK 기본값 사용)
```

| 설정 | 구성 키 | 기본값 | 설명 |
|---------|-----------|---------|-------------|
| 재연결 주기 | `ws_reconnect_interval` | 120s | 재연결 시도 사이의 대기 시간 |
| 핑 주기 | `ws_ping_interval` | _(SDK 기본값)_ | WebSocket 연결 유지를 위한 핑(ping) 발생 주기 |

## 그룹별 접근 제어 (Per-Group Access Control)

전역 `FEISHU_GROUP_POLICY` 외에도 config.yaml의 `group_rules`를 사용하면 각 그룹 채팅마다 세밀한 규칙을 설정할 수 있습니다.

```yaml
platforms:
  feishu:
    extra:
      default_group_policy: "open"     # group_rules에 없는 그룹을 위한 기본 정책
      admins:                          # 봇 설정을 관리할 수 있는 사용자들
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
          require_mention: false       # 이 채팅에서는 FEISHU_REQUIRE_MENTION 설정 재정의
```

| 정책 | 설명 |
|--------|-------------|
| `open` | 그룹 내 누구나 봇을 사용할 수 있음 |
| `allowlist` | 그룹의 `allowlist`에 포함된 사용자만 봇을 사용할 수 있음 |
| `blacklist` | 그룹의 `blacklist`에 포함된 사용자를 제외한 누구나 봇을 사용할 수 있음 |
| `admin_only` | 전역 `admins` 목록에 있는 사용자만 해당 그룹 내에서 봇을 사용할 수 있음 |
| `disabled` | 봇이 이 그룹 내의 모든 메시지를 무시함 |

`group_rules` 항목에서 `require_mention: false`를 설정하면 해당 채팅방의 @멘션 요구 사항을 건너뜁니다. 해당 값이 생략된 경우 전역 `FEISHU_REQUIRE_MENTION` 값을 상속받습니다.

`group_rules`에 기재되지 않은 그룹들은 `default_group_policy` 설정을 따릅니다(기본값은 `FEISHU_GROUP_POLICY`의 값을 따릅니다).

## 중복 제거 (Deduplication)

인바운드 메시지는 24시간 TTL을 갖는 메시지 ID를 사용하여 중복이 제거됩니다. 중복 제거 상태(dedup state)는 다시 시작 시 `~/.hermes/feishu_seen_message_ids.json`에 영구적으로 보존됩니다.

| 설정 | 환경 변수 | 기본값 |
|---------|---------|---------|
| 캐시 크기 | `HERMES_FEISHU_DEDUP_CACHE_SIZE` | 2048개 항목 |

## 모든 환경 변수

| 변수 | 필수 | 기본값 | 설명 |
|----------|----------|---------|-------------|
| `FEISHU_APP_ID` | ✅ | — | Feishu/Lark 앱 ID |
| `FEISHU_APP_SECRET` | ✅ | — | Feishu/Lark 앱 비밀 키(App Secret) |
| `FEISHU_DOMAIN` | — | `feishu` | `feishu` (중국) 또는 `lark` (인터내셔널) |
| `FEISHU_CONNECTION_MODE` | — | `websocket` | `websocket` 또는 `webhook` |
| `FEISHU_ALLOWED_USERS` | — | _(비어있음)_ | 사용자 허용 목록을 위한 쉼표로 구분된 open_id 목록 |
| `FEISHU_ALLOW_BOTS` | — | `none` | 다른 봇의 메시지 허용 모드: `none`, `mentions`, 또는 `all` |
| `FEISHU_REQUIRE_MENTION` | — | `true` | 그룹 메시지 전송 시 봇을 필수로 @멘션해야 하는지 여부 |
| `FEISHU_HOME_CHANNEL` | — | — | cron/알림 출력을 위한 채팅 ID |
| `FEISHU_ENCRYPT_KEY` | — | _(비어있음)_ | 웹훅 서명 검증을 위한 암호화 키 |
| `FEISHU_VERIFICATION_TOKEN` | — | _(비어있음)_ | 웹훅 페이로드 인증을 위한 검증 토큰 |
| `FEISHU_GROUP_POLICY` | — | `allowlist` | 그룹 메시지 처리 정책: `open`, `allowlist`, `disabled` |
| `FEISHU_BOT_OPEN_ID` | — | _(비어있음)_ | 봇의 open_id (@멘션 탐지 목적) |
| `FEISHU_BOT_USER_ID` | — | _(비어있음)_ | 봇의 user_id (@멘션 탐지 목적) |
| `FEISHU_BOT_NAME` | — | _(비어있음)_ | 봇의 표시 이름 (@멘션 탐지 목적) |
| `FEISHU_WEBHOOK_HOST` | — | `127.0.0.1` | 웹훅 서버 바인드 주소 |
| `FEISHU_WEBHOOK_PORT` | — | `8765` | 웹훅 서버 포트 |
| `FEISHU_WEBHOOK_PATH` | — | `/feishu/webhook` | 웹훅 엔드포인트 경로 |
| `HERMES_FEISHU_DEDUP_CACHE_SIZE` | — | `2048` | 추적할 중복 제거된 최대 메시지 ID 개수 |
| `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` | — | `0.6` | 텍스트 버스트 디바운스 대기 시간 |
| `HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES` | — | `8` | 단일 텍스트 일괄 처리 단위당 병합되는 최대 메시지 개수 |
| `HERMES_FEISHU_TEXT_BATCH_MAX_CHARS` | — | `4000` | 단일 텍스트 일괄 처리 단위당 병합되는 최대 글자 수 |
| `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | — | `0.8` | 미디어 버스트 디바운스 대기 시간 |

WebSocket 설정과 그룹별 ACL 설정은 `config.yaml` 파일 내 `platforms.feishu.extra` 아래에 정의됩니다 (위의 [WebSocket 튜닝](#websocket-tuning) 및 [그룹별 접근 제어](#per-group-access-control) 참조).

## 문제 해결

| 문제점 | 해결 방법 |
|---------|-----|
| `lark-oapi not installed` | SDK를 설치합니다: `pip install lark-oapi` |
| `websockets not installed; websocket mode unavailable` | websockets 패키지를 설치합니다: `pip install websockets` |
| `aiohttp not installed; webhook mode unavailable` | aiohttp 패키지를 설치합니다: `pip install aiohttp` |
| `FEISHU_APP_ID or FEISHU_APP_SECRET not set` | 두 환경 변수를 지정하거나 `hermes gateway setup`을 통해 설정합니다. |
| `Another local Hermes gateway is already using this Feishu app_id` | 하나의 app_id는 동시에 단일 Hermes 인스턴스에서만 구동할 수 있습니다. 이미 작동 중인 다른 게이트웨이를 중지하세요. |
| 봇이 그룹 내에서 응답하지 않음 | 봇이 제대로 @멘션되었는지 확인하고, `FEISHU_GROUP_POLICY` 설정을 점검하세요. 만약 `allowlist` 정책이라면, 메시지를 보낸 사람이 `FEISHU_ALLOWED_USERS` 내에 있는지 확인하세요. |
| `Webhook rejected: invalid verification token` | `FEISHU_VERIFICATION_TOKEN`의 값이 Feishu 앱 Event Subscriptions 설정에 등록된 토큰 값과 일치하는지 확인하세요. |
| `Webhook rejected: invalid signature` | `FEISHU_ENCRYPT_KEY`의 값이 Feishu 앱 설정의 encrypt key와 일치하는지 확인하세요. |
| Post 메시지가 일반 텍스트(plain text)로 표시됨 | Feishu API가 post 페이로드를 거부했을 때 일어나는 정상적인 폴백(fallback) 현상입니다. 자세한 오류 내용은 로그를 확인하세요. |
| 봇이 이미지나 파일을 수신하지 못함 | Feishu 앱에 `im:message` 와 `im:resource` 권한(scope)을 인가했는지 점검하세요. |
| 봇의 신원(Identity) 자동 파악 안 됨 | 일반적으로 Feishu의 bot info 엔드포인트에 접속하는 과정 중 나타나는 일시적인 네트워크 이슈입니다. 해결책으로 `FEISHU_BOT_OPEN_ID` 와 `FEISHU_BOT_NAME` 을 수동 지정할 수 있습니다. |
| `FEISHU_ALLOW_BOTS` 옵션을 켰으나 다른 봇의 메시지가 계속 무시됨 | Hermes가 아직 자신의 신원을 인지하지 못해서 일어나는 현상입니다. `FEISHU_BOT_OPEN_ID` 를 지정하세요 (앱에서 `sender_id_type=user_id`를 사용할 경우 `FEISHU_BOT_USER_ID` 도 필요함). |
| 다른 봇이 이름 대신 `ou_xxxxxx` 형태로 표시됨 | `application:bot.basic_info:read` 권한을 부여하세요. |
| 승인(Approval) 버튼 클릭 시 에러 200340 발생 | Feishu Developer Console에서 **Interactive Card** 기능을 활성화하고 **Card Request URL** 항목을 구성하세요. 위에 적힌 [Feishu 앱 필수 설정](#required-feishu-app-configuration) 파트를 참고하세요. |
| `Webhook rate limit exceeded` | 동일한 IP를 통한 요청이 1분당 120회를 넘었을 때 발생합니다. 대개 잘못된 설정이나 무한 루프 때문입니다. |

## 도구 세트 (Toolset)

Feishu / Lark는 Telegram 등 게이트웨이 기반의 타 플랫폼들과 동일하게 핵심 툴들이 담겨 있는 `hermes-feishu` 플랫폼 프리셋을 사용합니다.
