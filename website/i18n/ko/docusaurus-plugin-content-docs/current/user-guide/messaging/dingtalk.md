---
sidebar_position: 10
title: "DingTalk"
description: "Hermes Agent를 DingTalk 챗봇으로 설정하기"
---

# DingTalk 설정

Hermes Agent는 챗봇으로서 DingTalk(딩톡, 钉钉)에 통합되어, 다이렉트 메시지나 그룹 채팅을 통해 AI 비서와 대화할 수 있게 합니다. 봇은 공개 URL이나 웹훅 서버가 필요 없는 장기 활성 WebSocket 연결인 DingTalk의 Stream Mode를 통해 접속하며, DingTalk의 세션 웹훅 API를 통해 마크다운 형식의 메시지로 답변합니다.

설정하기 전에 대부분의 사람들이 가장 알고 싶어 하는 부분인 Hermes가 DingTalk 워크스페이스에서 어떻게 작동하는지에 대해 알아봅시다.

## Hermes의 동작 방식

| 컨텍스트 | 동작 방식 |
|---------|----------|
| **DM (1:1 채팅)** | Hermes는 모든 메시지에 응답합니다. `@mention`이 필요하지 않습니다. 각 DM은 고유한 세션을 갖습니다. |
| **그룹 채팅** | Hermes는 사용자가 봇을 `@mention`할 때 응답합니다. 멘션이 없으면 메시지를 무시합니다. |
| **여러 사용자가 있는 공유 그룹** | 기본적으로 Hermes는 그룹 내 각 사용자별로 세션 기록을 독립적으로 유지합니다. 동일한 그룹에서 대화하는 두 사람은 이를 명시적으로 비활성화하지 않는 한 하나의 대화 내용을 공유하지 않습니다. |

### DingTalk에서의 세션 모델

기본적으로:

- 각 DM은 고유한 세션을 가집니다.
- 공유 그룹 채팅의 각 사용자는 해당 그룹 내에서 본인만의 세션을 가집니다.

이는 `config.yaml`에 의해 제어됩니다:

```yaml
group_sessions_per_user: true
```

전체 그룹을 위한 하나의 공유된 대화를 원할 때만 `false`로 설정하십시오:

```yaml
group_sessions_per_user: false
```

이 가이드는 DingTalk 봇 생성부터 첫 메시지 전송까지의 전체 설정 과정을 안내합니다.

## 전제 조건

필요한 Python 패키지를 설치하세요:

```bash
pip install "hermes-agent[dingtalk]"
```

또는 개별적으로:

```bash
pip install dingtalk-stream httpx alibabacloud-dingtalk
```

- `dingtalk-stream` — Stream Mode(WebSocket 기반 실시간 메시징)를 위한 DingTalk 공식 SDK
- `httpx` — 세션 웹훅을 통해 응답을 보낼 때 사용되는 비동기 HTTP 클라이언트
- `alibabacloud-dingtalk` — AI 카드, 이모지 리액션 및 미디어 다운로드를 위한 DingTalk OpenAPI SDK

## 1단계: DingTalk 앱 생성

1. [DingTalk Developer Console](https://open-dev.dingtalk.com/)로 이동합니다.
2. DingTalk 관리자 계정으로 로그인합니다.
3. **Application Development** → **Custom Apps** → **Create App via H5 Micro-App** (또는 콘솔 버전에 따라 **Robot**)을 클릭합니다.
4. 내용을 채웁니다:
   - **App Name**: 예: `Hermes Agent`
   - **Description**: 선택 사항
5. 생성 후, **Credentials & Basic Info**로 이동하여 **Client ID**(AppKey)와 **Client Secret**(AppSecret)을 찾습니다. 둘 다 복사해 둡니다.

:::warning[자격 증명은 한 번만 표시됨]
Client Secret은 앱을 생성할 때 딱 한 번만 표시됩니다. 분실하면 다시 생성해야 합니다. 이 자격 증명을 공개적으로 공유하거나 Git에 커밋하지 마세요.
:::

## 2단계: Robot 기능 활성화

1. 앱 설정 페이지에서 **Add Capability** → **Robot**으로 이동합니다.
2. 로봇 기능을 활성화합니다.
3. **Message Reception Mode** 아래에서 **Stream Mode**를 선택합니다 (권장 — 공개 URL 불필요).

:::tip
Stream Mode 설정을 권장합니다. 로컬 컴퓨터에서 시작되는 장기 활성 WebSocket 연결을 사용하므로, 공인 IP나 도메인 이름 또는 웹훅 엔드포인트가 필요하지 않습니다. 이는 NAT, 방화벽 내부 및 로컬 시스템에서 잘 작동합니다.
:::

## 3단계: 본인의 DingTalk User ID 찾기

Hermes Agent는 여러분의 DingTalk User ID를 사용하여 누가 봇과 상호작용할 수 있는지 제어합니다. DingTalk User ID는 조직의 관리자가 설정한 영숫자 문자열입니다.

본인의 ID를 찾으려면:

1. DingTalk 조직 관리자에게 문의하세요 — User ID는 DingTalk 관리자 콘솔의 **Contacts** → **Members**에 구성되어 있습니다.
2. 대안으로, 봇은 수신된 각 메시지에 대해 `sender_id`를 기록합니다. 게이트웨이를 시작하고 봇에 메시지를 보낸 다음, 로그에서 ID를 확인하세요.

## 4단계: Hermes Agent 구성

### 옵션 A: 대화형 설정 (권장)

안내형 설정 명령을 실행합니다:

```bash
hermes gateway setup
```

프롬프트가 나타나면 **DingTalk**를 선택하세요. 설정 마법사는 두 가지 경로 중 하나를 통해 인증할 수 있습니다:

- **QR 코드 장치 인증 흐름 (권장).** DingTalk 모바일 앱으로 터미널에 표시된 QR을 스캔하세요 — 귀하의 Client ID와 Client Secret이 자동으로 반환되어 `~/.hermes/.env`에 쓰여집니다. 개발자 콘솔에 접속할 필요가 없습니다.
- **수동 붙여넣기.** 자격 증명이 이미 있거나 QR 스캔이 불편한 경우, 프롬프트에서 Client ID, Client Secret, 허용할 User ID들을 붙여넣으세요.

:::note openClaw 브랜딩 안내
DingTalk의 `verification_uri_complete`가 API 계층에서 openClaw 식별자에 하드코딩되어 있기 때문에, Alibaba / DingTalk-Real-AI 측에서 Hermes 전용 템플릿을 서버단에 등록해주기 전까지는 QR 코드가 `openClaw` 소스 문자열 아래에서 인증을 진행합니다. 이는 순전히 DingTalk이 동의 화면을 어떻게 렌더링하는지에 대한 것일 뿐 — 당신이 만드는 봇은 전적으로 본인 소유이며 본인의 테넌트 내에서 프라이빗하게 작동합니다.
:::

### 옵션 B: 수동 구성

`~/.hermes/.env` 파일에 다음을 추가합니다:

```bash
# 필수
DINGTALK_CLIENT_ID=your-app-key
DINGTALK_CLIENT_SECRET=your-app-secret

# 보안: 봇과 상호작용할 수 있는 사람 제한
DINGTALK_ALLOWED_USERS=user-id-1

# 다수의 허용된 사용자 (쉼표로 구분)
# DINGTALK_ALLOWED_USERS=user-id-1,user-id-2

# 선택 사항: 그룹 채팅 통제 (Slack/Telegram/Discord/WhatsApp과 동일)
# DINGTALK_REQUIRE_MENTION=true
# DINGTALK_FREE_RESPONSE_CHATS=cidABC==,cidDEF==
# DINGTALK_MENTION_PATTERNS=^小马
# DINGTALK_HOME_CHANNEL=cidXXXX==
# DINGTALK_ALLOW_ALL_USERS=true
```

`~/.hermes/config.yaml`의 선택적 동작 설정:

```yaml
group_sessions_per_user: true

gateway:
  platforms:
    dingtalk:
      extra:
        # 그룹에서 봇이 답변하기 위해 @mention을 요구함 (Slack/Telegram/Discord와의 일관성).
        # DM은 이를 무시합니다 — 봇은 1:1 채팅에서 항상 답변합니다.
        require_mention: true

        # 플랫폼별 허용 목록. 설정하면 명시된 DingTalk 사용자 ID들만 봇과 상호작용 가능
        # (.env 대신 여기로 범위가 지정된다는 점을 제외하면 DINGTALK_ALLOWED_USERS와 같은 의미).
        allowed_users:
          - user-id-1
          - user-id-2
```

- `group_sessions_per_user: true`는 공유 그룹 채팅 내에서 각 참가자의 컨텍스트를 분리된 상태로 유지합니다.
- `require_mention: true`는 봇이 모든 그룹 메시지에 반응하는 것을 막아줍니다 — 누군가 봇을 @-mention할 때만 답변합니다.
- `dingtalk.extra` 아래의 `allowed_users`는 `DINGTALK_ALLOWED_USERS`의 대안입니다; 두 가지가 모두 설정되어 있으면 결합(merge)됩니다.

### 게이트웨이 시작

구성이 완료되면 DingTalk 게이트웨이를 시작합니다:

```bash
hermes gateway
```

봇은 몇 초 내로 DingTalk의 Stream Mode에 연결되어야 합니다. 테스트하려면 봇에게 DM을 보내거나 봇이 추가된 그룹에 메시지를 보내보세요.

:::tip
지속적인 운영을 위해 `hermes gateway`를 백그라운드나 systemd 서비스로 실행할 수 있습니다. 자세한 내용은 배포 문서를 참조하세요.
:::

## 기능

### AI 카드

Hermes는 일반 마크다운 메시지 대신 DingTalk AI 카드를 사용하여 답변할 수 있습니다. 카드는 더 풍부하고 구조화된 디스플레이를 제공하며, 에이전트가 응답을 생성하는 동안 스트리밍 업데이트를 지원합니다.

AI 카드를 활성화하려면 `config.yaml`에 카드 템플릿 ID를 설정하세요:

```yaml
platforms:
  dingtalk:
    enabled: true
    extra:
      card_template_id: "your-card-template-id"
```

딩톡 개발자 콘솔(DingTalk Developer Console) 내 앱의 AI 카드 설정에서 카드 템플릿 ID를 찾을 수 있습니다. AI 카드가 활성화되면 모든 응답은 스트리밍 텍스트 업데이트와 함께 카드로 전송됩니다.

### 이모지 리액션

Hermes는 처리 상태를 보여주기 위해 메시지에 자동으로 이모지 리액션을 추가합니다:

- 🤔Thinking(생각 중) — 봇이 메시지를 처리하기 시작할 때 추가됨
- 🥳Done(완료) — 응답이 완료되었을 때 추가됨 (Thinking 리액션을 대체)

이 리액션들은 DM과 그룹 채팅 모두에서 작동합니다.

### 표시 설정

다른 플랫폼들과 별개로 DingTalk의 표시 동작을 사용자 맞춤 설정할 수 있습니다:

```yaml
display:
  platforms:
    dingtalk:
      show_reasoning: false   # 응답에 모델 추론(reasoning/thinking) 표시 여부
      streaming: true         # 응답 스트리밍 활성화 (AI 카드에서 작동)
      tool_progress: all      # 도구 실행 진행 상태 표시 (all/new/off)
      interim_assistant_messages: true  # 중간 코멘트 메시지 표시
```

더 깔끔한 경험을 위해 도구 진행 상태와 중간 메시지를 비활성화하려면:

```yaml
display:
  platforms:
    dingtalk:
      tool_progress: off
      interim_assistant_messages: false
```

## 문제 해결

### 봇이 메시지에 응답하지 않음

**원인**: Robot 기능이 활성화되지 않았거나 `DINGTALK_ALLOWED_USERS`에 사용자 ID가 포함되지 않았습니다.

**해결 방법**: 앱 설정에서 Robot 기능이 활성화되어 있고 Stream Mode가 선택되었는지 검증하세요. 귀하의 사용자 ID가 `DINGTALK_ALLOWED_USERS`에 있는지 확인하세요. 게이트웨이를 다시 시작하세요.

### "dingtalk-stream not installed" 오류

**원인**: `dingtalk-stream` Python 패키지가 설치되지 않았습니다.

**해결 방법**: 이를 설치하세요:

```bash
pip install dingtalk-stream httpx
```

### "DINGTALK_CLIENT_ID and DINGTALK_CLIENT_SECRET required"

**원인**: 환경 변수나 `.env` 파일에 자격 증명이 설정되지 않았습니다.

**해결 방법**: `~/.hermes/.env`에 `DINGTALK_CLIENT_ID` 및 `DINGTALK_CLIENT_SECRET`가 제대로 설정되어 있는지 확인하세요. Client ID는 귀하의 AppKey이며 Client Secret은 DingTalk 개발자 콘솔의 AppSecret입니다.

### 스트림 끊김 / 재연결 반복

**원인**: 네트워크 불안정성, DingTalk 플랫폼 유지 보수 또는 자격 증명 문제입니다.

**해결 방법**: 어댑터는 지수 백오프(2초 → 5초 → 10초 → 30초 → 60초)로 자동 재연결을 시도합니다. 귀하의 자격 증명이 유효한지 앱이 비활성화되지 않았는지 확인하세요. 아웃바운드 WebSocket 연결이 네트워크에서 허용되는지 검증하세요.

### 봇이 오프라인임

**원인**: Hermes 게이트웨이가 실행 중이 아니거나 연결에 실패했습니다.

**해결 방법**: `hermes gateway`가 실행 중인지 확인하세요. 터미널 출력에서 오류 메시지를 확인하세요. 흔한 문제들: 잘못된 자격 증명, 앱 비활성화됨, `dingtalk-stream` 또는 `httpx` 설치되지 않음.

### "No session_webhook available"

**원인**: 봇이 응답을 시도했지만 세션 웹훅 URL이 없습니다. 이는 일반적으로 웹훅이 만료되었거나 메시지 수신과 응답 전송 사이에 봇이 재시작된 경우 발생합니다.

**해결 방법**: 봇에게 새 메시지를 보내세요 — 들어오는 각 메시지는 응답을 위한 새로운 세션 웹훅을 제공합니다. 이는 DingTalk의 일반적인 제한 사항입니다; 봇은 최근에 받은 메시지에만 응답할 수 있습니다.

## 보안

:::warning
봇과 상호작용할 수 있는 사람을 제한하기 위해 항상 `DINGTALK_ALLOWED_USERS`를 설정하세요. 설정하지 않으면 게이트웨이는 안전 조치로 기본적으로 모든 사용자를 차단합니다. 도구 사용 및 시스템 접근을 포함하여 에이전트 기능에 대한 모든 권한을 허용받게 되므로 신뢰할 수 있는 사용자 ID만 추가하세요.
:::

Hermes Agent 배포 보안에 관한 더 자세한 정보는 [보안 가이드](../security.md)를 참고하세요.

## 참고 사항

- **Stream Mode**: 공인 URL, 도메인 이름, 웹훅 서버가 필요하지 않습니다. WebSocket을 통해 여러분의 머신에서 시작되는 연결이므로 NAT 및 방화벽 환경 뒤에서 동작합니다.
- **AI 카드**: 마크다운 평문 대신 풍부한 AI 카드로 응답을 보낼 수 있습니다. `card_template_id`를 통해 구성하세요.
- **이모지 리액션**: 처리 상태에 따른 자동 🤔생각 중/🥳완료 리액션 표시.
- **마크다운 응답**: 응답은 리치 텍스트 표시를 위해 DingTalk 마크다운 포맷으로 전송됩니다.
- **미디어 지원**: 수신 메시지의 이미지 및 파일은 자동 분해되어 비전 도구(vision tools)에서 처리할 수 있게 됩니다.
- **메시지 중복 처리 차단**: 어댑터는 같은 메시지의 이중 처리를 막기 위해 5분의 창(window) 내에서 메시지를 중복 제거합니다.
- **자동 재연결**: 스트림 연결이 끊어지면 어댑터가 기하급수적 백오프 알고리즘을 사용해 자동으로 재연결을 시도합니다.
- **메시지 길이 제한**: 응답은 메시지당 20,000자로 제한됩니다. 더 긴 응답의 경우 텍스트가 잘릴 수 있습니다.
