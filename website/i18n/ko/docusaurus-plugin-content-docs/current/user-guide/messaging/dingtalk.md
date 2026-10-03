---
sidebar_position: 10
title: "DingTalk"
description: "Hermes Agent를 DingTalk 챗봇으로 설정"
---

# DingTalk 설정

Hermes Agent는 DingTalk(钉钉)와 챗봇으로 통합되므로, 일대일 메시지나 그룹 채팅을 통해 AI 어시스턴트와 대화할 수 있습니다. 봇은 공개 URL이나 웹훅 서버가 필요 없는 장기 실행 WebSocket 연결인 DingTalk의 Stream Mode를 통해 연결되며, DingTalk의 세션 웹훅 API를 사용해 마크다운 형식의 메시지로 답변합니다.

설정하기 전에, 대부분의 사람들이 가장 궁금해하는 부분부터 알아보겠습니다. DingTalk 작업 공간에서 Hermes가 어떻게 동작하는지입니다.

## Hermes의 동작 방식

| 상황 | 동작 |
|---------|----------|
| **DM(일대일 채팅)** | Hermes는 모든 메시지에 응답합니다. `@mention`은 필요하지 않습니다. 각 DM에는 자체 세션이 있습니다. |
| **그룹 채팅** | Hermes는 사용자가 봇을 `@mention`할 때 응답합니다. 멘션이 없으면 Hermes는 메시지를 무시합니다. |
| **여러 사용자가 있는 공유 그룹** | 기본적으로 Hermes는 그룹 내 사용자별로 세션 기록을 격리합니다. 명시적으로 비활성화하지 않는 한, 같은 그룹에서 대화하는 두 사람은 하나의 대화 기록을 공유하지 않습니다. |

### DingTalk의 세션 모델

기본값은 다음과 같습니다.

- 각 DM에는 자체 세션이 있습니다.
- 공유 그룹 채팅의 각 사용자는 해당 그룹 안에서 자신만의 세션을 가집니다.

이는 `config.yaml`로 제어합니다.

```yaml
group_sessions_per_user: true
```

그룹 전체에서 하나의 공유 대화를 사용하려는 경우에만 `false`로 설정합니다.

```yaml
group_sessions_per_user: false
```

이 가이드는 DingTalk 봇 생성부터 첫 메시지 전송까지 전체 설정 과정을 안내합니다.

## 사전 요구 사항

필요한 Python 패키지를 설치합니다.

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[dingtalk]"
```

또는 개별적으로 설치합니다.

```bash
pip install dingtalk-stream httpx alibabacloud-dingtalk
```

- `dingtalk-stream` — Stream Mode(WebSocket 기반 실시간 메시징)를 위한 DingTalk 공식 SDK
- `httpx` — 세션 웹훅을 통해 답변을 보내는 데 사용하는 비동기 HTTP 클라이언트
- `alibabacloud-dingtalk` — AI Card, 이모지 리액션, 미디어 다운로드를 위한 DingTalk OpenAPI SDK

## 1단계: DingTalk 앱 생성

1. [DingTalk Developer Console](https://open-dev.dingtalk.com/)로 이동합니다.
2. DingTalk 관리자 계정으로 로그인합니다.
3. **Application Development** → **Custom Apps** → **Create App via H5 Micro-App**를 클릭합니다(콘솔 버전에 따라 **Robot**일 수 있음).
4. 다음 정보를 입력합니다.
   - **App Name**: 예: `Hermes Agent`
   - **Description**: 선택 사항
5. 앱을 생성한 후 **Credentials & Basic Info**로 이동해 **Client ID**(AppKey)와 **Client Secret**(AppSecret)을 확인합니다. 둘 다 복사합니다.

:::warning[자격 증명은 한 번만 표시됨]
Client Secret은 앱을 생성할 때 한 번만 표시됩니다. 잃어버린 경우 다시 생성해야 합니다. 이 자격 증명을 공개적으로 공유하거나 Git에 커밋하지 마세요.
:::

## 2단계: Robot 기능 활성화

1. 앱 설정 페이지에서 **Add Capability** → **Robot**으로 이동합니다.
2. Robot 기능을 활성화합니다.
3. **Message Reception Mode**에서 **Stream Mode**를 선택합니다(권장 — 공개 URL이 필요하지 않음).

:::tip
Stream Mode가 권장되는 설정입니다. 컴퓨터에서 시작하는 장기 실행 WebSocket 연결을 사용하므로 공개 IP, 도메인 이름 또는 웹훅 엔드포인트가 필요하지 않습니다. NAT, 방화벽 뒤나 로컬 컴퓨터에서도 작동합니다.
:::

## 3단계: DingTalk 사용자 ID 확인

Hermes Agent는 DingTalk User ID를 사용해 봇과 상호작용할 수 있는 사용자를 제어합니다. DingTalk User ID는 조직 관리자가 설정하는 영숫자 문자열입니다.

확인 방법은 다음과 같습니다.

1. DingTalk 조직 관리자에게 문의합니다. User ID는 DingTalk 관리자 콘솔의 **Contacts** → **Members**에서 설정됩니다.
2. 또는 봇이 수신 메시지마다 `sender_id`를 로그에 기록합니다. 게이트웨이를 시작하고 봇에 메시지를 보낸 다음 로그에서 ID를 확인합니다.

## 4단계: Hermes Agent 구성

### 옵션 A: 대화형 설정(권장)

안내형 설정 명령을 실행합니다.

```bash
hermes gateway setup
```

메시지가 표시되면 **DingTalk**를 선택합니다. 설정 마법사는 다음 두 경로 중 하나로 인증할 수 있습니다.

- **QR 코드 기기 플로우(권장).** 터미널에 출력되는 QR을 DingTalk 모바일 앱으로 스캔하면 Client ID와 Client Secret이 자동으로 반환되어 `~/.hermes/.env`에 기록됩니다. 개발자 콘솔에 들어갈 필요가 없습니다.
- **수동 붙여넣기.** 이미 자격 증명이 있거나 QR 스캔이 불편한 경우, 메시지가 표시될 때 Client ID, Client Secret 및 허용된 사용자 ID를 붙여넣습니다.

:::note openClaw 브랜딩 공개
DingTalk의 `verification_uri_complete`가 API 계층에서 openClaw ID로 하드코딩되어 있으므로, Alibaba / DingTalk-Real-AI가 서버 측에 Hermes 전용 템플릿을 등록할 때까지 QR은 현재 `openClaw` 소스 문자열로 인증합니다. 이는 DingTalk이 동의 화면을 표시하는 방식일 뿐이며, 생성한 봇은 전적으로 사용자의 것이고 테넌트에 비공개입니다.
:::

### 옵션 B: 수동 구성

다음 내용을 `~/.hermes/.env` 파일에 추가합니다.

```bash
# Required
DINGTALK_CLIENT_ID=your-app-key
DINGTALK_CLIENT_SECRET=your-app-secret

# Security: restrict who can interact with the bot
DINGTALK_ALLOWED_USERS=user-id-1

# Multiple allowed users (comma-separated)
# DINGTALK_ALLOWED_USERS=user-id-1,user-id-2

# Optional: group-chat gating (mirrors Slack/Telegram/Discord/WhatsApp)
# DINGTALK_REQUIRE_MENTION=true
# DINGTALK_FREE_RESPONSE_CHATS=cidABC==,cidDEF==
# DINGTALK_MENTION_PATTERNS=^小马
# DINGTALK_HOME_CHANNEL=cidXXXX==
# DINGTALK_ALLOW_ALL_USERS=true
```

`~/.hermes/config.yaml`의 선택적 동작 설정입니다.

```yaml
group_sessions_per_user: true

gateway:
  platforms:
    dingtalk:
      extra:
        # Require @mention in groups before the bot replies (parity with Slack/Telegram/Discord).
        # DMs ignore this — the bot always replies in 1:1 chats.
        require_mention: true

        # Per-platform allowlist. When set, only these DingTalk user IDs can interact with the bot
        # (same semantics as DINGTALK_ALLOWED_USERS, but scoped here instead of in .env).
        allowed_users:
          - user-id-1
          - user-id-2
```

- `group_sessions_per_user: true`는 공유 그룹 채팅 안에서 각 참여자의 컨텍스트를 격리합니다.
- `require_mention: true`는 봇이 모든 그룹 메시지에 응답하지 않도록 하며, 누군가 봇을 @mention할 때만 답변합니다.
- `dingtalk.extra` 아래의 `allowed_users`는 `DINGTALK_ALLOWED_USERS`의 대안입니다. 둘 중 하나를 설정하세요(둘 다 설정하면 두 목록에 모두 있는 사용자만 인증됩니다).

### 게이트웨이 시작

구성한 후 DingTalk 게이트웨이를 시작합니다.

```bash
hermes gateway
```

봇은 몇 초 안에 DingTalk의 Stream Mode에 연결되어야 합니다. 봇이 추가된 DM이나 그룹에서 메시지를 보내 테스트합니다.

:::tip
지속적으로 실행하려면 `hermes gateway`를 백그라운드나 systemd 서비스로 실행할 수 있습니다. 자세한 내용은 배포 문서를 참고하세요.
:::

## 기능

### AI Card

Hermes는 일반 마크다운 메시지 대신 DingTalk AI Card로 답변할 수 있습니다. Card는 더 풍부하고 구조화된 표시를 제공하며, 에이전트가 답변을 생성하는 동안 스트리밍 업데이트를 지원합니다.

AI Card를 활성화하려면 `config.yaml`에 카드 템플릿 ID를 구성합니다.

```yaml
platforms:
  dingtalk:
    enabled: true
    extra:
      card_template_id: "your-card-template-id"
```

DingTalk Developer Console의 앱 AI Card 설정에서 카드 템플릿 ID를 확인할 수 있습니다. AI Card를 활성화하면 모든 답변이 스트리밍 텍스트 업데이트가 포함된 카드로 전송됩니다.

### 이모지 리액션

Hermes는 처리 상태를 표시하기 위해 메시지에 이모지 리액션을 자동으로 추가합니다.

- 🤔Thinking — 봇이 메시지 처리를 시작할 때 추가됩니다.
- 🥳Done — 답변이 완료될 때 추가되며 Thinking 리액션을 대체합니다.

이 리액션은 DM과 그룹 채팅 모두에서 작동합니다.

### 표시 설정

다른 플랫폼과 독립적으로 DingTalk의 표시 동작을 사용자 지정할 수 있습니다.

```yaml
display:
  platforms:
    dingtalk:
      show_reasoning: false   # Show model reasoning/thinking in replies
      streaming: true         # Enable streaming responses (works with AI Cards)
      tool_progress: all      # Show tool execution progress (all/new/off)
      interim_assistant_messages: true  # Show intermediate commentary messages
```

더 깔끔한 경험을 위해 도구 진행 상황과 중간 메시지를 비활성화하려면 다음과 같이 설정합니다.

```yaml
display:
  platforms:
    dingtalk:
      tool_progress: off
      interim_assistant_messages: false
```

## 문제 해결

### 봇이 메시지에 응답하지 않음

**원인**: Robot 기능이 활성화되지 않았거나 `DINGTALK_ALLOWED_USERS`에 사용자의 User ID가 포함되어 있지 않습니다.

**해결 방법**: 앱 설정에서 Robot 기능이 활성화되어 있고 Stream Mode가 선택되어 있는지 확인합니다. User ID가 `DINGTALK_ALLOWED_USERS`에 있는지 확인합니다. 게이트웨이를 다시 시작합니다.

### "dingtalk-stream not installed" 오류

**원인**: `dingtalk-stream` Python 패키지가 설치되지 않았습니다.

**해결 방법**: 다음을 설치합니다.

```bash
pip install dingtalk-stream httpx
```

### "DINGTALK_CLIENT_ID and DINGTALK_CLIENT_SECRET required"

**원인**: 환경이나 `.env` 파일에 자격 증명이 설정되지 않았습니다.

**해결 방법**: `~/.hermes/.env`에 `DINGTALK_CLIENT_ID`와 `DINGTALK_CLIENT_SECRET`이 올바르게 설정되어 있는지 확인합니다. Client ID는 AppKey이고 Client Secret은 DingTalk Developer Console의 AppSecret입니다.

### Stream 연결 끊김 / 재연결 반복

**원인**: 네트워크 불안정, DingTalk 플랫폼 유지 보수 또는 자격 증명 문제입니다.

**해결 방법**: 어댑터는 지수 백오프(2s → 5s → 10s → 30s → 60s)로 자동 재연결합니다. 자격 증명이 유효하고 앱이 비활성화되지 않았는지 확인합니다. 네트워크에서 아웃바운드 WebSocket 연결을 허용하는지 확인합니다.

### 봇이 오프라인 상태임

**원인**: Hermes 게이트웨이가 실행 중이 아니거나 연결에 실패했습니다.

**해결 방법**: `hermes gateway`가 실행 중인지 확인합니다. 터미널 출력에서 오류 메시지를 확인합니다. 일반적인 문제는 잘못된 자격 증명, 비활성화된 앱, `dingtalk-stream` 또는 `httpx` 미설치입니다.

### "No session_webhook available"

**원인**: 봇이 답변을 시도했지만 세션 웹훅 URL이 없습니다. 일반적으로 웹훅이 만료되었거나 메시지를 받은 후 답변을 보내기 전에 봇이 다시 시작되었을 때 발생합니다.

**해결 방법**: 봇에 새 메시지를 보냅니다. 수신 메시지마다 답변을 위한 새 세션 웹훅이 제공됩니다. 이는 정상적인 DingTalk 제한 사항이며, 봇은 최근에 수신한 메시지에만 답변할 수 있습니다.

## 보안

:::warning
봇과 상호작용할 수 있는 사용자를 제한하려면 항상 `DINGTALK_ALLOWED_USERS`를 설정하세요. 이 설정이 없으면 안전 조치로 게이트웨이가 기본적으로 모든 사용자를 거부합니다. 신뢰하는 사람의 User ID만 추가하세요. 인증된 사용자는 도구 사용 및 시스템 액세스를 포함해 에이전트 기능에 완전히 접근할 수 있습니다.
:::

Hermes Agent 배포 보안에 관한 자세한 내용은 [보안 가이드](../security.md)를 참고하세요.

## 참고 사항

- **Stream Mode**: 공개 URL, 도메인 이름 또는 웹훅 서버가 필요하지 않습니다. WebSocket을 통해 컴퓨터에서 연결을 시작하므로 NAT와 방화벽 뒤에서도 작동합니다.
- **AI Card**: 일반 마크다운 대신 선택적으로 풍부한 AI Card로 답변할 수 있습니다. `card_template_id`로 구성합니다.
- **이모지 리액션**: 처리 상태를 표시하기 위해 🤔Thinking/🥳Done 리액션이 자동으로 추가됩니다.
- **마크다운 응답**: 풍부한 텍스트 표시를 위해 DingTalk의 마크다운 형식으로 답변합니다.
- **미디어 지원**: 수신 메시지의 이미지와 파일은 자동으로 확인되며 비전 도구로 처리할 수 있습니다.
- **메시지 중복 제거**: 어댑터는 동일한 메시지가 두 번 처리되지 않도록 5분의 시간 범위로 중복을 제거합니다.
- **자동 재연결**: Stream 연결이 끊기면 어댑터가 지수 백오프로 자동 재연결합니다.
- **메시지 길이 제한**: 메시지당 응답은 20,000자로 제한됩니다. 더 긴 응답은 잘립니다.
