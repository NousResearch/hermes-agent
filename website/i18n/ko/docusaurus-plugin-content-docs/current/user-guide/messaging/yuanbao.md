---
sidebar_position: 16
title: "Yuanbao"
description: "WebSocket 게이트웨이를 통해 Hermes Agent를 Yuanbao 엔터프라이즈 메시징 플랫폼에 연결합니다"
---

# Yuanbao

[Yuanbao](https://yuanbao.tencent.com/)는 Tencent의 엔터프라이즈 메시징 플랫폼입니다. Hermes를 Yuanbao에 연결하세요. 이 어댑터는 실시간 메시지 전달을 위해 WebSocket 게이트웨이를 사용하며, 1:1(C2C) 대화와 그룹 대화를 모두 지원합니다.

:::info
Yuanbao는 주로 Tencent 및 엔터프라이즈 환경에서 사용하는 엔터프라이즈 메시징 플랫폼입니다. 실시간 통신에 WebSocket을 사용하고, HMAC 기반 인증을 지원하며, 이미지, 파일, 음성 메시지 등의 리치 미디어를 지원합니다.
:::

## 사전 요구 사항

- 봇 생성 권한이 있는 Yuanbao 계정
- Yuanbao APP_ID 및 APP_SECRET (플랫폼 관리자로부터 발급)
- Python 패키지: `websockets` 및 `httpx`
- 미디어 지원: `aiofiles`

필요한 종속 항목을 설치하세요.

```bash
pip install websockets httpx aiofiles
```

## 설정

### 1. Yuanbao에서 봇 생성

1. [https://yuanbao.tencent.com/](https://yuanbao.tencent.com/)에서 Yuanbao 앱을 다운로드합니다.
2. 앱에서 **PAI → My Bot**으로 이동하여 새 봇을 생성합니다.
3. 봇이 생성되면 **APP_ID**와 **APP_SECRET**을 복사합니다.

### 2. 설정 마법사 실행

Yuanbao를 구성하는 가장 쉬운 방법은 대화형 설정을 사용하는 것입니다.

```bash
hermes gateway setup
```

메시지가 표시되면 **Yuanbao**를 선택합니다. 마법사가 다음 작업을 수행합니다.

1. APP_ID를 입력하라고 요청합니다.
2. APP_SECRET을 입력하라고 요청합니다.
3. 구성을 자동으로 저장합니다.

:::tip
WebSocket URL과 API Domain에는 합리적인 기본값이 내장되어 있습니다. 시작하려면 APP_ID와 APP_SECRET만 입력하면 됩니다.
:::

### 3. 환경 변수 구성

초기 설정 후 `~/.hermes/.env`에서 다음 변수를 확인하세요.

```bash
# Required
YUANBAO_APP_ID=your-app-id
YUANBAO_APP_SECRET=your-app-secret
YUANBAO_WS_URL=wss://api.yuanbao.example.com/ws
YUANBAO_API_DOMAIN=https://api.yuanbao.example.com

# Optional: bot account ID (normally obtained automatically from sign-token)
# YUANBAO_BOT_ID=your-bot-id

# Optional: internal routing environment (e.g. test/staging/production)
# YUANBAO_ROUTE_ENV=production

# Optional: home channel for cron/notifications (format: direct:<account> or group:<group_code>)
YUANBAO_HOME_CHANNEL=direct:bot_account_id
YUANBAO_HOME_CHANNEL_NAME="Bot Notifications"

# Optional: restrict access (legacy, see Access Control below for fine-grained policies)
YUANBAO_ALLOWED_USERS=user_account_1,user_account_2
```

### 4. 게이트웨이 시작

```bash
hermes gateway
```

어댑터가 Yuanbao WebSocket 게이트웨이에 연결되고, HMAC 서명을 사용해 인증한 뒤 메시지 처리를 시작합니다.

## 기능

- **WebSocket 게이트웨이** — 실시간 양방향 통신
- **HMAC 인증** — APP_ID/APP_SECRET을 사용한 안전한 요청 서명
- **C2C 메시징** — 사용자와 봇 간 1:1 대화
- **그룹 메시징** — 그룹 채팅에서의 대화
- **미디어 지원** — COS(Cloud Object Storage)를 통한 이미지, 파일, 음성 메시지
- **Markdown 서식** — Yuanbao의 크기 제한에 맞춰 메시지를 자동으로 분할
- **메시지 중복 제거** — 동일한 메시지의 중복 처리 방지
- **하트비트/연결 유지** — WebSocket 연결 안정성 유지
- **입력 중 표시** — 에이전트가 처리하는 동안 "입력 중…" 상태 표시
- **자동 재연결** — 지수 백오프를 사용한 WebSocket 연결 끊김 처리
- **그룹 정보 조회** — 그룹 세부 정보와 구성원 목록 조회
- **스티커/이모지 지원** — 대화에서 TIMFaceElem 스티커와 이모지 전송
- **WeChat 전달 대화 기록 지원** — 사용자가 WeChat 대화 기록 묶음을 Yuanbao로 전달하면 어댑터가 전달된 레코드(보낸 사람 닉네임, 텍스트, 중첩된 전달 내용을 포함한 멀티미디어 항목)를 디코딩하고 대화에 주입하여 에이전트가 전달된 전체 스레드를 읽을 수 있도록 함
- **자동 홈 설정** — 봇에게 처음 메시지를 보낸 사용자를 홈 채널 소유자로 자동 설정
- **느린 응답 알림** — 에이전트 응답이 예상보다 오래 걸릴 때 대기 메시지 전송

## 구성 옵션

### 채팅 ID 형식

Yuanbao는 대화 유형에 따라 접두사가 붙은 식별자를 사용합니다.

| 채팅 유형 | 형식 | 예시 |
|-----------|--------|---------|
| 1:1 메시지 (C2C) | `direct:<account>` | `direct:user123` |
| 그룹 메시지 | `group:<group_code>` | `group:grp456` |

### 미디어 업로드

Yuanbao 어댑터는 COS(Tencent Cloud Object Storage)를 통한 미디어 업로드를 자동으로 처리합니다.

- **이미지**: JPEG, PNG, GIF, WebP 지원
- **파일**: 일반적인 모든 문서 유형 지원
- **음성**: WAV, MP3, OGG 지원

SSRF 공격을 방지하기 위해 미디어 URL을 자동으로 검증하고 업로드 전에 다운로드합니다.

## 홈 채널

Yuanbao의 DM 또는 그룹 채팅에서 `/sethome` 명령을 사용하여 해당 채팅을 **홈 채널**로 지정합니다. 예약된 작업(cron 작업)의 결과가 이 채널로 전달됩니다.

:::tip Auto-sethome
홈 채널이 구성되지 않은 경우 봇에게 처음 메시지를 보낸 사용자가 자동으로 홈 채널 소유자로 설정됩니다. 현재 홈 채널이 그룹 채팅이면 첫 번째 DM을 통해 1:1 채널로 변경됩니다.
:::

`~/.hermes/.env`에서 수동으로 설정할 수도 있습니다.

```bash
YUANBAO_HOME_CHANNEL=direct:user_account_id
# or for a group:
# YUANBAO_HOME_CHANNEL=group:group_code
YUANBAO_HOME_CHANNEL_NAME="My Bot Updates"
```

### 예시: 홈 채널 설정

1. Yuanbao에서 봇과 대화를 시작합니다.
2. `/sethome` 명령을 보냅니다.
3. 봇이 다음과 같이 응답합니다: "홈 채널이 [chat_name]으로 설정되었으며 ID는 [chat_id]입니다. Cron 작업이 이 위치로 결과를 전달합니다."
4. 이후 cron 작업과 알림이 이 채널로 전송됩니다.

### 예시: Cron 작업 전달

cron 작업을 생성합니다.

```bash
/cron "0 9 * * *" Check server status
```

예약된 출력은 매일 오전 9시에 Yuanbao 홈 채널로 전달됩니다.

## 사용 팁

### 대화 시작

Yuanbao에서 봇에게 아무 메시지나 보내세요.

```
hello
```

봇이 같은 대화 스레드에서 응답합니다.

### 사용 가능한 명령

Yuanbao에서는 모든 표준 Hermes 명령을 사용할 수 있습니다.

| 명령 | 설명 |
|---------|-------------|
| `/new` | 새 대화 시작 |
| `/model [provider:model]` | 모델 표시 또는 변경 |
| `/sethome` | 이 채팅을 홈 채널로 설정 |
| `/status` | 세션 정보 표시 |
| `/help` | 사용 가능한 명령 표시 |

### 파일 보내기

파일을 봇에게 보내려면 Yuanbao 채팅에 파일을 직접 첨부하세요. 봇이 파일 첨부를 자동으로 다운로드하고 처리합니다.

첨부 파일과 함께 메시지를 보낼 수도 있습니다.

```
Please analyze this document
```

### 파일 받기

봇에게 파일을 만들거나 내보내도록 요청하면 파일을 Yuanbao 채팅으로 직접 보냅니다.

## 문제 해결

### 봇은 온라인이지만 메시지에 응답하지 않음

**원인**: WebSocket 핸드셰이크 중 인증에 실패했습니다.

**해결 방법**:
1. APP_ID와 APP_SECRET이 올바른지 확인합니다.
2. WebSocket URL에 접근할 수 있는지 확인합니다.
3. 봇 계정에 적절한 권한이 있는지 확인합니다.
4. 게이트웨이 로그를 확인합니다: `tail -f ~/.hermes/logs/gateway.log`

### "Connection refused" 오류

**원인**: WebSocket URL에 연결할 수 없거나 올바르지 않습니다.

**해결 방법**:
1. WebSocket URL 형식을 확인합니다(`wss://`로 시작해야 함).
2. Yuanbao API 도메인에 대한 네트워크 연결을 확인합니다.
3. 방화벽이 WebSocket 연결을 허용하는지 확인합니다.
4. 다음 명령으로 URL을 테스트합니다: `curl -I https://[YUANBAO_API_DOMAIN]`

### 미디어 업로드 실패

**원인**: COS 자격 증명이 유효하지 않거나 미디어 서버에 연결할 수 없습니다.

**해결 방법**:
1. API_DOMAIN이 올바른지 확인합니다.
2. 봇에 미디어 업로드 권한이 활성화되어 있는지 확인합니다.
3. 미디어 파일에 접근할 수 있고 손상되지 않았는지 확인합니다.
4. 플랫폼 관리자와 함께 COS 버킷 구성을 확인합니다.

### 홈 채널로 메시지가 전달되지 않음

**원인**: 홈 채널 ID 형식이 올바르지 않거나 cron 작업이 실행되지 않았습니다.

**해결 방법**:
1. YUANBAO_HOME_CHANNEL이 올바른 형식인지 확인합니다.
2. `/sethome` 명령으로 올바른 형식을 자동 감지하도록 테스트합니다.
3. `/status`로 cron 작업 일정을 확인합니다.
4. 대상 채팅에서 봇에 전송 권한이 있는지 확인합니다.

### 연결이 자주 끊김

**원인**: WebSocket 연결이 불안정하거나 네트워크가 안정적이지 않습니다.

**해결 방법**:
1. 오류 패턴을 확인하려면 게이트웨이 로그를 확인합니다.
2. 연결 설정에서 하트비트 타임아웃을 늘립니다.
3. Yuanbao API에 안정적으로 연결되는 네트워크인지 확인합니다.
4. 자세한 로깅을 활성화하는 것을 고려합니다: `hermes gateway run -vv`

## 액세스 제어

Yuanbao는 DM과 그룹 대화 모두에 대해 세밀한 액세스 제어를 지원합니다.

```bash
# DM policy: open (default) | allowlist | disabled
YUANBAO_DM_POLICY=open
# Comma-separated user IDs allowed to DM the bot (only used when DM_POLICY=allowlist)
YUANBAO_DM_ALLOW_FROM=user_id_1,user_id_2

# Group policy: open (default) | allowlist | disabled
YUANBAO_GROUP_POLICY=open
# Comma-separated group codes allowed (only used when GROUP_POLICY=allowlist)
YUANBAO_GROUP_ALLOW_FROM=group_code_1,group_code_2
```

다음과 같이 `config.yaml`에서도 설정할 수 있습니다.

```yaml
platforms:
  yuanbao:
    extra:
      dm_policy: allowlist
      dm_allow_from: "user1,user2"
      group_policy: open
      group_allow_from: ""
```

## 고급 구성

### 메시지 분할

Yuanbao에는 최대 메시지 크기가 있습니다. Hermes는 Markdown을 인식하여 긴 응답을 자동으로 분할합니다(코드 펜스, 표, 문단 경계를 존중함).

### 연결 매개변수

다음 연결 매개변수는 합리적인 기본값과 함께 어댑터에 내장되어 있습니다.

| 매개변수 | 기본값 | 설명 |
|-----------|-------------|-------------|
| WebSocket 연결 타임아웃 | 15초 | WS 핸드셰이크를 기다리는 시간 |
| 하트비트 간격 | 30초 | 연결 유지를 위한 ping 주기 |
| 최대 재연결 시도 횟수 | 100 | 재연결을 시도하는 최대 횟수 |
| 재연결 백오프 | 1초 → 60초 (지수 증가) | 재연결 시도 사이의 대기 시간 |
| 응답 하트비트 간격 | 2초 | RUNNING 상태 전송 주기 |
| 전송 타임아웃 | 30초 | 아웃바운드 WS 메시지의 타임아웃 |

:::note
현재 이 값들은 환경 변수를 통해 구성할 수 없습니다. 일반적인 Yuanbao 배포에 맞게 최적화되어 있습니다.
:::

### 상세 로깅

연결 문제를 해결하려면 디버그 로깅을 활성화하세요.

```bash
hermes gateway run -vv
```

## 다른 기능과의 통합

### Cron 작업

Yuanbao에서 실행되는 작업을 예약합니다.

```
/cron "0 */4 * * *" Report system health
```

결과는 홈 채널로 전달됩니다.

### 백그라운드 작업

대화를 차단하지 않고 오래 걸리는 작업을 실행합니다.

```
/background Analyze all files in the archive
```

### 플랫폼 간 메시지

CLI에서 Yuanbao로 메시지를 보냅니다.

```bash
hermes chat -q "Send 'Hello from CLI' to yuanbao:group:group_code"
```

## 관련 문서

- [메시징 게이트웨이 개요](./index.md)
- [슬래시 명령 참고](/reference/slash-commands)
- [Cron 작업](/user-guide/features/cron)
- [백그라운드 세션](/user-guide/cli#background-sessions)
