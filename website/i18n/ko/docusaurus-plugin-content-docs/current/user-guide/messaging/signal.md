---
sidebar_position: 6
title: "Signal"
description: "signal-cli 데몬을 통해 Hermes Agent를 Signal 메신저 봇으로 설정하기"
---

# Signal 설정

Hermes는 HTTP 모드로 실행되는 [signal-cli](https://github.com/AsamK/signal-cli) 데몬을 통해 Signal에 연결합니다. 어댑터는 SSE(Server-Sent Events)를 통해 실시간으로 메시지를 스트리밍하고 JSON-RPC를 통해 응답을 보냅니다.

Signal은 주류 메신저 중 개인정보 보호 수준이 가장 높은 서비스입니다 — 기본적으로 종단간 암호화가 적용되고, 오픈 소스 프로토콜을 사용하며, 수집하는 메타데이터가 최소화되어 있습니다. 따라서 보안에 민감한 에이전트 워크플로에 적합합니다.

:::info 새로운 Python 종속성 없음
Signal 어댑터는 모든 통신에 `httpx`(이미 Hermes의 핵심 종속성)를 사용합니다. 추가 Python 패키지는 필요하지 않습니다. 외부에 signal-cli만 설치하면 됩니다.
:::

---

## 사전 요구 사항

- **signal-cli** — Java 기반 Signal 클라이언트([GitHub](https://github.com/AsamK/signal-cli))
- **Java 17+** 런타임 — signal-cli에 필요
- **Signal이 설치된 전화번호** — 보조 기기로 연결할 때 사용

### signal-cli 설치

```bash
# macOS
brew install signal-cli

# Linux (download latest release)
VERSION=$(curl -Ls -o /dev/null -w %{url_effective} \
  https://github.com/AsamK/signal-cli/releases/latest | sed 's/^.*\/v//')
curl -L -O "https://github.com/AsamK/signal-cli/releases/download/v${VERSION}/signal-cli-${VERSION}.tar.gz"
sudo tar xf "signal-cli-${VERSION}.tar.gz" -C /opt
sudo ln -sf "/opt/signal-cli-${VERSION}/bin/signal-cli" /usr/local/bin/
```

:::caution
signal-cli는 apt 또는 snap 저장소에 **포함되어 있지 않습니다**. 위의 Linux 설치 방법은 [GitHub 릴리스](https://github.com/AsamK/signal-cli/releases)에서 직접 다운로드합니다.
:::

---

## 1단계: Signal 계정 연결

Signal-cli는 **연결된 기기**로 작동합니다 — Signal용 WhatsApp Web과 비슷합니다. 사용자의 전화가 기본 기기로 유지됩니다.

```bash
# Generate a linking URI (displays a QR code or link)
signal-cli link -n "HermesAgent"
```

1. 휴대전화에서 **Signal**을 엽니다
2. **설정 → 연결된 기기**로 이동합니다
3. **새 기기 연결**을 탭합니다
4. QR 코드를 스캔하거나 URI를 입력합니다

---

## 2단계: signal-cli 데몬 시작

```bash
# Replace +1234567890 with your Signal phone number (E.164 format)
signal-cli --account +1234567890 daemon --http 127.0.0.1:8080
```

:::tip
백그라운드에서 계속 실행하세요. `systemd`, `tmux`, `screen`을 사용하거나 서비스로 실행할 수 있습니다.
:::

실행 중인지 확인합니다.

```bash
curl http://127.0.0.1:8080/api/v1/check
# Should return: {"versions":{"signal-cli":...}}
```

---

## 3단계: Hermes 구성

가장 쉬운 방법은 다음과 같습니다.

```bash
hermes gateway setup
```

플랫폼 메뉴에서 **Signal**을 선택합니다. 마법사가 다음 작업을 수행합니다.

1. signal-cli가 설치되어 있는지 확인
2. HTTP URL 요청(기본값: `http://127.0.0.1:8080`)
3. 데몬 연결 테스트
4. 계정 전화번호 요청
5. 허용된 사용자 및 접근 정책 구성

### 수동 구성

`~/.hermes/.env`에 다음을 추가합니다.

```bash
# Required
SIGNAL_HTTP_URL=http://127.0.0.1:8080
SIGNAL_ACCOUNT=+1234567890

# Security (recommended)
SIGNAL_ALLOWED_USERS=+1234567890,+0987654321    # Comma-separated E.164 numbers or UUIDs

# Optional
SIGNAL_GROUP_ALLOWED_USERS=groupId1,groupId2     # Enable groups (omit to disable, * for all)
SIGNAL_HOME_CHANNEL=+1234567890                  # Default delivery target for cron jobs
```

그런 다음 게이트웨이를 시작합니다.

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

---

## 접근 제어

### DM 접근

DM 접근은 다른 모든 Hermes 플랫폼과 같은 방식을 따릅니다.

1. **`SIGNAL_ALLOWED_USERS`가 설정됨** → 해당 사용자만 메시지를 보낼 수 있습니다
2. **허용 목록이 설정되지 않음** → 알 수 없는 사용자에게 DM 페어링 코드가 전송됩니다(`hermes pairing approve signal CODE`로 승인)
3. **`SIGNAL_ALLOW_ALL_USERS=true`** → 누구나 메시지를 보낼 수 있습니다(주의해서 사용하세요)

### 그룹 접근

그룹 접근은 `SIGNAL_GROUP_ALLOWED_USERS` 환경 변수로 제어합니다.

| 구성 | 동작 |
|---------------|----------|
| 설정하지 않음(기본값) | 모든 그룹 메시지를 무시합니다. 봇은 DM에만 응답합니다. |
| 그룹 ID로 설정 | 나열된 그룹만 모니터링합니다(예: `groupId1,groupId2`). |
| `*`로 설정 | 봇이 멤버로 속한 모든 그룹에 응답합니다. |

---

## 기능

### 첨부 파일

어댑터는 양방향으로 미디어를 보내고 받을 수 있습니다.

**수신**(사용자 → 에이전트):

- **이미지** — PNG, JPEG, GIF, WebP(매직 바이트로 자동 감지)
- **오디오** — MP3, OGG, WAV, M4A(Whisper가 구성된 경우 음성 메시지를 텍스트로 변환)
- **문서** — PDF, ZIP 및 기타 파일 형식

**발신**(에이전트 → 사용자):

에이전트는 응답에 `MEDIA:` 태그를 사용해 미디어 파일을 보낼 수 있습니다. 다음 전달 방법이 지원됩니다.

- **이미지** — `send_multiple_images`와 `send_image_file`이 PNG, JPEG, GIF, WebP를 기본 Signal 첨부 파일로 전송
- **음성** — `send_voice`가 오디오 파일(OGG, MP3, WAV, M4A, AAC)을 첨부 파일로 전송
- **동영상** — `send_video`가 MP4 동영상 파일을 전송
- **문서** — `send_document`가 모든 파일 형식(PDF, ZIP 등)을 전송

모든 발신 미디어는 Signal의 표준 첨부 파일 API를 거칩니다. 일부 플랫폼과 달리 Signal은 프로토콜 수준에서 음성 메시지와 파일 첨부 파일을 구분하지 않습니다.

첨부 파일 크기 제한: **100MB**(양방향).
:::warning
**Signal 서버는 첨부 파일 업로드를 제한합니다.** 어댑터는 여러 이미지를 보낼 때 스케줄러를 사용해 이미지를 32개 단위로 묶고, Signal 서버 정책에 맞게 업로드 속도를 제한합니다.
:::

### 기본 서식, 답장 인용 및 리액션

Signal 메시지는 일반적인 마크다운 문자를 그대로 표시하는 대신 **기본 서식**으로 렌더링됩니다. 어댑터는 마크다운(`**굵게**`, `*기울임*`, `` `코드` ``, `~~취소선~~`, `||스포일러||`, 제목)을 Signal `bodyRanges`로 변환하므로, 수신자 클라이언트에서 보이는 `**` / `` ` `` 문자가 아니라 실제 서식으로 표시됩니다.

**답장 인용.** Hermes가 특정 메시지에 답장하면 원본을 인용하는 기본 답장을 게시합니다 — Signal 사용자가 직접 "답장"을 사용할 때 보게 되는 것과 같은 UI 기능입니다. 수신 메시지에 대한 응답으로 생성된 답장에는 이 기능이 자동으로 적용됩니다.

**리액션.** 에이전트는 표준 리액션 API를 통해 메시지에 반응할 수 있습니다. 리액션은 추가 텍스트가 아니라 참조된 메시지에 이모지 리액션으로 Signal에 표시됩니다.

이 기능에는 추가 구성이 필요하지 않습니다 — 최신 signal-cli 빌드에서는 기본으로 제공됩니다. `signal-cli` 버전이 너무 오래된 경우 Hermes는 일반 텍스트 전달로 대체하고 한 번만 경고를 기록합니다.

### 입력 중 표시

봇은 메시지를 처리하는 동안 입력 중 표시를 보내며, 8초마다 갱신합니다.

### 도구 진행 상황 표시

Signal은 이미 전송된 메시지의 편집을 지원하지 않습니다. 따라서 `/verbose`가 활성화되어 있고 플랫폼에 `off`가 아닌 모드가 저장되어 있더라도 Hermes는 Signal에서 게이트웨이 도구 진행 상황 말풍선을 숨깁니다.

CLI에서는 여전히 도구 활동을 볼 수 있으며, 최종 Signal 답장에는 일반적인 어시스턴트 출력이 포함될 수 있습니다. 채팅에서 도구별 실시간 진행 상황이 필요하다면 메시지 편집을 지원하는 메시징 플랫폼을 사용하세요.

### 전화번호 비식별화

모든 전화번호는 로그에서 자동으로 비식별화됩니다.
- `+15551234567` → `+155****4567`
- 이 기능은 Hermes 게이트웨이 로그와 전역 비식별화 시스템 모두에 적용됩니다

### 나에게 보내기(단일 번호 설정)

signal-cli를 별도의 봇 번호가 아닌 자신의 전화번호에 **연결된 보조 기기**로 실행하는 경우 Signal의 "나에게 보내기" 기능으로 Hermes와 상호작용할 수 있습니다.

휴대전화에서 자신에게 메시지를 보내기만 하면 됩니다 — signal-cli가 메시지를 수신하고 Hermes가 같은 대화에서 응답합니다.

**작동 방식:**
- "나에게 보내기" 메시지는 `syncMessage.sentMessage` envelope로 도착합니다
- 어댑터는 이 메시지가 봇 자신의 계정으로 전송되었는지 감지하고 일반적인 수신 메시지로 처리합니다
- 전송 타임스탬프 추적을 통한 에코 방지 기능이 무한 루프를 막습니다 — 봇 자신의 응답은 자동으로 필터링됩니다

**추가 구성이 필요하지 않습니다.** `SIGNAL_ACCOUNT`가 전화번호와 일치하기만 하면 자동으로 작동합니다.

### 상태 모니터링

어댑터는 SSE 연결을 모니터링하고 다음과 같은 경우 자동으로 다시 연결합니다.
- 연결이 끊긴 경우(지수 백오프: 2초 → 60초)
- 120초 동안 활동이 감지되지 않은 경우(signal-cli를 ping하여 확인)

---

## 문제 해결

| 문제 | 해결 방법 |
|---------|----------|
| 설정 중 **"signal-cli에 연결할 수 없음"** | signal-cli 데몬이 실행 중인지 확인합니다: `signal-cli --account +YOUR_NUMBER daemon --http 127.0.0.1:8080` |
| **메시지를 받지 못함** | `SIGNAL_ALLOWED_USERS`에 E.164 형식(`+` 접두사 포함)의 발신자 번호가 포함되어 있는지 확인합니다 |
| **"signal-cli가 PATH에 없음"** | signal-cli를 설치하고 PATH에 포함되어 있는지 확인하거나 Docker를 사용합니다 |
| **연결이 계속 끊김** | signal-cli 로그에서 오류를 확인합니다. Java 17+가 설치되어 있는지 확인합니다. |
| **그룹 메시지가 무시됨** | 특정 그룹 ID 또는 모든 그룹을 허용하는 `*`로 `SIGNAL_GROUP_ALLOWED_USERS`를 구성합니다. |
| **봇이 누구에게도 응답하지 않음** | `SIGNAL_ALLOWED_USERS`를 구성하거나, DM 페어링을 사용하거나, 더 넓은 접근을 원한다면 게이트웨이 정책에서 모든 사용자를 명시적으로 허용합니다. |
| **메시지 중복** | 전화번호를 수신하는 signal-cli 인스턴스가 하나만 실행 중인지 확인합니다 |

---

## 보안

:::warning
**항상 접근 제어를 구성하세요.** 봇은 기본적으로 터미널 접근 권한을 가집니다. `SIGNAL_ALLOWED_USERS` 또는 DM 페어링이 없으면 안전 조치로 게이트웨이가 모든 수신 메시지를 거부합니다.
:::

- 모든 로그 출력에서 전화번호가 비식별화됩니다
- 신규 사용자를 안전하게 등록하려면 DM 페어링 또는 명시적인 허용 목록을 사용합니다
- 그룹 기능이 꼭 필요할 때만 그룹을 활성화하고, 신뢰하는 그룹만 허용 목록에 추가합니다
- Signal의 종단간 암호화가 전송 중인 메시지 내용을 보호합니다
- `~/.local/share/signal-cli/`의 signal-cli 세션 데이터에는 계정 자격 증명이 포함되어 있으므로 비밀번호처럼 보호해야 합니다

---

## 환경 변수 참조

| 변수 | 필수 | 기본값 | 설명 |
|----------|----------|---------|-------------|
| `SIGNAL_HTTP_URL` | 예 | — | signal-cli HTTP 엔드포인트 |
| `SIGNAL_ACCOUNT` | 예 | — | 봇 전화번호(E.164) |
| `SIGNAL_ALLOWED_USERS` | 아니요 | — | 쉼표로 구분한 전화번호/UUID |
| `SIGNAL_GROUP_ALLOWED_USERS` | 아니요 | — | 모니터링할 그룹 ID 또는 모든 그룹을 뜻하는 `*`(그룹을 비활성화하려면 생략) |
| `SIGNAL_ALLOW_ALL_USERS` | 아니요 | `false` | 모든 사용자의 상호작용 허용(허용 목록 생략) |
| `SIGNAL_HOME_CHANNEL` | 아니요 | — | cron 작업의 기본 전달 대상 |
