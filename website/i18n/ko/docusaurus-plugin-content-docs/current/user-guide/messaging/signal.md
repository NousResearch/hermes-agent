---
sidebar_position: 6
title: "Signal"
description: "signal-cli 데몬을 통해 Hermes Agent를 Signal 메신저 봇으로 설정하기"
---

# Signal 설정

Hermes는 HTTP 모드로 실행되는 [signal-cli](https://github.com/AsamK/signal-cli) 데몬을 통해 Signal에 연결됩니다. 어댑터는 SSE(Server-Sent Events)를 통해 메시지를 실시간으로 스트리밍하고 JSON-RPC를 통해 응답을 보냅니다.

Signal은 기본적으로 종단간 암호화를 지원하고, 오픈소스 프로토콜을 사용하며, 메타데이터 수집을 최소화하는 등 주류 메신저 중 가장 프라이버시 중심적인 메신저입니다. 보안에 민감한 에이전트 작업 흐름에 이상적입니다.

:::info 새로운 Python 의존성 없음
Signal 어댑터는 모든 통신에 (이미 핵심 Hermes 의존성인) `httpx`를 사용합니다. 추가적인 Python 패키지가 필요하지 않습니다. 외부에 signal-cli만 설치하면 됩니다.
:::

---

## 전제 조건

- **signal-cli** — Java 기반 Signal 클라이언트 ([GitHub](https://github.com/AsamK/signal-cli))
- **Java 17+** 런타임 — signal-cli 실행에 필요
- Signal이 설치된 **전화번호** (보조 기기로 연결하기 위함)

### signal-cli 설치

```bash
# macOS
brew install signal-cli

# Linux (최신 릴리스 다운로드)
VERSION=$(curl -Ls -o /dev/null -w %{url_effective} \
  https://github.com/AsamK/signal-cli/releases/latest | sed 's/^.*\/v//')
curl -L -O "https://github.com/AsamK/signal-cli/releases/download/v${VERSION}/signal-cli-${VERSION}.tar.gz"
sudo tar xf "signal-cli-${VERSION}.tar.gz" -C /opt
sudo ln -sf "/opt/signal-cli-${VERSION}/bin/signal-cli" /usr/local/bin/
```

:::caution
signal-cli는 apt나 snap 저장소에 **없습니다**. 위의 Linux 설치 방법은 [GitHub 릴리스](https://github.com/AsamK/signal-cli/releases)에서 직접 다운로드합니다.
:::

---

## 1단계: Signal 계정 연결하기

Signal-cli는 **연결된 기기(linked device)**로 작동합니다 — WhatsApp Web과 비슷하지만 Signal을 위한 것입니다. 휴대폰은 여전히 기본 기기로 유지됩니다.

```bash
# 연결 URI 생성 (QR 코드 또는 링크 표시)
signal-cli link -n "HermesAgent"
```

1. 휴대폰에서 **Signal** 열기
2. **Settings(설정) → Linked Devices(연결된 기기)**로 이동
3. **Link New Device(새 기기 연결)** 탭하기
4. QR 코드를 스캔하거나 URI 입력

---

## 2단계: signal-cli 데몬 시작하기

```bash
# +1234567890을 여러분의 Signal 전화번호로 교체 (E.164 형식)
signal-cli --account +1234567890 daemon --http 127.0.0.1:8080
```

:::tip
이 프로세스를 백그라운드에서 계속 실행되도록 하세요. `systemd`, `tmux`, `screen`을 사용하거나 서비스로 실행할 수 있습니다.
:::

실행 중인지 확인:

```bash
curl http://127.0.0.1:8080/api/v1/check
# 다음과 같이 반환되어야 함: {"versions":{"signal-cli":...}}
```

---

## 3단계: Hermes 구성

가장 쉬운 방법:

```bash
hermes gateway setup
```

플랫폼 메뉴에서 **Signal**을 선택합니다. 마법사는 다음을 수행합니다:

1. signal-cli가 설치되어 있는지 확인
2. HTTP URL 입력 프롬프트 (기본값: `http://127.0.0.1:8080`)
3. 데몬과의 연결성 테스트
4. 계정 전화번호 입력 요청
5. 허용된 사용자 및 접근 정책 구성

### 수동 구성

`~/.hermes/.env`에 추가:

```bash
# 필수
SIGNAL_HTTP_URL=http://127.0.0.1:8080
SIGNAL_ACCOUNT=+1234567890

# 보안 (권장)
SIGNAL_ALLOWED_USERS=+1234567890,+0987654321    # 쉼표로 구분된 E.164 번호 또는 UUID

# 선택 사항
SIGNAL_GROUP_ALLOWED_USERS=groupId1,groupId2     # 그룹 허용 (비활성화하려면 생략, 모두 허용은 *)
SIGNAL_HOME_CHANNEL=+1234567890                  # 크론 작업의 기본 전달 대상
```

그 후 게이트웨이 시작:

```bash
hermes gateway              # 포그라운드 실행
hermes gateway install      # 사용자 서비스로 설치
sudo hermes gateway install --system   # Linux 전용: 부팅 시 시스템 서비스로 실행
```

---

## 접근 제어

### DM 접근

DM 접근은 다른 모든 Hermes 플랫폼과 동일한 패턴을 따릅니다:

1. **`SIGNAL_ALLOWED_USERS`가 설정된 경우** → 해당 사용자만 메시지를 보낼 수 있음
2. **허용 목록(allowlist)이 설정되지 않은 경우** → 알 수 없는 사용자는 DM 페이링 코드를 받음 (`hermes pairing approve signal CODE`를 통해 승인)
3. **`SIGNAL_ALLOW_ALL_USERS=true`인 경우** → 누구나 메시지를 보낼 수 있음 (주의해서 사용)

### 그룹 접근

그룹 접근은 `SIGNAL_GROUP_ALLOWED_USERS` 환경 변수로 제어됩니다:

| 구성 | 동작 |
|---------------|----------|
| 설정 안 함 (기본값) | 모든 그룹 메시지는 무시됩니다. 봇은 DM에만 응답합니다. |
| 그룹 ID 지정 | 나열된 그룹만 모니터링됩니다 (예: `groupId1,groupId2`). |
| `*`로 설정 | 봇이 멤버로 있는 모든 그룹에서 응답합니다. |

---

## 기능

### 첨부 파일

어댑터는 양방향 미디어 전송 및 수신을 지원합니다.

**수신** (사용자 → 에이전트):

- **이미지** — PNG, JPEG, GIF, WebP (매직 바이트를 통해 자동 감지)
- **오디오** — MP3, OGG, WAV, M4A (Whisper가 설정된 경우 음성 메시지 텍스트 변환됨)
- **문서** — PDF, ZIP 및 기타 파일 형식

**발신** (에이전트 → 사용자):

에이전트는 응답에 `MEDIA:` 태그를 통해 미디어 파일을 보낼 수 있습니다. 다음 전달 방법이 지원됩니다:

- **이미지** — `send_multiple_images` 및 `send_image_file`은 PNG, JPEG, GIF, WebP를 네이티브 Signal 첨부 파일로 전송
- **음성** — `send_voice`는 오디오 파일(OGG, MP3, WAV, M4A, AAC)을 첨부 파일로 전송
- **비디오** — `send_video`는 MP4 비디오 파일을 전송
- **문서** — `send_document`는 모든 파일 형식(PDF, ZIP 등)을 전송

모든 아웃바운드 미디어는 Signal의 표준 첨부 파일 API를 통과합니다. 일부 플랫폼과 달리 Signal은 프로토콜 수준에서 음성 메시지와 파일 첨부를 구별하지 않습니다.

첨부 파일 크기 제한: **100 MB** (양방향 모두).
:::warning
**Signal 서버는 첨부 파일 업로드에 속도 제한을 둡니다**. 어댑터는 다중 이미지 전송 시 스케줄러를 사용하여 이미지를 32개씩 그룹화하고 Signal 서버 정책에 맞게 업로드 속도를 조절합니다.
:::

### 네이티브 포매팅, 답글 인용 및 리액션

Signal 메시지는 문자 그대로의 마크다운 문자 대신 **네이티브 포매팅**으로 렌더링됩니다. 어댑터는 마크다운(`**굵게**`, `*기울임*`, `` `코드` ``, `~~취소선~~`, `||스포일러||`, 제목)을 Signal `bodyRanges`로 변환하여, 받는 사람의 클라이언트에 `**`나 `` ` `` 문자가 표시되는 대신 실제 스타일이 적용되어 표시됩니다.

**답글 인용.** Hermes가 특정 메시지에 답장할 때, 원래 메시지를 인용하는 네이티브 답장을 게시합니다 — 이는 Signal 사용자가 스스로 "답장"을 사용할 때 보는 UI와 동일합니다. 인바운드 메시지에 대한 응답으로 생성된 답변에는 자동으로 적용됩니다.

**리액션.** 에이전트는 표준 리액션 API를 통해 메시지에 리액션을 추가할 수 있습니다. 리액션은 추가 텍스트가 아닌 참조된 메시지에 이모티콘 반응으로 Signal에 나타납니다.

이러한 기능은 추가 구성이 필요 없으며 — 최신 signal-cli 빌드에서 기본적으로 켜져 있습니다. `signal-cli` 버전이 너무 오래된 경우, Hermes는 일반 텍스트 전송으로 전환하고 1회성 경고를 로깅합니다.

### 타이핑 인디케이터

봇은 메시지를 처리하는 동안 입력 인디케이터를 전송하며, 8초마다 갱신됩니다.

### 전화번호 교정(Redaction)

모든 전화번호는 로그에서 자동으로 교정됩니다:
- `+15551234567` → `+155****4567`
- 이는 Hermes 게이트웨이 로그 및 전역 교정 시스템 모두에 적용됩니다.

### 나와의 채팅 (단일 번호 설정)

signal-cli를 (별도의 봇 번호가 아닌) 본인의 전화번호에 **연결된 보조 기기**로 실행하는 경우, Signal의 "나와의 채팅(Note to Self)" 기능을 통해 Hermes와 상호작용할 수 있습니다.

휴대폰에서 자신에게 메시지를 보내기만 하면 됩니다 — signal-cli가 이를 포착하고 Hermes가 동일한 대화방에 응답합니다.

**작동 방식:**
- "나와의 채팅" 메시지는 `syncMessage.sentMessage` 봉투로 도착합니다.
- 어댑터는 이 메시지가 봇 자신의 계정으로 주소 지정된 시점을 감지하고 일반 수신 메시지로 처리합니다.
- 자체 에코 방지(보낸 타임스탬프 추적)가 무한 루프를 막습니다 — 봇 자신의 답변은 자동으로 필터링됩니다.

**추가 구성 불필요.** `SIGNAL_ACCOUNT`가 본인의 전화번호와 일치하는 한 이는 자동으로 작동합니다.

### 상태 모니터링(Health Monitoring)

어댑터는 SSE 연결을 모니터링하고 다음의 경우 자동으로 재연결합니다:
- 연결이 끊어진 경우 (기하급수적 백오프 사용: 2초 → 60초)
- 120초 동안 활동이 감지되지 않은 경우 (signal-cli에 핑을 보내 확인)

---

## 문제 해결

| 문제 | 해결책 |
|---------|----------|
| 설정 중 **"Cannot reach signal-cli"** | signal-cli 데몬이 실행 중인지 확인하세요: `signal-cli --account +YOUR_NUMBER daemon --http 127.0.0.1:8080` |
| **메시지가 수신되지 않음** | `SIGNAL_ALLOWED_USERS`에 보낸 사람의 번호가 E.164 형식(`+` 접두사 포함)으로 포함되어 있는지 확인하세요 |
| **"signal-cli not found on PATH"** | signal-cli를 설치하고 PATH에 있는지 확인하거나 Docker를 사용하세요 |
| **연결이 계속 끊어짐** | signal-cli 로그에서 오류를 확인하세요. Java 17+가 설치되어 있는지 확인하세요. |
| **그룹 메시지가 무시됨** | 특정 그룹 ID 또는 모든 그룹을 허용하는 `*`로 `SIGNAL_GROUP_ALLOWED_USERS`를 구성하세요. |
| **봇이 아무에게도 응답하지 않음** | `SIGNAL_ALLOWED_USERS`를 구성하거나 DM 페어링을 사용하거나, 더 넓은 접근을 원하면 게이트웨이 정책을 통해 명시적으로 모든 사용자를 허용하세요. |
| **중복 메시지** | 전화번호에서 오직 하나의 signal-cli 인스턴스만 리스닝 중인지 확인하세요 |

---

## 보안

:::warning
**항상 접근 제어를 구성하세요.** 봇은 기본적으로 터미널 접근 권한이 있습니다. `SIGNAL_ALLOWED_USERS` 또는 DM 페어링이 없으면 게이트웨이는 안전 조치로 들어오는 모든 메시지를 거부합니다.
:::

- 전화번호는 모든 로그 출력에서 교정됩니다.
- 새로운 사용자를 안전하게 맞이하려면 DM 페어링이나 명시적인 허용 목록을 사용하세요.
- 그룹 지원이 꼭 필요한 경우가 아니라면 그룹을 비활성화한 상태로 유지하거나 신뢰할 수 있는 그룹만 허용 목록에 올리세요.
- Signal의 종단간 암호화는 전송 중인 메시지 콘텐츠를 보호합니다.
- `~/.local/share/signal-cli/`의 signal-cli 세션 데이터에는 계정 자격 증명이 포함되어 있습니다 — 비밀번호처럼 보호하세요.

---

## 환경 변수 참조

| 변수 | 필수 | 기본값 | 설명 |
|----------|----------|---------|-------------|
| `SIGNAL_HTTP_URL` | 예 | — | signal-cli HTTP 엔드포인트 |
| `SIGNAL_ACCOUNT` | 예 | — | 봇 전화번호 (E.164) |
| `SIGNAL_ALLOWED_USERS` | 아니오 | — | 쉼표로 구분된 전화번호/UUID 목록 |
| `SIGNAL_GROUP_ALLOWED_USERS` | 아니오 | — | 모니터링할 그룹 ID 목록 또는 모두를 위한 `*` (비활성화하려면 생략) |
| `SIGNAL_ALLOW_ALL_USERS` | 아니오 | `false` | 누구나 상호작용할 수 있도록 허용 (허용 목록 건너뜀) |
| `SIGNAL_HOME_CHANNEL` | 아니오 | — | 크론 작업의 기본 전송 대상 |
