# BlueBubbles (iMessage)

[BlueBubbles](https://bluebubbles.app/)를 통해 Hermes를 Apple iMessage에 연결하세요. BlueBubbles는 iMessage를 모든 장치로 연결해 주는 무료 오픈소스 macOS 서버입니다.

## 사전 요구 사항

- [BlueBubbles Server](https://bluebubbles.app/)가 실행 중인 (항상 켜져 있는) **Mac**
- 해당 Mac의 메시지 앱(Messages.app)에 로그인된 Apple ID
- BlueBubbles Server v1.0.0+ (웹훅을 사용하려면 이 버전이 필요합니다)
- Hermes와 BlueBubbles 서버 간의 네트워크 연결

## 설정

### 1. BlueBubbles Server 설치

[bluebubbles.app](https://bluebubbles.app/)에서 다운로드하여 설치합니다. 설정 마법사를 완료하세요 — Apple ID로 로그인하고 연결 방법(로컬 네트워크, Ngrok, Cloudflare 또는 Dynamic DNS)을 구성합니다.

### 2. 서버 URL 및 비밀번호 가져오기

BlueBubbles Server의 **Settings → API**에서 다음 사항을 기록해 둡니다:
- **Server URL** (예: `http://192.168.1.10:1234`)
- **Server Password**

### 3. Hermes 구성

설정 마법사를 실행합니다:

```bash
hermes gateway setup
```

**BlueBubbles (iMessage)**를 선택하고 서버 URL과 비밀번호를 입력합니다.

또는 `~/.hermes/.env` 파일에 환경 변수를 직접 설정합니다:

```bash
BLUEBUBBLES_SERVER_URL=http://192.168.1.10:1234
BLUEBUBBLES_PASSWORD=your-server-password
```

#### 선택 사항: 그룹 채팅에서 멘션 요구하기

기본적으로 Hermes는 승인된 BlueBubbles/iMessage의 모든 DM 또는 그룹 메시지에 응답합니다. 그룹 채팅을 선택적으로 활성화하려면 멘션 기능을 활성화하세요:

```yaml
platforms:
  bluebubbles:
    enabled: true
    extra:
      require_mention: true
```

`require_mention: true`로 설정하면 DM은 정상적으로 작동하지만, 멘션 패턴이 일치하지 않는 한 그룹 채팅 메시지는 무시됩니다. 사용자 지정 패턴을 설정하지 않으면, Hermes는 `Hermes`와 `@Hermes agent` 변형 형태에 대해 보수적인 기본값을 사용합니다.

사용자 지정 에이전트 이름의 경우 정규식 패턴을 설정하세요:

```yaml
platforms:
  bluebubbles:
    extra:
      require_mention: true
      mention_patterns:
        - '(?<![\w@])@?amos\b[,:\-]?'
```

### 4. 사용자 권한 부여

다음 중 한 가지 방식을 선택하세요:

**DM 페어링 (권장):**
누군가 당신의 iMessage로 메시지를 보내면, Hermes는 자동으로 페어링 코드를 전송합니다. 다음 명령으로 이를 승인하세요:
```bash
hermes pairing approve bluebubbles <CODE>
```
대기 중인 코드와 승인된 사용자를 보려면 `hermes pairing list`를 사용하세요.

**특정 사용자 사전 승인** (`~/.hermes/.env`에서 설정):
```bash
BLUEBUBBLES_ALLOWED_USERS=user@icloud.com,+15551234567
```

**모든 접근 허용** (`~/.hermes/.env`에서 설정):
```bash
BLUEBUBBLES_ALLOW_ALL_USERS=true
```

### 5. 게이트웨이 시작

```bash
hermes gateway run
```

Hermes는 BlueBubbles 서버에 연결하고 웹훅을 등록한 후 iMessage 메시지 수신 대기를 시작합니다.

## 작동 방식

```
iMessage → Messages.app → BlueBubbles Server → 웹훅 → Hermes
Hermes → BlueBubbles REST API → Messages.app → iMessage
```

- **수신(Inbound):** BlueBubbles는 새 메시지가 도착하면 로컬 수신기로 웹훅 이벤트를 보냅니다. 폴링 없이 즉시 전송됩니다.
- **발신(Outbound):** Hermes는 BlueBubbles REST API를 통해 메시지를 보냅니다.
- **미디어(Media):** 이미지, 음성 메시지, 동영상, 문서는 양방향으로 지원됩니다. 수신된 첨부 파일은 에이전트가 처리할 수 있도록 로컬에 다운로드되어 캐시됩니다.

## 환경 변수

| 변수 | 필수 | 기본값 | 설명 |
|----------|----------|---------|-------------|
| `BLUEBUBBLES_SERVER_URL` | Yes | — | BlueBubbles 서버 URL |
| `BLUEBUBBLES_PASSWORD` | Yes | — | 서버 비밀번호 |
| `BLUEBUBBLES_WEBHOOK_HOST` | No | `127.0.0.1` | 웹훅 수신기 바인드 주소 |
| `BLUEBUBBLES_WEBHOOK_PORT` | No | `8645` | 웹훅 수신기 포트 |
| `BLUEBUBBLES_WEBHOOK_PATH` | No | `/bluebubbles-webhook` | 웹훅 URL 경로 |
| `BLUEBUBBLES_HOME_CHANNEL` | No | — | cron 전송을 위한 전화번호/이메일 |
| `BLUEBUBBLES_ALLOWED_USERS` | No | — | 쉼표로 구분된 승인된 사용자 |
| `BLUEBUBBLES_ALLOW_ALL_USERS` | No | `false` | 모든 사용자 허용 |
| `BLUEBUBBLES_REQUIRE_MENTION` | No | `false` | 그룹 채팅에서 응답하기 전 멘션 패턴 요구 |
| `BLUEBUBBLES_MENTION_PATTERNS` | No | Hermes wake words | 그룹 멘션 일치를 위한 JSON 배열, 개행 또는 쉼표로 구분된 정규식 패턴 |

자동으로 메시지를 읽음 처리하는 기능은 `~/.hermes/config.yaml` 의 `platforms.bluebubbles.extra` 하위에 있는 `send_read_receipts` 키를 통해 제어됩니다 (기본값: `true`). 해당 기능을 제어하는 환경 변수는 없습니다.

## 기능

### 문자 메시지
iMessage를 주고받습니다. 깔끔한 일반 텍스트 전송을 위해 마크다운은 자동으로 제거됩니다.

### 리치 미디어
- **이미지:** 사진이 iMessage 대화방에 기본형으로 표시됩니다
- **음성 메시지:** 오디오 파일이 iMessage 음성 메시지로 전송됩니다
- **동영상:** 동영상 첨부
- **문서:** iMessage 첨부 파일로 전송되는 파일

### 탭백 반응 (Tapback Reactions)
하트, 좋아요, 싫어요, 웃음, 강조, 물음표 반응을 보냅니다. BlueBubbles [Private API helper](https://docs.bluebubbles.app/helper-bundle/installation)가 필요합니다.

### 입력 중 표시 (Typing Indicators)
에이전트가 처리하는 동안 iMessage 대화방에 "입력 중..."을 표시합니다. Private API가 필요합니다.

### 읽음 확인 (Read Receipts)
처리 후 자동으로 메시지를 읽음으로 표시합니다. Private API가 필요합니다.

### 채팅 주소 지정
이메일이나 전화번호로 채팅을 지정할 수 있습니다. Hermes는 이들을 BlueBubbles 채팅 GUID로 자동 변환합니다. 변환되지 않은 GUID 형식을 직접 사용할 필요는 없습니다.

## Private API

일부 기능은 BlueBubbles [Private API helper](https://docs.bluebubbles.app/helper-bundle/installation)를 필요로 합니다:
- 탭백 반응
- 입력 중 표시
- 읽음 확인
- 주소를 통한 새 채팅 만들기

Private API가 없더라도 기본 문자 메시지와 미디어 전송은 작동합니다.

## 문제 해결

### "서버에 접근할 수 없습니다 (Cannot reach server)"
- 서버 URL이 올바르고 Mac이 켜져 있는지 확인하세요.
- BlueBubbles Server가 실행 중인지 확인하세요.
- 네트워크 연결(방화벽, 포트 포워딩)을 확인하세요.

### 메시지가 도착하지 않음
- BlueBubbles Server → Settings → API → Webhooks에서 웹훅이 등록되어 있는지 확인하세요.
- Mac에서 웹훅 URL에 접근할 수 있는지 확인하세요.
- `hermes logs gateway`를 통해 웹훅 오류를 확인하세요 (또는 실시간으로 확인하려면 `hermes logs -f` 사용).

### "Private API helper가 연결되지 않았습니다 (Private API helper not connected)"
- Private API helper를 설치하세요: [docs.bluebubbles.app](https://docs.bluebubbles.app/helper-bundle/installation)
- 이를 설치하지 않아도 기본 메시징은 작동합니다 — 오직 반응, 입력 중 표시, 읽음 확인에만 이 기능이 필요합니다.
