# BlueBubbles (iMessage)

[BlueBubbles](https://bluebubbles.app/)를 통해 Hermes를 Apple iMessage에 연결하세요. BlueBubbles는 iMessage를 모든 기기로 연결해 주는 무료 오픈 소스 macOS 서버입니다.

## 사전 요구 사항

- [BlueBubbles Server](https://bluebubbles.app/)를 실행하는 **Mac** (항상 켜져 있어야 함)
- 해당 Mac의 Messages.app에 로그인된 Apple ID
- BlueBubbles Server v1.0.0 이상 (웹훅에는 이 버전이 필요함)
- Hermes와 BlueBubbles 서버 간의 네트워크 연결

## 설정

### 1. BlueBubbles Server 설치

[bluebubbles.app](https://bluebubbles.app/)에서 다운로드하여 설치하세요. 설정 마법사를 완료하고 Apple ID로 로그인한 뒤 연결 방법(로컬 네트워크, Ngrok, Cloudflare 또는 Dynamic DNS)을 구성하세요.

### 2. 서버 URL 및 비밀번호 확인

BlueBubbles Server → **Settings → API**에서 다음을 확인하세요.
- **Server URL** (예: `http://192.168.1.10:1234`)
- **Server Password**

### 3. Hermes 구성

설정 마법사를 실행하세요.

```bash
hermes gateway setup
```

**BlueBubbles (iMessage)**를 선택하고 서버 URL과 비밀번호를 입력하세요.

또는 `~/.hermes/.env`에 환경 변수를 직접 설정하세요.

```bash
BLUEBUBBLES_SERVER_URL=http://192.168.1.10:1234
BLUEBUBBLES_PASSWORD=your-server-password
```

#### 그룹 채팅에서 멘션 필요 여부 (선택 사항)

기본적으로 Hermes는 인증된 모든 BlueBubbles/iMessage DM 또는 그룹 메시지에 응답합니다. 그룹 채팅을 옵트인 방식으로 바꾸려면 멘션 게이팅을 활성화하세요.

```yaml
platforms:
  bluebubbles:
    enabled: true
    extra:
      require_mention: true
```

`require_mention: true`이면 DM은 계속 정상적으로 작동하지만, 멘션 패턴과 일치하지 않는 그룹 채팅 메시지는 무시됩니다. 사용자 지정 패턴을 구성하지 않으면 Hermes는 `Hermes` 및 `@Hermes agent` 변형에 대한 보수적인 기본값을 사용합니다.

사용자 지정 에이전트 이름을 사용하려면 정규식 패턴을 설정하세요.

```yaml
platforms:
  bluebubbles:
    extra:
      require_mention: true
      mention_patterns:
        - '(?<![\w@])@?amos\b[,:\-]?'
```

### 4. 사용자 인증

다음 방법 중 하나를 선택하세요.

**DM 페어링 (권장):**
누군가 iMessage로 메시지를 보내면 Hermes가 자동으로 페어링 코드를 보냅니다. 다음 명령으로 승인하세요.
```bash
hermes pairing approve bluebubbles <CODE>
```
`hermes pairing list`를 사용하면 대기 중인 코드와 승인된 사용자를 볼 수 있습니다.

**특정 사용자를 사전 인증** (`~/.hermes/.env`에서):
```bash
BLUEBUBBLES_ALLOWED_USERS=user@icloud.com,+15551234567
```

**모든 사용자에게 공개** (`~/.hermes/.env`에서):
```bash
BLUEBUBBLES_ALLOW_ALL_USERS=true
```

### 5. 게이트웨이 시작

```bash
hermes gateway run
```

Hermes가 BlueBubbles 서버에 연결하고 웹훅을 등록한 뒤 iMessage 메시지 수신을 시작합니다.

## 작동 방식

```
iMessage → Messages.app → BlueBubbles Server → Webhook → Hermes
Hermes → BlueBubbles REST API → Messages.app → iMessage
```

- **수신:** 새 메시지가 도착하면 BlueBubbles가 로컬 리스너로 웹훅 이벤트를 보냅니다. 폴링이 없어 즉시 전달됩니다.
- **발신:** Hermes가 BlueBubbles REST API를 통해 메시지를 보냅니다.
- **미디어:** 이미지, 음성 메시지, 동영상 및 문서를 양방향으로 지원합니다. 수신 첨부 파일은 에이전트가 처리할 수 있도록 다운로드되어 로컬에 캐시됩니다.

## 환경 변수

| 변수 | 필수 | 기본값 | 설명 |
|----------|----------|---------|-------------|
| `BLUEBUBBLES_SERVER_URL` | 예 | — | BlueBubbles 서버 URL |
| `BLUEBUBBLES_PASSWORD` | 예 | — | 서버 비밀번호 |
| `BLUEBUBBLES_WEBHOOK_HOST` | 아니요 | `127.0.0.1` | 웹훅 리스너 바인드 주소 |
| `BLUEBUBBLES_WEBHOOK_PORT` | 아니요 | `8645` | 웹훅 리스너 포트 |
| `BLUEBUBBLES_WEBHOOK_PATH` | 아니요 | `/bluebubbles-webhook` | 웹훅 URL 경로 |
| `BLUEBUBBLES_HOME_CHANNEL` | 아니요 | — | cron 전달용 전화번호/이메일 |
| `BLUEBUBBLES_ALLOWED_USERS` | 아니요 | — | 쉼표로 구분한 인증된 사용자 |
| `BLUEBUBBLES_ALLOW_ALL_USERS` | 아니요 | `false` | 모든 사용자 허용 |
| `BLUEBUBBLES_REQUIRE_MENTION` | 아니요 | `false` | 그룹 채팅에서 응답하기 전에 멘션 패턴 필요 |
| `BLUEBUBBLES_MENTION_PATTERNS` | 아니요 | Hermes wake words | 그룹 멘션 일치에 사용할 JSON 배열, 줄바꿈 구분 또는 쉼표 구분 정규식 패턴 |

메시지를 읽음으로 자동 표시하는 기능은 `~/.hermes/config.yaml`의 `platforms.bluebubbles.extra` 아래 `send_read_receipts` 키로 제어합니다(기본값: `true`). 이에 대응하는 환경 변수는 없습니다.

## 기능

### 문자 메시지
 iMessage를 보내고 받습니다. 깔끔한 일반 텍스트로 전달되도록 마크다운은 자동으로 제거됩니다.

### 리치 미디어
- **이미지:** 사진이 iMessage 대화에 기본 형식으로 표시됩니다.
- **음성 메시지:** 오디오 파일이 iMessage 음성 메시지로 전송됩니다.
- **동영상:** 동영상 첨부 파일
- **문서:** 파일이 iMessage 첨부 파일로 전송됩니다.

### Tapback 반응
좋아요, 싫어요, 사랑해요, 웃겨요, 강조 및 질문 반응을 지원합니다. BlueBubbles [Private API helper](https://docs.bluebubbles.app/helper-bundle/installation)가 필요합니다.

### 입력 표시기
에이전트가 처리하는 동안 iMessage 대화에 "입력 중..."이 표시됩니다. Private API가 필요합니다.

### 읽음 확인
처리 후 메시지를 자동으로 읽음으로 표시합니다. Private API가 필요합니다.

### 채팅 주소 지정
이메일 또는 전화번호로 채팅을 지정할 수 있으며 Hermes가 이를 BlueBubbles 채팅 GUID로 자동 변환합니다. 원시 GUID 형식을 사용할 필요가 없습니다.

## Private API

일부 기능에는 BlueBubbles [Private API helper](https://docs.bluebubbles.app/helper-bundle/installation)가 필요합니다.
- Tapback 반응
- 입력 표시기
- 읽음 확인
- 주소로 새 채팅 만들기

Private API가 없어도 기본 문자 메시지와 미디어 기능은 작동합니다.

## 문제 해결

### "서버에 연결할 수 없음"
- 서버 URL이 올바르고 Mac이 켜져 있는지 확인하세요.
- BlueBubbles Server가 실행 중인지 확인하세요.
- 네트워크 연결을 확인하세요(방화벽, 포트 포워딩).

### 메시지가 도착하지 않음
- BlueBubbles Server → Settings → API → Webhooks에 웹훅이 등록되어 있는지 확인하세요.
- Mac에서 웹훅 URL에 접근할 수 있는지 확인하세요.
- 웹훅 오류는 `hermes logs gateway`에서 확인하세요(또는 `hermes logs -f`로 실시간 추적).

### "Private API helper가 연결되지 않음"
- Private API helper를 설치하세요: [docs.bluebubbles.app](https://docs.bluebubbles.app/helper-bundle/installation)
- 기본 메시징은 helper 없이도 작동합니다 — 반응, 입력 표시 및 읽음 확인에만 helper가 필요합니다.
