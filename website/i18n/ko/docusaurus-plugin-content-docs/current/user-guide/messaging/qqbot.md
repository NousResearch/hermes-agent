# QQ 봇 (QQ Bot)

Hermes를 공식 **QQ 봇 API (v2)**를 통해 QQ에 연결하세요 — 음성 전사를 지원하는 개인(C2C), 그룹 @멘션, 길드 및 다이렉트 메시지를 지원합니다.

## 개요

QQ 봇 어댑터는 [공식 QQ 봇 API](https://bot.q.qq.com/wiki/develop/api-v2/)를 사용하여 다음을 수행합니다:

- QQ 게이트웨이와 지속적인 **웹소켓(WebSocket)** 연결을 통해 메시지 수신
- **REST API**를 통한 텍스트 및 마크다운 답장 전송
- 이미지, 음성 메시지 및 파일 첨부 다운로드 및 처리
- 텐센트(Tencent)의 내장 ASR 또는 구성 가능한 STT 제공자를 사용하여 음성 메시지 전사

## 전제 조건

1. **QQ 봇 애플리케이션(QQ Bot Application)** — [q.qq.com](https://q.qq.com)에서 등록:
   - 새 애플리케이션을 생성하고 **App ID**와 **App Secret**을 메모해 두세요.
   - 필요한 인텐트 활성화: C2C 메시지, 그룹 @-메시지, 길드 메시지
   - 테스트를 위해 봇을 샌드박스 모드로 구성하거나 프로덕션용으로 게시하세요.

2. **종속성(Dependencies)** — 어댑터에는 `aiohttp` 및 `httpx`가 필요합니다:
   ```bash
   pip install aiohttp httpx
   ```

## 구성

### 대화형 설정

```bash
hermes gateway setup
```

플랫폼 목록에서 **QQ Bot**을 선택하고 프롬프트를 따릅니다.

### 수동 구성

`~/.hermes/.env`에 필수 환경 변수를 설정합니다:

```bash
QQ_APP_ID=your-app-id
QQ_CLIENT_SECRET=your-app-secret
```

## 환경 변수

| 변수 | 설명 | 기본값 |
|---|---|---|
| `QQ_APP_ID` | QQ 봇 App ID (필수) | — |
| `QQ_CLIENT_SECRET` | QQ 봇 App Secret (필수) | — |
| `QQBOT_HOME_CHANNEL` | cron/알림 전송을 위한 OpenID | — |
| `QQBOT_HOME_CHANNEL_NAME` | 홈 채널의 표시 이름 | `Home` |
| `QQ_ALLOWED_USERS` | DM 액세스를 위한 쉼표로 구분된 사용자 OpenID | open (모든 사용자) |
| `QQ_GROUP_ALLOWED_USERS` | 그룹 액세스를 위한 쉼표로 구분된 그룹 OpenID | — |
| `QQ_ALLOW_ALL_USERS` | 모든 DM을 허용하려면 `true`로 설정 | `false` |
| `QQ_PORTAL_HOST` | QQ 포털 호스트 재정의 (샌드박스 라우팅의 경우 `sandbox.q.qq.com`으로 설정) | `q.qq.com` |
| `QQ_STT_API_KEY` | 음성-텍스트(Voice-to-text) 제공자를 위한 API 키 | — |
| `QQ_STT_BASE_URL` | (직접 읽지 않음 — 대신 `config.yaml`에 `platforms.qqbot.extra.stt.baseUrl` 설정) | n/a |
| `QQ_STT_MODEL` | STT 모델 이름 | `glm-asr` |

## 고급 구성

세밀한 제어를 위해 `~/.hermes/config.yaml`에 플랫폼 설정을 추가합니다:

```yaml
platforms:
  qqbot:
    enabled: true
    extra:
      app_id: "your-app-id"
      client_secret: "your-secret"
      markdown_support: true       # QQ 마크다운 활성화 (msg_type 2). config 전용; 환경 변수로는 불가능.
      dm_policy: "open"          # open | allowlist | disabled
      allow_from:
        - "user_openid_1"
      group_policy: "open"       # open | allowlist | disabled
      group_allow_from:
        - "group_openid_1"
      stt:
        provider: "zai"          # zai (GLM-ASR), openai (Whisper) 등.
        baseUrl: "https://open.bigmodel.cn/api/coding/paas/v4"
        apiKey: "your-stt-key"
        model: "glm-asr"
```

## 음성 메시지 (STT)

음성 전사는 2단계로 작동합니다:

1. **QQ 내장 ASR** (무료, 항상 먼저 시도됨) — QQ는 텐센트 자체 음성 인식을 사용하는 `asr_refer_text`를 음성 메시지 첨부 파일에 제공합니다.
2. **구성된 STT 제공자** (대체) — QQ의 ASR이 텍스트를 반환하지 않으면 어댑터가 OpenAI 호환 STT API를 호출합니다:

   - **Zhipu/GLM (zai)**: 기본 제공자이며, `glm-asr` 모델을 사용합니다.
   - **OpenAI Whisper**: `QQ_STT_BASE_URL` 및 `QQ_STT_MODEL`을 설정합니다.
   - 기타 OpenAI 호환 STT 엔드포인트

## 문제 해결

### 봇이 즉시 연결 끊김 (빠른 연결 해제)

이는 일반적으로 다음을 의미합니다:
- **잘못된 App ID / Secret** — q.qq.com에서 자격 증명을 다시 확인하세요.
- **권한 누락** — 봇에 필요한 인텐트가 활성화되어 있는지 확인하세요.
- **샌드박스 전용 봇** — 봇이 샌드박스 모드인 경우 QQ의 샌드박스 테스트 채널에서만 메시지를 받을 수 있습니다.

### 음성 메시지가 전사되지 않음

1. 첨부 데이터에 QQ의 내장 `asr_refer_text`가 있는지 확인하세요.
2. 사용자 지정 STT 제공자를 사용하는 경우 `QQ_STT_API_KEY`가 올바르게 설정되었는지 확인하세요.
3. 게이트웨이 로그에서 STT 오류 메시지를 확인하세요.

### 메시지가 전송되지 않음

- q.qq.com에서 봇의 **인텐트(intents)**가 활성화되어 있는지 확인하세요.
- DM 액세스가 제한된 경우 `QQ_ALLOWED_USERS`를 확인하세요.
- 그룹 메시지의 경우 봇이 **@멘션**되었는지 확인하세요. (그룹 정책에 허용 목록이 필요할 수 있음)
- cron/알림 전송을 위해 `QQBOT_HOME_CHANNEL`을 확인하세요.

### 연결 오류

- `aiohttp` 및 `httpx`가 설치되어 있는지 확인하세요: `pip install aiohttp httpx`
- `api.sgroup.qq.com` 및 웹소켓 게이트웨이로의 네트워크 연결을 확인하세요.
- 자세한 오류 메시지와 재연결 동작을 보려면 게이트웨이 로그를 검토하세요.
