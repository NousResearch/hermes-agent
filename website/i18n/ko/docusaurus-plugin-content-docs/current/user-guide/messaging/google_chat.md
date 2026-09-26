---
sidebar_position: 12
title: "Google Chat"
description: "Cloud Pub/Sub를 사용하여 Hermes Agent를 Google Chat 봇으로 설정합니다"
---

# Google Chat 설정

Hermes Agent를 Google Chat의 봇으로 연결합니다. 이 통합은 인바운드 이벤트에 Cloud Pub/Sub
pull 구독을 사용하고 아웃바운드 메시지에 Chat REST API를 사용합니다.
Slack Socket Mode 또는 Telegram 롱 폴링과 비슷한 편의성을 제공합니다. Hermes
프로세스에 공개 URL, 터널 또는 TLS 인증서가 필요하지 않습니다. 연결하고,
인증한 뒤 구독을 수신합니다. Telegram 봇이 토큰으로 수신 대기하는 것과 같은 방식입니다.

> `hermes gateway setup`을 실행하고 안내에 따라 **Google Chat**을 선택하세요.

:::note Workspace 버전
Google Chat은 Google Workspace의 일부입니다. 개인 Workspace(Google을 통해 등록한
`@yourdomain.com`) 또는 앱을 게시할 관리자 권한이 있는 업무용
Workspace에서 이 통합을 사용할 수 있습니다. Gmail 전용 계정은 Chat 앱을 호스팅할 수 없습니다.
:::

## 개요

| 구성 요소 | 값 |
|-----------|-------|
| **라이브러리** | `google-cloud-pubsub`, `google-api-python-client`, `google-auth` |
| **인바운드 전송** | Cloud Pub/Sub pull 구독(공개 엔드포인트 없음) |
| **아웃바운드 전송** | Chat REST API(`chat.googleapis.com`) |
| **인증** | 구독에 `roles/pubsub.subscriber`가 설정된 서비스 계정 JSON |
| **사용자 식별** | Chat 리소스 이름(`users/{id}`) + 이메일 |

---

## 1단계: GCP 프로젝트를 만들거나 선택하기

Pub/Sub 주제를 호스팅하려면 Google Cloud 프로젝트가 필요합니다. 프로젝트가 없다면
[console.cloud.google.com](https://console.cloud.google.com)에서 만드세요.
개인 계정은 봇 트래픽을 충분히 처리할 수 있는 무료 등급을 제공합니다.

프로젝트 ID(예: `my-chat-bot-123`)를 기록해 두세요. 이후 모든 단계에서 사용합니다.

---

## 2단계: API 두 개 사용 설정하기

콘솔에서 **API 및 서비스 → 라이브러리**로 이동하고 다음 API를 사용 설정합니다.

- **Google Chat API**
- **Cloud Pub/Sub API**

두 API 모두 개인 봇이 생성하는 수준의 사용량에서는 무료입니다.

---

## 3단계: 서비스 계정 만들기

**IAM 및 관리자 → 서비스 계정 → 서비스 계정 만들기**로 이동합니다.

- 이름: `hermes-chat-bot`
- "이 서비스 계정에 프로젝트 액세스 권한 부여" 단계를 건너뜁니다. 특정
  구독에 대한 IAM 권한만 있으면 됩니다. 프로젝트 수준의 Pub/Sub 역할은 **부여하지 마세요**.

생성한 뒤 SA를 열고 **키 → 키 추가 → 새 키 만들기 → JSON**으로 이동하여
파일을 다운로드합니다. Hermes만 읽을 수 있는 곳에 저장하세요(예:
`~/.hermes/google-chat-sa.json`, `chmod 600`).

:::caution "Chat Bot Caller" 역할은 없습니다
흔히 Chat 전용 IAM 역할을 검색하여 프로젝트 수준에서 부여하는 실수를 합니다.
그런 역할은 존재하지 않습니다. Chat 봇의 권한은 IAM이 아니라 스페이스에 설치되어
있는지 여부에서 비롯됩니다. 다음 단계에서 만들 구독에 대한 Pub/Sub subscriber 권한만
SA에 있으면 됩니다.
:::

---

## 4단계: Pub/Sub 주제와 구독 만들기

**Pub/Sub → 주제 → 주제 만들기**로 이동합니다.

- 주제 ID: `hermes-chat-events`
- 나머지는 모두 기본값으로 둡니다.

생성 후 주제의 세부정보 페이지에 **구독** 탭이 표시됩니다. 하나를 만드세요.

- 구독 ID: `hermes-chat-events-sub`
- 전송 유형: **Pull**
- 메시지 보존: **7일**(hermes 재시작 후에도 백로그가 유지되도록)
- 나머지는 기본값으로 둡니다.

---

## 5단계: 주제에 IAM 바인딩 설정하기(중요)

**구독이 아닌 주제**에 IAM 주 구성원을 추가합니다.

- 주 구성원: `chat-api-push@system.gserviceaccount.com`
- 역할: `Pub/Sub Publisher`

이 설정이 없으면 Google Chat이 주제에 이벤트를 게시할 수 없으며 봇은 어떤 것도
수신하지 못합니다.

---

## 6단계: 구독에 IAM 바인딩 설정하기

**구독**에 자신의 서비스 계정을 주 구성원으로 추가합니다.

- 주 구성원: `hermes-chat-bot@<your-project>.iam.gserviceaccount.com`
- 역할: `Pub/Sub Subscriber`

같은 구독에 `Pub/Sub Viewer`도 부여합니다. Hermes는 시작 시 연결 가능성 확인을 위해
`subscription.get()`을 호출합니다.

---

## 7단계: Chat 앱 구성하기

**API 및 서비스 → Google Chat API → 구성**으로 이동합니다.

- **앱 이름**: 사용자가 보게 될 이름을 입력합니다("Hermes"가 무난합니다).
- **아바타 URL**: 공개 PNG URL을 입력합니다(Google에서 기본값도 제공합니다).
- **설명**: 앱 디렉터리에 표시할 짧은 문장을 입력합니다.
- **기능**: **1:1 메시지 수신** 및 **스페이스와 그룹 대화 참여**를 사용 설정합니다.
- **연결 설정**: **Cloud Pub/Sub**를 선택하고 주제 이름을
  `projects/<your-project>/topics/hermes-chat-events`로 입력합니다.
- **공개 범위**: 워크스페이스(또는 특정 사용자)로 제한합니다. 테스트 중에는
  모든 사용자에게 게시하지 마세요.

저장합니다.

---

## 8단계: 테스트 스페이스에 봇 설치하기

브라우저에서 Google Chat을 엽니다. **+ 새 채팅** 메뉴에서 앱 이름을 검색하여 앱과
DM을 시작합니다. 처음 메시지를 보내면 Google이 `ADDED_TO_SPACE` 이벤트를 전송하고,
Hermes는 자체 메시지 필터링을 위해 봇 자체의 `users/{id}`를 캐시합니다.

---

## 9단계: Hermes 구성하기

`~/.hermes/.env`에 Google Chat 섹션을 추가합니다.

```bash
# Required
GOOGLE_CHAT_PROJECT_ID=my-chat-bot-123
GOOGLE_CHAT_SUBSCRIPTION_NAME=projects/my-chat-bot-123/subscriptions/hermes-chat-events-sub
GOOGLE_CHAT_SERVICE_ACCOUNT_JSON=/home/you/.hermes/google-chat-sa.json

# Authorization — paste the emails of people allowed to talk to the bot
GOOGLE_CHAT_ALLOWED_USERS=you@yourdomain.com,coworker@yourdomain.com

# Optional
GOOGLE_CHAT_HOME_CHANNEL=spaces/AAAA...         # default delivery destination for cron jobs
GOOGLE_CHAT_MAX_MESSAGES=1                      # Pub/Sub FlowControl; 1 serializes commands per session
GOOGLE_CHAT_MAX_BYTES=16777216                  # 16 MiB — cap on in-flight message bytes
```

프로젝트 ID는 `GOOGLE_CLOUD_PROJECT`로도 대체할 수 있고 SA 경로는
`GOOGLE_APPLICATION_CREDENTIALS`로도 대체할 수 있습니다. 원하는 규칙을 사용하세요.

Google Chat 어댑터의 종속 항목은 유지 관리되는 설치 프로그램을 통해 설치합니다.
이 설치 프로그램은 런타임 검사에 사용되는 것과 동일한 고정 보안 최소 버전을 적용합니다.

```bash
python -m plugins.platforms.google_chat.oauth --install-deps
```

게이트웨이를 시작합니다.

```bash
hermes gateway
```

다음과 비슷한 로그가 표시되어야 합니다.

```
[GoogleChat] Connected; project=my-chat-bot-123, subscription=<redacted>,
             bot_user_id=users/XXXX, flow_control(msgs=1, bytes=16777216)
```

테스트 DM에서 "hola"를 전송합니다. 봇이 "Hermes is thinking…" 표시를 게시한 뒤,
같은 메시지를 실제 응답으로 제자리에서 수정합니다. "메시지 삭제됨" 흔적은 남지 않습니다.

### 작업 상태 표시 사용자 지정

표시 텍스트는 `~/.hermes/config.yaml`의 `typing_status_text`로 구성할 수 있습니다.
예를 들어 Ada라는 이름의 고양이 보조자라면 다음과 같습니다.

```yaml
platforms:
  google_chat:
    # Custom working-state marker text (default: "Hermes is thinking…").
    typing_status_text: "is pouncing… 🐾"
```

Slack의 일시적인 상태 표시줄과 달리 이는 **실제로 게시되는 메시지**이며
응답으로 제자리에서 수정됩니다. 따라서 여기에서 설정한 내용은 잠시 일반 메시지로
채팅에 나타납니다. 표시를 완전히 끄려면 `typing_indicator: false`로 설정하세요.

---

## 서식 및 기능

Google Chat은 제한된 Markdown 하위 집합을 렌더링합니다.

| 지원됨 | 지원되지 않음 |
|-----------|---------------|
| `*bold*`, `_italic_`, `~strike~`, `` `code` `` | 제목, 목록 |
| URL을 통한 인라인 이미지 | 대화형 Card v2 버튼(이 게이트웨이는 v1) |
| 기본 파일 첨부(`/setup-files` 후 — 10단계 참조) | 기본 음성 메모 / 원형 동영상 메모 |

에이전트의 시스템 프롬프트에는 Google Chat 전용 힌트가 포함되어 있으므로 이러한
제한을 알고 렌더링되지 않는 서식을 피합니다.

메시지 크기 제한: 메시지당 4000자. 더 긴 에이전트 응답은 자동으로 여러 메시지로
분할됩니다.

스레드 지원: 사용자가 스레드 안에서 답장하면 Hermes가 `thread.name`을 감지하고
같은 스레드에 답장을 게시하므로 각 스레드에 별도의 Hermes 세션이 생성됩니다.

### 대화형 카드로 명확화 질문하기

에이전트가 객관식 명확화 질문을 하면 어댑터는 일반적인 번호 매긴 텍스트 목록 대신
선택지마다 버튼 하나와 **"기타 / 답변 입력"** 버튼이 있는 기본 **Card v2**로
렌더링합니다. 버튼을 클릭하면 질문에 직접 답변합니다(`CARD_CLICKED` 이벤트가
대기 중인 세션으로 선택지를 전달합니다). 카드를 전송하지 못하거나 질문에 고정된
선택지가 없으면 어댑터는 표준 텍스트 명확화 방식으로 대체합니다. 별도 구성은 필요 없습니다.

---

## 10단계: 기본 첨부 파일 전송(선택 사항)

기본적으로 봇은 텍스트, URL을 통한 인라인 이미지, 오디오/동영상/문서용 다운로드
카드를 게시할 수 있습니다. 사람이 파일을 끌어다 놓을 때와 동일한 **기본 Chat 첨부
파일**을 전송하려면 각 사용자가 사용자별 OAuth 흐름을 통해 봇을 한 번 인증해야 합니다.

### 별도 흐름이 필요한 이유

Google Chat의 `media.upload` 엔드포인트는 서비스 계정 인증을 명시적으로 거부합니다.

> 이 메서드는 서비스 계정을 사용한 앱 인증을 지원하지 않습니다.
> 사용자 계정으로 인증하세요.

이를 해결할 IAM 역할이나 범위는 없습니다. 이 엔드포인트는 사용자 자격 증명만
허용합니다. 따라서 봇은 파일을 업로드할 때마다 *사용자*로 동작해야 하며,
구체적으로는 파일을 요청한 사용자로 동작해야 합니다.

### 일회성 설정(프로필별)

1. 같은 GCP 프로젝트에서 **API 및 서비스 → 사용자 인증 정보**로 이동합니다.
2. **사용자 인증 정보 만들기 → OAuth 클라이언트 ID → 데스크톱 앱**을 선택합니다.
3. JSON을 다운로드합니다. Hermes를 실행하는 호스트로 옮깁니다.
4. Hermes에 클라이언트를 등록합니다(범위를 지정하려는 프로필에서 실행).

```bash
# Default profile:
python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json

# A named profile gets its own separate registration:
hermes -p <profile> python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json
```

이 작업은 활성 프로필의 Hermes 홈에 클라이언트 시크릿을 기록합니다(예:
기본 프로필의 경우 `~/.hermes/google_chat_user_client_secret.json`). 클라이언트
시크릿은 **프로필 범위이며 프로필 간에 공유되지 않습니다**. 이는 의도된 동작입니다.
프로필은 인증 경계가 분리되어 있으므로 두 프로필이 서로 다른 Google OAuth 앱 또는
계정을 가리킬 수 있습니다. Google Chat 첨부 파일 전송이 필요한 각 프로필에서 한 번씩
등록하세요.

### 사용자별 인증(채팅에서)

각 사용자는 봇과의 개인 DM에서 다음 흐름을 한 번 실행합니다.

1. 봇에 `/setup-files`를 전송합니다. 봇이 상태와 다음 단계를 답장합니다.
2. `/setup-files start`를 전송합니다. 봇이 OAuth URL을 답장합니다.
3. URL을 열고 **허용**을 클릭한 뒤 브라우저가
   `http://localhost:1/?...&code=...`를 로드하지 못하는 것을 확인합니다. 이는 예상된
   동작이며 인증 코드는 URL 표시줄에 있습니다.
4. 로드에 실패한 URL(또는 `code=...` 값만)을 복사하여 `/setup-files <PASTED_URL>`로
   채팅에 다시 붙여 넣습니다. 봇이 이를 교환하여 갱신 토큰을 얻습니다.

토큰은 `~/.hermes/google_chat_user_tokens/<sanitized_email>.json`에 저장됩니다.
이후 해당 사용자의 DM에서 파일을 요청하면 봇이 *그 사용자로* 업로드하므로 메시지가
사용자의 스페이스에 도착합니다.

나중에 취소하려면 `/setup-files revoke`를 사용합니다. 해당 사용자의 토큰만 삭제됩니다.
다른 사용자의 토큰에는 영향을 주지 않습니다.

### 범위

이 흐름은 정확히 하나의 범위인 `chat.messages.create`를 요청합니다. 이 범위는
`media.upload`와 업로드된 `attachmentDataRef`를 참조하는 `messages.create`를 모두
처리합니다. Drive나 더 광범위한 Chat 범위는 요청하지 않습니다. 의도적으로 최소 권한만
사용합니다.

### 여러 사용자 동작

요청자에게 아직 사용자별 토큰이 없으면 봇은
`~/.hermes/google_chat_user_token.json`에 있는 레거시 단일 사용자 토큰(이전의
다중 사용자 설치에서 남아 있는 경우)으로 대체합니다. 둘 다 없으면 봇은 요청자에게
`/setup-files`를 실행하라는 명확한 텍스트 안내를 게시합니다.

사용자가 취소하면 자신의 슬롯만 삭제됩니다. 한 사용자의 토큰에서 발생한 401/403은
해당 사용자의 캐시만 제거합니다. 사용자끼리 서로 방해하지 않습니다.

---

## 문제 해결

**"hola"를 보낸 뒤 봇이 아무 반응이 없습니다.**

1. 콘솔에서 Pub/Sub 구독에 전달되지 않은 메시지가 있는지 확인합니다.
   있다면 Hermes가 인증되지 않은 것입니다. `GOOGLE_CHAT_SERVICE_ACCOUNT_JSON`과
   SA가 해당 구독의 `Pub/Sub Subscriber`로 등록되어 있는지 확인합니다.
2. 구독에 메시지가 없다면 Google Chat이 게시하지 않는 것입니다.
   **주제**의 IAM 바인딩을 다시 확인합니다.
   `chat-api-push@system.gserviceaccount.com`에 `Pub/Sub Publisher`가 있어야 합니다.
3. `[GoogleChat] Connected`가 있는지 `hermes gateway` 로그를 확인합니다. 
   `[GoogleChat] Config validation failed`가 보이면 오류 메시지에 수정할
   환경 변수가 표시됩니다.

**봇은 답장하지만 에이전트의 답변 대신 오류 메시지가 표시됩니다.**

`[GoogleChat] Pub/Sub stream died`가 있는지 로그를 확인합니다. 이 메시지가 반복되면
SA 자격 증명이 교체되었거나 구독이 삭제되었을 수 있습니다. 10회 시도 후 어댑터는
자체 상태를 치명적 오류로 표시합니다.

**모든 아웃바운드 메시지에서 "403 Forbidden"이 표시됩니다.**

봇이 스페이스에서 제거되었거나 Chat API 콘솔에서 취소되었습니다. 스페이스에 다시
설치하세요. 다음 `ADDED_TO_SPACE` 이벤트가 메시징을 자동으로 다시 활성화합니다.

**"Rate limit hit" 경고가 너무 많이 표시됩니다.**

Chat API의 기본 할당량은 스페이스당 분당 메시지 60개입니다. 에이전트가 이 한도를
초과하는 긴 스트리밍 응답을 생성하면 어댑터가 지수 백오프로 재시도하지만, 사용자가
체감하는 지연은 계속 발생합니다. 간결한 응답을 사용하거나 GCP 콘솔에서 할당량을
늘리는 방법을 고려하세요.

**봇이 파일 대신 "/setup-files" 안내를 계속 게시합니다.**

요청자에게 사용자별 OAuth 토큰이 없고 레거시 대체 토큰도 없습니다. 해당 사용자의
DM에서 `/setup-files`를 실행하고 10단계를 따르세요. 교환이 완료되면 다음 파일 요청부터
게이트웨이를 재시작하지 않아도 기본 방식으로 업로드됩니다.

**`/setup-files start`에 "저장된 클라이언트 자격 증명이 없습니다."가 표시됩니다.**

이 프로필에 대해 일회성 설정이 완료되지 않았습니다(클라이언트 시크릿은 프로필
범위이므로 한 프로필에 등록해도 다른 프로필에서 보이지 않습니다). 터미널에서
게이트웨이가 사용하는 프로필로 실행합니다.

```bash
# Default profile:
python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json

# Named profile:
hermes -p <profile> python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json
```

그런 다음 `/setup-files start`를 다시 전송합니다.

**`/setup-files <PASTED_URL>`에 "토큰 교환에 실패했습니다."가 표시됩니다.**

인증 코드는 한 번만 사용할 수 있고 수명이 짧습니다(일반적으로 몇 분). `/setup-files start`를
전송하여 새 URL을 받은 뒤 다시 시도하세요.

---

## 보안 참고 사항

- **서비스 계정 범위**: 어댑터는 `chat.bot` 및 `pubsub` 범위를 요청합니다.
  실제 적용은 IAM에서 이루어져야 합니다. SA에 최소 권한(구독에 대한
  `roles/pubsub.subscriber` + `roles/pubsub.viewer`)만 부여하고 프로젝트 수준이나
  조직 수준의 Pub/Sub 역할은 부여하지 마세요.
- **첨부 파일 다운로드 보호**: Hermes는 호스트가 Google 소유 도메인의 짧은 허용 목록
  (`googleapis.com`, `drive.google.com`, `lh[3-6].googleusercontent.com` 및
  기타 일부)과 일치하는 URL에만 SA bearer 토큰을 연결합니다. 그 밖의 호스트는 HTTP
  요청 전에 거부됩니다. 이는 조작된 이벤트가 bearer 토큰을 GCE 메타데이터 서비스로
  리디렉션할 수 있는 SSRF 시나리오를 방지하기 위한 것입니다.
- **비식별화**: 서비스 계정 이메일, 구독 경로 및 주제 경로는 `agent/redact.py`가
  로그 출력에서 제거합니다. 디버그 봉투 덤프(`GOOGLE_CHAT_DEBUG_RAW=1`)도 동일한
  비식별화 필터를 거치며 DEBUG 수준으로 기록됩니다.
- **컴플라이언스**: 이 봇을 규제 대상 워크스페이스(데이터 레지던시 또는 AI 거버넌스
  정책이 있는 모든 환경)에 연결하려면 최초 설치 전에 해당 승인을 받으세요.
- **사용자 OAuth 범위**: 사용자별 첨부 파일 흐름은 `chat.messages.create`만
  요청합니다. 이는 `media.upload`와 후속 `messages.create`를 처리하는 데 필요한
  최소 범위입니다. 토큰은 `~/.hermes/google_chat_user_tokens/<sanitized_email>.json`에
  일반 JSON으로 저장됩니다(파일 시스템 권한으로 보호하며 SA 키 파일과 동일한
  방식입니다). 각 토큰은 정확히 한 사용자의 소유이며 취소 범위도 해당 사용자로
  제한됩니다.
