---
sidebar_position: 12
title: "Google Chat"
description: "Cloud Pub/Sub을 사용하여 Hermes Agent를 Google Chat 봇으로 설정하기"
---

# Google Chat 설정

Hermes Agent를 Google Chat에 봇으로 연결합니다. 이 통합은 인바운드 이벤트를 위해 Cloud Pub/Sub의 풀(pull) 구독을 사용하고 아웃바운드 메시지를 위해 Chat REST API를 사용합니다. Slack Socket Mode나 Telegram의 롱 폴링(long-polling)과 동일한 편의성을 제공합니다. Hermes 프로세스는 퍼블릭 URL, 터널링, 또는 TLS 인증서가 필요하지 않습니다. 마치 Telegram 봇이 토큰을 사용해 수신 대기하는 것처럼 접속하고 인증하여 구독을 수신합니다.

> `hermes gateway setup`을 실행하고 **Google Chat**을 선택하면 안내에 따라 설정할 수 있습니다.

:::note Workspace 에디션
Google Chat은 Google Workspace의 일부입니다. 구글을 통해 등록된 개인용 Workspace (`@yourdomain.com`)나, 앱을 배포할 수 있는 관리자(Admin) 권한이 있는 회사 계정에서 이 통합 기능을 사용할 수 있습니다. Gmail 전용 계정으로는 Chat 앱을 호스팅할 수 없습니다.
:::

## 개요

| 컴포넌트 | 값 |
|-----------|-------|
| **라이브러리** | `google-cloud-pubsub`, `google-api-python-client`, `google-auth` |
| **인바운드 전송 수단** | Cloud Pub/Sub 풀 구독 (퍼블릭 엔드포인트 없음) |
| **아웃바운드 전송 수단** | Chat REST API (`chat.googleapis.com`) |
| **인증** | 서비스 계정 JSON (구독에 대해 `roles/pubsub.subscriber` 권한 필요) |
| **사용자 식별** | Chat 리소스 이름 (`users/{id}`) + 이메일 |

---

## 1단계: GCP 프로젝트 생성 또는 선택

Pub/Sub 주제(topic)를 호스팅하기 위해 Google Cloud 프로젝트가 필요합니다. 아직 프로젝트가 없다면 [console.cloud.google.com](https://console.cloud.google.com)에서 생성하세요. 개인 계정에 제공되는 무료 등급으로도 봇 트래픽을 감당하기에 충분합니다.

프로젝트 ID (예: `my-chat-bot-123`)를 메모해 두세요. 이 ID는 이후 모든 단계에서 사용됩니다.

---

## 2단계: 2가지 API 활성화하기

Google Cloud 콘솔에서 **API 및 서비스(APIs & Services) → 라이브러리(Library)**로 이동하여 다음을 활성화합니다:

- **Google Chat API**
- **Cloud Pub/Sub API**

개인 봇이 발생시키는 일반적인 수준의 사용량은 모두 무료로 처리됩니다.

---

## 3단계: 서비스 계정(Service Account) 생성

**IAM 및 관리자(IAM & Admin) → 서비스 계정(Service Accounts) → 서비스 계정 만들기(Create Service Account)** 메뉴로 이동합니다.

- 이름: `hermes-chat-bot`
- "이 서비스 계정에 프로젝트 액세스 권한 부여(Grant this service account access to project)" 단계는 건너뜁니다. 이어지는 단계에서 구성할 특정 구독에 대한 권한만 있으면 됩니다 — **절대** 프로젝트 레벨의 Pub/Sub 역할을 부여하지 마세요.

생성 후, 서비스 계정을 열고 **키(Keys) → 키 추가(Add Key) → 새 키 만들기(Create new key) → JSON**을 선택하여 파일을 다운로드합니다. 이 파일을 Hermes만 읽을 수 있는 안전한 위치에 저장합니다 (예: `~/.hermes/google-chat-sa.json`, `chmod 600`).

:::caution "Chat 봇 호출자(Chat Bot Caller)" 역할은 없습니다
Chat에 특화된 IAM 역할을 검색하여 프로젝트 수준에서 부여하려고 하는 흔한 실수가 있습니다. 그런 역할은 존재하지 않습니다. Chat 봇 권한은 IAM이 아닌 스페이스에 설치됨으로써 부여받습니다. 필요한 역할은 다음 단계에서 만들 구독에 대한 Pub/Sub 구독자 권한뿐입니다.
:::

---

## 4단계: Pub/Sub 주제(Topic) 및 구독(Subscription) 생성

**Pub/Sub → 주제(Topics) → 주제 만들기(Create topic)**로 이동합니다.

- 주제 ID: `hermes-chat-events`
- 나머지 설정은 모두 기본값으로 둡니다.

생성이 완료되면 주제 상세 페이지에 **구독(Subscriptions)** 탭이 나타납니다. 여기서 새로운 구독을 생성합니다:

- 구독 ID: `hermes-chat-events-sub`
- 전송 유형(Delivery type): **가져오기(Pull)**
- 메시지 보관 기간(Message retention): **7일** (Hermes를 재시작해도 대기열이 유지되도록 설정)
- 나머지는 기본값으로 둡니다.

---

## 5단계: 주제(Topic)에 대한 IAM 바인딩 (중요)

구독이 아닌 **주제(topic)** 설정에서 다음의 IAM 주 구성원(principal)을 추가합니다:

- 주 구성원: `chat-api-push@system.gserviceaccount.com`
- 역할: `Pub/Sub 게시자(Pub/Sub Publisher)`

이 설정이 없으면 Google Chat이 당신의 주제로 이벤트를 게시할 수 없고, 결국 봇이 어떠한 메시지도 받지 못하게 됩니다.

---

## 6단계: 구독(Subscription)에 대한 IAM 바인딩

이번에는 **구독(subscription)** 설정에서 위에서 생성했던 서비스 계정을 주 구성원(principal)으로 추가합니다:

- 주 구성원: `hermes-chat-bot@<your-project>.iam.gserviceaccount.com`
- 역할: `Pub/Sub 구독자(Pub/Sub Subscriber)`

동일한 구독에 대해 `Pub/Sub 뷰어(Pub/Sub Viewer)` 권한도 함께 부여하세요 — Hermes는 시작 시 도달 가능성 검증의 일환으로 `subscription.get()`을 호출합니다.

---

## 7단계: Chat 앱 구성

**API 및 서비스(APIs & Services) → Google Chat API → 구성(Configuration)** 메뉴로 이동합니다.

- **앱 이름(App name)**: 사용자가 볼 이름 ("Hermes" 권장)
- **아바타 URL(Avatar URL)**: 공개된 PNG 파일의 링크 (Google에서 제공하는 기본값도 있음)
- **설명(Description)**: 앱 디렉토리에 표시될 짧은 문장
- **기능(Functionality)**: **1:1 메시지 수신(Receive 1:1 messages)** 및 **스페이스 및 그룹 대화 참여(Join spaces and group conversations)** 항목을 활성화합니다.
- **연결 설정(Connection settings)**: **Cloud Pub/Sub**을 선택하고 `projects/<your-project>/topics/hermes-chat-events` 형식으로 방금 전 만든 주제 이름을 입력합니다.
- **가시성(Visibility)**: 조직 내부나 특정 사용자로만 제한하세요 — 테스트 중에는 모든 사용자에게 앱을 노출하지 마세요.

입력이 끝나면 저장합니다.

---

## 8단계: 테스트 스페이스에 봇 설치

웹 브라우저에서 Google Chat을 엽니다. **+ 새 채팅(+ New Chat)** 메뉴에서 구성했던 앱의 이름을 검색하여 다이렉트 메시지(DM)를 시작합니다. 봇에게 처음 메시지를 보내면, Google은 `ADDED_TO_SPACE` 이벤트를 발생시키며 Hermes는 이 이벤트에서 자신이 전송한 메시지(self-message) 필터링에 사용할 봇의 `users/{id}` 를 추출해 캐싱합니다.

---

## 9단계: Hermes 구성

`~/.hermes/.env` 파일에 Google Chat 관련 설정을 추가합니다:

```bash
# 필수 설정
GOOGLE_CHAT_PROJECT_ID=my-chat-bot-123
GOOGLE_CHAT_SUBSCRIPTION_NAME=projects/my-chat-bot-123/subscriptions/hermes-chat-events-sub
GOOGLE_CHAT_SERVICE_ACCOUNT_JSON=/home/you/.hermes/google-chat-sa.json

# 권한 관리 — 봇과 대화가 허용된 사람들의 이메일을 쉼표로 구분하여 기입하세요.
GOOGLE_CHAT_ALLOWED_USERS=you@yourdomain.com,coworker@yourdomain.com

# 선택적 설정
GOOGLE_CHAT_HOME_CHANNEL=spaces/AAAA...         # cron 작업의 결과를 수신할 기본 공간
GOOGLE_CHAT_MAX_MESSAGES=1                      # Pub/Sub FlowControl; 1로 설정하면 세션당 처리되는 명령들을 직렬화합니다.
GOOGLE_CHAT_MAX_BYTES=16777216                  # 16 MiB — 처리 중인 메시지의 바이트 상한
```

프로젝트 ID 설정은 `GOOGLE_CLOUD_PROJECT` 변수 이름을 활용해도 좋으며, 서비스 계정(SA)의 JSON 경로는 `GOOGLE_APPLICATION_CREDENTIALS` 변수를 사용하여 설정할 수도 있습니다 — 편한 방식을 취하세요.

Google Chat 어댑터 구동을 위해 필요한 의존성(Dependencies)들을 설치합니다. (현재 Hermes의 확장 요소로 직접 배포되고 있지 않으므로 다음과 같이 수동으로 설치해야 합니다):

```bash
pip install google-cloud-pubsub google-api-python-client google-auth google-auth-oauthlib
```

이제 게이트웨이를 시작합니다:

```bash
hermes gateway
```

이후 로그에 다음과 유사한 라인이 출력되어야 합니다:

```
[GoogleChat] Connected; project=my-chat-bot-123, subscription=<redacted>,
             bot_user_id=users/XXXX, flow_control(msgs=1, bytes=16777216)
```

이전에 만들었던 테스트 DM 채팅창에 "hola"를 보내보세요. 봇이 "Hermes is thinking…" 이라는 안내 메시지를 남기고, 얼마 후 해당 메시지 자체를 답변 내용으로 교체합니다 — 화면에서 "메시지 삭제됨"과 같은 불필요한 흔적을 남기지 않습니다.

---

## 포맷팅 및 제공 기능 한계

Google Chat의 마크다운 렌더링 범위는 제한적입니다:

| 지원됨 | 지원되지 않음 |
|-----------|---------------|
| `*굵게*`, `_기울임_`, `~취소선~`, `` `코드` `` | 제목 글씨(Headings), 목록(lists) |
| URL 기반의 인라인 이미지 | 대화형 카드 v2 버튼 기능 (현재의 게이트웨이는 v1에 해당함) |
| 네이티브 파일 첨부 (10단계의 `/setup-files` 작업 이후 가능) | 네이티브 음성 메모 / 원형 비디오(circular video) 메모 |

에이전트의 시스템 프롬프트(system prompt)에는 구글 챗에 특화된 힌트가 들어있어 에이전트 스스로 해당 한계를 인지하며, 렌더링되지 않는 마크다운 기호를 쓰지 않도록 만듭니다.

메시지 크기 제한: 메시지당 4,000자입니다. 에이전트의 답변이 이 길이 제한을 넘어설 시 여러 개의 메시지로 자동 분할됩니다.

스레드 지원: 사용자가 스레드 내에서 답변을 남기면 Hermes는 그 스레드 이름(`thread.name`)을 인지하여 해당 위치에 답변합니다. 이를 통해 각 스레드는 고유의 분리된 Hermes 세션처럼 작동하게 됩니다.

---

## 10단계: 네이티브 첨부 파일 전송 (선택 사항)

기본적으로 봇은 텍스트 전송, URL을 통한 인라인 이미지, 오디오/비디오/문서에 대한 다운로드 카드 처리를 할 수 있습니다. 이에 더하여 사람이 파일을 드래그 앤 드롭으로 보낼 때처럼 일반적인 파일 위젯인 **네이티브** Chat 첨부 파일을 전달하려면, 사용자는 계정별로 OAuth 플로우를 거쳐 봇에게 일회성 권한 승인을 내려야 합니다.

### 이 플로우가 별도로 구성된 이유

Google Chat의 `media.upload` 엔드포인트는 서비스 계정(service-account) 인증을 완전히 거부합니다:

> This method doesn't support app authentication with a service account.
> Authenticate with a user account.
> (번역: 이 메서드는 서비스 계정을 통한 앱 인증을 지원하지 않습니다. 사용자 계정으로 인증하세요.)

이 문제를 우회할 수 있는 IAM 역할이나 권한(scope)은 존재하지 않습니다. 이 엔드포인트는 오로지 사용자의 자격 증명만을 받아들입니다. 따라서 봇은 파일을 업로드할 때마다 *사용자 권한으로* 작업을 수행해야 하며 — 좀 더 명확히 말해, 파일을 요청한 그 사용자 권한으로 봇이 활동해야만 합니다.

### 단 1회 수행해야 할 초기 설정 (프로필별 설정)

1. 동일한 GCP 프로젝트의 **API 및 서비스(APIs & Services) → 사용자 인증 정보(Credentials)**로 이동합니다.
2. **사용자 인증 정보 만들기(Create credentials) → OAuth 클라이언트 ID(OAuth client ID) → 데스크톱 앱(Desktop app)**을 선택합니다.
3. 생성된 JSON 파일을 다운로드하여 Hermes가 구동 중인 호스트 컴퓨터로 가져옵니다.
4. 클라이언트를 Hermes에 등록합니다 (적용할 프로필을 명확히 설정하여 구동하세요):

```bash
# 기본 프로필일 때:
python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json

# 특정 이름을 가진 프로필에 등록할 때:
hermes -p <profile> python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json
```

위 작업이 끝나면 현재 설정된 프로필의 Hermes 홈 경로 내에 클라이언트 시크릿을 저장하게 됩니다 (예시: 기본 프로필이라면 `~/.hermes/google_chat_user_client_secret.json`에 저장됩니다). 주의할 점으로 클라이언트 시크릿은 **각 프로필 내로 제한(profile-scoped)되며 다른 프로필과 서로 공유되지 않습니다** — 따라서 각 프로필마다 별도로 등록해야 합니다. 이는 보안 목적으로 의도된 구조입니다: 프로필마다 독립된 권한(auth boundaries)을 갖기 때문에 두 가지 프로필이 서로 다른 Google OAuth 앱 / 계정에 개별적으로 연결될 수 있습니다. Google Chat에서 첨부 파일 발송 기능을 써야 할 각각의 프로필마다 한 번씩 이 등록 과정을 진행하세요.

### 사용자별 권한 획득 절차 (채팅 앱 내부)

파일 기능 사용을 원하는 각 개인은 봇과의 1:1 채팅(DM)을 통해 이 작업을 1회 수행합니다:

1. 사용자가 봇에게 `/setup-files`를 전송합니다. 봇은 상태 및 다음 단계를 회신합니다.
2. 이어서 사용자가 `/setup-files start`를 보냅니다. 봇은 인증 절차를 밟을 OAuth URL을 줍니다.
3. 사용자가 제공된 URL을 열어 **허용(Allow)** 버튼을 누르면, 브라우저가 `http://localhost:1/?...&code=...`을 로드하지 못하고 화면이 실패 상태에 머물게 되는 것을 보게 될 것입니다. 이는 예상된 결과입니다 — 핵심인 인증 코드(auth code)는 URL 주소창에 위치해 있습니다.
4. 실패한 페이지의 전체 URL을 (또는 `code=...` 부분만) 복사한 뒤, 채팅 앱으로 돌아가 봇에게 `/setup-files <복사한_URL>` 형태로 다시 말을 겁니다. 봇은 이 코드를 리프레시 토큰(refresh token)으로 성공적으로 교환(exchange)해냅니다.

교환이 끝난 토큰은 호스트 내 `~/.hermes/google_chat_user_tokens/<sanitized_email>.json` 파일에 안착합니다.
이후 해당 사용자가 DM에서 파일 첨부를 요청하면 *본인의* 토큰을 쓰게 됩니다. 봇은 해당 사용자의 계정 권한으로 파일 업로드를 수행하고, 그 결과가 성공적으로 사용자의 작업 스페이스에 담깁니다.

이후 취소를 원할 때: `/setup-files revoke` 명령을 치면 발신한 사용자의 토큰만 삭제합니다. 다른 사용자의 파일은 아무런 영향을 받지 않습니다.

### 범위 (Scope)

이 흐름은 정확히 단 하나의 권한(scope), 즉 `chat.messages.create` 만을 요청합니다. 이 권한은 `media.upload`를 수행할 때와 업로드 후 `attachmentDataRef` 데이터를 활용해 `messages.create`를 진행할 때 적용됩니다. Google Drive나 여타 폭넓은 권한을 묻지 않습니다 — 이는 가장 최소 권한만 사용하려는 목적에 부합합니다.

### 다중 사용자 환경에서의 동작 특성

사용자별 토큰이 없을 때 봇은 과거 단일 사용자(single-user) 시절의 데이터인 `~/.hermes/google_chat_user_token.json` 파일의 유무를 탐색하여(과거 설치 기록이 남아있다면) 폴백(fallback)을 시도합니다. 이 둘 모두를 사용할 수 없을 경우, 사용자에게 명확한 안내 텍스트로 `/setup-files` 명령을 치라는 알림을 남깁니다.

개인이 권한을 취소(revoke)할 땐 자신의 슬롯에 한해서만 데이터가 지워집니다. 특정 사용자의 토큰 권한에서 401이나 403 인증 실패 오류가 날 때도 그 사람의 캐시 정보만을 파기할 뿐입니다. 요약하자면 사용자들은 절대 서로에게 영향을 주지 않습니다.

---

## 문제 해결 (Troubleshooting)

**"hola" 전송 후에도 봇이 묵묵부답입니다.**

1. Google Cloud 콘솔에서 Pub/Sub 구독(Subscription)에 처리되지 않은(undelivered) 메시지가 남아있는지 확인합니다.
   메시지가 쌓여 있다면 Hermes의 인증 절차에 오류가 있는 것입니다 — `GOOGLE_CHAT_SERVICE_ACCOUNT_JSON` 값이 정확한지 확인하고, 구독의 `Pub/Sub 구독자(Pub/Sub Subscriber)` 명단에 사용된 SA(서비스 계정)가 있는지 살펴보세요.
2. 구독에 누적된 메시지가 없다면 Google Chat 시스템이 이벤트를 주제(topic)로 제대로 전송(publishing)하지 못하는 상태입니다.
   **주제(topic)**의 IAM 바인딩 상태를 검토하세요:
   `chat-api-push@system.gserviceaccount.com`은 반드시 `Pub/Sub 게시자(Pub/Sub Publisher)` 역할을 지녀야만 합니다.
3. `hermes gateway` 명령 실행 로그에서 `[GoogleChat] Connected` 라인이 출력되었는지 살펴보세요. 만약 `[GoogleChat] Config validation failed`이 찍혀 있다면 어느 환경 변수에서 문제를 일으켰는지 그 에러 로그가 함께 답을 알려줍니다.

**봇이 대답을 하긴 하지만, 에이전트의 답변 내용이 아닌 에러 메시지만 노출됩니다.**

로그에서 `[GoogleChat] Pub/Sub stream died` 오류가 발생했는지 확인하세요 — 이 문구가 계속 반복된다면 사용하는 SA 자격 증명이 변경(rotated)되었거나 관련된 구독 서비스가 삭제되었을 수도 있습니다. 10회의 재접속 시도 이후에도 실패하면 어댑터 스스로 치명적인 상태(fatal)라 결론짓습니다.

**봇의 모든 답변 메시지마다 "403 Forbidden" 에러를 내뱉습니다.**

이는 봇이 속해있던 스페이스에서 강제 퇴장(remove)을 당했거나, 아니면 Chat API 콘솔 내에서 접속 승인을 취소한 경우에 일어납니다. 해당 스페이스에서 봇을 다시 초대하여 설치하세요 (새로운 `ADDED_TO_SPACE` 이벤트가 발생하며 정상적인 기능이 곧장 회복될 것입니다).

**"Rate limit hit" (요청 횟수 제한 도달) 경고가 너무 많이 보입니다.**

Chat API의 기본 할당량은 하나의 스페이스당 1분에 60개 메시지 처리로 설정되어 있습니다. 귀하의 에이전트가 이를 상회할 만큼의 쉴 틈 없는 응답(streaming)을 발생시킨다면, 어댑터는 지수 백오프(exponential backoff) 방식으로 재시도를 하게 될 것입니다 — 그러나 사용자가 체감하는 지연 시간은 존재할 수밖에 없습니다. 해결책으로 에이전트의 답변을 조금 더 짧고 요약된 형태로 바꾸거나 GCP 콘솔에서 API 할당량을 증대하는 방법이 있습니다.

**봇이 파일을 주진 않고 "/setup-files"를 시작하라는 안내 문구만 계속 반복합니다.**

이는 명령을 내린 사용자의 고유한 OAuth 토큰이 존재하지 않고, 예전 방식의 단일 사용자 폴백 파일(legacy fallback) 또한 찾을 수 없는 경우입니다. 해결책으로 해당 사용자가 봇과의 1:1 대화(DM)에서 `/setup-files`를 전송하여 앞선 매뉴얼 10단계에 서술된 과정을 밟으면 됩니다. 토큰 교환 절차가 제대로 완료되면 게이트웨이를 따로 재시작하지 않아도 사용자의 다음 파일 전송 요구부터 네이티브 방식으로 곧장 업로드를 수행할 수 있습니다.

**`/setup-files start` 명령을 입력했더니 "No client credentials stored." 라고 합니다.**

*현재 켜져 있는 프로필 정보 기준*으로 단 1회 수행해야 할 초기 설정 단계가 제대로 끝나지 않은 것입니다 (클라이언트 시크릿은 각 프로필별로 따로 기록되며, 다른 프로필 아래에서 이뤄진 작업을 이쪽 프로필이 발견하지 못합니다). 현재 구동 중인 게이트웨이의 프로필 환경 하에서 터미널 접속 후 다음 명령을 수행하세요:

```bash
# 기본 프로필 기준:
python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json

# 특정 이름을 쓴 프로필 기준:
hermes -p <profile> python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json
```

위 과정이 다 끝났다면 `/setup-files start`를 통해 작업을 재개하세요.

**`/setup-files <복사한_URL>` 에 대해 "Token exchange failed."(토큰 교환 실패) 에러가 발생합니다.**

사용된 인증 코드는 단 한 번만 쓸 수 있으며 이마저도 유효 수명(통상 수 분)이 대단히 짧습니다. `/setup-files start` 명령을 입력하여 새로운 URL을 받은 뒤 다시 시도하세요.

---

## 보안 고려사항

- **서비스 계정(Service Account) 범위**: 이 어댑터는 `chat.bot` 과 `pubsub` 권한(scopes)을 요구합니다. 그러나 실제로 힘을 실어주는 것은 IAM의 적용 형태입니다 — 생성한 SA에게 최소한의 권한만을 허용하세요 (구독 내의 `roles/pubsub.subscriber` 및 `roles/pubsub.viewer` 역할만 주면 됩니다). 즉 전체 프로젝트나 조직(org-level) 단위의 Pub/Sub 역할을 남용해서 줘선 안 됩니다.
- **첨부파일 다운로드 경로 보호**: Hermes는 SA(서비스 계정) 베어러 토큰을 첨부할 대상 호스트 도메인이 오직 구글이 운영하는 곳인지(`googleapis.com`, `drive.google.com`, `lh[3-6].googleusercontent.com` 및 약간의 부가 도메인) 그 짧은 허용 목록 내에서만 판단합니다. 조건에 부합하지 않은 다른 외부 호스트로는 HTTP 요청을 시작도 하기 전 미리 거부하여 차단합니다. 이는 악의적으로 꾸며진 이벤트가 베어러 토큰 정보를 GCE의 메타데이터 서비스 외부 등으로 빼돌리려는 SSRF(Server-Side Request Forgery) 공격 시나리오를 원천 봉쇄합니다.
- **민감 정보 감추기(Redaction)**: 서비스 계정(SA)의 주요 이메일 정보, 그리고 각종 구독(subscription) 및 주제(topic) 접속 경로 내역들은 `agent/redact.py` 규칙에 따라 기본 로그 화면에서 가려집니다(stripped). 디버깅용 패킷 내역 덤프 기능(`GOOGLE_CHAT_DEBUG_RAW=1`) 구동 시에도 해당 기록들이 모두 정제 과정을 똑같이 거치며 DEBUG 환경에서 작동합니다.
- **규정 준수**: 이 봇을 각종 준수 의무가 따르는 워크스페이스에 접속시킬 예정이라면 (데이터 저장 공간이나 AI 거버넌스 정책의 영향을 받는 공간 등) 앱의 첫 번째 설치 단계 전에 관리자로부터 허가를 우선 취득해야 합니다.
- **사용자 개인의 OAuth 권한 범위**: 사용자별(per-user) 파일 첨부 플로우 과정에서 오직 `chat.messages.create` 단 한 개의 권한만을 묻게 됩니다 — 이는 사용자의 권한 범위를 오로지 `media.upload`를 하고 해당 작업 이후에 이어지는 `messages.create`에만 국한시키기 위함입니다. 받아낸 토큰은 아무런 2차 가공 없는 평문 JSON 형태로 `~/.hermes/google_chat_user_tokens/<sanitized_email>.json` 에 보관됩니다 (이는 SA 키 파일을 다루던 방식과 완벽히 동일하며 오직 운영 체제의 파일 관리 시스템의 권한(permissions)으로만 이를 보호합니다). 이 각각의 토큰은 특정한 딱 한 사람만 통제력을 갖고 있으며 소유권 철회(revoke)도 해당 사용자의 범위 안에서만 작동합니다.
