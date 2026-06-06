---
sidebar_position: 13
title: "Webhooks"
description: "GitHub, GitLab 및 기타 서비스의 이벤트를 수신하여 자동 Hermes 에이전트 실행 트리거"
---

# Webhooks

외부 서비스(GitHub, GitLab, JIRA, Stripe 등)로부터 이벤트를 수신하여 자동으로 Hermes 에이전트 실행을 트리거합니다. 웹훅 어댑터는 HTTP 서버를 실행하여 POST 요청을 수신하고, HMAC 서명을 검증하며, 페이로드를 에이전트 프롬프트로 변환한 후, 응답을 소스 또는 구성된 다른 플랫폼으로 전송합니다.

에이전트는 이벤트를 처리하고 PR에 주석을 달거나, Telegram/Discord로 메시지를 보내거나, 결과를 로깅하는 방식으로 응답할 수 있습니다.

## 비디오 튜토리얼

<div style={{position: 'relative', width: '100%', aspectRatio: '16 / 9', marginBottom: '1.5rem'}}>
  <iframe
    src="https://www.youtube.com/embed/WNYe5mD4fY8"
    title="Hermes Agent — Webhooks Tutorial"
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', border: 0}}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowFullScreen
  />
</div>

---

## 빠른 시작

1. `hermes gateway setup` 또는 환경 변수를 통해 활성화
2. `config.yaml`에 라우트를 정의**하거나** `hermes webhook subscribe`로 동적 생성
3. 서비스가 `http://your-server:8644/webhooks/<route-name>`를 가리키도록 설정

---

## 설정

웹훅 어댑터를 활성화하는 방법은 두 가지가 있습니다.

### 설정 마법사 사용

```bash
hermes gateway setup
```

프롬프트에 따라 웹훅을 활성화하고, 포트를 설정하며, 전역 HMAC 시크릿을 설정합니다.

### 환경 변수 사용

`~/.hermes/.env`에 추가:

```bash
WEBHOOK_ENABLED=true
WEBHOOK_PORT=8644        # 기본값
WEBHOOK_SECRET=your-global-secret
```

### 서버 검증

게이트웨이가 실행된 후:

```bash
curl http://localhost:8644/health
```

예상되는 응답:

```json
{"status": "ok", "platform": "webhook"}
```

---

## 라우트 구성 {#configuring-routes}

라우트는 서로 다른 웹훅 소스가 어떻게 처리될지 정의합니다. 각 라우트는 `config.yaml`의 `platforms.webhook.extra.routes` 아래에 이름이 지정된 항목입니다.

### 라우트 속성

| 속성 | 필수 | 설명 |
|----------|----------|-------------|
| `events` | 아니오 | 수락할 이벤트 유형 목록 (예: `["pull_request"]`). 비어있으면 모든 이벤트를 수락합니다. 이벤트 유형은 `X-GitHub-Event`, `X-GitLab-Event` 헤더 또는 페이로드의 `event_type`에서 읽어옵니다. |
| `secret` | **예** | 서명 검증을 위한 HMAC 시크릿. 라우트에 설정되지 않으면 전역 `secret`으로 대체됩니다. 테스트 전용으로만 `"INSECURE_NO_AUTH"` 설정 가능 (검증 건너뜀). |
| `prompt` | 아니오 | 도트(.) 표기법 페이로드 접근이 포함된 템플릿 문자열 (예: `{pull_request.title}`). 생략된 경우, 프롬프트에 전체 JSON 페이로드가 덤프됩니다. |
| `skills` | 아니오 | 에이전트 실행을 위해 로드할 스킬 이름 목록. |
| `deliver` | 아니오 | 응답 전송 위치: `github_comment`, `telegram`, `discord`, `slack`, `signal`, `sms`, `whatsapp`, `matrix`, `mattermost`, `homeassistant`, `email`, `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot`, 또는 `log` (기본값). |
| `deliver_extra` | 아니오 | 추가 전달 구성 — 키는 `deliver` 유형(예: `repo`, `pr_number`, `chat_id`)에 따라 다릅니다. 값은 `prompt`와 동일한 `{dot.notation}` 템플릿을 지원합니다. |
| `deliver_only` | 아니오 | `true`인 경우, 에이전트 실행을 완전히 건너뜁니다 — 렌더링된 `prompt` 템플릿이 그대로 문자열 메시지가 되어 전송됩니다. LLM 비용이 없고 1초 이내에 전달됩니다. 사용 사례는 [직접 전달 모드](#direct-delivery-mode)를 참조하세요. `deliver`를 진짜 대상(아닌 `log`)으로 설정해야 합니다. |

### 전체 예제

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644
      secret: "global-fallback-secret"
      routes:
        github-pr:
          events: ["pull_request"]
          secret: "github-webhook-secret"
          prompt: |
            이 Pull Request를 검토하세요:
            저장소: {repository.full_name}
            PR #{number}: {pull_request.title}
            작성자: {pull_request.user.login}
            URL: {pull_request.html_url}
            Diff URL: {pull_request.diff_url}
            동작: {action}
          skills: ["github-code-review"]
          deliver: "github_comment"
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
        deploy-notify:
          events: ["push"]
          secret: "deploy-secret"
          prompt: "{repository.full_name} 저장소의 {ref} 브랜치에 새 푸시 발생: {head_commit.message}"
          deliver: "telegram"
```

### 프롬프트 템플릿

프롬프트는 도트(.) 표기법을 사용하여 웹훅 페이로드 내부의 중첩된 필드에 접근합니다:

- `{pull_request.title}`은 `payload["pull_request"]["title"]`로 변환됩니다.
- `{repository.full_name}`은 `payload["repository"]["full_name"]`으로 변환됩니다.
- `{__raw__}` — 들여쓰기된 JSON 형태로 **전체 페이로드**를 덤프하는 특수 토큰입니다 (4000자로 잘림). 에이전트가 전체 맥락을 파악해야 하는 모니터링 알림 또는 일반 웹훅에 유용합니다.
- 누락된 키는 `{key}` 문자열 그대로 남습니다 (오류 없음).
- 중첩된 딕셔너리 및 목록은 JSON 형식으로 직렬화되며 2000자로 잘립니다.

일반 템플릿 변수와 `{__raw__}`를 혼합해서 사용할 수 있습니다:

```yaml
prompt: "PR #{pull_request.number} by {pull_request.user.login}: {__raw__}"
```

라우트에 구성된 `prompt` 템플릿이 없으면, 전체 페이로드가 들여쓰기된 JSON 형태로 덤프됩니다(4000자로 잘림).

동일한 도트 표기법 템플릿이 `deliver_extra` 값에서도 작동합니다.

### 포럼 토픽 전송

웹훅 응답을 Telegram으로 전송할 때, `deliver_extra`에 `message_thread_id` (또는 `thread_id`)를 포함하여 특정 포럼 토픽을 지정할 수 있습니다:

```yaml
webhooks:
  routes:
    alerts:
      events: ["alert"]
      prompt: "경고: {__raw__}"
      deliver: "telegram"
      deliver_extra:
        chat_id: "-1001234567890"
        message_thread_id: "42"
```

`deliver_extra`에 `chat_id`가 제공되지 않으면 전달 대상 플랫폼에 구성된 홈 채널을 사용합니다.

---

## GitHub PR 코드 리뷰 (단계별) {#github-pr-review}

모든 Pull Request에 대한 자동 코드 리뷰를 설정하는 워크스루입니다.

### 1. GitHub에서 웹훅 생성

1. 저장소 → **Settings** → **Webhooks** → **Add webhook**으로 이동
2. **Payload URL**을 `http://your-server:8644/webhooks/github-pr`로 설정
3. **Content type**을 `application/json`으로 설정
4. **Secret**을 라우트 설정(예: `github-webhook-secret`)과 동일하게 설정
5. **Which events?** 아래에서 **Let me select individual events**를 선택하고 **Pull requests**를 체크
6. **Add webhook** 클릭

### 2. 라우트 구성 추가

위 예제와 같이 `github-pr` 라우트를 `~/.hermes/config.yaml`에 추가합니다.

### 3. `gh` CLI 인증 상태 확인

`github_comment` 전달 유형은 `gh` CLI를 사용하여 댓글을 등록합니다:

```bash
gh auth login
```

### 4. 테스트

저장소에서 Pull Request를 오픈합니다. 웹훅이 발생하고 Hermes가 이벤트를 처리한 뒤 PR에 리뷰 댓글을 등록합니다.

---

## GitLab 웹훅 설정 {#gitlab-webhook-setup}

GitLab 웹훅도 유사하게 작동하지만 다른 인증 메커니즘을 사용합니다. GitLab은 HMAC가 아닌 일반 문자열인 `X-Gitlab-Token` 헤더로 시크릿을 전송합니다 (정확한 문자열 일치).

### 1. GitLab에서 웹훅 생성

1. 프로젝트 → **Settings** → **Webhooks**로 이동
2. **URL**을 `http://your-server:8644/webhooks/gitlab-mr`로 설정
3. **Secret token** 입력
4. **Merge request events**(및 원하는 다른 이벤트들)를 선택
5. **Add webhook** 클릭

### 2. 라우트 구성 추가

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        gitlab-mr:
          events: ["merge_request"]
          secret: "your-gitlab-secret-token"
          prompt: |
            이 Merge Request를 검토하세요:
            프로젝트: {project.path_with_namespace}
            MR !{object_attributes.iid}: {object_attributes.title}
            작성자: {object_attributes.last_commit.author.name}
            URL: {object_attributes.url}
            동작: {object_attributes.action}
          deliver: "log"
```

---

## 전송 옵션 (Delivery Options) {#delivery-options}

`deliver` 필드는 에이전트가 웹훅 이벤트를 처리한 후 응답이 전달되는 곳을 제어합니다.

| 전달 타입 | 설명 |
|-------------|-------------|
| `log` | 응답을 게이트웨이 로그 출력에 기록합니다. 기본값이며 테스트에 유용합니다. |
| `github_comment` | `gh` CLI를 통해 PR/Issue의 댓글로 응답을 게시합니다. `deliver_extra.repo` 및 `deliver_extra.pr_number`가 필수적입니다. 게이트웨이 호스트에 `gh` CLI가 설치되고 인증되어야 합니다 (`gh auth login`). |
| `telegram` | Telegram으로 응답을 전송합니다. 홈 채널을 사용하거나 `deliver_extra`의 `chat_id`를 지정합니다. |
| `discord` | Discord로 응답을 전송합니다. 홈 채널을 사용하거나 `deliver_extra`의 `chat_id`를 지정합니다. |
| `slack` | Slack으로 응답을 전송합니다. 홈 채널을 사용하거나 `deliver_extra`의 `chat_id`를 지정합니다. |
| `signal` | Signal로 응답을 전송합니다. 홈 채널을 사용하거나 `deliver_extra`의 `chat_id`를 지정합니다. |
| `sms` | Twilio를 통해 SMS 응답을 전송합니다. 홈 채널을 사용하거나 `deliver_extra`의 `chat_id`를 지정합니다. |
| `whatsapp` | WhatsApp으로 응답을 전송합니다. 홈 채널을 사용하거나 `deliver_extra`의 `chat_id`를 지정합니다. |
| `matrix` | Matrix로 응답을 전송합니다. 홈 채널을 사용하거나 `deliver_extra`의 `chat_id`를 지정합니다. |
| `mattermost` | Mattermost로 응답을 전송합니다. 홈 채널을 사용하거나 `deliver_extra`의 `chat_id`를 지정합니다. |
| `homeassistant` | Home Assistant로 응답을 전송합니다. 홈 채널을 사용하거나 `deliver_extra`의 `chat_id`를 지정합니다. |
| `email` | 이메일로 응답을 전송합니다. 홈 채널을 사용하거나 `deliver_extra`의 `chat_id`를 지정합니다. |
| `dingtalk` | DingTalk으로 응답을 전송합니다. 홈 채널을 사용하거나 `deliver_extra`의 `chat_id`를 지정합니다. |
| `feishu` | Feishu/Lark로 응답을 전송합니다. 홈 채널을 사용하거나 `deliver_extra`의 `chat_id`를 지정합니다. |
| `wecom` | WeCom으로 응답을 전송합니다. 홈 채널을 사용하거나 `deliver_extra`의 `chat_id`를 지정합니다. |
| `weixin` | Weixin (WeChat)으로 응답을 전송합니다. 홈 채널을 사용하거나 `deliver_extra`의 `chat_id`를 지정합니다. |
| `bluebubbles` | BlueBubbles (iMessage)로 응답을 전송합니다. 홈 채널을 사용하거나 `deliver_extra`의 `chat_id`를 지정합니다. |

플랫폼 간 교차 전송을 위해서는 대상 플랫폼도 게이트웨이에서 활성화되고 연결되어 있어야 합니다. `deliver_extra`에 `chat_id`가 지정되지 않으면 해당 플랫폼의 구성된 홈 채널로 전송됩니다.

---

## 직접 전달 모드 (Direct Delivery Mode) {#direct-delivery-mode}

기본적으로 웹훅의 모든 POST 요청은 에이전트 실행을 유도합니다 — 페이로드가 프롬프트가 되고, 에이전트가 처리하며 응답을 전달합니다. 이는 매 이벤트마다 LLM 토큰 비용을 소모합니다.

**일반적인 단순 알림 푸시** (사고(reasoning) 불필요, 에이전트 루프 없음, 단순 메시지 전송 목적) 의 경우 라우트에 `deliver_only: true`를 설정하세요. 렌더링된 `prompt` 템플릿이 실제 메시지 내용으로 활용되고 어댑터가 이를 설정된 대상 플랫폼에 직접 전송합니다.

### 언제 직접 전달을 써야 하는가

- **외부 서비스 푸시** — Supabase/Firebase 웹훅 데이터베이스 변경 시 → 즉시 Telegram으로 사용자에 알림
- **모니터링 경고** — Datadog/Grafana 알림 웹훅 → Discord 채널에 푸시
- **에이전트 간 핑** — 에이전트 A가 에이전트 B 사용자에게 장기 작업이 완료됨을 알림
- **백그라운드 작업 완료** — 크론 작업 완료 시 → Slack에 결과 보고

장점:

- **LLM 토큰 0소모** — 에이전트를 호출하지 않음
- **1초 이내 전달** — 단일 어댑터 호출, 사고 루프 불필요
- **에이전트 모드와 동일한 보안** — HMAC 인증, 속도 제한, 멱등성 및 본문 크기 제한 그대로 적용
- **동기 응답** — 성공 시 `200 OK`, 대상에서 거부할 경우 `502`가 반환되어 상위 시스템이 지능적으로 재시도 가능

### 예제: Supabase에서 Telegram 푸시

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644
      secret: "global-secret"
      routes:
        antenna-matches:
          secret: "antenna-webhook-secret"
          deliver: "telegram"
          deliver_only: true
          prompt: "🎉 새로운 매칭: {match.user_name}님과 매칭되었습니다!"
          deliver_extra:
            chat_id: "{match.telegram_chat_id}"
```

Supabase Edge Function은 HMAC-SHA256으로 페이로드에 서명하고 `https://your-server:8644/webhooks/antenna-matches`에 POST합니다. 웹훅 어댑터는 서명을 검증하고, 템플릿을 페이로드로 채우고, Telegram에 전송하며, `200 OK`를 반환합니다.

### 예제: CLI로 동적 구독 추가

```bash
hermes webhook subscribe antenna-matches \
  --deliver telegram \
  --deliver-chat-id "123456789" \
  --deliver-only \
  --prompt "🎉 새로운 매칭: {match.user_name}님과 매칭되었습니다!" \
  --description "Antenna match notifications"
```

### 응답 코드

| 상태 코드 | 의미 |
|--------|---------|
| `200 OK` | 성공적으로 전송. Body: `{"status": "delivered", "route": "...", "target": "...", "delivery_id": "..."}` |
| `200 OK` (status=duplicate) | 멱등성 만료 시간(1시간) 이내에 동일한 `X-GitHub-Delivery` ID 도착. 재전송하지 않음. |
| `401 Unauthorized` | HMAC 서명이 유효하지 않거나 없음. |
| `400 Bad Request` | 잘못된 JSON 본문. |
| `404 Not Found` | 알 수 없는 라우트 이름. |
| `413 Payload Too Large` | 본문 크기가 `max_body_bytes`를 초과. |
| `429 Too Many Requests` | 라우트 속도 제한 초과. |
| `502 Bad Gateway` | 대상 어댑터가 메시지를 거절하거나 예외 발생. 에러 내용은 게이트웨이에 기록되지만 어댑터 내부 정보 유출을 막기 위해 Body에는 일반 `Delivery failed` 반환. |

### 주의사항(Gotchas)

- `deliver_only: true`인 경우 `deliver`에 반드시 진짜 대상이 설정되어야 합니다. `deliver: log` (또는 미기재 시) 설정은 서버 구동 시 잘못된 설정으로 취급되어 기동이 차단됩니다.
- 직접 전달 모드에서는 `skills` 필드가 무시됩니다 (에이전트를 안 띄우므로 주입할 스킬 개념이 없음).
- 템플릿 렌더링 시 일반적인 `{__raw__}` 토큰 등 에이전트 모드와 동일한 `{dot.notation}` 문법이 사용됩니다.
- 멱등성은 `X-GitHub-Delivery` / `X-Request-ID` 헤더를 활용해 처리됩니다 — 동일 ID로 재시도 시 `status=duplicate`로 처리되며 재전송되지 않습니다.

---

## 동적 구독 (CLI) {#dynamic-subscriptions}

`config.yaml`에 지정된 정적 라우트 외에도, `hermes webhook` CLI 커맨드로 동적으로 웹훅 구독(subscription)을 생성할 수 있습니다. 에이전트 스스로 이벤트 기반 트리거 설정이 필요한 상황에서 특히 유용합니다.

### 구독 생성

```bash
hermes webhook subscribe github-issues \
  --events "issues" \
  --prompt "새 이슈 #{issue.number}: {issue.title}\nBy: {issue.user.login}\n\n{issue.body}" \
  --deliver telegram \
  --deliver-chat-id "-100123456789" \
  --description "GitHub 이슈 트리아지"
```

이 명령어는 생성된 웹훅 URL과 자동 생성된 HMAC 시크릿을 출력합니다. 사용하는 서비스가 해당 URL로 POST하도록 구성하세요.

### 구독 목록 확인

```bash
hermes webhook list
```

### 구독 제거

```bash
hermes webhook remove github-issues
```

### 구독 테스트

```bash
hermes webhook test github-issues
hermes webhook test github-issues --payload '{"issue": {"number": 42, "title": "Test"}}'
```

### 동적 구독 원리

- 구독 정보는 `~/.hermes/webhook_subscriptions.json` 파일에 저장됩니다.
- 웹훅 어댑터는 수신 요청마다 이 파일을 동적으로 로드합니다 (mtime 체크로 오버헤드 최소화).
- `config.yaml` 내 정적 라우트가 동적 라우트보다 항상 우선합니다.
- 동적 구독도 정적 라우트와 동일한 형식을 지니며 같은 기능들(events, prompt templates, skills, delivery)을 지원합니다.
- 게이트웨이 재시작이 필요 없습니다 — 등록 즉시 반영됩니다.

### 에이전트 기반 구독

에이전트가 `webhook-subscriptions` 스킬을 사용하여 터미널을 통해 스스로 구독을 생성할 수 있습니다. 에이전트에게 "GitHub 이슈 관련 웹훅 설정해 줘"라고 하면 적절한 `hermes webhook subscribe` 명령을 수행합니다.

---

## 보안 {#security}

웹훅 어댑터에는 다양한 계층의 보안이 적용되어 있습니다.

### HMAC 서명 검증

어댑터는 다음과 같이 소스에 알맞은 방식으로 수신 서명을 검증합니다:

- **GitHub**: `X-Hub-Signature-256` 헤더 — `sha256=` 접두어가 붙은 HMAC-SHA256 해시값.
- **GitLab**: `X-Gitlab-Token` 헤더 — 단순 문자열 시크릿 일치 비교.
- **일반 (Generic)**: `X-Webhook-Signature` 헤더 — 순수 HMAC-SHA256 해시값.

라우트에 시크릿이 구성되어 있으나 알맞은 서명 헤더가 없을 시 요청은 거부됩니다.

### 시크릿 필수 규칙

모든 라우트는 자체적인 혹은 전역적인 시크릿 값을 반드시 가져야 합니다. 이를 누락할 시 서버가 실행되지 않고 오류가 발생합니다. 개발 환경과 테스트 목적으로만 사용할 때는 값을 `"INSECURE_NO_AUTH"`로 지정하여 검증을 스킵할 수 있습니다.

`INSECURE_NO_AUTH`는 오직 게이트웨이가 루프백 호스트(`127.0.0.1`, `localhost`, `::1`)로 구성될 때만 사용될 수 있습니다. `0.0.0.0` 이나 외부 IP로 바인딩된 환경에서는 실수로 퍼블릭 네트워크에 미인증 엔드포인트가 열리지 않도록 기동 자체를 막습니다.

### 속도 제한(Rate Limiting)

기본적으로 각 라우트는 분당 **최대 30회 요청**의 속도 제한(고정 윈도우 방식)을 갖습니다. 전체적으로 속도를 재정의할 수 있습니다:

```yaml
platforms:
  webhook:
    extra:
      rate_limit: 60  # 분당 요청 수
```

제한을 넘으면 `429 Too Many Requests`가 반환됩니다.

### 멱등성 (Idempotency)

전달 ID(`X-GitHub-Delivery`, `X-Request-ID` 또는 백업용 타임스탬프 등)는 **1시간** 동안 캐시로 저장됩니다. 동일한 ID로 수신된 중복건의 경우 정상 상태(`200 OK`)를 돌려주고 실질적으로는 이벤트를 스킵하여 에이전트의 다중 호출을 방지합니다.

### 본문 크기 한도

1MB를 초과하는 데이터는 즉각 차단됩니다. 필요시 이를 조정할 수 있습니다:

```yaml
platforms:
  webhook:
    extra:
      max_body_bytes: 2097152  # 2 MB
```

### 프롬프트 인젝션 (Prompt Injection) 방어 위험

:::warning
웹훅 페이로드는 해커에 의해 조작된 PR 이름, 커밋 코멘트 등을 통해 악성 프롬프트 인젝션을 유발할 가능성이 있습니다. 악용 소지가 있으니 퍼블릭에 노출 시 샌드박스(Docker, VM) 내에서 운영하시길 권장합니다. 격리된 Docker나 SSH 터미널 백엔드 활용을 고려하십시오.
:::

---

## 문제 해결 {#troubleshooting}

### 웹훅이 전혀 도착하지 않음

- 포트가 제대로 외부에 열려 있는지, 접근 가능한지 점검합니다.
- 방화벽 설정 확인 — 기본 포트 `8644`(또는 사용자 설정 포트)가 개방 상태여야 합니다.
- URL 경로가 정확한지 확인하세요: `http://your-server:8644/webhooks/<route-name>`
- `/health`를 호출하여 서버가 제대로 작동 중인지 체크합니다.

### 서명 검증 에러 (Signature Validation Failed)

- 타사 서비스의 설정 값과 게이트웨이 내의 시크릿이 정확히 매치하는지 살핍니다.
- GitHub의 시크릿은 HMAC 기반임을 인지하고, `X-Hub-Signature-256` 헤더를 점검하세요.
- GitLab의 경우는 단순 토큰 매치 방식을 사용하므로 `X-Gitlab-Token`을 확인하세요.
- 게이트웨이 로그 중 `Invalid signature` 문구를 확인합니다.

### 이벤트가 지속적으로 스킵 (Ignore) 처리됨

- 보내진 이벤트 종류가 라우트의 `events` 옵션에 등록되었는지 봅니다.
- GitHub의 이벤트 값들은 `pull_request`, `push`, `issues` (`X-GitHub-Event` 헤더 값) 등과 같습니다.
- GitLab은 `merge_request`, `push` (`X-GitLab-Event` 헤더 값)을 씁니다.
- 등록 값을 기입하지 않으면 모든 이벤트를 받습니다.

### 에이전트가 응답하지 않음

- 백그라운드 모드가 아닌 직접 콘솔 포그라운드로 로그를 봅니다: `hermes gateway run`
- 템플릿 변수가 제대로 채워지고 있는지 분석합니다.
- 전송 목표 타깃(Discord, Slack 등) 플랫폼이 올바르게 연결되어 작동 중인지 확인합니다.

### 중복 응답 전송 현상

- 멱등성 처리가 켜져 있는지, 타사 서비스에서 전달 ID(`X-GitHub-Delivery`나 `X-Request-ID`)가 오는지 확인합니다.
- 전달된 ID 캐싱 기간은 1시간입니다.

### `gh` CLI 명령 실패 (GitHub Comment 관련)

- 게이트웨이가 작동하는 호스트 기기 내 터미널에서 `gh auth login`을 진행하세요.
- 사용 인증 계정이 해당 저장소의 수정 권한을 가지는지 봅니다.
- 환경 변수 PATH상에 `gh` 명령이 올라가 있는지 파악합니다.

---

## 환경 변수 {#environment-variables}

| 변수 이름 | 설명 | 기본값 |
|----------|-------------|---------|
| `WEBHOOK_ENABLED` | 웹훅 플랫폼 어댑터 활성화 여부 | `false` |
| `WEBHOOK_PORT` | HTTP 수신용 게이트웨이 포트 | `8644` |
| `WEBHOOK_SECRET` | 전역 HMAC 시크릿 (각 라우트에 시크릿이 미설정시 기본값으로 사용) | _(없음)_ |
