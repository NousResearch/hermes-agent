---
sidebar_position: 13
title: "웹훅"
description: "GitHub, GitLab 및 기타 서비스에서 이벤트를 받아 Hermes 에이전트 실행을 트리거합니다"
---

# 웹훅

외부 서비스(GitHub, GitLab, JIRA, Stripe 등)에서 이벤트를 받아 Hermes 에이전트 실행을 자동으로 트리거합니다. 웹훅 어댑터는 POST 요청을 수신하고, HMAC 서명을 검증하며, 페이로드를 에이전트 프롬프트로 변환하고, 응답을 이벤트 출처 또는 다른 구성된 플랫폼으로 전달하는 HTTP 서버를 실행합니다.

에이전트는 이벤트를 처리한 뒤 PR에 댓글을 게시하거나 Telegram/Discord로 메시지를 보내거나 결과를 로그에 기록할 수 있습니다.

## 동영상 튜토리얼

<div style={{position: 'relative', width: '100%', aspectRatio: '16 / 9', marginBottom: '1.5rem'}}>
  <iframe
    src="https://www.youtube.com/embed/WNYe5mD4fY8"
    title="Hermes Agent — 웹훅 튜토리얼"
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', border: 0}}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowFullScreen
  />
</div>

---

## 빠른 시작

1. `hermes gateway setup` 또는 환경 변수로 활성화합니다
2. `config.yaml`에서 라우트를 정의하거나 `hermes webhook subscribe`로 동적으로 생성합니다
3. 서비스를 `http://your-server:8644/webhooks/<route-name>`에 연결합니다

---

## 설정

웹훅 어댑터를 활성화하는 방법은 두 가지입니다.

### 설정 마법사 사용

```bash
hermes gateway setup
```

프롬프트에 따라 웹훅을 활성화하고, 포트를 설정하고, 전역 HMAC 시크릿을 지정합니다.

### 환경 변수 사용

`~/.hermes/.env`에 다음을 추가합니다.

```bash
WEBHOOK_ENABLED=true
WEBHOOK_PORT=8644        # default
WEBHOOK_SECRET=your-global-secret
```

### 서버 확인

게이트웨이가 실행되면 다음을 실행합니다.

```bash
curl http://localhost:8644/health
```

예상 응답:

```json
{"status": "ok", "platform": "webhook"}
```

---

## 라우트 구성 {#configuring-routes}

라우트는 서로 다른 웹훅 소스를 처리하는 방법을 정의합니다. 각 라우트는 `config.yaml`의 `platforms.webhook.extra.routes` 아래에 이름이 지정된 항목으로 작성합니다.

### 라우트 속성

| 속성 | 필수 | 설명 |
|----------|----------|-------------|
| `events` | 아니요 | 수락할 이벤트 유형 목록입니다(예: `["pull_request"]`). 비어 있으면 모든 이벤트를 수락합니다. 이벤트 유형은 페이로드의 `X-GitHub-Event`, `X-GitLab-Event` 또는 `event_type`에서 읽습니다. |
| `secret` | **예** | 서명 검증에 사용할 HMAC 시크릿입니다. 라우트에 설정하지 않으면 전역 `secret`을 사용합니다. 테스트 목적으로만 `"INSECURE_NO_AUTH"`로 설정할 수 있습니다(검증을 건너뜀). |
| `profile` | 아니요 | `gateway.multiplex_profiles`가 활성화된 경우 이 라우트를 실행할 권한이 있는 프로필입니다. 기본 프로필 전용 라우트로 사용하려면 생략하고, 프로필 이름(예: `coder`)을 지정하면 라우트와 시크릿이 `/p/coder/webhooks/<route>`에 연결됩니다. |
| `prompt` | 아니요 | 점 표기법으로 페이로드에 접근하는 템플릿 문자열입니다(예: `{pull_request.title}`). 생략하면 전체 JSON 페이로드가 프롬프트에 기록됩니다. 페이로드 필드는 신뢰할 수 없는 입력입니다. [인증되었다고 신뢰할 수 있는 것은 아닙니다](#authenticated-does-not-mean-trusted)를 참고하세요. |
| `filters` | 아니요 | 인증/본문/이벤트 필터링 후, 에이전트 또는 직접 전달 작업 전에 평가되는 선언적 페이로드 필터입니다. 일치하지 않으면 HTTP 200과 함께 `{"status":"ignored","reason":"filter"}`를 반환합니다. |
| `script` | 아니요 | `~/.hermes/scripts/` 아래의 필터/변환 스크립트입니다. 웹훅 페이로드는 stdin으로 JSON 형식으로 전달됩니다. JSON 객체 stdout은 템플릿 적용 전에 페이로드를 대체하고, 텍스트 stdout은 `script_output`으로 노출됩니다. stdout이 비어 있거나 `[SILENT]`이거나 종료 코드가 0이 아니면 웹훅을 무시합니다. |
| `skills` | 아니요 | 에이전트 실행에 로드할 스킬 이름 목록입니다. |
| `toolsets` | 아니요 | 이 라우트로 트리거된 실행에 대해 플랫폼 수준 웹훅 도구 세트를 **대체**하는 도구 세트 키 목록입니다(예: `["terminal", "file", "web"]`). 수동 설정 편집만 가능하며 `hermes webhook subscribe`로는 설정할 수 없으므로, 에이전트가 생성한 구독으로 권한이 높은 도구를 스스로 부여할 수 없습니다. 이름은 `platform_toolsets` 항목과 동일한 방식으로 검증됩니다(알 수 없거나 플랫폼에서 제한된 이름은 제거됨). [라우트별 도구 세트](#per-route-toolsets)를 참고하세요. |
| `deliver` | 아니요 | 응답을 보낼 위치입니다: `github_comment`, `telegram`, `discord`, `slack`, `signal`, `sms`, `whatsapp`, `matrix`, `mattermost`, `homeassistant`, `email`, `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot` 또는 `log`(기본값)입니다. |
| `deliver_extra` | 아니요 | 추가 전달 설정입니다. 키는 `deliver` 유형에 따라 다릅니다(예: `repo`, `pr_number`, `chat_id`). 값에는 `prompt`와 동일한 `{dot.notation}` 템플릿을 사용할 수 있습니다. |
| `deliver_only` | 아니요 | `true`이면 에이전트를 완전히 건너뛰고, 렌더링된 `prompt` 템플릿을 그대로 전달할 리터럴 메시지로 사용합니다. LLM 비용이 없고 1초 미만으로 전달됩니다. 사용 사례는 [직접 전달 모드](#direct-delivery-mode)를 참고하세요. `deliver`가 실제 대상이어야 하며 `log`일 수 없습니다. |

### 전체 예시

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
            Review this pull request:
            Repository: {repository.full_name}
            PR #{number}: {pull_request.title}
            Author: {pull_request.user.login}
            URL: {pull_request.html_url}
            Diff URL: {pull_request.diff_url}
            Action: {action}
          skills: ["github-code-review"]
          deliver: "github_comment"
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
        deploy-notify:
          events: ["push"]
          secret: "deploy-secret"
          prompt: "New push to {repository.full_name} branch {ref}: {head_commit.message}"
          filters:
            - field: "ref"
              equals: "refs/heads/main"
          deliver: "telegram"
```

### 페이로드 필터

공급자가 광범위한 이벤트 스트림을 보내지만 일부 페이로드만 에이전트를 깨우거나 `deliver_only` 전달을 트리거해야 할 때 `filters`를 사용합니다. 필터는 서명 검증, 본문 파싱, `events` 처리 후, 프롬프트 렌더링, 멱등성 검사, 에이전트 디스패치 또는 직접 전달 전에 실행됩니다.

```yaml
platforms:
  webhook:
    extra:
      routes:
        todoist:
          events: ["item:updated"]
          secret: "todoist-secret"
          filters:
            - field: "payload.labels"
              contains: "hermes"
            - any:
                - field: "payload.priority"
                  equals: 4
                - field: "payload.project_id"
                  in_file: "~/.hermes/data/todoist/watchlist.json"
          prompt: "Todoist task changed: {payload.content}"
```

지원되는 연산자:

- `exists: true|false`
- `missing: true`
- `equals` / `not_equals`
- 문자열, 목록, 딕셔너리 키에 사용하는 `contains`
- 인라인 목록에 사용하는 `in`
- JSON 배열, JSON 객체(키 사용) 또는 줄바꿈으로 구분된 텍스트 파일에 사용하는 `in_file`
- `regex`
- `all`, `any`, `not` 그룹

필드 경로에는 점 표기법을 사용합니다. `payload.foo`는 최상위 `payload` 객체가 있으면 그 객체에서 읽고, 평면 페이로드라면 웹훅 본문의 루트에서 읽습니다. `event` / `event_type`은 확인된 이벤트 유형과 일치하며, `headers.<Name>`은 요청 헤더를 읽습니다.

### 스크립트 필터 및 변환

선언적 필터만으로 충분하지 않을 때 `script`를 사용합니다. 스크립트는 활성 프로필의 `~/.hermes/scripts/` 아래에 있어야 하며, 상대 경로는 해당 디렉터리를 기준으로 해석되고 디렉터리 밖으로의 경로 탐색은 차단됩니다. `.sh` 및 `.bash` 스크립트는 bash로 실행되고, 그 외 확장자는 현재 Python 인터프리터로 실행됩니다.

라우트 페이로드는 JSON으로 stdin에 전달됩니다.

```python
# ~/.hermes/scripts/todoist-hermes-label.py
import json
import sys

payload = json.load(sys.stdin)
labels = payload.get("payload", {}).get("labels", [])
if "hermes" not in labels:
    print("[SILENT]")
    raise SystemExit(0)

payload["body"] = payload["payload"]["content"]
print(json.dumps(payload))
```

스크립트 결과:

- JSON 객체 stdout은 `prompt` 및 `deliver_extra`에 사용되는 페이로드를 대체합니다.
- JSON이 아닌 텍스트 stdout은 `script_output`으로 페이로드에 추가됩니다.
- stdout이 비어 있거나, 정확히 `[SILENT]`이거나, `{"__hermes_ignore__": true}`이거나, 시간 초과, 스크립트 없음 또는 종료 코드가 0이 아니면 `{"status":"ignored","reason":"script"}`와 함께 HTTP 200을 반환합니다.

### 프롬프트 템플릿

프롬프트는 점 표기법으로 웹훅 페이로드의 중첩 필드에 접근합니다.

- `{pull_request.title}`은 `payload["pull_request"]["title"]`로 해석됩니다.
- `{repository.full_name}`은 `payload["repository"]["full_name"]`으로 해석됩니다.
- `{__raw__}` — 전체 페이로드를 들여쓰기된 JSON으로 기록하는 **특수 토큰**입니다(4000자로 잘림). 모니터링 알림이나 에이전트에 전체 컨텍스트가 필요한 일반 웹훅에 유용합니다.
- 없는 키는 리터럴 `{key}` 문자열로 남습니다(오류 없음).
- 중첩된 딕셔너리와 목록은 JSON으로 직렬화되며 2000자로 잘립니다.

`{__raw__}`를 일반 템플릿 변수와 함께 사용할 수 있습니다.

```yaml
prompt: "PR #{pull_request.number} by {pull_request.user.login}: {__raw__}"
```

라우트에 `prompt` 템플릿이 구성되어 있지 않으면 전체 페이로드가 들여쓰기된 JSON으로 기록됩니다(4000자로 잘림).

동일한 점 표기법 템플릿을 `deliver_extra` 값에도 사용할 수 있습니다.

### 포럼 토픽 전달

웹훅 응답을 Telegram으로 전달할 때 `deliver_extra`에 `message_thread_id`(또는 `thread_id`)를 포함하면 특정 포럼 토픽을 대상으로 지정할 수 있습니다.

```yaml
webhooks:
  routes:
    alerts:
      events: ["alert"]
      prompt: "Alert: {__raw__}"
      deliver: "telegram"
      deliver_extra:
        chat_id: "-1001234567890"
        message_thread_id: "42"
```

`deliver_extra`에 `chat_id`가 없으면 대상 플랫폼에 구성된 홈 채널로 전달됩니다.

---

## GitHub PR 검토(단계별) {#github-pr-review}

모든 pull request에서 자동 코드 검토를 실행하도록 설정하는 과정입니다.

### 1. GitHub에서 웹훅 생성

1. 저장소로 이동 → **Settings** → **Webhooks** → **Add webhook**
2. **Payload URL**을 `http://your-server:8644/webhooks/github-pr`로 설정합니다
3. **Content type**을 `application/json`으로 설정합니다
4. **Secret**을 라우트 설정과 일치하도록 설정합니다(예: `github-webhook-secret`)
5. **Which events?**에서 **Let me select individual events**를 선택하고 **Pull requests**를 체크합니다
6. **Add webhook**을 클릭합니다

### 2. 라우트 설정 추가

위 예시와 같이 `github-pr` 라우트를 `~/.hermes/config.yaml`에 추가합니다.

### 3. `gh` CLI 인증 확인

`github_comment` 전달 유형은 GitHub CLI를 사용해 댓글을 게시합니다.

```bash
gh auth login
```

### 4. 테스트

저장소에서 pull request를 엽니다. 웹훅이 실행되고 Hermes가 이벤트를 처리한 다음 PR에 검토 댓글을 게시합니다.

---

## GitLab 웹훅 설정 {#gitlab-webhook-setup}

GitLab 웹훅도 비슷하게 작동하지만 인증 방식이 다릅니다. GitLab은 일반 `X-Gitlab-Token` 헤더로 시크릿을 보냅니다(HMAC이 아닌 정확한 문자열 일치).

### 1. GitLab에서 웹훅 생성

1. 프로젝트로 이동 → **Settings** → **Webhooks**
2. **URL**을 `http://your-server:8644/webhooks/gitlab-mr`로 설정합니다
3. **Secret token**을 입력합니다
4. **Merge request events**(및 필요한 다른 이벤트)를 선택합니다
5. **Add webhook**을 클릭합니다

### 2. 라우트 설정 추가

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
            Review this merge request:
            Project: {project.path_with_namespace}
            MR !{object_attributes.iid}: {object_attributes.title}
            Author: {object_attributes.last_commit.author.name}
            URL: {object_attributes.url}
            Action: {object_attributes.action}
          deliver: "log"
```

---

## 전달 옵션 {#delivery-options}

`deliver` 필드는 웹훅 이벤트를 처리한 뒤 에이전트 응답을 보낼 위치를 제어합니다.

| 전달 유형 | 설명 |
|-------------|-------------|
| `log` | 응답을 게이트웨이 로그 출력에 기록합니다. 기본값이며 테스트에 유용합니다. |
| `github_comment` | `gh` CLI를 통해 응답을 PR/이슈 댓글로 게시합니다. `deliver_extra.repo` 및 `deliver_extra.pr_number`가 필요합니다. 게이트웨이 호스트에 `gh` CLI가 설치되고 인증되어 있어야 합니다(`gh auth login`). |
| `telegram` | 응답을 Telegram으로 전달합니다. 홈 채널을 사용하거나 `deliver_extra`에 `chat_id`를 지정합니다. |
| `discord` | 응답을 Discord로 전달합니다. 홈 채널을 사용하거나 `deliver_extra`에 `chat_id`를 지정합니다. |
| `slack` | 응답을 Slack으로 전달합니다. 홈 채널을 사용하거나 `deliver_extra`에 `chat_id`를 지정합니다. |
| `signal` | 응답을 Signal로 전달합니다. 홈 채널을 사용하거나 `deliver_extra`에 `chat_id`를 지정합니다. |
| `sms` | Twilio를 통해 응답을 SMS로 전달합니다. 홈 채널을 사용하거나 `deliver_extra`에 `chat_id`를 지정합니다. |
| `whatsapp` | 응답을 WhatsApp으로 전달합니다. 홈 채널을 사용하거나 `deliver_extra`에 `chat_id`를 지정합니다. |
| `matrix` | 응답을 Matrix로 전달합니다. 홈 채널을 사용하거나 `deliver_extra`에 `chat_id`를 지정합니다. |
| `mattermost` | 응답을 Mattermost로 전달합니다. 홈 채널을 사용하거나 `deliver_extra`에 `chat_id`를 지정합니다. |
| `homeassistant` | 응답을 Home Assistant로 전달합니다. 홈 채널을 사용하거나 `deliver_extra`에 `chat_id`를 지정합니다. |
| `email` | 응답을 Email로 전달합니다. 홈 채널을 사용하거나 `deliver_extra`에 `chat_id`를 지정합니다. |
| `dingtalk` | 응답을 DingTalk로 전달합니다. 홈 채널을 사용하거나 `deliver_extra`에 `chat_id`를 지정합니다. |
| `feishu` | 응답을 Feishu/Lark로 전달합니다. 홈 채널을 사용하거나 `deliver_extra`에 `chat_id`를 지정합니다. |
| `wecom` | 응답을 WeCom으로 전달합니다. 홈 채널을 사용하거나 `deliver_extra`에 `chat_id`를 지정합니다. |
| `weixin` | 응답을 Weixin(WeChat)으로 전달합니다. 홈 채널을 사용하거나 `deliver_extra`에 `chat_id`를 지정합니다. |
| `bluebubbles` | 응답을 BlueBubbles(iMessage)로 전달합니다. 홈 채널을 사용하거나 `deliver_extra`에 `chat_id`를 지정합니다. |

플랫폼 간 전달을 사용하려면 대상 플랫폼도 게이트웨이에서 활성화되고 연결되어 있어야 합니다. `deliver_extra`에 `chat_id`가 없으면 응답은 해당 플랫폼에 구성된 홈 채널로 전송됩니다.

---

## 직접 전달 모드 {#direct-delivery-mode}

기본적으로 모든 웹훅 POST는 에이전트 실행을 트리거합니다. 페이로드가 프롬프트가 되고, 에이전트가 이를 처리하며, 에이전트의 응답이 전달됩니다. 따라서 모든 이벤트에 LLM 토큰 비용이 발생합니다.

**일반 알림만 전송**하고 싶은 사용 사례라면(추론이나 에이전트 루프 없이 메시지만 전달) 라우트에 `deliver_only: true`를 설정합니다. 렌더링된 `prompt` 템플릿이 리터럴 메시지 본문이 되고, 어댑터가 이를 구성된 전달 대상으로 직접 디스패치합니다.

### 직접 전달을 사용할 시점

- **외부 서비스 푸시** — Supabase/Firebase 웹훅이 데이터베이스 변경 시 실행 → Telegram 사용자에게 즉시 알림
- **모니터링 알림** — Datadog/Grafana 알림 웹훅 → Discord 채널로 전송
- **에이전트 간 핑** — 에이전트 A가 장시간 실행된 작업이 끝났음을 에이전트 B의 사용자에게 알림
- **백그라운드 작업 완료** — Cron 작업 종료 → Slack에 결과 게시

장점:

- **LLM 토큰 0개** — 에이전트를 호출하지 않습니다
- **1초 미만 전달** — 추론 루프 없이 어댑터를 한 번만 호출합니다
- **에이전트 모드와 동일한 보안** — HMAC 인증, 속도 제한, 멱등성 및 본문 크기 제한이 모두 그대로 적용됩니다
- **동기 응답** — 전달에 성공하면 POST가 `200 OK`를 반환하고, 대상이 거부하면 `502`를 반환하므로 업스트림 서비스가 지능적으로 재시도할 수 있습니다

### 예시: Supabase에서 Telegram으로 푸시

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
          prompt: "🎉 New match: {match.user_name} matched with you!"
          deliver_extra:
            chat_id: "{match.telegram_chat_id}"
```

Supabase edge function은 페이로드에 HMAC-SHA256으로 서명하고 `https://your-server:8644/webhooks/antenna-matches`로 POST합니다. 웹훅 어댑터가 서명을 검증하고, 페이로드에서 템플릿을 렌더링하고, Telegram으로 전달한 뒤 `200 OK`를 반환합니다.

### 예시: CLI를 통한 동적 구독

```bash
hermes webhook subscribe antenna-matches \
  --deliver telegram \
  --deliver-chat-id "123456789" \
  --deliver-only \
  --prompt "🎉 New match: {match.user_name} matched with you!" \
  --description "Antenna match notifications"
```

### 응답 코드

| 상태 | 의미 |
|--------|---------|
| `200 OK` | 성공적으로 전달되었습니다. 본문: `{"status": "delivered", "route": "...", "target": "...", "delivery_id": "..."}` |
| `200 OK` (status=duplicate) | 멱등성 TTL(1시간) 내에 `X-GitHub-Delivery` ID가 중복되었습니다. 다시 전달하지 않습니다. |
| `401 Unauthorized` | HMAC 서명이 유효하지 않거나 없습니다. |
| `400 Bad Request` | JSON 본문 형식이 잘못되었습니다. |
| `404 Not Found` | 알 수 없는 라우트 이름입니다. |
| `413 Payload Too Large` | 본문이 `max_body_bytes`를 초과했습니다. |
| `429 Too Many Requests` | 라우트 속도 제한을 초과했습니다. |
| `502 Bad Gateway` | 대상 어댑터가 메시지를 거부했거나 예외를 발생시켰습니다. 오류는 서버 측에 기록되며, 어댑터 내부 정보가 유출되지 않도록 응답 본문은 일반적인 `Delivery failed`로 반환됩니다. |

### 구성 시 주의 사항

- `deliver_only: true`에는 실제 대상인 `deliver`가 필요합니다. `deliver: log`(또는 `deliver` 생략)는 시작 시 거부됩니다. 어댑터는 잘못 구성된 라우트를 발견하면 시작하지 않습니다.
- 직접 전달 모드에서는 `skills` 필드를 무시합니다(에이전트가 실행되지 않으므로 스킬을 주입할 대상이 없습니다).
- 템플릿 렌더링에는 `{__raw__}` 토큰을 포함해 에이전트 모드와 동일한 `{dot.notation}` 구문을 사용합니다.
- 멱등성에는 동일한 `X-GitHub-Delivery` / `X-Request-ID` 헤더를 사용합니다. 동일한 ID로 재시도하면 `status=duplicate`를 반환하며 다시 전달하지 **않습니다**.

---

## 동적 구독(CLI) {#dynamic-subscriptions}

`config.yaml`의 정적 라우트 외에도 `hermes webhook` CLI 명령으로 웹훅 구독을 동적으로 생성할 수 있습니다. 에이전트가 이벤트 기반 트리거를 직접 설정해야 할 때 특히 유용합니다.

### 구독 생성

```bash
hermes webhook subscribe github-issues \
  --events "issues" \
  --prompt "New issue #{issue.number}: {issue.title}\nBy: {issue.user.login}\n\n{issue.body}" \
  --deliver telegram \
  --deliver-chat-id "-100123456789" \
  --description "Triage new GitHub issues"
```

이 명령은 웹훅 URL과 자동 생성된 HMAC 시크릿을 반환합니다. 해당 URL로 POST하도록 서비스를 구성합니다.

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

### 동적 구독의 작동 방식

- 구독은 `~/.hermes/webhook_subscriptions.json`에 저장됩니다.
- 웹훅 어댑터는 들어오는 각 요청마다 이 파일을 핫 리로드합니다(mtime으로 검사하며 오버헤드는 무시할 수 있는 수준입니다).
- 동일한 이름의 동적 라우트보다 `config.yaml`의 정적 라우트가 항상 우선합니다.
- 동적 구독은 정적 라우트와 동일한 형식과 기능을 사용합니다(events, 프롬프트 템플릿, 스킬, 전달).
- 게이트웨이를 재시작할 필요가 없습니다. 구독하면 즉시 적용됩니다.

### 에이전트 기반 구독

에이전트는 `webhook-subscriptions` 스킬의 안내를 받아 터미널 도구로 구독을 생성할 수 있습니다. 에이전트에게 “GitHub 이슈용 웹훅을 설정해 줘”라고 요청하면 적절한 `hermes webhook subscribe` 명령을 실행합니다.

---

## 라우트별 도구 세트 {#per-route-toolsets}

웹훅 에이전트 실행은 기본적으로 의도적으로 제한된 도구 세트(`web_search`, `web_extract`, `vision_analyze`, `clarify`)를 사용합니다. 웹훅 페이로드에는 신뢰할 수 없는 제3자 콘텐츠가 포함될 수 있기 때문입니다. 공개 PR 제목이나 이슈 댓글이 프롬프트 인젝션으로 터미널을 실행하도록 만들어서는 안 됩니다.

**신뢰할 수 있는** 라우트(시스템 알림을 보내는 localhost 모니터링 데몬, 내부 CI 시스템 등)라면 모든 웹훅 라우트를 넓히지 않고 해당 라우트에만 더 넓은 도구 세트를 부여할 수 있습니다.

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        oom-emergency:
          secret: "monitor-secret"
          prompt: "Memory emergency: {detail}. Diagnose with ps/free/py-spy and report."
          toolsets: ["terminal", "file", "code_execution", "web"]
          deliver: "telegram"
```

동적 구독에서는 `~/.hermes/webhook_subscriptions.json`을 직접 편집해 `toolsets` 키를 추가합니다.

```json
{
  "oom-emergency": {
    "secret": "...",
    "prompt": "...",
    "toolsets": ["terminal", "file", "web"],
    "deliver": "telegram"
  }
}
```

동작 및 안전 속성:

- 라우트 목록은 해당 라우트 실행에 대한 플랫폼 수준 웹훅 도구 세트 해석을 **대체**합니다(병합되지 않음).
- 이름은 `platform_toolsets` 설정과 동일한 경로로 검증됩니다. 알 수 없는 도구 세트와 플랫폼에서 제한된 도구 세트는 제거됩니다.
- `hermes webhook subscribe`는 의도적으로 toolsets 플래그를 지원하지 않습니다. 권한이 높은 도구를 부여하려면 설정 파일을 수동으로 편집해야 하므로, 런타임에 구독을 생성하는 에이전트가 스스로 `terminal`을 부여할 수 없습니다.
- 실제 HMAC 시크릿을 사용하고 발신자를 완전히 제어하는 라우트에만 권한이 높은 도구 세트를 부여합니다. 해당 라우트에 유효하게 서명된 페이로드를 POST할 수 있는 사람은 사실상 그 도구를 사용하는 에이전트를 실행할 수 있습니다.

---

## 보안 {#security}

웹훅 어댑터에는 여러 보안 계층이 포함되어 있습니다.

### HMAC 서명 검증

어댑터는 각 소스에 적합한 방식으로 들어오는 웹훅 서명을 검증합니다.

- **GitHub**: `X-Hub-Signature-256` 헤더 — `sha256=` 접두사가 붙은 HMAC-SHA256 16진수 다이제스트
- **GitLab**: `X-Gitlab-Token` 헤더 — 일반 시크릿 문자열 일치
- **일반(V2, 권장)**: `X-Webhook-Signature-V2` + `X-Webhook-Timestamp` 헤더 — `<timestamp>.<body>`의 HMAC-SHA256 16진수 다이제스트. 타임스탬프(Unix 초)는 서버 시계와 ±300초 이내여야 하므로, 탈취한 요청을 나중에 재생하는 것을 방지합니다.
- **일반(V1, 레거시)**: `X-Webhook-Signature` 헤더 — 본문만 사용한 원시 HMAC-SHA256 16진수 다이제스트입니다. 하위 호환성을 위해 계속 허용되지만 재생 방지 기능이 없습니다(탈취한 요청을 무기한 재생할 수 있음). 게이트웨이는 라우트당 한 번 사용 중단 경고를 기록합니다. 발신자를 V2로 전환하세요.

시크릿이 구성되어 있는데 인식 가능한 서명 헤더가 없으면 요청이 거부됩니다.

### 시크릿 필수

모든 라우트에는 시크릿이 있어야 합니다. 라우트에 직접 설정하거나 전역 `secret`을 상속할 수 있습니다. 시크릿이 없는 라우트가 있으면 어댑터가 오류와 함께 시작에 실패합니다. 개발/테스트 목적으로만 시크릿을 `"INSECURE_NO_AUTH"`로 설정해 검증을 완전히 건너뛸 수 있습니다.

멀티 프로필 라우팅이 활성화되면 라우트의 `profile` 필드도 해당 시크릿을 하나의 실행 대상으로 연결합니다. `profile`이 없는 라우트는 기본 프로필 전용입니다. 유효한 라우트 서명을 포함한 요청도 `/p/<profile>/` 접두사가 라우트 연결과 일치하지 않으면 거부됩니다.

`INSECURE_NO_AUTH`는 게이트웨이가 루프백 호스트(`127.0.0.1`, `localhost`, `::1`)에 바인딩된 경우에만 허용됩니다. `0.0.0.0` 또는 LAN IP 같은 비루프백 바인딩과 함께 사용하면 어댑터가 시작을 거부합니다. 이를 통해 인증되지 않은 엔드포인트를 공용 인터페이스에 실수로 노출하는 것을 방지합니다.

### 속도 제한

각 라우트는 기본적으로 **분당 30개 요청**으로 속도가 제한됩니다(고정 윈도우). 전역 설정:

```yaml
platforms:
  webhook:
    extra:
      rate_limit: 60  # requests per minute
```

제한을 초과하는 요청에는 `429 Too Many Requests` 응답이 반환됩니다.

### 멱등성

전달 ID(`X-GitHub-Delivery`, `X-Request-ID` 또는 타임스탬프 폴백)는 **1시간** 동안 캐시됩니다. 중복 전달(예: 웹훅 재시도)은 `200` 응답과 함께 조용히 건너뛰므로 에이전트가 중복 실행되는 것을 방지합니다.

### 본문 크기 제한

**1 MB**를 초과하는 페이로드는 본문을 읽기 전에 거부됩니다. 다음과 같이 설정합니다.

```yaml
platforms:
  webhook:
    extra:
      max_body_bytes: 2097152  # 2 MB
```

### 인증되었다고 신뢰할 수 있는 것은 아닙니다

:::warning
**HMAC 검증은 _발신자_를 인증할 뿐, _콘텐츠_를 인증하지 않습니다.** 유효한 서명은 요청이 라우트의 시크릿을 보유한 주체(예: GitHub)에서 왔다는 것만 증명합니다. 페이로드 내부의 _비즈니스 필드_를 누가 작성했는지는 알려주지 않습니다. PR 제목, 커밋 메시지, 이슈 설명 및 기타 업스트림 텍스트는 모두 임의의 제3자가 작성하므로 신뢰할 수 없는 입력으로 취급해야 합니다.

이는 에이전트가 읽는 모든 것에 적용되는 동일한 신뢰 모델입니다. 웹 페이지, 파일, 도구 출력은 모두 신뢰할 수 없는 입력입니다. Hermes는 차단 목록으로 신뢰할 수 없는 텍스트를 정제하지 않으며, 안정적으로 정제할 수도 없습니다. 표현, 인코딩, 번역으로 인해 차단 목록은 쉽게 우회할 수 있기 때문입니다. **신뢰 경계는 입력 채널이 아니라 에이전트의 기능 표면입니다.** 다음과 같이 기능 표면을 강화하세요.

- **런타임을 샌드박싱하세요.** 인터넷에 노출할 때는 Docker 또는 SSH 터미널 백엔드(또는 VM)에서 게이트웨이를 실행하여, 탈취된 턴이 호스트에 접근하지 못하게 하세요.
- **도구 세트의 범위를 제한하세요.** 라우트가 읽기와 요약만 필요하다면 웹훅으로 트리거된 세션에서 `terminal`, `file`, 외부 작업 도구를 비활성화하세요. 기능이 적을수록 페이로드 필드에 주입된 지시가 포함되어 있을 때 피해 범위가 작아집니다.
- **승인을 계속 활성화하세요.** 파괴적 작업이나 외부 작업에 승인을 요구하면 주입된 지시가 감독 없이 실행되지 않습니다.
- **템플릿을 좁게 지정하세요.** 전체 페이로드를 프롬프트에 기록하는 `{__raw__}` 또는 빈 템플릿보다 이름이 지정된 필드(`{pull_request.title}`)를 사용하는 구체적인 `prompt`를 우선하세요.
:::

---

## 문제 해결 {#troubleshooting}

### 웹훅이 도착하지 않음

- 포트가 노출되어 있고 웹훅 소스에서 접근 가능한지 확인합니다
- 방화벽 규칙을 확인합니다. 포트 `8644`(또는 구성한 포트)가 열려 있어야 합니다
- URL 경로가 일치하는지 확인합니다: `http://your-server:8644/webhooks/<route-name>`
- `/health` 엔드포인트로 서버 실행 여부를 확인합니다

### 서명 검증 실패

- 라우트 설정의 시크릿이 웹훅 소스에 구성된 시크릿과 정확히 일치하는지 확인합니다
- GitHub에서 시크릿은 HMAC 기반입니다. `X-Hub-Signature-256`을 확인합니다
- GitLab에서 시크릿은 일반 토큰 일치입니다. `X-Gitlab-Token`을 확인합니다
- 게이트웨이 로그에서 `Invalid signature` 경고를 확인합니다

### 이벤트가 무시됨

- 이벤트 유형이 라우트의 `events` 목록에 있는지 확인합니다
- GitHub 이벤트는 `pull_request`, `push`, `issues` 같은 값(`X-GitHub-Event` 헤더 값)을 사용합니다
- GitLab 이벤트는 `merge_request`, `push` 같은 값(`X-GitLab-Event` 헤더 값)을 사용합니다
- `events`가 비어 있거나 설정되지 않으면 모든 이벤트가 수락됩니다

### 에이전트가 응답하지 않음

- 로그를 확인할 수 있도록 게이트웨이를 포그라운드에서 실행합니다: `hermes gateway run`
- 프롬프트 템플릿이 올바르게 렌더링되는지 확인합니다
- 전달 대상이 구성되고 연결되어 있는지 확인합니다

### 중복 응답

- 멱등성 캐시가 이를 방지해야 합니다. 웹훅 소스가 전달 ID 헤더(`X-GitHub-Delivery` 또는 `X-Request-ID`)를 보내는지 확인합니다
- 전달 ID는 1시간 동안 캐시됩니다

### `gh` CLI 오류(GitHub 댓글 전달)

- 게이트웨이 호스트에서 `gh auth login`을 실행합니다
- 인증된 GitHub 사용자에게 저장소 쓰기 권한이 있는지 확인합니다
- `gh`가 설치되어 있고 PATH에 있는지 확인합니다

---

## 환경 변수 {#environment-variables}

| 변수 | 설명 | 기본값 |
|----------|-------------|---------|
| `WEBHOOK_ENABLED` | 웹훅 플랫폼 어댑터 활성화 | `false` |
| `WEBHOOK_PORT` | 웹훅 수신을 위한 HTTP 서버 포트 | `8644` |
| `WEBHOOK_SECRET` | 전역 HMAC 시크릿(라우트에 자체 시크릿이 지정되지 않았을 때 폴백으로 사용) | _(none)_ |
