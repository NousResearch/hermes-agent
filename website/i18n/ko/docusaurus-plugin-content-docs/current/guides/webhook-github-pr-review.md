---
sidebar_position: 11
sidebar_label: "Webhook을 통한 GitHub PR 리뷰"
title: "Webhook을 사용한 GitHub PR 댓글 자동화"
description: "Hermes를 GitHub에 연결하면 PR diff를 자동으로 가져오고 코드 변경 사항을 리뷰한 뒤 댓글을 게시합니다 — 수동 프롬프트 없이 webhook으로 트리거됩니다"
---

# Webhook을 사용한 GitHub PR 댓글 자동화

이 가이드에서는 Hermes Agent를 GitHub에 연결하여 pull request의 diff를 자동으로 가져오고 코드 변경 사항을 분석한 뒤 댓글을 게시하는 방법을 설명합니다. 수동 프롬프트 없이 webhook 이벤트로 트리거됩니다.

PR이 열리거나 업데이트되면 GitHub는 Hermes 인스턴스로 webhook POST를 보냅니다. Hermes는 `gh` CLI를 통해 diff를 가져오도록 지시하는 프롬프트와 함께 에이전트를 실행하고, 응답을 PR 스레드에 게시합니다.

:::tip 공개 엔드포인트 없이 더 간단하게 설정하고 싶으신가요?
공개 URL이 없거나 빠르게 시작하고 싶다면 [GitHub PR 리뷰 에이전트 빌드](./github-pr-review-agent.md)를 확인하세요. cron job을 사용하여 일정에 따라 PR을 폴링하므로 NAT와 방화벽 뒤에서도 작동합니다.
:::

:::info 참고 문서
전체 webhook 플랫폼 참고 문서(모든 구성 옵션, 전송 유형, 동적 구독, 보안 모델)는 [Webhooks](/user-guide/messaging/webhooks)를 참조하세요.
:::

:::warning 프롬프트 인젝션 위험
Webhook payload에는 공격자가 제어할 수 있는 데이터가 포함됩니다. PR 제목, 커밋 메시지, 설명에 악성 지시가 들어 있을 수 있습니다. webhook 엔드포인트를 인터넷에 노출할 때는 gateway를 샌드박스 환경(Docker, SSH backend)에서 실행하세요. 아래의 [보안 섹션](#security-notes)을 참조하세요.
:::

---

## 사전 요구 사항

- Hermes Agent가 설치되어 실행 중이어야 합니다(`hermes gateway`).
- [`gh` CLI](https://cli.github.com/)가 gateway 호스트에 설치되고 인증되어 있어야 합니다(`gh auth login`).
- Hermes 인스턴스에 공개적으로 접근 가능한 URL이 있어야 합니다(로컬에서 실행 중이라면 [ngrok을 사용한 로컬 테스트](#local-testing-with-ngrok) 참조).
- GitHub repository의 관리자 액세스 권한이 있어야 합니다(webhook 관리에 필요).

---

## 1단계 — webhook 플랫폼 활성화

다음 내용을 `~/.hermes/config.yaml`에 추가합니다.

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644          # default; change if another service occupies this port
      rate_limit: 30      # max requests per minute per route (not a global cap)

      routes:
        github-pr-review:
          secret: "your-webhook-secret-here"   # must match the GitHub webhook secret exactly
          events:
            - pull_request

          # The agent is instructed to fetch the actual diff before reviewing.
          # {number} and {repository.full_name} are resolved from the GitHub payload.
          prompt: |
            A pull request event was received (action: {action}).

            PR #{number}: {pull_request.title}
            Author: {pull_request.user.login}
            Branch: {pull_request.head.ref} → {pull_request.base.ref}
            Description: {pull_request.body}
            URL: {pull_request.html_url}

            If the action is "closed" or "labeled", stop here and do not post a comment.

            Otherwise:
            1. Run: gh pr diff {number} --repo {repository.full_name}
            2. Review the code changes for correctness, security issues, and clarity.
            3. Write a concise, actionable review comment and post it.

          deliver: github_comment
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
```

**주요 필드:**

| 필드 | 설명 |
|---|---|
| `secret` (route-level) | 이 route의 HMAC secret입니다. 생략하면 전역 `extra.secret`을 사용합니다. |
| `events` | 허용할 `X-GitHub-Event` header 값의 목록입니다. 빈 목록 = 모두 허용합니다. |
| `prompt` | 템플릿입니다. `{field}`와 `{nested.field}`는 GitHub payload에서 확인됩니다. |
| `deliver` | `github_comment`는 `gh pr comment`를 통해 게시합니다. `log`는 gateway log에 기록만 합니다. |
| `deliver_extra.repo` | payload에서 `org/repo`와 같은 값으로 확인됩니다. |
| `deliver_extra.pr_number` | payload에서 PR 번호로 확인됩니다. |

:::note Payload에는 코드가 포함되지 않습니다
GitHub webhook payload에는 PR 메타데이터(제목, 설명, branch 이름, URL)가 포함되지만 **diff는 포함되지 않습니다**. 위 프롬프트는 에이전트가 `gh pr diff`를 실행하여 실제 변경 사항을 가져오도록 지시합니다. 기본 `hermes-webhook` toolset은 webhook payload에 신뢰할 수 없는 콘텐츠가 포함될 수 있으므로 의도적으로 제한되어 있습니다(web search/extract, vision, clarify — **terminal 없음**). 이 route에서 `gh`를 실행하려면 route 구성에 route별 toolset 권한인 `toolsets: ["terminal", "web"]`을 추가하세요. [Route별 toolset](/docs/user-guide/messaging/webhooks#per-route-toolsets)을 참조하세요.
:::

---

## 2단계 — gateway 시작

```bash
hermes gateway
```

다음과 같이 표시되어야 합니다.

```
[webhook] Listening on 0.0.0.0:8644 — routes: github-pr-review
```

실행 중인지 확인합니다.

```bash
curl http://localhost:8644/health
# {"status": "ok", "platform": "webhook"}
```

---

## 3단계 — GitHub에 webhook 등록

1. repository로 이동하여 → **Settings** → **Webhooks** → **Add webhook**을 선택합니다.
2. 다음을 입력합니다.
   - **Payload URL:** `https://your-public-url.example.com/webhooks/github-pr-review`
   - **Content type:** `application/json`
   - **Secret:** route 구성의 `secret`에 설정한 값과 동일한 값을 입력합니다.
   - **Which events?** → Select individual events → **Pull requests**를 선택합니다.
3. **Add webhook**을 클릭합니다.

GitHub는 연결을 확인하기 위해 즉시 `ping` 이벤트를 보냅니다. `ping`은 `events` 목록에 없으므로 안전하게 무시되며 `{"status": "ignored", "event": "ping"}`를 반환합니다. DEBUG 레벨에만 기록되므로 기본 로그 레벨에서는 콘솔에 표시되지 않습니다.

---

## 4단계 — 테스트 PR 열기

branch를 만들고 변경 사항을 push한 다음 PR을 엽니다. 30~90초 이내에(PR 크기와 모델에 따라 다름) Hermes가 리뷰 댓글을 게시해야 합니다.

에이전트의 진행 상황을 실시간으로 확인하려면 다음을 실행합니다.

```bash
tail -f "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
```

---

## ngrok을 사용한 로컬 테스트

Hermes가 노트북에서 실행 중이라면 [ngrok](https://ngrok.com/)을 사용하여 노출합니다.

```bash
ngrok http 8644
```

`https://...ngrok-free.app` URL을 복사하여 GitHub Payload URL로 사용합니다. 무료 ngrok tier에서는 ngrok이 다시 시작될 때마다 URL이 변경되므로 세션마다 GitHub webhook을 업데이트해야 합니다. 유료 ngrok 계정에서는 고정 도메인을 사용할 수 있습니다.

정적 route는 `curl`로 직접 smoke-test할 수 있으므로 GitHub 계정이나 실제 PR이 필요하지 않습니다.

:::tip 로컬 테스트 시 `deliver: log` 사용
테스트하는 동안 구성에서 `deliver: github_comment`을 `deliver: log`로 변경하세요. 그렇지 않으면 에이전트가 테스트 payload의 가짜 `org/repo#99` repository에 댓글을 게시하려고 시도하다가 실패합니다. 프롬프트 출력이 만족스러우면 `deliver: github_comment`으로 되돌리세요.
:::

```bash
SECRET="your-webhook-secret-here"
BODY='{"action":"opened","number":99,"pull_request":{"title":"Test PR","body":"Adds a feature.","user":{"login":"testuser"},"head":{"ref":"feat/x"},"base":{"ref":"main"},"html_url":"https://github.com/org/repo/pull/99"},"repository":{"full_name":"org/repo"}}'
SIG=$(printf '%s' "$BODY" | openssl dgst -sha256 -hmac "$SECRET" -hex | awk '{print "sha256="$2}')

curl -s -X POST http://localhost:8644/webhooks/github-pr-review \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: pull_request" \
  -H "X-Hub-Signature-256: $SIG" \
  -d "$BODY"
# Expected: {"status":"accepted","route":"github-pr-review","event":"pull_request","delivery_id":"..."}
```

그런 다음 에이전트 실행을 확인합니다.
```bash
tail -f "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
```

:::note
`hermes webhook test <name>`은 `hermes webhook subscribe`로 생성한 **동적 구독**에서만 작동합니다. `config.yaml`의 route는 읽지 않습니다.
:::

---

## 특정 action으로 필터링

GitHub는 `opened`, `synchronize`, `reopened`, `closed`, `labeled` 등 다양한 action에 대해 `pull_request` 이벤트를 보냅니다. `events` 목록은 `X-GitHub-Event` header 값을 기준으로 필터링하며, route-level `filters`는 `action`과 같은 payload 필드로 범위를 좁힐 수 있습니다.

1단계의 프롬프트는 이미 `closed` 및 `labeled` 이벤트에서 일찍 중단하도록 에이전트에 지시하여 이 작업을 처리합니다.

:::warning 에이전트는 여전히 실행되고 token을 소비합니다
"여기서 중단" 지시는 의미 있는 리뷰를 방지하지만, action과 관계없이 모든 `pull_request` 이벤트에 대해 에이전트가 완료될 때까지 실행됩니다. 에이전트가 깨어나기 전에 필터링하는 것을 권장합니다.

```yaml
filters:
  - field: "action"
    in: ["opened", "synchronize", "reopened"]
```

트래픽이 많은 repository에서는 webhook URL을 조건부로 호출하는 GitHub Actions workflow를 사용하여 upstream에서 필터링할 수도 있습니다.
:::

> Jinja2나 조건부 템플릿 문법은 없습니다. 지원되는 치환은 `{field}`와 `{nested.field}`뿐입니다. 그 외의 모든 내용은 에이전트에 그대로 전달됩니다.

---

## 일관된 리뷰 스타일에 skill 사용

[Hermes skill](/user-guide/features/skills)을 로드하여 에이전트에 일관된 리뷰 페르소나를 부여할 수 있습니다. `config.yaml`의 `platforms.webhook.extra.routes` 아래 route에 `skills`를 추가합니다.

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        github-pr-review:
          secret: "your-webhook-secret-here"
          events: [pull_request]
          prompt: |
            A pull request event was received (action: {action}).
            PR #{number}: {pull_request.title} by {pull_request.user.login}
            URL: {pull_request.html_url}

            If the action is "closed" or "labeled", stop here and do not post a comment.

            Otherwise:
            1. Run: gh pr diff {number} --repo {repository.full_name}
            2. Review the diff using your review guidelines.
            3. Write a concise, actionable review comment and post it.
          skills:
            - review
          deliver: github_comment
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
```

> **참고:** 목록에서 발견되는 첫 번째 skill만 로드됩니다. Hermes는 여러 skill을 쌓지 않으므로 이후 항목은 무시됩니다.

---

## 대신 Slack 또는 Discord로 응답 보내기

route 내부의 `deliver` 및 `deliver_extra` 필드를 대상 플랫폼으로 변경합니다.

```yaml
# Inside platforms.webhook.extra.routes.<route-name>:

# Slack
deliver: slack
deliver_extra:
  chat_id: "C0123456789"   # Slack channel ID (omit to use the configured home channel)

# Discord
deliver: discord
deliver_extra:
  chat_id: "987654321012345678"  # Discord channel ID (omit to use home channel)
```

대상 플랫폼도 gateway에서 활성화되고 연결되어 있어야 합니다. `chat_id`를 생략하면 응답은 해당 플랫폼에 구성된 home channel로 전송됩니다.

유효한 `deliver` 값: `log` · `github_comment` · `telegram` · `discord` · `slack` · `signal` · `sms`

---

## GitLab 지원

동일한 adapter가 GitLab에서도 작동합니다. GitLab은 인증에 `X-Gitlab-Token`을 사용합니다(HMAC가 아닌 일반 문자열 일치). Hermes가 두 방식을 모두 자동으로 처리합니다.

이벤트 필터링에서 GitLab은 `X-GitLab-Event`를 `Merge Request Hook`, `Push Hook`, `Pipeline Hook`과 같은 값으로 설정합니다. `events`에는 정확한 header 값을 사용합니다.

```yaml
events:
  - Merge Request Hook
```

GitLab payload 필드는 GitHub와 다릅니다. 예를 들어 MR 제목에는 `{object_attributes.title}`, MR 번호에는 `{object_attributes.iid}`를 사용합니다. 전체 payload 구조를 확인하는 가장 쉬운 방법은 webhook 설정의 GitLab **Test** 버튼과 **Recent Deliveries** 로그를 함께 사용하는 것입니다. 또는 route 구성에서 `prompt`를 생략하면 Hermes가 형식이 지정된 JSON으로 전체 payload를 에이전트에 직접 전달하며, `deliver: log`를 사용하면 gateway log에 표시되는 에이전트 응답이 구조를 설명합니다.

---

## 보안 참고 사항

- 프로덕션에서 **절대 `INSECURE_NO_AUTH`를 사용하지 마세요**. signature validation을 완전히 비활성화합니다. 로컬 개발 전용입니다.
- webhook secret을 주기적으로 교체하고 GitHub(webhook settings)와 `config.yaml` 양쪽에서 업데이트하세요.
- **Rate limiting**은 기본적으로 route당 30 req/min입니다(`extra.rate_limit`으로 구성 가능). 초과하면 `429`를 반환합니다.
- **중복 전송**(webhook retry)은 1시간 idempotency cache를 통해 중복 제거됩니다. cache key는 `X-GitHub-Delivery`가 있으면 이를 사용하고, 다음으로 `X-Request-ID`, 그 다음으로 millisecond timestamp를 사용합니다. 두 delivery ID header가 모두 설정되지 않으면 retry는 **중복 제거되지 않습니다**.
- **Prompt injection:** PR 제목, 설명, 커밋 메시지는 공격자가 제어합니다. 악성 PR이 에이전트의 작업을 조작하려고 시도할 수 있습니다. 공개 인터넷에 노출할 때는 gateway를 샌드박스 환경(Docker, VM)에서 실행하세요.

---

## 문제 해결

| 증상 | 확인할 내용 |
|---|---|
| `401 Invalid signature` | config.yaml의 secret이 GitHub webhook secret과 일치하지 않음 |
| `404 Unknown route` | URL의 route 이름이 `routes:`의 key와 일치하지 않음 |
| `429 Rate limit exceeded` | route당 30 req/min 초과 — GitHub UI에서 테스트 이벤트를 재전송할 때 흔히 발생합니다. 1분 기다리거나 `extra.rate_limit`을 높이세요. |
| 댓글이 게시되지 않음 | `gh`가 설치되지 않았거나 PATH에 없거나 인증되지 않음(`gh auth login`) |
| 에이전트는 실행되지만 댓글이 없음 | gateway log를 확인하세요. 에이전트 출력이 비어 있거나 단순히 "SKIP"이어도 delivery는 시도됩니다. |
| Port already in use | config.yaml의 `extra.port`를 변경하세요. |
| 에이전트가 실행되지만 PR 설명만 리뷰함 | 프롬프트에 `gh pr diff` 지시가 포함되지 않음 — diff는 webhook payload에 없습니다. |
| ping 이벤트를 볼 수 없음 | 무시된 이벤트는 DEBUG 로그 레벨에서만 `{"status":"ignored","event":"ping"}`을 반환합니다. GitHub의 delivery log(repository → Settings → Webhooks → webhook → Recent Deliveries)를 확인하세요. |

**GitHub의 Recent Deliveries 탭**(repository → Settings → Webhooks → webhook)은 모든 delivery에 대한 정확한 request header, payload, HTTP status, response body를 보여 줍니다. 서버 로그를 확인하지 않고도 문제를 진단하는 가장 빠른 방법입니다.

---

## 전체 구성 참고

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644               # listen port (default: 8644)
      secret: ""               # optional global fallback secret
      rate_limit: 30           # requests per minute per route
      max_body_bytes: 1048576  # payload size limit in bytes (default: 1 MB)

      routes:
        <route-name>:
          secret: "required-per-route"
          events: []            # [] = accept all; otherwise list X-GitHub-Event values
          prompt: ""            # {field} / {nested.field} resolved from payload
          skills: []            # first matching skill is loaded (only one)
          deliver: "log"        # log | github_comment | telegram | discord | slack | signal | sms
          deliver_extra: {}     # repo + pr_number for github_comment; chat_id for others
```

---

## 다음 단계

- **[Cron 기반 PR 리뷰](./github-pr-review-agent.md)** — 일정에 따라 PR을 폴링하며 공개 엔드포인트가 필요하지 않습니다.
- **[Webhook 참고 문서](/user-guide/messaging/webhooks)** — webhook 플랫폼의 전체 구성 참고
- **[Plugin 빌드](/developer-guide/plugins)** — 리뷰 로직을 공유 가능한 plugin으로 패키징
- **[Profile](/user-guide/profiles)** — 자체 memory와 config를 사용하는 전용 reviewer profile 실행
