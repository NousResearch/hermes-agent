---
sidebar_position: 11
sidebar_label: "웹훅(Webhook)을 통한 GitHub PR 리뷰"
title: "웹훅을 사용한 자동 GitHub PR 코멘트"
description: "Hermes를 GitHub에 연결하여 PR 차이(diff)를 자동으로 가져오고, 코드 변경 사항을 검토하며, 코멘트를 게시합니다 — 수동 프롬프트 없이 웹훅 이벤트에 의해 트리거됩니다."
---

# 웹훅을 사용한 자동 GitHub PR 코멘트

이 가이드는 Hermes Agent를 GitHub에 연결하여 풀 리퀘스트의 diff를 자동으로 가져오고, 코드 변경 사항을 분석하며, 코멘트를 게시하는 과정을 안내합니다. 이 과정은 수동 프롬프트 없이 웹훅 이벤트에 의해 트리거됩니다.

PR이 열리거나 업데이트되면 GitHub는 사용자의 Hermes 인스턴스로 웹훅 POST 요청을 보냅니다. Hermes는 `gh` CLI를 통해 diff를 검색하도록 지시하는 프롬프트와 함께 에이전트를 실행하며, 응답은 PR 스레드에 다시 게시됩니다.

:::tip 공개 엔드포인트 없이 더 간단한 설정을 원하시나요?
공개 URL이 없거나 바로 시작하고 싶다면, [GitHub PR 리뷰 에이전트 구축하기](./github-pr-review-agent.md)를 확인하세요 — 크론(cron) 작업을 사용하여 일정에 따라 PR을 폴링하며, NAT 및 방화벽 뒤에서 작동합니다.
:::

:::info 레퍼런스 문서
전체 웹훅 플랫폼 레퍼런스(모든 구성 옵션, 전달 유형, 동적 구독, 보안 모델)에 대해서는 [웹훅(Webhooks)](/user-guide/messaging/webhooks)을 참조하세요.
:::

:::warning 프롬프트 인젝션 위험
웹훅 페이로드에는 공격자가 제어할 수 있는 데이터가 포함됩니다 — PR 제목, 커밋 메시지, 설명에 악의적인 지침이 포함될 수 있습니다. 웹훅 엔드포인트가 인터넷에 노출된 경우 게이트웨이를 샌드박스 환경(Docker, SSH 백엔드)에서 실행하세요. 아래의 [보안 참고 사항](#보안-참고-사항) 섹션을 참조하세요.
:::

---

## 사전 요구 사항

- Hermes Agent가 설치되고 실행 중일 것 (`hermes gateway`)
- 게이트웨이 호스트에 [`gh` CLI](https://cli.github.com/)가 설치되고 인증되었을 것 (`gh auth login`)
- Hermes 인스턴스에 퍼블릭으로 접근 가능한 URL (로컬에서 실행하는 경우 [ngrok을 사용한 로컬 테스트](#ngrok을-사용한-로컬-테스트) 참조)
- GitHub 저장소에 대한 관리자 액세스 권한 (웹훅을 관리하는 데 필요함)

---

## 1단계 — 웹훅 플랫폼 활성화

`~/.hermes/config.yaml`에 다음을 추가합니다:

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644          # 기본값; 다른 서비스가 이 포트를 사용하는 경우 변경
      rate_limit: 30      # 경로당 분당 최대 요청 수 (글로벌 제한이 아님)

      routes:
        github-pr-review:
          secret: "your-webhook-secret-here"   # GitHub 웹훅의 시크릿과 정확히 일치해야 합니다.
          events:
            - pull_request

          # 에이전트는 검토 전에 실제 diff를 가져오도록 지시받습니다.
          # {number} 및 {repository.full_name}은 GitHub 페이로드에서 해석됩니다.
          prompt: |
            풀 리퀘스트 이벤트가 수신되었습니다 (action: {action}).

            PR #{number}: {pull_request.title}
            Author: {pull_request.user.login}
            Branch: {pull_request.head.ref} → {pull_request.base.ref}
            Description: {pull_request.body}
            URL: {pull_request.html_url}

            만약 action이 "closed" 또는 "labeled"인 경우, 여기서 멈추고 코멘트를 게시하지 마세요.

            그렇지 않은 경우:
            1. 실행: gh pr diff {number} --repo {repository.full_name}
            2. 정확성, 보안 문제 및 코드의 명확성에 대해 코드 변경 사항을 검토합니다.
            3. 간결하고 실행 가능한 리뷰 코멘트를 작성하고 게시합니다.

          deliver: github_comment
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
```

**주요 필드:**

| 필드 | 설명 |
|---|---|
| `secret` (route-level) | 이 라우트에 대한 HMAC 시크릿. 생략된 경우 전역 `extra.secret`으로 대체됩니다. |
| `events` | 수락할 `X-GitHub-Event` 헤더 값 목록. 빈 목록 = 모두 수락. |
| `prompt` | 템플릿; `{field}` 및 `{nested.field}`는 GitHub 페이로드에서 변환됩니다. |
| `deliver` | `github_comment`는 `gh pr comment`를 통해 게시합니다. `log`는 게이트웨이 로그에만 씁니다. |
| `deliver_extra.repo` | 페이로드의 예: `org/repo`로 변환됩니다. |
| `deliver_extra.pr_number` | 페이로드의 PR 번호로 변환됩니다. |

:::note 페이로드에는 코드가 포함되어 있지 않습니다
GitHub 웹훅 페이로드에는 PR 메타데이터(제목, 설명, 브랜치 이름, URL)가 포함되지만 **diff 자체는 포함되지 않습니다**. 위의 프롬프트는 에이전트가 `gh pr diff`를 실행하여 실제 변경 사항을 가져오도록 지시합니다. 기본 `hermes-webhook` 도구 세트에 `terminal` 도구가 포함되어 있으므로 추가 구성이 필요하지 않습니다.
:::

---

## 2단계 — 게이트웨이 시작

```bash
hermes gateway
```

다음과 같이 출력되어야 합니다:

```
[webhook] Listening on 0.0.0.0:8644 — routes: github-pr-review
```

실행 중인지 확인합니다:

```bash
curl http://localhost:8644/health
# {"status": "ok", "platform": "webhook"}
```

---

## 3단계 — GitHub에 웹훅 등록

1. 저장소 → **Settings(설정)** → **Webhooks(웹훅)** → **Add webhook(웹훅 추가)** 로 이동합니다.
2. 다음 내용을 입력합니다:
   - **Payload URL:** `https://your-public-url.example.com/webhooks/github-pr-review`
   - **Content type:** `application/json`
   - **Secret:** 경로 구성(route config)에서 `secret`에 설정한 것과 동일한 값
   - **Which events?** → **Let me select individual events(개별 이벤트 선택)** 선택 → **Pull requests** 선택
3. **Add webhook(웹훅 추가)**을 클릭합니다.

GitHub는 연결을 확인하기 위해 즉시 `ping` 이벤트를 보냅니다. 이는 `events` 목록에 없으므로 무시되며 `{"status": "ignored", "event": "ping"}`을 반환합니다. 이는 DEBUG 수준에서만 로깅되므로 기본 로그 수준에서는 콘솔에 나타나지 않습니다.

---

## 4단계 — 테스트 PR 열기

브랜치를 생성하고 변경 사항을 푸시한 다음 PR을 엽니다. (PR 크기 및 모델에 따라) 30–90초 이내에 Hermes가 리뷰 코멘트를 게시해야 합니다.

에이전트의 진행 상황을 실시간으로 확인하려면:

```bash
tail -f "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
```

---

## ngrok을 사용한 로컬 테스트

랩탑에서 Hermes를 실행하는 경우 [ngrok](https://ngrok.com/)을 사용하여 로컬 환경을 노출하세요:

```bash
ngrok http 8644
```

`https://...ngrok-free.app` URL을 복사하여 GitHub의 Payload URL로 사용하세요. 무료 ngrok 계정의 경우 ngrok을 다시 시작할 때마다 URL이 변경되므로 매 세션마다 GitHub 웹훅을 업데이트해야 합니다. 유료 ngrok 계정은 고정 도메인을 얻습니다.

`curl`을 사용하여 GitHub 계정이나 실제 PR 없이도 정적 라우트를 직접 테스트할 수 있습니다.

:::tip 로컬 테스트 시 `deliver: log` 사용
테스트할 때는 구성의 `deliver: github_comment`를 `deliver: log`로 변경하세요. 그렇지 않으면 에이전트가 테스트 페이로드의 가짜 `org/repo#99` 저장소에 코멘트를 게시하려고 시도하다가 실패하게 됩니다. 프롬프트 결과가 만족스러우면 다시 `deliver: github_comment`로 전환하세요.
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
# 예상되는 결과: {"status":"accepted","route":"github-pr-review","event":"pull_request","delivery_id":"..."}
```

그 후 에이전트의 실행 로그를 확인합니다:
```bash
tail -f "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
```

:::note
`hermes webhook test <name>`은 `hermes webhook subscribe`로 생성된 **동적 구독**에만 작동합니다. `config.yaml`에 정의된 경로는 읽지 않습니다.
:::

---

## 특정 작업 필터링

GitHub는 `opened`, `synchronize`, `reopened`, `closed`, `labeled` 등 많은 작업(action)에 대해 `pull_request` 이벤트를 보냅니다. `events` 목록은 `X-GitHub-Event` 헤더 값으로만 필터링하며, 라우팅 수준에서 하위 유형(작업)으로 필터링할 수 없습니다.

1단계의 프롬프트는 에이전트에게 `closed` 및 `labeled` 이벤트에 대해 일찍 멈추도록 지시함으로써 이를 처리합니다.

:::warning 에이전트는 여전히 실행되고 토큰을 소비합니다.
"여기서 멈추기(stop here)" 지시문은 불필요한 리뷰를 방지하지만, 에이전트는 동작(action)과 관계없이 모든 `pull_request` 이벤트에 대해 완료될 때까지 실행됩니다. GitHub 웹훅은 `pull_request`, `push`, `issues` 등의 이벤트 유형(type)으로만 필터링할 수 있으며 `opened`, `closed`, `labeled` 등의 작업 하위 유형으로 필터링할 수 없습니다. 서브 액션을 필터링할 라우팅 수준의 방법은 없습니다. 트래픽이 많은 리포지토리의 경우, 이 비용을 수용하거나, 웹훅 URL을 조건부로 호출하는 GitHub Actions 워크플로를 사용하여 업스트림에서 필터링하세요.
:::

> Jinja2나 조건부 템플릿 구문은 지원되지 않습니다. `{field}`와 `{nested.field}`가 지원되는 유일한 치환입니다. 그 외의 모든 것은 에이전트에 문자 그대로 전달됩니다.

---

## 일관된 리뷰 스타일을 위한 스킬(Skill) 사용

[Hermes 스킬](/user-guide/features/skills)을 불러와 에이전트에 일관된 리뷰 페르소나를 부여하세요. `config.yaml`의 `platforms.webhook.extra.routes` 내부의 라우트에 `skills`를 추가하세요:

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
            풀 리퀘스트 이벤트가 수신되었습니다 (action: {action}).
            PR #{number}: {pull_request.title} by {pull_request.user.login}
            URL: {pull_request.html_url}

            만약 action이 "closed" 또는 "labeled"인 경우, 여기서 멈추고 코멘트를 게시하지 마세요.

            그렇지 않은 경우:
            1. 실행: gh pr diff {number} --repo {repository.full_name}
            2. 검토 가이드라인을 사용하여 diff를 검토합니다.
            3. 간결하고 실행 가능한 리뷰 코멘트를 작성하고 게시합니다.
          skills:
            - review
          deliver: github_comment
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
```

> **참고:** 목록에서 발견된 첫 번째 스킬만 로드됩니다. Hermes는 여러 스킬을 중첩하지 않으며 후속 항목은 무시됩니다.

---

## 응답을 Slack이나 Discord로 보내기

경로 내의 `deliver` 및 `deliver_extra` 필드를 대상 플랫폼으로 변경하세요:

```yaml
# platforms.webhook.extra.routes.<route-name> 내부:

# Slack
deliver: slack
deliver_extra:
  chat_id: "C0123456789"   # Slack 채널 ID (구성된 홈 채널을 사용하려면 생략)

# Discord
deliver: discord
deliver_extra:
  chat_id: "987654321012345678"  # Discord 채널 ID (홈 채널을 사용하려면 생략)
```

대상 플랫폼도 게이트웨이에서 활성화 및 연결되어 있어야 합니다. `chat_id`가 생략되면 구성된 플랫폼 홈 채널로 응답이 전송됩니다.

유효한 `deliver` 값: `log` · `github_comment` · `telegram` · `discord` · `slack` · `signal` · `sms`

---

## GitLab 지원

동일한 어댑터가 GitLab에서도 작동합니다. GitLab은 인증에 `X-Gitlab-Token`을 사용하며 (HMAC가 아닌 단순 문자열 일치) Hermes는 둘 다 자동으로 처리합니다.

이벤트 필터링의 경우, GitLab은 `X-GitLab-Event`를 `Merge Request Hook`, `Push Hook`, `Pipeline Hook` 등의 값으로 설정합니다. `events`에 정확한 헤더 값을 사용하세요:

```yaml
events:
  - Merge Request Hook
```

GitLab 페이로드 필드는 GitHub의 필드와 다릅니다 — 예를 들어, MR 제목은 `{object_attributes.title}`, MR 번호는 `{object_attributes.iid}`입니다. 전체 페이로드 구조를 파악하는 가장 쉬운 방법은 웹훅 설정의 GitLab **Test** 버튼과 **Recent Deliveries(최근 전송)** 로그를 결합하는 것입니다. 또는 라우트 구성에서 `prompt`를 생략하면 Hermes가 전체 페이로드를 포맷된 JSON으로 에이전트에 직접 전달하며, 에이전트의 응답( `deliver: log`로 게이트웨이 로그에 표시됨)이 해당 구조를 설명할 것입니다.

---

## 보안 참고 사항

- **프로덕션 환경에서는 절대 `INSECURE_NO_AUTH`를 사용하지 마세요** — 서명 검증이 완전히 비활성화됩니다. 이는 로컬 개발 전용입니다.
- **주기적으로 웹훅 시크릿을 교체**하고 GitHub(웹훅 설정) 및 `config.yaml` 모두에서 업데이트하세요.
- **속도 제한(Rate limit)**은 기본적으로 라우트당 분당 30회 요청입니다(`extra.rate_limit`을 통해 구성 가능). 이를 초과하면 `429`가 반환됩니다.
- **중복 전송**(웹훅 재시도)은 1시간의 멱등성(idempotency) 캐시를 통해 중복 제거됩니다. 캐시 키는 `X-GitHub-Delivery` 헤더(존재하는 경우), `X-Request-ID` 헤더, 그 다음으로 밀리초 단위의 타임스탬프 순으로 사용됩니다. 두 전송 ID 헤더가 모두 설정되지 않은 경우 재시도는 중복 제거되지 **않습니다**.
- **프롬프트 인젝션:** PR 제목, 설명 및 커밋 메시지는 공격자가 제어할 수 있습니다. 악의적인 PR이 에이전트의 동작을 조작하려 시도할 수 있습니다. 퍼블릭 인터넷에 노출되는 경우 게이트웨이를 샌드박스 환경(Docker, VM)에서 실행하세요.

---

## 문제 해결

| 증상 | 확인 사항 |
|---|---|
| `401 Invalid signature` | config.yaml의 시크릿이 GitHub 웹훅 시크릿과 일치하지 않음 |
| `404 Unknown route` | URL의 라우트 이름이 `routes:`의 키와 일치하지 않음 |
| `429 Rate limit exceeded` | 라우트당 분당 30건을 초과함 — GitHub UI에서 테스트 이벤트를 재전송할 때 흔히 발생; 1분 기다리거나 `extra.rate_limit` 값 늘리기 |
| 코멘트가 게시되지 않음 | `gh`가 설치되지 않았거나 PATH에 없거나, 인증되지 않음 (`gh auth login`) |
| 에이전트는 실행되지만 코멘트 없음 | 게이트웨이 로그 확인 — 에이전트 출력이 비어 있거나 단순히 "SKIP"만 있는 경우에도 전송은 시도됨 |
| Port already in use | config.yaml에서 `extra.port` 변경 |
| 에이전트가 실행되지만 PR 설명만 검토함 | 프롬프트에 `gh pr diff` 지시문이 포함되지 않음 — 웹훅 페이로드에 diff가 없음 |
| ping 이벤트가 보이지 않음 | 무시된 이벤트는 DEBUG 로그 수준에서만 `{"status":"ignored","event":"ping"}`을 반환함 — GitHub의 전송 로그(repo → Settings → Webhooks → 해당 웹훅 → Recent Deliveries)를 확인하세요 |

**GitHub의 Recent Deliveries 탭** (repo → Settings → Webhooks → 해당 웹훅)은 모든 전송에 대한 정확한 요청 헤더, 페이로드, HTTP 상태, 그리고 응답 본문을 보여줍니다. 서버 로그를 건드리지 않고 실패 원인을 진단하는 가장 빠른 방법입니다.

---

## 전체 구성 레퍼런스

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      host: "0.0.0.0"         # 바인드 주소 (기본값: 0.0.0.0)
      port: 8644               # 리슨 포트 (기본값: 8644)
      secret: ""               # 선택적 글로벌 대체 시크릿
      rate_limit: 30           # 라우트당 분당 요청 수
      max_body_bytes: 1048576  # 바이트 단위 페이로드 크기 제한 (기본값: 1 MB)

      routes:
        <route-name>:
          secret: "required-per-route"
          events: []            # [] = 모두 허용; 그렇지 않으면 X-GitHub-Event 값 목록
          prompt: ""            # 페이로드에서 {field} / {nested.field} 변환
          skills: []            # 발견된 첫 번째 스킬이 로드됨 (오직 하나만)
          deliver: "log"        # log | github_comment | telegram | discord | slack | signal | sms
          deliver_extra: {}     # github_comment의 경우 repo + pr_number; 나머지는 chat_id
```

---

## 다음 단계

- **[크론 기반 PR 리뷰](./github-pr-review-agent.md)** — 예약에 따라 PR을 폴링하며, 퍼블릭 엔드포인트가 필요하지 않음
- **[웹훅 레퍼런스](/user-guide/messaging/webhooks)** — 웹훅 플랫폼에 대한 전체 구성 레퍼런스
- **[플러그인 구축](/guides/build-a-hermes-plugin)** — 리뷰 로직을 공유 가능한 플러그인으로 패키징
- **[프로필](/user-guide/profiles)** — 자체 메모리와 구성을 가진 전용 리뷰어 프로필 실행
