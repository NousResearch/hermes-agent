---
sidebar_position: 15
title: "자동화 블루프린트"
description: "바로 사용할 수 있는 자동화 블루프린트 — 예약 작업, GitHub 이벤트 트리거, API 웹훅, 다중 스킬 워크플로"
---

# 자동화 블루프린트

일반적인 자동화 패턴을 위한 복사-붙여넣기용 블루프린트입니다. 각 블루프린트는 시간 기반 트리거에 Hermes의 내장 [cron 스케줄러](/user-guide/features/cron)를 사용하고, 이벤트 기반 트리거에 [웹훅 플랫폼](/user-guide/messaging/webhooks)을 사용합니다.

모든 블루프린트는 **어떤 모델에서든** 작동하며, 하나의 공급자에 종속되지 않습니다.

cron 구문 대신 폼을 사용하는 매개변수화된 블루프린트는 [자동화 블루프린트 카탈로그](/reference/automation-blueprints-catalog)를 참조하세요.

:::tip 세 가지 트리거 유형
| 트리거 | 방식 | 도구 |
|---------|-----|------|
| **예약** | 일정한 주기(매시간, 매일 밤, 매주)로 실행 | `cronjob` 도구 또는 `/cron` 슬래시 명령 |
| **GitHub 이벤트** | PR 열기, 푸시, 이슈, CI 결과 발생 시 실행 | 웹훅 플랫폼(`hermes webhook subscribe`) |
| **API 호출** | 외부 서비스가 엔드포인트로 JSON을 POST | 웹훅 플랫폼(config.yaml 라우트 또는 `hermes webhook subscribe`) |

세 유형 모두 Telegram, Discord, Slack, SMS, 이메일, GitHub 댓글 또는 로컬 파일로 전달할 수 있습니다.
:::

---

## 개발 워크플로

### 야간 백로그 분류

매일 밤 새로운 이슈에 라벨을 지정하고, 우선순위를 정하고, 요약합니다. 팀 채널로 다이제스트를 전달합니다.

**트리거:** 예약(매일 밤)

```bash
hermes cron create "0 2 * * *" \
  "You are a project manager triaging the NousResearch/hermes-agent GitHub repo.

1. Run: gh issue list --repo NousResearch/hermes-agent --state open --json number,title,labels,author,createdAt --limit 30
2. Identify issues opened in the last 24 hours
3. For each new issue:
   - Suggest a priority label (P0-critical, P1-high, P2-medium, P3-low)
   - Suggest a category label (bug, feature, docs, security)
   - Write a one-line triage note
4. Summarize: total open issues, new today, breakdown by priority

Format as a clean digest. If no new issues, respond with [SILENT]." \
  --name "Nightly backlog triage" \
  --deliver telegram
```

### 자동 PR 코드 리뷰

모든 풀 리퀘스트가 열릴 때 자동으로 리뷰합니다. PR에 직접 리뷰 댓글을 게시합니다.

**트리거:** GitHub 웹훅

**옵션 A — 동적 구독(CLI):**

```bash
hermes webhook subscribe github-pr-review \
  --events "pull_request" \
  --prompt "Review this pull request:
Repository: {repository.full_name}
PR #{pull_request.number}: {pull_request.title}
Author: {pull_request.user.login}
Action: {action}
Diff URL: {pull_request.diff_url}

Fetch the diff with: curl -sL {pull_request.diff_url}

Review for:
- Security issues (injection, auth bypass, secrets in code)
- Performance concerns (N+1 queries, unbounded loops, memory leaks)
- Code quality (naming, duplication, error handling)
- Missing tests for new behavior

Post a concise review. If the PR is a trivial docs/typo change, say so briefly." \
  --skills github-code-review \
  --deliver github_comment
```

**옵션 B — 정적 라우트(config.yaml):**

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644
      secret: "your-global-secret"
      routes:
        github-pr-review:
          events: ["pull_request"]
          secret: "github-webhook-secret"
          prompt: |
            Review PR #{pull_request.number}: {pull_request.title}
            Repository: {repository.full_name}
            Author: {pull_request.user.login}
            Diff URL: {pull_request.diff_url}
            Review for security, performance, and code quality.
          skills: ["github-code-review"]
          deliver: "github_comment"
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{pull_request.number}"
```

그런 다음 GitHub에서 **Settings → Webhooks → Add webhook**으로 이동하고, Payload URL에는 `http://your-server:8644/webhooks/github-pr-review`, Content type에는 `application/json`, Secret에는 `github-webhook-secret`을 입력하고, 이벤트로 **Pull requests**를 선택합니다.

### 문서 드리프트 감지

병합된 PR을 매주 스캔하여 문서 업데이트가 필요한 API 변경을 찾습니다.

**트리거:** 예약(매주)

```bash
hermes cron create "0 9 * * 1" \
  "Scan the NousResearch/hermes-agent repo for documentation drift.

1. Run: gh pr list --repo NousResearch/hermes-agent --state merged --json number,title,files,mergedAt --limit 30
2. Filter to PRs merged in the last 7 days
3. For each merged PR, check if it modified:
   - Tool schemas (tools/*.py) — may need docs/reference/tools-reference.md update
   - CLI commands (hermes_cli/commands.py, hermes_cli/main.py) — may need docs/reference/cli-commands.md update
   - Config options (hermes_cli/config.py) — may need docs/user-guide/configuration.md update
   - Environment variables — may need docs/reference/environment-variables.md update
4. Cross-reference: for each code change, check if the corresponding docs page was also updated in the same PR

Report any gaps where code changed but docs didn't. If everything is in sync, respond with [SILENT]." \
  --name "Docs drift detection" \
  --deliver telegram
```

### 의존성 보안 감사

프로젝트 의존성에서 알려진 취약점을 매일 스캔합니다.

**트리거:** 예약(매일)

```bash
hermes cron create "0 6 * * *" \
  "Run a dependency security audit on the hermes-agent project.

1. cd ~/.hermes/hermes-agent && source .venv/bin/activate
2. Run: pip audit --format json 2>/dev/null || pip audit 2>&1
3. Run: npm audit --json 2>/dev/null (in website/ directory if it exists)
4. Check for any CVEs with CVSS score >= 7.0

If vulnerabilities found:
- List each one with package name, version, CVE ID, severity
- Check if an upgrade is available
- Note if it's a direct dependency or transitive

If no vulnerabilities, respond with [SILENT]." \
  --name "Dependency audit" \
  --deliver telegram
```

---

## DevOps 및 모니터링

### 배포 확인

배포가 완료될 때마다 스모크 테스트를 트리거합니다. CI/CD 파이프라인은 배포가 완료되면 웹훅으로 POST합니다.

**트리거:** API 호출(웹훅)

```bash
hermes webhook subscribe deploy-verify \
  --events "deployment" \
  --prompt "A deployment just completed:
Service: {service}
Environment: {environment}
Version: {version}
Deployed by: {deployer}

Run these verification steps:
1. Check if the service is responding: curl -s -o /dev/null -w '%{http_code}' {health_url}
2. Search recent logs for errors: check the deployment payload for any error indicators
3. Verify the version matches: curl -s {health_url}/version

Report: deployment status (healthy/degraded/failed), response time, any errors found.
If healthy, keep it brief. If degraded or failed, provide detailed diagnostics." \
  --deliver telegram
```

CI/CD 파이프라인에서 다음을 트리거합니다.

```bash
curl -X POST http://your-server:8644/webhooks/deploy-verify \
  -H "Content-Type: application/json" \
  -H "X-Hub-Signature-256: sha256=$(echo -n '{"service":"api","environment":"prod","version":"2.1.0","deployer":"ci","health_url":"https://api.example.com/health"}' | openssl dgst -sha256 -hmac 'your-secret' | cut -d' ' -f2)" \
  -d '{"service":"api","environment":"prod","version":"2.1.0","deployer":"ci","health_url":"https://api.example.com/health"}'
```

### 알림 분류

모니터링 알림을 최근 변경 사항과 연관 지어 응답 초안을 작성합니다. JSON을 POST할 수 있는 Datadog, PagerDuty, Grafana 또는 모든 알림 시스템에서 작동합니다.

**트리거:** API 호출(웹훅)

```bash
hermes webhook subscribe alert-triage \
  --prompt "Monitoring alert received:
Alert: {alert.name}
Severity: {alert.severity}
Service: {alert.service}
Message: {alert.message}
Timestamp: {alert.timestamp}

Investigate:
1. Search the web for known issues with this error pattern
2. Check if this correlates with any recent deployments or config changes
3. Draft a triage summary with:
   - Likely root cause
   - Suggested first response steps
   - Escalation recommendation (P1-P4)

Be concise. This goes to the on-call channel." \
  --deliver slack
```

### 가동 시간 모니터

30분마다 엔드포인트를 확인합니다. 무언가 중단된 경우에만 알립니다.

**트리거:** 예약(30분마다)

```python title="~/.hermes/scripts/check-uptime.py"
import urllib.request, json, time

ENDPOINTS = [
    {"name": "API", "url": "https://api.example.com/health"},
    {"name": "Web", "url": "https://www.example.com"},
    {"name": "Docs", "url": "https://docs.example.com"},
]

results = []
for ep in ENDPOINTS:
    try:
        start = time.time()
        req = urllib.request.Request(ep["url"], headers={"User-Agent": "Hermes-Monitor/1.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        elapsed = round((time.time() - start) * 1000)
        results.append({"name": ep["name"], "status": resp.getcode(), "ms": elapsed})
    except Exception as e:
        results.append({"name": ep["name"], "status": "DOWN", "error": str(e)})

down = [r for r in results if r.get("status") == "DOWN" or (isinstance(r.get("status"), int) and r["status"] >= 500)]
if down:
    print("OUTAGE DETECTED")
    for r in down:
        print(f"  {r['name']}: {r.get('error', f'HTTP {r[\"status\"]}')} ")
    print(f"\nAll results: {json.dumps(results, indent=2)}")
else:
    print("NO_ISSUES")
```

```bash
hermes cron create "every 30m" \
  "If the script reports OUTAGE DETECTED, summarize which services are down and suggest likely causes. If NO_ISSUES, respond with [SILENT]." \
  --script ~/.hermes/scripts/check-uptime.py \
  --name "Uptime monitor" \
  --deliver telegram
```

---

## 리서치 및 인텔리전스

### 경쟁 저장소 탐색기

경쟁 저장소에서 흥미로운 PR, 기능 및 아키텍처 결정을 모니터링합니다.

**트리거:** 예약(매일)

```bash
hermes cron create "0 8 * * *" \
  "Scout these AI agent repositories for notable activity in the last 24 hours:

Repos to check:
- anthropics/claude-code
- openai/codex
- All-Hands-AI/OpenHands
- Aider-AI/aider

For each repo:
1. gh pr list --repo <repo> --state all --json number,title,author,createdAt,mergedAt --limit 15
2. gh issue list --repo <repo> --state open --json number,title,labels,createdAt --limit 10

Focus on:
- New features being developed
- Architectural changes
- Integration patterns we could learn from
- Security fixes that might affect us too

Skip routine dependency bumps and CI fixes. If nothing notable, respond with [SILENT].
If there are findings, organize by repo with brief analysis of each item." \
  --skill competitive-pr-scout \
  --name "Competitor scout" \
  --deliver telegram
```

### AI 뉴스 다이제스트

AI/ML 발전에 대한 주간 정리입니다.

**트리거:** 예약(매주)

```bash
hermes cron create "0 9 * * 1" \
  "Generate a weekly AI news digest covering the past 7 days:

1. Search the web for major AI announcements, model releases, and research breakthroughs
2. Search for trending ML repositories on GitHub
3. Check arXiv for highly-cited papers on language models and agents

Structure:
## Headlines (3-5 major stories)
## Notable Papers (2-3 papers with one-sentence summaries)
## Open Source (interesting new repos or major releases)
## Industry Moves (funding, acquisitions, launches)

Keep each item to 1-2 sentences. Include links. Total under 600 words." \
  --name "Weekly AI digest" \
  --deliver telegram
```

### 메모가 포함된 논문 다이제스트

메모 작성 시스템에 요약을 저장하는 매일 arXiv 스캔입니다.

**트리거:** 예약(매일)

```bash
hermes cron create "0 8 * * *" \
  "Search arXiv for the 3 most interesting papers on 'language model reasoning' OR 'tool-use agents' from the past day. For each paper, create an Obsidian note with the title, authors, abstract summary, key contribution, and potential relevance to Hermes Agent development." \
  --skill arxiv --skill obsidian \
  --name "Paper digest" \
  --deliver local
```

---

## GitHub 이벤트 자동화

### 이슈 자동 라벨링

새 이슈에 자동으로 라벨을 지정하고 응답합니다.

**트리거:** GitHub 웹훅

```bash
hermes webhook subscribe github-issues \
  --events "issues" \
  --prompt "New GitHub issue received:
Repository: {repository.full_name}
Issue #{issue.number}: {issue.title}
Author: {issue.user.login}
Action: {action}
Body: {issue.body}
Labels: {issue.labels}

If this is a new issue (action=opened):
1. Read the issue title and body carefully
2. Suggest appropriate labels (bug, feature, docs, security, question)
3. If it's a bug report, check if you can identify the affected component from the description
4. Post a helpful initial response acknowledging the issue

If this is a label or assignment change, respond with [SILENT]." \
  --deliver github_comment
```

### CI 실패 분석

CI 실패를 분석하고 PR에 진단 결과를 게시합니다.

**트리거:** GitHub 웹훅

```yaml
# config.yaml route
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        ci-failure:
          events: ["check_run"]
          secret: "ci-secret"
          prompt: |
            CI check failed:
            Repository: {repository.full_name}
            Check: {check_run.name}
            Status: {check_run.conclusion}
            PR: #{check_run.pull_requests.0.number}
            Details URL: {check_run.details_url}

            If conclusion is "failure":
            1. Fetch the log from the details URL if accessible
            2. Identify the likely cause of failure
            3. Suggest a fix
            If conclusion is "success", respond with [SILENT].
          deliver: "github_comment"
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{check_run.pull_requests.0.number}"
```

### 저장소 간 변경 사항 자동 포팅

한 저장소에서 PR이 병합되면 동등한 변경을 다른 저장소로 자동 포팅합니다.

**트리거:** GitHub 웹훅

```bash
hermes webhook subscribe auto-port \
  --events "pull_request" \
  --prompt "PR merged in the source repository:
Repository: {repository.full_name}
PR #{pull_request.number}: {pull_request.title}
Author: {pull_request.user.login}
Action: {action}
Merge commit: {pull_request.merge_commit_sha}

If action is 'closed' and pull_request.merged is true:
1. Fetch the diff: curl -sL {pull_request.diff_url}
2. Analyze what changed
3. Determine if this change needs to be ported to the Go SDK equivalent
4. If yes, create a branch, apply the equivalent changes, and open a PR on the target repo
5. Reference the original PR in the new PR description

If action is not 'closed' or not merged, respond with [SILENT]." \
  --skills github-pr-workflow \
  --deliver log
```

---

## 비즈니스 운영

### Stripe 결제 모니터링

결제 이벤트를 추적하고 실패 요약을 받습니다.

**트리거:** API 호출(웹훅)

```bash
hermes webhook subscribe stripe-payments \
  --events "payment_intent.succeeded,payment_intent.payment_failed,charge.dispute.created" \
  --prompt "Stripe event received:
Event type: {type}
Amount: {data.object.amount} cents ({data.object.currency})
Customer: {data.object.customer}
Status: {data.object.status}

For payment_intent.payment_failed:
- Identify the failure reason from {data.object.last_payment_error}
- Suggest whether this is a transient issue (retry) or permanent (contact customer)

For charge.dispute.created:
- Flag as urgent
- Summarize the dispute details

For payment_intent.succeeded:
- Brief confirmation only

Keep responses concise for the ops channel." \
  --deliver slack
```

### 일일 수익 요약

매일 아침 핵심 비즈니스 지표를 취합합니다.

**트리거:** 예약(매일)

```bash
hermes cron create "0 8 * * *" \
  "Generate a morning business metrics summary.

Search the web for:
1. Current Bitcoin and Ethereum prices
2. S&P 500 status (pre-market or previous close)
3. Any major tech/AI industry news from the last 12 hours

Format as a brief morning briefing, 3-4 bullet points max.
Deliver as a clean, scannable message." \
  --name "Morning briefing" \
  --deliver telegram
```

---

## 다중 스킬 워크플로

### 보안 감사 파이프라인

여러 스킬을 결합하여 매주 종합 보안 검토를 수행합니다.

**트리거:** 예약(매주)

```bash
hermes cron create "0 3 * * 0" \
  "Run a comprehensive security audit of the hermes-agent codebase.

1. Check for dependency vulnerabilities (pip audit, npm audit)
2. Search the codebase for common security anti-patterns:
   - Hardcoded secrets or API keys
   - SQL injection vectors (string formatting in queries)
   - Path traversal risks (user input in file paths without validation)
   - Unsafe deserialization (pickle.loads, yaml.load without SafeLoader)
3. Review recent commits (last 7 days) for security-relevant changes
4. Check if any new environment variables were added without being documented

Write a security report with findings categorized by severity (Critical, High, Medium, Low).
If nothing found, report a clean bill of health." \
  --skill codebase-security-audit \
  --name "Weekly security audit" \
  --deliver telegram
```

### 콘텐츠 파이프라인

일정에 따라 콘텐츠를 리서치하고, 초안을 작성하고, 준비합니다.

**트리거:** 예약(매주)

```bash
hermes cron create "0 10 * * 3" \
  "Research and draft a technical blog post outline about a trending topic in AI agents.

1. Search the web for the most discussed AI agent topics this week
2. Pick the most interesting one that's relevant to open-source AI agents
3. Create an outline with:
   - Hook/intro angle
   - 3-4 key sections
   - Technical depth appropriate for developers
   - Conclusion with actionable takeaway
4. Save the outline to ~/drafts/blog-$(date +%Y%m%d).md

Keep the outline to ~300 words. This is a starting point, not a finished post." \
  --name "Blog outline" \
  --deliver local
```

---

## 빠른 참조

### Cron 예약 구문

| 표현식 | 의미 |
|-----------|---------|
| `every 30m` | 30분마다 |
| `every 2h` | 2시간마다 |
| `0 2 * * *` | 매일 오전 2시 |
| `0 9 * * 1` | 매주 월요일 오전 9시 |
| `0 9 * * 1-5` | 평일 오전 9시 |
| `0 3 * * 0` | 매주 일요일 오전 3시 |
| `0 */6 * * *` | 6시간마다 |

### 전달 대상

| 대상 | 플래그 | 참고 |
|--------|------|-------|
| 동일한 채팅 | `--deliver origin` | 기본값 — 작업이 생성된 곳으로 전달 |
| 로컬 파일 | `--deliver local` | 출력을 저장하며 알림은 보내지 않음 |
| Telegram | `--deliver telegram` | 홈 채널 또는 특정 채널에는 `telegram:CHAT_ID` |
| Discord | `--deliver discord` | 홈 채널 또는 특정 채널에는 `discord:CHANNEL_ID` |
| Slack | `--deliver slack` | 홈 채널 |
| SMS | `--deliver sms:+15551234567` | 전화번호로 직접 전달 |
| 특정 스레드 | `--deliver telegram:-100123:456` | Telegram 포럼 주제 |

### 웹훅 템플릿 변수

| 변수 | 설명 |
|----------|-------------|
| `{pull_request.title}` | PR 제목 |
| `{issue.number}` | 이슈 번호 |
| `{repository.full_name}` | `owner/repo` |
| `{action}` | 이벤트 작업(열림, 닫힘 등) |
| `{__raw__}` | 전체 JSON 페이로드(4000자로 잘림) |
| `{sender.login}` | 이벤트를 트리거한 GitHub 사용자 |

### [SILENT] 패턴

cron 작업의 응답에 `[SILENT]`가 포함되면 전달이 억제됩니다. 조용한 실행에서 알림이 쏟아지는 것을 방지하려면 다음을 사용하세요.

```
If nothing noteworthy happened, respond with [SILENT].
```

즉, 에이전트가 보고할 내용이 있을 때만 알림을 받습니다.
