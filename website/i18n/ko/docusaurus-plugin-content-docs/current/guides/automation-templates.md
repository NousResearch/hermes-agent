---
sidebar_position: 15
title: "자동화 템플릿 (Automation Templates)"
description: "즉시 사용 가능한 자동화 레시피 — 예약된 작업, GitHub 이벤트 트리거, API 웹훅 및 다중 스킬 워크플로우"
---

# 자동화 템플릿 (Automation Templates)

일반적인 자동화 패턴을 위한 복사-붙여넣기 레시피입니다. 각 템플릿은 시간 기반 트리거를 위해 Hermes에 내장된 [크론 스케줄러](/user-guide/features/cron)를 사용하고, 이벤트 기반 트리거를 위해 [웹훅 플랫폼](/user-guide/messaging/webhooks)을 사용합니다.

모든 템플릿은 단일 제공자(provider)에 종속되지 않으며 **어떤 모델과도** 작동합니다.

:::tip 세 가지 트리거 유형
| 트리거 (Trigger) | 설명 (How) | 도구 (Tool) |
|---------|-----|------|
| **일정 (Schedule)** | 주기적으로 실행 (매시간, 매일 밤, 매주) | `cronjob` 도구 또는 `/cron` 슬래시 명령어 |
| **GitHub 이벤트** | PR 열기, 푸시, 이슈, CI 결과에 따라 실행 | 웹훅 플랫폼 (`hermes webhook subscribe`) |
| **API 호출 (API Call)** | 외부 서비스가 귀하의 엔드포인트로 JSON을 POST | 웹훅 플랫폼 (config.yaml 라우팅 또는 `hermes webhook subscribe`) |

세 가지 모두 Telegram, Discord, Slack, SMS, 이메일, GitHub 댓글 또는 로컬 파일로의 전송을 지원합니다.
:::

---

## 개발 워크플로우 (Development Workflow)

### 매일 밤 백로그 분류 (Nightly Backlog Triage)

매일 밤 새로운 이슈에 라벨을 지정하고, 우선순위를 매기고, 요약합니다. 팀 채널로 요약본을 전송합니다.

**트리거:** 일정 (매일 밤)

```bash
hermes cron create "0 2 * * *" \
  "당신은 NousResearch/hermes-agent GitHub 리포지토리를 분류하는 프로젝트 관리자입니다.

1. 실행: gh issue list --repo NousResearch/hermes-agent --state open --json number,title,labels,author,createdAt --limit 30
2. 지난 24시간 동안 열린 이슈 식별
3. 각 새 이슈에 대해:
   - 우선순위 라벨 제안 (P0-critical, P1-high, P2-medium, P3-low)
   - 카테고리 라벨 제안 (bug, feature, docs, security)
   - 한 줄 분류 노트 작성
4. 요약: 총 미해결 이슈, 오늘 새로 생성된 이슈, 우선순위별 분석

깔끔한 요약 형식으로 작성하세요. 새 이슈가 없으면 [SILENT]로 응답하세요." \
  --name "Nightly backlog triage" \
  --deliver telegram
```

### 자동 PR 코드 리뷰

풀 리퀘스트가 열릴 때마다 자동으로 리뷰합니다. PR에 직접 리뷰 댓글을 게시합니다.

**트리거:** GitHub 웹훅

**옵션 A — 동적 구독 (CLI):**

```bash
hermes webhook subscribe github-pr-review \
  --events "pull_request" \
  --prompt "이 풀 리퀘스트를 검토하세요:
리포지토리: {repository.full_name}
PR #{pull_request.number}: {pull_request.title}
작성자: {pull_request.user.login}
작업: {action}
Diff URL: {pull_request.diff_url}

다음 명령어로 diff를 가져오세요: curl -sL {pull_request.diff_url}

다음 사항을 검토하세요:
- 보안 문제 (인젝션, 인증 우회, 코드 내 기밀 정보)
- 성능 문제 (N+1 쿼리, 무한 루프, 메모리 누수)
- 코드 품질 (명명, 중복, 오류 처리)
- 새로운 동작에 대한 테스트 누락

간결한 리뷰를 게시하세요. PR이 사소한 문서/오타 수정인 경우 짧게 언급만 하세요." \
  --skill github-code-review \
  --deliver github_comment
```

**옵션 B — 정적 라우팅 (config.yaml):**

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
            PR 검토 #{pull_request.number}: {pull_request.title}
            리포지토리: {repository.full_name}
            작성자: {pull_request.user.login}
            Diff URL: {pull_request.diff_url}
            보안, 성능, 코드 품질을 검토하세요.
          skills: ["github-code-review"]
          deliver: "github_comment"
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{pull_request.number}"
```

그런 다음 GitHub에서: **Settings → Webhooks → Add webhook** → Payload URL: `http://your-server:8644/webhooks/github-pr-review`, Content type: `application/json`, Secret: `github-webhook-secret`, Events: **Pull requests**.

### 문서 드리프트 감지 (Docs Drift Detection)

병합된 PR을 매주 스캔하여 문서 업데이트가 필요한 API 변경 사항을 찾습니다.

**트리거:** 일정 (매주)

```bash
hermes cron create "0 9 * * 1" \
  "NousResearch/hermes-agent 리포지토리에서 문서 드리프트를 스캔하세요.

1. 실행: gh pr list --repo NousResearch/hermes-agent --state merged --json number,title,files,mergedAt --limit 30
2. 지난 7일 동안 병합된 PR로 필터링
3. 병합된 각 PR에 대해 다음 항목이 수정되었는지 확인:
   - 도구 스키마 (tools/*.py) — docs/reference/tools-reference.md 업데이트 필요 가능성
   - CLI 명령어 (hermes_cli/commands.py, hermes_cli/main.py) — docs/reference/cli-commands.md 업데이트 필요 가능성
   - 구성 옵션 (hermes_cli/config.py) — docs/user-guide/configuration.md 업데이트 필요 가능성
   - 환경 변수 — docs/reference/environment-variables.md 업데이트 필요 가능성
4. 교차 참조: 각 코드 변경 사항에 대해 해당 문서 페이지도 같은 PR에서 업데이트되었는지 확인

코드는 변경되었지만 문서가 변경되지 않은 간격을 보고하세요. 모든 것이 동기화되어 있으면 [SILENT]로 응답하세요." \
  --name "Docs drift detection" \
  --deliver telegram
```

### 종속성 보안 감사 (Dependency Security Audit)

프로젝트 종속성에서 알려진 취약점을 매일 스캔합니다.

**트리거:** 일정 (매일)

```bash
hermes cron create "0 6 * * *" \
  "hermes-agent 프로젝트에서 종속성 보안 감사를 실행하세요.

1. cd ~/.hermes/hermes-agent && source .venv/bin/activate
2. 실행: pip audit --format json 2>/dev/null || pip audit 2>&1
3. 실행: npm audit --json 2>/dev/null (website/ 디렉토리가 있는 경우 해당 디렉토리에서)
4. CVSS 점수가 7.0 이상인 CVE 확인

취약점이 발견된 경우:
- 패키지 이름, 버전, CVE ID, 심각도와 함께 각각 나열
- 업그레이드 사용 가능 여부 확인
- 직접 종속성인지 전이적(transitive) 종속성인지 기록

취약점이 없으면 [SILENT]로 응답하세요." \
  --name "Dependency audit" \
  --deliver telegram
```

---

## DevOps & 모니터링

### 배포 검증 (Deploy Verification)

모든 배포 후 스모크 테스트를 트리거합니다. CI/CD 파이프라인이 배포가 완료되면 웹훅으로 POST합니다.

**트리거:** API 호출 (웹훅)

```bash
hermes webhook subscribe deploy-verify \
  --events "deployment" \
  --prompt "배포가 방금 완료되었습니다:
서비스: {service}
환경: {environment}
버전: {version}
배포자: {deployer}

다음 검증 단계를 실행하세요:
1. 서비스가 응답하는지 확인: curl -s -o /dev/null -w '%{http_code}' {health_url}
2. 최근 로그에서 오류 검색: 배포 페이로드에서 오류 표시기를 확인
3. 버전이 일치하는지 확인: curl -s {health_url}/version

보고: 배포 상태(정상/저하/실패), 응답 시간, 발견된 오류.
정상이면 간략하게 유지하세요. 저하되거나 실패한 경우 자세한 진단 정보를 제공하세요." \
  --deliver telegram
```

CI/CD 파이프라인 트리거 예시:

```bash
curl -X POST http://your-server:8644/webhooks/deploy-verify \
  -H "Content-Type: application/json" \
  -H "X-Hub-Signature-256: sha256=$(echo -n '{"service":"api","environment":"prod","version":"2.1.0","deployer":"ci","health_url":"https://api.example.com/health"}' | openssl dgst -sha256 -hmac 'your-secret' | cut -d' ' -f2)" \
  -d '{"service":"api","environment":"prod","version":"2.1.0","deployer":"ci","health_url":"https://api.example.com/health"}'
```

### 경고 분류 (Alert Triage)

모니터링 알림과 최근 변경 사항을 연관시켜 대응 초안을 작성합니다. Datadog, PagerDuty, Grafana 또는 JSON을 POST할 수 있는 모든 알림 시스템과 작동합니다.

**트리거:** API 호출 (웹훅)

```bash
hermes webhook subscribe alert-triage \
  --prompt "모니터링 알림 수신됨:
알림: {alert.name}
심각도: {alert.severity}
서비스: {alert.service}
메시지: {alert.message}
타임스탬프: {alert.timestamp}

조사:
1. 웹에서 이 오류 패턴에 대해 알려진 문제를 검색
2. 이 알림이 최근 배포나 구성 변경과 상관관계가 있는지 확인
3. 다음이 포함된 분류 요약 초안 작성:
   - 예상되는 근본 원인
   - 제안하는 1차 대응 단계
   - 에스컬레이션 권장 사항 (P1-P4)

간결하게 작성하세요. 이는 온콜(on-call) 채널로 전달됩니다." \
  --deliver slack
```

### 업타임 모니터 (Uptime Monitor)

30분마다 엔드포인트를 확인합니다. 무언가 다운되었을 때만 알림을 보냅니다.

**트리거:** 일정 (30분마다)

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
    print("장애 감지됨 (OUTAGE DETECTED)")
    for r in down:
        print(f"  {r['name']}: {r.get('error', f'HTTP {r[\"status\"]}')} ")
    print(f"\n모든 결과: {json.dumps(results, indent=2)}")
else:
    print("NO_ISSUES")
```

```bash
hermes cron create "every 30m" \
  "스크립트가 OUTAGE DETECTED를 보고하면 다운된 서비스를 요약하고 예상되는 원인을 제안하세요. NO_ISSUES이면 [SILENT]로 응답하세요." \
  --script ~/.hermes/scripts/check-uptime.py \
  --name "Uptime monitor" \
  --deliver telegram
```

---

## 리서치 및 인텔리전스 (Research & Intelligence)

### 경쟁사 리포지토리 스카우트 (Competitive Repository Scout)

경쟁사의 리포지토리에서 흥미로운 PR, 기능 및 아키텍처 결정을 모니터링합니다.

**트리거:** 일정 (매일)

```bash
hermes cron create "0 8 * * *" \
  "다음 AI 에이전트 리포지토리에서 지난 24시간 동안 주목할 만한 활동을 스카우트하세요:

확인할 리포지토리:
- anthropics/claude-code
- openai/codex
- All-Hands-AI/OpenHands
- Aider-AI/aider

각 리포지토리에 대해:
1. gh pr list --repo <repo> --state all --json number,title,author,createdAt,mergedAt --limit 15
2. gh issue list --repo <repo> --state open --json number,title,labels,createdAt --limit 10

초점:
- 개발 중인 새로운 기능
- 아키텍처 변경
- 우리가 배울 수 있는 통합 패턴
- 우리에게도 영향을 미칠 수 있는 보안 수정

일상적인 종속성 업데이트와 CI 수정은 건너뛰세요. 주목할 만한 사항이 없으면 [SILENT]로 응답하세요.
발견된 사항이 있다면, 각 항목에 대한 간략한 분석과 함께 리포지토리별로 구성하세요." \
  --skill competitive-pr-scout \
  --name "Competitor scout" \
  --deliver telegram
```

### AI 뉴스 요약 (AI News Digest)

주간 AI/ML 개발 동향을 정리합니다.

**트리거:** 일정 (매주)

```bash
hermes cron create "0 9 * * 1" \
  "지난 7일 동안을 다루는 주간 AI 뉴스 요약을 생성하세요:

1. 주요 AI 발표, 모델 출시, 연구 돌파구를 웹에서 검색
2. GitHub에서 트렌딩 ML 리포지토리 검색
3. 언어 모델 및 에이전트에 대해 인용이 많은 논문을 arXiv에서 확인

구조:
## 헤드라인 (3-5개 주요 스토리)
## 주목할 만한 논문 (2-3개 논문과 한 문장 요약)
## 오픈 소스 (흥미로운 새 리포지토리 또는 주요 릴리스)
## 업계 동향 (투자, 인수, 출시)

각 항목은 1-2문장으로 유지하세요. 링크를 포함하세요. 전체 분량은 600단어 미만으로 작성하세요." \
  --name "Weekly AI digest" \
  --deliver telegram
```

### 논문 요약과 노트 (Paper Digest with Notes)

arXiv를 매일 스캔하여 노트 작성 시스템에 요약을 저장합니다.

**트리거:** 일정 (매일)

```bash
hermes cron create "0 8 * * *" \
  "지난 하루 동안 '언어 모델 추론(language model reasoning)' 또는 '도구 사용 에이전트(tool-use agents)'에 관한 가장 흥미로운 논문 3개를 arXiv에서 검색하세요. 각 논문에 대해 제목, 저자, 초록 요약, 주요 기여도, 그리고 Hermes Agent 개발과의 잠재적 관련성이 포함된 Obsidian 노트를 작성하세요." \
  --skill arxiv --skill obsidian \
  --name "Paper digest" \
  --deliver local
```

---

## GitHub 이벤트 자동화

### 이슈 자동 라벨링 (Issue Auto-Labeling)

새 이슈에 자동으로 라벨을 지정하고 응답합니다.

**트리거:** GitHub 웹훅

```bash
hermes webhook subscribe github-issues \
  --events "issues" \
  --prompt "새 GitHub 이슈 수신됨:
리포지토리: {repository.full_name}
이슈 #{issue.number}: {issue.title}
작성자: {issue.user.login}
작업: {action}
본문: {issue.body}
라벨: {issue.labels}

새 이슈인 경우 (action=opened):
1. 이슈 제목과 본문을 주의 깊게 읽음
2. 적절한 라벨 제안 (bug, feature, docs, security, question)
3. 버그 보고서인 경우, 설명을 통해 영향을 받는 구성 요소를 식별할 수 있는지 확인
4. 이슈를 확인하는 유용한 초기 응답 게시

라벨 또는 할당 변경인 경우 [SILENT]로 응답하세요." \
  --deliver github_comment
```

### CI 실패 분석 (CI Failure Analysis)

CI 실패를 분석하고 PR에 진단 결과를 게시합니다.

**트리거:** GitHub 웹훅

```yaml
# config.yaml 경로 설정
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        ci-failure:
          events: ["check_run"]
          secret: "ci-secret"
          prompt: |
            CI 체크 실패:
            리포지토리: {repository.full_name}
            체크: {check_run.name}
            상태: {check_run.conclusion}
            PR: #{check_run.pull_requests.0.number}
            상세 URL: {check_run.details_url}

            conclusion이 "failure"인 경우:
            1. 상세 URL에 접근 가능하면 로그를 가져옴
            2. 예상되는 실패 원인 파악
            3. 수정 사항 제안
            conclusion이 "success"인 경우 [SILENT]로 응답.
          deliver: "github_comment"
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{check_run.pull_requests.0.number}"
```

### 리포지토리 간 변경 사항 자동 포팅 (Auto-Port Changes Across Repos)

한 리포지토리에서 PR이 병합되면, 다른 리포지토리의 동등한 변경 사항으로 자동으로 포팅합니다.

**트리거:** GitHub 웹훅

```bash
hermes webhook subscribe auto-port \
  --events "pull_request" \
  --prompt "소스 리포지토리에서 PR이 병합됨:
리포지토리: {repository.full_name}
PR #{pull_request.number}: {pull_request.title}
작성자: {pull_request.user.login}
작업: {action}
병합 커밋: {pull_request.merge_commit_sha}

action이 'closed'이고 pull_request.merged가 true인 경우:
1. diff 가져오기: curl -sL {pull_request.diff_url}
2. 변경된 내용 분석
3. 이 변경 사항을 Go SDK 동등물로 포팅해야 하는지 판단
4. 맞다면, 브랜치를 생성하고 동등한 변경 사항을 적용한 후 대상 리포지토리에 PR을 열기
5. 새 PR 설명에 원본 PR 참조

action이 'closed'가 아니거나 병합되지 않은 경우 [SILENT]로 응답하세요." \
  --skill github-pr-workflow \
  --deliver log
```

---

## 비즈니스 운영 (Business Operations)

### Stripe 결제 모니터링

결제 이벤트를 추적하고 실패 요약을 받습니다.

**트리거:** API 호출 (웹훅)

```bash
hermes webhook subscribe stripe-payments \
  --events "payment_intent.succeeded,payment_intent.payment_failed,charge.dispute.created" \
  --prompt "Stripe 이벤트 수신됨:
이벤트 유형: {type}
금액: {data.object.amount} 센트 ({data.object.currency})
고객: {data.object.customer}
상태: {data.object.status}

payment_intent.payment_failed의 경우:
- {data.object.last_payment_error}에서 실패 원인 식별
- 일시적인 문제(재시도)인지 영구적인 문제(고객에게 연락)인지 제안

charge.dispute.created의 경우:
- 긴급(urgent)으로 표시
- 분쟁 세부 정보 요약

payment_intent.succeeded의 경우:
- 간략한 확인만

운영 채널을 위해 응답을 간결하게 유지하세요." \
  --deliver slack
```

### 일일 수익 요약 (Daily Revenue Summary)

매일 아침 주요 비즈니스 지표를 컴파일합니다.

**트리거:** 일정 (매일)

```bash
hermes cron create "0 8 * * *" \
  "아침 비즈니스 지표 요약을 생성하세요.

웹에서 검색:
1. 현재 비트코인 및 이더리움 가격
2. S&P 500 상태 (장전 또는 이전 종가)
3. 지난 12시간 동안의 주요 기술/AI 산업 뉴스

3-4개의 글머리 기호(bullet points)로 이루어진 짧은 아침 브리핑 형식으로 포맷하세요.
깔끔하고 스캔하기 쉬운 메시지로 전송하세요." \
  --name "Morning briefing" \
  --deliver telegram
```

---

## 다중 스킬 워크플로우 (Multi-Skill Workflows)

### 보안 감사 파이프라인 (Security Audit Pipeline)

종합적인 주간 보안 검토를 위해 여러 스킬을 결합합니다.

**트리거:** 일정 (매주)

```bash
hermes cron create "0 3 * * 0" \
  "hermes-agent 코드베이스에 대한 포괄적인 보안 감사를 실행하세요.

1. 종속성 취약점 확인 (pip audit, npm audit)
2. 코드베이스에서 일반적인 보안 안티 패턴 검색:
   - 하드코딩된 시크릿 또는 API 키
   - SQL 인젝션 벡터 (쿼리 내 문자열 포맷팅)
   - 경로 탐색(Path traversal) 위험 (유효성 검사 없는 파일 경로 내 사용자 입력)
   - 안전하지 않은 역직렬화(deserialization) (SafeLoader 없는 pickle.loads, yaml.load)
3. 최근 커밋(지난 7일)에서 보안과 관련된 변경 사항 검토
4. 문서화되지 않은 채 추가된 새 환경 변수가 있는지 확인

심각도(Critical, High, Medium, Low)별로 분류된 보안 보고서를 작성하세요.
아무것도 발견되지 않으면 건강한 상태라고 보고하세요." \
  --skill codebase-security-audit \
  --name "Weekly security audit" \
  --deliver telegram
```

### 콘텐츠 파이프라인 (Content Pipeline)

일정에 따라 콘텐츠를 리서치하고 초안을 작성하며 준비합니다.

**트리거:** 일정 (매주)

```bash
hermes cron create "0 10 * * 3" \
  "AI 에이전트의 트렌딩 주제에 대한 기술 블로그 게시물 개요를 리서치하고 초안을 작성하세요.

1. 이번 주에 가장 많이 논의된 AI 에이전트 주제를 웹에서 검색
2. 오픈 소스 AI 에이전트와 관련이 있고 가장 흥미로운 주제 선택
3. 다음이 포함된 개요 작성:
   - 후킹/도입부 관점
   - 3-4개의 핵심 섹션
   - 개발자에게 적합한 기술적 깊이
   - 실행 가능한 시사점이 있는 결론
4. 개요를 ~/drafts/blog-$(date +%Y%m%d).md 에 저장

개요는 약 300단어로 유지하세요. 이는 완성된 게시물이 아니라 시작점입니다." \
  --name "Blog outline" \
  --deliver local
```

---

## 빠른 참조 (Quick Reference)

### 크론 일정 구문 (Cron Schedule Syntax)

| 표현식 | 의미 |
|-----------|---------|
| `every 30m` | 30분마다 |
| `every 2h` | 2시간마다 |
| `0 2 * * *` | 매일 오전 2:00 |
| `0 9 * * 1` | 매주 월요일 오전 9:00 |
| `0 9 * * 1-5` | 평일 오전 9:00 |
| `0 3 * * 0` | 매주 일요일 오전 3:00 |
| `0 */6 * * *` | 6시간마다 |

### 전송 대상 (Delivery Targets)

| 대상 (Target) | 플래그 | 참고 사항 |
|--------|------|-------|
| 동일한 채팅 | `--deliver origin` | 기본값 — 작업이 생성된 곳으로 전송 |
| 로컬 파일 | `--deliver local` | 알림 없이 출력을 저장 |
| Telegram | `--deliver telegram` | 홈 채널, 또는 특정한 경우 `telegram:CHAT_ID` |
| Discord | `--deliver discord` | 홈 채널, 또는 `discord:CHANNEL_ID` |
| Slack | `--deliver slack` | 홈 채널 |
| SMS | `--deliver sms:+15551234567` | 전화번호로 직접 전송 |
| 특정 스레드 | `--deliver telegram:-100123:456` | Telegram 포럼 토픽 |

### 웹훅 템플릿 변수 (Webhook Template Variables)

| 변수 | 설명 |
|----------|-------------|
| `{pull_request.title}` | PR 제목 |
| `{issue.number}` | 이슈 번호 |
| `{repository.full_name}` | `owner/repo` |
| `{action}` | 이벤트 동작 (opened, closed 등) |
| `{__raw__}` | 전체 JSON 페이로드 (4000자로 제한됨) |
| `{sender.login}` | 이벤트를 트리거한 GitHub 사용자 |

### [SILENT] 패턴

크론 작업의 응답에 `[SILENT]`가 포함되어 있으면 전송이 억제됩니다. 조용한 실행 중에 알림 스팸을 방지하려면 이를 사용하세요:

```
주목할 만한 일이 일어나지 않으면 [SILENT]로 응답하세요.
```

이는 에이전트가 보고할 내용이 있을 때만 알림을 받는다는 것을 의미합니다.
