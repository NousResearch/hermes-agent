---
sidebar_position: 10
title: "튜토리얼: GitHub PR 리뷰 에이전트"
description: "저장소를 모니터링하고, pull request를 검토하며, 피드백을 자동으로 전달하는 AI 코드 리뷰어 만들기"
---

# 튜토리얼: GitHub PR 리뷰 에이전트 만들기

**문제:** 팀에서 검토할 수 있는 속도보다 빠르게 PR을 엽니다. PR은 검토자의 눈길을 기다리며 며칠씩 남아 있습니다. 아무도 확인할 시간이 없어 주니어 개발자가 버그를 병합합니다. 아침마다 코드를 만드는 대신 diff를 따라잡는 데 시간을 씁니다.

**해결책:** 24시간 저장소를 지켜보며 새 PR을 모두 검토하고 버그, 보안 문제, 코드 품질을 점검한 뒤 요약을 보내는 AI 에이전트입니다 — 실제로 사람의 판단이 필요한 PR에만 시간을 쓰면 됩니다.

**만들게 될 것:**

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   Cron Timer  ──▶  Hermes Agent  ──▶  GitHub API  ──▶  Review     │
│   (every 2h)       + gh CLI           (PR diffs)       delivery   │
│                    + skill                             (Telegram, │
│                    + memory                            Discord,   │
│                                                        local)     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

이 가이드는 **cron 작업**을 사용해 일정에 따라 PR을 조회하므로 서버나 공개 엔드포인트가 필요하지 않습니다. NAT와 방화벽 뒤에서도 작동합니다.

:::tip 실시간 리뷰를 원하시나요?
공개 엔드포인트를 사용할 수 있다면 [웹훅으로 자동 GitHub PR 댓글 달기](./webhook-github-pr-review.md)를 확인하세요 — PR이 열리거나 업데이트될 때 GitHub가 Hermes로 이벤트를 즉시 보냅니다.
:::

---

## 사전 요구 사항

- **Hermes Agent 설치** — [설치 가이드](/getting-started/installation) 참고
- **cron 작업을 위한 게이트웨이 실행:**
  ```bash
  hermes gateway install   # Install as a service
  # or
  hermes gateway           # Run in foreground
  ```
- **GitHub CLI(`gh`) 설치 및 인증:**
  ```bash
  # Install
  brew install gh        # macOS
  sudo apt install gh    # Ubuntu/Debian

  # Authenticate
  gh auth login
  ```
- **메시징 설정**(선택 사항) — [Telegram](/user-guide/messaging/telegram) 또는 [Discord](/user-guide/messaging/discord)

:::tip 메시징이 없나요? 문제없습니다
`deliver: "local"`을 사용하면 리뷰가 `~/.hermes/cron/output/`에 저장됩니다. 알림을 연결하기 전에 테스트하기에 좋습니다.
:::

---

## 1단계: 설정 확인

Hermes가 GitHub에 접근할 수 있는지 확인합니다. 채팅을 시작하세요.

```bash
hermes
```

간단한 명령으로 테스트합니다.

```
Run: gh pr list --repo NousResearch/hermes-agent --state open --limit 3
```

열린 PR 목록이 표시되어야 합니다. 작동한다면 준비가 된 것입니다.

---

## 2단계: 수동 리뷰 시도

채팅을 계속 열어 둔 상태에서 Hermes에 실제 PR을 검토해 달라고 요청합니다.

```
Review this pull request. Read the diff, check for bugs, security issues,
and code quality. Be specific about line numbers and quote problematic code.

Run: gh pr diff 3888 --repo NousResearch/hermes-agent
```

Hermes는 다음을 수행합니다.
1. `gh pr diff`를 실행해 코드 변경 사항을 가져옵니다.
2. 전체 diff를 읽습니다.
3. 구체적인 발견 사항이 포함된 구조화된 리뷰를 작성합니다.

품질이 만족스럽다면 자동화할 차례입니다.

---

## 3단계: 리뷰 스킬 만들기

스킬을 사용하면 여러 세션과 cron 실행에 걸쳐 Hermes가 일관된 리뷰 지침을 따릅니다. 스킬이 없으면 리뷰 품질이 달라집니다.

```bash
mkdir -p ~/.hermes/skills/code-review
```

`~/.hermes/skills/code-review/SKILL.md`를 만듭니다.

```markdown
---
name: code-review
description: Review pull requests for bugs, security issues, and code quality
---

# Code Review Guidelines

When reviewing a pull request:

## What to Check
1. **Bugs** — Logic errors, off-by-one, null/undefined handling
2. **Security** — Injection, auth bypass, secrets in code, SSRF
3. **Performance** — N+1 queries, unbounded loops, memory leaks
4. **Style** — Naming conventions, dead code, missing error handling
5. **Tests** — Are changes tested? Do tests cover edge cases?

## Output Format
For each finding:
- **File:Line** — exact location
- **Severity** — Critical / Warning / Suggestion
- **What's wrong** — one sentence
- **Fix** — how to fix it

## Rules
- Be specific. Quote the problematic code.
- Don't flag style nitpicks unless they affect readability.
- If the PR looks good, say so. Don't invent problems.
- End with: APPROVE / REQUEST_CHANGES / COMMENT
```

로드되었는지 확인합니다 — `hermes`를 시작하면 시작 시 스킬 목록에 `code-review`가 표시되어야 합니다.

---

## 4단계: 팀의 규칙 가르치기

리뷰어를 실제로 유용하게 만드는 단계입니다. 세션을 시작하고 Hermes에 팀의 표준을 가르치세요.

```
Remember: In our backend repo, we use Python with FastAPI.
All endpoints must have type annotations and Pydantic models.
We don't allow raw SQL — only SQLAlchemy ORM.
Test files go in tests/ and must use pytest fixtures.
```

```
Remember: In our frontend repo, we use TypeScript with React.
No `any` types allowed. All components must have props interfaces.
We use React Query for data fetching, never useEffect for API calls.
```

이 메모리는 영구적으로 유지됩니다 — 리뷰어는 매번 지시하지 않아도 팀의 규칙을 적용합니다.

---

## 5단계: 자동화된 Cron 작업 만들기

이제 모든 요소를 연결합니다. 2시간마다 실행되는 cron 작업을 만드세요.

```bash
hermes cron create "0 */2 * * *" \
  "Check for new open PRs and review them.

Repos to monitor:
- myorg/backend-api
- myorg/frontend-app

Steps:
1. Run: gh pr list --repo REPO --state open --limit 5 --json number,title,author,createdAt
2. For each PR created or updated in the last 4 hours:
   - Run: gh pr diff NUMBER --repo REPO
   - Review the diff using the code-review guidelines
3. Format output as:

## PR Reviews — today

### [repo] #[number]: [title]
**Author:** [name] | **Verdict:** APPROVE/REQUEST_CHANGES/COMMENT
[findings]

If no new PRs found, say: No new PRs to review." \
  --name "pr-review" \
  --deliver telegram \
  --skill code-review
```

예약되었는지 확인합니다.

```bash
hermes cron list
```

### 유용한 다른 일정

| 일정 | 시점 |
|----------|------|
| `0 */2 * * *` | 2시간마다 |
| `0 9,13,17 * * 1-5` | 평일 하루 세 번 |
| `0 9 * * 1` | 매주 월요일 아침 요약 |
| `30m` | 30분마다(트래픽이 많은 저장소) |

---

## 6단계: 필요할 때 실행

일정을 기다리고 싶지 않나요? 수동으로 트리거합니다.

```bash
hermes cron run pr-review
```

또는 채팅 세션 안에서 실행합니다.

```
/cron run pr-review
```

---

## 더 알아보기

### 리뷰를 GitHub에 바로 게시

Telegram으로 전달하는 대신 에이전트가 PR 자체에 댓글을 달도록 할 수 있습니다.

cron 프롬프트에 다음을 추가합니다.

```
After reviewing, post your review:
- For issues: gh pr review NUMBER --repo REPO --comment --body "YOUR_REVIEW"
- For critical issues: gh pr review NUMBER --repo REPO --request-changes --body "YOUR_REVIEW"
- For clean PRs: gh pr review NUMBER --repo REPO --approve --body "Looks good"
```

:::caution
`gh`에 `repo` 범위의 토큰이 있는지 확인하세요. 리뷰는 `gh`가 인증된 사용자로 게시됩니다.
:::

### 주간 PR 대시보드

모든 저장소의 월요일 아침 개요를 만듭니다.

```bash
hermes cron create "0 9 * * 1" \
  "Generate a weekly PR dashboard:
- myorg/backend-api
- myorg/frontend-app
- myorg/infra

For each repo show:
1. Open PR count and oldest PR age
2. PRs merged this week
3. Stale PRs (older than 5 days)
4. PRs with no reviewer assigned

Format as a clean summary." \
  --name "weekly-dashboard" \
  --deliver telegram
```

### 여러 저장소 모니터링

프롬프트에 저장소를 더 추가해 확장하세요. 에이전트가 순차적으로 처리하므로 추가 설정이 필요하지 않습니다.

---

## 문제 해결

### "gh: command not found"

게이트웨이는 최소 환경에서 실행됩니다. `gh`가 시스템 PATH에 있는지 확인하고 게이트웨이를 다시 시작하세요.

### 리뷰가 너무 일반적입니다
1. `code-review` 스킬을 추가합니다(3단계).
2. 메모리를 통해 Hermes에 팀의 규칙을 가르칩니다(4단계).
3. 기술 스택에 대한 맥락이 많을수록 리뷰가 좋아집니다.

### Cron 작업이 실행되지 않습니다
```bash
hermes gateway status    # Is the gateway running?
hermes cron list         # Is the job enabled?
```

### 속도 제한
GitHub는 인증된 사용자에게 시간당 5,000개의 API 요청을 허용합니다. PR 리뷰 하나에는 약 3–5개의 요청(목록 조회 + diff + 선택적 댓글)이 사용됩니다. 하루에 PR 100개를 검토해도 제한을 충분히 넘지 않습니다.

---

## 다음 단계는 무엇인가요?

- **[웹훅 기반 PR 리뷰](./webhook-github-pr-review.md)** — PR이 열릴 때 즉시 리뷰 받기(공개 엔드포인트 필요)
- **[일일 브리핑 봇](/guides/daily-briefing-bot)** — PR 리뷰와 아침 뉴스 다이제스트 결합
- **[플러그인 만들기](/developer-guide/plugins)** — 리뷰 로직을 공유 가능한 플러그인으로 감싸기
- **[프로필](/user-guide/profiles)** — 자체 메모리와 설정을 가진 전용 리뷰어 프로필 실행
- **[대체 프로바이더](/user-guide/features/fallback-providers)** — 한 프로바이더가 중단되어도 리뷰가 실행되도록 보장
