---
sidebar_position: 10
title: "튜토리얼: GitHub PR 리뷰 에이전트"
description: "저장소를 모니터링하고, 풀 리퀘스트를 검토하고, 피드백을 전달하는 자동 AI 코드 리뷰어 구축하기 — 전자동(hands-free)"
---

# 튜토리얼: GitHub PR 리뷰 에이전트 구축하기

**문제:** 팀에서 리뷰 속도보다 빠르게 PR을 올립니다. PR은 누군가의 확인을 기다리며 며칠씩 쌓여 있습니다. 주니어 개발자들은 버그를 발견할 시간이 없어서 그대로 병합합니다. 코드를 작성하는 대신 아침 시간을 차이(diff)를 확인하는 데 다 써버리게 됩니다.

**해결책:** 리포지토리를 24시간 모니터링하고, 모든 새로운 PR의 버그, 보안 문제 및 코드 품질을 검토하여 요약을 보내주는 AI 에이전트를 도입하세요 — 여러분은 사람의 판단이 실제로 필요한 PR에만 시간을 할애할 수 있습니다.

**구축할 내용:**

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   크론 타이머 ──▶ Hermes Agent  ──▶  GitHub API  ──▶    리뷰      │
│   (2시간마다)     + gh CLI           (PR diffs)         전달      │
│                   + 스킬                               (Telegram, │
│                   + 메모리                             Discord,   │
│                                                        로컬)      │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

이 가이드는 **크론(cron) 작업**을 사용하여 일정에 따라 PR을 확인(polling)합니다 — 서버나 공개(public) 엔드포인트가 필요하지 않습니다. NAT 및 방화벽 환경 내에서도 작동합니다.

:::tip 실시간 리뷰를 원하시나요?
공개 엔드포인트를 사용할 수 있다면 [웹훅을 사용한 자동 GitHub PR 코멘트](./webhook-github-pr-review.md)를 확인하세요 — PR이 열리거나 업데이트될 때마다 GitHub에서 즉시 Hermes로 이벤트를 푸시합니다.
:::

---

## 사전 요구 사항

- **Hermes Agent 설치됨** — [설치 가이드](/getting-started/installation) 참조
- 크론 작업을 위한 **게이트웨이 실행**:
  ```bash
  hermes gateway install   # 서비스로 설치
  # 또는
  hermes gateway           # 포그라운드에서 실행
  ```
- **GitHub CLI (`gh`) 설치 및 인증됨**:
  ```bash
  # 설치
  brew install gh        # macOS
  sudo apt install gh    # Ubuntu/Debian

  # 인증
  gh auth login
  ```
- **메시징 구성됨** (선택 사항) — [Telegram](/user-guide/messaging/telegram) 또는 [Discord](/user-guide/messaging/discord)

:::tip 메시징 앱이 없나요? 문제없습니다.
`deliver: "local"`을 사용하여 리뷰를 `~/.hermes/cron/output/`에 저장할 수 있습니다. 알림을 연동하기 전 테스트용으로 아주 좋습니다.
:::

---

## 1단계: 설정 확인

Hermes가 GitHub에 액세스할 수 있는지 확인합니다. 대화를 시작하세요:

```bash
hermes
```

간단한 명령어로 테스트합니다:

```
Run: gh pr list --repo NousResearch/hermes-agent --state open --limit 3
```

열린 PR 목록이 표시되어야 합니다. 이것이 작동하면 준비가 된 것입니다.

---

## 2단계: 수동 리뷰 시도하기

채팅창에서 Hermes에게 실제 PR을 리뷰해 달라고 요청해 보세요:

```
이 풀 리퀘스트를 리뷰해 줘. diff를 읽고 버그, 보안 문제, 코드 품질을 확인해 줘.
라인 번호를 구체적으로 명시하고 문제가 되는 코드를 인용해.

실행: gh pr diff 3888 --repo NousResearch/hermes-agent
```

Hermes는 다음을 수행합니다:
1. `gh pr diff`를 실행하여 코드 변경 사항을 가져옵니다.
2. 전체 diff를 꼼꼼히 읽습니다.
3. 구체적인 결과가 포함된 구조화된 리뷰를 생성합니다.

품질이 만족스럽다면, 이제 자동화할 시간입니다.

---

## 3단계: 리뷰 스킬 생성하기

스킬(skill)은 여러 세션 및 크론 실행 과정에서도 유지되는 일관된 리뷰 가이드라인을 제공합니다. 스킬이 없으면 리뷰 품질이 일정하지 않을 수 있습니다.

```bash
mkdir -p ~/.hermes/skills/code-review
```

`~/.hermes/skills/code-review/SKILL.md` 생성:

```markdown
---
name: code-review
description: 버그, 보안 문제, 코드 품질에 대한 풀 리퀘스트 리뷰
---

# Code Review Guidelines

풀 리퀘스트를 리뷰할 때:

## 무엇을 확인해야 하는가
1. **버그** — 로직 오류, 오프 바이 원(off-by-one), null/undefined 처리
2. **보안** — 인젝션, 인증 우회, 코드 내 시크릿 노출, SSRF
3. **성능** — N+1 쿼리, 무한 루프, 메모리 누수
4. **스타일** — 네이밍 컨벤션, 죽은 코드, 오류 처리 누락
5. **테스트** — 변경 사항이 테스트되었는가? 엣지 케이스를 다루는가?

## 출력 형식
각 문제(finding)에 대해:
- **File:Line** — 정확한 위치
- **Severity(심각도)** — Critical / Warning / Suggestion
- **What's wrong(문제점)** — 한 문장
- **Fix(수정 방법)** — 어떻게 수정할 것인가

## 규칙
- 구체적으로 작성하세요. 문제가 되는 코드를 인용하세요.
- 가독성에 영향을 주지 않는 사소한 스타일 지적은 하지 마세요.
- PR에 문제가 없어 보이면 그렇다고 말하세요. 문제를 지어내지 마세요.
- 마지막에는 다음 중 하나로 끝맺으세요: APPROVE / REQUEST_CHANGES / COMMENT
```

로드가 되었는지 확인 — `hermes`를 시작하면 시작 시 스킬 목록에 `code-review`가 보여야 합니다.

---

## 4단계: 당신의 컨벤션 가르치기

이것이 리뷰어를 실제로 유용하게 만드는 부분입니다. 세션을 시작하고 Hermes에게 팀의 표준을 가르치세요:

```
기억해 줘: 우리 백엔드 저장소는 Python과 FastAPI를 사용해.
모든 엔드포인트에는 타입 어노테이션과 Pydantic 모델이 있어야 해.
원시(raw) SQL은 허용하지 않고, 오직 SQLAlchemy ORM만 사용해.
테스트 파일은 tests/에 있어야 하고 pytest fixtures를 사용해야 해.
```

```
기억해 줘: 우리 프론트엔드 저장소는 TypeScript와 React를 사용해.
`any` 타입은 허용되지 않아. 모든 컴포넌트에는 props 인터페이스가 있어야 해.
데이터 패칭에는 React Query를 사용하고, API 호출에 useEffect를 절대 사용하지 않아.
```

이 메모리들은 영구적으로 지속됩니다 — 리뷰어는 매번 지시받지 않아도 여러분의 컨벤션을 강제할 것입니다.

---

## 5단계: 자동화된 크론 작업 생성하기

이제 모두 연결해 보겠습니다. 2시간마다 실행되는 크론 작업을 생성하세요:

```bash
hermes cron create "0 */2 * * *" \
  "열려 있는 새 PR을 확인하고 검토해 줘.

모니터링할 저장소:
- myorg/backend-api
- myorg/frontend-app

단계:
1. 실행: gh pr list --repo REPO --state open --limit 5 --json number,title,author,createdAt
2. 지난 4시간 내에 생성되거나 업데이트된 각 PR에 대해:
   - 실행: gh pr diff NUMBER --repo REPO
   - code-review 가이드라인을 사용하여 diff 검토
3. 다음과 같이 출력 형식을 지정해 줘:

## PR Reviews — today

### [repo] #[number]: [title]
**Author:** [name] | **Verdict:** APPROVE/REQUEST_CHANGES/COMMENT
[findings]

새로운 PR이 발견되지 않으면, '리뷰할 새로운 PR 없음'이라고 말해 줘." \
  --name "pr-review" \
  --deliver telegram \
  --skill code-review
```

예약되었는지 확인합니다:

```bash
hermes cron list
```

### 기타 유용한 일정 설정

| 일정 | 언제 |
|----------|------|
| `0 */2 * * *` | 2시간마다 |
| `0 9,13,17 * * 1-5` | 평일에만 하루 3번 |
| `0 9 * * 1` | 매주 월요일 아침 요약 |
| `30m` | 30분마다 (트래픽이 많은 저장소) |

---

## 6단계: 온디맨드로 실행하기

일정을 기다리고 싶지 않다면 수동으로 실행하세요:

```bash
hermes cron run pr-review
```

또는 채팅 세션 내에서:

```
/cron run pr-review
```

---

## 더 나아가기

### GitHub에 직접 리뷰 게시하기

Telegram으로 알림을 받는 대신, 에이전트가 PR 자체에 코멘트를 남기도록 합니다:

크론 프롬프트에 다음을 추가하세요:

```
리뷰한 후, 리뷰 내용을 게시해:
- 문제 발생 시: gh pr review NUMBER --repo REPO --comment --body "YOUR_REVIEW"
- 심각한 문제 시: gh pr review NUMBER --repo REPO --request-changes --body "YOUR_REVIEW"
- 깔끔한 PR인 경우: gh pr review NUMBER --repo REPO --approve --body "Looks good"
```

:::caution
`gh`가 `repo` 스코프를 가진 토큰을 갖고 있는지 확인하세요. 리뷰는 `gh`로 인증된 계정의 이름으로 게시됩니다.
:::

### 주간 PR 대시보드

모든 리포지토리에 대한 월요일 아침 개요(overview)를 생성합니다:

```bash
hermes cron create "0 9 * * 1" \
  "주간 PR 대시보드를 생성해 줘:
- myorg/backend-api
- myorg/frontend-app
- myorg/infra

각 저장소에 대해 다음을 보여줘:
1. 열린 PR 수 및 가장 오래된 PR의 생성 시점
2. 이번 주에 병합된 PR
3. 오래된 PR (5일 이상)
4. 리뷰어가 할당되지 않은 PR

깔끔하게 요약 형식으로 작성해." \
  --name "weekly-dashboard" \
  --deliver telegram
```

### 다중 저장소 모니터링

프롬프트에 더 많은 저장소를 추가하여 확장하세요. 에이전트는 이를 순차적으로 처리합니다 — 추가 설정이 필요하지 않습니다.

---

## 문제 해결

### "gh: command not found"
게이트웨이는 최소 환경에서 실행됩니다. `gh`가 시스템 PATH에 있는지 확인하고 게이트웨이를 재시작하세요.

### 리뷰 내용이 너무 일반적임
1. `code-review` 스킬 추가 (3단계)
2. 메모리를 통해 Hermes에게 여러분의 코딩 컨벤션을 가르치세요 (4단계)
3. 스택에 대한 컨텍스트가 많을수록 리뷰 품질이 향상됩니다.

### 크론 작업이 실행되지 않음
```bash
hermes gateway status    # 게이트웨이가 실행 중인가요?
hermes cron list         # 작업이 활성화되어 있나요?
```

### API 속도 제한
GitHub는 인증된 사용자에게 시간당 5,000건의 API 요청을 허용합니다. 각 PR 리뷰는 약 3-5개의 요청을 사용합니다 (list + diff + 선택적 comments). 하루에 100개의 PR을 리뷰하더라도 한도 내에 충분히 머물 수 있습니다.

---

## 다음 단계

- **[웹훅 기반 PR 리뷰](./webhook-github-pr-review.md)** — PR이 열릴 때 즉각적인 리뷰 받기 (공개 엔드포인트 필요)
- **[일일 브리핑 봇](/guides/daily-briefing-bot)** — 아침 뉴스 요약에 PR 리뷰를 결합
- **[플러그인 구축하기](/guides/build-a-hermes-plugin)** — 리뷰 로직을 공유 가능한 플러그인으로 패키징
- **[프로필](/user-guide/profiles)** — 자체 메모리와 구성을 가진 전용 리뷰어 프로필 실행
- **[대체 제공자(Fallback Providers)](/user-guide/features/fallback-providers)** — 특정 제공자가 다운되더라도 리뷰가 확실히 실행되도록 보장
