---
title: "Github Pr Workflow — GitHub PR 수명 주기: 브랜치, 커밋, 오픈, CI, 병합"
sidebar_label: "Github Pr Workflow"
description: "GitHub PR 수명 주기: 브랜치, 커밋, 오픈, CI, 병합"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py 스크립트에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Github Pr Workflow

GitHub PR 수명 주기: 브랜치, 커밋, 오픈, CI, 병합.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/github/github-pr-workflow` |
| Version | `1.1.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `GitHub`, `Pull-Requests`, `CI/CD`, `Git`, `Automation`, `Merge` |
| Related skills | [`github-auth`](/docs/user-guide/skills/bundled/github/github-github-auth), [`github-code-review`](/docs/user-guide/skills/bundled/github/github-github-code-review) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 것입니다.
:::

# GitHub Pull Request 워크플로우

PR 라이프사이클 관리를 위한 완벽한 가이드입니다. 각 섹션은 `gh` 방식을 먼저 보여준 다음, `gh`가 없는 시스템을 위한 `git` + `curl` 폴백을 보여줍니다.

## 전제 조건

- GitHub에 인증됨 (`github-auth` 스킬 참조)
- GitHub 리모트가 있는 git 저장소 내부

### 빠른 인증 감지

```bash
# 이 워크플로우 전반에 걸쳐 사용할 방법 결정
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  AUTH="gh"
else
  AUTH="git"
  # API 호출을 위한 토큰이 있는지 확인
  if [ -z "$GITHUB_TOKEN" ]; then
    if [ -f ~/.hermes/.env ] && grep -q "^GITHUB_TOKEN=" ~/.hermes/.env; then
      GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" ~/.hermes/.env | head -1 | cut -d= -f2 | tr -d '\n\r')
    elif grep -q "github.com" ~/.git-credentials 2>/dev/null; then
      GITHUB_TOKEN=$(grep "github.com" ~/.git-credentials 2>/dev/null | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|')
    fi
  fi
fi
echo "Using: $AUTH"
```

### Git 원격에서 Owner/Repo 추출

많은 `curl` 명령어에는 `owner/repo`가 필요합니다. git 원격에서 이를 추출합니다:

```bash
# HTTPS 및 SSH 원격 URL 모두에서 작동
REMOTE_URL=$(git remote get-url origin)
OWNER_REPO=$(echo "$REMOTE_URL" | sed -E 's|.*github\.com[:/]||; s|\.git$||')
OWNER=$(echo "$OWNER_REPO" | cut -d/ -f1)
REPO=$(echo "$OWNER_REPO" | cut -d/ -f2)
echo "Owner: $OWNER, Repo: $REPO"
```

---

## 1. 브랜치 생성

이 부분은 순수한 `git`입니다 — 어느 쪽이든 동일합니다:

```bash
# 최신 상태인지 확인
git fetch origin
git checkout main && git pull origin main

# 새로운 브랜치를 만들고 전환
git checkout -b feat/add-user-authentication
```

브랜치 명명 규칙:
- `feat/description` — 새로운 기능
- `fix/description` — 버그 수정
- `refactor/description` — 코드 구조 재편
- `docs/description` — 문서화
- `ci/description` — CI/CD 변경사항

## 2. 커밋하기

에이전트의 파일 도구(`write_file`, `patch`)를 사용하여 변경 사항을 만들고 커밋하세요:

```bash
# 특정 파일 스테이징
git add src/auth.py src/models/user.py tests/test_auth.py

# Conventional Commit 메시지로 커밋
git commit -m "feat: add JWT-based user authentication

- Add login/register endpoints
- Add User model with password hashing
- Add auth middleware for protected routes
- Add unit tests for auth flow"
```

커밋 메시지 형식 (Conventional Commits):
```
type(scope): short description

Longer explanation if needed. Wrap at 72 characters.
```

타입: `feat`, `fix`, `refactor`, `docs`, `test`, `ci`, `chore`, `perf`

## 3. 브랜치 푸시 및 PR 생성

### 브랜치 푸시 (어느 쪽이든 동일)

```bash
git push -u origin HEAD
```

### PR 생성

**gh 사용:**

```bash
gh pr create \
  --title "feat: add JWT-based user authentication" \
  --body "## Summary
- Adds login and register API endpoints
- JWT token generation and validation

## Test Plan
- [ ] Unit tests pass

Closes #42"
```

옵션: `--draft`, `--reviewer user1,user2`, `--label "enhancement"`, `--base develop`

**git + curl 사용:**

```bash
BRANCH=$(git branch --show-current)

curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/$OWNER/$REPO/pulls \
  -d "{
    \"title\": \"feat: add JWT-based user authentication\",
    \"body\": \"## Summary\nAdds login and register API endpoints.\n\nCloses #42\",
    \"head\": \"$BRANCH\",
    \"base\": \"main\"
  }"
```

응답 JSON에는 PR `number`가 포함됩니다 — 나중 명령어를 위해 이를 저장해두세요.

초안(draft)으로 만들려면 JSON 본문에 `"draft": true`를 추가하세요.

## 4. CI 상태 모니터링

### CI 상태 확인

**gh 사용:**

```bash
# 1회성 확인
gh pr checks

# 모든 검사가 완료될 때까지 지켜보기 (10초마다 폴링)
gh pr checks --watch
```

**git + curl 사용:**

```bash
# 현재 브랜치의 최신 커밋 SHA 가져오기
SHA=$(git rev-parse HEAD)

# 통합된 상태 쿼리
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/commits/$SHA/status \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Overall: {data['state']}\")
for s in data.get('statuses', []):
    print(f\"  {s['context']}: {s['state']} - {s.get('description', '')}\")"

# GitHub Actions 검사 실행도 확인 (별도 엔드포인트)
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/commits/$SHA/check-runs \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for cr in data.get('check_runs', []):
    print(f\"  {cr['name']}: {cr['status']} / {cr['conclusion'] or 'pending'}\")"
```

### 완료될 때까지 폴링 (git + curl)

```bash
# 간단한 폴링 루프 — 30초마다 최대 10분 동안 확인
SHA=$(git rev-parse HEAD)
for i in $(seq 1 20); do
  STATUS=$(curl -s \
    -H "Authorization: token $GITHUB_TOKEN" \
    https://api.github.com/repos/$OWNER/$REPO/commits/$SHA/status \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['state'])")
  echo "Check $i: $STATUS"
  if [ "$STATUS" = "success" ] || [ "$STATUS" = "failure" ] || [ "$STATUS" = "error" ]; then
    break
  fi
  sleep 30
done
```

## 5. CI 실패 자동 수정

CI가 실패하면 진단하고 수정하세요. 이 루프는 두 인증 방법 모두에서 작동합니다.

### 1단계: 실패 세부 정보 가져오기

**gh 사용:**

```bash
# 이 브랜치에서 최근 워크플로우 실행 목록
gh run list --branch $(git branch --show-current) --limit 5

# 실패한 로그 보기
gh run view <RUN_ID> --log-failed
```

**git + curl 사용:**

```bash
BRANCH=$(git branch --show-current)

# 이 브랜치의 워크플로우 실행 목록
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/$OWNER/$REPO/actions/runs?branch=$BRANCH&per_page=5" \
  | python3 -c "
import sys, json
runs = json.load(sys.stdin)['workflow_runs']
for r in runs:
    print(f\"Run {r['id']}: {r['name']} - {r['conclusion'] or r['status']}\")"

# 실패한 작업 로그 가져오기 (zip 다운로드, 압축 풀기, 읽기)
RUN_ID=<run_id>
curl -s -L \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/actions/runs/$RUN_ID/logs \
  -o /tmp/ci-logs.zip
cd /tmp && unzip -o ci-logs.zip -d ci-logs && cat ci-logs/*.txt
```

### 2단계: 수정 및 푸시

문제를 파악한 후, 파일 도구(`patch`, `write_file`)를 사용하여 코드를 수정하세요:

```bash
git add <fixed_files>
git commit -m "fix: resolve CI failure in <check_name>"
git push
```

### 3단계: 확인

위의 섹션 4 명령어를 사용하여 CI 상태를 다시 확인하세요.

### 자동 수정 루프 패턴

CI를 자동 수정하라는 요청을 받으면 이 루프를 따르세요:

1. CI 상태 확인 → 실패 파악
2. 실패 로그 읽기 → 오류 이해
3. `read_file` + `patch`/`write_file` 사용 → 코드 수정
4. `git add . && git commit -m "fix: ..." && git push`
5. CI 대기 → 상태 다시 확인
6. 계속 실패하면 반복 (최대 3회 시도 후 사용자에게 질문)

## 6. 병합

**gh 사용:**

```bash
# Squash merge + 브랜치 삭제 (기능 브랜치에 가장 깔끔함)
gh pr merge --squash --delete-branch

# 자동 병합 활성화 (모든 검사가 통과할 때 병합됨)
gh pr merge --auto --squash --delete-branch
```

**git + curl 사용:**

```bash
PR_NUMBER=<number>

# API를 통해 PR 병합 (squash)
curl -s -X PUT \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/merge \
  -d "{
    \"merge_method\": \"squash\",
    \"commit_title\": \"feat: add user authentication (#$PR_NUMBER)\"
  }"

# 병합 후 원격 브랜치 삭제
BRANCH=$(git branch --show-current)
git push origin --delete $BRANCH

# 로컬에서 main으로 다시 전환
git checkout main && git pull origin main
git branch -d $BRANCH
```

병합 방식(Merge methods): `"merge"` (merge commit), `"squash"`, `"rebase"`

### 자동 병합 활성화 (curl)

```bash
# 자동 병합을 하려면 레포지토리 설정에서 이 기능이 활성화되어 있어야 합니다.
# REST는 자동 병합을 지원하지 않으므로 GraphQL API를 사용합니다.
PR_NODE_ID=$(curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['node_id'])")

curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/graphql \
  -d "{\"query\": \"mutation { enablePullRequestAutoMerge(input: {pullRequestId: \\\"$PR_NODE_ID\\\", mergeMethod: SQUASH}) { clientMutationId } }\"}"
```

## 7. 전체 워크플로우 예시

```bash
# 1. 깔끔한 main에서 시작
git checkout main && git pull origin main

# 2. 브랜치 생성
git checkout -b fix/login-redirect-bug

# 3. (에이전트가 파일 도구를 사용하여 코드를 수정)

# 4. 커밋
git add src/auth/login.py tests/test_login.py
git commit -m "fix: correct redirect URL after login

Preserves the ?next= parameter instead of always redirecting to /dashboard."

# 5. 푸시
git push -u origin HEAD

# 6. PR 생성 (사용 가능한 것에 따라 gh 또는 curl 선택)
# ... (섹션 3 참조)

# 7. CI 모니터링 (섹션 4 참조)

# 8. 녹색(성공)이면 병합 (섹션 6 참조)
```

## 유용한 PR 명령어 참조

| 작업 (Action) | gh | git + curl |
|--------|-----|-----------|
| 내 PR 나열 | `gh pr list --author @me` | `curl -s -H "Authorization: token $GITHUB_TOKEN" "https://api.github.com/repos/$OWNER/$REPO/pulls?state=open"` |
| PR diff 보기 | `gh pr diff` | `git diff main...HEAD` (local) or `curl -H "Accept: application/vnd.github.diff" ...` |
| 댓글 추가 | `gh pr comment N --body "..."` | `curl -X POST .../issues/N/comments -d '{"body":"..."}'` |
| 리뷰 요청 | `gh pr edit N --add-reviewer user` | `curl -X POST .../pulls/N/requested_reviewers -d '{"reviewers":["user"]}'` |
| PR 닫기 | `gh pr close N` | `curl -X PATCH .../pulls/N -d '{"state":"closed"}'` |
| 다른 사람의 PR 체크아웃 | `gh pr checkout N` | `git fetch origin pull/N/head:pr-N && git checkout pr-N` |
