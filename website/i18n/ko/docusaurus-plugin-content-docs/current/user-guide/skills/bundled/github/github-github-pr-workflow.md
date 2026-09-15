---
title: "GitHub PR 워크플로 — 브랜치, 커밋, 열기, CI, 병합을 포함한 GitHub PR 수명 주기"
sidebar_label: "GitHub PR 워크플로"
description: "브랜치, 커밋, 열기, CI, 병합을 포함한 GitHub PR 수명 주기"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# GitHub PR 워크플로

브랜치 생성부터 커밋, 열기, CI, 병합까지 GitHub PR 수명 주기를 다룹니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 포함 (기본 설치됨) |
| 경로 | `skills/github/github-pr-workflow` |
| 버전 | `1.1.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `GitHub`, `Pull-Requests`, `CI/CD`, `Git`, `Automation`, `Merge` |
| 관련 스킬 | [`github-auth`](/docs/user-guide/skills/bundled/github/github-github-auth), [`github-code-review`](/docs/user-guide/skills/bundled/github/github-github-code-review) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 실행될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트에 표시되는 내용입니다.
:::

# GitHub 풀 리퀘스트 워크플로

PR 수명 주기를 관리하는 완전한 안내서입니다. 각 섹션에서는 먼저 `gh` 방식, 이어서 `gh`가 없는 시스템을 위한 `git` + `curl` 대체 방식을 설명합니다.

## 사전 요구 사항

- GitHub 인증 완료 (`github-auth` 스킬 참고)
- GitHub 원격 저장소가 있는 git 저장소 내부

### 빠른 인증 감지

```bash
# Determine which method to use throughout this workflow
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  AUTH="gh"
else
  AUTH="git"
  # Ensure we have a token for API calls
  if [ -z "$GITHUB_TOKEN" ]; then
    if _hermes_env="${HERMES_HOME:-$HOME/.hermes}/.env"; [ -f "$_hermes_env" ] && grep -q "^GITHUB_TOKEN=" "$_hermes_env"; then
      GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" "$_hermes_env" | head -1 | cut -d= -f2 | tr -d '\n\r')
    elif grep -q "github.com" ~/.git-credentials 2>/dev/null; then
      GITHUB_TOKEN=$(uv run python3 "${HERMES_HOME:-$HOME/.hermes}/skills/github/github-auth/scripts/git-credential-token.py")
    fi
  fi
fi
echo "Using: $AUTH"
```

### Git 원격 저장소에서 소유자/저장소 추출

많은 `curl` 명령에는 `owner/repo`가 필요합니다. git 원격 저장소에서 이를 추출하세요.

```bash
# Works for both HTTPS and SSH remote URLs
REMOTE_URL=$(git remote get-url origin)
OWNER_REPO=$(echo "$REMOTE_URL" | sed -E 's|.*github\.com[:/]||; s|\.git$||')
OWNER=$(echo "$OWNER_REPO" | cut -d/ -f1)
REPO=$(echo "$OWNER_REPO" | cut -d/ -f2)
echo "Owner: $OWNER, Repo: $REPO"
```

---

## 1. 브랜치 생성

이 부분은 순수한 `git` 작업이며 어느 방식에서나 동일합니다.

```bash
# Make sure you're up to date
git fetch origin
git checkout main && git pull origin main

# Create and switch to a new branch
git checkout -b feat/add-user-authentication
```

브랜치 이름 규칙:
- `feat/description` — 새 기능
- `fix/description` — 버그 수정
- `refactor/description` — 코드 구조 변경
- `docs/description` — 문서
- `ci/description` — CI/CD 변경

## 2. 커밋 작성

에이전트의 파일 도구(`write_file`, `patch`)로 변경한 다음 커밋하세요.

```bash
# Stage specific files
git add src/auth.py src/models/user.py tests/test_auth.py

# Commit with a conventional commit message
git commit -m "feat: add JWT-based user authentication

- Add login/register endpoints
- Add User model with password hashing
- Add auth middleware for protected routes
- Add unit tests for auth flow"
```

커밋 메시지 형식(Conventional Commits):
```
type(scope): short description

Longer explanation if needed. Wrap at 72 characters.
```

유형: `feat`, `fix`, `refactor`, `docs`, `test`, `ci`, `chore`, `perf`

## 3. 푸시 및 PR 생성

### 브랜치 푸시 (어느 방식에서나 동일)

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

응답 JSON에는 PR `number`가 포함됩니다. 이후 명령에서 사용할 수 있도록 저장하세요.

초안으로 생성하려면 JSON 본문에 `"draft": true`를 추가하세요.

## 4. CI 상태 모니터링

### CI 상태 확인

**gh 사용:**

```bash
# One-shot check
gh pr checks

# Watch until all checks finish (polls every 10s)
gh pr checks --watch
```

**git + curl 사용:**

```bash
# Get the latest commit SHA on the current branch
SHA=$(git rev-parse HEAD)

# Query the combined status
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/commits/$SHA/status \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Overall: {data['state']}\")
for s in data.get('statuses', []):
    print(f\"  {s['context']}: {s['state']} - {s.get('description', '')}\")"

# Also check GitHub Actions check runs (separate endpoint)
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
# Simple polling loop — check every 30 seconds, up to 10 minutes
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

CI가 실패하면 원인을 진단하고 수정하세요. 이 루프는 어느 인증 방식에서도 작동합니다.

### 1단계: 실패 세부 정보 가져오기

**gh 사용:**

```bash
# List recent workflow runs on this branch
gh run list --branch $(git branch --show-current) --limit 5

# View failed logs
gh run view <RUN_ID> --log-failed
```

**git + curl 사용:**

```bash
BRANCH=$(git branch --show-current)

# List workflow runs on this branch
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/$OWNER/$REPO/actions/runs?branch=$BRANCH&per_page=5" \
  | python3 -c "
import sys, json
runs = json.load(sys.stdin)['workflow_runs']
for r in runs:
    print(f\"Run {r['id']}: {r['name']} - {r['conclusion'] or r['status']}\")"

# Get failed job logs (download as zip, extract, read)
RUN_ID=<run_id>
curl -s -L \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/actions/runs/$RUN_ID/logs \
  -o /tmp/ci-logs.zip
cd /tmp && unzip -o ci-logs.zip -d ci-logs && cat ci-logs/*.txt
```

### 2단계: 수정 및 푸시

문제를 파악한 후 파일 도구(`patch`, `write_file`)로 수정하세요.

```bash
git add <fixed_files>
git commit -m "fix: resolve CI failure in <check_name>"
git push
```

### 3단계: 확인

위 4절의 명령을 사용해 CI 상태를 다시 확인하세요.

### 자동 수정 루프 패턴

CI를 자동 수정하라는 요청을 받으면 다음 루프를 따르세요.

1. CI 상태 확인 → 실패 식별
2. 실패 로그 읽기 → 오류 이해
3. `read_file` + `patch`/`write_file` 사용 → 코드 수정
4. `git add . && git commit -m "fix: ..." && git push`
5. CI 대기 → 상태 재확인
6. 여전히 실패하면 반복(최대 3회, 이후 사용자에게 질문)

## 6. 병합

**gh 사용:**

```bash
# Squash merge + delete branch (cleanest for feature branches)
gh pr merge --squash --delete-branch

# Enable auto-merge (merges when all checks pass)
gh pr merge --auto --squash --delete-branch
```

**git + curl 사용:**

```bash
PR_NUMBER=<number>

# Merge the PR via API (squash)
curl -s -X PUT \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/merge \
  -d "{
    \"merge_method\": \"squash\",
    \"commit_title\": \"feat: add user authentication (#$PR_NUMBER)\"
  }"

# Delete the remote branch after merge
BRANCH=$(git branch --show-current)
git push origin --delete $BRANCH

# Switch back to main locally
git checkout main && git pull origin main
git branch -d $BRANCH
```

병합 방식: `"merge"`(병합 커밋), `"squash"`, `"rebase"`

### 자동 병합 활성화 (curl)

```bash
# Auto-merge requires the repo to have it enabled in settings.
# This uses the GraphQL API since REST doesn't support auto-merge.
PR_NODE_ID=$(curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['node_id'])")

curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/graphql \
  -d "{\"query\": \"mutation { enablePullRequestAutoMerge(input: {pullRequestId: \\\"$PR_NODE_ID\\\", mergeMethod: SQUASH}) { clientMutationId } }\"}"
```

## 7. 전체 워크플로 예시

```bash
# 1. Start from clean main
git checkout main && git pull origin main

# 2. Branch
git checkout -b fix/login-redirect-bug

# 3. (Agent makes code changes with file tools)

# 4. Commit
git add src/auth/login.py tests/test_login.py
git commit -m "fix: correct redirect URL after login

Preserves the ?next= parameter instead of always redirecting to /dashboard."

# 5. Push
git push -u origin HEAD

# 6. Create PR (picks gh or curl based on what's available)
# ... (see Section 3)

# 7. Monitor CI (see Section 4)

# 8. Merge when green (see Section 6)
```

## 유용한 PR 명령어 참고

| 작업 | gh | git + curl |
|--------|-----|-----------|
| 내 PR 목록 보기 | `gh pr list --author @me` | `curl -s -H "Authorization: token $GITHUB_TOKEN" "https://api.github.com/repos/$OWNER/$REPO/pulls?state=open"` |
| PR diff 보기 | `gh pr diff` | `git diff main...HEAD` (로컬) 또는 `curl -H "Accept: application/vnd.github.diff" ...` |
| 댓글 추가 | `gh pr comment N --body "..."` | `curl -X POST .../issues/N/comments -d '{"body":"..."}'` |
| 리뷰 요청 | `gh pr edit N --add-reviewer user` | `curl -X POST .../pulls/N/requested_reviewers -d '{"reviewers":["user"]}'` |
| PR 닫기 | `gh pr close N` | `curl -X PATCH .../pulls/N -d '{"state":"closed"}'` |
| 다른 사람의 PR 체크아웃 | `gh pr checkout N` | `git fetch origin pull/N/head:pr-N && git checkout pr-N` |
