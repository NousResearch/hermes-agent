---
title: "GitHub 코드 리뷰 — gh 또는 REST를 통한 PR 검토: diff, 인라인 댓글"
sidebar_label: "GitHub 코드 리뷰"
description: "PR 검토: gh 또는 REST를 통한 diff와 인라인 댓글"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# GitHub 코드 리뷰

PR을 검토합니다: diff와 인라인 댓글을 gh 또는 REST를 통해 처리합니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 번들됨 (기본으로 설치됨) |
| 경로 | `skills/github/github-code-review` |
| 버전 | `1.1.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `GitHub`, `코드 리뷰`, `풀 리퀘스트`, `Git`, `품질` |
| 관련 스킬 | [`GitHub 인증`](/docs/user-guide/skills/bundled/github/github-github-auth), [`GitHub PR 워크플로`](/docs/user-guide/skills/bundled/github/github-github-pr-workflow) |

## 참조: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# GitHub 코드 리뷰

푸시하기 전에 로컬 변경 사항을 코드 리뷰하거나 GitHub에서 열린 PR을 검토합니다. 이 스킬의 대부분은 일반 `git`을 사용하며, `gh`/`curl` 구분은 PR 수준의 상호작용에서만 중요합니다.

## 사전 요구 사항

- GitHub 인증 완료 (`github-auth` 스킬 참고)
- Git 저장소 내부

### 설정 (PR 상호작용용)

```bash
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  AUTH="gh"
else
  AUTH="git"
  if [ -z "$GITHUB_TOKEN" ]; then
    if _hermes_env="${HERMES_HOME:-$HOME/.hermes}/.env"; [ -f "$_hermes_env" ] && grep -q "^GITHUB_TOKEN=" "$_hermes_env"; then
      GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" "$_hermes_env" | head -1 | cut -d= -f2 | tr -d '\n\r')
    elif grep -q "github.com" ~/.git-credentials 2>/dev/null; then
      GITHUB_TOKEN=$(uv run python3 "${HERMES_HOME:-$HOME/.hermes}/skills/github/github-auth/scripts/git-credential-token.py")
    fi
  fi
fi

REMOTE_URL=$(git remote get-url origin)
OWNER_REPO=$(echo "$REMOTE_URL" | sed -E 's|.*github\.com[:/]||; s|\.git$||')
OWNER=$(echo "$OWNER_REPO" | cut -d/ -f1)
REPO=$(echo "$OWNER_REPO" | cut -d/ -f2)
```

---

## 1. 로컬 변경 사항 검토 (푸시 전)

순수한 `git` 작업이므로 어디서나 사용할 수 있으며 API가 필요하지 않습니다.

### Diff 가져오기

```bash
# Staged changes (what would be committed)
git diff --staged

# All changes vs main (what a PR would contain)
git diff main...HEAD

# File names only
git diff main...HEAD --name-only

# Stat summary (insertions/deletions per file)
git diff main...HEAD --stat
```

### 검토 전략

1. **먼저 전체적인 맥락을 파악하세요:**

```bash
git diff main...HEAD --stat
git log main..HEAD --oneline
```

2. **파일별로 검토합니다** — 변경된 파일의 전체 맥락을 보려면 `read_file`을 사용하고, 무엇이 바뀌었는지 확인하려면 diff를 사용합니다.

```bash
git diff main...HEAD -- src/auth/login.py
```

3. **일반적인 문제를 확인합니다:**

```bash
# Debug statements, TODOs, console.logs left behind
git diff main...HEAD | grep -n "print(\|console\.log\|TODO\|FIXME\|HACK\|XXX\|debugger"

# Large files accidentally staged
git diff main...HEAD --stat | sort -t'|' -k2 -rn | head -10

# Secrets or credential patterns
git diff main...HEAD | grep -in "password\|secret\|api_key\|token.*=\|private_key"

# Merge conflict markers
git diff main...HEAD | grep -n "<<<<<<\|>>>>>>\|======="
```

4. **구조화된 피드백을 사용자에게 제시합니다.**

### 검토 결과 형식

```
## Code Review Summary

### Critical
- **src/auth.py:45** — SQL injection: user input passed directly to query.
  Suggestion: Use parameterized queries.

### Warnings
- **src/models/user.py:23** — Password stored in plaintext. Use bcrypt or argon2.
- **src/api/routes.py:112** — No rate limiting on login endpoint.

### Suggestions
- **src/utils/helpers.py:8** — Duplicates logic in `src/core/utils.py:34`. Consolidate.
- **tests/test_auth.py** — Missing edge case: expired token test.

### Looks Good
- Clean separation of concerns in the middleware layer
- Good test coverage for the happy path
```

## 2. GitHub에서 풀 리퀘스트 검토

### PR 세부 정보 보기

**gh 사용:**

```bash
gh pr view 123
gh pr diff 123
gh pr diff 123 --name-only
```

**git + curl 사용:**

```bash
PR_NUMBER=123

# Get PR details
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
  | python3 -c "
import sys, json
pr = json.load(sys.stdin)
print(f\"Title: {pr['title']}\")
print(f\"Author: {pr['user']['login']}\")
print(f\"Branch: {pr['head']['ref']} -> {pr['base']['ref']}\")
print(f\"State: {pr['state']}\")
print(f\"Body:\n{pr['body']}\")"

# List changed files
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/files \
  | python3 -c "
import sys, json
for f in json.load(sys.stdin):
    print(f\"{f['status']:10} +{f['additions']:-4} -{f['deletions']:-4}  {f['filename']}\")"
```

### 전체 검토를 위해 PR을 로컬에서 체크아웃

일반 `git`으로 처리할 수 있으므로 `gh`가 필요하지 않습니다.

```bash
# Fetch the PR branch and check it out
git fetch origin pull/123/head:pr-123
git checkout pr-123

# Now you can use read_file, search_files, run tests, etc.

# View diff against the base branch
git diff main...pr-123
```

**gh 사용 (단축 명령):**

```bash
gh pr checkout 123
```

### PR에 댓글 남기기

**일반 PR 댓글 — gh 사용:**

```bash
gh pr comment 123 --body "Overall looks good, a few suggestions below."
```

**일반 PR 댓글 — curl 사용:**

```bash
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/issues/$PR_NUMBER/comments \
  -d '{"body": "Overall looks good, a few suggestions below."}'
```

### 인라인 리뷰 댓글 남기기

**단일 인라인 댓글 — gh 사용 (API 경유):**

```bash
HEAD_SHA=$(gh pr view 123 --json headRefOid --jq '.headRefOid')

gh api repos/$OWNER/$REPO/pulls/123/comments \
  --method POST \
  -f body="This could be simplified with a list comprehension." \
  -f path="src/auth/login.py" \
  -f commit_id="$HEAD_SHA" \
  -f line=45 \
  -f side="RIGHT"
```

**단일 인라인 댓글 — curl 사용:**

```bash
# Get the head commit SHA
HEAD_SHA=$(curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['head']['sha'])")

curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/comments \
  -d "{
    \"body\": \"This could be simplified with a list comprehension.\",
    \"path\": \"src/auth/login.py\",
    \"commit_id\": \"$HEAD_SHA\",
    \"line\": 45,
    \"side\": \"RIGHT\"
  }"
```

### 공식 리뷰 제출 (승인 / 변경 요청)

**gh 사용:**

```bash
gh pr review 123 --approve --body "LGTM!"
gh pr review 123 --request-changes --body "See inline comments."
gh pr review 123 --comment --body "Some suggestions, nothing blocking."
```

**curl 사용 — 여러 댓글이 포함된 리뷰를 원자적으로 제출:**

```bash
HEAD_SHA=$(curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['head']['sha'])")

curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/reviews \
  -d "{
    \"commit_id\": \"$HEAD_SHA\",
    \"event\": \"COMMENT\",
    \"body\": \"Code review from Hermes Agent\",
    \"comments\": [
      {\"path\": \"src/auth.py\", \"line\": 45, \"body\": \"Use parameterized queries to prevent SQL injection.\"},
      {\"path\": \"src/models/user.py\", \"line\": 23, \"body\": \"Hash passwords with bcrypt before storing.\"},
      {\"path\": \"tests/test_auth.py\", \"line\": 1, \"body\": \"Add test for expired token edge case.\"}
    ]
  }"
```

이벤트 값: `"APPROVE"`, `"REQUEST_CHANGES"`, `"COMMENT"`

`line` 필드는 파일의 *새 버전*에 있는 줄 번호를 가리킵니다. 삭제된 줄에는 `"side": "LEFT"`를 사용하세요.

---

## 3. 검토 체크리스트

코드 리뷰(로컬 또는 PR)를 수행할 때 다음을 체계적으로 확인합니다.

### 정확성
- 코드가 주장하는 대로 동작합니까?
- 예외 상황(빈 입력, null, 대용량 데이터, 동시 접근)을 처리합니까?
- 오류 경로를 정상적으로 처리합니까?

### 보안
- 하드코딩된 비밀 값, 자격 증명 또는 API 키가 없습니까?
- 사용자에게 노출되는 입력을 검증합니까?
- SQL 인젝션, XSS 또는 경로 탐색이 없습니까?
- 필요한 곳에 인증/권한 부여 검사가 있습니까?

### 코드 품질
- 변수, 함수, 클래스 이름이 명확합니까?
- 불필요한 복잡성이나 성급한 추상화가 없습니까?
- DRY — 추출해야 하는 중복 로직이 없습니까?
- 함수가 단일 책임을 가지며 집중되어 있습니까?

### 테스트
- 새 코드 경로를 테스트합니까?
- 성공 경로와 오류 상황을 모두 다룹니까?
- 테스트가 읽기 쉽고 유지 관리 가능합니까?

### 성능
- N+1 쿼리나 불필요한 루프가 없습니까?
- 유용한 경우 적절한 캐싱을 적용했습니까?
- 비동기 코드 경로에서 블로킹 작업을 수행하지 않습니까?

### 문서화
- 공개 API를 문서화했습니까?
- 이해하기 어려운 로직에 "왜"를 설명하는 주석이 있습니까?
- 동작이 변경되었다면 README를 업데이트했습니까?

---

## 4. 푸시 전 검토 워크플로

사용자가 "코드를 검토해 주세요" 또는 "푸시하기 전에 확인해 주세요"라고 요청하면 다음을 수행합니다.

1. `git diff main...HEAD --stat` — 변경 범위를 확인합니다.
2. `git diff main...HEAD` — 전체 diff를 읽습니다.
3. 변경된 각 파일에 대해 더 많은 맥락이 필요하면 `read_file`을 사용합니다.
4. 위 체크리스트를 적용합니다.
5. 구조화된 형식(치명적 문제 / 경고 / 제안 / 문제 없음)으로 결과를 제시합니다.
6. 치명적인 문제가 발견되면 사용자가 푸시하기 전에 수정할 것을 제안합니다.

## 5. PR 검토 워크플로 (처음부터 끝까지)

사용자가 "PR #N을 검토해 주세요", "이 PR을 살펴봐 주세요"라고 요청하거나 PR URL을 제공하면 다음 절차를 따릅니다.

### 1단계: 환경 설정

```bash
source "${HERMES_HOME:-$HOME/.hermes}/skills/github/github-auth/scripts/gh-env.sh"
# Or run the inline setup block from the top of this skill
```

### 2단계: PR 맥락 수집

범위를 이해하기 전에 PR 메타데이터, 설명 및 변경된 파일 목록을 가져옵니다.

**gh 사용:**
```bash
gh pr view 123
gh pr diff 123 --name-only
gh pr checks 123
```

**curl 사용:**
```bash
PR_NUMBER=123

# PR details (title, author, description, branch)
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$GH_OWNER/$GH_REPO/pulls/$PR_NUMBER

# Changed files with line counts
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$GH_OWNER/$GH_REPO/pulls/$PR_NUMBER/files
```

### 3단계: PR을 로컬에서 체크아웃

이를 통해 `read_file`, `search_files` 및 테스트를 실행할 수 있습니다.

```bash
git fetch origin pull/$PR_NUMBER/head:pr-$PR_NUMBER
git checkout pr-$PR_NUMBER
```

### 4단계: diff를 읽고 변경 사항 이해

```bash
# Full diff against the base branch
git diff main...HEAD

# Or file-by-file for large PRs
git diff main...HEAD --name-only
# Then for each file:
git diff main...HEAD -- path/to/file.py
```

각 변경 파일에 대해 `read_file`을 사용하여 변경 부분 주변의 전체 맥락을 확인합니다. diff만으로는 주변 코드에서만 드러나는 문제를 놓칠 수 있습니다.

### 5단계: 자동화된 검사를 로컬에서 실행 (해당하는 경우)

```bash
# Run tests if there's a test suite
python -m pytest 2>&1 | tail -20
# or: npm test, cargo test, go test ./..., etc.

# Run linter if configured
ruff check . 2>&1 | head -30
# or: eslint, clippy, etc.
```

### 6단계: 검토 체크리스트 적용 (3절)

정확성, 보안, 코드 품질, 테스트, 성능, 문서화의 각 범주를 확인합니다.

### 7단계: GitHub에 검토 게시

결과를 취합하고 인라인 댓글이 포함된 공식 리뷰를 제출합니다.

**gh 사용:**
```bash
# If no issues — approve
gh pr review $PR_NUMBER --approve --body "Reviewed by Hermes Agent. Code looks clean — good test coverage, no security concerns."

# If issues found — request changes with inline comments
gh pr review $PR_NUMBER --request-changes --body "Found a few issues — see inline comments."
```

**curl 사용 — 여러 인라인 댓글을 포함한 원자적 리뷰:**
```bash
HEAD_SHA=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$GH_OWNER/$GH_REPO/pulls/$PR_NUMBER \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['head']['sha'])")

# Build the review JSON — event is APPROVE, REQUEST_CHANGES, or COMMENT
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$GH_OWNER/$GH_REPO/pulls/$PR_NUMBER/reviews \
  -d "{
    \"commit_id\": \"$HEAD_SHA\",
    \"event\": \"REQUEST_CHANGES\",
    \"body\": \"## Hermes Agent Review\n\nFound 2 issues, 1 suggestion. See inline comments.\",
    \"comments\": [
      {\"path\": \"src/auth.py\", \"line\": 45, \"body\": \"🔴 **Critical:** User input passed directly to SQL query — use parameterized queries.\"},
      {\"path\": \"src/models.py\", \"line\": 23, \"body\": \"⚠️ **Warning:** Password stored without hashing.\"},
      {\"path\": \"src/utils.py\", \"line\": 8, \"body\": \"💡 **Suggestion:** This duplicates logic in core/utils.py:34.\"}
    ]
  }"
```

### 8단계: 요약 댓글도 게시

인라인 댓글과 함께 PR 작성자가 전체 내용을 한눈에 볼 수 있도록 상위 수준의 요약도 남깁니다. `references/review-output-template.md`의 검토 결과 형식을 사용합니다.

**gh 사용:**
```bash
gh pr comment $PR_NUMBER --body "$(cat <<'EOF'
## Code Review Summary

**Verdict: Changes Requested** (2 issues, 1 suggestion)

### 🔴 Critical
- **src/auth.py:45** — SQL injection vulnerability

### ⚠️ Warnings
- **src/models.py:23** — Plaintext password storage

### 💡 Suggestions
- **src/utils.py:8** — Duplicated logic, consider consolidating

### ✅ Looks Good
- Clean API design
- Good error handling in the middleware layer

---
*Reviewed by Hermes Agent*
EOF
)"
```

### 9단계: 정리

```bash
git checkout main
git branch -D pr-$PR_NUMBER
```

### 결정: 승인 vs 변경 요청 vs 댓글

- **승인** — 치명적 문제나 경고 수준의 문제가 없고, 사소한 제안만 있거나 모든 것이 명확한 경우
- **변경 요청** — 병합 전에 수정해야 하는 치명적 문제나 경고 수준의 문제가 있는 경우
- **댓글** — 관찰 사항과 제안이 있지만 차단 요소는 없는 경우 (초안 PR이거나 확신이 서지 않을 때 사용)
