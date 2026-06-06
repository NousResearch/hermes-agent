---
title: "Github Code Review — PR 리뷰: gh 또는 REST를 통한 diff, 인라인 댓글"
sidebar_label: "Github Code Review"
description: "PR 리뷰: gh 또는 REST를 통한 diff, 인라인 댓글"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py 스크립트에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Github Code Review

PR 리뷰: gh 또는 REST를 통한 diff, 인라인 댓글.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/github/github-code-review` |
| Version | `1.1.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `GitHub`, `Code-Review`, `Pull-Requests`, `Git`, `Quality` |
| Related skills | [`github-auth`](/docs/user-guide/skills/bundled/github/github-github-auth), [`github-pr-workflow`](/docs/user-guide/skills/bundled/github/github-github-pr-workflow) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 것입니다.
:::

# GitHub 코드 리뷰 (GitHub Code Review)

푸시하기 전에 로컬 변경 사항을 코드 리뷰하거나 GitHub에서 열려 있는 PR을 리뷰합니다. 이 스킬의 대부분은 순수 `git`을 사용하며, `gh`/`curl` 구분은 PR 수준의 상호 작용에서만 중요합니다.

## 전제 조건

- GitHub에 인증됨 (`github-auth` 스킬 참조)
- git 리포지토리 내부

### 설정 (PR 상호 작용용)

```bash
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  AUTH="gh"
else
  AUTH="git"
  if [ -z "$GITHUB_TOKEN" ]; then
    if [ -f ~/.hermes/.env ] && grep -q "^GITHUB_TOKEN=" ~/.hermes/.env; then
      GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" ~/.hermes/.env | head -1 | cut -d= -f2 | tr -d '\n\r')
    elif grep -q "github.com" ~/.git-credentials 2>/dev/null; then
      GITHUB_TOKEN=$(grep "github.com" ~/.git-credentials 2>/dev/null | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|')
    fi
  fi
fi

REMOTE_URL=$(git remote get-url origin)
OWNER_REPO=$(echo "$REMOTE_URL" | sed -E 's|.*github\.com[:/]||; s|\.git$||')
OWNER=$(echo "$OWNER_REPO" | cut -d/ -f1)
REPO=$(echo "$OWNER_REPO" | cut -d/ -f2)
```

---

## 1. 로컬 변경 사항 리뷰 (푸시 전)

이것은 순수 `git`입니다 — API 없이 어디서나 작동합니다.

### Diff 가져오기

```bash
# 스테이징된 변경 사항 (커밋될 내용)
git diff --staged

# main 대비 모든 변경 사항 (PR에 포함될 내용)
git diff main...HEAD

# 파일 이름만
git diff main...HEAD --name-only

# Stat 요약 (파일당 추가/삭제 줄 수)
git diff main...HEAD --stat
```

### 리뷰 전략

1. **먼저 큰 그림을 봅니다:**

```bash
git diff main...HEAD --stat
git log main..HEAD --oneline
```

2. **파일별로 리뷰합니다** — 변경된 파일에 대해 전체 컨텍스트를 보려면 `read_file`을 사용하고, 무엇이 변경되었는지 보려면 diff를 사용합니다:

```bash
git diff main...HEAD -- src/auth/login.py
```

3. **일반적인 문제를 확인합니다:**

```bash
# 남겨진 디버그 문, TODO, console.log
git diff main...HEAD | grep -n "print(\|console\.log\|TODO\|FIXME\|HACK\|XXX\|debugger"

# 실수로 스테이징된 큰 파일
git diff main...HEAD --stat | sort -t'|' -k2 -rn | head -10

# 비밀 또는 자격 증명 패턴
git diff main...HEAD | grep -in "password\|secret\|api_key\|token.*=\|private_key"

# 병합 충돌 마커
git diff main...HEAD | grep -n "<<<<<<\|>>>>>>\|======="
```

4. 사용자가 이해할 수 있게 구조화된 피드백을 **제시**합니다.

### 리뷰 출력 형식

로컬 변경 사항을 리뷰할 때 다음 구조로 결과를 제시합니다:

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

---

## 2. GitHub에서 Pull Request 리뷰

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

# PR 세부 정보 가져오기
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

# 변경된 파일 목록
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/files \
  | python3 -c "
import sys, json
for f in json.load(sys.stdin):
    print(f\"{f['status']:10} +{f['additions']:-4} -{f['deletions']:-4}  {f['filename']}\")"
```

### 전체 리뷰를 위해 PR 로컬로 체크아웃

이것은 순수 `git`으로 작동합니다 — `gh`가 필요하지 않습니다:

```bash
# PR 브랜치를 가져와서 체크아웃
git fetch origin pull/123/head:pr-123
git checkout pr-123

# 이제 read_file, search_files, 테스트 실행 등을 사용할 수 있습니다.

# 기본 브랜치와 비교한 diff 보기
git diff main...pr-123
```

**gh 사용 (바로가기):**

```bash
gh pr checkout 123
```

### PR에 댓글 남기기

**일반적인 PR 댓글 — gh 사용:**

```bash
gh pr comment 123 --body "Overall looks good, a few suggestions below."
```

**일반적인 PR 댓글 — curl 사용:**

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
# 헤드 커밋 SHA 가져오기
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

**curl 사용 — 원자적으로 제출되는 다중 댓글 리뷰:**

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

이벤트(Event) 값: `"APPROVE"`, `"REQUEST_CHANGES"`, `"COMMENT"`

`line` 필드는 파일의 *새로운* 버전의 줄 번호를 참조합니다. 삭제된 줄의 경우 `"side": "LEFT"`를 사용하세요.

---

## 3. 리뷰 체크리스트

코드 리뷰(로컬 또는 PR)를 수행할 때 체계적으로 다음을 확인하세요:

### 정확성 (Correctness)
- 코드가 의도한 대로 동작합니까?
- 엣지 케이스 (빈 입력, null, 대량 데이터, 동시 액세스)를 처리합니까?
- 오류 경로를 우아하게 처리합니까?

### 보안 (Security)
- 하드코딩된 비밀, 자격 증명 또는 API 키가 없습니까?
- 사용자 대상 입력에 대한 입력 유효성 검사가 있습니까?
- SQL 인젝션, XSS 또는 경로 탐색 취약점이 없습니까?
- 필요한 경우 인증/인가 검사가 있습니까?

### 코드 품질 (Code Quality)
- 명확한 이름 지정 (변수, 함수, 클래스)
- 불필요한 복잡성이나 섣부른 추상화가 없습니까?
- DRY — 추출해야 할 중복 로직이 없습니까?
- 함수가 집중되어 있습니까 (단일 책임)?

### 테스팅 (Testing)
- 새로운 코드 경로가 테스트되었습니까?
- 해피 패스와 오류 케이스가 다루어졌습니까?
- 테스트가 가독성 있고 유지 관리하기 쉽습니까?

### 성능 (Performance)
- N+1 쿼리나 불필요한 루프가 없습니까?
- 유리한 경우 적절한 캐싱이 사용됩니까?
- 비동기 코드 경로에 차단 작업이 없습니까?

### 문서화 (Documentation)
- 공개 API가 문서화되었습니까?
- 명확하지 않은 로직에는 "이유"를 설명하는 주석이 있습니까?
- 동작이 변경된 경우 README가 업데이트되었습니까?

---

## 4. 푸시 전 리뷰 워크플로우

사용자가 "코드 리뷰해 줘" 또는 "푸시하기 전에 확인해 줘"라고 요청할 때:

1. `git diff main...HEAD --stat` — 변경 범위 확인
2. `git diff main...HEAD` — 전체 diff 읽기
3. 각 변경된 파일에 대해, 더 많은 컨텍스트가 필요하면 `read_file` 사용
4. 위의 체크리스트 적용
5. 결과를 구조화된 형식으로 제시 (Critical / Warnings / Suggestions / Looks Good)
6. 심각한 문제가 발견되면 사용자가 푸시하기 전에 수정하겠다고 제안

---

## 5. PR 리뷰 워크플로우 (종단간)

사용자가 "PR #N 리뷰해 줘", "이 PR 좀 봐줘"라고 요청하거나 PR URL을 제공할 때, 이 레시피를 따르세요:

### 1단계: 환경 설정

```bash
source "${HERMES_HOME:-$HOME/.hermes}/skills/github/github-auth/scripts/gh-env.sh"
# 또는 이 스킬의 상단에 있는 인라인 설정 블록 실행
```

### 2단계: PR 컨텍스트 수집

코드로 들어가기 전에 PR 메타데이터, 설명 및 변경된 파일 목록을 가져와 범위를 이해합니다.

**gh 사용:**
```bash
gh pr view 123
gh pr diff 123 --name-only
gh pr checks 123
```

**curl 사용:**
```bash
PR_NUMBER=123

# PR 세부 정보 (제목, 작성자, 설명, 브랜치)
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$GH_OWNER/$GH_REPO/pulls/$PR_NUMBER

# 줄 수가 포함된 변경된 파일
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$GH_OWNER/$GH_REPO/pulls/$PR_NUMBER/files
```

### 3단계: 로컬로 PR 체크아웃

이렇게 하면 `read_file`, `search_files`에 대한 전체 액세스 권한과 테스트 실행 기능이 제공됩니다.

```bash
git fetch origin pull/$PR_NUMBER/head:pr-$PR_NUMBER
git checkout pr-$PR_NUMBER
```

### 4단계: diff를 읽고 변경 사항 이해하기

```bash
# 기본 브랜치와 비교한 전체 diff
git diff main...HEAD

# 또는 큰 PR의 경우 파일별로
git diff main...HEAD --name-only
# 그 다음 각 파일에 대해:
git diff main...HEAD -- path/to/file.py
```

변경된 각 파일에 대해, `read_file`을 사용하여 변경 사항 주변의 전체 컨텍스트를 확인하세요 — diff만으로는 주변 코드에서만 볼 수 있는 문제를 놓칠 수 있습니다.

### 5단계: 로컬에서 자동화된 검사 실행 (해당하는 경우)

```bash
# 테스트 스위트가 있는 경우 테스트 실행
python -m pytest 2>&1 | tail -20
# 또는: npm test, cargo test, go test ./... 등

# 구성된 경우 린터 실행
ruff check . 2>&1 | head -30
# 또는: eslint, clippy 등
```

### 6단계: 리뷰 체크리스트 적용 (섹션 3)

각 카테고리를 살펴봅니다: 정확성, 보안, 코드 품질, 테스팅, 성능, 문서화.

### 7단계: GitHub에 리뷰 게시

결과를 모아 인라인 댓글과 함께 공식 리뷰로 제출합니다.

**gh 사용:**
```bash
# 문제가 없으면 — 승인
gh pr review $PR_NUMBER --approve --body "Reviewed by Hermes Agent. Code looks clean — good test coverage, no security concerns."

# 문제가 발견되면 — 인라인 댓글과 함께 변경 요청
gh pr review $PR_NUMBER --request-changes --body "Found a few issues — see inline comments."
```

**curl 사용 — 여러 인라인 댓글이 포함된 원자적 리뷰:**
```bash
HEAD_SHA=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$GH_OWNER/$GH_REPO/pulls/$PR_NUMBER \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['head']['sha'])")

# 리뷰 JSON 구성 — event는 APPROVE, REQUEST_CHANGES 또는 COMMENT입니다.
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

인라인 댓글 외에도 PR 작성자가 전체 상황을 한눈에 볼 수 있도록 최상위 요약을 남겨주세요. `references/review-output-template.md`의 리뷰 출력 형식을 사용하세요.

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

### 결정: 승인(Approve) vs 변경 요청(Request Changes) vs 댓글(Comment)

- **승인(Approve)** — Critical 또는 Warning 수준의 문제가 없으며, 사소한 제안만 있거나 모두 깨끗함
- **변경 요청(Request Changes)** — 병합하기 전에 해결해야 하는 Critical 또는 Warning 수준의 문제
- **댓글(Comment)** — 관찰 및 제안이지만 차단하는 문제는 없음 (확신이 서지 않거나 PR이 초안인 경우 사용)
