---
title: "Github Repo Management — GitHub CLI (`gh`): 저장소, 이슈, PR 등 GitHub 리소스 관리"
sidebar_label: "Github Repo Management"
description: "GitHub CLI (`gh`): 저장소, 이슈, PR 등 GitHub 리소스 관리"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Github Repo Management

GitHub CLI (`gh`): 저장소, 이슈, PR 등 GitHub 리소스 관리.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 내장 (기본으로 설치됨) |
| 경로 | `skills/github/github-repo-management` |
| 버전 | `1.0.0` |
| 작성자 | Hermes |
| 라이선스 | MIT |
| 의존성 | `gh` |
| 플랫폼 | linux, macos, windows |
| 태그 | `GitHub`, `Repository Management`, `Pull Requests`, `Issues`, `CLI` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# GitHub Repository Management

`gh` (GitHub CLI)를 사용하여 GitHub 리소스를 관리하기 위한 종합 가이드입니다.

## `gh` 사용 시기

**다음과 같은 경우에 `gh`를 사용하세요:**
- 저장소를 생성하거나 복제(clone)할 때
- 풀 리퀘스트(PR)를 관리할 때 (생성, 검토, 병합)
- 이슈를 처리할 때 (생성, 할당, 검색)
- 브라우저를 열지 않고 GitHub Actions를 실행하거나 모니터링할 때
- GitHub Releases를 관리할 때

**대신 다음 대안을 사용하는 것이 좋은 경우:**
- 로컬 파일 기록을 검사하거나 브랜치를 변경할 때 -> `git` 사용

## 빠른 시작

### 인증
```bash
# 대화형 로그인
gh auth login

# 상태 확인
gh auth status
```

### 일반적인 저장소 작업
```bash
# 저장소 복제 (URL이나 사용자/저장소 형식 사용 가능)
gh repo clone cli/cli

# 로컬 디렉토리에 새 저장소 생성
gh repo create my-project --public --source=. --remote=upstream
```

## 일반적인 워크플로우

### 워크플로우 1: 풀 리퀘스트(PR) 관리
```bash
# 로컬 브랜치에서 PR 생성
gh pr create --title "기능 추가" --body "이 PR은 새로운 기능을 추가합니다."

# 열려 있는 PR 목록 보기
gh pr list

# 특정 PR 확인(checkout)하기
gh pr checkout 123

# PR 상태 확인
gh pr status

# PR 병합(merge)
gh pr merge 123 --squash
```

### 워크플로우 2: 이슈 추적
```bash
# 새 이슈 생성
gh issue create --title "버그: 앱 크래시" --body "로그인 시 앱이 크래시됩니다."

# 열려 있는 이슈 목록 보기
gh issue list

# 이슈를 특정인에게 할당
gh issue edit 456 --add-assignee octocat
```

### 워크플로우 3: GitHub Actions
```bash
# 워크플로우 실행(run) 목록 보기
gh run list

# 특정 워크플로우의 로그 보기
gh run view 789 --log
```

## 스크립팅과 JSON 처리
`gh`는 스크립트 작성이나 상세 정보를 추출하는 데 적합한 뛰어난 JSON 출력 지원을 제공합니다. `jq`와 결합하여 효율적으로 파싱할 수 있습니다.

```bash
# 이슈에서 특정 필드를 JSON으로 추출
gh issue list --json number,title,assignees

# jq와 함께 사용하여 결과를 필터링
gh pr list --json title,url --jq '.[] | "- \(.title) (\(.url))"'
```

## 일반적인 문제

**문제:** `gh`가 브라우저를 열어 인증을 요청하는데, 터미널(헤드리스) 환경에 있을 때.
- **해결책:** 웹 브라우저를 사용할 수 없는 경우 `--with-token` 플래그나 `GH_TOKEN` 환경 변수를 사용하여 개인 액세스 토큰(Personal Access Token, PAT)으로 인증하세요.
  ```bash
  echo "YOUR_PAT" | gh auth login --with-token
  ```

**문제:** "GraphQL: Not Found" 에러
- **해결책:** 명령을 실행할 권한이 있는지 또는 저장소가 존재하는지 확인하세요. 또한 `gh auth status`를 통해 올바른 계정으로 로그인되어 있는지 확인하세요.
